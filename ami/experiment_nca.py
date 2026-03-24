# ami/experiment_nca.py
#
# Routing NCA experiment — drop-in replacement for experiment_zones.py.
# Uses trained routing NCA checkpoint instead of Physarum agent simulation.
#
# Same IPC interface: reads ami_trigger.json, writes ami/zone_state.json.
# watcher.py and responder.py are unchanged.
#
# Zone B (top right)  — Claude   (politics signal)
# Zone C (bot right)  — Gemini   (climate signal)
#
# Controls:
#   P  — fire politics signal  (→ Zone B / Claude)
#   C  — fire climate signal   (→ Zone C / Gemini)
#   R  — reset grid
#   Q  — quit
#
# Run:
#   source ~/ai-env/bin/activate
#   python ami/experiment_nca.py

import sys
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import pygame
from pathlib import Path
from scipy.ndimage import zoom

sys.path.insert(0, 'nca')
sys.path.insert(0, 'ami')

from model import UpdateNet, make_perception_kernel, nca_step
from physarum_core import (
    H as SIM_H, W as SIM_W,
    GAP_LEFT, GAP_RIGHT, RIGHT_MID,
    ANCHOR_A, ANCHOR_B, ANCHOR_C, ANCHOR_RADIUS,
    ZONE_B_THRESHOLD, ZONE_C_THRESHOLD,
    BRIDGE_HOLD_STEPS,
)
from train_routing import (
    TRAIN_H, TRAIN_W, ZONE_MAP_64, ZONE_MAP_64_JAX,
    CH_TRAIL, CH_SIGNAL, CH_ZONE, CH_PHYSICS, CH_F, CH_K,
    inject_zone_a_signal_only, generate_state,
)

CHECKPOINT      = Path("ami/routing_checkpoints/routing_020000.pkl")
TRIGGER_FILE    = Path("ami/ami_trigger.json")
ZONE_STATE_FILE = Path("ami/zone_state.json")

SCALE            = 8    # 64*8 = 512px window
FPS              = 30
RESEED_INTERVAL  = 50   # blend base trail every N steps to keep grid alive
BASE_BLEND       = 0.15  # how much base trail to blend in each reseed
NCA_THRESHOLD_B  = 0.005
NCA_THRESHOLD_C  = 0.005

# Zone boundaries at training resolution
GAP_L_64 = int(GAP_LEFT  * TRAIN_W / SIM_W)
GAP_R_64 = int(GAP_RIGHT * TRAIN_W / SIM_W)
MID_64   = TRAIN_H // 2


# ── Zone measurement ───────────────────────────────────────────────────────────

def measure_zones_64(grid):
    trail  = np.array(grid[:, :, CH_TRAIL])
    zone_a = float(trail[:, :GAP_L_64].mean())
    zone_b = float(trail[:MID_64, GAP_R_64:].mean())
    zone_c = float(trail[MID_64:, GAP_R_64:].mean())
    gap    = float(trail[:, GAP_L_64:GAP_R_64].mean())
    return zone_a, zone_b, zone_c, gap


# ── Signal injection ───────────────────────────────────────────────────────────

def inject_signal(grid_j, signal_type):
    """Inject Zone-A-only signal shape + set CH_SIGNAL. Returns updated grid."""
    trail     = np.array(grid_j[:, :, CH_TRAIL])
    trail_sim = zoom(trail, (SIM_H/TRAIN_H, SIM_W/TRAIN_W), order=1)
    trail_sim = inject_zone_a_signal_only(trail_sim, signal_type)
    trail_64  = zoom(trail_sim, (TRAIN_H/SIM_H, TRAIN_W/SIM_W), order=1)
    sig_val   = 0.5 if signal_type == "politics" else 1.0
    grid_j    = grid_j.at[:, :, CH_TRAIL].set(jnp.clip(jnp.array(trail_64), 0, 1))
    grid_j    = grid_j.at[:, :, CH_SIGNAL].set(sig_val)
    return grid_j


# ── Render ─────────────────────────────────────────────────────────────────────

def render(screen, grid_j, step_count, font, active_signal, zone_b, zone_c):
    trail = np.array(grid_j[:, :, CH_TRAIL])
    arr   = (np.clip(trail, 0, 1) * 255).astype(np.uint8)
    arr   = np.repeat(np.repeat(arr, SCALE, axis=0), SCALE, axis=1)
    rgb   = np.stack([arr, arr, arr], axis=-1)

    gap_l_px = GAP_L_64 * SCALE
    gap_r_px = GAP_R_64 * SCALE
    mid_px   = MID_64   * SCALE

    # Zone tints
    rgb[:mid_px, gap_r_px:, 2] = np.clip(
        rgb[:mid_px, gap_r_px:, 2].astype(int) + 25, 0, 255)
    rgb[mid_px:, gap_r_px:, 1] = np.clip(
        rgb[mid_px:, gap_r_px:, 1].astype(int) + 25, 0, 255)

    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))

    # Zone boundaries
    pygame.draw.line(screen, (255, 50, 50),
                     (gap_l_px, 0), (gap_l_px, TRAIN_H*SCALE), 1)
    pygame.draw.line(screen, (255, 50, 50),
                     (gap_r_px, 0), (gap_r_px, TRAIN_H*SCALE), 1)
    pygame.draw.line(screen, (80, 80, 80),
                     (gap_r_px, mid_px), (TRAIN_W*SCALE, mid_px), 1)

    # Labels
    screen.blit(font.render("ZONE A", True, (200, 200, 200)), (6, 6))
    screen.blit(font.render("ZONE B — Claude", True, (100, 160, 255)),
                (gap_r_px + 4, 6))
    screen.blit(font.render("ZONE C — Gemini", True, (100, 255, 160)),
                (gap_r_px + 4, mid_px + 6))

    # Zone measurements
    b_col = (0, 255, 80) if zone_b >= ZONE_B_THRESHOLD else (100, 160, 255)
    c_col = (0, 255, 80) if zone_c >= ZONE_C_THRESHOLD else (100, 255, 160)
    screen.blit(font.render(f"B={zone_b:.4f}", True, b_col),
                (gap_r_px + 4, 20))
    screen.blit(font.render(f"C={zone_c:.4f}", True, c_col),
                (gap_r_px + 4, mid_px + 20))

    # Status bar
    if active_signal == "politics":
        status = "POLITICS → Zone B (Claude)"
        sc = (100, 160, 255)
    elif active_signal == "climate":
        status = "CLIMATE → Zone C (Gemini)"
        sc = (100, 255, 160)
    else:
        status = "P=politics  C=climate  R=reset  Q=quit"
        sc = (160, 160, 160)

    screen.blit(font.render(f"step={step_count}  {status}", True, sc),
                (4, TRAIN_H*SCALE - 18))
    pygame.display.flip()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {CHECKPOINT}...")
    with open(CHECKPOINT, 'rb') as f:
        params = pickle.load(f)
    params = jax.device_put(params)
    print("  Loaded.")

    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()

    pygame.init()
    screen = pygame.display.set_mode((TRAIN_W*SCALE, TRAIN_H*SCALE))
    pygame.display.set_caption("AmI — Routing NCA")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 12)

    # Reset IPC files on startup
    ZONE_STATE_FILE.write_text(json.dumps({
        "zone_b_activated": False,
        "zone_c_activated": False,
        "signal_id":        None,
        "signal_type":      None,
        "step":             0,
    }))
    if TRIGGER_FILE.exists():
        try:
            t = json.loads(TRIGGER_FILE.read_text())
            t["consumed"] = True
            TRIGGER_FILE.write_text(json.dumps(t, indent=2))
        except Exception:
            pass

    # Bootstrap substrate from Physarum warmup
    print("  Warming up substrate...")
    rng       = np.random.default_rng(42)
    grid_0    = generate_state(rng, signal_type=None, warmup=200)
    grid_j    = jnp.array(grid_0)
    key       = jax.random.PRNGKey(0)
    # Save base trail for continuous re-injection to keep grid alive
    base_trail = grid_0[:, :, CH_TRAIL].copy()
    base_trail_j = jnp.array(base_trail)

    step_count       = 0
    active_signal    = None
    hold_steps       = 0
    active_signal_id = None
    llm_triggered    = False
    running          = True

    print(f"  Grid: {TRAIN_H}x{TRAIN_W}  |  Gap cols {GAP_L_64}-{GAP_R_64}")
    print(f"  Zone B (Claude): rows 0-{MID_64},   cols {GAP_R_64}-{TRAIN_W}")
    print(f"  Zone C (Gemini): rows {MID_64}-{TRAIN_H}, cols {GAP_R_64}-{TRAIN_W}")
    print(f"  Thresholds: B={NCA_THRESHOLD_B}  C={NCA_THRESHOLD_C}")
    print()
    print("  P = politics signal  (→ Zone B / Claude)")
    print("  C = climate signal   (→ Zone C / Gemini)")
    print("  R = reset | Q = quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    grid_0   = generate_state(rng, signal_type=None, warmup=200)
                    grid_j   = jnp.array(grid_0)
                    step_count       = 0
                    active_signal    = None
                    hold_steps       = 0
                    active_signal_id = None
                    llm_triggered    = False
                    print("  --- RESET ---")
                if event.key == pygame.K_p and not active_signal:
                    active_signal    = "politics"
                    hold_steps       = BRIDGE_HOLD_STEPS
                    active_signal_id = None   # manual fire — no signal_id
                    llm_triggered    = False
                    print(f"  step {step_count}: POLITICS signal → Zone B")
                if event.key == pygame.K_c and not active_signal:
                    active_signal    = "climate"
                    hold_steps       = BRIDGE_HOLD_STEPS
                    active_signal_id = None
                    llm_triggered    = False
                    print(f"  step {step_count}: CLIMATE signal → Zone C")

        # Check for incoming trigger from watcher.py
        if TRIGGER_FILE.exists():
            try:
                trigger = json.loads(TRIGGER_FILE.read_text())
                if not trigger.get("consumed") and not active_signal:
                    sig_type         = trigger.get("signal_type", "politics")
                    active_signal_id = trigger.get("signal_id")
                    topic            = trigger.get("topic", "unknown")
                    active_signal    = sig_type
                    hold_steps       = BRIDGE_HOLD_STEPS
                    llm_triggered    = False
                    print(f"  [nca] Signal '{sig_type}' received for '{topic}' "
                          f"(id={str(active_signal_id)[:8]}...) — routing")
            except Exception:
                pass

        # Re-inject signal every step during hold window
        if hold_steps > 0:
            grid_j     = inject_signal(grid_j, active_signal)
            hold_steps -= 1
            if hold_steps == 0:
                grid_j = grid_j.at[:, :, CH_SIGNAL].set(0.0)
                print(f"  step {step_count}: hold ended — signal released")
                # Keep active_signal set if from watcher so zone check still fires
                if active_signal_id is None:
                    active_signal = None

        # Continuously blend base trail to keep substrate alive
        if step_count % RESEED_INTERVAL == 0 and not active_signal:
            grid_j = grid_j.at[:, :, CH_TRAIL].set(
                jnp.clip(grid_j[:, :, CH_TRAIL] + base_trail_j * BASE_BLEND, 0, 1)
            )

        # NCA step
        grid_j, key = nca_step(grid_j, params, update_net, perception_kernel, key)

        # Re-inject control channels
        grid_j = grid_j.at[:, :, CH_ZONE].set(ZONE_MAP_64_JAX)
        grid_j = grid_j.at[:, :, CH_PHYSICS].set(0.5)

        step_count += 1

        # Measure zones and write state every 20 steps
        if step_count % 20 == 0:
            zone_a, zone_b, zone_c, gap = measure_zones_64(np.array(grid_j))

            zone_b_activated = (
                active_signal == "politics"
                and active_signal_id is not None
                and zone_b >= NCA_THRESHOLD_B
            )
            zone_c_activated = (
                active_signal == "climate"
                and active_signal_id is not None
                and zone_c >= NCA_THRESHOLD_C
            )

            if (zone_b_activated or zone_c_activated) and not llm_triggered:
                which = "B (Claude)" if zone_b_activated else "C (Gemini)"
                val   = zone_b if zone_b_activated else zone_c
                print(f"  [nca] Zone {which} activated ({val:.3f}) — "
                      f"signal {str(active_signal_id)[:8]}... reached destination")
                llm_triggered = True

            ZONE_STATE_FILE.write_text(json.dumps({
                "step":             step_count,
                "zone_a":           zone_a,
                "zone_b":           zone_b,
                "zone_c":           zone_c,
                "gap":              gap,
                "active_signal":    active_signal,
                "zone_b_activated": zone_b_activated,
                "zone_c_activated": zone_c_activated,
                "signal_id":        active_signal_id if (zone_b_activated or zone_c_activated) else None,
                "signal_type":      active_signal if (zone_b_activated or zone_c_activated) else None,
                "threshold_b":      NCA_THRESHOLD_B,
                "threshold_c":      NCA_THRESHOLD_C,
            }))

        # Log every 200 steps
        if step_count % 200 == 0:
            zone_a, zone_b, zone_c, gap = measure_zones_64(np.array(grid_j))
            print(f"  step {step_count:5d}  A={zone_a:.3f}  gap={gap:.3f}  "
                  f"B={zone_b:.3f}  C={zone_c:.3f}  signal={active_signal or 'none'}")

        render(screen, grid_j, step_count, font, active_signal,
               *measure_zones_64(np.array(grid_j))[1:3])
        clock.tick(FPS)

    pygame.quit()
    print(f"\nStopped at step {step_count}")


if __name__ == "__main__":
    main()
