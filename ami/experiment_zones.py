# ami/experiment_zones.py
#
# Three-zone Physarum routing experiment.
# Interactive pygame visualizer — uses physarum_core.py for all physics.
#
# Zone A (left)       — input / watcher
# Zone B (top right)  — Claude   (AI in politics signal)
# Zone C (bot right)  — Gemini   (climate tech signal)
#
# Controls:
#   P      — fire politics signal (horizontal bar → should reach Zone B)
#   C      — fire climate signal  (vertical bar   → should reach Zone C)
#   R      — reset
#   Q      — quit
#
# Run:
#   source ~/ai-env/bin/activate
#   python ami/experiment_zones.py

import numpy as np
import pygame
import json
from pathlib import Path
from physarum_core import (
    reset, step, inject_anchors, reseed_agents,
    inject_signal_politics, inject_signal_climate,
    reinforce_signal_politics, reinforce_signal_climate,
    measure_zones, ZONE_MAP,
    H, W, GAP_LEFT, GAP_RIGHT, RIGHT_MID,
    ANCHOR_A, ANCHOR_B, ANCHOR_C, ANCHOR_RADIUS,
    BRIDGE_HOLD_STEPS, ZONE_B_THRESHOLD, ZONE_C_THRESHOLD,
    RESEED_INTERVAL,
)

TRIGGER_FILE    = Path("ami/ami_trigger.json")
ZONE_STATE_FILE = Path("ami/zone_state.json")

SCALE = 3
FPS   = 60


# ── Render ────────────────────────────────────────────────────────────────────

def render(screen, trail, step_count, font, active_signal):
    # Trail as grayscale
    display_arr = (np.clip(trail, 0, 1) * 255).astype(np.uint8)
    display_arr = np.repeat(np.repeat(display_arr, SCALE, axis=0), SCALE, axis=1)
    rgb = np.stack([display_arr, display_arr, display_arr], axis=-1)

    # Tint Zone B (top right) slightly blue, Zone C (bottom right) slightly green
    rgb[                :RIGHT_MID*SCALE, GAP_RIGHT*SCALE:, 2] = np.clip(
        rgb[:RIGHT_MID*SCALE, GAP_RIGHT*SCALE:, 2].astype(int) + 20, 0, 255)
    rgb[RIGHT_MID*SCALE:,                GAP_RIGHT*SCALE:, 1] = np.clip(
        rgb[RIGHT_MID*SCALE:, GAP_RIGHT*SCALE:, 1].astype(int) + 20, 0, 255)

    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))

    # Anchor circles
    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    cx, cy = ANCHOR_C
    pygame.draw.circle(screen, (200, 200, 200),
                       (ay*SCALE, ax*SCALE), ANCHOR_RADIUS*SCALE, 2)
    pygame.draw.circle(screen, (100, 160, 255),
                       (by*SCALE, bx*SCALE), ANCHOR_RADIUS*SCALE, 2)
    pygame.draw.circle(screen, (100, 255, 160),
                       (cy*SCALE, cx*SCALE), ANCHOR_RADIUS*SCALE, 2)

    # Zone boundary lines
    pygame.draw.line(screen, (255, 50, 50),
                     (GAP_LEFT*SCALE, 0), (GAP_LEFT*SCALE, H*SCALE), 1)
    pygame.draw.line(screen, (255, 50, 50),
                     (GAP_RIGHT*SCALE, 0), (GAP_RIGHT*SCALE, H*SCALE), 1)
    pygame.draw.line(screen, (100, 100, 100),
                     (GAP_RIGHT*SCALE, RIGHT_MID*SCALE), (W*SCALE, RIGHT_MID*SCALE), 1)

    # Zone labels
    screen.blit(font.render("ZONE A  (input)", True, (200, 200, 200)), (8, 8))
    screen.blit(font.render("ZONE B — Claude", True, (100, 160, 255)),
                (GAP_RIGHT*SCALE + 8, 8))
    screen.blit(font.render("ZONE C — Gemini", True, (100, 255, 160)),
                (GAP_RIGHT*SCALE + 8, RIGHT_MID*SCALE + 8))
    screen.blit(font.render("GAP", True, (255, 255, 0)),
                (GAP_LEFT*SCALE + 5, 8))

    # Zone measurements
    zone_a, zone_b, zone_c, gap = measure_zones(trail)
    screen.blit(font.render(f"B={zone_b:.3f}", True, (100, 160, 255)),
                (GAP_RIGHT*SCALE + 8, 24))
    screen.blit(font.render(f"C={zone_c:.3f}", True, (100, 255, 160)),
                (GAP_RIGHT*SCALE + 8, RIGHT_MID*SCALE + 24))

    # Status bar
    if active_signal == "politics":
        status = "POLITICS SIGNAL → Zone B (Claude)"
        color  = (100, 160, 255)
    elif active_signal == "climate":
        status = "CLIMATE SIGNAL → Zone C (Gemini)"
        color  = (100, 255, 160)
    else:
        status = "P=politics  C=climate  R=reset  Q=quit"
        color  = (180, 180, 180)

    txt = font.render(
        f"step={step_count}  gap={gap:.3f}  |  {status}", True, color
    )
    screen.blit(txt, (5, H*SCALE - 20))

    pygame.display.flip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W*SCALE, H*SCALE))
    pygame.display.set_caption("AmI — Three Zone Routing Experiment")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 13)

    # Reset state files on startup
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

    rng = np.random.default_rng(42)
    trail, agent_x, agent_y, agent_heading, zone_ids = reset(rng)

    step_count       = 0
    active_signal    = None     # "politics", "climate", or None
    hold_steps       = 0
    active_signal_id = None
    llm_triggered    = False
    running          = True

    print("Three-zone routing experiment.")
    print(f"  Grid: {H}x{W}  |  Gap: cols {GAP_LEFT}-{GAP_RIGHT}")
    print(f"  Zone B (Claude):  rows 0-{RIGHT_MID},   cols {GAP_RIGHT}-{W}")
    print(f"  Zone C (Gemini):  rows {RIGHT_MID}-{H}, cols {GAP_RIGHT}-{W}")
    print()
    print("  P = fire politics signal (→ Zone B / Claude)")
    print("  C = fire climate signal  (→ Zone C / Gemini)")
    print("  R = reset | Q = quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    trail, agent_x, agent_y, agent_heading, zone_ids = reset(rng)
                    step_count    = 0
                    active_signal = None
                    hold_steps    = 0
                    print("  --- RESET ---")
                if event.key == pygame.K_p and not active_signal:
                    trail         = inject_signal_politics(trail)
                    active_signal = "politics"
                    hold_steps    = BRIDGE_HOLD_STEPS
                    print(f"  step {step_count}: POLITICS signal fired → routing to Zone B")
                if event.key == pygame.K_c and not active_signal:
                    trail         = inject_signal_climate(trail)
                    active_signal = "climate"
                    hold_steps    = BRIDGE_HOLD_STEPS
                    print(f"  step {step_count}: CLIMATE signal fired → routing to Zone C")

        trail, agent_x, agent_y, agent_heading = step(
            trail, agent_x, agent_y, agent_heading, rng
        )
        trail = inject_anchors(trail)
        step_count += 1

        if step_count % RESEED_INTERVAL == 0:
            agent_x, agent_y, agent_heading = reseed_agents(
                agent_x, agent_y, agent_heading, zone_ids, rng
            )

        # Check for incoming signal from watcher.py
        if TRIGGER_FILE.exists():
            try:
                trigger = json.loads(TRIGGER_FILE.read_text())
                if not trigger.get("consumed") and not active_signal:
                    sig_type      = trigger.get("signal_type", "politics")
                    active_signal_id = trigger.get("signal_id")
                    topic         = trigger.get("topic", "unknown")
                    if sig_type == "climate":
                        trail         = inject_signal_climate(trail)
                        active_signal = "climate"
                    else:
                        trail         = inject_signal_politics(trail)
                        active_signal = "politics"
                    hold_steps    = BRIDGE_HOLD_STEPS
                    llm_triggered = False
                    print(f"  [nca] Signal '{sig_type}' received for '{topic}' "
                          f"(id={str(active_signal_id)[:8]}...) — routing")
            except Exception:
                pass

        # Reinforce active signal during hold window, then clear when done
        if hold_steps > 0:
            if active_signal == "politics":
                trail = reinforce_signal_politics(trail)
            elif active_signal == "climate":
                trail = reinforce_signal_climate(trail)
            hold_steps -= 1
            if hold_steps == 0:
                active_signal    = None
                active_signal_id = None
                llm_triggered    = False

        # Write zone state every 50 steps
        if step_count % 50 == 0:
            zone_a, zone_b, zone_c, gap = measure_zones(trail)

            zone_b_activated = (
                active_signal == "politics"
                and active_signal_id is not None
                and zone_b >= ZONE_B_THRESHOLD
            )
            zone_c_activated = (
                active_signal == "climate"
                and active_signal_id is not None
                and zone_c >= ZONE_C_THRESHOLD
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
                "threshold_b":      ZONE_B_THRESHOLD,
                "threshold_c":      ZONE_C_THRESHOLD,
            }))

        # Log every 200 steps
        if step_count % 200 == 0:
            zone_a, zone_b, zone_c, gap = measure_zones(trail)
            print(f"  step {step_count:5d}  A={zone_a:.3f}  gap={gap:.3f}  "
                  f"B={zone_b:.3f}  C={zone_c:.3f}  signal={active_signal or 'none'}")

        render(screen, trail, step_count, font, active_signal)
        clock.tick(FPS)

    pygame.quit()
    print(f"\nStopped at step {step_count}")


if __name__ == "__main__":
    main()
