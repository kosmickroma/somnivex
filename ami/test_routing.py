# ami/test_routing.py
#
# Load a routing NCA checkpoint and visualize signal routing.
# Injects horizontal (politics) or vertical (climate) bar into Zone A only.
# Watch where the trail goes — does it route to Zone B or Zone C?
#
# Controls:
#   P  — inject horizontal bar (politics → should reach Zone B top-right)
#   C  — inject vertical bar  (climate  → should reach Zone C bottom-right)
#   R  — reset grid
#   Q  — quit
#
# Run:
#   source ~/ai-env/bin/activate
#   python ami/test_routing.py

import sys
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
    reset, step as phys_step, inject_anchors,
    measure_zones,
)
from train_routing import (
    TRAIN_H, TRAIN_W, ZONE_MAP_64, ZONE_MAP_64_JAX,
    CH_TRAIL, CH_SIGNAL, CH_ZONE, CH_PHYSICS, CH_F, CH_K,
    inject_zone_a_signal_only,
)

CHECKPOINT = Path("ami/routing_checkpoints/routing_010000.pkl")
SCALE      = 8   # 64x64 * 8 = 512px window
FPS        = 30


def make_initial_grid(rng, warmup=200):
    """Bootstrap a live Physarum state, downsample to 64x64 NCA grid."""
    sim_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    trail, ax, ay, ah, zone_ids = reset(sim_rng)
    for _ in range(warmup):
        trail, ax, ay, ah = phys_step(trail, ax, ay, ah, sim_rng)
        trail = inject_anchors(trail)
    trail_64 = zoom(trail, (TRAIN_H/SIM_H, TRAIN_W/SIM_W), order=1)
    trail_64 = np.clip(trail_64, 0, 1).astype(np.float32)
    grid = np.zeros((TRAIN_H, TRAIN_W, 16), dtype=np.float32)
    grid[:, :, CH_TRAIL]   = trail_64
    grid[:, :, CH_ZONE]    = ZONE_MAP_64
    grid[:, :, CH_PHYSICS] = 0.5
    grid[:, :, CH_F]       = 0.04
    grid[:, :, CH_K]       = 0.06
    return grid


def render(screen, grid, step_count, font, active_signal, zone_b, zone_c):
    gap_l_px = int(GAP_LEFT  * TRAIN_W / SIM_W) * SCALE
    gap_r_px = int(GAP_RIGHT * TRAIN_W / SIM_W) * SCALE
    mid_px   = (TRAIN_H // 2) * SCALE

    trail = grid[:, :, CH_TRAIL]
    arr   = (np.clip(trail, 0, 1) * 255).astype(np.uint8)
    arr   = np.repeat(np.repeat(arr, SCALE, axis=0), SCALE, axis=1)
    rgb   = np.stack([arr, arr, arr], axis=-1)

    # Zone tints
    rgb[:mid_px, gap_r_px:, 2] = np.clip(
        rgb[:mid_px, gap_r_px:, 2].astype(int) + 25, 0, 255)
    rgb[mid_px:, gap_r_px:, 1] = np.clip(
        rgb[mid_px:, gap_r_px:, 1].astype(int) + 25, 0, 255)

    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))

    # Zone boundary lines
    pygame.draw.line(screen, (255, 50, 50),
                     (gap_l_px, 0), (gap_l_px, TRAIN_H*SCALE), 1)
    pygame.draw.line(screen, (255, 50, 50),
                     (gap_r_px, 0), (gap_r_px, TRAIN_H*SCALE), 1)
    pygame.draw.line(screen, (80, 80, 80),
                     (gap_r_px, mid_px), (TRAIN_W*SCALE, mid_px), 1)

    # Labels
    screen.blit(font.render("ZONE A", True, (200,200,200)), (6, 6))
    screen.blit(font.render("ZONE B — Claude", True, (100,160,255)),
                (gap_r_px + 4, 6))
    screen.blit(font.render("ZONE C — Gemini", True, (100,255,160)),
                (gap_r_px + 4, mid_px + 6))

    # Zone measurements
    b_color = (0, 255, 0) if zone_b >= ZONE_B_THRESHOLD else (100, 160, 255)
    c_color = (0, 255, 0) if zone_c >= ZONE_C_THRESHOLD else (100, 255, 160)
    screen.blit(font.render(f"B={zone_b:.4f} (thresh {ZONE_B_THRESHOLD})",
                True, b_color), (gap_r_px + 4, 20))
    screen.blit(font.render(f"C={zone_c:.4f} (thresh {ZONE_C_THRESHOLD})",
                True, c_color), (gap_r_px + 4, mid_px + 20))

    # Status
    if active_signal == "politics":
        status = "HORIZONTAL BAR → should reach Zone B"
        sc = (100, 160, 255)
    elif active_signal == "climate":
        status = "VERTICAL BAR → should reach Zone C"
        sc = (100, 255, 160)
    else:
        status = "P=politics  C=climate  R=reset  Q=quit"
        sc = (160, 160, 160)
    screen.blit(font.render(f"step={step_count}  {status}", True, sc),
                (4, TRAIN_H*SCALE - 18))

    pygame.display.flip()


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
    pygame.display.set_caption("Routing NCA Test")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 12)

    rng  = np.random.default_rng(42)
    grid = make_initial_grid(rng)
    grid_j = jnp.array(grid)

    step_count    = 0
    active_signal = None
    hold_steps    = 0
    HOLD_WINDOW   = 300
    key           = jax.random.PRNGKey(0)

    print("Ready.")
    print("  P = inject horizontal bar (politics → Zone B)")
    print("  C = inject vertical bar   (climate  → Zone C)")
    print("  R = reset | Q = quit")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    grid   = make_initial_grid(rng)
                    grid_j = jnp.array(grid)
                    step_count    = 0
                    active_signal = None
                    print("  --- RESET ---")
                if event.key == pygame.K_p and not active_signal:
                    active_signal = "politics"
                    hold_steps    = HOLD_WINDOW
                    print(f"  step {step_count}: HORIZONTAL bar → watching Zone B ({HOLD_WINDOW} steps)")
                if event.key == pygame.K_c and not active_signal:
                    active_signal = "climate"
                    hold_steps    = HOLD_WINDOW
                    print(f"  step {step_count}: VERTICAL bar → watching Zone C ({HOLD_WINDOW} steps)")

        # Re-inject signal every step during hold window
        if hold_steps > 0:
            trail     = np.array(grid_j[:, :, CH_TRAIL])
            trail_sim = zoom(trail, (SIM_H/TRAIN_H, SIM_W/TRAIN_W), order=1)
            trail_sim = inject_zone_a_signal_only(trail_sim, active_signal)
            trail_64  = zoom(trail_sim, (TRAIN_H/SIM_H, TRAIN_W/SIM_W), order=1)
            grid_j    = grid_j.at[:, :, CH_TRAIL].set(
                jnp.clip(jnp.array(trail_64), 0, 1))
            sig_val   = 0.5 if active_signal == "politics" else 1.0
            grid_j    = grid_j.at[:, :, CH_SIGNAL].set(sig_val)
            hold_steps -= 1
            if hold_steps == 0:
                print(f"  step {step_count}: signal hold ended — watching for {active_signal} routing")
                active_signal = None

        # NCA step
        grid_j, key = nca_step(grid_j, params, update_net, perception_kernel, key)
        grid_j = grid_j.at[:, :, CH_ZONE].set(ZONE_MAP_64_JAX)
        grid_j = grid_j.at[:, :, CH_PHYSICS].set(0.5)
        step_count += 1

        # Measure zones
        trail  = np.array(grid_j[:, :, CH_TRAIL])
        mid    = TRAIN_H // 2
        gap_r  = int(GAP_RIGHT * TRAIN_W / SIM_W)
        zone_b = float(trail[:mid, gap_r:].mean())
        zone_c = float(trail[mid:, gap_r:].mean())

        if step_count % 50 == 0:
            print(f"  step {step_count:5d}  B={zone_b:.4f}  C={zone_c:.4f}  "
                  f"signal={active_signal or 'none'}")

        render(screen, np.array(grid_j), step_count, font,
               active_signal, zone_b, zone_c)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
