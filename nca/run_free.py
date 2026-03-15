# nca/run_free.py - Run the trained NCA freely, no GS needed. 
#
# Loads a trained checkpoint and runs the NCA on its own output forever,
# This is the test: did it learn enough to sustain itself?
#
# Run from the project root with:
#     python nca/run_free.py
#
# Controls: R=reset F=cycle f value K=cycle k value P=palette Q=quit

import os
import sys
import pickle
import numpy as np
import pygame
import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca.model import(
    N_CHANNELS, N_FILTERS,
    CH_A, CH_B, CH_F, CH_K,
    UpdateNet, make_perception_kernel, make_step_fn,
)
from gs.engine import GS_REGIMES
from nca.params import PALETTES

# ── Config ────────────────────────────────────────────────────────────────────
# Path to the checkpoint we want to load.
# Change this to any checkpoint in nca/checkpoints/ to compare stages,
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'params_050000.pkl'
)

GRID_H = 256 # run at full resolution - trained at 64x64 but works at any size
GRID_W = 256 # because all ops are local 3x3 convolutions
DISPLAY_W = 1920
DISPLAY_H = 1080 # upscale to fill screen
FPS = 30

# ── Grid initialization ───────────────────────────────────────────────────────
def init_nca_grid(key, H, W, f, k):
    """
    Build a fresh 256x256 NCA grid.
    
    Channels 0 (A) and 1 (B) are seeded like a GS grid - A=1 everywhere
    with tiny noise, B=0 with a few random patches to spark reactions.
    Channels 2-13 are zero (hidden state, NCA will fill these in).
    Channels 14 (f) and 15 (k) are set to the current control values.
    
    This gives the NCA a familiar starting state - the same kind of state
    it saw thousands of times during training.
    """
    grid = jnp.zeros((H, W, N_CHANNELS))

    # Seed A channel: 1.0 everywhere with tiny noise (GS layer)
    key, sk = random.split(key)
    A = jnp.ones((H, W)) - random.uniform(sk, (H, W)) * 0.04
    grid = grid.at[:, :, CH_A].set(A)

    # Seed B channel: random patches (the "spark" that starts the reaction)
    key, sk = random.split(key)
    n_patches = int(random.randint(sk, (), 8, 24))
    B = jnp.zeros((H, W))
    for _ in range(n_patches):
        key, sk1, sk2, sk3 = random.split(key, 4)
        y = int(random.randint(sk1, (), 0, H - 12))
        x = int(random.randint(sk2, (), 0, W- 12))
        sz = int(random.randint(sk3, (), 4, 12))
        B = B.at[y:y+sz, x:x+sz].set(0.25 + np.random.uniform(0, 0.1))
    grid = grid.at[:, :, CH_B].set(B)

    # Write control channels - same value broadcast to every cell
    grid = grid.at[:, :, CH_F].set(f)
    grid = grid.at[:, :, CH_K].set(k)

    return grid, key

# ── Rendering ─────────────────────────────────────────────────────────────────
def render(surface, grid, palette):
    """
    Map the B channel (channe 1) to color using the current palette.

    B is the "predator" chemical - the one that forms the visible patterns.
    Values run 0.0 to 1.0. We map that range through 4 palette colors
    using linear interpolation - same as the GS screensaver does.

    palette: list of 4 (R, G, B) tuples
    """
    B = np.array(grid[:, :, CH_B]) # (H, W) values in [0, 1]

    # Map B values to palette colors via lerp across 4 color stops
    # t=0.0 -> color[0], t=0.33 -> color[1], t=0.67 -> color[2], t=1.0 -> color[3]
    p = np.array(palette, dtype=np.float32) / 255.0 # (4, 3) normalized
    t = np.clip(B * 3.0, 0.0, 3.0) # scale to [0,3] for 4-stop lerp
    idx = np.floor(t).astype(int).clip(0, 2) # which segment [0,1,2]
    frac = (t - idx)[..., None] # fractional position in segment

    # Gather the two palette colors for each pixel's segment
    c0 = p[idx]       # (H, W, 3) — color at start of segment
    c1 = p[idx + 1]   # (H, W, 3) — color at end of segment

    # Linear interpolate and convert to uint8
    rgb = (c0 + frac * (c1 - c0))
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)   # (H, W, 3)

    # Blit to pygame surface — scale up to display resolution
    img = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    scaled = pygame.transform.scale(img, (DISPLAY_W, DISPLAY_H))
    surface.blit(scaled, (0, 0))

    # ── Main ──────────────────────────────────────────────────────────────────────
def run():
    # ── Load checkpoint ───────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Run nca/train.py first.")
        sys.exit(1)

    print(f"Loading checkpoint: {CHECKPOINT}")
    with open(CHECKPOINT, 'rb') as f:
        params = pickle.load(f)
    params = jax.device_put(params)   # move to GPU
    print("Loaded.")

    # ── Build model ───────────────────────────────────────────────────────
    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn           = make_step_fn(update_net, perception_kernel)
    # step_fn(grid, params, key) → (new_grid, new_key)
    # JIT-compiled, runs on GPU, reused every frame

    # ── Starting regime ───────────────────────────────────────────────────
    # Pick a starting GS regime for the control channels.
    # The NCA learned what to do with these values — let's see if it remembers.
    regime_names = list(GS_REGIMES.keys())
    regime_idx   = 0
    f, k         = GS_REGIMES[regime_names[regime_idx]]

    # ── Starting palette ──────────────────────────────────────────────────
    palette_names = list(PALETTES.keys())
    palette_idx   = 0
    palette       = PALETTES[palette_names[palette_idx]]

    # ── Init grid and JAX key ─────────────────────────────────────────────
    key = random.PRNGKey(42)
    grid, key = init_nca_grid(key, GRID_H, GRID_W, f, k)

    # ── Pygame setup ──────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.NOFRAME)
    pygame.display.set_caption("Somnivex — NCA Free Run")
    clock  = pygame.font.SysFont("monospace", 16)
    font   = pygame.font.SysFont("monospace", 16)
    ticker = pygame.time.Clock()

    print(f"\nRunning NCA free — no Gray-Scott.")
    print(f"Starting regime: {regime_names[regime_idx]}  f={f}  k={k}")
    print(f"Controls: R=reset  F=next regime  K=same regime new seed  P=palette  Q=quit\n")

    step_count = 0
    running    = True

    while running:

        # ── Events ────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_q:
                    running = False

                if event.key == pygame.K_r:
                    # Full reset — new random seed, same regime
                    key, sk = random.split(key)
                    grid, key = init_nca_grid(sk, GRID_H, GRID_W, f, k)
                    step_count = 0
                    print(f"Reset. Regime: {regime_names[regime_idx]}")

                if event.key == pygame.K_f:
                    # Cycle to next GS regime — changes f and k control channels
                    # This is the key test: does the NCA respond to control channel changes?
                    regime_idx = (regime_idx + 1) % len(regime_names)
                    f, k = GS_REGIMES[regime_names[regime_idx]]
                    # Write new f/k into every cell's control channels mid-run
                    grid = grid.at[:, :, CH_F].set(f)
                    grid = grid.at[:, :, CH_K].set(k)
                    print(f"Regime: {regime_names[regime_idx]}  f={f:.4f}  k={k:.4f}")

                if event.key == pygame.K_k:
                    # Keep regime, reset grid with new seed
                    key, sk = random.split(key)
                    grid, key = init_nca_grid(sk, GRID_H, GRID_W, f, k)
                    step_count = 0
                    print(f"New seed. Regime: {regime_names[regime_idx]}")

                if event.key == pygame.K_p:
                    # Cycle to next color palette
                    palette_idx = (palette_idx + 1) % len(palette_names)
                    palette = PALETTES[palette_names[palette_idx]]
                    print(f"Palette: {palette_names[palette_idx]}")

        # ── NCA step ──────────────────────────────────────────────────────
        # Run one NCA step. The NCA reads f/k from channels 14/15 of the grid
        # and uses what it learned during training to update channels 0 and 1.
        grid, key = step_fn(grid, params, key)
        step_count += 1

        # ── Render ────────────────────────────────────────────────────────
        render(screen, grid, palette)

        # HUD — small text overlay so you can see what's running
        regime_str  = regime_names[regime_idx]
        palette_str = palette_names[palette_idx]
        hud = font.render(
            f"step {step_count}  |  {regime_str}  f={f:.4f} k={k:.4f}  |  {palette_str}  |  R=reset F=regime P=palette Q=quit",
            True, (80, 80, 80)
        )
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        ticker.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    print(f"JAX devices: {jax.devices()}")
    run()
