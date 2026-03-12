"""
test_nca.py — Single-window NCA viewer.

Runs one piece from scratch: random params, init grid, grow forever.
Press R to randomize a new piece. Press Q or close window to quit.
Press SPACE to pause/unpause.

This is a throwaway test script — not part of the final screensaver.
It just lets us verify the NCA engine works and looks interesting
before we build the two-window display and stage system on top of it.
"""

import sys
import numpy as np
import pygame
from jax import random

# --- import our modules ---
from nca.model  import init_grid, make_perception_kernel, UpdateNet, make_step_fn
from nca.params import sample_params, init_net_params
from nca.rulesets import apply_ruleset

# --- display config ---
WINDOW_W = 768       # window width in pixels
WINDOW_H = 768       # window height in pixels
GRID_H   = 256       # NCA grid height (upscaled to window)
GRID_W   = 256       # NCA grid width

# --- test speed config ---
# Run this many NCA steps per display frame.
# 1 = slow/real-time feel. 10+ = fast for testing.
STEPS_PER_FRAME = 50
TEST_STEP_SIZE  = 0.7   # slightly lower step size balances the extra steps — less saturation

def grid_to_surface(grid, surface):
    """
    Convert the NCA grid's first 3 channels (RGB) to a pygame surface.

    grid shape: (H, W, N_CHANNELS)
    We take channels 0, 1, 2 as R, G, B and scale 0-1 float to 0-255 uint8.
    Then upscale from GRID_H x GRID_W to WINDOW_H x WINDOW_W.
    """
    # Extract RGB from first 3 channels, convert to 0-255 uint8
    rgb = np.array(grid[:, :, :3])          # (H, W, 3), float 0-1
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    # pygame wants (W, H, 3) — so transpose H and W axes
    rgb = rgb.transpose(1, 0, 2)            # (W, H, 3)

    # Blit into a small surface then scale up
    small = pygame.surfarray.make_surface(rgb)
    pygame.transform.scale(small, (WINDOW_W, WINDOW_H), surface)


def new_piece(seed=None):
    """
    Roll random params, initialize everything, return ready-to-run state.
    Returns (grid, params_dict, step_fn, nca_params, jax_key).
    """
    nca_params = sample_params(seed)
    print(f"\n--- New piece ---")
    print(f"  Subject:   {nca_params.subject} ({nca_params.category})")
    print(f"  Aesthetic: {nca_params.aesthetic}")
    print(f"  Palette:   {nca_params.palette_name}")
    print(f"  Mood:      {nca_params.mood}")
    print(f"  Grid:      {nca_params.grid_size}")
    print(f"  w_scale:   {nca_params.weight_scale:.3f}  step: {nca_params.step_size:.3f}")
    print(f"  Seed:      {nca_params.seed}")

    jax_key = random.PRNGKey(nca_params.seed)

    # Build the NCA components
    grid             = init_grid(jax_key, GRID_H, GRID_W)
    perception_kernel = make_perception_kernel()
    net              = UpdateNet()

    # Initialize + shape network weights
    net_params, jax_key = init_net_params(nca_params, jax_key)
    net_params, jax_key = apply_ruleset(net_params, nca_params.category, jax_key)

    # Compile the step function (first call will take a few seconds — that's JIT)
    step_fn = make_step_fn(net, perception_kernel)

    return grid, net_params, step_fn, nca_params, jax_key


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Somnivex — NCA Test")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 14)

    # Kick off first piece
    grid, net_params, step_fn, nca_params, jax_key = new_piece()

    paused    = False
    step_count = 0
    jit_warned = False

    print("\nControls: R = new piece  |  SPACE = pause  |  Q = quit")
    print("(First step will be slow — JAX is compiling. Subsequent steps are fast.)\n")

    while True:
        # --- events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    grid, net_params, step_fn, nca_params, jax_key = new_piece()
                    step_count = 0
                    jit_warned = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")

        # --- NCA steps (multiple per frame for testing speed) ---
        if not paused:
            if step_count == 0 and not jit_warned:
                print("Running first step — JIT compiling, may take 5-15 seconds...")
                jit_warned = True

            for _ in range(STEPS_PER_FRAME):
                grid, jax_key = step_fn(grid, net_params, jax_key, TEST_STEP_SIZE)
            step_count += STEPS_PER_FRAME

            if step_count == STEPS_PER_FRAME:
                print("JIT done. Running.\n")

        # --- draw ---
        grid_to_surface(grid, screen)

        # HUD — subject + step count in corner
        label = font.render(
            f"{nca_params.subject} | {nca_params.aesthetic} | step {step_count}",
            True, (220, 220, 220)
        )
        screen.blit(label, (10, 10))

        if paused:
            p_label = font.render("PAUSED", True, (255, 200, 0))
            screen.blit(p_label, (10, 28))

        pygame.display.flip()
        clock.tick(30)   # cap at 30fps — NCA step is the real bottleneck anyway


if __name__ == "__main__":
    main()
