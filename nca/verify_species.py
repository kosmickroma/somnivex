#!/usr/bin/env python3
"""
verify_species.py — Watch each Lenia species swim to confirm proof of life.

Shows all three creatures side by side in a pygame window.
Each panel runs its own independent simulation.
Press Q to quit, Space to reset all.

Usage:
    python nca/verify_species.py
"""

import os, sys
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca.lenia import (
    LENIA_CREATURES, CREATURE_NAMES, CREATURE_SEEDS,
    make_kernel_fft, lenia_step_np, seed_creature, LENIA_R,
)

GRID_H = 128
GRID_W = 128
PANEL_W = 384
PANEL_H = 384
MARGIN  = 8
FPS     = 30
STEPS_PER_FRAME = 4

def make_panel_surface(A):
    """Convert activation field to RGB surface."""
    img = np.clip(A * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([img, (img * 0.6).astype(np.uint8), (img * 0.2).astype(np.uint8)], axis=-1)
    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (PANEL_W, PANEL_H))

def init_creature(name, fK):
    rng = np.random.default_rng()
    A = seed_creature(GRID_H, GRID_W, rng, name)
    # Warm up so it's already moving when we start watching
    c = LENIA_CREATURES[name]
    for _ in range(200):
        A = lenia_step_np(A, fK, c['mu'], c['sigma'])
    return A

def main():
    pygame.init()
    fK = make_kernel_fft(LENIA_R, GRID_H, GRID_W)

    n = len(CREATURE_NAMES)
    win_w = n * PANEL_W + (n + 1) * MARGIN
    win_h = PANEL_H + MARGIN * 3 + 40
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Lenia Species Verification — Q=quit  Space=reset")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    # Init all species
    states = {name: init_creature(name, fK) for name in CREATURE_NAMES}
    steps  = {name: 200 for name in CREATURE_NAMES}

    print("Species verification running.")
    for name in CREATURE_NAMES:
        c = LENIA_CREATURES[name]
        seed_shape = CREATURE_SEEDS[name].shape
        print(f"  {name:12s}: mu={c['mu']:.3f}  sigma={c['sigma']:.4f}  "
              f"seed={seed_shape[0]}x{seed_shape[1]}")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    states = {name: init_creature(name, fK) for name in CREATURE_NAMES}
                    steps  = {name: 200 for name in CREATURE_NAMES}
                    print("Reset.")

        # Step each species
        for name in CREATURE_NAMES:
            c = LENIA_CREATURES[name]
            for _ in range(STEPS_PER_FRAME):
                states[name] = lenia_step_np(states[name], fK, c['mu'], c['sigma'])
            steps[name] += STEPS_PER_FRAME

        # Draw
        screen.fill((10, 10, 15))
        for i, name in enumerate(CREATURE_NAMES):
            x = MARGIN + i * (PANEL_W + MARGIN)
            y = MARGIN
            surf = make_panel_surface(states[name])
            screen.blit(surf, (x, y))

            # Label
            c = LENIA_CREATURES[name]
            label = f"{name}  mu={c['mu']:.3f} s={c['sigma']:.4f}  step={steps[name]}"
            txt = font.render(label, True, (200, 200, 200))
            screen.blit(txt, (x, y + PANEL_H + MARGIN))

            # Activity indicator
            activity = float(states[name].mean())
            color = (0, 220, 80) if activity > 0.005 else (220, 60, 60)
            status = "ALIVE" if activity > 0.005 else "DEAD"
            stxt = font.render(status, True, color)
            screen.blit(stxt, (x + PANEL_W - 60, y + PANEL_H + MARGIN))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()
