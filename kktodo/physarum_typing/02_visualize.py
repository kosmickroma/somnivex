# 02_visualize.py — Watch the Physarum run in real time
#
# TYPE THIS OUT.
#
# What this file does:
#   Same simulation as 01_trail_map.py but with a pygame window.
#   The trail map is displayed as a grayscale image (bright = high concentration).
#   Press Q to quit.
#
# What you should see:
#   - First few hundred steps: random scattered dots (agents depositing randomly)
#   - Around step 200-500: trails start connecting (agents following each other)
#   - After step 500+: branching network forms — thin bright lines connecting clusters
#   - Eventually: a stable spanning tree (the Physarum network that Adamatzky studied)
#
# This is the pattern the NCA will learn to predict.
#
# Run it with:
#   source ~/ai-env/bin/activate
#   python kktodo/physarum_typing/02_visualize.py

import numpy as np
import pygame
from scipy.ndimage import uniform_filter


# ── Parameters (same as 01, feel free to change) ──────────────────────────────

H, W         = 256, 256
N_AGENTS     = 5000
DECAY        = 0.95
DIFFUSE_R    = 1
DEPOSIT      = 1.5
SENSOR_DIST  = 9
SENSOR_ANGLE = 0.4
ROTATE_ANGLE = 0.3

# Display
SCALE = 3    # each cell = SCALE x SCALE pixels (256 * 3 = 768px window)
FPS   = 60


# ── Init ──────────────────────────────────────────────────────────────────────

rng   = np.random.default_rng(42)
trail = np.zeros((H, W), dtype=np.float32)

# Agents start in a circle at center
angles_init  = rng.uniform(0, 2 * np.pi, N_AGENTS)
radii_init   = rng.uniform(0, H * 0.1, N_AGENTS)
agent_x      = (H // 2 + radii_init * np.cos(angles_init)).astype(np.float32)
agent_y      = (W // 2 + radii_init * np.sin(angles_init)).astype(np.float32)
agent_heading = rng.uniform(0, 2 * np.pi, N_AGENTS).astype(np.float32)


# ── Vectorized sensor sampling ─────────────────────────────────────────────────
# Faster version of sample_trail — does all N_AGENTS at once using array indexing.
# This is the same idea as the NCA's perception kernel — parallel lookup.

def sample_all(trail, x, y):
    xi = x.astype(int) % H
    yi = y.astype(int) % W
    return trail[xi, yi]    # returns array of shape (N_AGENTS,)


# ── Step function ─────────────────────────────────────────────────────────────

def step(trail, agent_x, agent_y, agent_heading, rng):
    s_left   = agent_heading - SENSOR_ANGLE
    s_center = agent_heading
    s_right  = agent_heading + SENSOR_ANGLE

    lx = agent_x + SENSOR_DIST * np.cos(s_left)
    ly = agent_y + SENSOR_DIST * np.sin(s_left)
    cx = agent_x + SENSOR_DIST * np.cos(s_center)
    cy = agent_y + SENSOR_DIST * np.sin(s_center)
    rx = agent_x + SENSOR_DIST * np.cos(s_right)
    ry = agent_y + SENSOR_DIST * np.sin(s_right)

    # Vectorized sensor reads (fast — no Python loop)
    L = sample_all(trail, lx, ly)
    C = sample_all(trail, cx, cy)
    R = sample_all(trail, rx, ry)

    # Rotation decision (vectorized)
    rotate = np.where(C >= np.maximum(L, R), 0.0,
             np.where(L >= R, -ROTATE_ANGLE, ROTATE_ANGLE))

    # Random for ties
    tie_mask = (L == R) & (C < L)
    if tie_mask.any():
        rotate[tie_mask] = rng.choice([-ROTATE_ANGLE, ROTATE_ANGLE], size=tie_mask.sum())

    agent_heading = (agent_heading + rotate) % (2 * np.pi)

    # Move
    agent_x = (agent_x + np.cos(agent_heading)) % H
    agent_y = (agent_y + np.sin(agent_heading)) % W

    # Deposit
    xi = agent_x.astype(int) % H
    yi = agent_y.astype(int) % W
    np.add.at(trail, (xi, yi), DEPOSIT)

    # Diffuse + decay
    trail = uniform_filter(trail, size=2 * DIFFUSE_R + 1, mode='wrap')
    trail = np.clip(trail * DECAY, 0.0, 1.0)

    return trail, agent_x, agent_y, agent_heading


# ── Pygame display ─────────────────────────────────────────────────────────────

pygame.init()
screen = pygame.display.set_mode((W * SCALE, H * SCALE))
pygame.display.set_caption("Physarum Trail Simulation")
clock  = pygame.time.Clock()
font   = pygame.font.SysFont('monospace', 14)

step_count = 0
running    = True

print("Running — press Q to quit")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False

    # Simulate
    trail, agent_x, agent_y, agent_heading = step(
        trail, agent_x, agent_y, agent_heading, rng
    )
    step_count += 1

    # Render trail map as grayscale
    # trail is float32 in [0,1]. Convert to uint8 [0,255] for display.
    display_arr = (trail * 255).astype(np.uint8)

    # Scale up: repeat each cell SCALE times in both dimensions
    display_arr = np.repeat(np.repeat(display_arr, SCALE, axis=0), SCALE, axis=1)

    # pygame wants (W, H, 3) RGB. We have (H*SCALE, W*SCALE) grayscale.
    # Stack grayscale into RGB by repeating it 3 times.
    rgb = np.stack([display_arr, display_arr, display_arr], axis=-1)

    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))

    # HUD
    txt = font.render(
        f"step={step_count}  max={trail.max():.3f}  mean={trail.mean():.4f}",
        True, (0, 255, 0)
    )
    screen.blit(txt, (5, 5))
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
print(f"Stopped at step {step_count}")
print()
print("What you saw:")
print("  - Early steps: scattered random deposits")
print("  - Mid steps: trails starting to connect (positive feedback loop)")
print("  - Late steps: stable branching network — the Physarum spanning tree")
print()
print("This trail map, recorded at each step, is the teacher signal for the NCA.")
print("Next: open 03_generate_training_data.py")
