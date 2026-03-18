# 01_trail_map.py — The Physarum Trail Map
#
# TYPE THIS OUT. Don't copy-paste.
#
# What this file does:
#   Implements the core Physarum field dynamics — diffuse, decay, deposit.
#   No agents needed. Just the concentration field updating itself each step.
#
# The three operations:
#   DIFFUSE  — pheromone spreads to neighboring cells (convolution with a blur kernel)
#   DECAY    — pheromone evaporates over time (multiply by a number less than 1)
#   DEPOSIT  — organisms add pheromone where they are (add to specific cells)
#
# Run it with:
#   source ~/ai-env/bin/activate
#   python kktodo/physarum_typing/01_trail_map.py

import numpy as np
from scipy.ndimage import uniform_filter   # this is the blur/diffuse operation


# ── Parameters ────────────────────────────────────────────────────────────────
# These numbers control what kind of patterns form.
# Try changing them after you get it running.

H, W        = 128, 128    # grid size (start small, scale up later)
N_AGENTS    = 2000         # number of virtual organisms depositing trail
DECAY       = 0.92         # how much trail survives each step (0.9 = 10% evaporates)
DIFFUSE_R   = 1            # blur radius for spreading (1 = 3x3 neighborhood)
DEPOSIT     = 2.0          # how much trail each agent deposits per step
SENSOR_DIST = 5            # how far ahead agents sense (in cells)
SENSOR_ANGLE = 0.4         # angle between the three sensors (radians, ~23 degrees)
ROTATE_ANGLE = 0.3         # how much agent rotates per step (radians)
STEPS       = 500          # how many steps to run


# ── Initialize the trail map ───────────────────────────────────────────────────
# The trail map is just a 2D array of floats.
# Starts at zero everywhere — no pheromone yet.

trail = np.zeros((H, W), dtype=np.float32)


# ── Initialize agents ─────────────────────────────────────────────────────────
# Each agent has: x position, y position, heading angle (radians)
# They start clustered in the center, pointing in random directions.

rng = np.random.default_rng(42)

# Random positions in a circle at the center
angles_init = rng.uniform(0, 2 * np.pi, N_AGENTS)
radii_init  = rng.uniform(0, H * 0.15, N_AGENTS)   # within 15% of grid size from center
agent_x = (H // 2 + radii_init * np.cos(angles_init)).astype(np.float32)
agent_y = (W // 2 + radii_init * np.sin(angles_init)).astype(np.float32)
agent_heading = rng.uniform(0, 2 * np.pi, N_AGENTS).astype(np.float32)


# ── Helper: sample trail at a position (with wrap-around) ─────────────────────
# This is how an agent "smells" the pheromone at any position.
# np.clip keeps us inside the grid. int() converts float position to grid index.

def sample_trail(trail, x, y):
    # Wrap around the grid edges (toroidal boundary, same as the NCA)
    xi = int(x) % H
    yi = int(y) % W
    return trail[xi, yi]


# ── One simulation step ────────────────────────────────────────────────────────
# This is the core of the Physarum model.
# Call this once per frame.

def physarum_step(trail, agent_x, agent_y, agent_heading, rng):

    # ── AGENT SENSING ────────────────────────────────────────────────────
    # Each agent has three sensors: front-left, front-center, front-right.
    # The agent samples the trail at each sensor position.
    # It then turns toward whichever sensor reads the highest concentration.

    # Sensor positions (offset from agent center by SENSOR_DIST in each direction)
    s_left   = agent_heading - SENSOR_ANGLE
    s_center = agent_heading
    s_right  = agent_heading + SENSOR_ANGLE

    # Compute where each sensor is pointing (x, y offsets from agent)
    lx = agent_x + SENSOR_DIST * np.cos(s_left)
    ly = agent_y + SENSOR_DIST * np.sin(s_left)
    cx = agent_x + SENSOR_DIST * np.cos(s_center)
    cy = agent_y + SENSOR_DIST * np.sin(s_center)
    rx = agent_x + SENSOR_DIST * np.cos(s_right)
    ry = agent_y + SENSOR_DIST * np.sin(s_right)

    # Sample the trail at each sensor (vectorized over all agents)
    # Note: this loop is slow. We'll vectorize it properly later if needed.
    left_vals   = np.array([sample_trail(trail, lx[i], ly[i]) for i in range(N_AGENTS)])
    center_vals = np.array([sample_trail(trail, cx[i], cy[i]) for i in range(N_AGENTS)])
    right_vals  = np.array([sample_trail(trail, rx[i], ry[i]) for i in range(N_AGENTS)])

    # ── AGENT ROTATION ────────────────────────────────────────────────────
    # If center is highest: go straight (no rotation)
    # If left is highest:   rotate left
    # If right is highest:  rotate right
    # If left == right:     random rotation (breaks symmetry)

    rotate = np.zeros(N_AGENTS, dtype=np.float32)

    # Center is highest — stay straight
    straight = (center_vals >= left_vals) & (center_vals >= right_vals)
    rotate[straight] = 0.0

    # Left is highest — rotate left (negative angle)
    go_left = (left_vals > center_vals) & (left_vals >= right_vals)
    rotate[go_left] = -ROTATE_ANGLE

    # Right is highest — rotate right
    go_right = (right_vals > center_vals) & (right_vals > left_vals)
    rotate[go_right] = ROTATE_ANGLE

    # Tie between left and right — random
    tied = (left_vals == right_vals) & ~straight
    rotate[tied] = rng.choice([-ROTATE_ANGLE, ROTATE_ANGLE], size=tied.sum())

    agent_heading += rotate

    # ── AGENT MOVEMENT ────────────────────────────────────────────────────
    # Each agent moves 1 step forward in its heading direction.
    # Wrap around the grid edges.

    agent_x = (agent_x + np.cos(agent_heading)) % H
    agent_y = (agent_y + np.sin(agent_heading)) % W

    # ── DEPOSIT ───────────────────────────────────────────────────────────
    # Each agent deposits trail at its new position.
    # We use np.add.at for scatter-add (multiple agents can hit the same cell).

    xi = agent_x.astype(int) % H
    yi = agent_y.astype(int) % W
    np.add.at(trail, (xi, yi), DEPOSIT)

    # ── DIFFUSE ───────────────────────────────────────────────────────────
    # Blur the trail map so pheromone spreads to neighbors.
    # uniform_filter is a box blur — same as convolution with a flat kernel.
    # This is the equivalent of the Laplacian diffusion in Gray-Scott.

    trail = uniform_filter(trail, size=2 * DIFFUSE_R + 1, mode='wrap')

    # ── DECAY ─────────────────────────────────────────────────────────────
    # Pheromone evaporates. Each step, every cell keeps DECAY fraction of its value.
    # DECAY=0.92 means 8% evaporates per step. After 100 steps: 0.92^100 ≈ 0.00025.
    # Trails that agents stop reinforcing fade away in ~50-100 steps.

    trail *= DECAY

    # Clip to [0, 1] to avoid runaway values
    trail = np.clip(trail, 0.0, 1.0)

    return trail, agent_x, agent_y, agent_heading


# ── Run it ────────────────────────────────────────────────────────────────────

print(f"Running Physarum: {H}x{W} grid, {N_AGENTS} agents, {STEPS} steps")

for step in range(STEPS):
    trail, agent_x, agent_y, agent_heading = physarum_step(
        trail, agent_x, agent_y, agent_heading, rng
    )
    if step % 100 == 0:
        print(f"  step {step:4d}  trail max={trail.max():.3f}  mean={trail.mean():.4f}")

print("Done.")
print(f"Final trail: max={trail.max():.3f}  mean={trail.mean():.4f}  std={trail.std():.4f}")
print()
print("If mean > 0 and std > 0.01, the trails are forming.")
print("Next: open 02_visualize.py and watch it run.")
