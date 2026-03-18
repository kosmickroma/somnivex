# 03_generate_training_data.py — Save Physarum frames for NCA training
#
# TYPE THIS OUT.
#
# What this file does:
#   Runs the Physarum simulation and saves frames as a numpy array.
#   Each frame is a (H, W) float32 array — the trail concentration at that step.
#   This is the format train_lenia.py already knows how to read from GS and Lenia.
#
# Output:
#   nca/physarum_training_data.npz — contains frames array + metadata
#
# Run it with:
#   source ~/ai-env/bin/activate
#   python kktodo/physarum_typing/03_generate_training_data.py
#
# This takes a few minutes. Go get a coffee.

import os
import numpy as np
from scipy.ndimage import uniform_filter


# ── Parameters ────────────────────────────────────────────────────────────────

H, W         = 64, 64      # match NCA training grid size
N_AGENTS     = 1000
DECAY        = 0.95
DIFFUSE_R    = 1
DEPOSIT      = 1.5
SENSOR_DIST  = 5
SENSOR_ANGLE = 0.4
ROTATE_ANGLE = 0.3

N_TRAJECTORIES = 200       # how many independent runs to generate
STEPS_PER_RUN  = 200       # steps per run (skip first 50 warmup, save last 150)
WARMUP_STEPS   = 50        # discard early steps before trails form

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'nca', 'physarum_training_data.npz'
)


# ── Vectorized step (same as 02_visualize.py but no display) ──────────────────

def sample_all(trail, x, y):
    return trail[x.astype(int) % H, y.astype(int) % W]

def physarum_step(trail, agent_x, agent_y, agent_heading, rng):
    s_l = agent_heading - SENSOR_ANGLE
    s_c = agent_heading
    s_r = agent_heading + SENSOR_ANGLE

    L = sample_all(trail, agent_x + SENSOR_DIST * np.cos(s_l),
                          agent_y + SENSOR_DIST * np.sin(s_l))
    C = sample_all(trail, agent_x + SENSOR_DIST * np.cos(s_c),
                          agent_y + SENSOR_DIST * np.sin(s_c))
    R = sample_all(trail, agent_x + SENSOR_DIST * np.cos(s_r),
                          agent_y + SENSOR_DIST * np.sin(s_r))

    rotate = np.where(C >= np.maximum(L, R), 0.0,
             np.where(L >= R, -ROTATE_ANGLE, ROTATE_ANGLE))
    tie = (L == R) & (C < L)
    if tie.any():
        rotate[tie] = rng.choice([-ROTATE_ANGLE, ROTATE_ANGLE], size=tie.sum())

    agent_heading = (agent_heading + rotate) % (2 * np.pi)
    agent_x = (agent_x + np.cos(agent_heading)) % H
    agent_y = (agent_y + np.sin(agent_heading)) % W

    xi = agent_x.astype(int) % H
    yi = agent_y.astype(int) % W
    np.add.at(trail, (xi, yi), DEPOSIT)

    trail = uniform_filter(trail, size=2 * DIFFUSE_R + 1, mode='wrap')
    return np.clip(trail * DECAY, 0.0, 1.0), agent_x, agent_y, agent_heading


# ── Generate trajectories ──────────────────────────────────────────────────────
# A "trajectory" is a sequence of (trail_t, trail_t+1) pairs.
# The NCA will learn: given trail at step t, predict trail at step t+1.
# Same supervision structure as GS (given A_t, B_t → predict A_t+1, B_t+1).

print(f"Generating {N_TRAJECTORIES} trajectories x {STEPS_PER_RUN} steps at {H}x{W}...")
print(f"Output: {OUTPUT_PATH}")
print()

all_frames = []   # will be shape (N_TRAJECTORIES * STEPS_PER_RUN, H, W)

for traj in range(N_TRAJECTORIES):
    rng = np.random.default_rng(traj)   # different seed per trajectory

    # Init trail (empty) and agents (random circle start)
    trail = np.zeros((H, W), dtype=np.float32)
    angles  = rng.uniform(0, 2 * np.pi, N_AGENTS)
    radii   = rng.uniform(0, H * 0.15, N_AGENTS)
    ax = (H // 2 + radii * np.cos(angles)).astype(np.float32)
    ay = (W // 2 + radii * np.sin(angles)).astype(np.float32)
    ah = rng.uniform(0, 2 * np.pi, N_AGENTS).astype(np.float32)

    # Warmup — let trails form before we start saving
    for _ in range(WARMUP_STEPS):
        trail, ax, ay, ah = physarum_step(trail, ax, ay, ah, rng)

    # Save STEPS_PER_RUN frames
    traj_frames = []
    for _ in range(STEPS_PER_RUN):
        trail, ax, ay, ah = physarum_step(trail, ax, ay, ah, rng)
        traj_frames.append(trail.copy())

    all_frames.extend(traj_frames)

    if (traj + 1) % 20 == 0:
        print(f"  trajectory {traj+1}/{N_TRAJECTORIES}")

frames = np.stack(all_frames, axis=0)   # shape: (N_TRAJ * STEPS, H, W)
print(f"\nFrames array shape: {frames.shape}")
print(f"Value range: min={frames.min():.4f}  max={frames.max():.4f}  mean={frames.mean():.4f}")

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.savez(OUTPUT_PATH, frames=frames, H=H, W=W,
         decay=DECAY, sensor_dist=SENSOR_DIST)
print(f"\nSaved to: {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")
print()
print("What's in the file:")
print("  frames — (N, H, W) float32 array of trail concentrations")
print("  Each consecutive pair (frames[i], frames[i+1]) is one training example.")
print("  The NCA learns: given this trail map, predict the next one.")
print()
print("Next: read 04_what_next.md to see how this plugs into train_lenia.py")
