"""
Project Mycelium — Baseline Tests
Run from the somnivex/nca/ directory with the ai-env active.

Test 1: ch5 delta inspection
  Does the NCA produce nonzero ch5 deltas on its own?
  Near-zero = never learned to write ch5.

Test 2: Natural bridging test
  Seed two ch5 blobs 136px apart, run 3000 steps headlessly.
  Measure whether ch5 appears in the gap without painting it there.
  Run for both lenia_100000.pkl and physarum_100000.pkl.
"""

import os, sys, pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import UpdateNet, make_perception_kernel, nca_step, make_step_fn

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
GRID_H, GRID_W, N_CH = 256, 256, 16

def load_checkpoint(name):
    path = os.path.join(CHECKPOINT_DIR, name)
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return jax.device_put(params)

def make_blank_grid(key, f=0.046, k=0.065):
    """Blank grid with GS params set, A=1, B=0, everything else 0."""
    grid = np.zeros((GRID_H, GRID_W, N_CH), dtype=np.float32)
    grid[:, :, 0] = 1.0   # A = 1 (GS chemical A, full)
    grid[:, :, 1] = 0.0   # B = 0
    grid[:, :, 14] = f    # feed rate
    grid[:, :, 15] = k    # kill rate
    return jnp.array(grid)

def seed_ch5_blob(grid, cx, cy, radius=15, strength=0.8):
    """Paint a ch5 blob at (cx, cy). Returns updated grid."""
    grid_np = np.array(grid)
    for y in range(GRID_H):
        for x in range(GRID_W):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                grid_np[y, x, 5] = strength
    return jnp.array(grid_np)

def measure_ch5_in_gap(grid, cx1, cx2, cy, margin=20):
    """
    Measure mean ch5 in the gap region between two blobs.
    Gap = horizontal band between cx1+margin and cx2-margin at height cy±30.
    Returns: mean ch5 in gap, max ch5 in gap.
    """
    grid_np = np.array(grid)
    x_min = cx1 + margin
    x_max = cx2 - margin
    y_min = cy - 30
    y_max = cy + 30
    gap_region = grid_np[y_min:y_max, x_min:x_max, 5]
    return float(np.mean(gap_region)), float(np.max(gap_region))

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: ch5 delta inspection
# ─────────────────────────────────────────────────────────────────────────────

def test_ch5_delta(ckpt_name):
    print(f"\n{'═'*60}")
    print(f"TEST 1: ch5 delta inspection — {ckpt_name}")
    print(f"{'═'*60}")

    params = load_checkpoint(ckpt_name)
    update_net = UpdateNet()
    perception_kernel = make_perception_kernel()

    # Grid with ch5 = 0 everywhere
    key = random.PRNGKey(42)
    grid = make_blank_grid(key)

    from model import perceive
    perceived = perceive(grid, perception_kernel)
    delta = update_net.apply(params, perceived)

    ch5_delta = np.array(delta[:, :, 5])

    print(f"  ch5 delta — mean abs: {np.mean(np.abs(ch5_delta)):.6f}")
    print(f"  ch5 delta — max abs:  {np.max(np.abs(ch5_delta)):.6f}")
    print(f"  ch5 delta — std:      {np.std(ch5_delta):.6f}")

    if np.max(np.abs(ch5_delta)) < 0.001:
        print(f"  → NEAR ZERO. NCA has never learned to write ch5.")
        print(f"    Phase 1 training is required.")
    elif np.max(np.abs(ch5_delta)) < 0.01:
        print(f"  → VERY SMALL but nonzero. Trace ch5-writing behavior exists.")
        print(f"    May be trainable from here with small loss weight.")
    else:
        print(f"  → NONZERO. NCA already writes ch5 autonomously.")
        print(f"    Natural bridging test becomes very interesting.")

    # Also check all channels for comparison
    print(f"\n  All channel delta magnitudes (mean abs):")
    delta_np = np.array(delta)
    for ch in range(N_CH):
        mag = np.mean(np.abs(delta_np[:, :, ch]))
        bar = '█' * int(mag * 500)
        marker = ' ← ch5' if ch == 5 else ''
        print(f"    ch{ch:2d}: {mag:.6f}  {bar}{marker}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Natural bridging test
# ─────────────────────────────────────────────────────────────────────────────

def test_natural_bridging(ckpt_name, n_steps=3000, measure_every=500):
    print(f"\n{'═'*60}")
    print(f"TEST 2: Natural bridging — {ckpt_name}")
    print(f"{'═'*60}")

    params = load_checkpoint(ckpt_name)
    update_net = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn = make_step_fn(update_net, perception_kernel)

    key = random.PRNGKey(0)

    # Start from a blank grid
    grid = make_blank_grid(key)

    # Seed two ch5 blobs 136px apart, centered vertically
    cx1, cx2, cy = 60, 196, 128
    grid = seed_ch5_blob(grid, cx1, cy, radius=15, strength=0.8)
    grid = seed_ch5_blob(grid, cx2, cy, radius=15, strength=0.8)

    # Precompute blob mask for fast re-injection
    ys, xs = jnp.mgrid[0:GRID_H, 0:GRID_W]
    blob_mask = (((xs - cx1)**2 + (ys - cy)**2) <= 15**2) | \
                (((xs - cx2)**2 + (ys - cy)**2) <= 15**2)

    print(f"  Node A: ({cx1}, {cy})   Node B: ({cx2}, {cy})")
    print(f"  Gap: {cx2 - cx1 - 30}px (after subtracting blob radii)")
    print(f"  Running {n_steps} steps...")
    print(f"\n  {'Step':>6}  {'Gap mean ch5':>14}  {'Gap max ch5':>12}  {'Verdict'}")
    print(f"  {'─'*55}")

    gap_mean_0, gap_max_0 = measure_ch5_in_gap(grid, cx1, cx2, cy)
    print(f"  {'0':>6}  {gap_mean_0:>14.6f}  {gap_max_0:>12.6f}  (baseline)")

    bridging_detected = False

    for step in range(1, n_steps + 1):
        grid, key = step_fn(grid, params, key)

        # Re-inject ch5 at source blobs (simulating the trail clamp in run_free.py)
        grid = grid.at[:, :, 5].set(
            jnp.where(blob_mask, 0.8, grid[:, :, 5])
        )

        if step % measure_every == 0:
            gap_mean, gap_max = measure_ch5_in_gap(grid, cx1, cx2, cy)
            verdict = ''
            if gap_max > 0.3:
                verdict = '✓ STRONG BRIDGING'
                bridging_detected = True
            elif gap_max > 0.1:
                verdict = '~ weak signal'
            elif gap_max > 0.02:
                verdict = '· trace activity'
            else:
                verdict = '  nothing'
            print(f"  {step:>6}  {gap_mean:>14.6f}  {gap_max:>12.6f}  {verdict}")

    print(f"\n  {'─'*55}")
    if bridging_detected:
        print(f"  RESULT: Natural bridging detected. Phase 1 training may not be needed.")
        print(f"          The organism dynamics alone can connect nodes.")
    else:
        gap_mean_final, gap_max_final = measure_ch5_in_gap(grid, cx1, cx2, cy)
        if gap_max_final > 0.02:
            print(f"  RESULT: Trace activity in gap. Inconclusive.")
            print(f"          Run with more steps or check visually.")
        else:
            print(f"  RESULT: No bridging. ch5 does not propagate naturally.")
            print(f"          Phase 1 training is required.")

    return grid

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def test_obstacle(ckpt_name='lenia_100000.pkl', n_steps=3000, measure_every=500):
    """
    Place a vertical wall of high ch5 between two nodes.
    Measure ch5 ABOVE and BELOW the wall (routing around it) vs at the wall.
    If routing: ch5 appears in bypass lanes above/below.
    If diffusion only: ch5 piles at the wall and stops.
    """
    print(f"\n{'═'*60}")
    print(f"TEST 3: Obstacle routing — {ckpt_name}")
    print(f"{'═'*60}")

    params = load_checkpoint(ckpt_name)
    update_net = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn = make_step_fn(update_net, perception_kernel)

    key = random.PRNGKey(1)
    grid = make_blank_grid(key)

    cx1, cx2, cy = 60, 196, 128

    # Two nodes same as before
    grid = seed_ch5_blob(grid, cx1, cy, radius=15, strength=0.8)
    grid = seed_ch5_blob(grid, cx2, cy, radius=15, strength=0.8)

    # Vertical wall of ch5=0.8 down the center, blocking direct path
    # Wall: x=128, y=80 to y=176 (96px tall, leaves 52px gaps top/bottom)
    wall_x = 128
    wall_y_min, wall_y_max = 80, 176
    grid_np = np.array(grid)
    grid_np[wall_y_min:wall_y_max, wall_x-2:wall_x+3, 5] = 0.8
    grid = jnp.array(grid_np)

    # Precompute masks
    ys, xs = jnp.mgrid[0:GRID_H, 0:GRID_W]
    blob_mask = (((xs - cx1)**2 + (ys - cy)**2) <= 15**2) | \
                (((xs - cx2)**2 + (ys - cy)**2) <= 15**2)
    wall_mask = (xs >= wall_x-2) & (xs <= wall_x+2) & \
                (ys >= wall_y_min) & (ys < wall_y_max)

    print(f"  Nodes: ({cx1},{cy}) and ({cx2},{cy})")
    print(f"  Wall: x={wall_x}, y={wall_y_min}→{wall_y_max} (blocks center)")
    print(f"  Bypass lanes: y<{wall_y_min} (top) and y>{wall_y_max} (bottom)")
    print(f"\n  {'Step':>6}  {'Direct gap':>12}  {'Top bypass':>12}  {'Bot bypass':>12}  Verdict")
    print(f"  {'─'*65}")

    grid_np_init = np.array(grid)
    d0 = float(np.mean(grid_np_init[cy-10:cy+10, cx1+20:cx2-20, 5]))
    t0 = float(np.mean(grid_np_init[20:wall_y_min, cx1+20:cx2-20, 5]))
    b0 = float(np.mean(grid_np_init[wall_y_max:236, cx1+20:cx2-20, 5]))
    print(f"  {'0':>6}  {d0:>12.4f}  {t0:>12.4f}  {b0:>12.4f}  (baseline)")

    routing_detected = False

    for step in range(1, n_steps + 1):
        grid, key = step_fn(grid, params, key)
        # Re-inject nodes and wall
        grid = grid.at[:, :, 5].set(jnp.where(blob_mask | wall_mask, 0.8, grid[:, :, 5]))

        if step % measure_every == 0:
            g = np.array(grid)
            direct = float(np.mean(g[cy-10:cy+10, cx1+20:cx2-20, 5]))
            top    = float(np.mean(g[20:wall_y_min, cx1+20:cx2-20, 5]))
            bot    = float(np.mean(g[wall_y_max:236, cx1+20:cx2-20, 5]))
            bypass = max(top, bot)
            verdict = ''
            if bypass > 0.15 and direct < bypass * 1.5:
                verdict = '✓ ROUTING around wall'
                routing_detected = True
            elif direct > 0.15:
                verdict = '~ filling direct (wall ignored?)'
            elif bypass > 0.05:
                verdict = '· weak bypass signal'
            else:
                verdict = '  blocked'
            print(f"  {step:>6}  {direct:>12.4f}  {top:>12.4f}  {bot:>12.4f}  {verdict}")

    print(f"\n  {'─'*65}")
    if routing_detected:
        print(f"  RESULT: ROUTING CONFIRMED. ch5 navigates around obstacles.")
        print(f"          This is genuine path-finding, not just diffusion.")
        print(f"          *** Skip all training. Start building the node interface. ***")
    else:
        print(f"  RESULT: No routing detected. Obstacle stops or is ignored.")
        print(f"          Need to examine visually — may still be diffusion filling.")


def test_live_grid_delta(ckpt_name='lenia_100000.pkl', warmup_steps=500):
    """
    Run the ch5 delta inspection on a LIVE grid (active GS chemistry),
    not a blank one. Warm up GS for warmup_steps first, then inspect.
    This is the real test — blank grid results don't transfer to live system.
    """
    print(f"\n{'═'*60}")
    print(f"TEST 4: ch5 delta on LIVE grid — {ckpt_name}")
    print(f"{'═'*60}")

    params = load_checkpoint(ckpt_name)
    update_net = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn = make_step_fn(update_net, perception_kernel)

    # Start from a slightly randomized grid so GS activates
    key = random.PRNGKey(99)
    grid = make_blank_grid(key)
    # Seed a small GS perturbation to kick off chemistry
    grid_np = np.array(grid)
    grid_np[110:146, 110:146, 0] = 0.5
    grid_np[110:146, 110:146, 1] = 0.25
    grid = jnp.array(grid_np)

    print(f"  Warming up GS chemistry for {warmup_steps} steps...")
    for _ in range(warmup_steps):
        grid, key = step_fn(grid, params, key)

    # Now inspect ch5 delta on the live grid
    from model import perceive
    perceived = perceive(grid, perception_kernel)
    delta = update_net.apply(params, perceived)
    ch5_delta = np.array(delta[:, :, 5])

    print(f"  ch5 delta on live grid:")
    print(f"    mean abs: {np.mean(np.abs(ch5_delta)):.6f}")
    print(f"    max abs:  {np.max(np.abs(ch5_delta)):.6f}")
    print(f"    std:      {np.std(ch5_delta):.6f}")
    print(f"    mean:     {np.mean(ch5_delta):.6f}  (negative = NCA suppressing ch5)")

    if np.std(ch5_delta) > 0.01:
        print(f"  → SPATIALLY VARYING delta. NCA responds to local state for ch5.")
        print(f"    Routing behavior may emerge from gradient following.")
    else:
        print(f"  → UNIFORM delta (std near zero). ch5 output is state-independent.")
        print(f"    Confirms blank-grid test was degenerate. Training needed.")

    # Also plant two seeds and measure bridging on live grid
    print(f"\n  Planting two seeds and running 2000 steps on live grid...")
    cx1, cx2, cy = 60, 196, 128
    grid = seed_ch5_blob(grid, cx1, cy, radius=15, strength=0.8)
    grid = seed_ch5_blob(grid, cx2, cy, radius=15, strength=0.8)

    ys, xs = jnp.mgrid[0:GRID_H, 0:GRID_W]
    blob_mask = (((xs - cx1)**2 + (ys - cy)**2) <= 15**2) | \
                (((xs - cx2)**2 + (ys - cy)**2) <= 15**2)

    print(f"\n  {'Step':>6}  {'Gap mean ch5':>14}  {'Gap max ch5':>12}")
    print(f"  {'─'*40}")

    for step in range(1, 2001):
        grid, key = step_fn(grid, params, key)
        grid = grid.at[:, :, 5].set(jnp.where(blob_mask, 0.8, grid[:, :, 5]))
        if step % 500 == 0:
            gap_mean, gap_max = measure_ch5_in_gap(grid, cx1, cx2, cy)
            print(f"  {step:>6}  {gap_mean:>14.6f}  {gap_max:>12.6f}")

    print(f"\n  Gap max > 0.1 = bridging present on live grid.")


if __name__ == '__main__':
    print("Project Mycelium — Live Grid Tests")
    print("2026-03-20")

    test_live_grid_delta('lenia_100000.pkl', warmup_steps=1000)

    print(f"\n{'═'*60}")
    print("Done. This is the real result — live grid, not blank grid.")
    print(f"{'═'*60}")
