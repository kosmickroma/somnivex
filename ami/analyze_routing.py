# ami/analyze_routing.py
#
# Analyze hidden channel activations in a trained routing NCA checkpoint.
# Compares politics vs climate signal states to find the routing grammar.
#
# What we're looking for:
#   Channels that activate differently for politics vs climate signals.
#   Those differences ARE the routing tokens — the internal vocabulary
#   the NCA developed to distinguish signal types and route selectively.
#
# Usage:
#   python ami/analyze_routing.py
#   python ami/analyze_routing.py ami/routing_checkpoints/routing_020000.pkl

import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from scipy.ndimage import zoom

sys.path.insert(0, 'nca')
sys.path.insert(0, 'ami')

from model import UpdateNet, make_perception_kernel, nca_step
from physarum_core import (
    H as SIM_H, W as SIM_W,
    GAP_RIGHT, RIGHT_MID,
    reset, step as phys_step, inject_anchors,
)
from train_routing import (
    TRAIN_H, TRAIN_W, ZONE_MAP_64, ZONE_MAP_64_JAX,
    CH_TRAIL, CH_SIGNAL, CH_ZONE, CH_PHYSICS, CH_F, CH_K,
    inject_zone_a_signal_only, generate_state,
)

N_SAMPLES    = 16    # states per signal type
ROLLOUT      = 24    # steps to run before sampling hidden channels
CHECKPOINT   = Path("ami/routing_checkpoints/routing_020000.pkl")


# ── ASCII helpers ──────────────────────────────────────────────────────────────

def ascii_map(data, rows=16, cols=32, lo=None, hi=None):
    H, W   = data.shape
    rh, rw = max(1, H // rows), max(1, W // cols)
    chars  = " ░▒▓█"
    lo     = lo if lo is not None else float(np.min(data))
    hi     = hi if hi is not None else float(np.max(data))
    rng    = hi - lo + 1e-8
    lines  = []
    for r in range(rows):
        line = "│"
        for c in range(cols):
            cell = data[r*rh:min((r+1)*rh, H), c*rw:min((c+1)*rw, W)]
            v    = float(np.mean(cell))
            idx  = int(np.clip((v - lo) / rng * (len(chars) - 1), 0, len(chars)-1))
            line += chars[idx]
        line += "│"
        lines.append(line)
    return lines


def print_side_by_side(map_a, map_b, label_a, label_b):
    w = len(map_a[0]) if map_a else 34
    print(f"  {label_a:<{w}}  {label_b}")
    print(f"  {'─'*w}  {'─'*w}")
    for la, lb in zip(map_a, map_b):
        print(f"  {la}  {lb}")
    print(f"  {'─'*w}  {'─'*w}")


# ── State generation ───────────────────────────────────────────────────────────

def make_states(rng, signal_type, n=N_SAMPLES):
    states = []
    for _ in range(n):
        g = generate_state(rng, signal_type=signal_type)
        states.append(g)
    return np.stack(states)   # (N, H, W, 16)


def run_forward(states, params, update_net, perception_kernel, steps=ROLLOUT):
    """Run NCA forward for `steps` steps, re-injecting control channels."""
    grids = jnp.array(states)
    key   = jax.random.PRNGKey(0)
    sig   = float(states[0, 0, 0, CH_SIGNAL])   # uniform — same for all cells
    for _ in range(steps):
        key, subkey   = jax.random.split(key)
        batch_keys    = jax.random.split(subkey, grids.shape[0])
        grids, _      = jax.vmap(
            lambda g, k: nca_step(g, params, update_net, perception_kernel, k)
        )(grids, batch_keys)
        grids = grids.at[:, :, :, CH_ZONE].set(ZONE_MAP_64_JAX)
        grids = grids.at[:, :, :, CH_PHYSICS].set(0.5)
        grids = grids.at[:, :, :, CH_SIGNAL].set(sig)
    return np.array(grids)   # (N, H, W, 16)


# ── Analysis ───────────────────────────────────────────────────────────────────

def channel_stats(grids, label):
    """Print mean/std/range for every channel across a batch of grids."""
    print(f"\n── Channel stats — {label} ({'─'*(40-len(label))})")
    print(f"  {'ch':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'──':>4}  {'────':>8}  {'────':>8}  {'────':>8}  {'────':>8}")
    for ch in range(16):
        vals = grids[:, :, :, ch]
        print(f"  ch{ch:2d}  {np.mean(vals):+8.4f}  {np.std(vals):8.4f}  "
              f"{np.min(vals):+8.4f}  {np.max(vals):+8.4f}")


def routing_grammar(pol_grids, cli_grids):
    """
    Find channels that differentiate politics from climate.
    High separation = that channel encodes signal type = routing token.
    """
    print(f"\n── Routing grammar — channel separation ──────────────────────")
    print(f"  Channels ranked by |mean_politics - mean_climate|")
    print(f"  These are the routing tokens the NCA developed.\n")
    print(f"  {'ch':>4}  {'pol_mean':>10}  {'cli_mean':>10}  {'diff':>10}  {'signal?'}")
    print(f"  {'──':>4}  {'────────':>10}  {'────────':>10}  {'────':>10}  {'───────'}")

    diffs = []
    for ch in range(16):
        pm = float(np.mean(pol_grids[:, :, :, ch]))
        cm = float(np.mean(cli_grids[:, :, :, ch]))
        diffs.append((abs(pm - cm), ch, pm, cm))

    diffs.sort(reverse=True)
    for diff, ch, pm, cm in diffs:
        marker = " ← ROUTING TOKEN" if diff > 0.05 else ""
        print(f"  ch{ch:2d}  {pm:+10.4f}  {cm:+10.4f}  {pm-cm:+10.4f}{marker}")


def zone_activation(pol_grids, cli_grids):
    """Show Zone B vs C activation for each signal type."""
    gap_r = int(GAP_RIGHT * TRAIN_W / SIM_W)
    mid   = TRAIN_H // 2

    pol_b = float(np.mean(pol_grids[:, :mid, gap_r:, CH_TRAIL]))
    pol_c = float(np.mean(pol_grids[:, mid:, gap_r:, CH_TRAIL]))
    cli_b = float(np.mean(cli_grids[:, :mid, gap_r:, CH_TRAIL]))
    cli_c = float(np.mean(cli_grids[:, mid:, gap_r:, CH_TRAIL]))

    print(f"\n── Zone activation after {ROLLOUT} steps ──────────────────────────")
    print(f"  Signal      Zone B (Claude)   Zone C (Gemini)   Correct?")
    print(f"  ──────────  ───────────────   ───────────────   ────────")

    pol_ok = "✓ YES" if pol_b > pol_c else "✗ NO"
    cli_ok = "✓ YES" if cli_c > cli_b else "✗ NO"
    print(f"  politics    {pol_b:.4f}           {pol_c:.4f}           {pol_ok}")
    print(f"  climate     {cli_b:.4f}           {cli_c:.4f}           {cli_ok}")


def spatial_maps(pol_grids, cli_grids, top_channels):
    """Print side-by-side spatial maps of the most discriminating channels."""
    print(f"\n── Spatial maps — top routing channels ───────────────────────")
    for ch in top_channels[:4]:
        pol_map = np.mean(pol_grids[:, :, :, ch], axis=0)
        cli_map = np.mean(cli_grids[:, :, :, ch], axis=0)
        lo = min(pol_map.min(), cli_map.min())
        hi = max(pol_map.max(), cli_map.max())
        ma = ascii_map(pol_map, rows=8, cols=32, lo=lo, hi=hi)
        mb = ascii_map(cli_map, rows=8, cols=32, lo=lo, hi=hi)
        print(f"\n  ch{ch} (same scale  lo={lo:.3f} hi={hi:.3f})")
        print_side_by_side(ma, mb, "politics", "climate")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else CHECKPOINT

    if not ckpt.exists():
        # Fall back to latest available checkpoint
        available = sorted(Path("ami/routing_checkpoints").glob("routing_*.pkl"))
        if not available:
            print("No checkpoints found.")
            sys.exit(1)
        ckpt = available[-1]
        print(f"(using latest checkpoint: {ckpt})")

    print(f"\nLoading {ckpt}...")
    with open(ckpt, 'rb') as f:
        params = pickle.load(f)
    params = jax.device_put(params)

    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()

    rng = np.random.default_rng(99)

    print(f"Generating {N_SAMPLES} politics states...")
    pol_raw = make_states(rng, "politics")
    print(f"Generating {N_SAMPLES} climate states...")
    cli_raw = make_states(rng, "climate")
    print(f"Generating {N_SAMPLES} baseline states (no signal)...")
    base_raw = make_states(rng, None)

    print(f"Running {ROLLOUT} NCA steps forward...")
    pol_grids  = run_forward(pol_raw,  params, update_net, perception_kernel)
    cli_grids  = run_forward(cli_raw,  params, update_net, perception_kernel)
    base_grids = run_forward(base_raw, params, update_net, perception_kernel)

    print("\n" + "="*60)
    print("  ROUTING NCA ANALYSIS")
    print(f"  Checkpoint: {ckpt.name}")
    print(f"  Samples: {N_SAMPLES} per signal type  |  Rollout: {ROLLOUT} steps")
    print("="*60)

    # Zone activation — did routing work?
    zone_activation(pol_grids, cli_grids)

    # Which channels differentiate signal types?
    routing_grammar(pol_grids, cli_grids)

    # Full channel stats per signal type
    channel_stats(pol_grids,  "politics (horizontal bar → Zone B)")
    channel_stats(cli_grids,  "climate  (vertical bar   → Zone C)")
    channel_stats(base_grids, "baseline (no signal)")

    # Spatial maps of top discriminating channels
    diffs = []
    for ch in range(16):
        pm = float(np.mean(pol_grids[:, :, :, ch]))
        cm = float(np.mean(cli_grids[:, :, :, ch]))
        diffs.append((abs(pm - cm), ch))
    diffs.sort(reverse=True)
    top_channels = [ch for _, ch in diffs[:6]]
    spatial_maps(pol_grids, cli_grids, top_channels)

    print("\nDone.")


if __name__ == "__main__":
    main()
