#!/usr/bin/env python3
"""
analyze_save.py — inspect a grid save file and dump ch13 (physics bit) statistics.

Usage:
    python nca/analyze_save.py                    # loads most recent save
    python nca/analyze_save.py nca/saves/grid_0060530.pkl

Shows:
  - ch13 mean, std, min, max across the whole grid
  - What fraction of cells are "Lenia territory" (ch13 > 0.5) vs "GS territory"
  - ASCII heatmap of ch13 spatial distribution (16x16 buckets)
  - If matplotlib available: saves ch13_heatmap.png
"""

import os
import sys
import glob
import pickle
import numpy as np

def load_save(path=None):
    if path is None:
        saves_dir = os.path.join(os.path.dirname(__file__), 'saves')
        files = sorted(glob.glob(os.path.join(saves_dir, 'grid_*.pkl')))
        if not files:
            print("No saves found in nca/saves/")
            sys.exit(1)
        path = files[-1]
    print(f"Loading: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f), path

def ascii_heatmap(data, rows=16, cols=32, label=""):
    H, W = data.shape
    rh, rw = H // rows, W // cols
    chars = " ░▒▓█"
    print(f"\n  {label}")
    print("  " + "─" * (cols + 2))
    for r in range(rows):
        row_data = data[r*rh:(r+1)*rh, :]
        line = "  │"
        for c in range(cols):
            cell = row_data[:, c*rw:(c+1)*rw]
            v = float(np.mean(cell))
            idx = int(np.clip(v * (len(chars) - 1), 0, len(chars) - 1))
            line += chars[idx]
        line += "│"
        print(line)
    print("  " + "─" * (cols + 2))
    print(f"  0.0 (GS) {'─'*12} 1.0 (Lenia)   scale")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    data, fpath = load_save(path)

    grid = data['grid']          # (H, W, 16)
    step = data.get('step_count', '?')
    phys = data.get('physics_bit', '?')
    free = data.get('free_channels', False)

    print(f"\n{'='*60}")
    print(f"  Step:         {step}")
    print(f"  Physics bit:  {phys}  (injected value before release)")
    print(f"  Free channels:{free}")
    print(f"  Grid shape:   {grid.shape}")
    print(f"{'='*60}")

    ch13 = grid[:, :, 13]   # physics bit channel
    ch_b = grid[:, :, 1]    # B channel (visible activity)

    print(f"\n── ch13 (physics bit) stats ──────────────────────────────")
    print(f"  mean:  {np.mean(ch13):.4f}")
    print(f"  std:   {np.std(ch13):.4f}")
    print(f"  min:   {np.min(ch13):.4f}")
    print(f"  max:   {np.max(ch13):.4f}")

    lenia_frac = float(np.mean(ch13 > 0.5))
    gs_frac    = float(np.mean(ch13 < 0.5))
    mixed_frac = float(np.mean((ch13 >= 0.3) & (ch13 <= 0.7)))
    print(f"\n  Cells > 0.5  (Lenia territory):  {lenia_frac*100:.1f}%")
    print(f"  Cells < 0.5  (GS territory):      {gs_frac*100:.1f}%")
    print(f"  Cells 0.3-0.7 (mixed zone):       {mixed_frac*100:.1f}%")

    # Histogram
    hist, edges = np.histogram(ch13.ravel(), bins=10, range=(0, 1))
    print(f"\n── ch13 histogram ────────────────────────────────────────")
    for i, count in enumerate(hist):
        lo, hi = edges[i], edges[i+1]
        bar = "█" * int(count / max(hist) * 30)
        print(f"  {lo:.1f}-{hi:.1f}  {bar} {count}")

    # Spatial map
    ch13_norm = np.clip(ch13, 0.0, 1.0)
    ascii_heatmap(ch13_norm, rows=16, cols=48, label="ch13 spatial map  (dark=GS, bright=Lenia)")

    # B channel for comparison
    b_abs = np.abs(ch_b)
    b_norm = np.clip(b_abs / (np.max(b_abs) + 1e-8), 0.0, 1.0)
    ascii_heatmap(b_norm, rows=16, cols=48, label="B channel activity (dark=quiet, bright=active)")

    # Correlation between ch13 and activity
    corr = np.corrcoef(ch13.ravel(), b_abs.ravel())[0, 1]
    print(f"\n── Correlation ch13 vs |B| activity ──────────────────────")
    print(f"  r = {corr:.4f}", end="  ")
    if abs(corr) > 0.3:
        if corr > 0:
            print("(active regions write HIGHER ch13 — Lenia-mode dominates where things happen)")
        else:
            print("(active regions write LOWER ch13 — GS-mode dominates where things happen)")
    else:
        print("(weak correlation — physics bit not strongly tied to activity location)")

    # Hidden channel summary
    hidden = grid[:, :, 2:13]
    print(f"\n── Hidden channels (ch2-12) means ───────────────────────")
    for i in range(11):
        ch = hidden[:, :, i]
        print(f"  ch{i+2:2d}: mean={np.mean(ch):+.3f}  std={np.std(ch):.3f}  "
              f"range=[{np.min(ch):.3f}, {np.max(ch):.3f}]")

    # Optional: save matplotlib heatmap
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Save analysis — step {step}  free={free}", fontsize=12)

        im0 = axes[0].imshow(ch13, cmap='RdBu_r', vmin=0, vmax=1, origin='upper')
        axes[0].set_title("ch13 (physics bit)\nblue=GS  red=Lenia")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(b_abs, cmap='hot', origin='upper')
        axes[1].set_title("|B| activity")
        plt.colorbar(im1, ax=axes[1])

        # ch13 only where activity > threshold
        thresh = np.percentile(b_abs, 70)
        masked = np.ma.masked_where(b_abs < thresh, ch13)
        im2 = axes[2].imshow(ch13, cmap='gray', origin='upper', alpha=0.3)
        im2b = axes[2].imshow(masked, cmap='RdBu_r', vmin=0, vmax=1, origin='upper')
        axes[2].set_title("ch13 where |B|>70th pct\n(physics in active regions)")
        plt.colorbar(im2b, ax=axes[2])

        out = os.path.join(os.path.dirname(fpath), 'ch13_heatmap.png')
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        print(f"\nHeatmap saved → {out}")
    except ImportError:
        print("\n(matplotlib not available — skipping image output)")

if __name__ == '__main__':
    main()
