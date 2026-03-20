"""
decode.py — read values back from the NCA grid at known addresses.

Reads ch5 values from artist_state.json (zone map) OR directly samples
the center cell of each address line from the raw grid state.

Since we can't read the JAX array directly from outside run_free.py,
we use the ch5 heatmap approach: read the artist_state.json zone data
and also print a simple 16x16 ch5 summary if available.

For the cleanest read: samples the center x=128 of each address y.
run_free.py writes artist_state.json every AUTO_LOG_EVERY steps.

Usage:
    python nca/decode.py
"""

import os
import json
import numpy as np

STATE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artist_state.json')
HEATMAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artist_heatmap.npy')

# Must match encode.py
ADDRESSES = {
    0: 32,
    1: 96,
    2: 160,
    3: 224,
}

def decode():
    # Try heatmap first (most accurate)
    if os.path.exists(HEATMAP_FILE):
        heatmap = np.load(HEATMAP_FILE)
        print("Reading from ch5 heatmap (16x16):\n")
        print_heatmap(heatmap)
        print("\nDecoded values at known addresses:")
        for addr, y in ADDRESSES.items():
            # Sample the peak value across the full row (not just center)
            row = int(y / 256 * 16)
            val = float(np.max(heatmap[row, :]))
            print(f"  addr {addr} (y={y}) → {val:.3f}  (peak across row)")
        return

    # Fallback: zone map (coarse but available now)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        zones = state.get('zones', {})
        step = state.get('step', '?')
        print(f"Reading zone map at step {step} (coarse 4x4 — heatmap not yet built)\n")
        print("Zone densities (organism presence, not ch5 directly):")
        for row in range(4):
            line = ""
            for col in range(4):
                key = f"{row}{col}"
                val = zones.get(key, 0)
                line += f"  {val:.2f}"
            print(line)
        print("\nNote: run encode.py first, wait 5s, then decode.py reads back.")
        print("For exact ch5 readback, heatmap support needs to be added to run_free.py.")
    else:
        print("No state file found. Is run_free.py --artist --research running?")


def print_heatmap(h):
    """Print 16x16 heatmap as ASCII grid."""
    chars = ' ░▒▓█'
    for row in range(16):
        line = ""
        for col in range(16):
            v = h[row, col]
            idx = min(4, int(v * 5))
            line += chars[idx] * 2
        print(f"  {line}")


if __name__ == '__main__':
    decode()
