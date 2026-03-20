"""
encode.py — write values into the NCA grid at known addresses via ch5 trails.

Each "address" is a horizontal line at a fixed y coordinate.
The value is encoded as the trail strength (0.0 to 1.0).
The NCA holds these values permanently via trail_mask re-injection.

Usage:
    python nca/encode.py                  # encode default test values
    python nca/encode.py 0.3 0.8 0.5     # encode custom values at addresses 0,1,2

Addresses (y coordinates):
    addr 0 → y=32
    addr 1 → y=96
    addr 2 → y=160
    addr 3 → y=224
"""

import os
import sys
import time

CMD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_commands_artist.txt')

# Fixed y coordinates for each address (one per horizontal band)
ADDRESSES = {
    0: 32,
    1: 96,
    2: 160,
    3: 224,
}

def encode(values):
    """Write a list of values (0.0-1.0) into the grid at known addresses."""
    if len(values) > len(ADDRESSES):
        print(f"Max {len(ADDRESSES)} values supported")
        sys.exit(1)

    commands = []
    for i, val in enumerate(values):
        val = max(0.0, min(1.0, float(val)))
        y = ADDRESSES[i]
        # Full-width horizontal trail at this y, strength = val, width 1
        commands.append(f"trail 10 {y} 245 {y} {val:.2f} 4")
        print(f"  addr {i} (y={y}) ← {val:.2f}")

    block = "COMMANDS:\n" + "\n".join(commands)
    with open(CMD_FILE, 'w') as f:
        f.write(block)
    print(f"\nWritten to {CMD_FILE}")
    print("NCA will pick this up within ~1 second.")
    print("Wait 5 seconds then run decode.py to read back.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        vals = [float(v) for v in sys.argv[1:]]
    else:
        # Default test: encode 0.3 and 0.8
        vals = [0.3, 0.8]
        print("No values given — using defaults: 0.3, 0.8")

    print(f"\nEncoding {len(vals)} value(s) into NCA grid...")
    encode(vals)
