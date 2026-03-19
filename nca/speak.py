#!/usr/bin/env python3
# nca/speak.py — clean input terminal for artist mode video recording
# Run in a separate terminal: python nca/speak.py
# Type your direction and press Enter. The bridge picks it up on its next turn.

import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'human_input.txt')

# Clear on start
with open(OUT, 'w') as f:
    f.write('')

print("=" * 50)
print("  SOMNIVEX — Director Terminal")
print("  Type a direction and press Enter.")
print("  Gemini responds on its next turn (~6s).")
print("=" * 50)
print()

while True:
    try:
        msg = input("  > ")
        if msg.strip():
            with open(OUT, 'w') as f:
                f.write(msg.strip())
            print(f"  ✓ sent\n")
    except (EOFError, KeyboardInterrupt):
        print("\n  Director terminal closed.")
        break
