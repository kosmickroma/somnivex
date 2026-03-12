#!/bin/bash
# launch_screensaver.sh — launch Somnivex in screensaver mode.
# Bind this to Ctrl+Alt+S via GNOME Settings → Keyboard → Custom Shortcuts.

cd "$(dirname "$0")"
source /home/kk/ai-env/bin/activate
python main.py --screensaver
