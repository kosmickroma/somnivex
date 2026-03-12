#!/bin/bash
# Launch Somnivex screensaver manually.
# Mirrors displays, runs screensaver, restores layout on exit.

PRIMARY="DP-1"
SECONDARY="HDMI-0"

echo "Mirroring displays..."
xrandr --output $SECONDARY --same-as $PRIMARY --auto
sleep 0.5

source /home/kk/ai-env/bin/activate
python /home/kk/projects/ml/somnivex/main.py

echo "Restoring displays..."
xrandr --output $PRIMARY --auto --pos 0x0 --primary --output $SECONDARY --auto --right-of $PRIMARY
