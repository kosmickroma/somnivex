"""
daemon.py — Somnivex idle daemon.

When you walk away:
  - Mirrors your second monitor to your first automatically
  - Launches the screensaver

When you come back (or hit Q):
  - Kills the screensaver
  - Restores your normal side-by-side display layout

Usage:
    python daemon.py

Add to GNOME Startup Applications to run on login automatically.
"""

import subprocess
import time
import os
import sys
import signal
import re

# --- Config ---
IDLE_THRESHOLD = 60      # seconds idle before screensaver activates
POLL_INTERVAL  = 5       # how often to check idle time (seconds)

# Your display outputs — from xrandr --query
PRIMARY   = "DP-1"       # main monitor (left, primary)
SECONDARY = "HDMI-0"     # second monitor (gets mirrored to primary)

SCREENSAVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
PYTHON      = sys.executable
DISPLAY     = os.environ.get('DISPLAY', ':0')


def xrandr(*args):
    """Run an xrandr command."""
    subprocess.run(['xrandr'] + list(args), capture_output=True)


def mirror_displays():
    """Mirror secondary onto primary — both show the same thing."""
    print("  Mirroring displays...")
    xrandr('--output', SECONDARY, '--same-as', PRIMARY, '--auto')


def restore_displays():
    """Restore normal side-by-side layout."""
    print("  Restoring display layout...")
    xrandr('--output', PRIMARY,   '--auto', '--pos', '0x0', '--primary')
    xrandr('--output', SECONDARY, '--auto', '--right-of', PRIMARY)


def get_idle_seconds():
    """
    Get system idle time in seconds.
    Tries GNOME Mutter D-Bus first, falls back to xprintidle.
    Returns 0 on failure (safe — won't launch screensaver unexpectedly).
    """
    try:
        result = subprocess.run([
            'gdbus', 'call', '--session',
            '--dest',        'org.gnome.Mutter.IdleMonitor',
            '--object-path', '/org/gnome/Mutter/IdleMonitor/Core',
            '--method',      'org.gnome.Mutter.IdleMonitor.GetIdletime',
        ], capture_output=True, text=True, timeout=2)
        match = re.search(r'\d+', result.stdout)
        if match:
            return int(match.group()) / 1000.0
    except Exception:
        pass

    try:
        result = subprocess.run(['xprintidle'], capture_output=True, text=True, timeout=2)
        return int(result.stdout.strip()) / 1000.0
    except Exception:
        pass

    return 0


def main():
    screensaver_proc = None

    print(f"Somnivex daemon started.")
    print(f"Idle threshold : {IDLE_THRESHOLD}s")
    print(f"Displays       : {PRIMARY} (primary)  +  {SECONDARY} (mirrors on idle)")
    print(f"Watching... (Ctrl+C to stop)\n")

    def cleanup(sig, frame):
        if screensaver_proc and screensaver_proc.poll() is None:
            screensaver_proc.terminate()
        restore_displays()
        print("\nDaemon stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    while True:
        idle    = get_idle_seconds()
        running = screensaver_proc and screensaver_proc.poll() is None

        if idle >= IDLE_THRESHOLD and not running:
            print(f"Idle {idle:.0f}s — mirroring + launching screensaver.")
            mirror_displays()
            time.sleep(0.5)   # give xrandr a moment to settle
            env = os.environ.copy()
            env['DISPLAY'] = DISPLAY
            screensaver_proc = subprocess.Popen(
                [PYTHON, SCREENSAVER],
                env=env,
                cwd=os.path.dirname(SCREENSAVER),
            )

        elif idle < IDLE_THRESHOLD and running:
            print(f"User active — stopping screensaver + restoring displays.")
            screensaver_proc.terminate()
            screensaver_proc = None
            restore_displays()

        # Also check if main.py exited on its own (user hit Q)
        elif running and screensaver_proc.poll() is not None:
            print(f"Screensaver exited — restoring displays.")
            screensaver_proc = None
            restore_displays()

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
