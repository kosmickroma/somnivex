# config.py — all tunable constants in one place.
# Adjust these to dial in timing, speed, and behavior.

# --- Display ---
# Set DUAL_SCREEN = True to span both monitors (one wide 3840x1080 NOFRAME window).
# Left screen: current sim. Right screen: same sim, different palette + render mode.
# Set False for single fullscreen.
DUAL_SCREEN = True
SCREEN_W    = 1920    # width of one monitor
SCREEN_H    = 1080
DISPLAY_W   = SCREEN_W * 2 if DUAL_SCREEN else SCREEN_W
DISPLAY_H   = SCREEN_H
GRID_H = 256
GRID_W = 256
FPS = 15   # 30 for final/screensaver use, 15 for dev (halves CPU render overhead)

# --- Gray-Scott step rate ---
# Doubled to compensate for halved FPS — same total chemistry speed.
# 60 steps/frame at 15fps = 900 reaction steps/sec (same as 30 steps @ 30fps).
GS_STEPS_PER_FRAME = 60

# --- NCA (kept for reference, not used as main engine anymore) ---
STEPS_PER_FRAME = 15
STEP_SIZE = 0.3

# --- Poke (continuous micro-disruption) ---
# This is what keeps the background alive instead of freezing into rectangles.
# POKE_AMPLITUDE: 0.35 = aggressive (breathing mode feel), 0.18 = subtle churn
POKE_INTERVAL = 3.0
POKE_COUNT = 3
POKE_SIZE = 6
POKE_AMPLITUDE = 0.18

# --- Drift (world character shift) ---
# Every DRIFT_INTERVAL seconds, we mutate the weights slightly and crossfade
# to a new palette. The world doesn't reset — it BECOMES something else.
# DRIFT_NOISE controls how aggressive the mutation is:
#   0.05 = gentle nudge, almost imperceptible shift
#   0.15 = noticeable character change
#   0.30 = pretty wild shift
DRIFT_INTERVAL = 20     # 20s between character shifts (20s for testing, 120 for real use)
DRIFT_NOISE    = 0.10   # mutation intensity per drift event

# --- Palette crossfade ---
# Each frame we nudge the displayed palette slightly toward the target palette.
# 0.002 per frame at 30fps ≈ 17 seconds to fully crossfade. Feels organic.
PALETTE_FADE_SPEED = 0.002

# --- Stale detection ---
# Two kinds of stale:
#   Spatial stale:  B.std() < STALE_THRESHOLD  — dead or flooded (no structure)
#   Temporal stale: mean|B_now - B_snapshot| < FREEZE_THRESHOLD — has structure but frozen
#                   This catches worms/mazes that locked in and stopped evolving.
STALE_CHECK_INTERVAL = 20     # seconds between staleness checks
STALE_THRESHOLD      = 0.03   # spatial: grid std below this = dead/flooded
STALE_POKE_COUNT     = 12     # heavy blast when spatially stale
STALE_POKE_SIZE      = 14     # bigger patches too
STALE_AMPLITUDE      = 0.35   # full amplitude blast
SNAPSHOT_INTERVAL    = 15     # seconds between temporal freeze snapshots (15 for testing, 45 for real use)
FREEZE_THRESHOLD     = 0.003  # mean |B_now - B_prev| below this = frozen

# --- Drift disruption ---
# When the regime shifts, punch this many A=1/B=0 holes in the grid.
# New reaction fronts form at the hole edges and behave under the new F/K.
# Without this, stable patterns (worms, mazes) resist new chemistry and persist.
DRIFT_BLAST_PATCHES  = 12     # holes punched on each regime drift/shift
DRIFT_BLAST_SIZE     = 14     # size of each hole

# --- Ratings ---
# Minimum number of ratings before we start biasing toward liked stuff.
# Below this threshold: pure random (blank slate behavior).
MIN_RATINGS_TO_LEARN = 10
