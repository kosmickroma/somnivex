"""
main.py — Somnivex. Gray-Scott reaction-diffusion, always alive.

Two chemicals react forever. F and k parameters drift slowly between
named regimes (spirals, chaos, waves, worms...) — the world morphs
from one type of behavior to another without ever stopping.

Controls:
  U     — like this era  (saved to ratings, biases future choices)
  D     — dislike this era
  S     — force an immediate regime shift
  Space — pause / unpause
  Q     — quit
"""

import sys
import time
import json
import os
import numpy as np
import pygame
from jax import random

# Log file — tail -f somnivex.log in a second terminal to monitor while fullscreen
LOG_FILE = os.path.join(os.path.dirname(__file__), 'somnivex.log')

def log(msg):
    """Print to terminal AND append to log file."""
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

from config import (
    GRID_H, GRID_W, FPS,
    GS_STEPS_PER_FRAME,
    POKE_INTERVAL, POKE_COUNT, POKE_SIZE,
    DRIFT_INTERVAL,
    PALETTE_FADE_SPEED,
    STALE_CHECK_INTERVAL, STALE_THRESHOLD,
    STALE_POKE_COUNT, STALE_POKE_SIZE,
    SNAPSHOT_INTERVAL, FREEZE_THRESHOLD,
    DRIFT_BLAST_PATCHES, DRIFT_BLAST_SIZE,
)
from gs.engine    import GS_REGIMES, init_gs_grid, make_gs_step_fn
from nca.params   import sample_params, load_prefs, PALETTES
from display.windows import init_window, render_gs_frame, RENDER_MODES, EFFECTS

RATINGS_FILE = os.path.join(os.path.dirname(__file__), 'ratings_log.json')

# How fast F and K drift toward their targets each frame.
# 0.0005 per frame at 30fps ≈ 30s to fully transition between regimes.
# You'll actually WATCH the spirals morph into worms morph into chaos.
FK_DRIFT_SPEED = 0.005   # 0.005 for testing (fast morph), 0.0005 for real use (slow morph)


# ── Noise injection ───────────────────────────────────────────────────────────
def inject_gs_noise(B, key, n_pokes, patch_size):
    """
    Drop patches of B=0.25 into random locations.
    0.25 is the natural GS reaction value — 0.5 was too high and
    caused some regimes to immediately die after injection.
    """
    H, W = B.shape
    for _ in range(n_pokes):
        key, sk1, sk2 = random.split(key, 3)
        y = int(random.randint(sk1, (), 0, H - patch_size))
        x = int(random.randint(sk2, (), 0, W - patch_size))
        B = B.at[y:y+patch_size, x:x+patch_size].set(0.25)
    return B, key


def restore_gs_fields(A, B, key, n_patches=20):
    """
    Fix over-saturation: B has flooded the whole grid (yellow/solid screen).

    The problem: B spread everywhere, consumed all A, now nothing reacts.
    The fix: carve out patches where A=1.0 and B=0.0 — restore the substrate.
    The edges of these cleared patches immediately react with surrounding B,
    creating new reaction fronts. The pattern rebuilds from those edges.

    This is NOT a full reset — the surrounding B structure is preserved.
    It's more like punching holes in the flood so the reaction can restart.
    """
    H, W = B.shape
    for _ in range(n_patches):
        key, sk1, sk2, sk3 = random.split(key, 4)
        y  = int(random.randint(sk1, (), 0, H - 16))
        x  = int(random.randint(sk2, (), 0, W - 16))
        sz = int(random.randint(sk3, (), 8, 16))
        A  = A.at[y:y+sz, x:x+sz].set(1.0)
        B  = B.at[y:y+sz, x:x+sz].set(0.0)
    return A, B, key


# ── Rating ────────────────────────────────────────────────────────────────────
def save_rating(era_params, liked):
    entry = {
        'seed':      era_params.seed,
        'subject':   era_params.subject,
        'category':  era_params.category,
        'aesthetic': era_params.aesthetic,
        'palette':   era_params.palette_name,
        'mood':      era_params.mood,
        'liked':     liked,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ratings = []
    if os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE) as f:
            try:
                ratings = json.load(f)
            except json.JSONDecodeError:
                ratings = []
    ratings.append(entry)
    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)
    label = "LIKED" if liked else "DISLIKED"
    print(f"  {label}: {era_params.subject} / {era_params.palette_name}")


# ── HUD ───────────────────────────────────────────────────────────────────────
def draw_hud(screen, era_params, regime_name, f, k, step_count,
             render_mode, effect, toast=None):
    font_sm = pygame.font.SysFont("monospace", 14)
    font_lg = pygame.font.SysFont("monospace", 22, bold=True)

    info = (f"{era_params.subject} | {era_params.palette_name} | "
            f"{regime_name} | f={f:.4f} k={k:.4f} | "
            f"{render_mode}+{effect} | step {step_count}")
    screen.blit(font_sm.render(info, True, (70, 70, 70)), (10, 10))

    if toast:
        text, color = toast
        screen.blit(font_lg.render(text, True, color), (10, 30))


# ── World setup ───────────────────────────────────────────────────────────────
def setup_world(step_fn):
    """
    Initialize the living world. Called once at startup.
    Picks a random starting regime and palette (preference-biased if enough data).
    """
    prefs      = load_prefs(RATINGS_FILE)
    era_params = sample_params(prefs=prefs)

    regime_name         = np.random.choice(list(GS_REGIMES.keys()))
    f, k                = GS_REGIMES[regime_name]
    target_regime_name  = regime_name
    target_f, target_k  = f, k

    key          = random.PRNGKey(era_params.seed)
    A, B         = init_gs_grid(key, GRID_H, GRID_W)
    key, _       = random.split(key)

    pal_arr     = np.array(era_params.palette, dtype=np.float32)
    render_mode = np.random.choice(RENDER_MODES)
    effect      = np.random.choice(EFFECTS)

    print(f"\n{'='*60}")
    print(f"  Regime      : {regime_name}  (f={f}, k={k})")
    print(f"  Palette     : {era_params.palette_name}")
    print(f"  Subject     : {era_params.subject} / {era_params.mood}")
    print(f"  Mode        : {render_mode} + {effect}")
    print(f"{'='*60}\n")

    return dict(
        era_params         = era_params,
        A                  = A,
        B                  = B,
        f                  = f,
        k                  = k,
        target_f           = target_f,
        target_k           = target_k,
        regime_name        = regime_name,
        target_regime_name = target_regime_name,
        step_fn            = step_fn,
        jax_key            = key,
        step_count         = 0,
        current_pal        = pal_arr.copy(),
        target_pal         = pal_arr.copy(),
        render_mode        = render_mode,
        effect             = effect,
        last_poke          = time.time(),
        last_drift         = time.time(),
        last_stale         = time.time(),
        last_snapshot      = time.time(),
        B_snapshot         = np.array(B),
        stale_count        = 0,
    )


def pick_new_regime():
    """Pick a random regime name and its (f, k) values."""
    name    = np.random.choice(list(GS_REGIMES.keys()))
    f, k    = GS_REGIMES[name]
    return name, f, k


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    screen = init_window()
    clock  = pygame.time.Clock()
    pygame.font.init()

    print("\nStarting Somnivex — Gray-Scott reaction-diffusion.")
    print("U=like  D=dislike  S=shift now  Space=pause  Q=quit")
    print("(First frame takes a few seconds — JAX compiling. Normal.)\n")

    # Compile the step function once — reused forever
    _step_fn   = make_gs_step_fn(GS_STEPS_PER_FRAME)
    w          = setup_world(_step_fn)
    paused     = False
    toast_data = None

    while True:

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit(); sys.exit()

                if event.key == pygame.K_SPACE:
                    paused = not paused
                    log("Paused" if paused else "Resumed")

                if event.key == pygame.K_s:
                    # Force immediate regime shift — snap f/k NOW, don't drift
                    name, tf, tk    = pick_new_regime()
                    new_pal_name    = np.random.choice(list(PALETTES.keys()))
                    new_render_mode = np.random.choice(RENDER_MODES)
                    new_effect      = np.random.choice(EFFECTS)
                    w['regime_name']        = name
                    w['target_regime_name'] = name
                    w['f']          = tf
                    w['k']          = tk
                    w['target_f']   = tf
                    w['target_k']   = tk
                    w['target_pal'] = np.array(PALETTES[new_pal_name], dtype=np.float32)
                    w['era_params'] = sample_params(prefs=load_prefs(RATINGS_FILE))
                    w['render_mode'] = new_render_mode
                    w['effect']      = new_effect
                    w['last_drift']  = time.time()
                    w['A'], w['B'], w['jax_key'] = restore_gs_fields(
                        w['A'], w['B'], w['jax_key'], n_patches=DRIFT_BLAST_PATCHES
                    )
                    log(f"  Shifting → {name} (f={tf}, k={tk}) | {new_pal_name} / {new_render_mode}+{new_effect}")

                if event.key == pygame.K_u:
                    save_rating(w['era_params'], liked=True)
                    toast_data = ("  LIKED  ", (100, 255, 100), time.time() + 2.0)

                if event.key == pygame.K_d:
                    save_rating(w['era_params'], liked=False)
                    toast_data = (" DISLIKED ", (255, 80, 80), time.time() + 2.0)

        if paused:
            clock.tick(FPS); continue

        # ── Drift: shift regime every DRIFT_INTERVAL seconds ──────────────────
        # Sets new target F/K and palette. The actual F/K values ease toward
        # the target a tiny bit each frame (FK_DRIFT_SPEED) — smooth transition.
        if time.time() - w['last_drift'] >= DRIFT_INTERVAL:
            name, tf, tk    = pick_new_regime()
            new_pal_name    = np.random.choice(list(PALETTES.keys()))
            new_render_mode = np.random.choice(RENDER_MODES)
            new_effect      = np.random.choice(EFFECTS)
            w['target_regime_name'] = name
            w['target_f']    = tf
            w['target_k']    = tk
            w['target_pal']  = np.array(PALETTES[new_pal_name], dtype=np.float32)
            w['era_params']  = sample_params(prefs=load_prefs(RATINGS_FILE))
            w['render_mode'] = new_render_mode
            w['effect']      = new_effect
            w['last_drift']  = time.time()
            w['A'], w['B'], w['jax_key'] = restore_gs_fields(
                w['A'], w['B'], w['jax_key'], n_patches=DRIFT_BLAST_PATCHES
            )
            log(f"  Drifting → {name} (f={tf}, k={tk}) | {new_pal_name} / {new_render_mode}+{new_effect}")

        # ── F/K smooth drift ──────────────────────────────────────────────────
        # Each frame, nudge f and k a tiny bit toward their targets.
        # This is what makes patterns MORPH rather than snap.
        # You'll see spirals gradually warp into worms warp into chaos.
        w['f'] += (w['target_f'] - w['f']) * FK_DRIFT_SPEED
        w['k'] += (w['target_k'] - w['k']) * FK_DRIFT_SPEED
        # Update displayed regime name when we're close enough to the target
        if abs(w['f'] - w['target_f']) < 0.0005 and abs(w['k'] - w['target_k']) < 0.0005:
            w['regime_name'] = w['target_regime_name']

        # ── Poke: inject new reaction fronts periodically ─────────────────────
        if time.time() - w['last_poke'] >= POKE_INTERVAL:
            w['B'], w['jax_key'] = inject_gs_noise(
                w['B'], w['jax_key'], POKE_COUNT, POKE_SIZE
            )
            w['last_poke'] = time.time()

        # ── Temporal freeze check — catches patterns that have structure but stopped moving ─
        # Worms/mazes score fine on spatial std but are visually frozen.
        # Compare B to a snapshot taken SNAPSHOT_INTERVAL seconds ago.
        if time.time() - w['last_snapshot'] >= SNAPSHOT_INTERVAL:
            B_now  = np.array(w['B'])
            B_move = float(np.abs(B_now - w['B_snapshot']).mean())
            w['B_snapshot']    = B_now
            w['last_snapshot'] = time.time()
            if B_move < FREEZE_THRESHOLD:
                log(f"  Frozen (movement={B_move:.5f}) — blasting to break structure")
                w['A'], w['B'], w['jax_key'] = restore_gs_fields(
                    w['A'], w['B'], w['jax_key'], n_patches=DRIFT_BLAST_PATCHES
                )

        # ── Stale check: detect dead OR flooded, fix each, give up after 3 tries ─
        if time.time() - w['last_stale'] >= STALE_CHECK_INTERVAL:
            w['last_stale'] = time.time() + 30   # backoff — wait 30s extra before next check
            B_np   = np.array(w['B'])
            B_std  = float(B_np.std())
            B_mean = float(B_np.mean())

            if B_std < STALE_THRESHOLD:
                w['stale_count'] += 1

                if w['stale_count'] >= 3:
                    # Tried 3 times, this regime isn't working — force a shift
                    name, tf, tk = pick_new_regime()
                    new_pal_name    = np.random.choice(list(PALETTES.keys()))
                    new_render_mode = np.random.choice(RENDER_MODES)
                    new_effect      = np.random.choice(EFFECTS)
                    w['target_regime_name'] = name
                    w['target_f']    = tf
                    w['target_k']    = tk
                    w['target_pal']  = np.array(PALETTES[new_pal_name], dtype=np.float32)
                    w['era_params']  = sample_params(prefs=load_prefs(RATINGS_FILE))
                    w['render_mode'] = new_render_mode
                    w['effect']      = new_effect
                    w['last_drift']  = time.time()
                    w['stale_count'] = 0
                    key = w['jax_key']
                    w['A'], w['B']   = init_gs_grid(key, GRID_H, GRID_W)
                    log(f"  Gave up on stale — forcing {name} | {new_pal_name} | {new_render_mode} + {new_effect}")

                elif B_mean < 0.02:
                    # Dead — inject new B seeds
                    log(f"  Dead (mean={B_mean:.3f}, try {w['stale_count']}) — re-seeding...")
                    w['B'], w['jax_key'] = inject_gs_noise(
                        w['B'], w['jax_key'], STALE_POKE_COUNT, STALE_POKE_SIZE
                    )
                else:
                    # Flooded — punch holes so A can recover
                    log(f"  Flooded (mean={B_mean:.3f}, try {w['stale_count']}) — restoring...")
                    w['A'], w['B'], w['jax_key'] = restore_gs_fields(
                        w['A'], w['B'], w['jax_key']
                    )
            else:
                w['stale_count'] = 0   # healthy — reset the counter

        # ── Step Gray-Scott ───────────────────────────────────────────────────
        w['A'], w['B'] = w['step_fn'](w['A'], w['B'], w['f'], w['k'])
        w['step_count'] += GS_STEPS_PER_FRAME

        # ── Palette crossfade ─────────────────────────────────────────────────
        w['current_pal'] += (w['target_pal'] - w['current_pal']) * PALETTE_FADE_SPEED

        # ── Render ────────────────────────────────────────────────────────────
        render_gs_frame(screen, w['A'], w['B'],
                        w['current_pal'], w['render_mode'], w['effect'])

        toast = None
        if toast_data:
            text, color, expiry = toast_data
            if time.time() < expiry:
                toast = (text, color)
            else:
                toast_data = None

        draw_hud(screen, w['era_params'], w['regime_name'],
                 w['f'], w['k'], w['step_count'],
                 w['render_mode'], w['effect'], toast)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    run()
