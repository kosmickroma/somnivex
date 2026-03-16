# kktodo: nca/run_lenia.py — Lenia checkpoint free-run visualizer

## What this does

Tests the trained dual-teacher checkpoint. Three modes:
- **L mode (Lenia)**: seed an Orbium creature, physics_bit=1. NCA runs on its own — no Lenia kernel. The big question: does a creature appear and glide?
- **G mode (GS)**: standard GS warmup seed, physics_bit=0. Verify GS still works.
- **H mode (Hybrid)**: GS warmup + Orbium seeded on top, physics_bit=0. What happens at the boundary? Does GS chemistry kill the creature, or do they coexist?

In all modes the NCA runs completely on its own — no external physics, no target. We just re-inject ch13/ch14/ch15 after each step so the NCA always knows which regime it's in.

---

## How to run

```
python nca/run_lenia.py
```

Auto-loads the latest `lenia_XXXXXX.pkl` from `nca/checkpoints/`. If training hasn't finished yet, pass a specific checkpoint:

```
python nca/run_lenia.py --checkpoint nca/checkpoints/lenia_010000.pkl
```

---

## Controls

- **L** — switch to Lenia mode (Orbium seed, physics_bit=1)
- **G** — switch to GS mode (GS warmup, physics_bit=0)
- **H** — switch to Hybrid mode (GS + Orbium, physics_bit=0)
- **R** — reseed current mode
- **P** — cycle palette
- **+/-** or **]/[** — speed up/down (steps per frame)
- **Q** — quit

---

## Architecture reminder

What `nca_step` does in free run:
1. Each cell perceives its 3×3 neighborhood via Identity / Sobel X / Sobel Y / Laplacian → 64 inputs
2. UpdateNet maps 64 → 16 delta
3. 50% fire mask applied
4. new_grid = clip(old_grid + delta * mask, 0, 1)
5. **We then re-inject** ch13 (physics_bit), ch14 (f or mu), ch15 (k or sigma)
   — this is mandatory; without it the NCA forgets what physics it's running

The NCA never sees the Lenia kernel in free run. If Lenia dynamics appear, it's because
the UpdateNet learned to implement Lenia physics in its weights, using ch13=1 as the cue.

---

## Channel layout (reminder)

```
ch  0    = A  (GS food field / Lenia activation — the main visual)
ch  1    = B  (GS predator   / 0 for Lenia)
ch  2-12 = hidden (NCA uses these internally)
ch 13    = physics bit  (0.0 = GS, 1.0 = Lenia)
ch 14    = f (GS) or mu    (Lenia, 0.150 for Orbium)
ch 15    = k (GS) or sigma (Lenia, 0.014 for Orbium)
```

---

## Full code

Create file: `nca/run_lenia.py`

```python
# nca/run_lenia.py — Free-run visualizer for the dual-teacher Lenia checkpoint.
#
# Tests three modes:
#   L mode (Lenia):  Orbium seed, physics_bit=1 — does the NCA produce a moving creature?
#   G mode (GS):     GS warmup,   physics_bit=0 — is GS still intact?
#   H mode (Hybrid): GS + Orbium, physics_bit=0 — do they coexist or annihilate?
#
# Run from project root:
#     python nca/run_lenia.py
#     python nca/run_lenia.py --checkpoint nca/checkpoints/lenia_010000.pkl
#
# Controls: L=lenia  G=gs  H=hybrid  R=reseed  P=palette  ]/[=speed  Q=quit

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pygame
import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca.model import (
    N_CHANNELS, N_FILTERS,
    CH_A, CH_B, CH_F, CH_K,
    UpdateNet, make_perception_kernel, make_step_fn,
)
from nca.lenia import (
    CH_PHYSICS, LENIA_R, LENIA_CREATURES,
    seed_creature, make_kernel_fft,
)
from gs.engine import GS_REGIMES, gs_step, init_gs_grid
from nca.params import PALETTES

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

# Free-run grid is bigger than training (64×64) so creatures have room to roam.
# But not so big that the kernel radius doesn't cover the creature.
GRID_H = 128
GRID_W = 128

SCREEN_W        = 900
SCREEN_H        = 900
FPS             = 30
STEPS_PER_FRAME = 3

# Lenia parameters (canonical Orbium)
ORBIUM_MU    = 0.150
ORBIUM_SIGMA = 0.014

# GS regime for seeding
GS_REGIME    = "mitosis"   # interesting structured patterns
GS_WARMUP    = 300         # steps of real GS before handing off to NCA


def find_latest_checkpoint():
    """Return the latest lenia_XXXXXX.pkl, or None if none exist."""
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'lenia_*.pkl')))
    return files[-1] if files else None


# ── Grid builders ──────────────────────────────────────────────────────────────

def build_lenia_grid(H, W):
    """
    Lenia mode: place one Orbium creature, set physics_bit=1.

    ch0 = Orbium activation from the real 20×20 seed
    ch1 = 0  (Lenia has one state variable — GS owns ch1, leave it alone)
    ch13 = 1.0  (physics bit: Lenia)
    ch14 = mu, ch15 = sigma  (Orbium canonical params)
    """
    rng = np.random.default_rng()
    A   = seed_creature(H, W, rng)

    grid = np.zeros((H, W, N_CHANNELS), dtype=np.float32)
    grid[:, :, CH_A]       = A
    grid[:, :, CH_PHYSICS] = 1.0
    grid[:, :, CH_F]       = ORBIUM_MU
    grid[:, :, CH_K]       = ORBIUM_SIGMA
    return jnp.array(grid)


def build_gs_grid(key, H, W):
    """
    GS mode: warm up real GS, hand off to NCA.

    ch0 = A, ch1 = B  (GS chemicals)
    ch13 = 0.0  (physics bit: GS)
    ch14 = f, ch15 = k
    """
    f, k = GS_REGIMES[GS_REGIME]
    key, sk = random.split(key)
    A, B = init_gs_grid(sk, H, W)
    for _ in range(GS_WARMUP):
        A, B = gs_step(A, B, f, k)

    grid = np.zeros((H, W, N_CHANNELS), dtype=np.float32)
    grid[:, :, CH_A]       = np.array(A)
    grid[:, :, CH_B]       = np.array(B)
    grid[:, :, CH_PHYSICS] = 0.0
    grid[:, :, CH_F]       = f
    grid[:, :, CH_K]       = k
    return jnp.array(grid), key


def build_hybrid_grid(key, H, W):
    """
    Hybrid mode: GS state with an Orbium creature dropped on top, physics_bit=0.

    This is the oracle experiment: does GS chemistry interact with a Lenia seed
    when the NCA is running in GS mode?  Does the creature survive? Dissolve?
    Turn into a GS spot? Leave a trace?

    ch0 = GS A  + Orbium seed (overlaid — max blend)
    ch1 = GS B  (intact, no Lenia supervision of ch1)
    ch13 = 0.0  (physics bit: GS — NCA runs GS rules)
    ch14 = f, ch15 = k
    """
    f, k = GS_REGIMES[GS_REGIME]
    key, sk = random.split(key)
    A, B = init_gs_grid(sk, H, W)
    for _ in range(GS_WARMUP):
        A, B = gs_step(A, B, f, k)

    # Drop Orbium onto ch0 — placed randomly, max-blend with existing GS A
    rng      = np.random.default_rng()
    A_np     = np.array(A)
    creature = seed_creature(H, W, rng)
    A_np     = np.maximum(A_np, creature)   # max so GS structure shows through

    grid = np.zeros((H, W, N_CHANNELS), dtype=np.float32)
    grid[:, :, CH_A]       = A_np
    grid[:, :, CH_B]       = np.array(B)
    grid[:, :, CH_PHYSICS] = 0.0            # GS physics bit — key experiment
    grid[:, :, CH_F]       = f
    grid[:, :, CH_K]       = k
    return jnp.array(grid), key


# ── Rendering ──────────────────────────────────────────────────────────────────

def render(surface, grid, palette, mode):
    """
    Render the NCA grid to the pygame surface.

    Lenia mode:  ch0 only (creature activation) — sparse, creatures on dark bg
    GS/Hybrid:   ch0 + ch1 combined (same as run_free "combined" mode)
    """
    A_np = np.array(grid[:, :, CH_A])
    B_np = np.array(grid[:, :, CH_B])
    pal  = np.array(palette, dtype=np.float32) / 255.0

    if mode == 'lenia':
        # Render ch0 as a bright field against a dark background.
        # Creatures appear as glowing rings/blobs.
        # Stretch the [0,1] range through the palette: 0 → dark, 1 → bright.
        t   = np.clip(A_np * 4.0, 0.0, 3.0)   # use full palette range
        idx = np.floor(t).astype(int).clip(0, 2)
        frac = (t - idx)[..., None]
        rgb = pal[idx] + frac * (pal[idx + 1] - pal[idx])
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        # GS / Hybrid: B drives brightness, A modulates depth (same as run_free)
        t   = np.clip(B_np * 3.0, 0.0, 3.0)
        idx = np.floor(t).astype(int).clip(0, 2)
        frac = (t - idx)[..., None]
        rgb = pal[idx] + frac * (pal[idx + 1] - pal[idx])
        rgb = rgb * (0.6 + 0.4 * A_np)[..., None]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    img    = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    scaled = pygame.transform.scale(img, (SCREEN_W, SCREEN_H))
    surface.blit(scaled, (0, 0))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None,
                        help='Path to lenia checkpoint pkl. Defaults to latest.')
    args = parser.parse_args()

    # ── Find checkpoint ────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if ckpt_path is None:
        print("No lenia checkpoint found in nca/checkpoints/")
        print("Run nca/train_lenia.py first (or pass --checkpoint).")
        sys.exit(1)
    print(f"Loading: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        params = pickle.load(f)
    params = jax.device_put(params)
    print("Loaded.\n")

    # ── Model ─────────────────────────────────────────────────────────────
    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn           = make_step_fn(update_net, perception_kernel)

    # ── Palette ───────────────────────────────────────────────────────────
    palette_names = list(PALETTES.keys())
    # Pick a high-contrast palette that works well for both creatures and GS
    palette_idx   = palette_names.index('cosmic') if 'cosmic' in palette_names else 0
    palette       = PALETTES[palette_names[palette_idx]]

    # ── Initial state: Lenia mode ─────────────────────────────────────────
    key  = random.PRNGKey(int(np.random.randint(0, 2**31)))
    mode = 'lenia'

    print("Building Lenia grid (Orbium seed)...")
    grid = build_lenia_grid(GRID_H, GRID_W)
    physics_bit = 1.0
    mu_or_f     = ORBIUM_MU
    sigma_or_k  = ORBIUM_SIGMA
    print("Done. Launching.\n")

    # ── Pygame ────────────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Somnivex — Lenia Checkpoint Test")
    font   = pygame.font.SysFont("monospace", 15)
    ticker = pygame.time.Clock()

    print("Controls: L=lenia  G=gs  H=hybrid  R=reseed  P=palette  ]/[=speed  Q=quit")
    print(f"Starting in Lenia mode  mu={ORBIUM_MU}  sigma={ORBIUM_SIGMA}\n")

    step_count      = 0
    steps_per_frame = STEPS_PER_FRAME
    running         = True

    while running:

        # ── Events ────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_q:
                    running = False

                if event.key == pygame.K_l:
                    mode        = 'lenia'
                    grid        = build_lenia_grid(GRID_H, GRID_W)
                    physics_bit = 1.0
                    mu_or_f     = ORBIUM_MU
                    sigma_or_k  = ORBIUM_SIGMA
                    step_count  = 0
                    print(f"→ Lenia mode  mu={ORBIUM_MU}  sigma={ORBIUM_SIGMA}")

                if event.key == pygame.K_g:
                    mode = 'gs'
                    grid, key = build_gs_grid(key, GRID_H, GRID_W)
                    f, k      = GS_REGIMES[GS_REGIME]
                    physics_bit = 0.0
                    mu_or_f     = f
                    sigma_or_k  = k
                    step_count  = 0
                    print(f"→ GS mode  regime={GS_REGIME}  f={f:.4f}  k={k:.4f}")

                if event.key == pygame.K_h:
                    mode = 'hybrid'
                    grid, key = build_hybrid_grid(key, GRID_H, GRID_W)
                    f, k      = GS_REGIMES[GS_REGIME]
                    physics_bit = 0.0
                    mu_or_f     = f
                    sigma_or_k  = k
                    step_count  = 0
                    print(f"→ Hybrid mode  GS({GS_REGIME}) + Orbium  physics_bit=0")

                if event.key == pygame.K_r:
                    # Reseed current mode
                    if mode == 'lenia':
                        grid       = build_lenia_grid(GRID_H, GRID_W)
                        step_count = 0
                        print("Reseed: Lenia")
                    elif mode == 'gs':
                        grid, key  = build_gs_grid(key, GRID_H, GRID_W)
                        step_count = 0
                        print("Reseed: GS")
                    else:
                        grid, key  = build_hybrid_grid(key, GRID_H, GRID_W)
                        step_count = 0
                        print("Reseed: Hybrid")

                if event.key == pygame.K_p:
                    palette_idx = (palette_idx + 1) % len(palette_names)
                    palette     = PALETTES[palette_names[palette_idx]]
                    print(f"Palette: {palette_names[palette_idx]}")

                if event.key == pygame.K_RIGHTBRACKET:
                    steps_per_frame = min(steps_per_frame + 1, 20)
                    print(f"Speed: {steps_per_frame} steps/frame")

                if event.key == pygame.K_LEFTBRACKET:
                    steps_per_frame = max(steps_per_frame - 1, 1)
                    print(f"Speed: {steps_per_frame} steps/frame")

        # ── NCA steps ─────────────────────────────────────────────────────
        for _ in range(steps_per_frame):
            grid, key = step_fn(grid, params, key)

            # Re-inject control channels after EVERY step — mandatory.
            # Without this, the NCA's own delta will corrupt ch13/ch14/ch15.
            # For Lenia: if sigma drifts toward 0 it means nothing here (no Lenia
            # kernel in free run), but we still keep them stable for the NCA to read.
            grid = grid.at[:, :, CH_PHYSICS].set(physics_bit)
            grid = grid.at[:, :, CH_F].set(mu_or_f)
            grid = grid.at[:, :, CH_K].set(sigma_or_k)

            step_count += 1

        # ── Diagnostics every 100 steps ───────────────────────────────────
        if step_count % 100 == 0:
            A_std  = float(jnp.std(grid[:, :, CH_A]))
            A_mean = float(jnp.mean(grid[:, :, CH_A]))
            B_std  = float(jnp.std(grid[:, :, CH_B]))
            status = "ALIVE" if A_std > 0.01 else "flat"
            print(f"  step {step_count:5d}  ch0 std={A_std:.4f} mean={A_mean:.4f}  ch1 std={B_std:.4f}  [{status}]")

        # ── Render ────────────────────────────────────────────────────────
        render(screen, grid, palette, mode)

        A_std = float(jnp.std(grid[:, :, CH_A]))
        hud_txt = (
            f"step {step_count}  |  mode={mode.upper()}  "
            f"bit={physics_bit:.0f}  f/mu={mu_or_f:.4f}  k/sig={sigma_or_k:.4f}  "
            f"ch0_std={A_std:.4f}  |  spd={steps_per_frame}  "
            f"|  L=lenia G=gs H=hybrid R=reseed P=pal ]/[=spd Q=quit"
        )
        hud = font.render(hud_txt, True, (80, 80, 80))
        screen.blit(hud, (8, 8))

        pygame.display.flip()
        ticker.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    print(f"JAX: {jax.devices()}")
    main()
```

---

## What to look for

### Lenia mode (L)
Watch ch0. The Orbium seed is a 20×20 ring placed somewhere on the 128×128 grid.

**Best case**: The ring starts moving. You see a glider traveling across the grid.
That means the NCA internalized Lenia creature dynamics.

**Plausible case**: The ring stays roughly in place but pulses or maintains its shape.
The NCA learned "maintain this distribution" but not locomotion.

**Worst case**: ch0_std → 0.0000 within 50 steps. The NCA collapsed it to zero.
This means lenia=0.00000 in training was a trivial solution (output zero everywhere).

### GS mode (G)
Watch ch0+ch1. Should look similar to run_free.py. If it does — GS is still intact.
If it's all zeros or noise — catastrophic forgetting happened.

### Hybrid mode (H)
GS pattern in ch1, Orbium ring in ch0, physics_bit=0.
What happens? Options in rough order of interestingness:
1. **Coexistence**: creature floats in GS chemistry, both preserved
2. **Annihilation**: creature dissolves into GS, GS wins (ch0→GS A pattern)
3. **Infection**: creature disrupts GS, leaves a hole or different regime zone
4. **Synthesis**: something entirely new at the boundary

---

## Notes on grid size

We train on 64×64 but run on 128×128. This is fine because:
- The NCA is a 3×3 convolution applied identically to every cell
- It doesn't know or care about the total grid size
- The Orbium creature is 20×20 — same size in both grids
- On 128×128 there's more room for the creature to travel before wrapping
