# kktodo/05_run_fhn.md — FHN-driven NCA free run

## What this is

New file: `nca/run_fhn.py`

Same trained NCA model. Same checkpoint. Same rendering pipeline.

The difference: instead of a static sine-wave spatial f/k field, a live
**FitzHugh-Nagumo** reaction-diffusion simulation runs on the same grid and
its wave patterns drive the NCA's f/k control channels cell-by-cell.

FHN produces traveling waves — spirals, pulses, rotating arms. As each wave
sweeps through a region, that region's f/k shifts, pushing the NCA toward a
different GS regime. The NCA (trained on GS) now lives in a landscape that
breathes and rotates with FHN rhythm. The two systems interact. Neither is
in charge.

`run_free.py` is untouched. This is a separate experiment.

---

## FitzHugh-Nagumo equations

Two variables per cell:

```
∂u/∂t = Du·∇²u + (1/ε)·(u - u³/3 - v)
∂v/∂t = Dv·∇²v + ε·(u + β - γ·v)
```

- `u` = fast activator (like membrane voltage)
- `v` = slow inhibitor (like recovery)
- `Du`, `Dv` = diffusion coefficients
- `ε` = time-scale separation (small = v changes slowly = sharper waves)
- `β`, `γ` = recovery parameters

The interesting behaviors:
- Small ε + appropriate β/γ → sharp wavefronts, spirals, target waves
- Larger ε → softer oscillations
- Dv=0 → v doesn't diffuse, produces traveling pulses more than spirals

How it drives NCA:
- `u` modulates `f` (food parameter)
- `v` modulates `k` (kill parameter)
- FHN output is normalized via tanh, then scaled to ±AMP around a center value
- NCA's f/k field is now alive and moving

---

## File to create: `nca/run_fhn.py`

```python
# nca/run_fhn.py — FHN-driven NCA free run.
#
# Same trained NCA, different environment.
# FitzHugh-Nagumo reaction-diffusion runs live on the same grid.
# Its wave patterns drive the NCA's f/k control channels cell-by-cell.
# The NCA (trained on static GS) now inhabits a breathing, rotating landscape.
#
# Run from project root:
#     python nca/run_fhn.py
#
# Controls: R=reset  H=cycle FHN preset  M=render mode  E=effect
#           P=palette  [/]=speed  Q=quit

import os
import sys
import pickle
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
from gs.engine import GS_REGIMES, gs_step, init_gs_grid
from nca.params import PALETTES
from display.windows import compute_heat, apply_palette_heat, apply_effect

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'params_050000.pkl'
)

GRID_H          = 256
GRID_W          = 256
SCREEN_W        = 1920
SCREEN_H        = 1080
DUAL_SCREEN     = True
DISPLAY_W       = SCREEN_W * 2 if DUAL_SCREEN else SCREEN_W
DISPLAY_H       = SCREEN_H
FPS             = 30
STEPS_PER_FRAME = 5

GS_WARMUP_MIN = 50
GS_WARMUP_MAX = 800

# How many FHN integration steps to run per NCA step.
# More = FHN waves move faster relative to NCA patterns.
# Start at 3. Try 1-8 to taste.
FHN_STEPS_PER_NCA = 3

# FHN warmup: run this many FHN steps silently before starting NCA.
# Gives the FHN time to self-organize into spirals before the show begins.
FHN_WARMUP_STEPS = 800

# How much FHN modulates f and k around their center values.
# Too high = NCA gets stranded outside its trained range.
# Too low = FHN has no effect.
FHN_AMP_F = 0.025
FHN_AMP_K = 0.015

# NCA f/k center — the baseline the FHN modulates around.
# Start in the middle of the trained range.
F_CENTER = 0.040
K_CENTER = 0.058
F_MIN, F_MAX = 0.010, 0.080
K_MIN, K_MAX = 0.040, 0.075

# Palette crossfade
PALETTE_CHANGE_MIN  = 1800
PALETTE_CHANGE_MAX  = 4000
PALETTE_BLEND_STEPS = 300

# Render modes + effects — same as run_free.py
NCA_RENDER_MODES = ["combined", "B", "edges", "reaction", "differential", "A_inv"]
NCA_EFFECTS      = ["none", "bloom", "vignette", "chromatic", "grain", "scanlines"]

RENDER_MODE_CHANGE_MIN = 3000
RENDER_MODE_CHANGE_MAX = 8000
EFFECT_CHANGE_MIN      = 4000
EFFECT_CHANGE_MAX      = 10000

# Saturation guard — same logic as run_free.py
SATURATION_CHECK = 60
SATURATION_STD   = 0.010

# ── FHN parameter presets ─────────────────────────────────────────────────────
# Each preset produces a different wave character.
# H key cycles through them at runtime.
#
# Fields: Du, Dv, eps, beta, gamma, dt
#   Du   — activator diffusion (keep at 1.0 unless experimenting)
#   Dv   — inhibitor diffusion (0 = pulses, 0.5+ = spirals)
#   eps  — time-scale separation (smaller = sharper, faster wavefronts)
#   beta — recovery offset (higher = easier to excite)
#   gamma— recovery rate (higher = faster recovery, shorter waves)
#   dt   — integration step (lower = more stable, slower waves visually)

FHN_PRESETS = {
    "spirals":    dict(Du=1.0, Dv=0.5,  eps=0.08, beta=0.5, gamma=0.6, dt=0.08),
    "slow_rot":   dict(Du=1.0, Dv=0.3,  eps=0.10, beta=0.6, gamma=0.6, dt=0.08),
    "pulses":     dict(Du=1.0, Dv=0.0,  eps=0.10, beta=0.7, gamma=0.5, dt=0.08),
    "turbulence": dict(Du=0.5, Dv=0.1,  eps=0.05, beta=0.3, gamma=0.5, dt=0.05),
    "excitable":  dict(Du=0.8, Dv=0.3,  eps=0.08, beta=0.6, gamma=0.6, dt=0.06),
    "ghost":      dict(Du=1.0, Dv=0.8,  eps=0.12, beta=0.4, gamma=0.7, dt=0.08),
}
FHN_PRESET_NAMES = list(FHN_PRESETS.keys())

# ── Palette groups (same as run_free.py) ──────────────────────────────────────
PALETTE_GROUPS = {
    'dark':   ['void', 'event_horizon', 'kraken_ink', 'shadow_realm', 'black_water',
               'obsidian', 'oil_slick_dark', 'abyssal', 'inferno', 'deep_crimson'],
    'warm':   ['sunset_fire', 'amber_ember', 'molten_gold', 'solar_flare', 'magma_ocean',
               'blood_moon', 'coal_ember', 'dwarven_forge', 'phoenix', 'thermal_vent',
               'molten_steel', 'rust_iron', 'rust_decay', 'jasper', 'white_phosphor',
               'candlelight', 'golden_hour', 'solar_wind', 'dragon_fire'],
    'cool':   ['deep_ocean', 'ice_cave', 'ghost', 'titanium', 'storm_grey', 'arctic_melt',
               'blue_hour', 'blue_flame', 'welding_arc', 'quasar_jet', 'pulsar',
               'lightning_storm', 'fog_bank', 'factory_smoke', 'shale', 'angelic',
               'terminal_cyan', 'bone_dust'],
    'green':  ['forest_floor', 'acid', 'deep_jungle', 'neon_moss', 'copper_verdigris',
               'malachite', 'circuit_trace', 'terminal_green', 'phosphor_green',
               'fungal_glow', 'radiation', 'northern_lights', 'cell_wall', 'aurora',
               'toxic', 'deep_bio'],
    'cosmic': ['cosmic', 'void_bloom', 'andromeda', 'stardust', 'amethyst', 'acid_wash',
               'uv_rave', 'plasma_arc', 'vhs_bleed', 'nebula_red', 'oil_slick',
               'blood_vessel', 'holographic', 'sunset_lavender'],
    'vivid':  ['candy_chrome', 'neon_city', 'sakura', 'pollen_burst', 'terminal_amber',
               'pyrite', 'monochrome', 'sandstone', 'spore_cloud', 'mycelium',
               'dust_storm', 'bioluminescent_bay', 'tide_pool'],
}
_PAL_TO_GROUP = {p: g for g, pals in PALETTE_GROUPS.items() for p in pals}

def pick_next_palette(current_name, all_names):
    group = _PAL_TO_GROUP.get(current_name)
    if group and np.random.random() < 0.70:
        candidates = [p for p in PALETTE_GROUPS[group] if p in PALETTES and p != current_name]
        if candidates:
            return np.random.choice(candidates)
    return np.random.choice(all_names)

# ── FHN simulation ────────────────────────────────────────────────────────────

def fhn_laplacian(Z):
    """
    2D Laplacian using 5-point finite difference stencil.
    Periodic (wrap-around) boundary conditions — waves leave one edge and
    re-enter the other, so the grid feels infinite.
    """
    return (
        np.roll(Z,  1, axis=0) +
        np.roll(Z, -1, axis=0) +
        np.roll(Z,  1, axis=1) +
        np.roll(Z, -1, axis=1) -
        4.0 * Z
    )

def fhn_step(u, v, Du, Dv, eps, beta, gamma, dt):
    """
    One Euler integration step of FitzHugh-Nagumo.

    du/dt = Du·∇²u + (1/eps)·(u - u³/3 - v)
    dv/dt = Dv·∇²v + eps·(u + beta - gamma·v)

    u is the fast activator — drives sharp wavefronts.
    v is the slow inhibitor — creates the refractory period behind each wave.
    Together they produce the rotating/traveling wave behavior.
    """
    lap_u = fhn_laplacian(u)
    lap_v = fhn_laplacian(v)

    du = Du * lap_u + (u - u**3 / 3.0 - v) / eps
    dv = Dv * lap_v + eps * (u + beta - gamma * v)

    return u + du * dt, v + dv * dt

def init_fhn(H, W):
    """
    Initialize FHN with low-amplitude random noise.
    Small perturbations around the rest state (u≈0, v≈0) will self-organize
    into spirals and waves over the warmup period.
    """
    u = np.random.uniform(-0.2, 0.2, (H, W)).astype(np.float32)
    v = np.random.uniform(-0.2, 0.2, (H, W)).astype(np.float32)
    return u, v

def fhn_to_fk(u, v, f_center, k_center):
    """
    Map FHN state to NCA f/k control fields.

    tanh squeezes FHN output to [-1, 1] regardless of amplitude.
    Then scale by AMP and add to center value.
    Result: f/k fields that breathe and rotate with the FHN waves.
    """
    u_norm = np.tanh(u * 0.8)   # u roughly in [-2, 2] → tanh → [-1, 1]
    v_norm = np.tanh(v * 0.8)

    f_field = np.clip(
        f_center + u_norm * FHN_AMP_F,
        F_MIN, F_MAX
    ).astype(np.float32)

    k_field = np.clip(
        k_center + v_norm * FHN_AMP_K,
        K_MIN, K_MAX
    ).astype(np.float32)

    return f_field, k_field

# ── NCA grid init ─────────────────────────────────────────────────────────────

def init_nca_grid(key, H, W, f, k):
    """
    Seed NCA from a warmed-up GS state. Same as run_free.py.
    """
    key, sk = random.split(key)
    A, B = init_gs_grid(sk, H, W)

    warmup = int(np.random.randint(GS_WARMUP_MIN, GS_WARMUP_MAX))
    for _ in range(warmup):
        A, B = gs_step(A, B, f, k)

    grid = jnp.zeros((H, W, N_CHANNELS))
    grid = grid.at[:, :, CH_A].set(A)
    grid = grid.at[:, :, CH_B].set(B)
    grid = grid.at[:, :, CH_F].set(f)
    grid = grid.at[:, :, CH_K].set(k)

    return grid, key

# ── Rendering ─────────────────────────────────────────────────────────────────

def render(surface, grid, palette, render_mode="combined", effect="none"):
    A_np = np.array(grid[:, :, CH_A])
    B_np = np.array(grid[:, :, CH_B])
    pal  = np.array(palette, dtype=np.float32)

    if render_mode == "combined":
        p    = pal / 255.0
        t    = np.clip(B_np * 3.0, 0.0, 3.0)
        idx  = np.floor(t).astype(int).clip(0, 2)
        frac = (t - idx)[..., None]
        rgb  = p[idx] + frac * (p[idx + 1] - p[idx])
        rgb  = rgb * (0.6 + 0.4 * A_np)[..., None]
        rgb  = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        heat = compute_heat(A_np, B_np, render_mode)
        rgb  = apply_palette_heat(heat, pal)

    rgb = apply_effect(rgb, effect)

    img = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    if DUAL_SCREEN:
        scaled = pygame.transform.scale(img, (SCREEN_W, SCREEN_H))
        surface.blit(scaled, (0, 0))
        surface.blit(scaled, (SCREEN_W, 0))
    else:
        scaled = pygame.transform.scale(img, (DISPLAY_W, DISPLAY_H))
        surface.blit(scaled, (0, 0))

# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Run nca/train.py first.")
        sys.exit(1)

    print(f"Loading checkpoint: {CHECKPOINT}")
    with open(CHECKPOINT, 'rb') as fh:
        params = pickle.load(fh)
    params = jax.device_put(params)
    print("Loaded.")

    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn           = make_step_fn(update_net, perception_kernel)

    # ── Starting NCA regime (center point for FHN modulation) ─────────────
    regime_names = list(GS_REGIMES.keys())
    regime_idx   = np.random.randint(0, len(regime_names))
    f_center, k_center = GS_REGIMES[regime_names[regime_idx]]
    print(f"Starting NCA regime: {regime_names[regime_idx]}  f={f_center:.4f}  k={k_center:.4f}")

    # ── FHN init ──────────────────────────────────────────────────────────
    fhn_idx    = 0
    fhn_preset = FHN_PRESETS[FHN_PRESET_NAMES[fhn_idx]]
    u, v       = init_fhn(GRID_H, GRID_W)

    print(f"FHN preset: {FHN_PRESET_NAMES[fhn_idx]}")
    print(f"Warming up FHN ({FHN_WARMUP_STEPS} steps)...")
    for _ in range(FHN_WARMUP_STEPS):
        u, v = fhn_step(u, v, **fhn_preset)
    print("FHN ready.")

    # ── Initial f/k fields from FHN ───────────────────────────────────────
    f_field, k_field = fhn_to_fk(u, v, f_center, k_center)
    jf_field = jnp.array(f_field)
    jk_field = jnp.array(k_field)

    # ── NCA grid init ─────────────────────────────────────────────────────
    key = random.PRNGKey(int(np.random.randint(0, 2**31)))
    print(f"Warming up GS seed ({GS_WARMUP_MIN}–{GS_WARMUP_MAX} random steps)...")
    grid, key = init_nca_grid(key, GRID_H, GRID_W, f_center, k_center)
    print("Done. Launching.\n")

    # ── Palette ───────────────────────────────────────────────────────────
    palette_names = list(PALETTES.keys())
    palette_idx   = 0
    palette       = PALETTES[palette_names[palette_idx]]

    # ── Pygame ────────────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    if DUAL_SCREEN:
        os.environ.setdefault('SDL_VIDEO_WINDOW_POS', '0,0')
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.NOFRAME)
    pygame.display.set_caption("Somnivex — FHN Drive")
    font   = pygame.font.SysFont("monospace", 16)
    ticker = pygame.time.Clock()

    print(f"Controls: R=reset  H=FHN preset  M=mode  E=effect  P=palette  [/]=speed  Q=quit\n")

    step_count      = 0
    running         = True
    steps_per_frame = STEPS_PER_FRAME
    auto_nudges     = 0

    # Render mode + effect state
    render_mode_idx    = 0
    effect_idx         = 0
    render_mode        = NCA_RENDER_MODES[render_mode_idx]
    effect             = NCA_EFFECTS[effect_idx]
    next_mode_change   = np.random.randint(RENDER_MODE_CHANGE_MIN, RENDER_MODE_CHANGE_MAX)
    next_effect_change = np.random.randint(EFFECT_CHANGE_MIN, EFFECT_CHANGE_MAX)

    # Palette crossfade state
    palette_current     = np.array(palette, dtype=np.float32)
    palette_target      = palette_current.copy()
    palette_blend       = 0
    next_palette_change = np.random.randint(PALETTE_CHANGE_MIN, PALETTE_CHANGE_MAX)

    while running:

        # ── Events ────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_q:
                    running = False

                if event.key == pygame.K_r:
                    # Reset both NCA and FHN
                    regime_idx     = np.random.randint(0, len(regime_names))
                    f_center, k_center = GS_REGIMES[regime_names[regime_idx]]
                    key, sk        = random.split(key)
                    print(f"Resetting → {regime_names[regime_idx]}  f={f_center:.4f}  k={k_center:.4f}")
                    grid, key      = init_nca_grid(sk, GRID_H, GRID_W, f_center, k_center)
                    u, v           = init_fhn(GRID_H, GRID_W)
                    # Give FHN a quick re-warmup
                    for _ in range(FHN_WARMUP_STEPS):
                        u, v = fhn_step(u, v, **fhn_preset)
                    f_field, k_field = fhn_to_fk(u, v, f_center, k_center)
                    jf_field       = jnp.array(f_field)
                    jk_field       = jnp.array(k_field)
                    step_count     = 0
                    auto_nudges    = 0
                    print("Reset done.")

                if event.key == pygame.K_h:
                    # Cycle FHN preset — changes the wave character
                    fhn_idx    = (fhn_idx + 1) % len(FHN_PRESET_NAMES)
                    fhn_preset = FHN_PRESETS[FHN_PRESET_NAMES[fhn_idx]]
                    print(f"FHN preset → {FHN_PRESET_NAMES[fhn_idx]}")

                if event.key == pygame.K_p:
                    new_name       = pick_next_palette(palette_names[palette_idx], palette_names)
                    palette_idx    = palette_names.index(new_name)
                    palette_target = np.array(PALETTES[new_name], dtype=np.float32)
                    palette_blend  = 1
                    print(f"Palette: {new_name}")

                if event.key == pygame.K_m:
                    render_mode_idx = (render_mode_idx + 1) % len(NCA_RENDER_MODES)
                    render_mode     = NCA_RENDER_MODES[render_mode_idx]
                    print(f"Render mode: {render_mode}")

                if event.key == pygame.K_e:
                    effect_idx = (effect_idx + 1) % len(NCA_EFFECTS)
                    effect     = NCA_EFFECTS[effect_idx]
                    print(f"Effect: {effect}")

                if event.key == pygame.K_RIGHTBRACKET:
                    steps_per_frame = min(steps_per_frame + 1, 20)
                    print(f"Speed: {steps_per_frame} steps/frame")

                if event.key == pygame.K_LEFTBRACKET:
                    steps_per_frame = max(steps_per_frame - 1, 1)
                    print(f"Speed: {steps_per_frame} steps/frame")

        # ── FHN steps ─────────────────────────────────────────────────────
        # Run FHN_STEPS_PER_NCA integration steps per NCA step.
        # FHN is pure numpy on CPU — cheap. NCA is JAX on GPU — expensive.
        # This ratio controls how fast the FHN waves move relative to NCA.
        fhn_steps_this_frame = FHN_STEPS_PER_NCA * steps_per_frame
        for _ in range(fhn_steps_this_frame):
            u, v = fhn_step(u, v, **fhn_preset)

        # Recompute f/k fields from updated FHN state
        f_field, k_field = fhn_to_fk(u, v, f_center, k_center)
        jf_field = jnp.array(f_field)
        jk_field = jnp.array(k_field)

        # ── NCA steps ─────────────────────────────────────────────────────
        for _ in range(steps_per_frame):
            grid, key = step_fn(grid, params, key)
            # Inject FHN-driven f/k — overwrite whatever NCA wrote to those channels
            grid = grid.at[:, :, CH_F].set(jf_field)
            grid = grid.at[:, :, CH_K].set(jk_field)
            step_count += 1

        # ── Saturation guard ──────────────────────────────────────────────
        if step_count % SATURATION_CHECK == 0:
            b_std = float(jnp.std(grid[:, :, CH_B]))
            if b_std < SATURATION_STD:
                f_safe    = float(np.random.uniform(0.025, 0.060))
                k_safe    = float(np.random.uniform(0.050, 0.065))
                key, sk   = random.split(key)
                grid, key = init_nca_grid(sk, GRID_H, GRID_W, f_safe, k_safe)
                auto_nudges += 1
                print(f"Solid screen (std={b_std:.4f}) → reseed #{auto_nudges}")

        # ── Auto render mode rotation ──────────────────────────────────────
        if step_count >= next_mode_change:
            render_mode_idx  = (render_mode_idx + 1) % len(NCA_RENDER_MODES)
            render_mode      = NCA_RENDER_MODES[render_mode_idx]
            next_mode_change = step_count + np.random.randint(RENDER_MODE_CHANGE_MIN, RENDER_MODE_CHANGE_MAX)
            print(f"Auto render mode → {render_mode}")

        # ── Auto effect rotation ───────────────────────────────────────────
        if step_count >= next_effect_change:
            effect_idx         = (effect_idx + 1) % len(NCA_EFFECTS)
            effect             = NCA_EFFECTS[effect_idx]
            next_effect_change = step_count + np.random.randint(EFFECT_CHANGE_MIN, EFFECT_CHANGE_MAX)
            print(f"Auto effect → {effect}")

        # ── Autonomous palette crossfade ───────────────────────────────────
        if step_count >= next_palette_change and palette_blend == 0:
            new_name            = pick_next_palette(palette_names[palette_idx], palette_names)
            palette_idx         = palette_names.index(new_name)
            palette_target      = np.array(PALETTES[new_name], dtype=np.float32)
            palette_blend       = 1
            next_palette_change = step_count + np.random.randint(PALETTE_CHANGE_MIN, PALETTE_CHANGE_MAX)
            print(f"Palette → {new_name}")

        if palette_blend > 0:
            t               = palette_blend / PALETTE_BLEND_STEPS
            blended_palette = (palette_current * (1 - t) + palette_target * t).clip(0, 255)
            palette_blend  += 1
            if palette_blend >= PALETTE_BLEND_STEPS:
                palette_current = palette_target.copy()
                palette_blend   = 0
        else:
            blended_palette = palette_current

        # ── Render ────────────────────────────────────────────────────────
        render(screen, grid, blended_palette.astype(np.uint8).tolist(), render_mode, effect)

        hud = font.render(
            f"step {step_count}  |  FHN={FHN_PRESET_NAMES[fhn_idx]}  |  NCA f={f_center:.4f} k={k_center:.4f}  |  {palette_names[palette_idx]}  |  {render_mode}+{effect}  |  spd={steps_per_frame}  |  H=fhn R=reset M=mode E=effect P=pal Q=quit",
            True, (80, 80, 80)
        )
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        ticker.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    print(f"JAX devices: {jax.devices()}")
    run()
```

---

## What to expect when you run it

**First 30 seconds:** FHN spirals haven't fully developed yet, NCA just sees
gently modulated f/k. Looks similar to run_free.py.

**After ~1 minute:** FHN spirals are rotating. You'll see the NCA's patterns
change character as wave fronts sweep through — brief transitions between
GS regimes, following the FHN geometry. Regions behind a wavefront behave
differently from regions ahead of it.

**To tune the feel:**
- `FHN_AMP_F` / `FHN_AMP_K` — bigger = FHN has more control over NCA (try 0.01–0.04)
- `FHN_STEPS_PER_NCA` — bigger = FHN waves move faster (try 1–10)
- `H` key — cycle presets while running, find your favorite wave character
- "spirals" and "slow_rot" will be the most visually interesting starters

**If it looks chaotic/collapsed:** lower `FHN_AMP_F` and `FHN_AMP_K`. The NCA
is being pushed outside its trained range too aggressively.

**If FHN seems invisible:** raise `FHN_AMP_F` / `FHN_AMP_K`.

---

## Checklist

- [ ] Create `nca/run_fhn.py` with the code above
- [ ] Run it: `python nca/run_fhn.py`
- [ ] Watch for ~2 minutes, let FHN develop
- [ ] Try H key to cycle FHN presets
- [ ] Report back: what does it look like?
