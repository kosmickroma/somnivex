# nca/run_fhn.py — FHN-driven NCA free run.
#
# Same trained NCA, different environment.
# FitzHugh-Nagumo reaction-diffusion runs live on the same grid.
# Its wave patterns drive the NCA's f/k control channels cell-by-cell.
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

FHN_STEPS_PER_NCA = 3    # FHN integration steps per NCA step — tune wave speed
FHN_WARMUP_STEPS  = 800  # FHN steps before NCA starts — lets spirals form first

FHN_AMP_F = 0.025   # how much FHN u modulates f around center
FHN_AMP_K = 0.015   # how much FHN v modulates k around center

F_MIN, F_MAX = 0.010, 0.080
K_MIN, K_MAX = 0.040, 0.075

PALETTE_CHANGE_MIN  = 1800
PALETTE_CHANGE_MAX  = 4000
PALETTE_BLEND_STEPS = 300

NCA_RENDER_MODES = ["combined", "B", "edges", "reaction", "differential", "A_inv"]
NCA_EFFECTS      = ["none", "bloom", "vignette", "chromatic", "grain", "scanlines"]

RENDER_MODE_CHANGE_MIN = 3000
RENDER_MODE_CHANGE_MAX = 8000
EFFECT_CHANGE_MIN      = 4000
EFFECT_CHANGE_MAX      = 10000

SATURATION_CHECK = 60
SATURATION_STD   = 0.010

# ── FHN presets ───────────────────────────────────────────────────────────────
FHN_PRESETS = {
    "spirals":    dict(Du=1.0, Dv=0.5,  eps=0.08, beta=0.5, gamma=0.6, dt=0.08),
    "slow_rot":   dict(Du=1.0, Dv=0.3,  eps=0.10, beta=0.6, gamma=0.6, dt=0.08),
    "pulses":     dict(Du=1.0, Dv=0.0,  eps=0.10, beta=0.7, gamma=0.5, dt=0.08),
    "turbulence": dict(Du=0.5, Dv=0.1,  eps=0.05, beta=0.3, gamma=0.5, dt=0.05),
    "excitable":  dict(Du=0.8, Dv=0.3,  eps=0.08, beta=0.6, gamma=0.6, dt=0.06),
    "ghost":      dict(Du=1.0, Dv=0.8,  eps=0.12, beta=0.4, gamma=0.7, dt=0.08),
}
FHN_PRESET_NAMES = list(FHN_PRESETS.keys())

# ── Palette groups ────────────────────────────────────────────────────────────
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

# ── FHN ───────────────────────────────────────────────────────────────────────

def fhn_laplacian(Z):
    return (
        np.roll(Z,  1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z,  1, axis=1) + np.roll(Z, -1, axis=1) - 4.0 * Z
    )

def fhn_step(u, v, Du, Dv, eps, beta, gamma, dt):
    u = np.clip(u, -3.0, 3.0)  # prevent u**3 overflow before cubing
    v = np.clip(v, -3.0, 3.0)
    du = Du * fhn_laplacian(u) + (u - u**3 / 3.0 - v) / eps
    dv = Dv * fhn_laplacian(v) + eps * (u + beta - gamma * v)
    return np.clip(u + du * dt, -3.0, 3.0), np.clip(v + dv * dt, -3.0, 3.0)

def init_fhn(H, W):
    u = np.random.uniform(-0.1, 0.1, (H, W)).astype(np.float64)
    v = np.random.uniform(-0.1, 0.1, (H, W)).astype(np.float64)
    return u, v

def fhn_to_fk(u, v, f_center, k_center):
    f_field = np.clip(f_center + np.tanh(u * 0.8) * FHN_AMP_F, F_MIN, F_MAX).astype(np.float32)
    k_field = np.clip(k_center + np.tanh(v * 0.8) * FHN_AMP_K, K_MIN, K_MAX).astype(np.float32)
    return f_field, k_field

# ── NCA ───────────────────────────────────────────────────────────────────────

def init_nca_grid(key, H, W, f, k):
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

# ── Render ────────────────────────────────────────────────────────────────────

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
        sys.exit(1)

    print(f"Loading checkpoint: {CHECKPOINT}")
    with open(CHECKPOINT, 'rb') as fh:
        params = pickle.load(fh)
    params = jax.device_put(params)
    print("Loaded.")

    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn           = make_step_fn(update_net, perception_kernel)

    regime_names       = list(GS_REGIMES.keys())
    regime_idx         = np.random.randint(0, len(regime_names))
    f_center, k_center = GS_REGIMES[regime_names[regime_idx]]
    print(f"NCA regime: {regime_names[regime_idx]}  f={f_center:.4f}  k={k_center:.4f}")

    fhn_idx    = 0
    fhn_preset = FHN_PRESETS[FHN_PRESET_NAMES[fhn_idx]]
    u, v       = init_fhn(GRID_H, GRID_W)
    print(f"FHN preset: {FHN_PRESET_NAMES[fhn_idx]}  warming up {FHN_WARMUP_STEPS} steps...")
    for _ in range(FHN_WARMUP_STEPS):
        u, v = fhn_step(u, v, **fhn_preset)
    print("FHN ready.")

    f_field, k_field = fhn_to_fk(u, v, f_center, k_center)
    jf_field = jnp.array(f_field)
    jk_field = jnp.array(k_field)

    key = random.PRNGKey(int(np.random.randint(0, 2**31)))
    print(f"Warming up GS seed...")
    grid, key = init_nca_grid(key, GRID_H, GRID_W, f_center, k_center)
    print("Launching.\n")

    palette_names = list(PALETTES.keys())
    palette_idx   = 0

    pygame.init()
    pygame.font.init()
    if DUAL_SCREEN:
        os.environ.setdefault('SDL_VIDEO_WINDOW_POS', '0,0')
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.NOFRAME)
    pygame.display.set_caption("Somnivex — FHN Drive")
    font   = pygame.font.SysFont("monospace", 16)
    ticker = pygame.time.Clock()

    print("Controls: R=reset  H=FHN preset  M=mode  E=effect  P=palette  [/]=speed  Q=quit\n")

    step_count         = 0
    running            = True
    steps_per_frame    = STEPS_PER_FRAME
    auto_nudges        = 0

    render_mode_idx    = 0
    effect_idx         = 0
    render_mode        = NCA_RENDER_MODES[render_mode_idx]
    effect             = NCA_EFFECTS[effect_idx]
    next_mode_change   = np.random.randint(RENDER_MODE_CHANGE_MIN, RENDER_MODE_CHANGE_MAX)
    next_effect_change = np.random.randint(EFFECT_CHANGE_MIN, EFFECT_CHANGE_MAX)

    palette_current     = np.array(PALETTES[palette_names[palette_idx]], dtype=np.float32)
    palette_target      = palette_current.copy()
    palette_blend       = 0
    next_palette_change = np.random.randint(PALETTE_CHANGE_MIN, PALETTE_CHANGE_MAX)

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_q:
                    running = False

                if event.key == pygame.K_r:
                    regime_idx         = np.random.randint(0, len(regime_names))
                    f_center, k_center = GS_REGIMES[regime_names[regime_idx]]
                    key, sk            = random.split(key)
                    print(f"Reset → {regime_names[regime_idx]}  f={f_center:.4f}  k={k_center:.4f}")
                    grid, key          = init_nca_grid(sk, GRID_H, GRID_W, f_center, k_center)
                    u, v               = init_fhn(GRID_H, GRID_W)
                    for _ in range(FHN_WARMUP_STEPS):
                        u, v = fhn_step(u, v, **fhn_preset)
                    f_field, k_field   = fhn_to_fk(u, v, f_center, k_center)
                    jf_field           = jnp.array(f_field)
                    jk_field           = jnp.array(k_field)
                    step_count         = 0
                    auto_nudges        = 0

                if event.key == pygame.K_h:
                    fhn_idx    = (fhn_idx + 1) % len(FHN_PRESET_NAMES)
                    fhn_preset = FHN_PRESETS[FHN_PRESET_NAMES[fhn_idx]]
                    print(f"FHN → {FHN_PRESET_NAMES[fhn_idx]}")

                if event.key == pygame.K_p:
                    new_name        = pick_next_palette(palette_names[palette_idx], palette_names)
                    palette_idx     = palette_names.index(new_name)
                    palette_target  = np.array(PALETTES[new_name], dtype=np.float32)
                    palette_blend   = 1
                    print(f"Palette: {new_name}")

                if event.key == pygame.K_m:
                    render_mode_idx = (render_mode_idx + 1) % len(NCA_RENDER_MODES)
                    render_mode     = NCA_RENDER_MODES[render_mode_idx]
                    print(f"Mode: {render_mode}")

                if event.key == pygame.K_e:
                    effect_idx = (effect_idx + 1) % len(NCA_EFFECTS)
                    effect     = NCA_EFFECTS[effect_idx]
                    print(f"Effect: {effect}")

                if event.key == pygame.K_RIGHTBRACKET:
                    steps_per_frame = min(steps_per_frame + 1, 20)
                    print(f"Speed: {steps_per_frame}")

                if event.key == pygame.K_LEFTBRACKET:
                    steps_per_frame = max(steps_per_frame - 1, 1)
                    print(f"Speed: {steps_per_frame}")

        # FHN steps
        for _ in range(FHN_STEPS_PER_NCA * steps_per_frame):
            u, v = fhn_step(u, v, **fhn_preset)
        f_field, k_field = fhn_to_fk(u, v, f_center, k_center)
        jf_field = jnp.array(f_field)
        jk_field = jnp.array(k_field)

        # NCA steps
        for _ in range(steps_per_frame):
            grid, key = step_fn(grid, params, key)
            grid = grid.at[:, :, CH_F].set(jf_field)
            grid = grid.at[:, :, CH_K].set(jk_field)
            step_count += 1

        # Saturation guard
        if step_count % SATURATION_CHECK == 0:
            b_std = float(jnp.std(grid[:, :, CH_B]))
            if b_std < SATURATION_STD:
                key, sk   = random.split(key)
                grid, key = init_nca_grid(sk, GRID_H, GRID_W, f_center, k_center)
                auto_nudges += 1
                print(f"Solid screen (std={b_std:.4f}) → reseed #{auto_nudges}")

        if step_count >= next_mode_change:
            render_mode_idx  = (render_mode_idx + 1) % len(NCA_RENDER_MODES)
            render_mode      = NCA_RENDER_MODES[render_mode_idx]
            next_mode_change = step_count + np.random.randint(RENDER_MODE_CHANGE_MIN, RENDER_MODE_CHANGE_MAX)
            print(f"Auto mode → {render_mode}")

        if step_count >= next_effect_change:
            effect_idx         = (effect_idx + 1) % len(NCA_EFFECTS)
            effect             = NCA_EFFECTS[effect_idx]
            next_effect_change = step_count + np.random.randint(EFFECT_CHANGE_MIN, EFFECT_CHANGE_MAX)
            print(f"Auto effect → {effect}")

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

        render(screen, grid, blended_palette.astype(np.uint8).tolist(), render_mode, effect)

        hud = font.render(
            f"step {step_count}  |  FHN={FHN_PRESET_NAMES[fhn_idx]}  |  f={f_center:.4f} k={k_center:.4f}  |  {palette_names[palette_idx]}  |  {render_mode}+{effect}  |  spd={steps_per_frame}  |  H=fhn R=reset M=mode E=effect P=pal Q=quit",
            True, (80, 80, 80)
        )
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        ticker.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    print(f"JAX devices: {jax.devices()}")
    run()
