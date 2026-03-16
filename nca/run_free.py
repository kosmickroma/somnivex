# nca/run_free.py - Run the trained NCA freely, no GS needed.
#
# Loads a trained checkpoint and runs the NCA on its own output forever.
# Seeds from a warmed-up GS state so the NCA starts mid-reaction (not blobs).
# Autonomously drifts f/k over time so it never gets stuck.
# Auto-detects saturation and nudges the regime to escape.
#
# Run from the project root with:
#     python nca/run_free.py
#
# Controls: R=reset  F=cycle regime  P=palette  Q=quit

import os
import sys
import pickle
import numpy as np
import pygame
import jax
import jax.numpy as jnp
from jax import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca.model import(
    N_CHANNELS, N_FILTERS,
    CH_A, CH_B, CH_F, CH_K,
    UpdateNet, make_perception_kernel, make_step_fn,
)
from nca.lenia import CH_PHYSICS
from gs.engine import GS_REGIMES, gs_step, init_gs_grid
from nca.params import PALETTES
from display.windows import compute_heat, apply_palette_heat, apply_effect
from nca.sound import SoundEngine

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'lenia_050000.pkl'
)
GS_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'params_050000.pkl'
)

# Physics bit — 0.0 = GS mode, 1.0 = Lenia mode.
# The fused model absorbed both. Start at 0 (GS) where the interesting
# pulsing ring behavior lives. Press T to flip live and watch the transition.
PHYSICS_BIT = 0.0

# Quiet mode — disables all autonomous pokes, extreme bursts, and f/k drift.
# Set True to watch what the model does completely on its own.
# Set False to restore the full autonomous behaviour.
QUIET_MODE = True

SOUND_ENABLED = True   # set False to disable audio entirely

GRID_H          = 256
GRID_W          = 256
SCREEN_W        = 1920  # one monitor width
SCREEN_H        = 1080
DUAL_SCREEN     = False  # set False for single monitor
DISPLAY_W       = SCREEN_W * 2 if DUAL_SCREEN else SCREEN_W
DISPLAY_H       = SCREEN_H
FPS             = 30
STEPS_PER_FRAME = 5   # NCA steps per rendered frame — higher = faster evolution

GS_WARMUP_MIN      = 50    # short warmup = raw early chaos
GS_WARMUP_MAX      = 800   # long warmup = fully developed structure
DRIFT_EVERY        = 400   # steps between gentle f/k nudges
DRIFT_AMOUNT       = 0.004 # size of each nudge
F_MIN, F_MAX       = 0.01, 0.08
K_MIN, K_MAX       = 0.04, 0.075
SATURATION_CHECK   = 60    # steps between saturation checks (~2s at 30fps)
SATURATION_STD     = 0.010 # B std below this = truly solid screen, reseed

# Autonomous palette crossfading
PALETTE_CHANGE_MIN = 1800  # min steps between palette transitions
PALETTE_CHANGE_MAX = 4000  # max steps between palette transitions
PALETTE_BLEND_STEPS = 300  # steps to crossfade old → new (~10s at 30fps)

# Extreme regimes — furthest apart in f/k space + beyond training range
# The NCA has to extrapolate when given values it never trained on
EXTREME_REGIMES = [
    ("uskate",       0.010, 0.047),  # lowest f in training set
    ("coral",        0.060, 0.062),  # highest f in training set
    ("bacteria",     0.046, 0.065),  # highest k in training set
    ("beyond_low",   0.005, 0.040),  # below training range — uncharted
    ("beyond_high",  0.075, 0.070),  # above training range — uncharted
    ("beyond_wild",  0.008, 0.075),  # extreme diagonal — never seen
]

# Perturbation sequences — the main exploration mechanic.
# Every few minutes, poke the control channels 3-6 times with random spacing.
# Each poke disturbs the pattern mid-reaction. The NCA heals and reorganizes.
# Space them out — too fast = solid screen. Too slow = nothing happens.
PERTURB_INTERVAL_MIN = 1500  # min steps between sequences (~50s at 30fps)
PERTURB_INTERVAL_MAX = 3600  # max steps between sequences (~2min at 30fps)
PERTURB_POKES_MIN    = 3     # min pokes per sequence
PERTURB_POKES_MAX    = 6     # max pokes per sequence
PERTURB_SPACING_MIN  = 10    # min steps between pokes in a sequence
PERTURB_SPACING_MAX  = 30    # max steps between pokes in a sequence

EXTREME_INTERVAL_MIN = 4000  # min steps between autonomous extreme bursts (~2min)
EXTREME_INTERVAL_MAX = 9000  # max steps between autonomous extreme bursts (~5min)

RESEED_INTERVAL_MIN  = 8000  # min steps between autonomous reseeds (~4min)
RESEED_INTERVAL_MAX  = 18000 # max steps between autonomous reseeds (~10min)

# Render modes — different ways to read the NCA state
# "combined" is the NCA-specific A+B blend. The rest come from the GS pipeline.
NCA_RENDER_MODES = ["combined", "B", "edges", "reaction", "differential", "A_inv"]
NCA_EFFECTS      = ["none", "bloom", "vignette", "chromatic", "grain", "scanlines"]

RENDER_MODE_CHANGE_MIN = 3000  # steps between auto render mode changes
RENDER_MODE_CHANGE_MAX = 8000
EFFECT_CHANGE_MIN      = 4000  # steps between auto effect changes
EFFECT_CHANGE_MAX      = 10000

# Palette groups — when crossfading, 70% chance stay in same family for continuity
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
# Reverse map: palette name → group
_PAL_TO_GROUP = {p: g for g, pals in PALETTE_GROUPS.items() for p in pals}

def pick_next_palette(current_name, all_names):
    """70% chance: pick from same color family. 30%: pick anything."""
    group = _PAL_TO_GROUP.get(current_name)
    if group and np.random.random() < 0.70:
        candidates = [p for p in PALETTE_GROUPS[group] if p in PALETTES and p != current_name]
        if candidates:
            return np.random.choice(candidates)
    return np.random.choice(all_names)

# Spatial f/k variation — each cell gets its own f/k from a drifting noise field.
# Different regions behave in different parameter regimes simultaneously.
# The whole grid can never collapse to one state because regions are always in
# different territory. The field drifts slowly — patterns reorganize continuously.
FK_SPATIAL_AMP_F  = 0.015   # half-amplitude of f variation across grid
FK_SPATIAL_AMP_K  = 0.010   # half-amplitude of k variation across grid
FK_PHASE_DRIFT    = 0.0008  # phase advance per NCA step (one full cycle ≈ 7800 steps)

# ── Spatial field ─────────────────────────────────────────────────────────────
def make_fk_field(H, W, f_center, k_center, phase_fx, phase_fy, phase_kx, phase_ky):
    """
    Generate smooth 2D f and k arrays using sum-of-sines.
    Each cell gets its own f/k — regions live in different parameter regimes.
    The phase parameters drift slowly over time, shifting which regions get
    which behavior without any hard transitions.
    """
    xs = np.linspace(0, 2 * np.pi, W, endpoint=False)
    ys = np.linspace(0, 2 * np.pi, H, endpoint=False)
    xx, yy = np.meshgrid(xs, ys)  # (H, W)

    # Two overlapping sine waves per field — different frequencies and angles
    # so the resulting pattern has interesting large-scale structure
    f_noise = (
        np.sin(xx * 1.3 + phase_fx) * np.cos(yy * 0.9 + phase_fy) * 0.6 +
        np.cos(xx * 0.7 + phase_fy * 0.5) * np.sin(yy * 1.1 + phase_fx * 0.7) * 0.4
    )  # range roughly -1 to 1

    k_noise = (
        np.sin(xx * 0.8 + phase_kx + 1.0) * np.cos(yy * 1.2 + phase_ky) * 0.6 +
        np.cos(xx * 1.1 + phase_ky * 0.4) * np.sin(yy * 0.7 + phase_kx * 0.8) * 0.4
    )

    f_field = np.clip(f_center + f_noise * FK_SPATIAL_AMP_F, F_MIN, F_MAX).astype(np.float32)
    k_field = np.clip(k_center + k_noise * FK_SPATIAL_AMP_K, K_MIN, K_MAX).astype(np.float32)
    return f_field, k_field

# ── Grid initialization ───────────────────────────────────────────────────────
def init_nca_grid(key, H, W, f, k):
    """
    Build a fresh NCA grid seeded from a warmed-up GS simulation.

    Instead of dropping raw patches (which produce blobs in most regimes),
    we run GS_WARMUP_STEPS of real GS first. The NCA starts mid-reaction
    with actual structure already forming — the same kind of state it trained on.
    """
    key, sk = random.split(key)
    A, B = init_gs_grid(sk, H, W)

    # Warm up: random number of steps so every seed looks different
    warmup = int(np.random.randint(GS_WARMUP_MIN, GS_WARMUP_MAX))
    for _ in range(warmup):
        A, B = gs_step(A, B, f, k)

    # Pack into 16-channel NCA grid
    grid = jnp.zeros((H, W, N_CHANNELS))
    grid = grid.at[:, :, CH_A].set(A)
    grid = grid.at[:, :, CH_B].set(B)
    grid = grid.at[:, :, CH_F].set(f)
    grid = grid.at[:, :, CH_K].set(k)

    return grid, key

# ── Rendering ─────────────────────────────────────────────────────────────────
def render(surface, grid, palette, render_mode="combined", effect="none"):
    """
    Render the NCA grid through a chosen mode and effect.

    Modes:
      combined    — A+B blend (NCA-native: B drives foreground, A modulates depth)
      B           — raw B channel
      edges       — structure boundaries glow
      reaction    — only active chemistry zones light up
      differential— maximum contrast at the A/B interface
      A_inv       — inverted A channel

    Effects: none, bloom, vignette, chromatic, grain, scanlines
    """
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gs', action='store_true',
                        help='Load original GS-only checkpoint for comparison')
    parser.add_argument('--free', action='store_true',
                        help='Free channel experiment: stop injecting ch13/14/15 after warmup')
    args = parser.parse_args()

    ckpt = GS_CHECKPOINT if args.gs else CHECKPOINT
    label = "GS-only (params_050000)" if args.gs else "Lenia-fused (lenia_050000)"
    free_channels = args.free
    FREE_WARMUP   = 2000   # steps before releasing control channels

    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt}  [{label}]")
    if QUIET_MODE:
        print("QUIET MODE — autonomous pokes/drift disabled")
    with open(ckpt, 'rb') as f:
        params = pickle.load(f)
    params = jax.device_put(params)
    print("Loaded.")

    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn           = make_step_fn(update_net, perception_kernel)

    # ── Starting regime ───────────────────────────────────────────────────
    regime_names  = list(GS_REGIMES.keys())
    SEED_REGIMES  = ["chaos", "mitosis", "bacteria", "gliders", "uskate"]
    seed_name     = np.random.choice(SEED_REGIMES)
    regime_idx    = regime_names.index(seed_name)
    f, k          = GS_REGIMES[seed_name]

    # ── Starting palette ──────────────────────────────────────────────────
    palette_names = list(PALETTES.keys())
    palette_idx   = int(np.random.randint(0, len(palette_names)))
    palette       = PALETTES[palette_names[palette_idx]]

    # ── Init grid ─────────────────────────────────────────────────────────
    key = random.PRNGKey(int(np.random.randint(0, 2**31)))
    print(f"Warming up GS seed ({GS_WARMUP_MIN}–{GS_WARMUP_MAX} random steps)...")
    grid, key = init_nca_grid(key, GRID_H, GRID_W, f, k)
    print("Done. Launching.\n")

    # ── Pygame setup ──────────────────────────────────────────────────────
    pygame.init()
    pygame.font.init()
    if DUAL_SCREEN:
        os.environ.setdefault('SDL_VIDEO_WINDOW_POS', '0,0')
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.NOFRAME)
    pygame.display.set_caption("Somnivex — NCA Free Run")
    font   = pygame.font.SysFont("monospace", 16)
    ticker = pygame.time.Clock()

    print(f"Starting regime: {regime_names[regime_idx]}  f={f:.4f}  k={k:.4f}")
    print(f"Controls: R=reset  F=next regime  P=palette  Q=quit\n")

    sound = SoundEngine()
    if SOUND_ENABLED:
        sound.start()

    step_count      = 0
    running         = True
    auto_nudges     = 0
    steps_per_frame = STEPS_PER_FRAME
    physics_bit     = PHYSICS_BIT

    # Perturbation sequence state
    next_perturb    = np.random.randint(PERTURB_INTERVAL_MIN, PERTURB_INTERVAL_MAX)
    pokes_remaining = 0
    next_poke       = 0
    extreme_mode    = False
    pre_burst_f     = f   # f/k saved before an extreme burst so we can restore after
    pre_burst_k     = k

    # Autonomous extreme burst schedule
    next_extreme    = np.random.randint(EXTREME_INTERVAL_MIN, EXTREME_INTERVAL_MAX)

    # Autonomous reseed schedule
    next_reseed     = np.random.randint(RESEED_INTERVAL_MIN, RESEED_INTERVAL_MAX)

    # Palette crossfade state
    palette_current = np.array(palette, dtype=np.float32)
    palette_target  = palette_current.copy()
    palette_blend   = 0   # counts up to PALETTE_BLEND_STEPS, then resets
    next_palette_change = np.random.randint(PALETTE_CHANGE_MIN, PALETTE_CHANGE_MAX)

    # Render mode + effect state
    render_mode_idx  = 0   # start on "combined"
    effect_idx       = 0   # start on "none"
    render_mode      = NCA_RENDER_MODES[render_mode_idx]
    effect           = NCA_EFFECTS[effect_idx]
    next_mode_change = np.random.randint(RENDER_MODE_CHANGE_MIN, RENDER_MODE_CHANGE_MAX)
    next_effect_change = np.random.randint(EFFECT_CHANGE_MIN, EFFECT_CHANGE_MAX)

    # Spatial f/k field state — 4 independent phases drift at slightly different
    # speeds so the pattern never becomes periodic
    phase_fx = np.random.uniform(0, 2 * np.pi)
    phase_fy = np.random.uniform(0, 2 * np.pi)
    phase_kx = np.random.uniform(0, 2 * np.pi)
    phase_ky = np.random.uniform(0, 2 * np.pi)
    # Random phase velocity per axis — all drifting, but not in lockstep
    vel_fx = FK_PHASE_DRIFT * np.random.uniform(0.7, 1.3)
    vel_fy = FK_PHASE_DRIFT * np.random.uniform(0.7, 1.3)
    vel_kx = FK_PHASE_DRIFT * np.random.uniform(0.7, 1.3)
    vel_ky = FK_PHASE_DRIFT * np.random.uniform(0.7, 1.3)

    f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
    jf_field = jnp.array(f_field)
    jk_field = jnp.array(k_field)
    print(f"Spatial f/k active  f_center={f:.4f}±{FK_SPATIAL_AMP_F}  k_center={k:.4f}±{FK_SPATIAL_AMP_K}")

    while running:

        # ── Events ────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_q:
                    running = False

                if event.key == pygame.K_h:
                    grid_np = np.array(grid)
                    B_now = grid_np[:, :, CH_B]
                    grid_np[:, :, 2:13] = B_now[:, :, np.newaxis]
                    grid = jnp.array(grid_np)
                    print(f"H — hidden channels seeded from B. Model has full control.")

                if event.key == pygame.K_s:
                    try:
                        save_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 'saves',
                            f'grid_{step_count:07d}.pkl'
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        save_data = {
                            'grid': np.array(grid),
                            'step_count': step_count,
                            'f': f, 'k': k,
                            'physics_bit': physics_bit,
                            'free_channels': free_channels,
                            'phase_fx': phase_fx, 'phase_fy': phase_fy,
                            'phase_kx': phase_kx, 'phase_ky': phase_ky,
                        }
                        with open(save_path, 'wb') as fh:
                            pickle.dump(save_data, fh)
                        print(f"Saved → {save_path}")
                    except Exception as e:
                        print(f"Save FAILED: {e}")

                if event.key == pygame.K_l:
                    import glob
                    saves_dir = os.path.join(os.path.dirname(__file__), 'saves')
                    save_files = sorted(glob.glob(os.path.join(saves_dir, 'grid_*.pkl')))
                    if not save_files:
                        print("No saves found.")
                    else:
                        load_path = save_files[-1]  # most recent
                        with open(load_path, 'rb') as fh:
                            save_data = pickle.load(fh)
                        grid         = jnp.array(save_data['grid'])
                        step_count   = save_data['step_count']
                        f            = save_data['f']
                        k            = save_data['k']
                        physics_bit  = save_data['physics_bit']
                        free_channels = save_data.get('free_channels', False)
                        phase_fx     = save_data['phase_fx']
                        phase_fy     = save_data['phase_fy']
                        phase_kx     = save_data['phase_kx']
                        phase_ky     = save_data['phase_ky']
                        f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                        jf_field     = jnp.array(f_field)
                        jk_field     = jnp.array(k_field)
                        print(f"Loaded ← {load_path}  (step {step_count})")

                if event.key == pygame.K_r:
                    # Pick a fresh random regime so every reset looks different
                    regime_idx   = np.random.randint(0, len(regime_names))
                    f, k         = GS_REGIMES[regime_names[regime_idx]]
                    key, sk      = random.split(key)
                    print(f"Resetting → {regime_names[regime_idx]}  f={f:.4f} k={k:.4f}")
                    grid, key    = init_nca_grid(sk, GRID_H, GRID_W, f, k)
                    f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                    jf_field     = jnp.array(f_field)
                    jk_field     = jnp.array(k_field)
                    step_count   = 0
                    auto_nudges  = 0
                    print(f"Reset done.")

                if event.key == pygame.K_f:
                    # Jump to a random named GS regime — dramatic, guaranteed diverse
                    regime_idx = np.random.randint(0, len(regime_names))
                    f, k       = GS_REGIMES[regime_names[regime_idx]]
                    f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                    jf_field   = jnp.array(f_field)
                    jk_field   = jnp.array(k_field)
                    print(f"Regime jump → {regime_names[regime_idx]}  f={f:.4f} k={k:.4f}")

                if event.key == pygame.K_p:
                    new_name       = pick_next_palette(palette_names[palette_idx], palette_names)
                    palette_idx    = palette_names.index(new_name)
                    palette_target = np.array(PALETTES[new_name], dtype=np.float32)
                    palette_blend  = 1
                    print(f"Palette: {new_name}")

                if event.key == pygame.K_RIGHTBRACKET:
                    steps_per_frame = min(steps_per_frame + 1, 20)
                    print(f"Speed: {steps_per_frame} steps/frame")

                if event.key == pygame.K_LEFTBRACKET:
                    steps_per_frame = max(steps_per_frame - 1, 1)
                    print(f"Speed: {steps_per_frame} steps/frame")

                if event.key == pygame.K_m:
                    render_mode_idx = (render_mode_idx + 1) % len(NCA_RENDER_MODES)
                    render_mode     = NCA_RENDER_MODES[render_mode_idx]
                    print(f"Render mode: {render_mode}")

                if event.key == pygame.K_e:
                    effect_idx = (effect_idx + 1) % len(NCA_EFFECTS)
                    effect     = NCA_EFFECTS[effect_idx]
                    print(f"Effect: {effect}")

                if event.key == pygame.K_t:
                    physics_bit = 1.0 - physics_bit
                    print(f"Physics bit → {physics_bit:.0f}  ({'Lenia' if physics_bit == 1.0 else 'GS'})")

                if event.key == pygame.K_a:
                    if SOUND_ENABLED:
                        sound.toggle_mute()

                if event.key == pygame.K_x:
                    pre_burst_f     = f
                    pre_burst_k     = k
                    pokes_remaining = 6
                    next_poke       = step_count
                    extreme_mode    = True
                    print(f"EXTREME BURST fired (will restore f={f:.4f} k={k:.4f} after)")

                if event.key == pygame.K_z:
                    # Chaos injection — scramble hidden channels 2-13 directly.
                    # F/X only change f/k (channels 14-15) which the attractor ignores.
                    # This kicks the hidden state itself, forcing a new attractor search.
                    noise = jnp.array(
                        np.random.uniform(-0.5, 0.5, (GRID_H, GRID_W, 12)).astype(np.float32)
                    )
                    grid = grid.at[:, :, 2:14].add(noise)
                    print("Z: hidden channel chaos injection")

        # ── NCA steps ─────────────────────────────────────────────────────
        release_physics = free_channels and step_count >= FREE_WARMUP
        if free_channels and step_count == FREE_WARMUP:
            print(f"FREE CHANNELS ACTIVE — ch13 released. Model controls its own physics bit. f/k still injected.")
        for _ in range(steps_per_frame):
            grid, key = step_fn(grid, params, key)
            # Always keep f/k injected — life support
            grid = grid.at[:, :, CH_F].set(jf_field)
            grid = grid.at[:, :, CH_K].set(jk_field)
            # Only inject physics bit if not in free mode
            if not release_physics:
                grid = grid.at[:, :, CH_PHYSICS].set(physics_bit)
            step_count += 1

        # ── Spatial field phase drift ─────────────────────────────────────
        # Advance all four phases by their individual velocities each frame.
        # The field slowly scrolls across the grid — regions that were in
        # swirl territory drift toward coral territory, and vice versa.
        phase_fx += vel_fx * steps_per_frame
        phase_fy += vel_fy * steps_per_frame
        phase_kx += vel_kx * steps_per_frame
        phase_ky += vel_ky * steps_per_frame
        f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
        jf_field = jnp.array(f_field)
        jk_field = jnp.array(k_field)

        # ── Autonomous f/k drift ──────────────────────────────────────────
        # Every DRIFT_EVERY steps, nudge the center f/k values.
        # This shifts the whole spatial field — every region moves together
        # but they all stay offset from each other.
        if not QUIET_MODE and step_count % DRIFT_EVERY == 0:
            df = np.random.uniform(-DRIFT_AMOUNT, DRIFT_AMOUNT)
            dk = np.random.uniform(-DRIFT_AMOUNT, DRIFT_AMOUNT)
            f  = float(np.clip(f + df, F_MIN, F_MAX))
            k  = float(np.clip(k + dk, K_MIN, K_MAX))
            f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
            jf_field = jnp.array(f_field)
            jk_field = jnp.array(k_field)

        # ── Autonomous extreme burst ──────────────────────────────────────
        if not QUIET_MODE and step_count >= next_extreme and pokes_remaining == 0:
            pre_burst_f     = f
            pre_burst_k     = k
            pokes_remaining = 4
            next_poke       = step_count
            extreme_mode    = True
            next_extreme    = step_count + np.random.randint(EXTREME_INTERVAL_MIN, EXTREME_INTERVAL_MAX)
            print(f"Auto extreme burst (will restore after, next in {next_extreme - step_count} steps)")

        # ── Autonomous reseed — DISABLED ──────────────────────────────────
        # Spatial f/k variation prevents attractor lock-in structurally.
        # Timed reseeds now interrupt interesting states more than they help.
        # R key still works for manual reseeds. Re-enable if needed.

        # ── Perturbation sequences ────────────────────────────────────────
        # Periodically disturb the pattern with a burst of regime changes.
        # Each poke writes new f/k mid-reaction — the NCA treats it as damage
        # and reorganizes. Space them out to avoid solid-screen collapse.

        # Start a new sequence
        if not QUIET_MODE and pokes_remaining == 0 and step_count >= next_perturb:
            pokes_remaining = np.random.randint(PERTURB_POKES_MIN, PERTURB_POKES_MAX + 1)
            next_poke       = step_count
            print(f"Perturbation: {pokes_remaining} pokes incoming...")

        # Fire the next poke in the active sequence
        if pokes_remaining > 0 and step_count >= next_poke:
            if extreme_mode:
                # Adventurous but above F_MIN — going below 0.01 clips the
                # whole spatial field to the floor and collapses everything
                f = float(np.random.uniform(0.012, 0.072))
                k = float(np.random.uniform(0.043, 0.072))
            else:
                # Normal range — within trained territory
                f = float(np.random.uniform(F_MIN, F_MAX))
                k = float(np.random.uniform(K_MIN, K_MAX))
            f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
            jf_field = jnp.array(f_field)
            jk_field = jnp.array(k_field)
            pokes_remaining -= 1
            next_poke        = step_count + np.random.randint(PERTURB_SPACING_MIN, PERTURB_SPACING_MAX)
            label            = "EXTREME" if extreme_mode else "poke"
            print(f"  {label} → f={f:.4f} k={k:.4f}  ({pokes_remaining} remaining)")
            if pokes_remaining == 0:
                if extreme_mode:
                    # Restore the field to where it was before the burst
                    # so the extreme jolt doesn't permanently strand the system
                    f = pre_burst_f
                    k = pre_burst_k
                    f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                    jf_field = jnp.array(f_field)
                    jk_field = jnp.array(k_field)
                    print(f"  burst done — restored f={f:.4f} k={k:.4f}")
                extreme_mode = False
                next_perturb = step_count + np.random.randint(PERTURB_INTERVAL_MIN, PERTURB_INTERVAL_MAX)
                print(f"  next sequence in {next_perturb - step_count} steps")

        # ── Saturation detection ──────────────────────────────────────────
        # Free channel mode: drop small fresh GS patches at random positions.
        # Keeps all accumulated hidden state intact — just sparks new activity.
        if free_channels and step_count % SATURATION_CHECK == 0 and step_count >= FREE_WARMUP:
            b_std = float(jnp.std(grid[:, :, CH_B]))
            if b_std < SATURATION_STD:
                patch_size = 16
                n_patches  = np.random.randint(3, 7)
                grid_np    = np.array(grid)
                for _ in range(n_patches):
                    py = np.random.randint(0, GRID_H - patch_size)
                    px = np.random.randint(0, GRID_W - patch_size)
                    A_patch = np.ones((patch_size, patch_size), dtype=np.float32)
                    B_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
                    # Small random blob of B in the center of the patch
                    cy, cx = patch_size // 2, patch_size // 2
                    r = np.random.randint(3, 7)
                    for dy in range(-r, r+1):
                        for dx in range(-r, r+1):
                            if dy*dy + dx*dx <= r*r:
                                B_patch[cy+dy, cx+dx] = np.random.uniform(0.5, 1.0)
                    grid_np[py:py+patch_size, px:px+patch_size, CH_A] = A_patch
                    grid_np[py:py+patch_size, px:px+patch_size, CH_B] = B_patch
                grid = jnp.array(grid_np)
                print(f"Auto spark (free mode, std={b_std:.4f}) — {n_patches} patches dropped")

        # Only fires when the screen is truly solid — threshold lowered from
        # 0.02 to 0.005 so interesting dark/ghost-trace states are left alone.
        if not QUIET_MODE and step_count % SATURATION_CHECK == 0:
            b_std = float(jnp.std(grid[:, :, CH_B]))
            if b_std < SATURATION_STD:
                # Pick fresh safe f/k — current values may be extreme/clipped
                # which is what caused the collapse in the first place
                f         = float(np.random.uniform(0.025, 0.060))
                k         = float(np.random.uniform(0.050, 0.065))
                key, sk   = random.split(key)
                grid, key = init_nca_grid(sk, GRID_H, GRID_W, f, k)
                f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                jf_field  = jnp.array(f_field)
                jk_field  = jnp.array(k_field)
                auto_nudges += 1
                print(f"Solid screen (std={b_std:.4f}) → reseed #{auto_nudges}  f={f:.4f} k={k:.4f}")

        # ── Autonomous render mode rotation ───────────────────────────────
        if step_count >= next_mode_change:
            render_mode_idx  = (render_mode_idx + 1) % len(NCA_RENDER_MODES)
            render_mode      = NCA_RENDER_MODES[render_mode_idx]
            next_mode_change = step_count + np.random.randint(RENDER_MODE_CHANGE_MIN, RENDER_MODE_CHANGE_MAX)
            print(f"Auto render mode → {render_mode}")

        # ── Autonomous effect rotation ─────────────────────────────────────
        if step_count >= next_effect_change:
            effect_idx         = (effect_idx + 1) % len(NCA_EFFECTS)
            effect             = NCA_EFFECTS[effect_idx]
            next_effect_change = step_count + np.random.randint(EFFECT_CHANGE_MIN, EFFECT_CHANGE_MAX)
            print(f"Auto effect → {effect}")

        # ── Autonomous palette crossfade ──────────────────────────────────
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

        # ── Sound update ──────────────────────────────────────────────────
        if SOUND_ENABLED:
            sound.update(np.array(grid))

        # ── Render ────────────────────────────────────────────────────────
        render(screen, grid, blended_palette.astype(np.uint8).tolist(), render_mode, effect)

        palette_str = palette_names[palette_idx]
        free_str = "  |  FREE-CH" if (free_channels and step_count >= FREE_WARMUP) else ""
        hud = font.render(
            f"step {step_count}  |  bit={physics_bit:.0f}({'L' if physics_bit else 'G'}){free_str}  f={f:.4f}±{FK_SPATIAL_AMP_F} k={k:.4f}±{FK_SPATIAL_AMP_K}  |  {palette_str}  |  {render_mode}+{effect}  |  spd={steps_per_frame}  |  T=physics A=sound M=mode E=effect P=palette F=poke X=extreme Z=chaos H=seed-hidden S=save L=load R=reset Q=quit",
            True, (80, 80, 80)
        )
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        ticker.tick(FPS)

    if SOUND_ENABLED:
        sound.stop()
    pygame.quit()


if __name__ == '__main__':
    print(f"JAX devices: {jax.devices()}")
    run()
