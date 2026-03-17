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
import csv
import numpy as np
import pygame
import jax
import jax.numpy as jnp
from jax import random
from scipy import ndimage as _ndimage

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

# ── Research / decoder config ─────────────────────────────────────────────────
# Run with --research flag to lock display and enable auto-logging
# Note: RESEARCH_MODE is set after parse_args() inside run() — this is a placeholder
RESEARCH_MODE     = False  # overridden inside run() after argparse
AUTO_LOG_EVERY    = 200          # steps between feature vector logs
LOG_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
CLASSIFIER_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state_classifier.pkl')
RESEARCH_PALETTE  = 'neon_city'
RESEARCH_RENDER   = 'edges'
RESEARCH_EFFECT   = 'vignette'

STATE_NAMES = {
    0: 'Chaos/Init',
    1: 'Stable Ecosystem',
    2: 'Heat Death',
    3: 'Near Extinction',
    4: 'Rich Ecosystem',
    5: 'Predator Invasion',
    6: 'Pre-activation',
    7: 'Zombie',
}

def extract_features(grid_np):
    a, b = grid_np[:,:,0], grid_np[:,:,1]
    lo = a < 0.80
    labeled, n = _ndimage.label(lo)
    sizes = sorted([np.sum(labeled==i) for i in range(1,n+1)], reverse=True)
    ch4 = grid_np[:,:,4]
    corr = np.corrcoef(ch4.flatten(), b.flatten())[0,1] if ch4.std()>1e-6 else 0.0
    feat = [
        np.mean(a>0.97), np.mean(a<0.80), np.mean(b>0.08), a.std(), b.max(),
        grid_np[:,:,2].std(), grid_np[:,:,4].std(), grid_np[:,:,5].std(),
        n, sizes[0] if sizes else 0, sizes[1] if len(sizes)>1 else 0,
        np.std(sizes) if sizes else 0,
        np.mean(a[:,:32]<0.80), np.mean(a[:,32:]<0.80),
        abs(np.mean(a[:,:32]<0.80)-np.mean(a[:,32:]<0.80)), corr
    ]
    return [0.0 if (v != v) else float(v) for v in feat]  # replace NaN with 0

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'lenia_100000.pkl'
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
    parser.add_argument('--research', action='store_true',
                        help='Research mode: lock display to cell_wall+edges+vignette, enable auto-logging and state HUD')
    args = parser.parse_args()
    global RESEARCH_MODE
    RESEARCH_MODE = args.research

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
    if RESEARCH_MODE and RESEARCH_PALETTE in PALETTES:
        palette_idx = palette_names.index(RESEARCH_PALETTE)
    else:
        palette_idx = int(np.random.randint(0, len(palette_names)))
    palette = PALETTES[palette_names[palette_idx]]

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

    # Render mode + effect state
    if RESEARCH_MODE:
        render_mode_idx = NCA_RENDER_MODES.index(RESEARCH_RENDER) if RESEARCH_RENDER in NCA_RENDER_MODES else 0
        effect_idx      = NCA_EFFECTS.index(RESEARCH_EFFECT) if RESEARCH_EFFECT in NCA_EFFECTS else 0
    else:
        render_mode_idx = 0
        effect_idx      = 0
    render_mode      = NCA_RENDER_MODES[render_mode_idx]
    effect           = NCA_EFFECTS[effect_idx]
    # In research mode push auto-rotation far out so display stays locked
    next_mode_change   = np.random.randint(RENDER_MODE_CHANGE_MIN, RENDER_MODE_CHANGE_MAX) if not RESEARCH_MODE else 999999999
    next_effect_change = np.random.randint(EFFECT_CHANGE_MIN, EFFECT_CHANGE_MAX)           if not RESEARCH_MODE else 999999999
    next_palette_change = np.random.randint(PALETTE_CHANGE_MIN, PALETTE_CHANGE_MAX) if not RESEARCH_MODE else 999999999

    # ── Research: load classifier + open log file ──────────────────────────
    classifier        = None
    log_csv           = None
    intervention_log  = None
    current_state     = -1
    state_hud_str     = ''
    _ctrl_tgt_idx     = 0   # index into _CTRL_TARGETS for C key cycling
    _CTRL_TARGETS     = [4, 1, 5, 3, 7]  # Rich, Stable, Predator, Near Extinction, Zombie

    # ── Attractor seed (click-to-place) ───────────────────────────────────────
    # V key cycles which state to paint. Mouse click stamps that state's hidden
    # channel signature into a local region — a crystal seed the NCA expands outward.
    # Defined per-state as (ch2_amp, ch4_amp, ch4_ring, zero_hidden):
    #   ch2_amp   — noise amplitude for ch2 (0 = skip)
    #   ch4_amp   — noise amplitude for ch4 (0 = skip)
    #   ch4_ring  — if True, concentrate ch4 in a ring (border) instead of fill
    #   zero_out  — if True, zero all hidden channels in the region (death seed)
    SEED_RADIUS = 48   # half-size of stamp region — needs ~10% of grid to compete with attractor
    SEED_STATES = [4, 1, 5, 3]   # Rich, Stable, Predator, Near Extinction
    # Each seed writes to the VISIBLE chemistry (A, B) AND hidden channels.
    # Hidden-only injection gets swamped in a few steps — need to set the actual chemistry.
    # a_val/b_val: fixed values to write into A/B in the region (None = don't touch)
    # ch2_amp/ch4_amp: noise amplitude for hidden channels (0 = skip)
    # ch4_ring: concentrate ch4 in border ring (Rich Ecosystem signature)
    # zero_out: zero all hidden channels (Near Extinction signature)
    SEED_PARAMS = {
        4: dict(a_val=0.50, b_val=0.25, ch2_amp=0.04, ch4_amp=0.0, ch4_ring=True,  zero_out=False),  # Rich
        1: dict(a_val=0.85, b_val=0.10, ch2_amp=0.02, ch4_amp=0.02, ch4_ring=False, zero_out=False),  # Stable
        5: dict(a_val=0.30, b_val=0.40, ch2_amp=0.15, ch4_amp=0.15, ch4_ring=False, zero_out=False),  # Predator
        3: dict(a_val=0.99, b_val=0.00, ch2_amp=0.0,  ch4_amp=0.0,  ch4_ring=False, zero_out=True),   # Near Extinction
    }
    SEED_LABELS = {4:'Rich Ecosystem', 1:'Stable Ecosystem', 5:'Predator Invasion', 3:'Near Extinction'}
    _seed_state_idx = 0   # index into SEED_STATES
    if RESEARCH_MODE:
        if os.path.exists(CLASSIFIER_PATH):
            with open(CLASSIFIER_PATH, 'rb') as _f:
                classifier = pickle.load(_f)
            print(f"Classifier loaded: {CLASSIFIER_PATH}")
        else:
            print(f"WARNING: classifier not found at {CLASSIFIER_PATH} — state HUD disabled")
        os.makedirs(LOG_DIR, exist_ok=True)
        import datetime
        _run_ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(LOG_DIR, f'features_{_run_ts}.csv')
        log_csv  = open(log_path, 'w', newline='')
        _writer  = csv.writer(log_csv)
        _writer.writerow(['step','state_id','state_name','physics_bit',
                          'bg','dark','b_active','a_std','b_max',
                          'ch2','ch4','ch5','n_blobs','largest','second',
                          'size_std','left_dark','right_dark','asym','corr_ch4_b'])
        log_csv.flush()
        # Intervention log — records every keypress with step + effect
        _int_path   = os.path.join(LOG_DIR, f'interventions_{_run_ts}.csv')
        intervention_log = open(_int_path, 'w', newline='')
        csv.writer(intervention_log).writerow(['step','key','state_before','state_after','note'])
        intervention_log.flush()
        print(f"Logging features to: {log_path}")
        print(f"Logging interventions to: {_int_path}")
        print(f"Display locked: {RESEARCH_PALETTE} + {RESEARCH_RENDER} + {RESEARCH_EFFECT}")

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
                    if intervention_log: csv.writer(intervention_log).writerow([step_count,'H',current_state,'','seed hidden from B']); intervention_log.flush()
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
                    if intervention_log: csv.writer(intervention_log).writerow([step_count,'F',current_state,'','regime jump f/k']); intervention_log.flush()
                    # Jump to a random named GS regime — dramatic, guaranteed diverse
                    regime_idx = np.random.randint(0, len(regime_names))
                    f, k       = GS_REGIMES[regime_names[regime_idx]]
                    f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                    jf_field   = jnp.array(f_field)
                    jk_field   = jnp.array(k_field)
                    print(f"Regime jump → {regime_names[regime_idx]}  f={f:.4f} k={k:.4f}")

                # Number keys 1-9 for direct named regime selection
                _num_keys = {
                    pygame.K_1: 'spirals',
                    pygame.K_2: 'chaos',
                    pygame.K_3: 'waves',
                    pygame.K_4: 'worms',
                    pygame.K_5: 'mitosis',
                    pygame.K_6: 'gliders',
                    pygame.K_7: 'bacteria',
                    pygame.K_8: 'maze',
                    pygame.K_9: 'stripes',
                    pygame.K_0: 'uskate',
                }
                if event.key in _num_keys:
                    _rname = _num_keys[event.key]
                    f, k   = GS_REGIMES[_rname]
                    f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                    jf_field = jnp.array(f_field)
                    jk_field = jnp.array(k_field)
                    if intervention_log: csv.writer(intervention_log).writerow([step_count, f'KEY_{_rname}', current_state, '', f'regime={_rname} f={f:.4f} k={k:.4f}']); intervention_log.flush()
                    print(f"Regime → {_rname}  f={f:.4f} k={k:.4f}")

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
                    if intervention_log: csv.writer(intervention_log).writerow([step_count,'T',current_state,'',f'physics→{"Lenia" if physics_bit==1.0 else "GS"}']); intervention_log.flush()
                    if physics_bit == 1.0:
                        render_mode = 'A_inv'
                        render_mode_idx = NCA_RENDER_MODES.index('A_inv')
                    else:
                        render_mode = 'edges'
                        render_mode_idx = NCA_RENDER_MODES.index('edges')
                    print(f"Physics bit → {physics_bit:.0f}  ({'Lenia' if physics_bit == 1.0 else 'GS'})  render → {render_mode}")

                if event.key == pygame.K_a:
                    if SOUND_ENABLED:
                        sound.toggle_mute()

                if event.key == pygame.K_x:
                    if intervention_log: csv.writer(intervention_log).writerow([step_count,'X',current_state,'','extreme burst f/k']); intervention_log.flush()
                    pre_burst_f     = f
                    pre_burst_k     = k
                    pokes_remaining = 6
                    next_poke       = step_count
                    extreme_mode    = True
                    print(f"EXTREME BURST fired (will restore f={f:.4f} k={k:.4f} after)")

                if event.key == pygame.K_z:
                    if intervention_log: csv.writer(intervention_log).writerow([step_count,'Z',current_state,'','hidden channel chaos injection']); intervention_log.flush()
                    # Chaos injection — scramble hidden channels 2-13 directly.
                    # F/X only change f/k (channels 14-15) which the attractor ignores.
                    # This kicks the hidden state itself, forcing a new attractor search.
                    noise = jnp.array(
                        np.random.uniform(-0.5, 0.5, (GRID_H, GRID_W, 12)).astype(np.float32)
                    )
                    grid = grid.at[:, :, 2:14].add(noise)
                    print("Z: hidden channel chaos injection")

                if event.key == pygame.K_c:
                    # Directional hidden channel injection toward target cluster centroid.
                    # Each press cycles the target state, then steers ch2/ch4/ch5 toward it.
                    # Uses the delta between current and target centroids (in feature space)
                    # to compute injection amplitude — more precise than random Z chaos.
                    if classifier is not None:
                        _ctrl_tgt_idx = (_ctrl_tgt_idx + 1) % len(_CTRL_TARGETS)
                        _ctrl_target  = _CTRL_TARGETS[_ctrl_tgt_idx]
                        _ctrl_name    = STATE_NAMES.get(_ctrl_target, str(_ctrl_target))
                        _kmeans = classifier['kmeans']
                        _scaler = classifier['scaler']
                        _src_id = current_state if current_state >= 0 else 0
                        _src_center = _scaler.inverse_transform(
                            [_kmeans.cluster_centers_[_src_id]])[0]
                        _tgt_center = _scaler.inverse_transform(
                            [_kmeans.cluster_centers_[_ctrl_target]])[0]
                        # Feature indices: 5=ch2.std, 6=ch4.std, 7=ch5.std
                        _delta_ch2 = float(_tgt_center[5] - _src_center[5])
                        _delta_ch4 = float(_tgt_center[6] - _src_center[6])
                        _delta_ch5 = float(_tgt_center[7] - _src_center[7])
                        _grid_np_c = np.array(grid)
                        for _chi, _delta in [(2, _delta_ch2), (4, _delta_ch4), (5, _delta_ch5)]:
                            if abs(_delta) > 1e-5:
                                if _delta > 0:
                                    # Increase channel activity: add Gaussian noise at target amplitude
                                    # Amplifier 30x — deltas are small (~0.004-0.012) and need
                                    # enough force to actually displace the attractor basin
                                    _noise = np.random.normal(
                                        0, abs(_delta) * 30.0, (GRID_H, GRID_W)
                                    ).astype(np.float32)
                                    grid = grid.at[:, :, _chi].add(jnp.array(_noise))
                                else:
                                    # Decrease channel activity: strong dampen toward channel mean
                                    _cur_std = max(float(_grid_np_c[:, :, _chi].std()), 1e-6)
                                    _dampen  = max(0.0, 1.0 - abs(_delta) * 30.0 / _cur_std)
                                    _mean    = float(_grid_np_c[:, :, _chi].mean())
                                    grid = grid.at[:, :, _chi].set(
                                        grid[:, :, _chi] * _dampen + _mean * (1.0 - _dampen)
                                    )
                        if intervention_log:
                            csv.writer(intervention_log).writerow([
                                step_count, 'C', current_state, '',
                                f'steer→{_ctrl_name} ch2Δ={_delta_ch2:.4f} ch4Δ={_delta_ch4:.4f} amp=30'
                            ])
                            intervention_log.flush()
                        print(f"C: steering → {_ctrl_name}  "
                              f"ch2Δ={_delta_ch2:+.4f}  ch4Δ={_delta_ch4:+.4f}  ch5Δ={_delta_ch5:+.4f}")
                    else:
                        print("C: no classifier loaded — run with --research to enable steering")

                if event.key == pygame.K_v:
                    # Cycle the attractor seed paint state
                    _seed_state_idx = (_seed_state_idx + 1) % len(SEED_STATES)
                    _sname = SEED_LABELS[SEED_STATES[_seed_state_idx]]
                    print(f"V: paint state → {_sname}  (click to stamp on grid)")

            # ── Mouse click — stamp attractor seed ────────────────────────────
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # Map screen → grid coordinates (grid is scaled to fill display)
                gx = int(mx * GRID_W / DISPLAY_W)
                gy = int(my * GRID_H / DISPLAY_H)
                _sid   = SEED_STATES[_seed_state_idx]
                _sp    = SEED_PARAMS[_sid]
                _sname = SEED_LABELS[_sid]
                r      = SEED_RADIUS
                # Clamp region to grid bounds
                y0, y1 = max(0, gy - r), min(GRID_H, gy + r)
                x0, x1 = max(0, gx - r), min(GRID_W, gx + r)
                h, w   = y1 - y0, x1 - x0
                # Write visible chemistry (A and B) — this is the key part.
                # Hidden-channel-only injection gets swamped in a few steps.
                # Writing A/B sets the actual reaction state the NCA evolves from.
                if _sp['a_val'] is not None:
                    grid = grid.at[y0:y1, x0:x1, 0].set(_sp['a_val'])
                if _sp['b_val'] is not None:
                    grid = grid.at[y0:y1, x0:x1, 1].set(_sp['b_val'])
                if _sp['zero_out']:
                    # Death seed — kill all hidden channels in region
                    grid = grid.at[y0:y1, x0:x1, 2:13].set(0.0)
                else:
                    if _sp['ch2_amp'] > 0:
                        _n2 = np.random.normal(0, _sp['ch2_amp'], (h, w)).astype(np.float32)
                        grid = grid.at[y0:y1, x0:x1, 2].add(jnp.array(_n2))
                    if _sp['ch4_amp'] > 0:
                        if _sp['ch4_ring']:
                            # Concentrate ch4 in a ring around the border of the region
                            _n4 = np.zeros((h, w), dtype=np.float32)
                            ring = 6
                            _n4[:ring,  :]  = _sp['ch4_amp'] * 3.0
                            _n4[-ring:, :]  = _sp['ch4_amp'] * 3.0
                            _n4[:,  :ring]  = _sp['ch4_amp'] * 3.0
                            _n4[:, -ring:]  = _sp['ch4_amp'] * 3.0
                        else:
                            _n4 = np.random.normal(0, _sp['ch4_amp'], (h, w)).astype(np.float32)
                        grid = grid.at[y0:y1, x0:x1, 4].add(jnp.array(_n4))
                if intervention_log:
                    csv.writer(intervention_log).writerow([
                        step_count, 'SEED', current_state, '',
                        f'stamp {_sname} at grid ({gx},{gy}) r={r}'
                    ])
                    intervention_log.flush()
                print(f"SEED: stamped {_sname} at ({gx},{gy})  region [{x0}:{x1}, {y0}:{y1}]")

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

        # ── Research: auto feature logging + state HUD + transition saves ─
        if RESEARCH_MODE and step_count % AUTO_LOG_EVERY == 0:
            _grid_np = np.array(grid)
            _feat    = extract_features(_grid_np)
            _state_id = -1
            _state_name = 'unknown'
            if classifier is not None:
                _Xs = classifier['scaler'].transform([_feat])
                _state_id = int(classifier['kmeans'].predict(_Xs)[0])
                _state_name = STATE_NAMES.get(_state_id, str(_state_id))
            # Log to CSV
            if log_csv is not None:
                csv.writer(log_csv).writerow(
                    [step_count, _state_id, _state_name, int(physics_bit)] + [f'{v:.6f}' for v in _feat])
                log_csv.flush()
            # Transition detected — full save (pkl + screenshot)
            if _state_id != current_state and current_state != -1:
                _trans_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'saves',
                    f'transition_{step_count:07d}_to_{_state_name.replace(" ","_")}.pkl'
                )
                os.makedirs(os.path.dirname(_trans_path), exist_ok=True)
                with open(_trans_path, 'wb') as _fh:
                    pickle.dump({'grid': _grid_np, 'step_count': step_count,
                                 'f': f, 'k': k, 'physics_bit': physics_bit,
                                 'state_id': _state_id, 'state_name': _state_name,
                                 'prev_state': current_state,
                                 'free_channels': free_channels,
                                 'phase_fx': phase_fx, 'phase_fy': phase_fy,
                                 'phase_kx': phase_kx, 'phase_ky': phase_ky}, _fh)
                pygame.image.save(screen, _trans_path.replace('.pkl', '.png'))
                print(f"TRANSITION [{STATE_NAMES.get(current_state,'?')}] → [{_state_name}]  step {step_count}")
            current_state = _state_id
            # Update HUD string
            state_hud_str = (f"  |  [{_state_id}]{_state_name}"
                             f"  bg={_feat[0]*100:.0f}%"
                             f"  blobs={int(_feat[8])}"
                             f"  ch2={_feat[5]:.4f}"
                             f"  ch4={_feat[6]:.4f}")

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
            f"step {step_count}  |  bit={physics_bit:.0f}({'L' if physics_bit else 'G'}){free_str}  f={f:.4f}±{FK_SPATIAL_AMP_F} k={k:.4f}±{FK_SPATIAL_AMP_K}  |  {palette_str}  |  {render_mode}+{effect}  |  spd={steps_per_frame}  |  T=physics A=sound M=mode E=effect P=palette F=poke X=extreme Z=chaos H=seed-hidden S=save",
            True, (80, 80, 80)
        )
        screen.blit(hud, (10, 10))
        if RESEARCH_MODE and state_hud_str:
            hud2 = font.render(f"STATE{state_hud_str}", True, (60, 180, 120))
            screen.blit(hud2, (10, 28))

        pygame.display.flip()
        ticker.tick(FPS)

    if SOUND_ENABLED:
        sound.stop()
    if log_csv is not None:
        log_csv.close()
        print(f"Feature log closed.")
    pygame.quit()


if __name__ == '__main__':
    print(f"JAX devices: {jax.devices()}")
    run()
