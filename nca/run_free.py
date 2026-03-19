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
from collections import deque
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
CMD_FILE          = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_commands.txt')
CMD_FILE_KEEPER   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_commands_keeper.txt')
CMD_FILE_DESTROYER= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_commands_destroyer.txt')
CMD_FILE_ARTIST   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_commands_artist.txt')
TURN_FILE         = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'battle_turn.txt')
CMD_POLL_EVERY    = 30   # frames between command file checks
BATTLE_TURN_STEPS = 500   # NCA steps between battle turns (~8s on GPU)
RESEARCH_RENDER   = 'edges'
RESEARCH_EFFECT   = 'vignette'

STATE_NAMES = {
    0: 'Predator Invasion',
    1: 'Global Blob',
    2: 'Stable Ecosystem',
    3: 'Chaos/Transition',
}

_blob_centroid_history = deque(maxlen=2)  # rolling buffer for blob mobility (2 samples = 400 steps)

def extract_features(grid_np):
    a, b = grid_np[:,:,0], grid_np[:,:,1]
    ch2 = grid_np[:,:,2]
    ch4 = grid_np[:,:,4]
    lo = a < 0.80
    labeled, n = _ndimage.label(lo)
    sizes = sorted([np.sum(labeled==i) for i in range(1,n+1)], reverse=True)
    corr = np.corrcoef(ch4.flatten(), b.flatten())[0,1] if ch4.std()>1e-6 else 0.0

    # ── New features ──────────────────────────────────────────────────────────
    if n > 0:
        # Border ratio: ch4 and ch2 enrichment at largest blob edge vs global mean
        largest_label = max(range(1, n+1), key=lambda i: np.sum(labeled==i))
        blob_mask = (labeled == largest_label)
        eroded = _ndimage.binary_erosion(blob_mask, iterations=3)
        border_mask = blob_mask & ~eroded
        ch4_global = ch4.mean()
        ch4_border_ratio = (ch4[border_mask].mean() / (ch4_global + 1e-9)) if border_mask.any() else 1.0
        ch2_global = ch2.mean()
        ch2_border_ratio = (ch2[border_mask].mean() / (ch2_global + 1e-9)) if border_mask.any() else 1.0

        # Blob mobility: centroid displacement of top-2 blobs between samples
        top_labels = sorted(range(1, n+1), key=lambda i: np.sum(labeled==i), reverse=True)[:2]
        centroids = []
        for lbl in top_labels:
            coords = np.where(labeled == lbl)
            centroids.append((float(np.mean(coords[0])), float(np.mean(coords[1]))))
        while len(centroids) < 2:
            centroids.append((0.0, 0.0))
        _blob_centroid_history.append(centroids)
        if len(_blob_centroid_history) >= 2:
            prev, curr = _blob_centroid_history[-2], _blob_centroid_history[-1]
            d1 = np.sqrt((curr[0][0]-prev[0][0])**2 + (curr[0][1]-prev[0][1])**2)
            d2 = np.sqrt((curr[1][0]-prev[1][0])**2 + (curr[1][1]-prev[1][1])**2)
            blob_mobility = float((d1 + d2) / 2.0)
        else:
            blob_mobility = 0.0
    else:
        ch4_border_ratio = 1.0
        ch2_border_ratio = 1.0
        blob_mobility = 0.0

    # Suppression zone: ch4 elevated AND B depleted (predator hunting zone signature)
    ch4_mean = ch4.mean()
    suppression_zone_frac = float(np.mean((ch4 > 1.2 * ch4_mean) & (b < 0.05))) if ch4_mean > 1e-6 else 0.0

    feat = [
        np.mean(a>0.97), np.mean(a<0.80), np.mean(b>0.08), a.std(), b.max(),
        ch2.std(), ch4.std(), grid_np[:,:,5].std(),
        n, sizes[0] if sizes else 0, sizes[1] if len(sizes)>1 else 0,
        np.std(sizes) if sizes else 0,
        np.mean(a[:,:32]<0.80), np.mean(a[:,32:]<0.80),
        abs(np.mean(a[:,:32]<0.80)-np.mean(a[:,32:]<0.80)), corr,
        ch4_border_ratio, ch2_border_ratio, blob_mobility, suppression_zone_frac
    ]
    return [0.0 if (v != v) else float(v) for v in feat]  # replace NaN with 0

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'lenia_100000.pkl'
)
GS_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'gs_only_100000.pkl'
)
PHYSARUM_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'physarum_100000.pkl'
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
    parser.add_argument('--physarum', action='store_true',
                        help='Load GS+Physarum checkpoint (physarum_100000.pkl)')
    parser.add_argument('--free', action='store_true',
                        help='Free channel experiment: stop injecting ch13/14/15 after warmup')
    parser.add_argument('--research', action='store_true',
                        help='Research mode: lock display to cell_wall+edges+vignette, enable auto-logging and state HUD')
    parser.add_argument('--artist', action='store_true',
                        help='Artist mode: poll llm_commands_artist.txt for spatial brush commands from LLM painter')
    parser.add_argument('--battle', action='store_true',
                        help='Battle mode: enable keeper/destroyer turn polling')
    args = parser.parse_args()
    global RESEARCH_MODE
    RESEARCH_MODE = args.research
    ARTIST_MODE   = args.artist
    BATTLE_MODE   = args.battle

    if args.gs:
        ckpt  = GS_CHECKPOINT
        label = "GS-only (gs_only_100000)"
    elif args.physarum:
        ckpt  = PHYSARUM_CHECKPOINT
        label = "GS+Physarum (physarum_100000)"
    else:
        ckpt  = CHECKPOINT
        label = "Lenia-fused (lenia_100000)"
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

    step_count            = 0
    running               = True
    auto_nudges           = 0
    _battle_next_turn_step = 0   # step at which to advance battle turn
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

    # Autonomous reseed schedule (disabled — R key still works for manual reseeds)

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
    next_mode_change   = np.random.randint(RENDER_MODE_CHANGE_MIN, RENDER_MODE_CHANGE_MAX) if not (RESEARCH_MODE or ARTIST_MODE) else 999999999
    next_effect_change = np.random.randint(EFFECT_CHANGE_MIN, EFFECT_CHANGE_MAX)           if not (RESEARCH_MODE or ARTIST_MODE) else 999999999
    next_palette_change = np.random.randint(PALETTE_CHANGE_MIN, PALETTE_CHANGE_MAX) if not (RESEARCH_MODE or ARTIST_MODE) else 999999999

    # ── Research: load classifier + open log file ──────────────────────────
    classifier        = None
    log_csv           = None
    intervention_log  = None
    current_state     = -1
    state_hud_str     = ''
    _ctrl_tgt_idx     = 0   # index into _CTRL_TARGETS for C key cycling
    _CTRL_TARGETS     = [0, 1, 2, 3]  # Predator Invasion, Global Blob, Stable Ecosystem, Chaos

    # ── Wall / door system ────────────────────────────────────────────────────
    # W key: toggle wall-draw mode. Click+drag paints wall cells.
    # D key: clear all walls. Right-click in wall mode: erase wall cells.
    # Post-step: wall cells are forced to A=1.0, B=0.0 every frame.
    wall_mode    = False
    wall_mask    = np.zeros((GRID_H, GRID_W), dtype=bool)
    wall_drawing = False   # True while mouse button held in wall mode
    show_ch5_overlay = False  # X key — cyan heatmap of ch5 spatial values
    ch5_tint_color   = (0, 220, 255)  # RGB tint for ch5 overlay — artist can change with trailcolor command
    artist_mirror    = False           # mirror mode — paints symmetrically across vertical axis
    wall_erase   = False   # True for right-click erase
    jwall        = jnp.zeros((GRID_H, GRID_W), dtype=bool)
    WALL_BRUSH   = 2       # brush radius in grid cells

    # ── ch5 trail injection (Y key) ───────────────────────────────────────────
    trail_mode    = False
    trail_drawing = False
    trail_erase   = False
    trail_mask    = np.zeros((GRID_H, GRID_W), dtype=bool)
    trail_strength = 0.8    # injected ch5 value — [ / ] to adjust
    TRAIL_BRUSH   = 3        # brush radius in grid cells
    jtrail        = jnp.zeros((GRID_H, GRID_W), dtype=bool)
    # Drift — LLM can issue drift dx dy to shepherd organisms along a moving trail
    trail_drift_x  = 0      # pixels to shift trail per DRIFT_INTERVAL steps
    trail_drift_y  = 0
    DRIFT_INTERVAL = 30     # NCA steps between each drift shift

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
    BLOB_RADIUS   = 4     # single-blob stamp — small enough to seed exactly one ring
    blob_strength = 0.25  # B injection value — shift+[ / shift+] to adjust
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
                          'size_std','left_dark','right_dark','asym','corr_ch4_b',
                          'ch4_border_ratio','ch2_border_ratio','blob_mobility','suppression_zone_frac'])
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

                # Number keys — organized by function based on empirical testing
                # 5/6 = stabilizers, 8 = predator trigger, 0 = chaos escape
                # Others kept for exploration but less reliable
                _num_keys = {
                    pygame.K_1: 'mitosis',   # stable/rich
                    pygame.K_2: 'gliders',   # stable, glider-friendly
                    pygame.K_3: 'maze',      # predator trigger
                    pygame.K_4: 'worms',
                    pygame.K_5: 'spirals',
                    pygame.K_6: 'chaos',
                    pygame.K_7: 'bacteria',
                    pygame.K_8: 'waves',
                    pygame.K_9: 'stripes',
                    pygame.K_0: 'uskate',    # chaos escape
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

                if event.key == pygame.K_w:
                    wall_mode = not wall_mode
                    print(f"Wall mode: {'ON — click+drag to draw, right-click to erase' if wall_mode else 'OFF'}")

                if event.key == pygame.K_d:
                    wall_mask[:] = False
                    jwall = jnp.zeros((GRID_H, GRID_W), dtype=bool)
                    print("Walls cleared")

                if event.key == pygame.K_x:
                    show_ch5_overlay = not show_ch5_overlay
                    print(f"ch5 overlay: {'ON' if show_ch5_overlay else 'OFF'}")

                if event.key == pygame.K_y:
                    trail_mode = not trail_mode
                    print(f"Trail paint: {'ON (drag to paint ch5, right-click erase, [/] strength)' if trail_mode else 'OFF'}")

                if event.key == pygame.K_LEFTBRACKET:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        blob_strength = max(0.05, blob_strength - 0.05)
                        print(f"Blob strength: {blob_strength:.2f}")
                    else:
                        trail_strength = max(0.005, trail_strength - 0.005)
                        print(f"Trail strength: {trail_strength:.3f}")

                if event.key == pygame.K_RIGHTBRACKET:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        blob_strength = min(0.5, blob_strength + 0.05)
                        print(f"Blob strength: {blob_strength:.2f}")
                    else:
                        trail_strength = min(1.0, trail_strength + 0.005)
                        print(f"Trail strength: {trail_strength:.3f}")

                if event.key == pygame.K_n:
                    trail_mask[:] = False
                    jtrail = jnp.zeros((GRID_H, GRID_W), dtype=bool)
                    print("Trail cleared")

                if event.key == pygame.K_k:
                    # Nuke grid to extinction — A=1, B=0, hidden channels zeroed
                    # Trail and walls are preserved so you can drop blobs into a prepared arena
                    grid = grid.at[:, :, 0].set(1.0)
                    grid = grid.at[:, :, 1].set(0.0)
                    for _chi in range(2, 14):
                        grid = grid.at[:, :, _chi].set(0.0)
                    print("K: EXTINCTION — grid wiped. Drop a blob with B key + click.")


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

            # ── Mouse: wall draw / erase ──────────────────────────────────────
            if event.type == pygame.MOUSEBUTTONDOWN and wall_mode:
                wall_drawing = True
                wall_erase   = (event.button == 3)  # right-click = erase
                mx, my = event.pos
                gx = int(mx * GRID_W / DISPLAY_W)
                gy = int(my * GRID_H / DISPLAY_H)
                y0 = max(0, gy - WALL_BRUSH); y1 = min(GRID_H, gy + WALL_BRUSH + 1)
                x0 = max(0, gx - WALL_BRUSH); x1 = min(GRID_W, gx + WALL_BRUSH + 1)
                wall_mask[y0:y1, x0:x1] = not wall_erase
                jwall = jnp.array(wall_mask)

            if event.type == pygame.MOUSEMOTION and wall_drawing:
                mx, my = event.pos
                gx = int(mx * GRID_W / DISPLAY_W)
                gy = int(my * GRID_H / DISPLAY_H)
                y0 = max(0, gy - WALL_BRUSH); y1 = min(GRID_H, gy + WALL_BRUSH + 1)
                x0 = max(0, gx - WALL_BRUSH); x1 = min(GRID_W, gx + WALL_BRUSH + 1)
                wall_mask[y0:y1, x0:x1] = not wall_erase
                jwall = jnp.array(wall_mask)

            if event.type == pygame.MOUSEBUTTONUP:
                wall_drawing = False
                trail_drawing = False

            # ── Mouse: trail paint (left) or blob drop (right) ────────────────
            if event.type == pygame.MOUSEBUTTONDOWN and trail_mode and event.button == 3:
                # Right-click in trail mode = drop a single blob
                mx, my = event.pos
                gx = int(mx * GRID_W / DISPLAY_W)
                gy = int(my * GRID_H / DISPLAY_H)
                r  = BLOB_RADIUS
                y0, y1 = max(0, gy - r), min(GRID_H, gy + r)
                x0, x1 = max(0, gx - r), min(GRID_W, gx + r)
                h, w = y1 - y0, x1 - x0
                cy, cx = h / 2, w / 2
                yy, xx = np.ogrid[:h, :w]
                dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
                inside = dist <= r
                # A stays near 1 minus blob_strength so we don't flood the grid
                grid = grid.at[y0:y1, x0:x1, 0].set(jnp.array(np.where(inside, 1.0 - blob_strength, np.array(grid[y0:y1, x0:x1, 0]))))
                grid = grid.at[y0:y1, x0:x1, 1].set(jnp.array(np.where(inside, blob_strength, np.array(grid[y0:y1, x0:x1, 1]))))
                print(f"Blob dropped at ({gx},{gy})  B={blob_strength:.2f}")

            if event.type == pygame.MOUSEBUTTONDOWN and trail_mode and event.button == 1:
                trail_drawing = True
                trail_erase   = False
                mx, my = event.pos
                gx = int(mx * GRID_W / DISPLAY_W)
                gy = int(my * GRID_H / DISPLAY_H)
                y0 = max(0, gy - TRAIL_BRUSH); y1 = min(GRID_H, gy + TRAIL_BRUSH + 1)
                x0 = max(0, gx - TRAIL_BRUSH); x1 = min(GRID_W, gx + TRAIL_BRUSH + 1)
                trail_mask[y0:y1, x0:x1] = not trail_erase
                jtrail = jnp.array(trail_mask)

            if event.type == pygame.MOUSEMOTION and trail_drawing:
                mx, my = event.pos
                gx = int(mx * GRID_W / DISPLAY_W)
                gy = int(my * GRID_H / DISPLAY_H)
                y0 = max(0, gy - TRAIL_BRUSH); y1 = min(GRID_H, gy + TRAIL_BRUSH + 1)
                x0 = max(0, gx - TRAIL_BRUSH); x1 = min(GRID_W, gx + TRAIL_BRUSH + 1)
                trail_mask[y0:y1, x0:x1] = not trail_erase
                jtrail = jnp.array(trail_mask)

            # ── Mouse click — stamp attractor seed ────────────────────────────
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not wall_mode and not trail_mode:
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
                # Mask out wall cells — don't stamp over walls, only deposit on one side
                _stamp_mask = ~wall_mask[y0:y1, x0:x1]  # True where we CAN stamp
                # Write visible chemistry (A and B) — this is the key part.
                # Hidden-channel-only injection gets swamped in a few steps.
                # Writing A/B sets the actual reaction state the NCA evolves from.
                if _sp['a_val'] is not None:
                    _a_patch = np.where(_stamp_mask, _sp['a_val'], np.array(grid[y0:y1, x0:x1, 0]))
                    grid = grid.at[y0:y1, x0:x1, 0].set(jnp.array(_a_patch))
                if _sp['b_val'] is not None:
                    _b_patch = np.where(_stamp_mask, _sp['b_val'], np.array(grid[y0:y1, x0:x1, 1]))
                    grid = grid.at[y0:y1, x0:x1, 1].set(jnp.array(_b_patch))
                if _sp['zero_out']:
                    # Death seed — kill all hidden channels in region (skip wall cells)
                    for _chi in range(2, 13):
                        _cur = np.array(grid[y0:y1, x0:x1, _chi])
                        grid = grid.at[y0:y1, x0:x1, _chi].set(jnp.array(np.where(_stamp_mask, 0.0, _cur)))
                else:
                    if _sp['ch2_amp'] > 0:
                        _n2 = np.where(_stamp_mask, np.random.normal(0, _sp['ch2_amp'], (h, w)).astype(np.float32), 0.0)
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
            # Wall injection — force A=1.0, B=0.0 on wall cells every step
            if np.any(wall_mask):
                grid = grid.at[:, :, CH_A].set(jnp.where(jwall, 1.0, grid[:, :, CH_A]))
                grid = grid.at[:, :, CH_B].set(jnp.where(jwall, 0.0, grid[:, :, CH_B]))
            # Trail injection — force ch5 to trail_strength on painted cells every step
            if np.any(trail_mask):
                grid = grid.at[:, :, 5].set(jnp.where(jtrail, trail_strength, grid[:, :, 5]))
            # Trail drift — shift trail toward target direction every DRIFT_INTERVAL steps
            if (trail_drift_x != 0 or trail_drift_y != 0) and step_count % DRIFT_INTERVAL == 0:
                trail_mask = np.roll(trail_mask, shift=(trail_drift_y, trail_drift_x), axis=(0, 1))
                # Zero out wrapped edges so trail doesn't teleport
                if trail_drift_x > 0:  trail_mask[:, :trail_drift_x] = False
                elif trail_drift_x < 0: trail_mask[:, trail_drift_x:] = False
                if trail_drift_y > 0:  trail_mask[:trail_drift_y, :] = False
                elif trail_drift_y < 0: trail_mask[trail_drift_y:, :] = False
                jtrail = jnp.array(trail_mask)
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
            # Override: blank screen = extinction regardless of classifier
            if _feat[2] < 0.01 and _feat[8] == 0:  # b_active < 1%, n_blobs == 0
                _state_name = 'Extinction'
                _state_id   = -2
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
            # Write 4×4 zone density map for artist bridge
            if ARTIST_MODE:
                import json as _json
                _a_z = _grid_np[:,:,0]
                _zones = {}
                for _zr in range(4):
                    for _zc in range(4):
                        _zone_a = _a_z[_zr*64:(_zr+1)*64, _zc*64:(_zc+1)*64]
                        _zones[f'{_zr}{_zc}'] = round(float(np.mean(_zone_a < 0.80)), 3)
                _zone_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artist_state.json')
                with open(_zone_path, 'w') as _azf:
                    _json.dump({'zones': _zones, 'step': step_count}, _azf)

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

        # ── LLM command executor (shared by single-agent and battle mode) ──
        def _bresenham_cells(x0, y0, x1, y1, brush=2):
            """Return list of (y,x) grid cells along a line with given brush radius."""
            cells = set()
            dx, dy = abs(x1-x0), abs(y1-y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            cx, cy = x0, y0
            while True:
                for by in range(max(0,cy-brush), min(GRID_H,cy+brush+1)):
                    for bx in range(max(0,cx-brush), min(GRID_W,cx+brush+1)):
                        cells.add((by, bx))
                if cx == x1 and cy == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy; cx += sx
                if e2 < dx:
                    err += dx; cy += sy
            return list(cells)

        def _circle_cells(cx, cy, r):
            """Return list of (y,x) cells within radius r of (cx,cy)."""
            cells = []
            for gy in range(max(0,cy-r), min(GRID_H,cy+r+1)):
                for gx in range(max(0,cx-r), min(GRID_W,cx+r+1)):
                    if (gy-cy)**2 + (gx-cx)**2 <= r*r:
                        cells.append((gy,gx))
            return cells

        def _apply_mirror(fn):
            """If artist_mirror is on, call fn for both original and mirrored coords."""
            fn(False)
            if artist_mirror:
                fn(True)

        def _execute_llm_command(_cmd, _label='LLM'):
            nonlocal grid, f, k, f_field, k_field, jf_field, jk_field, key, regime_idx
            nonlocal step_count, auto_nudges
            if _cmd == 'inject_chaos':
                _noise = jnp.array(np.random.uniform(-0.5, 0.5, (GRID_H, GRID_W, 12)).astype(np.float32))
                grid = grid.at[:, :, 2:14].add(_noise)
                print(f"  [{_label}] inject_chaos")
            elif _cmd == 'reset':
                regime_idx = np.random.randint(0, len(regime_names))
                f, k = GS_REGIMES[regime_names[regime_idx]]
                key, sk = random.split(key)
                grid, key = init_nca_grid(sk, GRID_H, GRID_W, f, k)
                f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                jf_field = jnp.array(f_field)
                jk_field = jnp.array(k_field)
                print(f"  [{_label}] reset → {regime_names[regime_idx]}")
            elif _cmd in ('regime_0', 'regime_1', 'regime_2', 'regime_3'):
                _rmap = {'regime_0': 'uskate', 'regime_1': 'mitosis', 'regime_2': 'gliders', 'regime_3': 'maze'}
                _rname = _rmap[_cmd]
                f, k = GS_REGIMES[_rname]
                f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                jf_field = jnp.array(f_field)
                jk_field = jnp.array(k_field)
                print(f"  [{_label}] {_cmd} → {_rname}")
            else:
                # ── Spatial brush commands ─────────────────────────────────
                nonlocal trail_mask, jtrail, trail_strength, wall_mask, jwall
                nonlocal trail_drift_x, trail_drift_y
                nonlocal palette_idx, palette_target, palette_blend, render_mode, render_mode_idx
                nonlocal ch5_tint_color, artist_mirror, show_ch5_overlay
                # Strip any non-numeric words Gemini inserts (e.g. "strength 0.7" → "0.7")
                _parts = [p for p in _cmd.strip().split() if p not in ('strength', 'at', 'with', 'to')]
                _brush = _parts[0] if _parts else ''

                def _do_trail(mirror=False):
                    if len(_parts) < 5: return
                    x0,y0,x1,y1 = int(_parts[1]),int(_parts[2]),int(_parts[3]),int(_parts[4])
                    strength = float(_parts[5]) if len(_parts) > 5 else 0.5
                    if mirror:
                        x0,x1 = GRID_W-1-x0, GRID_W-1-x1
                    cells = _bresenham_cells(x0,y0,x1,y1, brush=2)
                    _g = np.array(grid[:,:,5])
                    for (gy,gx) in cells:
                        _g[gy,gx] = strength
                        trail_mask[gy,gx] = True
                    grid.__class__  # touch nonlocal
                    return _g, strength

                if _brush == 'trail':
                    _g = np.array(grid[:,:,5])
                    if len(_parts) >= 5:
                        x0,y0,x1,y1 = int(_parts[1]),int(_parts[2]),int(_parts[3]),int(_parts[4])
                        strength = float(_parts[5]) if len(_parts) > 5 else 0.5
                        for mirror in ([False,True] if artist_mirror else [False]):
                            mx0,mx1 = (GRID_W-1-x0,GRID_W-1-x1) if mirror else (x0,x1)
                            for (gy,gx) in _bresenham_cells(mx0,y0,mx1,y1,brush=2):
                                _g[gy,gx] = strength
                                trail_mask[gy,gx] = True
                        grid = grid.at[:,:,5].set(jnp.array(_g))
                        jtrail = jnp.array(trail_mask)
                        print(f"  [{_label}] trail ({x0},{y0})→({x1},{y1}) str={strength:.2f}")

                elif _brush == 'wall':
                    if len(_parts) >= 5:
                        x0,y0,x1,y1 = int(_parts[1]),int(_parts[2]),int(_parts[3]),int(_parts[4])
                        for mirror in ([False,True] if artist_mirror else [False]):
                            mx0,mx1 = (GRID_W-1-x0,GRID_W-1-x1) if mirror else (x0,x1)
                            for (gy,gx) in _bresenham_cells(mx0,y0,mx1,y1,brush=2):
                                wall_mask[gy,gx] = True
                        jwall = jnp.array(wall_mask)
                        print(f"  [{_label}] wall ({x0},{y0})→({x1},{y1})")

                elif _brush == 'pulse':
                    # Temporarily spike ch5 along a line — snap drifting organisms back
                    if len(_parts) >= 5:
                        x0,y0,x1,y1 = int(_parts[1]),int(_parts[2]),int(_parts[3]),int(_parts[4])
                        strength = float(_parts[5]) if len(_parts) > 5 else 0.8
                        _g = np.array(grid[:,:,5])
                        for mirror in ([False,True] if artist_mirror else [False]):
                            mx0,mx1 = (GRID_W-1-x0,GRID_W-1-x1) if mirror else (x0,x1)
                            for (gy,gx) in _bresenham_cells(mx0,y0,mx1,y1,brush=3):
                                _g[gy,gx] = strength
                        grid = grid.at[:,:,5].set(jnp.array(_g))
                        print(f"  [{_label}] pulse ({x0},{y0})→({x1},{y1}) str={strength:.2f}")

                elif _brush == 'wipe':
                    # wipe cx cy r — circular extinction zone (trail cells are protected)
                    if len(_parts) >= 4:
                        cx,cy,r = int(_parts[1]),int(_parts[2]),int(_parts[3])
                        _g_np = np.array(grid)
                        for mirror in ([False,True] if artist_mirror else [False]):
                            mcx = GRID_W-1-cx if mirror else cx
                            for (gy,gx) in _circle_cells(mcx,cy,r):
                                if trail_mask[gy,gx]:
                                    continue   # never kill trail cells
                                _g_np[gy,gx,0] = 1.0
                                _g_np[gy,gx,1] = 0.0
                                _g_np[gy,gx,2:14] = 0.0
                        grid = jnp.array(_g_np)
                        print(f"  [{_label}] wipe circle ({cx},{cy}) r={r} (trails protected)")

                elif _brush == 'wipe_rect':
                    # wipe_rect x1 y1 x2 y2 — clear rectangle (trail cells are protected)
                    if len(_parts) >= 5:
                        rx0,rx1 = min(int(_parts[1]),int(_parts[3])), max(int(_parts[1]),int(_parts[3]))
                        ry0,ry1 = min(int(_parts[2]),int(_parts[4])), max(int(_parts[2]),int(_parts[4]))
                        rx0,rx1 = max(0,rx0), min(GRID_W,rx1)
                        ry0,ry1 = max(0,ry0), min(GRID_H,ry1)
                        _g_np = np.array(grid)
                        # Build mask: cells to wipe = in rect AND not on trail
                        _wipe_region = np.zeros((GRID_H, GRID_W), dtype=bool)
                        _wipe_region[ry0:ry1, rx0:rx1] = True
                        _wipe_region &= ~trail_mask   # protect trail cells
                        _g_np[_wipe_region, 0] = 1.0
                        _g_np[_wipe_region, 1] = 0.0
                        _g_np[_wipe_region, 2:14] = 0.0
                        grid = jnp.array(_g_np)
                        print(f"  [{_label}] wipe_rect ({rx0},{ry0})→({rx1},{ry1}) (trails protected)")

                elif _brush == 'blob':
                    if len(_parts) >= 3:
                        bx,by = int(_parts[1]),int(_parts[2])
                        strength = float(_parts[3]) if len(_parts) > 3 else 0.25
                        r = 12   # larger than interactive BLOB_RADIUS — needs mass to bootstrap GS
                        _g_a = np.array(grid[:,:,0])
                        _g_b = np.array(grid[:,:,1])
                        for mirror in ([False,True] if artist_mirror else [False]):
                            mbx = GRID_W-1-bx if mirror else bx
                            for (gy,gx) in _circle_cells(mbx,by,r):
                                _g_a[gy,gx] = 1.0 - strength
                                _g_b[gy,gx] = strength
                        grid = grid.at[:,:,0].set(jnp.array(_g_a))
                        grid = grid.at[:,:,1].set(jnp.array(_g_b))
                        print(f"  [{_label}] blob ({bx},{by}) str={strength:.2f} r={r}")

                elif _brush == 'reset':
                    regime_idx = np.random.randint(0, len(regime_names))
                    _f, _k = GS_REGIMES[regime_names[regime_idx]]
                    _sk = random.PRNGKey(int(step_count) % 10000)
                    grid, _ = init_nca_grid(_sk, GRID_H, GRID_W, _f, _k)
                    f_field, k_field = make_fk_field(GRID_H, GRID_W, _f, _k, phase_fx, phase_fy, phase_kx, phase_ky)
                    jf_field = jnp.array(f_field)
                    jk_field = jnp.array(k_field)
                    step_count = 0
                    auto_nudges = 0
                    print(f"  [{_label}] reset → {regime_names[regime_idx]}  f={_f:.4f} k={_k:.4f}")

                elif _brush == 'shape':
                    # shape circle/ring/spiral cx cy r
                    # Also paints ch5 on the outline so organisms lock onto the shape
                    if len(_parts) >= 5:
                        shape_type = _parts[1]
                        cx,cy,r = int(_parts[2]),int(_parts[3]),int(_parts[4])
                        _g_a  = np.array(grid[:,:,0])
                        _g_b  = np.array(grid[:,:,1])
                        _g_ch5 = np.array(grid[:,:,5])
                        for mirror in ([False,True] if artist_mirror else [False]):
                            mcx = GRID_W-1-cx if mirror else cx
                            if shape_type == 'circle':
                                for (gy,gx) in _circle_cells(mcx,cy,r):
                                    _g_a[gy,gx] = 0.5; _g_b[gy,gx] = 0.5
                                    _g_ch5[gy,gx] = 0.8; trail_mask[gy,gx] = True
                            elif shape_type == 'ring':
                                for (gy,gx) in _circle_cells(mcx,cy,r):
                                    d = np.sqrt((gy-cy)**2+(gx-mcx)**2)
                                    if d >= r-3:
                                        _g_a[gy,gx] = 0.3; _g_b[gy,gx] = 0.6
                                        _g_ch5[gy,gx] = 0.8; trail_mask[gy,gx] = True
                            elif shape_type == 'spiral':
                                for angle in np.linspace(0, 4*np.pi, 300):
                                    rad = r * angle / (4*np.pi)
                                    gx = int(mcx + rad*np.cos(angle))
                                    gy = int(cy  + rad*np.sin(angle))
                                    if 0<=gy<GRID_H and 0<=gx<GRID_W:
                                        _g_a[gy,gx] = 0.3; _g_b[gy,gx] = 0.5
                                        _g_ch5[gy,gx] = 0.8; trail_mask[gy,gx] = True
                        grid = grid.at[:,:,0].set(jnp.array(_g_a))
                        grid = grid.at[:,:,1].set(jnp.array(_g_b))
                        grid = grid.at[:,:,5].set(jnp.array(_g_ch5))
                        jtrail = jnp.array(trail_mask)
                        print(f"  [{_label}] shape {shape_type} ({cx},{cy}) r={r} +ch5trail")

                elif _brush == 'wait':
                    pass   # no-op: Gemini uses this to skip turns and let organisms settle

                elif _brush == 'drift':
                    nonlocal trail_drift_x, trail_drift_y
                    if len(_parts) >= 2 and _parts[1] == 'stop':
                        trail_drift_x = trail_drift_y = 0
                        print(f"  [{_label}] drift stopped")
                    elif len(_parts) >= 3:
                        trail_drift_x = int(_parts[1])
                        trail_drift_y = int(_parts[2])
                        print(f"  [{_label}] drift ({trail_drift_x},{trail_drift_y}) px per {DRIFT_INTERVAL} steps")

                elif _brush == 'clear_trail':
                    trail_mask[:] = False
                    jtrail = jnp.zeros((GRID_H,GRID_W), dtype=bool)
                    trail_drift_x = trail_drift_y = 0
                    print(f"  [{_label}] clear_trail")

                elif _brush == 'clear_walls':
                    wall_mask[:] = False
                    jwall = jnp.zeros((GRID_H,GRID_W), dtype=bool)
                    print(f"  [{_label}] clear_walls")

                elif _brush == 'palette':
                    if len(_parts) >= 2:
                        _pname = _parts[1]
                        if _pname in PALETTES:
                            palette_idx    = palette_names.index(_pname)
                            palette_target = np.array(PALETTES[_pname], dtype=np.float32)
                            palette_blend  = 1   # trigger crossfade
                            print(f"  [{_label}] palette → {_pname}")
                        else:
                            print(f"  [{_label}] unknown palette: {_pname}. Options: {', '.join(palette_names)}")

                elif _brush == 'mode':
                    if len(_parts) >= 2 and _parts[1] in NCA_RENDER_MODES:
                        render_mode     = _parts[1]
                        render_mode_idx = NCA_RENDER_MODES.index(_parts[1])
                        print(f"  [{_label}] mode → {render_mode}")

                elif _brush == 'trailcolor':
                    if len(_parts) >= 4:
                        ch5_tint_color = (int(_parts[1]), int(_parts[2]), int(_parts[3]))
                        show_ch5_overlay = True
                        print(f"  [{_label}] trailcolor → {ch5_tint_color}")

                elif _brush == 'mirror':
                    if len(_parts) >= 2:
                        artist_mirror = (_parts[1] == 'on')
                        print(f"  [{_label}] mirror → {'ON' if artist_mirror else 'OFF'}")

        # ── LLM bridge command polling (single-agent) ─────────────────────
        if step_count % CMD_POLL_EVERY == 0 and os.path.exists(CMD_FILE):
            try:
                with open(CMD_FILE, 'r') as _cf:
                    _cmd = _cf.read().strip().lower()
                if _cmd and _cmd != 'none':
                    _execute_llm_command(_cmd)
                    with open(CMD_FILE, 'w') as _cf:
                        _cf.write('none')
            except Exception:
                pass   # never crash the main loop on bridge errors

        # ── Artist mode command polling ───────────────────────────────────
        if ARTIST_MODE and step_count % CMD_POLL_EVERY == 0 and os.path.exists(CMD_FILE_ARTIST):
            try:
                with open(CMD_FILE_ARTIST, 'r') as _cf:
                    _raw = _cf.read().strip()
                if _raw and _raw.lower() != 'none':
                    # Support multi-command COMMANDS: block or single line
                    _lines = []
                    _in_block = False
                    for _line in _raw.splitlines():
                        _line = _line.strip()
                        if _line.lower().startswith('commands:'):
                            _in_block = True
                            continue
                        if _line.lower().startswith('speech:') or _line.lower().startswith('reason:'):
                            _speech = _line.split(':', 1)[1].strip()
                            print(f"  [ARTIST] {_speech}")
                            continue
                        if _in_block and _line and not _line.startswith('#'):
                            _lines.append(_line)
                        elif not _in_block and _line and not _line.startswith('#'):
                            _lines.append(_line)
                    for _acmd in _lines:
                        if _acmd.lower() != 'none':
                            _execute_llm_command(_acmd.lower(), 'ARTIST')
                    with open(CMD_FILE_ARTIST, 'w') as _cf:
                        _cf.write('none')
            except Exception as _e:
                print(f"  [ARTIST] poll error: {_e}")

        # ── Battle mode command polling ───────────────────────────────────
        if BATTLE_MODE and step_count % CMD_POLL_EVERY == 0:
            try:
                # Read whose turn it is
                _battle_turn = None
                _battle_turn_num = 0
                if os.path.exists(TURN_FILE):
                    with open(TURN_FILE, 'r') as _tf:
                        _parts = _tf.read().strip().split()
                        if _parts:
                            _battle_turn = _parts[0]
                            _battle_turn_num = int(_parts[1]) if len(_parts) > 1 else 0

                # Execute keeper command if it's keeper's turn
                if _battle_turn == 'keeper' and os.path.exists(CMD_FILE_KEEPER):
                    with open(CMD_FILE_KEEPER, 'r') as _cf:
                        _bcmd = _cf.read().strip().lower()
                    if _bcmd and _bcmd != 'none':
                        _execute_llm_command(_bcmd, 'KEEPER')
                        with open(CMD_FILE_KEEPER, 'w') as _cf:
                            _cf.write('none')
                        # Advance turn after BATTLE_TURN_STEPS
                        _battle_next_turn_step = step_count + BATTLE_TURN_STEPS

                # Execute destroyer command if it's destroyer's turn
                elif _battle_turn == 'destroyer' and os.path.exists(CMD_FILE_DESTROYER):
                    with open(CMD_FILE_DESTROYER, 'r') as _cf:
                        _bcmd = _cf.read().strip().lower()
                    if _bcmd and _bcmd != 'none':
                        _execute_llm_command(_bcmd, 'DESTROYER')
                        with open(CMD_FILE_DESTROYER, 'w') as _cf:
                            _cf.write('none')
                        # Advance turn after BATTLE_TURN_STEPS
                        _battle_next_turn_step = step_count + BATTLE_TURN_STEPS
            except Exception:
                pass

        # Advance battle turn after N steps
        if BATTLE_MODE and os.path.exists(TURN_FILE) and _battle_next_turn_step == 0:
            _battle_next_turn_step = step_count + BATTLE_TURN_STEPS
        if _battle_next_turn_step > 0 and step_count >= _battle_next_turn_step:
            try:
                if os.path.exists(TURN_FILE):
                    with open(TURN_FILE, 'r') as _tf:
                        _parts = _tf.read().strip().split()
                        _cur = _parts[0] if _parts else 'keeper'
                        _tnum = int(_parts[1]) if len(_parts) > 1 else 1
                    _next = 'destroyer' if _cur == 'keeper' else 'keeper'
                    with open(TURN_FILE, 'w') as _tf:
                        _tf.write(f"{_next} {_tnum + 1}")
                    print(f"  [BATTLE] Turn {_tnum + 1} → {_next.upper()}")
                    _battle_next_turn_step = step_count + BATTLE_TURN_STEPS  # schedule next
            except Exception:
                pass

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

        # ── Wall overlay — draw wall cells as bright yellow lines ─────────────
        if np.any(wall_mask):
            cell_w = max(1, SCREEN_W // GRID_W)
            cell_h = max(1, SCREEN_H // GRID_H)
            wall_color = (255, 200, 0) if not wall_mode else (255, 100, 0)
            ys, xs = np.where(wall_mask)
            for gy, gx in zip(ys, xs):
                sx = gx * SCREEN_W // GRID_W
                sy = gy * SCREEN_H // GRID_H
                pygame.draw.rect(screen, wall_color, (sx, sy, cell_w, cell_h))

        # ── ch5 overlay — cyan heatmap showing spatial trail signal ───────────
        if show_ch5_overlay:
            ch5_np = np.array(grid[:, :, 5])
            ch5_min, ch5_max = ch5_np.min(), ch5_np.max()
            if ch5_max > ch5_min + 1e-8:
                ch5_norm = (ch5_np - ch5_min) / (ch5_max - ch5_min)
            else:
                ch5_norm = np.zeros_like(ch5_np)
            # pygame surfarray is (W, H) so transpose (1, 0)
            ch5_t = ch5_norm.T.astype(np.float32)
            ch5_surf_a = pygame.Surface((GRID_W, GRID_H), pygame.SRCALPHA)
            px = pygame.surfarray.pixels3d(ch5_surf_a)
            px[:, :, 0] = (ch5_t * ch5_tint_color[0]).astype(np.uint8)
            px[:, :, 1] = (ch5_t * ch5_tint_color[1]).astype(np.uint8)
            px[:, :, 2] = (ch5_t * ch5_tint_color[2]).astype(np.uint8)
            del px
            pa = pygame.surfarray.pixels_alpha(ch5_surf_a)
            pa[:, :] = (ch5_t * 180).astype(np.uint8)
            del pa
            ch5_scaled = pygame.transform.scale(ch5_surf_a, (DISPLAY_W, DISPLAY_H))
            screen.blit(ch5_scaled, (0, 0))

        palette_str = palette_names[palette_idx]
        free_str  = "  |  FREE-CH" if (free_channels and step_count >= FREE_WARMUP) else ""
        wall_str  = "  |  WALL-DRAW (D=clear)" if wall_mode else ("  |  walls" if np.any(wall_mask) else "")
        ch5_str   = "  |  CH5-OVERLAY" if show_ch5_overlay else ""
        trail_str = f"  |  TRAIL-PAINT str={trail_strength:.3f} (N=clear)" if trail_mode else ("  |  trail" if np.any(trail_mask) else "")
        hud = font.render(
            f"step {step_count}  |  bit={physics_bit:.0f}({'L' if physics_bit else 'G'}){free_str}{wall_str}{trail_str}{ch5_str}  f={f:.4f}±{FK_SPATIAL_AMP_F} k={k:.4f}±{FK_SPATIAL_AMP_K}  |  {palette_str}  |  {render_mode}+{effect}  |  spd={steps_per_frame}  |  K=wipe Y=trail(L=draw R=blob) [/]=str N=clear W=walls D=clear X=ch5 T=physics A=sound M=mode E=effect P=palette F=poke Z=chaos S=save",
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
