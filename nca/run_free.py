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
from gs.engine import GS_REGIMES, gs_step, init_gs_grid
from nca.params import PALETTES

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(
    os.path.dirname(__file__), 'checkpoints', 'params_050000.pkl'
)

GRID_H          = 256
GRID_W          = 256
SCREEN_W        = 1920  # one monitor width
SCREEN_H        = 1080
DUAL_SCREEN     = True  # set False for single monitor
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
SATURATION_CHECK   = 150   # steps between saturation checks
SATURATION_STD     = 0.02  # B std below this = stuck, escape

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
PERTURB_SPACING_MIN  = 5     # min steps between pokes in a sequence
PERTURB_SPACING_MAX  = 15    # max steps between pokes in a sequence

EXTREME_INTERVAL_MIN = 4000  # min steps between autonomous extreme bursts (~2min)
EXTREME_INTERVAL_MAX = 9000  # max steps between autonomous extreme bursts (~5min)

RESEED_INTERVAL_MIN  = 8000  # min steps between autonomous reseeds (~4min)
RESEED_INTERVAL_MAX  = 18000 # max steps between autonomous reseeds (~10min)

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
def render(surface, grid, palette):
    """
    Render both A and B channels.

    B (predator) drives the foreground — mapped through the palette as before.
    A (food) drives the background — it depletes where B is active, creating
    texture and depth in areas that used to be flat color.

    We blend them: the final color is B's palette color modulated by A's value.
    Where A is high (undepleted food) the background glows slightly.
    Where A is low (consumed by B) it goes dark, adding depth behind the patterns.
    """
    B = np.array(grid[:, :, CH_B])  # (H, W) foreground patterns
    A = np.array(grid[:, :, CH_A])  # (H, W) background texture

    p    = np.array(palette, dtype=np.float32) / 255.0
    t    = np.clip(B * 3.0, 0.0, 3.0)
    idx  = np.floor(t).astype(int).clip(0, 2)
    frac = (t - idx)[..., None]
    c0   = p[idx]
    c1   = p[idx + 1]
    rgb  = (c0 + frac * (c1 - c0))

    # Modulate by A channel — depleted food darkens the background slightly,
    # undepleted food adds a faint glow. Keeps it subtle so B still dominates.
    A_mod = (0.6 + 0.4 * A)[..., None]  # range 0.6–1.0, never fully dark
    rgb   = rgb * A_mod

    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    img    = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
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
    with open(CHECKPOINT, 'rb') as f:
        params = pickle.load(f)
    params = jax.device_put(params)
    print("Loaded.")

    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()
    step_fn           = make_step_fn(update_net, perception_kernel)

    # ── Starting regime ───────────────────────────────────────────────────
    regime_names = list(GS_REGIMES.keys())
    regime_idx   = np.random.randint(0, len(regime_names))
    f, k         = GS_REGIMES[regime_names[regime_idx]]

    # ── Starting palette ──────────────────────────────────────────────────
    palette_names = list(PALETTES.keys())
    palette_idx   = 0
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

    step_count      = 0
    running         = True
    auto_nudges     = 0
    steps_per_frame = STEPS_PER_FRAME

    # Perturbation sequence state
    next_perturb    = np.random.randint(PERTURB_INTERVAL_MIN, PERTURB_INTERVAL_MAX)
    pokes_remaining = 0
    next_poke       = 0
    extreme_mode    = False

    # Autonomous extreme burst schedule
    next_extreme    = np.random.randint(EXTREME_INTERVAL_MIN, EXTREME_INTERVAL_MAX)

    # Autonomous reseed schedule
    next_reseed     = np.random.randint(RESEED_INTERVAL_MIN, RESEED_INTERVAL_MAX)

    # Palette crossfade state
    palette_current = np.array(palette, dtype=np.float32)
    palette_target  = palette_current.copy()
    palette_blend   = 0   # counts up to PALETTE_BLEND_STEPS, then resets
    next_palette_change = np.random.randint(PALETTE_CHANGE_MIN, PALETTE_CHANGE_MAX)

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

                if event.key == pygame.K_r:
                    key, sk = random.split(key)
                    print(f"Resetting (GS warmup)...")
                    grid, key = init_nca_grid(sk, GRID_H, GRID_W, f, k)
                    f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                    jf_field = jnp.array(f_field)
                    jk_field = jnp.array(k_field)
                    step_count  = 0
                    auto_nudges = 0
                    print(f"Reset done.")

                if event.key == pygame.K_f:
                    f = float(np.random.uniform(F_MIN, F_MAX))
                    k = float(np.random.uniform(K_MIN, K_MAX))
                    f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                    jf_field = jnp.array(f_field)
                    jk_field = jnp.array(k_field)
                    print(f"Manual poke → f_center={f:.4f}  k_center={k:.4f}")

                if event.key == pygame.K_p:
                    palette_idx     = (palette_idx + 1) % len(palette_names)
                    palette_target  = np.array(PALETTES[palette_names[palette_idx]], dtype=np.float32)
                    palette_blend   = 0
                    print(f"Palette: {palette_names[palette_idx]}")

                if event.key == pygame.K_RIGHTBRACKET:
                    steps_per_frame = min(steps_per_frame + 1, 20)
                    print(f"Speed: {steps_per_frame} steps/frame")

                if event.key == pygame.K_LEFTBRACKET:
                    steps_per_frame = max(steps_per_frame - 1, 1)
                    print(f"Speed: {steps_per_frame} steps/frame")

                if event.key == pygame.K_x:
                    # Extreme burst — 4 rapid pokes from the wild regime list
                    # including values beyond the training range
                    pokes_remaining = 4
                    next_poke       = step_count
                    # Override normal perturbation to use extreme values
                    extreme_mode    = True
                    print(f"EXTREME BURST fired")

        # ── NCA steps ─────────────────────────────────────────────────────
        for _ in range(steps_per_frame):
            grid, key = step_fn(grid, params, key)
            # Re-inject spatial f/k after every NCA step so cells always read
            # their local value, not whatever the NCA accidentally wrote to those channels
            grid = grid.at[:, :, CH_F].set(jf_field)
            grid = grid.at[:, :, CH_K].set(jk_field)
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
        if step_count % DRIFT_EVERY == 0:
            df = np.random.uniform(-DRIFT_AMOUNT, DRIFT_AMOUNT)
            dk = np.random.uniform(-DRIFT_AMOUNT, DRIFT_AMOUNT)
            f  = float(np.clip(f + df, F_MIN, F_MAX))
            k  = float(np.clip(k + dk, K_MIN, K_MAX))
            f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
            jf_field = jnp.array(f_field)
            jk_field = jnp.array(k_field)

        # ── Autonomous extreme burst ──────────────────────────────────────
        if step_count >= next_extreme and pokes_remaining == 0:
            pokes_remaining = 4
            next_poke       = step_count
            extreme_mode    = True
            next_extreme    = step_count + np.random.randint(EXTREME_INTERVAL_MIN, EXTREME_INTERVAL_MAX)
            print(f"Auto extreme burst (next in {next_extreme - step_count} steps)")

        # ── Autonomous reseed ─────────────────────────────────────────────
        # Drop a fresh GS-warmed seed mid-run. The NCA gets a completely new
        # starting structure to grow from — breaks long attractor loops.
        # Also randomize the field phases so the spatial landscape is fresh.
        if step_count >= next_reseed:
            f           = float(np.random.uniform(F_MIN, F_MAX))
            k           = float(np.random.uniform(K_MIN, K_MAX))
            key, sk     = random.split(key)
            grid, key   = init_nca_grid(sk, GRID_H, GRID_W, f, k)
            # Fresh phase offsets — spatial landscape starts from a new configuration
            phase_fx = np.random.uniform(0, 2 * np.pi)
            phase_fy = np.random.uniform(0, 2 * np.pi)
            phase_kx = np.random.uniform(0, 2 * np.pi)
            phase_ky = np.random.uniform(0, 2 * np.pi)
            vel_fx = FK_PHASE_DRIFT * np.random.uniform(0.7, 1.3)
            vel_fy = FK_PHASE_DRIFT * np.random.uniform(0.7, 1.3)
            vel_kx = FK_PHASE_DRIFT * np.random.uniform(0.7, 1.3)
            vel_ky = FK_PHASE_DRIFT * np.random.uniform(0.7, 1.3)
            f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
            jf_field = jnp.array(f_field)
            jk_field = jnp.array(k_field)
            next_reseed = step_count + np.random.randint(RESEED_INTERVAL_MIN, RESEED_INTERVAL_MAX)
            print(f"Auto reseed → f={f:.4f} k={k:.4f}  (next in {next_reseed - step_count} steps)")

        # ── Perturbation sequences ────────────────────────────────────────
        # Periodically disturb the pattern with a burst of regime changes.
        # Each poke writes new f/k mid-reaction — the NCA treats it as damage
        # and reorganizes. Space them out to avoid solid-screen collapse.

        # Start a new sequence
        if pokes_remaining == 0 and step_count >= next_perturb:
            pokes_remaining = np.random.randint(PERTURB_POKES_MIN, PERTURB_POKES_MAX + 1)
            next_poke       = step_count
            print(f"Perturbation: {pokes_remaining} pokes incoming...")

        # Fire the next poke in the active sequence
        if pokes_remaining > 0 and step_count >= next_poke:
            if extreme_mode:
                # Wide random range — well beyond training data
                f = float(np.random.uniform(0.004, 0.080))
                k = float(np.random.uniform(0.038, 0.075))
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
                extreme_mode = False
                next_perturb = step_count + np.random.randint(PERTURB_INTERVAL_MIN, PERTURB_INTERVAL_MAX)
                print(f"  next sequence in {next_perturb - step_count} steps")

        # ── Saturation detection ──────────────────────────────────────────
        # If B channel goes nearly uniform (std too low), the NCA is stuck.
        # Jump to the next regime to escape.
        if step_count % SATURATION_CHECK == 0:
            b_std = float(jnp.std(grid[:, :, CH_B]))
            if b_std < SATURATION_STD:
                f           = float(np.random.uniform(F_MIN, F_MAX))
                k           = float(np.random.uniform(K_MIN, K_MAX))
                key, sk     = random.split(key)
                grid, key   = init_nca_grid(sk, GRID_H, GRID_W, f, k)
                f_field, k_field = make_fk_field(GRID_H, GRID_W, f, k, phase_fx, phase_fy, phase_kx, phase_ky)
                jf_field = jnp.array(f_field)
                jk_field = jnp.array(k_field)
                auto_nudges += 1
                print(f"Saturated (std={b_std:.4f}) → full reseed #{auto_nudges}  f={f:.4f} k={k:.4f}")

        # ── Autonomous palette crossfade ──────────────────────────────────
        if step_count >= next_palette_change and palette_blend == 0:
            new_idx             = np.random.randint(0, len(palette_names))
            palette_idx         = new_idx
            palette_target      = np.array(PALETTES[palette_names[new_idx]], dtype=np.float32)
            palette_blend       = 1
            next_palette_change = step_count + np.random.randint(PALETTE_CHANGE_MIN, PALETTE_CHANGE_MAX)
            print(f"Palette → {palette_names[new_idx]}")

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
        render(screen, grid, blended_palette.astype(np.uint8).tolist())

        palette_str = palette_names[palette_idx]
        hud = font.render(
            f"step {step_count}  |  f={f:.4f}±{FK_SPATIAL_AMP_F} k={k:.4f}±{FK_SPATIAL_AMP_K}  |  {palette_str}  |  speed={steps_per_frame}  |  [/] speed  R=reset F=poke X=extreme P=palette Q=quit",
            True, (80, 80, 80)
        )
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        ticker.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    print(f"JAX devices: {jax.devices()}")
    run()
