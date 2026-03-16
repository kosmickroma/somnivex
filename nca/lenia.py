# nca/lenia.py — Lenia simulator and pool builder.
#
# FitzHugh-Nagumo was too similar to GS (same reaction-diffusion family).
# Lenia is fundamentally different: discrete GS fills the whole grid with
# chemistry, Lenia produces discrete moving CREATURES in empty space.
#
# The experiment: fine-tune the GS NCA on both simultaneously.
# Hope: GS physics with Lenia creatures — solitons with internal chemistry.
#
# Channel layout for Lenia pool states:
#   ch 0       = A (Lenia activation field, [0,1])
#   ch 1       = 0 (not supervised by Lenia — GS owns ch1)
#   ch 2-12    = 0 (hidden, NCA fills during training)
#   ch 13      = 1.0 (physics bit: 0=GS, 1=Lenia — re-injected every step)
#   ch 14      = mu (Lenia growth center, like GS f)
#   ch 15      = sigma (Lenia growth width, like GS k)

import numpy as np
import jax
import jax.numpy as jnp

from nca.model import N_CHANNELS, CH_F, CH_K

LENIA_DT   = 0.1
LENIA_R    = 13    # kernel radius — covers all single-ring creatures
CH_PHYSICS = 13    # physics bit channel (re-injected from outside, not learned)

# ── Official Orbium initial condition ─────────────────────────────────────────
# Source: Bert Chan, animals.json ("O2b"), https://github.com/Chakazul/Lenia
# This is the exact published starting pattern that produces a stable glider.
# Parameters: R=13, mu=0.15, sigma=0.014 (canonical Orbium bicaudatus)

_ORBIUM_RLE = (
    "13.pK$14.qV$6.VpA.MpEpKpITqV$4.BpPpNrIrEqDpWpOpLpUqNvT$"
    "4.IqRrNsPsKqHJ3.GqOuC$4.TrLsTrPrLpS6.uUD$3.SpWqNrBqLpRqPqE6.vA$"
    "2.FpTpMLpHqPqHrVsPrS5.qUqA$K.pCpRG.ErFsRsVuSuPqN4.CrR$"
    "pA.pTU3.rWuBuRvXwTwKpF4.rCH$.tPqHH3.qFvAwUwVyJyKwNL2.DqLR$"
    ".pGsGA4.vPxSyDxE2yOuHS.XqJT$2.xIE4.sCyHyOvLvRyFxCsGpVpXqGP$"
    "2.VsU4.DxQyOvVuSwDwQuBrMqSqCF$3.vG5.tEyKwVvIvKvMtVrXqTpM$"
    "4.sU4.qFvDwMvNuUuDsUrKqDO$4.qCrDJ2.pPsKuGuHtOsQrNqKpC$"
    "5.pTqTpVpNqFrJsGsKrVrDqFpFD$6.QqCqJqPqVqXqRqHpOTC$8.LWpFpEXPG!"
)

def _ch2val(c):
    if c in '.b': return 0
    elif c == 'o': return 255
    elif len(c) == 1: return ord(c) - ord('A') + 1
    else: return (ord(c[0]) - ord('p')) * 24 + (ord(c[1]) - ord('A') + 25)

def _decode_orbium_rle(st):
    """Decode Lenia RLE string → 2D numpy array, values in [0, 1]."""
    rows, current_row = [], []
    last, count = '', ''
    for ch in st.rstrip('!'):
        if ch.isdigit():
            count += ch
        elif ch in 'pqrstuvwxy':
            last = ch
        elif ch == '$':
            n = int(count) if count else 1
            rows.append(current_row)
            rows.extend([[] for _ in range(n - 1)])
            current_row, last, count = [], '', ''
        else:
            val = _ch2val(last + ch) / 255.0
            n = int(count) if count else 1
            current_row.extend([val] * n)
            last, count = '', ''
    if current_row:
        rows.append(current_row)
    max_len = max(len(r) for r in rows)
    for r in rows:
        r.extend([0.0] * (max_len - len(r)))
    return np.array(rows, dtype=np.float32)

# Decode once at import time — 20x20 array, values in [0, 1]
ORBIUM_SEED = _decode_orbium_rle(_ORBIUM_RLE)

# Known creature parameters (canonical Lenia values from Chan 2019).
LENIA_CREATURES = {
    "orbium": dict(mu=0.150, sigma=0.014),   # canonical Orbium bicaudatus
}
CREATURE_NAMES = list(LENIA_CREATURES.keys())


# ── Kernel ─────────────────────────────────────────────────────────────────────

def _growth_np(U, m, s):
    """
    Gaussian growth function — Orbium uses gn=1 (field_func[1] in Chan's code).
    exp(-(U-m)^2 / (2*s^2)) * 2 - 1
    Range: (-1, +1). Peaks at U=m with value +1, falls to -1 far from m.
    """
    return np.exp(-((U - m) / s)**2 / 2.0) * 2.0 - 1.0

def make_kernel(R):
    """
    Exponential bump kernel — Orbium uses kn=1 (kernel_core[1] in Chan's code).
    shape: exp(4 - 4 / (4*D*(1-D)))  for D in (0,1), zero outside.
    Peaks sharply at D=0.5 (distance R/2 from center).
    This is the kernel the Orbium initial condition was evolved with.
    """
    mid = R
    r, c = np.ogrid[-mid:mid, -mid:mid]
    D = np.sqrt(r**2 + c**2) / R      # normalized distance [0, 1]
    inside = D < 1.0
    denom = 4.0 * D * (1.0 - D)
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(inside & (denom > 1e-6), np.exp(4.0 - 4.0 / denom), 0.0)
    K = K / K.sum()
    return K.astype(np.float32)

def make_kernel_fft(R, H, W):
    """
    Kernel FFT padded to grid size (H, W).

    The kernel must be embedded in the full H x W grid (centered at H//2, W//2),
    THEN fftshift moves the center to [0,0], THEN fft2.
    This correctly places negative lags at the wrap-around end of the array.

    The old approach (fftshift small kernel, then fft2 with s=(H,W)) was wrong:
    it padded zeros in the middle, putting negative lags at rows 13-25 instead
    of rows H-13 to H-1, causing U values 4x too low.
    """
    K_small = make_kernel(R)       # 26x26
    K_full  = np.zeros((H, W), dtype=np.float32)
    # Center the small kernel in the full grid
    r0 = H // 2 - R
    c0 = W // 2 - R
    K_full[r0:r0 + K_small.shape[0], c0:c0 + K_small.shape[1]] = K_small
    # fftshift on full grid moves center (H//2, W//2) to (0, 0)
    fK = np.fft.fft2(np.fft.fftshift(K_full))
    return fK.astype(np.complex64)


# ── Lenia step (numpy, CPU) ────────────────────────────────────────────────────

def lenia_step_np(A, fK_np, mu, sigma):
    """
    One Lenia step on CPU using numpy.
    Used for pool building — not training (no JAX/GPU needed here).

    U = convolution of A with kernel K  (the "neighborhood potential")
    G = growth function: bell(U, mu, sigma)*2 - 1  →  range (-1, +1)
    A_next = clip(A + dt * G, 0, 1)
    """
    U = np.real(np.fft.ifft2(fK_np * np.fft.fft2(A)))
    G = _growth_np(U, mu, sigma)
    return np.clip(A + LENIA_DT * G, 0.0, 1.0).astype(np.float32)


# ── Initial conditions ─────────────────────────────────────────────────────────

def seed_creature(H, W, rng):
    """
    Place the official Orbium initial condition (20x20) at a random position.

    Uses the exact published seed from Chan 2019 — the hand-crafted ring seed
    does NOT work because sigma=0.014 requires a precise starting distribution.
    Small noise added to break rotational symmetry so creature can start moving.
    """
    A = np.zeros((H, W), dtype=np.float32)
    sh, sw = ORBIUM_SEED.shape   # 20 x 20

    # Random placement with margins so creature doesn't start at edges
    margin = sh
    cx = int(rng.integers(margin, H - margin))
    cy = int(rng.integers(margin, W - margin))

    r0, r1 = cx, cx + sh
    c0, c1 = cy, cy + sw

    # Clamp to grid bounds (shouldn't happen with margins but be safe)
    seed_r0 = max(0, -cx)
    seed_c0 = max(0, -cy)
    r0, r1 = max(0, r0), min(H, r1)
    c0, c1 = max(0, c0), min(W, c1)

    A[r0:r1, c0:c1] = ORBIUM_SEED[seed_r0:seed_r0+(r1-r0), seed_c0:seed_c0+(c1-c0)]
    return A


# ── Pool state builder ─────────────────────────────────────────────────────────

def make_lenia_pool_state(H, W, fK_np, rng):
    """
    Generate one 16-channel NCA grid seeded with a Lenia state.

    Pool composition (to teach the NCA more than just 'maintain a clean ring'):
      10% empty grids    — zero is a valid Lenia stable state, learn it
      40% normal         — clean creatures from the known catalog
      30% mutated        — ±0.005 noise on mu/sigma, explores nearby params
      20% damaged        — sector removed mid-warmup, teaches repair not just continuation
    """
    r = rng.random()
    creature_name = rng.choice(CREATURE_NAMES)
    c = LENIA_CREATURES[creature_name]
    mu    = float(c['mu'])
    sigma = float(c['sigma'])

    if r < 0.10:
        # Empty grid — NCA must learn that zero is stable under Lenia rules
        A = np.zeros((H, W), dtype=np.float32)

    else:
        if 0.50 <= r < 0.80:
            # Mutated: small parameter drift
            mu    = float(np.clip(mu    + rng.uniform(-0.005, 0.005), 0.05, 0.40))
            sigma = float(np.clip(sigma + rng.uniform(-0.003, 0.003), 0.005, 0.08))
        # r >= 0.80: damaged (sector removed mid-warmup below)

        A = seed_creature(H, W, rng)
        warmup = rng.integers(150, 350)
        damaged = (r >= 0.80)

        for i in range(warmup):
            A = lenia_step_np(A, fK_np, mu, sigma)
            if damaged and i == warmup // 2:
                # Cut a random disk out of the creature at the halfway point
                cx  = rng.integers(H // 3, 2 * H // 3)
                cy  = rng.integers(W // 3, 2 * W // 3)
                rad = rng.integers(4, 12)
                yr, xr = np.ogrid[:H, :W]
                mask = (yr - cx)**2 + (xr - cy)**2 < rad**2
                A[mask] = 0.0

    # Pack into 16-channel NCA grid
    grid = np.zeros((H, W, N_CHANNELS), dtype=np.float32)
    grid[:, :, 0]          = A      # ch0 = Lenia activation
    # ch1 left at 0 — Lenia has one state variable, not two
    grid[:, :, CH_PHYSICS] = 1.0   # physics bit
    grid[:, :, CH_F]       = mu
    grid[:, :, CH_K]       = sigma
    return grid


def init_lenia_pool(pool_size, H, W):
    """Build the full Lenia pool. Returns (pool array, fK_np for reuse)."""
    fK_np = make_kernel_fft(LENIA_R, H, W)
    rng   = np.random.default_rng(seed=42)

    print(f"Initializing Lenia pool ({pool_size} states at {H}x{W})...")
    pool = np.zeros((pool_size, H, W, N_CHANNELS), dtype=np.float32)

    for i in range(pool_size):
        pool[i] = make_lenia_pool_state(H, W, fK_np, rng)
        if (i + 1) % 64 == 0:
            print(f"  {i+1}/{pool_size}")

    print("Lenia pool ready.\n")
    return pool, fK_np


# ── JAX training target fn ─────────────────────────────────────────────────────

def make_lenia_target_fn(fK_jax):
    """
    Returns a jitted function that runs one Lenia step on a batch.
    fK_jax is closed over — precomputed once, never changes.

    Usage:
        lenia_step_batch = make_lenia_target_fn(jnp.array(fK_np))
        A_next = lenia_step_batch(batch_A, batch_mu, batch_sigma)
    """
    @jax.jit
    def lenia_targets_batch(batch_A, batch_mu, batch_sigma):
        """
        batch_A:     (B, H, W) float32 — current Lenia state
        batch_mu:    (B,)      float32 — mu per sample
        batch_sigma: (B,)      float32 — sigma per sample
        Returns:     (B, H, W) float32 — next Lenia state
        """
        def step_one(A, mu, sigma):
            U = jnp.real(jnp.fft.ifft2(fK_jax * jnp.fft.fft2(A)))
            G = jnp.exp(-((U - mu) / sigma)**2 / 2.0) * 2.0 - 1.0
            return jnp.clip(A + LENIA_DT * G, 0.0, 1.0)
        return jax.vmap(step_one)(batch_A, batch_mu, batch_sigma)

    return lenia_targets_batch
