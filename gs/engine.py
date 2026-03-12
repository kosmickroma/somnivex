# gs/engine.py — Gray-Scott reaction-diffusion engine.
#
# Two chemicals A and B react and diffuse across a grid.
# A is the "food" (starts everywhere). B is the "predator" (starts in patches).
# B consumes A, but needs A to survive. The tension between them produces
# spirals, waves, spots, worms — all without any neural net or training.
#
# The magic is in F (feed rate) and k (kill rate).
# Different F/k values = completely different visual regimes.
# Slowly drifting F and k = the world morphs from one regime to another.

import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np

Da = 1.0   # diffusion rate of A (faster — spreads everywhere)
Db = 0.5   # diffusion rate of B (slower — concentrates into structures)
dt = 1.0   # time step (standard for this kernel scaling)

# The Laplacian kernel — measures how different a cell is from its neighbors.
# This is what makes B "diffuse" outward from high-concentration areas.
# The specific weights (0.05 corner, 0.20 edge, -1.0 center) are the
# standard Gray-Scott discrete approximation.
_LAPLACIAN = jnp.array([
    [0.05, 0.20, 0.05],
    [0.20, -1.0, 0.20],
    [0.05, 0.20, 0.05],
], dtype=jnp.float32)[None, None]  # shape (1,1,3,3) for conv

# ── Named parameter regimes ───────────────────────────────────────────────────
# These are the "characters" the world can have. Each (F, k) pair produces
# a visually distinct type of behavior. All are from Pearson's 1993 paper
# and the mrob.com xmorphia catalog — decades of known-good parameters.
GS_REGIMES = {
    # Original 7 — proven performers
    "spirals":     (0.012, 0.045),  # massive spinning spirals — never stops
    "chaos":       (0.020, 0.045),  # churning wave chaos — most energetic
    "waves":       (0.014, 0.047),  # clean interference wave patterns
    "worms":       (0.026, 0.055),  # branching worm-like structures
    "mitosis":     (0.030, 0.063),  # spots that divide like cells
    "solitons":    (0.022, 0.059),  # moving spots with long trails
    "coral":       (0.060, 0.062),  # coral/dendritic branching growth
    # New — from Pearson 1993 + xmorphia catalog
    "maze":        (0.029, 0.057),  # tight branching labyrinth — fills whole grid
    "fingerprint": (0.028, 0.053),  # flowing parallel curved stripes
    "gliders":     (0.034, 0.063),  # small autonomous structures drifting across grid
    "bacteria":    (0.046, 0.065),  # dense spot-division, feels cellular/alive
    "stripes":     (0.050, 0.063),  # long parallel wave stripes — clean and rhythmic
    "holes":       (0.039, 0.058),  # inverse of worms: dark holes in bright substrate
    "uskate":      (0.010, 0.047),  # u-skate world — extremely complex multi-spiral
    "pulsing":     (0.025, 0.060),  # stationary dots that breathe in and out
}


def _laplacian(u):
    """
    Apply Laplacian convolution to a 2D array with wrap padding.
    Wrap = torus topology, no edge artifacts.
    u shape: (H, W) → output: (H, W)
    """
    u_pad = jnp.pad(u[None, None], ((0,0),(0,0),(1,1),(1,1)), mode='wrap')
    out = lax.conv_general_dilated(
        u_pad, _LAPLACIAN,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    return out[0, 0]


def gs_step(A, B, f, k):
    """
    One Gray-Scott step. The core equations:

      reaction = A * B * B          (B catalyzes its own creation using A)
      dA = Da*∇²A  - reaction  + f*(1-A)   (A diffuses, gets consumed, gets fed)
      dB = Db*∇²B  + reaction  - (f+k)*B   (B diffuses, gets created, dies off)

    f (feed rate): how fast A is replenished.
      Low f = A runs out fast, B dies → stable spots
      High f = A abundant, B thrives → chaotic waves

    k (kill rate): how fast B dies.
      Low k = B persists → spirals, waves
      High k = B dies quickly → isolated spots, mitosis
    """
    lap_A = _laplacian(A)
    lap_B = _laplacian(B)

    reaction = A * B * B

    dA = Da * lap_A - reaction + f * (1.0 - A)
    dB = Db * lap_B + reaction - (f + k) * B

    A = jnp.clip(A + dt * dA, 0.0, 1.0)
    B = jnp.clip(B + dt * dB, 0.0, 1.0)
    return A, B


def init_gs_grid(key, H, W):
    """
    Initialize the Gray-Scott grid.
    A = 1.0 everywhere (food is abundant), tiny noise added.
    B = 0.0 everywhere, then random patches of B=0.25 seeded in.

    The B patches are the "spark" — they start the reaction.
    More patches = more fronts = more complex interference early on.
    """
    key, sk = random.split(key)
    A = jnp.ones((H, W)) - random.uniform(sk, (H, W)) * 0.04

    B = jnp.zeros((H, W))
    key, sk = random.split(key)
    n_patches = int(random.randint(sk, (), 8, 24))

    for _ in range(n_patches):
        key, sk1, sk2, sk3 = random.split(key, 4)
        y  = int(random.randint(sk1, (), 0, H - 12))
        x  = int(random.randint(sk2, (), 0, W - 12))
        sz = int(random.randint(sk3, (), 4, 12))
        B  = B.at[y:y+sz, x:x+sz].set(0.25 + np.random.uniform(0, 0.1))

    return A, B


def make_gs_step_fn(n_steps):
    """
    JIT-compiled multi-step Gray-Scott function.
    Runs n_steps per call. GS is cheap so we can afford 30+.
    f and k are arguments (not closed over) so drifting them mid-run
    doesn't cause recompilation — JAX traces them as abstract values.
    """
    @jax.jit
    def step_fn(A, B, f, k):
        for _ in range(n_steps):
            A, B = gs_step(A, B, f, k)
        return A, B
    return step_fn
