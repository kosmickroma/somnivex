# nca/model.py — NCA engine.
# A grid of cells each with 16 numbers describing their state.
# Every step: perceive neighbors → tiny neural net → update state.
# Iterate thousands of times → complex emergent patterns with no explicit rules.

import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn

# ── Constants ─────────────────────────────────────────────────────────────────
N_CHANNELS   = 16     # state values per cell (4 visible + 12 hidden)
N_FILTERS    = 4      # perception filters: identity, sobel_x, sobel_y, laplacian
HIDDEN_SIZE  = 128    # neurons in the update net hidden layer
ALIVE_THRESH = 0.1    # alpha above this = cell is alive
FIRE_RATE    = 0.75   # probability any cell actually updates each step
                      # stochastic async = organic irregular feel vs rigid wave-fronts


# ── Grid initialization ───────────────────────────────────────────────────────
def init_grid(key, H, W):
    """
    5 seed modes — picked randomly each world startup.
    Mode 4 (global_noise) is new: fills the whole grid with sparse random values
    so the world wakes up everywhere at once instead of growing from one point.
    """
    grid = jnp.zeros((H, W, N_CHANNELS))
    key, sk = random.split(key)
    mode = int(random.randint(sk, (), 0, 5))

    if mode == 0:
        # Center patch — classic, grows outward from middle
        shape_name = "patch"
        cx, cy = W // 2, H // 2
        r = 8
        key, sk = random.split(key)
        patch = random.uniform(sk, (r*2, r*2, N_CHANNELS))
        patch = patch.at[:, :, 3].set(0.5)
        grid  = grid.at[cy-r:cy+r, cx-r:cx+r, :].set(patch)

    elif mode == 1:
        # Single point — very slow careful growth from one cell
        shape_name = "point"
        key, sk = random.split(key)
        seed = random.uniform(sk, (1, 1, N_CHANNELS))
        seed = seed.at[:, :, 3].set(1.0)
        grid = grid.at[H//2, W//2, :].set(seed[0, 0])

    elif mode == 2:
        # Ring — growth happens inward AND outward simultaneously
        shape_name = "ring"
        cx, cy, r = W//2, H//2, 30
        ys = jnp.arange(H)
        xs = jnp.arange(W)
        yy, xx = jnp.meshgrid(ys, xs, indexing='ij')
        dist  = jnp.sqrt((yy - cy)**2 + (xx - cx)**2)
        mask  = ((dist > r - 4) & (dist < r + 4)).astype(jnp.float32)
        key, sk = random.split(key)
        noise = random.uniform(sk, (H, W, N_CHANNELS)) * mask[:, :, None]
        noise = noise.at[:, :, 3].set(mask * 0.7)
        grid  = grid + noise

    elif mode == 3:
        # Scattered — 8 random seeds, multiple collision fronts
        shape_name = "scattered"
        for _ in range(8):
            key, sk1, sk2 = random.split(key, 3)
            y = int(random.randint(sk1, (), 10, H - 10))
            x = int(random.randint(sk2, (), 10, W - 10))
            key, sk = random.split(key)
            seed = random.uniform(sk, (6, 6, N_CHANNELS))
            seed = seed.at[:, :, 3].set(0.6)
            grid = grid.at[y:y+6, x:x+6, :].set(seed)

    else:
        # Global noise — whole grid seeded with low sparse values.
        # The entire world is alive from frame 1. No frontier, no center.
        # Most "always alive" feeling of all the modes.
        shape_name = "global_noise"
        key, sk1, sk2, sk3 = random.split(key, 4)
        noise = random.uniform(sk1, (H, W, N_CHANNELS)) * 0.12
        # Sparse alpha: only ~15% of cells start alive so the network
        # has room to breathe and evolve rather than saturating immediately
        alive_mask = (random.uniform(sk2, (H, W)) < 0.15).astype(jnp.float32)
        alpha       = random.uniform(sk3, (H, W)) * 0.4 * alive_mask
        noise = noise.at[:, :, 3].set(alpha)
        grid  = noise

    return grid, shape_name


# ── Perception kernel ─────────────────────────────────────────────────────────
def make_perception_kernel():
    """
    Fixed (non-learned) 3x3 filters — the cell's sensory organs.
    Each of 4 filters is applied to all 16 channels independently (depthwise).
    Output per cell: 64 numbers (4 filters × 16 channels).

    Identity  — what am I right now?
    Sobel X   — left-right gradient (which way is the concentration higher?)
    Sobel Y   — up-down gradient
    Laplacian — am I a local peak or valley vs my neighbors?
                drives reaction-diffusion style patterns: rings, spirals, spots
    """
    identity  = jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=jnp.float32)
    sobel_x   = jnp.array([[-1,0,1],  [-2,0,2],  [-1,0,1]], dtype=jnp.float32) / 8.0
    sobel_y   = jnp.array([[-1,-2,-1],[0, 0, 0],  [1, 2, 1]], dtype=jnp.float32) / 8.0
    laplacian = jnp.array([[0,1,0],   [1,-4,1],   [0,1,0]], dtype=jnp.float32) / 4.0

    # Stack to (3,3,4) then tile across N_CHANNELS and reshape for depthwise conv
    kernel = jnp.stack([identity, sobel_x, sobel_y, laplacian], axis=-1)
    kernel = jnp.tile(kernel, (1, 1, N_CHANNELS))
    kernel = kernel.transpose(2, 0, 1)
    kernel = kernel[:, jnp.newaxis, :, :]
    return kernel


# ── Perception pass ───────────────────────────────────────────────────────────
def perceive(grid, kernel):
    """
    Apply all filters to all channels. Uses WRAP padding — torus topology,
    edges connect to opposite edges. No edge bleeding.
    Input: (H, W, 16) → Output: (H, W, 64)
    """
    x = grid.transpose(2, 0, 1)[None]
    x = jnp.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='wrap')
    out = lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1),
        padding='VALID',
        feature_group_count=N_CHANNELS,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    return out[0].transpose(1, 2, 0)  # (H, W, 64)


# ── Update network ────────────────────────────────────────────────────────────
class UpdateNet(nn.Module):
    """
    The neural net that IS the NCA rule.
    Input: 64-number perception → Output: 16-number delta (how to change state).

    WHY TANH INSTEAD OF RELU:
    ReLU only passes positive values → sharp fragmented saturation.
    tanh outputs -1 to +1 → allows oscillation and negative feedback.
    Result: wave-like fluid dynamics that keep moving instead of freezing.
    """
    hidden_size: int = HIDDEN_SIZE

    @nn.compact
    def __call__(self, perception):
        x = nn.Dense(self.hidden_size)(perception)
        x = jnp.tanh(x)                    # ← the one change that matters most
        x = nn.Dense(N_CHANNELS)(x)
        return x


# ── Single NCA step ───────────────────────────────────────────────────────────
def nca_step(grid, params, update_net, perception_kernel, key, step_size=1.0):
    """
    perceive → delta → alive mask → fire mask → apply → clip
    """
    H, W, _ = grid.shape
    perceived = perceive(grid, perception_kernel)
    delta     = update_net.apply(params, perceived)

    # Alive mask: only update cells where at least one neighbor is alive
    alpha        = grid[:, :, 3:4]
    alive        = (alpha > ALIVE_THRESH).astype(jnp.float32)
    alive_padded = jnp.pad(alive[:, :, 0], 1, mode='wrap')
    neighbor_max = jnp.zeros((H, W))
    for dy in range(3):
        for dx in range(3):
            neighbor_max = jnp.maximum(neighbor_max, alive_padded[dy:dy+H, dx:dx+W])
    alive_mask = (neighbor_max > 0.0)[:, :, jnp.newaxis]

    # Fire mask: stochastic async — 75% of cells update each step
    key, subkey = random.split(key)
    fire_mask   = (random.uniform(subkey, (H, W, 1)) < FIRE_RATE).astype(jnp.float32)

    new_grid = grid + step_size * delta * (alive_mask * fire_mask)
    new_grid = jnp.clip(new_grid, 0.0, 1.0)
    return new_grid, key


# ── JIT-compiled step factory ─────────────────────────────────────────────────
def make_step_fn(update_net, perception_kernel):
    """
    Call ONCE at startup. Reuse forever.
    Every jax.jit() call compiles a new XLA program. One call = no memory leak.
    """
    @jax.jit
    def step(grid, params, key, step_size=1.0):
        return nca_step(grid, params, update_net, perception_kernel, key, step_size)
    return step
