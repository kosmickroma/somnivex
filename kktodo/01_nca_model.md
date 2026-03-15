# 01 — nca/model.py (Full Rewrite)

This is the first file to retype. It replaces the entire contents of `nca/model.py`.

---

## What this file is

This is the NCA cell brain. Every cell in the 256×256 grid runs this same tiny neural network
every single step. It looks at its 3×3 neighborhood, runs the math, and outputs a small delta
(a nudge) to its own 16 state values. Do that a million times across a grid and you get emergence.

The key thing to understand: there is no "global rule" being applied. Every cell is local.
No cell knows what's happening across the grid. The patterns you see are entirely a product
of local interactions propagating outward. This is what makes NCAs wild.

---

## What changed from the old version and WHY

### 1. Channel roles are now explicit and meaningful

Old version: all 16 channels were anonymous numbers. No fixed meaning.

New version: specific channels have specific jobs:
- Channel 0 = A (Gray-Scott "food" chemical)
- Channel 1 = B (Gray-Scott "predator" chemical)
- Channels 2–13 = hidden (the NCA uses these however it wants)
- Channel 14 = f (feed rate — the "control knob" for GS behavior)
- Channel 15 = k (kill rate — the other control knob)

WHY: We're teaching the NCA Gray-Scott physics. During training, we'll set channels 14 and 15
to the current f and k values before every step. The NCA learns that "when channel 14 is 0.012
and channel 15 is 0.045, do spiral things." That's how control channels work — they're not
trained, they're written in from the outside, and the NCA figures out what to do with them.

### 2. Zero-init on the final Dense layer (CRITICAL — from the Growing NCA paper)

Old version: final layer initialized with random weights. NCA immediately does chaotic garbage.

New version: final Dense layer starts with ALL ZEROS. The NCA does absolutely nothing at start.

WHY: This is called a "do-nothing prior." When you start training, you want the model to
begin from a stable baseline and gradually learn to make changes — not start from chaos and
try to wrangle it into something useful. Zero-init means your loss starts at "NCA did nothing,
GS moved one step" and training teaches it to match that GS step. Much more stable.
Without this trick: early training is chaotic, gradients explode, often fails to learn at all.

### 3. Alive mask removed

Old version: cells could only update if a neighbor had alpha > 0.1. "Dead" cells couldn't fire.

New version: every cell updates every step (subject to the fire mask).

WHY: The alive mask was designed for "grow a pattern from a single seed pixel" — the paper's
original use case. It made sense there because you needed cells to be "dead" before the
growing frontier reached them. Gray-Scott doesn't work that way — the entire grid is always
chemically active. There are no dead cells. Keeping the alive mask would suppress updates in
regions where the NCA hasn't "grown" to yet, which would break GS physics learning entirely.

### 4. Fire rate changed: 0.75 → 0.5

WHY: The Growing NCA paper used 50% (not 75%). The idea is stochastic async updates —
cells don't all fire in lockstep like a clock. This forces the learned rule to be robust:
it has to work correctly even when some neighbors haven't updated yet this step.
50% is more aggressive and produces more robustness. Think of it like dropout for cells.

### 5. Perception kernel: Laplacian added (we kept this from before, paper didn't have it)

The paper used: Identity + Sobel X + Sobel Y = 3 filters × 16 channels = 48-dim perception.
We use:         Identity + Sobel X + Sobel Y + Laplacian = 4 filters × 16 channels = 64-dim.

WHY we kept the Laplacian: Gray-Scott's diffusion term IS the Laplacian (∇²A and ∇²B).
Giving the NCA explicit Laplacian perception means it can directly sense diffusion gradients —
exactly what it needs to learn GS physics. This is a helpful inductive bias. The paper's
target was growing static images; ours is learning reaction-diffusion equations. The Laplacian
kernel makes our job easier.

---

## The code — type this out completely into nca/model.py

```python
# nca/model.py — Neural Cellular Automata cell update rule.
#
# Every cell in the grid runs this same tiny neural network every step.
# Input:  what the cell can "see" (its 3x3 neighborhood through 4 filters)
# Output: a delta — how much to nudge each of the cell's 16 state values
#
# This file defines:
#   - The channel layout (which slot means what)
#   - The 4 fixed perception filters (the cell's "senses")
#   - The UpdateNet neural network (the cell's "brain")
#   - One NCA step function
#   - A JIT-compiled step factory for efficiency

import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn


# ── Channel layout ────────────────────────────────────────────────────────────
# Each cell carries 16 floating-point values ("channels").
# We've assigned meaning to specific slots:

N_CHANNELS = 16   # total state values per cell

CH_A = 0          # Gray-Scott chemical A ("food") — the thing B eats
CH_B = 1          # Gray-Scott chemical B ("predator") — the thing that makes patterns
# Channels 2–13: hidden state. The NCA figures out what to use these for.
# Nobody tells it. Training decides. This is where the magic lives.
CH_F = 14         # feed rate (f) — written in from outside before each step
CH_K = 15         # kill rate (k) — written in from outside before each step

# During training:  we set grid[:, :, CH_F] = f  and  grid[:, :, CH_K] = k
# The NCA reads these each step and learns to behave differently based on them.
# This is how one trained model covers all 15 GS regimes.


# ── Perception constants ──────────────────────────────────────────────────────
N_FILTERS   = 4     # identity + sobel_x + sobel_y + laplacian
HIDDEN_SIZE = 128   # neurons in the hidden layer of UpdateNet
FIRE_RATE   = 0.5   # fraction of cells that actually update each step
                    # 0.5 = stochastic async: forces robustness to neighbor timing


# ── Fixed perception kernels ──────────────────────────────────────────────────
def make_perception_kernel():
    """
    Build the 4 fixed 3x3 filters that define what each cell can "see."
    These are NOT learned — they're hand-crafted sensory organs.
    Applied to all 16 channels independently (depthwise convolution).
    Output per cell: 4 filters × 16 channels = 64 numbers.

    Identity:   what am I right now? (self-state)
    Sobel X:    left-right concentration gradient
    Sobel Y:    up-down concentration gradient
    Laplacian:  am I a local peak or valley vs my neighbors?
                This directly approximates ∇²  — the diffusion term in Gray-Scott.
                Including it gives the NCA a direct sense of diffusion gradients.
    """
    identity  = jnp.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=jnp.float32)

    sobel_x   = jnp.array([[-1,  0,  1],
                            [-2,  0,  2],
                            [-1,  0,  1]], dtype=jnp.float32) / 8.0

    sobel_y   = jnp.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=jnp.float32) / 8.0

    # Standard discrete Laplacian — same weights used in gs/engine.py
    # (0.05 corners, 0.20 edges, -1.0 center) approximates the continuous ∇²
    laplacian = jnp.array([[ 0.05,  0.20,  0.05],
                            [ 0.20, -1.00,  0.20],
                            [ 0.05,  0.20,  0.05]], dtype=jnp.float32)

    # Stack to (3, 3, 4) — four 3x3 kernels
    kernel = jnp.stack([identity, sobel_x, sobel_y, laplacian], axis=-1)

    # Tile across all N_CHANNELS so each filter is applied to every channel
    # Result: (3, 3, N_CHANNELS * N_FILTERS) = (3, 3, 64)
    kernel = jnp.tile(kernel, (1, 1, N_CHANNELS))

    # Reshape for JAX depthwise conv: (out_channels, 1, kH, kW)
    # Each output channel has its own kernel, no cross-channel mixing here
    kernel = kernel.transpose(2, 0, 1)
    kernel = kernel[:, jnp.newaxis, :, :]
    return kernel   # shape: (64, 1, 3, 3)


# ── Perception pass ───────────────────────────────────────────────────────────
def perceive(grid, kernel):
    """
    Apply all 4 filters to all 16 channels. WRAP padding = torus topology.
    No edge artifacts — the grid wraps around like Pac-Man.

    Input:  grid  (H, W, 16)
    Output: perceived  (H, W, 64)  — 64 numbers per cell describing its neighborhood
    """
    # Rearrange to (batch=1, channels=16, H, W) for JAX conv
    x = grid.transpose(2, 0, 1)[None]

    # Wrap-pad by 1 pixel on each side so 3x3 convolution covers edges
    x = jnp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='wrap')

    # Depthwise conv: each of the 64 output channels uses its own 3x3 kernel
    # feature_group_count=N_CHANNELS means no cross-channel mixing in perception
    out = lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1),
        padding='VALID',
        feature_group_count=N_CHANNELS,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )

    # Return as (H, W, 64) — spatial first, features last
    return out[0].transpose(1, 2, 0)


# ── Update network ────────────────────────────────────────────────────────────
class UpdateNet(nn.Module):
    """
    The neural network that IS the NCA rule. One tiny net, shared by every cell.
    Applied as a 1x1 convolution across the grid after the perception step.

    Input:  64-dim perception vector (what the cell can see)
    Output: 16-dim delta (how much to change each channel)

    Architecture:
        Dense(128) + tanh → Dense(16, zero-init)

    WHY TANH:
    tanh outputs [-1, +1], allowing both increases and decreases.
    Cells need to be able to lower concentrations (inhibition) not just raise them.
    ReLU (outputs 0+) would kill half the dynamics and cause saturation.

    WHY ZERO-INIT ON FINAL LAYER (critical!):
    At the start of training, every cell produces Δ = 0 (does nothing).
    This "do-nothing prior" means training begins from a stable baseline
    and incrementally learns to make useful changes.
    Without it: random initial deltas cause chaos, training is unstable.

    WHY RESIDUAL (delta, not replacement):
    new_state = old_state + delta
    This mirrors Gray-Scott's Euler integration: U_new = U + dt * dU
    The NCA is learning to approximate dU and dV, not U and V directly.
    """
    hidden_size: int = HIDDEN_SIZE

    @nn.compact
    def __call__(self, perception):
        x = nn.Dense(self.hidden_size)(perception)
        x = jnp.tanh(x)

        # Zero-initializer on weights: final layer starts outputting all zeros.
        # This is the single most important training stability trick.
        x = nn.Dense(N_CHANNELS,
                     kernel_init=nn.initializers.zeros,
                     bias_init=nn.initializers.zeros)(x)
        return x


# ── Single NCA step ───────────────────────────────────────────────────────────
def nca_step(grid, params, update_net, perception_kernel, key):
    """
    One full NCA update across the entire grid.

    Steps:
    1. perceive  — each cell reads its neighborhood through 4 filters → 64 numbers
    2. update    — UpdateNet maps 64 → 16 delta values
    3. fire mask — randomly zero out 50% of updates (stochastic async)
    4. apply     — new_grid = old_grid + delta * fire_mask
    5. clip      — keep all values in [0, 1]

    No alive mask. Every cell is always active.
    Gray-Scott has no concept of "dead" cells — the whole grid is chemically live.

    Returns: (new_grid, new_key)
    """
    H, W, _ = grid.shape

    # Step 1: perception — what does each cell see?
    perceived = perceive(grid, perception_kernel)   # (H, W, 64)

    # Step 2: compute delta for every cell simultaneously
    # update_net.apply runs the network as a 1x1 conv across the spatial grid
    delta = update_net.apply(params, perceived)     # (H, W, 16)

    # Step 3: stochastic fire mask
    # Only 50% of cells actually apply their update this step.
    # This forces the learned rule to work even when neighbors haven't updated yet.
    # Think of it like per-cell dropout applied to the update vector.
    key, subkey = random.split(key)
    fire_mask = (random.uniform(subkey, (H, W, 1)) < FIRE_RATE).astype(jnp.float32)

    # Step 4 + 5: apply delta, clip to valid range
    new_grid = jnp.clip(grid + delta * fire_mask, 0.0, 1.0)

    return new_grid, key


# ── JIT-compiled step factory ─────────────────────────────────────────────────
def make_step_fn(update_net, perception_kernel):
    """
    Call this ONCE at startup. It returns a JIT-compiled step function.
    Reuse the returned function forever — never call make_step_fn again in the loop.

    WHY: jax.jit compiles a new XLA program each time it's called on a new function.
    Calling make_step_fn repeatedly would cause recompilation on every step.
    One call here = one compilation = reuse forever = no memory leak, full speed.
    """
    @jax.jit
    def step(grid, params, key):
        return nca_step(grid, params, update_net, perception_kernel, key)
    return step
```

---

## After you type this out

The file is self-contained. It doesn't break anything — it's not imported by main.py yet
(main.py uses the GS engine directly). You're building the NCA's updated brain in parallel.

Next file will be `nca/train.py` — the training loop that teaches this brain GS physics.
