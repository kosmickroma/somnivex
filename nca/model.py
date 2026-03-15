# nca/model.py - Neural Cellular Automata cell update rule.
# 
# Every cell in the grid runs this same tiny neural network every step. 
# Input: what the cell can "see" (its 3x3 neighborhood through 4 filters)
# Output: a delta - how much to nudge each of the cell's 16 channels.
#
# This file defines:
# - The channel layout (which slot means what)
# - The 4 fixed perception filters (the cell's "senses")
# - The UpdateNet neural network (the cell's "brain")
# - One NCA step function
# - A JIT-compiled step factory for efficiency

import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn

# --- Channel layout: ------------------
# Each cell carries 16 floating-point values ("channels").
# We've assigned meaning to specific slots:

N_CHANNELS = 16 # total state values per cell

CH_A = 0 # Gray-Scott chemical A 
CH_B = 1 # Gray-Scott chemical B
# Channels 2-13 are "hidden" channels that the NCA can use for internal computations.
# Nobody tells it. Training decides. This is where the magic happens.
CH_F = 14 # feed rate (f) - written in from outside before each step
CH_K = 15 # kill rate (k) - written in from outside before each step

# During training:  we set grid[:, :, CH_F] = f  and  grid[:, :, CH_K] = k
# The NCA reads these each step and learns to behave differently based on them.
# This is how one trained model covers all 15 GS regimes.

# --- Perception constants: ---------------
N_FILTERS = 4 # identity + sobel_x + sobel_y + laplacian
HIDDEN_SIZE = 128 # number of hidden channels in the UpdateNet
FIRE_RATE = 0.5 # fraction of cells that actually update each step
                # 0.5 = stochastic async: forces robustness to neighbor timing

# --- Fixed perception kernels: -----------
def make_perception_kernel():
    """
    Build the 4 fixed 3x3 filters that define what each cell can "see."
    These are NOT learned - they're hand-crafted sensory organs.
    Applied to all 16 channels independently (depthwise convolution).
    Output per cell: 4 filters x 16 channels = 64 numbers.

    Identity: what am I right now? (self-state)
    Sobel X: left-right concentration gradient
    Sobel Y: up-down concentration gradient
    Laplacian: am I a local peak or valley vs my neighbors?
                This directly approximates ∇² - the diffusion term in Gray-Scott.
                Including it gives the NCA a direct sense of diffusion gradients.
    """
    identity = jnp.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]], dtype=jnp.float32)
    
    sobel_x = jnp.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=jnp.float32) / 8.0
    
    sobel_y = jnp.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=jnp.float32) / 8.0
    
    # Standard discrete Laplacian - same weights used in gs/engine.py
    # (0.05 corners, 0.20 edges, -1 center) approximates the continuous ∇².
    laplacian = jnp.array([[0.05, 0.20, 0.05],
                           [0.20, -1.0, 0.20],
                           [0.05, 0.20, 0.05]], dtype=jnp.float32)
                           
    # Stack to (3, 3, 4) - four 3x3 kernels
    kernel = jnp.stack([identity, sobel_x, sobel_y, laplacian], axis=-1)

    # Title across all N_CHANNELS so each filter is applied to every channel
    # Result: (3, 3, N_CHANNELS * N_FILTERS) = (3, 3, 64)
    kernel = jnp.tile(kernel, (1, 1, N_CHANNELS))

    # Reshape for JAX depthwise conv: (out_channels, 1, kH, kW)
    # Each output channel has its own kernel, no cross-channel mixing here
    kernel = kernel.transpose(2, 0, 1)
    kernel = kernel[:, jnp.newaxis, :, :]
    return kernel # shape: (64, 1, 3, 3)

# --- Perception pass ----------------
def perceive(grid, kernel):
    """
    Apply all 4 filters to all 16 channels. WRAP padding = tous topology.
    No edge artifacts - the grid wraps around like Pac-Man.

    Input: grid (H, W, 16)
    Output: perceived (H, W, 64) - 64 numbers per cell describing its neighborhood
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

    # Return as (H, W, 64) - spatial first, features last
    return out[0].transpose(1, 2, 0)

# --- Update network ----------------
class UpdateNet(nn.Module):
    """
    The neural network that IS the NCA rule. One tiny net, shared by every cell.
    Applied as 1x1 convolution across the grid after the perception step. 

    Input: 64-dim perception vector (what the cell can see)
    Output: 16-dim delta (how much to change each channel)

    Architecture:
    Dense(128) + tanh -> Dense(16, zero-init)
    
    WHY TANH:
    tanh outputs [-1, +1], allowing both increases and decreases.
    Cells need to be able to lower concentrations (inhibition) not just raise them.
    ReLU (outputs 0+) would kill half the dynamics and cause saturation. 
    WHY ZERO-INIT ON FINAL LAYER (critical):
    At the start of training, every cell produces zero delta - no change.
    This "do nothing prior" means training begings for a stable baseline
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

# --- Single NCA step ----------------
def nca_step(grid, params, update_net, perception_kernel, key):
    """
    One full NCA update across the entire grid.

    Steps: 
    1. perceive - each cell reads its neighborhood through 4 filters -> 64 numbers
    2. update - UpdateNet maps 64 -> 16 delta values
    3. fire mask - randomly zero out 50% of updates (stochastic async)
    4. apply - new_grid + delta * fire_mask
    5. clip - keep all values in [0,1]

    No alive mask, Every cell is always active.
    Grey-Scott has no concept of "dead" cells - the whole grid is chemically live.

    Returns: (new_grid, new_key)
    """
    H, W, _ = grid.shape

    # Step 1: perception - what does each cell see?
    perceived = perceive(grid, perception_kernel) # (H, W, 64)

    # Step 2: compute delta for every cell simultaneously
    # update_net.apply runs the network as 1x1 conv across the spatial grid
    delta = update_net.apply(params, perceived) # (H, W, 16)

    # Step 3: stochastic fire mask
    # Only 50% of cells actually apply their update this step. 
    # This forces the learned rule to work even when neighbors haven't updated yet.
    # Think of it like per-cell dropout applied to the update vector.
    key, subkey = random.split(key)
    fire_mask = (random.uniform(subkey, (H, W, 1)) < FIRE_RATE).astype(jnp.float32)

    # Step 4 + 5: apply delta, clip to valid range
    new_grid = jnp.clip(grid + delta * fire_mask, 0.0, 1.0)

    return new_grid, key


# --- JIT-compiled step factory --------------------------
def make_step_fn(update_net, perception_kernel):
        """
        Call this ONCE at startup. It returns a JIT-compiled step function.
        Reuse the returned function forever - never call make_step_fn again in the loop.
        
        WHY: jax.jit compiles a new XLA program each time it's called on a new function.
        Calling make_stepfn repeatedly would cause recompilation on every step.
        One call here = one compilation = reuse forever = no memory leak, full speed.
        """
        @jax.jit
        def step(grid, params, key):
            return nca_step(grid, params, update_net, perception_kernel, key)
        return step
