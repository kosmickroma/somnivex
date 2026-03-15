#nca/train.py - Pool based training script.
#
# Teaches the NCA to understand Gray-Scott reaction-diffusion physics
# using multi-step rollout: NCA runs on its own output for ROLLOUT_STEPS,
# loss is accumulated against the GS ground truth trajectory at each step.
#
# Run from the project root:
#           python nca/train.py
#
# Resumes from RESUME_FROM checkpoint if set (see Config below).
# Checkpoints save to nca/checkpoints/ every 1,000 steps.

import os
import sys
import time
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

# Make sure Python can find our project modelues when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs.engine import GS_REGIMES, gs_step, init_gs_grid
from nca.model import(
    N_CHANNELS, N_FILTERS, 
    CH_A, CH_B, CH_F, CH_K,
    UpdateNet, make_perception_kernel, nca_step,
)

# ── Training grid size ────────────────────────────────────────────────────────
# We train at 64x64, not 256x256.
# WHY: the pool would need 512 * 256 * 256 * 16 * 4 bytes = 2GB of RAM. Too much.
#      At 64x64: 512 * 64 * 64 * 16 * 4 bytes = 134MB. Fine.
# The NCA is fully convolutional — weights trained at 64x64 work at any resolution.
# All perception is local (3x3 kernels), so resolution doesn't affect what's learned.
TRAIN_H = 64
TRAIN_W = 64


# ── Hyperparameters ───────────────────────────────────────────────────────────
POOL_SIZE        = 512    # how many live GS states to maintain
BATCH_SIZE       = 32     # states processed per training step
TRAIN_STEPS      = 30000  # additional steps to run in this session
ROLLOUT_STEPS    = 8      # NCA steps per loss computation (multi-step rollout)
LEARNING_RATE    = 2e-4   # Adam step size — don't go higher, training becomes unstable
PERSIST_WEIGHT   = 0.1    # how much to penalize non-stability under small noise
PERSIST_NOISE    = 0.02   # amplitude of noise injected for persistence test
CHECKPOINT_EVERY = 1000   # save weights to disk every N steps
LOG_EVERY        = 100    # print loss every N steps

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

# ── Resume config ─────────────────────────────────────────────────────────────
# Set RESUME_FROM to a checkpoint path to continue training from that point.
# Set to None to start fresh from random weights.
RESUME_FROM = os.path.join(CHECKPOINT_DIR, 'params_020000.pkl')
RESUME_STEP = 20000   # the step number encoded in the filename above

# ── Pool state generation ─────────────────────────────────────────────────────
def make_pool_state(key, H, W):
    """
    Generate one GS simulation state and pack it into a 16-channel NCA grid.
    
    The pool is made of these. Each entry is a different moment in a different 
    GS simulation - different regime, different seed, different age.
    The NCA trains on all of them simultaneously, which is why it learns to handle the full range of GS behaviors, not just one regime.
    
    How the 16 channels are packed:
        ch 0 = A (food chemical from GS)
        ch 1 = B (predator chemical - this makes the visible patterns)
        ch 2-13 = zeros at init (NCA fills these in as training evolves)
        ch 14 = f (feed rate, same value brodcast to every cell)
        ch 15 = k (kill rate, same value broadcast to every cell)
        
    Returns: (grid as numpy array, f value, k value)
    """
    # Pick a random GS regime
    regime_names = list(GS_REGIMES.keys())
    key, sk = random.split(key)
    idx = int(random.randint(sk, (), 0, len(regime_names)))
    regime = regime_names[idx]
    f, k = GS_REGIMES[regime]

    # Init a 64x64 GS grid and warm it up
    # "Warm up" = run GS for a while so the pool starts full of real patterns,
    # not just the boring initial conditions. We want the NCA to see spirals,
    # worms, coral - not just uniform noise.
    key, sk = random.split(key)
    A, B = init_gs_grid(sk, H, W)

    key, sk = random.split(key)
    warmup_steps = int(random.randint(sk, (), 100, 500))
    for _ in range(warmup_steps):
        A, B = gs_step(A, B, f, k)

    # Pack into 16-channel NCA grid
    # jnp.zeros creates on GPU. We convert to numpy at the end for pool storage.
    grid = jnp.zeros((H, W, N_CHANNELS))
    grid = grid.at[:, :, CH_A].set(A)
    grid = grid.at[:, :, CH_B].set(B)
    grid = grid.at[:, :, CH_F].set(f)
    grid = grid.at[:, :, CH_K].set(k)

    return np.array(grid) # move to CPU numpy for pool storage

def init_pool(key, pool_size, H, W):
    """
    Build the initial pool of pool_size random GS states.
    Stored as a numpy array on CPU: shape (pool_size, H, W, N_CHANNELS).

    Keeping the pool on CPU lets us index and modify it easily.
    We move batches to GPU (jnp.array) when we need to compute on them.
    """
    print(f"Initializing pool ({pool_size} states at {H}x{W})...")
    pool = np.zeros((pool_size, H, W, N_CHANNELS), dtype=np.float32)

    for i in range(pool_size):
        key, sk = random.split(key)
        pool[i] = make_pool_state(sk, H, W)
        if (i + 1) % 64 == 0:
            print(f" {i + 1}/{pool_size}")

    print("Pool ready.\n")
    return pool

# ── Batched GS target computation (JIT-compiled) ──────────────────────────────
@jax.jit
def gs_targets_batch(batch_A, batch_B, batch_f, batch_k):
    """
    Run one GS step on a batch of (A, B) grids in parallel on GPU.
    Returns (A_next, B_next) - what GS actually produces in one step.
    This is the "ground truth" the NCA is trained to match.

    jax.vmap(fn, in_axes=(0,0,0,0)) means:
        "map fn over the first axis of each of these 4 arguments"
    So if batch_A is shape (32, 64, 64), vmap runs gs_step on each (64, 64) slice.
    All 32 slices run simultaneously on the GPU.
    """
    batched = jax.vmap(gs_step, in_axes=(0, 0, 0, 0))
    return batched(batch_A, batch_B, batch_f, batch_k)

# ── Loss function (JIT-compiled) ──────────────────────────────────────────────
def make_loss_fn(update_net, perception_kernel):
    """
    Build and return the JIT-compiled loss function.

    Uses MULTI-STEP ROLLOUT: NCA runs on its own output for ROLLOUT_STEPS.
    GS runs the same number of steps from the same start state.
    Loss is accumulated at each step — NCA must track GS across the whole trajectory.

    WHY ROLLOUT:
    1-step training only teaches "from a real GS state, predict one GS step."
    But free run feeds NCA outputs back into NCA — never real GS states.
    Errors from step 1 become the input to step 2, compounding until the NCA
    diverges from anything GS-like. That's the collapse you saw.

    Rollout training teaches "stay close to GS across a whole trajectory."
    The NCA must learn to be stable on its OWN outputs, not just GS outputs.
    """

    @jax.jit
    def loss_fn(params, batch_grids, batch_keys):
        """
        batch_grids: (BATCH, H, W, 16) - batch of pool states on GPU
        batch_keys:  (BATCH, 2)          - one PRNG key per grid in batch
        """
        def step_one(grid, key):
            return nca_step(grid, params, update_net, perception_kernel, key)

        # ── Extract GS parameters (fixed for this batch) ──────────────────
        # f and k are the same for every cell in a grid — grab from [0, 0]
        f_in = batch_grids[:, 0, 0, CH_F]   # (BATCH,)
        k_in = batch_grids[:, 0, 0, CH_K]   # (BATCH,)

        # ── Multi-step rollout ────────────────────────────────────────────
        # NCA trajectory: starts at batch_grids, runs on its own output each step
        # GS trajectory:  starts at same A/B, runs GS physics each step
        # We accumulate prediction loss at every step.
        nca_grids   = batch_grids
        gs_A        = batch_grids[:, :, :, CH_A]  # (BATCH, H, W)
        gs_B        = batch_grids[:, :, :, CH_B]  # (BATCH, H, W)
        total_pred_loss = jnp.zeros(())

        for _ in range(ROLLOUT_STEPS):
            # NCA step — runs on its own previous output
            nca_grids, batch_keys = jax.vmap(step_one)(nca_grids, batch_keys)

            # Re-inject original f/k after each NCA step.
            # The NCA's delta could drift channels 14/15. We don't want that —
            # f and k are control inputs written from outside, not learned state.
            nca_grids = nca_grids.at[:, :, :, CH_F].set(batch_grids[:, :, :, CH_F])
            nca_grids = nca_grids.at[:, :, :, CH_K].set(batch_grids[:, :, :, CH_K])

            # GS step — ground truth trajectory
            gs_A, gs_B = gs_targets_batch(gs_A, gs_B, f_in, k_in)

            # Accumulate MSE for channels A and B at this step
            total_pred_loss = total_pred_loss + jnp.mean(
                (nca_grids[:, :, :, CH_A] - gs_A) ** 2 +
                (nca_grids[:, :, :, CH_B] - gs_B) ** 2
            )

        pred_loss = total_pred_loss / ROLLOUT_STEPS

        # ── Persistence loss (on final rollout state) ─────────────────────
        # After ROLLOUT_STEPS, nudge the NCA's state slightly.
        # One more step on noisy vs clean should produce similar results.
        # Tests: does the NCA treat nearby states the same? (attractor stability)
        key_noise = batch_keys[0]
        noise = random.normal(key_noise, nca_grids.shape) * PERSIST_NOISE
        noisy = jnp.clip(nca_grids + noise, 0.0, 1.0)

        noisy_next, _ = jax.vmap(step_one)(noisy,     batch_keys)
        clean_next, _ = jax.vmap(step_one)(nca_grids, batch_keys)

        persist_loss = jnp.mean((noisy_next - clean_next) ** 2)

        # ── Total ─────────────────────────────────────────────────────────
        total = pred_loss + PERSIST_WEIGHT * persist_loss

        return total, (pred_loss, persist_loss)

    return loss_fn

# ── Per-variable gradient normalization ───────────────────────────────────────
def normalize_gradients(grads):
    """
    Normalize each parameter's gradient by its own L2 norm.
    This is per-variable, not global gradient clipping.

    Global clipping: if total gradient norm > threshold, scale ALL grads down.
    Per-variable:   for each weight matrix, grad = grad / ||grad||

    Why this instead of clipping:
    Late in training, a single weight might suddenly get a large gradient spike
    while everything else is small. Global clipping would scale the s mall
    useful gradients down along with the spike. Per-variable normalization
    handles each weight independently - the spike gets normalized, the small
    gradients are untouched.
    
    jax.tree_util.tree_map applies a function to every leaf of a pytree.
    Params are a nested dict (pytree) of JAX arrays. This hits every array.
    """

    return jax.tree_util.tree_map(
        lambda g: g / (jnp.sqrt(jnp.sum(g ** 2)) + 1e-8),
        grads
    )

# ── NCA batch step (for pool writes, no gradient needed) ─────────────────────
def make_nca_batch_step(update_net, perception_kernel):
    """
    Returns a compiled function that runs the NCA forward on a batch.
    Used after computing gradients to generate new pool states.
    We don't need gradients here - just the forward pass output.
    """
    @jax.jit
    def batch_step(params, batch_grids, batch_keys):
        def step_one(grid, key):
            return nca_step(grid, params, update_net, perception_kernel, key)
        return jax.vmap(step_one)(batch_grids, batch_keys)
    return batch_step

# ── Checkpointing ─────────────────────────────────────────────────────────────
def save_checkpoint(params, step):
    """
    Save network weights to disk.

    jax.device_get() moves JAX CPU numpy arrays.
    pickle serializes the nested dict of numpy arrays to a binary file.
    This is simple and reliable - no special checkpoint library needed.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'params_{step:06d}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    print(f" Saved: {path}")

def load_checkpoint(path):
    """
    Load weights from a checkpoint file. Returns params on GPU. 
    jax.device_put() moves CPU numpy arrays back to GPU.
    """
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return jax.device_put(params)

# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print(" Somnivex - NCA Physics Training")
    print(" Teaching the NCA Gray-Scott reaction-diffusion")
    print("=" * 60)
    print(f"\n JAX devices: {jax.devices()}")
    print(f" Training grid: {TRAIN_H}x{TRAIN_W}")
    print(f" Pool size: {POOL_SIZE} | Batch size: {BATCH_SIZE}")
    print(f" Steps: {TRAIN_STEPS} | Rollout: {ROLLOUT_STEPS} | LR: {LEARNING_RATE}")
    if RESUME_FROM:
        print(f" Resuming from: {RESUME_FROM}  (global step {RESUME_STEP})")
    print()

    # ── Initialize model ──────────────────────────────────────────────────
    update_net = UpdateNet()
    perception_kernel = make_perception_kernel()

    # Initialize params by running a dummy input through the network.
    # Flax is "lazy" - it doesn't know the weight shapes until it sees input shapes.
    # This dummy call tells it the input is 64-wide perception vectors.
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    dummy_input = jnp.zeros((TRAIN_H, TRAIN_W, N_CHANNELS * N_FILTERS))
    params = update_net.init(subkey, dummy_input)

    # ── Load checkpoint if resuming ────────────────────────────────────────
    # Load the trained weights. Optimizer state is NOT saved — Adam will
    # re-warm its momentum buffers over the first ~100 steps. This is fine.
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        params = load_checkpoint(RESUME_FROM)
        print(f" Loaded checkpoint: {RESUME_FROM}")
    elif RESUME_FROM:
        print(f" WARNING: RESUME_FROM set but file not found: {RESUME_FROM}")
        print(f" Starting from random weights.")

    start_step = RESUME_STEP if (RESUME_FROM and os.path.exists(RESUME_FROM)) else 0

    # Count total parameters (just for your info)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f" Network params: {n_params:,}")
    print()

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Adam: the standard adaptive gradient optimizer. Works well for NCAs.
    # We apply per-variable gradient normalization manually before each step.
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # ── Compiled functions ─────────────────────────────────────────────────
    # Build these once. They get JIT-compiled on first call and reused forever.
    loss_fn             = make_loss_fn(update_net, perception_kernel)
    loss_and_grad       = jax.value_and_grad(loss_fn, has_aux=True)
    nca_batch_step      = make_nca_batch_step(update_net, perception_kernel)

    # ── Initialize pool ───────────────────────────────────────────────────
    key, pool_key = random.split(key)
    pool = init_pool(pool_key, POOL_SIZE, TRAIN_H, TRAIN_W)

    # ── Training loop ──────────────────────────────────────────────────────
    end_step = start_step + TRAIN_STEPS
    print(f"Training steps {start_step} → {end_step}...")
    print(f"Watch pred_loss fall — that's the main signal.")
    print(f"persist_loss should stay small (< pred_loss).")
    print()

    t_start = time.time()

    for step in range(start_step, end_step):

        # Sample a random batch of indices (no replacement within a batch)
        batch_idx = np.random.choice(POOL_SIZE, BATCH_SIZE, replace=False)

        # Move batch from CPU pool to GPU for computation
        batch = jnp.array(pool[batch_idx])      # (BATCH, H, W, 16) on GPU

        # One PRNG key per grid in the batch (JAX needs explicit randomness)
        key, *batch_keys_list = random.split(key, BATCH_SIZE + 1)
        batch_keys = jnp.stack(batch_keys_list)     # (BATCH, 2)

        # ── Forward pass + backward pass ──────────────────────────────────
        # value_and_grad computes the loss AND the gradient in one efficient pass.
        # has_aux=True because loss_fn returns (total, (pred, persist)) - the
        # aux tuple gets passed through without being differentiated.
        (loss, (pred_loss, persist_loss)), grads = loss_and_grad(
            params, batch, batch_keys
        )

        # Normalize gradients (per-variable L2 norm)
        grads = normalize_gradients(grads)

        # Apply optimizer: Adam computes adaptive step sizes from gradient history
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # ── Write evolved states back to pool ─────────────────────────────
        # Run NCA forward (no gradient) to get the next-generation pool states.
        # Writing these back means: next time these indices are sampled,
        # the NCA trains on its OWN previous outputs, not fresh GS. 
        # This is the pool mechanism - it's what creates long-term stability.
        key, *step_keys_list = random.split(key, BATCH_SIZE + 1)
        step_keys = jnp.stack(step_keys_list)
        new_batch, _ = nca_batch_step(params, batch, step_keys)
        pool[batch_idx] = np.array(new_batch)       # move back to CPU

        # ── Reseed worst state ────────────────────────────────────────────
        # Every 10 steps: replace one pool state with a fresh GS state.
        # This prevents the pool from filling up entirely with NCA outputs
        # (which could drift away from GS if training isn't perfect yet).
        # The pool always stays grounded in real GS data.
        if step % 10 == 0:
            key, sk = random.split(key)
            pool[batch_idx[0]] = make_pool_state(sk, TRAIN_H, TRAIN_W)

        # ── Logging ───────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            steps_done      = step - start_step + 1
            elapsed         = time.time() - t_start
            rate            = steps_done / elapsed if elapsed > 0 else 1
            eta_minutes     = (end_step - step) / rate / 60
            print(
                f" step {step:6d}/{end_step}"
                f" loss={float(loss):.6f}"
                f" (pred={float(pred_loss):.6f}"
                f" persist={float(persist_loss):.6f})"
                f" {rate:.1f} steps/s"
                f" ETA {eta_minutes:.0f}m"
            )

        # ── Checkpointing ───────────────────────────────────────────────
        if step > start_step and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, step)

    # Final save
    save_checkpoint(params, end_step)
    total_min = (time.time() - t_start) / 60
    print(f"\nDone. Total time: {total_min:.1f} minutes")
    print(f"Best checkpoint is probably the last one: params_{end_step:06d}.pkl")

    return params

if __name__ == "__main__":
    train()