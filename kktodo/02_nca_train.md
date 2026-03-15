# 02 — nca/train.py (New File)

Type 01_nca_model.md first. This file depends on it.

This is a brand new file — it doesn't exist yet. Create it at: nca/train.py

---

## What this file does

This is the training loop. Run it once (from the project root) and it teaches the NCA
Gray-Scott physics over ~20,000 steps. Takes 30-60 minutes on your GPU.

    source /home/kk/ai-env/bin/activate
    python nca/train.py

Watch the loss numbers print. They should fall from ~0.01 toward ~0.001.
Checkpoints save automatically to nca/checkpoints/ every 1,000 steps.

---

## Key concepts before you type

### One-step prediction
Each training step: show NCA a GS grid → NCA predicts next frame → compare to actual GS next frame → measure error → adjust weights. Repeat. The NCA learns to approximate the GS update rule.

### The pool (read 00_overview.md if you haven't)
512 live GS states stored in memory. We train on them, write the NCA outputs back, repeat.
This is what teaches the NCA to maintain patterns, not just predict a single frame.

### Persistence loss
After NCA outputs a frame, we add tiny noise and run one more NCA step.
We want that noisy result to match the clean result. This teaches stability.
Without it: patterns that look correct during training slowly drift into garbage at runtime.

### jax.vmap
This function is used a lot in this file. It means: "run this function on a whole batch
of inputs in parallel on the GPU." Instead of looping over 32 grids one at a time,
vmap runs all 32 simultaneously. Way faster.

    # Without vmap (slow loop):
    results = [gs_step(A[i], B[i], f[i], k[i]) for i in range(32)]

    # With vmap (runs on GPU in parallel):
    batched_gs = jax.vmap(gs_step, in_axes=(0, 0, 0, 0))
    results = batched_gs(A, B, f, k)   # A, B, f, k all have a batch dimension

### jax.value_and_grad
Computes the loss value AND the gradients of that loss with respect to params in one call.
This is the core of backpropagation in JAX.

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, keys)

### Gradient normalization
After computing gradients, we normalize each weight's gradient by its own L2 norm.
This prevents one big gradient from drowning out small useful gradients.
The Growing NCA paper says this eliminates sudden loss spikes late in training.

---

## Before running: check optax is installed

    source /home/kk/ai-env/bin/activate
    python -c "import optax; print('optax ok:', optax.__version__)"

If that errors:  pip install optax

---

## The code — type this into nca/train.py

```python
# nca/train.py — Pool-based NCA training script.
#
# Teaches the NCA to understand Gray-Scott reaction-diffusion physics
# by watching thousands of GS one-step transitions and learning to predict them.
#
# Run from the project root:
#     python nca/train.py
#
# Loss should fall from ~0.01 to ~0.001 over 20,000 steps.
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

# Make sure Python can find our project modules when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs.engine import GS_REGIMES, gs_step, init_gs_grid
from nca.model import (
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
TRAIN_STEPS      = 20000  # total training steps (raise to 50k for higher quality)
LEARNING_RATE    = 2e-4   # Adam step size — don't go higher, training becomes unstable
PERSIST_WEIGHT   = 0.1    # how much to penalize non-stability under small noise
PERSIST_NOISE    = 0.02   # amplitude of noise injected for persistence test
CHECKPOINT_EVERY = 1000   # save weights to disk every N steps
LOG_EVERY        = 100    # print loss every N steps

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


# ── Pool state generation ─────────────────────────────────────────────────────
def make_pool_state(key, H, W):
    """
    Generate one GS simulation state and pack it into a 16-channel NCA grid.

    The pool is made of these. Each entry is a different moment in a different
    GS simulation — different regime, different seed, different age.
    The NCA trains on all of them simultaneously, which is why it learns to
    handle the full range of GS behaviors, not just one regime.

    How the 16 channels are packed:
        ch 0  = A (food chemical from GS)
        ch 1  = B (predator chemical — this makes the visible patterns)
        ch 2–13 = zeros at init (NCA fills these in as training evolves)
        ch 14 = f (feed rate, same value broadcast to every cell)
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
    # worms, coral — not just uniform noise.
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
    grid = grid.at[:, :, CH_F].set(f)   # broadcast scalar f to every cell
    grid = grid.at[:, :, CH_K].set(k)   # broadcast scalar k to every cell

    return np.array(grid)   # move to CPU numpy for pool storage


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
            print(f"  {i + 1}/{pool_size}")

    print("Pool ready.\n")
    return pool


# ── Batched GS target computation (JIT-compiled) ──────────────────────────────
@jax.jit
def gs_targets_batch(batch_A, batch_B, batch_f, batch_k):
    """
    Run one GS step on a batch of (A, B) grids in parallel on GPU.
    Returns (A_next, B_next) — what GS actually produces in one step.
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

    We build it as a closure (a function that returns another function) because
    jax.jit needs update_net and perception_kernel to be fixed at compile time.
    If we passed them as arguments, JAX would recompile every call.

    The returned loss_fn takes (params, batch_grids, batch_keys) and returns
    (total_loss, (pred_loss, persist_loss)).
    The 'has_aux=True' in value_and_grad tells JAX that the second return value
    is auxiliary data (not part of what we differentiate).
    """

    @jax.jit
    def loss_fn(params, batch_grids, batch_keys):
        """
        batch_grids: (BATCH, H, W, 16) — batch of pool states on GPU
        batch_keys:  (BATCH, 2)        — one PRNG key per grid in batch
        """

        # ── Run NCA forward on the whole batch ────────────────────────────
        # step_one wraps nca_step so vmap can map it over the batch dimension.
        # params is captured from the outer scope — shared across all cells/grids.
        def step_one(grid, key):
            return nca_step(grid, params, update_net, perception_kernel, key)

        # vmap maps step_one over axis 0 of both grid and key simultaneously.
        # Returns (new_grids, new_keys), both batched.
        nca_outputs, _ = jax.vmap(step_one)(batch_grids, batch_keys)
        # nca_outputs shape: (BATCH, H, W, 16)

        # ── Compute GS targets ────────────────────────────────────────────
        # Extract A, B, f, k from the batch grids for GS computation.
        # f and k are the same for every cell in a grid, so we just grab [0, 0].
        A_in  = batch_grids[:, :, :, CH_A]     # (BATCH, H, W)
        B_in  = batch_grids[:, :, :, CH_B]     # (BATCH, H, W)
        f_in  = batch_grids[:, 0, 0, CH_F]     # (BATCH,) — one value per grid
        k_in  = batch_grids[:, 0, 0, CH_K]     # (BATCH,)

        A_next, B_next = gs_targets_batch(A_in, B_in, f_in, k_in)
        # A_next, B_next: (BATCH, H, W) — what GS actually produced

        # ── Prediction loss ───────────────────────────────────────────────
        # NCA's A and B channels should match GS's output.
        # MSE = mean of (prediction - target)^2 across all pixels and batch items.
        # We only supervise channels 0 and 1 — the NCA is free to use ch 2-13 however it wants.
        pred_loss = jnp.mean(
            (nca_outputs[:, :, :, CH_A] - A_next) ** 2 +
            (nca_outputs[:, :, :, CH_B] - B_next) ** 2
        )

        # ── Persistence loss ──────────────────────────────────────────────
        # The goal: if you nudge a pattern slightly, the NCA should smooth it back.
        # Without this: patterns look correct on training data but slowly degrade
        # over thousands of steps at runtime (small errors accumulate).
        #
        # How it works:
        #   1. Take the NCA outputs (already computed above)
        #   2. Add tiny random noise to them
        #   3. Run one more NCA step on the noisy version
        #   4. Run one more NCA step on the clean version
        #   5. They should produce similar results — penalize the difference
        #
        # This is asking: "does the NCA treat nearby states similarly?"
        # If yes: it's stable. Small perturbations don't blow up.
        key_noise = batch_keys[0]   # grab any key for generating noise
        noise = random.normal(key_noise, nca_outputs.shape) * PERSIST_NOISE
        noisy = jnp.clip(nca_outputs + noise, 0.0, 1.0)

        # Step the noisy and clean versions both one more time
        noisy_next,  _ = jax.vmap(step_one)(noisy,       batch_keys)
        clean_next,  _ = jax.vmap(step_one)(nca_outputs,  batch_keys)

        persist_loss = jnp.mean((noisy_next - clean_next) ** 2)

        # ── Total ─────────────────────────────────────────────────────────
        total = pred_loss + PERSIST_WEIGHT * persist_loss

        # Return total loss + both components as auxiliary (for logging)
        return total, (pred_loss, persist_loss)

    return loss_fn


# ── Per-variable gradient normalization ───────────────────────────────────────
def normalize_gradients(grads):
    """
    Normalize each parameter's gradient by its own L2 norm.
    This is per-variable, not global gradient clipping.

    Global clipping: if total gradient norm > threshold, scale ALL grads down.
    Per-variable:    for each weight matrix, grad = grad / ||grad||

    Why this instead of clipping:
    Late in training, a single weight might suddenly get a large gradient spike
    while everything else is small. Global clipping would scale the small
    useful gradients down along with the spike. Per-variable normalization
    handles each weight independently — the spike gets normalized, the small
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
    We don't need gradients here — just the forward pass output.
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

    jax.device_get() moves JAX GPU arrays to CPU numpy arrays.
    pickle serializes the nested dict of numpy arrays to a binary file.
    This is simple and reliable — no special checkpoint library needed.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'params_{step:06d}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    print(f"  Saved: {path}")


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
    print("  Somnivex — NCA Physics Training")
    print("  Teaching the NCA Gray-Scott reaction-diffusion")
    print("=" * 60)
    print(f"\n  JAX devices: {jax.devices()}")
    print(f"  Training grid: {TRAIN_H}x{TRAIN_W}")
    print(f"  Pool size: {POOL_SIZE}  |  Batch size: {BATCH_SIZE}")
    print(f"  Steps: {TRAIN_STEPS}  |  LR: {LEARNING_RATE}")
    print()

    # ── Initialize model ──────────────────────────────────────────────────
    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()

    # Initialize params by running a dummy input through the network.
    # Flax is "lazy" — it doesn't know the weight shapes until it sees input shapes.
    # This dummy call tells it the input is 64-wide perception vectors.
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    dummy_input = jnp.zeros((TRAIN_H, TRAIN_W, N_CHANNELS * N_FILTERS))
    params = update_net.init(subkey, dummy_input)

    # Count total parameters (just for your info)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Network params: {n_params:,}")
    print()

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Adam: the standard adaptive gradient optimizer. Works well for NCAs.
    # We apply per-variable gradient normalization manually before each step.
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # ── Compiled functions ─────────────────────────────────────────────────
    # Build these once. They get JIT-compiled on first call and reused forever.
    loss_fn        = make_loss_fn(update_net, perception_kernel)
    loss_and_grad  = jax.value_and_grad(loss_fn, has_aux=True)
    nca_batch_step = make_nca_batch_step(update_net, perception_kernel)

    # ── Initialize pool ───────────────────────────────────────────────────
    key, pool_key = random.split(key)
    pool = init_pool(pool_key, POOL_SIZE, TRAIN_H, TRAIN_W)

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"Starting training for {TRAIN_STEPS} steps...")
    print(f"Watch pred_loss fall — that's the main signal.")
    print(f"persist_loss should stay small (< pred_loss).")
    print()

    t_start = time.time()

    for step in range(TRAIN_STEPS):

        # Sample a random batch of indices (no replacement within a batch)
        batch_idx = np.random.choice(POOL_SIZE, BATCH_SIZE, replace=False)

        # Move batch from CPU pool to GPU for computation
        batch = jnp.array(pool[batch_idx])   # (BATCH, H, W, 16) on GPU

        # One PRNG key per grid in the batch (JAX needs explicit randomness)
        key, *batch_keys_list = random.split(key, BATCH_SIZE + 1)
        batch_keys = jnp.stack(batch_keys_list)   # (BATCH, 2)

        # ── Forward pass + backward pass ──────────────────────────────────
        # value_and_grad computes the loss AND the gradient in one efficient pass.
        # has_aux=True because loss_fn returns (total, (pred, persist)) — the
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
        # This is the pool mechanism — it's what creates long-term stability.
        key, *step_keys_list = random.split(key, BATCH_SIZE + 1)
        step_keys  = jnp.stack(step_keys_list)
        new_batch, _ = nca_batch_step(params, batch, step_keys)
        pool[batch_idx] = np.array(new_batch)   # move back to CPU

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
            elapsed      = time.time() - t_start
            rate         = (step + 1) / elapsed if elapsed > 0 else 1
            eta_min      = (TRAIN_STEPS - step) / rate / 60
            print(
                f"  step {step:6d}/{TRAIN_STEPS}"
                f"  loss={float(loss):.6f}"
                f"  (pred={float(pred_loss):.6f}"
                f"  persist={float(persist_loss):.6f})"
                f"  {rate:.0f} steps/s"
                f"  ETA {eta_min:.0f}m"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, step)

    # Final save
    save_checkpoint(params, TRAIN_STEPS)
    total_min = (time.time() - t_start) / 60
    print(f"\nDone. Total time: {total_min:.1f} minutes")
    print(f"Best checkpoint is probably the last one: params_{TRAIN_STEPS:06d}.pkl")

    return params


if __name__ == '__main__':
    train()
```

---

## After you type this and run it

You should see something like:

    JAX devices: [CudaDevice(id=0)]
    Initializing pool (512 states at 64x64)...
      64/512
      128/512
      ...
    Pool ready.

    Starting training for 20000 steps...
    (First step takes 30-60 seconds — JAX compiling. Normal.)

      step      0/20000  loss=0.009832  (pred=0.008981  persist=0.008512)  0 steps/s  ETA ?m
      step    100/20000  loss=0.006211  (pred=0.005800  persist=0.004114)  12 steps/s  ETA 27m
      step    200/20000  loss=0.004103  ...
      ...
      step  20000/20000  loss=0.000873  ...

If loss is still above 0.005 after step 2000: something may be wrong. Come back and check.
If loss is going down steadily: let it run. Don't interrupt.

After it finishes: checkpoints are in nca/checkpoints/. Next step is 03_nca_run.md —
loading the best checkpoint and watching the NCA run on its own.
