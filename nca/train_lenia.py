# nca/train_lenia.py — Dual-teacher NCA fine-tuning: GS + Lenia.
#
# Loads the existing GS checkpoint and continues training on a mixed pool:
# some batches teach GS reaction-diffusion, some teach Lenia creature dynamics.
#
# The NCA must find a shared hidden representation that explains BOTH.
# Hope: in free-run, it invents a third thing — GS chemistry with Lenia creatures.
#
# Run from project root:
#     python nca/train_lenia.py
#
# Saves to nca/checkpoints/lenia_XXXXXX.pkl  (GS checkpoint untouched)

import os
import sys
import time
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gs.engine import GS_REGIMES, gs_step, init_gs_grid
from nca.model import (
    N_CHANNELS, N_FILTERS,
    CH_A, CH_B, CH_F, CH_K,
    UpdateNet, make_perception_kernel, nca_step,
)
from nca.lenia import (
    CH_PHYSICS, LENIA_R,
    init_lenia_pool, make_lenia_pool_state, make_lenia_target_fn,
    make_kernel_fft,
)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_H = 64
TRAIN_W = 64

POOL_SIZE        = 512
BATCH_SIZE       = 32
TRAIN_STEPS      = 50000
ROLLOUT_STEPS    = 8
LEARNING_RATE    = 1e-4      # lower than original — fine-tuning not cold training
PERSIST_WEIGHT   = 0.1
PERSIST_NOISE    = 0.02
CHECKPOINT_EVERY = 1000
LOG_EVERY        = 100

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
GS_CHECKPOINT  = os.path.join(CHECKPOINT_DIR, 'params_050000.pkl')

# Lenia ratio schedule: step threshold → n_lenia samples per batch of 32
# n_gs = BATCH_SIZE - n_lenia
# Start conservative — let GS knowledge stay dominant early.
LENIA_SCHEDULE = [
    (0,    5),    # steps    0–199:   5/32 = 15% Lenia
    (200,  6),    # steps  200–499:   6/32 = 19%
    (500, 10),    # steps  500–799:  10/32 = 31%
    (800, 13),    # steps  800+:     13/32 = 41%
]

# Lenia pool refresh: replace one Lenia pool state every N steps
LENIA_REFRESH_EVERY = 50
GS_REFRESH_EVERY    = 10    # same as original train.py


def get_n_lenia(step):
    n = LENIA_SCHEDULE[0][1]
    for threshold, val in LENIA_SCHEDULE:
        if step >= threshold:
            n = val
    return n


# ── GS pool (same as train.py, plus physics bit = 0.0) ───────────────────────

def make_gs_pool_state(key, H, W):
    regime_names = list(GS_REGIMES.keys())
    key, sk = random.split(key)
    idx = int(random.randint(sk, (), 0, len(regime_names)))
    f, k = GS_REGIMES[regime_names[idx]]

    key, sk = random.split(key)
    A, B = init_gs_grid(sk, H, W)
    key, sk = random.split(key)
    warmup = int(random.randint(sk, (), 100, 500))
    for _ in range(warmup):
        A, B = gs_step(A, B, f, k)

    grid = np.zeros((H, W, N_CHANNELS), dtype=np.float32)
    grid[:, :, CH_A]       = np.array(A)
    grid[:, :, CH_B]       = np.array(B)
    grid[:, :, CH_PHYSICS] = 0.0    # physics bit = GS
    grid[:, :, CH_F]       = f
    grid[:, :, CH_K]       = k
    return grid


def init_gs_pool(key, pool_size, H, W):
    print(f"Initializing GS pool ({pool_size} states at {H}x{W})...")
    pool = np.zeros((pool_size, H, W, N_CHANNELS), dtype=np.float32)
    for i in range(pool_size):
        key, sk = random.split(key)
        pool[i] = make_gs_pool_state(sk, H, W)
        if (i + 1) % 64 == 0:
            print(f"  {i+1}/{pool_size}")
    print("GS pool ready.\n")
    return pool


# ── GS target batch ───────────────────────────────────────────────────────────

@jax.jit
def gs_targets_batch(batch_A, batch_B, batch_f, batch_k):
    return jax.vmap(gs_step, in_axes=(0, 0, 0, 0))(batch_A, batch_B, batch_f, batch_k)


# ── Gradient normalization ────────────────────────────────────────────────────

def normalize_gradients(grads):
    return jax.tree_util.tree_map(
        lambda g: g / (jnp.sqrt(jnp.sum(g**2)) + 1e-8),
        grads
    )


# ── Loss functions ────────────────────────────────────────────────────────────

def make_gs_loss_fn(update_net, perception_kernel):
    """
    GS loss: same multi-step rollout as original train.py.
    Re-injects CH_PHYSICS=0, CH_F=f, CH_K=k after every NCA step.
    Loss = MSE on channels A and B across the trajectory.
    """
    @jax.jit
    def gs_loss_fn(params, batch_grids, batch_keys):
        def step_one(grid, key):
            return nca_step(grid, params, update_net, perception_kernel, key)

        f_in = batch_grids[:, 0, 0, CH_F]
        k_in = batch_grids[:, 0, 0, CH_K]

        nca_grids       = batch_grids
        gs_A            = batch_grids[:, :, :, CH_A]
        gs_B            = batch_grids[:, :, :, CH_B]
        total_pred_loss = jnp.zeros(())

        for _ in range(ROLLOUT_STEPS):
            nca_grids, batch_keys = jax.vmap(step_one)(nca_grids, batch_keys)
            # Re-inject all three control channels every step
            nca_grids = nca_grids.at[:, :, :, CH_PHYSICS].set(batch_grids[:, :, :, CH_PHYSICS])
            nca_grids = nca_grids.at[:, :, :, CH_F].set(batch_grids[:, :, :, CH_F])
            nca_grids = nca_grids.at[:, :, :, CH_K].set(batch_grids[:, :, :, CH_K])
            gs_A, gs_B = gs_targets_batch(gs_A, gs_B, f_in, k_in)
            total_pred_loss = total_pred_loss + jnp.mean(
                (nca_grids[:, :, :, CH_A] - gs_A)**2 +
                (nca_grids[:, :, :, CH_B] - gs_B)**2
            )

        pred_loss = total_pred_loss / ROLLOUT_STEPS

        # Persistence loss (same as original)
        noise      = random.normal(batch_keys[0], nca_grids.shape) * PERSIST_NOISE
        noisy      = jnp.clip(nca_grids + noise, 0.0, 1.0)
        noisy_next, _ = jax.vmap(step_one)(noisy,     batch_keys)
        clean_next, _ = jax.vmap(step_one)(nca_grids, batch_keys)
        persist_loss  = jnp.mean((noisy_next - clean_next)**2)

        total = pred_loss + PERSIST_WEIGHT * persist_loss
        return total, (pred_loss, persist_loss)

    return gs_loss_fn


def make_lenia_loss_fn(update_net, perception_kernel, lenia_targets_batch_fn):
    """
    Lenia loss: NCA must track Lenia trajectory on channel 0.
    Re-injects CH_PHYSICS=1, CH_F=mu, CH_K=sigma after every NCA step.
    Loss = MSE on channel 0 only (Lenia has one state variable, not two).

    The NCA sees the same 16-channel grid but with:
    - ch0 carrying Lenia activation instead of GS A
    - ch1 = 0 (no supervision from Lenia)
    - ch13 = 1.0 (physics bit tells NCA "this is Lenia physics")
    - ch14/15 carrying mu/sigma instead of f/k
    """
    @jax.jit
    def lenia_loss_fn(params, batch_grids, batch_keys):
        def step_one(grid, key):
            return nca_step(grid, params, update_net, perception_kernel, key)

        mu_in    = batch_grids[:, 0, 0, CH_F]
        sigma_in = batch_grids[:, 0, 0, CH_K]

        nca_grids  = batch_grids
        lenia_A    = batch_grids[:, :, :, CH_A]
        total_loss = jnp.zeros(())

        for _ in range(ROLLOUT_STEPS):
            nca_grids, batch_keys = jax.vmap(step_one)(nca_grids, batch_keys)
            # Re-inject control channels
            nca_grids = nca_grids.at[:, :, :, CH_PHYSICS].set(batch_grids[:, :, :, CH_PHYSICS])
            nca_grids = nca_grids.at[:, :, :, CH_F].set(batch_grids[:, :, :, CH_F])
            nca_grids = nca_grids.at[:, :, :, CH_K].set(batch_grids[:, :, :, CH_K])
            # Lenia ground truth step
            lenia_A = lenia_targets_batch_fn(lenia_A, mu_in, sigma_in)
            # Only supervise ch0 — Lenia owns ch0, GS owns ch1
            total_loss = total_loss + jnp.mean(
                (nca_grids[:, :, :, CH_A] - lenia_A)**2
            )

        return total_loss / ROLLOUT_STEPS, ()

    return lenia_loss_fn


# ── Batch step (for pool writes) ──────────────────────────────────────────────

def make_nca_batch_step(update_net, perception_kernel):
    @jax.jit
    def batch_step(params, batch_grids, batch_keys):
        def step_one(grid, key):
            return nca_step(grid, params, update_net, perception_kernel, key)
        return jax.vmap(step_one)(batch_grids, batch_keys)
    return batch_step


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(params, step):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'lenia_{step:06d}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    print(f"  Saved: {path}")


def load_checkpoint(path):
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return jax.device_put(params)


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print(" Somnivex — Dual-Teacher NCA (GS + Lenia)")
    print(" Fine-tuning GS checkpoint with Lenia creatures")
    print("=" * 60)
    print(f"\n JAX: {jax.devices()}")
    print(f" Grid: {TRAIN_H}x{TRAIN_W}  Pool: {POOL_SIZE}  Batch: {BATCH_SIZE}")
    print(f" Steps: {TRAIN_STEPS}  LR: {LEARNING_RATE}  Rollout: {ROLLOUT_STEPS}")
    print()

    # ── Model init ────────────────────────────────────────────────────────
    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    dummy = jnp.zeros((TRAIN_H, TRAIN_W, N_CHANNELS * N_FILTERS))
    params = update_net.init(subkey, dummy)

    if not os.path.exists(GS_CHECKPOINT):
        print(f"ERROR: GS checkpoint not found: {GS_CHECKPOINT}")
        print("Run nca/train.py first.")
        sys.exit(1)
    params = load_checkpoint(GS_CHECKPOINT)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f" Loaded: {GS_CHECKPOINT}  ({n_params:,} params)\n")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LEARNING_RATE),
    )
    opt_state = optimizer.init(params)

    # ── Lenia kernel (precomputed once, shared between pool builder + loss) ──
    fK_np  = make_kernel_fft(LENIA_R, TRAIN_H, TRAIN_W)
    fK_jax = jnp.array(fK_np)

    # ── Compile functions ─────────────────────────────────────────────────
    lenia_targets_batch_fn = make_lenia_target_fn(fK_jax)
    gs_loss_fn             = make_gs_loss_fn(update_net, perception_kernel)
    lenia_loss_fn          = make_lenia_loss_fn(update_net, perception_kernel, lenia_targets_batch_fn)
    gs_loss_and_grad       = jax.value_and_grad(gs_loss_fn,     has_aux=True)
    lenia_loss_and_grad    = jax.value_and_grad(lenia_loss_fn,  has_aux=True)
    nca_batch_step         = make_nca_batch_step(update_net, perception_kernel)

    # ── Init pools ────────────────────────────────────────────────────────
    key, pool_key = random.split(key)
    gs_pool             = init_gs_pool(pool_key, POOL_SIZE, TRAIN_H, TRAIN_W)
    lenia_pool, fK_np   = init_lenia_pool(POOL_SIZE, TRAIN_H, TRAIN_W)
    lenia_rng           = np.random.default_rng(seed=99)

    print(f" Training {TRAIN_STEPS} steps with ratio schedule:")
    for thresh, n in LENIA_SCHEDULE:
        print(f"  step {thresh:5d}+: {n}/{BATCH_SIZE} Lenia = {n/BATCH_SIZE*100:.0f}%")
    print()

    t_start = time.time()

    for step in range(TRAIN_STEPS):
        n_lenia = get_n_lenia(step)
        n_gs    = BATCH_SIZE - n_lenia

        # Sample from each pool
        gs_idx    = np.random.choice(POOL_SIZE, n_gs,    replace=False)
        lenia_idx = np.random.choice(POOL_SIZE, n_lenia, replace=False)

        gs_batch    = jnp.array(gs_pool[gs_idx])
        lenia_batch = jnp.array(lenia_pool[lenia_idx])

        key, *gs_key_list    = random.split(key, n_gs    + 1)
        key, *lenia_key_list = random.split(key, n_lenia + 1)
        gs_keys    = jnp.stack(gs_key_list)
        lenia_keys = jnp.stack(lenia_key_list)

        # ── Forward + backward for each physics ───────────────────────────
        # Two separate backward passes. Gradients are summed below.
        # This is mathematically equivalent to one combined loss pass,
        # but avoids dynamic batch size issues with JAX's JIT.
        (gs_loss, (gs_pred, gs_persist)), gs_grads = gs_loss_and_grad(
            params, gs_batch, gs_keys
        )
        (lenia_loss, _), lenia_grads = lenia_loss_and_grad(
            params, lenia_batch, lenia_keys
        )

        # Sum gradients — each already reflects its batch size
        combined_grads = jax.tree_util.tree_map(
            lambda g, l: g + l,
            gs_grads, lenia_grads
        )

        updates, opt_state = optimizer.update(combined_grads, opt_state)
        params = optax.apply_updates(params, updates)

        # ── Write evolved states back to pools ─────────────────────────────
        key, *gs_step_keys    = random.split(key, n_gs    + 1)
        key, *lenia_step_keys = random.split(key, n_lenia + 1)
        new_gs,    _ = nca_batch_step(params, gs_batch,    jnp.stack(gs_step_keys))
        new_lenia, _ = nca_batch_step(params, lenia_batch, jnp.stack(lenia_step_keys))
        new_gs_np    = np.array(new_gs)
        new_lenia_np = np.array(new_lenia)
        # Re-inject control channels after write-back so NCA can't corrupt them.
        # Critical for Lenia: if sigma (ch15) drifts to 0, the Lenia target
        # step computes (U-mu)/sigma → division by zero → NaN cascade.
        new_gs_np[:, :, :, CH_PHYSICS] = 0.0
        new_gs_np[:, :, :, CH_F]       = np.array(gs_batch[:, :, :, CH_F])
        new_gs_np[:, :, :, CH_K]       = np.array(gs_batch[:, :, :, CH_K])
        new_lenia_np[:, :, :, CH_PHYSICS] = 1.0
        new_lenia_np[:, :, :, CH_F]       = np.array(lenia_batch[:, :, :, CH_F])
        new_lenia_np[:, :, :, CH_K]       = np.array(lenia_batch[:, :, :, CH_K])
        if not np.any(np.isnan(new_gs_np)):
            gs_pool[gs_idx]       = new_gs_np
        if not np.any(np.isnan(new_lenia_np)):
            lenia_pool[lenia_idx] = new_lenia_np

        # ── Refresh pools with fresh states ───────────────────────────────
        # GS: inject one fresh GS state every GS_REFRESH_EVERY steps
        if step % GS_REFRESH_EVERY == 0:
            key, sk = random.split(key)
            gs_pool[gs_idx[0]] = make_gs_pool_state(sk, TRAIN_H, TRAIN_W)

        # Lenia: inject one fresh Lenia state every LENIA_REFRESH_EVERY steps
        if step % LENIA_REFRESH_EVERY == 0:
            lenia_pool[lenia_idx[0]] = make_lenia_pool_state(
                TRAIN_H, TRAIN_W, fK_np, lenia_rng
            )

        # ── Logging ───────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            rate    = (step + 1) / elapsed if elapsed > 0 else 1
            eta     = (TRAIN_STEPS - step) / rate / 60
            print(
                f" step {step:5d}/{TRAIN_STEPS}"
                f"  gs={float(gs_loss):.5f}"
                f"  lenia={float(lenia_loss):.5f}"
                f"  lenia%={n_lenia/BATCH_SIZE*100:.0f}%"
                f"  {rate:.1f}it/s  ETA {eta:.0f}m"
            )

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, step)

    save_checkpoint(params, TRAIN_STEPS)
    print(f"\nDone. {(time.time()-t_start)/60:.1f} minutes")
    print(f"Checkpoint: lenia_{TRAIN_STEPS:06d}.pkl")
    return params


if __name__ == "__main__":
    train()
