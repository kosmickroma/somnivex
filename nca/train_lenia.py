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
    init_lenia_pool, make_lenia_pool_state, make_lenia_pool_state_v2,
    make_lenia_target_fn, make_kernel_fft,
)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_H = 64
TRAIN_W = 64

POOL_SIZE        = 512
BATCH_SIZE       = 32
TRAIN_STEPS      = 100000   # 2x previous run — more time to internalize 3 species + continuous ch13
ROLLOUT_STEPS    = 8
LEARNING_RATE    = 5e-5     # lower than v1 — fine-tuning from strong checkpoint
PERSIST_WEIGHT   = 0.1
PERSIST_NOISE    = 0.02
CHECKPOINT_EVERY = 1000
LOG_EVERY        = 100

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
LENIA_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'lenia_050000.pkl')  # start from fused checkpoint
PHYSARUM_DATA_PATH = os.path.join(os.path.dirname(__file__), 'physarum_training_data.npz')

# Number of Physarum samples per batch when --physarum is active (taken from GS allocation)
PHYSARUM_N = 4

# Sample ch13 uniformly from [0, 1] during training instead of hard 0/1.
# Teaches the model the full interpolation spectrum between GS and Lenia.
CH13_CONTINUOUS = True

# Initialize hidden channels (ch2-12) with small random noise in pool states.
# Forces the model to learn to use hidden state rather than ignore it.
HIDDEN_INIT_NOISE = 0.05

# Lenia ratio schedule: step threshold → n_lenia samples per batch of 32
# More Lenia earlier — we already know the model handles GS well.
LENIA_SCHEDULE = [
    (0,     8),   # steps     0–499:   8/32 = 25% Lenia
    (500,  12),   # steps   500–999:  12/32 = 37%
    (1000, 16),   # steps  1000+:     16/32 = 50% — equal weighting
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
    grid[:, :, CH_PHYSICS] = float(np.random.uniform(0.0, 0.3)) if CH13_CONTINUOUS else 0.0
    grid[:, :, CH_F]       = f
    grid[:, :, CH_K]       = k
    # Hidden channel noise — forces model to learn to use ch2-12, not ignore them
    grid[:, :, 2:13] = np.random.normal(0.0, HIDDEN_INIT_NOISE, (H, W, 11)).astype(np.float32)
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


def make_physarum_loss_fn(update_net, perception_kernel):
    """
    Physarum loss: NCA must predict the next trail concentration frame on channel 0.
    Uses pre-computed (frame_t, frame_t+1) pairs from physarum_training_data.npz.
    Re-injects CH_PHYSICS=0.5 after step. Loss = MSE on ch0 only.
    ch13=0.5 puts Physarum exactly between GS (0.0) and Lenia (1.0).
    """
    @jax.jit
    def physarum_loss_fn(params, batch_grids, batch_targets, batch_keys):
        def step_one(grid, key):
            return nca_step(grid, params, update_net, perception_kernel, key)
        nca_grids, _ = jax.vmap(step_one)(batch_grids, batch_keys)
        nca_grids = nca_grids.at[:, :, :, CH_PHYSICS].set(0.5)
        loss = jnp.mean((nca_grids[:, :, :, CH_A] - batch_targets) ** 2)
        return loss, ()
    return physarum_loss_fn


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

def save_checkpoint(params, step, prefix='lenia'):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f'{prefix}_{step:06d}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    print(f"  Saved: {path}")


def load_checkpoint(path):
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return jax.device_put(params)


def find_latest_checkpoint(checkpoint_dir, prefix='lenia_'):
    """Return (path, step) of the most recently written lenia_XXXXXX.pkl, or (None, 0)."""
    import glob as _glob
    files = _glob.glob(os.path.join(checkpoint_dir, f'{prefix}??????.pkl'))
    if not files:
        return None, 0
    def step_of(p):
        try:
            return int(os.path.basename(p).replace(prefix, '').replace('.pkl', ''))
        except ValueError:
            return 0
    # Sort by mtime — the most recently written file is the true latest checkpoint
    files.sort(key=os.path.getmtime)
    latest = files[-1]
    return latest, step_of(latest)


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest lenia_XXXXXX.pkl checkpoint')
    parser.add_argument('--gs-only', action='store_true',
                        help='GS-only training: same setup as v2 but no Lenia teacher. '
                             'Starts from params_050000.pkl, saves to gs_only_XXXXXX.pkl. '
                             'Use to test whether the 8-state grammar requires multi-physics fusion.')
    parser.add_argument('--physarum', action='store_true',
                        help='Add Physarum as a teacher. Requires nca/physarum_training_data.npz. '
                             'With --gs-only: GS+Physarum grammar experiment (saves physarum_XXXXXX.pkl). '
                             'Without --gs-only: v3 three-teacher model (saves v3_XXXXXX.pkl).')
    args = parser.parse_args()

    gs_only  = args.gs_only
    physarum = args.physarum

    # In gs-only mode, force ch13=0.0 for GS pool states (no continuous interpolation).
    # Physarum will be at ch13=0.5 — we want clean separation, not 0.0-0.3 vs 0.5.
    global CH13_CONTINUOUS
    if gs_only:
        CH13_CONTINUOUS = False

    print("=" * 60)
    if gs_only and physarum:
        print(" Somnivex — GS+Physarum Grammar Experiment: No Lenia")
    elif gs_only:
        print(" Somnivex — GS-Only Training: Hidden Channel Noise, No Lenia")
    elif physarum:
        print(" Somnivex — v3 Training: GS + Lenia + Physarum (3 teachers)")
    else:
        print(" Somnivex — v2 Training: 3 Species + Continuous ch13 + Hidden Noise")
    print("=" * 60)
    print(f"\n JAX: {jax.devices()}")
    print(f" Grid: {TRAIN_H}x{TRAIN_W}  Pool: {POOL_SIZE}  Batch: {BATCH_SIZE}")
    print(f" Steps: {TRAIN_STEPS}  LR: {LEARNING_RATE}  Rollout: {ROLLOUT_STEPS}")
    print(f" Hidden noise: {HIDDEN_INIT_NOISE}  Continuous ch13: {CH13_CONTINUOUS}")
    print()

    # ── Model init ────────────────────────────────────────────────────────
    update_net        = UpdateNet()
    perception_kernel = make_perception_kernel()
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    dummy = jnp.zeros((TRAIN_H, TRAIN_W, N_CHANNELS * N_FILTERS))
    params = update_net.init(subkey, dummy)

    GS_ONLY_START = os.path.join(CHECKPOINT_DIR, 'params_050000.pkl')
    if gs_only and physarum:
        ckpt_prefix = 'physarum'
    elif gs_only:
        ckpt_prefix = 'gs_only'
    elif physarum:
        ckpt_prefix = 'v3'
    else:
        ckpt_prefix = 'lenia'

    if args.resume:
        resume_path, start_step = find_latest_checkpoint(CHECKPOINT_DIR, prefix=ckpt_prefix + '_')
        if resume_path is None:
            print(f"ERROR: --resume specified but no {ckpt_prefix}_XXXXXX.pkl found in checkpoints/")
            sys.exit(1)
        params = load_checkpoint(resume_path)
        print(f" RESUMING from: {resume_path}  (step {start_step} → {TRAIN_STEPS})")
    elif gs_only:
        start_step = 0
        if not os.path.exists(GS_ONLY_START):
            print(f"ERROR: GS-only start checkpoint not found: {GS_ONLY_START}")
            sys.exit(1)
        params = load_checkpoint(GS_ONLY_START)
        print(f" Loaded: {GS_ONLY_START}  [GS-only experiment]")
    else:
        start_step = 0
        if not os.path.exists(LENIA_CHECKPOINT):
            print(f"ERROR: Fused checkpoint not found: {LENIA_CHECKPOINT}")
            print("Expected lenia_050000.pkl from v1 training.")
            sys.exit(1)
        params = load_checkpoint(LENIA_CHECKPOINT)
        print(f" Loaded: {LENIA_CHECKPOINT}")

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f" Params: {n_params:,}\n")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LEARNING_RATE),
    )
    opt_state = optimizer.init(params)

    # ── Lenia kernel (precomputed once, shared between pool builder + loss) ──
    fK_np  = make_kernel_fft(LENIA_R, TRAIN_H, TRAIN_W)
    fK_jax = jnp.array(fK_np)

    # ── Load Physarum training data ────────────────────────────────────────
    if physarum:
        if not os.path.exists(PHYSARUM_DATA_PATH):
            print(f"ERROR: Physarum training data not found: {PHYSARUM_DATA_PATH}")
            print("Run: python kktodo/physarum_typing/03_generate_training_data.py")
            sys.exit(1)
        _phys_data = np.load(PHYSARUM_DATA_PATH)
        physarum_frames = _phys_data['frames']   # (N, H, W) float32
        physarum_max_idx = len(physarum_frames) - 1
        print(f" Physarum data: {len(physarum_frames)} frames  ({physarum_max_idx} pairs)")

    # ── Compile functions ─────────────────────────────────────────────────
    lenia_targets_batch_fn = make_lenia_target_fn(fK_jax)
    gs_loss_fn             = make_gs_loss_fn(update_net, perception_kernel)
    lenia_loss_fn          = make_lenia_loss_fn(update_net, perception_kernel, lenia_targets_batch_fn)
    physarum_loss_fn       = make_physarum_loss_fn(update_net, perception_kernel) if physarum else None
    gs_loss_and_grad       = jax.value_and_grad(gs_loss_fn,     has_aux=True)
    lenia_loss_and_grad    = jax.value_and_grad(lenia_loss_fn,  has_aux=True)
    physarum_loss_and_grad = jax.value_and_grad(physarum_loss_fn, argnums=0, has_aux=True) if physarum else None
    nca_batch_step         = make_nca_batch_step(update_net, perception_kernel)

    # ── Init pools ────────────────────────────────────────────────────────
    key, pool_key = random.split(key)
    gs_pool             = init_gs_pool(pool_key, POOL_SIZE, TRAIN_H, TRAIN_W)
    lenia_pool, fK_np   = init_lenia_pool(POOL_SIZE, TRAIN_H, TRAIN_W)
    lenia_rng           = np.random.default_rng(seed=99)

    if gs_only:
        print(f" Training {TRAIN_STEPS} steps — GS only, no Lenia teacher")
    else:
        print(f" Training {TRAIN_STEPS} steps with ratio schedule:")
        for thresh, n in LENIA_SCHEDULE:
            print(f"  step {thresh:5d}+: {n}/{BATCH_SIZE} Lenia = {n/BATCH_SIZE*100:.0f}%")
    print()

    t_start = time.time()

    for step in range(start_step, TRAIN_STEPS):
        n_physarum = PHYSARUM_N if physarum else 0
        n_lenia    = 0 if gs_only else get_n_lenia(step)
        n_gs       = BATCH_SIZE - n_lenia - n_physarum

        # Sample from GS pool
        gs_idx   = np.random.choice(POOL_SIZE, n_gs, replace=False)
        gs_batch = jnp.array(gs_pool[gs_idx])
        key, *gs_key_list = random.split(key, n_gs + 1)
        gs_keys  = jnp.stack(gs_key_list)

        (gs_loss, (gs_pred, gs_persist)), gs_grads = gs_loss_and_grad(
            params, gs_batch, gs_keys
        )

        if gs_only:
            combined_grads = gs_grads
            lenia_loss = 0.0
        else:
            lenia_idx   = np.random.choice(POOL_SIZE, n_lenia, replace=False)
            lenia_batch = jnp.array(lenia_pool[lenia_idx])
            key, *lenia_key_list = random.split(key, n_lenia + 1)
            lenia_keys  = jnp.stack(lenia_key_list)
            (lenia_loss, _), lenia_grads = lenia_loss_and_grad(
                params, lenia_batch, lenia_keys
            )
            combined_grads = jax.tree_util.tree_map(
                lambda g, l: g + l,
                gs_grads, lenia_grads
            )

        physarum_loss = 0.0
        if physarum:
            phys_idx     = np.random.choice(physarum_max_idx, n_physarum, replace=False)
            phys_grids   = np.zeros((n_physarum, TRAIN_H, TRAIN_W, N_CHANNELS), dtype=np.float32)
            phys_grids[:, :, :, CH_A]       = physarum_frames[phys_idx]
            phys_grids[:, :, :, CH_PHYSICS] = 0.5
            phys_grids[:, :, :, 2:13]       = np.random.normal(
                0.0, HIDDEN_INIT_NOISE, (n_physarum, TRAIN_H, TRAIN_W, 11)
            ).astype(np.float32)
            phys_targets = jnp.array(physarum_frames[phys_idx + 1])
            phys_batch   = jnp.array(phys_grids)
            key, *phys_key_list = random.split(key, n_physarum + 1)
            phys_keys = jnp.stack(phys_key_list)
            (physarum_loss, _), physarum_grads = physarum_loss_and_grad(
                params, phys_batch, phys_targets, phys_keys
            )
            combined_grads = jax.tree_util.tree_map(
                lambda g, p: g + p,
                combined_grads, physarum_grads
            )

        updates, opt_state = optimizer.update(combined_grads, opt_state)
        params = optax.apply_updates(params, updates)

        # ── Write evolved states back to GS pool ──────────────────────────
        key, *gs_step_keys = random.split(key, n_gs + 1)
        new_gs, _ = nca_batch_step(params, gs_batch, jnp.stack(gs_step_keys))
        new_gs_np = np.array(new_gs)
        new_gs_np[:, :, :, CH_PHYSICS] = 0.0
        new_gs_np[:, :, :, CH_F]       = np.array(gs_batch[:, :, :, CH_F])
        new_gs_np[:, :, :, CH_K]       = np.array(gs_batch[:, :, :, CH_K])
        if not np.any(np.isnan(new_gs_np)):
            gs_pool[gs_idx] = new_gs_np

        if not gs_only and n_lenia > 0:
            key, *lenia_step_keys = random.split(key, n_lenia + 1)
            new_lenia, _ = nca_batch_step(params, lenia_batch, jnp.stack(lenia_step_keys))
            new_lenia_np = np.array(new_lenia)
            new_lenia_np[:, :, :, CH_PHYSICS] = 1.0
            new_lenia_np[:, :, :, CH_F]       = np.array(lenia_batch[:, :, :, CH_F])
            new_lenia_np[:, :, :, CH_K]       = np.array(lenia_batch[:, :, :, CH_K])
            if not np.any(np.isnan(new_lenia_np)):
                lenia_pool[lenia_idx] = new_lenia_np

        # ── Refresh pools ─────────────────────────────────────────────────
        if step % GS_REFRESH_EVERY == 0:
            key, sk = random.split(key)
            gs_pool[gs_idx[0]] = make_gs_pool_state(sk, TRAIN_H, TRAIN_W)

        if not gs_only and step % LENIA_REFRESH_EVERY == 0:
            lenia_pool[lenia_idx[0]] = make_lenia_pool_state_v2(
                TRAIN_H, TRAIN_W, fK_np, lenia_rng,
                hidden_noise=HIDDEN_INIT_NOISE,
                ch13_max=1.0,
            )

        # ── Logging ───────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            rate    = (step + 1) / elapsed if elapsed > 0 else 1
            eta     = (TRAIN_STEPS - step) / rate / 60
            msg = f" step {step:5d}/{TRAIN_STEPS}  gs={float(gs_loss):.5f}"
            if not gs_only:
                msg += f"  lenia={float(lenia_loss):.5f}  lenia%={n_lenia/BATCH_SIZE*100:.0f}%"
            if physarum:
                msg += f"  phys={float(physarum_loss):.5f}"
            msg += f"  {rate:.1f}it/s  ETA {eta:.0f}m"
            print(msg)

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            save_checkpoint(params, step, prefix=ckpt_prefix)

    save_checkpoint(params, TRAIN_STEPS, prefix=ckpt_prefix)
    print(f"\nDone. {(time.time()-t_start)/60:.1f} minutes")
    print(f"Checkpoint: {ckpt_prefix}_{TRAIN_STEPS:06d}.pkl")
    return params


if __name__ == "__main__":
    train()
