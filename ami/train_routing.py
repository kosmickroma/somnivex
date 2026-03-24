# ami/train_routing.py
#
# Fine-tune physarum_100000.pkl to learn two-signal routing.
#
# WHAT WE'RE TEACHING:
#   Horizontal bar at Zone A → Zone B (top right / Claude) should activate
#   Vertical bar at Zone A   → Zone C (bottom right / Gemini) should activate
#
# HOW:
#   Start from physarum_100000.pkl (already knows Physarum tunnel building)
#   Add zone identity to ch12 — re-injected every step so NCA knows where it is
#   Add signal type to ch11 — 0.5=politics, 1.0=climate, 0.0=none
#   Two-teacher tension:
#     Teacher 1: Physarum trail physics (keep routing behavior alive)
#     Teacher 2: Routing loss — wrong zone activated = penalized
#   Loss forces hidden channels to distinguish signal types
#
# Run:
#   source ~/ai-env/bin/activate
#   cd /home/kk/projects/ml/axon
#   python ami/train_routing.py

import os
import sys
import pickle
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
from pathlib import Path
from functools import partial
from scipy.ndimage import uniform_filter, zoom

sys.path.insert(0, 'nca')
sys.path.insert(0, 'ami')

from model import UpdateNet, make_perception_kernel, make_step_fn, nca_step
from physarum_core import (
    H as SIM_H, W as SIM_W,
    GAP_LEFT, GAP_RIGHT, RIGHT_MID,
    ZONE_ID_A, ZONE_ID_GAP, ZONE_ID_B, ZONE_ID_C,
    reset, step as phys_step, inject_anchors,
    inject_signal_politics, inject_signal_climate,
)

# ── Hyperparameters ────────────────────────────────────────────────────────────

TRAIN_H        = 64
TRAIN_W        = 64
POOL_SIZE      = 256
BATCH_SIZE     = 8
TRAIN_STEPS    = 20000
ROLLOUT_STEPS  = 12
LEARNING_RATE  = 1e-4
PERSIST_WEIGHT = 0.05
PERSIST_NOISE  = 0.02
ROUTING_WEIGHT = 5.0
CHECKPOINT_EVERY = 500

PHYSARUM_CHECKPOINT = Path("nca/checkpoints/physarum_100000.pkl")
OUT_DIR             = Path("ami/routing_checkpoints")

# Channel indices (must match model's 16-channel layout)
CH_TRAIL   = 0    # Physarum trail — signal shape injected here (horizontal or vertical)
CH_SIGNAL  = 11   # signal type: 0.0=none, 0.5=politics, 1.0=climate (re-injected every step)
CH_ZONE    = 12   # zone identity (re-injected every step)
CH_PHYSICS = 13   # physics bit (0.5 = Physarum)
CH_F       = 14
CH_K       = 15


# ── Zone identity map at training resolution ───────────────────────────────────

def make_zone_map_64():
    zm    = np.zeros((TRAIN_H, TRAIN_W), dtype=np.float32)
    gap_l = int(GAP_LEFT  * TRAIN_W / SIM_W)
    gap_r = int(GAP_RIGHT * TRAIN_W / SIM_W)
    mid   = TRAIN_H // 2
    zm[:, :gap_l]      = ZONE_ID_A
    zm[:, gap_l:gap_r] = ZONE_ID_GAP
    zm[:mid, gap_r:]   = ZONE_ID_B
    zm[mid:, gap_r:]   = ZONE_ID_C
    return zm

ZONE_MAP_64     = make_zone_map_64()
ZONE_MAP_64_JAX = jnp.array(ZONE_MAP_64)


# ── Training state generation ──────────────────────────────────────────────────

def inject_zone_a_signal_only(trail_sim, signal_type):
    """
    Inject signal shape into Zone A ONLY — do NOT touch Zone B or Zone C.
    This is the training version: the NCA must learn to route from shape alone.
      politics → horizontal bar at anchor row
      climate  → vertical bar down from anchor col
    """
    from physarum_core import ANCHOR_A, GAP_LEFT, SIGNAL_STRENGTH, RIGHT_MID, H as SH
    ax, ay = ANCHOR_A
    if signal_type == "politics":
        # Horizontal bar across Zone A at anchor row
        trail_sim[ax-4:ax+4, :GAP_LEFT] = SIGNAL_STRENGTH
    elif signal_type == "climate":
        # Vertical bar down Zone A from anchor col
        trail_sim[ax:SH,  ay-4:ay+4] = SIGNAL_STRENGTH
        trail_sim[:ax,    ay-4:ay+4] = SIGNAL_STRENGTH * 0.2
    return trail_sim


def generate_state(rng, signal_type=None, warmup=200):
    """
    Run Physarum sim to build live substrate, inject signal shape in Zone A only.
    Zone B and Zone C start at natural Physarum trail levels — no pre-flooding.
    The NCA must learn to route to the correct zone from the signal shape.
    signal_type: None | "politics" | "climate"
    """
    sim_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    trail, ax, ay, ah, zone_ids = reset(sim_rng)

    for _ in range(warmup):
        trail, ax, ay, ah = phys_step(trail, ax, ay, ah, sim_rng)
        trail = inject_anchors(trail)

    if signal_type == "politics":
        trail   = inject_zone_a_signal_only(trail, "politics")
        sig_val = 0.5
    elif signal_type == "climate":
        trail   = inject_zone_a_signal_only(trail, "climate")
        sig_val = 1.0
    else:
        sig_val = 0.0

    # A few Physarum steps with signal present — let it start spreading naturally
    for _ in range(int(rng.integers(5, 20))):
        trail, ax, ay, ah = phys_step(trail, ax, ay, ah, sim_rng)
        trail = inject_anchors(trail)
        if signal_type is not None:
            trail = inject_zone_a_signal_only(trail, signal_type)

    # Downsample to training resolution
    trail_64 = zoom(trail, (TRAIN_H/SIM_H, TRAIN_W/SIM_W), order=1)
    trail_64 = np.clip(trail_64, 0, 1).astype(np.float32)

    grid = np.zeros((TRAIN_H, TRAIN_W, 16), dtype=np.float32)
    grid[:, :, CH_TRAIL]   = trail_64
    grid[:, :, CH_SIGNAL]  = sig_val
    grid[:, :, CH_ZONE]    = ZONE_MAP_64
    grid[:, :, CH_PHYSICS] = 0.5
    grid[:, :, CH_F]       = 0.04
    grid[:, :, CH_K]       = 0.06
    return grid


def make_target(state, signal_type, n_steps=8):
    """
    Pure trail diffusion — no zone painting.
    pred_loss teaches trail dynamics. routing_loss is the sole routing teacher.
    signal_type kept as parameter for pool_sigs bookkeeping but not used here.
    """
    trail = state[:, :, CH_TRAIL].copy()
    for _ in range(n_steps):
        trail = uniform_filter(trail, size=3, mode='reflect')
        trail = np.clip(trail * 0.95, 0, 1)
    target = state.copy()
    target[:, :, CH_TRAIL] = trail
    return target


def init_pool(rng):
    print(f"  Generating pool ({POOL_SIZE} states)...")
    pool  = np.zeros((POOL_SIZE, TRAIN_H, TRAIN_W, 16), dtype=np.float32)
    sigs  = []
    for i in range(POOL_SIZE):
        r   = rng.random()
        sig = None if r < 0.4 else ("politics" if r < 0.7 else "climate")
        pool[i] = generate_state(rng, signal_type=sig)
        sigs.append(sig)
        if (i+1) % 64 == 0:
            print(f"    {i+1}/{POOL_SIZE}")
    return pool, sigs


# ── Loss ───────────────────────────────────────────────────────────────────────

def compute_loss(params, update_net, perception_kernel,
                 batch, targets, sig_vals, key):
    """
    1. Trail prediction loss  — match Physarum trail evolution
    2. Routing loss           — politics→ZoneB high, climate→ZoneC high
    3. Persistence loss       — stable under noise

    sig_vals: float32 array (batch,) — 0.0=none, 0.5=politics, 1.0=climate
    CH_SIGNAL re-injected every rollout step so NCA always knows signal type.
    Routing grammar must emerge in hidden channels to route trail correctly.
    """
    gap_r = int(GAP_RIGHT * TRAIN_W / SIM_W)
    mid   = TRAIN_H // 2

    def run_rollout(grids, key):
        for _ in range(ROLLOUT_STEPS):
            key, subkey = jax.random.split(key)
            batch_keys  = jax.random.split(subkey, grids.shape[0])
            grids, _    = jax.vmap(
                lambda g, k: nca_step(g, params, update_net, perception_kernel, k)
            )(grids, batch_keys)
            # Re-inject control channels every step
            grids = grids.at[:, :, :, CH_ZONE].set(ZONE_MAP_64_JAX)
            grids = grids.at[:, :, :, CH_PHYSICS].set(0.5)
            # Re-inject signal type so NCA always knows what it's routing
            grids = grids.at[:, :, :, CH_SIGNAL].set(
                sig_vals[:, None, None] * jnp.ones((grids.shape[0], TRAIN_H, TRAIN_W))
            )
        return grids

    evolved = run_rollout(batch, key)

    # Trail prediction
    pred_loss = jnp.mean(
        (evolved[:, :, :, CH_TRAIL] - targets[:, :, :, CH_TRAIL]) ** 2
    )

    # Routing — measure zone activation after rollout
    zone_b = evolved[:, :mid,  gap_r:, CH_TRAIL].mean(axis=(1, 2))
    zone_c = evolved[:, mid:,  gap_r:, CH_TRAIL].mean(axis=(1, 2))

    # Politics: penalize if C >= B
    pol_mask = (sig_vals > 0.4) & (sig_vals < 0.6)
    pol_loss = jnp.where(pol_mask,
                         jnp.maximum(0.0, zone_c - zone_b + 0.05), 0.0).mean()

    # Climate: penalize if B >= C
    cli_mask = sig_vals > 0.9
    cli_loss = jnp.where(cli_mask,
                         jnp.maximum(0.0, zone_b - zone_c + 0.05), 0.0).mean()

    routing_loss = pol_loss + cli_loss

    # Persistence
    key, pk1, pk2, pk3 = jax.random.split(key, 4)
    noisy        = batch + jax.random.normal(pk1, batch.shape) * PERSIST_NOISE
    clean_e      = run_rollout(batch, pk2)
    noisy_e      = run_rollout(noisy, pk3)
    persist_loss = jnp.mean((clean_e - noisy_e) ** 2)

    total = pred_loss + ROUTING_WEIGHT * routing_loss + PERSIST_WEIGHT * persist_loss
    return total, (pred_loss, routing_loss, persist_loss)


def normalize_grads(grads):
    def norm(g):
        return g / (jnp.sqrt(jnp.sum(g**2)) + 1e-8)
    return jax.tree_util.tree_map(norm, grads)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("AmI Routing NCA — Fine-tuning from physarum_100000.pkl")
    print(f"  Resolution: {TRAIN_H}x{TRAIN_W}  Pool: {POOL_SIZE}  "
          f"Batch: {BATCH_SIZE}  Steps: {TRAIN_STEPS}")
    print(f"  Routing weight: {ROUTING_WEIGHT}  LR: {LEARNING_RATE}")
    print()

    # Build network
    update_net       = UpdateNet()
    perception_kernel = make_perception_kernel()

    # Load checkpoint — init params from checkpoint, not random
    print(f"Loading {PHYSARUM_CHECKPOINT}...")
    with open(PHYSARUM_CHECKPOINT, 'rb') as f:
        params = pickle.load(f)
    params = jax.device_put(params)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Loaded. Parameters: {n_params:,}")

    # JIT-compile loss+grad
    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda p, b, t, sv, k: compute_loss(
            p, update_net, perception_kernel, b, t, sv, k
        ),
        has_aux=True
    ))

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # Pool
    rng = np.random.default_rng(42)
    pool, pool_sigs = init_pool(rng)
    print(f"  Pool ready — none:{pool_sigs.count(None)} "
          f"politics:{pool_sigs.count('politics')} "
          f"climate:{pool_sigs.count('climate')}")
    print()

    loss_log = []
    t0 = time.time()

    for step_i in range(1, TRAIN_STEPS + 1):
        # Sample batch
        idx    = rng.choice(POOL_SIZE, BATCH_SIZE, replace=False)
        batch  = pool[idx]
        b_sigs = [pool_sigs[i] for i in idx]

        # Build targets
        targets = np.stack([make_target(s, sig) for s, sig in zip(batch, b_sigs)])

        # Encode signal types as float labels for loss (not injected into grid)
        sig_map   = {"politics": 0.5, "climate": 1.0, None: 0.0}
        sig_vals  = jnp.array([sig_map[s] for s in b_sigs], dtype=jnp.float32)

        batch_j   = jnp.array(batch)
        targets_j = jnp.array(targets)
        key       = jax.random.PRNGKey(step_i)

        (loss, (pred, routing, persist)), grads = loss_and_grad(
            params, batch_j, targets_j, sig_vals, key
        )

        grads     = normalize_grads(grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params    = optax.apply_updates(params, updates)

        # Write evolved states back to pool
        for j, i in enumerate(idx):
            g, _ = nca_step(
                batch_j[j], params, update_net, perception_kernel,
                jax.random.PRNGKey(step_i * BATCH_SIZE + j)
            )
            g = g.at[:, :, CH_ZONE].set(ZONE_MAP_64_JAX)
            g = g.at[:, :, CH_PHYSICS].set(0.5)
            g = g.at[:, :, CH_SIGNAL].set(float(sig_vals[j]))
            pool[i] = np.array(g)

        # Reseed one slot every 10 steps
        if step_i % 10 == 0:
            ri  = int(rng.integers(0, POOL_SIZE))
            r   = rng.random()
            sig = None if r < 0.4 else ("politics" if r < 0.7 else "climate")
            pool[ri]      = generate_state(rng, signal_type=sig)
            pool_sigs[ri] = sig

        loss_log.append({
            "step": step_i, "loss": float(loss),
            "pred": float(pred), "routing": float(routing),
            "persist": float(persist),
        })

        if step_i % 100 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / step_i * (TRAIN_STEPS - step_i)
            print(f"  step {step_i:6d}  loss={float(loss):.5f}  "
                  f"pred={float(pred):.5f}  routing={float(routing):.5f}  "
                  f"persist={float(persist):.5f}  "
                  f"eta={eta:.0f}s")

        if step_i % CHECKPOINT_EVERY == 0:
            p = OUT_DIR / f"routing_{step_i:06d}.pkl"
            with open(p, 'wb') as f:
                pickle.dump(jax.device_get(params), f)
            with open(OUT_DIR / "loss_log.json", 'w') as f:
                json.dump(loss_log, f)
            print(f"  Checkpoint: {p}")

    final = OUT_DIR / f"routing_{TRAIN_STEPS:06d}.pkl"
    with open(final, 'wb') as f:
        pickle.dump(jax.device_get(params), f)
    print(f"\nDone. Final: {final}")


if __name__ == "__main__":
    main()
