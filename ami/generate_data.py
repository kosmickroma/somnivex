# ami/generate_data.py
#
# Headless Physarum data generator for routing NCA training.
#
# Runs N simulations. Each simulation:
#   - Warms up for WARMUP_STEPS (zones settle, trails form)
#   - Optionally fires a bridge signal at a random step
#   - Captures trail snapshots at key moments
#   - Labels each snapshot: was bridge fired? is zone_b activated?
#
# Output: ami/training_data/
#   snapshots_XXXXXX.npy   — (T, H, W) trail state array
#   labels_XXXXXX.json     — per-snapshot metadata
#   summary.json           — dataset stats
#
# Run:
#   python ami/generate_data.py --runs 100 --steps 500
#   python ami/generate_data.py --runs 1000 --steps 800 --out ami/training_data

import numpy as np
import json
import argparse
import time
from pathlib import Path
from physarum_core import (
    reset, step, inject_anchors, reseed_agents,
    fire_bridge_signal, reinforce_bridge, measure_zones,
    BRIDGE_HOLD_STEPS, ZONE_B_THRESHOLD, RESEED_INTERVAL,
    H, W, N_AGENTS
)


def run_simulation(rng, total_steps, bridge_at=None, snapshot_every=50):
    """
    Run one simulation.

    Args:
        rng:            numpy RNG
        total_steps:    how many steps to run
        bridge_at:      step number to fire bridge signal (None = no bridge)
        snapshot_every: save trail state every N steps

    Returns:
        snapshots: list of (step, trail copy)
        labels:    list of dicts with per-snapshot metadata
    """
    trail, agent_x, agent_y, agent_heading = reset(rng)

    bridge_fired      = False
    bridge_hold       = 0
    zone_b_activated  = False

    snapshots = []
    labels    = []

    for s in range(1, total_steps + 1):
        trail, agent_x, agent_y, agent_heading = step(
            trail, agent_x, agent_y, agent_heading, rng
        )
        trail = inject_anchors(trail)

        # Stability: reseed drifted agents periodically
        if s % RESEED_INTERVAL == 0:
            agent_x, agent_y, agent_heading = reseed_agents(
                agent_x, agent_y, agent_heading, rng
            )

        # Fire bridge signal at scheduled step
        if bridge_at is not None and s == bridge_at and not bridge_fired:
            trail        = fire_bridge_signal(trail)
            bridge_fired = True
            bridge_hold  = BRIDGE_HOLD_STEPS

        # Reinforce while hold active
        if bridge_hold > 0:
            trail = reinforce_bridge(trail)
            bridge_hold -= 1

        # Measure zones
        zone_a, zone_b, gap = measure_zones(trail)

        if bridge_fired and zone_b >= ZONE_B_THRESHOLD:
            zone_b_activated = True

        # Save snapshot
        if s % snapshot_every == 0:
            snapshots.append(trail.copy())
            labels.append({
                "step":             s,
                "bridge_fired":     bridge_fired,
                "bridge_at":        bridge_at,
                "zone_b_activated": zone_b_activated,
                "zone_a":           round(zone_a, 5),
                "zone_b":           round(zone_b, 5),
                "gap":              round(gap, 5),
            })

    return snapshots, labels


def main():
    parser = argparse.ArgumentParser(description="Physarum routing data generator")
    parser.add_argument("--runs",          type=int,   default=100,
                        help="Number of simulations to run")
    parser.add_argument("--steps",         type=int,   default=600,
                        help="Steps per simulation")
    parser.add_argument("--warmup",        type=int,   default=200,
                        help="Warmup steps before bridge can fire (within --steps)")
    parser.add_argument("--snapshot-every",type=int,   default=50,
                        help="Save snapshot every N steps")
    parser.add_argument("--no-bridge-frac",type=float, default=0.3,
                        help="Fraction of runs with NO bridge signal (isolation baseline)")
    parser.add_argument("--out",           type=str,   default="ami/training_data",
                        help="Output directory")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    summary = {
        "runs":           args.runs,
        "steps":          args.steps,
        "warmup":         args.warmup,
        "snapshot_every": args.snapshot_every,
        "no_bridge_frac": args.no_bridge_frac,
        "H": H, "W": W,
        "N_AGENTS": N_AGENTS,
        "completed": 0,
        "bridge_runs": 0,
        "no_bridge_runs": 0,
        "zone_b_activated_runs": 0,
    }

    t0 = time.time()

    for run_idx in range(args.runs):
        # Decide whether to fire a bridge this run
        if rng.random() < args.no_bridge_frac:
            bridge_at = None
            summary["no_bridge_runs"] += 1
        else:
            # Fire at a random step after warmup
            bridge_at = int(rng.integers(args.warmup, args.steps))
            summary["bridge_runs"] += 1

        snapshots, labels = run_simulation(
            rng,
            total_steps    = args.steps,
            bridge_at      = bridge_at,
            snapshot_every = args.snapshot_every,
        )

        # Save snapshots as single (T, H, W) array
        run_id = f"{run_idx:06d}"
        snap_arr = np.stack(snapshots, axis=0).astype(np.float32)
        np.save(out_dir / f"snapshots_{run_id}.npy", snap_arr)

        # Save labels
        (out_dir / f"labels_{run_id}.json").write_text(
            json.dumps(labels, indent=2)
        )

        # Track activation
        if any(l["zone_b_activated"] for l in labels):
            summary["zone_b_activated_runs"] += 1

        summary["completed"] = run_idx + 1

        # Progress every 10 runs
        if (run_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            per_run = elapsed / (run_idx + 1)
            remaining = per_run * (args.runs - run_idx - 1)
            print(f"  run {run_idx+1:4d}/{args.runs}  "
                  f"bridge={bridge_at if bridge_at else 'none':>5}  "
                  f"activated={any(l['zone_b_activated'] for l in labels)}  "
                  f"elapsed={elapsed:.0f}s  eta={remaining:.0f}s")

    # Write summary
    summary["elapsed_s"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"Done. {args.runs} runs in {summary['elapsed_s']}s")
    print(f"  bridge runs:      {summary['bridge_runs']}")
    print(f"  no-bridge runs:   {summary['no_bridge_runs']}")
    print(f"  zone_b activated: {summary['zone_b_activated_runs']} / {summary['bridge_runs']} bridge runs")
    print(f"  saved to:         {out_dir}/")
    print()
    print("Each run saved as:")
    print("  snapshots_XXXXXX.npy  — shape (T, 256, 256) float32")
    print("  labels_XXXXXX.json    — step, bridge_fired, zone_b_activated, zone measurements")


if __name__ == "__main__":
    main()
