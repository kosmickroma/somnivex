# ami/physarum_core.py
#
# Shared Physarum simulation core — no pygame, no display.
# Used by both experiment_zones.py (interactive) and generate_data.py (headless).
#
# Jones 2010 agent-based slime mold physics.
# Two zones separated by a gap. Anchors act as persistent food sources.
# Bridge signal can be fired to connect Zone A → Zone B.

import numpy as np
from scipy.ndimage import uniform_filter

# ── Parameters ────────────────────────────────────────────────────────────────

H, W              = 256, 256
N_AGENTS          = 3000
DECAY             = 0.95
DIFFUSE_R         = 1
DEPOSIT           = 1.5
SENSOR_DIST       = 9
SENSOR_ANGLE      = 0.4
ROTATE_ANGLE      = 0.3

GAP_WIDTH         = 30
SIGNAL_STRENGTH   = 0.8
BRIDGE_HOLD_STEPS = 200
ZONE_B_THRESHOLD  = 0.052

ANCHOR_STRENGTH   = 1.0
ANCHOR_RADIUS     = 8
ANCHOR_SPREAD     = 20      # std dev for agent init clustering

# ── Zone boundaries ────────────────────────────────────────────────────────────

CENTER    = W // 2
GAP_LEFT  = CENTER - GAP_WIDTH // 2
GAP_RIGHT = CENTER + GAP_WIDTH // 2

# Anchor positions: center of each zone
ANCHOR_A  = (H // 2, GAP_LEFT // 2)
ANCHOR_B  = (H // 2, GAP_RIGHT + (W - GAP_RIGHT) // 2)

# ── Stability fix: reseed threshold ───────────────────────────────────────────
# If a zone loses too many agents to the other side, reseed near the anchor.
# Agents that wander into the gap or wrong zone get teleported back.
RESEED_INTERVAL   = 500     # steps between reseed checks
RESEED_FRACTION   = 0.1     # fraction of agents to reseed if zone is depleted


# ── Agent initialization ───────────────────────────────────────────────────────

def make_agents(rng):
    """Initialize agents clustered near anchor points."""
    n_left  = N_AGENTS // 2
    n_right = N_AGENTS - n_left

    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B

    lx = np.clip(rng.normal(ax, ANCHOR_SPREAD, n_left),  5, H-5).astype(np.float32)
    ly = np.clip(rng.normal(ay, ANCHOR_SPREAD, n_left),  5, GAP_LEFT-5).astype(np.float32)
    lh = rng.uniform(0, 2 * np.pi, n_left).astype(np.float32)

    rx = np.clip(rng.normal(bx, ANCHOR_SPREAD, n_right), 5, H-5).astype(np.float32)
    ry = np.clip(rng.normal(by, ANCHOR_SPREAD, n_right), GAP_RIGHT+5, W-5).astype(np.float32)
    rh = rng.uniform(0, 2 * np.pi, n_right).astype(np.float32)

    agent_x       = np.concatenate([lx, rx])
    agent_y       = np.concatenate([ly, ry])
    agent_heading = np.concatenate([lh, rh])
    return agent_x, agent_y, agent_heading


def reset(rng):
    trail = np.zeros((H, W), dtype=np.float32)
    agent_x, agent_y, agent_heading = make_agents(rng)
    return trail, agent_x, agent_y, agent_heading


# ── Sensor sampling ────────────────────────────────────────────────────────────

def sample_all(trail, x, y):
    xi = x.astype(int) % H
    yi = y.astype(int) % W
    return trail[xi, yi]


# ── Physarum step ──────────────────────────────────────────────────────────────

def step(trail, agent_x, agent_y, agent_heading, rng):
    s_left   = agent_heading - SENSOR_ANGLE
    s_center = agent_heading
    s_right  = agent_heading + SENSOR_ANGLE

    lx = agent_x + SENSOR_DIST * np.cos(s_left)
    ly = agent_y + SENSOR_DIST * np.sin(s_left)
    cx = agent_x + SENSOR_DIST * np.cos(s_center)
    cy = agent_y + SENSOR_DIST * np.sin(s_center)
    rx = agent_x + SENSOR_DIST * np.cos(s_right)
    ry = agent_y + SENSOR_DIST * np.sin(s_right)

    L = sample_all(trail, lx, ly)
    C = sample_all(trail, cx, cy)
    R = sample_all(trail, rx, ry)

    rotate = np.where(C >= np.maximum(L, R), 0.0,
             np.where(L >= R, -ROTATE_ANGLE, ROTATE_ANGLE))

    tie_mask = (L == R) & (C < L)
    if tie_mask.any():
        rotate[tie_mask] = rng.choice([-ROTATE_ANGLE, ROTATE_ANGLE], size=tie_mask.sum())

    agent_heading = (agent_heading + rotate) % (2 * np.pi)
    agent_x = (agent_x + np.cos(agent_heading)) % H
    agent_y = (agent_y + np.sin(agent_heading)) % W

    xi = agent_x.astype(int) % H
    yi = agent_y.astype(int) % W
    np.add.at(trail, (xi, yi), DEPOSIT)

    trail = uniform_filter(trail, size=2 * DIFFUSE_R + 1, mode='reflect')
    trail = np.clip(trail * DECAY, 0.0, 1.0)

    return trail, agent_x, agent_y, agent_heading


# ── Anchor injection ───────────────────────────────────────────────────────────

def inject_anchors(trail):
    """Re-inject anchor pheromone every step. Never fades."""
    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    for dx in range(-ANCHOR_RADIUS, ANCHOR_RADIUS + 1):
        for dy in range(-ANCHOR_RADIUS, ANCHOR_RADIUS + 1):
            if dx*dx + dy*dy <= ANCHOR_RADIUS*ANCHOR_RADIUS:
                trail[(ax+dx) % H, (ay+dy) % W] = ANCHOR_STRENGTH
                trail[(bx+dx) % H, (by+dy) % W] = ANCHOR_STRENGTH
    return trail


# ── Stability fix: agent reseeding ────────────────────────────────────────────

def reseed_agents(agent_x, agent_y, agent_heading, rng):
    """
    Detect agents that have drifted into the wrong zone or the gap.
    Teleport them back near their home anchor.

    Left agents (index 0..N/2-1) should stay in columns < GAP_LEFT.
    Right agents (index N/2..N-1) should stay in columns > GAP_RIGHT.
    """
    n_left  = N_AGENTS // 2
    ax, ay  = ANCHOR_A
    bx, by  = ANCHOR_B

    # Left agents that drifted right of gap
    left_mask = (agent_y[:n_left] >= GAP_LEFT - 5)
    if left_mask.any():
        n = left_mask.sum()
        agent_x[:n_left][left_mask] = np.clip(
            rng.normal(ax, ANCHOR_SPREAD, n), 5, H-5).astype(np.float32)
        agent_y[:n_left][left_mask] = np.clip(
            rng.normal(ay, ANCHOR_SPREAD, n), 5, GAP_LEFT-5).astype(np.float32)
        agent_heading[:n_left][left_mask] = rng.uniform(0, 2*np.pi, n).astype(np.float32)

    # Right agents that drifted left of gap
    right_mask = (agent_y[n_left:] <= GAP_RIGHT + 5)
    if right_mask.any():
        n = right_mask.sum()
        agent_x[n_left:][right_mask] = np.clip(
            rng.normal(bx, ANCHOR_SPREAD, n), 5, H-5).astype(np.float32)
        agent_y[n_left:][right_mask] = np.clip(
            rng.normal(by, ANCHOR_SPREAD, n), GAP_RIGHT+5, W-5).astype(np.float32)
        agent_heading[n_left:][right_mask] = rng.uniform(0, 2*np.pi, n).astype(np.float32)

    return agent_x, agent_y, agent_heading


# ── Bridge signal ──────────────────────────────────────────────────────────────

def fire_bridge_signal(trail):
    """Inject pheromone corridor from Zone A anchor to Zone B anchor."""
    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    trail[ax-4:ax+4, ay:by] = SIGNAL_STRENGTH
    trail[:, GAP_LEFT:GAP_RIGHT] = SIGNAL_STRENGTH * 0.5
    return trail


def reinforce_bridge(trail):
    """Keep corridor alive while bridge hold counter is active."""
    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    trail[ax-2:ax+2, ay:by] = SIGNAL_STRENGTH * 0.6
    return trail


# ── Zone state measurement ─────────────────────────────────────────────────────

def measure_zones(trail):
    zone_a = float(trail[:, :GAP_LEFT].mean())
    zone_b = float(trail[:, GAP_RIGHT:].mean())
    gap    = float(trail[:, GAP_LEFT:GAP_RIGHT].mean())
    return zone_a, zone_b, gap
