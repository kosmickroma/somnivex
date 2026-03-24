# ami/physarum_core.py
#
# Shared Physarum simulation core — no pygame, no display.
# Used by experiment_zones.py (interactive) and generate_data.py (headless).
#
# Three zones:
#   Zone A (left)       — input/watcher
#   Zone B (top right)  — Claude output (AI in politics)
#   Zone C (bot right)  — Gemini output (climate tech)
#
# Two signal shapes:
#   HORIZONTAL bar — politics signal → should route to Zone B
#   VERTICAL bar   — climate signal  → should route to Zone C
#
# Zone identity channel re-injected every step so the NCA knows where it is.

import numpy as np
from scipy.ndimage import uniform_filter

# ── Parameters ────────────────────────────────────────────────────────────────

H, W              = 256, 256
N_AGENTS          = 4000        # more agents — now covering 3 zones
DECAY             = 0.95
DIFFUSE_R         = 1
DEPOSIT           = 1.5
SENSOR_DIST       = 9
SENSOR_ANGLE      = 0.4
ROTATE_ANGLE      = 0.3

GAP_WIDTH         = 30
SIGNAL_STRENGTH   = 0.8
BRIDGE_HOLD_STEPS = 200

# Separate thresholds for each output zone
ZONE_B_THRESHOLD  = 0.052      # Claude zone (top right)
ZONE_C_THRESHOLD  = 0.052      # Gemini zone (bottom right)

ANCHOR_STRENGTH   = 1.0
ANCHOR_RADIUS     = 8
ANCHOR_SPREAD     = 20

RESEED_INTERVAL   = 500

# ── Zone boundaries ────────────────────────────────────────────────────────────

CENTER    = W // 2
GAP_LEFT  = CENTER - GAP_WIDTH // 2
GAP_RIGHT = CENTER + GAP_WIDTH // 2

# Right side split top/bottom for Zone B and Zone C
RIGHT_MID = H // 2

# ── Anchor positions ───────────────────────────────────────────────────────────
#
#  ANCHOR_A — center of Zone A (left zone)
#  ANCHOR_B — center of Zone B (top right)
#  ANCHOR_C — center of Zone C (bottom right)

ANCHOR_A = (H // 2,          GAP_LEFT // 2)
ANCHOR_B = (H // 4,          GAP_RIGHT + (W - GAP_RIGHT) // 2)
ANCHOR_C = (3 * H // 4,      GAP_RIGHT + (W - GAP_RIGHT) // 2)

# ── Zone identity values (re-injected every step as control channel) ──────────
#
# The NCA reads this channel to know which zone it is in.
# This is the spatial context that makes routing possible.
# Works exactly like ch13 (physics bit) in the original NCA.

ZONE_ID_A   = 0.0    # input zone
ZONE_ID_GAP = 0.3    # dead zone
ZONE_ID_B   = 0.7    # Claude zone (top right)
ZONE_ID_C   = 1.0    # Gemini zone (bottom right)


def make_zone_identity_map():
    """
    Build the static zone identity map — same shape as trail (H, W).
    Re-injected into a dedicated channel every simulation step.
    """
    zone_map = np.zeros((H, W), dtype=np.float32)
    zone_map[:, :GAP_LEFT]   = ZONE_ID_A
    zone_map[:, GAP_LEFT:GAP_RIGHT] = ZONE_ID_GAP
    # Top right = Zone B (Claude)
    zone_map[:RIGHT_MID, GAP_RIGHT:] = ZONE_ID_B
    # Bottom right = Zone C (Gemini)
    zone_map[RIGHT_MID:, GAP_RIGHT:] = ZONE_ID_C
    return zone_map

ZONE_MAP = make_zone_identity_map()


# ── Agent initialization ───────────────────────────────────────────────────────

def make_agents(rng):
    """
    Initialize agents clustered near their home anchors.
    Split evenly across three zones.
    """
    n_a = N_AGENTS // 3
    n_b = N_AGENTS // 3
    n_c = N_AGENTS - n_a - n_b

    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    cx, cy = ANCHOR_C

    # Zone A agents
    ax_ = np.clip(rng.normal(ax, ANCHOR_SPREAD, n_a), 5, H-5).astype(np.float32)
    ay_ = np.clip(rng.normal(ay, ANCHOR_SPREAD, n_a), 5, GAP_LEFT-5).astype(np.float32)
    ah_ = rng.uniform(0, 2*np.pi, n_a).astype(np.float32)

    # Zone B agents (top right)
    bx_ = np.clip(rng.normal(bx, ANCHOR_SPREAD, n_b), 5, RIGHT_MID-5).astype(np.float32)
    by_ = np.clip(rng.normal(by, ANCHOR_SPREAD, n_b), GAP_RIGHT+5, W-5).astype(np.float32)
    bh_ = rng.uniform(0, 2*np.pi, n_b).astype(np.float32)

    # Zone C agents (bottom right)
    cx_ = np.clip(rng.normal(cx, ANCHOR_SPREAD, n_c), RIGHT_MID+5, H-5).astype(np.float32)
    cy_ = np.clip(rng.normal(cy, ANCHOR_SPREAD, n_c), GAP_RIGHT+5, W-5).astype(np.float32)
    ch_ = rng.uniform(0, 2*np.pi, n_c).astype(np.float32)

    agent_x       = np.concatenate([ax_, bx_, cx_])
    agent_y       = np.concatenate([ay_, by_, cy_])
    agent_heading = np.concatenate([ah_, bh_, ch_])

    # Store zone membership for reseeding
    zone_ids = np.array(['A']*n_a + ['B']*n_b + ['C']*n_c)

    return agent_x, agent_y, agent_heading, zone_ids


def reset(rng):
    trail = np.zeros((H, W), dtype=np.float32)
    agent_x, agent_y, agent_heading, zone_ids = make_agents(rng)
    return trail, agent_x, agent_y, agent_heading, zone_ids


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

    trail = uniform_filter(trail, size=2*DIFFUSE_R+1, mode='reflect')
    trail = np.clip(trail * DECAY, 0.0, 1.0)

    return trail, agent_x, agent_y, agent_heading


# ── Anchor injection ───────────────────────────────────────────────────────────

def inject_anchors(trail):
    """Re-inject anchor pheromone every step. Never fades."""
    for anchor in [ANCHOR_A, ANCHOR_B, ANCHOR_C]:
        ax, ay = anchor
        for dx in range(-ANCHOR_RADIUS, ANCHOR_RADIUS + 1):
            for dy in range(-ANCHOR_RADIUS, ANCHOR_RADIUS + 1):
                if dx*dx + dy*dy <= ANCHOR_RADIUS*ANCHOR_RADIUS:
                    trail[(ax+dx) % H, (ay+dy) % W] = ANCHOR_STRENGTH
    return trail


# ── Agent reseeding ────────────────────────────────────────────────────────────

def reseed_agents(agent_x, agent_y, agent_heading, zone_ids, rng):
    """
    Teleport agents that drifted out of their home zone back to their anchor.
    Uses zone_ids array to know where each agent belongs.
    """
    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    cx, cy = ANCHOR_C

    for i, zone in enumerate(zone_ids):
        if zone == 'A':
            if agent_y[i] >= GAP_LEFT - 5:
                agent_x[i] = float(np.clip(rng.normal(ax, ANCHOR_SPREAD), 5, H-5))
                agent_y[i] = float(np.clip(rng.normal(ay, ANCHOR_SPREAD), 5, GAP_LEFT-5))
                agent_heading[i] = float(rng.uniform(0, 2*np.pi))
        elif zone == 'B':
            if agent_y[i] <= GAP_RIGHT + 5 or agent_x[i] >= RIGHT_MID - 5:
                agent_x[i] = float(np.clip(rng.normal(bx, ANCHOR_SPREAD), 5, RIGHT_MID-5))
                agent_y[i] = float(np.clip(rng.normal(by, ANCHOR_SPREAD), GAP_RIGHT+5, W-5))
                agent_heading[i] = float(rng.uniform(0, 2*np.pi))
        elif zone == 'C':
            if agent_y[i] <= GAP_RIGHT + 5 or agent_x[i] <= RIGHT_MID + 5:
                agent_x[i] = float(np.clip(rng.normal(cx, ANCHOR_SPREAD), RIGHT_MID+5, H-5))
                agent_y[i] = float(np.clip(rng.normal(cy, ANCHOR_SPREAD), GAP_RIGHT+5, W-5))
                agent_heading[i] = float(rng.uniform(0, 2*np.pi))

    return agent_x, agent_y, agent_heading


# ── Signal injection ───────────────────────────────────────────────────────────

def inject_signal_politics(trail):
    """
    HORIZONTAL bar — 'AI in politics' signal → routes to Zone B (top right / Claude).

    Injects pheromone across the full Zone B (top right).
    Does NOT inject into Zone C (bottom right).
    Zone A shape: horizontal bar at anchor row — what NCA training will use.
    """
    ax, ay = ANCHOR_A
    # Horizontal bar across Zone A
    trail[ax-4:ax+4, :GAP_LEFT]             = SIGNAL_STRENGTH
    # Bridge through gap at same row — targets top right
    trail[ax-4:ax+4, GAP_LEFT:GAP_RIGHT]    = SIGNAL_STRENGTH * 0.7
    # Flood Zone B (top right) — NOT Zone C
    trail[:RIGHT_MID, GAP_RIGHT:]           = SIGNAL_STRENGTH * 0.6
    return trail


def inject_signal_climate(trail):
    """
    VERTICAL bar — 'climate tech' signal → routes to Zone C (bottom right / Gemini).

    Injects pheromone across the full Zone C (bottom right).
    Does NOT inject into Zone B (top right).
    Zone A shape: vertical bar downward from anchor — what NCA training will use.
    """
    ax, ay = ANCHOR_A
    # Vertical bar down Zone A from anchor
    trail[ax:H,  ay-4:ay+4]                 = SIGNAL_STRENGTH
    trail[:ax,   ay-4:ay+4]                 = SIGNAL_STRENGTH * 0.2
    # Bridge through gap going downward — targets bottom right
    trail[ax:H,  GAP_LEFT:GAP_RIGHT]        = SIGNAL_STRENGTH * 0.7
    # Flood Zone C (bottom right) — NOT Zone B
    trail[RIGHT_MID:, GAP_RIGHT:]           = SIGNAL_STRENGTH * 0.6
    return trail


def reinforce_signal_politics(trail):
    """Keep politics corridor alive during hold window."""
    ax, ay = ANCHOR_A
    trail[ax-2:ax+2, :GAP_RIGHT]    = SIGNAL_STRENGTH * 0.5
    trail[:RIGHT_MID, GAP_RIGHT:]   = SIGNAL_STRENGTH * 0.4
    return trail


def reinforce_signal_climate(trail):
    """Keep climate corridor alive during hold window."""
    ax, ay = ANCHOR_A
    trail[ax:H, ay-2:ay+2]          = SIGNAL_STRENGTH * 0.5
    trail[RIGHT_MID:, GAP_RIGHT:]   = SIGNAL_STRENGTH * 0.4
    return trail


# ── Zone state measurement ─────────────────────────────────────────────────────

def measure_zones(trail):
    zone_a = float(trail[:, :GAP_LEFT].mean())
    zone_b = float(trail[:RIGHT_MID, GAP_RIGHT:].mean())   # top right
    zone_c = float(trail[RIGHT_MID:, GAP_RIGHT:].mean())   # bottom right
    gap    = float(trail[:, GAP_LEFT:GAP_RIGHT].mean())
    return zone_a, zone_b, zone_c, gap
