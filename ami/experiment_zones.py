# ami/experiment_zones.py
#
# EXPERIMENT: Zone isolation + signal-triggered bridging
#
# Two populations of Physarum agents — left zone and right zone.
# A gap runs down the center. No agents start in the gap.
#
# QUESTION 1: Do they stay isolated without a signal?
# QUESTION 2: Press SPACE to fire a pheromone pulse across the gap.
#             Do agents bridge toward it?
#
# Controls:
#   SPACE  — fire a bridge signal (injects pheromone strip across the gap)
#   R      — reset everything
#   Q      — quit
#
# Run:
#   source ~/ai-env/bin/activate
#   python ami/experiment_zones.py

import numpy as np
import pygame
import json
from pathlib import Path
from scipy.ndimage import uniform_filter

TRIGGER_FILE  = Path("ami/ami_trigger.json")
ZONE_STATE_FILE = Path("ami/zone_state.json")


# ── Parameters ────────────────────────────────────────────────────────────────

H, W          = 256, 256
N_AGENTS      = 3000        # total agents — split evenly left/right
DECAY         = 0.95        # trail evaporation rate
DIFFUSE_R     = 1           # blur radius
DEPOSIT       = 1.5         # trail deposited per agent per step
SENSOR_DIST   = 9           # how far agents sense
SENSOR_ANGLE  = 0.4         # sensor spread angle (radians)
ROTATE_ANGLE  = 0.3         # rotation per step (radians)

GAP_WIDTH     = 30          # width of the dead zone in the center (pixels)
SIGNAL_STRENGTH   = 0.8     # pheromone injected when SPACE is pressed
BRIDGE_HOLD_STEPS = 200     # how many steps to keep reinforcing the bridge
ZONE_B_THRESHOLD  = 0.052   # zone B activation level that triggers the LLM
                            # baseline ~0.035, bridge response ~0.060
SCALE         = 3           # display scale (256 * 3 = 768px window)
FPS           = 60


# ── Zone boundaries ────────────────────────────────────────────────────────────

CENTER        = W // 2
GAP_LEFT      = CENTER - GAP_WIDTH // 2    # left edge of gap
GAP_RIGHT     = CENTER + GAP_WIDTH // 2    # right edge of gap

# Left zone:  columns 0 to GAP_LEFT
# Gap:        columns GAP_LEFT to GAP_RIGHT  (no agents here)
# Right zone: columns GAP_RIGHT to W


# ── Initialize ────────────────────────────────────────────────────────────────

def make_agents(rng):
    """
    Start all agents clustered near the anchor points.
    Forces them to build outward from the food sources — no random trails elsewhere.
    """
    n_left  = N_AGENTS // 2
    n_right = N_AGENTS - n_left

    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    SPREAD = 20     # how tightly clustered around the anchor

    # Left agents — clustered near anchor A
    lx = np.clip(rng.normal(ax, SPREAD, n_left), 5, H-5).astype(np.float32)
    ly = np.clip(rng.normal(ay, SPREAD, n_left), 5, GAP_LEFT-5).astype(np.float32)
    lh = rng.uniform(0, 2 * np.pi, n_left).astype(np.float32)

    # Right agents — clustered near anchor B
    rx = np.clip(rng.normal(bx, SPREAD, n_right), 5, H-5).astype(np.float32)
    ry = np.clip(rng.normal(by, SPREAD, n_right), GAP_RIGHT+5, W-5).astype(np.float32)
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


# ── Step ──────────────────────────────────────────────────────────────────────

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


# ── Anchor points ─────────────────────────────────────────────────────────────
# Fixed high-pheromone sources — like food in real Physarum biology.
# Agents cluster here. Trails form between them.
# Re-injected every step so they never fade.

ANCHOR_STRENGTH = 1.0
ANCHOR_RADIUS   = 8     # pixels

# Zone A anchor — center of left zone
ANCHOR_A = (H // 2, GAP_LEFT // 2)

# Zone B anchor — center of right zone
ANCHOR_B = (H // 2, GAP_RIGHT + (W - GAP_RIGHT) // 2)


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


# ── Signal injection ───────────────────────────────────────────────────────────

def fire_bridge_signal(trail):
    """
    Inject a pheromone corridor from Zone A anchor across the gap to Zone B anchor.
    Runs along the anchor row so agents on both sides can actually sense it.
    """
    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    # Full corridor at anchor row height — Zone A edge to Zone B edge
    trail[ax-4:ax+4, ay:by] = SIGNAL_STRENGTH
    # Also flood the full gap so it's visually clear
    trail[:, GAP_LEFT:GAP_RIGHT] = SIGNAL_STRENGTH * 0.5
    return trail


def reinforce_bridge(trail):
    """Keep the corridor alive so Zone B agents have time to respond."""
    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    trail[ax-2:ax+2, ay:by] = SIGNAL_STRENGTH * 0.6
    return trail


# ── Render ────────────────────────────────────────────────────────────────────

def render(screen, trail, step_count, font, bridge_fired):
    # Trail as grayscale
    display_arr = (np.clip(trail, 0, 1) * 255).astype(np.uint8)
    display_arr = np.repeat(np.repeat(display_arr, SCALE, axis=0), SCALE, axis=1)
    rgb = np.stack([display_arr, display_arr, display_arr], axis=-1)

    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))

    # Draw anchor points
    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    pygame.draw.circle(screen, (0, 255, 100),
                       (ay * SCALE, ax * SCALE), ANCHOR_RADIUS * SCALE, 2)
    pygame.draw.circle(screen, (0, 100, 255),
                       (by * SCALE, bx * SCALE), ANCHOR_RADIUS * SCALE, 2)

    # Draw zone boundaries
    pygame.draw.line(screen, (255, 0, 0),
                     (GAP_LEFT * SCALE, 0), (GAP_LEFT * SCALE, H * SCALE), 1)
    pygame.draw.line(screen, (255, 0, 0),
                     (GAP_RIGHT * SCALE, 0), (GAP_RIGHT * SCALE, H * SCALE), 1)

    # Zone labels
    left_label  = font.render("ZONE A", True, (255, 100, 100))
    right_label = font.render("ZONE B", True, (100, 100, 255))
    gap_label   = font.render("GAP", True, (255, 255, 0))

    screen.blit(left_label,  (10, 10))
    screen.blit(right_label, (GAP_RIGHT * SCALE + 10, 10))
    screen.blit(gap_label,   (GAP_LEFT * SCALE + 5, 10))

    # Status
    status = "BRIDGE SIGNAL FIRED" if bridge_fired else "SPACE=fire signal  R=reset  Q=quit"
    color  = (0, 255, 0) if bridge_fired else (200, 200, 200)
    txt = font.render(
        f"step={step_count}  max={trail.max():.3f}  mean={trail.mean():.4f}  |  {status}",
        True, color
    )
    screen.blit(txt, (5, H * SCALE - 20))

    # Gap mean (tells us if signal is bleeding across)
    gap_mean = trail[:, GAP_LEFT:GAP_RIGHT].mean()
    gap_txt  = font.render(f"gap trail={gap_mean:.4f}", True, (255, 255, 0))
    screen.blit(gap_txt, (GAP_LEFT * SCALE - 60, 30))

    pygame.display.flip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W * SCALE, H * SCALE))
    pygame.display.set_caption("AmI Zone Experiment — Physarum Bridging Test")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 13)

    rng = np.random.default_rng(42)
    trail, agent_x, agent_y, agent_heading = reset(rng)

    step_count        = 0
    bridge_fired      = False
    bridge_hold_steps = 0   # counts down after bridge fires
    llm_triggered     = False
    running           = True

    print("Zone experiment running.")
    print(f"  Grid: {H}x{W}  |  Gap: cols {GAP_LEFT}-{GAP_RIGHT}  |  Agents: {N_AGENTS}")
    print("  SPACE = fire bridge signal across the gap")
    print("  R     = reset")
    print("  Q     = quit")
    print()
    print("  Watching for: does trail bleed across the gap WITHOUT a signal?")
    print("  Then:         does a signal trigger bridging FROM both sides?")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    trail, agent_x, agent_y, agent_heading = reset(rng)
                    step_count   = 0
                    bridge_fired = False
                    print("  --- RESET ---")
                if event.key == pygame.K_SPACE:
                    trail        = fire_bridge_signal(trail)
                    bridge_fired = True
                    print(f"  step {step_count}: BRIDGE SIGNAL FIRED — watching for crossing...")

        trail, agent_x, agent_y, agent_heading = step(
            trail, agent_x, agent_y, agent_heading, rng
        )
        trail = inject_anchors(trail)   # re-inject every step — never fades
        step_count += 1

        # Check for incoming signal from watcher.py
        if TRIGGER_FILE.exists():
            try:
                trigger = json.loads(TRIGGER_FILE.read_text())
                if not trigger.get("consumed") and not bridge_fired:
                    topic = trigger.get("topic", "unknown")
                    print(f"  [nca] Signal received: '{topic}' — firing bridge")
                    trail             = fire_bridge_signal(trail)
                    bridge_fired      = True
                    bridge_hold_steps = BRIDGE_HOLD_STEPS
                    llm_triggered     = False
            except Exception:
                pass

        # Keep bridge reinforced while hold counter is active
        if bridge_hold_steps > 0:
            trail = reinforce_bridge(trail)
            bridge_hold_steps -= 1

        # Write zone state every 50 steps
        if step_count % 50 == 0:
            zone_a_mean = float(trail[:, :GAP_LEFT].mean())
            zone_b_mean = float(trail[:, GAP_RIGHT:].mean())
            gap_mean    = float(trail[:, GAP_LEFT:GAP_RIGHT].mean())

            # Zone B threshold crossed — signal responder to fire LLM
            zone_b_activated = bridge_fired and zone_b_mean >= ZONE_B_THRESHOLD

            if zone_b_activated and not llm_triggered:
                print(f"  [nca] Zone B activated ({zone_b_mean:.3f}) — LLM trigger ready")
                llm_triggered = True

            state = {
                "step":             step_count,
                "zone_a":           zone_a_mean,
                "zone_b":           zone_b_mean,
                "gap":              gap_mean,
                "bridge_fired":     bridge_fired,
                "zone_b_activated": zone_b_activated,
                "threshold":        ZONE_B_THRESHOLD
            }
            ZONE_STATE_FILE.write_text(json.dumps(state, indent=2))

        # Log gap activity every 100 steps
        if step_count % 100 == 0:
            gap_mean = trail[:, GAP_LEFT:GAP_RIGHT].mean()
            left_mean  = trail[:, :GAP_LEFT].mean()
            right_mean = trail[:, GAP_RIGHT:].mean()
            print(f"  step {step_count:5d}  left={left_mean:.4f}  gap={gap_mean:.4f}  right={right_mean:.4f}")

        render(screen, trail, step_count, font, bridge_fired)
        clock.tick(FPS)

    pygame.quit()
    print(f"\nStopped at step {step_count}")
    print("\nWhat to look for in the results:")
    print("  PRE-SIGNAL:  gap trail should stay near 0 — zones isolated")
    print("  POST-SIGNAL: gap trail rises, then agents from both sides move toward it")
    print("  BRIDGING:    trail builds from both sides into the gap — connection forms")
    print("  NO BRIDGING: signal fades, agents ignore it — need different approach")


if __name__ == "__main__":
    main()
