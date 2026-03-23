# ami/experiment_zones.py
#
# EXPERIMENT: Zone isolation + signal-triggered bridging
# Interactive pygame visualizer — uses physarum_core.py for all physics.
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
from physarum_core import (
    reset, step, inject_anchors, reseed_agents,
    fire_bridge_signal, reinforce_bridge, measure_zones,
    H, W, GAP_LEFT, GAP_RIGHT, ANCHOR_A, ANCHOR_B,
    ANCHOR_RADIUS, BRIDGE_HOLD_STEPS, ZONE_B_THRESHOLD, RESEED_INTERVAL,
)

TRIGGER_FILE    = Path("ami/ami_trigger.json")
ZONE_STATE_FILE = Path("ami/zone_state.json")

SCALE = 3
FPS   = 60


# ── Render ────────────────────────────────────────────────────────────────────

def render(screen, trail, step_count, font, bridge_fired):
    display_arr = (np.clip(trail, 0, 1) * 255).astype(np.uint8)
    display_arr = np.repeat(np.repeat(display_arr, SCALE, axis=0), SCALE, axis=1)
    rgb = np.stack([display_arr, display_arr, display_arr], axis=-1)

    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    screen.blit(surf, (0, 0))

    ax, ay = ANCHOR_A
    bx, by = ANCHOR_B
    pygame.draw.circle(screen, (0, 255, 100),
                       (ay * SCALE, ax * SCALE), ANCHOR_RADIUS * SCALE, 2)
    pygame.draw.circle(screen, (0, 100, 255),
                       (by * SCALE, bx * SCALE), ANCHOR_RADIUS * SCALE, 2)

    pygame.draw.line(screen, (255, 0, 0),
                     (GAP_LEFT * SCALE, 0), (GAP_LEFT * SCALE, H * SCALE), 1)
    pygame.draw.line(screen, (255, 0, 0),
                     (GAP_RIGHT * SCALE, 0), (GAP_RIGHT * SCALE, H * SCALE), 1)

    screen.blit(font.render("ZONE A", True, (255, 100, 100)), (10, 10))
    screen.blit(font.render("ZONE B", True, (100, 100, 255)),
                (GAP_RIGHT * SCALE + 10, 10))
    screen.blit(font.render("GAP",    True, (255, 255, 0)),
                (GAP_LEFT * SCALE + 5, 10))

    status = "BRIDGE SIGNAL FIRED" if bridge_fired else "SPACE=fire signal  R=reset  Q=quit"
    color  = (0, 255, 0) if bridge_fired else (200, 200, 200)
    zone_a, zone_b, gap = measure_zones(trail)
    txt = font.render(
        f"step={step_count}  max={trail.max():.3f}  mean={trail.mean():.4f}  |  {status}",
        True, color
    )
    screen.blit(txt, (5, H * SCALE - 20))

    gap_txt = font.render(f"gap trail={gap:.4f}", True, (255, 255, 0))
    screen.blit(gap_txt, (GAP_LEFT * SCALE - 60, 30))

    pygame.display.flip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W * SCALE, H * SCALE))
    pygame.display.set_caption("AmI Zone Experiment — Physarum Bridging Test")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 13)

    # Clear stale state from previous runs — write clean slate with no signal_id
    ZONE_STATE_FILE.write_text(json.dumps({
        "zone_b_activated": False,
        "signal_id":        None,
        "step":             0
    }))
    if TRIGGER_FILE.exists():
        try:
            t = json.loads(TRIGGER_FILE.read_text())
            t["consumed"] = True
            TRIGGER_FILE.write_text(json.dumps(t, indent=2))
        except Exception:
            pass

    rng = np.random.default_rng(42)
    trail, agent_x, agent_y, agent_heading = reset(rng)

    step_count        = 0
    bridge_fired      = False
    bridge_hold_steps = 0
    llm_triggered     = False
    active_signal_id  = None   # ID of the signal currently traveling through the NCA
    running           = True

    print("Zone experiment running.")
    print(f"  Grid: {H}x{W}  |  Gap: cols {GAP_LEFT}-{GAP_RIGHT}")
    print("  SPACE = fire bridge signal | R = reset | Q = quit")

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
                    trail             = fire_bridge_signal(trail)
                    bridge_fired      = True
                    bridge_hold_steps = BRIDGE_HOLD_STEPS
                    print(f"  step {step_count}: BRIDGE SIGNAL FIRED")

        trail, agent_x, agent_y, agent_heading = step(
            trail, agent_x, agent_y, agent_heading, rng
        )
        trail = inject_anchors(trail)
        step_count += 1

        # Stability: reseed drifted agents
        if step_count % RESEED_INTERVAL == 0:
            agent_x, agent_y, agent_heading = reseed_agents(
                agent_x, agent_y, agent_heading, rng
            )

        # Check for incoming signal from watcher.py
        if TRIGGER_FILE.exists():
            try:
                trigger = json.loads(TRIGGER_FILE.read_text())
                if not trigger.get("consumed") and not bridge_fired:
                    topic             = trigger.get("topic", "unknown")
                    active_signal_id  = trigger.get("signal_id")
                    print(f"  [nca] Signal received: '{topic}' (id={str(active_signal_id)[:8]}...) — firing bridge")
                    trail             = fire_bridge_signal(trail)
                    bridge_fired      = True
                    bridge_hold_steps = BRIDGE_HOLD_STEPS
                    llm_triggered     = False
            except Exception:
                pass

        # Reinforce bridge while hold active
        if bridge_hold_steps > 0:
            trail = reinforce_bridge(trail)
            bridge_hold_steps -= 1

        # Write zone state every 50 steps
        if step_count % 50 == 0:
            zone_a, zone_b, gap = measure_zones(trail)

            # Zone B is only considered activated if:
            # 1. A bridge was fired for a specific signal
            # 2. Zone B has physically crossed the threshold AFTER that signal arrived
            # 3. We are still within the hold window (signal is still traveling)
            zone_b_activated = (
                bridge_fired
                and active_signal_id is not None
                and zone_b >= ZONE_B_THRESHOLD
            )

            if zone_b_activated and not llm_triggered:
                print(f"  [nca] Zone B activated ({zone_b:.3f}) — signal {str(active_signal_id)[:8]}... reached Zone B")
                llm_triggered = True

            state = {
                "step":             step_count,
                "zone_a":           zone_a,
                "zone_b":           zone_b,
                "gap":              gap,
                "bridge_fired":     bridge_fired,
                "zone_b_activated": zone_b_activated,
                # signal_id is only written when Zone B is genuinely activated by this signal
                # responder MUST match this ID to the trigger file before calling Claude
                "signal_id":        active_signal_id if zone_b_activated else None,
                "threshold":        ZONE_B_THRESHOLD,
            }
            ZONE_STATE_FILE.write_text(json.dumps(state, indent=2))

        # Log every 100 steps
        if step_count % 100 == 0:
            zone_a, zone_b, gap = measure_zones(trail)
            print(f"  step {step_count:5d}  left={zone_a:.4f}  gap={gap:.4f}  right={zone_b:.4f}")

        render(screen, trail, step_count, font, bridge_fired)
        clock.tick(FPS)

    pygame.quit()
    print(f"\nStopped at step {step_count}")


if __name__ == "__main__":
    main()
