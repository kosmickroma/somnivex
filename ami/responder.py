# ami/responder.py
#
# Monitors ami/ami_trigger.json.
# When a new topic signal arrives, calls Claude, writes results.
#
# Run:
#   python ami/responder.py

import time
import json
import anthropic
from pathlib import Path
from datetime import datetime

TRIGGER_FILE = Path("ami/ami_trigger.json")
ZONE_STATE   = Path("ami/zone_state.json")
RESULTS_FILE = Path("ami/results.txt")


def call_claude(topic):
    client = anthropic.Anthropic()
    print(f"  [responder] Calling Claude for: '{topic}'")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"The user is researching: {topic}\n\n"
                f"Give them:\n"
                f"1. A one-sentence summary of where this topic stands right now\n"
                f"2. 3-4 key developments or angles worth knowing\n"
                f"3. 2-3 specific things worth searching for\n\n"
                f"Be concise and useful. No fluff."
            )
        }]
    )
    return response.content[0].text


def write_results(topic, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider   = "=" * 60
    result    = f"\n{divider}\n"
    result   += f"TOPIC : {topic}\n"
    result   += f"TIME  : {timestamp}\n"
    result   += f"{divider}\n"
    result   += content
    result   += f"\n{divider}\n"

    with open(RESULTS_FILE, 'a') as f:
        f.write(result)

    print(f"\n{result}")
    print(f"  [responder] Written to {RESULTS_FILE}")


def main():
    RESULTS_FILE.touch()
    print("[responder] Started — watching NCA zone B for activation")
    print(f"[responder] Results will appear in ami/results.txt")
    print()

    last_topic = None

    while True:
        try:
            # Need both files to exist
            if not TRIGGER_FILE.exists() or not ZONE_STATE.exists():
                time.sleep(0.5)
                continue

            trigger = json.loads(TRIGGER_FILE.read_text())
            state   = json.loads(ZONE_STATE.read_text())

            # Skip if consumed
            if trigger.get("consumed"):
                time.sleep(0.5)
                continue

            topic = trigger.get("topic")
            if not topic or topic == last_topic:
                time.sleep(0.5)
                continue

            # Wait for NCA to actually route — zone B must activate
            zone_b_activated = state.get("zone_b_activated", False)
            zone_b_val       = state.get("zone_b", 0)

            if not zone_b_activated:
                time.sleep(0.5)
                continue

            # Zone B activated — NCA routed the signal — now call LLM
            print(f"\n  [responder] NCA routed signal — Zone B={zone_b_val:.3f}")
            print(f"  [responder] Topic: '{topic}'")

            trigger["consumed"] = True
            TRIGGER_FILE.write_text(json.dumps(trigger, indent=2))
            last_topic = topic

            result = call_claude(topic)
            write_results(topic, result)

        except KeyboardInterrupt:
            print("\n[responder] stopped")
            break
        except Exception as e:
            print(f"[responder] error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
