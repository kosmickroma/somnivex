# ami/watcher.py
#
# Watches ami/input.txt for topic signals.
# When you type a trigger phrase, extracts the topic and fires a signal.
#
# Trigger phrases:
#   "researching AI in global politics"
#   "writing about climate tech"
#   "# Article Title"
#   "looking into quantum computing"
#
# Run:
#   python ami/watcher.py

import time
import json
import re
from pathlib import Path
from datetime import datetime

INPUT_FILE   = Path("ami/input.txt")
TRIGGER_FILE = Path("ami/ami_trigger.json")

# How long to wait before re-triggering on same topic (seconds)
DEBOUNCE     = 15

TRIGGER_PATTERNS = [
    r"(?:researching|writing about|looking into|article on|notes on|studying)\s+(.+)",
    r"^#+\s+(.+)",
    r"(?:find|search for|get info on|tell me about)\s+(.+)",
]


def extract_topic(text):
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):
        line = line.strip()
        if not line:
            continue
        for pattern in TRIGGER_PATTERNS:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip('.,!?')
    return None


def fire_signal(topic):
    trigger = {
        "topic":    topic,
        "fired_at": datetime.now().isoformat(),
        "consumed": False
    }
    TRIGGER_FILE.write_text(json.dumps(trigger, indent=2))
    print(f"\n  [watcher] >>> TOPIC DETECTED: '{topic}'")
    print(f"  [watcher] >>> Signal fired — NCA routing...")


def main():
    INPUT_FILE.touch()
    print("[watcher] Started — watching ami/input.txt")
    print("[watcher] Type something like:")
    print("   'researching AI in global politics'")
    print("   '# My Article Title'")
    print("   'writing about climate and tech'")
    print()

    last_content      = INPUT_FILE.read_text()
    last_topic        = None
    last_trigger_time = 0

    while True:
        try:
            current = INPUT_FILE.read_text()
            if current != last_content:
                last_content = current
                topic = extract_topic(current)
                if topic:
                    now = time.time()
                    if topic != last_topic or (now - last_trigger_time) > DEBOUNCE:
                        fire_signal(topic)
                        last_topic        = topic
                        last_trigger_time = now
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[watcher] stopped")
            break
        except Exception as e:
            print(f"[watcher] error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
