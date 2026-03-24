# ami/watcher.py
#
# Watches ami/input.txt for topic signals.
# Detects intent type (politics vs climate) and fires signal with type included.
#
# Signal types:
#   "politics" — AI in politics, policy, government, election
#                → horizontal bar injection → routes to Zone B → Claude
#   "climate"  — climate tech, energy, renewable, environment
#                → vertical bar injection   → routes to Zone C → Gemini
#
# Trigger phrases:
#   "researching X", "writing about X", "looking into X",
#   "studying X", "# Header", "find X", "tell me about X"
#
# Run:
#   python ami/watcher.py

import time
import json
import re
import uuid
from pathlib import Path
from datetime import datetime

INPUT_FILE   = Path("ami/input.txt")
TRIGGER_FILE = Path("ami/ami_trigger.json")

DEBOUNCE = 15   # seconds before same topic can re-trigger

TRIGGER_PATTERNS = [
    r"(?:researching|writing about|looking into|article on|notes on|studying)\s+(.+)",
    r"^#+\s+(.+)",
    r"(?:find|search for|get info on|tell me about)\s+(.+)",
]

# Keywords that determine signal type.
# Politics → Zone B → Claude
# Climate  → Zone C → Gemini
POLITICS_KEYWORDS = [
    "politics", "political", "policy", "government", "election",
    "democracy", "congress", "senate", "legislation", "geopolitics",
    "AI in politics", "ai policy", "regulation",
]
CLIMATE_KEYWORDS = [
    "climate", "climate tech", "renewable", "energy", "solar",
    "wind power", "carbon", "emissions", "sustainability",
    "environment", "green tech", "clean energy",
]


def classify_signal(topic):
    """
    Determine signal type from topic text.
    Returns "politics", "climate", or "politics" as default.
    """
    topic_lower = topic.lower()
    for kw in CLIMATE_KEYWORDS:
        if kw in topic_lower:
            return "climate"
    for kw in POLITICS_KEYWORDS:
        if kw in topic_lower:
            return "politics"
    # Default to politics if no match — extend keyword lists as needed
    return "politics"


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
    signal_id   = str(uuid.uuid4())
    signal_type = classify_signal(topic)

    trigger = {
        "signal_id":   signal_id,
        "signal_type": signal_type,
        "topic":       topic,
        "fired_at":    datetime.now().isoformat(),
        "consumed":    False,
    }
    TRIGGER_FILE.write_text(json.dumps(trigger, indent=2))

    zone = "B (Claude)" if signal_type == "politics" else "C (Gemini)"
    print(f"\n  [watcher] >>> TOPIC DETECTED: '{topic}'")
    print(f"  [watcher] >>> Type: {signal_type} → Zone {zone}")
    print(f"  [watcher] >>> Signal ID: {signal_id[:8]}... — NCA routing...")


def main():
    INPUT_FILE.touch()
    print("[watcher] Started — watching ami/input.txt")
    print("[watcher] Trigger phrases: 'researching X', 'writing about X', '# Title'")
    print("[watcher] Signal types:")
    print("   politics → Zone B → Claude  (AI, policy, government, election...)")
    print("   climate  → Zone C → Gemini  (climate, energy, renewable, carbon...)")
    print()

    last_content      = INPUT_FILE.read_text()
    last_signal_id    = None
    last_trigger_time = 0

    while True:
        try:
            current = INPUT_FILE.read_text()
            if current != last_content:
                last_content = current
                topic = extract_topic(current)
                if topic:
                    now = time.time()
                    if (now - last_trigger_time) > DEBOUNCE:
                        fire_signal(topic)
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
