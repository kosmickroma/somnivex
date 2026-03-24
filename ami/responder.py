# ami/responder.py
#
# Monitors zone_state.json for NCA routing events.
#
# Zone B activated (politics signal routed) → calls Claude
# Zone C activated (climate signal routed)  → calls Gemini
#
# Claude fires ONLY after signal_id in zone_state matches trigger signal_id.
# This proves the signal physically traveled through the NCA substrate.
#
# Run:
#   python ami/responder.py

import time
import json
import anthropic
from google import genai as genai_new
import os
from pathlib import Path
from datetime import datetime

TRIGGER_FILE  = Path("ami/ami_trigger.json")
ZONE_STATE    = Path("ami/zone_state.json")
RESULTS_FILE  = Path("ami/results.txt")


# ── API clients ───────────────────────────────────────────────────────────────

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


def call_gemini(topic):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "[Gemini] No API key found — set GEMINI_API_KEY or GOOGLE_API_KEY"
    client = genai_new.Client(api_key=api_key)
    print(f"  [responder] Calling Gemini for: '{topic}'")
    prompt = (
        f"The user is researching: {topic}\n\n"
        f"Give them:\n"
        f"1. A one-sentence summary of where this topic stands right now\n"
        f"2. 3-4 key developments or angles worth knowing\n"
        f"3. 2-3 specific things worth searching for\n\n"
        f"Be concise and useful. No fluff."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text


# ── Result writing ────────────────────────────────────────────────────────────

def write_results(topic, api_name, zone, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider   = "=" * 60
    result    = f"\n{divider}\n"
    result   += f"TOPIC : {topic}\n"
    result   += f"API   : {api_name} (routed to Zone {zone})\n"
    result   += f"TIME  : {timestamp}\n"
    result   += f"{divider}\n"
    result   += content
    result   += f"\n{divider}\n"

    with open(RESULTS_FILE, 'a') as f:
        f.write(result)

    print(f"\n{result}")
    print(f"  [responder] Written to {RESULTS_FILE}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    RESULTS_FILE.touch()
    print("[responder] Started — watching NCA zone activations")
    print("[responder] Zone B → Claude  |  Zone C → Gemini")
    print(f"[responder] Results → {RESULTS_FILE}")
    print()

    last_signal_id = None

    while True:
        try:
            if not TRIGGER_FILE.exists() or not ZONE_STATE.exists():
                time.sleep(0.5)
                continue

            trigger = json.loads(TRIGGER_FILE.read_text())
            state   = json.loads(ZONE_STATE.read_text())

            if trigger.get("consumed"):
                time.sleep(0.5)
                continue

            topic       = trigger.get("topic")
            signal_id   = trigger.get("signal_id")
            signal_type = trigger.get("signal_type", "politics")

            if not topic or not signal_id:
                time.sleep(0.5)
                continue

            if signal_id == last_signal_id:
                time.sleep(0.5)
                continue

            # Check which zone activated and whether IDs match
            zone_b_activated = state.get("zone_b_activated", False)
            zone_c_activated = state.get("zone_c_activated", False)
            zone_state_id    = state.get("signal_id")
            zone_state_type  = state.get("signal_type")

            neither_activated = not zone_b_activated and not zone_c_activated
            if neither_activated:
                time.sleep(0.5)
                continue

            # Signal ID must match — proves this activation is from THIS signal
            if zone_state_id != signal_id:
                print(f"  [responder] Waiting — ID mismatch "
                      f"(state={str(zone_state_id)[:8] if zone_state_id else 'None'} "
                      f"want={str(signal_id)[:8]})")
                time.sleep(0.5)
                continue

            # Mark consumed before calling API
            trigger["consumed"] = True
            TRIGGER_FILE.write_text(json.dumps(trigger, indent=2))
            last_signal_id = signal_id

            # Route to correct API based on which zone activated
            if zone_b_activated:
                zone_b_val = state.get("zone_b", 0)
                print(f"\n  [responder] Zone B activated ({zone_b_val:.3f}) — "
                      f"signal {str(signal_id)[:8]}... → Claude")
                result = call_claude(topic)
                write_results(topic, "Claude", "B", result)

            elif zone_c_activated:
                zone_c_val = state.get("zone_c", 0)
                print(f"\n  [responder] Zone C activated ({zone_c_val:.3f}) — "
                      f"signal {str(signal_id)[:8]}... → Gemini")
                result = call_gemini(topic)
                write_results(topic, "Gemini", "C", result)

        except KeyboardInterrupt:
            print("\n[responder] stopped")
            break
        except Exception as e:
            print(f"[responder] error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
