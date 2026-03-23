# AmI — Brainstorm Log
*Raw ideas, conversations, connections. Don't clean this up.*

---

## 2026-03-23 — Origin conversation

**The coffee shop story:** User's brother switches between research mode and writing mode constantly. Has to manually manage all context switching. The system never sees what he's doing. Question: what if it could?

**Why LLMs are wrong for this:** They wait. They're invoked. They have no persistent presence. Stateless by design. You have to hand them everything. Miss one edge case and it breaks.

**The NCA instinct:** Different sections of the grid doing different things simultaneously. One section watching a notepad. Another hooked to scrapers. Another watching calendar. Signals propagate without central coordination.

---

## The Grammar Connection

From the grammar experiment (notes/grammar_experiment_results.md):

> "The grammar doesn't emerge from multi-physics per se — it emerges when the task cannot be solved by visible channels alone."

Key insight for AmI: if you train a routing NCA on two incompatible signal types, the visible channels won't be sufficient. The NCA will develop hidden channel abstractions to hold the tension. Those abstractions ARE the routing tokens. You don't design them. You let them emerge and then read them with k-means clustering — exactly the same method used for the behavioral grammar.

**The NCA creates its own tokens.** We already proved this with the physics grammar. Same principle applied to routing signals.

---

## Architecture Discussion

Grok's suggested layered approach:
1. Sensory layer (file watcher / screen OCR) → injects into ch7/ch8
2. Tiny routing head (~500-1k params) reads sensory signals, detects intent, propagates action signals
3. Action layer reads action channels, calls scrapers
4. LLM optional on top

Our pushback: the routing head is a classifier bolted on top, not the NCA doing the routing. Fine for Phase 0. Not the deep version.

The deep version: train a new NCA where routing IS the physics. Hidden channels emerge under signal tension = routing grammar. That's the research question for Phase 2.

---

## What The NCA Is Actually Good At (that LLMs cannot do)

- Massively parallel spatial computation (65,536 cells simultaneously)
- Persistent spatial memory (trail_mask holds indefinitely)
- Self-repair without being asked
- Always-on operation (runs whether anything talks to it or not)
- Network/path optimization (Physarum specialty — near optimal path finding)
- Emergent signal routing without central coordinator

**Summary:** Not smarter than an LLM. Completely different kind of computation. Analog, parallel, spatial, persistent, always-on. They cover each other's blind spots exactly.

---

## Naming

"Ambient Intelligence" — abbreviated AI, which is a problem.
"AmI" floated as alternative. Tabled. Build it first, name it when it exists.

---

## Self-Repair As Fault Tolerance (key differentiator)

Normal automation pipeline: connection breaks → it stops → you get an error → you fix it manually. No memory of what it was trying to do.

The NCA is different. The pheromone trail between two zones IS the connection. If an API key breaks, scraper goes down, signal gets interrupted — the trail doesn't disappear. The grid still knows that connection is supposed to exist. It keeps trying to re-establish it.

That's not replicable with explicit automation. The memory is baked into the physical state of the grid.

The deeper version: if routing grammar emerges from training, the NCA might not just reconnect the same path — it might find an alternative route. Different scraper. Cached result. Something that satisfies the same intent through a different path.

This is exactly what Physarum does in biology. Block one path — it doesn't stop. It reroutes around the obstacle. That behavior is already partially in the current NCA from Physarum training.

**This is the key property that separates AmI from every other automation system. Not just persistence. Active fault tolerance through physical memory.**

---

## 2026-03-23 — POC BUILT AND WORKING

### What we built today

Three scripts wired together into a working ambient intelligence loop:

- `ami/watcher.py` — watches `ami/input.txt` for trigger phrases (regex patterns: "researching X", "writing about X", "# Header", etc.). On detection writes `ami/ami_trigger.json` with topic + consumed=false.
- `ami/experiment_zones.py` — Physarum simulation running visually. Polls `ami/ami_trigger.json` every step. When unconsumed signal found, fires bridge signal across the gap (visual routing event). Writes `ami/zone_state.json` every 50 steps.
- `ami/responder.py` — monitors `ami/ami_trigger.json`. When unconsumed signal found, marks consumed, waits 3 seconds (visual delay for NCA bridge), calls Claude Haiku, writes structured results to `ami/results.txt` and terminal.

### The working flow

```
user types "researching AI in global politics" in ami/input.txt
  → watcher.py detects trigger phrase, extracts topic
  → writes ami/ami_trigger.json {topic: "AI in global politics", consumed: false}
  → experiment_zones.py reads trigger, fires bridge signal (visual)
  → responder.py reads trigger, marks consumed, waits 3s, calls Claude
  → Claude returns structured briefing (summary + key developments + search terms)
  → results written to ami/results.txt and printed to terminal
```

### First successful run output

Topic: "AI in global politics"
- One sentence summary of current state
- 4 key developments (great power race, governance vacuum, election warfare, autonomous weapons)
- 3 specific search terms worth pursuing

Total time from typing to results: ~5 seconds.

### What's working well

- File-based IPC is solid — same pattern as the existing NCA/LLM bridge, proven reliable
- Debounce (15 seconds) prevents double-firing on same topic
- Consumed flag prevents responder double-firing
- 3 second visual delay lets NCA bridge form before results appear — feels intentional
- Claude Haiku is fast and the prompt produces useful structured output

### What's rough / next improvements

- NCA routing is visual only — the bridge fires but doesn't actually gate the LLM call. Responder triggers on signal file, not on NCA zone threshold.
- Zone B activity too low to use as a real threshold (right zone ~0.035-0.040). Need stronger anchor reinforcement or different threshold mechanism.
- Topic detection is regex only — misses complex sentences, won't catch implicit intent
- Results go to a text file — needs better output (terminal popup, overlay, notification)
- Single input → single output — no multi-zone routing yet
- NCA is the Physarum simulation, not the trained NCA model

### The architecture that's now proven

```
SENSOR (file watcher)
  ↓ detects intent
SIGNAL (ami_trigger.json)
  ↓ two consumers simultaneously
NCA (visual routing, bridge fires)    RESPONDER (LLM call, results)
  ↓                                        ↓
zone_state.json                       results.txt
```

### What this proves

The plumbing works. File watching → signal injection → NCA activation → LLM call → results. That chain exists and runs. The architecture is sound.

What it does NOT yet prove: that the NCA is doing meaningful routing (it's currently visual only). That's Phase 2.

---

## Things To Decide Before Phase 0

1. Writing trigger or calendar trigger first?
2. Where are you actually typing — text file on disk or window on screen? (decides file watcher vs OCR)
3. Which zone of the current grid becomes the writing zone?
4. Same ch5 injection mechanism or new channel for sensory input?
