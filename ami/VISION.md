# AmI — Ambient Intelligence
*The nervous system layer nobody is building*

---

## The Origin

A conversation at a coffee shop. A researcher constantly switching modes — research day, writing day, different sources, different workflows. Having to consciously manage all of it himself. The system never sees what he's doing. He does all the coordination in his head.

The question that started everything:

**What if the system could see the work happening and move toward what's needed before you ask?**

---

## The Problem With LLMs

LLMs are flawed by design for ambient awareness.

They wait. They need to be invoked. They have no persistent presence. They know only what you hand them. You have to anticipate every exception in advance. Miss one instruction and the whole flow breaks.

Every AI company is racing to make the LLM the center of everything — bigger models, longer context, more tools. They're building smarter brains.

Nobody is building the nervous system.

---

## The Vision

An always-on substrate that connects your digital life. Different zones of the grid watching different things simultaneously. Signals propagating without a central brain. The LLM only wakes up when something actually needs reasoning.

**Examples of what it looks like working:**

- Calendar zone sees a meeting in 2 hours → signal propagates → file zone activates → relevant document surfaces → notification fires. You didn't ask. It just happened.

- Writing zone sees a header: "Researching dynamics of AI in global politics" → signal propagates → research zone activates → scrapers fire → latest articles start loading. You just started typing. The system was already moving.

- You're sick on a Sunday. The coffee run that was scheduled doesn't happen. Not because you cancelled it. Because the system noticed.

**This is not an AI assistant. Assistants wait. This is infrastructure. Infrastructure runs.**

---

## Why NCA Is The Right Substrate

- **Genuinely parallel** — 65,536 cells updating simultaneously. Different zones doing different things at the same time. No central coordinator.
- **Always-on** — runs whether anything is talking to it or not
- **Persistent memory** — trail_mask holds state indefinitely. The grid remembers.
- **Self-repair** — if a connection breaks, signals flow back along existing trails
- **Emergent routing** — signals propagate through local rules. No routing tables. No explicit addressing.
- **Proven** — we already demonstrated two LLMs coordinating through the grid without knowing each other existed

---

## The Architecture

```
SENSORS (always on)
  calendar watcher
  file watcher
  writing watcher (notepad / text editor)
  [future: screen, audio, health data, APIs]
        ↓
    inject signals into grid zones
        ↓
NCA GRID (the nervous system)
  different zones monitoring different concerns
  signals propagate through local rules
  urgency spreads without central coordination
  action channels activate when threshold crossed
        ↓
ACTION LAYER
  scrapers / RSS feeds
  file retrieval
  email / notifications
  calendar reads
        ↓ (only when reasoning needed)
LLM LAYER
  summarize these articles
  draft this email
  what does this mean
```

**The LLM is the prefrontal cortex. The NCA is the spinal cord. You don't talk to your nervous system. It just works.**

---

## What Makes This Different From Everything Else

The hidden channel grammar.

When trained on two incompatible signal types simultaneously — signals the visible channels alone cannot represent — the NCA develops internal abstractions to hold the tension. Those abstractions ARE the routing tokens. Not designed. Emerged.

This is already proven in the physics experiments:
- GS only → hidden channels flat
- GS + Physarum → ch5 emerges as pheromone channel
- The grammar came from the tension between incompatible teachers

For AmI: the tension is between incompatible signal types (intent signals vs. data signals). The routing grammar emerges the same way.

---

## What We Do NOT Know Yet

- Whether the routing grammar actually emerges from signal tension the same way physics grammar does
- How to encode text intent as a spatial signal the NCA can learn from
- Whether the routing needs to be trained or whether the current checkpoint is sufficient for a first version
- What the actual tokens look like when they emerge

**This is the experiment.**
