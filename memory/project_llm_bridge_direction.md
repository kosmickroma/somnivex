---
name: LLM Bridge — POC Priority
description: The LLM+NCA hybrid POC is now the top priority. Build a working loop where one LLM observes and acts on the NCA grid.
type: project
---

LLM bridge POC is top priority as of 2026-03-18. User wants to demonstrate LLM+NCA hybrid is viable before investing further in research directions.

**Why:** User needs to start generating value from this work. The POC is the foundation for everything — shared medium, multi-agent, parallel search, all of it requires this working first.

**The POC:** One LLM + one NCA. LLM reads grid state (feature CSV), reasons about it, injects commands via command file, NCA responds. Demonstrable in real time.

**Files to build:**
- `nca/llm_bridge.py` — asyncio script, tails feature CSV, calls Claude API, writes command file (~100 lines)
- Add command file polling to `nca/run_free.py` (~20 lines)

**Bigger vision documented in:** `notes/shared_medium_hypothesis.md`
- NCA as shared dynamic medium between multiple LLM agents
- LLM delegates parallel search/organization to NCA swarm
- LLMs sync at slow timescale (every ~1000 steps), NCA runs fast
- Not "processing" — delegation of parallel computation to swarm

**How to apply:** When starting next session, build llm_bridge.py first before anything else. The Physarum training and grammar comparison can run in background while this gets built.
