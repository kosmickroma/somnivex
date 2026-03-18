# Tomorrow — GitHub Issues Session

Set aside time to write up proper GitHub issues for the repo.
These are all documented internally but need public issues so collaborators
can find them and so the project looks active and serious.

## Issues to create

1. **Compartmentalized substrate: chemical walls, doors, multi-sector LLM control**
   - Full writeup already in README under "Future Directions"
   - Implementation plan in notes/ideas_backlog.md
   - The immune response experiment is runnable today

2. **Directional bias control — phase drift slider + per-sector flow**
   - Current rightward drift is from weight bias (Lenia training) + positive vel_fx
   - Slider -1 to +1 for direction control
   - Per-sector directional fields = circular currents, convergence zones

3. **Mouse-drawn pheromone trails**
   - Full concept in notes/ideas_backlog.md
   - Version 1 (today): left-drag deposits B=0.8 along path, decays at 0.95/frame, structures follow
   - Version 2 (post-Physarum): deposit into ch0 with ch13=0.5 — model was trained on this exact physics
   - Version 3: right-drag = high-A clear zone (corridor walls), both together = routed channel
   - LLM control angle: LLM draws waypoints to route creatures between sectors spatially

4. **Physarum as third teacher**
   - Full plan in session checkpoint and ideas backlog
   - Typing exercise already built in physarum/

5. **Universal grammar hypothesis — GS-only alignment test**
   - GS-only training currently running (gs_only_100000.pkl)
   - Hungarian matching on cluster centroids
   - Method already implemented

6. **LLM bridge (Option A — file-based)**
   - Asyncio process tails feature CSV
   - Formats plain English state summary
   - Calls API, injects commands via command file
   - run_free.py polls command file each frame

## Format tip
Each issue should have:
- Clear one-line title
- What it does (2-3 sentences)
- Why it matters (1-2 sentences)
- Implementation plan (bullet points)
- Labels: enhancement, research, or experiment
