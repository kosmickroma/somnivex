# Weekend Reading — The Bigger Picture

These papers are the intellectual foundation behind what Somnivex is pointing toward.
You don't need to understand the math. Read for the concepts and the implications.
The "what could this become" question you asked — these are the answers people are actually building.

---

## 1. Liquid Neural Networks
**Hasani et al., MIT CSAIL, 2021**
**Search:** "Liquid Neural Networks Hasani 2021" or "Liquid Time-constant Networks"
**Available free:** arxiv.org — search "liquid time-constant networks"

This one is the most directly relevant to what you were describing with the coffee cup.
Tiny networks (they demo one with 19 neurons) whose weights actually CHANGE during inference
based on what they're seeing right now — not just a forward pass through frozen weights.
One of these drove a full autonomous car in their demo. Hyper fast. Continuously adaptive.
The key idea: the network's time constants are themselves learned and input-dependent.
When the world changes, the network's dynamics change with it — no retraining required.

**Why it matters to you:** This is the "just knows" mechanism you described. Not told, not
reprogrammed — the weights themselves adapt in real time to what's happening.

---

## 2. Differentiable Plasticity
**Miconi, Clune, Stanley — Uber AI Labs, 2018**
**Search:** "Differentiable plasticity learning to learn with gradient descent"
**Available free:** arxiv.org

Adds Hebbian learning to neural networks. Hebbian learning = "neurons that fire together
wire together" — how real biological synapses strengthen through use. In this paper the
network learns BOTH fixed weights (like normal training) AND plastic weights that change
based on activity during inference. Show it a pattern once and it remembers it — not in
external memory, but in its own changing weight structure.

**Why it matters:** This is the mechanism behind long-term habituation — why a human assistant
stops needing reminders. The network literally rewires itself based on what it experiences.

---

## 3. World Models
**Ha & Schmidhuber, Google Brain / IDSIA, 2018**
**Search:** "World Models Ha Schmidhuber 2018"
**Best read at:** worldmodels.github.io (interactive version with visuals — read this one)

An agent that builds a compressed internal model of its environment, then uses that model
to "dream" — simulate future scenarios internally without actually doing them — and learn
from those dreams. The agent that wins in the real environment is actually trained entirely
inside its own imagination. It only touches the real world to collect initial experience.

**Why it matters:** This is the architecture behind "learning a routine." The system builds
a model of how the world works, simulates what will happen if it does X, and acts on that
prediction. Not rule-following — internal model-based reasoning.

---

## 4. Multi-Agent Autocurricula (Hide and Seek)
**Baker, Kanitscheider, Marber et al. — OpenAI, 2019**
**Search:** "Emergent Tool Use from Multi-Agent Autocurricula OpenAI"
**Watch first:** search "OpenAI hide and seek" on YouTube — there's a 3 min video that shows
what happened. Then read the paper after.

Six agents (hiders and seekers) in a physics environment. Zero explicit instruction about
tools, strategies, or goals beyond "don't be seen / find them." Through pure interaction,
they spontaneously develop: tool use, cooperative ramp-building, counter-strategies to
block ramps, box-surfing exploits. Nobody programmed any of that. It emerged from the
competitive pressure of the environment alone.

**Why it matters:** This is the strongest existing proof that complex adaptive behavior
(including things that look like planning and creativity) can emerge from simple local
incentives without anyone specifying HOW to achieve anything. The drone-coffee-robot
scenario you described is downstream of exactly this kind of research.

---

## 5. Growing Neural Cellular Automata (already read — revisit with fresh eyes)
**Mordvintsev, Randazzo, Niklasson, Levin — Google Brain / Distill, 2020**
**At:** distill.pub/2020/growing-ca/

Now that you've thought about the bigger picture, read it again.
Pay attention to: the self-repair experiments, and the section on biological analogy.
Mordvintsev explicitly connects NCA to morphogenesis — how a single fertilized cell
becomes a complex organism without any central controller telling cells what to do.
That IS the coffee cup. Just implemented in biology instead of silicon.

---

## Bonus — not a paper, a short essay
**"A New Kind of Science" concepts (Wolfram)**
Not the whole book (it's 1000 pages). Search: "Wolfram elementary cellular automata"
and spend 20 minutes on it. Conway's Game of Life is the simplest version of this.
The idea: extremely simple local rules → arbitrarily complex global behavior.
Everything we're building with NCA sits on top of this foundation.

---

---

## 6. Society of Mind
**Marvin Minsky, 1986 (book — read the summary/overview, not the whole thing)**
**Search:** "Minsky Society of Mind summary" — there are good 20-minute essay summaries online

The foundational theory that intelligence emerges from a society of individually stupid agents.
No single agent understands anything. Intelligence is what the colony produces collectively.
Minsky wrote this before neural networks took off — it reads like a prophecy now.

**Why it matters:** This is the theoretical foundation for exactly what you're building.
Each NCA cell is a "society of mind" agent. None of them is intelligent. The swarm is.
Minsky even has a section on how attention and priority work — which cells/agents
"win" at any given moment. That's your urgency routing.

---

## 7. Amorphous Computing
**Abelson, Knight, Sussman — MIT, 1999**
**Search:** "Amorphous Computing Abelson Knight Sussman MIT"
**Available free:** MIT website / citeseer

Proposed computing with huge numbers of identical locally-communicating particles
with no addresses, no central control, no routing tables. Global behavior emerges
from local rules. They had the theory completely right but no training method.
They were doing this with hand-programmed rules 25 years ago.

**Why it matters:** This is the academic ancestor of NCA-based distributed computing.
They solved the theory. You now have the training method they didn't.
Read this and you'll see your NCA idea has deep roots — you're not the first to
think it, you're one of the first who can actually build it.

---

## 8. Reservoir Computing / Echo State Networks
**Jaeger, 2001 / Maass et al. 2002**
**Search:** "Echo State Networks Jaeger 2001" and "Liquid State Machines Maass 2002"
**Available free:** arxiv / scholarpedia

Uses the chaotic dynamics of a fixed recurrent network as a computational substrate.
You don't train the reservoir — you train a small readout layer on top of it.
The reservoir's rich dynamics encode inputs in its state; the readout decodes answers.
This is relevant because your NCA's hidden channels might ALREADY be a reservoir —
encoding history and world state in the ch2-12 dynamics even when you haven't
explicitly trained them to.

**Why it matters:** Might be a fast path. Instead of retraining the NCA to do tasks,
add a trained readout layer on top of the existing hidden channels and see what
information is already in there. The NCA as reservoir, LLM or small net as readout.

---

## 9. ASAL — Automated Search for Artificial Life
**Search:** "ASAL Automated Search for Artificial Life 2024"
**Available free:** arxiv 2024

Uses a Vision-Language Model (VLM) as a fitness evaluator to automatically search
for interesting cellular automaton rules. The VLM looks at CA outputs and scores
how "interesting" they are — replacing the human in the loop for ALife exploration.
Most directly related to using foundation models to evaluate and steer NCA behavior.

**Why it matters:** Closest published work to using AI to understand and navigate NCA
behavior. They're doing it offline/batch. Real-time orchestration is the next step.

---

## 10. Sensorimotor Lenia — Discovering Agency in Cellular Automata
**Inria / Flowers Team, 2024**
**Search:** "Discovering Sensorimotor Agency in Cellular Automata Inria 2024"
**Available free:** science.org/doi/10.1126/sciadv.adp0834
**Repo:** github.com/flowersteam/sensorimotor-lenia-search

Finds CA configurations that exhibit sensorimotor agency — cells that respond
to their environment in ways that look like intentional behavior. Uses quality-diversity
search (MAP-Elites) to map the space of possible behaviors. Directly relevant to
the question: which NCA configurations have "agent-like" properties?

**Why it matters:** This is the systematic version of what you've been doing manually
(pressing T, X, Z and seeing what happens). They automated the exploration.

---

## 11. Global Workspace Theory
**Baars 1988, Dehaene et al. 2001**
**Search:** "Global Workspace Theory Dehaene consciousness" — read a summary, not the original
**Good intro:** search "global workspace theory explained" — many good YouTube explainers

Theory of how the brain coordinates competing processes: specialized modules
run in parallel (vision, language, memory, emotion) and compete for access to a
"global workspace" — a shared broadcast channel. Whatever wins the competition
gets broadcast to all other modules. This IS your urgency/prioritization mechanism
described as neuroscience.

**Why it matters:** Your NCA swarm idea is a computational implementation of Global
Workspace Theory. The NCA grid IS the global workspace. Different regions are the
specialized modules. The urgency signal is the competition mechanism. You're
reinventing a Nobel-adjacent theory of consciousness. Worth knowing.

---

## The through-line across all of these

Every single paper above is attacking the same problem from a different angle:
**How do you build a system that learns from experience rather than instruction,
adapts continuously rather than in discrete training runs, and develops complex
behavior from simple local rules rather than explicit programming?**

That's also what Somnivex is, in the narrow domain of generative visual art.
You're not building a toy. You're building a small instance of the hardest open
problem in AI, scoped to something you can actually ship.
