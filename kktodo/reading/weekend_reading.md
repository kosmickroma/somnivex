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

## The through-line across all of these

Every single paper above is attacking the same problem from a different angle:
**How do you build a system that learns from experience rather than instruction,
adapts continuously rather than in discrete training runs, and develops complex
behavior from simple local rules rather than explicit programming?**

That's also what Somnivex is, in the narrow domain of generative visual art.
You're not building a toy. You're building a small instance of the hardest open
problem in AI, scoped to something you can actually ship.
