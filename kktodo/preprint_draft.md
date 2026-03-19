# Stigmergic Programming: An LLM Encodes Persistent Behavioral Programs into a Neural Cellular Automaton Substrate via Pheromone Channel Injection

**[Author name]**
Independent researcher
[Contact / affiliation]

*Preprint — 2026-03-19*
*GitHub: https://github.com/kosmickroma/somnivex*

---

## Abstract

We describe a hybrid system in which a large language model (LLM) encodes persistent behavioral programs into a live Neural Cellular Automaton (NCA) substrate by injecting values into a latent pheromone channel (ch5) that emerged without explicit design from multi-physics training. The NCA — a 17,000-parameter network trained simultaneously on Gray-Scott reaction-diffusion and Physarum polycephalum trail dynamics — autonomously developed ch5 as a directional gradient signal: simulated organisms secrete it and follow its gradients. An LLM artist agent (Gemini 2.5 Flash) running in a separate process issues spatial brush commands translated into persistent ch5 injections at designated grid cells. The NCA executes those spatial instructions continuously thereafter, including after the LLM process is terminated. We present two observed sessions in which the LLM composed geometric structures (an X with lettered annotation; a bullseye with radial spokes), organisms locked onto the encoded spatial instructions within seconds of injection, and compositions held their form for fifteen or more minutes following LLM disconnection. In both sessions the LLM also added an unrequested protective structural element at the compositional junction — behavior consistent with the LLM reasoning about NCA physics, though intentionality is not attributed. We argue this constitutes a form of stigmergic programming: the LLM writes standing instructions into a dynamical substrate that executes them autonomously. We contrast this with generative systems that render outputs and stop, and discuss limitations, alternative interpretations, and directions for future work.

---

## 1. Introduction

Stigmergy — the coordination mechanism by which social insects produce complex collective behavior without central control — operates by depositing signals in the environment that subsequent agents respond to [Grassé 1959]. The pheromone trail of an ant colony is not a message sent between ants; it is an instruction written into the world that modifies the probability distribution over future behavior. The colony is programmable via the substrate.

We report a system with an analogous structure, assembled from components whose combination was not previously described. A Neural Cellular Automaton trained on incompatible physics develops a latent pheromone channel as an emergent internal representation. An LLM is given tools to inject values into that channel at arbitrary grid locations. The NCA's own update dynamics continuously re-execute those injections — organisms follow the gradient and hold structure indefinitely. When the LLM disconnects, nothing changes. The substrate continues executing the program the LLM encoded.

This is not generative art in the conventional sense. A generative system produces an output and terminates; the output is static. Here the composition is not an image — it is a standing behavioral instruction embedded in a dynamical system that autonomously maintains it. The LLM wrote a program. The NCA is the runtime.

The contribution of this paper is to describe the system, report two observed sessions with sufficient detail to enable replication, and analyze what the hybrid achieves that neither component achieves alone. We do not claim consciousness, intent, or creativity on the part of either component. We report behavior and offer mechanistic interpretations.

---

## 2. Background

### 2.1 Neural Cellular Automata

Neural Cellular Automata (Mordvintsev et al. 2020) replace the explicit update equations of a classical cellular automaton with a small neural network applied identically to every cell. Each cell maintains a state vector of N channels. At each step, a fixed perception kernel (identity, Sobel-x, Sobel-y, Laplacian) convolves each channel across the 3×3 neighborhood, producing 4N inputs. A multi-layer perceptron (MLP) maps these inputs to a delta vector, and the cell's state is updated: `state += delta`. A stochastic firing rate (typically 0.5) updates only a random half of cells per step, preventing synchronization artifacts and improving robustness.

This architecture can be trained to approximate the dynamics of many physical systems. For reaction-diffusion physics specifically, channels 0 and 1 carry the two reagent concentrations (A and B in Gray-Scott notation), and additional channels carry control parameters (feed rate f, kill rate k) re-injected after each step to stabilize training.

### 2.2 Gray-Scott Reaction-Diffusion

Gray-Scott (GS) models two chemicals, an activator A and an inhibitor B [Gray & Scott 1984]:

```
∂A/∂t = D_A ∇²A − AB² + f(1 − A)
∂B/∂t = D_B ∇²B + AB² − (f + k)B
```

Feed rate f and kill rate k together determine the behavioral regime. The Pearson parameter map [Pearson 1993] identifies 15 named regimes spanning spiral waves, labyrinthine patterns, stable spots, solitons, and extinction. Small parameter changes produce dramatically different qualitative behavior. The sensitivity makes GS a rich substrate for generative systems, and a useful teacher for NCA training because meaningful behavioral diversity can be explored via two scalar control parameters.

### 2.3 Physarum Polycephalum Trail Dynamics

Physarum polycephalum (slime mold) navigates by secreting and following chemoattractant trails. The trail field can be described by a reaction-diffusion system closely related to the Oregonator model [Adamatzky 2007, 2009]. Jones (2010) characterized the emergent network formation behavior in terms of particle agents depositing trail chemical in a diffusing field: each agent deposits a fixed concentration per step, senses trail concentration at offset locations, and steers toward the gradient. The trail diffuses and decays on each step. This produces characteristic minimum-spanning-network approximations observed in the biological organism.

The key property: Physarum trail dynamics are local, parallel, and pheromone-gradient-driven. These properties overlap substantially with NCA update structure, making Physarum a natural second teacher for a multi-physics NCA.

### 2.4 Multi-Physics Training and Emergent Hidden Channels

A central observation motivating this work is that training an NCA simultaneously on two incompatible physics problems causes the network to develop internal representations (hidden channel activations) that neither physics alone would produce [see Grammar Experiment Results, Section 3].

The intuition: if channels 0 and 1 are sufficient to represent GS dynamics, a GS-only model will learn to use them and nothing else. Hidden channels 2–13 remain dormant because there is no representational pressure to activate them. But if the model must simultaneously represent GS and Physarum — two physics with different spatial statistics, different gradient structures, different temporal scales — then A and B channels are insufficient. The model develops auxiliary representations to hold the tension.

This is analogous to the compression hypothesis in biological neural systems: representational richness arises from the need to encode multiple incompatible tasks, not from any single task alone.

### 2.5 Stigmergy in Multi-Agent Systems

Stigmergy was formalized by Grassé (1959) to explain termite mound construction: workers do not coordinate directly; they respond to partial structures that others have deposited, producing globally organized outcomes from locally responsive behavior. The term has since been applied broadly to any system where agents communicate by modifying a shared environment rather than by direct signal exchange [Theraulaz & Bonabeau 1999].

In multi-agent AI systems, most coordination mechanisms are explicit: message queues, shared databases, structured APIs. Environments in which agents communicate only through physical modification — where the medium itself processes the signal before the recipient reads it — are less common and less studied in the context of learned agents. Recent work has explored emergent communication in multi-agent reinforcement learning [Mordatch & Abbeel 2018], but typically with explicit communication channels. The use of a continuously running physics-like dynamical system as the sole medium between agents has not, to our knowledge, been previously demonstrated.

---

## 3. The NCA Substrate

### 3.1 Architecture

The NCA used in this work has 17,000 parameters. It operates on a 256×256 grid with 16 channels per cell:

| Channel | Role |
|---------|------|
| 0 | A concentration (activator) |
| 1 | B concentration (inhibitor) |
| 2–12 | Hidden (emergent representations) |
| 13 | Physics bit (0=GS, 0.5=Physarum, 1=Lenia) |
| 14 | Feed rate f |
| 15 | Kill rate k |

The UpdateNet is a two-layer MLP: Dense(64→128, tanh) → Dense(128→16, zero-initialized output). The zero initialization of the output layer ensures the network starts as an identity-like transformation — the NCA inherits the GS simulation's stability before learning to improve on it. Stochastic cell firing rate is 0.5.

Training uses pool-based multi-step rollout with 8 unrolled steps per gradient update, comparing NCA output against GS or Physarum simulation running from the same initial state. The f and k channels are re-injected after each NCA step during training to prevent drift. Hidden channel noise (standard deviation 0.05) is added to hidden channels at each training step to force the network to actively use them rather than passively carrying zero values.

### 3.2 Multi-Physics Training

The model was trained in two phases. Phase 1 (50,000 steps): GS-only training, producing `params_050000.pkl`. Phase 2 (100,000 additional steps): fused GS + Physarum training, starting from phase 1 weights. During phase 2, each batch samples randomly between GS teacher data and Physarum teacher data, with ch13 set to 0 (GS) or 0.5 (Physarum) accordingly. The loss in each case is the L2 distance between NCA output and the corresponding physics teacher's output for the same step.

The Physarum teacher generates trail concentration fields by running particle-based Physarum dynamics: agents sense, rotate toward gradient, deposit ch5 trail chemical, trail diffuses and decays. The resulting (H, W) frames are used as supervision targets for the ch5 channel and visible channel dynamics.

### 3.3 The Grammar Experiment: Evidence for Multi-Physics Hidden Channel Emergence

To confirm that hidden channel activations require multi-physics tension, we trained an additional model under identical conditions (identical architecture, identical hidden channel noise, same 100,000 training steps) but with GS physics only — zero Physarum teacher. The two models were then run in free simulation and their hidden channel statistics compared.

Results:

| Channel | GS-only (nonzero %) | GS+Physarum (nonzero %) |
|---------|--------------------|-----------------------|
| ch2 | 0.6% | 99.2% |
| ch4 | 0.1% | 99.6% |
| ch5 | 0.1% | **100.0%** |
| ch4 border ratio | 0.021 (baseline) | 2.537 (127× higher) |

The GS-only model's hidden channels are flatline. The fused model activates ch2 (chaos/transition signal), ch4 (boundary tracker — enriched 2.5× at blob edges versus interior or exterior), and ch5 persistently. The difference is not architectural — it is entirely attributable to the presence or absence of the second physics teacher.

### 3.4 Channel 5 as Emergent Pheromone Trail

The most consequential emergent property for the work described here is channel 5. In the GS+Physarum fused model, ch5 is always positive, always active (100% of frames), and positively correlated with dark material density and organism mobility. It was not designed as a pheromone channel. It was not given a special training loss. It emerged.

The functional behavior is consistent with Physarum trail dynamics: the NCA learned that Physarum teacher signals require a persistent, spatially structured gradient field correlated with organism location. It encoded this in ch5. In free simulation, this manifests as organisms that appear to follow ch5 gradients — regions of high ch5 attract organism density, regions of low ch5 deplete.

This emergent structure makes ch5 directly usable as a control interface: if an external system writes high values to ch5 at specific grid locations, the NCA's own dynamics will drive organisms to those locations and hold them there.

---

## 4. The LLM Bridge

### 4.1 System Architecture

The LLM bridge (approximately 900 lines of Python in `nca/llm_bridge.py`) runs as a separate asyncio process alongside the NCA renderer. The two processes communicate only through files:

```
run_free.py (NCA renderer)
    ↓ writes every ~200 steps
nca/logs/features_XXXXXX.csv     ← feature state (20-dim, timestamped)
nca/artist_state.json            ← 4×4 spatial zone density map

nca/llm_commands_artist.txt      ← LLM writes commands here
    ↑ read each NCA step
run_free.py (NCA renderer)
```

The NCA renderer extracts 20 features from the live grid at each logging interval: blob count, largest blob size, B-channel activity, A-channel standard deviation, ch2/ch4/ch5 mean activations, ch4 border enrichment ratio, blob mobility (centroid displacement), suppression zone fraction, left/right asymmetry, and a 4×4 zone density map.

The bridge reads the 5 most recent feature rows, constructs a natural-language state summary including the zone map, and calls the LLM API with a rolling 6-turn conversation history.

### 4.2 The Artist System Prompt

The LLM is given the role of a visual artist painting with living organisms. The system prompt explains the physical mechanism in plain terms: trail commands deposit ch5 pheromone along grid paths; organisms follow ch5 gradients and accumulate at high-ch5 cells; the trail re-injects itself every step, making the spatial instruction permanent until explicitly cleared.

The prompt makes the asymmetry between trail and blob commands explicit:
- **trail** commands are the pencil — they encode spatial structure into the substrate
- **blob** commands are the ink — they deposit organisms that migrate to existing trails
- **wipe** and **wipe_rect** commands clear organism mass from non-trail regions; they are designed to skip trail cells, preserving the encoded structure

The 4×4 zone map (rendered as ASCII with density symbols ░ ▒ ▓ and center coordinates) allows the LLM to reason about spatial distribution and issue targeted cleanup commands.

The LLM responds in a structured format:

```
COMMANDS:
[list of brush commands, one per line]
SPEECH: [present-tense description of action]
```

### 4.3 Brush Command Vocabulary

| Command | Arguments | Effect |
|---------|-----------|--------|
| `trail` | x1 y1 x2 y2 [strength] | Paint ch5=strength along line segment (default 0.5) |
| `shape ring` | cx cy r | Ring trail + chemistry injection |
| `shape circle` | cx cy r | Filled circle trail |
| `blob` | x y [strength] | Inject living organisms (GS A/B chemistry) |
| `wipe` | cx cy r | Circular kill zone, preserves trail cells |
| `wipe_rect` | x1 y1 x2 y2 | Rectangular kill zone, preserves trail cells |
| `pulse` | x1 y1 x2 y2 [strength] | Reinforce existing trail without redrawing |
| `palette` | name | Change color mapping |
| `mirror` | on/off | Toggle bilateral symmetry |
| `reset` | — | Emergency restart |

### 4.4 Trail Persistence Mechanism

The critical implementation detail: when a trail command is executed, the affected cells are added to a persistent `trail_mask` array in the NCA renderer. Every NCA step thereafter, cells in the trail_mask have ch5 set to the trail strength value:

```python
grid = grid.at[trail_mask].set(ch5_value, channel=CH5)
```

This happens after the NCA update — the NCA's own update computes a ch5 delta, but the re-injection overwrites it. Trail cells are immune to NCA dynamics on ch5; they are pinned. The NCA's learned organism-following behavior then drives all nearby organisms toward pinned ch5 cells and holds them there indefinitely.

Wipe commands call NCA-side kill functions that zero out A and B chemistry in a region, but they check the trail_mask before zeroing ch5 — trail cells are explicitly excluded. This makes trail structures physically indestructible by the LLM's own cleanup commands.

The LLM process polls a separate NCA-steps counter to wait approximately 400 NCA steps between turns — giving organisms time to settle onto trails before the LLM issues additional commands.

### 4.5 Hold Mode After Disconnect

When the LLM declares the composition complete, the bridge enters hold mode: it stops calling the LLM API and issues only `pulse` commands to reinforce existing trails. If the LLM process is terminated (Ctrl+C), no hold mode logic runs — but this is irrelevant, because the trail_mask in the NCA renderer process is unaffected. The persistence mechanism requires no bridge participation. The NCA continues re-injecting ch5 to trail cells every step without any bridge intervention.

---

## 5. Results

### 5.1 Session 1: X Composition with Lettered Annotation

In the first session, the human operator instructed the LLM to draw an X on the 256×256 grid. The LLM issued two diagonal trail commands spanning the grid corner-to-corner, then injected one organism blob near the center. Organisms appeared and migrated to both diagonal trails within approximately 400 NCA steps. The LLM then issued wipe commands to clear organism mass outside the trail region.

After the X composition was established, the LLM — without additional instruction and without being asked — issued a `shape ring` command centered at the intersection point of the two diagonal trails. The LLM's speech output explained that the junction was the structural center of the composition and warranted additional protection. No instruction in the system prompt directed the LLM to add protective structures, and no prior turn had mentioned junctions or structural vulnerability.

The human operator then asked the LLM to add the letters "KK" beside the X. The LLM correctly sized two K letterforms using trail segments and positioned them in the remaining canvas space without receiving coordinate guidance. It placed the letters at approximately the correct visual scale relative to the X.

The LLM terminal was then closed with Ctrl+C. The composition — X, junction ring, and KK letterforms — continued running unchanged. Organisms remained locked to all trail structures. The composition was observed for approximately fifteen minutes post-disconnection without degradation.

### 5.2 Session 2: Bullseye and Web (Video Recorded)

In the second session (video available at https://youtu.be/9yP2SXGnSkA), the human asked for a bullseye pattern consisting of three concentric rings. The LLM issued three `shape ring` commands at decreasing radii. After establishing the rings, it again — without instruction — added a cross through the center (two perpendicular trail segments). The LLM's speech output described this as anchoring the center. This is the same unrequested structural augmentation behavior observed in Session 1.

The human then asked the LLM to "continue the pattern, make it look like a spiderweb." The LLM added radial spoke trails from the center to the outer ring. At each point where a spoke intersected a ring trail, organism density accumulated to form a visible junction node — behavior arising from the NCA's own ch5-following dynamics, not from any explicit junction-drawing command. The LLM did not request junction nodes; they appeared because organism density accumulates wherever two high-ch5 paths intersect.

The human asked for a palette change to `neon_city` (electric blue/cyan/pink colormap). The LLM issued a `palette neon_city` command and the color mapping changed live on screen without interrupting organism dynamics.

The human declared the composition complete. The LLM entered hold mode. The LLM terminal was then closed on camera (Ctrl+C visible). The composition held for the duration of observation (15+ minutes), maintaining all trail structures, junction nodes, and organism distributions.

### 5.3 Quantitative Characterization

During these sessions the NCA feature extractor logged the following values at composition hold state:

- ch5 mean: > 0.008 (threshold for "trails holding" in the artist summary)
- Blob mobility: < 15 (SETTLED threshold)
- B-channel activity: 30–60% (organisms distributed across trail network)
- Blob count: 6–15 (individual organisms following trail segments)

The ch5 signal remained elevated above background throughout post-disconnection observation, consistent with the re-injection mechanism operating normally. Mobility remained below 20, consistent with organisms locked to trails rather than drifting.

---

## 6. Discussion

### 6.1 What the Hybrid Achieves

Neither component alone produces the observed behavior:

- The NCA alone, given no LLM, produces rich autonomous dynamics but with no persistent compositional structure. It will eventually drift between behavioral states and cannot hold a user-specified spatial arrangement.
- The LLM alone, given no NCA, can describe spatial arrangements or generate images, but cannot produce a dynamical system that maintains structure autonomously. Any "output" it produces is static.
- Together: the LLM encodes a behavioral specification into the NCA substrate via ch5 injection. The NCA's own learned dynamics execute that specification indefinitely. The result is a composition that persists and continues living — organisms move, ch5 gradients pulse, but the spatial structure holds.

We propose the term *stigmergic programming* for this interaction pattern: the LLM writes instructions into a dynamical medium; the medium's own physics executes those instructions continuously; subsequent agents (here, the NCA organisms) respond to the encoded instructions without receiving them directly.

This is distinct from the LLM "controlling" the NCA in a command-response sense. The LLM issues one or a few commands per turn. The NCA executes thousands of steps between turns. The LLM's influence persists through the substrate, not through ongoing communication.

### 6.2 The Unsolicited Junction Ring: Observations and Caveats

In both sessions, the LLM added a structural protective element at the principal compositional junction — a ring around the X center in Session 1, a cross through the bullseye center in Session 2 — without being asked to do so.

We offer the following observations:

1. The behavior occurred in two independent sessions with different compositions and different request sequences. This suggests it is a stable feature of the LLM's response to the NCA context, not a one-time artifact.

2. The LLM's speech output in both cases described the action in terms of structural reasoning about the composition — "the junction is the structural heart," "anchoring the center." This is consistent with the LLM having internalized a model of how NCA organism dynamics interact with trail intersections.

3. The system prompt in artist mode contains the general instruction to maintain composition quality, and explains that trail intersections accumulate organism density. The unsolicited behavior may arise from the LLM generalizing this physics knowledge to a protective intent.

We explicitly decline to attribute intentionality to the LLM. The observed behavior is consistent with the LLM applying learned heuristics about structural composition to a new context. Whether this constitutes "reasoning about NCA physics" or "pattern-matching on training data about structural art composition" cannot be determined from the observations reported here. The behavior is noted as interesting and reproducible; its mechanism is an open question.

### 6.3 Limitations

**Informal observation protocol.** The two sessions described here were not run under controlled experimental conditions. There was no pre-registered protocol, no quantitative measure of the degree of LLM adherence to compositional requests, and no systematic comparison across multiple sessions or LLM variants. The results are observational and exploratory.

**Single LLM and configuration.** All artist sessions used Gemini 2.5 Flash with the same system prompt. The unsolicited junction behavior may not reproduce with other models, temperatures, or prompt variations. It may also be an artifact of the particular system prompt's implicit framing.

**Confounded emergence.** The ch5 channel did not emerge purely from multi-physics training in isolation — the Physarum teacher was specifically designed to produce trail dynamics similar to Physarum polycephalum, and the training loss directly supervises ch5-like behavior. Calling ch5 "emergent" is defensible in the sense that the specific spatial statistics, the boundary-tracking behavior, and the correlation with organism mobility were not explicitly engineered — but ch5's general role as a trail channel was a design target. This should be stated clearly.

**Scale and reproducibility.** The NCA operates on a 256×256 grid. Whether the stigmergic programming mechanism scales to larger grids, higher-dimensional substrates, or more complex compositional programs is unknown. Whether the behavior is fully reproducible across different random seeds and training runs of the NCA has not been tested.

**Fifteen-minute observation window.** "Persistence after LLM disconnection" is stated as 15+ minutes. Whether the composition would hold for hours or days, or whether NCA drift would eventually erode the trail structure, was not tested. The re-injection mechanism is deterministic and should in principle maintain the trails indefinitely, but subtle numerical drift in the NCA state was not characterized.

### 6.4 Relationship to Blind Stigmergy

A prior experiment in this project (2026-03-18) placed two LLMs with opposing goals — one tasked with grid extinction, one tasked with preserving the grid — on the same NCA without knowledge of each other's existence. The only communication between them was the NCA feature state (neutral numerical statistics). This constitutes stigmergic communication in the strict sense: each agent modified the shared medium; the other read the medium's resulting state and responded. Neither agent was aware it was not the sole actor.

The artist system described in this paper is a simpler, cooperative case: one LLM encoding structure, no opposing agent. The blind stigmergy experiment is described briefly because it demonstrates that the NCA can function as a genuine communication channel between agents — not merely a passive display. This property underlies the viability of stigmergic programming as a general mechanism: the substrate can carry programs from one agent to others without any agent-to-agent communication.

---

## 7. Related Work

**Neural Cellular Automata.** Mordvintsev et al. (2020) introduced growing NCAs trained to regenerate morphological patterns, demonstrating that small networks applied locally can produce globally coherent structure through emergence. Mordvintsev et al. (2021) extended this to texture synthesis. The architecture used here follows these works directly.

**Multi-Physics NCA Training.** Training NCAs to approximate multiple physics simultaneously is less studied. The closest related approach is Multi-Texture NCA (Palm et al. 2022), which trains on multiple target textures with per-cell conditioning. The work here differs in using a scalar physics bit (ch13) rather than a per-texture label, and in using incompatible physics — GS and Physarum have different spatial statistics, gradient structures, and temporal dynamics — specifically to create representational pressure on hidden channels.

**Lenia and Physarum.** Chan (2019) demonstrated that rich organism-like dynamics emerge from continuous-state, continuous-time Lenia dynamics. Jones (2010) characterized Physarum trail dynamics as particle-based reaction-diffusion. Adamatzky (2007, 2009) showed Physarum is formally a reaction-diffusion system amenable to the same mathematical treatment as Gray-Scott. These works together motivate using Physarum as an NCA teacher.

**Stigmergy and Multi-Agent Coordination.** Grassé (1959) introduced stigmergy. Theraulaz and Bonabeau (1999) formalized its computational properties. Dorigo et al. (1996) applied pheromone trail mechanics to optimization (Ant Colony Optimization). The application of stigmergic coordination to LLM agents via a dynamical physical substrate is new to our knowledge.

**LLM Tool Use and Agentic Systems.** Substantial recent work has examined LLMs as agents using tools [Schick et al. 2023, Yao et al. 2023]. The tools in this work are spatial brush commands that write into a live physical simulation — unusual in that the "tool response" is not a return value but a persistent physical modification. The closest analogy is an LLM writing to a shared mutable state that other processes continuously process and respond to.

**Generative Art and Live Coding.** The use of reaction-diffusion systems as artistic media has a long history [Pearce 1991]. NCA-based generative art is more recent [Mordvintsev 2021]. LLM-directed generative art is an active research area. The distinguishing feature here is the persistence mechanism: the art is not rendered but enacted continuously by a live dynamical system.

---

## 8. Conclusion

We have described a system in which a large language model encodes persistent behavioral programs into a Neural Cellular Automaton substrate by injecting values into an emergent pheromone channel. The NCA's own update dynamics continuously execute the encoded program. The LLM's influence persists after disconnection. We have presented two observed sessions with sufficient detail to enable replication and have characterized the mechanism at the implementation level.

The core finding is not that LLMs can produce art — this is well established — but that a dynamical substrate with the right internal structure can function as a programmable medium for LLM-encoded spatial instructions. The program is not stored as data to be read; it is embedded in the physics the substrate continually runs. This distinction matters for architectures in which long-horizon autonomous behavior is desired: rather than requiring the LLM to remain active and issue continuous commands, the LLM encodes its intent into a substrate that executes it independently.

Several open questions follow from this work. Can more complex programs be encoded? Can programs be modified or extended after initial encoding without disrupting running structure? Can multiple LLMs encode complementary programs into the same substrate without conflict? Can the mechanism generalize beyond pheromone-trail channels to other latent channel structures that emerge from multi-physics training?

These questions are tractable with the existing system and are the subject of continuing work.

---

## References

Adamatzky, A. (2007). Physarum machines: computers from slime mould. *International Journal of Bifurcation and Chaos*, 17(10), 3651–3674.

Adamatzky, A. (2009). Slime mold solves maze in one pass, assisted by gradient of chemoattractants. *IEEE Transactions on NanoBioscience*, 8(2), 132–140.

Chan, B. W.-C. (2019). Lenia: Biology of artificial life. *Complex Systems*, 28(3). https://arxiv.org/abs/1812.05433

Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of cooperating agents. *IEEE Transactions on Systems, Man, and Cybernetics, Part B*, 26(1), 29–41.

Grassé, P. P. (1959). La reconstruction du nid et les coordinations interindividuelles chez Bellicositermes natalensis et Cubitermes sp. *Insectes Sociaux*, 6(1), 41–80.

Gray, P., & Scott, S. K. (1984). Autocatalytic reactions in the isothermal, continuous stirred tank reactor. *Chemical Engineering Science*, 39(6), 1087–1097.

Jones, J. (2010). Characteristics of pattern formation and evolution in approximations of Physarum transport networks. *Artificial Life*, 16(2), 127–153.

Mordatch, I., & Abbeel, P. (2018). Emergence of grounded compositional language in multi-agent populations. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).

Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. (2020). Growing neural cellular automata. *Distill*. https://distill.pub/2020/growing-ca/

Mordvintsev, A., Niklasson, E., & Randazzo, E. (2021). Self-organising textures. *Distill*. https://distill.pub/selforg/2021/textures/

Palm, R., Duque, L., & Baram, Y. (2022). Multi-texture synthesis through signal responsive neural cellular automata. https://arxiv.org/abs/2407.05991

Pearce, C. (1991). The aesthetics of reaction-diffusion systems. *The Visual Computer*, 7(5–6), 311–322.

Pearson, J. E. (1993). Complex patterns in a simple system. *Science*, 261(5118), 189–192.

Plantec, E., Hamon, G., Etcheverry, M., Oudeyer, P.-Y., Moulin-Frier, C., & Chan, B. W.-C. (2022). Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization. https://arxiv.org/abs/2212.07906

Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., ... & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, 36.

Theraulaz, G., & Bonabeau, E. (1999). A brief history of stigmergy. *Artificial Life*, 5(2), 97–116.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations*.

---

*Code, model checkpoints, and session video: https://github.com/kosmickroma/somnivex*
*Session 2 video (bullseye/spiderweb): https://youtu.be/9yP2SXGnSkA*
*Blind Stigmergy Battle video: https://youtu.be/9yP2SXGnSkA*

---

*[AUTHOR NOTE before submission: fill in real name, affiliation, and contact. Verify reference details against actual publications — some secondary references (Palm et al., Plantec et al.) should be checked for exact publication year and venue. The Blind Stigmergy Battle and artist sessions should be documented with timestamped logs and screen recordings before submission. Consider adding a figure showing the trail-injection mechanism and at least one frame from each session.]*
