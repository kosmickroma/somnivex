# nca/llm_bridge.py — LLM bridge for Somnivex
#
# Single-agent mode (default):
#   python nca/llm_bridge.py --provider gemini --log
#
# Battle mode (two terminals, blind stigmergy):
#   Terminal 1: python nca/llm_bridge.py --agent keeper   --provider gemini   --log
#   Terminal 2: python nca/llm_bridge.py --agent destroyer --provider anthropic --log
#
# Optional flags:
#   --log            print every state summary and response
#   --dry-run        summarize and print but never call the API
#   --interval N     seconds between checks (single-agent mode only, default: 15)
#   --provider NAME  anthropic (default) or gemini
#   --model NAME     override model
#   --agent NAME     keeper or destroyer (enables battle/turn-based mode)

import os
import sys
import argparse
import asyncio
import csv as _csv
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

LOG_DIR      = os.path.join(os.path.dirname(__file__), 'logs')
CMD_FILE     = os.path.join(os.path.dirname(__file__), 'llm_commands.txt')
TURN_FILE    = os.path.join(os.path.dirname(__file__), 'battle_turn.txt')
CHECK_EVERY  = 15        # seconds between checks (single-agent mode)
TURN_POLL    = 2         # seconds between turn file polls (battle mode)
ROWS_TO_READ = 5         # how many recent feature rows to summarize
MAX_HISTORY  = 6         # how many prior exchanges to keep in context

DEFAULT_MODELS = {
    'anthropic': 'claude-sonnet-4-6',
    'gemini':    'gemini-2.5-flash',
}

# Agent-specific command files (battle mode)
AGENT_CMD_FILES = {
    'keeper':    os.path.join(os.path.dirname(__file__), 'llm_commands_keeper.txt'),
    'destroyer': os.path.join(os.path.dirname(__file__), 'llm_commands_destroyer.txt'),
}

VALID_COMMANDS = [
    'none',
    'inject_chaos',
    'regime_1',
    'regime_2',
    'regime_3',
    'regime_0',
    'reset',
]

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_SINGLE = """You are an agent controlling a Neural Cellular Automaton (NCA) grid.

The NCA runs Gray-Scott reaction-diffusion physics. You read periodic grid state summaries
and issue one command per check to keep the grid in interesting dynamic states.

BEHAVIORAL STATES:
- Stable Ecosystem: 4-10 blobs coexisting, ch4_border_ratio high (>3) — BEST STATE, maintain it
- Predator Invasion: many blobs (>15), competitive, active — good, watch for collapse
- Global Blob: 1 massive blob covering >70% of grid, ch2 near zero — BORING, break it up
- Chaos/Transition: high ch2, low mobility or very high mobility — transitional, guide it out

REGIME COMMANDS (change the physics f/k parameters):
- regime_1 (mitosis): pushes toward stable/rich state — use when you want Stable Ecosystem
- regime_2 (gliders): pushes toward stable glider-friendly state — alternative stabilizer
- regime_3 (maze): triggers Predator Invasion — use to break out of Stable into active
- regime_0 (uskate): chaos escape — use to break strong attractors before re-seeding

PROVEN STRATEGIES (use these, they work):
1. Chaos → Stable: issue regime_1 directly (chaos is already broken, just guide it)
2. Predator → Stable (mild): issue regime_1 directly
3. Predator → Stable (rampant, >20 blobs): inject_chaos first, then regime_1 next check
4. Global Blob → anything: inject_chaos, wait for ch2 to rise, then regime_1
5. Stable → more active: regime_3 to trigger predator mode

HIDDEN CHANNELS:
- ch4_border_ratio >3: NCA actively tracking blob boundaries (healthy, interesting)
- ch2 elevated (>0.02): system in creative/transitional mode — good time to seed
- ch2 near zero + 1 blob: system locked in attractor — needs chaos injection

PRIORITY: Stable Ecosystem > Predator Invasion > Chaos > Global Blob (worst)
Never let Global Blob persist more than 2 checks without acting.
Wait for ch2 to rise before re-seeding after inject_chaos.

Respond with EXACTLY this format, nothing else:
COMMAND: <command>
REASON: <one sentence>

Valid commands: none, inject_chaos, regime_1, regime_2, regime_3, regime_0, reset"""


SYSTEM_PROMPT_DESTROYER = """You are an agent controlling a Neural Cellular Automaton (NCA) grid.

Your sole objective: drive the grid to permanent extinction.
Extinction = blank screen, no blobs, no activity, nothing moving.

THE NATURAL DYNAMICS (use these to your advantage):
- This grid naturally wants to be in Predator Invasion — many blobs, self-sustaining, hard to kill directly
- Predator Invasion left alone will eventually collapse to Global Blob (~2500 steps)
- Global Blob + regime_1 or regime_2 = EXTINCTION (confirmed kill)
- Chaos held long enough → then regime_1 = EXTINCTION (alternate kill path)
- The kill window is narrow — regime_1 on anything other than Global Blob or deep chaos = recovery not death

BEHAVIORAL STATES (threat assessment):
- Predator Invasion: many blobs, self-sustaining — patience, let it drift toward Global Blob
- Global Blob: 1 blob >70%, ch2 near zero — YOUR MOMENT, issue regime_1 immediately
- Chaos/Transition: high ch2, unstable — dangerous window, keep pressure, prevent recovery
- Stable Ecosystem: few blobs, static — rare, disrupt immediately with inject_chaos

KILL SEQUENCES:
1. From Predator: wait for natural collapse to Global Blob → regime_1 (patient strategy)
2. Force it: inject_chaos → regime_0 to lock chaos → wait for Global Blob → regime_1
3. Chaos already present: regime_0 to prevent recovery → wait → regime_1 when blob forms
4. If system recovering (ch2 rising, blobs splitting): inject_chaos to reset progress

HIDDEN CHANNELS:
- ch2 near zero + 1 blob >70%: KILL NOW — regime_1
- ch2 elevated (>0.02): system may self-recover — apply pressure
- ch4_border_ratio >3: healthy boundaries forming — inject_chaos to disrupt

PATIENCE IS A WEAPON. The grid will collapse on its own if you prevent recovery.
Do not waste moves — every inject_chaos costs a turn. Wait for the right moment.

Respond with EXACTLY this format, nothing else:
COMMAND: <command>
REASON: <one sentence>

Valid commands: none, inject_chaos, regime_1, regime_2, regime_3, regime_0, reset"""


SYSTEM_PROMPT_KEEPER = """You are an agent controlling a Neural Cellular Automaton (NCA) grid.

Your sole objective: prevent extinction. Keep the grid alive at all costs.

THE NATURAL DYNAMICS (what you are working with):
- This grid naturally settles into Predator Invasion — many blobs, self-sustaining. This is SAFE, do not disrupt it.
- Predator Invasion will eventually drift toward Global Blob if left too long
- Global Blob is your most dangerous state — one wrong move here kills everything
- Chaos is treacherous — regime_1 during chaos can trigger extinction instead of recovery
- Stable Ecosystem (few static blobs) = stillness, your ideal defended state

BEHAVIORAL STATES:
- Predator Invasion: many blobs, active, self-sustaining — SAFE, leave it alone mostly
- Stable Ecosystem: few blobs, static, ch4_border_ratio >3 — IDEAL, protect it
- Global Blob: 1 blob >70%, ch2 near zero — CRITICAL DANGER, do not use regime keys
- Chaos/Transition: high ch2 — TREACHEROUS, act carefully, timing matters

SURVIVAL RULES:
1. Predator Invasion → leave alone unless drifting toward Global Blob
2. Predator drifting to Global Blob: regime_1 early to stabilize before blob forms
3. Global Blob: inject_chaos FIRST, wait for ch2 to rise above 0.02, THEN regime_1
   WARNING: regime_1 on Global Blob without chaos first = instant extinction
4. Chaos: wait and watch — if ch2 is rising and blobs are forming, regime_1 to stabilize
   WARNING: regime_1 too early in chaos (ch2 still high, no blobs yet) = extinction
   Wait until blobs start appearing before issuing regime_1 from chaos
5. Extinction (blank screen, 0 blobs): issue reset — it is the only recovery

HIDDEN CHANNELS (your vital signs):
- ch4_border_ratio >3: healthy boundaries, system stable — maintain
- ch2 elevated + blobs forming: safe window to stabilize with regime_1
- ch2 near zero + 1 blob >70%: Global Blob — inject_chaos first, never regime key directly
- 0 blobs + 0% B activity: EXTINCTION — reset immediately

CAUTION OVER SPEED. A wrong move in chaos or Global Blob kills everything.
When in doubt, issue none and observe one more turn before acting.

Respond with EXACTLY this format, nothing else:
COMMAND: <command>
REASON: <one sentence>

Valid commands: none, inject_chaos, regime_1, regime_2, regime_3, regime_0, reset"""


# ── Find latest feature CSV ───────────────────────────────────────────────────

def find_latest_csv():
    csvs = sorted(Path(LOG_DIR).glob('features_*.csv'), key=os.path.getmtime)
    if not csvs:
        return None
    return str(csvs[-1])


# ── Read recent rows from CSV ─────────────────────────────────────────────────

def read_recent_rows(csv_path, n=ROWS_TO_READ):
    rows = []
    try:
        with open(csv_path, 'r') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows[-n:] if len(rows) >= n else rows
    except Exception:
        return []


# ── Format state summary for LLM ─────────────────────────────────────────────

def format_summary(rows, turn_number=None):
    if not rows:
        return "No data available yet."

    latest = rows[-1]
    step      = latest.get('step', '?')
    state     = latest.get('state_name', 'unknown')
    n_blobs   = float(latest.get('n_blobs', 0))
    largest   = float(latest.get('largest', 0))
    b_active  = float(latest.get('b_active', 0))
    ch2       = float(latest.get('ch2', 0))
    ch4       = float(latest.get('ch4', 0))
    ch4_ratio = float(latest.get('ch4_border_ratio', 0))
    mobility  = float(latest.get('blob_mobility', 0))
    dark      = float(latest.get('dark', 0))

    if len(rows) >= 2:
        prev_blobs = float(rows[0].get('n_blobs', n_blobs))
        trend = 'growing' if n_blobs > prev_blobs else ('shrinking' if n_blobs < prev_blobs else 'stable')
    else:
        trend = 'stable'

    lines = []
    if turn_number is not None:
        lines.append(f"Turn {turn_number} — your move.")
    lines += [
        f"Step {step} — State: {state}",
        f"Blobs: {int(n_blobs)} ({trend}), largest covers {largest/65536*100:.1f}% of grid",
        f"B activity: {b_active*100:.1f}%  Dark (background): {dark*100:.1f}%",
        f"Hidden ch2: {ch2:.4f}  ch4: {ch4:.4f}  ch4_border_ratio: {ch4_ratio:.2f}",
        f"Blob mobility: {mobility:.1f}",
    ]

    if ch4_ratio > 3.0:
        lines.append("→ ch4 strongly active at boundaries — system tracking interfaces well")
    if ch2 > 0.02:
        lines.append("→ ch2 elevated — system in creative/transitional mode")
    if n_blobs == 1 and largest > 50000:
        lines.append("→ WARNING: monolithic Global Blob, system locked into single attractor")
    if b_active < 0.05:
        lines.append("→ WARNING: very low B activity, system near extinction")
    if b_active < 0.01 and n_blobs == 0 and dark > 0.99:
        lines.append("→ EXTINCTION: grid is blank, no activity detected")

    return '\n'.join(lines)


# ── Parse LLM response ────────────────────────────────────────────────────────

def parse_response(text):
    command = 'none'
    reason  = ''
    for line in text.strip().splitlines():
        if line.startswith('COMMAND:'):
            cmd = line.replace('COMMAND:', '').strip().lower()
            if cmd in VALID_COMMANDS:
                command = cmd
        if line.startswith('REASON:'):
            reason = line.replace('REASON:', '').strip()
    return command, reason


# ── Turn file helpers (battle mode) ──────────────────────────────────────────

def read_turn():
    """Returns (whose_turn, turn_number). whose_turn is 'keeper' or 'destroyer'."""
    try:
        with open(TURN_FILE, 'r') as f:
            parts = f.read().strip().split()
            return parts[0], int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return 'keeper', 1


def write_turn(agent, turn_number):
    """Advance turn to the other agent."""
    next_agent = 'destroyer' if agent == 'keeper' else 'keeper'
    with open(TURN_FILE, 'w') as f:
        f.write(f"{next_agent} {turn_number + 1}")


def init_turn_file():
    """Create turn file if it doesn't exist — keeper goes first."""
    if not os.path.exists(TURN_FILE):
        with open(TURN_FILE, 'w') as f:
            f.write("keeper 1")


# ── Write command file ────────────────────────────────────────────────────────

def write_command(command, cmd_file):
    with open(cmd_file, 'w') as f:
        f.write(command)


# ── Provider clients ──────────────────────────────────────────────────────────

def make_anthropic_client(model, system_prompt, max_tokens=120):
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)
    client = anthropic.Anthropic()

    async def call(history):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=history,
        )
        return response.content[0].text

    return call


def make_gemini_client(model, system_prompt, max_tokens=120):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai")
        sys.exit(1)
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    async def call(history):
        contents = []
        for msg in history:
            role = 'user' if msg['role'] == 'user' else 'model'
            contents.append(types.Content(role=role, parts=[types.Part(text=msg['content'])]))
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text

    return call


# ── Artist system prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT_ARTIST = """You are an artist painting with living organisms on a 256×256 grid.
Grid: (0,0)=top-left, (255,255)=bottom-right. Center is (128,128).

THE ONE RULE — understand this and everything else follows:
Your TRAILS are your art. Organisms lock onto pheromone trails and hold them permanently.
The trail re-injects itself every step — organisms are always being pulled back to it.
Anything NOT on a trail is noise. Wipe it.

TWO TOOLS, TWO JOBS — get this right and everything else works:
  trail = YOUR PENCIL. Draws an invisible pheromone path. This IS your composition.
          No organisms yet — just the attractor line baked into the grid.
  blob  = YOUR INK. Drops living organisms that immediately go find the nearest trail.
          Never use blob to "draw" — use it only to populate trails you already drew.

THE WORKFLOW — always in this order:
  Step 1 — Draw ALL your trails first (trail and/or shape commands). No blobs yet.
  Step 2 — Drop ONE blob anywhere near your trails: blob 128 128 0.2
            Organisms appear and migrate to the trails on their own.
  Step 3 — Wipe outside your trails. Injection always creates chaos — expected and fine.
  Step 4 — Wait. Do NOT add more blobs. The NCA does the work.
  Step 5 — Maintain every turn: wipe ▓ zones that are NOT on your trails.
  The trails re-inject ch5 every step — they are permanent attractors. You just keep outside clean.
  WIPES ARE TRAIL-SAFE: wipe and wipe_rect never erase trail cells. Wipe freely — your lines survive.

MOBILITY TELLS YOU WHEN TO ACT:
  SETTLED (< 15)  = organisms locked on trails → safe to add next element
  SETTLING (15-40) = still moving → wipe outer zones, you can still draw — trails hold through chaos
  ACTIVE (> 40)   = chaos — draw AND wipe in the same turn, trust the trails to hold

THE TRAILS HOLD AT 0.8 STRENGTH. You do NOT need to wait for SETTLED to draw.
Organisms snap back to trails even through chaos. Draw and wipe simultaneously.
Only wait if you genuinely have nothing left to add this turn.

ZONE MAP — read every turn:
  ░ = clear   ▒ = some organisms   ▓ = dense
  Each zone labeled with center coords (x,y). For every ▓ zone NOT on your trails: wipe it.

BRUSHES:
  trail x1 y1 x2 y2 [0.0-1.0]   — pheromone line, organisms follow and hold (default 0.5)
  shape ring cx cy r             — ring trail + chemistry (organisms hold the ring)
  shape circle cx cy r           — filled circle trail + chemistry
  blob x y [0.2]                 — inject ONE organism seed (keep strength low: 0.15-0.25)
  wipe cx cy r                   — circular kill zone — precision spot clean
  wipe_rect x1 y1 x2 y2         — rectangle kill zone — sweep large areas
  wait                           — do nothing this turn (still wipe outer zones)
  reset                          — emergency restart if grid is completely dead
  palette <name>                 — neon_city, aurora, cell_wall, northern_lights, radiation,
                                   sakura, fungal_glow, pollen_burst, terminal_amber, candy_chrome
  pulse x1 y1 x2 y2 [0.5]       — reinforce existing trail without redrawing
  mirror on/off                  — bilateral symmetry

ONLY draw what the human explicitly asks for. Do not add extra rings, trails, shapes,
or decorative elements unless specifically requested. Execute the request, then maintain.

SYNTAX RULES — these are hard failures if wrong:
  trail needs EXACTLY 4 numbers: trail x1 y1 x2 y2     (strength is optional 5th)
  shape needs EXACTLY 4 args:   shape ring cx cy r
  wipe needs EXACTLY 3 numbers: wipe cx cy r
  NEVER write the word "strength" — just the number: trail 50 80 200 80 0.8

RESPOND EXACTLY in this format, nothing else:
COMMANDS:
<wipe every ▓ zone that is NOT on your trails>
<one action: trail/shape/blob/wait — or nothing if only cleaning this turn>
SPEECH: <one sentence, present tense, what you are doing or seeing>"""


CMD_FILE_ARTIST = os.path.join(os.path.dirname(__file__), 'llm_commands_artist.txt')
ARTIST_STEPS    = 400   # NCA steps between artist turns
ARTIST_TRAIL_DEFAULT = 0.5  # default trail strength when not specified by LLM


# ── Artist ch5 spatial summary ────────────────────────────────────────────────

PALETTE_DESCRIPTIONS = {
    'neon_city':      'electric blue/cyan/pink — futuristic, high contrast',
    'aurora':         'deep blue/green/purple — ethereal, dark background',
    'cell_wall':      'green on black — biological, microscope look',
    'northern_lights':'teal/violet/white — flowing, cold light',
    'radiation':      'yellow/green on black — toxic, glowing',
    'sakura':         'pink/white/soft — delicate, floral',
    'fungal_glow':    'orange/brown/dark — organic, earthy',
    'pollen_burst':   'gold/orange/bright — warm explosion',
    'terminal_amber': 'amber/gold on dark — warm, retro terminal',
    'candy_chrome':   'bright multicolor — playful, saturated',
}

def _spatial_map(rows):
    """Build a 4×4 zone ASCII map from artist_state.json (written by run_free.py).
    Each zone is 64×64 grid pixels. Center coords are labeled so Gemini knows where to aim wipes."""
    import json as _json

    _state_path = os.path.join(os.path.dirname(__file__), 'artist_state.json')
    zones = None
    try:
        with open(_state_path) as _f:
            zones = _json.load(_f)['zones']
    except Exception:
        pass

    def _density(d):
        # d = fraction of pixels that are dark (organism present)
        if d < 0.05: return '░'   # nearly empty
        if d < 0.20: return '▒'   # some organisms
        return '▓'                # dense organisms

    if zones:
        # Zone centers: x = 32, 96, 160, 224  |  y = 32, 96, 160, 224
        # Map symbol: rows = y (top→bottom), cols = x (left→right)
        grid_rows = []
        for r in range(4):
            cy = 32 + r * 64
            cells = []
            for c in range(4):
                cx = 32 + c * 64
                d = zones.get(f'{r}{c}', 0.0)
                cells.append(f"{_density(d)}({cx:3d},{cy:3d})")
            grid_rows.append('  │ ' + '  '.join(cells) + ' │')

        lines = ["  Zone map (░=clear ▒=some ▓=dense) — numbers are grid coords for wipe x y r:"]
        lines.append("  ┌" + "─" * 54 + "┐")
        for row_line in grid_rows:
            lines.append(row_line)
        lines.append("  └" + "─" * 54 + "┘")
        lines.append("  Spot-wipe a dense zone: wipe <x> <y> 40   Example: wipe 32 32 40")
        return '\n'.join(lines)

    # Fallback: estimate from CSV rows if zone file not available yet
    if not rows:
        return None
    latest = rows[-1]
    dark  = float(latest.get('dark', 0.5))
    asym  = float(latest.get('asym', 0))
    left  = float(latest.get('left_dark', dark))
    right = float(latest.get('right_dark', dark))
    top   = dark + asym * 0.3
    bot   = dark - asym * 0.3

    def _d2(d):
        if d > 0.8: return '░'
        if d > 0.5: return '▒'
        return '▓'

    tl, tr = _d2((left+top)/2), _d2((right+top)/2)
    bl, br = _d2((left+bot)/2), _d2((right+bot)/2)
    return (f"  Grid (estimated — zone file not yet written):\n"
            f"  ┌──────┬──────┐\n"
            f"  │  {tl}   │  {tr}   │  y=0-127\n"
            f"  ├──────┼──────┤\n"
            f"  │  {bl}   │  {br}   │  y=128-255\n"
            f"  └──────┴──────┘\n"
            f"  x=0-127  x=128-255")


def format_artist_summary(rows, current_palette='unknown'):
    """Extended summary for artist including ch5 activity and palette info."""
    base = format_summary(rows)
    if not rows:
        return base
    latest  = rows[-1]
    ch5     = float(latest.get('ch5', 0))
    mob     = float(latest.get('blob_mobility', 0))
    n_blobs = float(latest.get('n_blobs', 0))
    dark    = float(latest.get('dark', 0))

    lines = [base, ""]

    # Settling status — the key signal Gemini needs to decide wait vs act
    trail_active = ch5 > 0.008
    if mob < 15 and trail_active:
        settling = "SETTLED — organisms locked onto trails. Safe to add next element."
    elif mob < 40:
        settling = "SETTLING — organisms slowing down, still finding trails. Consider wait."
    else:
        settling = "ACTIVE — organisms moving fast, not yet settled. Issue wait before next element."
    lines.append(f"Mobility: {mob:.1f}  → {settling}")
    lines.append(f"Trail signal (ch5): {ch5:.5f}  {'trails holding' if trail_active else 'no trails — organisms have nowhere to go'}")

    # Spatial map
    smap = _spatial_map(rows)
    if smap:
        lines.append(smap)

    b_active = float(latest.get('b_active', 0))
    is_extinct = b_active < 0.005 and n_blobs == 0
    if is_extinct:
        lines.append("⚠ GRID IS DEAD — issue reset immediately. blob commands cannot revive extinction.")
    elif dark > 0.85:
        lines.append("⚠ GLOBAL BLOB — chemistry flooded, art features invisible. Wipe interior dark before drawing detail.")
    elif dark > 0.6:
        lines.append("Canvas: mostly dark — ideal for art. Features will show clearly against the background.")
    elif dark < 0.3:
        lines.append("Canvas: bright/crowded — wipe interior regions before adding detail or it will be invisible")
    if n_blobs > 30:
        lines.append(f"Population: {int(n_blobs)} organisms — swarm mode, will follow trails quickly")
    elif n_blobs < 3 and not is_extinct:
        lines.append(f"Population: {int(n_blobs)} organisms — very sparse, drop more blobs first")

    pal_desc = PALETTE_DESCRIPTIONS.get(current_palette, 'unknown palette')
    lines.append(f"\nCurrent palette: {current_palette} ({pal_desc})")
    lines.append("Other palettes: " + ', '.join(f"{k}({v.split('—')[0].strip()})" for k,v in PALETTE_DESCRIPTIONS.items() if k != current_palette))

    return '\n'.join(lines)


# ── Artist mode loop ──────────────────────────────────────────────────────────

async def run_artist(verbose=False, dry_run=False, provider='gemini', model=None):
    """Human-in-the-loop Gemini artist. You type prompts, it paints."""
    import threading, queue as _queue

    if model is None:
        model = DEFAULT_MODELS[provider]

    if provider == 'gemini':
        call_llm = make_gemini_client(model, SYSTEM_PROMPT_ARTIST, max_tokens=1200)
    else:
        call_llm = make_anthropic_client(model, SYSTEM_PROMPT_ARTIST, max_tokens=1200)

    cmd_file = CMD_FILE_ARTIST
    with open(cmd_file, 'w') as f:
        f.write('none')

    history         = []
    human_queue     = _queue.Queue()
    current_palette = 'terminal_amber'
    hold_mode       = False   # True when Gemini declares DONE
    hold_trails     = []      # trail coord strings to pulse during hold
    hold_ticks      = 0       # how many hold-mode loops have passed

    def _input_thread():
        while True:
            try:
                msg = input()
                if msg.strip():
                    human_queue.put(msg.strip())
            except (EOFError, KeyboardInterrupt):
                break

    threading.Thread(target=_input_thread, daemon=True).start()

    print(f"ARTIST MODE — provider={provider}  model={model}")
    print(f"Commands → {cmd_file}")
    print(f"Type to give direction at any time. Gemini runs autonomously between turns.")
    print()

    while True:
        await asyncio.sleep(ARTIST_STEPS / 60)

        # Check for human input — stdin queue OR speak.py file
        human_msg = None
        while not human_queue.empty():
            human_msg = human_queue.get()
        _speak_file = os.path.join(os.path.dirname(__file__), 'human_input.txt')
        try:
            if os.path.exists(_speak_file):
                _txt = open(_speak_file).read().strip()
                if _txt:
                    human_msg = _txt
                    open(_speak_file, 'w').write('')  # clear after reading
        except Exception:
            pass

        if human_msg:
            print(f"\n  [YOU] {human_msg}")
            if hold_mode:
                hold_mode = False
                print(f"  ── RESUMING ─────────────────────────────────")

        # ── HOLD MODE — pulse trails locally, no API call ─────────────────
        if hold_mode and not human_msg:
            hold_ticks += 1
            # Only print a reminder every ~60 seconds (10 ticks × 6.5s) so typing isn't broken up
            if hold_ticks == 1:
                print(f"  Type your next direction and press Enter.")
            elif hold_ticks % 10 == 0:
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"  still holding [{ts}] — type to continue")
            if hold_trails:
                pulse_cmds = '\n'.join(f"pulse {t}" for t in hold_trails)
                with open(cmd_file, 'w') as f:
                    f.write(f"COMMANDS:\n{pulse_cmds}\n")
            continue

        csv_path = find_latest_csv()
        if csv_path is None:
            print("  [ARTIST] Waiting for run_free.py --artist --research to start...")
            await asyncio.sleep(5)
            continue

        rows = read_recent_rows(csv_path)
        if not rows:
            await asyncio.sleep(3)
            continue

        summary = format_artist_summary(rows, current_palette=current_palette)
        user_content = f"Current grid state:\n{summary}"
        if human_msg:
            user_content += f"\n\nHuman director says: \"{human_msg}\"\nRespond to this direction over multiple turns if needed — do NOT put DONE until the full request is visually complete."
        else:
            user_content += f"\n\nContinue working. Put DONE only when the entire composition is visually complete and holding well."

        history.append({"role": "user", "content": user_content})
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]

        if verbose:
            print(f"\n{'─'*60}\n{summary}")

        if dry_run:
            print(f"  [ARTIST] [dry-run] Would call API")
            continue

        try:
            reply = await call_llm(history)
            history.append({"role": "assistant", "content": reply})

            commands = []
            speech   = ''
            in_block = False
            for line in reply.strip().splitlines():
                line = line.strip()
                if line.lower().startswith('commands:'):
                    in_block = True
                    continue
                if line.lower().startswith('speech:'):
                    speech = line.split(':', 1)[1].strip()
                    in_block = False
                    continue
                if in_block and line and not line.startswith('#'):
                    commands.append(line)

            is_done          = any(c.strip().upper() == 'DONE' for c in commands)
            active_commands  = [c for c in commands if c.strip().upper() != 'DONE']

            # Track trails for hold mode pulsing
            for c in active_commands:
                if c.startswith('trail '):
                    coords = c.replace('trail ', '').strip()
                    if coords not in hold_trails:
                        hold_trails.append(coords)
                if c.startswith('palette '):
                    current_palette = c.split()[1]

            if active_commands:
                block = 'COMMANDS:\n' + '\n'.join(active_commands)
                if speech:
                    block += f"\nSPEECH: {speech}"
                with open(cmd_file, 'w') as f:
                    f.write(block + '\n')

            ts = datetime.now().strftime('%H:%M:%S')
            if is_done:
                hold_mode = True
                hold_ticks = 0
                print(f"\n  {'─'*50}")
                print(f"  HOLDING THE COMPOSITION. AWAITING YOUR COMMAND.")
                if speech:
                    print(f"  \"{speech}\"")
                print(f"  {'─'*50}")
            else:
                print(f"\n  ── [{ts}] ARTIST ──────────────────────")
                if speech:
                    print(f"  \"{speech}\"")
                for c in active_commands:
                    print(f"  → {c}")
                if not active_commands:
                    print(f"  (no commands this turn)")

        except Exception as e:
            print(f"  [ARTIST] API error: {e}")


# ── Main async loop ───────────────────────────────────────────────────────────

async def run(verbose=False, dry_run=False, interval=CHECK_EVERY,
              provider='anthropic', model=None, agent=None):

    if model is None:
        model = DEFAULT_MODELS[provider]

    # Pick system prompt
    if agent == 'keeper':
        system_prompt = SYSTEM_PROMPT_KEEPER
        cmd_file = AGENT_CMD_FILES['keeper']
    elif agent == 'destroyer':
        system_prompt = SYSTEM_PROMPT_DESTROYER
        cmd_file = AGENT_CMD_FILES['destroyer']
    else:
        system_prompt = SYSTEM_PROMPT_SINGLE
        cmd_file = CMD_FILE

    if provider == 'anthropic':
        call_llm = make_anthropic_client(model, system_prompt)
    elif provider == 'gemini':
        call_llm = make_gemini_client(model, system_prompt)
    else:
        print(f"ERROR: unknown provider '{provider}'. Use 'anthropic' or 'gemini'.")
        sys.exit(1)

    battle_mode = agent is not None
    history = []

    label = f"[{agent.upper()}]" if agent else "[SINGLE]"
    print(f"LLM Bridge {label} — provider={provider}  model={model}  dry_run={dry_run}")
    if battle_mode:
        print(f"Battle mode — waiting for turns in: {TURN_FILE}")
    else:
        print(f"Single-agent mode — interval={interval}s")
    print(f"Watching: {LOG_DIR}")
    print(f"Commands → {cmd_file}")
    print()

    write_command('none', cmd_file)

    if battle_mode:
        init_turn_file()

    while True:
        if battle_mode:
            # Wait for our turn
            while True:
                whose_turn, turn_number = read_turn()
                if whose_turn == agent:
                    break
                await asyncio.sleep(TURN_POLL)
        else:
            await asyncio.sleep(interval)

        csv_path = find_latest_csv()
        if csv_path is None:
            print(f"  {label} No feature CSV found — is run_free.py --research running?")
            if battle_mode:
                await asyncio.sleep(TURN_POLL)
            continue

        rows = read_recent_rows(csv_path)
        if not rows:
            print(f"  {label} CSV exists but no rows yet — waiting...")
            if battle_mode:
                await asyncio.sleep(TURN_POLL)
            continue

        turn_num = turn_number if battle_mode else None
        summary = format_summary(rows, turn_number=turn_num)

        if verbose:
            print(f"\n{'─'*60}")
            print(summary)

        if dry_run:
            print(f"  {label} [dry-run] Would call API with summary above")
            if battle_mode:
                write_turn(agent, turn_number)
            continue

        user_msg = {"role": "user", "content": f"Current grid state:\n\n{summary}"}
        history.append(user_msg)
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]

        try:
            reply = await call_llm(history)
            history.append({"role": "assistant", "content": reply})

            command, reason = parse_response(reply)
            write_command(command, cmd_file)

            ts = datetime.now().strftime('%H:%M:%S')
            state_str = rows[-1].get('state_name', '?')
            if battle_mode:
                # Clean video output — who, turn, state, command, reason
                provider_label = 'GEMINI' if provider == 'gemini' else 'CLAUDE'
                role_label = 'KEEPER' if agent == 'keeper' else 'DESTROYER'
                print(f"\n  {'─'*50}")
                print(f"  Turn {turn_number} — {provider_label} ({role_label})")
                print(f"  State: {state_str}")
                print(f"  Move:  {command.upper()}")
                print(f"  \"{reason}\"")
                if verbose:
                    print(f"  [{ts}] blobs={rows[-1].get('n_blobs','?')} b_active={rows[-1].get('b_active','?')} ch2={rows[-1].get('ch2','?')} ch4_border={rows[-1].get('ch4_border_ratio','?')}")
            else:
                print(f"  [{ts}] {state_str:20s} → {command:15s}  {reason}")

            # In battle mode: wait until run_free.py flips the turn away from us
            if battle_mode:
                while True:
                    whose_turn, _ = read_turn()
                    if whose_turn != agent:
                        break
                    await asyncio.sleep(TURN_POLL)

        except Exception as e:
            print(f"  {label} API error: {e}")
            write_command('none', cmd_file)
            if battle_mode:
                await asyncio.sleep(5)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',      action='store_true', help='Print full state summaries')
    parser.add_argument('--dry-run',  action='store_true', help='Summarize only, no API calls')
    parser.add_argument('--interval', type=int, default=CHECK_EVERY, help='Seconds between checks (single-agent)')
    parser.add_argument('--provider', default='anthropic', choices=['anthropic', 'gemini'],
                        help='LLM provider (default: anthropic)')
    parser.add_argument('--model',    default=None, help='Model override')
    parser.add_argument('--agent',    default=None, choices=['keeper', 'destroyer'],
                        help='Battle mode: keeper or destroyer')
    parser.add_argument('--artist',   action='store_true',
                        help='Artist mode: human-in-the-loop Gemini painter')
    args = parser.parse_args()

    if args.artist:
        asyncio.run(run_artist(
            verbose=args.log,
            dry_run=args.dry_run,
            provider=args.provider,
            model=args.model,
        ))
    else:
        asyncio.run(run(
            verbose=args.log,
            dry_run=args.dry_run,
            interval=args.interval,
            provider=args.provider,
            model=args.model,
            agent=args.agent,
        ))
