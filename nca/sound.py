# nca/sound.py — Ambient sound engine for Somnivex.
#
# Two independent drones tracking different hidden channel groups.
# Spatial stereo follows blob center of mass left/right.
# Bell fires on absorption/collision events (rolling-window detection).
# Slow LFO breathing so the sound feels alive even during calm phases.

import numpy as np
from collections import deque

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except (ImportError, OSError):
    _SD_AVAILABLE = False
    print("sounddevice/PortAudio not available — audio disabled.")

SAMPLE_RATE = 44100
BLOCK_SIZE  = 1024   # ~23ms

# ── Frequency ranges ──────────────────────────────────────────────────────────
# Drone 1: tracks mean of hidden ch2-6  (lower, slower)
D1_FREQ_MIN = 110.0   # A2
D1_FREQ_MAX = 330.0   # E4

# Drone 2: tracks mean of hidden ch7-11 (higher, moves independently)
D2_FREQ_MIN = 165.0   # E3
D2_FREQ_MAX = 495.0   # B4

BELL_FREQ   = 432.0   # event bell
BELL_DECAY  = 0.9996  # ~2.5s tail

# ── Volumes ───────────────────────────────────────────────────────────────────
D1_VOL   = 0.10
D2_VOL   = 0.08
BELL_VOL = 0.12

# ── Slow volume LFO — the sound "breathes" at 0.06Hz (~17s per cycle) ────────
LFO_RATE  = 0.06
LFO_DEPTH = 0.25

# ── Pitch LFOs — each drone wanders independently so sound never settles ──────
# Out of phase and different rates — they beat against each other unpredictably
D1_PITCH_LFO_RATE  = 0.05    # Hz — one wander every ~20s
D1_PITCH_LFO_DEPTH = 25.0    # ±25Hz
D2_PITCH_LFO_RATE  = 0.077   # Hz — prime-ish ratio to D1, never syncs
D2_PITCH_LFO_DEPTH = 18.0    # ±18Hz

# ── Smoothing ─────────────────────────────────────────────────────────────────
D1_ALPHA  = 0.03
D2_ALPHA  = 0.06
PAN_ALPHA = 0.05

# ── Event detection ───────────────────────────────────────────────────────────
# Track B channel std over a rolling window.
# Bell fires when current std deviates from rolling mean by > threshold.
EVENT_WINDOW  = 30    # frames (~1s at 30fps)
EVENT_THRESH  = 0.025 # deviation needed to trigger bell


class SoundEngine:
    def __init__(self):
        # Targets (written by main thread)
        self._t_d1_freq  = 220.0
        self._t_d2_freq  = 330.0
        self._t_pan      = 0.5
        self._t_vol      = 0.8
        self._bell_pend  = False
        self._muted      = False

        # Audio thread state
        self._d1_freq    = 220.0
        self._d2_freq    = 330.0
        self._pan        = 0.5
        self._ph_d1        = 0.0
        self._ph_d2        = 0.0
        self._ph_bell      = 0.0
        self._ph_lfo       = 0.0
        self._ph_pitch_d1  = 0.0
        self._ph_pitch_d2  = np.pi   # start out of phase with D1
        self._bell_amp     = 0.0

        # Event detection (main thread)
        self._b_std_hist = deque(maxlen=EVENT_WINDOW)

        self._stream = None

    def start(self):
        if not _SD_AVAILABLE:
            return
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype='float32',
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()
        print("Sound engine started.  A=mute/unmute")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def toggle_mute(self):
        self._muted = not self._muted
        print(f"Sound {'muted' if self._muted else 'unmuted'}")
        return not self._muted

    def update(self, grid_np):
        """Call once per frame. grid_np: (H, W, 16) numpy array."""
        if not _SD_AVAILABLE or self._muted:
            return

        W      = grid_np.shape[1]
        hidden = grid_np[:, :, 2:13]
        B      = grid_np[:, :, 1]

        # Drone 1 — ch2-6 mean → frequency
        h1 = float(np.tanh(np.mean(hidden[:, :, 0:5]) * 3.0))
        self._t_d1_freq = D1_FREQ_MIN + (h1 + 1.0) * 0.5 * (D1_FREQ_MAX - D1_FREQ_MIN)

        # Drone 2 — ch7-11 mean → frequency (independent movement)
        h2 = float(np.tanh(np.mean(hidden[:, :, 5:10]) * 3.0))
        self._t_d2_freq = D2_FREQ_MIN + (h2 + 1.0) * 0.5 * (D2_FREQ_MAX - D2_FREQ_MIN)

        # Stereo pan — center of mass of |B| along x axis
        B_abs   = np.abs(B)
        b_total = np.sum(B_abs) + 1e-8
        x_norm  = np.linspace(0.0, 1.0, W, dtype=np.float32)
        self._t_pan = float(np.sum(B_abs * x_norm[None, :]) / b_total)

        # Volume — activity level
        b_mean = float(np.mean(B_abs))
        self._t_vol = float(np.clip(0.4 + b_mean * 4.0, 0.3, 1.0))

        # Event detection — rolling window on B std
        b_std = float(np.std(B))
        self._b_std_hist.append(b_std)
        if len(self._b_std_hist) == EVENT_WINDOW:
            roll_mean = float(np.mean(self._b_std_hist))
            if abs(b_std - roll_mean) > EVENT_THRESH:
                self._bell_pend = True

    def _callback(self, outdata, frames, time_info, status):
        t_d1  = self._t_d1_freq
        t_d2  = self._t_d2_freq
        t_pan = self._t_pan
        t_vol = 0.0 if self._muted else self._t_vol

        if self._bell_pend:
            self._bell_amp = BELL_VOL
            self._bell_pend = False

        n = np.arange(frames, dtype=np.float64)

        # Smooth tracking
        self._d1_freq += (t_d1 - self._d1_freq) * D1_ALPHA
        self._d2_freq += (t_d2 - self._d2_freq) * D2_ALPHA
        self._pan     += (t_pan - self._pan)     * PAN_ALPHA

        # Pitch LFOs — independent wander on each drone
        p1_dp = 2.0 * np.pi * D1_PITCH_LFO_RATE / SAMPLE_RATE
        p2_dp = 2.0 * np.pi * D2_PITCH_LFO_RATE / SAMPLE_RATE
        pitch_offset_d1 = np.sin(self._ph_pitch_d1) * D1_PITCH_LFO_DEPTH
        pitch_offset_d2 = np.sin(self._ph_pitch_d2) * D2_PITCH_LFO_DEPTH
        self._ph_pitch_d1 = (self._ph_pitch_d1 + frames * p1_dp) % (2.0 * np.pi)
        self._ph_pitch_d2 = (self._ph_pitch_d2 + frames * p2_dp) % (2.0 * np.pi)

        d1 = max(self._d1_freq + pitch_offset_d1, 20.0)
        d2 = max(self._d2_freq + pitch_offset_d2, 20.0)
        pan = float(np.clip(self._pan, 0.0, 1.0))

        # LFO — slow breathing
        lfo_dp   = 2.0 * np.pi * LFO_RATE / SAMPLE_RATE
        lfo_env  = 1.0 - LFO_DEPTH + LFO_DEPTH * np.sin(self._ph_lfo + n * lfo_dp)
        self._ph_lfo = (self._ph_lfo + frames * lfo_dp) % (2.0 * np.pi)

        # Drone 1 (root + fifth + octave)
        dp1 = 2.0 * np.pi * d1 / SAMPLE_RATE
        sig1 = (
            np.sin(self._ph_d1 + n * dp1) * 0.60 +
            np.sin(self._ph_d1 + n * dp1 * 1.5) * 0.25 +
            np.sin(self._ph_d1 + n * dp1 * 2.0) * 0.15
        ) * D1_VOL
        self._ph_d1 = (self._ph_d1 + frames * dp1) % (2.0 * np.pi)

        # Drone 2 (root + third)
        dp2 = 2.0 * np.pi * d2 / SAMPLE_RATE
        sig2 = (
            np.sin(self._ph_d2 + n * dp2) * 0.70 +
            np.sin(self._ph_d2 + n * dp2 * 1.25) * 0.30
        ) * D2_VOL
        self._ph_d2 = (self._ph_d2 + frames * dp2) % (2.0 * np.pi)

        # Bell
        bell_sig = np.zeros(frames)
        if self._bell_amp > 1e-6:
            b_dp     = 2.0 * np.pi * BELL_FREQ / SAMPLE_RATE
            decay    = self._bell_amp * (BELL_DECAY ** n)
            bell_sig = np.sin(self._ph_bell + n * b_dp) * decay
            self._ph_bell = (self._ph_bell + frames * b_dp) % (2.0 * np.pi)
            self._bell_amp *= (BELL_DECAY ** frames)

        mono = (sig1 + sig2 + bell_sig) * lfo_env * t_vol

        # Constant-power pan
        outdata[:, 0] = (mono * np.sqrt(1.0 - pan)).astype(np.float32)
        outdata[:, 1] = (mono * np.sqrt(pan)).astype(np.float32)
