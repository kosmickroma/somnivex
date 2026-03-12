# display/windows.py — window setup, rendering modes, and visual effects.
#
# RENDER MODES — different ways to read the Gray-Scott state:
#   "B"            — raw B concentration (classic, bright = active B)
#   "A_inv"        — inverted A (bright where A was consumed = where B is)
#   "edges"        — gradient magnitude of B (bright = structure boundaries/fronts)
#   "reaction"     — A*B² (the actual reaction rate — only lights up active zones)
#   "differential" — (A-B) normalized (high contrast at the A/B interface)
#
# EFFECTS — post-processing applied after palette mapping:
#   "none"       — clean output
#   "bloom"      — bright spots glow outward (organic, dreamy)
#   "vignette"   — edges darkened, draws focus to center
#   "chromatic"  — R/B channels shifted apart (subtle psychedelic fringe)
#   "grain"      — film grain noise (gritty, analog feel)
#   "scanlines"  — faint horizontal lines (retro CRT look)

import os
import numpy as np
import pygame
from config import DISPLAY_W, DISPLAY_H, DUAL_SCREEN, SCREEN_W, SCREEN_H

RENDER_MODES = ["B", "A_inv", "edges", "reaction", "differential"]
EFFECTS      = ["none", "bloom", "vignette", "chromatic", "grain", "scanlines"]


# ── Half-screen surface cache (dual mode) ─────────────────────────────────────
# Pre-allocated so we don't create a new Surface every frame.
_half_surf = None

def _get_half_surf():
    global _half_surf
    if _half_surf is None:
        _half_surf = pygame.Surface((SCREEN_W, SCREEN_H))
    return _half_surf


# ── Window ────────────────────────────────────────────────────────────────────
def init_window():
    """
    Single fullscreen window (DUAL_SCREEN=False) or one wide NOFRAME window
    spanning both monitors (DUAL_SCREEN=True, positions at 0,0 → 3840x1080).
    """
    pygame.init()
    if DUAL_SCREEN:
        os.environ.setdefault('SDL_VIDEO_WINDOW_POS', '0,0')
        screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.NOFRAME)
    else:
        screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), pygame.FULLSCREEN)
    pygame.display.set_caption("Somnivex")
    return screen


# ── Internal blur helper ──────────────────────────────────────────────────────
def _blur3(a):
    """
    One pass of 3x3 mean blur. Same-size output, edges copied unchanged.
    Works on (H, W) or (H, W, C) arrays. No dependencies beyond numpy.
    """
    a = a.astype(np.float32)
    out = a.copy()
    out[1:-1, 1:-1] = (
        a[:-2, :-2] + a[:-2, 1:-1] + a[:-2, 2:] +
        a[1:-1, :-2] + a[1:-1, 1:-1] + a[1:-1, 2:] +
        a[2:, :-2] + a[2:, 1:-1] + a[2:, 2:]
    ) / 9.0
    return out


# ── Vignette mask cache ────────────────────────────────────────────────────────
_vignette_cache = {}

def _get_vignette(H, W):
    """
    Precompute and cache a radial vignette mask for (H, W).
    Center = 1.0 (full brightness), edges fade to 0.15.
    """
    if (H, W) not in _vignette_cache:
        ys = np.linspace(-1, 1, H)
        xs = np.linspace(-1, 1, W)
        xx, yy = np.meshgrid(xs, ys)
        dist = np.sqrt(xx**2 + yy**2)
        mask = np.clip(1.0 - dist * 0.75, 0.15, 1.0).astype(np.float32)
        _vignette_cache[(H, W)] = mask[:, :, np.newaxis]
    return _vignette_cache[(H, W)]


# ── Scanline mask cache ────────────────────────────────────────────────────────
_scanline_cache = {}

def _get_scanlines(H, W):
    """Alternating row mask: even rows slightly dimmed (CRT scanline look)."""
    if (H, W) not in _scanline_cache:
        mask = np.ones((H, W, 1), dtype=np.float32)
        mask[::2, :, :] = 0.75
        _scanline_cache[(H, W)] = mask
    return _scanline_cache[(H, W)]


# ── Render modes ──────────────────────────────────────────────────────────────
_levels_cache = {}   # key → (lo, hi, frame_counter)
_levels_frame  = [0] # global frame counter

def _auto_levels(heat, cache_key):
    """
    Stretch heat to fill [0, 1] using 2nd/98th percentile.
    Percentile is recomputed every 5 frames and cached — sorting 65k values
    every frame at 15fps adds up, especially with two screens.
    """
    _levels_frame[0] += 1
    cached = _levels_cache.get(cache_key)
    if cached is None or (_levels_frame[0] - cached[2]) >= 5:
        lo = float(np.percentile(heat, 2))
        hi = float(np.percentile(heat, 98))
        _levels_cache[cache_key] = (lo, hi, _levels_frame[0])
    else:
        lo, hi, _ = cached
    if hi - lo < 0.01:
        return heat
    return np.clip((heat - lo) / (hi - lo), 0, 1)


def compute_heat(A_np, B_np, render_mode):
    """
    Extract a (H, W) float32 heat map in [0, 1] from the GS state arrays.
    Auto-levels is applied to all modes so the full palette is always used —
    prevents solid-color screens when a mode's natural range is very narrow.
    """
    if render_mode == "B":
        heat = B_np

    elif render_mode == "A_inv":
        heat = 1.0 - A_np

    elif render_mode == "edges":
        gx   = np.gradient(B_np, axis=1)
        gy   = np.gradient(B_np, axis=0)
        heat = np.sqrt(gx**2 + gy**2)

    elif render_mode == "reaction":
        heat = A_np * B_np * B_np * 6.0

    elif render_mode == "differential":
        heat = (A_np - B_np) * 0.5 + 0.5

    else:
        heat = B_np

    return _auto_levels(heat, render_mode)


# ── Palette mapping ───────────────────────────────────────────────────────────
def apply_palette_heat(heat, palette):
    """
    Map a (H, W) heat array [0, 1] through a 4-color palette.
    Returns (H, W, 3) uint8.

    palette: numpy float array (4, 3) in 0–255 range.
    Heat 0.0 → palette[0], 0.33 → palette[1], 0.66 → palette[2], 1.0 → palette[3].
    """
    p      = palette / 255.0
    t      = (heat * 3.0).clip(0, 3)
    idx_lo = np.floor(t).astype(np.int32).clip(0, 2)
    idx_hi = (idx_lo + 1).clip(0, 3)
    frac   = (t - idx_lo)[:, :, np.newaxis]
    color  = p[idx_lo] + frac * (p[idx_hi] - p[idx_lo])
    return (color * 255).clip(0, 255).astype(np.uint8)


# ── Visual effects ─────────────────────────────────────────────────────────────
def apply_effect(rgb, effect):
    """
    Apply a post-processing visual effect to an (H, W, 3) uint8 image.
    Returns (H, W, 3) uint8.
    """
    if effect == "none":
        return rgb

    elif effect == "bloom":
        # Isolate bright pixels (> 40% brightness), blur them out, add back.
        # Makes bright spots glow softly into surrounding darkness.
        f = rgb.astype(np.float32) / 255.0
        bright = np.maximum(f - 0.4, 0) * (1.0 / 0.6)
        for _ in range(4):
            bright = _blur3(bright)
        result = np.clip(f + bright * 0.9, 0, 1)
        return (result * 255).astype(np.uint8)

    elif effect == "vignette":
        H, W = rgb.shape[:2]
        mask = _get_vignette(H, W)
        return (rgb.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)

    elif effect == "chromatic":
        # Shift R channel right, B channel left by a few pixels.
        # Creates a subtle color fringe around high-contrast edges.
        out   = rgb.copy()
        shift = 4
        out[:, shift:, 0]  = rgb[:, :-shift, 0]   # R: shift right
        out[:, :shift, 0]  = rgb[:, 0:1, 0]
        out[:, :-shift, 2] = rgb[:, shift:, 2]     # B: shift left
        out[:, -shift:, 2] = rgb[:, -1:, 2]
        return out

    elif effect == "grain":
        # Film grain: random noise ±12 per channel.
        noise  = np.random.normal(0, 12, rgb.shape).astype(np.float32)
        return (rgb.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

    elif effect == "scanlines":
        H, W = rgb.shape[:2]
        mask = _get_scanlines(H, W)
        return (rgb.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)

    return rgb


# ── Main render call ──────────────────────────────────────────────────────────
def render_gs_frame(screen, A, B, palette, render_mode="B", effect="none"):
    """
    Full render pipeline: GS state → heat → palette → effect → screen.

    Single screen: fills the whole window.
    Dual screen:   renders once, blits the result to both halves.
                   One GPU→CPU transfer, one palette+effect pass, two cheap blits.
    """
    A_np  = np.array(A)
    B_np  = np.array(B)
    heat  = compute_heat(A_np, B_np, render_mode)
    rgb   = apply_palette_heat(heat, palette)
    rgb   = apply_effect(rgb, effect)
    rgb_t = rgb.transpose(1, 0, 2)
    small = pygame.surfarray.make_surface(rgb_t)

    if DUAL_SCREEN:
        half = _get_half_surf()
        pygame.transform.scale(small, (SCREEN_W, SCREEN_H), half)
        screen.blit(half, (0, 0))           # left monitor
        screen.blit(half, (SCREEN_W, 0))    # right monitor — free copy
    else:
        pygame.transform.scale(small, (DISPLAY_W, DISPLAY_H), screen)


# ── NCA renderer (kept, not currently used as main engine) ────────────────────
def apply_palette(grid_np, palette):
    """Map NCA channel values through a 4-color palette. Returns (H, W, 3) uint8."""
    p     = palette / 255.0
    alpha = grid_np[:, :, 3]
    heat  = (grid_np[:, :, 0] + grid_np[:, :, 1]) * 0.5
    t      = (heat * 3.0).clip(0, 3)
    idx_lo = np.floor(t).astype(np.int32).clip(0, 2)
    idx_hi = (idx_lo + 1).clip(0, 3)
    frac   = (t - idx_lo)[:, :, np.newaxis]
    color  = p[idx_lo] + frac * (p[idx_hi] - p[idx_lo])
    rgb    = (color * alpha[:, :, np.newaxis] * 255).clip(0, 255).astype(np.uint8)
    return rgb


def render_frame(screen, grid, palette):
    """Render an NCA grid frame."""
    grid_np = np.array(grid)
    rgb     = apply_palette(grid_np, palette)
    rgb_t   = rgb.transpose(1, 0, 2)
    small   = pygame.surfarray.make_surface(rgb_t)
    pygame.transform.scale(small, (DISPLAY_W, DISPLAY_H), screen)
