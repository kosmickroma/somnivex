# Finds Lenia parameter sets where a simple seed produces a living creature.
# We don't need Orbium specifically — we need ANY stable Lenia dynamics for training.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from nca.lenia import make_kernel_fft, LENIA_R

def growth_poly(U, m, s):
    return np.maximum(0.0, 1.0 - (U - m)**2 / (9.0 * s**2))**4 * 2.0 - 1.0

def growth_gauss(U, m, s):
    return np.exp(-((U - m) / s)**2 / 2.0) * 2.0 - 1.0

def lenia_step(A, fK, m, s, dt=0.1, growth_fn=growth_poly):
    U = np.real(np.fft.ifft2(fK * np.fft.fft2(A)))
    return np.clip(A + dt * growth_fn(U, m, s), 0.0, 1.0).astype(np.float32)

def make_seed_blob(H, W, rng):
    """Gaussian blob — simpler than a ring."""
    cx, cy = H // 2, W // 2
    r, c   = np.ogrid[:H, :W]
    dist   = np.sqrt((r - cx)**2 + (c - cy)**2)
    A = np.exp(-(dist / (LENIA_R * 0.5))**2).astype(np.float32)
    A += rng.uniform(0, 0.05, (H, W)).astype(np.float32)
    return np.clip(A, 0.0, 1.0)

def make_seed_noise(H, W, rng):
    """Random noise in a disk — lets dynamics self-organize."""
    cx, cy = H // 2, W // 2
    r, c   = np.ogrid[:H, :W]
    dist   = np.sqrt((r - cx)**2 + (c - cy)**2)
    A = (dist < LENIA_R * 1.2) * rng.uniform(0, 0.5, (H, W)).astype(np.float32)
    return A.astype(np.float32)

H, W = 128, 128
rng  = np.random.default_rng(42)

mu_vals    = [0.10, 0.14, 0.15, 0.16, 0.20, 0.28]
sigma_vals = [0.014, 0.025, 0.04, 0.06, 0.10]
seeds      = [("blob", make_seed_blob), ("noise", make_seed_noise)]
growths    = [("poly", growth_poly), ("gauss", growth_gauss)]

winners = []

for seed_name, seed_fn in seeds:
    for gname, gfn in growths:
        fK = make_kernel_fft(LENIA_R, H, W)
        for mu in mu_vals:
            for sigma in sigma_vals:
                A = seed_fn(H, W, rng)
                for _ in range(500):
                    A = lenia_step(A, fK, mu, sigma, growth_fn=gfn)
                std = A.std()
                mean = A.mean()
                alive = std > 0.02 and mean > 0.001 and mean < 0.5
                if alive:
                    winners.append((std, seed_name, gname, mu, sigma, A.copy()))
                    print(f"ALIVE  seed={seed_name:5s}  growth={gname:5s}  mu={mu:.3f}  sigma={sigma:.3f}  std={std:.4f}  mean={mean:.4f}")

print(f"\n{len(winners)} working configs found.")

if winners:
    winners.sort(reverse=True)  # sort by std (most interesting first)
    n = min(len(winners), 6)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1: axes = [axes]
    for i, (std, sn, gn, mu, sigma, A) in enumerate(winners[:n]):
        axes[i].imshow(A, cmap='viridis')
        axes[i].set_title(f"{sn}/{gn}\nmu={mu} s={sigma:.3f}", fontsize=8)
    plt.tight_layout()
    plt.savefig('/tmp/lenia_search.png')
    print("Saved: /tmp/lenia_search.png")
else:
    print("No working configs. Simulator may have a deeper issue.")
