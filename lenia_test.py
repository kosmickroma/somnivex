import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '.')
from nca.lenia import make_kernel_fft, lenia_step_np, seed_creature, LENIA_R, _growth_np

H, W = 128, 128
fK   = make_kernel_fft(LENIA_R, H, W)
rng  = np.random.default_rng(42)
A    = seed_creature(H, W, rng)
mu, sigma = 0.15, 0.014

U = np.real(np.fft.ifft2(fK * np.fft.fft2(A)))
G = _growth_np(U, mu, sigma)
thresh = 0.1
mask = A > thresh
print(f"Seed: A max={A.max():.4f}  mean={A.mean():.4f}  nonzero={mask.sum()}")
print(f"U in creature (A>{thresh}): {U[mask].mean():.4f}  (need ~0.15)")
print(f"G in creature (A>{thresh}): {G[mask].mean():.4f}  (need > 0, positive = alive)")
print()

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
axes[0].imshow(A, cmap='viridis')
axes[0].set_title('step 0 (seed)')

for i, n in enumerate([50, 100, 200, 400, 600]):
    for _ in range(50):
        A = lenia_step_np(A, fK, mu=mu, sigma=sigma)
    axes[i+1].imshow(A, cmap='viridis')
    axes[i+1].set_title(f'~step {n}')
    print(f"step ~{n}: A std={A.std():.4f}  {'ALIVE' if A.std()>0.01 else 'DEAD'}")

plt.tight_layout()
plt.savefig('/tmp/lenia_test.png')
print('Saved: /tmp/lenia_test.png')
