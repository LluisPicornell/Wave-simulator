"""Tomografía de impedancia eléctrica (EIT) de una pierna con fractura a 50 kHz.

A baja frecuencia no hay propagación: se resuelve div(sigma* grad(phi)) = 0 con
16 electrodos en la piel, inyectando corriente entre pares adyacentes y
midiendo las tensiones entre los demás pares. La fractura (hematoma, que
conduce mucho más que la cortical) cambia el camino de la corriente.

Uso:  python examples/10_impedancia_eit.py   → outputs/10_impedancia_eit.png  (~3 min)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.em import QuasiStaticSolver, reconstruct_difference, ring_electrodes, sensitivity_matrix, simulate_eit
from wavesim.phantoms import RingFracture, cross_section_limb
from wavesim.viz import save

OUT = Path(__file__).resolve().parent.parent / "outputs"
F = 50e3
N_ELECTRODES = 16
CURRENT = 1e-3  # A por metro de profundidad (el problema es 2D)
DX = 0.5e-3
FRAME = 0  # inyección entre los electrodos 0 (arriba, junto a la tibia) y 1

healthy = cross_section_limb("pierna", None, dx=DX, seed=3)
broken = cross_section_limb("pierna", RingFracture(angle_deg=0.0), dx=DX, seed=3)
electrodes = ring_electrodes(healthy, N_ELECTRODES, width=8e-3)

runs = {}
for name, ph in (("sana", healthy), ("fracturada", broken)):
    runs[name] = simulate_eit(QuasiStaticSolver(ph, F), electrodes, current=CURRENT)
    print(f"{name}: {runs[name].iterations} iteraciones COCG, residuo {runs[name].residual:.1e}")
v0, v1 = runs["sana"].voltages, runs["fracturada"].voltages

# --- Cambio de las tensiones de borde por la fractura
rel = np.abs(v1 - v0) / np.abs(v0)
print(f"\nTensión típica entre electrodos: {np.median(np.abs(v0)) * 1e3:.2f} mV (corriente {CURRENT * 1e3:.1f} mA/m)")
print(f"Cambio máximo de una tensión: {rel.max() * 100:.2f} %  (respecto a su valor sano)")
print(f"Cambio máximo respecto a la mayor tensión: {np.abs(v1 - v0).max() / np.abs(v0).max() * 100:.3f} %")
print(f"Datos que cambian más de 0.1 %: {(rel > 1e-3).sum()} de {rel.size}")

# --- Reconstrucción de diferencias de un paso (Jacobiano de la pierna sana)
sens = sensitivity_matrix(runs["sana"], factor=6)
delta = reconstruct_difference(sens, v1 - v0, reg=1e-2)
img = sens.to_image(delta)
cell = sens.factor * DX
jj, ii = np.unravel_index(np.argmax(img), img.shape)
g = healthy.grid
x_rec, z_rec = g.x0 + (ii + 0.5) * cell, g.z0 + (jj + 0.5) * cell
X, Z = g.meshgrid()
mask = broken.masks["hematoma"]
x_true, z_true = X[mask].mean(), Z[mask].mean()
blob = (img >= 0.5 * img.max()).sum() * cell**2
print(f"\nReconstrucción: máximo en ({x_rec * 1e3:.0f}, {z_rec * 1e3:.0f}) mm; "
      f"hematoma real en ({x_true * 1e3:.0f}, {z_true * 1e3:.0f}) mm → error de posición "
      f"{np.hypot(x_rec - x_true, z_rec - z_true) * 1e3:.0f} mm")
print(f"Resolución aproximada (diámetro a media altura): {2 * np.sqrt(blob / np.pi) * 1e3:.0f} mm "
      f"(celda de reconstrucción {cell * 1e3:.0f} mm; el hematoma mide ~{broken.meta['fracture']['hematoma_length'] * 1e3:.0f} mm)")

# --- Figuras
ext = g.extent_mm()
fig, axs = plt.subplots(2, 3, figsize=(17, 11))
for row, (name, ph) in enumerate((("sana", healthy), ("fracturada", broken))):
    run = runs[name]
    phi = run.fields[FRAME]
    jmag = run.solver.current_density_magnitude(phi)
    tis = run.solver.tissue
    ax = axs[row, 0]
    im = ax.imshow(np.where(tis, np.abs(phi) * 1e3, np.nan), cmap="viridis", extent=ext)
    ax.contour(g.x * 1e3, g.z * 1e3, np.real(phi), 14, colors="w", linewidths=0.4)
    plt.colorbar(im, ax=ax, fraction=0.046, label="|φ| (mV)")
    ax.set_title(f"Potencial, pierna {name}")
    ax = axs[row, 1]
    im = ax.imshow(np.where(tis, jmag, np.nan), cmap="inferno", extent=ext, vmax=np.percentile(jmag[tis], 99.5))
    plt.colorbar(im, ax=ax, fraction=0.046, label="|J| (A/m²)")
    ax.set_title(f"Densidad de corriente, pierna {name}")
    for a in axs[row, :2]:
        a.plot(electrodes.centers[:, 0] * 1e3, electrodes.centers[:, 1] * 1e3, "c.", ms=6)
        a.plot(*(electrodes.centers[[FRAME, FRAME + 1]].T * 1e3), "wo", ms=5, mfc="none")
        a.set_xlabel("x (mm)")
        a.set_ylabel("z (mm)")

ax = axs[0, 2]
ax.bar(np.arange(rel.size), rel * 100, width=1.0, color="C3")
ax.set_xlabel("índice del dato (trama × medida)")
ax.set_ylabel("cambio de la tensión (%)")
ax.set_title(f"Cambio por la fractura (máx {rel.max() * 100:.1f} %)")

ax = axs[1, 2]
ax.imshow(broken.labels, cmap="gray", extent=ext, alpha=0.35)
up = sens.upsample(img, g.shape)
im = ax.imshow(np.where(up != 0, up, np.nan), cmap="RdBu_r", extent=ext, alpha=0.8, vmin=-np.abs(img).max(), vmax=np.abs(img).max())
ax.contour(g.x * 1e3, g.z * 1e3, broken.masks["hematoma"], [0.5], colors="k", linewidths=1.2)
plt.colorbar(im, ax=ax, fraction=0.046, label="cambio relativo de admitividad")
ax.set_title("Reconstrucción de diferencias (contorno: hematoma real)")
ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")
fig.tight_layout()
print("\nGuardado:", save(fig, OUT / "10_impedancia_eit.png", dpi=90))
