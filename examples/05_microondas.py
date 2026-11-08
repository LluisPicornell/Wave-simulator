"""Ondas de radio y microondas (Maxwell FDTD) a través del antebrazo.

Campo eléctrico en régimen estacionario y SAR (energía absorbida) para varias
frecuencias, con y sin fractura. Muestra por qué las microondas "ven" poco:
la longitud de onda en el tejido es de centímetros y el agua las absorbe.

Uso:  python examples/05_microondas.py   → outputs/05_microondas.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.em import MaxwellTMSolver, recommended_dx
from wavesim.phantoms import Fracture, longitudinal_limb
from wavesim.viz import save

OUT = Path(__file__).resolve().parent.parent / "outputs"
freqs = [128e6, 915e6, 2.45e9, 10e9]
fig, axs = plt.subplots(3, len(freqs), figsize=(5 * len(freqs), 14))
for c, f0 in enumerate(freqs):
    maps = []
    for fr in (None, Fracture(displacement=1e-3, gap=0.6e-3)):
        tissues = ["water", "skin", "fat", "muscle", "cortical_bone", "marrow", "blood", "hematoma", "edema"]
        dx = min(recommended_dx(f0, tissues, ppw=15), 0.5e-3)
        p = longitudinal_limb("antebrazo", fr, width=60e-3, dx=dx, seed=1).replace_tissue("water", "air")
        s = MaxwellTMSolver(p, f0)
        s.set_plane_wave_source()
        r = s.run()
        print(f"{f0 / 1e9:.3g} GHz: malla {p.grid.nz}x{p.grid.nx}, {r.steps} pasos, {r.elapsed_s:.0f} s")
        maps.append((p, r))
    (p0, r0), (p1, r1) = maps
    ext = p0.grid.extent_mm()
    axs[0, c].imshow(p0.labels, cmap="gray", extent=ext, alpha=0.5)
    im = axs[0, c].imshow(20 * np.log10(r0.amplitude + 1e-9), cmap="viridis", vmin=-40, vmax=6, extent=ext, alpha=0.8)
    axs[0, c].set_title(f"|E| a {f0 / 1e9:.3g} GHz (dB re 1 V/m incidente)")
    plt.colorbar(im, ax=axs[0, c], fraction=0.046)
    im = axs[1, c].imshow(np.log10(r0.sar + 1e-12), cmap="magma", extent=ext, vmin=-5, vmax=-1)
    axs[1, c].set_title("log10 SAR [W/kg] (1 V/m)")
    plt.colorbar(im, ax=axs[1, c], fraction=0.046)
    if p1.grid.shape == p0.grid.shape:
        diff = 20 * np.log10((r1.amplitude + 1e-12) / (r0.amplitude + 1e-12))
        # Donde el campo es < -40 dB el cociente es ruido numérico: se oculta
        valid = np.maximum(r0.amplitude, r1.amplitude) > 1e-2
        im = axs[2, c].imshow(np.where(valid, diff, np.nan), cmap="RdBu_r", vmin=-3, vmax=3, extent=ext)
        axs[2, c].set_title("Cambio de |E| por fractura + hematoma + edema (dB)")
        plt.colorbar(im, ax=axs[2, c], fraction=0.046)
        body = valid & (p0.labels != p0.labels[0, 0])
        print(f"   cambio mediano en el cuerpo por la fractura: {np.median(np.abs(diff[body])):.2f} dB "
              f"(máx. {np.abs(diff[body]).max():.1f} dB)")
fig.tight_layout()
print("Guardado:", save(fig, OUT / "05_microondas.png", dpi=80))
