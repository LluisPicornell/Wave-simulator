"""Luz en el antebrazo: Monte Carlo de fotones frente a la aproximación de difusión (800 nm).

Uso:  python examples/13_luz_montecarlo.py   → outputs/13_luz_montecarlo.png

Izquierda: fluencia MC (haz ancho colimado); centro: fluencia por difusión; derecha: cociente MC / difusión,
donde se ve que la difusión falla cerca de la piel iluminada y en las fronteras. Las zonas con pocos fotones
(error estándar > 30 %) se enmascaran en el cociente.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.em import montecarlo, optics
from wavesim.phantoms import Fracture, longitudinal_limb
from wavesim.viz import save

OUT = Path(__file__).resolve().parent.parent / "outputs"
LAM = 800.0
p = longitudinal_limb("antebrazo", Fracture(displacement=1e-3), dx=0.5e-3, width=40e-3, seed=1)
p = p.replace_tissue("water", "air")

mc = montecarlo.simulate(montecarlo.medium_from_phantom(p, LAM), 300_000, seed=1)
df = optics.solve_diffusion(p, LAM)
print(f"MC ({mc.backend}): {mc.photons_per_second:,.0f} fotones/s; reflectancia {mc.reflectance:.3f} ± "
      f"{mc.sem['reflectance']:.3f}, absorbida {mc.absorbed_total:.3f}, lateral {mc.lateral:.3f}, "
      f"transmitida {mc.transmittance:.1e}")

ext = p.grid.extent_mm()
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for ax, f, title in ((axs[0], mc.fluence, "Monte Carlo"), (axs[1], df.fluence, "Difusión")):
    im = ax.imshow(np.log10(np.maximum(f, 1e-6)), cmap="inferno", vmin=-4, vmax=1, extent=ext)
    ax.set_title(f"{title}, {LAM:.0f} nm")
    plt.colorbar(im, ax=ax, fraction=0.046, label="log10 fluencia relativa")
good = (mc.fluence > 0) & (mc.fluence_sem < 0.3 * mc.fluence) & (df.fluence > 0)
ratio = np.where(good, mc.fluence / np.where(df.fluence > 0, df.fluence, 1.0), np.nan)
im = axs[2].imshow(np.log2(ratio), cmap="RdBu_r", vmin=-2, vmax=2, extent=ext)
axs[2].set_title("log2 (MC / difusión)")
plt.colorbar(im, ax=axs[2], fraction=0.046, label="log2 cociente")
for ax in axs:
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
fig.tight_layout()
print("Guardado:", save(fig, OUT / "13_luz_montecarlo.png", dpi=90))
