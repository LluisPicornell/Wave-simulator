"""Luz visible e infrarroja a través del antebrazo (aproximación de difusión).

Uso:  python examples/08_luz.py   → outputs/08_luz.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.em import optics
from wavesim.phantoms import Fracture, longitudinal_limb
from wavesim.viz import save

OUT = Path(__file__).resolve().parent.parent / "outputs"
p = longitudinal_limb("antebrazo", Fracture(displacement=1e-3), dx=0.2e-3, width=40e-3, seed=1)
p = p.replace_tissue("water", "air")
wavelengths = [450, 530, 650, 800, 1064, 1450]
fig, axs = plt.subplots(2, 3, figsize=(17, 11))
for ax, lam in zip(axs.ravel(), wavelengths):
    r = optics.solve_diffusion(p, lam)
    ax.imshow(p.labels, cmap="gray", extent=p.grid.extent_mm(), alpha=0.4)
    im = ax.imshow(np.log10(np.maximum(r.fluence, r.floor)), cmap="inferno", vmin=-14, vmax=1, extent=p.grid.extent_mm(),
                   alpha=0.85)
    ax.set_title(f"{lam} nm — transmitido {r.transmission_text()}\n"
                 f"(balístico: {optics.ballistic_transmission(p, lam):.0e})")
    plt.colorbar(im, ax=ax, fraction=0.046, label="log10 fluencia relativa")
    print(f"{lam} nm: transmisión difusa {r.transmission_text()}, {r.iterations} iteraciones CG")
fig.tight_layout()
print("Guardado:", save(fig, OUT / "08_luz.png", dpi=90))
