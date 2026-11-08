"""Fantomas de extremidad: anatomías, tipos de fractura y cortes transversales.

Uso:  python examples/01_fantomas.py   → outputs/01_fantomas.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.phantoms import (FRACTURE_KINDS, cross_section_limb, longitudinal_limb, random_fracture,
                              RingFracture)
from wavesim.viz import plot_phantom, save

OUT = Path(__file__).resolve().parent.parent / "outputs"

fig, axs = plt.subplots(3, 4, figsize=(20, 14))
for ax, preset in zip(axs[0, :3], ["tibia", "antebrazo", "muslo"]):
    plot_phantom(longitudinal_limb(preset, dx=0.1e-3, width=40e-3), ax, f"{preset} (sana)", legend=False)
plot_phantom(cross_section_limb("pierna", RingFracture(angle_deg=5)), axs[0, 3], "pierna, corte transversal")
for ax, kind in zip(axs[1:].ravel(), FRACTURE_KINDS):
    fr = random_fracture(np.random.default_rng(1), kind, x_range=1e-3)
    p = longitudinal_limb("tibia", fr, dx=0.05e-3, width=24e-3, depth=26e-3, seed=2)
    plot_phantom(p, ax, f"tibia: fractura {kind}", legend=False)
plot_phantom(cross_section_limb("antebrazo"), axs[2, 2], "antebrazo, corte transversal")
plot_phantom(cross_section_limb("muslo", RingFracture(angle_deg=-30)), axs[2, 3], "muslo, corte transversal")
fig.tight_layout()
print("Guardado:", save(fig, OUT / "01_fantomas.png", dpi=90))
