"""Rayos X: radiografía (proyección) y TAC de una pierna con fractura.

Uso:  python examples/06_rayos_x_tac.py   → outputs/06_rayos_x.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

from wavesim.em import xray
from wavesim.phantoms import Fracture, RingFracture, cross_section_limb, longitudinal_limb
from wavesim.viz import plot_image, plot_phantom, save

OUT = Path(__file__).resolve().parent.parent / "outputs"
fig, axs = plt.subplots(2, 3, figsize=(18, 11))

cross = cross_section_limb("pierna", RingFracture(angle_deg=-10, gap=0.6e-3), dx=0.25e-3).replace_tissue("water", "air")
plot_phantom(cross, axs[0, 0], "Corte transversal (fractura en la tibia)")
ct = xray.ct_scan(cross, n_angles=360, photons=2e5)
plot_image(ct.hu, cross.grid.extent_mm(), axs[0, 1], "TAC 120 kVp (ventana ósea)", vmin=-400, vmax=1800, label="HU")
plot_image(ct.hu, cross.grid.extent_mm(), axs[0, 2], "TAC (ventana de partes blandas)", vmin=-160, vmax=240,
           label="HU")

lon = longitudinal_limb("tibia", Fracture(displacement=1.2e-3, angle_deg=25), dx=0.1e-3, width=60e-3).replace_tissue(
    "water", "air")
plot_phantom(lon, axs[1, 0], "Corte longitudinal (fractura oblicua)")
ct2 = xray.ct_scan(lon, n_angles=360, photons=2e5)
plot_image(ct2.hu, lon.grid.extent_mm(), axs[1, 1], "TAC reconstruido en el plano longitudinal", vmin=-400,
           vmax=1800, label="HU")
for ang in (0.0, 90.0):
    rad = xray.radiograph(lon, ang, photons=1e5)
    axs[1, 2].plot(rad.det * 1e3, rad.intensity[0], label=f"proyección a {ang:.0f}°")
axs[1, 2].set_xlabel("posición en el detector [mm]")
axs[1, 2].set_ylabel("I / I0")
axs[1, 2].set_title("Radiografía (60 kVp) de este corte: perfiles 1D")
axs[1, 2].legend()
fig.tight_layout()
print("Guardado:", save(fig, OUT / "06_rayos_x.png", dpi=90))
