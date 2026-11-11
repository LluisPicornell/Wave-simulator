"""Artefactos reales de RM en la pierna fracturada con el simulador de Bloch espacial.

Fila de arriba: imagen sin artefactos (equivale a las ecuaciones de señal de ``mri_scan``).
Fila de abajo: con desplazamiento químico de la grasa (BW 130 Hz/píxel: 1.7 píxeles en la
lectura, horizontal) y campo B0 de la susceptibilidad de los tejidos con B0 vertical en el
plano (imán abierto o extremidad perpendicular al túnel). Las dos con ruido (SNR 40) y la
misma escala de grises. Última columna: la diferencia en STIR y el mapa de ΔB0 (Hz).

Lo que se ve (cifras en docs/validacion/rm.md): la médula y la grasa subcutánea se desplazan
1.7 píxeles a la izquierda (bandas blanca y negra en sus bordes; en T1 la cortical de la
tibia parece de 2 píxeles a la izquierda y de 7 a la derecha, frente a 4-5 reales); la línea
de la fractura se desplaza ~1 píxel por el ΔB0 de la grieta (+80 Hz) pero sigue visible; el
STIR, que anula la grasa por T1, apenas cambia y el edema sigue igual de visible. La
saturación grasa espectral (DP-FS) no falla aquí: dentro de una extremidad redonda el campo
es casi uniforme y ninguna celda de grasa sale de la banda del pulso.

Uso:  python examples/14_rm_artefactos.py   → outputs/14_rm_artefactos.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.mri import mri_scan
from wavesim.mri.spatial import BlochPhysics
from wavesim.phantoms import RingFracture, cross_section_limb
from wavesim.viz import plot_image, save

OUT = Path(__file__).resolve().parent.parent / "outputs"
p = cross_section_limb("pierna", RingFracture(angle_deg=-10, gap=0.6e-3, edema_length=12e-3), dx=0.2e-3)
p = p.replace_tissue("water", "air")
names = ["T1", "STIR", "GRE", "PD_FS"]
rows = {"sin artefactos": BlochPhysics.none(), "con artefactos (B0 vertical)": BlochPhysics(b0_direction="z")}
fig, axs = plt.subplots(2, 5, figsize=(25, 10))
stir, vmax = {}, {}
for i, (label, physics) in enumerate(rows.items()):
    for j, name in enumerate(names):
        im = mri_scan(p, name, b0=1.5, snr=40, engine="bloch", matrix=(128, 128), physics=physics)
        # La misma escala de grises en las dos filas: la de la imagen sin artefactos
        vmax.setdefault(name, np.percentile(im.ideal, 99.5))
        if name == "STIR":
            stir[label] = im
        plot_image(im.magnitude, im.extent_mm, axs[i, j], f"{name} ({label})\n{im.timings['total']:.1f} s, "
                   f"{im.n_isochromats} isocromatos", vmax=vmax[name])
    if i:
        plot_image(im.field_hz, p.grid.extent_mm(), axs[i, 4], "ΔB0 de la susceptibilidad (Hz)", cmap="RdBu_r",
                   vmin=-150, vmax=150, label="Hz")
diff = stir["con artefactos (B0 vertical)"].ideal - stir["sin artefactos"].ideal
lim = np.abs(diff).max()
plot_image(diff, stir["sin artefactos"].extent_mm, axs[0, 4], "STIR: con - sin artefactos (sin ruido)",
           cmap="RdBu_r", vmin=-lim, vmax=lim)
fig.tight_layout()
print("Guardado:", save(fig, OUT / "14_rm_artefactos.png", dpi=80))
