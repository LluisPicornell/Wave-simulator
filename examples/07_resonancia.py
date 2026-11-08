"""Resonancia magnética de una pierna fracturada con distintas secuencias.

La fractura se ve sobre todo por el edema de la médula (brillante en STIR y
T2 con saturación grasa, oscuro en T1). El hueso cortical es negro salvo en UTE.

Uso:  python examples/07_resonancia.py   → outputs/07_resonancia.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

from wavesim.mri import SEQUENCES, mri_scan
from wavesim.phantoms import RingFracture, cross_section_limb
from wavesim.viz import plot_image, plot_phantom, save

OUT = Path(__file__).resolve().parent.parent / "outputs"
p = cross_section_limb("pierna", RingFracture(angle_deg=-10, gap=0.6e-3, edema_length=12e-3), dx=0.15e-3)
p = p.replace_tissue("water", "air")
names = ["T1", "T2", "PD_FS", "STIR", "GRE", "UTE"]
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
plot_phantom(p, axs[0, 0], "Fantoma", legend=False)
for ax, name in zip(list(axs[0, 1:]) + list(axs[1]), names):
    im = mri_scan(p, name, b0=1.5, voxel=0.4e-3, snr=40)
    seq = im.sequence
    extra = f", TI={seq.ti:.0f}" if seq.ti else ""
    plot_image(im.magnitude, im.extent_mm, ax, f"{seq.name}\nTR={seq.tr:.0f} TE={seq.te:g}{extra} ms",
               vmax=im.magnitude.max() * 0.8)
axs[1, 3].axis("off")
fig.tight_layout()
print("Guardado:", save(fig, OUT / "07_resonancia.png", dpi=90))
print({k: SEQUENCES[k].name for k in names})
