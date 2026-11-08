"""Ecografía (modo B) de una tibia sana y fracturada con la misma anatomía.

Simula una sonda lineal de 3 MHz con 3 ondas planas (-8°, 0°, +8°) compuestas.

Uso:  python examples/03_ecografia_fractura.py   → outputs/03_ecografia.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

from wavesim.analysis import dilate, fracture_visibility, resample_mask
from wavesim.phantoms import Fracture, longitudinal_limb
from wavesim.ultrasound import simulate_bmode
from wavesim.viz import plot_image, plot_phantom, save

OUT = Path(__file__).resolve().parent.parent / "outputs"
f0 = 3e6
dx = 1450 / f0 / 8
cases = {
    "sana": None,
    "fractura desplazada": Fracture(displacement=1.0e-3, gap=0.5e-3),
    "fisura (0.15 mm)": Fracture(kind="fisura", gap=0.15e-3, depth_fraction=0.3, hematoma_thickness=0.3e-3,
                                 hematoma_length=4e-3),
}
fig, axs = plt.subplots(2, len(cases), figsize=(6 * len(cases), 10))
images = {}
for k, (name, fr) in enumerate(cases.items()):
    p = longitudinal_limb("tibia", fr, width=24e-3, depth=22e-3, dx=dx, seed=1)
    sc = simulate_bmode(p, f0, angles=(-8.0, 0.0, 8.0), progress=True)
    images[name] = (p, sc.image)
    plot_phantom(p, axs[0, k], name, legend=False)
    plot_image(sc.image.db, sc.image.extent_mm(), axs[1, k], f"Modo B — {name}", vmin=-50, vmax=0)
fig.tight_layout()
print("Guardado:", save(fig, OUT / "03_ecografia.png", dpi=90))

_, healthy = images["sana"]
for name in list(cases)[1:]:
    p, img = images[name]
    roi = dilate(resample_mask(p, p.fracture_mask | p.masks["hematoma"], img.x, img.z), 4)
    v = fracture_visibility(healthy.db, img.db, roi)
    print(f"{name}: cambio medio en la zona de la fractura = {v['mean_abs_change']:.1f} dB")
