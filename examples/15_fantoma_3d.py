"""Fantoma 3D de pierna (tibia y peroné) con fractura: cortes ortogonales, proyección y vista 3D.

Uso:  PYTHONPATH=. python examples/15_fantoma_3d.py   → docs/img/fantoma3d.png

Pierna de 60 mm (x) con la metáfisis proximal de la tibia hacia x > 0 y una
fractura oblicua desplazada en la diáfisis (x = -12 mm), a dx = 0.25 mm
(38 M celdas, unos 4 s).
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from wavesim.materials.tissues import TISSUES
from wavesim.phantoms import Fracture3D, limb_3d
from wavesim.phantoms.phantom import TISSUE_INDEX
from wavesim.viz import plot_phantom, save

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
X_FRACTURE, X_METAPHYSIS = -12e-3, 20e-3

fr = Fracture3D(kind="desplazada", position=X_FRACTURE, angle_deg=25.0, displacement=2e-3, displacement_y=1e-3,
                gap=0.6e-3, roughness=0.12e-3)
t0 = time.perf_counter()
ph = limb_3d("pierna_metafisis", fr, length=60e-3, dx=0.25e-3, seed=4)
print(f"limb_3d: {ph.grid.shape} = {ph.grid.size / 1e6:.1f} M celdas en {time.perf_counter() - t0:.1f} s")
g = ph.grid

fig = plt.figure(figsize=(17, 10.5))
ax = [fig.add_subplot(2, 3, k + 1) for k in range(5)] + [fig.add_subplot(2, 3, 6, projection="3d")]

lon = ph.slice_y(0.0)
plot_phantom(lon, ax[0], "corte longitudinal y = 0 (slice_y)", legend=False)
ax[0].set_ylim(70, 0)

for a, x, name in ((ax[1], X_FRACTURE, "en el foco"), (ax[2], X_METAPHYSIS, "en la metáfisis")):
    plot_phantom(ph.slice_x(x), a, f"corte transversal x = {x * 1e3:.0f} mm (slice_x), {name}", legend=False)
    a.set_xlabel("y [mm]")

cor = ph.slice_z(ph.meta["axis_depth"])
plot_phantom(cor, ax[3], f"corte coronal z = {ph.meta['axis_depth'] * 1e3:.0f} mm (slice_z)", legend=False)
ax[3].set_ylabel("y [mm]")
ax[3].set_ylim(-25, 30)

# Proyección lateral: longitud de hueso atravesada a lo largo de y (como una radiografía lateral)
bone = np.isin(ph.labels, [TISSUE_INDEX[t] for t in ("cortical_bone", "trabecular_bone")])
path = bone.sum(axis=1) * g.dx * 1e3
im = ax[4].imshow(path, cmap="gray", extent=lon.grid.extent_mm(), interpolation="nearest")
hem = ph.masks["hematoma"].any(axis=1)
ax[4].contour(g.x * 1e3, g.z * 1e3, hem.astype(float), levels=[0.5], colors="red", linewidths=0.8)
ax[4].set_ylim(70, 0)
ax[4].set_title("proyección lateral del hueso (Σ a lo largo de y) y hematoma")
ax[4].set_xlabel("x [mm]")
ax[4].set_ylabel("profundidad z [mm]")
fig.colorbar(im, ax=ax[4], label="hueso atravesado [mm]", shrink=0.8)

# Vista 3D: superficie de los huesos (vóxeles de 1.5 mm) y la fractura
s = 6
crop = (slice(0, int(70e-3 / g.dx)), slice(None), slice(None))
solid = np.isin(ph.labels[crop], [TISSUE_INDEX[t] for t in ("cortical_bone", "trabecular_bone", "marrow", "edema")])
hemat = ph.masks["hematoma"][crop]


def coarse(m):
    nz, ny, nx = (n // s for n in m.shape)
    return m[:nz * s, :ny * s, :nx * s].reshape(nz, s, ny, s, nx, s).mean(axis=(1, 3, 5)) > 0.4


vb, vh = coarse(solid), coarse(hemat)
vb &= ~vh
vox = np.transpose(vb | vh, (2, 1, 0))  # (x, y, z)
colors = np.empty(vox.shape, dtype=object)
colors[np.transpose(vb, (2, 1, 0))] = TISSUES["cortical_bone"].color
colors[np.transpose(vh, (2, 1, 0))] = "#d62728"
ax[5].voxels(vox, facecolors=colors, edgecolor=None, linewidth=0)
ax[5].set_xlabel("x")
ax[5].set_ylabel("y")
ax[5].set_zlabel("z")
ax[5].invert_zaxis()
ax[5].set_box_aspect(vox.shape)
ax[5].set_xticks([])
ax[5].set_yticks([])
ax[5].set_zticks([])
ax[5].view_init(elev=25, azim=-60)
ax[5].set_title("huesos y hematoma de la fractura (vóxeles de 1.5 mm)")

present = [k for k in TISSUE_INDEX if (ph.labels == TISSUE_INDEX[k]).any()]
handles = [Patch(facecolor=TISSUES[k].color, edgecolor="k", lw=0.3, label=TISSUES[k].name) for k in present]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
fig.suptitle("limb_3d('pierna_metafisis'): tibia y peroné, metáfisis con hueso trabecular, fractura oblicua "
             "desplazada (2 mm en z, 1 mm en y)")
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
print("Guardado:", save(fig, OUT / "fantoma3d.png", dpi=80))
