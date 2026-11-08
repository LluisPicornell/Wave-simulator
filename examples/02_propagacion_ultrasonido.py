"""Propagación de un pulso de ultrasonido a través de la tibia fracturada (animación).

Se ve cómo la onda plana atraviesa gel, piel y grasa, rebota con fuerza en el
hueso (impedancia ~4 veces mayor), una parte entra en la cortical convertida en
ondas P y S, y cómo la grieta y el escalón de la fractura dispersan la onda.

Uso:  python examples/02_propagacion_ultrasonido.py
      → outputs/02_propagacion.gif, 02_propagacion.png, 02_presion_maxima.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.phantoms import Fracture, longitudinal_limb
from wavesim.ultrasound import ElasticSolver, LinearArray
from wavesim.viz import animate_wavefield, plot_phantom, plot_wavefield, save

OUT = Path(__file__).resolve().parent.parent / "outputs"
f0 = 2e6
phantom = longitudinal_limb("tibia", Fracture(displacement=1.2e-3, angle_deg=15, gap=0.6e-3),
                            width=30e-3, depth=30e-3, dx=1450 / f0 / 8, seed=1)
solver = ElasticSolver(phantom, f0)
array = LinearArray(n_elements=64, pitch=0.38e-3, z=1.6e-3, f0=f0)
nt = int(32e-6 / solver.dt)
t = solver.time_axis(nt)
solver.add_source(array.transmit_source(phantom.grid, t, array.plane_wave_delays(0.0, 1540)))
print(f"Malla {phantom.grid.nz}x{phantom.grid.nx}, {nt} pasos, dt = {solver.dt * 1e9:.1f} ns")
res = solver.run(nt, snapshot_every=nt // 80, snapshot_decimate=1, snapshot_field="p", progress=True)
print(f"Tiempo de cálculo: {res.elapsed_s:.0f} s")

animate_wavefield(phantom, res.snapshots, res.snapshot_times, OUT / "02_propagacion.gif",
                  title="Presión (escala log.)", log_compress=True)

fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for ax, k in zip(axs.ravel(), np.linspace(8, len(res.snapshots) - 1, 6).astype(int)):
    plot_wavefield(phantom, res.snapshots[k], ax, f"t = {res.snapshot_times[k] * 1e6:.1f} µs", clip=0.25)
fig.tight_layout()
save(fig, OUT / "02_propagacion.png", dpi=90)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plot_phantom(phantom, axs[0], "Fantoma")
peak_db = 20 * np.log10(res.peak_pressure / res.peak_pressure.max() + 1e-12)
im = axs[1].imshow(peak_db, cmap="inferno", vmin=-60, vmax=0, extent=phantom.grid.extent_mm())
axs[1].set_title("Presión máxima alcanzada (dB)")
plt.colorbar(im, ax=axs[1], label="dB")
save(fig, OUT / "02_presion_maxima.png", dpi=90)
print("Resultados en", OUT)
