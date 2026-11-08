"""¿Qué ondas atraviesan el cuerpo? Penetración en todo el espectro electromagnético y en ultrasonido.

Uso:  python examples/04_espectro_penetracion.py
      → outputs/04_penetracion.png y tabla de pérdidas a través del antebrazo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.em import spectrum
from wavesim.materials import TISSUES
from wavesim.phantoms import longitudinal_limb
from wavesim.viz import save

OUT = Path(__file__).resolve().parent.parent / "outputs"
tissues = ["skin", "fat", "muscle", "cortical_bone", "marrow", "blood"]
freqs = spectrum.survey_frequencies(12)
fig, ax = plt.subplots(figsize=(14, 7))
for k in tissues:
    d = np.array([spectrum.penetration_depth(k, f) for f in freqs])
    ax.loglog(freqs, d, label=TISSUES[k].name, color=TISSUES[k].color if k != "cortical_bone" else "k", lw=1.6)
us = np.logspace(5, 7.5, 40)
ax.loglog(us, [spectrum.ultrasound_penetration_depth("muscle", f) for f in us], "--", color="#c0392b",
          label="ULTRASONIDO en músculo (intensidad)")
ax.loglog(us, [spectrum.ultrasound_penetration_depth("cortical_bone", f) for f in us], "--", color="k",
          label="ULTRASONIDO en hueso cortical")
for b in spectrum.BANDS:
    ax.axvspan(b.f_min, b.f_max, alpha=0.06 if spectrum.BANDS.index(b) % 2 else 0.0, color="gray")
    ax.text(np.sqrt(b.f_min * b.f_max), 3e2, b.name, rotation=90, fontsize=7, ha="center", va="top")
ax.axhline(0.05, color="gray", ls=":")
ax.text(2e3, 0.055, "grosor típico de un antebrazo (5 cm)", fontsize=8)
ax.set_xlabel("frecuencia [Hz]  (rayos X: 1 keV ≈ 2.4e17 Hz)")
ax.set_ylabel("profundidad a la que la intensidad cae a 1/e [m]")
ax.set_ylim(1e-7, 3e2)
ax.set_title("Penetración de las ondas en los tejidos")
ax.legend(fontsize=8, loc="lower left")
print("Guardado:", save(fig, OUT / "04_penetracion.png", dpi=110))

layers = spectrum.column_layers(longitudinal_limb("antebrazo", dx=0.1e-3, width=10e-3))
print("\nAntebrazo:", ", ".join(f"{TISSUES[k].name} {t * 1e3:.1f} mm" for k, t in layers))
print(f"\n{'onda':38s} {'banda':30s} {'pérdida al cruzar':>18s}")
cases = [
    ("Ultrasonido 1 MHz", 1e6, "ultrasound"), ("Ultrasonido 5 MHz", 5e6, "ultrasound"),
    ("Corriente 50 kHz (bioimpedancia)", 5e4, "em"), ("RF de RM a 1.5 T (64 MHz)", 64e6, "em"),
    ("Microondas 915 MHz", 915e6, "em"), ("Microondas 2.45 GHz", 2.45e9, "em"), ("Ondas mm 60 GHz", 60e9, "em"),
    ("Terahercios 1 THz", 1e12, "em"), ("Infrarrojo cercano 800 nm", spectrum.C0 / 800e-9, "em"),
    ("Luz roja 650 nm", spectrum.C0 / 650e-9, "em"), ("Luz verde 530 nm", spectrum.C0 / 530e-9, "em"),
    ("Ultravioleta 300 nm", spectrum.C0 / 300e-9, "em"), ("Rayos X 30 keV", 30 * spectrum.HZ_PER_KEV, "em"),
    ("Rayos X 60 keV", 60 * spectrum.HZ_PER_KEV, "em"), ("Gamma 662 keV (Cs-137)", 662 * spectrum.HZ_PER_KEV, "em"),
]
for name, f, mod in cases:
    loss = spectrum.through_transmission_db(layers, f, mod)["total_db"]
    band = "ultrasonido" if mod == "ultrasound" else spectrum.band_of(f).name
    print(f"{name:38s} {band:30s} {loss:14.1f} dB")
print("\n(1 dB ≈ 26 % de pérdida de intensidad; 100 dB = 1e-10 de la intensidad inicial)")
