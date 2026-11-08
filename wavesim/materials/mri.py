"""Parámetros de resonancia magnética de los tejidos: PD, T1, T2 (y T2*).

Fuentes (valores in vivo, aproximados):
  - Músculo y grasa: Gold et al., AJR 183, 343 (2004), y de Bazelaire et al.,
    Radiology 230, 652 (2004).
  - Sangre: Stanisz et al., MRM 54, 507 (2005).
  - Hueso cortical: estudios UTE (Du & Bydder 2013): T2* ≈ 0.3-0.5 ms, por eso
    sale negro en secuencias convencionales.
  - Médula amarilla de la diáfisis ≈ grasa.
  - Edema y hematoma: valores representativos; en la realidad dependen mucho
    del tiempo de evolución (el hematoma cambia de señal en días).

PD es la densidad protónica relativa al agua. Los tiempos están en ms a 1.5 T
y 3 T; para otros campos se interpola en escala log-log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MRIProps:
    pd: float
    t1: tuple[float, float]  # (1.5 T, 3 T) en ms
    t2: tuple[float, float]  # (1.5 T, 3 T) en ms
    t2_prime_15: float = 40.0  # desfase por inhomogeneidad (T2') a 1.5 T, ms; ∝ 1/B0

    def at_field(self, b0: float) -> tuple[float, float, float, float]:
        """``(PD, T1, T2, T2*)`` en ms al campo ``b0`` (teslas)."""
        t1 = _loglog(b0, self.t1)
        t2 = _loglog(b0, self.t2)
        t2p = self.t2_prime_15 * 1.5 / b0
        t2s = 1.0 / (1.0 / t2 + 1.0 / t2p)
        return self.pd, t1, t2, t2s


def _loglog(b0: float, pair: tuple[float, float]) -> float:
    (v15, v30) = pair
    slope = np.log(v30 / v15) / np.log(3.0 / 1.5)
    return float(v15 * (b0 / 1.5) ** slope)


MRI: dict[str, MRIProps] = {
    "water": MRIProps(1.0, (3000.0, 3200.0), (2000.0, 1800.0)),
    "air": MRIProps(0.0, (1.0, 1.0), (1.0, 1.0)),
    "skin": MRIProps(0.7, (700.0, 850.0), (35.0, 30.0)),
    "fat": MRIProps(0.9, (343.0, 382.0), (58.0, 68.0)),
    "muscle": MRIProps(0.8, (1130.0, 1420.0), (35.0, 32.0)),
    "blood": MRIProps(0.85, (1441.0, 1932.0), (290.0, 275.0)),
    "cortical_bone": MRIProps(0.15, (200.0, 250.0), (0.5, 0.4), t2_prime_15=1.0),
    "trabecular_bone": MRIProps(0.7, (370.0, 400.0), (40.0, 38.0), t2_prime_15=8.0),
    "marrow": MRIProps(0.9, (370.0, 400.0), (55.0, 60.0)),
    "hematoma": MRIProps(0.85, (1000.0, 1300.0), (100.0, 80.0)),
    "edema": MRIProps(0.95, (1000.0, 1300.0), (90.0, 80.0)),
}
