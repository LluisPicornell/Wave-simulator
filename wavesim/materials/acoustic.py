"""Propiedades acústicas y elásticas de los tejidos (ultrasonido).

Valores representativos a ~37 °C dentro de los rangos publicados en:
  - F. A. Duck, *Physical Properties of Tissue* (1990).
  - IT'IS Foundation Tissue Properties Database v4.1 (Hasgall et al., 2022).
  - T. D. Mast, "Empirical relationships between acoustic parameters in human
    soft tissues", ARLO 1, 37 (2000).
  - Velocidad de cizalla del hueso cortical: Bossy et al. (2004), Grimal et al.

La variabilidad entre personas es grande (±5 % en velocidad, ±50 % en
atenuación). El hueso trabecular se trata como medio efectivo homogéneo y sus
valores son especialmente inciertos.

Atenuación: ``alpha(f) = alpha0 * f[MHz]**y`` en dB/cm. Para el hueso se
supone el mismo factor de calidad Q para ondas P y S, lo que da
``alpha_s = alpha_p * c_p / c_s``.

Los tejidos blandos tienen módulos de cizalla de kPa (ondas S de pocos m/s que
se atenúan en micras a frecuencias de MHz), así que se tratan como fluidos
(``c_s = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DB_PER_NEPER = 20.0 / np.log(10.0)  # 8.686


@dataclass(frozen=True)
class AcousticProps:
    density: float  # kg/m^3
    c_p: float  # velocidad de onda longitudinal (P), m/s
    c_s: float = 0.0  # velocidad de onda de cizalla (S), m/s; 0 = fluido
    alpha_p: float = 0.0  # dB/(cm·MHz^y)
    alpha_s: float = 0.0  # dB/(cm·MHz^y)
    y: float = 1.0  # exponente de la ley de potencia

    @property
    def impedance(self) -> float:
        """Impedancia acústica Z = rho * c (Rayl = kg/(m²·s))."""
        return self.density * self.c_p

    def alpha_np_per_m(self, f_hz: float, shear: bool = False) -> float:
        """Atenuación de amplitud en Np/m a la frecuencia ``f_hz``."""
        a0 = self.alpha_s if shear else self.alpha_p
        return a0 * (f_hz / 1e6) ** self.y * 100.0 / DB_PER_NEPER


ACOUSTIC: dict[str, AcousticProps] = {
    # agua a ~22 °C (como gel de acoplamiento); atenuación ∝ f^2
    "water": AcousticProps(1000.0, 1500.0, 0.0, 0.0022, 0.0, 2.0),
    # el aire refleja prácticamente todo el ultrasonido (por eso se usa gel)
    "air": AcousticProps(1.2, 343.0, 0.0, 1.6, 0.0, 1.0),
    "skin": AcousticProps(1109.0, 1615.0, 0.0, 1.8, 0.0, 1.0),
    "fat": AcousticProps(911.0, 1450.0, 0.0, 0.6, 0.0, 1.0),
    "muscle": AcousticProps(1090.0, 1580.0, 0.0, 1.09, 0.0, 1.0),
    "blood": AcousticProps(1050.0, 1578.0, 0.0, 0.2, 0.0, 1.2),
    "cortical_bone": AcousticProps(1908.0, 3500.0, 1800.0, 5.0, 9.7, 1.0),
    "trabecular_bone": AcousticProps(1178.0, 2100.0, 1000.0, 15.0, 31.5, 1.0),
    # médula amarilla de la diáfisis: casi toda grasa
    "marrow": AcousticProps(980.0, 1450.0, 0.0, 0.6, 0.0, 1.0),
    # sangre coagulada en el foco de fractura
    "hematoma": AcousticProps(1060.0, 1580.0, 0.0, 0.3, 0.0, 1.2),
    # médula con más agua por el edema
    "edema": AcousticProps(1010.0, 1500.0, 0.0, 0.5, 0.0, 1.0),
}


def acoustic(key: str) -> AcousticProps:
    return ACOUSTIC[key]


def reflection_coefficient(key1: str, key2: str) -> float:
    """Coeficiente de reflexión de presión a incidencia normal de 1 hacia 2."""
    z1, z2 = ACOUSTIC[key1].impedance, ACOUSTIC[key2].impedance
    return (z2 - z1) / (z2 + z1)


def intensity_transmission(key1: str, key2: str) -> float:
    """Fracción de intensidad transmitida a incidencia normal (1 - R²)."""
    z1, z2 = ACOUSTIC[key1].impedance, ACOUSTIC[key2].impedance
    return 4.0 * z1 * z2 / (z1 + z2) ** 2
