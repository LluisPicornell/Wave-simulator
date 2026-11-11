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
  - Fascia y periostio: estimación a partir del tendón y el ligamento, cuyo
    agua ligada al colágeno tiene T2 de 1-10 ms (Du & Bydder 2013; Gold et
    al. 2004): salen oscuros en todas las secuencias convencionales y se ven
    en UTE.

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
    "fascia": MRIProps(0.6, (700.0, 800.0), (12.0, 10.0), t2_prime_15=10.0),
    "periosteum": MRIProps(0.6, (800.0, 900.0), (15.0, 12.0), t2_prime_15=10.0),
}


# --------------------------------------------------------------------------
# Datos para el simulador de Bloch espacial (``wavesim.mri.spatial``)
# --------------------------------------------------------------------------
#: Razón giromagnética del protón entre 2π (Hz/T), CODATA 2018.
GAMMA_BAR = 42.577478518e6

#: Fracción de la señal que procede de protones de grasa (saturación grasa,
#: desplazamiento químico y susceptibilidad). Fascia y periostio: algo de
#: grasa entre las láminas de colágeno (estimación).
FAT_FRACTION: dict[str, float] = {
    "fat": 0.95, "marrow": 0.85, "trabecular_bone": 0.6, "edema": 0.4, "skin": 0.1, "muscle": 0.03,
    "fascia": 0.05, "periosteum": 0.02,
}

#: Espectro de la grasa: (desplazamiento respecto al agua en ppm, amplitud relativa).
#: Un pico: el metileno (-CH2-)n a -3.4 ppm, el modelo de los libros de texto.
#: Seis picos: los valores por defecto del ISMRM Fat-Water Toolbox (código de
#: Hernando et al., comprobado en sus copias públicas de GitHub), que proceden
#: del espectro in vivo de Hamilton et al., NMR Biomed 24:784 (2011); que la
#: tabla sea exactamente la de ese artículo no se pudo verificar (PubMed no era
#: accesible). Las amplitudes suman 0.999 y se normalizan al usarlas.
FAT_PEAKS: dict[str, tuple[tuple[float, float], ...]] = {
    "single": ((-3.4, 1.0),),
    "multipeak": ((-3.80, 0.087), (-3.40, 0.693), (-2.60, 0.128), (-1.94, 0.004), (-0.39, 0.039), (0.60, 0.048)),
}


def blood_susceptibility(saturation: float, hct: float = 0.42) -> float:
    """Susceptibilidad volumétrica SI de la sangre (ppm) con saturación de O2 ``saturation``.

    Fórmula de Spees et al., MRM 45:533 (2001): ``Δχ_do = 4π·0.264 ppm``
    por unidad de hematocrito entre glóbulos rojos desoxigenados y
    oxigenados, ``χ = 4π {[0.264 (1 - Y) - 0.736] Hct - 0.722 (1 - Hct)}``.
    """
    return 4.0 * np.pi * ((0.264 * (1.0 - saturation) - 0.736) * hct - 0.722 * (1.0 - hct))


#: Susceptibilidad del agua y del aire (SI, ppm): Schenck, Med Phys 23:815 (1996);
#: el aire es paramagnético por el oxígeno (+0.358 ppm a 20 °C).
CHI_WATER = -9.05
CHI_AIR = 0.36
#: Hueso cortical bovino: -0.90e-6 (cgs) = -11.3 ppm SI (Hopkins & Wehrli, MRM 37:494, 1997).
CHI_CORTICAL = -11.3
#: Lípido: 0.61 ppm más paramagnético que el agua (Karampinos et al., MRM 2012).
CHI_LIPID = CHI_WATER + 0.61


def _soft_chi(key: str) -> float:
    """Tejido blando como mezcla de agua y lípido según su fracción grasa.

    Las diferencias entre tejidos blandos acuosos son de 0.01-0.1 ppm (cifra
    de la literatura de mapas de susceptibilidad, QSM), despreciables aquí
    frente a las de la grasa (0.6 ppm), el hueso (2.3 ppm) y el aire (9.4 ppm).
    """
    ff = FAT_FRACTION.get(key, 0.0)
    return (1.0 - ff) * CHI_WATER + ff * CHI_LIPID


#: Susceptibilidad magnética volumétrica (SI, ppm) por tejido. El hueso
#: trabecular es la mezcla lineal de un 20 % de matriz cortical y un 80 % de
#: médula (la BV/TV de ``materials.porous``). La sangre de los vasos es una
#: mezcla arteriovenosa (Y = 0.8); el hematoma, hiperagudo (oxihemoglobina,
#: Y = 0.95), coherente con el T2 largo de la tabla: en fase aguda
#: (desoxihemoglobina, días 1-3) sería ``blood_susceptibility(0)`` ≈ -7.8 ppm.
SUSCEPTIBILITY_PPM: dict[str, float] = {
    "water": CHI_WATER,
    "air": CHI_AIR,
    "skin": _soft_chi("skin"),
    "fat": _soft_chi("fat"),
    "muscle": _soft_chi("muscle"),
    "blood": blood_susceptibility(0.8),
    "cortical_bone": CHI_CORTICAL,
    "trabecular_bone": 0.2 * CHI_CORTICAL + 0.8 * _soft_chi("marrow"),
    "marrow": _soft_chi("marrow"),
    "hematoma": blood_susceptibility(0.95),
    "edema": _soft_chi("edema"),
    "fascia": _soft_chi("fascia"),
    "periosteum": _soft_chi("periosteum"),
}
