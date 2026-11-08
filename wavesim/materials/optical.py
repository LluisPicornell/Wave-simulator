"""Propiedades ópticas de los tejidos (UV, visible, infrarrojo; 200 nm - 300 µm).

Modelo por composición (S. L. Jacques, "Optical properties of biological
tissues: a review", Phys. Med. Biol. 58, R37, 2013):

    mu_a  = B·[S·mu_a(HbO2) + (1-S)·mu_a(Hb)] + W·mu_a(agua) + F·mu_a(lípido)
            + M·mu_a(melanina) + mu_a(base)
    mu_s' = a·(lambda / 500 nm)^(-b)

B, S, W, F, M = fracciones de sangre, saturación de O2, agua, grasa y melanina.

AVISO SOBRE LA PRECISIÓN. Estas tablas son las menos precisas del paquete:
  - Hemoglobina 400-1000 nm: coeficientes de extinción de S. Prahl (OMLC),
    transcritos para un subconjunto de longitudes de onda. En el UV (< 400 nm)
    los valores son aproximados.
  - Agua: Hale & Querry (1973), Segelstein (1981), Pope & Fry (1997). En el
    infrarrojo medio y lejano los valores son de orden de magnitud (±50 %).
  - Lípido en el infrarrojo medio: aproximación grosera (0.5 × agua).
  - Parámetros de scattering (a, b) y composiciones: representativos.
  - Fascia y periostio: estimación como dermis sin melanina (colágeno denso,
    poca sangre), con el scattering de la piel.
Sirven para estimar cuánto penetra la luz (orden de magnitud), no para
dosimetría óptica. Sustitúyelos por datos medidos si necesitas precisión.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# λ [nm], HbO2, Hb: coeficiente de extinción molar [cm⁻¹/M]
_HB = np.array([
    [250, 106112, 112736], [300, 9.0e4, 9.0e4], [350, 7.0e4, 7.5e4],
    [400, 266232, 223296], [410, 466840, 303956], [420, 480360, 407560], [430, 214120, 528600],
    [440, 102580, 413280], [450, 62816, 103292], [460, 44480, 23388.8], [480, 26629.2, 14550],
    [500, 20932.8, 20035.2], [520, 24202.4, 31589.6], [540, 53236, 46592], [560, 32613.2, 53788],
    [580, 50104, 37020], [600, 3200, 14677.2], [620, 942, 6509.6], [650, 368, 3750.12],
    [700, 290, 1794.28], [750, 518, 1405.24], [800, 816, 761.72], [850, 1058, 691.32],
    [900, 1198, 761.84], [950, 1204, 693.44], [1000, 1032, 428],
])
# 150 g/L de hemoglobina, 64500 g/mol: mu_a = ln(10)·eps·C
_HB_FACTOR = np.log(10.0) * 150.0 / 64500.0  # cm⁻¹ por (cm⁻¹/M)

# λ [nm], mu_a del agua [cm⁻¹]
_WATER = np.array([
    [200, 0.07], [250, 0.012], [300, 0.003], [350, 0.0004], [400, 6.6e-5], [450, 9.2e-5],
    [500, 2.04e-4], [550, 5.65e-4], [600, 2.24e-3], [650, 3.4e-3], [700, 6.2e-3], [750, 0.026],
    [800, 0.020], [850, 0.043], [900, 0.068], [950, 0.39], [975, 0.45], [1000, 0.36],
    [1064, 0.13], [1100, 0.17], [1200, 1.0], [1300, 1.1], [1450, 31], [1550, 10], [1650, 5.5],
    [1800, 8.0], [1940, 120], [2100, 25], [2300, 25], [2500, 50], [2700, 1000], [2950, 12000],
    [3300, 1500], [3800, 120], [4500, 300], [5000, 200], [6100, 2700], [7000, 600], [8000, 500],
    [10000, 600], [15000, 2500], [20000, 3000], [30000, 2500], [50000, 1500], [100000, 800],
    [300000, 230],
])

# λ [nm], mu_a del lípido puro [cm⁻¹] (van Veen et al. 2005, aproximado)
_LIPID = np.array([
    [400, 0.1], [430, 0.07], [500, 0.01], [600, 0.004], [700, 0.003], [800, 0.005], [930, 0.11],
    [1040, 0.03], [1210, 1.1], [1400, 1.0], [1730, 10.0], [2000, 10.0],
])


def _loginterp(table: np.ndarray, lam_nm, col: int = 1):
    lam = np.asarray(lam_nm, dtype=float)
    return np.exp(np.interp(np.log(lam), np.log(table[:, 0]), np.log(table[:, col])))


def mu_a_water(lam_nm):
    return _loginterp(_WATER, lam_nm)


def mu_a_lipid(lam_nm):
    lam = np.asarray(lam_nm, dtype=float)
    return np.where(lam <= 2000, _loginterp(_LIPID, lam), 0.5 * mu_a_water(lam))


def mu_a_blood(lam_nm, so2: float):
    """Sangre completa (150 g/L Hb). Por encima de 1000 nm la Hb es despreciable frente al agua."""
    lam = np.clip(np.asarray(lam_nm, dtype=float), _HB[0, 0], _HB[-1, 0])
    oxy = _loginterp(_HB, lam, 1) * _HB_FACTOR
    deoxy = _loginterp(_HB, lam, 2) * _HB_FACTOR
    mu = so2 * oxy + (1 - so2) * deoxy
    return np.where(np.asarray(lam_nm) > 1000, mu * (1000.0 / np.asarray(lam_nm, float)) ** 4, mu)


def mu_a_melanin(lam_nm):
    """Interior del melanosoma (Jacques)."""
    return 519.0 * (np.asarray(lam_nm, dtype=float) / 500.0) ** -3.5


def mu_a_baseline(lam_nm):
    """Absorción de fondo de tejido sin sangre (proteínas), Jacques 1998."""
    return 0.244 + 85.3 * np.exp(-(np.asarray(lam_nm, dtype=float) - 154.0) / 66.2)


@dataclass(frozen=True)
class OpticalComposition:
    blood: float = 0.0  # B
    so2: float = 0.7  # S
    water: float = 0.0  # W
    fat: float = 0.0  # F
    melanin: float = 0.0  # M
    scat_a: float = 0.0  # mu_s' a 500 nm [cm⁻¹]
    scat_b: float = 1.0
    baseline: bool = True
    anisotropy: float = 0.9  # g


OPTICAL: dict[str, OpticalComposition] = {
    "water": OpticalComposition(water=1.0, baseline=False),
    "air": OpticalComposition(baseline=False),
    # piel clara: melanina promediada sobre 1.5 mm de piel (epidermis ~0.1 mm)
    "skin": OpticalComposition(blood=0.02, so2=0.7, water=0.65, melanin=0.003, scat_a=46.0, scat_b=1.421),
    "fat": OpticalComposition(blood=0.01, so2=0.8, water=0.15, fat=0.8, scat_a=18.4, scat_b=0.672),
    "muscle": OpticalComposition(blood=0.05, so2=0.7, water=0.75, fat=0.05, scat_a=13.0, scat_b=0.9),
    "blood": OpticalComposition(blood=1.0, so2=0.75, water=0.55, scat_a=22.0, scat_b=0.66, baseline=False),
    "cortical_bone": OpticalComposition(blood=0.01, so2=0.7, water=0.12, scat_a=22.9, scat_b=0.716),
    "trabecular_bone": OpticalComposition(blood=0.05, so2=0.7, water=0.25, fat=0.5, scat_a=25.0, scat_b=0.7),
    "marrow": OpticalComposition(blood=0.03, so2=0.7, water=0.15, fat=0.8, scat_a=18.4, scat_b=0.672),
    "hematoma": OpticalComposition(blood=1.0, so2=0.3, water=0.55, scat_a=22.0, scat_b=0.66, baseline=False),
    "edema": OpticalComposition(blood=0.05, so2=0.7, water=0.45, fat=0.5, scat_a=16.0, scat_b=0.7),
    "fascia": OpticalComposition(blood=0.01, so2=0.7, water=0.65, fat=0.02, scat_a=46.0, scat_b=1.421),
    "periosteum": OpticalComposition(blood=0.03, so2=0.7, water=0.65, fat=0.02, scat_a=46.0, scat_b=1.421),
}


def mu_a(key: str, lam_nm):
    """Coeficiente de absorción en 1/m."""
    c = OPTICAL[key]
    val = (
        c.blood * mu_a_blood(lam_nm, c.so2)
        + c.water * mu_a_water(lam_nm)
        + c.fat * mu_a_lipid(lam_nm)
        + c.melanin * mu_a_melanin(lam_nm)
    )
    if c.baseline:
        val = val + mu_a_baseline(lam_nm)
    return np.asarray(val) * 100.0


def mu_s_reduced(key: str, lam_nm):
    """Coeficiente de scattering reducido ``mu_s' = mu_s (1-g)`` en 1/m."""
    c = OPTICAL[key]
    return c.scat_a * (np.asarray(lam_nm, dtype=float) / 500.0) ** (-c.scat_b) * 100.0


def mu_effective(key: str, lam_nm):
    """Atenuación efectiva de la luz difusa ``sqrt(3 mu_a (mu_a + mu_s'))`` en 1/m."""
    a, s = mu_a(key, lam_nm), mu_s_reduced(key, lam_nm)
    return np.sqrt(3.0 * a * (a + s))


def mu_total(key: str, lam_nm):
    """Atenuación de fotones balísticos (sin desviar) ``mu_a + mu_s`` en 1/m."""
    c = OPTICAL[key]
    return mu_a(key, lam_nm) + mu_s_reduced(key, lam_nm) / (1.0 - c.anisotropy)
