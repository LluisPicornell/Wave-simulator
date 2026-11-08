"""Atenuación de rayos X y gamma en los tejidos (10 keV - 10 MeV).

A estas energías la longitud de onda es de 1e-4 a 0.1 nm: no tiene sentido
resolver la onda en una malla. La física relevante es la atenuación a lo largo
de rayos (Beer-Lambert) por efecto fotoeléctrico, Compton, Rayleigh y
producción de pares, resumida en el coeficiente másico ``mu/rho(E)``.

Tablas (cm²/g, con dispersión coherente) de J. H. Hubbell & S. M. Seltzer,
NIST Standard Reference Database 126, para agua, aire, aluminio, músculo
esquelético y hueso cortical (composiciones ICRU-44). Transcritas de la tabla
publicada; conviene contrastarlas con la base de datos oficial antes de un uso
cuantitativo.

El resto de tejidos se obtiene con descomposición en base agua+hueso
(Alvarez & Macovski 1976): se igualan la densidad electrónica (Compton) y el
término fotoeléctrico ``sum w_i Z_i^4 / A_i`` del tejido. Con músculo, cuya
tabla se conoce, el método reproduce los valores con un error < 1 %
(ver ``tests/test_materials.py``).
"""

from __future__ import annotations

import numpy as np

ENERGIES_KEV = np.array(
    [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 600, 800,
     1000, 1250, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000], dtype=float
)

TABLES: dict[str, np.ndarray] = {
    "water": np.array([5.329, 1.673, 0.8096, 0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707, 0.1505, 0.1370,
                       0.1186, 0.1061, 0.09687, 0.08956, 0.07865, 0.07072, 0.06323, 0.05754, 0.04942, 0.03969,
                       0.03403, 0.03031, 0.02770, 0.02429, 0.02219]),
    "air": np.array([5.120, 1.614, 0.7779, 0.3538, 0.2485, 0.2080, 0.1875, 0.1662, 0.1541, 0.1356, 0.1233,
                     0.1067, 0.09549, 0.08712, 0.08055, 0.07074, 0.06358, 0.05687, 0.05175, 0.04447, 0.03581,
                     0.03079, 0.02751, 0.02522, 0.02225, 0.02045]),
    "aluminum": np.array([26.23, 7.955, 3.441, 1.128, 0.5685, 0.3681, 0.2778, 0.2018, 0.1704, 0.1378, 0.1223,
                          0.1042, 0.09276, 0.08445, 0.07802, 0.06841, 0.06146, 0.05496, 0.05006, 0.04324, 0.03541,
                          0.03106, 0.02836, 0.02655, 0.02437, 0.02318]),
    "muscle": np.array([5.356, 1.693, 0.8205, 0.3783, 0.2685, 0.2262, 0.2048, 0.1823, 0.1693, 0.1492, 0.1358,
                        0.1176, 0.1052, 0.09602, 0.08877, 0.07796, 0.07010, 0.06268, 0.05704, 0.04900, 0.03938,
                        0.03379, 0.03012, 0.02754, 0.02418, 0.02211]),
    "cortical_bone": np.array([28.51, 9.032, 4.001, 1.331, 0.6655, 0.4242, 0.3148, 0.2229, 0.1855, 0.1480,
                               0.1309, 0.1113, 0.09908, 0.09022, 0.08332, 0.07308, 0.06566, 0.05871, 0.05346,
                               0.04607, 0.03745, 0.03257, 0.02946, 0.02734, 0.02467, 0.02314]),
}

ATOMIC = {  # Z, masa atómica
    "H": (1, 1.008), "C": (6, 12.011), "N": (7, 14.007), "O": (8, 15.999), "Na": (11, 22.990),
    "Mg": (12, 24.305), "P": (15, 30.974), "S": (16, 32.06), "Cl": (17, 35.45), "K": (19, 39.098),
    "Ca": (20, 40.078), "Fe": (26, 55.845),
}

# Composición elemental en % de masa (ICRU-44; médula: Woodard & White 1986;
# fascia y periostio: "tejido conectivo" de ICRU-44)
COMPOSITION: dict[str, dict[str, float]] = {
    "water": {"H": 11.19, "O": 88.81},
    "muscle": {"H": 10.2, "C": 14.3, "N": 3.4, "O": 71.0, "Na": 0.1, "P": 0.2, "S": 0.3, "Cl": 0.1, "K": 0.4},
    "cortical_bone": {"H": 3.4, "C": 15.5, "N": 4.2, "O": 43.5, "Na": 0.1, "Mg": 0.2, "P": 10.3, "S": 0.3, "Ca": 22.5},
    "fat": {"H": 11.4, "C": 59.8, "N": 0.7, "O": 27.8, "Na": 0.1, "S": 0.1, "Cl": 0.1},
    "skin": {"H": 10.0, "C": 20.4, "N": 4.2, "O": 64.5, "Na": 0.2, "P": 0.1, "S": 0.2, "Cl": 0.3, "K": 0.1},
    "blood": {"H": 10.2, "C": 11.0, "N": 3.3, "O": 74.5, "Na": 0.1, "P": 0.1, "S": 0.2, "Cl": 0.3, "K": 0.2, "Fe": 0.1},
    "marrow": {"H": 11.5, "C": 64.4, "N": 0.7, "O": 23.1, "Na": 0.1, "S": 0.1, "Cl": 0.1},
    "fascia": {"H": 9.4, "C": 20.7, "N": 6.2, "O": 62.2, "Na": 0.6, "S": 0.6, "Cl": 0.3},
}

# Densidad másica en g/cm³ (ICRU-44 / IT'IS)
DENSITY_G_CM3: dict[str, float] = {
    "water": 1.0, "air": 0.0012, "skin": 1.09, "fat": 0.95, "muscle": 1.05, "blood": 1.06,
    "cortical_bone": 1.92, "marrow": 0.98, "aluminum": 2.699, "fascia": 1.12,
}

# Tejidos definidos como mezcla por volumen de otros (fracciones de volumen).
VOLUME_MIXTURES: dict[str, tuple[tuple[str, float], ...]] = {
    "trabecular_bone": (("cortical_bone", 0.2), ("marrow", 0.8)),  # BV/TV ≈ 20 %
    "hematoma": (("blood", 1.0),),
    "edema": (("marrow", 0.6), ("blood", 0.4)),
    "periosteum": (("fascia", 1.0),),
}

_PE_EXPONENT = 4.0


def _descriptors(comp: dict[str, float]) -> np.ndarray:
    total = sum(comp.values())
    ne = sum(w / total * ATOMIC[e][0] / ATOMIC[e][1] for e, w in comp.items())
    pe = sum(w / total * ATOMIC[e][0] ** _PE_EXPONENT / ATOMIC[e][1] for e, w in comp.items())
    return np.array([ne, pe])


def decomposed_mass_attenuation(comp: dict[str, float]) -> np.ndarray:
    """``mu/rho`` (cm²/g) de una composición elemental vía base agua + hueso."""
    basis = np.column_stack([_descriptors(COMPOSITION["water"]), _descriptors(COMPOSITION["cortical_bone"])])
    a, b = np.linalg.solve(basis, _descriptors(comp))
    return a * TABLES["water"] + b * TABLES["cortical_bone"]


def _mass_table(key: str) -> np.ndarray:
    if key in TABLES:
        return TABLES[key]
    return decomposed_mass_attenuation(COMPOSITION[key])


def linear_attenuation(key: str, energy_kev) -> np.ndarray:
    """Coeficiente de atenuación lineal ``mu`` en 1/m (interpolación log-log)."""
    e = np.asarray(energy_kev, dtype=float)
    if np.any((e < ENERGIES_KEV[0]) | (e > ENERGIES_KEV[-1])):
        raise ValueError("Energía fuera del rango tabulado (10 keV - 10 MeV)")
    if key in VOLUME_MIXTURES:
        return sum(frac * linear_attenuation(k, e) for k, frac in VOLUME_MIXTURES[key])
    table = _mass_table(key)
    mu_rho = np.exp(np.interp(np.log(e), np.log(ENERGIES_KEV), np.log(table)))
    return mu_rho * DENSITY_G_CM3[key] * 100.0  # cm²/g · g/cm³ = 1/cm → 1/m


def hounsfield(mu: np.ndarray, mu_water: float, mu_air: float = 0.0) -> np.ndarray:
    """Convierte coeficientes de atenuación a unidades Hounsfield (HU)."""
    return 1000.0 * (mu - mu_water) / (mu_water - mu_air)
