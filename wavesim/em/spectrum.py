"""Todo el espectro electromagnético (y el ultrasonido) frente a los tejidos.

Cada banda necesita una física distinta; este módulo elige la adecuada:

| Banda                   | Frecuencia          | Modelo físico                                    | Solver de imagen          |
|-------------------------|---------------------|--------------------------------------------------|---------------------------|
| Baja frecuencia         | < 10 MHz            | cuasi-estático (λ ≫ cuerpo), Gabriel             | sólo atenuación analítica |
| Radiofrecuencia / micro | 10 MHz - 300 GHz    | onda completa, Maxwell, Gabriel                  | ``fdtd_tm``               |
| Terahercios             | 0.3 - 10 THz        | absorción del agua (doble Debye / tablas)        | ``projection`` (balística)|
| Infrarrojo / visible    | 10 THz - 790 THz    | transporte de luz: difusión o balística          | ``optics``                |
| Ultravioleta            | 790 THz - 1.5 PHz   | absorción muy fuerte (balística)                 | ``projection``            |
| UV extremo / X blandos  | 6 eV - 1 keV        | sin datos tabulados (penetración < 1 µm)         | -                         |
| Rayos X blandos         | 1 - 10 keV          | Beer-Lambert (agua escalada)                     | ``projection``            |
| Rayos X / gamma         | 10 keV - 10 MeV     | Beer-Lambert, tablas NIST                        | ``xray``                  |

La profundidad de penetración que se devuelve es la distancia a la que la
*intensidad* (energía) cae a 1/e. En la luz difusa la energía llega más lejos
que la información nítida (fotones balísticos), y se distinguen ambas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import dielectric, optical
from ..materials import xray as xd
from ..materials.acoustic import ACOUSTIC
from ..phantoms.phantom import TISSUE_INDEX, Phantom

H_PLANCK = 6.62607015e-34
EV = 1.602176634e-19
C0 = dielectric.C0
HZ_PER_KEV = 1e3 * EV / H_PLANCK  # 2.418e17 Hz

DIELECTRIC_MAX_F = 1e12  # hasta 1 THz se usa la permitividad (Gabriel + agua)
OPTICAL_MIN_NM = 200.0

# mu/rho del agua entre 1 y 10 keV (NIST), para extender los rayos X blandos
_WATER_SOFT = np.array([[1.0, 4078.0], [1.5, 1376.0], [2.0, 617.3], [3.0, 192.9], [4.0, 82.78],
                        [5.0, 42.58], [6.0, 24.64], [8.0, 10.37], [10.0, 5.329]])


@dataclass(frozen=True)
class Band:
    name: str
    f_min: float
    f_max: float
    physics: str
    solver: str


BANDS: tuple[Band, ...] = (
    Band("Baja frecuencia (cuasi-estático)", 1.0, 1e7, "cuasi-estático", "analítico"),
    Band("Radiofrecuencia", 1e7, 3e8, "onda completa (Maxwell)", "em.fdtd_tm"),
    Band("Microondas", 3e8, 3e11, "onda completa (Maxwell)", "em.fdtd_tm"),
    Band("Terahercios", 3e11, 1e13, "absorción del agua", "em.projection"),
    Band("Infrarrojo", 1e13, 3.85e14, "transporte de luz", "em.optics"),
    Band("Visible", 3.85e14, 7.9e14, "difusión de luz", "em.optics"),
    Band("Ultravioleta", 7.9e14, 1.5e15, "absorción fuerte", "em.projection"),
    Band("UV extremo / X muy blandos", 1.5e15, 1.0 * HZ_PER_KEV, "sin datos", "-"),
    Band("Rayos X blandos", 1.0 * HZ_PER_KEV, 10 * HZ_PER_KEV, "Beer-Lambert", "em.projection"),
    Band("Rayos X", 10 * HZ_PER_KEV, 150 * HZ_PER_KEV, "Beer-Lambert", "em.xray"),
    Band("Rayos gamma", 150 * HZ_PER_KEV, 1e4 * HZ_PER_KEV, "Beer-Lambert", "em.xray"),
)


def band_of(f: float) -> Band:
    for b in BANDS:
        if b.f_min <= f < b.f_max:
            return b
    return BANDS[-1] if f >= BANDS[-1].f_min else BANDS[0]


def wavelength_nm(f):
    return C0 / np.asarray(f, dtype=float) * 1e9


def energy_kev(f):
    return np.asarray(f, dtype=float) / HZ_PER_KEV


def attenuation(tissue: str, f: float) -> dict:
    """Atenuación de intensidad (1/m) del tejido a la frecuencia ``f``.

    Devuelve ``{"mu": energía, "mu_imaging": fotones/onda sin desviar, "regime": str}``.
    ``mu`` y ``mu_imaging`` son NaN donde no hay datos.
    """
    f = float(f)
    if f <= DIELECTRIC_MAX_F:
        _, a = dielectric.wave_numbers(tissue, f)
        mu = 2.0 * float(a)
        return {"mu": mu, "mu_imaging": mu, "regime": "onda (permitividad)"}
    lam = float(wavelength_nm(f))
    if lam >= OPTICAL_MIN_NM:
        if tissue == "air":
            return {"mu": 0.0, "mu_imaging": 0.0, "regime": "aire"}
        mua = float(optical.mu_a(tissue, lam))
        mus = float(optical.mu_s_reduced(tissue, lam))
        mut = float(optical.mu_total(tissue, lam))
        if mus > mua:
            return {"mu": float(np.sqrt(3 * mua * (mua + mus))), "mu_imaging": mut, "regime": "difusión"}
        return {"mu": mut, "mu_imaging": mut, "regime": "balístico"}
    e = float(energy_kev(f))
    if e < 1.0:
        return {"mu": np.nan, "mu_imaging": np.nan, "regime": "sin datos"}
    if e < 10.0:
        # agua escalada por densidad y por la relación tejido/agua a 10 keV
        water = np.exp(np.interp(np.log(e), np.log(_WATER_SOFT[:, 0]), np.log(_WATER_SOFT[:, 1]))) * 100.0
        ratio = float(xd.linear_attenuation(tissue, 10.0) / xd.linear_attenuation("water", 10.0))
        mu = float(water * ratio)
        return {"mu": mu, "mu_imaging": mu, "regime": "Beer-Lambert (aprox.)"}
    if e <= 1e4:
        mu = float(xd.linear_attenuation(tissue, e))
        return {"mu": mu, "mu_imaging": mu, "regime": "Beer-Lambert"}
    return {"mu": np.nan, "mu_imaging": np.nan, "regime": "fuera de rango"}


def penetration_depth(tissue: str, f: float, imaging: bool = False) -> float:
    """Profundidad (m) a la que la intensidad cae a 1/e."""
    a = attenuation(tissue, f)
    mu = a["mu_imaging"] if imaging else a["mu"]
    return 1.0 / mu if mu and np.isfinite(mu) and mu > 0 else (np.inf if mu == 0 else np.nan)


def ultrasound_penetration_depth(tissue: str, f: float) -> float:
    """Profundidad (m) a la que la intensidad del ultrasonido cae a 1/e."""
    p = ACOUSTIC[tissue]
    alpha = p.alpha_np_per_m(f)
    return 1.0 / (2.0 * alpha) if alpha > 0 else np.inf


def column_layers(phantom: Phantom, x: float = 0.0, skip=("water", "air")) -> list[tuple[str, float]]:
    """Secuencia de capas ``(tejido, grosor)`` que atraviesa una línea vertical en ``x``."""
    _, i = phantom.grid.index_of(x, 0.0)
    keys = list(TISSUE_INDEX)
    col = phantom.labels[:, i]
    layers: list[tuple[str, float]] = []
    for lab in col:
        k = keys[lab]
        if k in skip:
            continue
        if layers and layers[-1][0] == k:
            layers[-1] = (k, layers[-1][1] + phantom.grid.dx)
        else:
            layers.append((k, phantom.grid.dx))
    return layers


def through_transmission_db(layers: list[tuple[str, float]], f: float, modality: str = "em",
                            include_interfaces: bool = True, outer: str = "water") -> dict:
    """Pérdida (dB) de una onda que cruza las capas a incidencia normal.

    ``modality``: "em" (cualquier frecuencia electromagnética) o "ultrasound".
    Las pérdidas por reflexión en las interfaces se incluyen donde el modelo
    de onda plana es válido (ultrasonido y EM entre 10 MHz y 1 THz).
    """
    absorb = 0.0
    for k, th in layers:
        if modality == "ultrasound":
            mu = 2.0 * ACOUSTIC[k].alpha_np_per_m(f)
        else:
            mu = attenuation(k, f)["mu"]
        absorb += 10 * np.log10(np.e) * mu * th
    refl = 0.0
    if include_interfaces and layers:
        seq = [outer] + [k for k, _ in layers] + [outer]
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                continue
            if modality == "ultrasound":
                za, zb = ACOUSTIC[a].impedance, ACOUSTIC[b].impedance
                t = 4 * za * zb / (za + zb) ** 2
            elif f < 1e7:
                t = 1.0  # cuasi-estático: λ ≫ cuerpo, la reflexión de onda plana no tiene sentido
            elif f <= DIELECTRIC_MAX_F:
                ea = dielectric.intrinsic_impedance(a, f)
                eb = dielectric.intrinsic_impedance(b, f)
                gamma = (eb - ea) / (eb + ea)
                t = 1 - abs(gamma) ** 2
            else:
                t = 1.0
            refl += -10 * np.log10(max(t, 1e-30))
    return {"absorption_db": absorb, "reflection_db": refl, "total_db": absorb + refl}


def survey_frequencies(n_per_decade: int = 8) -> np.ndarray:
    """Frecuencias de 1 kHz a 10 MeV (≈2.4e21 Hz) para barrer el espectro."""
    return np.logspace(3, np.log10(1e4 * HZ_PER_KEV), int((np.log10(1e4 * HZ_PER_KEV) - 3) * n_per_decade))


def penetration_survey(tissues: list[str], freqs: np.ndarray | None = None, imaging: bool = False) -> dict:
    """Profundidad de penetración (m) de cada tejido en todo el espectro EM."""
    freqs = survey_frequencies() if freqs is None else freqs
    return {"freqs": freqs, **{k: np.array([penetration_depth(k, f, imaging) for f in freqs]) for k in tissues}}
