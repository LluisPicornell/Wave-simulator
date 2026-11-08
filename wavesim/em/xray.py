"""Rayos X y gamma: radiografía (proyección) y TAC (retroproyección filtrada).

- Haz monoenergético o polienergético (tubo de wolframio, espectro de Kramers
  filtrado por aluminio; se omiten las líneas características K del W).
- Endurecimiento del haz: el espectro pierde fotones de baja energía al
  atravesar el cuerpo; produce el artefacto de "cupping" en el TAC.
- Ruido cuántico de Poisson según el número de fotones por detector.
- No se simula la radiación dispersa (scatter), que en la realidad reduce el
  contraste.

Para que sea rápido, cada tejido se descompone en base agua + hueso
(``mu_t(E) ≈ a_t mu_agua(E) + b_t mu_hueso(E)``): bastan dos proyecciones
por ángulo para cualquier espectro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import xray as xd
from ..phantoms.phantom import Phantom
from .projection import detector_axis, fbp, project


@dataclass
class XraySpectrum:
    energies_kev: np.ndarray
    weights: np.ndarray  # fracción de fotones en cada energía (suma 1)

    @property
    def mean_energy(self) -> float:
        return float(np.sum(self.energies_kev * self.weights))


def monoenergetic(energy_kev: float) -> XraySpectrum:
    return XraySpectrum(np.array([float(energy_kev)]), np.array([1.0]))


def tube_spectrum(kvp: float = 80.0, al_filter_mm: float = 2.5, bin_kev: float = 1.0) -> XraySpectrum:
    """Espectro aproximado de un tubo de W (ley de Kramers) filtrado por aluminio."""
    e = np.arange(10.0 + bin_kev / 2, kvp, bin_kev)
    phi = (kvp - e) / e
    phi *= np.exp(-xd.linear_attenuation("aluminum", e) * al_filter_mm * 1e-3)
    return XraySpectrum(e, phi / phi.sum())


def basis_coefficients(tissue: str, spectrum: XraySpectrum) -> tuple[float, float]:
    """Coeficientes (a, b) tales que mu_t ≈ a·mu_agua + b·mu_hueso en las energías del espectro."""
    e = spectrum.energies_kev
    w = np.sqrt(spectrum.weights + 1e-6)
    A = np.column_stack([xd.linear_attenuation("water", e), xd.linear_attenuation("cortical_bone", e)]) * w[:, None]
    y = xd.linear_attenuation(tissue, e) * w
    if e.size == 1:
        mu_w = xd.linear_attenuation("water", e)[0]
        return float(xd.linear_attenuation(tissue, e)[0] / mu_w), 0.0
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


@dataclass
class Projections:
    angles_deg: np.ndarray
    det: np.ndarray
    line_integrals: np.ndarray  # -ln(I/I0) medido (con ruido)
    intensity: np.ndarray  # I/I0
    spectrum: XraySpectrum


def acquire(phantom: Phantom, angles_deg, spectrum: XraySpectrum | None = None, photons: float | None = 1e5,
            det_spacing: float | None = None, rng: np.random.Generator | None = None) -> Projections:
    """Adquiere proyecciones de rayos X del fantoma.

    ``photons``: fotones por elemento de detector y proyección sin objeto
    (None = sin ruido). Valores típicos: 1e4-1e6.
    """
    spectrum = spectrum or tube_spectrum()
    g = phantom.grid
    det = detector_axis(g, det_spacing)
    # Los coeficientes son lineales en mu: la mezcla lineal de la porosidad es exacta
    coef = {k: basis_coefficients(k, spectrum) for k in phantom.property_tissues()}
    maps = np.stack([phantom.property_map(lambda k: coef[k][0], physics="xray"),
                     phantom.property_map(lambda k: coef[k][1], physics="xray")])
    lw, lb = project(maps, g, angles_deg, det)  # longitudes equivalentes de agua y hueso (m)
    e = spectrum.energies_kev
    mu_w = xd.linear_attenuation("water", e)
    mu_b = xd.linear_attenuation("cortical_bone", e)
    expo = np.exp(-(lw[..., None] * mu_w + lb[..., None] * mu_b))
    intensity = np.sum(spectrum.weights * expo, axis=-1)
    if photons is not None:
        rng = rng or np.random.default_rng(0)
        counts = np.maximum(rng.poisson(intensity * photons), 1)
        intensity = counts / photons
    return Projections(np.asarray(angles_deg, float), det, -np.log(intensity), intensity, spectrum)


def radiograph(phantom: Phantom, angle_deg: float = 0.0, spectrum: XraySpectrum | None = None, **kw) -> Projections:
    """Radiografía de un corte 2D: un perfil 1D de transmisión (por defecto 60 kVp, típico de extremidades)."""
    return acquire(phantom, [angle_deg], spectrum or tube_spectrum(60.0, 2.5), **kw)


@dataclass
class CTImage:
    mu: np.ndarray  # 1/m
    hu: np.ndarray
    projections: Projections


def ct_scan(phantom: Phantom, n_angles: int = 360, spectrum: XraySpectrum | None = None, photons: float | None = 1e5,
            beam_hardening_correction: bool = True, rng: np.random.Generator | None = None) -> CTImage:
    """TAC de haz paralelo sobre 180° y reconstrucción por FBP en la malla del fantoma.

    Con ``beam_hardening_correction`` las proyecciones se linealizan para agua
    (como hace un escáner real): el agua y los tejidos blandos salen sin
    "cupping"; el hueso conserva el endurecimiento residual (bandas oscuras
    entre huesos), igual que en la clínica.
    """
    spectrum = spectrum or tube_spectrum(120.0, 6.0)  # TAC clínico típico
    angles = np.arange(n_angles) * 180.0 / n_angles
    proj = acquire(phantom, angles, spectrum, photons, rng=rng)
    e_ref = float(np.clip(spectrum.mean_energy, xd.ENERGIES_KEV[0], xd.ENERGIES_KEV[-1]))
    mu_ref = float(xd.linear_attenuation("water", e_ref))
    sino = proj.line_integrals
    if beam_hardening_correction and spectrum.energies_kev.size > 1:
        sino = water_linearize(sino, spectrum, mu_ref)
        mu_w = mu_ref
    else:
        mu_w = effective_water_mu(spectrum, typical_path(proj)) if spectrum.energies_kev.size > 1 else mu_ref
    mu = fbp(sino, angles, proj.det, phantom.grid)
    return CTImage(mu, xd.hounsfield(mu, mu_w, 0.0), proj)


def water_linearize(line_integrals: np.ndarray, spectrum: XraySpectrum, mu_ref: float) -> np.ndarray:
    """Convierte ``-ln(I/I0)`` polienergético en ``mu_ref × longitud de agua equivalente``."""
    mu_w = xd.linear_attenuation("water", spectrum.energies_kev)
    l_max = 2.0 * max(float(line_integrals.max()), 1.0) / float(mu_w.min())
    L = np.linspace(0.0, l_max, 4000)
    poly = -np.log(np.sum(spectrum.weights[None, :] * np.exp(-np.outer(L, mu_w)), axis=1))
    return np.interp(line_integrals, poly, mu_ref * L)


def effective_water_mu(spectrum: XraySpectrum, path: float) -> float:
    """Atenuación efectiva del agua tras ``path`` metros (incluye el endurecimiento del haz)."""
    mu = xd.linear_attenuation("water", spectrum.energies_kev)
    return float(-np.log(np.sum(spectrum.weights * np.exp(-mu * path))) / path)


def typical_path(proj: Projections) -> float:
    """Longitud de agua equivalente típica de los rayos que atraviesan el objeto."""
    mu_w = effective_water_mu(proj.spectrum, 1e-3)
    li = proj.line_integrals[proj.line_integrals > 0.05]
    return float(np.median(li) / mu_w) if li.size else 0.05
