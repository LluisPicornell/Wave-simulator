"""Rayos X y gamma: radiografía (proyección) y TAC (retroproyección filtrada).

- Haz monoenergético o polienergético (tubo de wolframio, espectro de Kramers
  filtrado por aluminio y, opcionalmente, con líneas características K del W,
  filtración de cobre y autoabsorción en el ánodo según su ángulo).
- Endurecimiento del haz: el espectro pierde fotones de baja energía al
  atravesar el cuerpo; produce el artefacto de "cupping" en el TAC.
- Ruido cuántico de Poisson según el número de fotones por detector.
- La radiación dispersa (scatter), que en la realidad reduce el contraste, se simula
  como opción con un Monte Carlo de fotones (``xray_scatter``).

Para que sea rápido, cada tejido se descompone en base agua + hueso
(``mu_t(E) ≈ a_t mu_agua(E) + b_t mu_hueso(E)``): bastan dos proyecciones
por ángulo para cualquier espectro.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import xray as xd
from ..phantoms.phantom import Phantom
from .fanbeam import FanGeometry, fan_fbp, fan_project, os_sart
from .projection import detector_axis, fbp, project
from .xray_detector import DetectorModel, blur_signal, detect_counts
from .xray_scatter import ScatterOptions, scatter_sinogram


@dataclass
class XraySpectrum:
    energies_kev: np.ndarray
    weights: np.ndarray  # fracción de fotones en cada energía (suma 1)

    @property
    def mean_energy(self) -> float:
        return float(np.sum(self.energies_kev * self.weights))


def monoenergetic(energy_kev: float) -> XraySpectrum:
    return XraySpectrum(np.array([float(energy_kev)]), np.array([1.0]))


# Líneas K del wolframio: energía (keV) e intensidad relativa a Kα1. Energías e intensidades
# recordadas de la tabla de líneas de rayos X de Lawrence Berkeley (X-Ray Data Booklet);
# NO verificadas en esta sesión (sin acceso a la fuente).
W_K_EDGE_KEV = 69.525
W_K_LINES = ((59.318, 1.00), (57.982, 0.58), (67.245, 0.22), (69.067, 0.08))
# Fotones de las líneas K por electrón: proporcionales a (U-1)^1.67 con U = kVp / borde K
# (exponente de Green & Cosslett para la ionización de la capa K), no a la intensidad del
# continuo. La constante está anclada a que valgan el 4 % de los fotones del continuo a
# 100 kVp: calibración heurística, no un ajuste a datos publicados.
_K_YIELD_FRAC_100, _K_YIELD_P = 0.04, 1.67
_W_CM3 = 19.3
# Profundidad media de creación de los fotones: fracción del alcance de Kanaya-Okayama. Ajustada
# (junto con la fracción de líneas K) para que la HVL concuerde con SpekPy 2.5.4 en 6 casos
# (50-120 kVp, 12°, 2.5 y 4 mm Al; ver docs/validacion/rayos_x.md); no es un valor medido.
_DEPTH_FRACTION = 0.27


def _kanaya_okayama_um(e_kev):
    """Alcance de electrones en W (µm), fórmula de Kanaya-Okayama (A=183.84, Z=74)."""
    return 0.0276 * 183.84 * np.asarray(e_kev, float) ** 1.67 / (74.0**0.889 * _W_CM3)


def anode_self_attenuation(e_kev: np.ndarray, kvp: float, anode_angle_deg: float,
                           takeoff_offset_deg: float = 0.0) -> np.ndarray:
    """Transmisión de los fotones a través del propio ánodo de W (autofiltración).

    Los fotones de energía E se crean a profundidad uniforme entre 0 y una fracción del alcance
    que pierde el electrón al bajar de ``kvp`` a E (modelo simple, con la fracción ajustada a SpekPy); salen con el
    ángulo de despegue ``anode_angle_deg - takeoff_offset_deg``: ``takeoff_offset_deg > 0`` es el
    lado del ánodo (rayos más rasantes: efecto talón) y ``< 0`` el del cátodo.
    """
    take = np.radians(anode_angle_deg - takeoff_offset_deg)
    if take <= 0:
        raise ValueError("El ángulo de despegue debe ser positivo")
    depth_m = _DEPTH_FRACTION * 1e-6 * (_kanaya_okayama_um(kvp) - _kanaya_okayama_um(e_kev))
    x = xd.filter_attenuation("tungsten", e_kev) * np.maximum(depth_m, 1e-12) / np.sin(take)
    return (1.0 - np.exp(-x)) / x


def tube_spectrum(kvp: float = 80.0, al_filter_mm: float = 2.5, bin_kev: float = 1.0, *, cu_filter_mm: float = 0.0,
                  characteristic: bool = False, anode_angle_deg: float | None = None,
                  takeoff_offset_deg: float = 0.0) -> XraySpectrum:
    """Espectro aproximado de un tubo de W (ley de Kramers) filtrado por aluminio (y cobre).

    Con los valores por defecto es el espectro de siempre. Opciones:

    - ``cu_filter_mm``: filtración adicional de cobre.
    - ``characteristic``: añade las líneas Kα1, Kα2, Kβ1 y Kβ2 del W si ``kvp`` supera el borde K
      (69.525 keV); sus energías son líneas aparte, no se reparten en el histograma.
    - ``anode_angle_deg``: autoabsorción en el ánodo (ver ``anode_self_attenuation``); con
      ``takeoff_offset_deg`` se modela el efecto talón fuera del eje.
    """
    e = np.arange(10.0 + bin_kev / 2, kvp, bin_kev)
    phi = (kvp - e) / e
    energies = e
    if characteristic and kvp > W_K_EDGE_KEV:
        line_e = np.array([le for le, _ in W_K_LINES])
        rel = np.array([r for _, r in W_K_LINES])
        e100 = np.arange(10.0 + bin_kev / 2, 100.0, bin_kev)
        n_brems_100 = ((100.0 - e100) / e100).sum()
        u_ratio = (kvp / W_K_EDGE_KEV - 1.0) / (100.0 / W_K_EDGE_KEV - 1.0)
        total_lines = _K_YIELD_FRAC_100 * n_brems_100 * u_ratio**_K_YIELD_P
        energies = np.concatenate([e, line_e])
        phi = np.concatenate([phi, total_lines * rel / rel.sum()])
    if anode_angle_deg is not None:
        phi = phi * anode_self_attenuation(energies, kvp, anode_angle_deg, takeoff_offset_deg)
    phi = phi * np.exp(-xd.linear_attenuation("aluminum", energies) * al_filter_mm * 1e-3)
    if cu_filter_mm:
        phi = phi * np.exp(-xd.filter_attenuation("copper", energies) * cu_filter_mm * 1e-3)
    order = np.argsort(energies)
    energies, phi = energies[order], phi[order]
    return XraySpectrum(energies, phi / phi.sum())


def air_kerma_weights(spectrum: XraySpectrum) -> np.ndarray:
    """Pesos de kerma en aire de cada energía del espectro: ``w_i E_i (mu_en/rho)_aire``."""
    return spectrum.weights * spectrum.energies_kev * xd.air_mass_energy_absorption(spectrum.energies_kev)


def half_value_layer(spectrum: XraySpectrum, material: str = "aluminum", fraction: float = 0.5) -> float:
    """Capa hemirreductora (mm) del kerma en aire: espesor que lo reduce a ``fraction`` (0.5 = HVL).

    Se resuelve por bisección sobre el espectro discreto (sin aproximar por un haz monoenergético).
    """
    kw = air_kerma_weights(spectrum)
    kw = kw / kw.sum()
    mu = xd.filter_attenuation(material, spectrum.energies_kev)

    def kerma(t_mm):
        return float(np.sum(kw * np.exp(-mu * t_mm * 1e-3)))

    lo, hi = 0.0, 1.0
    while kerma(hi) > fraction:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if kerma(mid) > fraction else (lo, mid)
    return 0.5 * (lo + hi)


def fit_al_filtration(kvp: float, hvl_mm: float, **kw) -> float:
    """Espesor de Al (mm) de filtración total con el que ``tube_spectrum`` da la HVL ``hvl_mm``."""
    lo, hi = 0.0, 40.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if half_value_layer(tube_spectrum(kvp, mid, **kw)) < hvl_mm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
    geometry: FanGeometry | None = None  # None = haz paralelo
    primary: np.ndarray | None = None  # I/I0 del haz primario (sin ruido); None si no se pidió scatter
    scatter: np.ndarray | None = None  # I/I0 de la radiación dispersa (sin ruido)


def acquire(phantom: Phantom, angles_deg, spectrum: XraySpectrum | None = None, photons: float | None = 1e5,
            det_spacing: float | None = None, rng: np.random.Generator | None = None,
            geometry: FanGeometry | None = None, scatter: ScatterOptions | None = None,
            detector: DetectorModel | None = None) -> Projections:
    """Adquiere proyecciones de rayos X del fantoma.

    ``photons``: fotones por elemento de detector y proyección sin objeto
    (None = sin ruido). Valores típicos: 1e4-1e6. Con ``geometry`` (``FanGeometry``) el haz es en
    abanico con fuente puntual; ``det`` son entonces las posiciones de los elementos del detector.
    Con ``scatter`` (``ScatterOptions``) se suma la radiación dispersa calculada por Monte Carlo
    (``xray_scatter``) a la intensidad primaria antes del ruido de Poisson. Con ``detector``
    (``DetectorModel``) se aplica el desenfoque del detector a la señal esperada y se añade su ruido
    electrónico.
    """
    spectrum = spectrum or tube_spectrum()
    g = phantom.grid
    det = detector_axis(g, det_spacing) if geometry is None else geometry.u
    # Los coeficientes son lineales en mu: la mezcla lineal de la porosidad es exacta
    coef = {k: basis_coefficients(k, spectrum) for k in phantom.property_tissues()}
    maps = np.stack([phantom.property_map(lambda k: coef[k][0], physics="xray"),
                     phantom.property_map(lambda k: coef[k][1], physics="xray")])
    if geometry is None:
        lw, lb = project(maps, g, angles_deg, det)  # longitudes equivalentes de agua y hueso (m)
    else:
        lw, lb = fan_project(maps, g, geometry, angles_deg)
    e = spectrum.energies_kev
    mu_w = xd.linear_attenuation("water", e)
    mu_b = xd.linear_attenuation("cortical_bone", e)
    expo = np.exp(-(lw[..., None] * mu_w + lb[..., None] * mu_b))
    intensity = np.sum(spectrum.weights * expo, axis=-1)
    primary = scattered = None
    if scatter is not None:
        primary = intensity
        scattered, _ = scatter_sinogram(phantom, angles_deg, spectrum, geometry, None if geometry else det, scatter)
        intensity = primary + scattered
    spacing = float(det[1] - det[0])
    if photons is not None:
        rng = rng or np.random.default_rng(0)
        if detector is None:
            counts = np.maximum(rng.poisson(intensity * photons), 1)
        else:
            counts = detect_counts(intensity * photons, spacing, detector, rng)
        intensity = counts / photons
    elif detector is not None:
        intensity = blur_signal(intensity, spacing, detector)
    return Projections(np.asarray(angles_deg, float), det, -np.log(intensity), intensity, spectrum, geometry,
                       primary, scattered)


def radiograph(phantom: Phantom, angle_deg: float = 0.0, spectrum: XraySpectrum | None = None, **kw) -> Projections:
    """Radiografía de un corte 2D: un perfil 1D de transmisión (por defecto 60 kVp, típico de extremidades)."""
    return acquire(phantom, [angle_deg], spectrum or tube_spectrum(60.0, 2.5), **kw)


@dataclass
class CTImage:
    mu: np.ndarray  # 1/m
    hu: np.ndarray
    projections: Projections


def ct_scan(phantom: Phantom, n_angles: int = 360, spectrum: XraySpectrum | None = None, photons: float | None = 1e5,
            beam_hardening_correction: bool = True, rng: np.random.Generator | None = None,
            geometry: FanGeometry | None = None, method: str = "fbp", scatter: ScatterOptions | None = None,
            detector: DetectorModel | None = None, **recon_kw) -> CTImage:
    """TAC y reconstrucción en la malla del fantoma. Por defecto haz paralelo sobre 180° y FBP.

    Con ``geometry`` (``FanGeometry``) el haz es en abanico y ``n_angles`` vistas cubren 360°.
    ``method="sart"`` reconstruye con ``fanbeam.os_sart`` (``recon_kw``: ``n_iter``, ``subset_size``,
    ``tv_weight``...) y exige ``geometry``; sirve para pocos ángulos o ruido alto. ``scatter``
    (``ScatterOptions``) añade la radiación dispersa sin corregirla: baja el contraste y produce
    *cupping*. ``detector`` (``DetectorModel``) añade desenfoque y ruido electrónico.

    Con ``beam_hardening_correction`` las proyecciones se linealizan para agua
    (como hace un escáner real): el agua y los tejidos blandos salen sin
    "cupping"; el hueso conserva el endurecimiento residual (bandas oscuras
    entre huesos), igual que en la clínica.
    """
    spectrum = spectrum or tube_spectrum(120.0, 6.0)  # TAC clínico típico
    if method not in ("fbp", "sart"):
        raise ValueError(f"método desconocido '{method}' (use 'fbp' o 'sart')")
    if method == "sart" and geometry is None:
        raise ValueError("method='sart' requiere una geometría en abanico")
    angles = np.arange(n_angles) * (180.0 if geometry is None else 360.0) / n_angles
    proj = acquire(phantom, angles, spectrum, photons, rng=rng, geometry=geometry, scatter=scatter,
                   detector=detector)
    e_ref = float(np.clip(spectrum.mean_energy, xd.ENERGIES_KEV[0], xd.ENERGIES_KEV[-1]))
    mu_ref = float(xd.linear_attenuation("water", e_ref))
    sino = proj.line_integrals
    if beam_hardening_correction and spectrum.energies_kev.size > 1:
        sino = water_linearize(sino, spectrum, mu_ref)
        mu_w = mu_ref
    else:
        mu_w = effective_water_mu(spectrum, typical_path(proj)) if spectrum.energies_kev.size > 1 else mu_ref
    if geometry is None:
        mu = fbp(sino, angles, proj.det, phantom.grid)
    elif method == "fbp":
        mu = fan_fbp(sino, angles, geometry, phantom.grid)
    else:
        mu = os_sart(sino, angles, geometry, phantom.grid, **recon_kw)
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
