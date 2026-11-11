"""Secciones eficaces por proceso (fotoeléctrico, Compton, Rayleigh) y distribuciones angulares.

Origen de cada una (importante: ninguna sale de la base XCOM, que no era accesible):

- **Total** ``mu(E)``: el de ``materials.xray.linear_attenuation`` (tablas de Hubbell y Seltzer para
  agua, músculo y hueso y descomposición agua+hueso para el resto). Los tres procesos suman
  exactamente este total, así que el haz primario del Monte Carlo coincide con Beer-Lambert.
- **Compton (incoherente)**: Klein-Nishina por electrón multiplicada por la función de dispersión
  incoherente ``S(x, Z)`` de cada átomo (electrones ligados; corrige a baja energía y ángulo
  pequeño). Se integra sobre el ángulo y se suma por elementos con la composición ICRU-44 del
  tejido. No hay ensanchamiento Doppler (las energías salientes son las de Compton libre).
- **Rayleigh (coherente)**: ``r_e²/2 (1 + cos²θ) F(x, Z)²`` por átomo, integrado igual, con ``F`` el
  factor de forma atómico. Se suma por átomos independientes (sin interferencia molecular).
- **Fotoeléctrico**: lo que falta para llegar al total tabulado, ``mu - mu_coh - mu_inc`` (con un suelo
  del 0.1 % del total). Incluye por tanto cualquier error de los dos modelos anteriores; a energías
  altas donde el fotoeléctrico es despreciable sirve de control (ver ``tests/test_xray_scatter.py``).
  No se simula la fluorescencia: el fotón fotoeléctrico se absorbe localmente.

``S`` y ``F`` son un modelo de Slater aproximado: cada capa de ``N_s`` electrones se trata como un
orbital con densidad radial ``r^(2n-2) exp(-2ζr)`` y ``ζ = Zeff / n`` (reglas de Slater); su factor de
forma es analítico, ``F = sum N_s f_s`` y ``S = Z - sum N_s f_s²``. Es un factor de forma de manual, no el de Hubbell o
Waller-Hartree: se espera un error de decenas por ciento en la sección coherente (el 2-6 % de la
atenuación total entre 20 y 100 keV en agua), es decir, ~1 % del total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import xray as xd

R_E2_CM2 = 7.9407877e-26  # r_e² en cm²
N_A = 6.02214076e23
MEC2_KEV = 510.99895
HC_KEV_A = 12.398419843  # hc en keV·Å
A0_ANGSTROM = 0.529177210903
_CAPACITY = (2, 8, 18, 32)

_AIR = {"N": 75.5, "O": 24.5}  # sin argón: sólo para el reparto de procesos del aire
_EXTRA_COMPOSITION = {"air": _AIR}


def _atomic(symbol: str) -> tuple[int, float]:
    if symbol == "Al":
        return 13, 26.982
    return xd.ATOMIC[symbol]


def shells(z: int) -> list[tuple[int, int, float]]:
    """Capas (n, electrones, exponente ζ en 1/a0) de un átomo neutro con las reglas de Slater."""
    occ, left = [], z
    for cap in _CAPACITY:
        if left <= 0:
            break
        occ.append(min(cap, left))
        left -= cap
    out = []
    for n, ne in enumerate(occ, start=1):
        inner = sum(occ[:n - 2]) if n > 2 else 0
        prev = occ[n - 2] if n > 1 else 0
        screen = 0.30 * (ne - 1) if n == 1 else 0.35 * (ne - 1) + 0.85 * prev + 1.0 * inner
        out.append((n, ne, max(z - screen, 1.0) / n))
    return out


def _shell_factors(x: np.ndarray, z: int) -> tuple[np.ndarray, np.ndarray]:
    """``F(x, Z)`` y ``S(x, Z)``; ``x = sin(θ/2)/λ`` en 1/Å."""
    q = 4.0 * np.pi * np.asarray(x, float) * A0_ANGSTROM  # q·a0
    f_total = np.zeros_like(q)
    f_sq = np.zeros_like(q)
    q = np.maximum(q, 1e-8)
    for n, ne, zeta in shells(z):
        # Factor de forma de un orbital de Slater con densidad r^(2n-2) exp(-2 ζ r) (s de capa n)
        a = 2.0 * zeta
        f = a ** (2 * n + 1) * np.sin(2 * n * np.arctan(q / a)) / (2 * n * q * (a * a + q * q) ** n)
        f_total += ne * f
        f_sq += ne * f * f
    return f_total, z - f_sq


def form_factor(x, z: int) -> np.ndarray:
    return _shell_factors(x, z)[0]


def incoherent_function(x, z: int) -> np.ndarray:
    return _shell_factors(x, z)[1]


def composition_of(key: str) -> dict[str, float]:
    return xd.COMPOSITION.get(key) or _EXTRA_COMPOSITION[key]


def _atoms_per_gram(comp: dict[str, float]) -> dict[str, float]:
    total = sum(comp.values())
    return {e: w / total * N_A / _atomic(e)[1] for e, w in comp.items()}


_GL_X, _GL_W = np.polynomial.legendre.leggauss(400)


def _element_cross_sections(e_kev: np.ndarray, z: int) -> tuple[np.ndarray, np.ndarray]:
    """(``sigma_coh``, ``sigma_inc``) por átomo en cm² para cada energía."""
    k = e_kev[:, None] / MEC2_KEV
    c = _GL_X[None, :]  # cos θ
    ratio = 1.0 / (1.0 + k * (1.0 - c))
    x = e_kev[:, None] / HC_KEV_A * np.sqrt(np.maximum(0.5 * (1.0 - c), 0.0))
    f, s = _shell_factors(x, z)
    kn = 0.5 * R_E2_CM2 * ratio**2 * (ratio + 1.0 / ratio - (1.0 - c**2))
    th = 0.5 * R_E2_CM2 * (1.0 + c**2)
    sigma_inc = 2.0 * np.pi * np.sum(kn * s * _GL_W, axis=1)
    sigma_coh = 2.0 * np.pi * np.sum(th * f * f * _GL_W, axis=1)
    return sigma_coh, sigma_inc


def process_attenuation(key: str, energy_kev) -> dict[str, np.ndarray]:
    """Coeficientes lineales (1/m) ``photoelectric``, ``incoherent``, ``coherent`` y ``total`` de un tejido.

    Los tejidos definidos como mezcla por volumen son la suma ponderada de sus componentes.
    """
    e = np.atleast_1d(np.asarray(energy_kev, float))
    if key in xd.VOLUME_MIXTURES:
        parts = [(f, process_attenuation(k, e)) for k, f in xd.VOLUME_MIXTURES[key]]
        return {n: sum(f * p[n] for f, p in parts) for n in ("photoelectric", "incoherent", "coherent", "total")}
    rho = xd.DENSITY_G_CM3[key]
    coh = np.zeros_like(e)
    inc = np.zeros_like(e)
    for el, n_per_g in _atoms_per_gram(composition_of(key)).items():
        sc, si = _element_cross_sections(e, _atomic(el)[0])
        coh += n_per_g * sc
        inc += n_per_g * si
    coh *= rho * 100.0  # cm²/g · g/cm³ = 1/cm → 1/m
    inc *= rho * 100.0
    total = xd.linear_attenuation(key, e)
    pe = np.maximum(total - coh - inc, 1e-3 * total)
    return {"photoelectric": pe, "incoherent": inc, "coherent": coh, "total": total}


def compton_s_over_z(key: str, x_grid: np.ndarray) -> np.ndarray:
    """``S(x)`` del material dividida por su número de electrones, para el rechazo en Compton."""
    if key in xd.VOLUME_MIXTURES:
        # Promedio ponderado por la densidad electrónica de cada componente (≈ fracciones de volumen)
        acc, norm = np.zeros_like(x_grid), 0.0
        for k, f in xd.VOLUME_MIXTURES[key]:
            w = f * xd.DENSITY_G_CM3[k] * sum(
                n * _atomic(e)[0] for e, n in _atoms_per_gram(composition_of(k)).items())
            acc += w * compton_s_over_z(k, x_grid)
            norm += w
        return acc / norm
    num, den = np.zeros_like(x_grid), 0.0
    for el, n_per_g in _atoms_per_gram(composition_of(key)).items():
        z = _atomic(el)[0]
        num += n_per_g * incoherent_function(x_grid, z)
        den += n_per_g * z
    return num / den


def rayleigh_cdf(key: str, energy_kev: float, n_cos: int = 1024) -> np.ndarray:
    """CDF de ``cos θ`` (rejilla uniforme en [-1, 1]) de la dispersión coherente del tejido a esa energía."""
    c = np.linspace(-1.0, 1.0, n_cos)
    x = energy_kev / HC_KEV_A * np.sqrt(0.5 * (1.0 - c))
    if key in xd.VOLUME_MIXTURES:
        pdf = np.zeros_like(c)
        for k, f in xd.VOLUME_MIXTURES[key]:
            cdf = rayleigh_cdf(k, energy_kev, n_cos)
            w = f * process_attenuation(k, energy_kev)["coherent"][0]
            pdf += w * np.gradient(cdf)
        cum = np.cumsum(pdf)
    else:
        pdf = np.zeros_like(c)
        for el, n_per_g in _atoms_per_gram(composition_of(key)).items():
            pdf += n_per_g * form_factor(x, _atomic(el)[0]) ** 2
        cum = np.cumsum(pdf * (1.0 + c**2))
    return cum / cum[-1]


@dataclass
class ProcessTables:
    """Tablas para el Monte Carlo: energías de 1 keV, coeficientes por material y distribuciones."""

    materials: list[str]
    energies_kev: np.ndarray  # rejilla uniforme (paso 1 keV)
    mu_pe: np.ndarray  # (n_mat, n_e) 1/m
    mu_coh: np.ndarray
    mu_inc: np.ndarray
    x_grid: np.ndarray  # 1/Å, uniforme
    s_over_z: np.ndarray  # (n_mat, n_x)
    ray_energies_kev: np.ndarray  # rejilla gruesa de energías del CDF de Rayleigh
    ray_cdf: np.ndarray  # (n_mat, n_ray_e, n_cos)

    @property
    def mu_total(self) -> np.ndarray:
        return self.mu_pe + self.mu_coh + self.mu_inc


def build_tables(materials: list[str], e_max_kev: float = 150.0, e_min_kev: float = 10.0,
                 ray_step_kev: float = 5.0) -> ProcessTables:
    """Calcula las tablas de ``materials`` entre ``e_min_kev`` y ``e_max_kev`` (<= 10 MeV; el Monte Carlo
    de este módulo se valida hasta 200 keV)."""
    e = np.arange(e_min_kev, np.ceil(e_max_kev) + 2.0, 1.0)
    procs = [process_attenuation(m, e) for m in materials]
    x_grid = np.linspace(0.0, 1.05 * e[-1] / HC_KEV_A, 2048)
    ray_e = np.arange(e_min_kev, e[-1] + ray_step_kev, ray_step_kev)
    return ProcessTables(
        list(materials), e,
        np.array([p["photoelectric"] for p in procs]), np.array([p["coherent"] for p in procs]),
        np.array([p["incoherent"] for p in procs]), x_grid,
        np.array([compton_s_over_z(m, x_grid) for m in materials]), ray_e,
        np.array([[rayleigh_cdf(m, ee) for ee in ray_e] for m in materials]),
    )
