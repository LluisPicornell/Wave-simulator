"""Propiedades dieléctricas de los tejidos (ondas de radio, microondas, ondas mm).

Modelo de Cole-Cole de 4 polos de Gabriel, Lau & Gabriel, "The dielectric
properties of biological tissues: III. Parametric models for the dielectric
spectrum of tissues", Phys. Med. Biol. 41, 2271 (1996):

    eps(w) = eps_inf + sum_n  d_eps_n / (1 + (j w tau_n)^(1 - alpha_n))  + sigma_i / (j w eps0)

Válido aproximadamente entre 10 Hz y 100 GHz. Por encima, la absorción la
domina el agua y se usa el modelo de doble Debye del agua (Kindt &
Schmuttenmaer 1996), escalado por la fracción de agua del tejido.

Convención: ``eps = eps' - j eps''`` y conductividad efectiva
``sigma = w eps0 eps''`` (incluye las pérdidas dieléctricas).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi
C0 = 299_792_458.0

PS, NS, US, MS = 1e-12, 1e-9, 1e-6, 1e-3


@dataclass(frozen=True)
class ColeCole:
    eps_inf: float
    poles: tuple[tuple[float, float, float], ...]  # (delta_eps, tau [s], alpha)
    sigma_ionic: float  # S/m

    def complex_permittivity(self, f_hz):
        w = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
        eps = np.full(w.shape, self.eps_inf, dtype=complex)
        for d_eps, tau, alpha in self.poles:
            eps = eps + d_eps / (1.0 + (1j * w * tau) ** (1.0 - alpha))
        with np.errstate(divide="ignore", invalid="ignore"):
            eps = eps + self.sigma_ionic / (1j * w * EPS0)
        return eps


# Parámetros de la tabla 1 de Gabriel et al. (1996c).
COLE_COLE: dict[str, ColeCole] = {
    "skin": ColeCole(4.0, ((32.0, 7.234 * PS, 0.0), (1100.0, 32.481 * NS, 0.20)), 0.0002),  # piel seca
    "fat": ColeCole(
        2.5,
        ((3.0, 7.958 * PS, 0.20), (15.0, 15.915 * NS, 0.10), (3.3e4, 159.155 * US, 0.05), (1e7, 7.958 * MS, 0.01)),
        0.010,
    ),
    "muscle": ColeCole(
        4.0,
        ((50.0, 7.234 * PS, 0.10), (7000.0, 353.678 * NS, 0.10), (1.2e6, 318.31 * US, 0.10), (2.5e7, 2.274 * MS, 0.0)),
        0.20,
    ),
    "blood": ColeCole(4.0, ((56.0, 8.377 * PS, 0.10), (5200.0, 132.629 * NS, 0.10)), 0.70),
    "cortical_bone": ColeCole(
        2.5,
        ((10.0, 13.263 * PS, 0.20), (180.0, 79.577 * NS, 0.20), (5e3, 159.155 * US, 0.20), (1e5, 15.915 * MS, 0.0)),
        0.020,
    ),
    "trabecular_bone": ColeCole(
        2.5,
        ((18.0, 13.263 * PS, 0.22), (300.0, 79.577 * NS, 0.25), (2e4, 159.155 * US, 0.20), (2e7, 15.915 * MS, 0.0)),
        0.070,
    ),
    "marrow": ColeCole(  # "bone marrow (not infiltrated)"
        2.5,
        ((3.0, 7.958 * PS, 0.20), (25.0, 15.915 * NS, 0.10), (5e3, 1591.549 * US, 0.10), (2e6, 15.915 * MS, 0.0)),
        0.0016,
    ),
    # agua pura a 25 °C (Debye simple); conductividad de agua ligeramente mineralizada
    "water": ColeCole(5.2, ((73.2, 8.27 * PS, 0.0),), 0.005),
    "air": ColeCole(1.0, (), 0.0),
}

# Tejidos que se modelan como mezcla lineal (por volumen) de permitividades.
# Es una aproximación simple; las reglas de mezcla rigurosas (Maxwell-Garnett,
# Bruggeman) dan valores algo distintos.
MIXTURES: dict[str, tuple[tuple[str, float], ...]] = {
    "hematoma": (("blood", 1.0),),
    "edema": (("marrow", 0.6), ("blood", 0.4)),
}

# Fracción de agua en volumen, usada por encima de 100 GHz (aprox., Duck 1990).
WATER_FRACTION: dict[str, float] = {
    "water": 1.0,
    "air": 0.0,
    "skin": 0.65,
    "fat": 0.15,
    "muscle": 0.75,
    "blood": 0.80,
    "cortical_bone": 0.12,
    "trabecular_bone": 0.25,
    "marrow": 0.15,
    "hematoma": 0.78,
    "edema": 0.45,
}

GABRIEL_MAX_FREQ = 100e9


def water_double_debye(f_hz):
    """Permitividad del agua líquida a 25 °C, válida hasta ~2 THz.

    Kindt & Schmuttenmaer, J. Phys. Chem. 100, 10373 (1996).
    """
    w = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    eps_s, eps_2, eps_inf = 78.36, 4.93, 3.48
    tau1, tau2 = 8.24 * PS, 0.18 * PS
    return eps_inf + (eps_s - eps_2) / (1 + 1j * w * tau1) + (eps_2 - eps_inf) / (1 + 1j * w * tau2)


def complex_permittivity(key: str, f_hz):
    """Permitividad relativa compleja ``eps' - j eps''`` del tejido ``key``.

    Por debajo de 100 GHz usa Gabriel 1996. Por encima, la del agua escalada
    por la fracción de agua del tejido (el agua domina la absorción en THz).
    """
    f = np.asarray(f_hz, dtype=float)
    if key in MIXTURES:
        return sum(frac * complex_permittivity(k, f) for k, frac in MIXTURES[key])
    model = COLE_COLE[key]
    eps = model.complex_permittivity(f)
    if np.any(f > GABRIEL_MAX_FREQ):
        wf = WATER_FRACTION[key]
        dry = 2.5 if key != "air" else 1.0
        high = wf * water_double_debye(f) + (1.0 - wf) * dry
        eps = np.where(f > GABRIEL_MAX_FREQ, high, eps)
    return eps


def permittivity_and_conductivity(key: str, f_hz: float) -> tuple[float, float]:
    """``(eps_r', sigma_eff [S/m])`` a la frecuencia dada."""
    eps = complex(complex_permittivity(key, f_hz))
    return eps.real, -eps.imag * 2.0 * np.pi * f_hz * EPS0


def wave_numbers(key: str, f_hz):
    """Constante de fase ``beta`` [rad/m] y atenuación de amplitud ``alpha`` [Np/m]
    de una onda plana en el tejido."""
    f = np.asarray(f_hz, dtype=float)
    k = 2.0 * np.pi * f / C0 * np.sqrt(complex_permittivity(key, f))
    # Con eps = eps' - j eps'', sqrt(eps) = n - j kappa → alpha = k0 * kappa
    return k.real, -k.imag


def penetration_depth(key: str, f_hz):
    """Profundidad a la que la *intensidad* cae a 1/e: ``1 / (2 alpha)`` en metros."""
    _, alpha = wave_numbers(key, f_hz)
    with np.errstate(divide="ignore"):
        return 1.0 / (2.0 * alpha)


def intrinsic_impedance(key: str, f_hz):
    return np.sqrt(MU0 / (EPS0 * complex_permittivity(key, f_hz)))
