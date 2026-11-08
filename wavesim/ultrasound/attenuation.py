"""Atenuación viscoelástica en el dominio del tiempo (sólido lineal estándar generalizado).

Los tejidos atenúan el ultrasonido casi en proporción a la frecuencia
(``alpha ∝ f^y`` con ``y ≈ 1``), es decir, con un factor de calidad Q casi
constante. En FDTD esto se modela con L mecanismos de relajación (GSLS) y el
"método tau" de Blanch, Robertsson & Symes (Geophysics 60, 176, 1995):

    M(w) = M_R [1 + sum_l  tau · i w tau_l / (1 + i w tau_l)]

``tau_l`` (tiempos de relajación) se reparten en la banda del pulso y ``tau``
se ajusta por mínimos cuadrados para reproducir la Q objetivo de cada tejido.
El módulo relajado ``M_R`` se corrige para que la velocidad de fase a la
frecuencia central ``f0`` sea exactamente la de la tabla (dispersión causal).
"""

from __future__ import annotations

import numpy as np


def relaxation_times(f0: float, n_mechanisms: int) -> np.ndarray:
    """Tiempos de relajación repartidos logarítmicamente alrededor de ``f0``."""
    if n_mechanisms <= 0:
        return np.zeros(0)
    if n_mechanisms == 1:
        freqs = np.array([f0])
    else:
        freqs = np.geomspace(f0 / 2.5, f0 * 2.5, n_mechanisms)
    return 1.0 / (2.0 * np.pi * freqs)


def fit_band(f0: float) -> np.ndarray:
    """Frecuencias en las que se ajusta Q (banda útil de un pulso típico)."""
    return np.linspace(0.5 * f0, 1.8 * f0, 24)


def target_qinv(alpha_np_m: np.ndarray, c: float, f: np.ndarray) -> np.ndarray:
    """Q⁻¹ exacto correspondiente a una atenuación ``alpha`` y velocidad ``c``."""
    a = alpha_np_m * c / (2.0 * np.pi * f)
    return 2.0 * a / (1.0 - a**2)


def model_qinv(tau: float, tau_sigma: np.ndarray, f: np.ndarray) -> np.ndarray:
    w = 2.0 * np.pi * np.asarray(f)[:, None]
    wt = w * tau_sigma[None, :]
    g = np.sum(wt / (1 + wt**2), axis=1)
    h = np.sum(wt**2 / (1 + wt**2), axis=1)
    return tau * g / (1.0 + tau * h)


def fit_tau(qinv: np.ndarray, tau_sigma: np.ndarray, f: np.ndarray) -> float:
    """Ajusta ``tau`` por mínimos cuadrados a un Q⁻¹ objetivo en las frecuencias ``f``.

    Q⁻¹ = tau·g / (1 + tau·h)  ⇔  tau·(g - Q⁻¹·h) = Q⁻¹  (lineal en tau).
    """
    if tau_sigma.size == 0 or np.all(qinv <= 0):
        return 0.0
    w = 2.0 * np.pi * np.asarray(f)[:, None]
    wt = w * tau_sigma[None, :]
    g = np.sum(wt / (1 + wt**2), axis=1)
    h = np.sum(wt**2 / (1 + wt**2), axis=1)
    a = g - qinv * h
    return float(max(np.sum(a * qinv) / np.sum(a * a), 0.0))


def modulus_ratio(tau, tau_sigma: np.ndarray, f):
    """``M(w) / M_R`` (complejo, convención e^{iwt})."""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    r = np.ones(np.broadcast(w, np.asarray(tau)).shape, dtype=complex)
    for ts in tau_sigma:
        r = r + tau * (1j * w * ts) / (1 + 1j * w * ts)
    return r


def relaxed_modulus_factor(tau, tau_sigma: np.ndarray, f0: float):
    """Factor que multiplica ``rho c0²`` para obtener ``M_R`` con velocidad de fase c0 en f0."""
    r0 = modulus_ratio(tau, tau_sigma, f0)
    return np.real(r0 ** -0.5) ** 2


def model_wave(c0: float, tau: float, tau_sigma: np.ndarray, f0: float, f):
    """Velocidad de fase y atenuación (Np/m) que produce el modelo a la frecuencia ``f``."""
    k_factor = modulus_ratio(tau, tau_sigma, f) ** -0.5 / np.real(modulus_ratio(tau, tau_sigma, f0) ** -0.5)
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    k = w / c0 * k_factor
    return w / np.real(k), -np.imag(k)
