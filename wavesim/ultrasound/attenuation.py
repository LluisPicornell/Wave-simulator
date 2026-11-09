"""Atenuación viscoelástica en el dominio del tiempo (sólido lineal estándar generalizado).

Los tejidos atenúan el ultrasonido según una ley de potencia
``alpha(f) = alpha0 · f^y``: ``y ≈ 1`` (factor de calidad Q casi constante) en
los tejidos blandos y el hueso, ``y ≈ 1.2`` en la sangre y ``y = 2`` en el
agua. En FDTD se modela con L mecanismos de relajación (GSLS):

    M(w) = M_R [1 + sum_l  y_l · i w tau_l / (1 + i w tau_l)]

- ``tau_l``: tiempos de relajación, comunes a toda la malla (así las memorias
  se actualizan con coeficientes escalares por mecanismo).
- ``y_l``: peso (intensidad de relajación) de cada mecanismo, propio de cada
  tejido o celda. Con pesos no negativos el material es pasivo (disipa
  energía) y el esquema es estable con el paso calculado con el módulo no
  relajado ``M_U = M_R (1 + sum_l y_l)``.

El módulo relajado ``M_R`` se corrige para que la velocidad de fase a la
frecuencia central ``f0`` sea exactamente la de la tabla. La variación de la
velocidad con la frecuencia es la que impone la causalidad del modelo; en la
banda ajustada coincide con la de Kramers-Kronig para la ley de potencia
(``power_law_phase_speed``).

Ajuste de los pesos (``mechanism_weights``):

- **1 mecanismo**: "método tau" de Blanch, Robertsson & Symes (Geophysics 60,
  176, 1995) con ``tau_1 = 1/(2π f0)`` y el peso ajustado por mínimos
  cuadrados de Q⁻¹ en [0.5, 1.8]·f0. Un solo pico de Debye no puede seguir la
  ley de potencia en esa banda (error máximo del 14 % con y = 1 y del ~95 %
  con y = 2). Se deja como estaba, y los resultados por defecto no cambian:
  en la cortical un ajuste minimax baja el error máximo del 13.8 al 11.2 %,
  pero sube el de f0 del +7 al +11 % y el error cuadrático ponderado por el
  espectro de un pulso del 60 % de ancho de banda del 6 al 9 %.
- **2 o más**: un peso por mecanismo, ajuste minimax del error relativo en
  ``alpha(f)`` con pesos no negativos y con el Q⁻¹ objetivo calculado con la
  velocidad de fase del propio modelo (no con la de la tabla), en la banda de
  ``MECHANISM_DESIGN``. Los tiempos de relajación de esa tabla se optimizaron
  para que el error sea pequeño a la vez para cualquier exponente entre 1 y 2
  sin encarecer mucho el paso de tiempo (ver ``MECHANISM_DESIGN``).
"""

from __future__ import annotations

import itertools

import numpy as np

#: Diseño de los mecanismos para cada L: frecuencias de relajación
#: ``1/(2π tau_l)`` y banda de ajuste, ambas en unidades de f0.
#:
#: Las frecuencias (salvo L = 1) se obtuvieron minimizando el peor error
#: relativo en alpha(f) para y = 1.0 (Q alto y bajo), 1.1, 1.2, 1.5 y 2.0 a la
#: vez, más una penalización por el aumento del módulo no relajado del hueso
#: cortical (que fija el paso de tiempo): ``benchmarks/attenuation_power_law.py
#: --disenar`` lo repite. Peor error del modelo entre esos exponentes en
#: [0.5, 1.8]·f0 / [0.3, 2.5]·f0 (``docs/validacion/atenuacion.md``):
#:
#: - L = 2: 1.9 % / 20 %. Un mecanismo muy por encima de la banda es necesario
#:   para que y = 2 (agua) siga la ley; a cambio, el paso de tiempo es un 7 %
#:   menor que con L = 1 en un fantoma con cortical.
#: - L = 3: 1.7 % / 1.7 %, con un paso un 2 % menor que con L = 1.
#: - L = 4: 0.4 % / 0.4 %.
#:
#: Con y = 2 y pérdidas muy altas (Q⁻¹(f0) ≳ 0.05, que no tiene ningún tejido
#: de la tabla) el error sube al 3-4 %.
MECHANISM_DESIGN: dict[int, tuple[tuple[float, ...], tuple[float, float]]] = {
    1: ((1.0,), (0.5, 1.8)),
    2: ((0.543, 9.17), (0.5, 1.8)),
    3: ((0.311, 2.66, 20.0), (0.3, 2.5)),
    4: ((0.180, 0.885, 4.61, 42.0), (0.25, 3.0)),
}

#: Iteraciones del ajuste: recálculo de Q⁻¹ con la velocidad del modelo y
#: repesado de Lawson hacia el minimax (converge en ~10-20).
_DISPERSION_ITERS = 3
_MINIMAX_ITERS = 20


def relaxation_times(f0: float, n_mechanisms: int) -> np.ndarray:
    """Tiempos de relajación (s) de ``n_mechanisms`` mecanismos para la frecuencia central ``f0``."""
    if n_mechanisms <= 0:
        return np.zeros(0)
    if n_mechanisms not in MECHANISM_DESIGN:
        raise ValueError(f"n_mechanisms debe estar entre 0 y {max(MECHANISM_DESIGN)} (con 4 el error ya es < 0.5 %)")
    freqs = f0 * np.asarray(MECHANISM_DESIGN[n_mechanisms][0])
    return 1.0 / (2.0 * np.pi * freqs)


def fit_band(f0: float, n_mechanisms: int = 1) -> np.ndarray:
    """Frecuencias en las que se ajustan los pesos (banda útil del pulso, más ancha con más mecanismos)."""
    if n_mechanisms <= 1:
        return np.linspace(0.5 * f0, 1.8 * f0, 24)
    lo, hi = MECHANISM_DESIGN[n_mechanisms][1]
    return f0 * np.geomspace(lo, hi, 32)


def target_qinv(alpha_np_m: np.ndarray, c, f: np.ndarray) -> np.ndarray:
    """Q⁻¹ exacto correspondiente a una atenuación ``alpha`` y velocidad de fase ``c`` (escalar o por frecuencia)."""
    a = alpha_np_m * c / (2.0 * np.pi * f)
    return 2.0 * a / (1.0 - a**2)


def _per_mechanism(weights, n: int) -> list:
    """Pesos de cada mecanismo: un escalar vale para todos (método tau); un array lleva el mecanismo en el eje 0."""
    w = np.asarray(weights, dtype=float)
    if w.ndim == 0:
        return [w] * n
    if w.shape[0] != n:
        raise ValueError(f"se esperaban {n} pesos en el eje 0, no {w.shape[0]}")
    return list(w)


def modulus_ratio(weights, tau_sigma: np.ndarray, f):
    """``M(w) / M_R`` (complejo, convención e^{iwt}).

    ``weights``: escalar (el mismo peso en todos los mecanismos) o array con
    el mecanismo en el eje 0; el resto de sus ejes se combina con ``f`` por
    *broadcasting*.
    """
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    per = _per_mechanism(weights, tau_sigma.size)
    r = np.ones(np.broadcast(w, *per).shape if per else w.shape, dtype=complex)
    for wl, ts in zip(per, tau_sigma):
        r = r + wl * (1j * w * ts) / (1 + 1j * w * ts)
    return r


def model_qinv(weights, tau_sigma: np.ndarray, f):
    """Q⁻¹ = Im M / Re M del modelo."""
    r = modulus_ratio(weights, tau_sigma, f)
    return r.imag / r.real


def relaxed_modulus_factor(weights, tau_sigma: np.ndarray, f0: float):
    """Factor que multiplica ``rho c0²`` para obtener ``M_R`` con velocidad de fase c0 en f0."""
    r0 = modulus_ratio(weights, tau_sigma, f0)
    return np.real(r0 ** -0.5) ** 2


def model_wave(c0, weights, tau_sigma: np.ndarray, f0: float, f):
    """Velocidad de fase y atenuación (Np/m) que produce el modelo a la frecuencia ``f``."""
    k_factor = modulus_ratio(weights, tau_sigma, f) ** -0.5 / np.real(modulus_ratio(weights, tau_sigma, f0) ** -0.5)
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    k = w / c0 * k_factor
    return w / np.real(k), -np.imag(k)


def fit_tau(qinv: np.ndarray, tau_sigma: np.ndarray, f: np.ndarray) -> float:
    """Método tau: un único peso para todos los mecanismos, por mínimos cuadrados de Q⁻¹ en ``f``.

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


def _nnls_small(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mínimos cuadrados no negativos de muchos problemas pequeños a la vez.

    ``a``: (n, m, L), ``b``: (n, m). Con L ≤ 4 la solución exacta se obtiene
    probando todos los conjuntos de pesos activos (la de menor residuo entre
    las que no tienen pesos negativos), sin el algoritmo iterativo de
    Lawson-Hanson.
    """
    n, _, L = a.shape
    best_res = np.full(n, np.inf)
    best = np.zeros((n, L))
    for k in range(1, L + 1):
        for sub in itertools.combinations(range(L), k):
            s = a[:, :, sub]
            m = np.einsum("nfi,nfj->nij", s, s) + 1e-14 * np.eye(k)
            y = np.linalg.solve(m, np.einsum("nfi,nf->ni", s, b)[..., None])[..., 0]
            res = np.linalg.norm(np.einsum("nfi,ni->nf", s, y) - b, axis=1)
            ok = np.all(y >= 0.0, axis=1) & (res < best_res)
            best_res[ok] = res[ok]
            full = np.zeros((n, L))
            full[:, sub] = y
            best[ok] = full[ok]
    return best


def _minimax_weights(alpha: np.ndarray, c: np.ndarray, f: np.ndarray, tau_sigma: np.ndarray, f0: float) -> np.ndarray:
    """Pesos (n, L) que minimizan el mayor error relativo en alpha (filas con alpha > 0)."""
    wt = 2.0 * np.pi * f[:, None] * tau_sigma[None, :]
    g, h = wt / (1 + wt**2), wt**2 / (1 + wt**2)
    speed = np.repeat(c[:, None], f.size, axis=1)
    lawson = np.ones(alpha.shape)
    ones = np.ones(alpha.shape)
    y = np.zeros((alpha.shape[0], tau_sigma.size))
    for it in range(_DISPERSION_ITERS + _MINIMAX_ITERS):
        q = target_qinv(alpha, speed, f[None, :])
        # Q⁻¹(1 + Σ y h) = Σ y g es lineal en los pesos; al dividir por Q⁻¹ el
        # residuo es el error relativo de Q⁻¹ (≈ el de alpha)
        a = (g[None] - q[:, :, None] * h[None]) / q[:, :, None]
        sw = np.sqrt(lawson)
        y = _nnls_small(a * sw[:, :, None], ones * sw)
        speed, model = model_wave(c[:, None], y.T[..., None], tau_sigma, f0, f[None, :])
        if it >= _DISPERSION_ITERS:
            # Lawson: se refuerzan las frecuencias con más error hasta igualarlo
            lawson = lawson * (np.abs(model / alpha - 1.0) + 1e-12)
            lawson /= lawson.mean(axis=1, keepdims=True)
    return y


def mechanism_weights(alpha: np.ndarray, c, f: np.ndarray, tau_sigma: np.ndarray, f0: float) -> np.ndarray:
    """Pesos ``y_l`` (n, L) que reproducen la atenuación ``alpha`` (n, len(f)) en Np/m con velocidad ``c`` (n,).

    Filas con ``c <= 0`` o sin atenuación dan pesos nulos.
    """
    alpha = np.atleast_2d(np.asarray(alpha, dtype=float))
    c = np.broadcast_to(np.asarray(c, dtype=float), alpha.shape[:1])
    f = np.asarray(f, dtype=float)
    out = np.zeros((alpha.shape[0], tau_sigma.size))
    rows = (c > 0) & np.all(alpha > 0, axis=1)
    if tau_sigma.size == 0 or not rows.any():
        return out
    if tau_sigma.size == 1:
        out[rows, 0] = [fit_tau(target_qinv(a, ck, f), tau_sigma, f) for a, ck in zip(alpha[rows], c[rows])]
    else:
        out[rows] = _minimax_weights(alpha[rows], c[rows], f, tau_sigma, f0)
    return out


def power_law_phase_speed(c0: float, alpha_ref: float, y: float, f0: float, f, f_ref: float = 1e6):
    """Velocidad de fase causal de la ley ``alpha = alpha_ref (f/f_ref)^y`` (Np/m), con ``c(f0) = c0``.

    Relaciones de Kramers-Kronig aproximadas para la ley de potencia (Szabo,
    JASA 96:491, 1994; Waters et al., JASA 118:3349, 2005), con
    ``a0 = alpha_ref / (2π f_ref)^y`` en Np/m/(rad/s)^y:

    - ``y ≠ 1``: ``1/c(w) = 1/c0 + a0 tan(πy/2) (w^(y-1) - w0^(y-1))``
    - ``y = 1``: ``1/c(w) = 1/c0 - (2/π) a0 ln(w/w0)``
    """
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    w0 = 2.0 * np.pi * f0
    a0 = alpha_ref / (2.0 * np.pi * f_ref) ** y
    if abs(y - 1.0) < 1e-9:
        inv = 1.0 / c0 - (2.0 / np.pi) * a0 * np.log(w / w0)
    else:
        inv = 1.0 / c0 + a0 * np.tan(np.pi * y / 2.0) * (w ** (y - 1.0) - w0 ** (y - 1.0))
    return 1.0 / inv
