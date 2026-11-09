"""Soluciones analíticas exactas para validar el FDTD de Maxwell (modo TM).

- ``multilayer_rt``: reflexión y transmisión complejas de una pila de capas
  planas a incidencia normal (matriz de transferencia), con medios de entrada
  y salida distintos y permitividades arbitrarias por capa (Gabriel, ajuste de
  Debye...). Con el mismo medio a los dos lados reproduce
  ``spectrum._coherent_transmission`` (que sólo da ``|t|^2``).
- ``mie_cylinder_tm``: campo total de una onda plana que incide sobre un
  cilindro dieléctrico circular infinito con el campo eléctrico paralelo a su
  eje (modo TM del paquete: Ey perpendicular al plano), en serie de Bessel y
  Hankel (Bohren y Huffman, *Absorption and Scattering of Light by Small
  Particles*, 1983, §8.4, con el convenio ``e^{+jwt}``).

Convenio de todo el paquete: ``e^{+jwt}``, ``eps = eps' - j eps''``; una onda
que avanza hacia ``+z`` es ``e^{-jkz}`` y la onda saliente es ``H_n^(2)``.
"""

from __future__ import annotations

import numpy as np

from ..materials import dielectric
from . import bessel

C0, EPS0, MU0 = dielectric.C0, dielectric.EPS0, dielectric.MU0
ETA0 = np.sqrt(MU0 / EPS0)


# --------------------------------------------------------------------------
# Capas planas
# --------------------------------------------------------------------------
def multilayer_rt(eps_layers, thicknesses, f, eps_in=1.0, eps_out=1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(r, t)`` de una pila de capas a incidencia normal.

    ``eps_layers[i]`` es la permitividad relativa compleja de la capa ``i``
    (escalar o array con la forma de ``f``) y ``thicknesses[i]`` su grosor
    (m). ``r = E_r / E_i`` en la cara de entrada y ``t = E_t / E_i`` con
    ``E_t`` en la cara de salida y ``E_i`` en la de entrada. Cada capa aporta
    ``[[cos kd, j eta sin kd], [j sin kd / eta, cos kd]]`` sobre ``(E, H)``.
    """
    f = np.asarray(f, dtype=float)
    k0 = 2.0 * np.pi * f / C0
    one = np.ones_like(k0, dtype=complex)
    m00, m01, m10, m11 = one.copy(), 0 * one, 0 * one, one.copy()
    for eps, d in zip(eps_layers, thicknesses):
        n = np.sqrt(np.asarray(eps, dtype=complex) * one)
        kd, eta = k0 * n * d, ETA0 / n
        c, s = np.cos(kd), np.sin(kd)
        a00, a01, a10, a11 = c, 1j * eta * s, 1j * s / eta, c
        m00, m01, m10, m11 = (m00 * a00 + m01 * a10, m00 * a01 + m01 * a11,
                              m10 * a00 + m11 * a10, m10 * a01 + m11 * a11)
    eta_in = ETA0 / np.sqrt(np.asarray(eps_in, dtype=complex) * one)
    eta_out = ETA0 / np.sqrt(np.asarray(eps_out, dtype=complex) * one)
    e1 = m00 + m01 / eta_out  # E en la entrada por unidad de E_t
    h1 = m10 + m11 / eta_out  # H en la entrada por unidad de E_t
    t = 2.0 / (e1 + eta_in * h1)
    r = (e1 - eta_in * h1) / (e1 + eta_in * h1)
    return r, t


def tissue_layers_eps(layers: list[tuple[str, float]], f, permittivity=None) -> list[np.ndarray]:
    """Permitividades de las capas ``(tejido, grosor)``; por defecto Gabriel (``dielectric``)."""
    get = permittivity or (lambda k, ff: dielectric.complex_permittivity(k, ff))
    return [get(k, f) for k, _ in layers]


def insertion_transmission(layers: list[tuple[str, float]], f, outer: str = "air", permittivity=None) -> np.ndarray:
    """Cociente del campo transmitido con la pila y sin ella, en un mismo punto detrás de la pila.

    Es lo que mide el FDTD comparando con una simulación de referencia sin
    capas (todo ``outer``): ``t e^{j k_out D}``, con ``D`` el grosor total.
    """
    f = np.asarray(f, dtype=float)
    get = permittivity or (lambda k, ff: dielectric.complex_permittivity(k, ff))
    eps_out = get(outer, f)
    _, t = multilayer_rt(tissue_layers_eps(layers, f, get), [d for _, d in layers], f, eps_out, eps_out)
    k_out = 2.0 * np.pi * f / C0 * np.sqrt(np.asarray(eps_out, dtype=complex))
    return t * np.exp(1j * k_out * sum(d for _, d in layers))


# --------------------------------------------------------------------------
# Cilindro (Mie 2D, modo TM)
# --------------------------------------------------------------------------
def mie_coefficients_tm(k_out: float, k_in: complex, radius: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Coeficientes ``(a_n, c_n)``, ``n = 0..n_max``, del campo dispersado y del interior.

    Continuidad de ``E_y`` y de ``H_phi ∝ dE_y/drho`` (mismo ``mu0``) en ``rho = a``:

        J_n(x) + a_n H_n(x) = c_n J_n(m x)
        k J_n'(x) + a_n k H_n'(x) = c_n k_in J_n'(m x)

    con ``x = k_out a`` (real: medio exterior sin pérdidas). ``c_n`` sale del
    wronskiano ``J_n H_n' - J_n' H_n = -2j / (pi x)`` para no dividir por
    ``J_n(m x)``.
    """
    x = k_out * radius
    xi = k_in * radius
    jo = bessel.besselj_orders(n_max + 1, np.array([x]))[:, 0]
    ho = bessel.hankel2_orders(n_max + 1, np.array([x]))[:, 0]
    ji = bessel.besselj_orders(n_max + 1, np.array([xi]))[:, 0]
    djo, dho, dji = (bessel.derivative_orders(v) for v in (jo, ho, ji))
    jo, ho, ji = jo[:-1], ho[:-1], ji[:-1]
    den = k_out * dho * ji - k_in * dji * ho
    a = (k_in * dji * jo - k_out * djo * ji) / den
    c = k_out * (-2j / (np.pi * x)) / den
    return a, c


def mie_cylinder_tm(x, z, radius: float, eps_in: complex, f: float, eps_out: float = 1.0,
                    center: tuple[float, float] = (0.0, 0.0), n_max: int | None = None) -> np.ndarray:
    """Campo total ``E_y`` de una onda plana ``e^{-j k (z - z_c)}`` (amplitud 1 en el centro) sobre el cilindro.

    ``eps_in``: permitividad relativa compleja del cilindro; ``eps_out``, la del
    medio exterior (real, sin pérdidas). ``x`` y ``z`` son arrays de posiciones.
    """
    if np.iscomplexobj(eps_out) and np.imag(eps_out) != 0:
        raise ValueError("el medio exterior debe ser sin pérdidas (Y_n sólo con argumento real)")
    k0 = 2.0 * np.pi * f / C0
    k_out = k0 * np.sqrt(float(np.real(eps_out)))
    k_in = k0 * np.sqrt(complex(eps_in))
    x = np.asarray(x, dtype=float) - center[0]
    z = np.asarray(z, dtype=float) - center[1]
    rho = np.hypot(x, z)
    phi = np.arctan2(x, z)  # ángulo desde el eje +z (dirección de incidencia)
    if n_max is None:
        size = abs(k_in) * radius
        n_max = int(np.ceil(size + 4.0 * size ** (1.0 / 3.0) + 15))
    a, c = mie_coefficients_tm(k_out, k_in, radius, n_max)
    n = np.arange(n_max + 1)
    weight = np.where(n == 0, 1.0, 2.0) * (-1j) ** n  # términos n y -n juntos: 2 (-j)^n cos(n phi)
    out = np.empty(rho.shape, dtype=complex)
    inside = rho <= radius
    if inside.any():
        jn = bessel.besselj_orders(n_max, k_in * rho[inside])
        cos_n = np.cos(n[:, None] * phi[inside][None, :])
        out[inside] = np.sum((weight * c)[:, None] * jn * cos_n, axis=0)
    outside = ~inside
    if outside.any():
        hn = bessel.hankel2_orders(n_max, k_out * rho[outside])
        cos_n = np.cos(n[:, None] * phi[outside][None, :])
        scat = np.sum((weight * a)[:, None] * hn * cos_n, axis=0)
        out[outside] = np.exp(-1j * k_out * z[outside]) + scat
    return out
