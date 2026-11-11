"""Soluciones analíticas de referencia para validar el solver elástico.

Sin SciPy: todo son fórmulas cerradas o raíces halladas con NumPy.

- ``discrete_wavenumber``: relación de dispersión del propio esquema (4.º orden
  en espacio, leapfrog de 2.º orden en tiempo, malla escalonada) para una onda
  plana alineada con un eje de la malla. Permite separar el error del esquema
  (dispersión numérica, que se puede predecir) de un error de implementación.
- ``fluid_solid_coefficients``: reflexión y transmisión de una onda P plana
  que incide desde un fluido sobre un semiespacio sólido elástico (Zoeppritz
  para fluido-sólido; forma cerrada de Brekhovskikh en
  ``fluid_solid_reflection_closed``).
- ``lamb_wavenumbers`` y ``lamb_fluid_loaded``: ondas de Lamb (simétricas y
  antisimétricas) en una placa libre (Rayleigh-Lamb) y en una placa con el
  mismo fluido a ambos lados (raíces complejas: modos con fuga).
- ``rayleigh_velocity``: onda de Rayleigh en la superficie libre de un sólido.

Convenciones: dependencia temporal ``exp(-i w t)``, ``x`` a lo largo de la
interfaz o de la placa y ``z`` normal a ella (hacia el sólido en el caso
fluido-sólido). Potenciales: ``u = grad(phi) + rot(psi y)`` con
``u_x = dphi/dx - dpsi/dz`` y ``u_z = dphi/dz + dpsi/dx``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Coeficientes del operador escalonado de 4.º orden (los mismos que ``elastic_fdtd``).
C1, C2 = 9.0 / 8.0, -1.0 / 24.0


# ----------------------------------------------------------------------------
# Dispersión numérica del esquema
# ----------------------------------------------------------------------------
def discrete_wavenumber(omega, c: float, dx: float, dt: float):
    """Número de onda ``k`` que el esquema FDTD asocia a ``omega`` a lo largo de un eje.

    Relación de dispersión del leapfrog escalonado con diferencias de 4.º orden:
    ``sin(w dt/2) / (c dt/2) = (2/dx) [C1 sin(k dx/2) + C2 sin(3 k dx/2)]``.
    Se resuelve por Newton desde ``k = w/c`` (la rama física). Para ``w dt`` y
    ``k dx`` pequeños, ``c_num/c - 1 ≈ (w dt)²/24 - 3 (k dx)⁴/640``: el
    término temporal (2.º orden) acelera la onda y el espacial (4.º) la frena.
    """
    omega = np.asarray(omega, dtype=float)
    rhs = np.sin(omega * dt / 2.0) * dx / (c * dt)
    return _invert_stencil(rhs, omega / c, dx)


def _stencil(k, dx):
    """Símbolo del operador escalonado de 4.º orden por ``dx/2``: ``C1 sin(k dx/2) + C2 sin(3 k dx/2)``."""
    return C1 * np.sin(k * dx / 2.0) + C2 * np.sin(1.5 * k * dx)


def _invert_stencil(rhs, k0, dx):
    """``k`` con ``_stencil(k, dx) = rhs`` por Newton desde ``k0`` (complejo si ``rhs`` lo es)."""
    k = np.asarray(k0) + 0 * np.asarray(rhs)
    for _ in range(60):
        th = k * dx / 2.0
        df = (C1 * np.cos(th) + 3.0 * C2 * np.cos(3.0 * th)) * dx / 2.0
        k = k - (_stencil(k, dx) - rhs) / df
    return k


def discrete_kz(omega, kx, c: float, dx: float, dt: float):
    """``kz`` de una onda plana del esquema con número de onda horizontal ``kx`` (exacto en la malla).

    Relación de dispersión 2D: ``[sin(w dt/2)/(c dt/2)]² = Kx² + Kz²`` con
    ``K(k) = (2/dx) [C1 sin(k dx/2) + C2 sin(3k dx/2)]``. Complejo (Im > 0)
    para ondas evanescentes.
    """
    omega, kx = np.asarray(omega, dtype=float), np.asarray(kx, dtype=float)
    big_k = 2.0 * np.sin(omega * dt / 2.0) / (c * dt)
    kx_s = 2.0 / dx * _stencil(kx, dx)
    kz_s = np.sqrt(np.asarray(big_k**2 - kx_s**2, dtype=complex))
    kz0 = np.sqrt(np.asarray((omega / c) ** 2 - kx**2, dtype=complex))
    kz = _invert_stencil(kz_s * dx / 2.0, kz0, dx)
    return np.where(np.imag(kz) < 0, -kz, kz)


# ----------------------------------------------------------------------------
# Interfaz fluido-sólido
# ----------------------------------------------------------------------------
def _kz(k, kx):
    """Componente normal ``sqrt(k² - kx²)`` con Im >= 0 (onda que se aleja o decae)."""
    kz = np.sqrt(np.asarray(k, dtype=complex) ** 2 - np.asarray(kx, dtype=complex) ** 2)
    return np.where(np.imag(kz) < 0, -kz, kz)


@dataclass
class FluidSolidCoefficients:
    """Coeficientes de una onda P plana desde un fluido sobre un sólido, por unidad de onda incidente.

    - ``R``: reflexión de presión (= de potencial: ``p = rho1 w² phi``).
    - ``Tp``, ``Ts``: amplitudes de potencial transmitidas (P y S).
    - ``vp``, ``vs``: amplitud de velocidad de partícula de la P y la S
      transmitidas dividida por la de la onda incidente (``|v| = w k |pot|``).
      Sólo tienen sentido como amplitud de onda plana donde esa onda se propaga.
    - ``energy_p``, ``energy_s``: fracción del flujo de energía normal
      incidente que se llevan la P y la S (``|R|² + energy_p + energy_s = 1``).
    - ``theta_p``, ``theta_s``: ángulos de refracción (complejos pasado el crítico).
    """

    R: np.ndarray
    Tp: np.ndarray
    Ts: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    energy_p: np.ndarray
    energy_s: np.ndarray
    theta_p: np.ndarray
    theta_s: np.ndarray


def fluid_solid_coefficients(theta, rho1: float, c1: float, rho2: float, cp: float, cs: float,
                             omega: float = 1.0) -> FluidSolidCoefficients:
    """Resuelve las condiciones de contorno en ``z = 0`` para cada ángulo de incidencia ``theta`` (rad).

    Condiciones: continuidad de ``u_z``, ``sigma_zz = -p`` y ``sigma_xz = 0``
    (el fluido no transmite cizalla). Es un sistema lineal 3x3 en
    ``(R, Tp, Ts)``; resolverlo así evita depender de una convención de signos
    copiada de la literatura, y ``fluid_solid_reflection_closed`` lo contrasta
    con la forma cerrada. Los coeficientes no dependen de ``omega`` (no hay
    escala de longitud), que sólo fija las unidades intermedias.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    k1, kp, ks = omega / c1, omega / cp, omega / cs
    mu = rho2 * cs**2
    kx = k1 * np.sin(theta)
    kz1 = k1 * np.cos(theta)
    kzp, kzs = _kz(kp, kx), _kz(ks, kx)
    n = theta.size
    M = np.zeros((n, 3, 3), dtype=complex)
    b = np.zeros((n, 3), dtype=complex)
    # u_z: i kz1 (1 - R) = i kzp Tp + i kx Ts
    M[:, 0, 0], M[:, 0, 1], M[:, 0, 2] = kz1, kzp, kx
    b[:, 0] = kz1
    # sigma_zz = -p:  -mu (ks² - 2kx²) Tp - 2 mu kx kzs Ts = -rho1 w² (1 + R)
    M[:, 1, 0] = -rho1 * omega**2
    M[:, 1, 1] = mu * (ks**2 - 2 * kx**2)
    M[:, 1, 2] = 2 * mu * kx * kzs
    b[:, 1] = rho1 * omega**2
    # sigma_xz = 0:  -2 kx kzp Tp + (kzs² - kx²) Ts = 0
    M[:, 2, 1] = -2 * kx * kzp
    M[:, 2, 2] = kzs**2 - kx**2
    R, Tp, Ts = np.moveaxis(np.linalg.solve(M, b[..., None])[..., 0], -1, 0)
    vp = np.abs(Tp) * kp / k1
    vs = np.abs(Ts) * ks / k1
    flux_in = rho1 * kz1
    energy_p = rho2 * np.real(kzp) * np.abs(Tp) ** 2 / flux_in
    energy_s = rho2 * np.real(kzs) * np.abs(Ts) ** 2 / flux_in
    theta_p = np.arcsin(np.asarray(np.sin(theta) * cp / c1, dtype=complex))
    theta_s = np.arcsin(np.asarray(np.sin(theta) * cs / c1, dtype=complex))
    return FluidSolidCoefficients(R, Tp, Ts, vp, vs, energy_p, energy_s, theta_p, theta_s)


def fluid_solid_reflection_closed(theta, rho1: float, c1: float, rho2: float, cp: float, cs: float):
    """Forma cerrada de Brekhovskikh: ``R = (Zp cos²2θs + Zs sin²2θs - Z1) / (... + Z1)``.

    ``Z1 = rho1 c1 / cos θ``, ``Zp = rho2 cp / cos θp``, ``Zs = rho2 cs / cos θs``.
    Pasado un ángulo crítico el coseno correspondiente es imaginario puro; el
    signo se elige para que la onda transmitida decaiga hacia el sólido
    (``cos θ = kz / k`` con Im(kz) >= 0, igual que en ``fluid_solid_coefficients``).
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    s = np.sin(theta)
    cos_p = _kz(1.0 / cp, s / c1) * cp
    cos_s = _kz(1.0 / cs, s / c1) * cs
    sin_s = s * cs / c1
    cos2s = 1 - 2 * sin_s**2
    sin2s = 2 * sin_s * cos_s
    z1 = rho1 * c1 / np.cos(theta)
    zp = rho2 * cp / cos_p
    zs = rho2 * cs / cos_s
    ztot = zp * cos2s**2 + zs * sin2s**2
    return (ztot - z1) / (ztot + z1)


def critical_angles(c1: float, cp: float, cs: float) -> tuple[float, float]:
    """Ángulos críticos P y S (rad) de una incidencia desde un fluido de velocidad ``c1``."""
    return float(np.arcsin(c1 / cp)), float(np.arcsin(c1 / cs)) if cs > c1 else float("nan")


# ----------------------------------------------------------------------------
# Onda de Rayleigh
# ----------------------------------------------------------------------------
def rayleigh_velocity(cp: float, cs: float) -> float:
    """Raíz real de ``(2 - x)² = 4 sqrt(1 - x cs²/cp²) sqrt(1 - x)`` con ``x = (c/cs)²`` (bisección)."""
    r = (cs / cp) ** 2

    def f(x):
        return (2 - x) ** 2 - 4 * np.sqrt(1 - x * r) * np.sqrt(1 - x)

    lo, hi = 0.5, 1.0 - 1e-15  # c_R está entre ~0.87 cs y cs para todo coeficiente de Poisson >= 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float(cs * np.sqrt(0.5 * (lo + hi)))


# ----------------------------------------------------------------------------
# Ondas de Lamb
# ----------------------------------------------------------------------------
def _sinc(x):
    """sin(x)/x para argumentos complejos (vale 1 en 0)."""
    x = np.asarray(x, dtype=complex)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def lamb_function(k, omega: float, thickness: float, cp: float, cs: float, family: str,
                  rho: float = 1.0, rho_f: float = 0.0, c_f: float = 1500.0):
    """Función característica de Rayleigh-Lamb (con fluido si ``rho_f > 0``): sus raíces en ``k`` son los modos.

    Placa de espesor ``thickness = 2h`` con el mismo fluido a ambos lados. Se
    divide por ``q`` (simétricos) o por ``p`` (antisimétricos) para que, sin
    fluido, sea real tanto si ``p``, ``q`` son reales como imaginarios:

    - simétricos (S): ``(q²-k²)² cos(ph) sin(qh)/q + 4k² p sin(ph) cos(qh)
      - i (rho_f/rho) ks⁴ (p/q) sin(ph) sin(qh) / kz_f``
    - antisimétricos (A): ``(q²-k²)² sin(ph)/p cos(qh) + 4k² q cos(ph) sin(qh)
      + i (rho_f/rho) ks⁴ cos(ph) cos(qh) / kz_f``

    con ``p² = w²/cp² - k²``, ``q² = w²/cs² - k²`` y ``kz_f`` el número de onda
    normal en el fluido. Sin fluido son las ecuaciones de Rayleigh-Lamb de los
    textos: ``tan(qh)/tan(ph) = -4k²pq/(q²-k²)²`` y su inversa.
    """
    h = thickness / 2
    k = np.asarray(k, dtype=complex)
    p = np.sqrt((omega / cp) ** 2 - k**2)
    q = np.sqrt((omega / cs) ** 2 - k**2)
    a, b = (q**2 - k**2) ** 2, 4 * k**2
    ks4 = (omega / cs) ** 4
    if family == "S":
        val = a * np.cos(p * h) * h * _sinc(q * h) + b * p * np.sin(p * h) * np.cos(q * h)
        if rho_f > 0:
            val = val - 1j * (rho_f / rho) * ks4 * p * np.sin(p * h) * h * _sinc(q * h) / _kz_fluid(k, omega / c_f)
    elif family == "A":
        val = a * h * _sinc(p * h) * np.cos(q * h) + b * q * np.cos(p * h) * np.sin(q * h)
        if rho_f > 0:
            val = val + 1j * (rho_f / rho) * ks4 * np.cos(p * h) * np.cos(q * h) / _kz_fluid(k, omega / c_f)
    else:
        raise ValueError("family debe ser 'S' o 'A'")
    return val


def _kz_fluid(k, kf):
    """``kz`` en el fluido: real positivo (radia, modo con fuga) si Re(k) < kf; +i·real (decae) si no."""
    k = np.asarray(k, dtype=complex)
    leaky = np.sqrt(kf**2 - k**2)
    bound = 1j * np.sqrt(k**2 - kf**2)
    return np.where(np.real(k) < kf, leaky, bound)


def lamb_wavenumbers(f: float, thickness: float, cp: float, cs: float, family: str, n_grid: int = 4000) -> np.ndarray:
    """Números de onda reales (rad/m, de mayor a menor: el primero es el modo 0) de la placa libre a ``f``.

    Barrido de ``k`` buscando cambios de signo de ``lamb_function`` (que es
    real y no tiene polos en esta forma) y bisección. A0 es siempre más lento
    que la onda de Rayleigh y que la flexión de Kirchhoff, que son cotas
    inferiores de su ``k``: el barrido llega a 2.5 veces la mayor de ellas.
    """
    w = 2 * np.pi * f
    c_plate2 = 4 * cs**2 * (1 - cs**2 / cp**2)  # = E / (rho (1 - nu²)), velocidad de placa al cuadrado
    k_flex = (12 * w**2 / (c_plate2 * thickness**2)) ** 0.25
    k_max = 2.5 * max(w / rayleigh_velocity(cp, cs), k_flex)
    ks = np.linspace(k_max / n_grid, k_max, n_grid)

    def g(k):
        return np.real(lamb_function(k, w, thickness, cp, cs, family))

    vals = g(ks)
    roots = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        lo, hi = ks[i], ks[i + 1]
        flo = vals[i]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = g(mid)
            if np.sign(fm) == np.sign(flo):
                lo, flo = mid, fm
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    return np.array(sorted(roots, reverse=True))


def lamb_fluid_loaded(f: float, thickness: float, cp: float, cs: float, rho: float, rho_f: float, c_f: float,
                      family: str, k0: complex) -> complex:
    """Raíz compleja ``k`` cerca de ``k0`` (Im > 0: el modo pierde energía hacia el fluido), por la secante."""
    w = 2 * np.pi * f

    def g(k):
        return complex(lamb_function(np.array([k]), w, thickness, cp, cs, family, rho, rho_f, c_f)[0])

    k_a, k_b = complex(k0), complex(k0) * (1 + 1e-4) + 1e-6j * abs(k0)
    g_a, g_b = g(k_a), g(k_b)
    for _ in range(100):
        if g_b == g_a:
            break
        k_c = k_b - g_b * (k_b - k_a) / (g_b - g_a)
        k_a, g_a = k_b, g_b
        k_b, g_b = k_c, g(k_c)
        if abs(k_b - k_a) < 1e-12 * abs(k_b):
            break
    return k_b


def lamb_fundamental(freqs, thickness: float, cp: float, cs: float, family: str, rho: float = 1.0,
                     rho_f: float = 0.0, c_f: float = 1500.0, n_continuation: int = 40) -> np.ndarray:
    """``k(f)`` (complejo) del modo fundamental S0 o A0, libre (``rho_f = 0``) o con carga de fluido.

    Con fluido, la raíz se sigue por continuación para no saltar a otra rama:
    en la primera frecuencia se aumenta ``rho_f`` desde 0 en ``n_continuation``
    pasos partiendo de la raíz de la placa libre, y después se avanza en
    frecuencia usando la raíz anterior (escalada por ``f``) como semilla. Así
    A0 sigue siendo el modo ligado (más lento que el fluido, sin fuga) que a
    alta frecuencia tiende a la onda de Scholte, y S0 el modo con fuga.
    ``freqs`` debe ir en orden creciente y empezar a baja frecuencia.
    """
    freqs = np.asarray(freqs, dtype=float)
    out = np.empty(freqs.size, dtype=complex)
    if rho_f <= 0:
        for n, f in enumerate(freqs):
            out[n] = lamb_wavenumbers(f, thickness, cp, cs, family)[0]
        return out
    k = complex(lamb_wavenumbers(freqs[0], thickness, cp, cs, family)[0])
    for rf in np.linspace(0.0, rho_f, n_continuation + 1)[1:]:
        k = lamb_fluid_loaded(freqs[0], thickness, cp, cs, rho, rf, c_f, family, k)
    out[0] = k
    for n in range(1, freqs.size):
        k = lamb_fluid_loaded(freqs[n], thickness, cp, cs, rho, rho_f, c_f, family, k * freqs[n] / freqs[n - 1])
        out[n] = k
    return out
