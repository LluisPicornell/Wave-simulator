"""Funciones de Bessel de orden entero (sin SciPy) para las soluciones analíticas 2D.

- ``J_n(z)`` para ``z`` complejo, con la representación integral de Bessel

      J_n(z) = (1/pi) ∫_0^pi cos(n tau - z sin tau) dtau = (1/2pi) ∫_-pi^pi e^{j(z sin tau - n tau)} dtau.

  El integrando es periódico y analítico, así que la regla del trapecio con
  ``N`` nodos es espectralmente exacta: su único error es el *aliasing*
  ``J_{n±N}(z)``, despreciable si ``N > |z| + n + 40``. Los ``N`` nodos dan a
  la vez todos los órdenes ``0..N-1`` con una FFT (son los coeficientes de
  Fourier de ``e^{j z sin tau}``, la expansión de Jacobi-Anger). El trapecio
  sólo tiene precisión *absoluta* (~1e-16 e^{|Im z|}); para los órdenes
  ``n > |z|``, donde ``J_n`` es diminuta, se usa la recurrencia hacia atrás de
  Miller normalizada con los órdenes bajos del trapecio.
- ``Y_n(z)`` para ``Re z > 0`` (pensada para argumento real): ``Y_0`` e ``Y_1``
  con la representación integral (DLMF 10.9.7)

      Y_n(z) = (1/pi) ∫_0^pi sin(z sin t - n t) dt - (1/pi) ∫_0^inf (e^{nt} + (-1)^n e^{-nt}) e^{-z sinh t} dt

  con cuadratura de Gauss-Legendre compuesta, y los órdenes superiores con la
  recurrencia ``Y_{n+1} = (2n/z) Y_n - Y_{n-1}``, estable hacia arriba porque
  ``Y_n`` es la solución dominante.
- ``H_n^(2) = J_n - j Y_n``: onda saliente con el convenio ``e^{+jwt}`` del paquete.

Precisión comprobada en ``tests/test_em_dispersive.py`` frente a los valores
tabulados de Abramowitz y Stegun (tablas 9.1, 9.2 y 9.8) y la identidad del
wronskiano: ~1e-13 relativo.
"""

from __future__ import annotations

import numpy as np


def _trapezoid_nodes(z_abs_max: float, n_max: int) -> int:
    n = int(np.ceil(z_abs_max + n_max + 40))
    return 1 << max(int(np.ceil(np.log2(max(n, 64)))), 6)


def _besselj_fft(n_max: int, flat: np.ndarray) -> np.ndarray:
    """Trapecio (FFT) de la integral de Bessel: error absoluto ~1e-16 e^{|Im z|}."""
    n_nodes = _trapezoid_nodes(float(np.abs(flat).max(initial=0.0)), n_max)
    tau = 2.0 * np.pi * np.arange(n_nodes) / n_nodes
    out = np.empty((n_max + 1, flat.size), dtype=complex)
    chunk = max(1, 2_000_000 // n_nodes)
    for s in range(0, flat.size, chunk):
        e = np.exp(1j * flat[s:s + chunk, None] * np.sin(tau)[None, :])
        out[:, s:s + chunk] = (np.fft.fft(e, axis=1)[:, : n_max + 1] / n_nodes).T
    return out


def _besselj_miller(n_max: int, flat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Recurrencia hacia atrás (Miller), normalizada con los órdenes bajos de ``reference``.

    ``J_n`` es la solución mínima de ``C_{n-1} = (2n/z) C_n - C_{n+1}`` cuando
    ``n`` crece, así que la recurrencia hacia atrás da ``J_n`` con precisión
    *relativa* también donde es diminuta (``n >> |z|``), que es justo donde el
    trapecio sólo tiene precisión absoluta.
    """
    n_start = n_max + int(np.abs(flat).max(initial=0.0)) + 60
    seq = np.zeros((n_start + 2, flat.size), dtype=complex)
    seq[n_start] = 1e-30
    for n in range(n_start, 0, -1):
        seq[n - 1] = (2.0 * n / flat) * seq[n] - seq[n + 1]
        big = np.abs(seq[n - 1]) > 1e250
        if big.any():
            seq[:, big] *= 1e-250
    seq = seq[: n_max + 1]
    seq /= np.abs(seq).max(axis=0)[None, :]
    # Escala: mínimos cuadrados con los órdenes n <= |z| + 1, donde el trapecio es exacto
    m = np.minimum(np.floor(np.abs(flat)).astype(int) + 1, n_max)
    use = np.arange(n_max + 1)[:, None] <= m[None, :]
    num = np.sum(np.where(use, np.conj(seq) * reference, 0.0), axis=0)
    den = np.sum(np.where(use, np.abs(seq) ** 2, 0.0), axis=0)
    return seq * (num / den)[None, :]


def besselj_orders(n_max: int, z) -> np.ndarray:
    """``J_0(z) .. J_{n_max}(z)``: array de forma ``(n_max + 1,) + z.shape`` (complejo).

    Trapecio para los órdenes ``n <= |z|`` y recurrencia de Miller para el resto.
    """
    z = np.asarray(z, dtype=complex)
    flat = z.ravel()
    out = _besselj_fft(max(n_max, 1), flat)
    nz = flat != 0
    if nz.any():
        out[:, nz] = _besselj_miller(max(n_max, 1), flat[nz], out[:, nz])
    if (~nz).any():
        out[:, ~nz] = 0.0
        out[0, ~nz] = 1.0
    return out[: n_max + 1].reshape((n_max + 1,) + z.shape)


def besselj(n: int, z) -> np.ndarray:
    """``J_n(z)`` de orden entero (negativo incluido: ``J_{-n} = (-1)^n J_n``) y ``z`` complejo."""
    m = abs(int(n))
    j = besselj_orders(m, z)[m]
    return j * (-1) ** m if n < 0 else j


def _gauss_legendre(a: float, b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _composite_gauss(a: float, b: float, n_seg: int, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = [_gauss_legendre(a + (b - a) * k / n_seg, a + (b - a) * (k + 1) / n_seg, n_nodes) for k in range(n_seg)]
    return np.concatenate([x for x, _ in nodes]), np.concatenate([w for _, w in nodes])


def _y0_y1(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``Y_0`` e ``Y_1`` por la representación integral (``Re z > 0``), por bloques de puntos."""
    if np.any(z.real <= 0):
        raise ValueError("Y_n sólo está implementada para Re z > 0")
    y0 = np.empty(z.size, dtype=complex)
    y1 = np.empty(z.size, dtype=complex)
    order = np.argsort(np.abs(z))  # bloques de |z| parecido: menos nodos por bloque
    for s in range(0, z.size, 512):
        idx = order[s:s + 512]
        zz = z[idx][:, None]
        z_max, zr_min = float(np.abs(zz).max()), float(zz.real.min())
        # [0, pi]: tramos en los que la fase z sin t avanza < 4 rad, 20 nodos por tramo
        th, wt = _composite_gauss(0.0, np.pi, int(np.ceil(z_max * np.pi / 4.0)) + 2, 20)
        sn = np.sin(th)[None, :]
        i0 = (np.sin(zz * sn) * wt).sum(axis=1) / np.pi
        i1 = (np.sin(zz * sn - th[None, :]) * wt).sum(axis=1) / np.pi
        # [0, T]: e^{-z sinh t} < e^-60 a partir de T; tramos en los que el exponente cambia < 4
        t_end = float(np.arcsinh(60.0 / zr_min)) + 1.0
        h = min(0.5, 4.0 / max(z_max, 1e-300))
        tt, wt = _composite_gauss(0.0, t_end, int(np.ceil(t_end / h)), 16)
        decay = np.exp(-zz * np.sinh(tt)[None, :]) * wt
        k0 = 2.0 * decay.sum(axis=1) / np.pi
        k1 = 2.0 * (np.sinh(tt)[None, :] * decay).sum(axis=1) / np.pi
        y0[idx], y1[idx] = i0 - k0, i1 - k1
    return y0, y1


def bessely_orders(n_max: int, z) -> np.ndarray:
    """``Y_0(z) .. Y_{n_max}(z)`` para ``Re z > 0``; forma ``(n_max + 1,) + z.shape``.

    Con ``z`` real devuelve valores reales.
    """
    z_in = np.asarray(z)
    real = not np.iscomplexobj(z_in)
    zc = z_in.astype(complex).ravel()
    out = np.empty((n_max + 1, zc.size), dtype=complex)
    y0, y1 = _y0_y1(zc)
    out[0] = y0
    if n_max >= 1:
        out[1] = y1
    for n in range(1, n_max):
        out[n + 1] = (2.0 * n / zc) * out[n] - out[n - 1]
    out = out.reshape((n_max + 1,) + z_in.shape)
    return out.real if real else out


def bessely(n: int, z) -> np.ndarray:
    """``Y_n(z)`` de orden entero (``Y_{-n} = (-1)^n Y_n``) para ``Re z > 0``."""
    m = abs(int(n))
    y = bessely_orders(m, z)[m]
    return y * (-1) ** m if n < 0 else y


def hankel2_orders(n_max: int, x) -> np.ndarray:
    """``H_n^(2)(x) = J_n(x) - j Y_n(x)``, ``n = 0..n_max`` (``Re x > 0``)."""
    return besselj_orders(n_max, x) - 1j * bessely_orders(n_max, x)


def derivative_orders(f_orders: np.ndarray) -> np.ndarray:
    """Derivadas ``C_n'(z) = (C_{n-1} - C_{n+1}) / 2`` a partir de los órdenes ``0..N`` (válidas hasta ``N-1``).

    Para ``n = 0`` usa ``C_0' = -C_1``.
    """
    d = np.empty_like(f_orders[:-1])
    d[0] = -f_orders[1]
    d[1:] = 0.5 * (f_orders[:-2] - f_orders[2:])
    return d
