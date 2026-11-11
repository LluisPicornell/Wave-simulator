"""Kernels Numba del multigrid de agregación (``em.multigrid``): suavizado Gauss-Seidel rojo-negro y transferencias.

Cada nivel es un operador de 5 puntos simétrico ``A = D - F`` con conductancias de cara ``fx[j, i]`` (entre las
celdas ``(j, i)`` y ``(j, i + 1)``) y ``fz[j, i]`` (entre ``(j, i)`` y ``(j + 1, i)``), nulas en el último
índice. Sirven igual para coeficientes reales (difusión óptica) y complejos (admitividad).
"""

from __future__ import annotations

from numba import njit


@njit(cache=True)
def gs_color(x, b, fx, fz, invd, color):
    """Una pasada de Gauss-Seidel sobre las celdas de un color del tablero ((j + i) % 2 == color)."""
    nz, nx = x.shape
    for j in range(nz):
        for i in range((j + color) % 2, nx, 2):
            s = b[j, i]
            if i > 0:
                s += fx[j, i - 1] * x[j, i - 1]
            if i < nx - 1:
                s += fx[j, i] * x[j, i + 1]
            if j > 0:
                s += fz[j - 1, i] * x[j - 1, i]
            if j < nz - 1:
                s += fz[j, i] * x[j + 1, i]
            x[j, i] = s * invd[j, i]


@njit(cache=True)
def residual(r, b, x, fx, fz, d):
    """``r = b - A x``."""
    nz, nx = x.shape
    for j in range(nz):
        for i in range(nx):
            s = d[j, i] * x[j, i]
            if i > 0:
                s -= fx[j, i - 1] * x[j, i - 1]
            if i < nx - 1:
                s -= fx[j, i] * x[j, i + 1]
            if j > 0:
                s -= fz[j - 1, i] * x[j - 1, i]
            if j < nz - 1:
                s -= fz[j, i] * x[j + 1, i]
            r[j, i] = b[j, i] - s


@njit(cache=True)
def restrict_sum(rc, r):
    """Restricción por suma sobre agregados 2x2 (la traspuesta de la prolongación constante)."""
    nz, nx = r.shape
    rc[:, :] = 0
    for j in range(nz):
        for i in range(nx):
            rc[j // 2, i // 2] += r[j, i]


@njit(cache=True)
def prolong_add(x, xc, w):
    """``x += w * P xc`` con ``P`` constante a trozos sobre los agregados 2x2."""
    nz, nx = x.shape
    for j in range(nz):
        for i in range(nx):
            x[j, i] += w * xc[j // 2, i // 2]


@njit(cache=True)
def dense_apply(out, m, v):
    """``out = m @ v`` sin BLAS: en el nivel más grueso las llamadas multihilo cuestan más que el producto."""
    n = v.size
    for a in range(n):
        s = m[a, 0] * v[0]
        for c in range(1, n):
            s += m[a, c] * v[c]
        out[a] = s
