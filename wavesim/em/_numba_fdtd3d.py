"""Kernels Numba del FDTD de Maxwell 3D (backend opcional de ``em.fdtd3d``).

Como en 2D (``_numba_fdtd``), cada paso son dos barridos fusionados,
paralelos por planos z (``prange``):

  1. ``step_h``: seis derivadas de E (hacia delante) con CPML y avance de Hx, Hy, Hz.
  2. ``step_e`` / ``step_e_debye``: seis derivadas de H (hacia atrás) con CPML,
     rotacional y actualización de Ex, Ey, Ez (con las corrientes de
     polarización de los polos de Debye en la variante de banda ancha).

Cada plano escribe sólo en sí mismo (campos, corrientes y ``psi`` de la CPML
en sus celdas) y lee campos que ese barrido no modifica: no hay carreras. Dentro
de un plano se recorre por filas (z, y fijos) con buffers de una fila, de modo
que los bucles internos (a lo largo de x, contiguo) son sencillos.

Para coincidir con el backend NumPy salvo redondeo: sin ``fastmath``, las
derivadas sólo existen donde las escribe NumPy (hacia delante ``[0, n-2]``,
hacia atrás ``[1, n-1]``, cero fuera), el orden de las operaciones es el de
``fdtd3d`` y los escalares (``ch``, ``1/dx``) llegan con el dtype de los campos.

Las fuentes y las correcciones de campo total/campo dispersado se aplican
fuera de los kernels, como en 2D.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _dx(f, out, k, j, forward, inv_dx, p):
    """Derivada en x de la fila ``(k, j)`` de ``f`` con CPML (``p``: tupla de ``CPML3D.pack``)."""
    nx = out.shape[0]
    if forward:
        for i in range(nx - 1):
            out[i] = (f[k, j, i + 1] - f[k, j, i]) * inv_dx
        out[nx - 1] = 0.0
    else:
        out[0] = 0.0
        for i in range(1, nx):
            out[i] = (f[k, j, i] - f[k, j, i - 1]) * inv_dx
    for i in range(min(p[0], nx)):
        psi = p[3][i] * p[6][k, j, i] + p[2][i] * out[i]
        p[6][k, j, i] = psi
        out[i] += psi
    for i in range(p[1], nx):
        q = i - p[1]
        psi = p[5][q] * p[7][k, j, q] + p[4][q] * out[i]
        p[7][k, j, q] = psi
        out[i] += psi


@njit(cache=True)
def _cpml_row(out, psi_row, a, b):
    for i in range(out.shape[0]):
        psi = b * psi_row[i] + a * out[i]
        psi_row[i] = psi
        out[i] += psi


@njit(cache=True)
def _dy(f, out, k, j, forward, inv_dx, p):
    """Derivada en y de ``f`` en la fila ``(k, j)`` con CPML."""
    ny, nx = f.shape[1], f.shape[2]
    if forward:
        if j < ny - 1:
            for i in range(nx):
                out[i] = (f[k, j + 1, i] - f[k, j, i]) * inv_dx
        else:
            out[:] = 0.0
    else:
        if j > 0:
            for i in range(nx):
                out[i] = (f[k, j, i] - f[k, j - 1, i]) * inv_dx
        else:
            out[:] = 0.0
    if j < p[0]:
        _cpml_row(out, p[6][k, j], p[2][j], p[3][j])
    elif j >= p[1]:
        q = j - p[1]
        _cpml_row(out, p[7][k, q], p[4][q], p[5][q])


@njit(cache=True)
def _dz(f, out, k, j, forward, inv_dx, p):
    """Derivada en z de ``f`` en la fila ``(k, j)`` con CPML."""
    nz, nx = f.shape[0], f.shape[2]
    if forward:
        if k < nz - 1:
            for i in range(nx):
                out[i] = (f[k + 1, j, i] - f[k, j, i]) * inv_dx
        else:
            out[:] = 0.0
    else:
        if k > 0:
            for i in range(nx):
                out[i] = (f[k, j, i] - f[k - 1, j, i]) * inv_dx
        else:
            out[:] = 0.0
    if k < p[0]:
        _cpml_row(out, p[6][k, j], p[2][k], p[3][k])
    elif k >= p[1]:
        q = k - p[1]
        _cpml_row(out, p[7][q, j], p[4][q], p[5][q])


@njit(cache=True, parallel=True)
def step_h(ex, ey, ez, hx, hy, hz, ch, inv_dx, p_ey_z, p_ez_y, p_ez_x, p_ex_z, p_ex_y, p_ey_x):
    """``H += dt/mu0 (-rot E)``: Hx con dEy/dz - dEz/dy, Hy con dEz/dx - dEx/dz y Hz con dEx/dy - dEy/dx."""
    nz, ny, nx = ex.shape
    for kp in prange(nz):
        k = np.int64(kp)  # el índice de prange es uint64: restar un int64 daría float64
        a = np.empty(nx, ex.dtype)
        b = np.empty(nx, ex.dtype)
        for j in range(ny):
            _dz(ey, a, k, j, True, inv_dx, p_ey_z)
            _dy(ez, b, k, j, True, inv_dx, p_ez_y)
            for i in range(nx):
                hx[k, j, i] += (a[i] - b[i]) * ch
            _dx(ez, a, k, j, True, inv_dx, p_ez_x)
            _dz(ex, b, k, j, True, inv_dx, p_ex_z)
            for i in range(nx):
                hy[k, j, i] += (a[i] - b[i]) * ch
            _dy(ex, a, k, j, True, inv_dx, p_ex_y)
            _dx(ey, b, k, j, True, inv_dx, p_ey_x)
            for i in range(nx):
                hz[k, j, i] += (a[i] - b[i]) * ch


@njit(cache=True, parallel=True)
def step_e(ex, ey, ez, hx, hy, hz, cax, cbx, cay, cby, caz, cbz, inv_dx,
           p_hz_y, p_hy_z, p_hx_z, p_hz_x, p_hy_x, p_hx_y):
    """``E <- ca E + cb rot H`` (medio de banda estrecha) en las tres componentes."""
    nz, ny, nx = ex.shape
    for kp in prange(nz):
        k = np.int64(kp)
        a = np.empty(nx, ex.dtype)
        b = np.empty(nx, ex.dtype)
        for j in range(ny):
            _dy(hz, a, k, j, False, inv_dx, p_hz_y)
            _dz(hy, b, k, j, False, inv_dx, p_hy_z)
            for i in range(nx):
                ex[k, j, i] = ex[k, j, i] * cax[k, j, i] + cbx[k, j, i] * (a[i] - b[i])
            _dz(hx, a, k, j, False, inv_dx, p_hx_z)
            _dx(hz, b, k, j, False, inv_dx, p_hz_x)
            for i in range(nx):
                ey[k, j, i] = ey[k, j, i] * cay[k, j, i] + cby[k, j, i] * (a[i] - b[i])
            _dx(hy, a, k, j, False, inv_dx, p_hy_x)
            _dy(hx, b, k, j, False, inv_dx, p_hx_y)
            for i in range(nx):
                ez[k, j, i] = ez[k, j, i] * caz[k, j, i] + cbz[k, j, i] * (a[i] - b[i])


@njit(cache=True)
def _debye_row(e, ca, cb, jp, beta, kappa, g, k, j, c):
    """Una fila de una componente: ``c -= sum g_k J_k``, ``E <- ca E + cb c`` y ``J_k <- kappa J_k + beta dE``."""
    npol = jp.shape[0]
    nx = c.shape[0]
    for m in range(npol):
        for i in range(nx):
            c[i] -= g[m] * jp[m, k, j, i]
    for i in range(nx):
        old = e[k, j, i]
        new = old * ca[k, j, i] + cb[k, j, i] * c[i]
        e[k, j, i] = new
        c[i] = new - old
    for m in range(npol):
        for i in range(nx):
            jp[m, k, j, i] = jp[m, k, j, i] * kappa[m] + beta[m, k, j, i] * c[i]


@njit(cache=True, parallel=True)
def step_e_debye(ex, ey, ez, hx, hy, hz, cax, cbx, cay, cby, caz, cbz, inv_dx,
                 p_hz_y, p_hy_z, p_hx_z, p_hz_x, p_hy_x, p_hx_y,
                 jx, jy, jz, bx, by, bz, kappa, g):
    """Actualiza E y las corrientes de polarización de los polos de Debye (ADE), componente a componente."""
    nz, ny, nx = ex.shape
    for kp in prange(nz):
        k = np.int64(kp)
        a = np.empty(nx, ex.dtype)
        b = np.empty(nx, ex.dtype)
        for j in range(ny):
            _dy(hz, a, k, j, False, inv_dx, p_hz_y)
            _dz(hy, b, k, j, False, inv_dx, p_hy_z)
            for i in range(nx):
                a[i] = a[i] - b[i]
            _debye_row(ex, cax, cbx, jx, bx, kappa, g, k, j, a)
            _dz(hx, a, k, j, False, inv_dx, p_hx_z)
            _dx(hz, b, k, j, False, inv_dx, p_hz_x)
            for i in range(nx):
                a[i] = a[i] - b[i]
            _debye_row(ey, cay, cby, jy, by, kappa, g, k, j, a)
            _dx(hy, a, k, j, False, inv_dx, p_hy_x)
            _dy(hx, b, k, j, False, inv_dx, p_hx_y)
            for i in range(nx):
                a[i] = a[i] - b[i]
            _debye_row(ez, caz, cbz, jz, bz, kappa, g, k, j, a)


@njit(cache=True, parallel=True)
def accumulate_dft(acc, f, ph_re, ph_im):
    """``acc[m] += f * (ph_re[m] + j ph_im[m])`` para cada frecuencia ``m`` (DFT corriente)."""
    nf = acc.shape[0]
    nz, ny, nx = f.shape
    for kp in prange(nz):
        k = np.int64(kp)
        for m in range(nf):
            w = complex(ph_re[m], ph_im[m])
            for j in range(ny):
                for i in range(nx):
                    acc[m, k, j, i] += f[k, j, i] * w


@njit(cache=True, parallel=True)
def max_abs(f):
    """Máximo de ``|f|`` (para el criterio de parada de banda ancha sin crear temporales)."""
    nz = f.shape[0]
    out = np.zeros(nz)
    for kp in prange(nz):
        k = np.int64(kp)
        m = 0.0
        for j in range(f.shape[1]):
            for i in range(f.shape[2]):
                v = abs(f[k, j, i])
                if v > m:
                    m = v
        out[k] = m
    return out.max()
