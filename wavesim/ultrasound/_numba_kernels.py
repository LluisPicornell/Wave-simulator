"""Kernels Numba del paso FDTD elástico (backend opcional de ``ElasticSolver``).

El backend NumPy recorre la malla muchas veces (una por derivada, por
producto y por suma) y es limitado por ancho de banda de memoria. Aquí cada
paso se reduce a dos barridos fusionados, paralelizados por filas:

  1. ``velocity_kernel``: derivadas de tensión con CPML y actualización de vx, vz.
  2. ``stress_kernel``: derivadas de velocidad con CPML, tensiones y memorias
     viscoelásticas.

Cada celda sólo escribe en sí misma (campo, memorias y psi de la CPML en su
posición) y lee de campos que ese barrido no modifica, así que no hay
condiciones de carrera con ``prange`` sobre filas.

Para que el resultado coincida con NumPy salvo redondeo:
  - no se usa ``fastmath`` (sin reordenaciones ni contracciones FMA);
  - las derivadas sólo existen en las mismas celdas que escribe el backend
    NumPy (``[1, n-3]`` hacia delante, ``[2, n-2]`` hacia atrás); fuera valen
    cero, igual que los bordes nunca escritos de sus buffers;
  - el orden de las operaciones en float32 es el mismo que en ``step``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

F32 = np.float32


def pack_pml(cp) -> tuple:
    """Empaqueta una ``CPMLDerivative`` en una tupla que Numba sabe recibir.

    Los coeficientes de las franjas propias se pasan como arrays 2D de la
    forma de su ``psi`` (con M-PML dependen de las dos coordenadas en las
    esquinas) y los de las franjas de la otra dirección, aplanados a 1D. Los
    ``psi`` se pasan tal cual para que el kernel los actualice in situ.
    """

    def full(c, psi):
        return np.ascontiguousarray(np.broadcast_to(c, psi.shape))

    if cp.ratio > 0:
        cross = (cp.ml, cp.mr_start, cp.a_cl.ravel(), cp.b_cl.ravel(), cp.a_cr.ravel(), cp.b_cr.ravel(),
                 cp.psi_cl, cp.psi_cr)
    else:
        e1, e2 = np.zeros(0, np.float32), np.zeros((0, 0), np.float32)
        other = cp.psi_l.shape[0] if cp.axis == 1 else cp.psi_l.shape[1]
        cross = (0, other, e1, e1, e1, e1, e2, e2)
    return (
        cp.nl,
        cp.nr_start,
        full(cp.a_l, cp.psi_l),
        full(cp.b_l, cp.psi_l),
        full(cp.a_r, cp.psi_r),
        full(cp.b_r, cp.psi_r),
        cp.psi_l,
        cp.psi_r,
        *cross,
    )


@njit(cache=True)
def _row_dx(f, out, i, forward, c1, c2, p):
    """Derivada en x de la fila ``i`` de ``f`` en ``out`` (0 fuera de lo que escribe NumPy) con CPML."""
    nx = out.shape[0]
    lo, hi = (1, nx - 2) if forward else (2, nx - 1)
    out[:] = F32(0.0)
    if forward:
        for j in range(lo, hi):
            out[j] = (f[i, j + 1] - f[i, j]) * c1 + (f[i, j + 2] - f[i, j - 1]) * c2
    else:
        for j in range(lo, hi):
            out[j] = (f[i, j] - f[i, j - 1]) * c1 + (f[i, j + 1] - f[i, j - 2]) * c2
    # La CPML sólo actúa en las franjas de los extremos
    for j in range(lo, min(p[0], hi)):
        psi = p[3][i, j] * p[6][i, j] + p[2][i, j] * out[j]
        p[6][i, j] = psi
        out[j] += psi
    for j in range(max(p[1], lo), hi):
        k = j - p[1]
        psi = p[5][i, k] * p[7][i, k] + p[4][i, k] * out[j]
        p[7][i, k] = psi
        out[j] += psi
    # M-PML: filas de las capas de arriba y abajo, entre las franjas propias (ninguna si ratio = 0)
    if i < p[8]:
        _cross_row(out, i, p[10][i], p[11][i], p[14], max(p[0], lo), min(p[1], hi), p[0])
    elif i >= p[9]:
        r = i - p[9]
        _cross_row(out, r, p[12][r], p[13][r], p[15], max(p[0], lo), min(p[1], hi), p[0])


@njit(cache=True, inline="always")
def _cross_row(out, r, a, b, psi_c, j0, j1, offset):
    """Recursión CPML con coeficientes ``(a, b)`` en ``out[j0:j1]``, memoria en la fila ``r`` de ``psi_c``."""
    for j in range(j0, j1):
        k = j - offset
        psi = b * psi_c[r, k] + a * out[j]
        psi_c[r, k] = psi
        out[j] += psi


@njit(cache=True)
def _row_dz(f, out, i, forward, c1, c2, p):
    """Derivada en z de ``f`` en la fila ``i`` (hacia ``out``) con CPML; 0 si la fila no se escribe."""
    nz, nx = f.shape
    out[:] = F32(0.0)
    if forward:
        if i < 1 or i > nz - 3:
            return
        for j in range(nx):
            out[j] = (f[i + 1, j] - f[i, j]) * c1 + (f[i + 2, j] - f[i - 1, j]) * c2
    else:
        if i < 2 or i > nz - 2:
            return
        for j in range(nx):
            out[j] = (f[i, j] - f[i - 1, j]) * c1 + (f[i + 1, j] - f[i - 2, j]) * c2
    if i < p[0]:
        for j in range(nx):
            psi = p[3][i, j] * p[6][i, j] + p[2][i, j] * out[j]
            p[6][i, j] = psi
            out[j] += psi
    elif i >= p[1]:
        k = i - p[1]
        for j in range(nx):
            psi = p[5][k, j] * p[7][k, j] + p[4][k, j] * out[j]
            p[7][k, j] = psi
            out[j] += psi
    else:
        # M-PML: columnas de las capas laterales (vacío si ratio = 0)
        r = i - p[0]
        for j in range(p[8]):
            psi = p[11][j] * p[14][r, j] + p[10][j] * out[j]
            p[14][r, j] = psi
            out[j] += psi
        for j in range(p[9], nx):
            k = j - p[9]
            psi = p[13][k] * p[15][r, k] + p[12][k] * out[j]
            p[15][r, k] = psi
            out[j] += psi


@njit(parallel=True, cache=True)
def velocity_kernel(vx, vz, sxx, szz, sxz, dtbx, dtbz, c1, c2, p_sxx_x, p_sxz_z, p_sxz_x, p_szz_z):
    """vx += dt/rho (dsxx/dx + dsxz/dz);  vz += dt/rho (dsxz/dx + dszz/dz)."""
    nz, nx = vx.shape
    for i in prange(nz):
        a = np.empty(nx, np.float32)
        b = np.empty(nx, np.float32)
        _row_dx(sxx, a, i, True, c1, c2, p_sxx_x)
        _row_dz(sxz, b, i, False, c1, c2, p_sxz_z)
        for j in range(nx):
            vx[i, j] += (a[j] + b[j]) * dtbx[i, j]
        _row_dx(sxz, a, i, False, c1, c2, p_sxz_x)
        _row_dz(szz, b, i, True, c1, c2, p_szz_z)
        for j in range(nx):
            vz[i, j] += (a[j] + b[j]) * dtbz[i, j]


@njit(cache=True, inline="always")
def _relax(stress, r_arr, m, i, j, forcing, a, b):
    """Memoria ``m`` por Crank-Nicolson: devuelve ``stress + (r_new + r_old)/2`` y guarda ``r_new = a r - b F``.

    Los escalares de memoria son float64 (como en NumPy): se multiplica en
    doble precisión y se redondea al guardar.
    """
    r = r_arr[m, i, j]
    stress = stress + F32(r * (0.5 * (1.0 + a)))
    stress = stress - F32(forcing * (0.5 * b))
    r_arr[m, i, j] = F32(r * a) - F32(forcing * b)
    return stress


@njit(parallel=True, cache=True)
def stress_kernel(
    vx, vz, sxx, szz, sxz, rxx, rzz, rxz,
    dt_pi, dt_mu2, dt_mu_xz, dt_pi_t, dt_mu2_t, dt_mu_t_xz, mem_a, mem_b,
    has_shear, c1, c2, p_exx, p_ezz, p_exz_a, p_exz_b,
    aniso, dt_c11, dt_c33, dt_c13, dt_c11_t, dt_c33_t, dt_c13_t,
):  # fmt: skip
    """Deformaciones, tensiones y memorias viscoelásticas (Crank-Nicolson).

    ``rxx``, ``rzz``, ``rxz`` y los coeficientes ``*_t`` (uno por mecanismo)
    son pilas ``(n_mech, nz, nx)``; con ``n_mech = 0`` el bucle de memorias no
    se ejecuta y los coeficientes no se leen.

    Con ``aniso`` las tensiones normales usan ``dt_c11``, ``dt_c33`` y
    ``dt_c13`` (y sus memorias ``*_t``) en lugar de ``dt_pi`` y ``dt_mu2``. La
    rama se elige por fila: el camino isótropo no paga nada por ella.
    """
    nz, nx = vx.shape
    n_mech = mem_a.shape[0]
    for i in prange(nz):
        row_exx = np.empty(nx, np.float32)
        row_ezz = np.empty(nx, np.float32)
        row_a = np.empty(nx, np.float32)
        row_b = np.empty(nx, np.float32)
        _row_dx(vx, row_exx, i, False, c1, c2, p_exx)
        _row_dz(vz, row_ezz, i, False, c1, c2, p_ezz)
        _row_dz(vx, row_a, i, True, c1, c2, p_exz_a)
        _row_dx(vz, row_b, i, True, c1, c2, p_exz_b)
        if aniso:
            for j in range(nx):
                exx, ezz, exz = row_exx[j], row_ezz[j], row_a[j] + row_b[j]
                sx = (sxx[i, j] + dt_c11[i, j] * exx) + dt_c13[i, j] * ezz
                sz = (szz[i, j] + dt_c13[i, j] * exx) + dt_c33[i, j] * ezz
                sxz_v = sxz[i, j]
                if has_shear:
                    sxz_v += dt_mu_xz[i, j] * exz
                shear_xz = has_shear and dt_mu_xz[i, j] != F32(0.0)
                for m in range(n_mech):
                    a, b = mem_a[m], mem_b[m]
                    f_xx = dt_c11_t[m, i, j] * exx + dt_c13_t[m, i, j] * ezz
                    f_zz = dt_c13_t[m, i, j] * exx + dt_c33_t[m, i, j] * ezz
                    sx = _relax(sx, rxx, m, i, j, f_xx, a, b)
                    sz = _relax(sz, rzz, m, i, j, f_zz, a, b)
                    if shear_xz:
                        sxz_v = _relax(sxz_v, rxz, m, i, j, dt_mu_t_xz[m, i, j] * exz, a, b)
                sxx[i, j] = sx
                szz[i, j] = sz
                sxz[i, j] = sxz_v
        else:
            for j in range(nx):
                exx, ezz, exz = row_exx[j], row_ezz[j], row_a[j] + row_b[j]

                div = exx + ezz
                a1 = dt_pi[i, j] * div
                sx = (sxx[i, j] + a1) - dt_mu2[i, j] * ezz
                sz = (szz[i, j] + a1) - dt_mu2[i, j] * exx
                sxz_v = sxz[i, j]
                if has_shear:
                    sxz_v += dt_mu_xz[i, j] * exz

                # Sin cizalla (fluido) los coeficientes de cizalla de las memorias son
                # nulos y rxz no sale de cero: no se leen. Es la mayoría de la malla
                # y el bucle está limitado por el ancho de banda de memoria.
                fluid = dt_mu2[i, j] == F32(0.0)
                shear_xz = has_shear and dt_mu_xz[i, j] != F32(0.0)
                for m in range(n_mech):
                    # Cada mecanismo tiene su propio peso: coeficientes (L, nz, nx)
                    a, b = mem_a[m], mem_b[m]
                    vol = dt_pi_t[m, i, j] * div
                    if fluid:
                        f_xx = vol
                        f_zz = vol
                    else:
                        f_xx = vol - dt_mu2_t[m, i, j] * ezz
                        f_zz = vol - dt_mu2_t[m, i, j] * exx
                    sx = _relax(sx, rxx, m, i, j, f_xx, a, b)
                    sz = _relax(sz, rzz, m, i, j, f_zz, a, b)
                    if shear_xz:
                        sxz_v = _relax(sxz_v, rxz, m, i, j, dt_mu_t_xz[m, i, j] * exz, a, b)

                sxx[i, j] = sx
                szz[i, j] = sz
                sxz[i, j] = sxz_v
