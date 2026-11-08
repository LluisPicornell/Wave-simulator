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

    Los coeficientes se aplanan a 1D (a lo largo del eje de la derivada); los
    ``psi`` se pasan tal cual para que el kernel los actualice in situ.
    """
    return (
        cp.nl,
        cp.nr_start,
        cp.a_l.ravel(),
        cp.b_l.ravel(),
        cp.a_r.ravel(),
        cp.b_r.ravel(),
        cp.psi_l,
        cp.psi_r,
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
        psi = p[3][j] * p[6][i, j] + p[2][j] * out[j]
        p[6][i, j] = psi
        out[j] += psi
    for j in range(max(p[1], lo), hi):
        k = j - p[1]
        psi = p[5][k] * p[7][i, k] + p[4][k] * out[j]
        p[7][i, k] = psi
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
        a, b = p[2][i], p[3][i]
        for j in range(nx):
            psi = b * p[6][i, j] + a * out[j]
            p[6][i, j] = psi
            out[j] += psi
    elif i >= p[1]:
        k = i - p[1]
        a, b = p[4][k], p[5][k]
        for j in range(nx):
            psi = b * p[7][k, j] + a * out[j]
            p[7][k, j] = psi
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


@njit(parallel=True, cache=True)
def stress_kernel(
    vx, vz, sxx, szz, sxz, rxx, rzz, rxz,
    dt_pi, dt_mu2, dt_mu_xz, dt_pi_t, dt_mu2_t, dt_mu_t_xz, mem_a, mem_b,
    has_shear, c1, c2, p_exx, p_ezz, p_exz_a, p_exz_b,
):  # fmt: skip
    """Deformaciones, tensiones y memorias viscoelásticas (Crank-Nicolson).

    ``rxx``, ``rzz``, ``rxz`` son pilas ``(n_mech, nz, nx)``; con ``n_mech = 0``
    el bucle de memorias no se ejecuta y los coeficientes ``*_t`` no se leen.
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
        for j in range(nx):
            exx, ezz, exz = row_exx[j], row_ezz[j], row_a[j] + row_b[j]

            div = exx + ezz
            a1 = dt_pi[i, j] * div
            sx = (sxx[i, j] + a1) - dt_mu2[i, j] * ezz
            sz = (szz[i, j] + a1) - dt_mu2[i, j] * exx
            sxz_v = sxz[i, j]
            if has_shear:
                sxz_v += dt_mu_xz[i, j] * exz

            if n_mech > 0:
                vol = dt_pi_t[i, j] * div
                f_xx = vol - dt_mu2_t[i, j] * ezz
                f_zz = vol - dt_mu2_t[i, j] * exx
                f_xz = dt_mu_t_xz[i, j] * exz
                for l in range(n_mech):
                    a, b = mem_a[l], mem_b[l]
                    half_a, half_b = 0.5 * (1.0 + a), 0.5 * b
                    # Los escalares de memoria son float64 (como en NumPy): se
                    # multiplica en doble precisión y se redondea al guardar
                    r = rxx[l, i, j]
                    sx = sx + F32(r * half_a)
                    sx = sx - F32(f_xx * half_b)
                    rxx[l, i, j] = F32(r * a) - F32(f_xx * b)
                    r = rzz[l, i, j]
                    sz = sz + F32(r * half_a)
                    sz = sz - F32(f_zz * half_b)
                    rzz[l, i, j] = F32(r * a) - F32(f_zz * b)
                    if has_shear:
                        r = rxz[l, i, j]
                        sxz_v = sxz_v + F32(r * half_a)
                        sxz_v = sxz_v - F32(f_xz * half_b)
                        rxz[l, i, j] = F32(r * a) - F32(f_xz * b)

            sxx[i, j] = sx
            szz[i, j] = sz
            sxz[i, j] = sxz_v
