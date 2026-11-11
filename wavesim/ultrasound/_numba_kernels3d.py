"""Kernels Numba del paso FDTD elástico 3D (backend opcional de ``ElasticSolver3D``).

Como en 2D (``_numba_kernels``), cada paso son dos barridos fusionados,
paralelos por planos z (``prange``):

  1. ``velocity_kernel``: 9 derivadas de tensión con CPML y actualización de vx, vy, vz.
  2. ``stress_kernel``: 9 derivadas de velocidad con CPML, tensiones y memorias viscoelásticas.

Cada plano escribe sólo en sí mismo (campos, memorias y ``psi`` de la CPML en
sus celdas) y lee de campos que ese barrido no modifica: no hay condiciones de
carrera. Dentro de un plano se recorre por filas (z, y fijos) con buffers de
una fila, de modo que los bucles internos (a lo largo de x, contiguo) son
sencillos.

Para que el resultado coincida con el backend NumPy salvo redondeo: sin
``fastmath``, las derivadas sólo existen en las celdas que escribe NumPy
(``[1, n-3]`` hacia delante, ``[2, n-2]`` hacia atrás, cero fuera) y el orden
de las operaciones en float32 es el de ``ElasticSolver3D._step_numpy``.

``slip_velocity`` y ``slip_stress`` aplican la corrección de interfaz
fluido-sólido (``elastic3d._slip_terms``) después de cada barrido: las
actualizaciones son lineales en cada derivada y los campos que se derivan no
cambian dentro del barrido, así que basta sumar su efecto en los nodos de la
interfaz.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

F32 = np.float32


@njit(cache=True)
def _cpml_row(out, psi_row, a, b):
    """CPML de una fila entera de una derivada a lo largo de y o z (``psi_row``: su fila de ``psi``)."""
    for i in range(out.shape[0]):
        psi = b * psi_row[i] + a * out[i]
        psi_row[i] = psi
        out[i] += psi


@njit(cache=True)
def _dx(f, out, k, j, forward, c1, c2, p):
    """Derivada en x de la fila ``(k, j)`` de ``f`` en ``out`` con CPML (0 donde NumPy no escribe)."""
    nx = out.shape[0]
    out[:] = F32(0.0)
    if forward:
        lo, hi = 1, nx - 2
        for i in range(lo, hi):
            out[i] = (f[k, j, i + 1] - f[k, j, i]) * c1 + (f[k, j, i + 2] - f[k, j, i - 1]) * c2
    else:
        lo, hi = 2, nx - 1
        for i in range(lo, hi):
            out[i] = (f[k, j, i] - f[k, j, i - 1]) * c1 + (f[k, j, i + 1] - f[k, j, i - 2]) * c2
    for i in range(lo, min(p[0], hi)):
        psi = p[3][i] * p[6][k, j, i] + p[2][i] * out[i]
        p[6][k, j, i] = psi
        out[i] += psi
    for i in range(max(p[1], lo), hi):
        q = i - p[1]
        psi = p[5][q] * p[7][k, j, q] + p[4][q] * out[i]
        p[7][k, j, q] = psi
        out[i] += psi


@njit(cache=True)
def _dy(f, out, k, j, forward, c1, c2, p):
    """Derivada en y de ``f`` en la fila ``(k, j)`` con CPML; 0 si la fila no se escribe."""
    ny, nx = f.shape[1], f.shape[2]
    out[:] = F32(0.0)
    if forward:
        if j < 1 or j > ny - 3:
            return
        for i in range(nx):
            out[i] = (f[k, j + 1, i] - f[k, j, i]) * c1 + (f[k, j + 2, i] - f[k, j - 1, i]) * c2
    else:
        if j < 2 or j > ny - 2:
            return
        for i in range(nx):
            out[i] = (f[k, j, i] - f[k, j - 1, i]) * c1 + (f[k, j + 1, i] - f[k, j - 2, i]) * c2
    if j < p[0]:
        _cpml_row(out, p[6][k, j], p[2][j], p[3][j])
    elif j >= p[1]:
        q = j - p[1]
        _cpml_row(out, p[7][k, q], p[4][q], p[5][q])


@njit(cache=True)
def _dz(f, out, k, j, forward, c1, c2, p):
    """Derivada en z de ``f`` en la fila ``(k, j)`` con CPML; 0 si el plano no se escribe."""
    nz, nx = f.shape[0], f.shape[2]
    out[:] = F32(0.0)
    if forward:
        if k < 1 or k > nz - 3:
            return
        for i in range(nx):
            out[i] = (f[k + 1, j, i] - f[k, j, i]) * c1 + (f[k + 2, j, i] - f[k - 1, j, i]) * c2
    else:
        if k < 2 or k > nz - 2:
            return
        for i in range(nx):
            out[i] = (f[k, j, i] - f[k - 1, j, i]) * c1 + (f[k + 1, j, i] - f[k - 2, j, i]) * c2
    if k < p[0]:
        _cpml_row(out, p[6][k, j], p[2][k], p[3][k])
    elif k >= p[1]:
        q = k - p[1]
        _cpml_row(out, p[7][q, j], p[4][q], p[5][q])


@njit(parallel=True, cache=True)
def velocity_kernel(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, dtbx, dtby, dtbz, c1, c2, has_shear,
                    p_sxx_x, p_sxy_y, p_sxz_z, p_sxy_x, p_syy_y, p_syz_z, p_sxz_x, p_syz_y, p_szz_z):
    """v_i += dt/rho d(s_ij)/dx_j. Sin sólido las cizallas son nulas y sus derivadas no se calculan."""
    nz, ny, nx = vx.shape
    for k in prange(nz):
        a = np.empty(nx, np.float32)
        b = np.empty(nx, np.float32)
        c = np.empty(nx, np.float32)
        for j in range(ny):
            _dx(sxx, a, k, j, True, c1, c2, p_sxx_x)
            if has_shear:
                _dy(sxy, b, k, j, False, c1, c2, p_sxy_y)
                _dz(sxz, c, k, j, False, c1, c2, p_sxz_z)
                for i in range(nx):
                    vx[k, j, i] += ((a[i] + b[i]) + c[i]) * dtbx[k, j, i]
                _dx(sxy, a, k, j, False, c1, c2, p_sxy_x)
                _dy(syy, b, k, j, True, c1, c2, p_syy_y)
                _dz(syz, c, k, j, False, c1, c2, p_syz_z)
                for i in range(nx):
                    vy[k, j, i] += ((a[i] + b[i]) + c[i]) * dtby[k, j, i]
                _dx(sxz, a, k, j, False, c1, c2, p_sxz_x)
                _dy(syz, b, k, j, False, c1, c2, p_syz_y)
                _dz(szz, c, k, j, True, c1, c2, p_szz_z)
                for i in range(nx):
                    vz[k, j, i] += ((a[i] + b[i]) + c[i]) * dtbz[k, j, i]
            else:
                for i in range(nx):
                    vx[k, j, i] += a[i] * dtbx[k, j, i]
                _dy(syy, b, k, j, True, c1, c2, p_syy_y)
                for i in range(nx):
                    vy[k, j, i] += b[i] * dtby[k, j, i]
                _dz(szz, c, k, j, True, c1, c2, p_szz_z)
                for i in range(nx):
                    vz[k, j, i] += c[i] * dtbz[k, j, i]


@njit(cache=True, inline="always")
def _relax(stress, r_arr, m, k, j, i, forcing, a, b):
    """Memoria ``m`` por Crank-Nicolson: ``stress + (r_new + r_old)/2`` y guarda ``r_new = a r - b F``.

    Los escalares de memoria son float64 (como en NumPy): se multiplica en doble
    precisión y se redondea al guardar.
    """
    r = r_arr[m, k, j, i]
    stress = stress + F32(r * (0.5 * (1.0 + a)))
    stress = stress - F32(forcing * (0.5 * b))
    r_arr[m, k, j, i] = F32(r * a) - F32(forcing * b)
    return stress


@njit(cache=True)
def _shear_row(stress, r_arr, ea, eb, coef, coef_t, mem_a, mem_b, k, j):
    """Una componente de cizalla en la fila ``(k, j)``: ``s += dt mu (ea + eb)`` y sus memorias."""
    n_mech = mem_a.shape[0]
    for i in range(ea.shape[0]):
        e = ea[i] + eb[i]
        mu = coef[k, j, i]
        s = stress[k, j, i] + mu * e
        # Sin cizalla (fluido) los coeficientes de memoria son nulos y la memoria no sale de cero
        if mu != F32(0.0):
            for m in range(n_mech):
                s = _relax(s, r_arr, m, k, j, i, coef_t[m, k, j, i] * e, mem_a[m], mem_b[m])
        stress[k, j, i] = s


@njit(parallel=True, cache=True)
def stress_kernel(
    vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, rxx, ryy, rzz, rxy, rxz, ryz,
    dt_pi, dt_mu2, dt_pi_t, dt_mu2_t,
    dt_mu_xy, dt_mu_xz, dt_mu_yz, dt_mu_t_xy, dt_mu_t_xz, dt_mu_t_yz,
    aniso, dt_c11, dt_ct, dt_c12, dt_c23, dt_c11_t, dt_ct_t, dt_c12_t, dt_c23_t,
    mem_a, mem_b, has_shear, c1, c2,
    p_exx, p_eyy, p_ezz, p_exy_a, p_exy_b, p_exz_a, p_exz_b, p_eyz_a, p_eyz_b,
):  # fmt: skip
    """Deformaciones, tensiones y memorias viscoelásticas (Crank-Nicolson).

    Las memorias ``r**`` y los coeficientes ``*_t`` son pilas ``(n_mech, nz,
    ny, nx)``; con ``n_mech = 0`` no se leen. Con ``aniso`` las tensiones
    normales usan las rigideces transversalmente isótropas (eje x) en lugar de
    ``dt_pi`` y ``dt_mu2``.
    """
    nz, ny, nx = vx.shape
    n_mech = mem_a.shape[0]
    for k in prange(nz):
        e1 = np.empty(nx, np.float32)
        e2 = np.empty(nx, np.float32)
        e3 = np.empty(nx, np.float32)
        ra = np.empty(nx, np.float32)
        rb = np.empty(nx, np.float32)
        for j in range(ny):
            _dx(vx, e1, k, j, False, c1, c2, p_exx)
            _dy(vy, e2, k, j, False, c1, c2, p_eyy)
            _dz(vz, e3, k, j, False, c1, c2, p_ezz)
            if aniso:
                for i in range(nx):
                    # siempre en el orden exx, eyy, ezz (ver ``ElasticSolver3D._step_numpy``)
                    x1, x2, x3 = e1[i], e2[i], e3[i]
                    c11, ct, c12, c23 = dt_c11[k, j, i], dt_ct[k, j, i], dt_c12[k, j, i], dt_c23[k, j, i]
                    sx = ((sxx[k, j, i] + c11 * x1) + c12 * x2) + c12 * x3
                    sy = ((syy[k, j, i] + c12 * x1) + ct * x2) + c23 * x3
                    sz = ((szz[k, j, i] + c12 * x1) + c23 * x2) + ct * x3
                    for m in range(n_mech):
                        a, b = mem_a[m], mem_b[m]
                        c11t, ctt = dt_c11_t[m, k, j, i], dt_ct_t[m, k, j, i]
                        c12t, c23t = dt_c12_t[m, k, j, i], dt_c23_t[m, k, j, i]
                        f_xx = (c11t * x1 + c12t * x2) + c12t * x3
                        f_yy = (c12t * x1 + ctt * x2) + c23t * x3
                        f_zz = (c12t * x1 + c23t * x2) + ctt * x3
                        sx = _relax(sx, rxx, m, k, j, i, f_xx, a, b)
                        sy = _relax(sy, ryy, m, k, j, i, f_yy, a, b)
                        sz = _relax(sz, rzz, m, k, j, i, f_zz, a, b)
                    sxx[k, j, i] = sx
                    syy[k, j, i] = sy
                    szz[k, j, i] = sz
            else:
                for i in range(nx):
                    x1, x2, x3 = e1[i], e2[i], e3[i]
                    div = (x1 + x2) + x3
                    a1 = dt_pi[k, j, i] * div
                    mu2 = dt_mu2[k, j, i]
                    sx = (sxx[k, j, i] + a1) - mu2 * (x2 + x3)
                    sy = (syy[k, j, i] + a1) - mu2 * (x1 + x3)
                    sz = (szz[k, j, i] + a1) - mu2 * (x1 + x2)
                    fluid = mu2 == F32(0.0)
                    for m in range(n_mech):
                        a, b = mem_a[m], mem_b[m]
                        vol = dt_pi_t[m, k, j, i] * div
                        if fluid:
                            f_xx = vol
                            f_yy = vol
                            f_zz = vol
                        else:
                            mt = dt_mu2_t[m, k, j, i]
                            f_xx = vol - (x2 + x3) * mt
                            f_yy = vol - (x1 + x3) * mt
                            f_zz = vol - (x1 + x2) * mt
                        sx = _relax(sx, rxx, m, k, j, i, f_xx, a, b)
                        sy = _relax(sy, ryy, m, k, j, i, f_yy, a, b)
                        sz = _relax(sz, rzz, m, k, j, i, f_zz, a, b)
                    sxx[k, j, i] = sx
                    syy[k, j, i] = sy
                    szz[k, j, i] = sz
            if has_shear:
                _dy(vx, ra, k, j, True, c1, c2, p_exy_a)
                _dx(vy, rb, k, j, True, c1, c2, p_exy_b)
                _shear_row(sxy, rxy, ra, rb, dt_mu_xy, dt_mu_t_xy, mem_a, mem_b, k, j)
                _dz(vx, ra, k, j, True, c1, c2, p_exz_a)
                _dx(vz, rb, k, j, True, c1, c2, p_exz_b)
                _shear_row(sxz, rxz, ra, rb, dt_mu_xz, dt_mu_t_xz, mem_a, mem_b, k, j)
                _dz(vy, ra, k, j, True, c1, c2, p_eyz_a)
                _dy(vz, rb, k, j, True, c1, c2, p_eyz_b)
                _shear_row(syz, ryz, ra, rb, dt_mu_yz, dt_mu_t_yz, mem_a, mem_b, k, j)


@njit(parallel=True, cache=True)
def slip_velocity(out, coef, src, t, S, W):
    """``out[t] += coef[t] * sum(W * src[S])`` (arrays planos; cada destino ``t`` una sola vez)."""
    for n in prange(t.size):
        acc = F32(0.0)
        for m in range(S.shape[1]):
            acc += W[n, m] * src[S[n, m]]
        out[t[n]] += coef[t[n]] * acc


@njit(parallel=True, cache=True)
def slip_stress(stress, coef, coef_t, r, mem_b, src, t, S, W):
    """Corrección de una deformación de cizalla en la tensión y en sus memorias (Crank-Nicolson, lineal)."""
    n_mech = mem_b.shape[0]
    for n in prange(t.size):
        acc = F32(0.0)
        for m in range(S.shape[1]):
            acc += W[n, m] * src[S[n, m]]
        tt = t[n]
        s = stress[tt] + coef[tt] * acc
        for m in range(n_mech):
            forcing = coef_t[m, tt] * acc
            s = s - F32(forcing * (0.5 * mem_b[m]))
            r[m, tt] = r[m, tt] - F32(forcing * mem_b[m])
        stress[tt] = s
