"""Kernels Numba del proyector 3D de rayos X (backend opcional de ``xray3d``).

Trabajan en **coordenadas de índice**: el centro de la celda ``[k, j, i]`` está en ``(i, j, k)`` y la celda
ocupa ``[i-1/2, i+1/2]`` en cada eje; las longitudes salen en unidades de celda y ``xray3d`` las multiplica
por ``dx``. Un rayo es ``o + t·d`` con ``d`` unitario.

- ``forward``: integrales de línea exactas del volumen constante a trozos (Siddon, recorrido de la rejilla
  celda a celda al estilo Amanatides-Woo: los mismos tramos que Siddon sin ordenar los cruces de planos).
  Recorre varios mapas a la vez con un único recorrido de rayo.
- ``adjoint``: su transpuesta exacta (cada celda recibe ``valor_del_rayo × longitud_del_tramo``).
- ``backproject_fdk``: retroproyección dirigida por vóxel (proyecta cada vóxel al detector) con el peso
  de Feldkamp; paraleliza sin carreras porque cada vóxel es de un hilo.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

_EPS = 1e-12


@njit(cache=True)
def _clip_box(o0, o1, o2, d0, d1, d2, n0, n1, n2):
    """Intervalo ``[tmin, tmax]`` del rayo dentro de la caja ``[-1/2, n-1/2]³`` (``tmax <= tmin`` si no la cruza)."""
    tmin, tmax = -1e300, 1e300
    for a in range(3):
        o = o0 if a == 0 else (o1 if a == 1 else o2)
        d = d0 if a == 0 else (d1 if a == 1 else d2)
        n = n0 if a == 0 else (n1 if a == 1 else n2)
        lo, hi = -0.5, n - 0.5
        if abs(d) < _EPS:
            if o < lo or o > hi:
                return 1.0, 0.0
        else:
            ta, tb = (lo - o) / d, (hi - o) / d
            if ta > tb:
                ta, tb = tb, ta
            tmin = max(tmin, ta)
            tmax = min(tmax, tb)
    return tmin, tmax


@njit(cache=True)
def _walk(o, d, n0, n1, n2, out_idx, out_len):
    """Rellena ``out_idx``/``out_len`` con las celdas planas y longitudes que atraviesa el rayo; devuelve su número.

    Los índices son ``(k * ny + j) * nx + i`` (``n0 = nx, n1 = ny, n2 = nz``).
    """
    tmin, tmax = _clip_box(o[0], o[1], o[2], d[0], d[1], d[2], n0, n1, n2)
    if tmax <= tmin:
        return 0
    cnt = 0
    c = np.empty(3, np.int64)
    step = np.empty(3, np.int64)
    tn = np.empty(3)
    td = np.empty(3)
    ns = (n0, n1, n2)
    tm = tmin + 1e-9
    for a in range(3):
        p = o[a] + tm * d[a]
        c[a] = min(max(int(np.floor(p + 0.5)), 0), ns[a] - 1)
        if d[a] > _EPS:
            step[a] = 1
            td[a] = 1.0 / d[a]
            tn[a] = (c[a] + 0.5 - o[a]) / d[a]
        elif d[a] < -_EPS:
            step[a] = -1
            td[a] = -1.0 / d[a]
            tn[a] = (c[a] - 0.5 - o[a]) / d[a]
        else:
            step[a] = 0
            td[a] = 1e300
            tn[a] = 1e300
    t = tmin
    while t < tmax:
        a = 0
        if tn[1] < tn[a]:
            a = 1
        if tn[2] < tn[a]:
            a = 2
        t_next = min(tn[a], tmax)
        seg = t_next - t
        if seg > 0.0:
            out_idx[cnt] = (c[2] * n1 + c[1]) * n0 + c[0]
            out_len[cnt] = seg
            cnt += 1
        t = t_next
        c[a] += step[a]
        tn[a] += td[a]
        if c[a] < 0 or c[a] >= ns[a]:
            break
    return cnt


@njit(parallel=True, cache=True)
def forward(vol, org, dirs, out):
    """``out[m, r] = sum_celdas vol[m] · longitud`` para cada rayo ``r`` (``vol``: ``(m, nz, ny, nx)``)."""
    nm, nz, ny, nx = vol.shape
    flat = vol.reshape(nm, nz * ny * nx)
    nr = org.shape[0]
    maxseg = nx + ny + nz + 4
    for r in prange(nr):
        idx = np.empty(maxseg, np.int64)
        ln = np.empty(maxseg)
        cnt = _walk(org[r], dirs[r], nx, ny, nz, idx, ln)
        for m in range(nm):
            s = 0.0
            for q in range(cnt):
                s += flat[m, idx[q]] * ln[q]
            out[m, r] = s


@njit(cache=True)
def adjoint(vals, org, dirs, acc, nx, ny, nz):
    """Suma en ``acc`` (plano ``nz*ny*nx``) ``vals[r] · longitud`` por celda: transpuesta exacta de ``forward``."""
    maxseg = nx + ny + nz + 4
    idx = np.empty(maxseg, np.int64)
    ln = np.empty(maxseg)
    for r in range(org.shape[0]):
        v = vals[r]
        if v == 0.0:
            continue
        cnt = _walk(org[r], dirs[r], nx, ny, nz, idx, ln)
        for q in range(cnt):
            acc[idx[q]] += v * ln[q]


@njit(parallel=True, cache=True)
def backproject_fdk(q, vol, x0, y0, z0, dx, c, d, u, v, sid, sdd, du, dv, cone, weight):
    """Suma en ``vol (nz, ny, nx)`` la retroproyección de la vista ``q (n_v, n_u)`` ya filtrada.

    ``c``: isocentro (x, y, z) en metros; ``d``, ``u``, ``v``: rayo central y ejes del detector (x, y, z).
    Cono: el vóxel a profundidad ``r`` (a lo largo de ``d``) se proyecta en ``(u, v)·sdd/(sid + r)`` y pesa
    ``(sid/(sid + r))²``. Haz paralelo (``cone = False``): proyección ortogonal, peso 1.
    """
    nz, ny, nx = vol.shape
    nv, nu = q.shape
    for k in prange(nz):
        pz = z0 + k * dx - c[2]
        for j in range(ny):
            py = y0 + j * dx - c[1]
            for i in range(nx):
                px = x0 + i * dx - c[0]
                r = px * d[0] + py * d[1] + pz * d[2]
                t = px * u[0] + py * u[1] + pz * u[2]
                w = px * v[0] + py * v[1] + pz * v[2]
                if cone:
                    den = sid + r
                    if den <= 1e-9:
                        continue
                    mag = sdd / den
                    f = (sid / den) ** 2
                else:
                    mag = 1.0
                    f = 1.0
                fu = t * mag / du + 0.5 * (nu - 1)
                fv = w * mag / dv + 0.5 * (nv - 1)
                iu = int(np.floor(fu))
                iv = int(np.floor(fv))
                if iu < 0 or iu >= nu - 1 or iv < 0 or iv >= nv - 1:
                    continue
                au, av = fu - iu, fv - iv
                val = ((1 - av) * ((1 - au) * q[iv, iu] + au * q[iv, iu + 1])
                       + av * ((1 - au) * q[iv + 1, iu] + au * q[iv + 1, iu + 1]))
                vol[k, j, i] += weight * f * val
