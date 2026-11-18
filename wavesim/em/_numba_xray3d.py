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
def backproject_fdk(q, vol, x0, y0, z0, dx, c, d, u, v, sid, sdd, du, dv, cone, weight, vmag_fixed):
    """Suma en ``vol (nz, ny, nx)`` la retroproyección de la vista ``q (n_v, n_u)`` ya filtrada.

    ``c``: isocentro (x, y, z) en metros; ``d``, ``u``, ``v``: rayo central y ejes del detector (x, y, z).
    Cono: el vóxel a profundidad ``r`` (a lo largo de ``d``) se proyecta en ``(u, v)·sdd/(sid + r)`` y pesa
    ``(sid/(sid + r))²``. Haz paralelo (``cone = False``): proyección ortogonal, peso 1. ``vmag_fixed`` > 0
    fija la magnificación en v (apilado de cortes de abanico: la fila no depende de la profundidad).
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
                fv = w * (vmag_fixed if vmag_fixed > 0.0 else mag) / dv + 0.5 * (nv - 1)
                iu = int(np.floor(fu))
                iv = int(np.floor(fv))
                if iu < 0 or iu >= nu - 1 or iv < 0 or iv >= nv - 1:
                    continue
                au, av = fu - iu, fv - iv
                val = ((1 - av) * ((1 - au) * q[iv, iu] + au * q[iv, iu + 1])
                       + av * ((1 - au) * q[iv + 1, iu] + au * q[iv + 1, iu + 1]))
                vol[k, j, i] += weight * f * val


# --------------------------------------------------------------------------------------
# Monte Carlo de fotones en 3D (scatter): generalización de ``_numba_xray.run_photons``
# --------------------------------------------------------------------------------------
from ._numba_xray import HC, MEC2, _interp_uniform, _rotate  # noqa: E402

# Contadores en ``acc``: 0 N emitidos, 1 peso de señal emitido, 2 n fotoeléctrico, 3 n bajo el corte de energía,
# 4 n fuera del detector, 5 n colisiones reales, 6 n virtuales, 7 n Compton, 8 n Rayleigh
NACC3 = 9


@njit(cache=True)
def _exit3(x, y, z, ux, uy, uz, b):
    """Parámetro ``t`` al que la trayectoria sale de la caja ``b = (x0, x1, y0, y1, z0, z1)``."""
    t = 1e300
    if ux > 1e-14:
        t = min(t, (b[1] - x) / ux)
    elif ux < -1e-14:
        t = min(t, (b[0] - x) / ux)
    if uy > 1e-14:
        t = min(t, (b[3] - y) / uy)
    elif uy < -1e-14:
        t = min(t, (b[2] - y) / uy)
    if uz > 1e-14:
        t = min(t, (b[5] - z) / uz)
    elif uz < -1e-14:
        t = min(t, (b[4] - z) / uz)
    return t


@njit(cache=True)
def run_photons3d(seed, nphot, mat, grid_f, e0, mu_pe, mu_coh, mu_inc, mu_max, x_step, s_over_z, ray_e0, ray_de,
                  ray_cdf, spec_e, spec_cdf, det_w, gv, prim_n, scat_n, prim_w, scat_w, acc):
    """Lanza ``nphot`` fotones de una vista y acumula en los mapas de bins ``(nbv, nbu)``.

    ``grid_f = (x0, y0, z0, dx)``; ``mat (nz, ny, nx)``; ``det_w``: peso de señal por energía (rejilla de 1 keV
    desde ``e0``). ``gv`` (geometría de la vista): 0-2 isocentro, 3-5 rayo central ``d``, 6-8 eje ``u``, 9-11
    eje ``v``, 12 ``sid``, 13 ``sdd``, 14-15 semianchuras del detector en u y v, 16-17 número de bins (u, v),
    18 radio del cilindro circunscrito (haz paralelo), 19 = 1 si es haz paralelo.
    """
    np.random.seed(seed)
    nz, ny, nx = mat.shape
    x0, y0, z0, dx = grid_f[0], grid_f[1], grid_f[2], grid_f[3]
    box = np.array([x0 - 0.5 * dx, x0 + (nx - 0.5) * dx, y0 - 0.5 * dx, y0 + (ny - 0.5) * dx, z0 - 0.5 * dx,
                    z0 + (nz - 0.5) * dx])
    cx, cy, cz = gv[0], gv[1], gv[2]
    dxr, dyr, dzr = gv[3], gv[4], gv[5]
    uxr, uyr, uzr = gv[6], gv[7], gv[8]
    vxr, vyr, vzr = gv[9], gv[10], gv[11]
    sid, sdd, half_u, half_v = gv[12], gv[13], gv[14], gv[15]
    nbu, nbv = int(gv[16]), int(gv[17])
    rad = gv[18]
    parallel = gv[19] > 0.5
    bw_u, bw_v = 2.0 * half_u / nbu, 2.0 * half_v / nbv
    sx, sy, sz = cx - sid * dxr, cy - sid * dyr, cz - sid * dzr
    # centro del detector (cono) o plano de detección (paralelo)
    if parallel:
        pcx, pcy, pcz = cx + rad * dxr, cy + rad * dyr, cz + rad * dzr
    else:
        pcx, pcy, pcz = cx + (sdd - sid) * dxr, cy + (sdd - sid) * dyr, cz + (sdd - sid) * dzr
    e_min = e0
    nspec = spec_e.shape[0]
    nre = ray_cdf.shape[1]
    ncos = ray_cdf.shape[2]
    for _ in range(nphot):
        r = np.random.random()
        k = np.searchsorted(spec_cdf, r)
        if k >= nspec:
            k = nspec - 1
        en = spec_e[k]
        acc[0] += 1.0
        acc[1] += _interp_uniform(det_w, en - e0)
        ue = (2.0 * np.random.random() - 1.0) * half_u
        ve = (2.0 * np.random.random() - 1.0) * half_v
        if parallel:
            x = cx + ue * uxr + ve * vxr - rad * dxr
            y = cy + ue * uyr + ve * vyr - rad * dyr
            z = cz + ue * uzr + ve * vzr - rad * dzr
            ux, uy, uz = dxr, dyr, dzr
        else:
            px = pcx + ue * uxr + ve * vxr
            py = pcy + ue * uyr + ve * vyr
            pz = pcz + ue * uzr + ve * vzr
            ux, uy, uz = px - sx, py - sy, pz - sz
            nrm = np.sqrt(ux * ux + uy * uy + uz * uz)
            ux, uy, uz = ux / nrm, uy / nrm, uz / nrm
            x, y, z = sx, sy, sz
        n_int = 0
        alive = True
        # ---- hasta la caja (el origen está siempre fuera)
        tmin, tmax = 0.0, 1e300
        ok = True
        for a in range(3):
            o = x if a == 0 else (y if a == 1 else z)
            u_ = ux if a == 0 else (uy if a == 1 else uz)
            lo, hi = box[2 * a], box[2 * a + 1]
            if abs(u_) > 1e-14:
                ta, tb = (lo - o) / u_, (hi - o) / u_
                tmin = max(tmin, min(ta, tb))
                tmax = min(tmax, max(ta, tb))
            elif o < lo or o > hi:
                ok = False
        inside = ok and tmax > tmin
        if inside:
            x += (tmin + 1e-9 * dx) * ux
            y += (tmin + 1e-9 * dx) * uy
            z += (tmin + 1e-9 * dx) * uz
        # ---- transporte por delta tracking
        while inside and alive:
            f = en - e0
            mmax = _interp_uniform(mu_max, f)
            s = -np.log(1.0 - np.random.random()) / mmax
            t_exit = _exit3(x, y, z, ux, uy, uz, box)
            if s >= t_exit:
                x += t_exit * ux
                y += t_exit * uy
                z += t_exit * uz
                break
            x += s * ux
            y += s * uy
            z += s * uz
            i = int((x - box[0]) / dx)
            j = int((y - box[2]) / dx)
            kk = int((z - box[4]) / dx)
            if i < 0 or i >= nx or j < 0 or j >= ny or kk < 0 or kk >= nz:
                break
            m = mat[kk, j, i]
            a_pe = _interp_uniform(mu_pe[m], f)
            a_coh = _interp_uniform(mu_coh[m], f)
            a_inc = _interp_uniform(mu_inc[m], f)
            a_tot = a_pe + a_coh + a_inc
            if np.random.random() * mmax > a_tot:
                acc[6] += 1.0
                continue
            acc[5] += 1.0
            n_int += 1
            r = np.random.random() * a_tot
            if r < a_pe:
                acc[2] += 1.0
                alive = False
                break
            if r < a_pe + a_coh:
                ir = int((en - ray_e0) / ray_de + 0.5)
                ir = min(max(ir, 0), nre - 1)
                row = ray_cdf[m, ir]
                q = np.random.random()
                idx = np.searchsorted(row, q)
                if idx <= 0:
                    cost = -1.0
                elif idx >= ncos:
                    cost = 1.0
                else:
                    den = row[idx] - row[idx - 1]
                    frac = (q - row[idx - 1]) / den if den > 0 else 0.0
                    cost = -1.0 + 2.0 * (idx - 1 + frac) / (ncos - 1)
                acc[8] += 1.0
                ux, uy, uz = _rotate(ux, uy, uz, cost, 2.0 * np.pi * np.random.random())
            else:
                kq = en / MEC2
                emin = 1.0 / (1.0 + 2.0 * kq)
                a1 = -np.log(emin)
                a2 = 0.5 * (1.0 - emin * emin)
                while True:
                    if np.random.random() * (a1 + a2) < a1:
                        eps = emin ** np.random.random()
                    else:
                        eps = np.sqrt(emin * emin + (1.0 - emin * emin) * np.random.random())
                    cost = 1.0 - (1.0 / eps - 1.0) / kq
                    sin2 = 1.0 - cost * cost
                    tacc = 1.0 - eps * sin2 / (1.0 + eps * eps)
                    xq = en / HC * np.sqrt(max(0.5 * (1.0 - cost), 0.0))
                    szv = _interp_uniform(s_over_z[m], xq / x_step)
                    if np.random.random() < tacc * szv:
                        break
                acc[7] += 1.0
                en = en * eps
                ux, uy, uz = _rotate(ux, uy, uz, cost, 2.0 * np.pi * np.random.random())
                if en < e_min + 1e-9:
                    acc[3] += 1.0
                    alive = False
                    break
        if not alive:
            continue
        # ---- detección: recta desde el punto de salida hasta el plano del detector
        den = ux * dxr + uy * dyr + uz * dzr
        hit = False
        iu = 0
        iv = 0
        if den > 1e-12:
            if parallel:
                t = (rad - ((x - cx) * dxr + (y - cy) * dyr + (z - cz) * dzr)) / den
            else:
                t = (sdd - ((x - sx) * dxr + (y - sy) * dyr + (z - sz) * dzr)) / den
            if t >= 0.0:
                hx, hy, hz = x + t * ux - pcx, y + t * uy - pcy, z + t * uz - pcz
                lu = hx * uxr + hy * uyr + hz * uzr
                lv = hx * vxr + hy * vyr + hz * vzr
                iu = int(np.floor((lu + half_u) / bw_u))
                iv = int(np.floor((lv + half_v) / bw_v))
                hit = lu > -half_u and lv > -half_v and iu < nbu and iv < nbv
        if not hit:
            acc[4] += 1.0
            continue
        w = _interp_uniform(det_w, en - e0)
        if n_int == 0:
            prim_n[iv, iu] += 1.0
            prim_w[iv, iu] += w
        else:
            scat_n[iv, iu] += 1.0
            scat_w[iv, iu] += w
