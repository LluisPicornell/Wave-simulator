"""Kernel Numba del Monte Carlo de fotones X (backend opcional de ``xray_scatter``).

Misma física que el backend NumPy (``xray_scatter._run_numpy``), un fotón cada vez. Transporte por
*delta tracking* (Woodcock): se muestrea la distancia con la atenuación máxima del medio a esa energía
y cada colisión es real con probabilidad ``mu(voxel)/mu_max``; así no hace falta recorrer vóxeles.
Los dos backends usan generadores distintos: coinciden en sentido estadístico, no fotón a fotón.

Coordenadas 3D ``(x, y, z)``: ``(x, z)`` es el plano de la sección (malla 2D) e ``y`` la dirección
extruida, infinita. Vector ``gf`` de geometría (float): ver ``xray_scatter._geometry_vector``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MEC2 = 510.99895
HC = 12.398419843
# Contadores en ``acc``: 0 N emitidos, 1 E emitida, 2 n fotoeléctrico, 3 E fotoeléctrico, 4 E del electrón
# Compton, 5 n bajo el corte, 6 E bajo el corte, 7 n fuera del detector, 8 E fuera del detector,
# 9 n rechazados por la rejilla, 10 E rechazada por la rejilla, 11 n colisiones reales, 12 n virtuales,
# 13 n Compton, 14 n Rayleigh, 15 suma de cos θ de Compton, 16 suma de cos θ de Rayleigh
NACC = 17


@njit(cache=True)
def _interp_uniform(arr, f):
    i = int(f)
    i = min(max(i, 0), arr.shape[0] - 2)
    w = f - i
    return arr[i] * (1.0 - w) + arr[i + 1] * w


@njit(cache=True)
def _rotate(ux, uy, uz, cost, phi):
    sint = np.sqrt(max(1.0 - cost * cost, 0.0))
    cp, sp = np.cos(phi), np.sin(phi)
    if abs(uz) > 0.99999:
        s = 1.0 if uz > 0 else -1.0
        return sint * cp, sint * sp, s * cost
    tmp = np.sqrt(1.0 - uz * uz)
    nx = sint * (ux * uz * cp - uy * sp) / tmp + ux * cost
    ny = sint * (uy * uz * cp + ux * sp) / tmp + uy * cost
    nz = -sint * cp * tmp + uz * cost
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / n, ny / n, nz / n


@njit(cache=True)
def _exit_distance(x, z, ux, uz, bx0, bx1, bz0, bz1):
    """Parámetro 3D ``t`` al que la trayectoria sale de la caja (infinito si no sale en el plano)."""
    t = 1e300
    if ux > 1e-14:
        t = min(t, (bx1 - x) / ux)
    elif ux < -1e-14:
        t = min(t, (bx0 - x) / ux)
    if uz > 1e-14:
        t = min(t, (bz1 - z) / uz)
    elif uz < -1e-14:
        t = min(t, (bz0 - z) / uz)
    return t


@njit(cache=True)
def run_photons(seed, nphot, mat, scale, grid_f, e0, mu_pe, mu_coh, mu_inc, mu_max, x_step, s_over_z, ray_e0, ray_de,
                ray_cdf, spec_e, spec_cdf, gf, kind, n_det, prim, scat, scat_comp, scat_ray, eprim, escat, acc):
    """Lanza ``nphot`` fotones y acumula en ``prim``/``scat``/... (n_det) y ``acc``.

    ``kind``: 0 abanico con detector curvo, 1 abanico con detector plano, 2 haz paralelo.
    """
    np.random.seed(seed)
    nz, nx = mat.shape
    x0, z0, dx = grid_f[0], grid_f[1], grid_f[2]
    bx0, bx1 = x0 - 0.5 * dx, x0 + (nx - 0.5) * dx
    bz0, bz1 = z0 - 0.5 * dx, z0 + (nz - 0.5) * dx
    cx, cz = grid_f[3], grid_f[4]
    dxr, dzr, nxr, nzr = gf[0], gf[1], gf[2], gf[3]  # eje del rayo central y del detector
    sid, sdd, spacing, half_t, half_h, cos_acc = gf[4], gf[5], gf[6], gf[7], gf[8], gf[9]
    rad = gf[10]
    e_min = e0
    nspec = spec_e.shape[0]
    nre = ray_cdf.shape[1]
    ncos = ray_cdf.shape[2]
    sx, sz = cx - sid * dxr, cz - sid * dzr  # fuente (abanico) o plano de partida (paralelo)
    if kind == 2:
        sx, sz = cx - rad * dxr, cz - rad * dzr
        sdd_eff = 2.0 * rad
    else:
        sdd_eff = sdd
    for _ in range(nphot):
        # ---- energía y emisión
        r = np.random.random()
        k = np.searchsorted(spec_cdf, r)
        if k >= nspec:
            k = nspec - 1
        en = spec_e[k]
        acc[0] += 1.0
        acc[1] += en
        y_iso = (2.0 * np.random.random() - 1.0) * half_t
        u_pos = (np.random.random() - 0.5) * n_det * spacing
        if kind == 2:
            x = sx + u_pos * nxr
            z = sz + u_pos * nzr
            y = y_iso
            ux, uy, uz = dxr, 0.0, dzr
        else:
            if kind == 0:
                gam = u_pos / sdd
            else:
                gam = np.arctan(u_pos / sdd)
            ph_x = np.cos(gam) * dxr + np.sin(gam) * nxr
            ph_z = np.cos(gam) * dzr + np.sin(gam) * nzr
            tanphi = y_iso * np.cos(gam) / sid
            nrm = np.sqrt(1.0 + tanphi * tanphi)
            ux, uy, uz = ph_x / nrm, tanphi / nrm, ph_z / nrm
            x, z, y = sx, sz, 0.0
        n_int = 0
        had_compton = False
        alive = True
        # ---- hasta la caja
        inside = x >= bx0 and x <= bx1 and z >= bz0 and z <= bz1
        if not inside:
            tmin, tmax = 0.0, 1e300
            ok = True
            if abs(ux) > 1e-14:
                ta, tb = (bx0 - x) / ux, (bx1 - x) / ux
                tmin = max(tmin, min(ta, tb))
                tmax = min(tmax, max(ta, tb))
            elif x < bx0 or x > bx1:
                ok = False
            if abs(uz) > 1e-14:
                ta, tb = (bz0 - z) / uz, (bz1 - z) / uz
                tmin = max(tmin, min(ta, tb))
                tmax = min(tmax, max(ta, tb))
            elif z < bz0 or z > bz1:
                ok = False
            if ok and tmax > tmin:
                x += (tmin + 1e-9 * dx) * ux
                y += (tmin + 1e-9 * dx) * uy
                z += (tmin + 1e-9 * dx) * uz
                inside = True
        # ---- transporte
        while inside and alive:
            f = en - e0  # índice fraccionario en la rejilla de 1 keV
            mmax = _interp_uniform(mu_max, f)
            s = -np.log(1.0 - np.random.random()) / mmax
            t_exit = _exit_distance(x, z, ux, uz, bx0, bx1, bz0, bz1)
            if s >= t_exit:
                x += t_exit * ux
                y += t_exit * uy
                z += t_exit * uz
                inside = False
                break
            x += s * ux
            y += s * uy
            z += s * uz
            i = int((x - bx0) / dx)
            j = int((z - bz0) / dx)
            if i < 0 or i >= nx or j < 0 or j >= nz:
                inside = False
                break
            m = mat[j, i]
            sc = scale[j, i]
            a_pe = sc * _interp_uniform(mu_pe[m], f)
            a_coh = sc * _interp_uniform(mu_coh[m], f)
            a_inc = sc * _interp_uniform(mu_inc[m], f)
            a_tot = a_pe + a_coh + a_inc
            if np.random.random() * mmax > a_tot:
                acc[12] += 1.0
                continue
            acc[11] += 1.0
            n_int += 1
            r = np.random.random() * a_tot
            if r < a_pe:
                acc[2] += 1.0
                acc[3] += en
                alive = False
                break
            if r < a_pe + a_coh:
                # Rayleigh: cos θ del CDF tabulado a la energía más cercana
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
                acc[14] += 1.0
                acc[16] += cost
                ux, uy, uz = _rotate(ux, uy, uz, cost, 2.0 * np.pi * np.random.random())
            else:
                had_compton = True
                kk = en / MEC2
                emin = 1.0 / (1.0 + 2.0 * kk)
                a1 = -np.log(emin)
                a2 = 0.5 * (1.0 - emin * emin)
                while True:
                    if np.random.random() * (a1 + a2) < a1:
                        eps = emin ** np.random.random()
                    else:
                        eps = np.sqrt(emin * emin + (1.0 - emin * emin) * np.random.random())
                    cost = 1.0 - (1.0 / eps - 1.0) / kk
                    sin2 = 1.0 - cost * cost
                    tacc = 1.0 - eps * sin2 / (1.0 + eps * eps)
                    xq = en / HC * np.sqrt(max(0.5 * (1.0 - cost), 0.0))
                    szv = _interp_uniform(s_over_z[m], xq / x_step)
                    if np.random.random() < tacc * szv:
                        break
                acc[4] += en * (1.0 - eps)
                acc[13] += 1.0
                acc[15] += cost
                en = en * eps
                ux, uy, uz = _rotate(ux, uy, uz, cost, 2.0 * np.pi * np.random.random())
                if en < e_min + 1e-9:
                    acc[5] += 1.0
                    acc[6] += en
                    alive = False
                    break
        if not alive:
            continue
        # ---- detección: la trayectoria recta desde (x, y, z) hasta la superficie del detector
        h2 = ux * ux + uz * uz
        a, b = x - sx, z - sz
        hit = False
        jdet = 0
        t = 0.0
        if h2 > 1e-24:
            if kind == 0:
                bb = a * ux + b * uz
                cc = a * a + b * b - sdd * sdd
                disc = bb * bb - h2 * cc
                if disc >= 0.0:
                    t = (-bb + np.sqrt(disc)) / h2
                    ha, hb = a + t * ux, b + t * uz
                    along = ha * dxr + hb * dzr
                    if along > 0.0 and t >= 0.0:
                        lat = ha * nxr + hb * nzr
                        gam = np.arctan2(lat, along)
                        jdet = int(np.floor(gam / (spacing / sdd) + 0.5 * n_det))
                        hit = True
            else:
                den = ux * dxr + uz * dzr
                if den > 1e-12:
                    t = (sdd_eff - (a * dxr + b * dzr)) / den
                    if t >= 0.0:
                        lat = (a + t * ux) * nxr + (b + t * uz) * nzr
                        jdet = int(np.floor(lat / spacing + 0.5 * n_det))
                        hit = True
        if hit:
            yh = y + t * uy
            hit = jdet >= 0 and jdet < n_det and abs(yh) <= half_h
        if not hit:
            acc[7] += 1.0
            acc[8] += en
            continue
        if cos_acc > 0.0 and n_int > 0:
            if kind == 2:
                ca = ux * dxr + uz * dzr
            else:
                vx, vy, vz = a + t * ux, y + t * uy, b + t * uz
                ca = (ux * vx + uy * vy + uz * vz) / np.sqrt(vx * vx + vy * vy + vz * vz)
            if ca < cos_acc:
                acc[9] += 1.0
                acc[10] += en
                continue
        if n_int == 0:
            prim[jdet] += 1.0
            eprim[jdet] += en
        else:
            scat[jdet] += 1.0
            escat[jdet] += en
            if had_compton:
                scat_comp[jdet] += 1.0
            else:
                scat_ray[jdet] += 1.0
