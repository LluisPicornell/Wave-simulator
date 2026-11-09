"""Kernel Numba del Monte Carlo de fotones (backend opcional de ``montecarlo.simulate``).

Misma física que el backend NumPy (``montecarlo._run_batch_numpy``), pero un
fotón cada vez en un bucle escalar: sin vectorizar, Numba lo compila a código
máquina y no hay que arrastrar arrays de estado ni compactarlos. Las dos
implementaciones usan generadores distintos, así que no dan la misma secuencia
de fotones: coinciden sólo en sentido estadístico (los tests lo comprueban).
"""

from __future__ import annotations

import numpy as np
from numba import njit

C0 = 299_792_458.0
WTH = 1e-4  # ruleta rusa por debajo de este peso (como MCML)
CHANCE = 0.1


@njit(cache=True)
def _fresnel(n1, n2, ci):
    """Reflectancia de Fresnel no polarizada y coseno transmitido (``ci`` = coseno de incidencia, >= 0)."""
    if n1 == n2:
        return 0.0, ci
    ci = max(ci, 1e-12)
    st = n1 / n2 * np.sqrt(max(1.0 - ci * ci, 0.0))
    if st >= 1.0:
        return 1.0, 0.0
    ct = np.sqrt(1.0 - st * st)
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    return 0.5 * (rs * rs + rp * rp), ct


@njit(cache=True)
def run_batch(seed, nphot, mua, mus, gmap, nmap, tissue, dx, n_out, periodic, kind, src_iz, src_ix, src_x, src_z,
              dt_bin, flu, absorbed, flu_t, prof_top, prof_bot, prof_side, tof_top, tof_bot, acc):
    """Lanza ``nphot`` fotones y acumula en los arrays de salida (todos float64, se suman in situ).

    ``kind``: 0 haz ancho (columnas ``src_ix``/``src_iz``), 1 haz fino (``src_x`` fija), 2 fuente isótropa en
    ``(src_x, src_z)``. ``acc[0]`` = reflexión especular, ``acc[1]`` = peso transmitido sin
    haberse dispersado nunca (balístico). Los arrays con tiempo tienen tamaño 0 (``flu_t``) o 1
    (``tof_*``) si no se piden.
    """
    np.random.seed(seed)
    nz, nx = mua.shape
    nt = flu_t.shape[0]
    use_t = nt > 0
    ntof = tof_top.shape[0]
    wdx = nx * dx
    for _ in range(nphot):
        # ---- lanzamiento
        uy = 0.0
        ux = 0.0
        if kind == 2:
            ix = min(max(int(src_x / dx), 0), nx - 1)
            iz = min(max(int(src_z / dx), 0), nz - 1)
            x = src_x
            z = src_z
            uz = 2.0 * np.random.random() - 1.0
            ph = 2.0 * np.pi * np.random.random()
            s = np.sqrt(max(1.0 - uz * uz, 0.0))
            ux = s * np.cos(ph)
            uy = s * np.sin(ph)
            w = 1.0
        else:
            k = int(np.random.random() * src_ix.shape[0]) if kind == 0 else 0
            ix = src_ix[k]
            iz = src_iz[k]
            x = (ix + np.random.random()) * dx if kind == 0 else src_x
            z = iz * dx
            uz = 1.0
            rsp, _ct = _fresnel(n_out, nmap[iz, ix], 1.0)
            acc[0] += rsp
            w = 1.0 - rsp
        tau = -np.log(1.0 - np.random.random())
        t = 0.0
        scattered = False
        alive = True
        while alive:
            # el paso libre usa mu_s: la absorción se aplica de forma continua a lo largo del segmento,
            # de modo que la atenuación del haz balístico sigue siendo exp(-mu_t l)
            ms = mus[iz, ix]
            s_ev = tau / ms if ms > 0.0 else 1e300
            # distancia a la siguiente pared de celda
            if ux > 0.0:
                dxb = ((ix + 1) * dx - x) / ux
            elif ux < 0.0:
                dxb = (ix * dx - x) / ux
            else:
                dxb = 1e300
            if uz > 0.0:
                dzb = ((iz + 1) * dx - z) / uz
            elif uz < 0.0:
                dzb = (iz * dx - z) / uz
            else:
                dzb = 1e300
            axis_x = dxb < dzb
            db = max(dxb if axis_x else dzb, 0.0)
            hit = db < s_ev
            ell = db if hit else s_ev
            # absorción continua: peso perdido y fluencia por estimador de longitud de traza
            a = mua[iz, ix] * ell
            if a > 1e-8:
                f = -np.expm1(-a)
                dw = w * f
                trk = w * ell * f / a
            else:
                dw = w * a
                trk = w * ell * (1.0 - 0.5 * a)
            flu[iz, ix] += trk
            absorbed[iz, ix] += dw
            if use_t:
                it = int((t + 0.5 * nmap[iz, ix] * ell / C0) / dt_bin)
                if it < nt:
                    flu_t[it, iz, ix] += trk
            w -= dw
            t += nmap[iz, ix] * ell / C0
            x += ux * ell
            z += uz * ell
            if not hit:
                # ---- dispersión Henyey-Greenstein
                g = gmap[iz, ix]
                u = np.random.random()
                if g == 0.0:
                    ct = 2.0 * u - 1.0
                else:
                    q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
                    ct = (1.0 + g * g - q * q) / (2.0 * g)
                ct = min(max(ct, -1.0), 1.0)
                st = np.sqrt(1.0 - ct * ct)
                ph = 2.0 * np.pi * np.random.random()
                cp = np.cos(ph)
                sp = np.sin(ph)
                if abs(uz) > 0.99999:
                    ux = st * cp
                    uy = st * sp
                    uz = ct if uz > 0.0 else -ct
                else:
                    tmp = np.sqrt(1.0 - uz * uz)
                    nux = st * (ux * uz * cp - uy * sp) / tmp + ux * ct
                    nuy = st * (uy * uz * cp + ux * sp) / tmp + uy * ct
                    uz = -st * cp * tmp + uz * ct
                    ux = nux
                    uy = nuy
                scattered = True
                tau = -np.log(1.0 - np.random.random())
            else:
                tau -= ms * ell
                # ---- cruce de pared: celda vecina
                if axis_x:
                    sgn = 1 if ux > 0.0 else -1
                    nix = ix + sgn
                    niz = iz
                    ci = abs(ux)
                    x = (ix + 1) * dx if sgn > 0 else ix * dx
                else:
                    sgn = 1 if uz > 0.0 else -1
                    nix = ix
                    niz = iz + sgn
                    ci = abs(uz)
                    z = (iz + 1) * dx if sgn > 0 else iz * dx
                wrapped = False
                if axis_x and periodic:
                    if nix < 0:
                        nix = nx - 1
                        wrapped = True
                    elif nix >= nx:
                        nix = 0
                        wrapped = True
                outside = nix < 0 or nix >= nx or niz < 0 or niz >= nz
                if not outside:
                    outside = not tissue[niz, nix]
                n1 = nmap[iz, ix]
                n2 = n_out if outside else nmap[niz, nix]
                rf, ctr = _fresnel(n1, n2, ci)
                if outside:
                    we = w * (1.0 - rf)
                    if axis_x:
                        prof_side[0 if sgn < 0 else 1, iz] += we
                    else:
                        it = 0
                        if ntof > 1:
                            it = min(int(t / dt_bin), ntof - 1)  # el último hueco es el desbordamiento
                        if sgn < 0:
                            prof_top[ix] += we
                            tof_top[it] += we
                        else:
                            prof_bot[ix] += we
                            tof_bot[it] += we
                            if not scattered:
                                acc[1] += we
                    w *= rf
                    if axis_x:
                        ux = -ux
                    else:
                        uz = -uz
                elif np.random.random() < rf:
                    if axis_x:
                        ux = -ux
                    else:
                        uz = -uz
                else:
                    if n1 != n2:
                        r = n1 / n2
                        if axis_x:
                            ux = sgn * ctr
                            uy *= r
                            uz *= r
                        else:
                            uz = sgn * ctr
                            ux *= r
                            uy *= r
                    ix = nix
                    iz = niz
                    if wrapped:
                        x = 0.0 if nix == 0 else wdx
            if w < WTH:
                if w <= 0.0:
                    alive = False
                elif np.random.random() < CHANCE:
                    w /= CHANCE
                else:
                    alive = False
