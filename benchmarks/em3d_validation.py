"""Validación del FDTD de Maxwell 3D (``wavesim.em.fdtd3d``) y caso de aplicación en la pierna.

Uso (desde la raíz del repositorio; Numba recomendado):

    NUMBA_NUM_THREADS=1 python benchmarks/em3d_validation.py               # todo (~25-30 min en 1 núcleo)
    python benchmarks/em3d_validation.py --case plane      # beta, alpha en medio homogéneo y dipolo (~1 min)
    python benchmarks/em3d_validation.py --case slab       # losa frente a matriz de transferencia y 2D (~2 min)
    python benchmarks/em3d_validation.py --case yinv       # caso invariante en y frente al 2D TM (~1 min)
    python benchmarks/em3d_validation.py --case mie        # esfera frente a la serie de Mie (~8 min)
    python benchmarks/em3d_validation.py --case broadband  # un pulso frente a varias ondas continuas (~2 min)
    python benchmarks/em3d_validation.py --case speed      # Mceldas·paso/s y memoria (~1 min)
    python benchmarks/em3d_validation.py --case leg        # pierna a 915 MHz y 2.45 GHz, 3D frente a 2D (~15 min)

Casos (las tablas de ``docs/validacion/maxwell3d.md`` salen de aquí):

1. ``plane``: lámina de corriente en músculo y grasa homogéneos (20 puntos
   por longitud de onda), las dos polarizaciones: beta y alpha ajustadas a lo
   largo de z frente a Gabriel. Dipolo puntual en vacío y en músculo frente al
   campo exacto del dipolo de Hertz (onda esférica, campo cercano incluido).
2. ``slab``: losa piel 2 / grasa 5 / músculo 15 / cortical 6 mm en aire, onda
   plana TF/SF a incidencia normal (E ∥ x y E ∥ y): |T|, fase y |R| frente a la
   matriz de transferencia (``analytic``) y |T| frente al 2D TM
   (``MaxwellTMSolver``, con y sin losa).
3. ``yinv``: cilindro de músculo con núcleo de cortical extruido en y, E ∥ y:
   (a) ny = 1 con el dt del 2D (igualdad salvo redondeo) y (b) 3D de verdad
   (ny = 16, CPML en y, dt 3D, float32) frente al 2D.
4. ``mie``: esfera de cortical (a = 15 mm) y de músculo (a = 10 mm) en aire a
   2.45 GHz: sección eficaz de absorción y error L2 del campo dentro y fuera
   frente a la serie de Mie, de 10 a 30 puntos por longitud de onda en la
   esfera, con escalera y con promedio sub-celda.
5. ``broadband``: un pulso de 0.5-4 GHz sobre la losa frente a ondas continuas
   con la misma permitividad (el ajuste de Debye) y frente a la matriz de
   transferencia; esfera de cortical con el pulso frente a Mie.
6. ``speed``: rendimiento con 1 y 4 hilos, NumPy/Numba, float32/float64.
7. ``leg``: pierna de ``limb_3d("pierna")`` (tibia y peroné, sin rugosidad,
   fractura transversal de 2 mm en la tibia) en agua, onda plana desde arriba
   a 915 MHz y 2.45 GHz con E ∥ y E ⊥ al hueso: SAR pico y promediado (1 g,
   10 g), fracción de la potencia en el hueso y a más de 2 cm; frente al 2D TM
   en el corte transversal (E ∥ hueso) y en el longitudinal (E ⊥ hueso, la
   pierna como capas planas), y contraste de la fractura en 3D y en 2D.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

from wavesim.core.grid import Grid2D
from wavesim.core.grid3d import Grid3D
from wavesim.em import MaxwellTMSolver, analytic
from wavesim.em.fdtd3d import (
    ETA0, BroadbandMaxwell3DSolver, Maxwell3DSolver, subcell_node_permittivity,
)
from wavesim.materials import dielectric
from wavesim.materials.dispersive import tissue_debye_model
from wavesim.phantoms import Fracture3D, LIMB3D_PRESETS, limb_3d, limb_grid
from wavesim.phantoms.limb3d import Phantom3D, extrude, homogeneous_phantom
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

C0 = dielectric.C0
LAYERS = [("skin", 2e-3), ("fat", 5e-3), ("muscle", 15e-3), ("cortical_bone", 6e-3)]


def db(x):
    return 20 * np.log10(np.abs(x))


# ----------------------------------------------------------------------------
# 1. Medio homogéneo y dipolo
# ----------------------------------------------------------------------------
def fit_beta_alpha(col, z):
    phase = np.unwrap(np.angle(col))
    return -np.polyfit(z, phase, 1)[0], -np.polyfit(z, np.log(np.abs(col)), 1)[0]


def hertz_dipole_ez(x, z, f, eps):
    """``E_z`` del dipolo de Hertz de 1 A·m a lo largo de z en el origen (``e^{+jwt}``), en el plano y = 0."""
    k = 2 * np.pi * f / C0 * np.sqrt(complex(eps))
    eta = ETA0 / np.sqrt(complex(eps))
    r = np.hypot(x, z)
    ct, st = z / r, x / r
    g = np.exp(-1j * k * r)
    e_r = eta * ct / (2 * np.pi * r ** 2) * (1 + 1 / (1j * k * r)) * g
    e_t = 1j * eta * k * st / (4 * np.pi * r) * (1 + 1 / (1j * k * r) - 1 / (k * r) ** 2) * g
    return e_r * ct - e_t * st


def case_plane():
    print("\n## 1. Medio homogéneo: beta y alpha (lámina de corriente, 20 puntos por longitud de onda)\n")
    print("| tejido | f | polarización | beta FDTD / Gabriel - 1 | alpha FDTD / Gabriel - 1 |")
    print("|---|---|---|---|---|")
    for tissue in ("muscle", "fat"):
        for f0 in (915e6, 2.45e9):
            beta, alpha = dielectric.wave_numbers(tissue, f0)
            dx = 2 * np.pi / beta / 20
            nz = int(round(5 * 2 * np.pi / beta / dx)) + 80
            g = Grid3D(2, 2, nz, dx)
            for pol in ("x", "y"):
                s = Maxwell3DSolver(homogeneous_phantom(g, tissue), f0, npml=(20, 0, 0), dtype=np.float64)
                s.set_current_sheet(z=25 * dx, polarization=pol)
                r = s.run()
                col = r.e[0 if pol == "x" else 1][:, 0, 0]
                sel = slice(35, nz - 30)
                b, a = fit_beta_alpha(col[sel], g.z[sel])
                print(f"| {tissue} | {f0 / 1e9:.3g} GHz | E ∥ {pol} | {b / beta - 1:+.3%} | {a / alpha - 1:+.3%} |")
    print("\n## 1b. Dipolo puntual frente al dipolo de Hertz (plano ecuatorial y = 0, Ez a lo largo de x)\n")
    print("| medio | f | dx | r (celdas) | error L2 de Ez | error L2, r >= 8 celdas | error de \\|Ez\\| máx. | malla "
          "| tiempo |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tissue, f0, ppw in (("air", 2.45e9, 20), ("muscle", 2.45e9, 20)):
        eps = complex(dielectric.complex_permittivity(tissue, f0))
        dx = C0 / f0 / abs(np.sqrt(eps)) / ppw
        n = 81
        c = (n - 1) / 2 * dx
        g = Grid3D(n, n, n, dx, -c, -c, -c)
        s = Maxwell3DSolver(homogeneous_phantom(g, tissue), f0, npml=12, dtype=np.float64)
        s.set_dipole(0.0, 0.0, 0.0, "z")
        t0 = time.time()
        r = s.run(settle_time=15 / f0)
        el = time.time() - t0
        mid = n // 2
        ii = np.arange(mid + 4, n - 16)
        x = g.x[ii]
        fd = r.e[2, mid, mid, ii]
        # Ez centrado = media de los nodos z = -dx/2 y +dx/2 de la celda; el dipolo está en el nodo +dx/2 (z = 0)
        ex = 0.5 * (hertz_dipole_ez(x, -dx, f0, eps) + hertz_dipole_ez(x, 0 * x, f0, eps))
        far = ii - mid >= 8
        err = np.linalg.norm(fd - ex) / np.linalg.norm(ex)
        err_far = np.linalg.norm(fd[far] - ex[far]) / np.linalg.norm(ex[far])
        err_mag = np.max(np.abs(np.abs(fd) / np.abs(ex) - 1))
        print(f"| {tissue} | {f0 / 1e9:.3g} GHz | {dx * 1e3:.2f} mm | {ii[0] - mid}-{ii[-1] - mid} | {err:.2%} | "
              f"{err_far:.2%} | {err_mag:.2%} | {n}³ | {el:.0f} s |")


# ----------------------------------------------------------------------------
# 2. Losa
# ----------------------------------------------------------------------------
def slab_phantom_3d(dx, z_top=30e-3, margin=40e-3):
    depth = z_top + sum(d for _, d in LAYERS) + margin
    g = Grid3D(2, 2, int(round(depth / dx)), dx)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    z = z_top
    for t, d in LAYERS:
        lab[(g.z >= z - 1e-12) & (g.z < z + d - 1e-12)] = TISSUE_INDEX[t]
        z += d
    return Phantom3D(g, lab)


def slab_2d_t(f0, dx, z_top=30e-3, margin=40e-3):
    """``T`` del 2D TM: campo detrás de la losa con y sin ella (como ``benchmarks/em_validation.py``)."""
    p3 = slab_phantom_3d(dx, z_top, margin)
    g3 = p3.grid
    g2 = Grid2D(4, g3.nz, dx)
    lab = np.repeat(p3.labels[:, 0, :1], 4, axis=1)
    kb = g3.index_of(0, 0, z_top + sum(d for _, d in LAYERS) + 10e-3)[0]
    out = []
    for labels in (lab, np.full_like(lab, TISSUE_INDEX["air"])):
        s = MaxwellTMSolver(Phantom(g2, labels), f0, npml=20)
        s.set_plane_wave_source(z=g2.z0 + 23 * dx)
        r = s.run(settle_time=40 / f0, n_periods_steady=8)
        out.append(r.phasor[kb, 2])
    return out[0] / out[1]


def case_slab():
    print("\n## 2. Losa piel/grasa/músculo/cortical en aire, onda plana TF/SF\n")
    print("| f | dx | pol. | \\|T\\| exacto | 3D: error \\|T\\| | 3D: fase | 2D TM: error \\|T\\| | 2D: fase "
          "| \\|R\\| exacto | 3D: \\|R\\| | tiempo 3D |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    D = sum(d for _, d in LAYERS)
    for f0, dx in ((915e6, 0.5e-3), (2.45e9, 0.5e-3), (915e6, 0.25e-3), (2.45e9, 0.25e-3)):
        p = slab_phantom_3d(dx)
        g = p.grid
        kb = g.index_of(0, 0, 30e-3 + D + 10e-3)[0]
        ka = 21  # por encima de la cara superior de la caja TF/SF (k0 = npml + 3 = 23): sólo la onda reflejada
        t_ex = analytic.insertion_transmission(LAYERS, f0, outer="air")
        r_ex, _ = analytic.multilayer_rt([dielectric.complex_permittivity(k, f0) for k, _ in LAYERS],
                                         [d for _, d in LAYERS], f0)
        t2 = slab_2d_t(f0, dx)
        for pol in ("x", "y"):
            s = Maxwell3DSolver(p, f0, npml=(20, 0, 0), dtype=np.float64)
            s.set_plane_wave(pol)
            r = s.run(settle_time=40 / f0, n_periods_steady=8)
            col = r.e[0 if pol == "x" else 1][:, 1, 1]
            t3 = col[kb] / r.incident[kb]
            r3 = col[ka] / r.incident[ka]
            print(f"| {f0 / 1e9:.3g} GHz | {dx * 1e3:.2f} mm | E ∥ {pol} | {db(t_ex):.3f} dB | "
                  f"{db(t3) - db(t_ex):+.3f} dB | {np.degrees(np.angle(t3 / t_ex)):+.2f}° | "
                  f"{db(t2) - db(t_ex):+.3f} dB | {np.degrees(np.angle(t2 / t_ex)):+.2f}° | {abs(r_ex):.4f} | "
                  f"{abs(r3):.4f} | {r.elapsed_s:.1f} s |")


# ----------------------------------------------------------------------------
# 3. Invariante en y
# ----------------------------------------------------------------------------
def cylinder_2d(dx):
    g = Grid2D.from_size(90e-3, 100e-3, dx, x0=-45e-3)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    X, Z = g.meshgrid()
    lab[np.hypot(X, Z - 50e-3) < 20e-3] = TISSUE_INDEX["muscle"]
    lab[np.hypot(X, Z - 50e-3) < 8e-3] = TISSUE_INDEX["cortical_bone"]
    return Phantom(g, lab)


def case_yinv():
    print("\n## 3. Caso invariante en y (cilindro de músculo con núcleo de cortical, E ∥ y) frente al 2D TM\n")
    print("| f | dx | variante | diferencia máx. / máx. | mediana | SAR máx. 3D/2D - 1 | tiempo 2D | tiempo 3D |")
    print("|---|---|---|---|---|---|---|---|")
    for f0, dx in ((915e6, 1e-3), (2.45e9, 0.5e-3)):
        p2 = cylinder_2d(dx)
        s2 = MaxwellTMSolver(p2, f0, npml=20)
        s2.set_plane_wave_source()
        r2 = s2.run()
        ref = r2.phasor
        for label, ny, npml, dtype, dt in (("ny = 1, dt del 2D, float64", 1, (20, 0, 20), np.float64, s2.dt),
                                           ("ny = 16, CPML en y, dt 3D, float32", 16, (20, 4, 20), np.float32,
                                            None)):
            s3 = Maxwell3DSolver(extrude(p2, ny=ny), f0, npml=npml, dtype=dtype, dt=dt)
            s3.set_current_sheet(polarization="y")
            r3 = s3.run(exact_dft=False)
            ey = r3.e[1][:, ny // 2, :]
            inner = (slice(20, -20), slice(20, -20))
            d = np.abs(ey - ref)[inner]
            m = np.abs(ref)[inner].max()
            sar3, sar2 = r3.sar[:, ny // 2, :][inner].max(), r2.sar[inner].max()
            print(f"| {f0 / 1e9:.3g} GHz | {dx * 1e3:.1f} mm | {label} | {d.max() / m:.2e} | {np.median(d) / m:.2e} | "
                  f"{sar3 / sar2 - 1:+.2e} | {r2.elapsed_s:.1f} s | {r3.elapsed_s:.1f} s |")


# ----------------------------------------------------------------------------
# 4. Esfera frente a Mie
# ----------------------------------------------------------------------------
def sphere_run(tissue, a, ppw, f=2.45e9, subcell=False, settle_periods=25, broadband=None):
    eps = complex(dielectric.complex_permittivity(tissue, f))
    lam0 = C0 / f
    dx = lam0 / abs(np.sqrt(eps)) / ppw
    npml = 10
    half = a + lam0 / 4
    n = int(np.ceil(2 * half / dx)) + 2 * npml + 8
    n += 1 - n % 2
    c = (n - 1) / 2 * dx
    g = Grid3D(n, n, n, dx, -c, -c, -c)
    zz, yy, xx = np.meshgrid(g.z, g.y, g.x, indexing="ij")
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[r <= a] = TISSUE_INDEX[tissue]
    p = Phantom3D(g, lab)
    sel_in = r <= a - dx
    shell = (r > a + dx) & (r < half) & (np.abs(xx) < half) & (np.abs(yy) < half) & (np.abs(zz) < half)
    if broadband is not None:
        band, freqs, model = broadband
        s = BroadbandMaxwell3DSolver(p, band, model=model, npml=npml)
        s.set_plane_wave("x", z_ref=0.0)
        res = s.run(freqs=freqs)
        rows = []
        for m, ff in enumerate(freqs):
            rf = res.at(m)
            for name, eps_f in (("Gabriel", complex(dielectric.complex_permittivity(tissue, ff))),
                                ("Debye", complex(model.tissues[tissue].complex_permittivity(ff)))):
                e_in = analytic.mie_sphere_field(xx[sel_in], yy[sel_in], zz[sel_in], a, eps_f, ff)
                e_out = analytic.mie_sphere_field(xx[shell], yy[shell], zz[shell], a, eps_f, ff)
                rows.append((ff, name, np.linalg.norm(rf.e[:, sel_in] - e_in) / np.linalg.norm(e_in),
                             np.linalg.norm(rf.e[:, shell] - e_out) / np.linalg.norm(e_out)))
        return rows, res, dx, n
    node = None
    if subcell:
        node = subcell_node_permittivity(g, lambda x, y, z: np.where(x * x + y * y + z * z <= a * a, eps, 1.0 + 0j))
    s = Maxwell3DSolver(p, f, npml=npml, node_eps=node)
    s.set_plane_wave("x", z_ref=0.0)
    res = s.run(settle_time=settle_periods / f)
    q_fdtd = res.absorbed_power() / (1 / (2 * ETA0)) / (np.pi * a ** 2)
    q_mie = analytic.mie_sphere_efficiencies(eps, a, f)[2]
    e_in = analytic.mie_sphere_field(xx[sel_in], yy[sel_in], zz[sel_in], a, eps, f)
    e_out = analytic.mie_sphere_field(xx[shell], yy[shell], zz[shell], a, eps, f)
    err_in = np.linalg.norm(res.e[:, sel_in] - e_in) / np.linalg.norm(e_in)
    err_out = np.linalg.norm(res.e[:, shell] - e_out) / np.linalg.norm(e_out)
    vol = (r <= a).sum() * dx ** 3 / (4 / 3 * np.pi * a ** 3)
    return dict(dx=dx, n=n, steps=res.steps, t=res.elapsed_s, q=q_fdtd, q_mie=q_mie, err_in=err_in, err_out=err_out,
                vol=vol, mem=s.memory_bytes()["total"])


def case_mie(quick=False):
    print("\n## 4a. Serie de Mie frente a valores conocidos\n")
    lam, a = 0.6328e-6, 0.525e-6
    q = analytic.mie_sphere_efficiencies(1.55 ** 2, a, C0 / lam)
    print(f"- m = 1.55, x = {2 * np.pi * a / lam:.4f}: Q_ext = {q[0]:.7f}, Q_sca = {q[1]:.7f} (referencia 3.1054255)")
    for x in (0.01, 0.05, 0.1):
        m = 1.5 + 0.1j
        q = analytic.mie_sphere_efficiencies(np.conj(m) ** 2, x * lam / (2 * np.pi), C0 / lam)
        ray = 4 * x * ((m ** 2 - 1) / (m ** 2 + 2)).imag
        print(f"- Rayleigh, m = 1.5 + 0.1i, x = {x}: Q_abs = {q[2]:.6e}, 4x Im((m²-1)/(m²+2)) = {ray:.6e} "
              f"({q[2] / ray - 1:+.2e})")
    for tissue, a in (("cortical_bone", 15e-3), ("muscle", 10e-3)):
        eps = complex(dielectric.complex_permittivity(tissue, 2.45e9))
        k0 = 2 * np.pi * 2.45e9 / C0
        print(f"\n## 4b. Esfera de {tissue} (a = {a * 1e3:.0f} mm, eps = {eps.real:.2f} - {-eps.imag:.2f}j, "
              f"|m| x = {abs(np.sqrt(eps)) * k0 * a:.2f}) en aire a 2.45 GHz; Q_abs Mie = "
              f"{analytic.mie_sphere_efficiencies(eps, a, 2.45e9)[2]:.4f}\n")
        print("| puntos/lambda | dx | malla | volumen escalera | Q_abs escalera | Q_abs sub-celda | escalera: dentro "
              "| fuera | sub-celda: dentro | fuera | pasos | tiempo (2 casos) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        ppws = (10, 15, 20) if (quick or tissue == "muscle") else (10, 15, 20, 30)
        for ppw in ppws:
            st = sphere_run(tissue, a, ppw)
            sc = sphere_run(tissue, a, ppw, subcell=True)
            print(f"| {ppw} | {st['dx'] * 1e3:.3f} mm | {st['n']}³ | {st['vol']:.3f} | "
                  f"{st['q'] / st['q_mie'] - 1:+.2%} | {sc['q'] / sc['q_mie'] - 1:+.2%} | "
                  f"{st['err_in']:.2%} | {st['err_out']:.2%} | {sc['err_in']:.2%} | "
                  f"{sc['err_out']:.2%} | {st['steps']} | {st['t'] + sc['t']:.0f} s |")


# ----------------------------------------------------------------------------
# 5. Banda ancha
# ----------------------------------------------------------------------------
def case_broadband():
    print("\n## 5a. Losa con un pulso de 0.5-4 GHz frente a ondas continuas (misma permitividad: el ajuste de Debye)\n")
    band = (0.5e9, 4e9)
    model = tissue_debye_model(band)
    freqs = np.array([0.5e9, 0.915e9, 1.5e9, 2.45e9, 3.0e9, 4.0e9])
    D = sum(d for _, d in LAYERS)
    for dx in (0.5e-3,):
        p = slab_phantom_3d(dx)
        kb = p.grid.index_of(0, 0, 30e-3 + D + 10e-3)[0]
        sb = BroadbandMaxwell3DSolver(p, band, model=model, npml=(20, 0, 0), dtype=np.float64)
        sb.set_plane_wave("y")
        rb = sb.run(freqs=freqs)
        print(f"{model.n_poles} polos, {rb.steps} pasos, {rb.elapsed_s:.1f} s (un pulso)\n")
        print("| f (GHz) | \\|T\\| Debye exacto | pulso: error \\|T\\| | fase | onda continua: error \\|T\\| | fase "
              "| pulso frente a onda continua | tiempo onda continua |")
        print("|---|---|---|---|---|---|---|---|")
        for m, f in enumerate(freqs):
            perm = lambda k, ff: complex(model.tissues[k].complex_permittivity(ff))  # noqa: E731
            sn = Maxwell3DSolver(p, f, npml=(20, 0, 0), dtype=np.float64, permittivity=perm)
            sn.set_plane_wave("y")
            rn = sn.run(settle_time=40 / f, n_periods_steady=8)
            tb = rb.e[m, 1, kb, 1, 1] / rb.incident[m, kb]
            tn = rn.e[1, kb, 1, 1] / rn.incident[kb]
            t_ex = analytic.insertion_transmission(LAYERS, f, outer="air", permittivity=perm)
            print(f"| {f / 1e9:.3g} | {db(t_ex):.3f} dB | {db(tb) - db(t_ex):+.3f} dB | "
                  f"{np.degrees(np.angle(tb / t_ex)):+.2f}° | {db(tn) - db(t_ex):+.3f} dB | "
                  f"{np.degrees(np.angle(tn / t_ex)):+.2f}° | {abs(tb / tn - 1):.2e} | {rn.elapsed_s:.1f} s |")
    print("\n## 5b. Esfera de cortical (a = 15 mm) con un pulso de 1-3 GHz frente a Mie (20 puntos/lambda a 3 GHz)\n")
    band = (1e9, 3e9)
    model = tissue_debye_model(band)
    freqs = np.array([1.0e9, 1.5e9, 2.0e9, 2.45e9, 3.0e9])
    eps3 = complex(dielectric.complex_permittivity("cortical_bone", 3e9))
    ppw = 20 * abs(np.sqrt(eps3)) / abs(np.sqrt(complex(dielectric.complex_permittivity("cortical_bone", 2.45e9)))) \
        * 3e9 / 2.45e9
    rows, res, dx, n = sphere_run("cortical_bone", 15e-3, ppw, broadband=(band, freqs, model))
    print(f"dx = {dx * 1e3:.3f} mm, {n}³ celdas, {res.steps} pasos, {res.elapsed_s:.0f} s, escalera\n")
    print("| f (GHz) | Mie con | dentro | fuera |")
    print("|---|---|---|---|")
    for ff, name, ei, eo in rows:
        print(f"| {ff / 1e9:.3g} | {name} | {ei:.2%} | {eo:.2%} |")


# ----------------------------------------------------------------------------
# 6. Rendimiento
# ----------------------------------------------------------------------------
SPEED_SNIPPET = r"""
import sys, time, numpy as np
from wavesim.core.grid3d import Grid3D
from wavesim.em.fdtd3d import Maxwell3DSolver, BroadbandMaxwell3DSolver
from wavesim.materials.dispersive import tissue_debye_model
from wavesim.phantoms.limb3d import homogeneous_phantom
n, backend, bits, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
dtype = np.float32 if bits == "32" else np.float64
g = Grid3D(n, n, n, 1e-3)
p = homogeneous_phantom(g, "muscle")
if mode == "nb":
    s = Maxwell3DSolver(p, 2.45e9, npml=10, dtype=dtype, backend=backend)
else:
    s = BroadbandMaxwell3DSolver(p, (1e9, 4e9), model=tissue_debye_model((1e9, 4e9), n_poles=4), npml=10,
                                 dtype=dtype, backend=backend)
s.set_plane_wave("x")
for k in range(3):
    s.step(0.1)
nt = 40 if backend == "numba" else 6
t = time.time()
for k in range(nt):
    s.step(np.sin(0.1 * k))
el = time.time() - t
print(g.size * nt / el / 1e6, s.memory_bytes()["total"] / g.size)
"""


def case_speed():
    print("\n## 6. Rendimiento (malla 120³ de músculo, CPML de 10 celdas, onda plana TF/SF; Numba ya compilado)\n")
    print("| modo | backend | dtype | hilos | Mceldas·paso/s | memoria del solver (B/celda) |")
    print("|---|---|---|---|---|---|")
    env = dict(os.environ)
    for mode, backend, bits, threads in (("nb", "numpy", "32", 1), ("nb", "numpy", "64", 1),
                                         ("nb", "numba", "32", 1), ("nb", "numba", "32", 4),
                                         ("nb", "numba", "64", 1), ("nb", "numba", "64", 4),
                                         ("bb", "numba", "32", 1), ("bb", "numba", "32", 4)):
        env["NUMBA_NUM_THREADS"] = str(threads)
        out = subprocess.run([sys.executable, "-c", SPEED_SNIPPET, "120", backend, bits, mode], env=env,
                             capture_output=True, text=True, check=True).stdout.split()
        name = "banda estrecha" if mode == "nb" else "Debye (4 polos)"
        print(f"| {name} | {backend} | float{bits} | {threads} | {float(out[0]):.1f} | {float(out[1]):.0f} |")


# ----------------------------------------------------------------------------
# 7. Pierna
# ----------------------------------------------------------------------------
BONE_KEYS = ("cortical_bone", "trabecular_bone", "marrow", "hematoma", "edema")


def power_fractions(pd, labels, z, z_skin, deep=20e-3):
    """Fracción de la potencia absorbida en el tejido que cae en el hueso y a más de ``deep`` bajo la piel."""
    tissue = ~np.isin(labels, [TISSUE_INDEX["water"], TISSUE_INDEX["air"]])
    bone = np.isin(labels, [TISSUE_INDEX[k] for k in BONE_KEYS])
    zz = z.reshape((-1,) + (1,) * (labels.ndim - 1))
    total = pd[tissue].sum()
    return pd[bone].sum() / total, pd[tissue & (zz > z_skin + deep)].sum() / total


def leg_phantom(fracture, dx, length, pad):
    """``limb_3d("pierna")`` sin rugosidad, con ``pad`` celdas más de agua arriba y abajo (caras de la TF/SF)."""
    spec = LIMB3D_PRESETS["pierna"]
    g0 = limb_grid(spec, length, dx)
    grid = Grid3D(g0.nx, g0.ny, g0.nz + 2 * pad, dx, g0.x0, g0.y0, g0.z0 - pad * dx)
    return limb_3d("pierna", fracture, length=length, dx=dx, grid=grid, roughness=False)


def case_leg(freqs=(915e6, 2.45e9)):
    """Pierna de ``limb_3d`` (tibia y peroné) con fractura transversal, onda plana desde arriba, 3D frente a 2D."""
    npml = 10
    frac = Fracture3D(kind="transversal", gap=2e-3)
    print("\n## 7. Pierna 3D (limb_3d 'pierna', sin rugosidad, fractura transversal de 2 mm en la tibia) en agua\n")
    rows = []
    for f0 in freqs:
        dx, length, x_far = (2e-3, 100e-3, -25e-3) if f0 < 1.5e9 else (1e-3, 80e-3, -25e-3)
        p_frac = leg_phantom(frac, dx, length, npml + 6)
        p_int = leg_phantom(None, dx, length, npml + 6)
        g = p_frac.grid
        z_skin, z_bone = p_frac.meta["skin_depth"], p_frac.meta["bone_top_depth"]
        k_skin = g.index_of(0, 0, z_skin)[0]
        j0, i0 = g.index_of(0, 0, 0)[1], g.index_of(0, 0, 0)[2]
        i_far = g.index_of(x_far, 0, 0)[2]
        out = {}
        for name, ph, pol in (("par_frac", p_frac, "parallel"), ("perp_frac", p_frac, "perpendicular"),
                              ("perp_int", p_int, "perpendicular")):
            s = Maxwell3DSolver(ph, f0, npml=npml)
            s.set_plane_wave(pol, z_ref=z_skin)
            out[name] = (s.run(), s.memory_bytes()["total"])
        # 2D TM: corte transversal en x_far (E ∥ hueso) y longitudinal en y = 0 (E ⊥ hueso: capas planas)
        res2 = {}
        for name, ph in (("cross", p_frac.slice_x(x_far)), ("longi", p_frac.slice_y(0.0)),
                         ("longi_int", p_int.slice_y(0.0))):
            s2 = MaxwellTMSolver(ph, f0, npml=npml)
            s2.set_plane_wave_source(z=ph.grid.z0 + (npml + 3) * dx)
            r2 = s2.run()
            # Incidente en la piel: la misma simulación sin la pierna (todo agua)
            s0 = MaxwellTMSolver(Phantom(ph.grid, np.full_like(ph.labels, TISSUE_INDEX["water"])), f0, npml=npml)
            s0.set_plane_wave_source(z=ph.grid.z0 + (npml + 3) * dx)
            res2[name] = (r2, s0.run().phasor[k_skin, ph.grid.nx // 2], ph.labels)
        rows.append(dict(f0=f0, g=g, out=out, res2=res2, k_skin=k_skin, j0=j0, i0=i0, i_far=i_far, z_skin=z_skin,
                         z_bone=z_bone))
        print(f"f = {f0 / 1e9:.3g} GHz, dx = {dx * 1e3:.1f} mm, malla {g.nx}x{g.ny}x{g.nz} = {g.size / 1e6:.2f} M "
              f"celdas, memoria del solver {out['par_frac'][1] / 2 ** 20:.0f} MiB, "
              + ", ".join(f"{k}: {v[0].steps} pasos {v[0].elapsed_s:.0f} s" for k, v in out.items()), flush=True)
    print("\nNormalización: 1 V/m de onda incidente (en el agua) en el plano de la piel; SAR en W/kg por (V/m)². "
          "SAR 1 g y 10 g: cubo centrado sólo sobre el tejido (§2.4).\n")
    print("| f | caso | SAR pico | SAR 1 g | SAR 10 g | potencia en el hueso | potencia a > 2 cm bajo la piel "
          "| \\|E\\| 2 mm dentro de la cortical (x lejos, y = 0) |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        f0, g, out, res2, j0, i_far = row["f0"], row["g"], row["out"], row["res2"], row["j0"], row["i_far"]
        z_skin, z_bone = row["z_skin"], row["z_bone"]
        k_bone = g.index_of(0, 0, z_bone + 2e-3)[0]
        for name, label in (("par_frac", "3D, E ∥ hueso, con fractura"), ("perp_frac", "3D, E ⊥ hueso, con fractura"),
                            ("perp_int", "3D, E ⊥ hueso, sin fractura")):
            r = out[name][0]
            tissue = r.tissue_mask()
            fb, fd = power_fractions(r.power_density, r.labels, g.z, z_skin)
            print(f"| {f0 / 1e9:.3g} GHz | {label} | {r.sar[tissue].max():.3e} | {r.averaged_sar(1e-3).max():.3e} | "
                  f"{r.averaged_sar(10e-3).max():.3e} | {fb:.2%} | {fd:.2%} | "
                  f"{np.sqrt(r.e_squared[k_bone, j0, i_far]):.3f} |")
        for name, label in (("cross", "2D TM transversal (E ∥ hueso)"),
                            ("longi", "2D TM longitudinal (E ⊥ hueso, capas planas)")):
            r2, norm, labels2 = res2[name]
            n2 = abs(norm) ** 2
            fb, fd = power_fractions(0.5 * r2.sigma * np.abs(r2.phasor) ** 2, labels2, g.z, z_skin)
            c = j0 if name == "cross" else i_far
            print(f"| {f0 / 1e9:.3g} GHz | {label} | {(r2.sar / n2).max():.3e} | - | - | {fb:.2%} | {fd:.2%} | "
                  f"{abs(r2.phasor[k_bone, c] / norm):.3f} |")
    print("\n### 3D frente a 2D (mismo plano, misma normalización)\n")
    print("| f | comparación | \\|E\\| en la vertical y = 0: diferencia máx. (piel - 2 mm en la cortical) "
          "| SAR máx. del plano 3D/2D | contraste de la fractura en \\|E\\| (centro del hueco) 3D | 2D |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        f0, g, out, res2, j0, i0, i_far = (row[k] for k in ("f0", "g", "out", "res2", "j0", "i0", "i_far"))
        k_skin, z_bone = row["k_skin"], row["z_bone"]
        k_bone = g.index_of(0, 0, z_bone + 2e-3)[0]
        zsel = slice(k_skin, k_bone + 1)
        r3 = out["par_frac"][0]
        r2, norm, _ = res2["cross"]
        e3 = np.sqrt(r3.e_squared[:, :, i_far])
        e2 = np.abs(r2.phasor / norm)
        d = np.abs(e3[zsel, j0] - e2[zsel, j0]).max() / e2[zsel, j0].max()
        tissue2 = ~np.isin(r3.labels[:, :, i_far], [TISSUE_INDEX["water"]])
        s3 = r3.sar[:, :, i_far][tissue2].max()
        s2 = (r2.sar / abs(norm) ** 2)[tissue2].max()
        print(f"| {f0 / 1e9:.3g} GHz | E ∥ hueso: 3D en x = {g.x[i_far] * 1e3:.0f} mm frente a 2D transversal | "
              f"{d:.2%} | {s3 / s2:.3f} | - | - |")
        r3, r3i = out["perp_frac"][0], out["perp_int"][0]
        r2, norm, _ = res2["longi"]
        r2i, norm_i, _ = res2["longi_int"]
        e3 = np.sqrt(r3.e_squared[:, j0, :])
        e2 = np.abs(r2.phasor / norm)
        d = np.abs(e3[zsel, i_far] - e2[zsel, i_far]).max() / e2[zsel, i_far].max()
        tissue2 = ~np.isin(r3.labels[:, j0, :], [TISSUE_INDEX["water"]])
        s3 = r3.sar[:, j0, :][tissue2].max()
        s2 = (r2.sar / abs(norm) ** 2)[tissue2].max()
        c3 = np.sqrt(r3.e_squared[k_bone, j0, i0] / r3i.e_squared[k_bone, j0, i0]) - 1
        c2 = abs(r2.phasor[k_bone, i0] / norm) / abs(r2i.phasor[k_bone, i0] / norm_i) - 1
        print(f"| {f0 / 1e9:.3g} GHz | E ⊥ hueso: 3D en y = 0 frente a 2D longitudinal | {d:.2%} | {s3 / s2:.3f} | "
              f"{c3:+.1%} | {c2:+.1%} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="all",
                    choices=["all", "plane", "slab", "yinv", "mie", "broadband", "speed", "leg"])
    ap.add_argument("--quick", action="store_true", help="mallas menos finas en el caso mie")
    args = ap.parse_args()
    t0 = time.time()
    cases = {"plane": case_plane, "slab": case_slab, "yinv": case_yinv, "mie": lambda: case_mie(args.quick),
             "broadband": case_broadband, "speed": case_speed, "leg": case_leg}
    for name, fn in cases.items():
        if args.case in ("all", name):
            fn()
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 10
    print(f"\nTiempo total: {time.time() - t0:.0f} s; memoria máxima del proceso {rss:.0f} MiB")


if __name__ == "__main__":
    main()
