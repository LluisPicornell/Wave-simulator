"""Validación del módulo de rayos X / TAC en 3D (``wavesim.em.xray3d`` y ``xray3d_scatter``).

Uso (desde la raíz del repositorio, con un solo hilo para medir tiempos):

    NUMBA_NUM_THREADS=1 python benchmarks/xray3d_validation.py                 # completo
    NUMBA_NUM_THREADS=1 python benchmarks/xray3d_validation.py --only chords adjoint
    NUMBA_NUM_THREADS=1 python benchmarks/xray3d_validation.py --only leg --figure docs/img/rayos_x3d.png

Imprime las tablas de ``docs/validacion/rayos_x3d.md``. Secciones (``--only``):
  chords    esfera y cilindro frente a la longitud de cuerda analítica (Siddon y muestreo trilineal, cono y paralelo);
  adjoint   <A x, y> = <x, Aᵀ y> en todas las combinaciones de método, backend y geometría;
  extruded  volumen extruido en y proyectado en paralelo frente al 2D (``projection.project``) y TAC frente a ``fbp``;
  poly      radiografía policromática de losas frente a Beer-Lambert integrado en energía;
  crack     visibilidad de una grieta plana frente al ángulo entre el haz y su plano;
  fdk       FDK frente al ángulo de cono sobre un fantoma de elipsoides tipo Shepp-Logan 3D y una esfera;
  sart      OS-SART 3D con TV frente a FDK con pocas vistas y ruido;
  scatter   Monte Carlo 3D: balance, primaria frente a Beer-Lambert, losa frente al MC 2D y SPR de la pierna;
  leg       pierna 3D: radiografías AP y lateral y TAC de haz cónico (tiempo, memoria y figura).
"""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np

from wavesim.core.grid import Grid2D
from wavesim.core.grid3d import Grid3D
from wavesim.em import xray, xray_scatter as xs3
from wavesim.em import xray3d as x3
from wavesim.em import xray3d_scatter as sc3
from wavesim.em.projection import detector_axis, fbp
from wavesim.materials import xray as xd
from wavesim.phantoms import Fracture, TISSUE_INDEX, longitudinal_limb
from wavesim.phantoms.limb3d import Phantom3D, TransverseFracture3D, extrude, homogeneous_phantom, leg_segment
from wavesim.phantoms.phantom import Phantom


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def sec(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


# --------------------------------------------------------------------------------------
def chords() -> None:
    sec("Cuerdas analíticas: esfera (R = 15 mm) y cilindro (eje x, R = 12 mm), error en rayos con rho < 0.8 R")
    print("| objeto | dx (mm) | geometría | método | rms (mm) | máx (mm) | rms / 2R | t (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for dx in (1e-3, 0.5e-3):
        n = int(round(0.056 / dx))
        g = Grid3D(n, n, n, dx)
        c = x3.center(g)
        lo = np.array([g.x0, g.y0, g.z0]) - 0.5 * dx
        hi = lo + np.array([g.nx, g.ny, g.nz]) * dx
        R = {"esfera": 15e-3, "cilindro": 12e-3}
        vols = {"esfera": x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (R["esfera"],) * 3, 1.0)], 3),
                "cilindro": x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (1.0, R["cilindro"], R["cilindro"]),
                                                                 1.0)], 3)}
        for label, par, over in (("paralelo, paso = dx", True, 1.0), ("paralelo, paso = dx/1.37", True, 1.37),
                                 ("cónico", False, 1.0)):
            geo = x3.Geometry3D.covering(g, 0.4, 0.5, parallel=par, oversample=over)
            ang = 37.0
            org, dirs = geo.rays(g, ang)
            for name, vol in vols.items():
                if name == "esfera":
                    ref = x3.sphere_chords(org, dirs, c, R[name])
                    rho = x3.ray_point_distance(org, dirs, c)
                else:
                    ref = x3.cylinder_chords(org, dirs, c, R[name], "x", lo, hi)
                    ax = np.array([1.0, 0.0, 0.0])
                    cr = np.cross(dirs, ax)
                    rho = np.abs(((org - c) * cr).sum(1)) / np.linalg.norm(cr, axis=1)  # distancia recta-eje
                sel = rho < 0.8 * R[name]
                for method, be in (("siddon", "numba"), ("sampling", "numpy")):
                    t = time.time()
                    p = x3.project(vol, g, geo, [ang], method=method, backend=be)[0].reshape(-1)
                    dt = time.time() - t
                    err = (p - ref)[sel]
                    print(f"| {name} | {dx*1e3:.1f} | {label} | {method} | "
                          f"{np.sqrt((err**2).mean())*1e3:.3f} | {np.abs(err).max()*1e3:.3f} | "
                          f"{np.sqrt((err**2).mean())/(2*R[name]):.2e} | {dt:.2f} |", flush=True)
    # Exactitud del algoritmo en sí: volumen homogéneo (caja) frente a la cuerda de la caja
    g = Grid3D(30, 24, 20, 1e-3)
    ones = np.ones(g.shape)
    geo = x3.Geometry3D.covering(g, 0.3, 0.4)
    worst = 0.0
    for ang in (0.0, 13.0, 37.0, 90.0, 211.0):
        org, dirs = geo.rays(g, ang)
        lo = np.array([g.x0, g.y0, g.z0]) - 0.5 * g.dx
        hi = lo + np.array([g.nx, g.ny, g.nz]) * g.dx
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - org) / dirs, (hi - org) / dirs
        par = np.abs(dirs) < 1e-12
        ins = (org >= lo) & (org <= hi)
        t0 = np.where(par, np.where(ins, -np.inf, np.inf), np.minimum(ta, tb)).max(axis=1)
        t1 = np.where(par, np.where(ins, np.inf, -np.inf), np.maximum(ta, tb)).min(axis=1)
        ref = np.maximum(t1 - t0, 0.0)
        for be in ("numba", "numpy"):
            p = x3.project(ones, g, geo, [ang], backend=be)[0].reshape(-1)
            worst = max(worst, np.abs(p - ref).max())
    print(f"\nCaja homogénea 30×24×20 (cono, 5 ángulos, ambos backends): error máx = {worst:.1e} m "
          f"(cuerda exacta de la caja).")


# --------------------------------------------------------------------------------------
def adjoint() -> None:
    sec("Adjunción <A x, y> = <x, Aᵀ y>")
    print("| método | backend | geometría | <Ax, y> | <x, Aᵀy> | error relativo |")
    print("|---|---|---|---|---|---|")
    g = Grid3D(24, 20, 18, 1e-3)
    rng = np.random.default_rng(1)
    x = rng.random(g.shape)
    ang = [0.0, 20.0, 75.0, 130.0, 250.0]
    for par in (False, True):
        geo = x3.Geometry3D.covering(g, 0.15, 0.2, parallel=par)
        y = rng.random((len(ang), geo.n_v, geo.n_u))
        for method, be in (("siddon", "numba"), ("siddon", "numpy"), ("sampling", "numpy")):
            lhs = np.sum(x3.project(x, g, geo, ang, method, be) * y)
            rhs = np.sum(x * x3.backproject(y, g, geo, ang, method, be))
            print(f"| {method} | {be} | {'paralelo' if par else 'cónico'} | {lhs:.10g} | {rhs:.10g} | "
                  f"{abs(lhs-rhs)/abs(lhs):.1e} |", flush=True)


# --------------------------------------------------------------------------------------
def extruded() -> None:
    sec("Volumen extruido en y, haz paralelo, frente al 2D (pierna longitudinal, dx = 0.5 mm, 60 kVp)")
    ph = longitudinal_limb("tibia", Fracture(), width=0.04, depth=0.04, dx=0.5e-3)
    p3 = extrude(ph, ny=6)
    spec = xray.tube_spectrum(60, 2.5)
    sp = 0.37 * ph.grid.dx  # paso genérico: ningún rayo cae justo sobre una cara de celda
    det = detector_axis(ph.grid, sp)
    geo = x3.Geometry3D(0.5, 0.5, det.size, 3, sp, 1e-3, "y", True)
    ang = [0, 30, 60, 90, 123]
    a2 = xray.acquire(ph, ang, spec, photons=None, det_spacing=sp)
    inner = np.abs(det) < 0.0185
    print("| ángulo | Siddon: L2 rel. | Siddon: máx | muestreo: L2 rel. | muestreo: máx |")
    print("|---|---|---|---|---|")
    res = {m: x3.acquire3d(p3, ang, geo, spec, photons=None, method=m) for m in ("siddon", "sampling")}
    for a, th in enumerate(ang):
        cells = []
        for m in ("siddon", "sampling"):
            d = (res[m].line_integrals[a, 1] - a2.line_integrals[a])[inner]
            cells += [f"{np.linalg.norm(d)/np.linalg.norm(a2.line_integrals[a][inner]):.4f}",
                      f"{np.abs(d).max():.3f}"]
        print(f"| {th}° | " + " | ".join(cells) + " |")
    # TAC: FDK de haz paralelo de un corte frente a la FBP 2D del mismo sinograma
    g2 = Grid2D(64, 64, 1e-3)
    from wavesim.phantoms.phantom import TISSUE_INDEX as TI
    yy, xx = np.meshgrid(g2.z - g2.z.mean(), g2.x - g2.x.mean(), indexing="ij")
    lab = np.zeros((64, 64), np.int16)
    lab[:] = TI["water"]
    lab[np.hypot(yy, xx) < 0.02] = TI["muscle"]
    lab[np.hypot(yy, xx - 0.008) < 0.006] = TI["cortical_bone"]
    ph2 = Phantom(g2, lab, {}, np.zeros((64, 64), np.float32), {})
    p3b = extrude(ph2, ny=8)
    e = xray.monoenergetic(60.0)
    angs = np.arange(180.0)
    s2 = xray.acquire(ph2, angs, e, photons=None)
    mu2 = fbp(s2.line_integrals, angs, s2.det, g2)
    geo3 = x3.Geometry3D(0.5, 0.5, s2.det.size, 8, s2.det[1] - s2.det[0], 1e-3, "y", True)
    s3 = x3.acquire3d(p3b, angs, geo3, e, photons=None)
    mu3 = x3.fdk(s3.line_integrals, angs, geo3, p3b.grid, mode="parallel")
    c = mu3[:, p3b.grid.ny // 2, :]
    mask = np.hypot(yy, xx) < 0.025
    print(f"\nTAC paralelo: FDK 3D (corte central del extruido) frente a fbp 2D: "
          f"error rms relativo = {np.sqrt(((c-mu2)[mask]**2).mean())/np.sqrt((mu2[mask]**2).mean()):.4f}")


# --------------------------------------------------------------------------------------
def poly() -> None:
    sec("Radiografía policromática de losas frente a Beer-Lambert integrado en energía (-ln T)")
    print("| losa | kVp / Al | espesor | detector | -ln T simulado | -ln T de referencia | error |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for tissue, thick in (("water", 40e-3), ("cortical_bone", 10e-3), ("muscle", 40e-3), ("fat", 40e-3)):
        nzs = int(round(thick / 1e-3))
        ph = homogeneous_phantom(Grid3D(24, 24, nzs, 1e-3), tissue)
        for kvp, al in ((60, 2.5), (120, 6.0)):
            spec = xray.tube_spectrum(kvp, al)
            geo = x3.Geometry3D(0.5, 0.5, 5, 5, 1e-3, 1e-3, "x", True)
            for det_name, det in (("ideal (fotones)", None), ("CsI 0.5 mm, integra E", x3.FlatPanel())):
                r = x3.acquire3d(ph, [0.0], geo, spec, photons=None, detector=det)
                panel = x3._as_panel(det)
                sig_w, _ = panel.signal_weights(spec)
                mu = xd.linear_attenuation(tissue, spec.energies_kev)
                ref = -np.log(np.sum(sig_w * np.exp(-mu * thick)))
                sim = r.line_integrals[0, 2, 2]
                rows.append(abs(sim - ref))
                print(f"| {tissue} | {kvp} / {al} mm | {thick*1e3:.0f} mm | {det_name} | {sim:.5f} | {ref:.5f} | "
                      f"{sim-ref:+.1e} |")
    print(f"\nPeor |error| = {max(rows):.1e} en -ln T (el de la descomposición agua + hueso: exacto en agua y hueso).")
    # Endurecimiento del haz: atenuación efectiva vs espesor (agua)
    spec = xray.tube_spectrum(80, 2.5)
    print("\nEndurecimiento del haz (80 kVp, 2.5 mm Al, agua): mu efectiva = -ln T / espesor")
    print("| espesor (mm) | mu efectiva (1/m) |")
    print("|---|---|")
    for t in (5e-3, 20e-3, 40e-3):
        nzs = int(round(t / 1e-3))
        r = x3.acquire3d(homogeneous_phantom(Grid3D(8, 8, nzs, 1e-3), "water"), [0.0],
                         x3.Geometry3D(0.5, 0.5, 3, 3, 1e-3, 1e-3, "x", True), spec, photons=None)
        print(f"| {t*1e3:.0f} | {r.line_integrals[0, 1, 1]/t:.2f} |")


# --------------------------------------------------------------------------------------
def crack_phantom(grid: Grid3D, width: float, block: float, with_crack: bool = True) -> Phantom3D:
    """Bloque cúbico de hueso cortical con una grieta plana ``|y| < width/2`` (plano xz) rellena de agua."""
    ph = homogeneous_phantom(grid, "water")
    lab = ph.labels.copy()
    zz, yy, xx = np.meshgrid(grid.z - grid.z.mean(), grid.y - grid.y.mean(), grid.x - grid.x.mean(), indexing="ij")
    inside = (np.abs(zz) < block / 2) & (np.abs(yy) < block / 2) & (np.abs(xx) < block / 2)
    lab[inside] = TISSUE_INDEX["cortical_bone"]
    if with_crack:
        lab[inside & (np.abs(yy) < width / 2)] = TISSUE_INDEX["water"]
    return Phantom3D(grid, lab)


def crack() -> None:
    sec("Grieta plana (0.5 mm de agua en un bloque cortical de 20 mm): señal frente al ángulo haz-plano")
    dx = 0.25e-3
    g = Grid3D(100, 100, 100, dx)
    w, blk = 0.5e-3, 20e-3
    e = xray.monoenergetic(60.0)
    geo = x3.Geometry3D(0.5, 0.5, 281, 5, dx, dx, "x", True)
    mu_b = float(xd.linear_attenuation("cortical_bone", 60.0))
    mu_w = float(xd.linear_attenuation("water", 60.0))
    ph1, ph0 = crack_phantom(g, w, blk, True), crack_phantom(g, w, blk, False)
    print(f"mu(60 keV): hueso {mu_b:.1f} 1/m, agua {mu_w:.1f} 1/m. "
          "Señal = máx de [-ln T(sin grieta) − -ln T(con grieta)] (la grieta atenúa menos).")
    print("| ángulo haz-plano | señal simulada | modelo (mu_b − mu_w)·min(w/sin φ, Z/cos φ) | CNR (1e5 fotones/px) |")
    print("|---|---|---|---|")
    for phi in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0):
        a1 = x3.acquire3d(ph1, [phi], geo, e, photons=None).line_integrals[0, 2]
        a0 = x3.acquire3d(ph0, [phi], geo, e, photons=None).line_integrals[0, 2]
        sig = float(np.max(a0 - a1))
        s, c = np.sin(np.radians(phi)), np.cos(np.radians(phi))
        model = (mu_b - mu_w) * min(w / max(s, 1e-9), blk / max(c, 1e-9))
        # CNR: diferencia de I/I0 entre el píxel de la grieta y su vecindad, frente al ruido de Poisson
        k = np.argmax(a0 - a1)
        i1, i0 = np.exp(-a1[k]), np.exp(-a0[k])
        n = 1e5
        cnr = (i1 - i0) * n / np.sqrt(n * (i1 + i0) / 2)
        print(f"| {phi:.0f}° | {sig:.4f} | {model:.4f} | {abs(cnr):.1f} |", flush=True)


# --------------------------------------------------------------------------------------
def shepp_error(rec, truth, g, mask_slab, support):
    d = (rec - truth)[mask_slab & support]
    return np.sqrt((d**2).mean()) / np.sqrt((truth[mask_slab & support] ** 2).mean())


def fdk_study() -> None:
    sec("FDK frente al ángulo de cono (fantoma de elipsoides tipo Shepp-Logan 3D, 64³, dx = 1 mm, 360 vistas)")
    n = 64
    g = Grid3D(n, n, n, 1e-3)
    gf = Grid3D(2 * n, 2 * n, 2 * n, 0.5e-3, g.x0 - 0.25e-3, g.y0 - 0.25e-3, g.z0 - 0.25e-3)  # malla fina de datos
    truth, ell = x3.shepp_logan_3d(g)
    fine, _ = x3.shepp_logan_3d(gf)
    assert np.isclose(x3.center(g), x3.center(gf)).all()
    zz, yy, xx = np.meshgrid(g.z - g.z.mean(), g.y - g.y.mean(), g.x - g.x.mean(), indexing="ij")
    ctr, sa = np.array(ell[1].center), np.array(ell[1].semi_axes) - 2e-3  # interior del cráneo, 2 mm de margen
    sup = ((xx - ctr[0]) / sa[0]) ** 2 + ((yy - ctr[1]) / sa[1]) ** 2 + ((zz - ctr[2]) / sa[2]) ** 2 < 1
    central = np.zeros(g.shape, bool)
    central[:, :, n // 2 - 3:n // 2 + 4] = True  # corte(s) axial(es) central(es): x = n/2 ± 3
    outer = np.zeros(g.shape, bool)
    outer[:, :, : n // 2 - n // 4] = True
    outer[:, :, n // 2 + n // 4:] = True
    print("| geometría | sid (mm) | semiángulo de cono | error rms plano central | error rms planos exteriores | "
          "t proy (s) | t FDK (s) |")
    print("|---|---|---|---|---|---|---|")
    ang360 = np.arange(360) * 1.0
    cases = [("paralelo (apilado)", None, None, "parallel")]
    for sid in (1.0, 0.4, 0.2, 0.12):
        cases.append((f"cónico, sid {int(sid*1e3)}", sid, 1.4 * sid, "fdk"))
        cases.append((f"cónico apilado, sid {int(sid*1e3)}", sid, 1.4 * sid, "stack"))
    cache = {}
    for name, sid, sdd, mode in cases:
        par = sid is None
        ang = np.arange(180) * 1.0 if par else ang360
        key = (sid,)
        if key not in cache:
            geo = x3.Geometry3D.covering(g, 0.4 if par else sid, sdd, parallel=par, oversample=1.0)
            t = time.time()
            sino = x3.project(fine, gf, geo, ang)  # datos de la malla fina: evita el "crimen inverso"
            cache[key] = (geo, sino, time.time() - t)
        geo, sino, tp = cache[key]
        t = time.time()
        rec = x3.fdk(sino, ang, geo, g, mode=mode)
        tf = time.time() - t
        half = np.degrees(np.arctan(0.5 * geo.n_v * geo.dv / geo.sdd)) if not par else 0.0
        print(f"| {name} | {'-' if par else int(sid*1e3)} | {half:.1f}° | "
              f"{shepp_error(rec, truth, g, central, sup):.3f} | {shepp_error(rec, truth, g, outer, sup):.3f} | "
              f"{tp:.1f} | {tf:.1f} |", flush=True)
    # Esfera homogénea: valor central y error en los polos
    R, mu0 = 0.02, 50.0
    sph = x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (R, R, R), mu0)], 3)
    sphf = x3.ellipsoid_volume(gf, [x3.Ellipsoid((0, 0, 0), (R, R, R), mu0)], 3)
    print("\nEsfera homogénea (R = 20 mm, mu = 50 1/m): valor reconstruido en el centro y a 0.8 R del centro a lo "
          "largo del eje x (polo)")
    print("| geometría | centro | polo (0.8 R) | error rms dentro de la esfera |")
    print("|---|---|---|---|")
    c = n // 2
    pole = c + int(round(0.8 * R / 1e-3))
    inside = sph > 0.99 * mu0
    for name, sid, sdd, mode in [("paralelo", None, None, "parallel"), ("cónico FDK, sid 400", 0.4, 0.56, "fdk"),
                                 ("cónico FDK, sid 200", 0.2, 0.28, "fdk"),
                                 ("cónico FDK, sid 120", 0.12, 0.168, "fdk")]:
        par = sid is None
        ang = np.arange(180) * 1.0 if par else ang360
        geo = x3.Geometry3D.covering(g, 0.4 if par else sid, sdd, parallel=par)
        rec = x3.fdk(x3.project(sphf, gf, geo, ang), ang, geo, g, mode=mode)
        err = np.sqrt(((rec - sph)[inside] ** 2).mean()) / mu0
        print(f"| {name} | {rec[c, c, c]/mu0:.3f} | {rec[c, c, pole]/mu0:.3f} | {err:.3f} |", flush=True)


    # Cilindro largo a lo largo del eje de giro (invariante en x): FDK es exacto (el coseno compensa la inclinación
    # de los rayos); el apilado de abanico, que ignora el cono, falla en los planos fuera del central
    cylf = x3.ellipsoid_volume(gf, [x3.Ellipsoid((0, 0, 0), (1.0, 0.015, 0.015), mu0)], 3)
    print("\nCilindro de R = 15 mm a lo largo del eje de giro (mu = 50 1/m): valor reconstruido / mu en el eje, a "
          "x = 0, 15 y 24 mm del plano central")
    print("| geometría | modo | x = 0 | x = 15 mm | x = 24 mm |")
    print("|---|---|---|---|---|")
    for sid in (0.4, 0.2, 0.12):
        geo = x3.Geometry3D.covering(g, sid, 1.4 * sid)
        sino = x3.project(cylf, gf, geo, ang360)
        for mode in ("fdk", "stack"):
            rec = x3.fdk(sino, ang360, geo, g, mode=mode)
            v = [rec[c, c, c + d] / mu0 for d in (0, 15, 24)]
            print(f"| cónico, sid {int(sid*1e3)} | {mode} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} |", flush=True)


def sart_study() -> None:
    sec("OS-SART 3D (con TV) frente a FDK: pocas vistas y ruido (Shepp-Logan 3D, 48³, dx = 1.3 mm, cono sid 400 mm)")
    n = 48
    g = Grid3D(n, n, n, 1.3e-3)
    gf = Grid3D(2 * n, 2 * n, 2 * n, 0.65e-3, g.x0 - 0.325e-3, g.y0 - 0.325e-3, g.z0 - 0.325e-3)
    truth, ell = x3.shepp_logan_3d(g, mu=50.0)
    fine, _ = x3.shepp_logan_3d(gf, mu=50.0)
    zz, yy, xx = np.meshgrid(g.z - g.z.mean(), g.y - g.y.mean(), g.x - g.x.mean(), indexing="ij")
    ctr, sa = np.array(ell[1].center), np.array(ell[1].semi_axes) - 2.5e-3  # interior del cráneo
    sup = ((xx - ctr[0]) / sa[0]) ** 2 + ((yy - ctr[1]) / sa[1]) ** 2 + ((zz - ctr[2]) / sa[2]) ** 2 < 1
    geo = x3.Geometry3D.covering(g, 0.4, 0.56)
    rng = np.random.default_rng(0)

    def run(name, sino, ang):
        if name == "FDK":
            return x3.fdk(sino, ang, geo, g)
        return x3.os_sart3d(sino, ang, geo, g, n_iter=4, subset_size=6, tv_weight=0.003 if "TV" in name else 0.0)

    print("| vistas | fotones/px | método | error rms relativo (interior) | t (s) |")
    print("|---|---|---|---|---|")
    for nv, n0 in ((120, None), (36, None), (36, 300.0)):
        ang = np.arange(nv) * 360.0 / nv
        sino = x3.project(fine, gf, geo, ang)
        if n0 is not None:  # ruido de Poisson monoenergético
            sino = -np.log(np.maximum(rng.poisson(n0 * np.exp(-sino)), 1) / n0)
        for name in ("FDK", "OS-SART (4 it)", "OS-SART + TV (4 it)"):
            t = time.time()
            rec = run(name, sino, ang)
            err = np.sqrt(((rec - truth)[sup] ** 2).mean()) / 50.0
            print(f"| {nv} | {'sin ruido' if n0 is None else int(n0)} | {name} | {err:.3f} | {time.time()-t:.1f} |",
                  flush=True)


# --------------------------------------------------------------------------------------
def scatter_study() -> None:
    sec("Monte Carlo 3D de scatter")
    spec = xray.tube_spectrum(60, 2.5)
    # 1. balance y primaria frente a Beer-Lambert, con un cubo de agua
    g = Grid3D(40, 40, 40, 1e-3)
    ph = homogeneous_phantom(g, "water")
    geo = x3.Geometry3D.covering(g, 0.5, 0.6)
    t = time.time()
    r = sc3.simulate_view(ph, 0.0, geo, spec, None, 4_000_000, 10)
    dt = time.time() - t
    bl = x3.acquire3d(ph, [0.0], geo, spec, photons=None).intensity[0]
    nb = 10
    cu = geo.n_u // 2
    sel_u = slice(cu - geo.n_u // (2 * nb), cu + geo.n_u // (2 * nb) + 1)
    sel_v = slice(geo.n_v // 2 - geo.n_v // (2 * nb), geo.n_v // 2 + geo.n_v // (2 * nb) + 1)
    inner = r.primary_rel[nb // 2 - 1:nb // 2 + 1, nb // 2 - 1:nb // 2 + 1].mean()
    print(f"Cubo de agua de 40 mm (60 kVp, 4·10⁶ fotones, {dt:.1f} s, {4e6/dt/1e6:.2f} M fotones/s con 1 hilo): "
          f"balance de fotones = {r.balance():.1e}; primaria MC (4 bins centrales) = {inner:.4f}, "
          f"Beer-Lambert (píxel central) = {bl[sel_v, sel_u].mean():.4f}")
    # 2. losa ancha frente al MC 2D (haz paralelo, rendija de 10 mm)
    dx, nx, nz, ny = 2e-3, 60, 20, 120
    g3 = Grid3D(nx, ny, nz, dx)
    p3 = homogeneous_phantom(g3, "water")
    g2 = Grid2D(nx, nz, dx)
    p2 = Phantom(g2, np.full((nz, nx), TISSUE_INDEX["water"], np.int16), {}, np.zeros((nz, nx), np.float32), {})
    geo2 = x3.Geometry3D(0.3, 0.3, nx, 5, dx, dx, "y", True)
    n_ph = 8_000_000
    r3 = sc3.simulate_view(p3, 0, geo2, spec, None, n_ph, 12)
    det = (np.arange(nx) - (nx - 1) / 2) * dx
    r2 = xs3.simulate_scatter(p2, 0, spec, None, det, n_ph, slice_thickness=0.01, backend="numba")
    s2 = r2.scatter_rel.reshape(12, 5).mean(axis=1)
    s3 = r3.scatter_rel.mean(axis=0)
    spr2 = r2.scat.sum() / r2.prim.sum()
    spr3 = r3.scat_n.sum() / r3.prim_n.sum()
    print("\nLosa de agua de 40 mm, 120 mm × 240 mm, rendija de 10 mm, haz paralelo: scatter relativo por bin "
          "(8·10⁶ fotones)")
    print("| bin | MC 2D | MC 3D |")
    print("|---|---|---|")
    for k in (0, 2, 4, 6, 8, 11):
        print(f"| {k} | {s2[k]:.4f} | {s3[k]:.4f} |")
    print(f"SPR global 2D = {spr2:.4f}, 3D = {spr3:.4f}; diferencia máxima por bin = "
          f"{np.max(np.abs(s3 - s2) / s2):.1%}; media = {np.mean(s3) / np.mean(s2) - 1:+.1%}")
    # 3. SPR de la pierna y efecto en el contraste de la fractura
    leg = leg_segment("tibia", TransverseFracture3D(gap=1e-3), length=0.05, dx=0.5e-3)
    geol = x3.Geometry3D.covering(leg.grid, 1.0, 1.1, oversample=1.0)

    def spr(r, lo, hi):
        sel = (r.primary[0] > lo) & (r.primary[0] < hi)
        return float(r.scatter[0][sel].sum() / r.primary[0][sel].sum())

    rr = {}
    for view in ("ap", "lateral"):
        t = time.time()
        rr[view] = x3.radiograph(leg, view, geometry=geol, photons=None, scatter=sc3.Scatter3DOptions(6_000_000, 8, 20))
        print(f"Pierna, vista {view}: SPR en la zona de hueso (T < 0.2) = {spr(rr[view], 0, 0.2):.3f}, en tejido "
              f"blando (0.2 < T < 0.5) = {spr(rr[view], 0.2, 0.5):.3f} ({time.time()-t:.1f} s)", flush=True)
    # contraste de la fractura (hueco): perfil a lo largo del hueso con y sin scatter
    ap = rr["ap"]
    li_p, li_s = -np.log(ap.primary[0]), -np.log(ap.intensity[0])
    col = int(np.argmax(li_p.mean(axis=0)))
    j0 = int(np.argmin(np.abs(geol.v)))
    win = slice(j0 - 3, j0 + 4)
    contrast = [float(np.r_[pr[j0 - 40:j0 - 15], pr[j0 + 15:j0 + 40]].mean() - pr[win].min())
                for pr in (li_p[:, col], li_s[:, col])]
    print(f"Contraste de la fractura en -ln I (columna de mayor atenuación): sin scatter {contrast[0]:.4f}, "
          f"con scatter {contrast[1]:.4f}")


# --------------------------------------------------------------------------------------
def leg_study(figure: str | None) -> None:
    sec("Pierna 3D: radiografías AP/lateral y TAC de haz cónico")
    frac = TransverseFracture3D(gap=1.2e-3, position=0.0, arc_deg=360.0)
    # --- radiografías (dx = 0.5 mm, panel de 0.25 mm de píxel, ruido de Poisson)
    leg = leg_segment("tibia", frac, length=0.06, dx=0.5e-3)
    print(f"Fantoma de radiografía: {leg.grid.shape} celdas de 0.5 mm ({leg.grid.size/1e6:.2f} M)")
    panel = x3.FlatPanel(x3.DetectorModel(pixel_pitch=0.25e-3, blur_sigma=0.1e-3, electronic_noise=3.0))
    out = {}
    for view in ("ap", "lateral"):
        t = time.time()
        out[view] = x3.radiograph(leg, view, photons=2e4, detector=panel, rng=np.random.default_rng(0))
        print(f"radiografía {view}: {out[view].line_integrals.shape[1:]} px, {time.time()-t:.2f} s (sin scatter)")
    sc = sc3.Scatter3DOptions(4_000_000, 8, 20)
    t = time.time()
    rs = x3.radiograph(leg, "ap", photons=2e4, detector=panel, rng=np.random.default_rng(0), scatter=sc)
    print(f"radiografía ap con scatter MC (4·10⁶ fotones): {time.time()-t:.1f} s")
    # --- TAC cónico 128³
    g = Grid3D(128, 128, 128, 0.4e-3)
    ct = leg_segment("tibia", frac, length=51.2e-3, dx=0.4e-3, width=51.2e-3, depth=51.2e-3)
    assert ct.grid.shape == g.shape, ct.grid.shape
    geo = x3.Geometry3D.covering(ct.grid, 0.5, 0.9, oversample=1.0)
    n_ang = 360
    t = time.time()
    res = x3.ct_scan3d(ct, n_ang, geo, photons=2e5, rng=np.random.default_rng(1))
    t_ct = time.time() - t
    mem = rss_mb()
    mu_w = float(xd.linear_attenuation("water", float(np.clip(res.projections.spectrum.mean_energy, 10, 10000))))
    truth = ct.property_map(lambda k: float(xd.linear_attenuation(k, res.projections.spectrum.mean_energy)),
                            dtype=np.float64)
    hu_true = xd.hounsfield(truth, mu_w, 0.0)
    def erode(m, k=2):
        """Erosión en el plano (y, z): descarta las celdas a menos de k celdas de otro tejido (volumen parcial)."""
        out = m.copy()
        for dz in range(-k, k + 1):
            for dy in range(-k, k + 1):
                out &= np.roll(np.roll(m, dz, axis=0), dy, axis=1)
        return out

    away = np.abs(ct.grid.x) > 8e-3  # fuera de la zona de la fractura y sus bandas
    away[:12] = away[-12:] = False
    for name, mask in (("agua (fondo)", ct.mask("water")), ("músculo (anillo de 2 mm)", ct.mask("muscle")),
                       ("grasa", ct.mask("fat")), ("cortical", ct.mask("cortical_bone"))):
        m = erode(mask, 1 if "músculo" in name else 2) & away[None, None, :]
        print(f"  {name}: HU medio {res.hu[m].mean():8.1f} (modelo {hu_true[m].mean():8.1f}), "
              f"σ = {res.hu[m].std():6.1f} HU, n = {int(m.sum())}")
    print(f"TAC cónico 128³ ({n_ang} vistas, panel {geo.n_u}×{geo.n_v}, sid {geo.sid} m, sdd {geo.sdd} m, 120 kVp): "
          f"{t_ct:.0f} s con 1 hilo, memoria máxima {mem:.0f} MB")
    t = time.time()
    st = x3.fdk(res.projections.line_integrals, np.arange(n_ang) * 360.0 / n_ang, geo, ct.grid, mode="stack")
    print(f"(para comparar) apilado de abanico sobre las mismas proyecciones: {time.time()-t:.1f} s")
    del st
    if figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 3, figsize=(13, 8.6), constrained_layout=True)
        for a, view, ttl in ((axs[0, 0], "ap", "Radiografía AP"), (axs[0, 1], "lateral", "Radiografía lateral")):
            r = out[view]
            ext = [r.geometry.u[0] * 1e3, r.geometry.u[-1] * 1e3, r.geometry.v[-1] * 1e3, r.geometry.v[0] * 1e3]
            a.imshow(r.line_integrals[0], cmap="gray_r", extent=ext, aspect="equal")
            a.set_title(ttl + " (60 kVp, con ruido)")
            a.set_xlabel("u (mm, en el detector)")
            a.set_ylabel("v = x (mm)")
        li_p, li_s = -np.log(rs.primary[0]), -np.log(rs.primary[0] + rs.scatter[0])
        col = int(np.argmax(li_p.mean(axis=0)))
        vv = rs.geometry.v * 1e3
        axs[0, 2].plot(vv, li_p[:, col], label="primaria")
        axs[0, 2].plot(vv, li_s[:, col], label="primaria + scatter (Monte Carlo)")
        axs[0, 2].set_title("Perfil AP a lo largo del hueso: el scatter\nreduce el contraste de la fractura")
        axs[0, 2].set_xlabel("v = x (mm)")
        axs[0, 2].set_ylabel("-ln(I/I0)")
        axs[0, 2].legend(fontsize=8)
        gx = ct.grid
        ext_yz = [gx.y[0] * 1e3, gx.y[-1] * 1e3, gx.z[-1] * 1e3, gx.z[0] * 1e3]
        ext_xz = [gx.x[0] * 1e3, gx.x[-1] * 1e3, gx.z[-1] * 1e3, gx.z[0] * 1e3]
        ax_i = int(np.argmin(np.abs(gx.x - 12e-3)))
        j0 = ct.grid.ny // 2
        axs[1, 0].imshow(res.hu[:, :, ax_i], cmap="gray", vmin=-200, vmax=1200, extent=ext_yz)
        axs[1, 0].set_title("TAC cónico: corte axial (x = 12 mm), HU")
        axs[1, 0].set_xlabel("y (mm)")
        axs[1, 0].set_ylabel("z (mm)")
        axs[1, 1].imshow(res.hu[:, j0, :], cmap="gray", vmin=-200, vmax=1200, extent=ext_xz, aspect="equal")
        axs[1, 1].set_title("TAC cónico: corte longitudinal (y = 0), con la fractura")
        axs[1, 1].set_xlabel("x (mm)")
        axs[1, 1].set_ylabel("z (mm)")
        axs[1, 2].imshow(res.hu[:, :, ct.grid.nx // 2], cmap="gray", vmin=-200, vmax=1200, extent=ext_yz)
        axs[1, 2].set_title("TAC cónico: corte axial por el hueco de la fractura")
        axs[1, 2].set_xlabel("y (mm)")
        fig.suptitle("Pierna 3D (tibia con fractura transversal de 1.2 mm): rayos X y TAC de haz cónico")
        fig.savefig(figure, dpi=110)
        print(f"figura: {figure}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--figure", default="docs/img/rayos_x3d.png")
    args = ap.parse_args()
    todo = args.only or ["chords", "adjoint", "extruded", "poly", "crack", "fdk", "sart", "scatter", "leg"]
    t0 = time.time()
    if "chords" in todo:
        chords()
    if "adjoint" in todo:
        adjoint()
    if "extruded" in todo:
        extruded()
    if "poly" in todo:
        poly()
    if "crack" in todo:
        crack()
    if "fdk" in todo:
        fdk_study()
    if "sart" in todo:
        sart_study()
    if "scatter" in todo:
        scatter_study()
    if "leg" in todo:
        leg_study(args.figure)
    print(f"\nTiempo total: {time.time()-t0:.0f} s; memoria máxima {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
