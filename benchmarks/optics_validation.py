"""Validación completa del Monte Carlo de fotones (``wavesim.em.montecarlo``) y comparación con la difusión.

Uso (desde la raíz del repositorio, con un solo hilo para medir fotones/s):

    NUMBA_NUM_THREADS=1 python benchmarks/optics_validation.py            # completo, ~10 min con Numba
    NUMBA_NUM_THREADS=1 python benchmarks/optics_validation.py --quick     # 1/10 de fotones, ~1 min

Imprime las tablas de ``docs/validacion/optica.md``. Casos:
  (a) losa de van de Hulst con índice ajustado (R = 0.09739, T = 0.66096);
  (b) losa con n = 1.5: caso absorbente exacto (serie de Fresnel) y caso dispersivo sin referencia verificada;
  (c) atenuación balística con Fresnel;
  (d) pendiente de log(fluencia) frente a mu_eff de la difusión, y discrepancia cerca de la fuente / la frontera;
  (e) la pierna (fantoma ``cross_section_limb("pierna")``) a 800 nm: transmisión MC frente a difusión;
  (f) rendimiento (fotones/s con un hilo) de los dos backends.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from wavesim.core.grid import Grid2D
from wavesim.em import montecarlo as mc
from wavesim.em import optics
from wavesim.materials import optical
from wavesim.phantoms import TISSUE_INDEX, Phantom, cross_section_limb

VDH = dict(mua=1000.0, mus=9000.0, g=0.75, thickness=2e-4)  # 10 /cm, 90 /cm, 0.02 cm
R_VDH, T_VDH = 0.09739, 0.66096  # van de Hulst (1980), tabla 35


def series(r_f, mu, d):
    e = np.exp(-mu * d)
    return (1 - r_f) ** 2 * e / (1 - r_f**2 * e**2)


def line(name, got, sem, ref, tol_note=""):
    z = (got - ref) / sem if sem > 0 else float("nan")
    print(f"| {name} | {got:.5f} ± {sem:.5f} | {ref:.5f} | {got - ref:+.5f} ({z:+.1f} σ) | {tol_note} |")


def caso_a(n):
    print("\n## (a) Losa de van de Hulst, n = 1 dentro y fuera (n fotones =", n, ")")
    print("| magnitud | MC (± error estándar) | referencia | diferencia | |\n|---|---|---|---|---|")
    m = mc.slab_medium(**VDH, dx=5e-6)
    for be in ("numba", "numpy"):
        nn = n if be == "numba" else n // 20
        try:
            r = mc.simulate(m, nn, periodic_x=True, backend=be, seed=2024, n_batches=40)
        except ImportError:
            continue
        line(f"R difusa ({be}, {nn:.0e})", r.reflectance, r.sem["reflectance"], R_VDH)
        line(f"T total ({be}, {nn:.0e})", r.transmittance, r.sem["transmittance"], T_VDH)
        print(f"|  balance ({be}) | {r.balance:.7f} | 1 | | |")


def caso_b(n):
    print("\n## (b) Desajuste de índice, n = 1.5 en la losa (exterior n = 1)")
    print("| caso | magnitud | MC | referencia | diferencia |\n|---|---|---|---|---|")
    a, d, nn = 400.0, 1e-3, 1.5
    rf = ((nn - 1) / (nn + 1)) ** 2
    e = np.exp(-a * d)
    m = mc.slab_medium(a, 0.0, 0.0, d, n=nn, dx=d / 20)
    r = mc.simulate(m, n // 20, periodic_x=True, seed=5)
    t_ref = series(rf, a, d)
    r_ref = rf * (1 - rf) ** 2 * e**2 / (1 - rf**2 * e**2)
    print(f"| absorbente puro (exacto) | T | {r.transmittance:.7f} | {t_ref:.7f} | {r.transmittance - t_ref:+.1e} |")
    print(f"| absorbente puro (exacto) | R difusa | {r.reflectance:.7f} | {r_ref:.7f} | {r.reflectance - r_ref:+.1e} |")
    print(f"| absorbente puro (exacto) | R especular | {r.specular:.7f} | {rf:.7f} | {r.specular - rf:+.1e} |")
    rd = mc.simulate(mc.slab_medium(**VDH, n=1.5, dx=5e-6), n, periodic_x=True, seed=6, n_batches=40)
    print(f"| van de Hulst con n=1.5 (SIN referencia verificada) | R especular | {rd.specular:.5f} | {rf:.5f} | |")
    print(f"| van de Hulst con n=1.5 (SIN referencia verificada) | R difusa | {rd.reflectance:.5f} ± "
          f"{rd.sem['reflectance']:.5f} | n/d | |")
    print(f"| van de Hulst con n=1.5 (SIN referencia verificada) | T total | {rd.transmittance:.5f} ± "
          f"{rd.sem['transmittance']:.5f} | n/d | |")
    print(f"| van de Hulst con n=1.5 (SIN referencia verificada) | absorbida | {rd.absorbed_total:.5f} ± "
          f"{rd.sem['absorbed_total']:.5f} | n/d | |")
    print(f"| van de Hulst con n=1.5 (SIN referencia verificada) | balance | {rd.balance:.7f} | 1 | |")


def caso_c(n):
    print("\n## (c) Atenuación balística (fotones transmitidos sin dispersarse nunca)")
    print("| n losa | mu_a, mu_s [1/m], d | MC (± e.e.) | serie de Fresnel exp(-mu_t d) | diferencia |\n"
          "|---|---|---|---|---|")
    for nn in (1.0, 1.5):
        for a, s, d in ((300.0, 500.0, 2e-3), (50.0, 2000.0, 2e-3)):
            m = mc.slab_medium(a, s, 0.8, d, n=nn, dx=d / 40)
            r = mc.simulate(m, n, periodic_x=True, seed=3, n_batches=40)
            ref = series(((nn - 1) / (nn + 1)) ** 2, a + s, d)
            z = (r.ballistic_transmittance - ref) / r.sem["ballistic_transmittance"]
            print(f"| {nn} | {a:g}, {s:g}, {d * 1e3:g} mm | {r.ballistic_transmittance:.6f} ± "
                  f"{r.sem['ballistic_transmittance']:.6f} | {ref:.6f} | {r.ballistic_transmittance - ref:+.1e} "
                  f"({z:+.1f} σ) |")


def slab_fantoma(key, dx, width, depth, rim=5):
    g = Grid2D(width + 2 * rim, depth + 2 * rim, dx)
    lab = np.full(g.shape, TISSUE_INDEX["air"])
    lab[rim:-rim, rim:-rim] = TISSUE_INDEX[key]
    return Phantom(g, lab), rim


def caso_d(n):
    print("\n## (d) Monte Carlo frente a difusión, medio difusivo (800 nm)")
    lam, dx = 800.0, 0.5e-3
    print("\n### Pendiente de log(fluencia) lejos de fuente y fronteras (haz ancho; columnas centrales, promediadas)")
    print("| tejido | mu_a/mu_s' | ventana de profundidad [mm] | pendiente MC / mu_eff - 1 "
          "| pendiente difusión / mu_eff - 1 |"
          "\n|---|---|---|---|---|")
    saved = {}
    for key in ("fat", "muscle"):
        p, rim = slab_fantoma(key, dx, 160, 100)
        med = mc.medium_from_phantom(p, lam)
        t0 = time.perf_counter()
        r = mc.simulate(med, n // 4, seed=11, n_batches=40)
        dif = optics.solve_diffusion(p, lam)
        saved[key] = (p, r, dif, rim)
        meff = float(optical.mu_effective(key, lam))
        ratio = float(optical.mu_a(key, lam) / optical.mu_s_reduced(key, lam))
        z = (np.arange(p.grid.nz) - rim + 0.5) * dx
        cols = slice(rim + 60, rim + 100)
        for lo, hi in ((8e-3, 18e-3), (10e-3, 25e-3), (15e-3, 30e-3)):
            sel = (z > lo) & (z < hi)
            out = []
            for f in (r.fluence, dif.fluence):
                prof = f[:, cols].mean(axis=1)
                out.append(-np.polyfit(z[sel], np.log(prof[sel]), 1)[0] / meff - 1)
            print(f"| {key} | {ratio:.3f} | {lo * 1e3:g}-{hi * 1e3:g} | {out[0]:+.2%} | {out[1]:+.2%} |")
        print(f"|  ({key}: mu_eff = {meff / 100:.3f} /cm, {r.n_photons:.0e} fotones, "
              f"{time.perf_counter() - t0:.0f} s) | | | | |")

    p, r, dif, rim = saved["fat"]
    z = (np.arange(p.grid.nz) - rim + 0.5) * dx
    cols = slice(rim + 60, rim + 100)
    print("\n### Cerca de la cara iluminada (grasa, haz ancho): fluencia MC / difusión por profundidad")
    print("| profundidad [mm] | MC | difusión | MC / difusión |\n|---|---|---|---|")
    for dmm in (0.25, 0.75, 1.25, 2.25, 3.25, 5.25, 10.25):
        j = int(round(dmm * 1e-3 / dx - 0.5)) + rim
        a, b = r.fluence[j, cols].mean(), dif.fluence[j, cols].mean()
        print(f"| {dmm:g} | {a:.4f} | {b:.4f} | {a / b:.3f} |")
    print(f"\nReflectancia difusa MC (grasa, 800 nm): {r.reflectance:.4f} ± {r.sem['reflectance']:.4f}; "
          f"fracción absorbida {r.absorbed_total:.4f}")

    print("\n### Cerca de una fuente puntual (fibra isótropa a 10 mm de profundidad; grasa)")
    pf, rimf = slab_fantoma("fat", dx, 100, 60)
    xs, zs = pf.grid.x[rimf + 50], pf.grid.z[rimf + 20]
    med = mc.medium_from_phantom(pf, lam)
    rf = mc.simulate(med, n // 4, source="isotropic", source_xz=(xs, zs), seed=12, n_batches=40)
    df = optics.solve_diffusion(pf, lam, source=(xs, zs))
    jf, i0 = pf.grid.index_of(xs, zs)
    print("| distancia a la fuente [mm] (en x) | MC [1/m] | difusión [1/m] | MC / difusión | error estándar MC |"
          "\n|---|---|---|---|---|")
    for k in (1, 2, 4, 6, 10, 16, 24):
        a, b = rf.fluence[jf, i0 + k], df.fluence[jf, i0 + k]
        print(f"| {k * dx * 1e3:.1f} | {a:.2f} | {b:.2f} | {a / b:.3f} | {rf.fluence_sem[jf, i0 + k] / a:.1%} |")


def caso_e(n):
    print("\n## (e) La pierna a 800 nm: transmisión total MC frente a difusión")
    dx = 1e-3
    p = cross_section_limb("pierna", dx=dx, seed=1).replace_tissue("water", "air")
    dif = optics.solve_diffusion(p, 800.0)
    med = mc.medium_from_phantom(p, 800.0)
    nruns = 5
    t0 = time.perf_counter()
    rs = [mc.simulate(med, n // nruns, seed=100 + k, n_batches=20) for k in range(nruns)]
    secs = time.perf_counter() - t0
    thick = (p.labels != TISSUE_INDEX["air"]).sum(axis=0) * dx
    cols = np.nonzero(thick > 0)[0]
    print(f"Malla {p.grid.shape}, dx = {dx * 1e3:g} mm, {n:.0e} fotones en {nruns} corridas independientes "
          f"({secs:.0f} s).")
    print("\n| columnas (espesor de tejido) | MC T (media ± e.e. entre corridas) | difusión T | MC / difusión |"
          "\n|---|---|---|---|")
    for lo, hi in ((0, 1e9), (0, 5e-3), (5e-3, 20e-3), (20e-3, 40e-3), (40e-3, 1e9)):
        sel = cols[(thick[cols] > lo) & (thick[cols] <= hi)]
        if sel.size == 0:
            continue
        mcv = np.array([r.transmittance_profile[sel].mean() for r in rs])
        d = float(np.mean(dif.transmitted_profile[sel]))
        mean, sem = mcv.mean(), mcv.std(ddof=1) / np.sqrt(nruns)
        txt = f"{mean:.2e} ± {sem:.1e}" if mean > 0 else f"0 fotones de {n:.0e}"
        ratio = f"{mean / d:.2f}" if mean > 0 else "n/d"
        name = f"{lo * 1e3:g}-{hi * 1e3:g} mm" if hi < 1e9 else f"> {lo * 1e3:g} mm"
        if (lo, hi) == (0, 1e9):
            name = "todas"
        print(f"| {name} ({sel.size} col.) | {txt} | {d:.2e} | {ratio} |")
    mean_all = np.mean([r.transmittance for r in rs])
    print(f"\nBalance MC: R = {np.mean([r.reflectance for r in rs]):.4f}, T = {mean_all:.2e}, "
          f"laterales = {np.mean([r.lateral for r in rs]):.4f}, "
          f"absorbida = {np.mean([r.absorbed_total for r in rs]):.4f}, "
          f"especular = {rs[0].specular:.4f}; difusión T = {dif.transmission:.2e} "
          f"(suelo fiable {dif.transmission_floor:.0e}).")
    print(f"Balística (analítica, mu_a + mu_s, columna central): {optics.ballistic_transmission(p, 800.0):.1e}")


def caso_f(n):
    print("\n## (f) Rendimiento con un hilo")
    print("| caso | backend | fotones/s |\n|---|---|---|")
    m = mc.slab_medium(**VDH, dx=5e-6)
    p = cross_section_limb("pierna", dx=1e-3, seed=1).replace_tissue("water", "air")
    med = mc.medium_from_phantom(p, 800.0)
    for name, medium, kw, nn in (("losa van de Hulst (40 celdas)", m, dict(periodic_x=True), n),
                                 ("pierna 800 nm (100x100)", med, {}, max(n // 50, 2000))):
        for be in ("numpy", "numba"):
            try:
                k = nn if be == "numba" else max(nn // 20, 2000)
                mc.simulate(medium, 200, backend=be, seed=0, **kw)  # compilación / calentamiento
                r = mc.simulate(medium, k, backend=be, seed=0, **kw)
            except ImportError:
                continue
            print(f"| {name} | {be} | {r.photons_per_second:,.0f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="un décimo de los fotones")
    args = ap.parse_args()
    scale = 0.1 if args.quick else 1.0
    t0 = time.perf_counter()
    caso_a(int(4_000_000 * scale))
    caso_b(int(2_000_000 * scale))
    caso_c(int(1_000_000 * scale))
    caso_d(int(2_000_000 * scale))
    caso_e(int(2_500_000 * scale))
    caso_f(int(2_000_000 * scale))
    print(f"\nTiempo total: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
