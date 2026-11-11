"""Validación del FDTD de Maxwell (modo TM): banda ancha dispersiva y soluciones analíticas del campo.

Uso (desde la raíz del repositorio; sólo NumPy):

    NUMBA_NUM_THREADS=1 python benchmarks/em_validation.py              # todo (~5-6 min en 1 núcleo)
    python benchmarks/em_validation.py --case fit                       # sólo el ajuste de Debye (~10 s)
    python benchmarks/em_validation.py --case propagation               # beta(f), alpha(f) en músculo y grasa (~10 s)
    python benchmarks/em_validation.py --case slab                      # losa piel/grasa/músculo/hueso (~40 s)
    python benchmarks/em_validation.py --case mie                       # cilindro: Bessel, Mie, convergencia (~4.5 min)
    python benchmarks/em_validation.py --case mie --quick               # cilindro sin las mallas más finas

Casos (las tablas de ``docs/validacion/maxwell.md`` salen de aquí):

1. ``fit``: error relativo máximo de ``eps'`` y ``sigma`` del modelo de Debye
   (polos comunes) frente al Cole-Cole de Gabriel en 0.1-10 GHz para cada
   tejido y para las mezclas porosas de Maxwell Garnett (tabla de porosidad
   interpolada, comparada con la mezcla exacta en porosidades aleatorias), y
   número de polos en otras bandas.
2. ``propagation``: medio homogéneo (músculo, grasa), un solo pulso de
   0.1-10 GHz con 20 puntos por longitud de onda a 10 GHz. Constante de fase
   y atenuación ajustadas a lo largo de z a 21 frecuencias frente a las del
   Cole-Cole (``dielectric.wave_numbers``).
3. ``slab``: losa piel 2 mm / grasa 5 mm / músculo 15 mm / cortical 6 mm en
   aire, incidencia normal. Coeficiente de transmisión complejo (campo detrás
   de la losa dividido por el de una simulación sin losa) frente a la matriz
   de transferencia exacta (``em.analytic.insertion_transmission``): banda
   estrecha a 128 MHz, 915 MHz y 2.45 GHz; banda ancha 0.1-10 GHz con
   Gabriel y con el propio ajuste de Debye (separa el error del FDTD del
   error del ajuste). Dos tamaños de celda.
4. ``mie``: funciones de Bessel frente a valores tabulados; cilindro de
   músculo y de cortical (radio 20 mm) en aire a 2.45 GHz, onda plana TM;
   error L2 relativo del campo dentro y fuera (en un cuadro de lado
   ``2 (a + lambda0/4)``) frente a la serie de Mie, con y sin promedio
   sub-celda, de 10 a 40 puntos por longitud de onda en el cilindro; barrido
   del radio a 15 puntos por longitud de onda; y un cilindro de músculo con
   el solver de banda ancha (0.5-3 GHz).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavesim.core.grid import Grid2D  # noqa: E402
from wavesim.em import analytic, bessel, subpixel  # noqa: E402
from wavesim.em.fdtd_dispersive import BroadbandMaxwellTMSolver  # noqa: E402
from wavesim.em.fdtd_tm import MaxwellTMSolver  # noqa: E402
from wavesim.materials import dielectric, dispersive  # noqa: E402
from wavesim.materials.tissues import tissue_keys  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402

C0 = dielectric.C0
NPML = 20
BAND = (1e8, 1e10)
AIR = TISSUE_INDEX["air"]


def column(dx: float, labels_z, z0: float = 0.0) -> Phantom:
    """Malla 1D (4 columnas, la CPML lateral no actúa sobre una onda plana uniforme en x)."""
    g = Grid2D(4, len(labels_z), dx, x0=-2 * dx, z0=z0)
    return Phantom(g, np.repeat(np.asarray(labels_z, np.int16)[:, None], 4, axis=1))


def db(x):
    return 20.0 * np.log10(np.abs(x))


# --------------------------------------------------------------------------
# 1. Ajuste de Debye
# --------------------------------------------------------------------------
def case_fit() -> None:
    t0 = time.time()
    model = dispersive.tissue_debye_model(BAND)
    print(f"\n## Ajuste de Debye en {BAND[0]/1e9:g}-{BAND[1]/1e9:g} GHz: {model.n_poles} polos comunes "
          f"({time.time() - t0:.1f} s)")
    print("Frecuencias de relajación f_k = 1/(2 pi tau_k) [GHz]:",
          ", ".join(f"{1 / (2 * np.pi * t) / 1e9:.4g}" for t in model.tau))
    print("\n| Tejido | eps_inf | sigma (S/m) | error máx. eps' | error máx. sigma |")
    print("|---|---|---|---|---|")
    for k in tissue_keys():
        m, (ee, es) = model.tissues[k], model.errors[k]
        print(f"| {k} | {m.eps_inf:.2f} | {m.sigma:.3f} | {ee * 100:.2f} % | {es * 100:.2f} % |")
    print(f"\nMáximo: {model.max_error * 100:.2f} %")

    print("\n### Celdas porosas (Maxwell Garnett): tabla de porosidad interpolada frente a la mezcla exacta")
    print("| Tejido | relleno | porosidades | error máx. eps' | error máx. sigma |")
    print("|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    f = np.geomspace(*BAND, 200)
    for tissue, fill in (("cortical_bone", "blood"), ("cortical_bone", "marrow"), ("trabecular_bone", "blood")):
        phi_tab, par = dispersive.porous_level_table(model, tissue, fill)
        phis = np.concatenate([rng.uniform(0, 1, 40), [0.05, 0.8]])
        exact = dispersive.porous_permittivity(tissue, phis, f, fill)
        worst = np.zeros(2)
        for i, phi in enumerate(phis):
            p = np.array([np.interp(phi, phi_tab, par[:, q]) for q in range(par.shape[1])])
            med = dispersive.DebyeMedium(p[0], p[1], p[2:], model.tau)
            worst = np.maximum(worst, dispersive.fit_errors(med, exact[i], f))
        print(f"| {tissue} | {fill} | {phis.size} aleatorias en [0, 1] | {worst[0] * 100:.2f} % | "
              f"{worst[1] * 100:.2f} % |")

    print("\n### Otras bandas (número de polos elegido para un error < 1 %)")
    print("| Banda | polos | error máx. (todos los tejidos) | tiempo de ajuste |")
    print("|---|---|---|---|")
    for band in ((1e8, 3e9), (5e8, 5e9), (1e9, 2e10), (1e7, 1e10)):
        t0 = time.time()
        m = dispersive.tissue_debye_model(band)
        print(f"| {band[0]/1e9:g}-{band[1]/1e9:g} GHz | {m.n_poles} | {m.max_error * 100:.2f} % | "
              f"{time.time() - t0:.1f} s |")


# --------------------------------------------------------------------------
# 2. Propagación en medio homogéneo
# --------------------------------------------------------------------------
def propagation(tissue: str, band=BAND, ppw: float = 20.0, depth: float | None = None, nf: int = 21):
    n_hi = abs(np.sqrt(complex(dielectric.complex_permittivity(tissue, band[1]))))
    dx = C0 / (band[1] * n_hi * ppw)
    if depth is None:  # unas 13 longitudes de atenuación a f_max: el ajuste no llega al ruido
        depth = min(0.15, 13.0 / float(dielectric.wave_numbers(tissue, band[1])[1]) + 0.01)
    nz = 2 * NPML + 10 + int(depth / dx)
    p = column(dx, np.full(nz, TISSUE_INDEX[tissue]))
    g = p.grid
    t0 = time.time()
    s = BroadbandMaxwellTMSolver(p, band)
    z_src = g.z0 + (NPML + 3) * dx
    s.set_plane_wave_source(z=z_src)
    mon = np.zeros(g.shape, bool)
    mon[:, 2] = True
    freqs = np.geomspace(*band, nf)
    r = s.run(freqs, monitor=mon)
    z = g.z[r.points[0]]
    beta, alpha = dielectric.wave_numbers(tissue, freqs)
    eb, ea = np.empty(nf), np.empty(nf)
    for k in range(nf):
        col = r.phasor[k]
        sel = (z > z_src + 8 * dx) & (z < g.z0 + g.depth - (NPML + 5) * dx)
        sel &= np.abs(col) > 1e-5 * np.abs(col[sel]).max()
        eb[k] = abs(np.polyfit(z[sel], np.unwrap(np.angle(col[sel])), 1)[0]) / beta[k] - 1
        ea[k] = -np.polyfit(z[sel], np.log(np.abs(col[sel])), 1)[0] / alpha[k] - 1
    return dict(freqs=freqs, err_beta=eb, err_alpha=ea, dx=dx, nz=nz, steps=r.steps, poles=s.j_pol.shape[0],
                time=time.time() - t0)


def case_propagation() -> None:
    print(f"\n## Propagación en medio homogéneo, pulso de {BAND[0]/1e9:g}-{BAND[1]/1e9:g} GHz, "
          "20 puntos/lambda a f_max")
    res = {t: propagation(t) for t in ("muscle", "fat")}
    for t, r in res.items():
        print(f"{t}: dx = {r['dx']*1e3:.3f} mm, {r['nz']} celdas, {r['poles']} polos, {r['steps']} pasos, "
              f"{r['time']:.1f} s")
    print("\n| f (GHz) | músculo: error beta | músculo: error alpha | grasa: error beta | grasa: error alpha |")
    print("|---|---|---|---|---|")
    m, f_ = res["muscle"], res["fat"]
    for k, f in enumerate(m["freqs"]):
        if k % 2 == 0 or k == len(m["freqs"]) - 1:
            print(f"| {f/1e9:.3g} | {m['err_beta'][k]*100:+.2f} % | {m['err_alpha'][k]*100:+.2f} % | "
                  f"{f_['err_beta'][k]*100:+.2f} % | {f_['err_alpha'][k]*100:+.2f} % |")
    for t, r in res.items():
        print(f"{t}: máx |error beta| = {np.abs(r['err_beta']).max()*100:.2f} %, "
              f"máx |error alpha| = {np.abs(r['err_alpha']).max()*100:.2f} %  (tolerancia 2 % y 5 %)")


# --------------------------------------------------------------------------
# 3. Losa multicapa
# --------------------------------------------------------------------------
SLAB = [("skin", 2e-3), ("fat", 5e-3), ("muscle", 15e-3), ("cortical_bone", 6e-3)]


def slab_phantoms(dx: float):
    gap = 20
    cells = [int(round(d / dx)) for _, d in SLAB]
    lab = [AIR] * (NPML + gap)
    for (k, _), n in zip(SLAB, cells):
        lab += [TISSUE_INDEX[k]] * n
    j_obs = len(lab) + gap // 2
    lab += [AIR] * (gap + NPML + 10)
    return column(dx, lab), column(dx, [AIR] * len(lab)), j_obs, [(k, n * dx) for (k, _), n in zip(SLAB, cells)]


def slab_narrowband(dx: float, f0: float):
    slab, ref, j_obs, layers = slab_phantoms(dx)
    z_src = slab.grid.z0 + (NPML + 3) * dx
    out = []
    for p in (slab, ref):
        s = MaxwellTMSolver(p, f0)
        s.set_plane_wave_source(z=z_src)
        out.append(s.run().phasor[j_obs, 2])
    t = out[0] / out[1]
    exact = complex(analytic.insertion_transmission(layers, f0))
    return t, exact


def slab_broadband(dx: float, band=BAND, nf: int = 25):
    slab, ref, j_obs, layers = slab_phantoms(dx)
    z_src = slab.grid.z0 + (NPML + 3) * dx
    mon = np.zeros(slab.grid.shape, bool)
    mon[j_obs, 2] = True
    freqs = np.geomspace(*band, nf)
    model = dispersive.tissue_debye_model(band)
    out = []
    for p in (slab, ref):
        s = BroadbandMaxwellTMSolver(p, band, model=model)
        s.set_plane_wave_source(z=z_src)
        out.append(s.run(freqs, monitor=mon).phasor[:, 0])
    t = out[0] / out[1]
    gab = analytic.insertion_transmission(layers, freqs)
    deb = analytic.insertion_transmission(layers, freqs,
                                          permittivity=lambda k, ff: model.tissues[k].complex_permittivity(ff))
    return freqs, t, gab, deb


def case_slab() -> None:
    print("\n## Losa piel 2 / grasa 5 / músculo 15 / cortical 6 mm en aire, incidencia normal")
    print("\n### Banda estrecha (onda continua)")
    print("| f | dx | |T| FDTD | |T| exacto | error |T| | error de fase |")
    print("|---|---|---|---|---|---|")
    for dx in (0.5e-3, 0.25e-3):
        for f0 in (128e6, 915e6, 2.45e9):
            t0 = time.time()
            t, ex = slab_narrowband(dx, f0)
            print(f"| {f0/1e9:g} GHz | {dx*1e3:g} mm | {db(t):.3f} dB | {db(ex):.3f} dB | {db(t / ex):+.3f} dB | "
                  f"{np.degrees(np.angle(t / ex)):+.2f}° |  ({time.time() - t0:.1f} s)")
    print(f"\n### Banda ancha {BAND[0]/1e9:g}-{BAND[1]/1e9:g} GHz (un pulso por malla)")
    for dx in (0.25e-3, 0.125e-3):
        t0 = time.time()
        freqs, t, gab, deb = slab_broadband(dx)
        print(f"\ndx = {dx*1e3:g} mm ({time.time() - t0:.1f} s)")
        print("| f (GHz) | |T| FDTD | |T| Gabriel | error frente a Gabriel | error frente a Debye "
              "| fase frente a Gabriel |")
        print("|---|---|---|---|---|---|")
        for k, f in enumerate(freqs):
            if k % 3 == 0 or k == len(freqs) - 1:
                print(f"| {f/1e9:.3g} | {db(t[k]):.2f} dB | {db(gab[k]):.2f} dB | {db(t[k] / gab[k]):+.3f} dB | "
                      f"{db(t[k] / deb[k]):+.3f} dB | {np.degrees(np.angle(t[k] / gab[k])):+.2f}° |")
        print(f"máx |error| frente a Gabriel: {np.abs(db(t / gab)).max():.3f} dB y "
              f"{np.abs(np.degrees(np.angle(t / gab))).max():.2f}°; "
              f"frente al Debye: {np.abs(db(t / deb)).max():.3f} dB")


# --------------------------------------------------------------------------
# 4. Bessel y cilindro
# --------------------------------------------------------------------------
# Abramowitz y Stegun (1964), tablas 9.1, 9.2 y 9.8 (I_n: J_n(ix) = i^n I_n(x)), 10 decimales
BESSEL_TABLE = [
    ("J", 0, 1.0, 0.7651976866), ("J", 1, 1.0, 0.4400505857), ("J", 2, 1.0, 0.1149034849),
    ("Y", 0, 1.0, 0.0882569642), ("Y", 1, 1.0, -0.7812128213), ("Y", 2, 1.0, -1.6506826068),
    ("J", 0, 5.0, -0.1775967713), ("Y", 0, 5.0, -0.3085176252),
    ("J", 0, 10.0, -0.2459357645), ("J", 1, 10.0, 0.0434727462), ("J", 5, 10.0, -0.2340615282),
    ("Y", 0, 10.0, 0.0556711673), ("Y", 1, 10.0, 0.2490154242), ("Y", 5, 10.0, 0.1354030477),
]


def cylinder_grid(dx: float, a: float, f: float):
    half = a + max(0.25 * C0 / f, 0.5 * a)
    n_half = int(np.ceil(half / dx))
    n = 2 * n_half + 1 + 2 * NPML
    g = Grid2D(n, n + 6, dx, x0=-(n_half + NPML) * dx, z0=-(n_half + NPML + 6) * dx)
    return g, half


def disk_labels(g: Grid2D, a: float, tissue: str) -> np.ndarray:
    xx, zz = g.meshgrid()
    return np.where(np.hypot(xx, zz) <= a, TISSUE_INDEX[tissue], AIR).astype(np.int16)


def cylinder_narrowband(tissue: str, f: float, a: float, ppw: float, smooth: bool, factor: int = 9):
    eps = complex(dielectric.complex_permittivity(tissue, f))
    dx = C0 / f / abs(np.sqrt(eps)) / ppw
    g, half = cylinder_grid(dx, a, f)
    eps_map = None
    if smooth:
        gf = subpixel.fine_grid(g, factor)
        eps_map = subpixel.smoothed_permittivity(Phantom(gf, disk_labels(gf, a, tissue)), factor, f)
    z_src = g.z0 + (NPML + 2) * dx
    # Régimen permanente bien asentado: el de por defecto deja ~0.5 % de error fuera del cilindro
    settle = 40 / f + 3.0 * 2 * g.depth * abs(np.sqrt(eps)) / C0
    s = MaxwellTMSolver(Phantom(g, disk_labels(g, a, tissue)), f, eps_map=eps_map)
    s.set_plane_wave_source(z=z_src)
    e = s.run(n_periods_steady=8, settle_time=settle).phasor
    ref = MaxwellTMSolver(column(dx, [AIR] * g.nz, g.z0), f)
    ref.set_plane_wave_source(z=z_src)
    e = e / ref.run(n_periods_steady=8, settle_time=settle).phasor[g.index_of(0.0, 0.0)[0], 2]
    xx, zz = g.meshgrid()
    box = (np.abs(xx) <= half) & (np.abs(zz) <= half)
    mie = analytic.mie_cylinder_tm(xx[box], zz[box], a, eps, f)
    inside = np.hypot(xx, zz)[box] <= a
    err = [float(np.linalg.norm(e[box][m] - mie[m]) / np.linalg.norm(mie[m])) for m in (inside, ~inside)]
    return err[0], err[1], dx, g.shape


def cylinder_broadband(tissue: str = "muscle", a: float = 0.015, band=(5e8, 3e9), ppw: float = 20.0, nf: int = 6):
    model = dispersive.tissue_debye_model(band)
    n_hi = abs(np.sqrt(complex(dielectric.complex_permittivity(tissue, band[1]))))
    dx = C0 / band[1] / n_hi / ppw
    g, half = cylinder_grid(dx, a, band[1])
    xx, zz = g.meshgrid()
    box = (np.abs(xx) <= half) & (np.abs(zz) <= half)
    z_src = g.z0 + (NPML + 2) * dx
    freqs = np.geomspace(*band, nf)
    s = BroadbandMaxwellTMSolver(Phantom(g, disk_labels(g, a, tissue)), band, model=model)
    s.set_plane_wave_source(z=z_src)
    r = s.run(freqs, monitor=box)
    ref = BroadbandMaxwellTMSolver(column(dx, [AIR] * g.nz, g.z0), band, model=model)
    ref.set_plane_wave_source(z=z_src)
    mon = np.zeros((g.nz, 4), bool)
    mon[g.index_of(0.0, 0.0)[0], 2] = True
    e_inc = ref.run(freqs, monitor=mon).phasor[:, 0]
    inside = np.hypot(xx, zz)[box] <= a
    rows = []
    for k, f in enumerate(freqs):
        e = r.phasor[k] / e_inc[k]
        errs = []
        for eps in (complex(dielectric.complex_permittivity(tissue, f)),
                    complex(model.tissues[tissue].complex_permittivity(f))):
            mie = analytic.mie_cylinder_tm(xx[box], zz[box], a, eps, f)
            errs += [float(np.linalg.norm(e[m] - mie[m]) / np.linalg.norm(mie[m])) for m in (inside, ~inside)]
        rows.append((f, *errs))
    return rows, dx, g.shape, r.steps


def case_mie(quick: bool = False) -> None:
    print("\n## Funciones de Bessel frente a Abramowitz y Stegun (tablas 9.1, 9.2)")
    worst = 0.0
    for kind, n, x, v in BESSEL_TABLE:
        got = float(np.real(bessel.besselj(n, x) if kind == "J" else bessel.bessely(n, x)))
        worst = max(worst, abs(got - v))
    print(f"{len(BESSEL_TABLE)} valores (J_0-J_5, Y_0-Y_5 en x = 1, 5, 10): diferencia máxima {worst:.1e} "
          "(la tabla tiene 10 decimales)")
    x = np.geomspace(0.01, 150, 300)
    j, y = bessel.besselj_orders(41, x).real, bessel.bessely_orders(41, x)
    w = max(float(np.abs((j[n + 1] * y[n] - j[n] * y[n + 1]) * np.pi * x / 2 - 1).max()) for n in range(41))
    print(f"Wronskiano J_(n+1) Y_n - J_n Y_(n+1) = 2/(pi x), n = 0..40, x = 0.01..150: error relativo máx. {w:.1e}")

    f = 2.45e9
    ppws = (10, 15, 20, 30) if quick else (10, 15, 20, 30, 40)
    for tissue in ("muscle", "cortical_bone"):
        print(f"\n### Cilindro de {tissue}, radio 20 mm, en aire a {f/1e9:g} GHz (modo TM)")
        print("| puntos/lambda | dx | malla | escalera: dentro | escalera: fuera | sub-celda: dentro "
              "| sub-celda: fuera |")
        print("|---|---|---|---|---|---|---|")
        res = []
        for ppw in ppws:
            t0 = time.time()
            p_in, p_out, dx, shape = cylinder_narrowband(tissue, f, 0.02, ppw, False)
            s_in, s_out, _, _ = cylinder_narrowband(tissue, f, 0.02, ppw, True)
            res.append((dx, p_in, p_out, s_in, s_out))
            print(f"| {ppw} | {dx*1e3:.3f} mm | {shape[0]}x{shape[1]} | {p_in*100:.2f} % | {p_out*100:.2f} % | "
                  f"{s_in*100:.2f} % | {s_out*100:.2f} % |  ({time.time() - t0:.0f} s)")
        res = np.array(res)
        names = ((1, "escalera dentro"), (3, "sub-celda dentro"), (2, "escalera fuera"), (4, "sub-celda fuera"))
        for col, name in names:
            order = np.polyfit(np.log(res[:, 0]), np.log(res[:, col]), 1)[0]
            print(f"orden observado ({name}): {order:.2f}")

    print("\n### Barrido del radio: músculo a 2.45 GHz, 15 puntos/lambda (error dentro del cilindro)")
    print("| radio | escalera | sub-celda |")
    print("|---|---|---|")
    radii = (0.012, 0.016, 0.020) if quick else (0.012, 0.014, 0.016, 0.018, 0.020, 0.022)
    for a in radii:
        p_in = cylinder_narrowband("muscle", f, a, 15, False)[0]
        s_in = cylinder_narrowband("muscle", f, a, 15, True)[0]
        print(f"| {a*1e3:g} mm | {p_in*100:.2f} % | {s_in*100:.2f} % |")

    print("\n### Cilindro de músculo (radio 15 mm) con el solver de banda ancha, 0.5-3 GHz, 20 puntos/lambda a 3 GHz")
    t0 = time.time()
    rows, dx, shape, steps = cylinder_broadband()
    print(f"dx = {dx*1e3:.3f} mm, malla {shape[0]}x{shape[1]}, {steps} pasos, {time.time() - t0:.0f} s")
    print("| f (GHz) | Mie con Gabriel: dentro | fuera | Mie con el ajuste de Debye: dentro | fuera |")
    print("|---|---|---|---|---|")
    for f_, gi, go, di, do in rows:
        print(f"| {f_/1e9:.3g} | {gi*100:.2f} % | {go*100:.2f} % | {di*100:.2f} % | {do*100:.2f} % |")


CASES = {"fit": case_fit, "propagation": case_propagation, "slab": case_slab, "mie": case_mie}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=list(CASES) + ["all"], default="all")
    ap.add_argument("--quick", action="store_true", help="cilindro sin las mallas más finas")
    args = ap.parse_args()
    t0 = time.time()
    for name, fn in CASES.items():
        if args.case in (name, "all"):
            fn(args.quick) if name == "mie" else fn()
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
