"""Atenuación de ley de potencia del solver elástico: modelo, simulación y coste.

Uso (desde la raíz del repositorio; sólo numpy, y numba opcional):

    NUMBA_NUM_THREADS=1 python benchmarks/attenuation_power_law.py            # todo (~3 min con Numba)
    python benchmarks/attenuation_power_law.py --parte modelo                  # sólo el modelo analítico (segundos)
    python benchmarks/attenuation_power_law.py --parte simulacion              # onda plana medida (~1 min)
    python benchmarks/attenuation_power_law.py --parte tiempo                  # coste de simulate_bmode (~2 min)
    python benchmarks/attenuation_power_law.py --disenar                       # re-optimiza MECHANISM_DESIGN (~1 min)

Partes:

1. ``modelo``: error relativo máximo de alpha(f) del GSLS frente a la ley
   ``alpha0 f^y`` en [0.5, 1.8]·f0 y [0.3, 2.5]·f0 para y = 1.0, 1.1, 1.5 y
   2.0 y para los tejidos de la tabla; velocidad del pulso no relajado (fija
   el paso de tiempo) y dispersión frente a Kramers-Kronig. Compara con el
   "método tau" anterior (un peso común y tiempos de relajación en
   f0/2.5-2.5·f0).
2. ``simulacion``: onda plana en un medio homogéneo (malla de pocas columnas:
   con la fuente en todas ellas el campo no varía en x y la simulación es 1D
   exacta). alpha(f) y c(f) salen del cociente espectral entre dos receptores
   separados 15 mm. Una simulación sin pérdidas idéntica sirve de referencia:
   se resta su atenuación aparente (ventana finita, < 0.02 dB/cm) y su
   velocidad corrige la dispersión numérica de la malla.
3. ``tiempo``: ``simulate_bmode`` de una emisión sobre una tibia con fractura
   de 24×22 mm a 3 MHz (dx = 0.06 mm) con 1, 2 y 3 mecanismos.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavesim.core.grid import Grid2D  # noqa: E402
from wavesim.core.signals import gaussian_pulse  # noqa: E402
from wavesim.materials.acoustic import ACOUSTIC  # noqa: E402
from wavesim.phantoms import Fracture, longitudinal_limb  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402
from wavesim.ultrasound import ElasticMedium, ElasticSolver, Receiver, Source, simulate_bmode  # noqa: E402
from wavesim.ultrasound import attenuation as att  # noqa: E402

DB_PER_NP = 20.0 / np.log(10.0)
F0 = 1e6
NARROW = (0.5, 1.8)
WIDE = (0.3, 2.5)
EXPONENTS = (1.0, 1.1, 1.5, 2.0)
NPML = 10


def alpha_law(a0_db: float, y: float, f) -> np.ndarray:
    """Ley de potencia en Np/m (``a0_db`` en dB/(cm·MHz^y))."""
    return a0_db * (np.asarray(f) / 1e6) ** y * 100.0 / DB_PER_NP


def design(n_mech: int, f0: float, method: str):
    """Tiempos de relajación, banda y pesos de un método: "nuevo" (actual) o "tau" (el anterior)."""
    if method == "nuevo":
        return att.relaxation_times(f0, n_mech), att.fit_band(f0, n_mech)
    freqs = np.array([f0]) if n_mech == 1 else np.geomspace(f0 / 2.5, f0 * 2.5, n_mech)
    return 1.0 / (2.0 * np.pi * freqs), np.linspace(0.5 * f0, 1.8 * f0, 24)


def fit(a0_db, y, c, n_mech, f0=F0, method="nuevo"):
    ts, band = design(n_mech, f0, method)
    if method == "nuevo":
        w = att.mechanism_weights(alpha_law(a0_db, y, band)[None], np.array([c]), band, ts, f0)[0]
    else:
        tau = att.fit_tau(att.target_qinv(alpha_law(a0_db, y, band), c, band), ts, band)
        w = np.full(n_mech, tau)
    return w, ts


def band_freqs(band, f0=F0, n=200):
    return f0 * np.linspace(*band, n)


def model_errors(a0_db, y, c, n_mech, method="nuevo", f0=F0):
    """Error máximo de alpha (banda estrecha y ancha), c_U/c0 y desviación de c respecto a Kramers-Kronig."""
    w, ts = fit(a0_db, y, c, n_mech, f0, method)
    out = {}
    for name, band in (("estrecha", NARROW), ("ancha", WIDE)):
        f = band_freqs(band, f0)
        c_m, a_m = att.model_wave(c, w, ts, f0, f)
        c_kk = att.power_law_phase_speed(c, alpha_law(a0_db, y, 1e6), y, f0, f)
        out[name] = float(np.abs(a_m / alpha_law(a0_db, y, f) - 1).max())
        out["kk_" + name] = float(np.abs(c_m - c_kk).max())
        out["disp_" + name] = (float(c_m[-1] - c_m[0]), float(c_kk[-1] - c_kk[0]))
    out["cu"] = float(np.sqrt(att.relaxed_modulus_factor(w, ts, f0) * (1 + w.sum())))
    return out


# ----------------------------------------------------------------------------
# 1. modelo
# ----------------------------------------------------------------------------
def part_model():
    print("\n=== 1. Modelo analítico: error máximo de alpha(f) frente a la ley de potencia ===")
    print("Medio genérico: c = 1540 m/s, alpha(1 MHz) = 1 dB/cm. Bandas: estrecha [0.5, 1.8]·f0, ancha [0.3, 2.5]·f0")
    print(f"{'método':10s}{'L':>3s}" + "".join(f"{'y=' + str(y):>16s}" for y in EXPONENTS) + "   (estrecha / ancha, %)")
    for method, ls in (("tau", (1, 2, 3)), ("nuevo", (1, 2, 3, 4))):
        for n in ls:
            row = [model_errors(1.0, y, 1540.0, n, method) for y in EXPONENTS]
            print(f"{method:10s}{n:3d}" + "".join(f"{r['estrecha']*100:8.2f}/{r['ancha']*100:6.1f}" for r in row))

    print("\nTejidos de la tabla (onda P y S), método nuevo: error máx. estrecha/ancha (%) y c_U/c0")
    tissues = [(k, s) for k in ("cortical_bone", "trabecular_bone", "muscle", "skin", "fat", "blood", "water")
               for s in ((False, True) if ACOUSTIC[k].c_s > 0 else (False,))]
    print(f"{'tejido':22s}{'y':>5s}" + "".join(f"{'L=' + str(n):>22s}" for n in (1, 2, 3)))
    for k, shear in tissues:
        p = ACOUSTIC[k]
        a0, c = (p.alpha_s, p.c_s) if shear else (p.alpha_p, p.c_p)
        row = [model_errors(a0, p.y, c, n) for n in (1, 2, 3)]
        print(f"{k + (' S' if shear else ' P'):22s}{p.y:5.1f}"
              + "".join(f"{r['estrecha']*100:7.2f}/{r['ancha']*100:6.2f} {r['cu']:6.4f}" for r in row))

    p = ACOUSTIC["cortical_bone"]
    ref = model_errors(p.alpha_p, p.y, p.c_p, 1)["cu"]
    print("\nPaso de tiempo (lo fija la cortical P) respecto al de L = 1 actual: "
          + ", ".join(f"L={n} {method}: {ref / model_errors(p.alpha_p, p.y, p.c_p, n, method)['cu']:.3f}"
                      for method in ("tau", "nuevo") for n in (1, 2, 3)))

    print("\nDispersión: c(fin de banda) - c(inicio) del modelo y de Kramers-Kronig (m/s); máx |c_modelo - c_KK| (m/s)")
    for k, a0, y, c in (("cortical P", p.alpha_p, 1.0, p.c_p), ("genérico y=1.1", 2.0, 1.1, 1580.0),
                        ("genérico y=1.5", 1.0, 1.5, 1540.0), ("genérico y=2.0", 1.0, 2.0, 1540.0)):
        for n in (1, 2, 3):
            r = model_errors(a0, y, c, n)
            print(f"  {k:16s} L={n}: estrecha {r['disp_estrecha'][0]:+7.2f} vs {r['disp_estrecha'][1]:+7.2f} "
                  f"(máx dif {r['kk_estrecha']:.2f}); ancha {r['disp_ancha'][0]:+7.2f} vs {r['disp_ancha'][1]:+7.2f} "
                  f"(máx dif {r['kk_ancha']:.2f})")


# ----------------------------------------------------------------------------
# 2. simulación
# ----------------------------------------------------------------------------
def plane_wave(c, w, ts, ppw=20, dz=15e-3, wave="p", f0=F0, backend="auto"):
    """Onda plana 1D en un medio homogéneo; devuelve (t, trazas de los 2 receptores, separación)."""
    dx = c / (f0 * ppw)
    nx = 2 * NPML + 6
    js = NPML + 6
    j1 = js + int(round(4e-3 / dx))
    j2 = j1 + int(round(dz / dx))
    # Fondo lejos: el registro termina 4 µs después del pulso, antes de que vuelva el reflejo débil de la CPML
    nz = j2 + int(round(4e-6 * c / dx)) + NPML
    g = Grid2D(nx, nz, dx)
    one = np.ones(g.shape)
    wmap = np.asarray(w)[:, None, None] * one
    zero = np.zeros_like(wmap)
    if wave == "p":
        med = ElasticMedium(1000.0 * one, c * one, 0 * one, wmap, zero, ts, f0)
    else:  # onda S en un sólido con cp = 2 cs
        med = ElasticMedium(1900.0 * one, 2 * c * one, c * one, zero, wmap, ts, f0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ElasticSolver(Phantom(g, np.full(g.shape, TISSUE_INDEX["water"], np.int16)), f0, npml=NPML, medium=med,
                          backend=backend)
    bw = 2.0  # pulso muy ancho: hay señal de 0.3 a 2.5 f0
    nt = int(np.ceil((8.0 / (bw * f0) + (j2 - js) * dx / c + 4e-6) / s.dt))
    t = s.time_axis(nt)
    kind, q = ("pressure", "p") if wave == "p" else ("force_x", "vx")
    s.add_source(Source(np.full(nx, js), np.arange(nx), np.zeros(nx, int), gaussian_pulse(t, f0, bw)[None, :], kind))
    cols = np.arange(NPML, nx - NPML)
    s.add_receiver("r", Receiver(np.r_[np.full(cols.size, j1), np.full(cols.size, j2)], np.r_[cols, cols],
                                 np.repeat([0, 1], cols.size), 2, q))
    r = s.run(nt, track_peak=False)
    return r.t, r.traces["r"], (j2 - j1) * dx


def spectral_ratio(t, tr, dz, c0, freqs):
    """alpha (Np/m) y velocidad de fase entre los dos receptores (fase desenrollada con c ≈ c0)."""
    spec = np.fft.rfft(tr, 1 << 16, axis=1)
    fr = np.fft.rfftfreq(1 << 16, t[1] - t[0])
    k = np.array([np.argmin(np.abs(fr - f)) for f in freqs])
    ratio = spec[1, k] / spec[0, k]
    dphi = (-np.angle(ratio)) % (2 * np.pi)
    dphi = dphi + 2 * np.pi * np.round((2 * np.pi * freqs * dz / c0 - dphi) / (2 * np.pi))
    return -np.log(np.abs(ratio)) / dz, 2 * np.pi * freqs * dz / dphi


def measure(a0_db, y, c, n_mech, wave="p", ppw=20, backend="auto", freqs=None, method="nuevo"):
    """alpha y c medidos en simulación (calibrados con la onda sin pérdidas), del modelo y de la ley."""
    freqs = F0 * np.linspace(*WIDE, 23) if freqs is None else freqs
    w, ts = fit(a0_db, y, c, n_mech, method=method)
    t0, tr0, dz = plane_wave(c, np.zeros(0), np.zeros(0), ppw, wave=wave, backend=backend)
    a_ref, c_ref = spectral_ratio(t0, tr0, dz, c, freqs)
    t, tr, dz = plane_wave(c, w, ts, ppw, wave=wave, backend=backend)
    a_sim, c_sim = spectral_ratio(t, tr, dz, c, freqs)
    c_model, a_model = att.model_wave(c, w, ts, F0, freqs)
    return dict(f=freqs, alpha=a_sim - a_ref, c=c_sim * c / c_ref, alpha_model=a_model, c_model=c_model,
                alpha_law=alpha_law(a0_db, y, freqs),
                c_kk=att.power_law_phase_speed(c, alpha_law(a0_db, y, 1e6), y, F0, freqs))


def part_simulation():
    print("\n=== 2. Simulación: onda plana, cociente espectral entre receptores separados 15 mm (20 puntos/λ a f0) ===")
    print("Errores máximos en la banda estrecha [0.5, 1.8]·f0 | ancha [0.3, 2.5]·f0")
    cases = [(f"P c=1540 y={y}", 2.0, y, 1540.0, "p") for y in EXPONENTS]
    cases += [("cortical P (5 dB/cm/MHz)", 5.0, 1.0, 3500.0, "p"),
              ("cortical S (9.7 dB/cm/MHz)", 9.7, 1.0, 1800.0, "s")]
    print(f"{'caso':28s}{'L':>3s}{'alpha vs ley (%)':>20s}{'alpha vs modelo (%)':>22s}{'c vs modelo (m/s)':>20s}"
          f"{'c vs K-K (m/s)':>18s}")
    for name, a0, y, c, wave in cases:
        for n in (1, 2, 3):
            t0 = time.time()
            m = measure(a0, y, c, n, wave)
            narrow = (m["f"] >= NARROW[0] * F0 - 1) & (m["f"] <= NARROW[1] * F0 + 1)
            # la onda S no tiene señal útil por encima de ~1.8 f0 (se atenúa ~40 dB en 15 mm)
            wide = np.ones_like(narrow) if wave == "p" else narrow
            cols = []
            for a, b, rel in (("alpha", "alpha_law", True), ("alpha", "alpha_model", True), ("c", "c_model", False),
                              ("c", "c_kk", False)):
                e = m[a] / m[b] - 1 if rel else m[a] - m[b]
                scale = 100 if rel else 1
                cols.append(f"{np.abs(e[narrow]).max()*scale:8.2f} |{np.abs(e[wide]).max()*scale:7.2f}")
            print(f"{name:28s}{n:3d}" + "".join(f"{x:>20s}" for x in cols) + f"   [{time.time()-t0:.1f} s]")


# ----------------------------------------------------------------------------
# 3. tiempo
# ----------------------------------------------------------------------------
def part_time():
    print("\n=== 3. Coste: simulate_bmode, tibia con fractura 24×22 mm, 3 MHz, dx = 0.06 mm, 1 emisión ===")
    p = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=24e-3, depth=22e-3, dx=0.06e-3)
    base = None
    for n in (1, 2, 3):
        t0 = time.time()
        sc = simulate_bmode(p, 3e6, angles=(0.0,), n_mechanisms=n)
        el = time.time() - t0
        base = base or el
        r = sc.results[0]
        print(f"L={n}: {el:6.1f} s ({el/base:.2f}× L=1), {r.t.size} pasos de {r.dt*1e9:.3f} ns, "
              f"{r.elapsed_s / r.t.size * 1e3:.2f} ms/paso")


# ----------------------------------------------------------------------------
# diseño de MECHANISM_DESIGN
# ----------------------------------------------------------------------------
def nelder_mead(fun, x0, step=0.3, iters=300, tol=1e-7):
    """Nelder-Mead mínimo (no hay SciPy)."""
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * np.eye(len(x0))[i] for i in range(len(x0))]
    vals = [fun(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if vals[-1] - vals[0] < tol:
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = 2 * cen - pts[-1]
        fr = fun(xr)
        if fr < vals[0]:
            xe = 3 * cen - 2 * pts[-1]
            fe = fun(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = 0.5 * (cen + pts[-1])
            fc = fun(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [fun(p) for p in pts]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def part_design(penalty=0.05):
    """Optimiza las frecuencias de relajación: peor error en alpha + penalty·(c_U²/c0² - 1) de la cortical."""
    print("\n=== Diseño de las frecuencias de relajación (MECHANISM_DESIGN) ===")
    # (alpha0 dB/cm/MHz^y, y, c): Q alto y bajo con y = 1 (el ajuste depende algo de Q), sangre, y = 1.5 y 2
    cases = np.array([(5.0, 1.0, 3500.0), (15.0, 1.0, 2100.0), (1.0, 1.0, 1540.0), (2.0, 1.1, 1580.0),
                      (0.2, 1.2, 1578.0), (1.0, 1.5, 1500.0), (0.5, 2.0, 1500.0)])
    starts = {2: [-0.6, 2.0], 3: [-1.1, 1.0, 3.0], 4: [-1.5, -0.3, 1.0, 3.0]}
    for n, x0 in starts.items():
        lo, hi = att.MECHANISM_DESIGN[n][1]
        band = F0 * np.geomspace(lo, hi, 32)
        alpha = cases[:, :1] * (band[None] / 1e6) ** cases[:, 1:2] * 100 / DB_PER_NP

        def cost(x, alpha=alpha, band=band):
            ts = 1.0 / (2 * np.pi * F0 * np.exp(x))
            w = att.mechanism_weights(alpha, cases[:, 2], band, ts, F0)
            _, a_m = att.model_wave(cases[:, 2:3], w.T[..., None], ts, F0, band[None])
            stiff = att.relaxed_modulus_factor(w[0], ts, F0) * (1 + w[0].sum())
            return float(np.abs(a_m / alpha - 1).max() + penalty * (stiff - 1))

        x, v = nelder_mead(cost, np.array(x0))
        print(f"L={n}, banda {lo}-{hi}·f0: frecuencias/f0 = {np.round(np.sort(np.exp(x)), 3)} (coste {v:.4f}); "
              f"tabla actual {att.MECHANISM_DESIGN[n][0]} (coste {cost(np.log(att.MECHANISM_DESIGN[n][0])):.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parte", choices=("todo", "modelo", "simulacion", "tiempo"), default="todo")
    ap.add_argument("--disenar", action="store_true", help="re-optimiza las frecuencias de relajación y sale")
    args = ap.parse_args()
    if args.disenar:
        part_design()
        return
    if args.parte in ("todo", "modelo"):
        part_model()
    if args.parte in ("todo", "simulacion"):
        part_simulation()
    if args.parte in ("todo", "tiempo"):
        part_time()


if __name__ == "__main__":
    main()
