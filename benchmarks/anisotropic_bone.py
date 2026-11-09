"""Hueso cortical transversalmente isótropo: velocidades medidas, estabilidad y coste.

Uso (desde la raíz del repositorio; sólo numpy, y numba opcional):

    NUMBA_NUM_THREADS=1 python benchmarks/anisotropic_bone.py                  # todo (~4 min con Numba)
    python benchmarks/anisotropic_bone.py --parte velocidades                 # ondas planas (~1 min)
    python benchmarks/anisotropic_bone.py --parte estabilidad                 # tibia y pierna con fractura (~3 min)

Partes:

1. ``velocidades``: medio homogéneo con las rigideces del corte longitudinal
   (``CORTICAL_STIFFNESS``, x = eje del hueso). Ondas planas con la normal a
   0-90° del eje x (fuente de línea perpendicular a la normal; a 0° y 90° la
   fuente ocupa una fila o columna entera y la simulación es 1D). La
   velocidad sale del retardo por correlación cruzada entre dos receptores a
   lo largo de la normal, con una ventana en la llegada de cada onda. Se
   compara con la ecuación de Christoffel (a 0° y 90°, sqrt(C/ρ)).
2. ``estabilidad``: ``simulate_bmode`` de una emisión sobre una tibia con
   fractura desplazada (modo realista, con poros) a 3 MHz, 24×22 mm, con la
   cortical isótropa y anisótropa, 1 y 3 mecanismos, con Numba y NumPy
   (comparación de las RF entre backends), y un corte transversal de pierna
   con fractura a 1.5 MHz. Tiempo por paso iso/aniso (mínimo de 5 bloques).
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
from wavesim.core.signals import gaussian_pulse, pulse_peak_time  # noqa: E402
from wavesim.materials.acoustic import ACOUSTIC, CORTICAL_STIFFNESS  # noqa: E402
from wavesim.phantoms import Fracture, RingFracture, cross_section_limb, longitudinal_limb  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402
from wavesim.ultrasound import ElasticMedium, ElasticSolver, Receiver, Source, default_array, simulate_bmode  # noqa: E402
from wavesim.ultrasound.medium import christoffel_speeds  # noqa: E402

F0 = 1e6
BW = 1.5
NPML = 10
RHO = ACOUSTIC["cortical_bone"].density
TI = np.array(CORTICAL_STIFFNESS.plane("longitudinal")) / RHO


def lag(a, b, dt):
    """Retardo de ``b`` respecto a ``a`` (correlación cruzada con ajuste parabólico)."""
    n = a.size
    xc = np.fft.irfft(np.fft.rfft(b, 2 * n) * np.conj(np.fft.rfft(a, 2 * n)))
    k = int(np.argmax(xc[:n]))
    y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
    return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * dt


def homogeneous_solver(g, backend="auto"):
    one = np.ones(g.shape)
    z = np.zeros((0, *g.shape))
    med = ElasticMedium(RHO * one, np.sqrt(TI[0]) * one, np.sqrt(TI[3]) * one, z, z.copy(), np.zeros(0), F0,
                        TI[:, None, None] * one)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElasticSolver(Phantom(g, np.full(g.shape, TISSUE_INDEX["water"], np.int16)), F0, npml=NPML,
                             medium=med, backend=backend)


def plane_wave_speed(theta_deg: float, wave: str, dx=0.1e-3, behind=6e-3, d1=8e-3, d2=18e-3, backend="auto"):
    """Velocidad de fase medida de la onda ``wave`` ("qP" o "qSV") con la normal a ``theta_deg`` del eje x."""
    th = np.radians(theta_deg)
    n_vec = np.array([np.cos(th), np.sin(th)])
    axis = theta_deg in (0.0, 90.0)
    if axis:
        # 1D: la fuente ocupa una fila o columna entera y el campo no varía en la otra dirección
        long_n = int(round((behind + d2 + 6e-3) / dx)) + 2 * NPML
        short_n = 2 * NPML + 4
        start = NPML + 4 + int(round(behind / dx))
        if theta_deg == 0:
            g, c = Grid2D(long_n, short_n, dx), np.array([start, short_n // 2])  # c = (i, j)
        else:
            g, c = Grid2D(short_n, long_n, dx), np.array([short_n // 2, start])
    else:
        n = int(round(2 * (behind + d2 + 4e-3) / dx))
        g = Grid2D(n, n, dx)
        c = np.array([n // 2 - int(round((d2 - behind) / 2 * np.cos(th) / dx)),
                      n // 2 - int(round((d2 - behind) / 2 * np.sin(th) / dx))])
    s = homogeneous_solver(g, backend)
    v_qp, v_qs = (float(v) for v in christoffel_speeds(TI, th))
    v = v_qp if wave == "qP" else v_qs
    nt = int(np.ceil((pulse_peak_time(F0, BW) * 2 + (behind + d2) / v + 2e-6) / s.dt))
    t = s.time_axis(nt)
    jj, ii = np.mgrid[0:g.nz, 0:g.nx]
    # línea de fuentes: celdas cuya proyección sobre la normal está a ``behind`` por detrás del centro
    proj = ((ii - c[0]) * n_vec[0] + (jj - c[1]) * n_vec[1]) * dx
    line = np.abs(proj + behind) < 0.5 * dx * max(abs(n_vec[0]), abs(n_vec[1]))
    js, is_ = np.nonzero(line)
    pulse = gaussian_pulse(t, F0, BW)[None, :]
    zero = np.zeros(js.size, int)
    if wave == "qP":
        s.add_source(Source(js, is_, zero, pulse, "pressure"))
    else:  # fuerza perpendicular a la normal: excita sobre todo la qSV
        s.add_source(Source(js, is_, zero, -np.sin(th) * pulse, "force_x"))
        s.add_source(Source(js, is_, zero, np.cos(th) * pulse, "force_z"))
    rec = [np.round(c + (d - behind) / dx * n_vec).astype(int) for d in (d1, d2)]
    rj, ri = np.array([r[1] for r in rec]), np.array([r[0] for r in rec])
    for q in ("p", "vx", "vz"):
        s.add_receiver(q, Receiver(rj, ri, np.array([0, 1]), 2, q))
    res = s.run(nt, track_peak=False)
    tr = res.traces["p"] if wave == "qP" else -np.sin(th) * res.traces["vx"] + np.cos(th) * res.traces["vz"]
    dist = np.array([((r[0] - c[0]) * n_vec[0] + (r[1] - c[1]) * n_vec[1]) * dx + behind for r in rec])
    # ventana en la llegada de la onda medida, sin la otra (qP antes, qSV después)
    t0 = pulse_peak_time(F0, BW)  # el pulso dura 2·t0
    for k, d in enumerate(dist):
        w = min(1.2 * t0, 0.5 * (d / v_qs - d / v_qp))
        tr[k] = tr[k] * (np.abs(res.t - (t0 + d / v)) < w)
    return (dist[1] - dist[0]) / lag(tr[0], tr[1], s.dt), v


def part_speeds():
    print("\n=== 1. Velocidades de fase medidas frente a Christoffel (cortical longitudinal, x = eje del hueso) ===")
    print(f"C11, C33, C13, C55 = {', '.join(f'{c/1e9:.2f}' for c in CORTICAL_STIFFNESS.plane('longitudinal'))} GPa,"
          f" rho = {RHO:.0f} kg/m³; f0 = 1 MHz, dx = 0.1 mm")
    print(f"{'ángulo':>7s}{'qP medida':>11s}{'qP Christ.':>12s}{'error':>8s}{'qSV medida':>12s}{'qSV Christ.':>13s}"
          f"{'error':>8s}")
    for ang in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        t0 = time.time()
        vp, ep = plane_wave_speed(ang, "qP")
        vs, es = plane_wave_speed(ang, "qSV")
        print(f"{ang:6.0f}°{vp:11.1f}{ep:12.1f}{(vp/ep-1)*100:+7.2f}%{vs:12.1f}{es:13.1f}{(vs/es-1)*100:+7.2f}%"
              f"   [{time.time()-t0:.1f} s]")


def one_emission(p, f0, n_mech, aniso, backend):
    """RF de una emisión a 0° con ``ElasticSolver`` (como ``simulate_bmode``) y su tiempo."""
    s = ElasticSolver(p, f0, n_mechanisms=n_mech, anisotropic_bone=aniso, backend=backend)
    arr = default_array(p, f0)
    g = p.grid
    depth = g.z0 + g.depth - (20 + 3) * g.dx - arr.z
    nt = int(np.ceil((arr.pulse_peak_time + (depth + np.hypot(depth, arr.aperture)) / 1400.0) / s.dt))
    t = s.time_axis(nt)
    s.add_source(arr.transmit_source(g, t, arr.plane_wave_delays(0.0, 1540.0)))
    s.add_receiver("rf", arr.receiver(g))
    t0 = time.time()
    r = s.run(nt, track_peak=True)
    return s, r, time.time() - t0


def step_time(p, f0, n_mech, aniso, blocks=5, steps=60):
    s = ElasticSolver(p, f0, n_mechanisms=n_mech, anisotropic_bone=aniso)
    arr = default_array(p, f0)
    t = s.time_axis(blocks * steps + 20)
    s.add_source(arr.transmit_source(p.grid, t, arr.plane_wave_delays(0.0, 1540.0)))
    s.run(20, track_peak=False)
    best = np.inf
    for b in range(blocks):
        t0 = time.time()
        for n in range(steps):
            s.step(20 + b * steps + n)
        best = min(best, (time.time() - t0) / steps)
    return best, s.dt


def part_stability():
    print("\n=== 2. Estabilidad: tibia con fractura desplazada 1 mm, modo realista, 3 MHz, 24×22 mm, 1 emisión ===")
    p = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=24e-3, depth=22e-3, dx=0.06e-3,
                          realism="realista")
    ref = {}
    for aniso in (False, True):
        for n_mech, backend in ((1, "numba"), (1, "numpy"), (3, "numba")):
            s, r, el = one_emission(p, 3e6, n_mech, aniso, backend)
            rf = r.traces["rf"]
            ok = np.isfinite(rf).all() and np.isfinite(s.vx).all()
            line = (f"{'aniso' if aniso else 'iso':5s} L={n_mech} {backend:5s}: {r.t.size} pasos de {s.dt*1e9:.3f} ns, "
                    f"{el:5.1f} s, finita={ok}, |p| final máx = {np.abs(s.pressure()).max():.3e} Pa, "
                    f"pico = {r.peak_pressure.max():.3e} Pa")
            if (aniso, n_mech) in ref:
                a = ref[(aniso, n_mech)][1]
                line += f", RF numba vs numpy: dif. máx relativa {np.abs(a - rf).max() / np.abs(a).max():.1e}"
            else:
                ref[(aniso, n_mech)] = (r.t, rf)
            print(line, flush=True)
    (ta, a), (tb, b) = ref[(False, 1)], ref[(True, 1)]  # pasos de tiempo distintos: se interpola
    b = np.array([np.interp(ta, tb, ch) for ch in b])
    change = np.linalg.norm(a - b) / np.linalg.norm(a)
    print(f"Cambio de la RF por la anisotropía (L = 1): {change * 100:.0f} % (norma L2)")
    for n_mech in (1, 3):
        ti, dti = step_time(p, 3e6, n_mech, False)
        ta, dta = step_time(p, 3e6, n_mech, True)
        print(f"Tiempo por paso (Numba, mínimo) L={n_mech}: iso {ti*1e3:.2f} ms, aniso {ta*1e3:.2f} ms "
              f"(×{ta/ti:.2f}); dt iso {dti*1e9:.3f} ns, aniso {dta*1e9:.3f} ns")

    print("\n=== Corte transversal de pierna con fractura (1.5 MHz, dx = 0.12 mm): plano de isotropía ===")
    q = cross_section_limb("pierna", RingFracture(), dx=0.12e-3, realism="realista")
    for aniso in (False, True):
        t0 = time.time()
        sc = simulate_bmode(q, 1.5e6, angles=(0.0,), anisotropic_bone=aniso)
        r = sc.results[0]
        print(f"{'aniso' if aniso else 'iso':5s}: malla {q.grid.shape}, {r.t.size} pasos, {time.time()-t0:.1f} s, "
              f"RF finita={np.isfinite(sc.rf[0]).all()}, pico = {r.peak_pressure.max():.3e} Pa")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parte", choices=("todo", "velocidades", "estabilidad"), default="todo")
    args = ap.parse_args()
    if args.parte in ("todo", "velocidades"):
        part_speeds()
    if args.parte in ("todo", "estabilidad"):
        part_stability()


if __name__ == "__main__":
    main()
