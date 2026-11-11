"""Comparación numérica wavesim (FDTD elástico) frente a k-Wave (pseudoespectral).

Instalación de la dependencia (sólo para este benchmark; NO está en pyproject.toml):

    pip install k-wave-python scipy          # probado con k-wave-python 0.6.2
    # k-wave-python 0.6.2 trae los binarios C++ (OMP y CUDA) en el propio wheel;
    # si no hay binario utilizable se puede usar ``--backend python`` (más lento).

Uso (desde la raíz del repositorio):

    python benchmarks/kwave_comparison.py                  # todo
    python benchmarks/kwave_comparison.py --case layers    # sólo las dos capas
    python benchmarks/kwave_comparison.py --backend cpp   # binario C++ (necesita libhdf5_serial_hl.so.100)

Casos:

A) ``layers``: onda plana en dos capas fluidas sin pérdidas
   (c1=1500, rho1=1000 | c2=1800, rho2=1100). Compara con la solución analítica
   (coeficientes de reflexión y transmisión de presión) y entre los dos
   simuladores: tiempos de llegada, amplitudes y error relativo de la forma de
   onda. Se repite a 3 resoluciones (dx = 0.20, 0.15 y 0.10 mm).
B) ``point``: fuente puntual (línea en 2D) en un medio homogéneo con
   atenuación de ley de potencia (alpha = 2 dB/cm/MHz^1.1, c = 1580 m/s).
   Compara atenuación y velocidad de fase en función de la frecuencia,
   medidas entre dos receptores a distinta distancia (corrigiendo la
   expansión cilíndrica 1/sqrt(r)).

Notas de diseño:
  * La amplitud absoluta de una fuente "aditiva" no está definida igual en
    ambos códigos, así que las comparaciones de amplitud se hacen
    normalizando por la onda incidente medida por el mismo simulador.
  * k-Wave usa un exponente alpha_power = 1.1 (no 1.0) porque la dispersión
    causal de k-Wave diverge para y = 1.
  * Las PML de ambos códigos se dejan fuera de la región activa (wavesim las
    lleva dentro de su malla, k-Wave fuera por defecto).
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
from wavesim.core.signals import analytic_signal, gaussian_pulse, pulse_peak_time  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402
from wavesim.ultrasound import attenuation as att  # noqa: E402
from wavesim.ultrasound.elastic_fdtd import ElasticSolver, Receiver, Source  # noqa: E402
from wavesim.ultrasound.medium import ElasticMedium  # noqa: E402

F0 = 1.0e6
BW = 0.6
NPML = 20
DB_PER_NP = 20.0 / np.log(10.0)


# ----------------------------------------------------------------------------
# wavesim
# ----------------------------------------------------------------------------
def ws_medium(rho, cp, f0, alpha_db=None, y=1.1, n_mech=1) -> ElasticMedium:
    """Medio fluido para wavesim. ``rho``, ``cp``: (nz, nx) de la región activa."""
    rho = np.pad(rho, NPML, mode="edge")
    cp = np.pad(cp, NPML, mode="edge")
    if alpha_db is None or n_mech == 0:
        z = np.zeros_like(rho)
        return ElasticMedium(rho, cp, z, z.copy(), z.copy(), np.zeros(0), f0)
    tau_sigma = att.relaxation_times(f0, n_mech)
    band = att.fit_band(f0, n_mech)
    alpha = alpha_db * (band / 1e6) ** y * 100.0 / DB_PER_NP
    c0 = float(cp.flat[0])
    w = att.mechanism_weights(alpha[None], np.array([c0]), band, tau_sigma, f0)[0]
    return ElasticMedium(rho, cp, np.zeros_like(rho), w[:, None, None] * np.ones_like(rho),
                         np.zeros((n_mech, *rho.shape)), tau_sigma, f0)


def ws_run(rho, cp, dx, src_cells, rec_cells, t_end, alpha_db=None, n_mech=1):
    """Ejecuta wavesim. ``src_cells``/``rec_cells``: listas de (j, i) en la región activa."""
    nz, nx = rho.shape
    grid = Grid2D(nx + 2 * NPML, nz + 2 * NPML, dx)
    phantom = Phantom(grid, np.full(grid.shape, TISSUE_INDEX["water"]))
    med = ws_medium(rho, cp, F0, alpha_db, n_mech=n_mech)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = ElasticSolver(phantom, F0, npml=NPML, medium=med, attenuation=alpha_db is not None)
    nt = int(np.ceil(t_end / solver.dt))
    t = solver.time_axis(nt)
    sig = gaussian_pulse(t, F0, BW)[None, :]
    sj = np.array([c[0] + NPML for c in src_cells])
    si = np.array([c[1] + NPML for c in src_cells])
    solver.add_source(Source(sj, si, np.zeros(len(sj), int), sig, "pressure"))
    rj = np.array([c[0] + NPML for c in rec_cells])
    ri = np.array([c[1] + NPML for c in rec_cells])
    solver.add_receiver("r", Receiver(rj, ri, np.arange(len(rj)), len(rj), "p"))
    res = solver.run(nt, track_peak=False)
    return res.t, res.traces["r"], solver.dt, res.elapsed_s


# ----------------------------------------------------------------------------
# k-Wave
# ----------------------------------------------------------------------------
def kw_run(rho, cp, dx, src_mask, rec_cells, t_end, alpha_db=None, y=1.1, backend="cpp"):
    """Ejecuta k-Wave. Convención de ejes: primera dimensión = z (profundidad)."""
    from kwave.data import Vector
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    nz, nx = rho.shape
    kgrid = kWaveGrid(Vector([nz, nx]), Vector([dx, dx]))
    dt = 0.3 * dx / float(cp.max())  # CFL 0.3 (valor por defecto de k-Wave)
    nt = int(np.ceil(t_end / dt))
    kgrid.setTime(nt, dt)
    t = np.arange(nt) * dt
    kw = dict(sound_speed=cp, density=rho)
    if alpha_db is not None:
        kw.update(alpha_coeff=alpha_db, alpha_power=y)
    medium = kWaveMedium(**kw)
    source = kSource()
    source.p_mask = src_mask.astype(bool)
    n_src = int(src_mask.sum())
    source.p = np.tile(gaussian_pulse(t, F0, BW)[None, :], (n_src, 1))
    sensor_mask = np.zeros((nz, nx), bool)
    for j, i in rec_cells:
        sensor_mask[j, i] = True
    sensor = kSensor(mask=sensor_mask)
    t0 = time.time()
    out = kspaceFirstOrder(kgrid, medium, source, sensor, pml_size=NPML, pml_inside=False,
                           backend=backend, device="cpu", quiet=True, smooth_p0=False)
    el = time.time() - t0
    p = np.asarray(out["p"])
    # los sensores salen en orden C del mapa (j, i): devolverlos en el orden pedido
    order = sorted(range(len(rec_cells)), key=lambda k: rec_cells[k])
    out_p = np.empty_like(p)
    for pos, k in enumerate(order):
        out_p[k] = p[pos]
    return t, out_p, dt, el


# ----------------------------------------------------------------------------
# métricas
# ----------------------------------------------------------------------------
def env(x):
    return np.abs(analytic_signal(x))


def peak(t, x, lo=None, hi=None):
    """Instante (parabólico) y valor del máximo de la envolvente en [lo, hi]."""
    e = env(x)
    m = np.ones_like(t, bool)
    if lo is not None:
        m &= t >= lo
    if hi is not None:
        m &= t <= hi
    k = np.flatnonzero(m)[np.argmax(e[m])]
    a, b, c = e[k - 1], e[k], e[k + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return t[k] + d * (t[1] - t[0]), b + 0.25 * (a - c) * d


def resample(t_src, x, t_dst):
    return np.interp(t_dst, t_src, x, left=0.0, right=0.0)


def best_shape_error(ref, x, t, max_shift=0.1e-6):
    """Error L2 relativo tras alinear en tiempo y ajustar amplitud (mínimos cuadrados)."""
    best = None
    dt = t[1] - t[0]
    for s in np.arange(-max_shift, max_shift + dt / 2, dt):
        xs = resample(t + s, x, t)
        a = np.dot(xs, ref) / np.dot(xs, xs)
        e = np.linalg.norm(a * xs - ref) / np.linalg.norm(ref)
        if best is None or e < best[0]:
            best = (e, s, a)
    return best


# ----------------------------------------------------------------------------
# caso A: dos capas
# ----------------------------------------------------------------------------
def case_layers(dx, backend):
    c1, r1, c2, r2 = 1500.0, 1000.0, 1800.0, 1100.0
    width, depth = 40e-3, 20e-3
    z_src, z_r0, z_int, z_r1 = 2e-3, 4e-3, 11e-3, 15e-3  # distancias cortas: las ondas de borde de la fuente finita llegan tarde
    nz, nx = int(round(depth / dx)), int(round(width / dx))
    j_int = int(round(z_int / dx))
    rho = np.full((nz, nx), r1)
    cp = np.full((nz, nx), c1)
    rho[j_int:], cp[j_int:] = r2, c2
    j_src, j_r0, j_r1 = (int(round(z / dx)) for z in (z_src, z_r0, z_r1))
    ic = nx // 2
    src_cells = [(j_src, i) for i in range(nx)]
    src_mask = np.zeros((nz, nx), bool)
    src_mask[j_src, :] = True
    rec_cells = [(j_r0, ic), (j_r1, ic)]
    t_end = 18e-6  # la reflejada llega a ~13 us: el registro debe cubrir todo el pulso
    tw, pw, dtw, elw = ws_run(rho, cp, dx, src_cells, rec_cells, t_end)
    tk, pk, dtk, elk = kw_run(rho, cp, dx, src_mask, rec_cells, t_end, backend=backend)

    Z1, Z2 = r1 * c1, r2 * c2
    R_an, T_an = (Z2 - Z1) / (Z2 + Z1), 2 * Z2 / (Z1 + Z2)
    zi = (j_int - 0.5) * dx  # la interfaz queda entre los centros de las celdas j_int-1 y j_int
    z_r0e, z_r1e = j_r0 * dx, j_r1 * dx
    # instante intermedio entre la onda incidente y la reflejada en el receptor 1
    split = pulse_peak_time(F0, BW) + (zi - j_src * dx) / c1
    out = {}
    for name, t, p in (("wavesim", tw, pw), ("k-Wave", tk, pk)):
        ti, ai = peak(t, p[0], hi=split)
        tr, ar = peak(t, p[0], lo=split)
        tt, at = peak(t, p[1])
        out[name] = dict(t_inc=ti, a_inc=ai, t_ref=tr, a_ref=ar, t_tr=tt, a_tr=at,
                         R=ar / ai, T=at / ai, dt_ref=tr - ti, dt_tr=tt - ti)
    exp_ref = 2 * (zi - z_r0e) / c1
    exp_tr = (zi - z_r0e) / c1 + (z_r1e - zi) / c2

    tc = np.arange(0, t_end, 5e-9)
    shape = {}
    for k, label in ((0, "receptor 1 (incidente + reflejada)"), (1, "receptor 2 (transmitida)")):
        a = resample(tw, pw[k], tc)
        b = resample(tk, pk[k], tc)
        raw = np.linalg.norm(a - b) / np.linalg.norm(b)
        e_shape, shift, amp = best_shape_error(b, a, tc)
        shape[label] = dict(raw=raw, shape=e_shape, shift=shift, amp=amp)
    return dict(dx=dx, ppw=c1 / (F0 * dx), R_an=R_an, T_an=T_an, exp_ref=exp_ref, exp_tr=exp_tr,
                res=out, shape=shape, steps=(len(tw), len(tk)), dts=(dtw, dtk), elapsed=(elw, elk))


def print_layers(r):
    print(f"\n--- dos capas, dx = {r['dx']*1e3:.2f} mm ({r['ppw']:.0f} puntos/long. de onda en la capa 1) ---")
    print(f"pasos: wavesim {r['steps'][0]} (dt={r['dts'][0]*1e9:.1f} ns, {r['elapsed'][0]:.1f} s), "
          f"k-Wave {r['steps'][1]} (dt={r['dts'][1]*1e9:.1f} ns, {r['elapsed'][1]:.1f} s)")
    print(f"analítico: R = {r['R_an']:.4f}, T = {r['T_an']:.4f}, "
          f"t(refl)-t(inc) = {r['exp_ref']*1e6:.3f} us, t(trans)-t(inc) = {r['exp_tr']*1e6:.3f} us")
    print(f"{'':10s}{'R':>9s}{'T':>9s}{'dt_ref us':>11s}{'dt_tr us':>10s}{'t_inc us':>10s}")
    for k, v in r["res"].items():
        print(f"{k:10s}{v['R']:9.4f}{v['T']:9.4f}{v['dt_ref']*1e6:11.3f}{v['dt_tr']*1e6:10.3f}{v['t_inc']*1e6:10.3f}")
    w, k = r["res"]["wavesim"], r["res"]["k-Wave"]
    print(f"err. rel. vs analítico  R: wavesim {abs(w['R']/r['R_an']-1)*100:.2f} %, k-Wave {abs(k['R']/r['R_an']-1)*100:.2f} %"
          f" | T: wavesim {abs(w['T']/r['T_an']-1)*100:.2f} %, k-Wave {abs(k['T']/r['T_an']-1)*100:.2f} %")
    print(f"dif. wavesim-kWave  dt_ref: {(w['dt_ref']-k['dt_ref'])*1e9:+.1f} ns, dt_tr: {(w['dt_tr']-k['dt_tr'])*1e9:+.1f} ns, "
          f"t_inc absoluto: {(w['t_inc']-k['t_inc'])*1e9:+.1f} ns, amplitud incidente (cociente): {w['a_inc']/k['a_inc']:.3f}")
    for lab, s in r["shape"].items():
        print(f"error L2 relativo {lab}: crudo {s['raw']*100:.1f} %, "
              f"tras mejor desfase ({s['shift']*1e9:+.0f} ns) y escala ({s['amp']:.3f}) {s['shape']*100:.1f} %")


# ----------------------------------------------------------------------------
# caso B: fuente puntual con atenuación
# ----------------------------------------------------------------------------
def case_point(dx, backend, n_mechs=(1, 2)):
    c0, rho0, a0, y = 1580.0, 1090.0, 2.0, 1.1
    side = 50e-3
    n = int(round(side / dx))
    rho = np.full((n, n), rho0)
    cp = np.full((n, n), c0)
    js = is_ = n // 2
    rs = [5e-3, 10e-3, 15e-3]
    rec = [(js + int(round(r / dx)), is_) for r in rs]
    src_mask = np.zeros((n, n), bool)
    src_mask[js, is_] = True
    t_end = 17e-6
    runs = {}
    t, p, dt, el = kw_run(rho, cp, dx, src_mask, rec, t_end, alpha_db=a0, y=y, backend=backend)
    runs["k-Wave"] = (t, p, el)
    for nm in n_mechs:
        t, p, dt, el = ws_run(rho, cp, dx, [(js, is_)], rec, t_end, alpha_db=a0, n_mech=nm)
        runs[f"wavesim ({nm} mec.)"] = (t, p, el)
    r_eff = [(j - js) * dx for j, _ in rec]
    freqs = np.array([0.7e6, 0.85e6, 1.0e6, 1.15e6, 1.3e6])
    out = {}
    for name, (t, p, el) in runs.items():
        nfft = 1 << 15
        dtt = t[1] - t[0]
        P = np.fft.rfft(p, nfft, axis=1)
        fr = np.fft.rfftfreq(nfft, dtt)
        a_est, c_est = [], []
        dr = r_eff[2] - r_eff[0]
        for f in freqs:
            k = np.argmin(abs(fr - f))
            ratio = P[2, k] / P[0, k]
            # un retardo tau resta fase 2*pi*f*tau; hay que fijar la ambigüedad de 2*pi con c ~ c0
            dphi = (-np.angle(ratio)) % (2 * np.pi)
            n2pi = np.round((2 * np.pi * f * dr / c0 - dphi) / (2 * np.pi))
            c_est.append(2 * np.pi * f * dr / (dphi + 2 * np.pi * n2pi))
            # amplitud ~ exp(-alpha r)/sqrt(r)
            a_est.append(-(np.log(abs(ratio)) + 0.5 * np.log(r_eff[2] / r_eff[0])) / dr)
        out[name] = dict(alpha_np=np.array(a_est), c=np.array(c_est), elapsed=el)
    alpha_th = a0 * (freqs / 1e6) ** y * 100.0 / DB_PER_NP
    return dict(freqs=freqs, alpha_th=alpha_th, runs=out, r=r_eff)


def print_point(r):
    print("\n--- fuente puntual en medio con atenuación (alpha = 2 dB/cm/MHz^1.1, c = 1580 m/s) ---")
    print(f"receptores a {', '.join(f'{x*1e3:.0f}' for x in r['r'])} mm; atenuación medida entre el 1.º y el 3.º")
    print("f (MHz)".ljust(26) + "".join(f"{f/1e6:10.2f}" for f in r["freqs"]))
    print("alpha teórica (dB/cm)".ljust(26) + "".join(f"{a*DB_PER_NP/100:10.3f}" for a in r["alpha_th"]))
    ref = r["runs"]["k-Wave"]
    for name, v in r["runs"].items():
        print(f"alpha {name} (dB/cm)".ljust(26) + "".join(f"{a*DB_PER_NP/100:10.3f}" for a in v["alpha_np"])
              + f"   [{v['elapsed']:.1f} s]")
    for name, v in r["runs"].items():
        err = (v["alpha_np"] / r["alpha_th"] - 1) * 100
        print(f"err vs teoría {name} (%)".ljust(26) + "".join(f"{e:+10.1f}" for e in err))
    for name, v in r["runs"].items():
        print(f"c_fase {name} (m/s)".ljust(26) + "".join(f"{c:10.1f}" for c in v["c"]))
    for n, v in r["runs"].items():
        if n != "k-Wave":
            print(f"alpha {n} / k-Wave - 1 (%)".ljust(26)
                  + "".join(f"{(v['alpha_np'][i]/ref['alpha_np'][i]-1)*100:+10.1f}" for i in range(len(r['freqs']))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=("all", "layers", "point"), default="all")
    ap.add_argument("--backend", choices=("python", "cpp"), default="python",
                    help="python (por defecto, puro NumPy) o cpp (binario OMP/CUDA incluido en el wheel)")
    args = ap.parse_args()
    try:
        import kwave  # noqa: F401
    except ImportError:
        sys.exit("Falta k-wave-python: pip install k-wave-python scipy")
    warnings.filterwarnings("ignore", category=FutureWarning)
    if args.case in ("all", "layers"):
        for dx in (0.20e-3, 0.15e-3, 0.10e-3):
            print_layers(case_layers(dx, args.backend))
    if args.case in ("all", "point"):
        print_point(case_point(0.1e-3, args.backend))


if __name__ == "__main__":
    main()
