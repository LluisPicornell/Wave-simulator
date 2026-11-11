"""Comparación numérica wavesim (FDTD elástico, aquí en fluido) frente a k-Wave (pseudoespectral).

Instalación de la dependencia (sólo para este benchmark; NO está en pyproject.toml), en un entorno aparte:

    python -m venv --system-site-packages venv_kwave
    venv_kwave/bin/pip install k-wave-python scipy      # probado con k-wave-python 0.6.2 (la última en PyPI)

Uso (desde la raíz del repositorio; para guardar las trazas de referencia, ``cross_ultrasonido.py``):

    NUMBA_NUM_THREADS=1 PYTHONPATH=$(pwd) venv_kwave/bin/python benchmarks/kwave_comparison.py               # todo
    ... benchmarks/kwave_comparison.py --case layers    # sólo las dos capas (A, ~3 min)
    ... benchmarks/kwave_comparison.py --case point     # fuente puntual con atenuación (B, ~4 min)
    ... benchmarks/kwave_comparison.py --case tissue    # medio tipo tejido (C, ~2 min)
    ... --backend cpp   # binario C++ de k-Wave (necesita libhdf5_serial_hl.so.100; aquí era más lento)

Casos (resultados en ``docs/validacion/cruzada_ultrasonido.md``):

A) ``layers``: onda plana en dos capas fluidas sin pérdidas (c1=1500, rho1=1000 | c2=1800, rho2=1100).
   R y T (cociente de envolventes y R(f) espectral), tiempos de llegada y forma de onda frente a la
   solución analítica y entre códigos, a dx = 0.20, 0.15, 0.10 y 0.05 mm. wavesim se ejecuta con su CFL
   por defecto (0.9) y con 0.3 (la de k-Wave), para separar su dispersión temporal.
B) ``point``: fuente puntual (línea en 2D) en un medio homogéneo con atenuación de ley de potencia
   (2 dB/cm/MHz^1.1, c = 1580 m/s). Atenuación y velocidad de fase entre receptores a 5 y 15 mm,
   corrigiendo la expansión cilíndrica. Cada código se compara con su propio modelo continuo
   (``kwave_model``, ``attenuation.model_wave``) y entre sí con k-Wave "ajustado" (``kwave_matched``):
   sus ``sound_speed`` y ``alpha_coeff`` se eligen para que su modelo dé 1580 m/s y 2 dB/cm a 1 MHz.
C) ``tissue``: agua, piel con superficie curva, grasa con fondo ondulado y rugoso, músculo; fluido sin
   pérdidas. Fuente puntual en el agua; trazas de presión en una fila de 32 receptores en el agua (ecos,
   tras quitar la onda directa) y otra en el músculo (transmisión). Error L2 crudo, tras alinear y tras
   alinear y escalar; desfase de cada traza.

Notas de diseño y convenciones:
  * La amplitud absoluta de una fuente "aditiva" se define igual en los dos códigos (incremento
    2 c dt / dx por paso, que emite la presión de la señal hacia cada lado); aun así, R, T y α se
    normalizan con la onda medida por el mismo código.
  * Orígenes de tiempo: wavesim etiqueta la muestra n con (n + 1) dt (la de la última fuente
    inyectada) y k-Wave con n dt. Se comparan diferencias de tiempos dentro de cada código (no
    dependen del origen) y, en las formas de onda, el desfase que mejor alinea.
  * Las comparaciones de forma de onda se hacen en frecuencia (DFT de cada traza en su propio muestreo,
    0-4 MHz): el desfase entre trazas es continuo y no hay error de interpolación.
  * k-Wave usa alpha_power = 1.1 (no 1.0) porque su término de dispersión lleva tan(pi y / 2), que
    diverge en y = 1.
  * PML: 20 celdas fuera de la región activa en ambos códigos (wavesim la lleva dentro de su malla
    ampliada; k-Wave con ``pml_inside=False``).
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
from wavesim.materials.acoustic import ACOUSTIC  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402
from wavesim.ultrasound import analytic as an  # noqa: E402
from wavesim.ultrasound import attenuation as att  # noqa: E402
from wavesim.ultrasound.elastic_fdtd import ElasticSolver, Receiver, Source  # noqa: E402
from wavesim.ultrasound.medium import ElasticMedium  # noqa: E402

F0 = 1.0e6
BW = 0.6
NPML = 20
DB_PER_NP = 20.0 / np.log(10.0)
#: Frecuencias de la comparación de formas de onda: el pulso (BW 0.6) no tiene energía apreciable sobre 3 MHz
F_CMP = np.arange(10e3, 4e6, 10e3)


# ----------------------------------------------------------------------------
# wavesim
# ----------------------------------------------------------------------------
def ws_medium(rho, cp, f0, alpha_db=None, y=1.1, n_mech=2) -> ElasticMedium:
    """Medio fluido para wavesim. ``rho``, ``cp``: (nz, nx) de la región activa (se amplía con la CPML)."""
    rho = np.pad(rho, NPML, mode="edge")
    cp = np.pad(cp, NPML, mode="edge")
    if alpha_db is None or n_mech == 0:
        z = np.zeros_like(rho)
        return ElasticMedium(rho, cp, z, z.copy(), z.copy(), np.zeros(0), f0)
    tau_sigma = att.relaxation_times(f0, n_mech)
    w = ws_weights(float(cp.flat[0]), alpha_db, y, n_mech, f0)
    return ElasticMedium(rho, cp, np.zeros_like(rho), w[:, None, None] * np.ones_like(rho),
                         np.zeros((n_mech, *rho.shape)), tau_sigma, f0)


def ws_weights(c0, alpha_db, y, n_mech, f0=F0):
    """Pesos GSLS que wavesim ajusta a ``alpha_db (f/1 MHz)^y`` dB/cm en su banda."""
    band = att.fit_band(f0, n_mech)
    alpha = alpha_db * (band / 1e6) ** y * 100.0 / DB_PER_NP
    return att.mechanism_weights(alpha[None], np.array([c0]), band, att.relaxation_times(f0, n_mech), f0)[0]


def ws_run(rho, cp, dx, src_cells, rec_cells, t_end, alpha_db=None, n_mech=2, cfl=0.9):
    """Ejecuta wavesim. ``src_cells``/``rec_cells``: listas de (j, i) en la región activa."""
    nz, nx = rho.shape
    grid = Grid2D(nx + 2 * NPML, nz + 2 * NPML, dx)
    phantom = Phantom(grid, np.full(grid.shape, TISSUE_INDEX["water"]))
    med = ws_medium(rho, cp, F0, alpha_db, n_mech=n_mech)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = ElasticSolver(phantom, F0, npml=NPML, medium=med, cfl=cfl)
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


def ws_group_delay_error(c, dx, dt, f=F0, df=1e3):
    """Retardo de grupo por metro del esquema de wavesim menos el exacto (s/m; < 0: llega antes).

    Derivada numérica de ``analytic.discrete_wavenumber`` (onda plana a lo largo de un eje). La
    envolvente viaja a la velocidad de grupo, cuyo error es ~3 veces el de la de fase cuando domina
    el término temporal (``(w dt)²/24`` en la fase, ``(w dt)²/8`` en el grupo).
    """
    w, dw = 2 * np.pi * f, 2 * np.pi * df
    k = an.discrete_wavenumber(np.array([w - dw, w + dw]), c, dx, dt)
    return float((k[1] - k[0]) / (2 * dw)) - 1.0 / c


# ----------------------------------------------------------------------------
# k-Wave
# ----------------------------------------------------------------------------
def kw_run(rho, cp, dx, src_mask, rec_cells, t_end, alpha_db=None, y=1.1, backend="python"):
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


def kwave_model(f, c0, alpha_db, y):
    """Velocidad de fase (m/s) y atenuación (Np/m) del modelo continuo de k-Wave para ``sound_speed = c0``.

    Ecuación de estado de k-Wave (``kspace_solver.py``): ``p = c0² (rho + tau L_y-2 (rho0 div u)
    - eta L_y-1 rho)``, con ``L_s = (-∇²)^(s/2)``, ``tau = -2 a0 c0^(y-1)`` y
    ``eta = 2 a0 c0^y tan(pi y / 2)``. Con ``e^{i(kx - wt)}`` queda
    ``w² = c0² k² - 2i w a0 c0^(y+1) k^y - 2 a0 c0^(y+2) tan(pi y/2) k^(y+1)``, que se resuelve
    por Newton desde ``k = w / c0``. A primer orden, ``Im k = a0 w^y`` y
    ``1/c = 1/c0 + a0 tan(pi y/2) w^(y-1)``: ``c0`` es la velocidad a frecuencia cero, no la de f0
    (con y = 1.1, tan < 0 y la onda va más deprisa que ``c0``).
    """
    w = 2 * np.pi * np.asarray(f, dtype=float)
    a0 = alpha_db * 100.0 / DB_PER_NP / (2 * np.pi * 1e6) ** y
    tn = np.tan(np.pi * y / 2)
    k = (w / c0).astype(complex)
    for _ in range(50):
        g = c0**2 * k**2 - 2j * w * a0 * c0 ** (y + 1) * k**y - 2 * a0 * c0 ** (y + 2) * tn * k ** (y + 1) - w**2
        dg = 2 * c0**2 * k - 2j * w * a0 * c0 ** (y + 1) * y * k ** (y - 1) \
            - 2 * a0 * c0 ** (y + 2) * tn * (y + 1) * k**y
        k = k - g / dg
    return w / k.real, k.imag


def kwave_matched(c_target, alpha_db_target, y, f0=F0):
    """``(sound_speed, alpha_coeff)`` de k-Wave con los que su modelo da ``c_target`` y ``alpha_db_target`` en f0."""
    c0, a = c_target, alpha_db_target
    for _ in range(60):
        c, al = kwave_model(np.array([f0]), c0, a, y)
        c0 *= c_target / c[0]
        a *= alpha_db_target * 100.0 / DB_PER_NP * (f0 / 1e6) ** y / al[0]
    return float(c0), float(a)


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


def spectrum(t, x, f=F_CMP):
    """Transformada de Fourier (``e^{-i w t}``) de trazas muestreadas en ``t``, en las frecuencias ``f``."""
    x = np.atleast_2d(x)
    return (x @ np.exp(-2j * np.pi * np.outer(t, f))) * (t[1] - t[0])


def gate(t, lo=None, hi=None, taper=0.5e-6):
    """Ventana 1 en [lo, hi] con bordes coseno de ``taper``."""
    w = np.ones_like(t)
    if lo is not None:
        w *= 0.5 * (1 + np.cos(np.pi * np.clip((lo - t) / taper, 0, 1)))
    if hi is not None:
        w *= 0.5 * (1 + np.cos(np.pi * np.clip((t - hi) / taper, 0, 1)))
    return w


def compare(W, K, max_shift=150e-9, f=F_CMP):
    """Diferencia L2 relativa (Parseval) de espectros ``W`` (wavesim) frente a ``K`` (referencia), por traza.

    ``raw``: tal cual; ``aligned``: con el desfase de wavesim que la minimiza (``shift`` > 0: hay
    que retrasar wavesim, que llega antes); ``scaled``: además con el factor de amplitud óptimo ``amp``.
    """
    W, K = np.atleast_2d(W), np.atleast_2d(K)
    shifts = np.arange(-max_shift, max_shift + 1e-13, 0.25e-9)
    ph = np.exp(-2j * np.pi * np.outer(shifts, f))
    nk = np.linalg.norm(K, axis=1)
    out = {k: [] for k in ("raw", "aligned", "shift", "scaled", "amp")}
    for w, k, n in zip(W, K, nk):
        ws = w[None, :] * ph
        e1 = np.linalg.norm(ws - k, axis=1) / n
        a = np.real(np.sum(np.conj(ws) * k, axis=1)) / np.sum(np.abs(w) ** 2)
        e2 = np.linalg.norm(a[:, None] * ws - k, axis=1) / n
        m = int(np.argmin(e1))
        out["raw"].append(np.linalg.norm(w - k) / n)
        out["aligned"].append(e1[m])
        out["shift"].append(shifts[m])
        out["scaled"].append(e2.min())
        out["amp"].append(a[int(np.argmin(e2))])
    return {k: np.array(v) for k, v in out.items()}


def total_error(W, K, shifts):
    """Error L2 relativo de todo el array a la vez, con el desfase de cada traza."""
    ws = W * np.exp(-2j * np.pi * np.asarray(shifts)[:, None] * F_CMP[None, :])
    return float(np.linalg.norm(ws - K) / np.linalg.norm(K)), float(np.linalg.norm(W - K) / np.linalg.norm(K))


# ----------------------------------------------------------------------------
# caso A: dos capas
# ----------------------------------------------------------------------------
LAYERS = dict(c1=1500.0, r1=1000.0, c2=1800.0, r2=1100.0, width=40e-3, depth=20e-3, z_src=2e-3, z_r0=4e-3,
              z_int=11e-3, z_r1=15e-3, t_end=18e-6)  # distancias cortas: las ondas de borde de la fuente llegan tarde


def layers_setup(dx):
    """Mapas, fuentes y receptores del caso A a la resolución ``dx``."""
    g = LAYERS
    nz, nx = int(round(g["depth"] / dx)), int(round(g["width"] / dx))
    j_int = int(round(g["z_int"] / dx))
    rho = np.full((nz, nx), g["r1"])
    cp = np.full((nz, nx), g["c1"])
    rho[j_int:], cp[j_int:] = g["r2"], g["c2"]
    j_src, j_r0, j_r1 = (int(round(g[k] / dx)) for k in ("z_src", "z_r0", "z_r1"))
    src_mask = np.zeros((nz, nx), bool)
    src_mask[j_src, :] = True
    rec = [(j_r0, nx // 2), (j_r1, nx // 2)]
    return dict(rho=rho, cp=cp, src=[(j_src, i) for i in range(nx)], src_mask=src_mask, rec=rec,
                j_int=j_int, j_src=j_src, j_r0=j_r0, j_r1=j_r1)


def layers_metrics(t, p, dx, j_int, j_src):
    """R, T (envolventes), R(f) espectral y tiempos de un código en el caso A."""
    g = LAYERS
    zi = (j_int - 0.5) * dx  # la interfaz queda entre los centros de las celdas j_int-1 y j_int
    # instante intermedio entre la onda incidente y la reflejada en el receptor 1
    split = pulse_peak_time(F0, BW) + (zi - j_src * dx) / g["c1"]
    ti, ai = peak(t, p[0], hi=split)
    tr, ar = peak(t, p[0], lo=split)
    tt, at = peak(t, p[1])
    f = np.array([0.7e6, 1.0e6, 1.3e6])
    inc = spectrum(t, p[0] * gate(t, hi=split), f)[0]
    ref = spectrum(t, p[0] * gate(t, lo=split), f)[0]
    return dict(t_inc=ti, a_inc=ai, t_ref=tr, a_ref=ar, t_tr=tt, a_tr=at, R=ar / ai, T=at / ai,
                dt_ref=tr - ti, dt_tr=tt - ti, R_f=np.abs(ref / inc), f=f)


def case_layers(dx, backend, cfls=(0.9, 0.3)):
    g = LAYERS
    s = layers_setup(dx)
    runs = {}
    for cfl in cfls:
        tw, pw, dtw, elw = ws_run(s["rho"], s["cp"], dx, s["src"], s["rec"], g["t_end"], cfl=cfl)
        runs[f"wavesim CFL {cfl}"] = (tw, pw, dtw, elw)
    tk, pk, dtk, elk = kw_run(s["rho"], s["cp"], dx, s["src_mask"], s["rec"], g["t_end"], backend=backend)
    runs["k-Wave"] = (tk, pk, dtk, elk)

    Z1, Z2 = g["r1"] * g["c1"], g["r2"] * g["c2"]
    R_an, T_an = (Z2 - Z1) / (Z2 + Z1), 2 * Z2 / (Z1 + Z2)
    zi = (s["j_int"] - 0.5) * dx
    z_r0e, z_r1e = s["j_r0"] * dx, s["j_r1"] * dx
    exp_ref = 2 * (zi - z_r0e) / g["c1"]
    exp_tr = (zi - z_r0e) / g["c1"] + (z_r1e - zi) / g["c2"]
    res, shape, pred = {}, {}, {}
    K = spectrum(tk, pk)
    for name, (t, p, dt, el) in runs.items():
        res[name] = layers_metrics(t, p, dx, s["j_int"], s["j_src"]) | dict(steps=len(t), dt=dt, elapsed=el)
        if name.startswith("wavesim"):
            shape[name] = compare(spectrum(t, p), K)
            e1, e2 = ws_group_delay_error(g["c1"], dx, dt), ws_group_delay_error(g["c2"], dx, dt)
            pred[name] = (2 * (zi - z_r0e) * e1, (zi - z_r0e) * e1 + (z_r1e - zi) * e2)
    return dict(dx=dx, ppw=g["c1"] / (F0 * dx), R_an=R_an, T_an=T_an, exp_ref=exp_ref, exp_tr=exp_tr,
                res=res, shape=shape, pred=pred, traces={k: (v[0], v[1]) for k, v in runs.items()})


def print_layers(r):
    print(f"\n--- dos capas, dx = {r['dx']*1e3:.2f} mm ({r['ppw']:.0f} puntos/long. de onda en la capa 1) ---")
    print("pasos: " + ", ".join(f"{k} {v['steps']} (dt={v['dt']*1e9:.1f} ns, {v['elapsed']:.1f} s)"
                                for k, v in r["res"].items()))
    print(f"analítico: R = {r['R_an']:.4f}, T = {r['T_an']:.4f}, "
          f"t(refl)-t(inc) = {r['exp_ref']*1e6:.3f} us, t(trans)-t(inc) = {r['exp_tr']*1e6:.3f} us")
    print(f"{'':18s}{'R':>8s}{'T':>8s}{'err R':>8s}{'err T':>8s}{'R(0.7/1.0/1.3 MHz)':>24s}"
          f"{'dt_ref us':>11s}{'dt_tr us':>10s}{'t_inc us':>10s}")
    for k, v in r["res"].items():
        print(f"{k:18s}{v['R']:8.4f}{v['T']:8.4f}{(v['R']/r['R_an']-1)*100:+7.2f}%{(v['T']/r['T_an']-1)*100:+7.2f}%"
              f"   {' '.join(f'{x:.4f}' for x in v['R_f'])}  {v['dt_ref']*1e6:9.3f}{v['dt_tr']*1e6:10.3f}"
              f"{v['t_inc']*1e6:10.3f}")
    k = r["res"]["k-Wave"]
    for name, sh in r["shape"].items():
        w = r["res"][name]
        pr, ptr = r["pred"][name]
        print(f"{name} - k-Wave: R {(w['R']/k['R']-1)*100:+.2f} %, T {(w['T']/k['T']-1)*100:+.2f} %, "
              f"dt_ref {(w['dt_ref']-k['dt_ref'])*1e9:+.1f} ns ({(w['dt_ref']/k['dt_ref']-1)*100:+.2f} %), "
              f"dt_tr {(w['dt_tr']-k['dt_tr'])*1e9:+.1f} ns ({(w['dt_tr']/k['dt_tr']-1)*100:+.2f} %); "
              f"retardo de grupo del esquema predicho: {pr*1e9:+.1f} / {ptr*1e9:+.1f} ns")
        for i, lab in enumerate(("receptor 1 (incidente + reflejada)", "receptor 2 (transmitida)")):
            print(f"    L2 {lab}: crudo {sh['raw'][i]*100:.1f} %, alineado ({sh['shift'][i]*1e9:+.1f} ns) "
                  f"{sh['aligned'][i]*100:.2f} %, alineado y escalado ({sh['amp'][i]:.4f}) {sh['scaled'][i]*100:.2f} %")


# ----------------------------------------------------------------------------
# caso B: fuente puntual con atenuación
# ----------------------------------------------------------------------------
POINT = dict(c0=1580.0, rho0=1090.0, a0=2.0, y=1.1, side=50e-3, rs=(5e-3, 10e-3, 15e-3), t_end=17e-6)
POINT_FREQS = np.array([0.7e6, 0.85e6, 1.0e6, 1.15e6, 1.3e6])


def point_setup(dx):
    g = POINT
    n = int(round(g["side"] / dx))
    js = n // 2
    rec = [(js + int(round(r / dx)), js) for r in g["rs"]]
    src_mask = np.zeros((n, n), bool)
    src_mask[js, js] = True
    return dict(rho=np.full((n, n), g["rho0"]), cp=np.full((n, n), g["c0"]), src=[(js, js)], src_mask=src_mask,
                rec=rec, r=[(j - js) * dx for j, _ in rec])


def point_estimate(t, p, r, c_guess, freqs=POINT_FREQS):
    """Atenuación (Np/m) y velocidad de fase entre el 1.er y el 3.er receptor (expansión ~ 1/sqrt(r))."""
    P = spectrum(t, p[[0, 2]], freqs)
    dr = r[2] - r[0]
    ratio = P[1] / P[0]
    # un retardo tau resta fase 2*pi*f*tau; la ambigüedad de 2*pi se fija con c ~ c_guess
    dphi = (-np.angle(ratio)) % (2 * np.pi)
    n2pi = np.round((2 * np.pi * freqs * dr / c_guess - dphi) / (2 * np.pi))
    c = 2 * np.pi * freqs * dr / (dphi + 2 * np.pi * n2pi)
    alpha = -(np.log(np.abs(ratio)) + 0.5 * np.log(r[2] / r[0])) / dr
    return alpha, c


def case_point(dx, backend, n_mechs=(1, 2, 3)):
    g = POINT
    s = point_setup(dx)
    c_kw, a_kw = kwave_matched(g["c0"], g["a0"], g["y"])
    runs, models = {}, {}
    for name, c0, a0 in (("k-Wave nominal", g["c0"], g["a0"]), ("k-Wave ajustado", c_kw, a_kw)):
        t, p, dt, el = kw_run(s["rho"], np.full_like(s["cp"], c0), dx, s["src_mask"], s["rec"], g["t_end"],
                              alpha_db=a0, y=g["y"], backend=backend)
        runs[name] = (t, p, el)
        models[name] = kwave_model(POINT_FREQS, c0, a0, g["y"])
    for nm in n_mechs:
        t, p, dt, el = ws_run(s["rho"], s["cp"], dx, s["src"], s["rec"], g["t_end"], alpha_db=g["a0"], n_mech=nm)
        name = f"wavesim {nm} mec."
        runs[name] = (t, p, el)
        models[name] = att.model_wave(g["c0"], ws_weights(g["c0"], g["a0"], g["y"], nm)[:, None],
                                      att.relaxation_times(F0, nm), F0, POINT_FREQS)
    out = {}
    for name, (t, p, el) in runs.items():
        alpha, c = point_estimate(t, p, s["r"], g["c0"])
        out[name] = dict(alpha_np=alpha, c=c, elapsed=el, model_c=models[name][0], model_alpha=models[name][1])
    alpha_th = g["a0"] * (POINT_FREQS / 1e6) ** g["y"] * 100.0 / DB_PER_NP
    c_th = att.power_law_phase_speed(g["c0"], g["a0"] * 100.0 / DB_PER_NP, g["y"], F0, POINT_FREQS)
    return dict(freqs=POINT_FREQS, alpha_th=alpha_th, c_th=c_th, runs=out, r=s["r"], kw_matched=(c_kw, a_kw),
                traces={k: (v[0], v[1]) for k, v in runs.items()})


def print_point(r):
    g = POINT
    print(f"\n--- fuente puntual, alpha = {g['a0']} dB/cm/MHz^{g['y']}, c = {g['c0']:.0f} m/s ---")
    print(f"receptores a {', '.join(f'{x*1e3:.0f}' for x in r['r'])} mm; medida entre el 1.º y el 3.º; "
          f"k-Wave ajustado: sound_speed = {r['kw_matched'][0]:.2f} m/s, alpha_coeff = {r['kw_matched'][1]:.4f}")

    def row(label, vals, fmt="{:10.3f}"):
        print(label.ljust(34) + "".join(fmt.format(v) for v in vals))

    row("f (MHz)", r["freqs"] / 1e6, "{:10.2f}")
    row("alpha ley nominal (dB/cm)", r["alpha_th"] * DB_PER_NP / 100)
    for name, v in r["runs"].items():
        row(f"alpha {name} (dB/cm)", v["alpha_np"] * DB_PER_NP / 100)
        row("   su modelo (dB/cm)", v["model_alpha"] * DB_PER_NP / 100)
        row("   medida/modelo - 1 (%)", (v["alpha_np"] / v["model_alpha"] - 1) * 100, "{:+10.2f}")
        row("   medida/ley nominal - 1 (%)", (v["alpha_np"] / r["alpha_th"] - 1) * 100, "{:+10.2f}")
    row("c ley nominal + Kramers-Kronig", r["c_th"], "{:10.1f}")
    for name, v in r["runs"].items():
        row(f"c {name} (m/s)", v["c"], "{:10.1f}")
        row("   su modelo (m/s)", v["model_c"], "{:10.1f}")
    ref = r["runs"]["k-Wave ajustado"]
    for name, v in r["runs"].items():
        if name.startswith("wavesim"):
            row(f"{name} / k-Wave ajustado: alpha (%)", (v["alpha_np"] / ref["alpha_np"] - 1) * 100, "{:+10.2f}")
            row("   c (%)", (v["c"] / ref["c"] - 1) * 100, "{:+10.3f}")
    print("tiempos: " + ", ".join(f"{k} {v['elapsed']:.1f} s" for k, v in r["runs"].items()))


# ----------------------------------------------------------------------------
# caso C: medio heterogéneo tipo tejido
# ----------------------------------------------------------------------------
TISSUE = dict(width=32e-3, depth=26e-3, z_src=1.0e-3, z_echo=1.5e-3, z_trans=23e-3, n_rec=32, rec_margin=3e-3,
              t_end=24e-6)


def tissue_interfaces(x):
    """Profundidades (m) de las interfaces agua|piel, piel|grasa y grasa|músculo en la posición lateral ``x``."""
    xc = TISSUE["width"] / 2
    z_skin = 5.0e-3 + 1.2e-3 * ((x - xc) / xc) ** 2  # superficie curva (cuenco)
    z_fat = z_skin + 1.8e-3
    # fondo de la grasa ondulado y rugoso (dos escalas)
    z_muscle = 13.0e-3 + 1.2e-3 * np.sin(2 * np.pi * x / 14e-3) + 0.3e-3 * np.sin(2 * np.pi * x / 2.3e-3 + 1.0)
    return z_skin, z_fat, z_muscle


def tissue_setup(dx):
    g = TISSUE
    nz, nx = int(round(g["depth"] / dx)), int(round(g["width"] / dx))
    x = np.arange(nx) * dx
    z = np.arange(nz)[:, None] * dx
    z_skin, z_fat, z_muscle = tissue_interfaces(x)
    labels = np.where(z < z_skin, 0, np.where(z < z_fat, 1, np.where(z < z_muscle, 2, 3)))
    keys = ("water", "skin", "fat", "muscle")
    rho = np.choose(labels, [ACOUSTIC[k].density for k in keys]).astype(float)
    cp = np.choose(labels, [ACOUSTIC[k].c_p for k in keys]).astype(float)
    i_rec = np.round(np.linspace(g["rec_margin"], g["width"] - g["rec_margin"], g["n_rec"]) / dx).astype(int)
    j_src, j_e, j_t = (int(round(g[k] / dx)) for k in ("z_src", "z_echo", "z_trans"))
    src_mask = np.zeros((nz, nx), bool)
    src_mask[j_src, nx // 2] = True
    rec = [(j_e, i) for i in i_rec] + [(j_t, i) for i in i_rec]
    # llegada (pico) de la onda directa a la fila de ecos, para separarla de los ecos
    dist = np.hypot((i_rec - nx // 2) * dx, (j_e - j_src) * dx)
    t_direct = pulse_peak_time(F0, BW) + dist / ACOUSTIC["water"].c_p
    return dict(rho=rho, cp=cp, labels=labels, src=[(j_src, nx // 2)], src_mask=src_mask, rec=rec,
                x_rec=i_rec * dx, t_direct=t_direct)


#: Semiduración de la onda directa que se quita de las trazas de eco (el pulso de BW 0.6 dura ~5 us)
ECHO_GATE = 2.6e-6


def tissue_spectra(t, p, t_direct):
    """Espectros de las trazas de eco (sin la onda directa) y de transmisión."""
    n = TISSUE["n_rec"]
    echo = p[:n] * np.stack([gate(t, lo=td + ECHO_GATE) for td in t_direct])
    return spectrum(t, echo), spectrum(t, p[n:])


def tissue_arrivals(t, p):
    """Instante del máximo de la envolvente de cada traza de transmisión (primera llegada, la más fuerte)."""
    return np.array([peak(t, x)[0] for x in p[TISSUE["n_rec"]:]])


def case_tissue(dx, backend, cfls=(0.9, 0.3)):
    g = TISSUE
    s = tissue_setup(dx)
    runs = {}
    tk, pk, dtk, elk = kw_run(s["rho"], s["cp"], dx, s["src_mask"], s["rec"], g["t_end"], backend=backend)
    runs["k-Wave"] = (tk, pk, elk)
    for cfl in cfls:
        tw, pw, dtw, elw = ws_run(s["rho"], s["cp"], dx, s["src"], s["rec"], g["t_end"], cfl=cfl)
        runs[f"wavesim CFL {cfl}"] = (tw, pw, elw)
    Ke, Kt = tissue_spectra(tk, pk, s["t_direct"])
    t_k = tissue_arrivals(tk, pk)
    out = {}
    for name, (t, p, el) in runs.items():
        if name == "k-Wave":
            continue
        We, Wt = tissue_spectra(t, p, s["t_direct"])
        ce, ct = compare(We, Ke), compare(Wt, Kt)
        t_w = tissue_arrivals(t, p)
        travel = t_k - pulse_peak_time(F0, BW)
        out[name] = dict(echo=ce, trans=ct, echo_total=total_error(We, Ke, ce["shift"]),
                         trans_total=total_error(Wt, Kt, ct["shift"]), dt_arrival=t_w - t_k,
                         rel_arrival=(t_w - t_k) / travel, elapsed=el)
    return dict(dx=dx, res=out, elapsed_kw=elk, traces={k: (v[0], v[1]) for k, v in runs.items()},
                t_direct=s["t_direct"], x_rec=s["x_rec"])


def print_tissue(r):
    print(f"\n--- medio tipo tejido (agua | piel curva | grasa con fondo rugoso | músculo), dx = {r['dx']*1e3:.2f} mm,"
          f" {TISSUE['n_rec']} receptores de eco (z = {TISSUE['z_echo']*1e3:.1f} mm) y de transmisión "
          f"(z = {TISSUE['z_trans']*1e3:.0f} mm); k-Wave {r['elapsed_kw']:.1f} s ---")
    for name, v in r["res"].items():
        print(f"{name} ({v['elapsed']:.1f} s) frente a k-Wave:")
        for lab, key in (("ecos", "echo"), ("transmisión", "trans")):
            c = v[key]
            tot_al, tot_raw = v[f"{key}_total"]
            print(f"  {lab:12s} L2 del array: crudo {tot_raw*100:.1f} %, alineado {tot_al*100:.2f} % | por traza "
                  f"(mediana / máx): crudo {np.median(c['raw'])*100:.1f} / {c['raw'].max()*100:.1f} %, alineado "
                  f"{np.median(c['aligned'])*100:.2f} / {c['aligned'].max()*100:.2f} %, alineado y escalado "
                  f"{np.median(c['scaled'])*100:.2f} / {c['scaled'].max()*100:.2f} %; desfase "
                  f"{np.median(c['shift'])*1e9:+.1f} ns [{c['shift'].min()*1e9:+.1f}, {c['shift'].max()*1e9:+.1f}],"
                  f" amplitud {np.median(c['amp']):.4f}")
        print(f"  primera llegada en transmisión, wavesim - k-Wave: {np.median(v['dt_arrival'])*1e9:+.1f} ns "
              f"[{v['dt_arrival'].min()*1e9:+.1f}, {v['dt_arrival'].max()*1e9:+.1f}], relativa al tiempo de vuelo "
              f"{np.median(v['rel_arrival'])*100:+.3f} % [{v['rel_arrival'].min()*100:+.3f}, "
              f"{v['rel_arrival'].max()*100:+.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=("all", "layers", "point", "tissue"), default="all")
    ap.add_argument("--backend", choices=("python", "cpp"), default="python",
                    help="python (por defecto, puro NumPy) o cpp (binario OMP/CUDA incluido en el wheel)")
    args = ap.parse_args()
    try:
        import kwave  # noqa: F401
    except ImportError:
        sys.exit("Falta k-wave-python: pip install k-wave-python scipy (en un entorno aparte, ver la cabecera)")
    warnings.filterwarnings("ignore", category=FutureWarning)
    if args.case in ("all", "layers"):
        for dx in (0.20e-3, 0.15e-3, 0.10e-3, 0.05e-3):
            print_layers(case_layers(dx, args.backend))
    if args.case in ("all", "point"):
        print_point(case_point(0.1e-3, args.backend))
    if args.case in ("all", "tissue"):
        print_tissue(case_tissue(0.1e-3, args.backend))


if __name__ == "__main__":
    main()
