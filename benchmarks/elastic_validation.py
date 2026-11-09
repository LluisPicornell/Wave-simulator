"""Validación cuantitativa del solver elástico frente a soluciones analíticas (sin software externo).

Casos U3, U4 y U5 del plan de ``docs/comparativa_simuladores.md`` (§11.1) y la
onda de Rayleigh; método, tablas y lectura en ``docs/validacion/elastico.md``.
Todos los medios son isótropos y sin atenuación (``ElasticMedium`` construido a
mano, ``tau = 0``); las propiedades son las de ``wavesim.materials.acoustic``.

  u3  Convergencia al variar dx y la CFL:
      (a) onda plana en medio homogéneo (agua; P y S en cortical): velocidad de
          fase en f0, error L2 de la forma de onda y comparación con la
          relación de dispersión del propio esquema;
      (b) reflexión y transmisión a incidencia normal en dos capas fluidas y en
          agua-cortical, con variantes de diagnóstico (salto sólo de c o sólo
          de rho) y la posición efectiva de la interfaz.
  u4  Interfaz agua-cortical: R(θ) de presión y amplitud de la onda S
      transmitida frente a la solución exacta fluido-sólido (Zoeppritz /
      Brekhovskikh), incluidos los ángulos críticos P y S y la fase compleja
      pasado el crítico. Haces gaussianos inclinados; descomposición en ondas
      planas a lo largo de una línea (DFT en x y en t).
  u5  Ondas de Lamb en una placa de cortical de 3 mm, libre (en aire) y en
      agua: velocidad de fase de S0 y A0 entre 100 y 500 kHz por transformada
      f-k frente a las raíces de Rayleigh-Lamb (con carga de fluido en agua).
  rayleigh  Onda de Rayleigh en la superficie libre (aire) de la cortical.
  backends  El backend Numba ("auto") da lo mismo que ``backend="numpy"``.

Uso (desde la raíz del repositorio):

    NUMBA_NUM_THREADS=1 PYTHONPATH=$(pwd) python benchmarks/elastic_validation.py            # todo
    NUMBA_NUM_THREADS=1 PYTHONPATH=$(pwd) python benchmarks/elastic_validation.py --case u4  # un caso
    ... --backend numpy      # fuerza la referencia NumPy (unas 3 veces más lento)

Tiempos con 1 hilo de Numba: u3 ~1 min, u4 ~X min, u5 ~X min, rayleigh ~X min.
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
from wavesim.core.signals import envelope, gaussian_pulse  # noqa: E402
from wavesim.materials.acoustic import ACOUSTIC  # noqa: E402
from wavesim.phantoms.phantom import Phantom  # noqa: E402
from wavesim.ultrasound import analytic as an  # noqa: E402
from wavesim.ultrasound.elastic_fdtd import ElasticSolver, Source  # noqa: E402
from wavesim.ultrasound.medium import ElasticMedium  # noqa: E402

F0 = 1.0e6
NPML = 20


def _props(key: str) -> tuple[float, float, float]:
    p = ACOUSTIC[key]
    return (p.density, p.c_p, p.c_s)


WATER = _props("water")  # (rho, cp, cs) = (1000, 1500, 0)
BONE = _props("cortical_bone")  # (1908, 3500, 1800)
AIR = _props("air")  # (1.2, 343, 0)


# ----------------------------------------------------------------------------
# utilidades comunes
# ----------------------------------------------------------------------------
def make_solver(rho, cp, cs, dx, f0=F0, cfl=0.9, backend="auto", npml=NPML) -> ElasticSolver:
    """Solver sin atenuación sobre mapas (nz, nx) de rho, cp y cs (el fantoma sólo aporta la malla)."""
    rho, cp, cs = (np.asarray(a, dtype=float) for a in (rho, cp, cs))
    grid = Grid2D(rho.shape[1], rho.shape[0], dx)
    zero = np.zeros_like(rho)
    medium = ElasticMedium(rho=rho, cp=cp, cs=cs, tau_p=zero, tau_s=zero.copy(), tau_sigma=np.zeros(0), f0=f0)
    with warnings.catch_warnings():
        # el aviso de pocos puntos por longitud de onda es intencionado en los barridos de dx
        # y en el aire (onda que no lleva energía)
        warnings.simplefilter("ignore")
        return ElasticSolver(Phantom(grid, np.zeros(grid.shape, np.int16)), f0, npml=npml, cfl=cfl, medium=medium,
                             backend=backend)


def layered(nz: int, nx: int, layers: list[tuple[int, tuple[float, float, float]]]):
    """Mapas (rho, cp, cs) por filas: ``layers = [(fila_inicial, (rho, cp, cs)), ...]`` en orden creciente."""
    maps = [np.zeros((nz, nx)) for _ in range(3)]
    for n, (j0, props) in enumerate(layers):
        j1 = layers[n + 1][0] if n + 1 < len(layers) else nz
        for m, v in zip(maps, props):
            m[j0:j1] = v
    return maps


def sample(solver: ElasticSolver, field: str, j, i):
    """Valor de un campo en celdas ``(j, i)``; "p" = presión, "-szz" = tracción normal cambiada de signo."""
    if field == "p":
        return -0.5 * (solver.sxx[j, i] + solver.szz[j, i])
    if field == "-szz":
        return -solver.szz[j, i]
    return getattr(solver, field)[j, i]


def run_traces(solver: ElasticSolver, nt: int, probes: dict) -> dict[str, np.ndarray]:
    """Avanza ``nt`` pasos y guarda ``probes = {nombre: (campo, j, i)}`` en cada paso (arrays (nt, ...))."""
    out = {k: np.zeros((nt,) + np.shape(sample(solver, f, j, i))) for k, (f, j, i) in probes.items()}
    for n in range(nt):
        solver.step(n)
        for k, (f, j, i) in probes.items():
            out[k][n] = sample(solver, f, j, i)
    return out


def dft(x: np.ndarray, t: np.ndarray, f: float) -> complex:
    """Componente de Fourier en ``f`` con la convención física ``exp(-i w t)`` (fase = +w t)."""
    return complex(np.sum(x * np.exp(2j * np.pi * f * t)))


def gate(t: np.ndarray, center: float, half: float, taper: float = 1e-6) -> np.ndarray:
    """Ventana plana de semiancho ``half`` alrededor de ``center`` con bordes coseno de ``taper``."""
    d = np.abs(t - center) - half
    w = np.where(d <= 0, 1.0, 0.5 * (1 + np.cos(np.pi * np.clip(d / taper, 0, 1))))
    return np.where(d >= taper, 0.0, w)


def slopes(dx, err) -> list[float]:
    """Pendientes locales log-log de |err| frente a dx entre resoluciones consecutivas."""
    dx, err = np.asarray(dx), np.abs(np.asarray(err))
    return list(np.log(err[:-1] / err[1:]) / np.log(dx[:-1] / dx[1:]))


def fit_slope(dx, err) -> float:
    """Pendiente de mínimos cuadrados de log|err| frente a log dx."""
    return float(np.polyfit(np.log(dx), np.log(np.abs(err)), 1)[0])


def _fmt_slopes(s) -> str:
    return " ".join(f"{v:4.2f}" for v in s)


# ----------------------------------------------------------------------------
# U3 (a): onda plana en medio homogéneo
# ----------------------------------------------------------------------------
def plane_wave_1d(medium=WATER, kind="pressure", ppw=12.0, cfl=0.9, dt=None, n_lambda=20, nx=6,
                  backend="auto", f0=F0) -> dict:
    """Onda plana en z en un medio homogéneo; velocidad de fase en ``f0`` y error L2 frente a la exacta.

    La fuente es una fila entera de celdas y el medio no varía en x, así que el
    problema 2D es exactamente 1D (las derivadas en x son nulas en todo el
    dominio, también dentro de la CPML lateral) y basta una franja de ``nx``
    columnas. ``kind="force_x"`` lanza una onda S (sólo en sólidos). Con ``dt``
    se fija el paso temporal (la CFL se ajusta) para aislar el error espacial.

    Métricas, entre dos receptores separados ``n_lambda`` longitudes de onda:
    - ``err``: c medida (fase de la DFT en f0) / c - 1;
    - ``pred``: la misma cantidad predicha por ``analytic.discrete_wavenumber``;
    - ``l2``: ||b(t) - a(t - L/c)|| / ||b|| (forma de onda frente a la propagación exacta).
    """
    rho, cp, cs = medium
    c = cs if kind == "force_x" else cp
    lam = c / f0
    dx = lam / ppw
    if dt is not None:
        cfl = dt * cp * np.sqrt(2.0) * (an.C1 + abs(an.C2)) / dx
    z_src, z_a = 4 * lam, 7 * lam
    j_s = NPML + int(round(z_src / dx))
    j_a = NPML + int(round(z_a / dx))
    j_b = j_a + int(round(n_lambda * lam / dx))
    L = (j_b - j_a) * dx
    nz = j_b + int(round(6 * lam / dx)) + NPML
    s = make_solver(np.full((nz, nx), rho), np.full((nz, nx), cp), np.full((nz, nx), cs), dx, f0, cfl, backend)
    nt = int(np.ceil(((j_b - j_s) * dx / c + 8 / f0) / s.dt))
    t = s.time_axis(nt)
    s.add_source(Source(np.full(nx, j_s), np.arange(nx), np.zeros(nx, int), gaussian_pulse(t, f0)[None, :], kind))
    field = "vx" if kind == "force_x" else "p"
    ic = nx // 2
    tr = run_traces(s, nt, {"a": (field, j_a, ic), "b": (field, j_b, ic)})
    a, b = tr["a"], tr["b"]
    ratio = dft(b, t, f0) / dft(a, t, f0)
    # retardo = fase / w; la ambigüedad de 2π se resuelve con la velocidad nominal
    phase = np.angle(ratio)
    n2pi = np.round((2 * np.pi * f0 * L / c - phase) / (2 * np.pi))
    c_meas = 2 * np.pi * f0 * L / (phase + 2 * np.pi * n2pi)
    c_pred = 2 * np.pi * f0 / an.discrete_wavenumber(2 * np.pi * f0, c, dx, s.dt)
    nfft = 1 << int(np.ceil(np.log2(4 * nt)))
    w = 2 * np.pi * np.fft.rfftfreq(nfft, s.dt)
    a_exact = np.fft.irfft(np.fft.rfft(a, nfft) * np.exp(-1j * w * L / c), nfft)[:nt]
    return dict(dx=dx, dt=s.dt, cfl=cfl, ppw=ppw, err=c_meas / c - 1, pred=c_pred / c - 1,
                l2=float(np.linalg.norm(b - a_exact) / np.linalg.norm(b)), amp=abs(ratio), nt=nt)


# ----------------------------------------------------------------------------
# U3 (b): reflexión y transmisión a incidencia normal
# ----------------------------------------------------------------------------
def layers_1d(m1=WATER, m2=BONE, ppw=12.0, cfl=0.9, nx=6, backend="auto", f0=F0) -> dict:
    """R y T de presión (tracción normal en el sólido) a incidencia normal entre dos semiespacios.

    Medio 2 en las filas ``j >= j_int``: la interfaz queda en ``(j_int - 1/2) dx``,
    sobre la fila de nodos ``vz`` (densidad media aritmética). Dos ejecuciones con
    el mismo ``dt``: con interfaz y sólo con el medio 1. La incidente en el
    receptor ``a`` es la de la segunda; la reflejada, la diferencia. En 1D sin
    pérdidas la amplitud de cada onda no cambia al propagarse, así que
    ``|R| = |U(f0)| / |D(f0)|`` y ``|T| = |P_b(f0)| / |D(f0)|`` no dependen de
    la dispersión numérica. La fase de R, corregida con el ``k`` discreto del
    medio 1, da el desplazamiento efectivo de la interfaz ``delta`` (en celdas).
    Se compara también con el cociente de máximos de envolvente (como en §10.2).
    """
    (r1, c1, _), (r2, c2, _) = m1, m2
    lam = c1 / f0
    dx = lam / ppw
    j_s, j_a, j_int, j_b = (NPML + int(round(z * lam / dx)) for z in (4, 9, 15, 21))
    nz = j_b + int(round(9 * lam / dx)) + NPML
    cmax = max(c1, c2)
    runs = {}
    for name, two in (("ref", False), ("int", True)):
        lay = [(0, m1)] + ([(j_int, m2)] if two else [])
        rho, cp, cs = layered(nz, nx, lay)
        s = make_solver(rho, cp, cs, dx, f0, cfl if two else cfl * c1 / cmax, backend)
        nt = int(np.ceil((2 * (j_int - j_s) * dx / c1 + 10 / f0) / s.dt))
        t = s.time_axis(nt)
        s.add_source(Source(np.full(nx, j_s), np.arange(nx), np.zeros(nx, int), gaussian_pulse(t, f0)[None, :]))
        runs[name] = (s.dt, t, run_traces(s, nt, {"a": ("p", j_a, nx // 2), "b": ("-szz", j_b, nx // 2)}))
    (dt_r, _, ref), (dt, t, two) = runs["ref"], runs["int"]
    assert abs(dt_r / dt - 1) < 1e-12, "las dos ejecuciones deben compartir dt"
    # ventanas que aíslan cada pulso de los ecos (débiles) de la CPML
    t_pk = 4 * (2 * np.sqrt(2 * np.log(2)) / (2 * np.pi * 0.6 * f0))  # instante del máximo del pulso
    half = 3.5 / f0
    D = ref["a"] * gate(t, t_pk + (j_a - j_s) * dx / c1, half)
    U = (two["a"] - ref["a"]) * gate(t, t_pk + (2 * j_int - j_a - j_s) * dx / c1, half)
    P = two["b"] * gate(t, t_pk + (j_int - j_s) * dx / c1 + (j_b - j_int) * dx / c2, half)
    Z1, Z2 = r1 * c1, r2 * c2
    R_an, T_an = (Z2 - Z1) / (Z2 + Z1), 2 * Z2 / (Z1 + Z2)
    h = (j_int - 0.5 - j_a) * dx
    k1 = an.discrete_wavenumber(2 * np.pi * f0, c1, dx, dt)
    Rc = dft(U, t, f0) / dft(D, t, f0) * np.exp(-2j * k1 * h)
    R = abs(Rc) * np.sign(np.real(Rc))
    T = abs(dft(P, t, f0) / dft(D, t, f0))
    delta = -np.angle(Rc * np.sign(R_an)) / (2 * k1) / dx
    R_env = np.sign(R_an) * envelope(U).max() / envelope(D).max()
    T_env = envelope(P).max() / envelope(D).max()
    return dict(dx=dx, ppw=ppw, R=R, T=T, R_an=R_an, T_an=T_an, eR=R / R_an - 1, eT=T / T_an - 1, delta=delta,
                eR_env=R_env / R_an - 1, eT_env=T_env / T_an - 1)


def run_u3(backend="auto", quick=False) -> dict:
    ppws = (6, 8, 12, 16, 24, 32) if not quick else (8, 12, 16)
    out = {}
    print("\n=== U3 (a) onda plana en agua, CFL 0.9: velocidad de fase en f0 = 1 MHz ===")
    print("ppw   dx(um)   c_med/c-1   predicho(esquema)   error L2   |b|/|a|")
    rows = [plane_wave_1d(WATER, "pressure", p, backend=backend) for p in ppws]
    for r in rows:
        print(f"{r['ppw']:3.0f} {r['dx']*1e6:8.2f} {r['err']:+11.3e} {r['pred']:+15.3e} {r['l2']:12.3e} "
              f"{r['amp']:9.6f}")
    dxs = [r["dx"] for r in rows]
    print("pendientes locales err c:", _fmt_slopes(slopes(dxs, [r["err"] for r in rows])),
          "| L2:", _fmt_slopes(slopes(dxs, [r["l2"] for r in rows])))
    out["water_cfl"] = rows

    dt_fixed = plane_wave_1d(WATER, "pressure", 32, cfl=0.1, backend=backend)["dt"]
    print(f"\n--- mismo caso con dt fijo = {dt_fixed*1e9:.2f} ns (CFL 0.1 a 32 ppw): error espacial ---")
    print("ppw   CFL     c_med/c-1   predicho(esquema)   error L2")
    rows = [plane_wave_1d(WATER, "pressure", p, dt=dt_fixed, backend=backend) for p in (4, 6, 8, 12, 16)]
    for r in rows:
        print(f"{r['ppw']:3.0f} {r['cfl']:6.3f} {r['err']:+11.3e} {r['pred']:+15.3e} {r['l2']:12.3e}")
    print("pendientes locales err c:", _fmt_slopes(slopes([r["dx"] for r in rows], [r["err"] for r in rows])))
    out["water_dt"] = rows

    print("\n--- 10 ppw, CFL variable ---")
    print("CFL    c_med/c-1   predicho   error L2")
    rows = [plane_wave_1d(WATER, "pressure", 10, cfl=c, backend=backend) for c in (0.2, 0.4, 0.6, 0.8, 0.9, 0.99)]
    for r in rows:
        print(f"{r['cfl']:4.2f} {r['err']:+11.3e} {r['pred']:+10.3e} {r['l2']:10.3e}")
    out["water_cfl_sweep"] = rows

    print("\n--- cortical: ondas P (fuente de presión) y S (fuerza horizontal), CFL 0.9 (dt fijado por cp) ---")
    print("ppw(propio)  P: c/c-1   predicho   L2      |  S: c/c-1   predicho   L2")
    rows = []
    for p in ppws:
        rp = plane_wave_1d(BONE, "pressure", p, backend=backend)
        rs = plane_wave_1d(BONE, "force_x", p, backend=backend)
        rows.append((rp, rs))
        print(f"{p:6.0f}     {rp['err']:+10.3e} {rp['pred']:+10.3e} {rp['l2']:8.2e} | {rs['err']:+10.3e} "
              f"{rs['pred']:+10.3e} {rs['l2']:8.2e}")
    out["bone"] = rows

    print("\n=== U3 (b) incidencia normal: R y T de presión frente a la impedancia ===")
    cases = {
        "fluido-fluido 1500/1000 | 1800/1100": (WATER, (1100.0, 1800.0, 0.0)),
        "agua | cortical": (WATER, BONE),
        "diagnóstico: sólo salto de c (1500 | 1980)": (WATER, (1000.0, 1980.0, 0.0)),
        "diagnóstico: sólo salto de rho (1000 | 1320)": (WATER, (1320.0, 1500.0, 0.0)),
    }
    out["layers"] = {}
    for name, (m1, m2) in cases.items():
        rows = [layers_1d(m1, m2, p, backend=backend) for p in ppws]
        r0 = rows[0]
        print(f"\n{name}: R = {r0['R_an']:.5f}, T = {r0['T_an']:.5f}")
        print("ppw    R          err R      T          err T     delta/dx  | envolvente: err R   err T")
        for r in rows:
            print(f"{r['ppw']:3.0f} {r['R']:9.5f} {r['eR']:+10.2e} {r['T']:9.5f} {r['eT']:+10.2e} {r['delta']:+9.4f}"
                  f"  | {r['eR_env']:+10.2e} {r['eT_env']:+10.2e}")
        dxs = [r["dx"] for r in rows]
        print("pendientes locales err R:", _fmt_slopes(slopes(dxs, [r["eR"] for r in rows])),
              "| err T:", _fmt_slopes(slopes(dxs, [r["eT"] for r in rows])))
        out["layers"][name] = rows

    print("\n--- agua | cortical a 12 ppw, CFL variable ---")
    print("CFL    err R      err T")
    rows = [layers_1d(WATER, BONE, 12, cfl=c, backend=backend) for c in (0.3, 0.6, 0.9)]
    for c, r in zip((0.3, 0.6, 0.9), rows):
        print(f"{c:4.2f} {r['eR']:+10.2e} {r['eT']:+10.2e}")
    out["layers_cfl"] = rows
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=("all", "u3", "u4", "u5", "rayleigh", "backends"), default="all")
    ap.add_argument("--backend", choices=("auto", "numpy", "numba"), default="auto")
    ap.add_argument("--quick", action="store_true", help="menos resoluciones (para probar el script)")
    args = ap.parse_args()
    t0 = time.time()
    if args.case in ("all", "u3"):
        run_u3(args.backend, args.quick)
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
