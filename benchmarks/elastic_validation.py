"""Validación cuantitativa del solver elástico frente a soluciones analíticas (sin software externo).

Casos U3, U4 y U5 del plan de ``docs/comparativa_simuladores.md`` (§11.1);
método, tablas y lectura en ``docs/validacion/elastico.md``.
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
      planas ascendentes y descendentes con dos filas de receptores.
  u5  Ondas de Lamb en una placa de cortical de 3 mm, libre (en aire) y en
      agua: velocidad de fase de S0 y A0 entre 100 y 500 kHz por transformada
      f-k frente a las raíces de Rayleigh-Lamb (con carga de fluido en agua).
  backends  El backend Numba ("auto") da lo mismo que ``backend="numpy"``.

Uso (desde la raíz del repositorio):

    NUMBA_NUM_THREADS=1 PYTHONPATH=$(pwd) python benchmarks/elastic_validation.py            # todo
    NUMBA_NUM_THREADS=1 PYTHONPATH=$(pwd) python benchmarks/elastic_validation.py --case u4  # un caso
    ... --backend numpy      # fuerza la referencia NumPy (unas 2-3 veces más lento)
    ... --quick              # resoluciones gruesas, para comprobar que el script funciona (~2 min)

Tiempos con 1 hilo de Numba (CPU compartida): u3 ~1 min, u4 ~4.5 min, u5 ~1.5 min, backends ~10 s.
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
# U4: interfaz agua-cortical, R(θ) y onda S transmitida
# ----------------------------------------------------------------------------
def run_dft_rows(solver: ElasticSolver, nt: int, rows: dict, freqs) -> dict[str, np.ndarray]:
    """Avanza ``nt`` pasos acumulando la DFT temporal (``exp(-i w t)``) de filas enteras: ``{nombre: (campo, j)}``.

    La presión se toma en los instantes de las tensiones, ``(n + 1) dt``, y las
    velocidades a ``(n + 1/2) dt`` (leapfrog).
    """
    w = 2 * np.pi * np.asarray(freqs, dtype=float)
    acc = {k: np.zeros((w.size, solver.grid.nx), complex) for k in rows}
    for n in range(nt):
        solver.step(n)
        for k, (field, j) in rows.items():
            tn = (n + 1) * solver.dt if field == "p" else (n + 0.5) * solver.dt
            acc[k] += np.exp(1j * w * tn)[:, None] * sample(solver, field, j, slice(None))[None, :]
    return acc


def _kz(k, kx):
    kz = np.sqrt(np.asarray(k, complex) ** 2 - np.asarray(kx, complex) ** 2)
    return np.where(kz.imag < 0, -kz, kz)


def _updown(X0, X1, kz, z0, z1):
    """Separa ``X(z) = A e^{i kz z} + B e^{-i kz z}`` (A desciende, B asciende) a partir de dos filas z0, z1."""
    det = np.exp(1j * kz * (z0 - z1)) - np.exp(1j * kz * (z1 - z0))
    A = (X0 * np.exp(-1j * kz * z1) - X1 * np.exp(-1j * kz * z0)) / det
    B = (X1 * np.exp(1j * kz * z0) - X0 * np.exp(1j * kz * z1)) / det
    return A, B


def beam_reflection(theta_deg: float, ppw: float = 15.0, sigma_lam: float = 4.0,
                    freqs=(0.8e6, 0.9e6, 1.0e6, 1.1e6, 1.2e6), dtheta: float = 12.0, n_theta: int = 97,
                    backend: str = "auto", solid_lam: float = 6.0, f0: float = F0) -> dict:
    """R(θ) y amplitud de la S transmitida para un haz gaussiano que incide desde el agua sobre la cortical.

    Montaje (de arriba abajo): CPML, fila de fuentes de presión con retardo
    lineal ``x sinθ / c`` y envolvente gaussiana de anchura ``sigma_lam``
    longitudes de onda (truncada a ±3.5σ), dos filas de receptores de presión
    separadas 3 celdas, la interfaz a 1.5 λ y ``solid_lam`` λ de cortical.

    En cada frecuencia de ``freqs`` y cada número de onda horizontal ``kx = w
    sinθ / c`` (DFT a lo largo de toda la fila) el campo de cada fila es la
    suma de una onda descendente y otra ascendente; con dos filas y el ``kz``
    exacto del esquema en agua (``analytic.discrete_kz``) se separan. Entre las
    filas y la interfaz no hay más fuentes ni dispersores, así que
    ``R = ascendente / descendente`` referidas al plano de la interfaz
    ``(j_int - 1/2) dx``, para cualquier haz y aunque la CPML superior
    devuelva parte del campo (a incidencia rasante, ~1-3 %: restando una
    ejecución sin hueso, ese eco reaparecía como un rizado de ±3 % en R(θ)).
    Es exacto mientras el campo se recoja entero en la fila (dominio y
    duración holgados). La S transmitida se separa de la P con un ajuste por
    mínimos cuadrados de ondas planas P y S descendentes a seis filas de
    ``vx``, ``vz`` en el sólido (polarizaciones y ``kz`` del continuo) y se da
    como velocidad de partícula relativa a la de la onda descendente en el
    agua (``|p| / (rho c)``).

    Sólo se devuelven los pares (f, θ) en los que el haz incidente tiene al
    menos el 30 % de su máximo a esa frecuencia: el haz cubre unos ±2σ_θ
    alrededor de ``theta_deg``, con ``σ_θ = 1 / (k σ cosθ)``.
    """
    rho1, c1, _ = WATER
    rho2, cp2, cs2 = BONE
    lam = c1 / f0
    dx = lam / ppw
    th0 = np.radians(theta_deg)
    sig = sigma_lam * lam
    sep = 3
    j_s = NPML + int(round(1.5 * lam / dx))
    j_r = j_s + int(round(1.0 * lam / dx))
    j_int = j_r + sep + int(round(1.5 * lam / dx))
    nz = j_int + int(round(solid_lam * lam / dx)) + NPML
    tan = np.tan(th0)
    x_c = 4.0 * sig
    width = x_c + (2.5 + 2 * 1.5) * lam * tan + 4.0 * sig + (8 if theta_deg > 45 else 2) * lam
    nx = int(np.ceil(width / dx)) + 2 * NPML
    x = (np.arange(nx) - NPML) * dx
    src = np.nonzero(np.abs(x - x_c) <= 3.5 * sig)[0]
    delay = (x[src] - x_c) * np.sin(th0) / c1
    delay -= delay.min()
    amp = np.exp(-0.5 * ((x[src] - x_c) / sig) ** 2)
    t_end = delay.max() + 6 * lam / (c1 * np.cos(th0)) + 12e-6 + (12e-6 if theta_deg > 45 else 0.0)
    solid_rows = [j_int + 2 + m for m in range(6)]
    rho, cp, cs = layered(nz, nx, [(0, WATER), (j_int, BONE)])
    s = make_solver(rho, cp, cs, dx, f0, 0.9, backend)
    dt = s.dt
    nt = int(np.ceil(t_end / dt))
    t = s.time_axis(nt)
    sigs = amp[:, None] * gaussian_pulse(t[None, :] - delay[:, None], f0)
    s.add_source(Source(np.full(src.size, j_s), src, np.arange(src.size), sigs))
    rows = {"p0": ("p", j_r), "p1": ("p", j_r + sep)}
    rows.update({f"vx{m}": ("vx", j) for m, j in enumerate(solid_rows)})
    rows.update({f"vz{m}": ("vz", j) for m, j in enumerate(solid_rows)})
    t0 = time.time()
    two = run_dft_rows(s, nt, rows, freqs)
    elapsed = time.time() - t0
    z_int = (j_int - 0.5) * dx
    z0, z1 = j_r * dx - z_int, (j_r + sep) * dx - z_int  # posiciones de las filas respecto a la interfaz
    thetas = np.radians(np.linspace(theta_deg - dtheta, theta_deg + dtheta, n_theta))
    thetas = thetas[(thetas >= 0) & (thetas < np.radians(88))]
    out = dict(f=[], theta=[], weight=[], R=[], vs=[])
    for fi, f in enumerate(freqs):
        w = 2 * np.pi * f
        kx = w * np.sin(thetas) / c1
        E = np.exp(-1j * kx[:, None] * x[None, :])
        D, U = _updown(E @ two["p0"][fi], E @ two["p1"][fi], an.discrete_kz(w, kx, c1, dx, dt), z0, z1)
        R = U / D
        # onda S transmitida: ajuste de P y S descendentes en el sólido
        kp, ks = w / cp2, w / cs2
        kzp, kzs = _kz(kp, kx), _kz(ks, kx)
        vs = np.zeros(thetas.size)
        for it in range(thetas.size):
            A, b = [], []
            ex = np.exp(-1j * kx[it] * (x + 0.5 * dx))  # vx está media celda a la derecha
            for m, j in enumerate(solid_rows):
                zx, zz = j * dx - z_int, (j + 0.5) * dx - z_int  # vz está media celda más abajo
                A.append([kx[it] / kp * np.exp(1j * kzp[it] * zx), kzs[it] / ks * np.exp(1j * kzs[it] * zx)])
                b.append(ex @ two[f"vx{m}"][fi])
                A.append([kzp[it] / kp * np.exp(1j * kzp[it] * zz), -kx[it] / ks * np.exp(1j * kzs[it] * zz)])
                b.append(E[it] @ two[f"vz{m}"][fi])
            sol = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
            vs[it] = abs(sol[1]) / (abs(D[it]) / (rho1 * c1))
        keep = np.abs(D) >= 0.3 * np.abs(D).max()
        out["f"] += [f] * int(keep.sum())
        out["theta"] += list(np.degrees(thetas[keep]))
        out["weight"] += list(np.abs(D[keep]) / np.abs(D).max())
        out["R"] += list(R[keep])
        out["vs"] += list(vs[keep])
    res = {k: np.array(v) for k, v in out.items()}
    res.update(dx=dx, nx=nx, nz=nz, elapsed=elapsed)
    return res


def r_theta_analytic(theta_deg):
    return an.fluid_solid_coefficients(np.radians(theta_deg), WATER[0], WATER[1], *BONE)


def critical_angle_from_peak(res: dict) -> np.ndarray:
    """Ángulo del máximo de |R| (el crítico P: allí |R| = 1) en cada frecuencia, con ajuste parabólico."""
    out = []
    for f in np.unique(res["f"]):
        m = res["f"] == f
        th, r = res["theta"][m], np.abs(res["R"][m])
        k = int(np.argmax(r))
        k = min(max(k, 1), th.size - 2)
        y0, y1, y2 = r[k - 1], r[k], r[k + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        out.append(th[k] + d * (th[1] - th[0]))
    return np.array(out)


U4_ANGLES = (0.0, 10.0, 20.0, 30.0, 38.0, 46.0, 52.0, 60.0, 68.0, 76.0)


def best_by_angle(beams: list[dict], f: float, step: float = 0.25) -> dict[str, np.ndarray]:
    """Combina los haces: en cada ángulo (rejilla de ``step`` grados) a la frecuencia ``f``, la muestra de más peso."""
    th = np.concatenate([b["theta"][b["f"] == f] for b in beams])
    w = np.concatenate([b["weight"][b["f"] == f] for b in beams])
    R = np.concatenate([b["R"][b["f"] == f] for b in beams])
    vs = np.concatenate([b["vs"][b["f"] == f] for b in beams])
    bins = np.round(th / step).astype(int)
    out = dict(theta=[], weight=[], R=[], vs=[])
    for b in np.unique(bins):
        m = np.flatnonzero(bins == b)
        k = m[np.argmax(w[m])]
        for key, arr in (("theta", th), ("weight", w), ("R", R), ("vs", vs)):
            out[key].append(arr[k])
    return {k: np.array(v) for k, v in out.items()}


def u4_zones(th_p, th_s):
    """Zonas angulares en las que se resume el error: lejos de los críticos (±3°), cerca de ellos y rasante."""
    return (("0° - θp-3°", 0.0, th_p - 3), ("θp ± 3°", th_p - 3, th_p + 3), ("θp+3° - θs-3°", th_p + 3, th_s - 3),
            ("θs ± 3°", th_s - 3, th_s + 3), ("θs+3° - 70°", th_s + 3, 70.0), ("70° - 80°", 70.0, 80.0))


def run_u4(backend="auto", quick=False) -> dict:
    th_p, th_s = (np.degrees(a) for a in an.critical_angles(WATER[1], BONE[1], BONE[2]))
    print("\n=== U4 interfaz agua | cortical (1000/1500 | 1908/3500/1800): R(θ) de presión ===")
    print(f"ángulos críticos teóricos: P {th_p:.2f}°, S {th_s:.2f}°; haces gaussianos (σ = 4 λ) centrados en "
          + ", ".join(f"{a:.0f}°" for a in U4_ANGLES))
    ppws = (15.0, 25.0) if not quick else (12.0,)
    out = {}
    for ppw in ppws:
        t0 = time.time()
        beams = [beam_reflection(th0, ppw, backend=backend) for th0 in U4_ANGLES]
        rc = beam_reflection(th_p, ppw, dtheta=4.0, n_theta=161, backend=backend)
        th_c = critical_angle_from_peak(rc)
        print(f"\n--- {ppw:.0f} puntos por longitud de onda en agua (dx = {1.5 / ppw:.3f} mm), "
              f"{time.time() - t0:.0f} s ---")
        print("frecuencia 0.8-1.2 MHz, para cada θ el haz con más energía en ese ángulo")
        print("zona              máx|err |R||  err medio   máx|Δfase| (°)   máx|err vS/vi|")
        zones = {}
        for name, lo, hi in u4_zones(th_p, th_s):
            e, ph, ev_ = [], [], []
            for f in np.unique(beams[0]["f"]):
                b = best_by_angle(beams, f)
                m = (b["theta"] >= lo) & (b["theta"] <= hi)
                a = r_theta_analytic(b["theta"][m])
                e += list(np.abs(b["R"][m]) / np.abs(a.R) - 1)
                ph += list(np.degrees(np.angle(b["R"][m] / a.R)))
                ok = a.vs > 0.05  # la S sólo se compara donde se propaga y lleva amplitud
                if hi <= th_s - 3 and ok.any():
                    ev_ += list(b["vs"][m][ok] / a.vs[ok] - 1)
            e, ph = np.array(e), np.array(ph)
            ev_ = np.abs(np.array(ev_)).max() if ev_ else np.nan
            zones[name] = dict(err_max=np.abs(e).max(), err_mean=e.mean(), dph_max=np.abs(ph).max(), evs_max=ev_)
            print(f"{name:16s} {np.abs(e).max():12.4f} {e.mean():+10.4f} {np.abs(ph).max():14.2f} {ev_:16.4f}")
        print("crítico P medido (máximo de |R|) a 0.8, 0.9, 1.0, 1.1 y 1.2 MHz: "
              + " ".join(f"{v:.2f}°" for v in th_c) + f"  (teórico {th_p:.2f}°)")
        b = best_by_angle(beams, 1e6)
        print("R(θ) a 1 MHz: θ, |R| medido / exacto, fase medida / exacta (°), vS/vi medida / exacta")
        for target in (0, 10, 20, 24, 27, 30, 35, 40, 45, 50, 54, 58, 60, 63.5, 65, 70, 75, 80):
            k = int(np.argmin(np.abs(b["theta"] - target)))
            a = r_theta_analytic(np.array([b["theta"][k]]))
            vs_a = a.vs[0] if b["theta"][k] < th_s else np.nan
            print(f"  {b['theta'][k]:5.2f}  {abs(b['R'][k]):7.4f} / {abs(a.R[0]):7.4f}"
                  f"  {np.degrees(np.angle(b['R'][k])):8.2f} / {np.degrees(np.angle(a.R[0])):8.2f}"
                  f"  {b['vs'][k]:7.4f} / {vs_a:7.4f}")
        out[ppw] = dict(beams=beams, critical=th_c, zones=zones)
    return out


# ----------------------------------------------------------------------------
# U5: ondas de Lamb en una placa de cortical
# ----------------------------------------------------------------------------
def lamb_plate(fluid=AIR, thickness=3e-3, dx=0.15e-3, length=200e-3, fluid_layer=None, f0=300e3,
               bandwidth=1.2, t_end=260e-6, backend="auto", decimate=4) -> dict:
    """Placa de cortical en ``fluid`` excitada en una columna de celdas; registro a lo largo de ella.

    La placa (``thickness`` × ``length``) termina en dos extremos libres a 5 mm
    de la CPML lateral, rodeada de fluido: si entra en la CPML, los modos
    guiados la vuelven inestable (crecimiento exponencial en la CPML tras
    ~3000 pasos, con aire o con agua y con o sin la corrección de interfaz).
    El eco del extremo derecho viaja hacia -x y el filtro f-k lo descarta; el
    del izquierdo es el mismo modo que la onda directa. Las fuerzas (pulso
    gaussiano de banda ancha centrado en ``f0``) actúan a 10 mm del extremo
    izquierdo. Se registran ``vx`` y ``vz`` en
    dos filas simétricas respecto al plano medio (a 2 celdas de cada cara); las
    combinaciones pares separan las familias sin ambigüedad: en los modos
    simétricos (S) ``vx`` es par en z, en los antisimétricos (A) lo es ``vz``.
    Devuelve las señales ``S`` y ``A`` (tiempo diezmado × posición) desde 10 mm
    después de la fuente hasta 10 mm antes del extremo derecho.
    """
    n_plate = int(round(thickness / dx))
    layer = fluid_layer if fluid_layer is not None else (1e-3 if fluid[0] < 10 else 6e-3)
    n_fl = int(round(layer / dx))
    j_a = NPML + n_fl
    j_b = j_a + n_plate - 1
    nz = j_b + 1 + n_fl + NPML
    gap = int(round(5e-3 / dx))
    nx = int(round(length / dx)) + 2 * (NPML + gap)
    rho, cp, cs = layered(nz, nx, [(0, fluid), (j_a, BONE), (j_b + 1, fluid)])
    for a, v in zip((rho, cp, cs), fluid):
        a[:, : NPML + gap] = v
        a[:, nx - NPML - gap:] = v
    s = make_solver(rho, cp, cs, dx, f0, 0.9, backend)
    nt = int(np.ceil(t_end / s.dt))
    t = s.time_axis(nt)
    i_s = NPML + gap + int(round(10e-3 / dx))
    # fuerza horizontal y vertical en toda la columna de la placa: la primera excita sobre todo los
    # modos simétricos (S0) y la segunda los antisimétricos (A0)
    col = np.arange(j_a, j_b + 1)
    pulse = gaussian_pulse(t, f0, bandwidth)[None, :]
    for kind in ("force_x", "force_z"):
        s.add_source(Source(col, np.full(col.size, i_s), np.zeros(col.size, int), pulse, kind))
    i0, i1 = i_s + int(round(10e-3 / dx)), nx - NPML - gap - int(round(10e-3 / dx))
    cols = slice(i0, i1)
    rows = {"vx_t": ("vx", j_a + 2), "vx_b": ("vx", j_b - 2), "vz_t": ("vz", j_a + 2), "vz_b": ("vz", j_b - 3)}
    rec = {k: np.zeros((nt // decimate + 1, i1 - i0), np.float32) for k in rows}
    t0 = time.time()
    vmax = np.zeros(2)  # máximo de |v| en todo el campo en la primera y en la segunda mitad (estabilidad)
    for n in range(nt):
        s.step(n)
        if n % decimate == 0:
            for k, (field, j) in rows.items():
                rec[k][n // decimate] = sample(s, field, j, cols)
        if n % 200 == 0:
            vmax[2 * n // nt] = max(vmax[2 * n // nt], float(np.abs(s.vz).max()), float(np.abs(s.vx).max()))
    elapsed = time.time() - t0
    finite = bool(np.isfinite(s.vx).all() and np.isfinite(s.sxx).all())
    return dict(S=0.5 * (rec["vx_t"] + rec["vx_b"]), A=0.5 * (rec["vz_t"] + rec["vz_b"]),
                dt=s.dt * decimate, dx=dx, x=(np.arange(i0, i1) - i_s) * dx, elapsed=elapsed, finite=finite,
                nz=nz, nx=nx, nt=nt, peak=float(vmax[0]), final=float(vmax[1]))


def fk_phase_velocity(u: np.ndarray, dt: float, dx: float, freqs, k_guess, search=0.2, pad=16, band=0.25):
    """Velocidad de fase por la transformada f-k, afinada con la pendiente de la fase del modo filtrado.

    1. FFT en t; en cada frecuencia, FFT en x (ventana de Hann, relleno ×``pad``)
       y máximo de |U(k)| en ``k_guess·(1 ± search)``: estimación f-k.
    2. Filtro f-k: se conserva sólo la banda ``±band`` (k > 0) alrededor de ese máximo
       (separa el modo de los demás de su familia, p. ej. A1 de A0) y se vuelve
       a x; la pendiente de la fase desenrollada (ajuste ponderado por la
       amplitud, sin el 10 % de cada extremo) da ``k`` sin el sesgo de la
       ventana ni la limitación de resolución 2π/L del máximo.

    Devuelve ``c = 2π f / k`` de los dos pasos y la prominencia del máximo
    (frente al mayor valor del espectro en esa frecuencia).
    """
    nt, nxr = u.shape
    nfft = 1 << int(np.ceil(np.log2(4 * nt)))
    spec_t = np.fft.rfft(u, nfft, axis=0)
    ft = np.fft.rfftfreq(nfft, dt)
    win = np.hanning(nxr)
    nk = pad * (1 << int(np.ceil(np.log2(nxr))))
    k_axis = 2 * np.pi * np.fft.rfftfreq(nk, dx)
    n2 = 4 * nxr
    k_full = 2 * np.pi * np.fft.fftfreq(n2, dx)
    x = np.arange(nxr) * dx
    inner = slice(nxr // 10, nxr - nxr // 10)
    c_fk, c_ph, prom = [], [], []
    for f, kg in zip(freqs, k_guess):
        # la onda viaja hacia +x: con e^{-i w t} su fase espacial es +k x (numpy da la conjugada en t)
        a = np.conj(spec_t[int(np.argmin(np.abs(ft - f)))])
        sk = np.abs(np.fft.fft(a * win, nk))[: k_axis.size]
        m = np.flatnonzero((k_axis > kg * (1 - search)) & (k_axis < kg * (1 + search)))
        k = m[np.argmax(sk[m])]
        y0, y1, y2 = np.log(sk[k - 1: k + 2])
        kp = k_axis[k] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (k_axis[1] - k_axis[0])
        prom.append(sk[k] / sk.max())
        c_fk.append(2 * np.pi * f / kp)
        # banda de al menos ±6 celdas de resolución (2π/L): más estrecha, el filtro emborrona la fase en x
        half = max(band * kp, 6 * 2 * np.pi / (nxr * dx))
        filt = np.fft.ifft(np.fft.fft(a, n2) * (np.abs(k_full - kp) < half) * (k_full > 0))[:nxr]
        ph = np.unwrap(np.angle(filt[inner]))
        k_fit = np.polyfit(x[inner], ph, 1, w=np.abs(filt[inner]))[0]
        c_ph.append(2 * np.pi * f / k_fit)
    return np.array(c_fk), np.array(c_ph), np.array(prom)


U5_FREQS = np.arange(100e3, 501e3, 25e3)


def lamb_case(fluid, thickness=3e-3, dx=0.15e-3, freqs=U5_FREQS, backend="auto", **kw) -> dict:
    """Simulación de la placa y comparación de S0 y A0 con Rayleigh-Lamb (libre o con carga de fluido)."""
    sim = lamb_plate(fluid, thickness, dx, backend=backend, **kw)
    rho_s, cp_s, cs_s = BONE
    loaded = fluid[0] > 10
    out = dict(sim=sim, f=np.asarray(freqs))
    fr = np.concatenate([np.linspace(20e3, freqs[0], 8)[:-1], freqs])  # continuación desde baja frecuencia
    for fam in ("S", "A"):
        k_free = an.lamb_fundamental(fr, thickness, cp_s, cs_s, fam)[7:]
        k_ref = an.lamb_fundamental(fr, thickness, cp_s, cs_s, fam, rho_s, fluid[0], fluid[1])[7:] if loaded \
            else k_free
        c_fk, c_ph, prom = fk_phase_velocity(sim[fam], sim["dt"], sim["dx"], freqs, k_ref.real)
        w = 2 * np.pi * np.asarray(freqs)
        out[fam] = dict(c=c_ph, c_fk=c_fk, prom=prom, c_ref=w / k_ref.real, c_free=w / k_free.real,
                        att=np.abs(k_ref.imag))
    return out


#: Prominencia mínima del máximo f-k para dar el modo por resuelto (ver ``fk_phase_velocity``)
U5_MIN_PROMINENCE = 0.3


def run_u5(backend="auto", quick=False) -> dict:
    print("\n=== U5 ondas de Lamb en una placa de cortical de 3 mm (1908/3500/1800), 100-500 kHz ===")
    print("velocidad de fase por f-k + pendiente de fase (registro de 180 mm) frente a las raíces de Rayleigh-Lamb;")
    print(f"'-' = modo no resuelto (máximo f-k con prominencia < {U5_MIN_PROMINENCE}: poco excitado o con mucha fuga)")
    configs = [("placa libre (en aire)", AIR, 0.15e-3), ("placa libre (en aire)", AIR, 0.10e-3),
               ("placa en agua", WATER, 0.15e-3)]
    if quick:
        configs = [("placa libre (en aire)", AIR, 0.25e-3), ("placa en agua", WATER, 0.25e-3)]
    out = {}
    for name, fluid, dx in configs:
        r = lamb_case(fluid, dx=dx, backend=backend)
        sim = r["sim"]
        print(f"\n--- {name}, dx = {dx * 1e3:.2f} mm ({int(round(3e-3 / dx))} celdas en el espesor): malla "
              f"{sim['nz']}x{sim['nx']}, {sim['nt']} pasos, {sim['elapsed']:.0f} s; campo finito: {sim['finite']}, "
              f"máx|v| 2.ª mitad / 1.ª = {sim['final'] / sim['peak']:.2f} ---")
        ref = "Rayleigh-Lamb libre" if fluid is AIR else "con carga de agua (parte real de la raíz compleja)"
        extra = "" if fluid is AIR else "   S0 libre  A0 libre  Im k S0 (1/m)"
        print(f"f (kHz)   S0 med   S0 ref    err   |  A0 med   A0 ref    err    [ref: {ref}]{extra}")
        errs = {"S": [], "A": []}
        for i, f in enumerate(r["f"]):
            cells = []
            for fam in ("S", "A"):
                d = r[fam]
                if d["prom"][i] >= U5_MIN_PROMINENCE:
                    e = d["c"][i] / d["c_ref"][i] - 1
                    errs[fam].append(e)
                    cells.append(f"{d['c'][i]:8.1f} {d['c_ref'][i]:8.1f} {e:+7.4f}")
                else:
                    cells.append(f"{'-':>8s} {d['c_ref'][i]:8.1f} {'-':>7s}")
            line = f"{f / 1e3:6.0f}  {cells[0]}  | {cells[1]}"
            if fluid is not AIR:
                line += f"   {r['S']['c_free'][i]:8.1f}  {r['A']['c_free'][i]:8.1f}  {r['S']['att'][i]:9.1f}"
            print(line)
        summary = {fam: (np.abs(errs[fam]).max() if errs[fam] else np.nan, len(errs[fam])) for fam in errs}
        print("máx|err| donde está resuelto: " + ", ".join(
            f"{fam}0 {v[0] * 100:.2f} % ({v[1]}/{r['f'].size} frecuencias)" for fam, v in summary.items()))
        out[(name, dx)] = dict(result=r, summary=summary)
    return out


def run_backends() -> dict:
    """El backend Numba ("auto") debe dar lo mismo que la referencia NumPy en los montajes de validación."""
    print("\n=== backends: Numba frente a NumPy ===")
    out = {}
    a, b = (layers_1d(WATER, BONE, 12, backend=k) for k in ("numba", "numpy"))
    out["u3"] = (a["R"], b["R"], a["T"], b["T"])
    print(f"U3 agua|cortical 12 ppw: R numba {a['R']:.7f} numpy {b['R']:.7f}; T numba {a['T']:.7f} numpy {b['T']:.7f}")
    ra, rb = (beam_reflection(46.0, 12, backend=k, freqs=(1e6,)) for k in ("numba", "numpy"))
    d = np.abs(ra["R"] - rb["R"]).max() / np.abs(rb["R"]).max()
    out["u4"] = d
    print(f"U4 haz a 46°, 12 ppw: máx|R numba - R numpy| / máx|R| = {d:.1e} ({ra['elapsed']:.1f} s frente a "
          f"{rb['elapsed']:.1f} s)")
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=("all", "u3", "u4", "u5", "backends"), default="all")
    ap.add_argument("--backend", choices=("auto", "numpy", "numba"), default="auto")
    ap.add_argument("--quick", action="store_true", help="menos resoluciones (para probar el script)")
    args = ap.parse_args()
    t0 = time.time()
    if args.case in ("all", "u3"):
        run_u3(args.backend, args.quick)
    if args.case in ("all", "u4"):
        run_u4(args.backend, args.quick)
    if args.case in ("all", "u5"):
        run_u5(args.backend, args.quick)
    if args.case in ("all", "backends"):
        run_backends()
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
