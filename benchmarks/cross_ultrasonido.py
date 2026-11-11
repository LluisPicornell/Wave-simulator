"""Comparación cruzada del solver elástico 2D de wavesim con k-Wave y con Devito, ejecutados aquí en CPU.

Las herramientas de referencia NO son dependencias de wavesim (no están en ``pyproject.toml``):
se instalan en entornos aparte que ven el NumPy/Numba del sistema (para wavesim):

    python -m venv --system-site-packages venv_kwave && venv_kwave/bin/pip install k-wave-python scipy
    python -m venv --system-site-packages venv_devito && venv_devito/bin/pip install devito
    # probado con k-wave-python 0.6.2 (NumPy 2.2.6, backend "python") y Devito 4.8.23 (gcc, sin OpenMP)

Uso (desde la raíz del repositorio; ``--save`` reescribe ``benchmarks/data/cross_ultrasonido/``):

    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$(pwd) venv_kwave/bin/python \\
        benchmarks/cross_ultrasonido.py --tool kwave [--case A,B,C] [--save]       # ~18 min
    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$(pwd) venv_devito/bin/python \\
        benchmarks/cross_ultrasonido.py --tool devito [--case D1,D2] [--save]      # ~23 min

Sin las herramientas, ``tests/test_cross_ultrasonido.py`` compara wavesim con los datos guardados.

Casos (método y resultados en ``docs/validacion/cruzada_ultrasonido.md``):

A) k-Wave, dos capas fluidas: R, T, tiempos y forma de onda (``kwave_comparison.case_layers``).
B) k-Wave, fuente puntual con atenuación de ley de potencia: α(f) y c(f) de cada código frente a su
   propio modelo continuo y entre sí (``kwave_comparison.case_point``).
C) k-Wave, medio tipo tejido (agua, piel, grasa con interfaz ondulada, músculo con interfaz curva),
   fluido: trazas en dos filas de 32 receptores (``kwave_comparison.case_tissue``).
D) Devito (k-wave-python 0.6.2 no trae solver elástico: ``kWaveSimulation`` rechaza ``pstdElastic*``),
   velocidad-tensión en malla escalonada con orden espacial 4 (el estencil de wavesim) y 8:
   D1 haces gaussianos sobre la interfaz agua-cortical a 0°, 30° y 45° (``elastic_validation.beam_reflection``),
   D2 placa de cortical de 3 mm en agua, velocidades de fase S0 y A0 (``elastic_validation.lamb_case``).
   Las dos implementaciones reciben los mismos mapas, el mismo dt y las mismas fuentes; Devito usa una
   esponja absorbente en lugar de la CPML y no tiene la corrección de deslizamiento de la interfaz.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wavesim.ultrasound.elastic_fdtd import C1, C2  # noqa: E402

DATA = ROOT / "benchmarks" / "data" / "cross_ultrasonido"


def _load(name: str):
    """Importa otro script de ``benchmarks/`` (no es un paquete)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("elastic_validation")


# ----------------------------------------------------------------------------
# Devito: solver velocidad-tensión con la interfaz de ``ElasticSolver``
# ----------------------------------------------------------------------------
class DevitoElastic:
    """Esquema de Virieux en malla escalonada generado por Devito, con los atributos que usan los benchmarks.

    Mismas posiciones que wavesim (``sxx``, ``szz`` en (i, j); ``vx`` en (i+1/2, j); ``vz`` en (i, j+1/2);
    ``sxz`` en (i+1/2, j+1/2)), los mismos coeficientes del medio en cada nodo (densidad media aritmética en
    los nodos de velocidad, ``mu`` media armónica de 4 celdas en ``sxz``, nula si alguna es fluida) y el
    mismo dt. Lo que cambia: las derivadas las genera Devito con orden ``space_order`` (4 = el estencil de
    wavesim, 8 = uno más ancho), no hay corrección de deslizamiento en la interfaz fluido-sólido y, en lugar
    de la CPML, una esponja (atenuación exponencial de todos los campos) en una franja del borde de
    ``sponge_lambdas`` longitudes de onda (de la onda más rápida del borde, a ``f0``; al menos ``npml + 40``
    celdas): la malla de Devito se amplía por cada lado lo que haga falta replicando el borde y los campos que
    se leen son los de la región original. Con una franja fija de 60 celdas, a 20-30 ppw (2-3 λ) el eco de
    la esponja cambiaba |R| a 45° hasta un 14 % y la velocidad de S0 de la placa hasta un 1 %. Las fuentes
    se suman desde NumPy entre pasos: las de presión en el mismo instante que wavesim; las de fuerza, tras la
    actualización de las tensiones (medio paso más tarde que en wavesim, indiferente para medir velocidades
    de fase).
    """

    def __init__(self, rho, cp, cs, dx, f0, cfl=0.9, space_order=8, npml=ev.NPML, sponge_lambdas=3.0,
                 r_sponge=1e-4):
        from devito import (NODE, Eq, Function, Grid, Operator, TensorTimeFunction, VectorTimeFunction,
                            configuration, div, grad)
        from wavesim.ultrasound.elastic_fdtd import _harmonic_xz

        configuration["log-level"] = "WARNING"
        border = np.ones(np.shape(cp), bool)
        border[npml:-npml, npml:-npml] = False
        c_edge = float(np.asarray(cp)[border].max())
        width = max(npml + 40, int(np.ceil(sponge_lambdas * c_edge / (f0 * dx))))
        e = width - npml
        rho, cp, cs = (np.pad(np.asarray(a, float), e, mode="edge") for a in (rho, cp, cs))
        nz, nx = rho.shape
        self.e, self.dt, self.f0 = e, cfl * dx / (float(cp.max()) * np.sqrt(2.0) * (C1 + abs(C2))), f0
        self.grid = type("G", (), {"nx": nx - 2 * e, "nz": nz - 2 * e, "dx": dx})()
        # Devito: dimensiones (x, y) con y = profundidad; los arrays de wavesim son (z, x)
        grid = Grid(shape=(nx, nz), extent=((nx - 1) * dx, (nz - 1) * dx), dtype=np.float32)
        x, y = grid.dimensions
        so = space_order
        self._v = VectorTimeFunction(name="v", grid=grid, space_order=so, time_order=1)
        self._t = TensorTimeFunction(name="t", grid=grid, space_order=so, time_order=1)

        def fun(name, values, staggered):
            f = Function(name=name, grid=grid, space_order=so, staggered=staggered)
            f.data[:] = np.asarray(values, np.float32).T
            return f

        mu = rho * cs**2
        lam = rho * cp**2 - 2 * mu
        bx = fun("bx", 2.0 / (rho + np.concatenate([rho[:, 1:], rho[:, -1:]], 1)), x)
        bz = fun("bz", 2.0 / (rho + np.concatenate([rho[1:], rho[-1:]], 0)), y)
        f_lam, f_mu = fun("lam", lam, NODE), fun("mu", mu, NODE)
        f_muxz = fun("muxz", _harmonic_xz(mu), (x, y))
        # esponja: d(r) = d0 (r/W)², d0 de la fórmula de la PML para un reflejo teórico r_sponge
        dist = np.maximum(np.maximum(width - np.arange(nz), np.arange(nz) - (nz - 1 - width)), 0)[:, None]
        dist = np.maximum(dist, np.maximum(width - np.arange(nx), np.arange(nx) - (nx - 1 - width))[None, :])
        d0 = 3.0 * float(cp.max()) * np.log(1.0 / r_sponge) / (2.0 * width * dx)
        damp = fun("damp", np.exp(-d0 * (dist / width) ** 2 * self.dt), NODE)
        s = grid.stepping_dim.spacing
        v, t = self._v, self._t
        dvt = div(t)
        strain = grad(v.forward) + grad(v.forward).transpose(inner=False)
        dvf = div(v.forward)
        eqs = [Eq(v[0].forward, damp * (v[0] + s * bx * dvt[0])),
               Eq(v[1].forward, damp * (v[1] + s * bz * dvt[1])),
               Eq(t[0, 0].forward, damp * (t[0, 0] + s * (f_lam * dvf + f_mu * strain[0, 0]))),
               Eq(t[1, 1].forward, damp * (t[1, 1] + s * (f_lam * dvf + f_mu * strain[1, 1]))),
               Eq(t[0, 1].forward, damp * (t[0, 1] + s * f_muxz * strain[0, 1]))]
        self._op = Operator(eqs)
        self._cp = cp[e:-e, e:-e]
        self._buf = 0
        self.sources = []

    def _field(self, f):
        # vista ndarray simple (las de ``devito.Data`` no admiten este corte traspuesto)
        return f.data[self._buf].view(np.ndarray)[self.e:-self.e, self.e:-self.e].T

    vx = property(lambda self: self._field(self._v[0]))
    vz = property(lambda self: self._field(self._v[1]))
    sxx = property(lambda self: self._field(self._t[0, 0]))
    szz = property(lambda self: self._field(self._t[1, 1]))
    sxz = property(lambda self: self._field(self._t[0, 1]))

    def time_axis(self, nt):
        return (np.arange(nt) + 1) * self.dt

    def add_source(self, src):
        self.sources.append(src)

    def step(self, n):
        self._op.apply(time_m=n, time_M=n, dt=np.float32(self.dt))
        self._buf = (n + 1) % 2
        k = 2.0 * self.dt / self.grid.dx
        for src in self.sources:
            inc = src.signals[src.channel, n] * k * self._cp[src.j, src.i]
            if src.kind == "pressure":
                for f in (self.sxx, self.szz):
                    np.subtract.at(f, (src.j, src.i), inc)
            else:
                np.add.at(self.vx if src.kind == "force_x" else self.vz, (src.j, src.i), inc)


def devito_factory(space_order):
    def make(rho, cp, cs, dx, f0=ev.F0, cfl=0.9, backend=None):
        return DevitoElastic(rho, cp, cs, dx, f0, cfl, space_order)
    return make


def no_slip_factory(rho, cp, cs, dx, f0=ev.F0, cfl=0.9, backend="auto"):
    """wavesim sin la corrección de deslizamiento de la interfaz (diagnóstico: lo que Devito no tiene)."""
    s = ev.make_solver(rho, cp, cs, dx, f0, cfl, backend)
    s._slip = {}
    return s


def codes(with_devito=True):
    """Implementaciones que se comparan en D: nombre -> ``factory`` de ``elastic_validation``."""
    out = {"wavesim": None, "wavesim sin corrección": no_slip_factory}
    if with_devito:
        out.update({"Devito orden 4": devito_factory(4), "Devito orden 8": devito_factory(8)})
    return out


# ----------------------------------------------------------------------------
# D1: interfaz agua-cortical
# ----------------------------------------------------------------------------
D1_ANGLES = (0.0, 30.0, 45.0)
D1_FREQS = (0.8e6, 0.9e6, 1.0e6, 1.1e6, 1.2e6)
#: 12 ppw es la resolución del test rápido; 20 y 30, las del informe
D1_PPWS = (12.0, 20.0, 30.0)


def beam_at(theta, ppw, factory=None, freqs=D1_FREQS):
    """R (complejo) y amplitud de la S transmitida en el ángulo central del haz, en cada frecuencia."""
    r = ev.beam_reflection(theta, ppw, freqs=freqs, factory=factory)
    R, vs = [], []
    for f in freqs:
        m = np.flatnonzero(r["f"] == f)
        k = m[np.argmin(np.abs(r["theta"][m] - theta))]
        R.append(r["R"][k])
        vs.append(r["vs"][k])
    return np.array(R), np.array(vs), r["elapsed"]


def run_d1(ppws=D1_PPWS, with_devito=True):
    out = {}
    for ppw in ppws:
        res = {}
        for name, fac in codes(with_devito).items():
            R, vs, el = zip(*(beam_at(th, ppw, fac) for th in D1_ANGLES))
            res[name] = dict(R=np.array(R), vs=np.array(vs), elapsed=float(sum(el)))
            print(f"  D1 {ppw:.0f} ppw, {name}: {sum(el):.0f} s", flush=True)
        out[ppw] = res
    return out


def d1_analytic():
    a = ev.r_theta_analytic(np.array(D1_ANGLES))
    return a.R, a.vs


def print_d1(out):
    R_an, vs_an = d1_analytic()
    i1 = D1_FREQS.index(1e6)
    print("\n=== D1 interfaz agua | cortical: |R| y vS/vi a 1 MHz (máx|err| en 0.8-1.2 MHz entre corchetes) ===")
    for ppw, res in out.items():
        print(f"--- {ppw:.0f} puntos por longitud de onda en agua ---")
        print(f"{'':24s}" + "".join(f"{th:>14.0f}°" for th in D1_ANGLES) + "  |  vS/vi a "
              + ", ".join(f"{th:.0f}°" for th in D1_ANGLES[1:]))
        print(f"{'analítico':24s}" + "".join(f"{abs(r):15.4f}" for r in R_an) + "  |  "
              + "  ".join(f"{v:.4f}" for v in vs_an[1:]))
        for name, v in res.items():
            eR = np.abs(v["R"]) / np.abs(R_an)[:, None] - 1
            evs = v["vs"][1:] / vs_an[1:, None] - 1
            print(f"{name:24s}" + "".join(f"{abs(v['R'][a, i1]):7.4f}[{np.abs(eR[a]).max()*100:4.2f}%]"
                                          for a in range(len(D1_ANGLES)))
                  + "  |  " + "  ".join(f"{v['vs'][a + 1, i1]:.4f}[{np.abs(evs[a]).max()*100:4.2f}%]"
                                        for a in range(len(D1_ANGLES) - 1))
                  + f"   ({v['elapsed']:.0f} s)")
        ref = res["wavesim"]["R"]
        for name, v in res.items():
            if name == "wavesim":
                continue
            d = np.abs(v["R"]) / np.abs(ref) - 1
            dph = np.degrees(np.angle(v["R"] / ref))
            print(f"  {name} / wavesim - 1 en |R| (1 MHz): " + ", ".join(
                f"{th:.0f}° {d[a, i1]*100:+.2f} %" for a, th in enumerate(D1_ANGLES))
                + "; Δfase " + ", ".join(f"{dph[a, i1]:+.2f}°" for a in range(len(D1_ANGLES))))


# ----------------------------------------------------------------------------
# D2: placa de cortical en agua (Lamb)
# ----------------------------------------------------------------------------
#: 0.25 mm es la resolución del test rápido (12 celdas en el espesor); 0.15 y 0.10, las del informe
D2_DXS = (0.25e-3, 0.15e-3, 0.10e-3)


def run_d2(dxs=D2_DXS, with_devito=True):
    out = {}
    for dx in dxs:
        res = {}
        for name, fac in codes(with_devito).items():
            r = ev.lamb_case(ev.WATER, dx=dx, factory=fac)
            res[name] = {fam: dict(c=r[fam]["c"], prom=r[fam]["prom"]) for fam in "SA"}
            res[name]["elapsed"] = r["sim"]["elapsed"]
            res[name]["growth"] = r["sim"]["final"] / r["sim"]["peak"]
            print(f"  D2 dx = {dx*1e3:.2f} mm, {name}: {r['sim']['elapsed']:.0f} s", flush=True)
        res["ref"] = {fam: r[fam]["c_ref"] for fam in "SA"}
        out[dx] = res
    return out


#: Por encima, S0 en agua pierde tanta energía (Im k = 77 1/m a 375 kHz, 107 a 400) que el máximo f-k supera el$
#: umbral de prominencia sin ser el modo: a 0.10 mm, 375-400 kHz daban de −3 a −8 % en todos los códigos
D2_S0_MAX = 350e3


def d2_errors(res, name, fam):
    """Error relativo frente a Rayleigh-Lamb con carga de agua donde el modo está resuelto (NaN si no)."""
    d = res[name][fam]
    e = d["c"] / res["ref"][fam] - 1
    ok = d["prom"] >= ev.U5_MIN_PROMINENCE
    if fam == "S":
        ok &= ev.U5_FREQS <= D2_S0_MAX
    return np.where(ok, e, np.nan)


def print_d2(out):
    f = ev.U5_FREQS
    print("\n=== D2 placa de cortical de 3 mm en agua: velocidad de fase de S0 y A0 frente a Rayleigh-Lamb ===")
    for dx, res in out.items():
        print(f"--- dx = {dx*1e3:.2f} mm ({int(round(3e-3 / dx))} celdas en el espesor) ---")
        names = [k for k in res if k != "ref"]
        for fam in "SA":
            print(f"{fam}0  f (kHz)".ljust(28) + "".join(f"{x/1e3:7.0f}" for x in f))
            print("    analítico (m/s)".ljust(28) + "".join(f"{x:7.0f}" for x in res["ref"][fam]))
            for name in names:
                e = d2_errors(res, name, fam) * 100
                print(f"    {name} (%)".ljust(28) + "".join("      -" if np.isnan(x) else f"{x:+7.2f}" for x in e)
                      + f"  máx {np.nanmax(np.abs(e)):.2f} %")
        for name in names:
            if name == "wavesim":
                continue
            parts = []
            for fam in "SA":
                d = (res[name][fam]["c"] / res["wavesim"][fam]["c"] - 1) * 100
                ok = ~np.isnan(d2_errors(res, "wavesim", fam) + d2_errors(res, name, fam))
                parts.append(f"{fam}0 media {d[ok].mean():+.2f} % (máx|·| {np.abs(d[ok]).max():.2f} %)")
            print(f"  {name} / wavesim - 1: " + ", ".join(parts) + f"; {res[name]['elapsed']:.0f} s, "
                  f"máx|v| 2.ª/1.ª mitad {res[name]['growth']:.2f}")


# ----------------------------------------------------------------------------
# datos de referencia para el test rápido
# ----------------------------------------------------------------------------
def save_kwave(case, r):
    """Trazas de k-Wave (float32) y metadatos; las de wavesim no se guardan, el test las recalcula."""
    import kwave

    DATA.mkdir(parents=True, exist_ok=True)
    if case == "B":
        extra = {}
        for name, key in (("k-Wave nominal", "nominal"), ("k-Wave ajustado", "matched")):
            t, p = r["traces"][name]
            extra[f"t_{key}"], extra[f"p_{key}"] = t, p.astype(np.float32)
        extra["kw_matched"] = np.array(r["kw_matched"])
    else:
        t, p = r["traces"]["k-Wave"]
        extra = dict(t=t, p=p.astype(np.float32))
    np.savez_compressed(DATA / f"kwave_{case}.npz", version=kwave.__version__, dx=r["dx"], **extra)


def save_devito(d1, d2):
    import devito

    DATA.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for ppw, res in d1.items():
        for name, v in res.items():
            if name.startswith("Devito"):
                tag = f"o{name[-1]}_ppw{ppw:.0f}"
                arrays[f"R_{tag}"], arrays[f"vs_{tag}"] = v["R"], v["vs"]
    for dx, res in d2.items():
        for name, v in res.items():
            if name.startswith("Devito"):
                tag = f"o{name[-1]}_dx{dx*1e6:.0f}"
                for fam in "SA":
                    arrays[f"c{fam}_{tag}"], arrays[f"prom{fam}_{tag}"] = v[fam]["c"], v[fam]["prom"]
    np.savez_compressed(DATA / "devito_D.npz", version=devito.__version__, angles=np.array(D1_ANGLES),
                        freqs=np.array(D1_FREQS), lamb_freqs=ev.U5_FREQS, **arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", choices=("kwave", "devito"), required=True)
    ap.add_argument("--case", default=None, help="lista separada por comas (A,B,C o D1,D2); por defecto, todos")
    ap.add_argument("--save", action="store_true", help="guarda las trazas/resultados de referencia")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    t0 = time.time()
    if args.tool == "kwave":
        try:
            import kwave  # noqa: F401
        except ImportError:
            sys.exit("Falta k-wave-python (ver la cabecera para instalarlo en un entorno aparte)")
        kw = _load("kwave_comparison")
        cases = (args.case or "A,B,C").split(",")
        if "A" in cases:
            for dx in (0.20e-3, 0.15e-3, 0.10e-3, 0.05e-3):
                r = kw.case_layers(dx, "python")
                kw.print_layers(r)
                if args.save and abs(dx - 0.10e-3) < 1e-9:
                    save_kwave("A", r)
        if "B" in cases:
            r = kw.case_point(0.1e-3, "python")
            kw.print_point(r)
            if args.save:
                save_kwave("B", r | {"dx": 0.1e-3})
        if "C" in cases:
            for dx in (0.10e-3, 0.05e-3):
                r = kw.case_tissue(dx, "python")
                kw.print_tissue(r)
                if args.save and abs(dx - 0.10e-3) < 1e-9:
                    save_kwave("C", r)
    else:
        try:
            import devito  # noqa: F401
        except ImportError:
            sys.exit("Falta Devito (ver la cabecera para instalarlo en un entorno aparte)")
        cases = (args.case or "D1,D2").split(",")
        d1 = run_d1() if "D1" in cases else {}
        if d1:
            print_d1(d1)
        d2 = run_d2() if "D2" in cases else {}
        if d2:
            print_d2(d2)
        if args.save and d1 and d2:
            save_devito(d1, d2)
    print(f"\ntiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
