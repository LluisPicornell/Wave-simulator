"""Rendimiento de los solvers electromagnéticos: tiempos antes/después y comprobación de que no cambia el resultado.

Uso (desde la raíz del repositorio; Numba opcional, sin él sólo se mide NumPy):

    NUMBA_NUM_THREADS=1 python benchmarks/rendimiento_em.py --case fdtd    # Maxwell FDTD: NumPy frente a Numba (~3 min)
    NUMBA_NUM_THREADS=1 python benchmarks/rendimiento_em.py --case fdtd --quick   # sin los casos más grandes (~1 min)

Casos de ``fdtd`` (los de ``tests/test_em.py`` y ``benchmarks/em_validation.py``):

- ``plana``: onda plana en músculo a 2.45 GHz (el test de ``test_em.py``).
- ``losa``: losa piel/grasa/músculo/cortical, banda estrecha 2.45 GHz a dx = 0.25 mm y banda ancha 0.1-10 GHz a dx = 0.25 mm.
- ``mie``: cilindro de músculo (a = 20 mm) a 2.45 GHz con 15 puntos por longitud de onda y el de banda ancha 0.5-3 GHz.

Cada caso se ejecuta con el backend NumPy y con el de Numba (ya compilado: se calienta antes) y se imprime la
discrepancia relativa máxima entre los campos, que debe estar al nivel del redondeo de float64.
Los tiempos con 1 hilo se obtienen con ``NUMBA_NUM_THREADS=1``.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import em_validation as ev  # noqa: E402
from wavesim.core.grid import Grid2D  # noqa: E402
from wavesim.em import MaxwellTMSolver, recommended_dx  # noqa: E402
from wavesim.em.fdtd_dispersive import BroadbandMaxwellTMSolver  # noqa: E402
from wavesim.materials import dielectric  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402


def _con_backend(backend: str):
    """Hace que ``em_validation`` construya sus solvers con el backend pedido."""
    ev.MaxwellTMSolver = functools.partial(MaxwellTMSolver, backend=backend)
    ev.BroadbandMaxwellTMSolver = functools.partial(BroadbandMaxwellTMSolver, backend=backend)


def _plana(backend: str):
    f0, tissue = 2.45e9, "muscle"
    dx = recommended_dx(f0, [tissue], ppw=20)
    beta, _ = dielectric.wave_numbers(tissue, f0)
    g = Grid2D.from_size(20 * dx, 5 * 2 * np.pi / beta + 60 * dx, dx, x0=-10 * dx)
    s = MaxwellTMSolver(Phantom(g, np.full(g.shape, TISSUE_INDEX[tissue], np.int16)), f0, backend=backend)
    s.set_plane_wave_source(z=g.z0 + 25 * dx)
    return s.run().phasor, g.shape


def _losa_estrecha(backend: str):
    _con_backend(backend)
    return np.array([ev.slab_narrowband(0.25e-3, 2.45e9)[0]]), "dx 0.25 mm"


def _losa_ancha(backend: str):
    _con_backend(backend)
    return ev.slab_broadband(0.25e-3)[1], "dx 0.25 mm"


def _mie_estrecha(backend: str):
    _con_backend(backend)
    return np.array(ev.cylinder_narrowband("muscle", 2.45e9, 0.02, 15, False)[:2]), "a 20 mm, 15 ppw"


def _mie_ancha(backend: str):
    _con_backend(backend)
    rows, _, shape, _ = ev.cylinder_broadband()
    return np.array(rows), shape


CASOS = [("plana 2.45 GHz (músculo)", _plana, False), ("losa banda estrecha 2.45 GHz", _losa_estrecha, False),
         ("losa banda ancha 0.1-10 GHz", _losa_ancha, True), ("Mie banda estrecha 2.45 GHz", _mie_estrecha, True),
         ("Mie banda ancha 0.5-3 GHz", _mie_ancha, True)]


def bench_fdtd(quick: bool) -> None:
    warnings.simplefilter("ignore")
    print("| caso | NumPy (s) | Numba (s) | aceleración | discrepancia relativa |")
    print("|---|---|---|---|---|")
    for nombre, fn, grande in CASOS:
        if quick and grande:
            continue
        fn("numba")  # calienta (compila o carga la caché de Numba) con el mismo caso
        t0 = time.time()
        a, _ = fn("numpy")
        t_np = time.time() - t0
        t0 = time.time()
        b, _ = fn("numba")
        t_nb = time.time() - t0
        err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)
        print(f"| {nombre} | {t_np:.1f} | {t_nb:.1f} | {t_np / t_nb:.1f}x | {err:.1e} |", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=["fdtd"], default="fdtd")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.case == "fdtd":
        bench_fdtd(args.quick)


if __name__ == "__main__":
    main()
