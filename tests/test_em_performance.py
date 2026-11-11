"""Rendimiento de los solvers electromagnéticos: los backends rápidos dan los mismos resultados que los de referencia.

Las tablas de tiempos (antes/después) están en ``docs/validacion/rendimiento_em.md`` y salen de
``benchmarks/rendimiento_em.py``; aquí sólo se comprueba la equivalencia en casos pequeños y rápidos.
"""

import warnings

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.em.fdtd_dispersive import BroadbandMaxwellTMSolver
from wavesim.em.fdtd_tm import MaxwellTMSolver
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

pytest.importorskip("numba")


def _cilindro(n=36, dx=1.5e-3, radio=7e-3):
    """Cilindro de músculo con una inclusión de grasa en aire: medios muy distintos y CPML en los cuatro bordes."""
    g = Grid2D(n, n, dx, x0=-n * dx / 2, z0=-n * dx / 2)
    xx, zz = g.meshgrid()
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[np.hypot(xx, zz) <= radio] = TISSUE_INDEX["muscle"]
    lab[np.hypot(xx - 2e-3, zz + 2e-3) <= 2.5e-3] = TISSUE_INDEX["fat"]
    return Phantom(g, lab)


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(a).max()


def test_maxwell_banda_estrecha_numba_igual_que_numpy():
    p = _cilindro()
    out = {}
    for be in ("numpy", "numba"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pocos puntos por longitud de onda: da igual para comparar backends
            s = MaxwellTMSolver(p, 2.45e9, npml=8, backend=be)
        s.set_point_source(-8e-3, -12e-3)
        out[be] = s.run(n_periods_steady=1.0, settle_time=3e-9)
    assert out["numba"].steps == out["numpy"].steps
    assert np.abs(out["numpy"].phasor).max() > 0
    assert _rel(out["numpy"].phasor, out["numba"].phasor) < 1e-12
    assert _rel(out["numpy"].sar, out["numba"].sar) < 1e-12


def test_maxwell_banda_ancha_numba_igual_que_numpy():
    """Pulso de banda ancha con polos de Debye (ADE), fuente puntual y monitor parcial."""
    p = _cilindro()
    mon = np.zeros(p.grid.shape, bool)
    mon[10:26, 10:26] = True
    freqs = np.array([1.0e9, 2.0e9, 3.0e9])
    out = {}
    for be in ("numpy", "numba"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = BroadbandMaxwellTMSolver(p, (1e9, 3e9), npml=8, backend=be)
        assert s.j_pol.shape[0] > 0
        s.set_point_source(-8e-3, -12e-3)
        out[be] = s.run(freqs, monitor=mon, t_max=2.5e-9)
    assert out["numba"].steps == out["numpy"].steps
    assert _rel(out["numpy"].phasor, out["numba"].phasor) < 1e-11


def test_maxwell_onda_plana_numba_igual_que_numpy():
    """La fuente de lámina (una fila entera) también pasa por la corrección de J en banda ancha."""
    p = _cilindro()
    out = {}
    for be in ("numpy", "numba"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = BroadbandMaxwellTMSolver(p, (1e9, 3e9), npml=8, backend=be)
            s.set_plane_wave_source(z=p.grid.z0 + 11 * p.grid.dx)
        out[be] = s.run(np.array([1.5e9, 2.5e9]), t_max=2e-9).phasor
    assert _rel(out["numpy"], out["numba"]) < 1e-11


def test_maxwell_backend_invalido():
    p = _cilindro(n=20)
    with pytest.raises(ValueError, match="backend"):
        MaxwellTMSolver(p, 2.45e9, npml=4, backend="gpu")
    with pytest.raises(ValueError, match="float64"):
        MaxwellTMSolver(p, 2.45e9, npml=4, backend="numba", dtype=np.float32)
    # auto con float32 cae en NumPy en vez de fallar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert MaxwellTMSolver(p, 2.45e9, npml=4, dtype=np.float32).backend == "numpy"
        assert MaxwellTMSolver(p, 2.45e9, npml=4).backend == "numba"
