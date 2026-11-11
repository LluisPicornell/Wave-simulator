"""Regresión de estabilidad a tiempos largos del solver elástico (``docs/validacion/estabilidad.md``).

Con la CPML clásica (``pml_ratio=0``) una placa de cortical que atraviesa la capa lateral crecía
exponencialmente (modos de Lamb retrógrados amplificados dentro de la capa); la tibia realista
osteoporótica, cuya cortical fina entra en la CPML lateral, también. Con la M-PML por defecto la energía
no crece.
"""

import warnings

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.core.signals import gaussian_pulse
from wavesim.phantoms import Fracture, longitudinal_limb
from wavesim.phantoms.phantom import Phantom
from wavesim.ultrasound import ElasticMedium, ElasticSolver, LinearArray, Source

BACKENDS = ["numba", "numpy"]


def _energia(s):
    """Energía cinética más una medida de la elástica (cuadrática en las tensiones)."""
    m = s.medium
    kin = 0.5 * np.sum(m.rho * (s.vx.astype(float) ** 2 + s.vz.astype(float) ** 2))
    mod = m.rho * m.cp**2
    el = 0.5 * np.sum((s.sxx.astype(float) ** 2 + s.szz.astype(float) ** 2 + s.sxz.astype(float) ** 2) / mod)
    return kin + el


def _historia(s, nt, cada):
    e = []
    for n in range(nt):
        s.step(n)
        if n % cada == 0:
            e.append(_energia(s))
    return np.array(e)


def _placa(backend, ratio=None):
    """Placa de cortical de 1.5 mm en agua que atraviesa las dos CPML laterales, excitada en el centro."""
    dx, f0, npml = 0.15e-3, 300e3, 20
    nz, nx, ja, jb = 70, 90, 30, 40
    vals = [np.full((nz, nx), v) for v in (1000.0, 1500.0, 0.0)]
    for a, v in zip(vals, (1908.0, 3500.0, 1800.0)):
        a[ja:jb] = v
    z = np.zeros((nz, nx))
    med = ElasticMedium(*vals, z, z.copy(), np.zeros(0), f0)
    g = Grid2D(nx, nz, dx)
    kw = {} if ratio is None else {"pml_ratio": ratio}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ElasticSolver(Phantom(g, np.zeros(g.shape, np.int16)), f0, npml=npml, medium=med, backend=backend, **kw)
    nt = 12000
    col = np.arange(ja, jb)
    pulse = gaussian_pulse(s.time_axis(nt), f0, 1.2)[None, :]
    for kind in ("force_x", "force_z"):
        s.add_source(Source(col, np.full(col.size, nx // 2), np.zeros(col.size, int), pulse, kind))
    return s, nt


@pytest.mark.parametrize("backend", BACKENDS)
def test_placa_que_entra_en_la_cpml_lateral_no_crece(backend):
    s, nt = _placa(backend)
    e = _historia(s, nt, 500)
    assert np.isfinite(e).all()
    # tras el pulso la energía sólo puede bajar (en la CPML clásica crecía ~x40 cada 500 pasos)
    assert e[-1] < 1e-2 * e.max() and e[-1] <= 1.5 * e[len(e) // 2]


def test_placa_con_cpml_clasica_crece():
    """El montaje sí detecta la inestabilidad: con ``pml_ratio=0`` la energía se dispara."""
    s, _ = _placa("auto", ratio=0.0)
    e = _historia(s, 6000, 500)
    assert not np.isfinite(e[-1]) or e[-1] > 1e3 * e.max(initial=0, where=np.arange(e.size) < 4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tibia_realista_osteoporotica_sin_atenuacion_no_crece(backend):
    p = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=8e-3, depth=14e-3, dx=1450 / 2e6 / 6,
                          body="anciano_osteoporotico", realism="realista")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ElasticSolver(p, 2e6, n_mechanisms=0, backend=backend)
    nt = 12000
    arr = LinearArray(n_elements=12, pitch=0.35e-3, z=1.2e-3, f0=2e6)
    s.add_source(arr.transmit_source(p.grid, s.time_axis(nt), arr.plane_wave_delays(0, 1540)))
    e = _historia(s, nt, 1000)
    assert np.isfinite(e).all()
    assert e[-1] < 1e-3 * e.max() and e[-1] <= 1.5 * e[len(e) // 2]
