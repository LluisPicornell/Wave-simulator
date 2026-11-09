"""El backend Numba debe reproducir al de NumPy (salvo redondeo float32)."""

import numpy as np
import pytest

pytest.importorskip("numba")

from wavesim.core.grid import Grid2D  # noqa: E402
from wavesim.core.signals import gaussian_pulse  # noqa: E402
from wavesim.phantoms import Fracture, longitudinal_limb  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402
from wavesim.ultrasound import ElasticSolver, Receiver, Source  # noqa: E402

RTOL = 1e-4


def _run_both(phantom, f0, kind, nt, **kw):
    """Ejecuta el mismo caso con ambos backends y devuelve (solver, resultado) de cada uno."""
    out = {}
    for backend in ("numpy", "numba"):
        s = ElasticSolver(phantom, f0, npml=10, backend=backend, **kw)
        g = phantom.grid
        # Fila de fuentes cerca de la superficie y receptores en otra más profunda
        i = np.arange(g.nx // 4, 3 * g.nx // 4)
        j = np.full(i.size, 14)
        t = s.time_axis(nt)
        sig = gaussian_pulse(t, f0)[None, :]
        s.add_source(Source(j, i, np.zeros(i.size, int), sig, kind))
        s.add_receiver("p", Receiver(np.full(i.size, g.nz // 2), i, np.zeros(i.size, int), 1, "p"))
        s.add_receiver("vx", Receiver(np.full(i.size, g.nz // 2), i, np.zeros(i.size, int), 1, "vx"))
        out[backend] = (s, s.run(nt, track_peak=True))
    return out


def _assert_close(out, fields=("sxx", "szz", "sxz", "vx", "vz")):
    (sa, ra), (sb, rb) = out["numpy"], out["numba"]
    assert sb.backend == "numba" and sa.backend == "numpy"
    for name in fields:
        a, b = getattr(sa, name), getattr(sb, name)
        scale = np.abs(a).max()
        assert scale > 0
        assert np.abs(a - b).max() <= RTOL * scale, name
    for k in ra.traces:
        scale = np.abs(ra.traces[k]).max()
        assert np.abs(ra.traces[k] - rb.traces[k]).max() <= RTOL * scale, k
    assert np.abs(ra.peak_pressure - rb.peak_pressure).max() <= RTOL * ra.peak_pressure.max()


def test_homogeneous_water():
    g = Grid2D.from_size(12e-3, 12e-3, 1500.0 / 1e6 / 10, x0=-6e-3)
    ph = Phantom(g, np.full(g.shape, TISSUE_INDEX["water"], np.int16))
    _assert_close(_run_both(ph, 1e6, "pressure", 150, attenuation=False), ("sxx", "szz", "vx", "vz"))


def test_heterogeneous_bone_fracture_with_attenuation():
    f0 = 2e6
    ph = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=8e-3, depth=14e-3, dx=1450 / f0 / 8)
    out = _run_both(ph, f0, "pressure", 220, attenuation=True, n_mechanisms=2)
    assert out["numpy"][0].has_shear and out["numpy"][0].n_mech == 2
    _assert_close(out)


def test_per_mechanism_weights_in_porous_bone():
    """Tres mecanismos con pesos distintos por celda (poros de la cortical realista)."""
    f0 = 2e6
    ph = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=8e-3, depth=14e-3, dx=1450 / f0 / 8,
                           realism="realista")
    out = _run_both(ph, f0, "pressure", 220, attenuation=True, n_mechanisms=3)
    w = out["numpy"][0].dt_pi_t
    assert w.shape[0] == 3 and not np.allclose(w[0], w[1])  # cada mecanismo con su peso
    _assert_close(out)


def test_force_source():
    f0 = 2e6
    ph = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=8e-3, depth=14e-3, dx=1450 / f0 / 8)
    _assert_close(_run_both(ph, f0, "force_x", 220, attenuation=True))


def test_backend_selection():
    g = Grid2D.from_size(4e-3, 4e-3, 1500.0 / 1e6 / 10, x0=-2e-3)
    ph = Phantom(g, np.full(g.shape, TISSUE_INDEX["water"], np.int16))
    assert ElasticSolver(ph, 1e6, backend="auto").backend == "numba"
    assert ElasticSolver(ph, 1e6, backend="numpy").backend == "numpy"
    with pytest.raises(ValueError):
        ElasticSolver(ph, 1e6, backend="cuda")
    with pytest.raises(ValueError):
        ElasticSolver(ph, 1e6, backend="numba", dtype=np.float64)
