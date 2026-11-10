"""Simulador de Bloch espacial (``wavesim.mri.spatial``): motor por eventos, secuencias y reconstrucción."""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.mri import bloch, mri_scan, spatial, tissue_signal
from wavesim.mri.spatial import ADC, RF, Free, Isochromats, Spoil
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

GRID = Grid2D.from_size(8e-3, 8e-3, 0.5e-3, x0=-4e-3, z0=-4e-3)


def uniform(tissue: str, grid: Grid2D = GRID, porosity: float | None = None, fill: str = "blood") -> Phantom:
    lab = np.full(grid.shape, TISSUE_INDEX[tissue], np.int16)
    por = None if porosity is None else np.full(grid.shape, porosity)
    return Phantom(grid, lab, porosity=por, pore_fill=fill)


def test_rotation_convention_matches_bloch_module():
    # 90°x lleva Mz a +My, como bloch.rotate_x; y un espín-eco de un isocromato da lo mismo que bloch.py
    iso = Isochromats([0.0], [0.0], [1.0], [1e9], [1e9])
    _, st = spatial.simulate(iso, [RF(np.pi / 2)])
    assert st.f[0, 0] == pytest.approx(1j, abs=1e-15)
    t1, t2, tr, te = 0.6, 0.08, 0.5, 0.015
    iso = Isochromats([0.0], [0.0], [1.0], [t1], [t2])
    events = [RF(np.pi / 2), Free(te / 2), RF(np.pi, np.pi / 2), Free(te / 2), ADC(1, 1e-9), Spoil(),
              Free(tr - te - 1e-9)]
    mz = spatial.steady_state_mz(iso, events)
    sig, _ = spatial.simulate(iso, events, spatial._State.longitudinal(mz))
    ref = bloch.spin_echo_steady_state(t1 * 1e3, t2 * 1e3, tr * 1e3, te * 1e3)
    assert abs(sig[0][0]) == pytest.approx(float(ref), rel=1e-9)


def test_rect_pulse_off_resonance_follows_rabi_formula():
    # Pulso rectangular de duración T con desvío Δ: Mz = (Δ² + w1² cos(Ωe T)) / Ωe², Ωe² = w1² + Δ²
    dur, flip = 2e-3, np.pi / 2
    w1 = flip / dur
    delta = np.array([0.0, 300.0, 1000.0, -700.0])
    iso = Isochromats(np.zeros(4), np.zeros(4), np.ones(4), np.full(4, 1e9), np.full(4, 1e9), dw=delta)
    _, st = spatial.simulate(iso, [RF(flip, duration=dur)])
    we = np.hypot(w1, delta)
    np.testing.assert_allclose(st.mz(), (delta**2 + w1**2 * np.cos(we * dur)) / we**2, atol=1e-12)


@pytest.mark.parametrize("seq", ["T1", "T2", "PD", "PD_FS", "STIR", "GRE"])
@pytest.mark.parametrize("tissue", ["muscle", "fat", "marrow", "edema", "hematoma"])
def test_uniform_phantom_matches_signal_equation(seq, tissue):
    # Sin artefactos (sin desplazamiento químico ni B0), la imagen de un tejido uniforme es su señal analítica
    im = spatial.bloch_scan(uniform(tissue), seq, 1.5, snr=0, texture=0, matrix=(16, 16), backend="numpy")
    np.testing.assert_allclose(im.magnitude, tissue_signal(tissue, seq, 1.5), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("tissue, phi, fill", [("cortical_bone", 0.3, "blood"), ("trabecular_bone", 0.5, "blood"),
                                               ("cortical_bone", 0.6, "marrow")])
def test_porous_cells_follow_linear_mri_mixing(tissue, phi, fill):
    p = uniform(tissue, porosity=phi, fill=fill)
    sig = {k: tissue_signal(k, "PD", 1.5) for k in p.property_tissues()}
    ref = p.property_map(lambda k: sig[k], physics="mri")
    im = spatial.bloch_scan(p, "PD", 1.5, snr=0, texture=0, matrix=(16, 16), backend="numpy")
    np.testing.assert_allclose(im.magnitude, ref, rtol=1e-9)


def _small_isochromats():
    g = Grid2D.from_size(6e-3, 6e-3, 0.5e-3, x0=-3e-3, z0=-3e-3)
    lab = np.full(g.shape, TISSUE_INDEX["muscle"], np.int16)
    X, Z = g.meshgrid()
    lab[X**2 + Z**2 < 2e-3**2] = TISSUE_INDEX["marrow"]
    iso = spatial.phantom_isochromats(Phantom(g, lab), 1.5, 0.0)
    iso.dw[:] = np.linspace(-300.0, 500.0, iso.n)  # desvíos de frecuencia arbitrarios
    iso.b1[:] = np.linspace(0.8, 1.15, iso.n)  # B1+ no uniforme: pulsos imperfectos, ecos estimulados
    return iso


@pytest.mark.parametrize("seq", ["T1", "STIR", "GRE"])
def test_steady_state_shortcut_matches_event_by_event(seq):
    iso = _small_isochromats()
    cseq = spatial.cartesian_sequence(seq, 1.5, (12, 12), (6e-3, 6e-3))
    fast = spatial.acquire(iso, cseq, "steady", "numpy")
    slow = spatial.acquire(iso, cseq, "events", "numpy")
    assert np.abs(fast - slow).max() < 1e-12 * np.abs(fast).max()


def test_dummy_scans_converge_to_exact_steady_state():
    iso = _small_isochromats()
    cseq = spatial.cartesian_sequence("GRE", 1.5, (8, 8), (6e-3, 6e-3))
    fast = spatial.acquire(iso, cseq, "steady", "numpy")
    few = spatial.acquire(iso, cseq, "events", "numpy", n_dummy=2)
    many = spatial.acquire(iso, cseq, "events", "numpy", n_dummy=400)
    scale = np.abs(fast).max()
    assert np.abs(few - fast).max() > 1e-3 * scale  # transitorio visible con pocos TR de preparación
    assert np.abs(many - fast).max() < 1e-8 * scale


def test_numba_backend_matches_numpy():
    pytest.importorskip("numba")
    iso = _small_isochromats()
    cseq = spatial.cartesian_sequence("STIR", 1.5, (12, 12), (6e-3, 6e-3))
    a = spatial.acquire(iso, cseq, "steady", "numpy")
    b = spatial.acquire(iso, cseq, "steady", "numba")
    assert np.abs(a - b).max() < 1e-12 * np.abs(a).max()


@pytest.mark.parametrize("readout", ["x", "z"])
def test_bloch_axes_locate_a_block_like_mri_scan(readout):
    g = Grid2D.from_size(24e-3, 22e-3, 0.2e-3, x0=-12e-3, z0=1e-3)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    X, Z = g.meshgrid()
    blk = (X >= -3e-3) & (X < 5e-3) & (Z >= 6e-3) & (Z < 14e-3)
    lab[blk] = TISSUE_INDEX["water"]
    p = Phantom(g, lab)
    an = mri_scan(p, "PD", voxel=0.5e-3, snr=0, texture=0)
    im = mri_scan(p, "PD", voxel=0.5e-3, snr=0, texture=0, engine="bloch", readout=readout, backend="numpy")
    assert im.magnitude.shape == an.magnitude.shape
    np.testing.assert_allclose(im.x, an.x, atol=1e-12)
    np.testing.assert_allclose(im.z, an.z, atol=1e-12)
    mag = im.magnitude
    cx = np.sum(mag.sum(axis=0) * im.x) / mag.sum()
    cz = np.sum(mag.sum(axis=1) * im.z) / mag.sum()
    assert abs(cx - X[blk].mean()) < 0.25 * im.voxel
    assert abs(cz - Z[blk].mean()) < 0.25 * im.voxel
    # Dentro del bloque coincide con el modo analítico; en los bordes no del todo: el desfase T2'
    # durante la lectura (exp(-|t - TE|/T2')) atenúa los extremos del k-space y suaviza el Gibbs
    inner = (np.abs(im.x[None, :] - 1e-3) < 2e-3) & (np.abs(im.z[:, None] - 10e-3) < 2e-3)
    np.testing.assert_allclose(mag[inner], an.magnitude[inner], rtol=1e-2)


def test_noise_level_matches_requested_snr():
    g = Grid2D.from_size(32e-3, 32e-3, 0.5e-3, x0=-16e-3, z0=-16e-3)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    X, Z = g.meshgrid()
    lab[X**2 + Z**2 < 6e-3**2] = TISSUE_INDEX["muscle"]
    im = spatial.bloch_scan(Phantom(g, lab), "T2", snr=20, texture=0, matrix=(64, 64), backend="numpy")
    corner = im.complex_image[:12, :12]  # aire: sólo ruido
    sigma = tissue_signal("muscle", "T2") / 20
    assert np.std(corner.real) == pytest.approx(sigma, rel=0.15)
    assert np.std(corner.imag) == pytest.approx(sigma, rel=0.15)


def test_infeasible_timing_and_unknown_engine_raise():
    with pytest.raises(ValueError, match="UTE"):
        spatial.cartesian_sequence("UTE", 1.5, (16, 16), (8e-3, 8e-3))
    with pytest.raises(ValueError, match="ancho de banda"):
        spatial.cartesian_sequence("GRE", 1.5, (16, 16), (8e-3, 8e-3), bandwidth=50.0)
    with pytest.raises(ValueError, match="motor"):
        mri_scan(uniform("muscle"), engine="koma")
