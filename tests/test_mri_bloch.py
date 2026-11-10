"""Simulador de Bloch espacial (``wavesim.mri.spatial``): motor por eventos, secuencias y reconstrucción."""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.materials import TISSUES
from wavesim.materials.mri import GAMMA_BAR, SUSCEPTIBILITY_PPM
from wavesim.mri import bloch, mri_scan, signal, spatial, tissue_signal
from wavesim.mri import susceptibility as sus
from wavesim.mri.sequences import Sequence
from wavesim.mri.spatial import ADC, RF, BlochPhysics, Free, Isochromats, Spoil, SurfaceCoil
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

GRID = Grid2D.from_size(8e-3, 8e-3, 0.5e-3, x0=-4e-3, z0=-4e-3)
NONE = BlochPhysics.none()


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
    im = spatial.bloch_scan(uniform(tissue), seq, 1.5, snr=0, texture=0, matrix=(16, 16), physics=NONE,
                            backend="numpy")
    np.testing.assert_allclose(im.magnitude, tissue_signal(tissue, seq, 1.5), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("tissue, phi, fill", [("cortical_bone", 0.3, "blood"), ("trabecular_bone", 0.5, "blood"),
                                               ("cortical_bone", 0.6, "marrow")])
def test_porous_cells_follow_linear_mri_mixing(tissue, phi, fill):
    p = uniform(tissue, porosity=phi, fill=fill)
    sig = {k: tissue_signal(k, "PD", 1.5) for k in p.property_tissues()}
    ref = p.property_map(lambda k: sig[k], physics="mri")
    im = spatial.bloch_scan(p, "PD", 1.5, snr=0, texture=0, matrix=(16, 16), physics=NONE, backend="numpy")
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
    # Sin T2' (que atenúa los extremos del k-space durante la lectura y suaviza el Gibbs) y con el T2
    # de 2 s del agua, la imagen de Bloch debe ser la del k-space recortado del modo analítico
    flat = BlochPhysics(chemical_shift=False, susceptibility=False, t2_prime=False)
    im = mri_scan(p, "PD", voxel=0.5e-3, snr=0, texture=0, engine="bloch", readout=readout, physics=flat,
                  backend="numpy")
    assert im.magnitude.shape == an.magnitude.shape
    np.testing.assert_allclose(im.x, an.x, atol=1e-12)
    np.testing.assert_allclose(im.z, an.z, atol=1e-12)
    mag = im.magnitude
    cx = np.sum(mag.sum(axis=0) * im.x) / mag.sum()
    cz = np.sum(mag.sum(axis=1) * im.z) / mag.sum()
    assert abs(cx - X[blk].mean()) < 0.25 * im.voxel
    assert abs(cz - Z[blk].mean()) < 0.25 * im.voxel
    assert np.abs(mag - an.magnitude).max() < 3e-3 * an.magnitude.max()


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


# --------------------------------------------------------------------------
# Física de los artefactos
# --------------------------------------------------------------------------
def test_susceptibility_table_covers_every_tissue():
    assert set(TISSUES) <= set(SUSCEPTIBILITY_PPM)
    assert SUSCEPTIBILITY_PPM["water"] == -9.05 and SUSCEPTIBILITY_PPM["air"] == 0.36
    # Orden físico: hueso más diamagnético que el agua, grasa menos, aire paramagnético
    assert SUSCEPTIBILITY_PPM["cortical_bone"] < SUSCEPTIBILITY_PPM["muscle"] < SUSCEPTIBILITY_PPM["fat"] < 0


def _circular_centroid(img: np.ndarray) -> float:
    """Centroide (píxeles) a lo largo de las columnas por la fase del primer armónico respecto del DC.

    Para un objeto simétrico de banda limitada desplazado es exacto, también con desplazamientos sub-píxel.
    """
    n = img.shape[1]
    h1 = (img * np.exp(-2j * np.pi * np.arange(n) / n)[None, :]).sum()
    return (-np.angle(h1 / img.sum()) * n / (2 * np.pi)) % n


@pytest.mark.parametrize("b0, bw", [(1.5, 130.0), (3.0, 260.0), (1.5, 80.0)])
def test_fat_chemical_shift_equals_df_over_bandwidth(b0, bw):
    n, fov = 32, 16e-3
    c = -fov / 2 + (np.arange(2 * n) + 0.5) * fov / (2 * n)
    X, Z = np.meshgrid(c, c)
    blk = (np.abs(X) < 3e-3) & (np.abs(Z) < 3e-3)
    seq = Sequence("SE", "SE", tr=500.0, te=30.0)
    cseq = spatial.cartesian_sequence(seq, b0, (n, n), (fov, fov), bw, "x")
    origin = (-fov / 2 + fov / n / 2,) * 2
    cen = {}
    for name, ppm in (("agua", 0.0), ("grasa", -3.4)):
        iso = Isochromats(X[blk], Z[blk], 1.0, 0.3, 1e3, 0.0, 2 * np.pi * ppm * 1e-6 * GAMMA_BAR * b0)
        img = spatial.reconstruct(spatial.acquire(iso, cseq, "steady", "numpy"), cseq, origin)
        cen[name] = _circular_centroid(img), np.sum(np.abs(img).sum(0) * np.arange(n)) / np.abs(img).sum()
    expected = -3.4e-6 * GAMMA_BAR * b0 / bw
    assert cen["grasa"][0] - cen["agua"][0] == pytest.approx(expected, abs=0.01)
    assert cen["grasa"][1] - cen["agua"][1] == pytest.approx(expected, abs=0.2)  # lo que se ve en la magnitud
    im = spatial.bloch_scan(uniform("fat"), seq, b0, snr=0, matrix=(16, 16), bandwidth=bw)
    assert im.fat_shift_px == pytest.approx(expected)


def _offset_phantom(offsets_hz: np.ndarray, weights: np.ndarray, r2p: float = 0.0) -> Isochromats:
    X, Z = GRID.meshgrid()
    k = offsets_hz.size
    return Isochromats(np.repeat(X.ravel(), k), np.repeat(Z.ravel(), k), np.tile(weights, X.size), 1.0, 0.05,
                       r2p, 2 * np.pi * (37.0 + np.tile(offsets_hz, X.size)))


def _uniform_value(iso: Isochromats, seq: Sequence) -> float:
    cseq = spatial.cartesian_sequence(seq, 1.5, (16, 16), (8e-3, 8e-3), 1000.0)
    img = spatial.reconstruct(spatial.acquire(iso, cseq, "steady", "numpy"), cseq, (-4e-3, -4e-3))
    return float(np.abs(img).mean() * img.size / GRID.nx / GRID.nz)


@pytest.mark.parametrize("te", [4.0, 12.0, 30.0])
def test_spin_echo_refocuses_and_gre_decays_with_the_offset_distribution(te):
    t1, t2 = 1000.0, 50.0
    hwhm = 25.0  # Hz: T2' = 1/(2π·25 Hz) = 6.4 ms
    nodes, w = np.polynomial.hermite_e.hermegauss(48)
    sigma = 15.0
    cases = {
        # Lorentziana exacta (estados de desfase) -> exp(-TE/T2')
        "lorentz": (_offset_phantom(np.zeros(1), np.ones(1), 2 * np.pi * hwhm), np.exp(-2 * np.pi * hwhm * te * 1e-3)),
        # Gaussiana con cuadratura de Gauss-Hermite -> exp(-(2π σ TE)²/2)
        "gauss": (_offset_phantom(sigma * nodes, w / w.sum()), np.exp(-0.5 * (2 * np.pi * sigma * te * 1e-3) ** 2)),
    }
    se = Sequence("SE", "SE", tr=200.0, te=te)
    gre = Sequence("GRE", "GRE", tr=200.0, te=te, flip_deg=30.0)
    for iso, decay in cases.values():
        # Espín-eco: la señal no depende del desvío (37 Hz global) ni de su dispersión
        assert _uniform_value(iso, se) == pytest.approx(float(signal(1.0, t1, t2, t2, se)), rel=1e-9)
        assert _uniform_value(iso, gre) == pytest.approx(float(signal(1.0, t1, t2, t2, gre)) * decay, rel=1e-9)


def test_dipole_field_of_a_cylinder_perpendicular_to_b0():
    n, a = 128, 16.0
    sub = (np.arange(8) + 0.5) / 8 - 0.5
    c = np.arange(n) - n / 2 + 0.5
    frac = ((c[None, :, None, None] + sub[None, None, None, :]) ** 2
            + (c[:, None, None, None] + sub[None, None, :, None]) ** 2 <= a**2).mean(axis=(2, 3))
    X, Z = np.meshgrid(c, c)
    far = np.abs(np.hypot(X, Z) - a) > 2  # fuera de la escalera del borde
    for direction in ("z", "x"):
        err = np.abs(sus.dipole_field(frac + sus.CHI_WATER, direction) - sus.cylinder_field(X, Z, a, 1.0, direction))
        assert err[far].max() < 0.01 * 0.5  # < 1 % de Δχ/2 (se mide 0.64 %)
    # B0 oblicuo a la malla: la escalera del borde deja estrías que decaen despacio (5 % a 2-4 celdas
    # del borde); el error cuadrático medio sigue siendo pequeño
    err = sus.dipole_field(frac + sus.CHI_WATER, (1.0, 0.0, 1.0)) - sus.cylinder_field(X, Z, a, 1.0, (1, 0, 1))
    assert np.sqrt(np.mean(err[far] ** 2)) < 0.01 * 0.5
    # B0 perpendicular al corte: sólo el desplazamiento de Lorentz Δχ/3 dentro, nada fuera
    f = sus.dipole_field(frac + sus.CHI_WATER, "y")
    np.testing.assert_allclose(f, frac / 3.0, atol=1e-12)


def test_auto_b0_direction_follows_bore_geometry():
    ph = BlochPhysics()
    assert ph.direction(Phantom(GRID, np.zeros(GRID.shape, np.int16), meta={"view": "cross_section"})) == "y"
    assert ph.direction(Phantom(GRID, np.zeros(GRID.shape, np.int16), meta={"view": "longitudinal"})) == "x"
    assert BlochPhysics(b0_direction="z").direction(uniform("muscle")) == "z"


def test_gre_loses_signal_near_air_but_spin_echo_does_not():
    g = Grid2D.from_size(32e-3, 32e-3, 0.25e-3, x0=-16e-3, z0=-16e-3)
    X, Z = g.meshgrid()
    lab = np.full(g.shape, TISSUE_INDEX["muscle"], np.int16)
    lab[np.hypot(X, Z) < 4e-3] = TISSUE_INDEX["air"]
    p = Phantom(g, lab)
    ratios = {}
    for name, seq in (("SE", "PD"), ("GRE", Sequence("GRE", "GRE", tr=50.0, te=15.0, flip_deg=20.0))):
        with_b0 = spatial.bloch_scan(p, seq, snr=0, texture=0, matrix=(64, 64),
                                     physics=BlochPhysics(chemical_shift=False, b0_direction="z"))
        ref = spatial.bloch_scan(p, seq, snr=0, texture=0, matrix=(64, 64), physics=NONE)
        xi, zi = np.meshgrid(with_b0.x, with_b0.z)
        # Región que incluye el hueco: la distorsión mueve señal pero en espín-eco no la destruye
        region = np.hypot(xi, zi) < 8e-3
        ratios[name] = with_b0.magnitude[region].sum() / ref.magnitude[region].sum()
    assert ratios["SE"] == pytest.approx(1.0, abs=0.02)
    assert ratios["GRE"] < 0.9


def test_fat_saturation_misses_fat_shifted_by_b0():
    p = uniform("fat")
    cseq = spatial.cartesian_sequence("PD_FS", 1.5, (16, 16), (8e-3, 8e-3))
    vals = []
    for off in (0.0, 120.0):  # la banda de saturación es de ±1 ppm (±64 Hz a 1.5 T)
        iso = spatial.phantom_isochromats(p, 1.5, 0.0, 1.0, field_hz=np.full(GRID.shape, off))
        vals.append(np.abs(spatial.reconstruct(spatial.acquire(iso, cseq), cseq, (-4e-3, -4e-3))).mean())
    assert vals[0] == pytest.approx(tissue_signal("fat", "PD_FS"), rel=1e-9)
    assert vals[1] == pytest.approx(tissue_signal("fat", "PD"), rel=1e-9)  # grasa sin saturar


def test_surface_coil_weights_the_image():
    coil = SurfaceCoil((0.0, -6e-3), 5e-3)
    im = spatial.bloch_scan(uniform("muscle"), "PD", snr=0, texture=0, matrix=(16, 16),
                            physics=BlochPhysics(coil=coil), backend="numpy")
    xi, zi = np.meshgrid(im.x, im.z)
    ref = tissue_signal("muscle", "PD") * coil.sensitivity(xi, zi)
    assert np.abs(im.magnitude - ref).max() < 0.03 * ref.max()
