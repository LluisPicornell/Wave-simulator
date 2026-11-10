"""Solver elástico 3D y fantomas 3D: validación rápida con mallas diminutas.

Los casos completos (más finos y con tablas) están en
``benchmarks/elastic3d_validation.py`` y ``docs/validacion/elastico3d.md``.
"""

import warnings

import numpy as np
import pytest

from wavesim.core.grid3d import Grid3D
from wavesim.core.signals import amplitude_at, gaussian_pulse
from wavesim.materials.acoustic import ACOUSTIC, CORTICAL_STIFFNESS
from wavesim.phantoms.limb3d import Phantom3D, homogeneous_phantom
from wavesim.phantoms.phantom import TISSUE_INDEX
from wavesim.ultrasound.elastic3d import (
    ElasticMedium3D, ElasticSolver3D, Receiver3D, Source3D, isotropic_stiffness_3d, point_source_pressure,
)

F0 = 1e6
WATER = (1000.0, 1500.0, 0.0)
BONE = (1908.0, 3500.0, 1800.0)


def _medium(shape, rho, cp, cs, stiffness=None):
    full = lambda v: np.broadcast_to(np.asarray(v, np.float32), shape).copy()  # noqa: E731
    z = np.zeros((0, *shape), np.float32)
    return ElasticMedium3D(full(rho), full(cp), full(cs), z, z.copy(), np.zeros(0), F0, stiffness)


def _solver(grid, medium, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElasticSolver3D(homogeneous_phantom(grid), F0, medium=medium, **kw)


def _xcorr_lag(a, b, dt):
    n = a.size
    xc = np.fft.irfft(np.fft.rfft(b, 2 * n) * np.conj(np.fft.rfft(a, 2 * n)))
    k = int(np.argmax(xc[:n]))
    y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
    return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * dt


def _plane_wave_speed(axis, kind, quantity, rho, cp, cs, stiffness=None, ppw=10, c_ref=1500.0):
    """Velocidad de una onda plana a lo largo de ``axis`` (0 = z, 1 = y, 2 = x): campos uniformes en los otros ejes."""
    dx = c_ref / F0 / ppw
    n = [4, 4, 4]
    n[axis] = int(round(36e-3 / dx)) + 20
    grid = Grid3D(n[2], n[1], n[0], dx)
    npml = [0, 0, 0]
    npml[axis] = 10
    shape = grid.shape
    st = None if stiffness is None else np.broadcast_to(np.asarray(stiffness, np.float32)[:, None, None, None],
                                                        (5, *shape)).copy()
    s = _solver(grid, _medium(shape, rho, cp, cs, st), npml=tuple(npml))
    nt = int(26e-6 / s.dt)
    t = s.time_axis(nt)

    def plane(index):
        idx = [np.arange(4)] * 3
        idx[axis] = np.array([index])
        k, j, i = (a.ravel() for a in np.meshgrid(*idx, indexing="ij"))
        return k, j, i

    k, j, i = plane(14)
    s.add_source(Source3D(k, j, i, np.zeros(k.size, int), gaussian_pulse(t, F0)[None, :], kind))
    ia, ib = 14 + int(round(9e-3 / dx)), 14 + int(round(24e-3 / dx))
    for name, index in (("a", ia), ("b", ib)):
        k, j, i = plane(index)
        s.add_receiver(name, Receiver3D(k, j, i, np.zeros(k.size, int), 1, quantity))
    r = s.run(nt)
    return (ib - ia) * dx / _xcorr_lag(r.traces["a"][0], r.traces["b"][0], s.dt)


# ----------------------------------------------------------------------------
# Velocidades y reflexión (ondas planas: campos uniformes en dos ejes)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_velocidad_p_en_agua_a_lo_largo_de_cada_eje(axis):
    assert _plane_wave_speed(axis, "pressure", "p", *WATER) == pytest.approx(1500.0, rel=0.01)


@pytest.mark.parametrize("axis, kind, quantity", [(0, "force_x", "vx"), (0, "force_y", "vy"), (2, "force_z", "vz"),
                                                  (1, "force_x", "vx")])
def test_velocidad_s_en_hueso_con_cada_polarizacion(axis, kind, quantity):
    assert _plane_wave_speed(axis, kind, quantity, *BONE, ppw=12, c_ref=1800.0) == pytest.approx(1800.0, rel=0.01)


def test_velocidades_de_la_cortical_anisotropa():
    """Eje del hueso en x: P axial sqrt(C_A/rho), P transversal sqrt(C_T/rho), S con el eje sqrt(G_A/rho)."""
    rho = ACOUSTIC["cortical_bone"].density
    ti = CORTICAL_STIFFNESS
    st = np.array([ti.c_a, ti.c_t, ti.c_ta, ti.g_a, ti.g_t]) / rho
    args = (rho, np.sqrt(st[0]), np.sqrt(st[3]), st)
    assert _plane_wave_speed(2, "pressure", "p", *args, ppw=10, c_ref=3221.0) == pytest.approx(np.sqrt(st[0]), rel=0.01)
    assert _plane_wave_speed(0, "pressure", "p", *args, ppw=10, c_ref=3221.0) == pytest.approx(np.sqrt(st[1]), rel=0.01)
    assert _plane_wave_speed(0, "force_x", "vx", *args, ppw=12, c_ref=1690.0) == pytest.approx(np.sqrt(st[3]), rel=0.01)
    # S que se propaga en el plano transversal (y-z): C44 = G_T
    assert _plane_wave_speed(0, "force_y", "vy", *args, ppw=12, c_ref=1457.0) == pytest.approx(np.sqrt(st[4]), rel=0.01)


@pytest.mark.parametrize("aniso", [False, True])
def test_reflexion_a_incidencia_normal_agua_cortical(aniso):
    dx = 1500.0 / F0 / 12
    nz = int(round(34e-3 / dx))
    grid = Grid3D(4, 4, nz, dx)
    zc = 24e-3
    bone = (grid.z >= zc)[:, None, None] & np.ones(grid.shape, bool)
    rho = np.where(bone, BONE[0], WATER[0])
    cp = np.where(bone, BONE[1], WATER[1])
    cs = np.where(bone, BONE[2], WATER[2])
    st, z_bone = None, BONE[0] * BONE[1]
    if aniso:
        ti = CORTICAL_STIFFNESS
        st = isotropic_stiffness_3d(cp, cs)
        st[:, bone] = (np.array([ti.c_a, ti.c_t, ti.c_ta, ti.g_a, ti.g_t]) / BONE[0])[:, None]
        z_bone = np.sqrt(BONE[0] * ti.c_t)  # a lo largo de z (transversal al eje)
    s = _solver(grid, _medium(grid.shape, rho, cp, cs, None if st is None else st.astype(np.float32)),
                npml=(12, 0, 0))
    nt = int(30e-6 / s.dt)
    t = s.time_axis(nt)
    k, j, i = grid.cells_in_box((0, 1), (0, 1), (5e-3, 5e-3))
    s.add_source(Source3D(k, j, i, np.zeros(k.size, int), gaussian_pulse(t, F0)[None, :]))
    k, j, i = grid.cells_in_box((0, 1), (0, 1), (12e-3, 12e-3))
    s.add_receiver("r", Receiver3D(k, j, i, np.zeros(k.size, int), 1))
    tr = s.run(nt).traces["r"][0]
    # cociente espectral en f0 (el de los picos depende de la dispersión del pulso)
    t_split = (zc - 12e-3) / 1500 + 6e-6
    inc, refl = np.where(t < t_split, tr, 0.0), np.where(t >= t_split, tr, 0.0)
    r_meas = abs(amplitude_at(refl, s.dt, F0) / amplitude_at(inc, s.dt, F0))
    z_w = WATER[0] * WATER[1]
    assert r_meas == pytest.approx((z_bone - z_w) / (z_bone + z_w), rel=0.006)


# ----------------------------------------------------------------------------
# Fuente puntual: función de Green 3D
# ----------------------------------------------------------------------------
def test_fuente_puntual_decae_como_1_r_y_sigue_la_funcion_de_green():
    """Fuente de presión de una celda en agua: p = dx² s'(t - r/c) / (2 pi r c) (``point_source_pressure``).

    A 10 puntos por longitud de onda la amplitud coincide con la analítica a
    < 1 %; el error de la forma de onda crece con la distancia (~2 % por
    longitud de onda) por la dispersión numérica del esquema.
    """
    f0, c = 1e6, 1500.0
    dx = c / f0 / 10
    n = 72
    grid = Grid3D(n, n, n, dx)
    s = _solver(grid, _medium(grid.shape, *WATER), npml=8)
    nt = int(9e-6 / s.dt)
    t = s.time_axis(nt)
    centre = n // 2 - 14
    s.add_source(Source3D(np.array([centre]), np.array([n // 2]), np.array([n // 2]), np.array([0]),
                          gaussian_pulse(t, f0)[None, :]))
    radii = np.array([10, 14, 18, 22])
    s.add_receiver("r", Receiver3D(centre + radii, np.full(4, n // 2), np.full(4, n // 2), np.arange(4), 4))
    # a la misma distancia en diagonal (x-y): comprueba la isotropía del esquema
    d = 13
    s.add_receiver("diag", Receiver3D(np.array([centre]), np.array([n // 2 + d]), np.array([n // 2 + d]),
                                      np.array([0]), 1))
    res = s.run(nt)
    tr = res.traces["r"]
    h = 1e-12
    deriv = lambda tt: (gaussian_pulse(tt + h, f0) - gaussian_pulse(tt - h, f0)) / (2 * h)  # noqa: E731
    for q, r_cells in enumerate(radii):
        ref = point_source_pressure(r_cells * dx, t, deriv, dx, c, s.dt)
        assert np.abs(tr[q]).max() == pytest.approx(np.abs(ref).max(), rel=0.01)
        err = np.sqrt(np.mean((tr[q] - ref) ** 2)) / np.sqrt(np.mean(ref**2))
        assert err < 0.05, (r_cells, err)
    amp = np.abs(tr).max(axis=1)
    slope = np.polyfit(np.log(radii), np.log(amp), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.01)
    amp_diag = np.abs(res.traces["diag"][0]).max()
    assert amp_diag == pytest.approx(np.interp(np.hypot(d, d), radii, amp), rel=0.02)


# ----------------------------------------------------------------------------
# Backends, interfaz fluido-sólido y estabilidad
# ----------------------------------------------------------------------------
def _tube(dx=0.25e-3):
    """Tubo de cortical con médula, en agua, con un hueco transversal en medio perímetro."""
    grid = Grid3D(28, 26, 26, dx, -14 * dx, -13 * dx, 0.0)
    z, y, x = np.meshgrid(grid.z - 13 * dx, grid.y, grid.x, indexing="ij")
    r = np.hypot(y, z)
    lab = np.full(grid.shape, TISSUE_INDEX["water"], np.int16)
    lab[r <= 1.5e-3] = TISSUE_INDEX["cortical_bone"]
    lab[r <= 0.9e-3] = TISSUE_INDEX["marrow"]
    gap = (np.abs(x - 0.4e-3) <= 0.3e-3) & (z < 0.2e-3) & (r > 0.9e-3) & (r <= 1.5e-3)
    lab[gap] = TISSUE_INDEX["hematoma"]
    return Phantom3D(grid, lab, meta={"bone_axis": "x"})


@pytest.mark.parametrize("aniso, kind", [(False, "pressure"), (True, "force_y")])
def test_backends_numpy_y_numba_coinciden(aniso, kind):
    pytest.importorskip("numba")
    ph = _tube()
    out = {}
    for backend in ("numpy", "numba"):
        s = ElasticSolver3D(ph, 1.5e6, npml=4, n_mechanisms=2, anisotropic_bone=aniso, backend=backend)
        g = ph.grid
        t = s.time_axis(160)
        k, j, i = g.index_of(-1.5e-3, 0.0, 1.1e-3)
        s.add_source(Source3D(np.array([k, k]), np.array([j, j + 3]), np.array([i, i]), np.zeros(2, int),
                              gaussian_pulse(t, 1.5e6)[None, :], kind))
        s.add_receiver("p", Receiver3D(np.array([k + 8]), np.array([j]), np.array([i + 6]), np.zeros(1, int), 1))
        out[backend] = (s, s.run(160))
    (sa, ra), (sb, rb) = out["numpy"], out["numba"]
    assert sb.backend == "numba" and sa._slip and sa.n_mech == 2 and sa.anisotropic == aniso
    for name in ("vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz"):
        a, b = getattr(sa, name), getattr(sb, name)
        assert np.abs(a).max() > 0
        assert np.abs(a - b).max() <= 1e-4 * np.abs(a).max(), name
    for name in ("sxx", "sxz", "syz"):
        a, b = sa.mem[name], sb.mem[name]
        assert np.abs(a - b).max() <= 1e-4 * np.abs(a).max(), name
    assert np.abs(ra.traces["p"] - rb.traces["p"]).max() <= 1e-4 * np.abs(ra.traces["p"]).max()


def test_correccion_de_interfaz_conserva_la_energia():
    """Las derivadas de cizalla corregidas siguen siendo adjuntas, <s, D_f v> = -<v, D_b s>, en las 3 componentes.

    Inclusión esférica de hueso en agua (contorno escalonado en los tres ejes),
    campos aleatorios nulos cerca de los bordes, float64 y backend NumPy.
    """
    n = 28
    grid = Grid3D(n, n, n, 1e-4)
    zz, yy, xx = np.mgrid[:n, :n, :n]
    bone = (zz - 13.3) ** 2 + (yy - 14.6) ** 2 + (xx - 13.8) ** 2 < 8.7**2
    rho, cp, cs = (np.where(bone, b, w) for b, w in zip(BONE, WATER))
    med = ElasticMedium3D(rho, cp, cs, np.zeros((0, *grid.shape)), np.zeros((0, *grid.shape)), np.zeros(0), F0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ElasticSolver3D(homogeneous_phantom(grid), F0, npml=3, medium=med, backend="numpy", dtype=np.float64)
    rng = np.random.default_rng(3)
    win = np.zeros(grid.shape)
    win[7:-7, 7:-7, 7:-7] = 1.0
    out = np.zeros(grid.shape)
    pairs = (("exy_a", "sxy_y"), ("exy_b", "sxy_x"), ("exz_a", "sxz_z"), ("exz_b", "sxz_x"),
             ("eyz_a", "syz_z"), ("eyz_b", "syz_y"))
    from wavesim.ultrasound.elastic3d import STRESS_DERIVS, VELOCITY_DERIVS

    for strain, momentum in pairs:
        assert strain in s._slip and momentum in s._slip
        v, sig = rng.standard_normal((2, *grid.shape)) * win
        _, axis, fwd = STRESS_DERIVS[strain]
        a = np.sum(sig * s._derivative(strain, v, out, axis, fwd))
        _, axis_b, fwd_b, _ = VELOCITY_DERIVS[momentum]
        b = np.sum(v * s._derivative(momentum, sig, out, axis_b, fwd_b))
        assert abs(a + b) < 1e-12 * abs(a), strain


def test_seleccion_de_backend_y_errores():
    grid = Grid3D(6, 6, 6, 1.5e-4)
    ph = homogeneous_phantom(grid)
    assert ElasticSolver3D(ph, F0, npml=0, backend="numpy").backend == "numpy"
    with pytest.raises(ValueError):
        ElasticSolver3D(ph, F0, backend="cuda")
    s = ElasticSolver3D(ph, F0, npml=0, backend="numpy")
    with pytest.raises(ValueError):
        ElasticSolver3D(ph, F0, npml=0, backend="numpy", dt=1.01 * s.dt / 0.9)
    with pytest.raises(ValueError):
        ElasticSolver3D(homogeneous_phantom(Grid3D(3, 6, 6, 1.5e-4)), F0)
    with pytest.raises(ValueError):
        ElasticSolver3D(ph, F0, anisotropic_bone=True)  # sin meta["bone_axis"]
