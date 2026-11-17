"""Tests rápidos del FDTD de Maxwell 3D (``em.fdtd3d``) y de la serie de Mie de la esfera."""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.core.grid3d import Grid3D
from wavesim.em import BroadbandMaxwell3DSolver, Maxwell3DSolver, MaxwellTMSolver, analytic, bessel
from wavesim.em.fdtd3d import averaged_sar, node_average, polarization_axis
from wavesim.materials import dielectric
from wavesim.materials.dispersive import tissue_debye_model
from wavesim.phantoms.limb3d import Phantom3D, extrude, homogeneous_phantom
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

try:
    import numba  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

LAYERS = [("skin", 2e-3), ("fat", 5e-3), ("muscle", 15e-3), ("cortical_bone", 6e-3)]


def _slab_phantom(dx, z_top=30e-3, margin=40e-3, n_lat=2):
    depth = z_top + sum(d for _, d in LAYERS) + margin
    g = Grid3D(n_lat, n_lat, int(round(depth / dx)), dx)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    z = z_top
    for t, d in LAYERS:
        lab[(g.z >= z - 1e-12) & (g.z < z + d - 1e-12)] = TISSUE_INDEX[t]
        z += d
    return Phantom3D(g, lab), z_top


def _cylinder_2d(dx=1e-3):
    g = Grid2D(50, 60, dx, x0=-25e-3, z0=0.0)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    X, Z = g.meshgrid()
    lab[np.hypot(X, Z - 30e-3) < 9e-3] = TISSUE_INDEX["muscle"]
    lab[np.hypot(X, Z - 30e-3) < 4e-3] = TISSUE_INDEX["cortical_bone"]
    return Phantom(g, lab)


def test_estabilidad_energia_vacio_cpml():
    """Un pulso de dipolo en vacío con CPML en las seis caras: la energía no crece y acaba absorbida."""
    g = Grid3D(30, 30, 30, 1e-3, -15e-3, -15e-3, -15e-3)
    s = Maxwell3DSolver(homogeneous_phantom(g, "air"), 10e9, npml=8, dtype=np.float64)
    s.set_dipole(0.0, 0.0, 0.0, "z")
    t = (np.arange(600) + 0.5) * s.dt
    t0, tau = 60 * s.dt, 15 * s.dt
    drive = -(t - t0) / tau * np.exp(-0.5 * ((t - t0) / tau) ** 2)
    energy = []
    for n in range(600):
        s.step(drive[n])
        energy.append(s.energy())
    energy = np.array(energy)
    # Tras el pulso sólo queda el campo electrostático residual del dipolo (~1e-7 del máximo): no crece
    after = energy[150:]
    assert np.all(np.diff(after) <= 1e-9 * energy.max())
    assert after[-1] <= after[100] * (1 + 1e-6)
    assert after[-1] < 1e-4 * energy.max()


@pytest.mark.parametrize("pol", ["x", "y"])
def test_velocidad_de_fase_y_atenuacion_onda_plana(pol):
    """Lámina de corriente en grasa homogénea: beta y alpha frente a Gabriel (20 puntos por longitud de onda)."""
    f0 = 2.45e9
    beta, alpha = dielectric.wave_numbers("fat", f0)
    dx = 2 * np.pi / beta / 20
    g = Grid3D(2, 2, 160, dx)
    s = Maxwell3DSolver(homogeneous_phantom(g, "fat"), f0, npml=(20, 0, 0), dtype=np.float64)
    s.set_current_sheet(z=25 * dx, polarization=pol)
    r = s.run()
    c = 0 if pol == "x" else 1
    col = r.e[c][:, 0, 0]
    sel = slice(35, 130)
    zz = g.z[sel]
    phase = np.unwrap(np.angle(col[sel]))
    b_fit = -np.polyfit(zz, phase, 1)[0]
    a_fit = -np.polyfit(zz, np.log(np.abs(col[sel])), 1)[0]
    assert abs(b_fit / beta - 1) < 0.01
    assert abs(a_fit / alpha - 1) < 0.03


def test_tfsf_sin_fugas_y_onda_unitaria():
    """En vacío el campo dispersado fuera de la caja TF/SF es redondeo y dentro la onda vale 1."""
    g = Grid3D(3, 3, 120, 1e-3)
    s = Maxwell3DSolver(homogeneous_phantom(g, "air"), 3e9, npml=(10, 0, 0), dtype=np.float64)
    s.set_plane_wave("x")
    r = s.run()
    col = r.e[0][:, 1, 1]
    k0, k1 = 13, g.nz - 14
    assert np.abs(col[10:k0 - 1]).max() < 1e-10
    assert np.abs(col[k1 + 2:g.nz - 10]).max() < 1e-10
    assert np.abs(np.abs(col[k0:k1 + 1]) - 1).max() < 1e-3
    assert np.abs(r.e[1:]).max() < 1e-10


@pytest.mark.parametrize("pol", ["x", "y"])
def test_losa_frente_a_matriz_de_transferencia(pol):
    """Losa piel/grasa/músculo/hueso a 915 MHz, onda plana TF/SF: |T| y |R| frente a la solución exacta."""
    f0, dx = 915e6, 1e-3
    p, z_top = _slab_phantom(dx)
    s = Maxwell3DSolver(p, f0, npml=(20, 0, 0), dtype=np.float64)
    s.set_plane_wave(pol)
    r = s.run(settle_time=30 / f0)
    c = 0 if pol == "x" else 1
    col = r.e[c][:, 1, 1]
    D = sum(d for _, d in LAYERS)
    kb = p.grid.index_of(0, 0, z_top + D + 10e-3)[0]
    # Por encima de la cara superior de la caja (k0 = 23) sólo hay campo dispersado: la onda reflejada
    ka = p.grid.index_of(0, 0, z_top - 8e-3)[0]
    t_fdtd = col[kb] / r.incident[kb]
    r_fdtd = col[ka] / r.incident[ka]
    t_ex = analytic.insertion_transmission(LAYERS, f0, outer="air")
    r_ex, _ = analytic.multilayer_rt([dielectric.complex_permittivity(k, f0) for k, _ in LAYERS],
                                     [d for _, d in LAYERS], f0)
    assert abs(20 * np.log10(abs(t_fdtd) / abs(t_ex))) < 0.2
    assert abs(np.degrees(np.angle(t_fdtd / t_ex))) < 3.0
    assert abs(abs(r_fdtd) - abs(r_ex)) < 0.02


def test_igual_al_2d_en_caso_extruido():
    """Con ny = 1, la lámina del 2D y el mismo dt, el 3D reproduce MaxwellTMSolver salvo redondeo."""
    p2 = _cylinder_2d()
    s2 = MaxwellTMSolver(p2, 2.45e9, npml=10, backend="numpy")
    s2.set_plane_wave_source()
    r2 = s2.run()
    s3 = Maxwell3DSolver(extrude(p2, ny=1), 2.45e9, npml=(10, 0, 10), dtype=np.float64, dt=s2.dt)
    s3.set_current_sheet(polarization="y")
    r3 = s3.run(exact_dft=False)
    ey = r3.e[1][:, 0, :]
    assert np.abs(ey - r2.phasor).max() < 1e-10 * np.abs(r2.phasor).max()
    assert np.abs(r3.e[[0, 2]]).max() == 0.0
    np.testing.assert_allclose(r3.sar[:, 0, :], r2.sar, rtol=1e-9, atol=1e-12 * r2.sar.max())


def test_3d_invariante_en_y_frente_al_2d():
    """Un 3D de verdad (ny = 8, CPML en y, dt 3D, float32) con E ∥ y sobre un fantoma extruido: ≤ 1 % del 2D."""
    p2 = _cylinder_2d()
    s2 = MaxwellTMSolver(p2, 2.45e9, npml=10)
    s2.set_plane_wave_source()
    r2 = s2.run()
    s3 = Maxwell3DSolver(extrude(p2, ny=8), 2.45e9, npml=(10, 2, 10), dtype=np.float32)
    s3.set_current_sheet(polarization="y")
    r3 = s3.run(exact_dft=False)
    ey = r3.e[1][:, 4, :]
    inner = (slice(12, -12), slice(12, -12))
    err = np.abs(ey - r2.phasor)[inner].max() / np.abs(r2.phasor)[inner].max()
    assert err < 0.01


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba no instalado")
def test_numpy_y_numba_iguales():
    """Mismo campo con los dos backends (float64): TF/SF, CPML en las seis caras y un cuerpo con pérdidas."""
    g = Grid3D(22, 20, 30, 1e-3, -11e-3, -10e-3, 0.0)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    zz, yy, xx = np.meshgrid(g.z - 15e-3, g.y, g.x, indexing="ij")
    lab[xx ** 2 + yy ** 2 + zz ** 2 < (4e-3) ** 2] = TISSUE_INDEX["muscle"]
    p = Phantom3D(g, lab)
    out = {}
    for backend in ("numpy", "numba"):
        s = Maxwell3DSolver(p, 2.45e9, npml=5, dtype=np.float64, backend=backend)
        s.set_plane_wave("x")
        for n in range(150):
            s.step(np.sin(2 * np.pi * 2.45e9 * (n + 0.5) * s.dt))
        out[backend] = np.stack([s.f[c] for c in ("ex", "ey", "ez", "hx", "hy", "hz")])
    scale = np.abs(out["numpy"]).max(axis=(1, 2, 3), keepdims=True)
    assert np.abs(out["numpy"] - out["numba"]).max() / scale.max() < 1e-12
    # Banda ancha con polos de Debye
    model = tissue_debye_model((1e9, 4e9), n_poles=3)
    for backend in ("numpy", "numba"):
        s = BroadbandMaxwell3DSolver(p, (1e9, 4e9), model=model, npml=5, dtype=np.float64, backend=backend)
        s.set_plane_wave("y")
        for n in range(150):
            s.step(np.sin(2 * np.pi * 2.45e9 * (n + 0.5) * s.dt))
        out[backend] = np.stack([s.f[c] for c in ("ex", "ey", "ez")] + [s.jpol["ey"][0]])
    for q in range(4):
        ref = np.abs(out["numpy"][q]).max()
        assert np.abs(out["numpy"][q] - out["numba"][q]).max() < 1e-11 * ref


def test_sar_no_negativo_y_cero_en_el_aire():
    p2 = _cylinder_2d(dx=1.5e-3)
    p3 = extrude(p2, ny=8)
    s = Maxwell3DSolver(p3, 2.45e9, npml=(8, 0, 8))
    s.set_plane_wave("x")
    r = s.run(n_periods_steady=2)
    sar = r.sar
    assert np.all(sar >= 0)
    assert np.all(sar[p3.labels == TISSUE_INDEX["air"]] == 0)
    assert sar[p3.labels == TISSUE_INDEX["muscle"]].max() > 0
    avg = r.averaged_sar(mass=1e-3)
    assert np.all(avg >= 0) and np.all(avg[~r.tissue_mask()] == 0)
    assert r.absorbed_power() > 0


def test_banda_ancha_frente_a_banda_estrecha():
    """Un pulso de 1-4 GHz da la transmisión de una losa a varias frecuencias como varias ondas continuas."""
    dx = 1e-3
    p, z_top = _slab_phantom(dx)
    band = (1e9, 4e9)
    model = tissue_debye_model(band, n_poles=4)
    freqs = np.array([1.5e9, 2.45e9, 3.5e9])
    sb = BroadbandMaxwell3DSolver(p, band, model=model, npml=(20, 0, 0), dtype=np.float64)
    sb.set_plane_wave("x")
    rb = sb.run(freqs=freqs)
    kb = p.grid.index_of(0, 0, z_top + 38e-3)[0]
    for m, f in enumerate(freqs):
        sn = Maxwell3DSolver(p, f, npml=(20, 0, 0), dtype=np.float64,
                             permittivity=lambda k, ff: complex(model.tissues[k].complex_permittivity(ff)))
        sn.set_plane_wave("x")
        rn = sn.run(settle_time=30 / f)
        tb = rb.e[m, 0, kb, 1, 1] / rb.incident[m, kb]
        tn = rn.e[0, kb, 1, 1] / rn.incident[kb]
        assert abs(tb / tn - 1) < 0.01


def test_polarizacion_respecto_al_hueso():
    assert polarization_axis("parallel", {"bone_axis": "x"}) == "x"
    assert polarization_axis("perpendicular", {"bone_axis": "x"}) == "y"
    assert polarization_axis("perpendicular", {"bone_axis": "y"}) == "x"
    with pytest.raises(ValueError):
        polarization_axis("parallel", {"bone_axis": None})


def test_promedio_en_nodos_y_sar_promediado():
    a = np.arange(24.0).reshape(2, 3, 4)
    m = node_average(a, 2)
    assert m[0, 0, 0] == 0.5 and m[0, 0, -1] == a[0, 0, -1]
    sar = np.ones((9, 9, 9))
    rho = np.full(sar.shape, 1000.0)
    tissue = np.ones(sar.shape, bool)
    np.testing.assert_allclose(averaged_sar(sar, rho, 1e-3, tissue, mass=27e-6), 1.0)


# --------------------------------------------------------------------------
# Serie de Mie de la esfera
# --------------------------------------------------------------------------
def test_bessel_esfericas():
    x = np.array([0.1, 1.0, 5.0, 20.0])
    j = bessel.spherical_jn_orders(30, x)
    y = bessel.spherical_yn_orders(30, x)
    # (la forma cerrada pierde dígitos por cancelación en x = 0.1: tolerancia absoluta)
    np.testing.assert_allclose(j[2], 3 / x ** 3 * np.sin(x) - 3 / x ** 2 * np.cos(x) - np.sin(x) / x, atol=1e-13)
    # Wronskiano j_n y_{n-1} - j_{n-1} y_n = 1/x²
    np.testing.assert_allclose((j[1:] * y[:-1] - j[:-1] * y[1:]) * x ** 2, 1.0, rtol=1e-12)
    z = np.array([2 + 1j, 7.5 - 0.9j])
    term = z ** 12 / np.prod(np.arange(1.0, 26.0, 2.0))
    series = term.copy()
    for k in range(1, 80):
        term = term * (-z * z / 2) / (k * (2 * 12 + 2 * k + 1))
        series += term
    np.testing.assert_allclose(bessel.spherical_jn_orders(12, z)[12], series, rtol=1e-12)


def test_mie_valores_conocidos():
    """m = 1.55, x = 5.213 (ejemplo de BHMIE): Q_ext = Q_sca = 3.10543; límite de Rayleigh con pérdidas."""
    lam, a = 0.6328e-6, 0.525e-6
    q_ext, q_sca, q_abs = analytic.mie_sphere_efficiencies(1.55 ** 2, a, analytic.C0 / lam)
    assert abs(q_ext - 3.10543) < 1e-5 and abs(q_sca - 3.10543) < 1e-5 and abs(q_abs) < 1e-12
    m = 1.5 + 0.1j  # convenio de B&H (Im > 0); en wavesim eps = conj(m)²
    x = 0.01
    q = analytic.mie_sphere_efficiencies(np.conj(m) ** 2, x * lam / (2 * np.pi), analytic.C0 / lam)
    assert abs(q[2] / (4 * x * ((m ** 2 - 1) / (m ** 2 + 2)).imag) - 1) < 1e-3


def test_mie_campo_continuidad_y_absorcion():
    """Campo de Mie: continuidad en la superficie y absorción por integral de volumen igual a la serie."""
    f, a = 2.45e9, 10e-3
    eps = complex(dielectric.complex_permittivity("muscle", f))
    rng = np.random.default_rng(1)
    th, ph = rng.uniform(0.1, 3.0, 30), rng.uniform(0, 2 * np.pi, 30)
    rhat = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    e_in = analytic.mie_sphere_field(*(a * (1 - 1e-9) * rhat), a, eps, f)
    e_out = analytic.mie_sphere_field(*(a * (1 + 1e-9) * rhat), a, eps, f)
    n_in, n_out = np.sum(e_in * rhat, 0), np.sum(e_out * rhat, 0)
    t_in, t_out = e_in - n_in * rhat, e_out - n_out * rhat
    assert np.abs(t_in - t_out).max() < 1e-6 * np.abs(t_out).max()
    assert np.abs(eps * n_in - n_out).max() < 1e-6 * np.abs(n_out).max()
    xr, wr = np.polynomial.legendre.leggauss(40)
    r, wr = 0.5 * a * (xr + 1), 0.5 * a * wr
    mu, wt = np.polynomial.legendre.leggauss(40)
    phis = np.arange(24) * 2 * np.pi / 24
    R, MU, PH = np.meshgrid(r, mu, phis, indexing="ij")
    S = np.sqrt(1 - MU ** 2)
    e2 = np.sum(np.abs(analytic.mie_sphere_field(R * S * np.cos(PH), R * S * np.sin(PH), R * MU, a, eps, f)) ** 2, 0)
    integral = np.sum(wr[:, None, None] * R ** 2 * wt[None, :, None] * (2 * np.pi / 24) * e2)
    k0 = 2 * np.pi * f / analytic.C0
    q_vol = k0 * (-eps.imag) * integral / (np.pi * a ** 2)
    assert abs(q_vol / analytic.mie_sphere_efficiencies(eps, a, f)[2] - 1) < 1e-6
