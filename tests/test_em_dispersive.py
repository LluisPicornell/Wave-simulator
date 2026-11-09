"""Maxwell de banda ancha (Debye multipolo + ADE) y validación analítica del campo (capas planas y Mie 2D).

Las versiones completas (toda la banda 0.1-10 GHz, convergencia con dx) están
en ``benchmarks/em_validation.py``; aquí van configuraciones reducidas.
"""

import warnings

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.em import analytic, bessel, spectrum, subpixel
from wavesim.em.fdtd_dispersive import BroadbandMaxwellTMSolver, wideband_pulse
from wavesim.em.fdtd_tm import MaxwellTMSolver
from wavesim.materials import dielectric, dispersive, porous
from wavesim.phantoms import cross_section_limb, longitudinal_limb
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

C0 = dielectric.C0
NPML = 20


def _column(dx, labels_z, z0=0.0):
    """Malla 1D (4 columnas) con las etiquetas ``labels_z`` a lo largo de z."""
    g = Grid2D(4, len(labels_z), dx, x0=-2 * dx, z0=z0)
    return Phantom(g, np.repeat(np.asarray(labels_z, np.int16)[:, None], 4, axis=1))


# --------------------------------------------------------------------------
# Ajuste de Debye
# --------------------------------------------------------------------------
def test_nnls_respeta_las_restricciones():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 6))
    x_true = np.array([1.0, 0.0, 2.5, 0.0, 0.3, 4.0])
    x = dispersive.nnls(a, a @ x_true)
    np.testing.assert_allclose(x, x_true, atol=1e-10)
    x = dispersive.nnls(a, -a @ x_true)  # la solución sin restricciones sería negativa
    assert np.all(x >= 0)


def test_ajuste_debye_reproduce_gabriel_en_la_banda():
    model = dispersive.tissue_debye_model((1e8, 1e10))
    assert model.n_poles <= 8
    for k, (err_eps, err_sigma) in model.errors.items():
        assert err_eps < 0.02 and err_sigma < 0.02, k
        m = model.tissues[k]
        assert m.eps_inf >= 1.0 and m.sigma >= 0.0 and np.all(m.delta_eps >= 0.0)
    # conductividad iónica: a baja frecuencia sigma_eff -> sigma del modelo (sin d_eps)
    f = np.geomspace(1e8, 1e10, 200)
    eps_g = dielectric.complex_permittivity("muscle", f)
    eps_m = model.tissues["muscle"].complex_permittivity(f)
    np.testing.assert_allclose(eps_m.real, eps_g.real, rtol=0.02)
    np.testing.assert_allclose(eps_m.imag * f, eps_g.imag * f, rtol=0.02)


def test_mapas_debye_de_celdas_porosas_siguen_a_maxwell_garnett():
    p = cross_section_limb("pierna", dx=1.0e-3, seed=3, realism="realista")
    assert p.porosity is not None
    model = dispersive.tissue_debye_model((1e8, 1e10))
    med = dispersive.debye_maps(p, model)
    for f in (1.5e8, 9.15e8, 2.45e9, 8e9):
        eps_c = {k: complex(dielectric.complex_permittivity(k, f)) for k in p.property_tissues()}
        ref = p.property_map(lambda k, e=eps_c: e[k], dtype=complex, physics="dielectric")
        got = med.complex_permittivity(f)
        assert np.max(np.abs(got.real / ref.real - 1)) < 0.02
        assert np.max(np.abs(got.imag / ref.imag - 1)) < 0.02
    porous_cells = np.isin(p.labels, [TISSUE_INDEX[k] for k in porous.POROUS_TISSUES]) & np.isfinite(p.porosity)
    assert porous_cells.sum() > 100


# --------------------------------------------------------------------------
# Solver de banda ancha
# --------------------------------------------------------------------------
def test_pulso_sin_componente_continua_y_con_la_banda_dentro():
    dt = 1e-12
    t = np.arange(20000) * dt
    p = wideband_pulse(t, 1e8, 1e10)
    spec = np.abs(np.exp(-2j * np.pi * np.outer([0.0, 1e8, 5.05e9, 1e10, 2e10], t)) @ p) * dt
    assert spec[0] < 1e-6 * spec[2]
    assert spec[1] > 1e-2 * spec[2] and spec[3] > 0.05 * spec[2]
    assert spec[4] < 1e-6 * spec[2]


def test_sin_polos_coincide_con_banda_estrecha():
    """Con ``eps`` y ``sigma`` constantes el ADE es el FDTD de siempre: misma onda, DFT frente a régimen permanente."""
    dx, f0 = 0.5e-3, 2e9
    nz = 2 * NPML + 120
    lab = np.full(nz, TISSUE_INDEX["air"])
    lab[NPML + 30:] = TISSUE_INDEX["muscle"]
    p = _column(dx, lab)
    eps_r, sigma = 30.0, 0.8
    shape = p.grid.shape
    air = p.labels == TISSUE_INDEX["air"]
    medium = dispersive.DebyeMedium(np.where(air, 1.0, eps_r), np.where(air, 0.0, sigma), np.zeros((0,) + shape),
                                    np.zeros(0))
    z_src = p.grid.z0 + (NPML + 3) * dx
    bb = BroadbandMaxwellTMSolver(p, (5e8, 4e9), medium=medium)
    bb.set_plane_wave_source(z=z_src)
    r_bb = bb.run(freqs=[f0])
    nb = MaxwellTMSolver(p, f0, eps_map=medium.complex_permittivity(f0))
    nb.set_plane_wave_source(z=z_src)
    r_nb = nb.run(n_periods_steady=8)
    # La fuente de banda estrecha toma el seno en t_{n+1} y no en t_{n+1/2}: adelanta w dt / 2
    shift = np.exp(-1j * np.pi * f0 * nb.dt)
    col = slice(NPML + 5, nz - NPML - 5)
    # Queda ~0.3 % junto a la fuente: las CPML no son iguales (término CFS con f_ref = f_min frente a f0)
    np.testing.assert_allclose(r_bb.phasor[0][col, 2], r_nb.phasor[col, 2] * shift, atol=5e-3)
    assert np.median(np.abs(r_bb.phasor[0][col, 2] - r_nb.phasor[col, 2] * shift)) < 2e-3


@pytest.mark.parametrize("tissue", ["muscle", "fat"])
def test_banda_ancha_beta_y_alpha_frente_a_cole_cole(tissue):
    """Una sola simulación da la constante de fase y la atenuación en toda la banda (< 2 % y < 5 %)."""
    band = (2e8, 5e9)
    n_hi = abs(np.sqrt(complex(dielectric.complex_permittivity(tissue, band[1]))))
    dx = C0 / (band[1] * n_hi * 15)
    nz = 2 * NPML + 10 + int(0.04 / dx)
    p = _column(dx, np.full(nz, TISSUE_INDEX[tissue]))
    g = p.grid
    s = BroadbandMaxwellTMSolver(p, band)
    z_src = g.z0 + (NPML + 3) * dx
    s.set_plane_wave_source(z=z_src)
    mon = np.zeros(g.shape, bool)
    mon[:, 2] = True
    freqs = np.geomspace(*band, 9)
    r = s.run(freqs, monitor=mon)
    z = g.z[r.points[0]]
    sel = (z > z_src + 8 * dx) & (z < g.z0 + g.depth - (NPML + 5) * dx)
    beta, alpha = dielectric.wave_numbers(tissue, freqs)
    for k in range(freqs.size):
        col = r.phasor[k][sel]
        a_fit = -np.polyfit(z[sel], np.log(np.abs(col)), 1)[0]
        b_fit = abs(np.polyfit(z[sel], np.unwrap(np.angle(col)), 1)[0])
        assert b_fit == pytest.approx(beta[k], rel=0.02)
        assert a_fit == pytest.approx(alpha[k], rel=0.05)


def test_fantoma_poroso_banda_ancha_coincide_con_banda_estrecha():
    """Antebrazo realista (cortical porosa, fascias): el fasor del pulso a f0 es el de la onda continua."""
    p = cross_section_limb("antebrazo", dx=0.9e-3, seed=1, realism="realista", margin=30e-3)
    assert p.porosity is not None
    f0 = 2.45e9
    nb = MaxwellTMSolver(p, f0)
    nb.set_plane_wave_source()
    r_nb = nb.run()
    # Desfase de medio paso de la fuente de banda estrecha y su normalización con la impedancia
    # real: en el agua emite sqrt(eps'/eps) V/m en vez de 1
    eps_src = complex(dielectric.complex_permittivity("water", f0))
    gain = np.sqrt(eps_src.real / eps_src) * np.exp(1j * np.pi * f0 * nb.dt)
    e_nb = r_nb.phasor / gain
    bb = BroadbandMaxwellTMSolver(p, (1.5e9, 3e9))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # la fuente está en el agua: no debe avisar
        bb.set_plane_wave_source()
    r = bb.run(freqs=[f0])
    inner = (slice(NPML + 5, -NPML - 5), slice(NPML + 5, -NPML - 5))
    diff = np.abs(r.phasor[0][inner] - e_nb[inner]) / np.abs(e_nb[inner]).max()
    assert np.median(diff) < 0.005 and diff.max() < 0.02
    # SAR con la conductividad del ajuste de Debye frente a la de Gabriel (< 1 % de diferencia)
    sar_nb = r_nb.sar[inner] / abs(gain) ** 2
    assert np.abs(r.at(0).sar[inner] - sar_nb).max() < 0.03 * sar_nb.max()


def test_fuente_que_cruza_tejidos_avisa():
    p = cross_section_limb("antebrazo", dx=0.9e-3, seed=1, margin=2e-3)
    bb = BroadbandMaxwellTMSolver(p, (1.5e9, 3e9))
    with pytest.warns(UserWarning, match="cruza medios distintos"):
        bb.set_plane_wave_source(z=0.0)


# --------------------------------------------------------------------------
# Losa multicapa a incidencia normal
# --------------------------------------------------------------------------
SLAB = [("skin", 2e-3), ("fat", 5e-3), ("muscle", 15e-3), ("cortical_bone", 6e-3)]


def _slab_phantoms(dx):
    gap = 20
    cells = [int(round(d / dx)) for _, d in SLAB]
    lab = [TISSUE_INDEX["air"]] * (NPML + gap)
    for (k, _), n in zip(SLAB, cells):
        lab += [TISSUE_INDEX[k]] * n
    j_obs = len(lab) + gap // 2
    lab += [TISSUE_INDEX["air"]] * (gap + NPML + 10)
    slab = _column(dx, lab)
    ref = _column(dx, [TISSUE_INDEX["air"]] * len(lab))
    return slab, ref, j_obs, [(k, n * dx) for (k, _), n in zip(SLAB, cells)]


@pytest.mark.parametrize("f0", [128e6, 915e6, 2.45e9])
def test_losa_multicapa_banda_estrecha_frente_a_matriz_de_transferencia(f0):
    dx = 0.5e-3
    slab, ref, j_obs, layers = _slab_phantoms(dx)
    z_src = slab.grid.z0 + (NPML + 3) * dx
    out = []
    for p in (slab, ref):
        s = MaxwellTMSolver(p, f0)
        s.set_plane_wave_source(z=z_src)
        out.append(s.run().phasor[j_obs, 2])
    t_fdtd = out[0] / out[1]
    t_exact = complex(analytic.insertion_transmission(layers, f0))
    assert abs(20 * np.log10(abs(t_fdtd / t_exact))) < 0.5
    assert abs(np.degrees(np.angle(t_fdtd / t_exact))) < 3.0


def test_losa_multicapa_banda_ancha_frente_a_matriz_de_transferencia():
    band, dx = (1e8, 3e9), 0.5e-3
    slab, ref, j_obs, layers = _slab_phantoms(dx)
    z_src = slab.grid.z0 + (NPML + 3) * dx
    mon = np.zeros(slab.grid.shape, bool)
    mon[j_obs, 2] = True
    freqs = np.geomspace(*band, 15)
    model = dispersive.tissue_debye_model(band)
    out = []
    for p in (slab, ref):
        s = BroadbandMaxwellTMSolver(p, band, model=model)
        s.set_plane_wave_source(z=z_src)
        out.append(s.run(freqs, monitor=mon).phasor[:, 0])
    ratio = (out[0] / out[1]) / analytic.insertion_transmission(layers, freqs)
    assert np.max(np.abs(20 * np.log10(np.abs(ratio)))) < 0.5
    assert np.max(np.abs(np.degrees(np.angle(ratio)))) < 3.0


def test_matriz_de_transferencia_general_reproduce_spectrum():
    layers = spectrum.column_layers(longitudinal_limb("antebrazo", dx=0.1e-3, width=10e-3))
    for f in (64e6, 915e6, 2.45e9):
        for outer in ("water", "air"):
            e_out = dielectric.complex_permittivity(outer, f)
            _, t = analytic.multilayer_rt(analytic.tissue_layers_eps(layers, f), [d for _, d in layers], f,
                                          e_out, e_out)
            assert abs(t) ** 2 == pytest.approx(spectrum._coherent_transmission(layers, f, outer), rel=1e-10)
    # una interfaz: r de Fresnel
    eps = complex(dielectric.complex_permittivity("muscle", 1e9))
    r, _ = analytic.multilayer_rt([], [], 1e9, 1.0, eps)
    assert complex(r) == pytest.approx((1 - np.sqrt(eps)) / (1 + np.sqrt(eps)), rel=1e-12)


# --------------------------------------------------------------------------
# Funciones de Bessel y Mie 2D
# --------------------------------------------------------------------------
# Abramowitz y Stegun, Handbook of Mathematical Functions (1964), tablas 9.1 y 9.2
# (J_n, Y_n de argumento real) y 9.8 (I_0, I_1: J_n(i x) = i^n I_n(x)).
TABLE = [
    ("J", 0, 1.0, 0.7651976866), ("J", 1, 1.0, 0.4400505857), ("J", 2, 1.0, 0.1149034849),
    ("Y", 0, 1.0, 0.0882569642), ("Y", 1, 1.0, -0.7812128213), ("Y", 2, 1.0, -1.6506826068),
    ("J", 0, 5.0, -0.1775967713), ("Y", 0, 5.0, -0.3085176252),
    ("J", 0, 10.0, -0.2459357645), ("J", 1, 10.0, 0.0434727462), ("J", 5, 10.0, -0.2340615282),
    ("Y", 0, 10.0, 0.0556711673), ("Y", 1, 10.0, 0.2490154242), ("Y", 5, 10.0, 0.1354030477),
]


@pytest.mark.parametrize("kind,n,x,value", TABLE)
def test_bessel_frente_a_valores_tabulados(kind, n, x, value):
    got = bessel.besselj(n, x) if kind == "J" else bessel.bessely(n, x)
    assert complex(got).real == pytest.approx(value, abs=2e-10)


def test_bessel_argumento_complejo_y_wronskiano():
    assert complex(bessel.besselj(0, 1j)) == pytest.approx(1.2660658778, abs=1e-9)  # I_0(1)
    assert complex(bessel.besselj(1, 1j)) == pytest.approx(0.5651591040j, abs=1e-9)  # i I_1(1)
    x = np.geomspace(0.01, 150, 300)
    j, y = bessel.besselj_orders(41, x).real, bessel.bessely_orders(41, x)
    for n in (0, 3, 15, 40):
        w = j[n + 1] * y[n] - j[n] * y[n + 1]  # = 2 / (pi x)
        np.testing.assert_allclose(w * np.pi * x / 2, 1.0, rtol=1e-10)
    # serie de potencias para z complejo y orden alto (la parte que el trapecio no resuelve)
    from math import factorial

    z = np.array([0.5 + 0.2j, 7.5 - 0.9j])
    ser = sum((-1) ** m / (factorial(m) * factorial(m + 12)) * (z / 2) ** (2 * m + 12) for m in range(60))
    np.testing.assert_allclose(bessel.besselj(12, z), ser, rtol=1e-11)


def test_mie_es_continua_en_la_superficie_y_trivial_sin_contraste():
    f, a = 2.45e9, 0.02
    eps = complex(dielectric.complex_permittivity("muscle", f))
    th = np.linspace(0, 2 * np.pi, 40)
    x, z = a * np.sin(th), a * np.cos(th)
    e_in = analytic.mie_cylinder_tm(x * (1 - 1e-9), z * (1 - 1e-9), a, eps, f)
    e_out = analytic.mie_cylinder_tm(x * (1 + 1e-9), z * (1 + 1e-9), a, eps, f)
    np.testing.assert_allclose(e_in, e_out, atol=1e-7)
    xx, zz = np.meshgrid(np.linspace(-0.04, 0.04, 17), np.linspace(-0.04, 0.04, 17))
    e = analytic.mie_cylinder_tm(xx, zz, a, 1.0 + 0j, f)
    np.testing.assert_allclose(e, np.exp(-2j * np.pi * f / C0 * zz), atol=1e-12)


def _cylinder_error(tissue, f, a, ppw, smooth, factor=9):
    """Error L2 relativo del FDTD de banda estrecha frente a Mie (dentro, fuera)."""
    eps = complex(dielectric.complex_permittivity(tissue, f))
    dx = C0 / f / abs(np.sqrt(eps)) / ppw
    half = a + max(0.25 * C0 / f, 0.5 * a)
    n_half = int(np.ceil(half / dx))
    n = 2 * n_half + 1 + 2 * NPML
    g = Grid2D(n, n + 6, dx, x0=-(n_half + NPML) * dx, z0=-(n_half + NPML + 6) * dx)
    xx, zz = g.meshgrid()
    lab = np.where(np.hypot(xx, zz) <= a, TISSUE_INDEX[tissue], TISSUE_INDEX["air"]).astype(np.int16)
    eps_map = None
    if smooth:
        gf = subpixel.fine_grid(g, factor)
        xf, zf = gf.meshgrid()
        fine = Phantom(gf, np.where(np.hypot(xf, zf) <= a, TISSUE_INDEX[tissue], TISSUE_INDEX["air"]).astype(np.int16))
        eps_map = subpixel.smoothed_permittivity(fine, factor, f)
    z_src = g.z0 + (NPML + 2) * dx
    settle = 40 / f + 3.0 * 2 * g.depth * abs(np.sqrt(eps)) / C0
    s = MaxwellTMSolver(Phantom(g, lab), f, eps_map=eps_map)
    s.set_plane_wave_source(z=z_src)
    e = s.run(n_periods_steady=8, settle_time=settle).phasor
    # onda incidente: la misma malla sin cilindro es uniforme en x, basta una columna
    ref = MaxwellTMSolver(_column(dx, [TISSUE_INDEX["air"]] * g.nz, g.z0), f)
    ref.set_plane_wave_source(z=z_src)
    e = e / ref.run(n_periods_steady=8, settle_time=settle).phasor[g.index_of(0.0, 0.0)[0], 2]
    box = (np.abs(xx) <= half) & (np.abs(zz) <= half)
    mie = analytic.mie_cylinder_tm(xx[box], zz[box], a, eps, f)
    inside = np.hypot(xx, zz)[box] <= a
    err = [np.linalg.norm(e[box][m] - mie[m]) / np.linalg.norm(mie[m]) for m in (inside, ~inside)]
    return err[0], err[1]


def test_cilindro_de_hueso_frente_a_mie():
    err_in, err_out = _cylinder_error("cortical_bone", 2.45e9, 0.015, ppw=20, smooth=False)
    assert err_in < 0.05 and err_out < 0.03


def test_promedio_subcelda_reduce_el_error_de_escalera():
    """El error de escalera depende de cómo cae la circunferencia en la malla: se promedia en dos radios.

    Barrido completo (5-6 radios, 10-40 puntos por longitud de onda) en ``benchmarks/em_validation.py``.
    """
    plain, smooth = [], []
    for a in (0.012, 0.018):
        plain.append(_cylinder_error("muscle", 2.45e9, a, ppw=10, smooth=False)[0])
        smooth.append(_cylinder_error("muscle", 2.45e9, a, ppw=10, smooth=True)[0])
    assert np.mean(smooth) < 0.75 * np.mean(plain)
    assert max(smooth) < 0.09
