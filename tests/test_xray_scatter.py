"""Radiación dispersa: secciones eficaces por proceso y Monte Carlo de fotones X."""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.em import xray
from wavesim.em import xray_scatter as xs
from wavesim.em.fanbeam import FanGeometry, fan_project
from wavesim.materials import xray as xd
from wavesim.materials import xray_processes as xpr
from wavesim.phantoms import TISSUE_INDEX, Phantom

try:
    import numba  # noqa: F401
    BACKENDS = ["numpy", "numba"]
except ImportError:  # pragma: no cover
    BACKENDS = ["numpy"]


def cylinder(radius=0.09, tissue="water", half=0.1, dx=0.004):
    g = Grid2D.from_size(2 * half, 2 * half, dx, x0=-half, z0=-half)
    X, Z = g.meshgrid()
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[np.hypot(X, Z) <= radius] = TISSUE_INDEX[tissue]
    return Phantom(g, lab), g


# ----------------------------------------------------------------------------- secciones eficaces
def test_procesos_suman_el_total_y_son_positivos():
    e = np.array([10.0, 20.0, 30.0, 60.0, 100.0, 150.0, 200.0])
    for key in ("water", "muscle", "fat", "cortical_bone", "trabecular_bone", "air", "blood"):
        p = xpr.process_attenuation(key, e)
        total = xd.linear_attenuation(key, e)
        np.testing.assert_allclose(p["total"], total)
        np.testing.assert_allclose(p["photoelectric"] + p["incoherent"] + p["coherent"], total, rtol=1e-12 + 5e-3)
        assert all(np.all(p[n] > 0) for n in ("photoelectric", "incoherent", "coherent"))
        # Sin suelo: el modelo de Compton + Rayleigh no puede superar la atenuación total tabulada
        assert np.all(p["incoherent"] + p["coherent"] < total)


def test_limites_fisicos_del_factor_de_forma_y_de_la_funcion_s():
    for z in (1, 8, 20):
        assert xpr.form_factor(0.0, z) == pytest.approx(z, rel=1e-6)
        assert xpr.incoherent_function(0.0, z) == pytest.approx(0.0, abs=1e-6)
        assert xpr.form_factor(50.0, z) == pytest.approx(0.0, abs=1e-3)
        assert xpr.incoherent_function(50.0, z) == pytest.approx(z, rel=1e-3)


def test_compton_libre_es_klein_nishina():
    # Hidrógeno a 100 keV y 1 MeV: Klein-Nishina por electrón (0.4927 b y 0.2112 b) casi sin ligadura
    _, inc = xpr._element_cross_sections(np.array([100.0, 1000.0]), 1)
    np.testing.assert_allclose(inc * 1e24, [0.4923, 0.2112], rtol=3e-3)


def test_fotoelectrico_domina_a_baja_energia_y_se_apaga_a_alta():
    lo = xpr.process_attenuation("water", 20.0)
    hi = xpr.process_attenuation("water", 200.0)
    assert lo["photoelectric"][0] > 2 * lo["incoherent"][0]
    assert hi["photoelectric"][0] < 0.01 * hi["total"][0]
    assert hi["coherent"][0] < 0.05 * hi["total"][0]
    bone = xpr.process_attenuation("cortical_bone", 30.0)
    assert bone["photoelectric"][0] > 2 * xpr.process_attenuation("water", 30.0)["photoelectric"][0]


def test_cdf_de_rayleigh_valida():
    cdf = xpr.rayleigh_cdf("water", 60.0)
    assert cdf[-1] == pytest.approx(1.0) and np.all(np.diff(cdf) >= 0)
    # Dispersión muy hacia delante a 60 keV, pero menos a 20 keV
    c = np.linspace(-1, 1, cdf.size)
    mean = lambda e: float(np.sum(c * np.diff(xpr.rayleigh_cdf("water", e), prepend=0.0)))  # noqa: E731
    assert mean(60.0) > mean(20.0) > 0.5


# ----------------------------------------------------------------------------- Monte Carlo
@pytest.mark.parametrize("backend", BACKENDS)
def test_primario_mc_igual_a_beer_lambert_y_balance(backend):
    p, g = cylinder()
    sp = xray.tube_spectrum(100.0, 3.0)
    geom = FanGeometry.covering(g, 0.6, 1.1, True)
    n = 400_000 if backend == "numba" else 200_000
    r = xs.simulate_scatter(p, 0.0, sp, geom, n_photons=n, slice_thickness=0.01, backend=backend, seed=2)
    bal = r.balance()
    assert abs(bal["photons"]) < 1e-12 and abs(bal["energy"]) < 1e-9
    # Beer-Lambert exacto sobre el mismo mapa de vóxeles (agua) y el aire de la caja
    lw = fan_project((p.labels == TISSUE_INDEX["water"]).astype(float), g, geom, [0.0], step=0.0005)[0, 0]
    lbox = fan_project(np.ones(g.shape), g, geom, [0.0], step=0.0005)[0, 0]
    mu_w, mu_a = (xd.linear_attenuation(k, sp.energies_kev) for k in ("water", "air"))
    expo = np.array([np.sum(sp.weights * np.exp(-mu_w * a - mu_a * (b - a))) for a, b in zip(lw, lbox)])
    c = geom.n_det // 2
    for sl in (slice(c - 10, c + 11), slice(c + 22, c + 42)):
        obs, exp = r.prim[sl].sum(), expo[sl].sum() * r.open_beam
        assert abs(obs - exp) < 4.0 * np.sqrt(exp) + 0.01 * exp  # error estadístico (+1 % de geometría de vóxeles)
    assert r.scat.sum() > 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_distribucion_angular_y_energia_de_compton_y_rayleigh(backend):
    p, g = cylinder(radius=0.05)
    energy = 60.0
    sp = xray.monoenergetic(energy)
    prep = xs.prepare(p, sp)
    prep.scale = prep.scale * 0.03  # medio muy tenue: casi sólo una interacción por fotón
    geom = FanGeometry.covering(g, 0.6, 1.1, True)
    n = 1_500_000 if backend == "numba" else 600_000
    r = xs.simulate_scatter(p, 0.0, sp, geom, n_photons=n, backend=backend, seed=5, prepared=prep)
    c = r.counters
    assert c["n_compton"] > 4000 and c["n_rayleigh"] > 200
    # Valores esperados con las mismas S y F que el muestreo (agua, 60 keV), por integración directa
    cosg = np.linspace(-1, 1, 20001)
    k = energy / xpr.MEC2_KEV
    eps = 1 / (1 + k * (1 - cosg))
    xq = energy / xpr.HC_KEV_A * np.sqrt(0.5 * (1 - cosg))
    kn = eps**2 * (eps + 1 / eps - (1 - cosg**2)) * np.interp(xq, prep.tables.x_grid, prep.tables.s_over_z[0])
    mean_cos = np.sum(cosg * kn) / np.sum(kn)
    mean_loss = energy * np.sum((1 - eps) * kn) / np.sum(kn)
    sigma_cos = np.sqrt(np.sum(cosg**2 * kn) / np.sum(kn) - mean_cos**2) / np.sqrt(c["n_compton"])
    assert c["sum_cos_compton"] / c["n_compton"] == pytest.approx(mean_cos, abs=4 * sigma_cos)
    assert c["e_compton_electron"] / c["n_compton"] == pytest.approx(mean_loss, rel=0.01)
    cdf = prep.tables.ray_cdf[0, int(round((energy - prep.tables.ray_energies_kev[0]) / 5.0))]
    ray_mean = float(np.sum(np.linspace(-1, 1, cdf.size) * np.diff(cdf, prepend=0.0)))
    assert c["sum_cos_rayleigh"] / c["n_rayleigh"] == pytest.approx(ray_mean, abs=4 * 0.5 / np.sqrt(c["n_rayleigh"]))


def test_backends_coinciden_estadisticamente():
    if len(BACKENDS) < 2:
        pytest.skip("Numba no está instalado")
    p, g = cylinder()
    sp = xray.tube_spectrum(100.0, 3.0)
    geom = FanGeometry.covering(g, 0.6, 1.1, True)
    a = xs.simulate_scatter(p, 0.0, sp, geom, n_photons=400_000, slice_thickness=0.04, backend="numpy", seed=1)
    b = xs.simulate_scatter(p, 0.0, sp, geom, n_photons=400_000, slice_thickness=0.04, backend="numba", seed=1)
    assert abs(a.scat.sum() - b.scat.sum()) < 4.5 * np.sqrt(a.scat.sum() + b.scat.sum())
    assert abs(a.prim.sum() - b.prim.sum()) < 4.5 * np.sqrt(a.prim.sum() + b.prim.sum())
    with pytest.raises(ValueError):
        xs.simulate_scatter(p, 0.0, sp, geom, n_photons=10, backend="gpu")


def test_spr_crece_con_el_grosor_el_tamano_y_baja_con_la_rejilla():
    sp = xray.tube_spectrum(100.0, 3.0)

    def spr(radius, thickness, grid_deg=None):
        p, g = cylinder(radius=radius)
        geom = FanGeometry.covering(g, 0.6, 1.1, True)
        r = xs.simulate_scatter(p, 0.0, sp, geom, n_photons=600_000, slice_thickness=thickness,
                                grid_acceptance_deg=grid_deg, backend=BACKENDS[-1], seed=7)
        c = geom.n_det // 2
        s = slice(c - 6, c + 7)
        return r.scat[s].sum() / r.prim[s].sum()

    base = spr(0.09, 0.02)
    assert spr(0.09, 0.08) > 1.5 * base  # colimación más abierta
    assert spr(0.05, 0.02) < 0.7 * base  # objeto más pequeño
    assert spr(0.09, 0.02, grid_deg=1.0) < 0.5 * base  # rejilla antidispersora


def test_haz_paralelo_y_porosidad_en_el_scatter():
    p, g = cylinder(dx=0.005)
    r = xs.simulate_scatter(p, 30.0, xray.tube_spectrum(100.0, 3.0), None, n_photons=200_000, slice_thickness=0.02,
                            backend=BACKENDS[-1])
    assert abs(r.balance()["photons"]) < 1e-12
    assert r.scat.sum() > 0 and r.prim.sum() > 0
    # El scatter de una vista a 30° es el de la vista a 210° con el detector invertido
    r2 = xs.simulate_scatter(p, 210.0, xray.tube_spectrum(100.0, 3.0), None, n_photons=200_000,
                             slice_thickness=0.02, backend=BACKENDS[-1], seed=9)
    assert abs(r.scat.sum() - r2.scat.sum()) < 5 * np.sqrt(r.scat.sum() + r2.scat.sum())


def test_ct_con_scatter_cupping_y_perdida_de_contraste():
    p, g = cylinder(radius=0.07, tissue="muscle", dx=0.005)
    X, Z = g.meshgrid()
    p.labels[np.hypot(X - 0.035, Z) <= 0.012] = TISSUE_INDEX["cortical_bone"]
    sp = xray.tube_spectrum(100.0, 3.0)
    geom = FanGeometry.covering(g, 0.6, 1.1, True)
    opt = xs.ScatterOptions(n_photons=150_000, n_views=4, slice_thickness=0.04, backend=BACKENDS[-1])
    ct0 = xray.ct_scan(p, 120, sp, None, geometry=geom)
    ct1 = xray.ct_scan(p, 120, sp, None, geometry=geom, scatter=opt)
    r = np.hypot(X, Z)
    centro = (r < 0.015)
    borde = (r > 0.05) & (r < 0.065) & (np.hypot(X - 0.035, Z) > 0.02)
    assert np.allclose(ct0.hu[centro].mean(), ct0.hu[borde].mean(), atol=15)  # sin scatter: plano
    cupping = ct1.hu[borde].mean() - ct1.hu[centro].mean()
    assert cupping > 15  # el centro baja más que el borde
    bone = np.hypot(X - 0.035, Z) < 0.008
    assert ct1.hu[bone].mean() - ct1.hu[centro].mean() < 0.97 * (ct0.hu[bone].mean() - ct0.hu[centro].mean())
    assert ct1.projections.scatter.shape == ct1.projections.line_integrals.shape
    assert np.all(ct1.projections.scatter >= 0)


def test_acquire_sin_scatter_no_cambia():
    p, g = cylinder(dx=0.005)
    a = xray.acquire(p, [0.0, 45.0], xray.tube_spectrum(80.0, 2.5), photons=None)
    assert a.primary is None and a.scatter is None
