"""Haz en abanico: proyector, FBP de abanico y reconstrucción iterativa (SART)."""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.em import fanbeam as fb
from wavesim.em import xray, xray_phantoms as xp
from wavesim.em.projection import detector_axis, fbp
from wavesim.materials import xray as xd
from wavesim.phantoms import cross_section_limb


@pytest.fixture(scope="module")
def setup():
    g = Grid2D.from_size(0.2, 0.2, 0.003125, x0=-0.1, z0=-0.1)  # 64 x 64
    X, Z = g.meshgrid()
    return g, np.hypot(X, Z)


def test_elipse_chord_exacta():
    d = xp.disk(0.05, 20.0)
    val = xp.line_integrals(d, np.array([[0.02, -1.0]]), np.array([[0.0, 1.0]]))[0]
    assert val == pytest.approx(20.0 * 2 * np.sqrt(0.05**2 - 0.02**2), rel=1e-12)
    assert xp.line_integrals(d, np.array([[0.06, -1.0]]), np.array([[0.0, 1.0]]))[0] == 0.0


def test_geometria_cubre_el_dominio_y_valida(setup):
    g, _ = setup
    geom = fb.FanGeometry.covering(g, 0.6, 1.1, curved=True)
    assert geom.n_det % 2 == 1
    half = np.arcsin(0.5 * np.hypot(g.width, g.depth) / 0.6)
    assert geom.gamma.max() >= half
    with pytest.raises(ValueError):
        fb.FanGeometry(1.0, 0.5, 10, 1e-3)
    with pytest.raises(ValueError):
        fb.FanGeometry.covering(g, 0.1, 1.0)


def test_proyector_de_abanico_y_adjunto(setup):
    g, _ = setup
    d = xp.disk(0.05, 20.0)
    img = xp.rasterize(d, g)
    for curved in (True, False):
        geom = fb.FanGeometry.covering(g, 0.6, 1.1, curved)
        ang = [0.0, 37.0, 123.0]
        pj = fb.fan_project(img, g, geom, ang)[0]
        ex = fb.fan_exact_sinogram(d, g, geom, ang)
        assert np.abs(pj - ex).max() < 0.1 * ex.max()
        # El detector (curvo o plano) ve la fuente a la distancia pedida
        pts = geom.detector_points(g, 37.0)
        src, _ = geom.view_rays(g, 37.0)
        assert np.hypot(*(pts[geom.n_det // 2] - src)) == pytest.approx(1.1)
        rng = np.random.default_rng(0)
        x, y = rng.random(g.shape), rng.random((3, geom.n_det))
        lhs = np.sum(fb.fan_project(x, g, geom, ang)[0] * y)
        assert lhs == pytest.approx(np.sum(x * fb.fan_back(y, g, geom, ang)), rel=1e-10)


@pytest.mark.parametrize("curved", [True, False])
def test_fbp_de_abanico_en_un_disco(setup, curved):
    g, r = setup
    d = xp.disk(0.05, 20.0)
    geom = fb.FanGeometry.covering(g, 0.6, 1.1, curved)
    ang = np.arange(360.0)
    rec = fb.fan_fbp(fb.fan_exact_sinogram(d, g, geom, ang), ang, geom, g)
    assert rec[r < 0.04].mean() == pytest.approx(20.0, rel=0.01)
    assert abs(rec[r > 0.065].mean()) < 0.3
    # Frente al haz paralelo con el mismo objeto exacto
    det = detector_axis(g)
    a2 = np.arange(180.0)
    par = fbp(xp.parallel_sinogram(d, g, a2, det), a2, det, g)
    assert np.sqrt(np.mean((rec - par)[r < 0.06] ** 2)) < 1.0


def test_shepp_logan_abanico_frente_a_paralelo_y_exacto(setup):
    g, r = setup
    sl = xp.shepp_logan(0.08, 20.0)
    ref = xp.rasterize(sl, g)
    m = r < 0.08 * 0.95
    rmse = lambda a: float(np.sqrt(np.mean((a - ref)[m] ** 2)))  # noqa: E731
    det = detector_axis(g)
    a2 = np.arange(360.0) / 2
    par = fbp(xp.parallel_sinogram(sl, g, a2, det), a2, det, g)
    ang = np.arange(360.0)
    for curved in (True, False):
        geom = fb.FanGeometry.covering(g, 0.6, 1.1, curved)
        rec = fb.fan_fbp(fb.fan_exact_sinogram(sl, g, geom, ang), ang, geom, g)
        # El error está dominado por la discretización de 64 píxeles: el abanico no debe empeorarlo
        assert rmse(rec) < 1.2 * rmse(par) + 0.1
        assert rmse(rec) < 0.3 * np.sqrt(np.mean(ref[m] ** 2))
        # Región uniforme: el cerebro vale (1 - 0.8)·mu0 = 4 /m
        brain = (np.abs(ref - 4.0) < 1e-9) & (r < 0.04)
        assert np.mean(rec[brain]) == pytest.approx(4.0, abs=0.6)


def test_sart_con_pocos_angulos_gana_a_fbp(setup):
    g, r = setup
    sl = xp.shepp_logan(0.08, 20.0)
    ref = xp.rasterize(sl, g)
    m = r < 0.08 * 0.95
    rmse = lambda a: float(np.sqrt(np.mean((a - ref)[m] ** 2)))  # noqa: E731
    geom = fb.FanGeometry.covering(g, 0.6, 1.1, True)
    for n in (30, 60):
        ang = np.arange(n) * 360.0 / n
        sino = fb.fan_exact_sinogram(sl, g, geom, ang)
        e_fbp = rmse(fb.fan_fbp(sino, ang, geom, g))
        sart = fb.os_sart(sino, ang, geom, g, n_iter=10, subset_size=5)
        assert sart.min() >= 0.0  # no negatividad
        assert rmse(sart) < 0.85 * e_fbp


def test_tv_ayuda_con_ruido(setup):
    g, r = setup
    sl = xp.shepp_logan(0.08, 20.0)
    ref = xp.rasterize(sl, g)
    m = r < 0.08 * 0.95
    geom = fb.FanGeometry.covering(g, 0.6, 1.1, True)
    ang = np.arange(60) * 6.0
    sino = fb.fan_exact_sinogram(sl, g, geom, ang)
    sino = sino + 0.05 * np.random.default_rng(1).standard_normal(sino.shape)
    base = fb.os_sart(sino, ang, geom, g, n_iter=10, subset_size=5)
    tv = fb.os_sart(sino, ang, geom, g, n_iter=10, subset_size=5, tv_weight=1e-3)
    rm = lambda a: float(np.sqrt(np.mean((a - ref)[m] ** 2)))  # noqa: E731
    assert rm(tv) < rm(base) < rm(fb.fan_fbp(sino, ang, geom, g))


def test_ct_scan_abanico_en_el_tac_del_fantoma():
    p = cross_section_limb("pierna", dx=1.0e-3).replace_tissue("water", "air")
    geom = fb.FanGeometry.covering(p.grid, 0.5, 0.9, True)
    ct = xray.ct_scan(p, n_angles=240, spectrum=xray.monoenergetic(60.0), photons=None, geometry=geom)
    mu_w = xd.linear_attenuation("water", 60.0)
    expect = 1000 * (xd.linear_attenuation("muscle", 60.0) - mu_w) / mu_w
    assert float(np.median(ct.hu[p.mask("muscle")])) == pytest.approx(expect, abs=30)
    assert ct.projections.geometry is geom
    with pytest.raises(ValueError):
        xray.ct_scan(p, n_angles=10, method="sart")
