"""Tests rápidos de rayos X / TAC 3D (``wavesim.em.xray3d``): cuerdas analíticas, adjunción, coherencia con el 2D,
FDK, radiografía policromática, ruido de Poisson, visibilidad de la grieta y scatter."""

import numpy as np
import pytest

from wavesim.core.grid3d import Grid3D
from wavesim.em import xray, xray3d as x3
from wavesim.em.projection import detector_axis
from wavesim.materials import xray as xd
from wavesim.phantoms import Fracture, TISSUE_INDEX, longitudinal_limb
from wavesim.phantoms.limb3d import Phantom3D, extrude, homogeneous_phantom, leg_segment

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


def _box(grid):
    lo = np.array([grid.x0, grid.y0, grid.z0]) - 0.5 * grid.dx
    return lo, lo + np.array([grid.nx, grid.ny, grid.nz]) * grid.dx


def test_cuerda_de_una_caja_homogenea_es_exacta_en_los_dos_backends():
    g = Grid3D(20, 16, 14, 1e-3)
    lo, hi = _box(g)
    geo = x3.Geometry3D.covering(g, 0.2, 0.3)
    for ang in (13.0, 37.0, 211.0):
        org, dirs = geo.rays(g, ang)
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - org) / dirs, (hi - org) / dirs
        par = np.abs(dirs) < 1e-12
        ins = (org >= lo) & (org <= hi)
        t0 = np.where(par, np.where(ins, -np.inf, np.inf), np.minimum(ta, tb)).max(axis=1)
        t1 = np.where(par, np.where(ins, np.inf, -np.inf), np.maximum(ta, tb)).min(axis=1)
        ref = np.maximum(t1 - t0, 0.0)
        for be in ("numpy",) + (("numba",) if HAS_NUMBA else ()):
            p = x3.project(np.ones(g.shape), g, geo, [ang], backend=be)[0].reshape(-1)
            assert np.abs(p - ref).max() < 1e-12


def test_esfera_y_cilindro_frente_a_la_cuerda_analitica():
    g = Grid3D(40, 40, 40, 1e-3)
    c = x3.center(g)
    lo, hi = _box(g)
    geo = x3.Geometry3D.covering(g, 0.4, 0.5)
    org, dirs = geo.rays(g, 37.0)
    R = 0.012
    sph = x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (R, R, R), 1.0)], 3)
    cyl = x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (1.0, R, R), 1.0)], 3)
    cr = np.cross(dirs, np.array([1.0, 0, 0]))
    rho_c = np.abs(((org - c) * cr).sum(1)) / np.linalg.norm(cr, axis=1)
    for vol, ref, rho in ((sph, x3.sphere_chords(org, dirs, c, R), x3.ray_point_distance(org, dirs, c)),
                          (cyl, x3.cylinder_chords(org, dirs, c, R, "x", lo, hi), rho_c)):
        sel = rho < 0.8 * R
        for method in ("siddon", "sampling"):
            p = x3.project(vol, g, geo, [37.0], method=method)[0].reshape(-1)
            rms = np.sqrt(((p - ref)[sel] ** 2).mean())
            assert rms < 0.01 * 2 * R, (method, rms)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("method,backend", [("siddon", "numpy"), ("siddon", "numba"), ("sampling", "numpy")])
def test_adjuncion_ax_y_igual_x_aty(parallel, method, backend):
    if backend == "numba" and not HAS_NUMBA:
        pytest.skip("sin Numba")
    g = Grid3D(14, 12, 10, 1e-3)
    rng = np.random.default_rng(5)
    x = rng.random(g.shape)
    geo = x3.Geometry3D.covering(g, 0.15, 0.2, parallel=parallel)
    ang = [0.0, 33.0, 120.0]
    y = rng.random((3, geo.n_v, geo.n_u))
    lhs = np.sum(x3.project(x, g, geo, ang, method, backend) * y)
    rhs = np.sum(x * x3.backproject(y, g, geo, ang, method, backend))
    assert abs(lhs - rhs) < 1e-10 * abs(lhs)


def test_numpy_y_numba_dan_la_misma_proyeccion():
    if not HAS_NUMBA:
        pytest.skip("sin Numba")
    g = Grid3D(16, 14, 12, 1e-3)
    vol = np.random.default_rng(2).random(g.shape)
    geo = x3.Geometry3D.covering(g, 0.2, 0.3)
    a = x3.project(vol, g, geo, [10.0, 100.0], backend="numpy")
    b = x3.project(vol, g, geo, [10.0, 100.0], backend="numba")
    assert np.abs(a - b).max() < 1e-12 * np.abs(a).max()


def test_volumen_extruido_proyectado_en_paralelo_coincide_con_el_2d():
    ph = longitudinal_limb("tibia", Fracture(), width=0.03, depth=0.03, dx=0.5e-3)
    p3 = extrude(ph, ny=4)
    spec = xray.tube_spectrum(60, 2.5)
    sp = 0.37 * ph.grid.dx  # ningún rayo cae justo sobre una cara de celda
    det = detector_axis(ph.grid, sp)
    geo = x3.Geometry3D(0.5, 0.5, det.size, 3, sp, 1e-3, "y", True)
    ang = [30, 90]
    a2 = xray.acquire(ph, ang, spec, photons=None, det_spacing=sp)
    inner = np.abs(det) < 0.0135
    # El muestreo trilineal es el mismo esquema que el 2D (bilineal): coinciden a ~0.2 %
    s3 = x3.acquire3d(p3, ang, geo, spec, photons=None, method="sampling")
    for a in range(2):
        d = (s3.line_integrals[a, 1] - a2.line_integrals[a])[inner]
        assert np.linalg.norm(d) < 3e-3 * np.linalg.norm(a2.line_integrals[a][inner])
    # Siddon (exacto) difiere del 2D sólo por el suavizado del bilineal 2D: < 1 % en un ángulo oblicuo
    d = (x3.acquire3d(p3, [30], geo, spec, photons=None).line_integrals[0, 1] - a2.line_integrals[0])[inner]
    assert np.linalg.norm(d) < 1e-2 * np.linalg.norm(a2.line_integrals[0][inner])
    # las filas del detector (eje y) son idénticas: el volumen es invariante en y
    assert np.abs(s3.line_integrals[:, 0] - s3.line_integrals[:, 2]).max() < 1e-12


def test_fdk_reconstruye_una_esfera_homogenea():
    n = 32
    g = Grid3D(n, n, n, 1.5e-3)
    gf = Grid3D(2 * n, 2 * n, 2 * n, 0.75e-3, g.x0 - 0.375e-3, g.y0 - 0.375e-3, g.z0 - 0.375e-3)
    R, mu0 = 0.016, 40.0
    sph = x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (R, R, R), mu0)], 2)
    sphf = x3.ellipsoid_volume(gf, [x3.Ellipsoid((0, 0, 0), (R, R, R), mu0)], 2)
    geo = x3.Geometry3D.covering(g, 0.5, 0.7)
    ang = np.arange(120) * 3.0
    rec = x3.fdk(x3.project(sphf, gf, geo, ang), ang, geo, g)
    c = n // 2
    assert abs(rec[c, c, c] / mu0 - 1) < 0.02  # centro
    zz, yy, xx = np.meshgrid(g.z - g.z.mean(), g.y - g.y.mean(), g.x - g.x.mean(), indexing="ij")
    inside = np.sqrt(xx**2 + yy**2 + zz**2) < 0.75 * R  # lejos del borde (oscilación de Gibbs)
    assert np.sqrt(((rec - sph)[inside] ** 2).mean()) / mu0 < 0.03
    assert abs(rec[c, c, c + 8] / mu0 - 1) < 0.05  # fuera del plano central
    # el fondo fuera de la esfera queda cerca de cero
    assert abs(rec[c, c, 1]) < 0.05 * mu0


def test_fdk_numpy_igual_a_numba():
    if not HAS_NUMBA:
        pytest.skip("sin Numba")
    n = 16
    g = Grid3D(n, n, n, 2e-3)
    sph = x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (0.012, 0.012, 0.012), 1.0)], 1)
    geo = x3.Geometry3D.covering(g, 0.4, 0.5)
    ang = np.arange(24) * 15.0
    s = x3.project(sph, g, geo, ang)
    a = x3.fdk(s, ang, geo, g, backend="numpy")
    b = x3.fdk(s, ang, geo, g, backend="numba")
    assert np.abs(a - b).max() < 1e-9 * np.abs(a).max()


def test_fdk_es_exacto_en_un_cilindro_a_lo_largo_del_eje_y_el_apilado_no():
    n = 32
    g = Grid3D(n, n, n, 1.5e-3)
    gf = Grid3D(2 * n, 2 * n, 2 * n, 0.75e-3, g.x0 - 0.375e-3, g.y0 - 0.375e-3, g.z0 - 0.375e-3)
    cylf = x3.ellipsoid_volume(gf, [x3.Ellipsoid((0, 0, 0), (1.0, 0.014, 0.014), 40.0)], 2)
    geo = x3.Geometry3D.covering(g, 0.18, 0.25)  # cono amplio
    ang = np.arange(120) * 3.0
    s = x3.project(cylf, gf, geo, ang)
    c = n // 2
    fd = x3.fdk(s, ang, geo, g, mode="fdk")
    sk = x3.fdk(s, ang, geo, g, mode="stack")
    assert abs(fd[c, c, c + 8] / 40.0 - 1) < 0.02
    assert abs(sk[c, c, c + 8] / 40.0 - 1) > abs(fd[c, c, c + 8] / 40.0 - 1)


def test_tac_paralelo_apilado_reconstruye_el_corte():
    n = 32
    g = Grid3D(n, n, n, 1.5e-3)
    sph = x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (0.016, 0.016, 0.016), 40.0)], 2)
    geo = x3.Geometry3D.covering(g, 0.3, parallel=True)
    ang = np.arange(60) * 3.0
    rec = x3.fdk(x3.project(sph, g, geo, ang), ang, geo, g, mode="parallel")
    assert abs(rec[n // 2, n // 2, n // 2] / 40.0 - 1) < 0.03


def test_radiografia_policromatica_de_una_losa_frente_a_beer_lambert_integrado():
    spec = xray.tube_spectrum(70, 2.5)
    geo = x3.Geometry3D(0.5, 0.5, 3, 3, 1e-3, 1e-3, "x", True)
    for tissue, nz in (("water", 30), ("cortical_bone", 8)):
        ph = homogeneous_phantom(Grid3D(6, 6, nz, 1e-3), tissue)
        mu = xd.linear_attenuation(tissue, spec.energies_kev)
        for det in (None, x3.FlatPanel()):
            w, _ = x3._as_panel(det).signal_weights(spec)
            ref = -np.log(np.sum(w * np.exp(-mu * nz * 1e-3)))
            sim = x3.acquire3d(ph, [0.0], geo, spec, photons=None, detector=det).line_integrals[0, 1, 1]
            assert abs(sim - ref) < 1e-9
    # endurecimiento del haz: la atenuación efectiva por mm baja con el espesor
    mu_eff = [x3.acquire3d(homogeneous_phantom(Grid3D(6, 6, n, 1e-3), "water"), [0.0], geo, spec,
                           photons=None).line_integrals[0, 1, 1] / (n * 1e-3) for n in (5, 20, 40)]
    assert mu_eff[0] > mu_eff[1] > mu_eff[2]


def test_magnificacion_conica():
    g = Grid3D(20, 20, 20, 1e-3)
    ph = homogeneous_phantom(g, "water")
    ph.labels[8:12, 8:12, 8:12] = TISSUE_INDEX["cortical_bone"]  # cubo central de 4 mm
    geo = x3.Geometry3D(0.5, 0.75, 121, 121, 0.25e-3, 0.25e-3, "x", False)
    img = x3.acquire3d(ph, [0.0], geo, xray.monoenergetic(60.0), photons=None).line_integrals[0]
    prof = img[60]
    agua = prof[40]  # dentro de la caja de agua pero fuera del cubo
    edge = np.nonzero(prof > agua + 0.5 * (prof.max() - agua))[0]
    width = (edge.max() - edge.min() + 1) * geo.du
    assert width == pytest.approx(4e-3 * 1.5, rel=0.1)  # magnificación sdd/sid = 1.5


def test_ruido_de_poisson_en_la_radiografia():
    g = Grid3D(48, 48, 8, 1e-3)
    ph = homogeneous_phantom(g, "water")
    geo = x3.Geometry3D(0.5, 0.5, 40, 40, 1e-3, 1e-3, "x", True)  # todos los rayos cruzan el agua
    spec = xray.tube_spectrum(70, 2.5)
    n0 = 5e3
    r = x3.acquire3d(ph, [0.0], geo, spec, photons=n0, rng=np.random.default_rng(4))
    t = float(x3.acquire3d(ph, [0.0], geo, spec, photons=None).intensity.mean())
    counts = r.intensity[0] * n0
    assert counts.mean() == pytest.approx(n0 * t, rel=0.02)
    assert counts.var() == pytest.approx(counts.mean(), rel=0.15)  # varianza = media (Poisson)
    # con el panel: se detecta una fracción de los fotones, la varianza sigue a la media de fotones absorbidos
    panel = x3.FlatPanel(csi_thickness=0.3e-3, energy_integrating=False)
    r2 = x3.acquire3d(ph, [0.0], geo, spec, photons=n0, detector=panel, rng=np.random.default_rng(4))
    _, absorbed = panel.signal_weights(spec)
    c2 = r2.intensity[0] * n0 * absorbed
    assert c2.var() == pytest.approx(c2.mean(), rel=0.15)
    assert absorbed < 0.99


def test_la_grieta_solo_se_ve_con_el_haz_paralelo_a_su_plano():
    g = Grid3D(40, 40, 40, 0.5e-3)
    lab = homogeneous_phantom(g, "water").labels.copy()
    zz, yy, xx = np.meshgrid(g.z - g.z.mean(), g.y - g.y.mean(), g.x - g.x.mean(), indexing="ij")
    blk = (np.abs(zz) < 8e-3) & (np.abs(yy) < 8e-3) & (np.abs(xx) < 8e-3)
    lab_c = lab.copy()
    lab[blk] = TISSUE_INDEX["cortical_bone"]
    lab_c[blk & (np.abs(yy) >= 0.5e-3)] = TISSUE_INDEX["cortical_bone"]  # grieta de 1 mm en el plano y = 0
    geo = x3.Geometry3D(0.5, 0.5, 81, 3, 0.25e-3, 0.25e-3, "x", True)
    e = xray.monoenergetic(60.0)

    def signal(phi):
        a0 = x3.acquire3d(Phantom3D(g, lab), [phi], geo, e, photons=None).line_integrals[0, 1]
        a1 = x3.acquire3d(Phantom3D(g, lab_c), [phi], geo, e, photons=None).line_integrals[0, 1]
        return float(np.max(a0 - a1))

    mu_b, mu_w = (float(xd.linear_attenuation(t, 60.0)) for t in ("cortical_bone", "water"))
    s0, s45, s90 = signal(0.0), signal(45.0), signal(90.0)
    assert s0 == pytest.approx((mu_b - mu_w) * 16e-3, rel=0.02)  # haz en el plano: atraviesa todo el bloque
    assert s90 == pytest.approx((mu_b - mu_w) * 1e-3, rel=0.05)  # haz perpendicular: sólo el grosor de la grieta
    assert s0 > 10 * s90 and s45 < 0.2 * s0


def test_scatter_balance_primaria_y_losa_frente_a_la_primaria_de_beer_lambert():
    if not HAS_NUMBA:
        pytest.skip("sin Numba")
    from wavesim.em import xray3d_scatter as sc

    g = Grid3D(24, 24, 24, 1e-3)
    ph = homogeneous_phantom(g, "water")
    spec = xray.tube_spectrum(60, 2.5)
    geo = x3.Geometry3D.covering(g, 0.5, 0.6)
    r = sc.simulate_view(ph, 0.0, geo, spec, None, 300_000, 6)
    assert abs(r.balance()) < 1e-12
    bl = x3.acquire3d(ph, [0.0], geo, spec, photons=None).intensity[0]
    c = bl[geo.n_v // 2 - 2:geo.n_v // 2 + 3, geo.n_u // 2 - 2:geo.n_u // 2 + 3].mean()
    assert r.primary_rel[2:4, 2:4].mean() == pytest.approx(c, rel=0.05)
    assert 0.0 < r.scat_n.sum() / r.prim_n.sum() < 0.2
    # el scatter se suma a la primaria en acquire3d y reduce el contraste (-ln I)
    a = x3.acquire3d(ph, [0.0], geo, spec, photons=None, scatter=sc.Scatter3DOptions(200_000, 1, 6))
    assert a.scatter is not None and a.intensity.mean() > a.primary.mean()


def test_pierna_3d_radiografia_ap_lateral_y_geometria():
    ph = leg_segment("tibia", None, length=0.02, dx=1e-3)
    ap = x3.radiograph(ph, "ap", photons=None)
    lat = x3.radiograph(ph, "lateral", photons=None)
    assert ap.line_integrals.shape == lat.line_integrals.shape and ap.line_integrals.shape[0] == 1
    assert ap.line_integrals.min() >= -1e-9 and ap.line_integrals.max() > 0.5
    with pytest.raises(ValueError):
        x3.Geometry3D(1.0, 0.5, 4, 4, 1e-3, 1e-3)
    with pytest.raises(ValueError):
        x3.project(np.zeros((2, 2, 2)), Grid3D(3, 3, 3, 1e-3), x3.Geometry3D(1.0, 1.2, 4, 4, 1e-3, 1e-3), [0.0])


def test_os_sart3d_reconstruye_una_esfera_y_el_tv_ayuda_con_ruido():
    n = 20
    g = Grid3D(n, n, n, 2.2e-3)
    sph = x3.ellipsoid_volume(g, [x3.Ellipsoid((0, 0, 0), (0.016, 0.016, 0.016), 40.0)], 2)
    geo = x3.Geometry3D.covering(g, 0.4, 0.55)
    ang = np.arange(24) * 15.0
    s = x3.project(sph, g, geo, ang)
    rec = x3.os_sart3d(s, ang, geo, g, n_iter=4, subset_size=6)
    zz, yy, xx = np.meshgrid(g.z - g.z.mean(), g.y - g.y.mean(), g.x - g.x.mean(), indexing="ij")
    inside = np.sqrt(xx**2 + yy**2 + zz**2) < 0.012
    assert np.sqrt(((rec - sph)[inside] ** 2).mean()) / 40.0 < 0.05
    assert rec.min() >= 0.0  # no negatividad
    # con ruido de Poisson fuerte, la TV reduce el error respecto a no regularizar
    rng = np.random.default_rng(1)
    noisy = -np.log(np.maximum(rng.poisson(200.0 * np.exp(-s)), 1) / 200.0)
    e0 = np.sqrt(((x3.os_sart3d(noisy, ang, geo, g, n_iter=6, subset_size=6) - sph)[inside] ** 2).mean())
    e1 = np.sqrt(((x3.os_sart3d(noisy, ang, geo, g, n_iter=6, subset_size=6, tv_weight=0.003) - sph)[inside] ** 2)
                 .mean())
    assert e1 < e0
