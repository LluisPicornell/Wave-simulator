"""Tests del Monte Carlo de fotones (``wavesim.em.montecarlo``).

Los umbrales estadísticos son de la forma ``|valor - referencia| <= k·sem``
con ``sem`` el error estándar entre tandas del propio resultado y una semilla
fija; el benchmark completo (``benchmarks/optics_validation.py``) usa muchos más fotones.
"""

import importlib.util

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.em import montecarlo as mc
from wavesim.em import optics
from wavesim.materials import optical
from wavesim.phantoms import TISSUE_INDEX, Phantom, cross_section_limb

HAS_NUMBA = importlib.util.find_spec("numba") is not None
BACKENDS = ["numpy"] + (["numba"] if HAS_NUMBA else [])

# Losa de van de Hulst: mu_a = 10 /cm, mu_s = 90 /cm, g = 0.75, 0.02 cm (tau = 2, albedo 0.9)
SLAB = dict(mua=1000.0, mus=9000.0, g=0.75, thickness=2e-4)


def _n(backend: str, n_numba: int) -> int:
    """El backend NumPy es ~10 veces más lento: menos fotones para mantener cada test por debajo de unos segundos."""
    return n_numba if backend == "numba" else n_numba // 4


def _series(r_f: float, mu: float, d: float) -> float:
    """Transmitancia de una losa sin dispersión con reflexiones múltiples (incidencia normal)."""
    e = np.exp(-mu * d)
    return (1 - r_f) ** 2 * e / (1 - r_f**2 * e**2)


@pytest.mark.parametrize("g", [0.0, 0.75, 0.95])
def test_henyey_greenstein_media(g):
    """<cos> = g y <P2(cos)> = g² (propiedades de la función de fase de Henyey-Greenstein)."""
    n = 400_000
    c = mc.henyey_greenstein_cos(g, np.random.default_rng(5).random(n))
    assert abs(c.mean() - g) < 5 * c.std() / np.sqrt(n) + 1e-12
    p2 = 0.5 * (3 * c**2 - 1)
    assert abs(p2.mean() - g**2) < 5 * p2.std() / np.sqrt(n) + 1e-12


def test_fresnel():
    r, ct = mc.fresnel(1.0, 1.5, 1.0)
    assert r == pytest.approx(0.04) and ct == pytest.approx(1.0)
    # de denso a menos denso: reflexión total interna por encima del ángulo crítico (41.8° para 1.5 -> 1)
    r_tir, _ = mc.fresnel(1.5, 1.0, np.cos(np.radians(45.0)))
    r_ok, _ = mc.fresnel(1.5, 1.0, np.cos(np.radians(30.0)))
    assert r_tir == 1.0 and r_ok < 0.1
    r_eq, _ = mc.fresnel(1.4, 1.4, 0.3)
    assert r_eq == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_van_de_hulst_indice_ajustado(backend):
    """(a) Losa con n = 1 dentro y fuera: R = 0.09739, T = 0.66096 (van de Hulst, tabla 35, doubling)."""
    m = mc.slab_medium(**SLAB, dx=5e-6)
    r = mc.simulate(m, _n(backend, 80_000), periodic_x=True, backend=backend, seed=7)
    assert abs(r.reflectance - 0.09739) < 4 * r.sem["reflectance"] + 2e-4
    assert abs(r.transmittance - 0.66096) < 4 * r.sem["transmittance"] + 2e-4
    assert r.specular == 0.0
    assert r.balance == pytest.approx(1.0, abs=1e-5)  # sólo la ruleta rusa rompe el balance exacto
    assert r.lateral == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [1.0, 1.5])
def test_atenuacion_balistica(n, backend):
    """(c) Transmitancia sin dispersar = serie de Fresnel con exp(-mu_t d)."""
    a, s, d = 300.0, 500.0, 2e-3
    m = mc.slab_medium(a, s, 0.8, d, n=n, dx=d / 40)
    r = mc.simulate(m, _n(backend, 24_000), periodic_x=True, backend=backend, seed=3)
    rf = ((n - 1) / (n + 1)) ** 2
    assert r.specular == pytest.approx(rf, rel=1e-9)
    ref = _series(rf, a + s, d)
    assert abs(r.ballistic_transmittance - ref) < 4 * r.sem["ballistic_transmittance"] + 1e-9
    assert r.ballistic_transmittance < r.transmittance  # hay luz dispersada que también sale


@pytest.mark.parametrize("backend", BACKENDS)
def test_losa_absorbente_con_desajuste_de_indice(backend):
    """(b) Sin dispersión y con n = 1.5, el resultado es exacto: serie de reflexiones múltiples de Fresnel."""
    a, d, n = 400.0, 1e-3, 1.5
    m = mc.slab_medium(a, 0.0, 0.0, d, n=n, dx=d / 20)
    r = mc.simulate(m, 5_000, periodic_x=True, backend=backend, seed=1)
    rf = ((n - 1) / (n + 1)) ** 2
    t_ref = _series(rf, a, d)
    e = np.exp(-a * d)
    r_ref = rf * (1 - rf) ** 2 * e**2 / (1 - rf**2 * e**2)  # reflexión difusa (sin la especular)
    assert r.transmittance == pytest.approx(t_ref, rel=1e-6)
    assert r.reflectance == pytest.approx(r_ref, rel=1e-3)  # la ruleta rusa del rabo introduce ruido ~1e-5
    assert r.balance == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_losa_dispersiva_con_desajuste_es_coherente(backend):
    """Con dispersión y n = 1.5 no hay referencia verificada: se exige balance, Fresnel de entrada y sentido físico."""
    kw = dict(periodic_x=True, backend=backend, seed=2)
    matched = mc.simulate(mc.slab_medium(**SLAB, dx=5e-6), _n(backend, 10_000), **kw)
    mism = mc.simulate(mc.slab_medium(**SLAB, n=1.5, dx=5e-6), _n(backend, 10_000), **kw)
    assert mism.specular == pytest.approx(0.04)
    assert mism.balance == pytest.approx(1.0, abs=1e-5)
    # la reflexión interna atrapa luz: más absorbida y menos transmitida que con índice ajustado
    assert mism.transmittance < matched.transmittance - 4 * (mism.sem["transmittance"] + matched.sem["transmittance"])
    assert mism.absorbed_total > matched.absorbed_total


def test_semilla_reproducible():
    m = mc.slab_medium(**SLAB, dx=1e-5)
    a = mc.simulate(m, 3_000, periodic_x=True, backend="numpy", seed=11)
    b = mc.simulate(m, 3_000, periodic_x=True, backend="numpy", seed=11)
    c = mc.simulate(m, 3_000, periodic_x=True, backend="numpy", seed=12)
    assert a.reflectance == b.reflectance and np.array_equal(a.fluence, b.fluence)
    assert a.reflectance != c.reflectance
    if HAS_NUMBA:
        d1 = mc.simulate(m, 3_000, periodic_x=True, backend="numba", seed=11)
        d2 = mc.simulate(m, 3_000, periodic_x=True, backend="numba", seed=11)
        assert d1.reflectance == d2.reflectance and np.array_equal(d1.fluence, d2.fluence)


@pytest.mark.skipif(not HAS_NUMBA, reason="requiere Numba")
def test_backends_numpy_y_numba_coinciden_estadisticamente():
    m = mc.slab_medium(**SLAB, n=1.4, dx=5e-6)
    a = mc.simulate(m, 8_000, periodic_x=True, backend="numpy", seed=5)
    b = mc.simulate(m, 8_000, periodic_x=True, backend="numba", seed=5)
    for k in ("reflectance", "transmittance", "absorbed_total"):
        diff = abs(getattr(a, k) - getattr(b, k))
        assert diff < 4 * np.hypot(a.sem[k], b.sem[k]) + 1e-9, k


def test_backend_desconocido():
    with pytest.raises(ValueError):
        mc.simulate(mc.slab_medium(**SLAB), 10, backend="cuda")


def test_resolucion_temporal_balistica():
    """Sin dispersión, todo lo transmitido llega en el instante n d / c; la suma temporal reproduce lo no resuelto."""
    a, d, n = 200.0, 1e-3, 1.4
    m = mc.slab_medium(a, 0.0, 0.0, d, n=n, dx=d / 10)
    t_flight = n * d / mc.C0
    r = mc.simulate(m, 2_000, periodic_x=True, backend="numpy", time_bins=39, t_max=4 * t_flight)
    k = int(np.argmax(r.transmittance_t))
    assert r.time_edges[k] <= t_flight < r.time_edges[k + 1]
    # reflexiones internas: el eco llega a 3 n d / c
    assert r.transmittance_t.sum() + r.reflectance_t.sum() + r.late == pytest.approx(
        r.transmittance + r.reflectance, rel=1e-9)
    # lo registrado después de t_max se pierde en la fluencia resuelta (aquí, el rabo de reflexiones ~1e-6)
    assert r.fluence_t.sum(axis=0) == pytest.approx(r.fluence, rel=1e-4)


def test_resolucion_temporal_dispersiva_coherente():
    m = mc.slab_medium(**SLAB, dx=1e-5)
    r = mc.simulate(m, 6_000, periodic_x=True, backend="numpy", time_bins=30, t_max=40e-12)
    assert r.transmittance_t.sum() + r.reflectance_t.sum() + r.late == pytest.approx(
        r.transmittance + r.reflectance, rel=1e-9)
    # la luz retrodispersada tarda más que la transmitida balística: la media de tiempos de la reflectancia es mayor
    tc = 0.5 * (r.time_edges[1:] + r.time_edges[:-1])
    assert (tc * r.reflectance_t).sum() / r.reflectance_t.sum() > 0


def _fat_slab(dx=1e-3, width=100, depth=70, rim=5):
    g = Grid2D(width + 2 * rim, depth + 2 * rim, dx)
    lab = np.full(g.shape, TISSUE_INDEX["air"])
    lab[rim:-rim, rim:-rim] = TISSUE_INDEX["fat"]
    return Phantom(g, lab)


def test_medio_desde_fantoma_reproduce_la_tabla():
    p = _fat_slab(dx=2e-3, width=10, depth=10)
    m = mc.medium_from_phantom(p, 800.0)
    tis = m.tissue
    assert not tis[0, 0] and tis[10, 10]
    g = optical.OPTICAL["fat"].anisotropy
    assert np.allclose(m.mus[tis] * (1 - m.g[tis]), optical.mu_s_reduced("fat", 800.0))
    assert np.allclose(m.mua[tis], optical.mu_a("fat", 800.0))
    assert np.allclose(m.g[tis], g)
    assert m.n_outer == 1.0


def test_pendiente_coincide_con_difusion():
    """(d) Lejos de la fuente y de las fronteras, log(fluencia) decae con mu_eff (<= 5 %), igual que la difusión."""
    p = _fat_slab(dx=1e-3, width=80, depth=45)
    med = mc.medium_from_phantom(p, 800.0)
    r = mc.simulate(med, 40_000, seed=4)
    diff = optics.solve_diffusion(p, 800.0)
    z = (np.arange(p.grid.nz) - 5 + 0.5) * p.grid.dx  # profundidad bajo la superficie (z = 0 en la cara)
    cols = slice(25, 65)
    sel = (z > 0.008) & (z < 0.018)
    meff = float(optical.mu_effective("fat", 800.0))
    slopes = {}
    for name, f in (("mc", r.fluence), ("diff", diff.fluence)):
        slopes[name] = -np.polyfit(z[sel], np.log(f[:, cols].mean(axis=1)[sel]), 1)[0]
    assert abs(slopes["mc"] / meff - 1) < 0.05
    assert abs(slopes["diff"] / meff - 1) < 0.05
    # cerca de la cara iluminada la difusión sobrestima o subestima: el MC difiere más de un 20 %
    j = 5 + 1  # segunda celda bajo la superficie
    assert abs(r.fluence[j, cols].mean() / diff.fluence[j, cols].mean() - 1) > 0.2


@pytest.mark.parametrize("source", ["top", "pencil", "isotropic"])
def test_balance_en_fantoma_con_fugas_laterales(source):
    # 450 nm: la grasa absorbe más y las historias son cortas (el test es rápido también con NumPy)
    p = _fat_slab(dx=2e-3, width=20, depth=8)
    med = mc.medium_from_phantom(p, 450.0)
    xz = {"top": None, "pencil": (p.grid.x[6], 0.0), "isotropic": (p.grid.x[6], p.grid.z[8])}[source]
    r = mc.simulate(med, 600, source=source, source_xz=xz, seed=2)
    assert r.balance == pytest.approx(1.0, abs=1e-4)
    assert r.lateral > 0.0  # el fantoma es finito: parte de la luz escapa por los lados
    # las exitancias por columna reproducen los totales
    assert r.reflectance_profile.sum() * p.grid.dx / r.source_width == pytest.approx(r.reflectance)
    assert r.absorbed.sum() == pytest.approx(r.absorbed_total)
    assert (r.fluence >= 0).all()


def test_fuente_fuera_del_tejido():
    p = _fat_slab(dx=2e-3, width=10, depth=10)
    med = mc.medium_from_phantom(p, 800.0)
    with pytest.raises(ValueError):
        mc.simulate(med, 10, source="isotropic", source_xz=(p.grid.x[0], p.grid.z[0]))
    with pytest.raises(ValueError):
        mc.simulate(med, 10, source="isotropic")


def test_pierna_corre_y_conserva_energia():
    p = cross_section_limb("pierna", dx=3e-3, seed=1).replace_tissue("water", "air")
    med = mc.medium_from_phantom(p, 800.0)
    r = mc.simulate(med, 1_000, seed=1)
    assert r.balance == pytest.approx(1.0, abs=1e-4)
    assert 0.0 < r.reflectance < 1.0
    # la luz que sale por los lados y la absorbida acaparan casi todo; la transmitida es mínima
    assert r.transmittance < r.reflectance
