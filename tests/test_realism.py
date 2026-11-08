"""Modo realista: porosidad cortical, red trabecular, rugosidad multiescala y tejido blando."""

from dataclasses import replace

import numpy as np
import pytest

from wavesim.core.signals import gaussian_pulse
from wavesim.em import xray
from wavesim.materials import ACOUSTIC, TISSUES, dielectric, mri, optical, porous
from wavesim.materials import xray as xray_tables
from wavesim.mri.sequences import FAT_FRACTION, tissue_signal
from wavesim.phantoms import (
    BODY_TYPES, REALISM_PRESETS, CorticalPorosity, FractalRoughness, Fracture, LimbSpec, RealismSpec, TISSUE_INDEX,
    cross_section_limb, longitudinal_limb, random_body_type, random_fracture,
)
from wavesim.phantoms.roughness import RoughProfile
from wavesim.ultrasound import ElasticSolver, Source, line_receiver
from wavesim.ultrasound.medium import SCATTER_STRENGTH

DX_US = 1450 / 3e6 / 8  # malla de los ejemplos a 3 MHz
NEW_TISSUES = ("fascia", "periosteum")


# --------------------------------------------------------------------------
# Retrocompatibilidad
# --------------------------------------------------------------------------
def test_basic_preset_is_the_default_phantom():
    fr = random_fracture(np.random.default_rng(1), "desplazada", x_range=1e-3)
    kw = dict(dx=0.1e-3, width=24e-3, depth=26e-3, seed=2)
    a, b = longitudinal_limb("tibia", fr, **kw), longitudinal_limb("tibia", fr, realism="basico", **kw)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.porosity is None and b.porosity is None and "realism" not in b.meta
    c = cross_section_limb("pierna", dx=0.5e-3, seed=1)
    d = cross_section_limb("pierna", dx=0.5e-3, seed=1, realism=REALISM_PRESETS["basico"])
    np.testing.assert_array_equal(c.labels, d.labels)


def test_porosity_alone_keeps_labels_and_layers():
    """La porosidad vive en su mapa: las etiquetas (y las máscaras) son las del fantoma básico."""
    spec = RealismSpec("solo_poros", cortical=CorticalPorosity())
    kw = dict(dx=0.1e-3, width=24e-3, seed=4)
    a = longitudinal_limb("tibia", Fracture(), **kw)
    b = longitudinal_limb("tibia", Fracture(), realism=spec, **kw)
    np.testing.assert_array_equal(a.labels, b.labels)
    for k in a.masks:
        np.testing.assert_array_equal(a.masks[k], b.masks[k])
    assert np.all(np.isnan(b.porosity[~b.mask("cortical_bone")]))
    assert np.all(np.isfinite(b.porosity[b.mask("cortical_bone")]))
    # sin porosidad pedida no hay mapa: la cortical conserva las propiedades de su tabla
    assert longitudinal_limb("tibia", realism=RealismSpec("r", roughness=FractalRoughness()), **kw).porosity is None


def test_reference_porosity_reproduces_the_tables():
    """Una cortical con la porosidad de referencia tiene exactamente las propiedades de la tabla."""
    p = longitudinal_limb("tibia", dx=0.2e-3, width=10e-3, seed=0)
    getters = {
        "acoustic": lambda k: ACOUSTIC[k].density,
        "xray": lambda k: float(xray_tables.linear_attenuation(k, 60.0)),
        "dielectric": lambda k: complex(dielectric.complex_permittivity(k, 1e9)),
        "optical": lambda k: float(optical.mu_a(k, 800.0)),
        "mri": lambda k: tissue_signal(k, "T1"),
    }
    for physics, ref in porous.CORTICAL_REFERENCE_POROSITY.items():
        q = replace(p, porosity=np.full(p.grid.shape, ref, np.float32))
        getter = getters[physics]
        dtype = complex if physics == "dielectric" else float
        cort = p.mask("cortical_bone")
        np.testing.assert_allclose(q.property_map(getter, dtype, physics)[cort], getter("cortical_bone"), rtol=1e-7)


# --------------------------------------------------------------------------
# Porosidad cortical
# --------------------------------------------------------------------------
@pytest.mark.parametrize("body, requested", [("normal", 0.05), ("anciano_osteoporotico", 0.05 * 3.6)])
def test_cortical_porosity_matches_request(body, requested):
    measured = [longitudinal_limb("tibia", width=60e-3, depth=12e-3, dx=DX_US, seed=s, body=body,
                                  realism="realista").meta["microstructure"]["cortical_porosity"] for s in (0, 1)]
    assert np.mean(measured) == pytest.approx(requested, rel=0.12)


def test_cortical_porosity_grows_towards_endosteum_and_with_age():
    p = longitudinal_limb("tibia", width=60e-3, depth=12e-3, dx=DX_US, seed=0, realism="realista")
    old = longitudinal_limb("tibia", width=60e-3, depth=12e-3, dx=DX_US, seed=0, realism="realista",
                            body="anciano_osteoporotico")
    for ph in (p, old):
        cort = ph.mask("cortical_bone")
        z = ph.grid.z[:, None] * np.ones((1, ph.grid.nx))
        top, bottom = z[cort].min(), z[cort].max()
        outer = cort & (z < top + 0.3 * (bottom - top))
        inner = cort & (z > top + 0.7 * (bottom - top))
        assert ph.porosity[inner].mean() > 1.5 * ph.porosity[outer].mean()
    assert old.pore_fill == "marrow" and p.pore_fill == "blood"
    assert old.meta["realism"]["cortical"]["trabecularized"] == pytest.approx(0.25)


def test_explicit_and_effective_pores_give_the_same_mean_density():
    """Misma semilla, poros dibujados explícitos (malla fina) o como fracción por celda: misma densidad media."""
    kw = dict(width=12e-3, depth=12e-3, dx=20e-6, seed=3, body="anciano_osteoporotico")
    rho = {}
    for name, cells in (("explicit", 0.0), ("effective", np.inf)):
        p = longitudinal_limb("tibia", realism=replace(REALISM_PRESETS["realista"], resolve_cells=cells), **kw)
        cort = p.mask("cortical_bone")
        phi = p.porosity[cort]
        if name == "explicit":
            assert np.all((phi == 0) | (phi == 1)) and p.meta["microstructure"]["explicit_pores"] > 0
        else:
            assert np.any((phi > 0.05) & (phi < 0.95)) and p.meta["microstructure"]["explicit_pores"] == 0
        rho[name] = p.property_map(lambda k: ACOUSTIC[k].density, physics="acoustic")[cort].mean()
    assert rho["explicit"] == pytest.approx(rho["effective"], rel=0.005)
    assert rho["effective"] < 0.95 * ACOUSTIC["cortical_bone"].density


def test_pores_follow_the_displaced_fragment():
    fr = Fracture(displacement=1.5e-3, hematoma_thickness=0.0)
    p = longitudinal_limb("tibia", fr, width=24e-3, depth=16e-3, dx=0.05e-3, seed=2, realism="realista",
                          body="anciano_osteoporotico")
    cort = p.mask("cortical_bone")
    x = p.grid.x[None, :] * np.ones((p.grid.nz, 1))
    for side in (x < -2e-3, x > 2e-3):
        assert p.porosity[cort & side].mean() == pytest.approx(0.18, rel=0.35)
    assert p.has_fracture and np.all(p.labels[p.fracture_mask] == TISSUE_INDEX["hematoma"])


def test_cross_section_porosity():
    p = cross_section_limb("pierna", dx=0.25e-3, seed=1, realism="realista")
    old = cross_section_limb("pierna", dx=0.25e-3, seed=1, realism="realista", body="anciano_osteoporotico")
    assert p.meta["microstructure"]["cortical_porosity"] == pytest.approx(0.05, rel=0.15)
    assert old.meta["microstructure"]["cortical_porosity"] == pytest.approx(0.18, rel=0.15)


# --------------------------------------------------------------------------
# Red trabecular
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dx, body", [(0.05e-3, "normal"), (0.05e-3, "anciano_osteoporotico"), (0.2e-3, "normal")])
def test_trabecular_bvtv_matches_request(dx, body):
    p = longitudinal_limb("tibia_metafisis", width=20e-3, depth=30e-3, dx=dx, seed=1, realism="realista", body=body)
    spec = p.meta["realism"]["trabecular"]
    trab = p.mask("trabecular_bone")
    bvtv = 1.0 - p.porosity[trab].mean()
    assert bvtv == pytest.approx(spec["bvtv"], abs=0.005)
    info = p.meta["microstructure"]
    assert info["trabecular_explicit"] == (dx == 0.05e-3)
    if info["trabecular_explicit"]:
        assert np.all((p.porosity[trab] == 0) | (p.porosity[trab] == 1))
        # grosor del orden del pedido; separación en el rango medido (0.5-1.5 mm)
        assert info["trabecular_thickness"] == pytest.approx(spec["thickness"], rel=0.3)
        assert 0.5e-3 <= info["trabecular_separation"] <= 1.5e-3
    if body == "anciano_osteoporotico":
        assert spec["bvtv"] == pytest.approx(0.12)


def test_trabecular_network_is_oriented_along_the_bone_axis():
    p = longitudinal_limb("tibia_metafisis", width=20e-3, depth=30e-3, dx=0.05e-3, seed=2, realism="realista")
    trab = p.mask("trabecular_bone")
    solid = (trab & (p.porosity < 0.5)).astype(float)
    nz, nx = solid.shape

    def correlation(dj, di):
        a, b = solid[: nz - dj, : nx - di], solid[dj:, di:]
        m = trab[: nz - dj, : nx - di] & trab[dj:, di:]
        return np.corrcoef(a[m], b[m])[0, 1]

    # 0.3 mm (6 celdas): las trabéculas siguen el eje x más que la profundidad
    assert correlation(0, 6) > correlation(6, 0) + 0.05


def test_bone_bruise_struts_cross_the_edema():
    fr = random_fracture(np.random.default_rng(0), "contusion_osea")
    p = longitudinal_limb("tibia_metafisis", fr, width=30e-3, dx=0.05e-3, seed=1, realism="realista")
    healthy = longitudinal_limb("tibia_metafisis", None, width=30e-3, dx=0.05e-3, seed=1, realism="realista")
    zone = p.masks["edema"] | p.masks["fracture"]
    rows, cols = np.nonzero(zone)
    box = np.zeros_like(zone)
    box[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
    struts = p.mask("trabecular_bone") & (p.porosity < 0.5) & box
    healthy_struts = healthy.mask("trabecular_bone") & (healthy.porosity < 0.5) & box
    assert np.count_nonzero(struts) > 0.5 * np.count_nonzero(healthy_struts)
    assert not (p.mask("trabecular_bone") & p.masks["edema"]).any()


# --------------------------------------------------------------------------
# Rugosidad multiescala
# --------------------------------------------------------------------------
def test_power_law_roughness_has_requested_rms_and_slope():
    length, rms, xi, hurst, ds = 40e-3, 30e-6, 0.5e-3, 0.7, 10e-6
    s = np.arange(0, length, ds)
    k = 2 * np.pi * np.fft.rfftfreq(s.size, ds)
    # Periódico (un tramo abierto tendría un salto entre sus extremos que añade una cola k⁻²) y
    # promediado sobre 8 realizaciones: el periodograma de una sola tiene un error de ±0.25 en la pendiente.
    psd = np.zeros(k.size)
    for seed in range(8):
        h = RoughProfile(length, rms, xi, np.random.default_rng(seed), periodic=True, spectrum="power_law",
                         hurst=hurst)(s)
        assert h.std() == pytest.approx(rms, rel=0.05)
        psd += np.abs(np.fft.rfft(h)) ** 2
    band = (k > 10 / xi) & (k < 2 * np.pi / 80e-6)
    slope = np.polyfit(np.log(k[band]), np.log(psd[band]), 1)[0]
    assert slope == pytest.approx(-(1 + 2 * hurst), abs=0.1)
    # la gaussiana de siempre apenas tiene rugosidad por debajo de ξ/4
    g = RoughProfile(length, rms, xi, np.random.default_rng(0), periodic=True)(s)
    fine = k > 8 / xi

    def fine_fraction(profile):
        p = np.abs(np.fft.rfft(profile)) ** 2
        return p[fine].sum() / p.sum()

    assert fine_fraction(h) > 0.02 > 100 * fine_fraction(g)


# --------------------------------------------------------------------------
# Tejido blando
# --------------------------------------------------------------------------
def test_new_tissues_are_in_every_table():
    for k in NEW_TISSUES:
        assert k in TISSUES and k in ACOUSTIC and k in mri.MRI and k in optical.OPTICAL
        assert k in SCATTER_STRENGTH and k in FAT_FRACTION
        assert k in dielectric.COLE_COLE or k in dielectric.MIXTURES
        assert k in xray_tables.COMPOSITION or k in xray_tables.VOLUME_MIXTURES
        eps, sigma = dielectric.permittivity_and_conductivity(k, 1e9)
        assert 30 < eps < 60 and 0.3 < sigma < 1.5  # entre la grasa y el músculo, como el tendón
        assert xray_tables.linear_attenuation("muscle", 60) < xray_tables.linear_attenuation(k, 60)
    # ecogénicas: reflejan más frente a la grasa que el músculo y son lo más dispersivo de las partes
    # blandas; en RM salen oscuras
    from wavesim.materials import reflection_coefficient
    assert reflection_coefficient("fat", "fascia") > reflection_coefficient("fat", "muscle")
    assert SCATTER_STRENGTH["fascia"] > max(SCATTER_STRENGTH[k] for k in ("skin", "fat", "muscle"))
    assert tissue_signal("fascia", "T2") < tissue_signal("muscle", "T2")


def test_soft_tissue_structures_in_the_longitudinal_cut():
    p = longitudinal_limb("antebrazo", width=24e-3, dx=DX_US, seed=1, realism="realista")
    fas, per = p.mask("fascia"), p.mask("periosteum")
    assert per.any() and fas.any()
    # periostio pegado a la cortical: justo encima de la cortical cercana en casi todas las columnas
    cort = p.mask("cortical_bone")
    top = np.argmax(cort, axis=0)
    above = np.array([per[max(j - 3, 0):j, i].any() for i, j in enumerate(top)])
    assert above.mean() > 0.8
    # perimisio: una fracción del músculo del orden de grosor / separación × presencia
    soft = p.meta["realism"]["soft"]
    expected = soft["perimysium_thickness"] / soft["perimysium_spacing"] * 0.75
    z = p.grid.z[:, None]
    far_muscle = (z > p.meta["bone_bottom_depth"] + 2e-3) & (z < p.meta["bone_bottom_depth"] + 15e-3)
    region = far_muscle & (p.mask("muscle") | fas)
    frac = np.count_nonzero(fas & region) / np.count_nonzero(region)
    assert 0.5 * expected < frac < 2.0 * expected


def test_soft_tissue_structures_in_the_cross_section():
    p = cross_section_limb("pierna", dx=0.2e-3, seed=1, realism="realista")
    f = p.tissue_fractions()
    assert f["fascia"] > 0.01 and f["periosteum"] > 0.001
    assert f["muscle"] > 0.8 * cross_section_limb("pierna", dx=0.2e-3, seed=1).tissue_fractions()["muscle"]


# --------------------------------------------------------------------------
# Física
# --------------------------------------------------------------------------
def test_porosity_reaches_every_modality():
    kw = dict(width=20e-3, depth=12e-3, dx=0.1e-3, seed=0)
    basic = longitudinal_limb("tibia", body="anciano_osteoporotico", **kw)
    real = longitudinal_limb("tibia", body="anciano_osteoporotico", realism="realista", **kw)
    cort = real.mask("cortical_bone")

    def cortex(p, getter, physics, dtype=float):
        return np.mean(p.property_map(getter, dtype, physics)[p.mask("cortical_bone")])

    rho = lambda k: ACOUSTIC[k].density  # noqa: E731
    assert cortex(real, rho, "acoustic") < 0.96 * cortex(basic, rho, "acoustic")
    spectrum = xray.tube_spectrum(120.0, 6.0)
    coef = {k: xray.basis_coefficients(k, spectrum) for k in real.property_tissues()}
    assert cortex(real, lambda k: coef[k][1], "xray") < 0.9 * cortex(basic, lambda k: coef[k][1], "xray")
    eps = lambda k: complex(dielectric.complex_permittivity(k, 2.45e9))  # noqa: E731
    assert abs(cortex(real, eps, "dielectric", complex) - cortex(basic, eps, "dielectric", complex)) > 0.5
    t1 = lambda k: tissue_signal(k, "T1")  # noqa: E731
    assert cortex(real, t1, "mri") > 5 * cortex(basic, t1, "mri") + 0.02  # médula grasa en los poros
    assert real.porosity[cort].mean() > 0.1


def _transmission(realism, seed: int) -> float:
    """Amplitud coherente a f0 de una onda plana de 3 MHz tras cruzar 4 mm de cortical (de agua a médula).

    Las interfaces de gran escala son lisas: la rugosidad endóstica básica
    (0.25 mm rms) desfasa ~2 rad la onda a 3 MHz y su realización dominaría
    la amplitud coherente. Así sólo cuentan los poros y la rugosidad fina.
    """
    f0 = 3e6
    spec = LimbSpec(name="placa", standoff=2.5e-3, skin=0.0, fat=0.0, muscle=0.0, cortex=4e-3, medulla=20e-3,
                    muscle_far=0.0, vessels=(), periosteal_rough=(0.0, 0.4e-3), endosteal_rough=(0.0, 1.5e-3))
    p = longitudinal_limb(spec, width=8e-3, depth=12e-3, dx=DX_US, seed=seed, realism=realism)
    g = p.grid
    s = ElasticSolver(p, f0, scattering=0.0)
    nt = int(9e-6 / s.dt)
    t = s.time_axis(nt)
    s.add_source(Source(np.full(g.nx, g.index_of(0, 1.6e-3)[0]), np.arange(g.nx), np.zeros(g.nx, int),
                        gaussian_pulse(t, f0)[None, :]))
    s.add_receiver("tx", line_receiver(g, 9.5e-3, -2e-3, 2e-3))
    tr = s.run(nt, track_peak=False).traces["tx"].mean(0)
    return abs(np.sum(tr * np.exp(-2j * np.pi * f0 * t)))


def test_realistic_cortex_attenuates_more_ultrasound():
    """Los poros dispersan: la onda coherente que cruza la cortical sale más débil, y más con la osteoporosis."""
    seed = 2
    basic = _transmission(None, seed)
    healthy = _transmission("realista", seed)
    old = REALISM_PRESETS["realista"].scaled(**{k: getattr(BODY_TYPES["anciano_osteoporotico"], k)
                                                for k in ("porosity", "pore_size", "trabecularization", "bvtv",
                                                          "pore_fill")})
    osteoporotic = _transmission(old, seed)
    loss_healthy = 20 * np.log10(basic / healthy)
    loss_old = 20 * np.log10(basic / osteoporotic)
    # sana: 4-7 dB más en 4 mm a 3 MHz según la semilla (≈ 3-5 dB/(cm·MHz) sobre la tabla)
    assert 1.0 < loss_healthy < 9.0
    assert loss_old > loss_healthy + 6.0


def test_random_body_type_keeps_bone_quality():
    b = random_body_type(np.random.default_rng(0), ["anciano_osteoporotico"])
    assert b.porosity == BODY_TYPES["anciano_osteoporotico"].porosity and b.pore_fill == "marrow"
