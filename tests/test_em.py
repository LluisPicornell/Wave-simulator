"""Validación de los modelos electromagnéticos (Maxwell, rayos X, óptica, espectro)."""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.em import MaxwellTMSolver, optics, recommended_dx, spectrum, xray
from wavesim.em.projection import detector_axis, fbp, project
from wavesim.materials import dielectric
from wavesim.materials import xray as xd
from wavesim.phantoms import cross_section_limb, longitudinal_limb, RingFracture
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom


def _uniform(g, tissue):
    return Phantom(g, np.full(g.shape, TISSUE_INDEX[tissue], np.int16))


@pytest.mark.parametrize("tissue", ["muscle", "fat"])
def test_maxwell_plane_wave_matches_complex_permittivity(tissue):
    f0 = 2.45e9
    dx = recommended_dx(f0, [tissue], ppw=20)
    beta, alpha = dielectric.wave_numbers(tissue, f0)
    depth = 5 * 2 * np.pi / beta + 60 * dx
    g = Grid2D.from_size(20 * dx, depth, dx, x0=-10 * dx)
    s = MaxwellTMSolver(_uniform(g, tissue), f0)
    s.set_plane_wave_source(z=g.z0 + 25 * dx)
    r = s.run()
    col = r.phasor[:, g.nx // 2]
    sel = (g.z > g.z0 + 35 * dx) & (g.z < g.z0 + g.depth - 30 * dx)
    a_fit = -np.polyfit(g.z[sel], np.log(np.abs(col[sel])), 1)[0]
    b_fit = abs(np.polyfit(g.z[sel], np.unwrap(np.angle(col[sel])), 1)[0])
    assert b_fit == pytest.approx(beta, rel=0.02)
    assert a_fit == pytest.approx(alpha, rel=0.05)


def test_fbp_reconstructs_uniform_disk():
    g = Grid2D.from_size(60e-3, 60e-3, 0.4e-3, x0=-30e-3, z0=-30e-3)
    X, Z = g.meshgrid()
    mu = np.where(np.hypot(X, Z) <= 20e-3, 20.0, 0.0)
    det = detector_axis(g)
    ang = np.arange(180.0)
    rec = fbp(project(mu, g, ang, det)[0], ang, det, g)
    assert rec[np.hypot(X, Z) < 15e-3].mean() == pytest.approx(20.0, rel=0.005)
    assert abs(rec[np.hypot(X, Z) > 25e-3].mean()) < 0.1


def test_beer_lambert_slab():
    g = Grid2D.from_size(20e-3, 60e-3, 0.25e-3, x0=-10e-3)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[(g.z >= 10e-3) & (g.z < 60e-3), :] = TISSUE_INDEX["water"]
    p = Phantom(g, lab)
    proj = xray.radiograph(p, 0.0, spectrum=xray.monoenergetic(60.0), photons=None)
    center = proj.line_integrals[0, proj.det.size // 2]
    expected = xd.linear_attenuation("water", 60.0) * 0.05 + xd.linear_attenuation("air", 60.0) * 0.01
    assert center == pytest.approx(expected, rel=0.01)


def test_ct_hounsfield_units_monoenergetic():
    p = cross_section_limb("pierna", dx=0.4e-3).replace_tissue("water", "air")
    ct = xray.ct_scan(p, n_angles=180, spectrum=xray.monoenergetic(60.0), photons=None)
    hu = {k: float(np.median(ct.hu[p.mask(k)])) for k in ("air", "muscle", "fat", "cortical_bone")}
    mu_w = xd.linear_attenuation("water", 60.0)
    expect = {k: 1000 * (xd.linear_attenuation(k, 60.0) - mu_w) / mu_w for k in ("muscle", "fat", "cortical_bone")}
    assert hu["air"] == pytest.approx(-1000, abs=30)
    assert hu["muscle"] == pytest.approx(expect["muscle"], abs=15)
    assert hu["fat"] == pytest.approx(expect["fat"], abs=20)
    assert hu["cortical_bone"] == pytest.approx(expect["cortical_bone"], rel=0.05)


def test_ct_shows_fracture_gap():
    p = cross_section_limb("pierna", RingFracture(angle_deg=0, gap=1.0e-3), dx=0.25e-3).replace_tissue("water", "air")
    ct = xray.ct_scan(p, n_angles=180, photons=None)
    crack = ct.hu[p.fracture_mask].mean()
    bone = np.median(ct.hu[p.mask("cortical_bone")])
    assert crack < 0.6 * bone


def test_diffusion_decay_matches_effective_attenuation():
    g = Grid2D.from_size(40e-3, 40e-3, 0.2e-3, x0=-20e-3, z0=-20e-3)
    p = _uniform(g, "muscle")
    lam = 800.0
    r = optics.solve_diffusion(p, lam, source=(0.0, 0.0))
    prof = r.fluence[g.nz // 2, g.nx // 2:]
    x = g.x[g.nx // 2:]
    sel = (x > 4e-3) & (x < 12e-3)
    # Solución 2D: K0(mu_eff r) ~ exp(-mu_eff r)/sqrt(r)
    slope = -np.polyfit(x[sel], np.log(prof[sel] * np.sqrt(x[sel])), 1)[0]
    from wavesim.materials import optical

    assert slope == pytest.approx(float(optical.mu_effective("muscle", lam)), rel=0.05)


def test_spectrum_bands_and_penetration_ordering():
    assert spectrum.band_of(2.45e9).name == "Microondas"
    assert spectrum.band_of(60 * spectrum.HZ_PER_KEV).name == "Rayos X"
    d = {name: spectrum.penetration_depth("muscle", f) for name, f in [
        ("thz", 1e12), ("nir", spectrum.C0 / 800e-9), ("mw", 2.45e9), ("xray", 60 * spectrum.HZ_PER_KEV)]}
    assert d["thz"] < 1e-3
    assert d["thz"] < d["nir"] < d["mw"] < d["xray"]
    assert d["xray"] == pytest.approx(1 / (0.2048 * 1.05 * 100), rel=0.02)


def test_through_limb_losses_are_finite_and_ordered():
    p = longitudinal_limb("antebrazo", dx=0.2e-3, width=10e-3)
    layers = spectrum.column_layers(p)
    loss = {f: spectrum.through_transmission_db(layers, f)["total_db"] for f in (2.45e9, 1e12, 60 * spectrum.HZ_PER_KEV)}
    assert loss[60 * spectrum.HZ_PER_KEV] < loss[2.45e9] < loss[1e12]
