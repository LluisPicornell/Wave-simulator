import numpy as np
import pytest

from wavesim.materials import ACOUSTIC, TISSUES, dielectric, mri, optical, xray
from wavesim.ultrasound.medium import SCATTER_STRENGTH


@pytest.mark.parametrize("table", [ACOUSTIC, mri.MRI, optical.OPTICAL, dielectric.WATER_FRACTION, SCATTER_STRENGTH])
def test_every_tissue_has_properties(table):
    assert set(TISSUES) <= set(table)


def test_dielectric_and_xray_cover_all_tissues():
    for k in TISSUES:
        eps = dielectric.complex_permittivity(k, 1e9)
        assert np.isfinite(eps)
        assert np.isfinite(xray.linear_attenuation(k, 60.0))


# Valores de referencia a 1 GHz (Gabriel 1996 / base de datos IT'IS)
@pytest.mark.parametrize(
    "tissue, eps_ref, sigma_ref",
    [
        ("muscle", 54.81, 0.978),
        ("fat", 5.45, 0.0535),
        ("cortical_bone", 12.36, 0.156),
        ("trabecular_bone", 20.58, 0.364),
        ("blood", 61.07, 1.58),
        ("skin", 40.94, 0.90),
    ],
)
def test_gabriel_reference_values_at_1ghz(tissue, eps_ref, sigma_ref):
    eps, sigma = dielectric.permittivity_and_conductivity(tissue, 1e9)
    assert eps == pytest.approx(eps_ref, rel=0.01)
    assert sigma == pytest.approx(sigma_ref, rel=0.02)


def test_microwave_penetration_ordering():
    # A 2.45 GHz (horno microondas) la grasa es casi transparente y el músculo no
    d_fat = dielectric.penetration_depth("fat", 2.45e9)
    d_muscle = dielectric.penetration_depth("muscle", 2.45e9)
    assert d_fat > 4 * d_muscle
    assert 5e-3 < d_muscle < 20e-3


def test_water_is_opaque_to_terahertz():
    assert dielectric.penetration_depth("water", 1e12) < 100e-6


def test_xray_basis_decomposition_reproduces_muscle_table():
    est = xray.decomposed_mass_attenuation(xray.COMPOSITION["muscle"])
    np.testing.assert_allclose(est, xray.TABLES["muscle"], rtol=0.015)


def test_xray_bone_attenuates_more_than_soft_tissue():
    for e in (30.0, 60.0, 100.0):
        mu = {k: xray.linear_attenuation(k, e) for k in ("fat", "muscle", "cortical_bone")}
        assert mu["fat"] < mu["muscle"] < mu["cortical_bone"]


def test_acoustic_impedance_reflection():
    # Hueso-músculo refleja mucho más que grasa-músculo
    from wavesim.materials import reflection_coefficient

    assert abs(reflection_coefficient("muscle", "cortical_bone")) > 0.5
    assert abs(reflection_coefficient("fat", "muscle")) < 0.15


def test_optical_window_in_near_infrared():
    # La "ventana terapéutica" (650-950 nm) penetra más que el verde o el infrarrojo de 1450 nm
    eff = {lam: optical.mu_effective("muscle", lam) for lam in (550, 800, 1450)}
    assert eff[800] < eff[550]
    assert eff[800] < eff[1450]
