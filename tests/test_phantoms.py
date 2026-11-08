import numpy as np
import pytest

from wavesim.phantoms import (
    CROSS_PRESETS, FRACTURE_KINDS, PRESETS, Fracture, RingFracture, TISSUE_INDEX, cross_section_limb,
    longitudinal_limb, random_fracture,
)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_longitudinal_presets_build(preset):
    p = longitudinal_limb(preset, dx=0.2e-3)
    assert {"skin", "fat", "cortical_bone", "marrow"} <= set(p.present_tissues())
    assert not p.has_fracture


@pytest.mark.parametrize("preset", list(CROSS_PRESETS))
def test_cross_section_presets_build(preset):
    p = cross_section_limb(preset, dx=0.5e-3)
    assert {"skin", "fat", "muscle", "cortical_bone"} <= set(p.present_tissues())


def test_layer_thickness_matches_spec():
    spec = PRESETS["tibia"]
    p = longitudinal_limb(spec, dx=0.05e-3)
    col = p.labels[:, p.grid.nx // 2]
    cort = np.count_nonzero(col == TISSUE_INDEX["cortical_bone"]) * p.grid.dx
    # dos corticales (cercana y lejana) ± rugosidad endóstica
    assert cort == pytest.approx(2 * spec.cortex, abs=1.2e-3)


def test_same_seed_same_anatomy_at_any_resolution():
    a = longitudinal_limb("antebrazo", Fracture(), dx=0.1e-3, seed=4).tissue_fractions()
    b = longitudinal_limb("antebrazo", Fracture(), dx=0.05e-3, seed=4).tissue_fractions()
    for k in ("fat", "muscle", "cortical_bone", "marrow"):
        assert a[k] == pytest.approx(b[k], abs=0.01)


def test_reproducible_with_seed():
    a = longitudinal_limb("tibia", Fracture(angle_deg=20), dx=0.1e-3, seed=7)
    b = longitudinal_limb("tibia", Fracture(angle_deg=20), dx=0.1e-3, seed=7)
    np.testing.assert_array_equal(a.labels, b.labels)


@pytest.mark.parametrize("kind", FRACTURE_KINDS)
def test_every_fracture_kind_cracks_the_cortex(kind):
    fr = random_fracture(np.random.default_rng(3), kind)
    p = longitudinal_limb("tibia", fr, dx=0.05e-3, seed=1)
    assert p.has_fracture
    assert p.masks["hematoma"].any()
    # la grieta está dentro de la zona que antes era hueso
    assert np.all(p.labels[p.fracture_mask] == TISSUE_INDEX["hematoma"])


def test_displacement_creates_cortical_step():
    p = longitudinal_limb("tibia", Fracture(displacement=2e-3, hematoma_thickness=0), dx=0.05e-3, seed=2)
    g = p.grid
    cort = p.labels == TISSUE_INDEX["cortical_bone"]

    def top(x):
        i = g.index_of(x, 0)[1]
        return g.z[np.argmax(cort[:, i])]

    assert top(5e-3) - top(-5e-3) == pytest.approx(2e-3, abs=0.4e-3)


def test_ring_fracture_and_edema():
    p = cross_section_limb("pierna", RingFracture(angle_deg=0), dx=0.2e-3)
    assert p.has_fracture and p.masks["edema"].any()
