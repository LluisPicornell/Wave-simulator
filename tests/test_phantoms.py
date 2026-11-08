import hashlib
from dataclasses import replace

import numpy as np
import pytest

from wavesim.analysis import residual_db, xcorr_delay
from wavesim.datasets.generate import make_phantom
from wavesim.phantoms import (
    BODY_TYPES, CORTICAL_FRACTURE_KINDS, CROSS_PRESETS, FRACTURE_KINDS, INTERNAL_FRACTURE_KINDS, PRESETS, Fracture,
    RingFracture, TISSUE_INDEX, cross_section_limb, longitudinal_limb, random_fracture, random_limb_spec,
    random_ring_fracture, with_body_type,
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


# --------------------------------------------------------------------------
# Retrocompatibilidad: huellas de las etiquetas calculadas con el código
# anterior a las fracturas internas y los tipos de cuerpo (commit 6a0d034).
# --------------------------------------------------------------------------
LEGACY_DIGESTS = {
    "transversal": "7ce2f80e7c8151a1",
    "oblicua": "a23c994b2ef18dc5",
    "fisura": "ee45776cda2eb50a",
    "conminuta": "72cd473e2fb2e0f1",
    "incompleta": "20ebb0337d4bba9a",
    "desplazada": "b27b614e63d4feea",
}


def _digest(p):
    h = hashlib.sha256(p.labels.tobytes())
    for k in ("fracture", "hematoma", "edema"):
        h.update(p.masks[k].tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("kind", CORTICAL_FRACTURE_KINDS)
def test_legacy_fractures_are_unchanged(kind):
    fr = random_fracture(np.random.default_rng(1), kind, x_range=1e-3)
    p = longitudinal_limb("tibia", fr, dx=0.1e-3, width=24e-3, depth=26e-3, seed=2)
    assert _digest(p) == LEGACY_DIGESTS[kind]


def test_legacy_random_choice_and_other_phantoms_are_unchanged():
    fr = random_fracture(np.random.default_rng(5), kinds=CORTICAL_FRACTURE_KINDS)
    assert fr.kind == "incompleta" and fr.position == pytest.approx(0.004927052635783901, rel=1e-15)
    assert _digest(longitudinal_limb("antebrazo", fr, dx=0.1e-3, width=24e-3, seed=3)) == "455f41ab0ef532cb"
    assert _digest(longitudinal_limb("tibia", Fracture(), dx=0.1e-3, width=24e-3, seed=0)) == "5d10d8cf74178c74"
    healthy = {"tibia": "2ddabb5b3ab3bd00", "antebrazo": "3cbdafdcca6098e5", "muslo": "e7cc96eebf3a4dd0"}
    for preset, d in healthy.items():
        assert _digest(longitudinal_limb(preset, dx=0.2e-3, width=24e-3, seed=4)) == d
    cross = {"pierna": "943567ec54e32213", "antebrazo": "121f7387c70c67ec", "muslo": "517046fe8971194f"}
    for preset, d in cross.items():
        assert _digest(cross_section_limb(preset, RingFracture(angle_deg=5), dx=0.5e-3, seed=1)) == d


# --------------------------------------------------------------------------
# Fracturas internas
# --------------------------------------------------------------------------
def _changed_near_periosteum(fr, preset="tibia", margin_cells=2, seed=3):
    """Celdas que la fractura cambia en la capa externa de la cortical cercana o por encima de ella."""
    kw = dict(dx=0.05e-3, width=24e-3, seed=seed)
    p, healthy = longitudinal_limb(preset, fr, **kw), longitudinal_limb(preset, None, **kw)
    top = np.argmax(healthy.labels == TISSUE_INDEX["cortical_bone"], axis=0)
    outer = np.arange(p.grid.nz)[:, None] <= top[None, :] + margin_cells
    return p, int(np.count_nonzero((p.labels != healthy.labels) & outer))


INTERNAL_CASES = [(k, s) for k in ("intracortical", "cortical_lejana", "contusion_osea") for s in range(3)]


@pytest.mark.parametrize("kind,seed", INTERNAL_CASES)
def test_internal_fractures_spare_the_near_periosteum(kind, seed):
    fr = random_fracture(np.random.default_rng(seed), kind, x_range=2e-3)
    preset = "tibia_metafisis" if kind == "contusion_osea" else "tibia"
    p, n_changed = _changed_near_periosteum(fr, preset)
    assert p.has_fracture
    assert not fr.opens_near_surface
    assert n_changed == 0
    # sin grieta abierta a la sonda no hay hematoma en las partes blandas de encima
    above = p.grid.z[:, None] < p.meta["bone_top_depth"] - 0.5e-3
    assert not (p.masks["hematoma"] & above).any()


def test_endosteal_stress_fracture_spares_periosteum_and_adds_reaction():
    fr = Fracture(kind="estres", span=(0.6, 1.0), gap=0.08e-3, min_gap_cells=1, hematoma_thickness=0.0,
                  endosteal_reaction=1e-3, reaction_length=15e-3, edema_depth=5e-3)
    p, n_changed = _changed_near_periosteum(fr)
    assert p.has_fracture and n_changed == 0
    healthy = longitudinal_limb("tibia", None, dx=0.05e-3, width=24e-3, seed=3)
    cort = TISSUE_INDEX["cortical_bone"]
    assert np.count_nonzero(p.labels == cort) > np.count_nonzero(healthy.labels == cort)  # esclerosis endóstica


def test_periosteal_stress_fracture_reaches_periosteum_with_new_bone():
    fr = Fracture(kind="estres", span=(0.0, 0.4), gap=0.08e-3, min_gap_cells=1, hematoma_thickness=0.0,
                  periosteal_reaction=1e-3, reaction_length=15e-3)
    p, n_changed = _changed_near_periosteum(fr)
    assert fr.opens_near_surface and n_changed > 0
    # hueso perióstico nuevo (trabecular) y su ribete de edema
    assert p.mask("trabecular_bone").any() and p.masks["edema"].any()


def test_thin_crack_is_continuous_with_min_gap_cells():
    fr = Fracture(span=(0.2, 0.8), gap=0.03e-3, hematoma_thickness=0.0, edema_length=0.0)
    kw = dict(dx=0.1e-3, width=24e-3, seed=1)
    exact = longitudinal_limb("tibia", fr, **kw).fracture_mask
    one_cell = longitudinal_limb("tibia", replace(fr, min_gap_cells=1), **kw).fracture_mask

    def rows(m):
        return np.count_nonzero(m.any(axis=1))

    # 0.6 · 4 mm de grieta ≈ 24 filas: con una celda mínima no queda punteada
    assert rows(one_cell) >= 20 and rows(exact) < rows(one_cell)


@pytest.mark.parametrize("span", [(2.0, 3.0), (2.3, 3.0)])
def test_far_cortex_fracture_breaks_the_far_cortex(span):
    fr = Fracture(kind="cortical_lejana", span=span, gap=0.3e-3, edema_depth=4e-3)
    p = longitudinal_limb("tibia", fr, dx=0.05e-3, width=24e-3, seed=2)
    m = p.fracture_mask
    j, i = np.nonzero(m)
    soft = [TISSUE_INDEX[t] for t in ("muscle", "fat", "hematoma")]
    medullary = [TISSUE_INDEX[t] for t in ("marrow", "edema", "trabecular_bone")]
    below_free = ~m[j + 1, i]
    above_free = ~m[j - 1, i]
    # sale al periostio lejano (a su hematoma o al músculo) y está toda por debajo de la médula
    assert np.isin(p.labels[j + 1, i][below_free], soft).any()
    assert p.grid.z[j].min() > p.meta["bone_top_depth"] + PRESETS["tibia"].cortex + 5e-3
    # llega al endostio lejano sólo si empieza en él (ζ0 = 2)
    assert np.isin(p.labels[j - 1, i][above_free], medullary).any() == (span[0] <= 2.0)
    assert (p.masks["hematoma"] & (p.grid.z[:, None] > p.grid.z[j].max())).any()


def test_bone_bruise_keeps_trabecular_frame():
    fr = random_fracture(np.random.default_rng(0), "contusion_osea")
    kw = dict(dx=0.1e-3, width=30e-3, seed=1)
    p = longitudinal_limb("tibia_metafisis", fr, **kw)
    healthy = longitudinal_limb("tibia_metafisis", None, **kw)
    trab = healthy.labels == TISSUE_INDEX["trabecular_bone"]
    zone = p.masks["edema"] | p.masks["fracture"]
    lost = trab & (p.labels != TISSUE_INDEX["trabecular_bone"])
    # dentro de la zona de contusión sólo una parte del hueso trabecular pasa a edema o sangre
    assert 0 < np.count_nonzero(lost) < 0.6 * np.count_nonzero(trab & _bbox(zone))
    assert np.all(p.labels[p.fracture_mask] == TISSUE_INDEX["hematoma"])


def _bbox(mask):
    jj, ii = np.nonzero(mask)
    out = np.zeros_like(mask)
    out[jj.min():jj.max() + 1, ii.min():ii.max() + 1] = True
    return out


@pytest.mark.parametrize("kind", INTERNAL_FRACTURE_KINDS)
def test_ring_internal_fractures(kind):
    fr = random_ring_fracture(np.random.default_rng(2), kind)
    p = cross_section_limb("pierna", fr, dx=0.2e-3, seed=1)
    healthy = cross_section_limb("pierna", None, dx=0.2e-3, seed=1)
    assert p.has_fracture
    b = CROSS_PRESETS["pierna"].bones[0]
    X, Z = p.grid.meshgrid()
    rb = np.hypot(X - b.center[0], Z - b.center[1])
    if kind in ("intracortical", "contusion_osea"):
        assert not ((p.labels != healthy.labels) & (rb > b.outer_radius - 0.1 * b.cortex)).any()
    if kind == "cortical_lejana":
        assert np.all(Z[p.fracture_mask] > b.center[1])  # en el lado opuesto a la sonda


# --------------------------------------------------------------------------
# Tipos de cuerpo
# --------------------------------------------------------------------------
def _near_side_thickness(p, tissue):
    """Grosor (m) de un tejido en la columna central, desde la piel hasta 15 mm bajo la cortical cercana."""
    col = p.labels[:, p.grid.nx // 2]
    top = int(np.argmax(col == TISSUE_INDEX["cortical_bone"]))
    seg = col[: top + int(15e-3 / p.grid.dx)] if tissue == "cortical_bone" else col[:top]
    return np.count_nonzero(seg == TISSUE_INDEX[tissue]) * p.grid.dx


def test_body_types_change_thicknesses_in_the_expected_direction():
    specs = {b: with_body_type("tibia", b) for b in BODY_TYPES}
    normal, old = specs["normal"], specs["anciano_osteoporotico"]
    assert specs["delgado"].fat < normal.fat < specs["sobrepeso"].fat < specs["obeso"].fat
    assert specs["musculoso"].muscle_far > normal.muscle_far > old.muscle_far
    assert specs["musculoso"].cortex > normal.cortex
    assert old.cortex < 0.7 * normal.cortex and old.medulla > normal.medulla  # resorción endóstica
    assert old.skin < normal.skin < specs["obeso"].skin
    # el diámetro exterior del hueso apenas cambia en la osteoporosis
    assert 2 * old.cortex + old.medulla == pytest.approx(2 * normal.cortex + normal.medulla, rel=0.05)
    ph = {b: longitudinal_limb("tibia", dx=0.05e-3, width=10e-3, seed=1, body=b)
          for b in ("normal", "obeso", "anciano_osteoporotico")}
    assert _near_side_thickness(ph["obeso"], "fat") == pytest.approx(3 * _near_side_thickness(ph["normal"], "fat"),
                                                                     rel=0.2)
    assert (_near_side_thickness(ph["anciano_osteoporotico"], "cortical_bone")
            < 0.75 * _near_side_thickness(ph["normal"], "cortical_bone"))


def test_normal_body_type_is_the_preset():
    for preset in PRESETS:
        a = longitudinal_limb(preset, dx=0.2e-3, width=20e-3, seed=5)
        b = longitudinal_limb(preset, dx=0.2e-3, width=20e-3, seed=5, body="normal")
        np.testing.assert_array_equal(a.labels, b.labels)


@pytest.mark.parametrize("body", list(BODY_TYPES))
def test_vessels_stay_in_soft_tissue_for_every_body(body):
    p = longitudinal_limb("tibia", dx=0.1e-3, width=24e-3, seed=0, body=body)
    _, _, r = PRESETS["tibia"].vessels[0]
    assert np.count_nonzero(p.mask("blood")) * p.grid.dx**2 > 0.8 * np.pi * r**2


def test_body_types_in_cross_section():
    normal = with_body_type(CROSS_PRESETS["pierna"], "normal")
    obese = with_body_type(CROSS_PRESETS["pierna"], "obeso")
    old = with_body_type(CROSS_PRESETS["pierna"], "anciano_osteoporotico")
    assert obese.radius > normal.radius and obese.fat == pytest.approx(3 * normal.fat)
    assert old.bones[0].cortex < 0.7 * normal.bones[0].cortex
    frac = {b: cross_section_limb("pierna", dx=0.5e-3, seed=1, body=b).tissue_fractions() for b in ("normal", "obeso")}
    assert frac["obeso"]["fat"] > 1.5 * frac["normal"]["fat"]


def test_random_limb_spec_samples_body_types():
    rng = np.random.default_rng(0)
    assert len({random_limb_spec(rng, "tibia").body for _ in range(40)}) >= 5
    obese = [random_limb_spec(rng, "tibia", body="obeso") for _ in range(10)]
    assert all(s.body == "obeso" and s.fat > 2 * PRESETS["tibia"].fat for s in obese)
    assert longitudinal_limb(obese[0], dx=0.2e-3, width=20e-3).meta["body_type"] == "obeso"


def test_dataset_phantoms_with_body_and_internal_kinds():
    rng = np.random.default_rng(4)
    for _ in range(8):
        p = make_phantom(rng, "tibia", True, 0.2e-3, 24e-3, 22e-3, body="obeso",
                         fracture_kinds=INTERNAL_FRACTURE_KINDS)
        assert p.has_fracture and p.meta["body_type"] == "obeso"
        # la cortical lejana de una pierna obesa queda fuera de los 22 mm de imagen
        assert p.meta["fracture"]["kind"] in INTERNAL_FRACTURE_KINDS and p.meta["fracture"]["kind"] != "cortical_lejana"


def test_xcorr_delay_and_residual():
    dt = 1e-8
    t = np.arange(2000) * dt
    a = np.exp(-(((t - 8e-6) / 1e-6) ** 2)) * np.sin(2 * np.pi * 1e6 * t)
    b = np.exp(-(((t - 8.0123e-6) / 1e-6) ** 2)) * np.sin(2 * np.pi * 1e6 * (t - 0.0123e-6))
    assert xcorr_delay(a, b, dt) == pytest.approx(12.3e-9, abs=0.5e-9)
    assert residual_db(a, 0.9 * a) == pytest.approx(-20.0, abs=1e-6)
