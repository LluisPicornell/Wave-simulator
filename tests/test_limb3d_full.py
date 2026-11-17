"""Fantomas 3D de extremidad completos (``phantoms.limb3d_full``)."""

from dataclasses import replace

import numpy as np
import pytest

from wavesim.phantoms import (
    BODY_TYPES, CROSS_PRESETS, FRACTURE_KINDS_3D, LIMB3D_PRESETS, Fracture, Fracture3D, Metaphysis,
    cross_section_limb, equivalent_longitudinal_spec, limb_3d, limb_3d_fractions, longitudinal_limb,
    random_fracture_3d, random_limb_3d_spec,
)
from wavesim.phantoms.phantom import TISSUE_INDEX

NAMES = {v: k for k, v in TISSUE_INDEX.items()}


def _runs(col):
    """[(tejido, índice de la primera celda)] de una columna de etiquetas."""
    out = []
    for i, lab in enumerate(col):
        if not out or out[-1][0] != NAMES[int(lab)]:
            out.append((NAMES[int(lab)], i))
    return out


def _centroid(ph, mask):
    k, j, i = np.nonzero(mask)
    g = ph.grid
    return g.x0 + i.mean() * g.dx, g.y0 + j.mean() * g.dx, g.z0 + k.mean() * g.dx


# ----------------------------------------------------------------------------
# Anatomía, semilla y resolución
# ----------------------------------------------------------------------------
def test_pierna_tiene_tibia_y_perone_en_su_sitio():
    ph = limb_3d("pierna", length=2e-3, dx=0.5e-3)
    g = ph.grid
    assert ph.labels.dtype == np.int16 and ph.meta["bone_axis"] == "x"
    assert set(ph.present_tissues()) == {"water", "skin", "fat", "muscle", "blood", "cortical_bone", "marrow"}
    (ty, tz), (py, pz) = ph.meta["bone_centers"]
    assert ty == 0.0 and pz > tz and py > 15e-3  # peroné lateral y posterior a la tibia
    # La tibia está casi bajo la piel: entre la piel y la cortical sólo hay piel y grasa
    col = ph.labels[:, g.index_of(0, 0, 0)[1], g.nx // 2]
    assert [t for t, _ in _runs(col)][:5] == ["water", "skin", "fat", "cortical_bone", "marrow"]
    assert ph.meta["bone_top_depth"] - ph.meta["skin_depth"] < 8e-3
    # cortical: anillos de tibia y peroné, π(R² − r²) cada uno
    ring = ph.mask("cortical_bone")[:, :, 0].sum() * g.dx**2
    bones = CROSS_PRESETS["pierna"].bones
    expected = sum(np.pi * (b.outer_radius**2 - (b.outer_radius - b.cortex) ** 2) for b in bones)
    assert ring == pytest.approx(expected, rel=0.05)
    # margen de fondo alrededor del miembro
    assert (ph.labels[:3] == TISSUE_INDEX["water"]).all() and (ph.labels[:, :2] == TISSUE_INDEX["water"]).all()


def test_presets_y_fondo_de_aire():
    for name, bones in (("antebrazo", 2), ("muslo", 1)):
        ph = limb_3d(name, length=3e-3, dx=1e-3, background="air")
        assert len(ph.meta["bone_centers"]) == bones
        assert ph.labels[0, 0, 0] == TISSUE_INDEX["air"] and "water" not in ph.present_tissues()
        assert ph.mask("cortical_bone").any()


def test_misma_semilla_misma_anatomia_a_cualquier_dx():
    fr = Fracture3D(kind="desplazada", position=1e-3, angle_deg=20.0, displacement=1.5e-3, roughness=0.3e-3)
    kw = dict(length=8e-3, seed=7)
    a = limb_3d("pierna", fr, dx=1e-3, **kw)
    b = limb_3d("pierna", fr, dx=1e-3, **kw)
    assert (a.labels == b.labels).all() and all((a.masks[k] == b.masks[k]).all() for k in a.masks)
    c = limb_3d("pierna", fr, dx=1e-3, length=8e-3, seed=8)
    assert (a.labels != c.labels).any()  # otra semilla, otra rugosidad
    fine = limb_3d("pierna", fr, dx=0.5e-3, **kw)
    # las celdas gruesas son celdas finas (x = 0, y = 0 y z = 0 son centros de celda en ambas mallas)
    ga, gf = a.grid, fine.grid
    i = np.round((ga.x - gf.x0) / gf.dx).astype(int)
    j = np.round((ga.y - gf.y0) / gf.dx).astype(int)
    k = np.round((ga.z - gf.z0) / gf.dx).astype(int)
    ok_i, ok_j, ok_k = (i >= 0) & (i < gf.nx), (j >= 0) & (j < gf.ny), (k >= 0) & (k < gf.nz)
    sub = fine.labels[np.ix_(k[ok_k], j[ok_j], i[ok_i])]
    ref = a.labels[np.ix_(ok_k, ok_j, ok_i)]
    assert (sub == ref).mean() > 0.999
    assert (fine.masks["hematoma"][np.ix_(k[ok_k], j[ok_j], i[ok_i])] == a.masks["hematoma"][np.ix_(ok_k, ok_j, ok_i)]
            ).mean() > 0.999


def test_anatomia_aleatoria_reproducible_y_dentro_del_rango():
    s1 = random_limb_3d_spec(np.random.default_rng(3), "pierna", body="normal")
    s2 = random_limb_3d_spec(np.random.default_rng(3), "pierna", body="normal")
    assert s1 == s2
    ref = CROSS_PRESETS["pierna"]
    specs = [random_limb_3d_spec(np.random.default_rng(s), "pierna", body="normal") for s in range(30)]
    for s in specs:
        # tipo de cuerpo (σ ≈ 10 %) más la variación individual ±10 %
        assert 0.6 < s.cross.fat / ref.fat < 1.6
        assert 0.6 < s.cross.bones[0].cortex / ref.bones[0].cortex < 1.6
    assert np.std([s.bone_tilt_deg for s in specs]) > 0.5 and np.std([s.rotation_deg for s in specs]) > 1
    ph = limb_3d(specs[0], length=3e-3, dx=1e-3, seed=3)
    assert ph.meta["bone_centers"][0][0] == pytest.approx(0.0, abs=1e-12)  # la tibia sigue en y = 0


def test_metafisis_ensancha_el_hueso_con_trabecular():
    ph = limb_3d("pierna_metafisis", length=40e-3, dx=1e-3, roughness=False)
    g = ph.grid
    dia, met = g.index_of(-18e-3, 0, 0)[2], g.index_of(18e-3, 0, 0)[2]
    trab = ph.mask("trabecular_bone")
    assert not trab[:, :, dia].any() and trab[:, :, met].sum() > 100
    bone = np.isin(ph.labels, [TISSUE_INDEX[t] for t in ("cortical_bone", "trabecular_bone", "marrow")])
    j0 = g.index_of(0, 0, 0)[1]
    top = [np.argmax(bone[:, j0, i]) for i in (dia, met)]
    height = [bone[:, j0, i].sum() for i in (dia, met)]
    assert abs(top[0] - top[1]) <= 1  # la cara subcutánea no se mueve
    assert height[1] / height[0] == pytest.approx(1.68, rel=0.1)


# ----------------------------------------------------------------------------
# Coherencia con los cortes 2D
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("preset,body", [("pierna", None), ("pierna", "obeso"), ("antebrazo", None),
                                         ("muslo", "delgado")])
def test_corte_y0_coincide_con_el_longitudinal_2d(preset, body):
    dx = 0.5e-3
    ph = limb_3d(preset, length=4e-3, dx=dx, roughness=False, body=body)
    l2 = longitudinal_limb(equivalent_longitudinal_spec(preset, body), width=4e-3, dx=dx, depth=ph.grid.nz * dx)
    r3 = _runs(ph.slice_y(0.0).labels[:, ph.grid.nx // 2])
    r2 = _runs(l2.labels[:, l2.grid.nx // 2])
    assert [t for t, _ in r3] == [t for t, _ in r2]
    assert all(abs(a[1] - b[1]) <= 1 for a, b in zip(r3, r2))


@pytest.mark.parametrize("preset,body", [("pierna", None), ("antebrazo", "obeso")])
def test_corte_x0_coincide_con_el_transversal_2d(preset, body):
    dx = 0.5e-3
    ph = limb_3d(preset, length=2e-3, dx=dx, roughness=False, body=body)
    zero = (0.0, 1e-3)
    cs = replace(CROSS_PRESETS[preset], skin_rough=zero, soft_rough=zero, periosteal_rough=zero, endosteal_rough=zero)
    c2 = cross_section_limb(cs, dx=dx, body=body)
    sx = ph.slice_x(0.0)
    assert sx.meta["view"] == "cross_section" and sx.labels.shape == (ph.grid.nz, ph.grid.ny)
    yl, zl = ph.meta["leg_center"]
    Y, Z = np.meshgrid(sx.grid.x - yl, sx.grid.z - zl)
    ii = np.round((Y - c2.grid.x0) / dx).astype(int)
    jj = np.round((Z - c2.grid.z0) / dx).astype(int)
    leg = np.hypot(Y, Z) <= ph.meta["leg_radius"]
    agree = (c2.labels[jj[leg], ii[leg]] == sx.labels[leg]).mean()
    assert agree > 0.95  # las mallas difieren en un desplazamiento de subcelda
    for t in c2.present_tissues():
        a3, a2 = (sx.labels == TISSUE_INDEX[t]).sum(), (c2.labels == TISSUE_INDEX[t]).sum()
        assert a3 == pytest.approx(a2, rel=0.05, abs=4), t


def test_corte_coronal():
    ph = limb_3d("pierna", length=6e-3, dx=1e-3)
    sz = ph.slice_z(ph.meta["axis_depth"])
    assert sz.labels.shape == (ph.grid.ny, ph.grid.nx) and sz.meta["view"] == "coronal"
    assert (sz.labels[:, 0] == TISSUE_INDEX["marrow"]).any()


# ----------------------------------------------------------------------------
# Tipos de cuerpo
# ----------------------------------------------------------------------------
def test_grosores_por_tipo_de_cuerpo():
    dx = 0.5e-3
    thick = {}
    for body in ("delgado", "normal", "obeso", "anciano_osteoporotico"):
        ph = limb_3d("pierna", length=2e-3, dx=dx, roughness=False, body=body)
        r = _runs(ph.slice_y(0).labels[:, ph.grid.nx // 2])
        t = {name: (r[n + 1][1] - start) * dx for n, (name, start) in enumerate(r[:-1])}
        # grasa posterior (el penúltimo tramo antes de la piel) y cortical anterior
        t["fat_far"] = (r[-2][1] - r[-3][1]) * dx
        sp = equivalent_longitudinal_spec("pierna", body)
        assert t["cortical_bone"] == pytest.approx(sp.cortex, abs=1.01 * dx)
        assert t["fat_far"] == pytest.approx(sp.fat_far, abs=1.01 * dx)
        thick[body] = t
    assert thick["delgado"]["fat_far"] < thick["normal"]["fat_far"] < thick["obeso"]["fat_far"]
    assert thick["obeso"]["fat_far"] / thick["normal"]["fat_far"] == pytest.approx(BODY_TYPES["obeso"].fat, rel=0.15)
    assert thick["anciano_osteoporotico"]["cortical_bone"] < 0.7 * thick["normal"]["cortical_bone"]


# ----------------------------------------------------------------------------
# Fracturas
# ----------------------------------------------------------------------------
def _fractured(kind, seed=0, dx=0.5e-3, preset="pierna", **extra):
    fr = random_fracture_3d(np.random.default_rng(seed), kind, x_range=2e-3)
    fr = replace(fr, **extra)
    return fr, limb_3d(preset, fr, length=24e-3, dx=dx, seed=seed)


@pytest.mark.parametrize("kind", FRACTURE_KINDS_3D)
def test_cada_tipo_de_fractura_tiene_mascara_en_su_sitio(kind):
    preset = "pierna_metafisis" if kind == "contusion_osea" else "pierna"
    fr, ph = _fractured(kind, preset=preset, position=1e-3)
    fx, fy, fz = ph.meta["fracture_center"]
    assert fx == pytest.approx(1e-3) and abs(fy) < 1e-3  # la metáfisis crece hacia el centro del miembro
    frac, hem, ede = ph.masks["fracture"], ph.masks["hematoma"], ph.masks["edema"]
    assert frac.any() and (ph.labels[frac] == TISSUE_INDEX["hematoma"]).all()
    cx, cy, cz = _centroid(ph, frac)
    ro = 18.5e-3 if preset == "pierna_metafisis" else 11e-3
    assert abs(cx - 1e-3) < 6e-3 and np.hypot(cy - fy, cz - fz) < ro
    if kind == "cortical_lejana":
        assert cz > fz  # la grieta está en la cortical lejana
    if kind in ("fisura", "incompleta", "intracortical"):
        assert cz < fz  # lado de la sonda, sin llegar al otro lado
    if kind == "contusion_osea":
        assert not frac[ph.mask("cortical_bone")].any() and ede.any()
        # las microfracturas (hematoma) cruzan el edema, como en 2D
        assert np.isin(ph.labels[ede], [TISSUE_INDEX["edema"], TISSUE_INDEX["hematoma"]]).all()
    if fr.gap > 0 and fr.opens_near_surface and fr.hematoma_thickness > 0:
        assert (hem & ~frac).any()  # hematoma perióstico
    if fr.edema_length > 0:
        assert ede.any()


def test_fractura_transversal_corta_todo_el_perimetro_y_la_incompleta_un_arco():
    dx = 0.5e-3
    base = dict(position=0.0, gap=1.5e-3, roughness=0.0, hematoma_thickness=0.0, edema_length=0.0)
    full = limb_3d("pierna", Fracture3D(**base), length=8e-3, dx=dx, roughness=False)
    g = full.grid
    in_gap = np.abs(g.x) < 0.5e-3
    j_tib = np.abs(g.y) < 12e-3  # sólo la tibia: el peroné no se rompe
    assert not full.mask("cortical_bone")[:, j_tib][:, :, in_gap].any()
    assert full.mask("cortical_bone")[:, ~j_tib][:, :, in_gap].any()
    arc = limb_3d("pierna", Fracture3D(**base, arc_deg=90.0), length=8e-3, dx=dx, roughness=False)
    _, cy, cz = _centroid(arc, arc.masks["fracture"])
    assert cz < arc.meta["axis_depth"] - 5e-3 and abs(cy) < 2e-3
    side = limb_3d("pierna", Fracture3D(**base, arc_deg=90.0, arc_center_deg=90.0), length=8e-3, dx=dx,
                   roughness=False)
    _, cy, cz = _centroid(side, side.masks["fracture"])
    assert cy > 5e-3 and abs(cz - side.meta["axis_depth"]) < 2e-3  # 90°: hacia +y
    assert arc.masks["fracture"].sum() < 0.4 * full.masks["fracture"][:, j_tib].sum()


def test_desplazada_traslada_el_fragmento_distal():
    dx = 0.5e-3
    fr = Fracture3D(kind="desplazada", position=0.0, displacement=2e-3, displacement_y=1e-3, roughness=0.0)
    ph = limb_3d("pierna", fr, length=20e-3, dx=dx, roughness=False)
    g = ph.grid
    cort = ph.mask("cortical_bone")
    j0 = g.index_of(0, 0, 0)[1]
    top = [np.argmax(cort[:, j0, g.index_of(x, 0, 0)[2]]) for x in (-8e-3, 8e-3)]
    assert (top[1] - top[0]) * dx == pytest.approx(2e-3, abs=dx)
    # equivalente al 2D: Fracture con el mismo escalón da el mismo corte longitudinal
    assert limb_3d("pierna", Fracture(displacement=2e-3, roughness=0.0), length=4e-3, dx=1e-3).meta["fracture"][
        "displacement"] == 2e-3


def test_fisura_fina_es_una_lamina_de_una_celda_parcial():
    fr, ph = _fractured("fisura", dx=0.5e-3)
    frac = ph.masks["fracture"]
    assert fr.gap < 0.25e-3 and fr.min_gap_cells == 1.0 and fr.arc_deg < 181
    # lámina delgada: pocas celdas por columna a lo largo de x
    per_col = frac.sum(axis=2)
    assert per_col[per_col > 0].mean() < 3


def test_estres_con_reaccion_y_contusion_con_microfracturas():
    fr, ph = _fractured("estres", seed=1, periosteal_reaction=1.5e-3, endosteal_reaction=0.0, span=(0.0, 0.4))
    new_bone = ph.mask("trabecular_bone")  # hueso perióstico nuevo (no hay trabecular en la diáfisis)
    assert new_bone.any() and _centroid(ph, new_bone)[2] < ph.meta["bone_top_depth"] + 1e-3
    fr, ph = _fractured("estres", seed=1, periosteal_reaction=0.0, endosteal_reaction=1.5e-3, span=(0.6, 1.0))
    assert ph.masks["edema"].any()
    fr, ph = _fractured("contusion_osea", seed=2, preset="pierna_metafisis", microcracks=12, edema_fraction=0.3)
    assert ph.masks["fracture"].any() and not ph.masks["fracture"][ph.mask("cortical_bone")].any()
    # el edema ocupa los espacios intertrabeculares: el armazón trabecular sigue ahí
    k, j, i = (np.unique(a) for a in np.nonzero(ph.masks["edema"]))
    box = np.ix_(k, j, i)
    trab = ph.mask("trabecular_bone")[box].sum()
    assert 0.1 < ph.masks["edema"][box].sum() / (trab + ph.masks["edema"][box].sum()) < 0.7


def test_fracciones_de_volumen():
    ph, frac = limb_3d_fractions("antebrazo", length=2e-3, dx=1e-3, supersample=2)
    total = sum(frac.values())
    assert np.allclose(total, 1.0)
    # fracciones intermedias sólo en las interfaces, y la cortical media coincide con la del fantoma
    mixed = ((total > 0) & (frac["cortical_bone"] > 0) & (frac["cortical_bone"] < 1)).mean()
    assert 0 < mixed < 0.05
    assert frac["cortical_bone"].mean() == pytest.approx(ph.mask("cortical_bone").mean(), rel=0.1)


def test_metafisis_parametrizable():
    spec = replace(LIMB3D_PRESETS["pierna"], metaphysis=Metaphysis(position=-5e-3, direction=-1, flare=1.3))
    ph = limb_3d(spec, length=20e-3, dx=1e-3, roughness=False)
    g = ph.grid
    bone = ph.mask("cortical_bone") | ph.mask("trabecular_bone") | ph.mask("marrow")
    j0 = g.index_of(0, 0, 0)[1]
    h = [bone[:, j0, g.index_of(x, 0, 0)[2]].sum() for x in (-9e-3, 9e-3)]
    assert h[0] > h[1]


# ----------------------------------------------------------------------------
# Solver elástico 3D sobre un limb_3d
# ----------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:Sólo")
def test_solver_elastico_3d_corre_sobre_limb_3d():
    from wavesim.ultrasound.elastic3d import ElasticSolver3D, Source3D
    from wavesim.core.signals import gaussian_pulse

    fr = Fracture3D(position=0.0, gap=1e-3)
    ph = limb_3d("antebrazo", fr, length=6e-3, dx=1e-3, margin=1e-3)
    s = ElasticSolver3D(ph, 0.1e6, npml=4)
    nt = 60
    t = s.time_axis(nt)
    k, j, i = ph.grid.index_of(0.0, 0.0, 2e-3)
    pulse = gaussian_pulse(t, 0.1e6)[None, :]
    s.add_source(Source3D(np.array([k]), np.array([j]), np.array([i]), np.array([0]), pulse))
    s.run(nt)
    assert np.isfinite(s.vx).all() and np.isfinite(s.pressure()).all() and np.abs(s.vz).max() > 0


def test_espiroidea_gira_alrededor_del_hueso():
    dx = 0.5e-3
    fr = Fracture3D(kind="espiroidea", position=0.0, gap=0.8e-3, roughness=0.0, spiral_pitch=40e-3,
                    crack_length=20e-3, edema_length=0.0)
    ph = limb_3d("pierna", fr, length=24e-3, dx=dx, roughness=False)
    g = ph.grid
    yc, zc = ph.meta["bone_centers"][0]
    angles = []
    for x in (-5e-3, 0.0, 5e-3):
        k, j = np.nonzero(ph.masks["fracture"][:, :, g.index_of(x, 0, 0)[2]])
        angles.append(np.degrees(np.arctan2((g.y0 + j * dx - yc).mean(), -(g.z0 + k * dx - zc).mean())))
    # 5 mm de 40 mm por vuelta = 45°
    assert np.diff(angles) == pytest.approx([45.0, 45.0], abs=8.0)
    assert not ph.masks["fracture"][:, :, np.abs(g.x) > 10.5e-3].any()
    with pytest.raises(ValueError):
        limb_3d("pierna", replace(fr, displacement=1e-3), length=4e-3, dx=1e-3)
