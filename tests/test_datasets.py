import json

import numpy as np
import pytest

from wavesim.datasets import generate_dataset, generate_sample


@pytest.mark.parametrize("modality", ["ct", "mri"])
def test_sample_has_aligned_mask(modality):
    s = generate_sample(np.random.default_rng(0), modality, fracture=True)
    assert s["label"] == 1
    assert s["image"].shape == s["mask"].shape
    assert s["mask"].any()
    assert np.isfinite(s["image"]).all()


def test_dataset_is_written_and_indexed(tmp_path):
    out = generate_dataset(tmp_path, 4, "mri", seed=3, verbose=False)
    idx = json.loads((out / "index.json").read_text())
    assert len(idx["samples"]) == 4
    d = np.load(out / idx["samples"][0]["file"])
    assert {"image", "mask", "label"} <= set(d.files)


def test_thigh_dataset_contains_bone_and_fracture():
    # Antes, con 22 mm fijos de profundidad, el muslo (hueso a 40-50 mm) nunca tenía fractura
    s = generate_sample(np.random.default_rng(1), "mri", "muslo", fracture=True)
    assert s["label"] == 1 and s["mask"].any()


def test_mri_mask_lines_up_with_the_crack():
    from wavesim.datasets.generate import make_phantom

    rng = np.random.default_rng(7)
    ph = make_phantom(np.random.default_rng(7), "tibia", True, 0.05e-3, 24e-3, "auto")
    s = generate_sample(rng, "mri", "tibia", fracture=True)
    # misma semilla → mismo fantoma; los píxeles marcados deben caer sobre la grieta (± 1 vóxel)
    j, i = np.nonzero(s["mask"])
    crack_x = ph.grid.x[np.nonzero(ph.fracture_mask)[1]] * 1e3
    crack_z = ph.grid.z[np.nonzero(ph.fracture_mask)[0]] * 1e3
    voxel = s["x_mm"][1] - s["x_mm"][0]
    assert abs(s["x_mm"][i].mean() - crack_x.mean()) < voxel
    assert abs(s["z_mm"][j].mean() - crack_z.mean()) < voxel


def test_ultrasound_mask_uses_apparent_depth():
    from wavesim.analysis import resample_mask_apparent_depth
    from wavesim.core.grid import Grid2D
    from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

    g = Grid2D.from_size(4e-3, 20e-3, 0.05e-3, x0=-2e-3)
    ph = Phantom(g, np.full(g.shape, TISSUE_INDEX["water"], np.int16))  # agua: c = 1500 m/s
    mask = np.zeros(g.shape, bool)
    j_true = g.index_of(0, 12e-3)[0]
    mask[j_true, :] = True
    z_img = np.arange(0, 20e-3, 0.025e-3)
    out = resample_mask_apparent_depth(ph, mask, g.x, z_img, z_probe=1e-3, c0=1540.0)
    z_app = z_img[np.nonzero(out[:, 0])[0]].mean()
    expected = 1e-3 + 1540.0 / 1500.0 * (g.z[j_true] - 1e-3)
    assert z_app == pytest.approx(expected, abs=0.05e-3)
