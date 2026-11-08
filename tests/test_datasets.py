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
