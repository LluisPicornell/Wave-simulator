import numpy as np
import pytest

from wavesim.mri import SEQUENCES, bloch, mri_scan, resolve, signal, tissue_signal
from wavesim.phantoms import RingFracture, cross_section_limb

T1 = np.array([250.0, 600.0, 1200.0, 3000.0])
T2 = np.array([40.0, 60.0, 100.0, 1500.0])


def test_spin_echo_equation_matches_bloch():
    seq = SEQUENCES["T1"]
    analytic = signal(1.0, T1, T2, T2, seq)
    numeric = bloch.spin_echo_steady_state(T1, T2, seq.tr, seq.te)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


def test_inversion_recovery_equation_matches_bloch():
    seq = resolve("STIR", 1.5)
    analytic = signal(1.0, T1, T2, T2, seq)
    numeric = bloch.inversion_recovery_steady_state(T1, T2, seq.tr, seq.te, seq.ti)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize("b0", [1.5, 3.0])
def test_stir_nulls_fat_and_shows_edema(b0):
    assert tissue_signal("fat", "STIR", b0) < 0.02 * tissue_signal("muscle", "STIR", b0)
    assert tissue_signal("edema", "STIR", b0) > 1.5 * tissue_signal("marrow", "STIR", b0)


def test_cortical_bone_dark_except_ute():
    assert tissue_signal("cortical_bone", "T2") < 0.01 * tissue_signal("muscle", "T2")
    assert tissue_signal("cortical_bone", "UTE") > 0.2 * tissue_signal("muscle", "UTE")


def test_mri_scan_resolution_and_noise():
    p = cross_section_limb("antebrazo", RingFracture(), dx=0.2e-3).replace_tissue("water", "air")
    im = mri_scan(p, "STIR", voxel=0.5e-3, snr=20)
    assert im.magnitude.shape[0] == pytest.approx(p.grid.depth / 0.5e-3, abs=1)
    air = im.magnitude[:5, :5]
    assert air.mean() > 0  # ruido de Rice en el fondo
