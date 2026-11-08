import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.mri import SEQUENCES, bloch, mri_scan, resolve, signal, tissue_signal
from wavesim.phantoms import RingFracture, cross_section_limb
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

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


@pytest.mark.parametrize("voxel", [0.4e-3, 0.5e-3, 0.8e-3])
def test_mri_axes_locate_a_block_within_a_quarter_voxel(voxel):
    # Con k-space recortado, la muestra k está en x0 + k·W/n: el centroide de un bloque
    # reconstruido, medido con im.x / im.z, debe coincidir con el real.
    g = Grid2D.from_size(24e-3, 22e-3, 0.2e-3, x0=-12e-3, z0=1e-3)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    X, Z = g.meshgrid()
    blk = (X >= -3e-3) & (X < 5e-3) & (Z >= 6e-3) & (Z < 14e-3)
    lab[blk] = TISSUE_INDEX["water"]
    im = mri_scan(Phantom(g, lab), "PD", voxel=voxel, snr=0, texture=0)
    mag = im.magnitude
    assert im.x.size == mag.shape[1] and im.z.size == mag.shape[0]
    cx = np.sum(mag.sum(axis=0) * im.x) / mag.sum()
    cz = np.sum(mag.sum(axis=1) * im.z) / mag.sum()
    assert abs(cx - X[blk].mean()) < 0.25 * im.voxel
    assert abs(cz - Z[blk].mean()) < 0.25 * im.voxel
    ext = im.extent_mm
    assert ext[0] == pytest.approx((im.x[0] - im.voxel / 2) * 1e3, rel=1e-6)
    assert ext[3] == pytest.approx((im.z[0] - np.diff(im.z)[0] / 2) * 1e3, rel=1e-6)
