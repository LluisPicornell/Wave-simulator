"""Captura de matriz completa (FMC): síntesis de emisiones por linealidad, STA de subconjuntos y ruido de canal.

Agua con un dispersor puntual a 2 MHz, 8 elementos: cada FMC son 8 simulaciones pequeñas.
"""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.core.signals import analytic_signal
from wavesim.ultrasound import ElasticSolver, LinearArray
from wavesim.ultrasound.beamforming import Transmit, bmode_channels
from wavesim.ultrasound.fmc import (FMCData, acquire_fmc, band_noise, blank_direct, delayed_pairs, element_weights,
                                    fractional_delay, pair_coherence, pair_image, synthesize_rf)
from wavesim.ultrasound.quality import point_scatterer_medium, water_phantom

C, F0, NPML = 1500.0, 2e6, 12


@pytest.fixture(scope="module")
def setup():
    g = Grid2D.from_size(6e-3, 7e-3, C / F0 / 10, x0=-3e-3)
    ph = water_phantom(g)
    medium = point_scatterer_medium(g, F0, [(0.3e-3, 4.5e-3)], contrast=3.0)
    arr = LinearArray(n_elements=8, pitch=C / F0 / 2, z=g.z0 + (NPML + 3) * g.dx, x_center=0.0, f0=F0)
    fmc = acquire_fmc(ph, arr, F0, npml=NPML, medium=medium, n_mechanisms=0)
    return g, ph, medium, arr, fmc


def test_fractional_delay_matches_analytic_shift():
    dt = 1e-8
    t = np.arange(600) * dt
    x = np.exp(-((t - 2e-6) / 0.3e-6) ** 2) * np.sin(2 * np.pi * F0 * t)
    y = fractional_delay(x, 0.37e-6, dt)
    ref = np.exp(-((t - 2.37e-6) / 0.3e-6) ** 2) * np.sin(2 * np.pi * F0 * (t - 0.37e-6))
    assert np.max(np.abs(y - ref)) < 1e-6


@pytest.mark.parametrize("angle", [0.0, 12.0])
def test_plane_wave_from_fmc_equals_direct_simulation(setup, angle):
    """Σ_i S_i(t − τ_i) reproduce la onda plana simulada directamente (linealidad del FDTD)."""
    g, ph, medium, arr, fmc = setup
    delays = arr.plane_wave_delays(angle, C)
    synth = synthesize_rf(fmc, delays)
    solver = ElasticSolver(ph, F0, npml=NPML, n_mechanisms=0, medium=medium)
    nt = fmc.t.size
    t = solver.time_axis(nt)
    solver.add_source(arr.transmit_source(g, t, delays))
    solver.add_receiver("rf", arr.receiver(g))
    direct = solver.run(nt, track_peak=False).traces["rf"]
    err = np.linalg.norm(synth - direct) / np.linalg.norm(direct)
    # Medido: 2e-7 a 0° y 7e-5 a 12° (en el eco, 6e-6 y 2e-4): redondeo float32 del FDTD y retardo
    # fraccionario por FFT del pulso emitido, que empieza bruscamente en t = 0
    assert err < 3e-4
    # El eco del dispersor (tras borrar la onda directa) también coincide
    late = fmc.t > fmc.t_peak + 2 * 3.5e-3 / C
    err_echo = np.linalg.norm((synth - direct)[:, late]) / np.linalg.norm(direct[:, late])
    assert err_echo < 1e-3


def test_fmc_is_reciprocal(setup):
    """S_ij ≈ S_ji: el mismo trayecto en sentido inverso (fuente y receptor de presión en las mismas celdas)."""
    *_, fmc = setup
    rf = blank_direct(fmc, C)
    err = np.linalg.norm(rf - rf.transpose(1, 0, 2)) / np.linalg.norm(rf)
    assert err < 1e-4  # medido: 7e-6


def test_pair_image_equals_sta_beamformer(setup):
    """La STA completa con el cubo de pares coincide con ``bmode_channels`` (mismo DAS, salvo la normalización)."""
    g, ph, medium, arr, fmc = setup
    rf = blank_direct(fmc, C)
    x_img = np.linspace(-1.5e-3, 1.5e-3, 25)
    z_img = np.linspace(3.0e-3, 5.5e-3, 30)
    D = delayed_pairs(rf, fmc, x_img, z_img, C, f_number=1.0, window="rect")
    mine = pair_image(D, np.arange(8), np.arange(8))
    ref = bmode_channels(list(rf), [Transmit("element", element=k) for k in range(8)], fmc.t, fmc.element_x,
                         fmc.array_z, x_img, z_img, C, fmc.t_peak, f_number=1.0, rx_window="rect").iq
    w_sum = element_weights(fmc.element_x, fmc.array_z, x_img, z_img, 1.0, "rect").sum(0)
    err = np.linalg.norm(mine - ref * w_sum) / np.linalg.norm(mine)
    assert err < 1e-5
    # El máximo cae en el dispersor
    j, i = np.unravel_index(np.argmax(np.abs(mine)), mine.shape)
    assert abs(x_img[i] - 0.3e-3) < 0.2e-3 and abs(z_img[j] - 4.5e-3) < 0.15e-3
    # Factor de coherencia: ~1 en el dispersor, menor fuera de él
    cf = pair_coherence(D, np.arange(8), np.arange(8))
    assert cf[j, i] > 0.8 and np.median(cf) < cf[j, i]


def test_band_noise_level_and_band():
    rng = np.random.default_rng(0)
    dt = 1e-8
    n = band_noise((200, 4096), dt, F0, 1.0, rng)
    assert np.std(n) == pytest.approx(1.0, rel=0.03)
    spec = np.mean(np.abs(np.fft.rfft(n, axis=-1)) ** 2, axis=0)
    f = np.fft.rfftfreq(4096, dt)
    assert abs(f[np.argmax(spec)] - F0) < 0.15 * F0
    # |H| = 1/2 (−6 dB) en f0 ± f0/2
    k = np.argmin(np.abs(f - 1.5 * F0))
    assert 10 * np.log10(spec[k] / spec.max()) == pytest.approx(-6.0, abs=1.0)


def test_fmc_save_load_roundtrip(tmp_path, setup):
    *_, fmc = setup
    fmc.save(tmp_path / "f.npz")
    back = FMCData.load(tmp_path / "f.npz")
    assert np.array_equal(back.rf, fmc.rf) and back.t_peak == fmc.t_peak and back.array_z == fmc.array_z
    assert np.abs(analytic_signal(back.rf[0, 0])).max() > 0
