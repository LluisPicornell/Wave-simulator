"""Validación física del solver elástico frente a soluciones analíticas."""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.core.signals import envelope, gaussian_pulse
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom
from wavesim.ultrasound import ElasticMedium, ElasticSolver, LinearArray, Source, line_receiver
from wavesim.ultrasound import attenuation as att
from wavesim.ultrasound.beamforming import bmode

F0 = 1e6


def _grid(depth, width=24e-3, ppw=10, c=1500.0):
    dx = c / F0 / ppw
    return Grid2D.from_size(width, depth, dx, x0=-width / 2)


def _medium(g, rho, cp, cs, tau_p=None, tau_s=None, n_mech=0):
    z = np.zeros(g.shape)
    full = lambda v: v if isinstance(v, np.ndarray) else np.full(g.shape, float(v))  # noqa: E731
    return ElasticMedium(full(rho), full(cp), full(cs), z if tau_p is None else full(tau_p),
                         z if tau_s is None else full(tau_s), att.relaxation_times(F0, n_mech), F0)


def _plane_source(g, solver, z, nt, kind="pressure"):
    j = g.index_of(0, z)[0]
    t = solver.time_axis(nt)
    return Source(np.full(g.nx, j), np.arange(g.nx), np.zeros(g.nx, int), gaussian_pulse(t, F0)[None, :], kind)


def _dummy_phantom(g):
    return Phantom(g, np.full(g.shape, TISSUE_INDEX["water"], np.int16))


def _xcorr_lag(a, b, dt):
    n = a.size
    xc = np.fft.irfft(np.fft.rfft(b, 2 * n) * np.conj(np.fft.rfft(a, 2 * n)))
    k = int(np.argmax(xc[:n]))
    # refinamiento parabólico del máximo
    y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
    return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * dt


def test_compressional_wave_speed():
    g = _grid(40e-3)
    s = ElasticSolver(_dummy_phantom(g), F0, medium=_medium(g, 1000, 1500, 0))
    nt = int(26e-6 / s.dt)
    s.add_source(_plane_source(g, s, 6e-3, nt))
    s.add_receiver("a", line_receiver(g, 15e-3, -2e-3, 2e-3))
    s.add_receiver("b", line_receiver(g, 30e-3, -2e-3, 2e-3))
    r = s.run(nt, track_peak=False)
    lag = _xcorr_lag(r.traces["a"].mean(0), r.traces["b"].mean(0), s.dt)
    dz = g.z[g.index_of(0, 30e-3)[0]] - g.z[g.index_of(0, 15e-3)[0]]
    assert dz / lag == pytest.approx(1500.0, rel=0.01)


def test_shear_wave_speed_in_bone():
    g = _grid(40e-3, ppw=12, c=1800)
    s = ElasticSolver(_dummy_phantom(g), F0, medium=_medium(g, 1900, 3500, 1800))
    nt = int(24e-6 / s.dt)
    s.add_source(_plane_source(g, s, 6e-3, nt, kind="force_x"))  # fuerza horizontal → onda S
    ra = line_receiver(g, 15e-3, -2e-3, 2e-3)
    rb = line_receiver(g, 30e-3, -2e-3, 2e-3)
    ra.quantity = rb.quantity = "vx"
    s.add_receiver("a", ra)
    s.add_receiver("b", rb)
    r = s.run(nt, track_peak=False)
    lag = _xcorr_lag(r.traces["a"].mean(0), r.traces["b"].mean(0), s.dt)
    dz = g.z[g.index_of(0, 30e-3)[0]] - g.z[g.index_of(0, 15e-3)[0]]
    assert dz / lag == pytest.approx(1800.0, rel=0.015)


def test_reflection_coefficient_at_fluid_interface():
    g = _grid(40e-3)
    rho = np.where(g.meshgrid()[1] < 25e-3, 1000.0, 2000.0)
    cp = np.where(g.meshgrid()[1] < 25e-3, 1500.0, 2500.0)
    s = ElasticSolver(_dummy_phantom(g), F0, medium=_medium(g, rho, cp, 0))
    nt = int(30e-6 / s.dt)
    s.add_source(_plane_source(g, s, 5e-3, nt))
    s.add_receiver("r", line_receiver(g, 12e-3, -2e-3, 2e-3))
    r = s.run(nt, track_peak=False)
    tr = r.traces["r"].mean(0)
    t = r.t
    t_split = (12e-3 - 5e-3) / 1500 + (25e-3 - 12e-3) / 1500 + 2e-6
    inc = np.abs(tr[t < t_split]).max()
    refl = tr[t >= t_split]
    sign = np.sign(refl[np.argmax(np.abs(refl))])
    r_meas = sign * np.abs(refl).max() / inc
    r_theory = (2000 * 2500 - 1000 * 1500) / (2000 * 2500 + 1000 * 1500)
    assert r_meas == pytest.approx(r_theory, rel=0.04)


def test_viscoelastic_attenuation_matches_target():
    """α(f0) medido en la simulación ≈ α del modelo ≈ α de la tabla (5 dB/cm/MHz)."""
    g = _grid(40e-3, ppw=12)
    c = 1500.0
    alpha_db = 5.0
    alpha = alpha_db * 100 / (20 / np.log(10))  # Np/m a 1 MHz
    ts = att.relaxation_times(F0, 1)
    band = att.fit_band(F0)
    tau = att.fit_tau(att.target_qinv(alpha * band / F0, c, band), ts, band)

    def run(tau_p):
        s = ElasticSolver(_dummy_phantom(g), F0, medium=_medium(g, 1000, c, 0, tau_p=tau_p, n_mech=1))
        nt = int(26e-6 / s.dt)
        s.add_source(_plane_source(g, s, 6e-3, nt))
        s.add_receiver("a", line_receiver(g, 15e-3, -2e-3, 2e-3))
        s.add_receiver("b", line_receiver(g, 30e-3, -2e-3, 2e-3))
        r = s.run(nt, track_peak=False)
        tt = r.t
        amp = [abs(np.sum(r.traces[k].mean(0) * np.exp(-2j * np.pi * F0 * tt))) for k in ("a", "b")]
        return amp[1] / amp[0]

    dz = g.z[g.index_of(0, 30e-3)[0]] - g.z[g.index_of(0, 15e-3)[0]]
    alpha_sim = -np.log(run(tau) / run(0.0)) / dz
    _, alpha_model = att.model_wave(c, tau, ts, F0, F0)
    assert alpha_sim == pytest.approx(alpha_model, rel=0.05)
    assert alpha_model == pytest.approx(alpha, rel=0.10)


def test_pml_absorbs_outgoing_waves():
    g = _grid(20e-3, width=20e-3)
    s = ElasticSolver(_dummy_phantom(g), F0, medium=_medium(g, 1000, 1500, 0))
    nt = int(40e-6 / s.dt)
    t = s.time_axis(nt)
    j, i = g.index_of(0, 10e-3)
    s.add_source(Source(np.array([j]), np.array([i]), np.array([0]), gaussian_pulse(t, F0)[None, :]))
    r = s.run(nt)
    assert np.abs(s.pressure()).max() < 0.01 * r.peak_pressure[g.nz // 4: -g.nz // 4, g.nx // 4: -g.nx // 4].max()


def test_heterogeneous_limb_is_stable():
    from wavesim.phantoms import Fracture, longitudinal_limb

    p = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=8e-3, depth=14e-3, dx=1450 / 2e6 / 8)
    s = ElasticSolver(p, 2e6)
    arr = LinearArray(n_elements=16, pitch=0.35e-3, z=1.2e-3, f0=2e6)
    nt = 2500
    t = s.time_axis(nt)
    s.add_source(arr.transmit_source(p.grid, t, arr.plane_wave_delays(0, 1540)))
    s.run(nt)
    assert np.isfinite(s.vx).all() and np.abs(s.pressure()).max() < 1e6


def test_das_focuses_point_scatterer():
    """Beamforming sobre RF sintética de un dispersor puntual."""
    c0, f0 = 1540.0, 5e6
    arr = LinearArray(n_elements=64, pitch=c0 / f0 / 2, z=0.0, f0=f0)
    dt = 1 / (40 * f0)
    t = np.arange(4000) * dt
    xs, zs = 1.5e-3, 12e-3
    rf = np.stack([
        gaussian_pulse(t - (zs / c0 + np.hypot(xs - xe, zs) / c0), f0) for xe in arr.element_x
    ])
    x = np.linspace(-4e-3, 4e-3, 161)
    z = np.linspace(8e-3, 16e-3, 161)
    img = bmode([rf], [0.0], t, arr.element_x, 0.0, x, z, c0=c0, t_peak=arr.pulse_peak_time)
    jz, ix = np.unravel_index(np.argmax(img.envelope), img.envelope.shape)
    lam = c0 / f0
    assert abs(x[ix] - xs) < lam / 2 and abs(z[jz] - zs) < lam / 2


def test_envelope_of_pulse_peaks_at_center():
    t = np.arange(2000) * 1e-8
    p = gaussian_pulse(t, 1e6, t0=8e-6)
    assert t[np.argmax(envelope(p))] == pytest.approx(8e-6, abs=2e-8)
