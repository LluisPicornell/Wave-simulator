"""Cadena de imagen de ultrasonido: transductor realista, formadores de haz y métricas de calidad.

Las simulaciones son pequeñas (1-2 MHz, agua con dispersores de una celda) para que el fichero
tarde menos de un minuto. Las referencias son teóricas: diagrama de apertura (1.206·λ·F# en
rectangular), pulso de ida y vuelta, SNR de Rayleigh (1.913).
"""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.core.signals import analytic_signal, gaussian_pulse
from wavesim.ultrasound import (ElementResponse, LinearArray, TwoLayerSpeed, apodization, bmode, default_array,
                                simulate_bmode, simulate_scan)
from wavesim.ultrasound.beamforming import Transmit, channel_cube, combine_channels
from wavesim.ultrasound.quality import (RAYLEIGH_SNR, _water_medium, contrast_db, disk_mask, fwhm_1d, gcnr,
                                        point_scatterer_medium, predicted_grain_fwhm, psf_fwhm, speckle_grain_fwhm,
                                        speckle_medium, speckle_snr, theoretical_axial_fwhm,
                                        theoretical_lateral_fwhm, water_phantom)
from wavesim.ultrasound.transducer import window_function

C = 1500.0  # velocidad del agua del simulador


def _setup(f0, width, depth, response=None):
    """Malla de agua a 10 puntos por longitud de onda y sonda λ/2 que cabe bajo la PML."""
    g = Grid2D.from_size(width, depth, C / f0 / 10, x0=-width / 2)
    ph = water_phantom(g)
    a = default_array(ph, f0, 20, C)
    arr = LinearArray(n_elements=a.n_elements, pitch=a.pitch, z=a.z, x_center=a.x_center, f0=f0, response=response)
    return g, ph, arr


# --------------------------------------------------------------------------
# Transductor
# --------------------------------------------------------------------------
def test_element_response_is_odd_centered_and_bandpass():
    r = ElementResponse(2e6, "windowed", 2.0, "hann")
    dt = 1 / (40 * 2e6)
    h = r.impulse(dt)
    assert h.size % 2 == 1 and np.allclose(h, -h[::-1], atol=1e-12)  # impar: sin(2π f0 τ) con ventana par
    f = np.linspace(0.1e6, 5e6, 200)
    spec = r.spectrum(f, dt)
    assert abs(f[np.argmax(spec)] - 2e6) < 0.25e6
    assert spec[0] < 0.2 * spec.max() and spec[-1] < 0.2 * spec.max()  # paso de banda: sin continua ni HF


def test_receive_filter_does_not_delay_the_echo():
    arr = LinearArray(f0=2e6, response=ElementResponse(2e6, "windowed", 2.0))
    dt = 1 / (40 * 2e6)
    t = np.arange(2000) * dt
    echo = gaussian_pulse(t, 2e6, t0=t[1000])
    out = arr.receive_filter(echo[None, :], dt)[0]
    # La envolvente se ensancha pero su centro no se mueve
    assert abs(t[np.argmax(np.abs(analytic_signal(out)))] - t[1000]) < 2 * dt
    assert np.array_equal(LinearArray(f0=2e6).receive_filter(echo[None, :], dt)[0], echo)  # sin ``response``, identidad


def test_kerf_and_width():
    assert LinearArray(pitch=1e-3).width == pytest.approx(0.85e-3)
    assert LinearArray(pitch=1e-3, kerf=0.1e-3).width == pytest.approx(0.9e-3)
    assert LinearArray(pitch=1e-3, kerf=0.1e-3, element_width=0.5e-3).width == pytest.approx(0.5e-3)
    g = Grid2D.from_size(8e-3, 6e-3, 0.1e-3, x0=-4e-3)
    narrow = LinearArray(n_elements=8, pitch=0.7e-3, kerf=0.4e-3, z=1e-3)
    wide = LinearArray(n_elements=8, pitch=0.7e-3, kerf=0.1e-3, z=1e-3)
    assert narrow.element_cells(g)[1].size < wide.element_cells(g)[1].size


def test_apodization_windows():
    assert np.all(apodization(16, "rect") == 1.0)
    h = apodization(17, "hann")
    assert h.argmax() == 8 and np.allclose(h, h[::-1]) and h[0] < 0.05 < h[8] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        window_function(np.zeros(3), "kaiser")


def test_focused_subaperture_focuses_and_limits_aperture():
    arr = LinearArray(n_elements=32, pitch=0.3e-3, z=0.0)
    delays, apod = arr.focused_subaperture(0.5e-3, 12e-3, 2.0, C)
    active = apod > 0
    assert active.sum() < arr.n_elements  # F#=2 a 12 mm: apertura de 6 mm de 9.6
    # Todos los elementos activos llegan al foco a la vez
    t_arrival = delays + np.hypot(arr.element_x - 0.5e-3, 12e-3) / C
    assert np.ptp(t_arrival[active]) < 1e-12
    assert abs(arr.element_x[active].mean() - 0.5e-3) < arr.pitch
    assert delays[~active].max() == 0.0


# --------------------------------------------------------------------------
# Beamforming sobre RF sintética
# --------------------------------------------------------------------------
def _synthetic_rf(arr, t, xs, zs, c_of_path):
    """Eco de un dispersor puntual para una onda plana a 0°: ida ``tx`` y vuelta ``rx`` dadas por ``c_of_path``."""
    tx, rx = c_of_path
    return np.stack([gaussian_pulse(t - (tx + rx[e]), arr.f0) for e in range(arr.n_elements)])


def _layers_fermat_bruteforce(layers, xs, zs, X, Z):
    """Tiempo mínimo por búsqueda exhaustiva del punto de cruce de la interfaz."""
    xc = np.linspace(min(xs, X) - 2e-3, max(xs, X) + 2e-3, 20001)
    h1, h2 = layers.z_interface - zs, Z - layers.z_interface
    return (np.hypot(xc - xs, h1) / layers.c_top + np.hypot(X - xc, h2) / layers.c_bottom).min()


def test_two_layer_travel_time_matches_fermat_bruteforce():
    ly = TwoLayerSpeed(10e-3, 1450.0, 1580.0)
    X, Z = np.array([[3e-3, -4e-3, 0.0]]), np.array([[25e-3, 18e-3, 12e-3]])
    t = ly.point_source_time(-2e-3, 0.0, X, Z)
    ref = [_layers_fermat_bruteforce(ly, -2e-3, 0.0, X[0, k], Z[0, k]) for k in range(3)]
    assert np.allclose(t[0], ref, rtol=1e-5)
    # Sin contraste de velocidad se reduce a la línea recta
    flat = TwoLayerSpeed(10e-3, 1540.0, 1540.0)
    assert np.allclose(flat.point_source_time(0.0, 0.0, X, Z), np.hypot(X, Z) / 1540.0, rtol=1e-9)


def test_two_layer_plane_wave_matches_snell_geometry():
    ly = TwoLayerSpeed(8e-3, 1450.0, 1600.0)
    ang = 10.0
    X, Z = np.array([[2e-3]]), np.array([[20e-3]])
    t = ly.plane_wave_time(X, Z, ang, xr=-5e-3, array_z=0.0)[0, 0]
    # Mínimo del tiempo (frente plano hasta el punto de cruce + trayecto a P) sobre el punto de cruce
    xc = np.linspace(-20e-3, 20e-3, 200001)
    th = np.radians(ang)
    ref = (((xc + 5e-3) * np.sin(th) + 8e-3 * np.cos(th)) / 1450.0 + np.hypot(2e-3 - xc, 12e-3) / 1600.0).min()
    assert t == pytest.approx(ref, rel=1e-6)


def _point_rf(layers=None, c0=1540.0, f0=3e6, xs=1.0e-3, zs=22e-3):
    arr = LinearArray(n_elements=48, pitch=c0 / f0 / 2, z=0.0, f0=f0)
    dt = 1 / (30 * f0)
    t = np.arange(int(60e-6 / dt)) * dt
    P = (np.array([[xs]]), np.array([[zs]]))
    if layers is None:
        tx = zs / 1540.0
        rx = np.hypot(arr.element_x - xs, zs) / 1540.0
    else:
        tx = layers.plane_wave_time(*P, 0.0, arr.element_x.min(), 0.0)[0, 0]
        rx = np.array([layers.point_source_time(xe, 0.0, *P)[0, 0] for xe in arr.element_x])
    rf = np.stack([gaussian_pulse(t - (tx + rx[e]), f0) for e in range(arr.n_elements)])
    return arr, t, rf, xs, zs


def test_layered_speed_correction_refocuses_and_constant_c_does_not():
    ly = TwoLayerSpeed(15e-3, 1450.0, 1650.0)  # capa lenta (grasa) sobre medio rápido
    arr, t, rf, xs, zs = _point_rf(ly)
    x = np.linspace(-2e-3, 4e-3, 121)
    z = np.linspace(19e-3, 25e-3, 121)
    kw = dict(t_peak=arr.pulse_peak_time, f_number=1.5)
    fixed = bmode([rf], [0.0], t, arr.element_x, 0.0, x, z, c0=1540.0, **kw)
    corr = bmode([rf], [0.0], t, arr.element_x, 0.0, x, z, c0=1540.0, layers=ly, **kw)
    pk = lambda im: np.unravel_index(np.argmax(im.envelope), im.envelope.shape)  # noqa: E731
    jf, if_ = pk(fixed)
    jc, ic = pk(corr)
    err_fixed = np.hypot(z[jf] - zs, x[if_] - xs)
    err_corr = np.hypot(z[jc] - zs, x[ic] - xs)
    assert err_corr < 0.15e-3 < 0.3e-3 < err_fixed  # con 1540 fijo el eco aparece desplazado/borroso


def test_general_path_matches_original_das_with_default_options():
    arr, t, rf, xs, zs = _point_rf()
    x = np.linspace(-3e-3, 5e-3, 60)
    z = np.linspace(19e-3, 25e-3, 60)
    kw = dict(t_peak=arr.pulse_peak_time)
    a = bmode([rf, rf], [-5.0, 5.0], t, arr.element_x, 0.0, x, z, **kw)
    uniform = TwoLayerSpeed(5e-3, 1540, 1540)  # sin contraste: debe dar lo mismo que 1540 fijo
    b = bmode([rf, rf], [-5.0, 5.0], t, arr.element_x, 0.0, x, z, beamformer="das", layers=uniform, **kw)
    assert np.allclose(a.envelope, b.envelope, rtol=1e-7, atol=1e-9 * a.envelope.max())
    c = bmode([rf, rf], [-5.0, 5.0], t, arr.element_x, 0.0, x, z, rx_window="hann", **kw)
    assert np.array_equal(a.envelope, c.envelope)  # el valor por defecto es el de siempre


def test_coherence_factor_on_coherent_and_incoherent_channels():
    arr, t, rf, xs, zs = _point_rf()
    x = np.array([xs, xs + 4e-3])
    z = np.array([zs])
    iq = [analytic_signal(rf, axis=1)]
    C_, W = channel_cube(iq, [Transmit("plane")], t, arr.element_x, 0.0, x, z, 1540.0, arr.pulse_peak_time, 1.5)
    das = np.abs(combine_channels(C_, W, "das"))
    cf = np.abs(combine_channels(C_, W, "cf")) / np.maximum(das, 1e-30)
    assert cf[0, 0] > 0.9  # en el dispersor los canales están en fase: CF ≈ 1
    # Canales con fase aleatoria: CF ≈ 1/N
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((48, 4000)) + 1j * rng.standard_normal((48, 4000))
    Cn = noise.reshape(48, 40, 100)
    Wn = np.ones_like(Cn.real)
    cfn = np.abs(combine_channels(Cn, Wn, "cf")) / np.maximum(np.abs(combine_channels(Cn, Wn, "das")), 1e-30)
    assert cfn.mean() == pytest.approx(1 / 48, rel=0.35)


def test_dmas_peaks_at_scatterer_and_slsc_measures_coherence():
    arr, t, rf, xs, zs = _point_rf()
    x = np.linspace(-2e-3, 4e-3, 49)
    z = np.linspace(20e-3, 24e-3, 49)
    kw = dict(t_peak=arr.pulse_peak_time)
    im = bmode([rf], [0.0], t, arr.element_x, 0.0, x, z, beamformer="dmas", **kw)
    j, i = np.unravel_index(np.argmax(im.envelope), im.envelope.shape)
    assert abs(x[i] - xs) < 0.6e-3 and abs(z[j] - zs) < 0.6e-3
    # SLSC no es una amplitud sino una coherencia: ~max_lag con canales en fase, ~0 con canales independientes
    sl = bmode([rf], [0.0], t, arr.element_x, 0.0, np.array([xs]), np.array([zs]), beamformer="slsc", **kw)
    max_lag = arr.n_elements // 5
    assert sl.envelope[0, 0] > 0.7 * max_lag
    rng = np.random.default_rng(1)
    Cn = rng.standard_normal((48, 80, 30)) + 1j * rng.standard_normal((48, 80, 30))
    s_noise = combine_channels(Cn, np.ones_like(Cn.real), "slsc", kernel_px=8)
    assert abs(s_noise.mean()) < 0.05 * max_lag and s_noise.max() < 0.4 * max_lag
    with pytest.raises(ValueError):
        bmode([rf], [0.0], t, arr.element_x, 0.0, x, z, beamformer="mv", **kw)


# --------------------------------------------------------------------------
# Resolución: PSF de un dispersor puntual frente a la teoría
# --------------------------------------------------------------------------
F0_PSF = 1e6
XS, ZS = 0.0, 16e-3


def _psf_scan(mode, window, response=None, **kw):
    g, ph, arr = _setup(F0_PSF, 20e-3, 22e-3, response)
    med = point_scatterer_medium(g, F0_PSF, [(XS, ZS)])
    lam = C / F0_PSF
    sc = simulate_scan(ph, F0_PSF, mode, arr, c0=C, medium=med, n_mechanisms=0, tgc_db_per_cm=0, pixel=lam / 12,
                       max_depth=18e-3, rx_window=window, **kw)
    return sc, arr, lam


def test_plane_wave_psf_matches_theory():
    resp = ElementResponse(F0_PSF, "windowed", 2.0, "hann")
    sc, arr, lam = _psf_scan("plane", "rect", resp)
    r = psf_fwhm(sc.image.envelope, sc.image.x, sc.image.z, XS, ZS)
    # Lateral: 1.206·λ·F# (apertura rectangular, sólo recepción focalizada)
    assert r["lateral"] == pytest.approx(1.206 * lam * 1.5, rel=0.10)
    assert r["lateral"] == pytest.approx(
        theoretical_lateral_fwhm(arr.element_x, ZS, lam, 1.5, "rect", False, XS, arr.z), rel=0.10)
    # Axial: envolvente del pulso de ida y vuelta (h ∗ h)
    dt = sc.t[1] - sc.t[0]
    assert r["axial"] == pytest.approx(theoretical_axial_fwhm(arr.round_trip_pulse(dt), dt, C), rel=0.10)
    assert abs(r["offset_x"]) < lam / 8 and abs(r["offset_z"]) < lam / 4  # bien localizado


def test_synthetic_aperture_psf_is_two_way_narrower():
    sc, arr, lam = _psf_scan("sta", "rect", ElementResponse(F0_PSF, "windowed", 2.0, "hann"))
    r = psf_fwhm(sc.image.envelope, sc.image.x, sc.image.z, XS, ZS)
    assert len(sc.transmits) == arr.n_elements  # una emisión por elemento
    # Diagrama de ida y vuelta (sinc²): 0.886·λ·F# en paraxial; con F# = 1.5 la geometría exacta da ~8 % menos
    assert r["lateral"] == pytest.approx(
        theoretical_lateral_fwhm(arr.element_x, ZS, lam, 1.5, "rect", True, XS, arr.z), rel=0.10)
    assert r["lateral"] < 0.8 * 1.206 * lam * 1.5  # y bastante más estrecho que sólo con recepción focalizada


def test_focused_line_by_line_psf_at_focus():
    # Dispersor sobre el eje de una línea de emisión (las líneas están en las posiciones de los elementos)
    xs = _setup(F0_PSF, 20e-3, 22e-3)[2].element_x[7]
    g, ph, arr = _setup(F0_PSF, 20e-3, 22e-3, ElementResponse(F0_PSF, "windowed", 2.0, "hann"))
    lam = C / F0_PSF
    sc = simulate_scan(ph, F0_PSF, "focused", arr, c0=C, medium=point_scatterer_medium(g, F0_PSF, [(xs, ZS)]),
                       n_mechanisms=0, tgc_db_per_cm=0, pixel=lam / 12, max_depth=18e-3, rx_window="hann",
                       line_step=1, focus_depth=ZS, tx_f_number=2.0, tx_window="hann")
    r = psf_fwhm(sc.image.envelope, sc.image.x, sc.image.z, xs, ZS)
    th = theoretical_lateral_fwhm(arr.element_x, ZS, lam, 1.5, "hann", True, xs, arr.z, tx_f_number=2.0,
                                  tx_window="hann")
    assert r["lateral"] == pytest.approx(th, rel=0.10)
    assert all(tx.kind == "focused" for tx in sc.transmits) and len(sc.transmits) == arr.n_elements


def test_default_pulse_axial_resolution_and_hann_widens_lateral():
    sc, arr, lam = _psf_scan("plane", "hann")
    r = psf_fwhm(sc.image.envelope, sc.image.x, sc.image.z, XS, ZS)
    dt = sc.t[1] - sc.t[0]
    assert r["axial"] == pytest.approx(theoretical_axial_fwhm(arr.round_trip_pulse(dt), dt, C), rel=0.10)
    assert r["lateral"] > 1.206 * lam * 1.5 * 1.3  # Hann ensancha el lóbulo principal frente a rectangular


def test_axial_fwhm_of_a_step_reflector_is_the_round_trip_pulse():
    """Sin la física 2D del dispersor puntual: una interfaz plana da exactamente el pulso de ida y vuelta."""
    resp = ElementResponse(F0_PSF, "windowed", 2.0, "hann")
    g, ph, arr = _setup(F0_PSF, 20e-3, 22e-3, resp)
    med = _water_medium(g, F0_PSF)
    med.rho[g.z >= 14e-3, :] *= 1.3
    sc = simulate_scan(ph, F0_PSF, "plane", arr, c0=C, medium=med, n_mechanisms=0, tgc_db_per_cm=0,
                       pixel=C / F0_PSF / 24, max_depth=18e-3, rx_window="hann")
    z = sc.image.z
    prof = sc.image.envelope[:, np.argmin(np.abs(sc.image.x))]
    sel = np.abs(z - 14e-3) < 3e-3
    dt = sc.t[1] - sc.t[0]
    th = theoretical_axial_fwhm(arr.round_trip_pulse(dt), dt, C)
    assert fwhm_1d(z[sel], prof[sel]) == pytest.approx(th, rel=0.06)


def test_simulate_bmode_is_simulate_scan_plane():
    g, ph, arr = _setup(F0_PSF, 14e-3, 16e-3)
    med = point_scatterer_medium(g, F0_PSF, [(0.0, 10e-3)])
    kw = dict(array=arr, c0=C, medium=med, n_mechanisms=0, max_depth=14e-3)
    a = simulate_bmode(ph, F0_PSF, angles=(-4.0, 4.0), **kw)
    b = simulate_scan(ph, F0_PSF, "plane", angles=(-4.0, 4.0), **kw)
    assert np.array_equal(a.image.envelope, b.image.envelope)
    assert a.angles == [-4.0, 4.0] and len(a.rf) == 2


# --------------------------------------------------------------------------
# Speckle y quiste
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cyst_scan():
    f0 = 2e6
    g, ph, arr = _setup(f0, 20e-3, 24e-3, ElementResponse(f0, "windowed", 2.0, "hann"))
    X, Z = g.meshgrid()
    cyst = ((X - 0.0) ** 2 + (Z - 14e-3) ** 2) <= (3e-3) ** 2
    med = speckle_medium(g, f0, 0.05, seed=3, anechoic=cyst)
    sc = simulate_scan(ph, f0, "plane", arr, c0=C, medium=med, n_mechanisms=0, tgc_db_per_cm=0,
                       pixel=C / f0 / 8, max_depth=20e-3, rx_window="hann")
    return sc, arr


def test_fully_developed_speckle_follows_rayleigh_snr(cyst_scan):
    sc, _ = cyst_scan
    im = sc.image
    xx, zz = np.meshgrid(im.x, im.z)
    roi = (np.abs(xx) < 6e-3) & (zz > 6e-3) & (zz < 10.5e-3)  # lejos del quiste y con poca variación de nivel
    assert speckle_snr(im.envelope, roi) == pytest.approx(RAYLEIGH_SNR, rel=0.12)


def test_speckle_grain_matches_psf_autocorrelation(cyst_scan):
    sc, arr = cyst_scan
    f0 = 2e6
    g, ph, _ = _setup(f0, 20e-3, 24e-3)
    px = C / f0 / 8
    sc2 = simulate_scan(ph, f0, "plane", arr, c0=C, medium=point_scatterer_medium(g, f0, [(0.0, 8e-3)]),
                        n_mechanisms=0, tgc_db_per_cm=0, pixel=px, max_depth=20e-3, rx_window="hann")
    pred = predicted_grain_fwhm(sc2.image.iq, sc2.image.x, sc2.image.z, 0.0, 8e-3)
    im = sc.image
    xx, zz = np.meshgrid(im.x, im.z)
    roi = (np.abs(xx) < 6e-3) & (zz > 5e-3) & (zz < 10.5e-3)
    meas = speckle_grain_fwhm(im.envelope, roi, px, px)
    # Una sola realización: el tamaño del grano se estima con ~25 % de incertidumbre
    assert meas[0] == pytest.approx(pred[0], rel=0.25) and meas[1] == pytest.approx(pred[1], rel=0.30)


def test_anechoic_cyst_contrast_and_coherence_factor(cyst_scan):
    sc, arr = cyst_scan
    im = sc.image
    dist = np.hypot(*np.meshgrid(im.x, im.z - 14e-3))
    inside, back = dist < 2e-3, (dist > 4e-3) & (dist < 6.5e-3)
    assert contrast_db(im.envelope, inside, back) < -12.0  # el quiste se ve oscuro
    assert gcnr(im.envelope, inside, back) > 0.8
    iq = [analytic_signal(r, axis=1) for r in sc.rf]
    Cc, W = channel_cube(iq, sc.transmits, sc.t, arr.element_x, arr.z, im.x, im.z, C, arr.pulse_peak_time, 1.5, "hann")
    cf = np.abs(combine_channels(Cc, W, "cf"))
    # El factor de coherencia oscurece más el interior del quiste (clutter incoherente) que el fondo
    assert contrast_db(cf, inside, back) < contrast_db(im.envelope, inside, back) - 6.0
    dx = im.x[1] - im.x[0]
    assert disk_mask(im.x, im.z, (0, 14e-3), 2e-3).sum() * dx**2 == pytest.approx(np.pi * (2e-3) ** 2, rel=0.05)
