"""Detector de rayos X, MTF por el método del borde y espectro de ruido (NPS)."""

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.em import xray
from wavesim.em import xray_detector as xdet
from wavesim.em.xray_phantoms import disk, rasterize
from wavesim.phantoms import TISSUE_INDEX, Phantom


def erf(x):
    # Aproximación de Abramowitz-Stegun 7.1.26 (error < 1.5e-7), sin SciPy
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    y = 1 - poly * np.exp(-x * x)
    return np.sign(x) * y


def tilted_edge(n=96, angle_deg=4.0, sigma=0.0):
    """Borde recto inclinado, bordeado por la gaussiana de desviación ``sigma`` (en unidades de ``px``)."""
    jj, ii = np.indices((n, n))
    d = ((ii - n / 2) * np.cos(np.radians(angle_deg)) - (jj - n / 2) * np.sin(np.radians(angle_deg)))
    if sigma == 0:
        return (d > 0).astype(float)
    return 0.5 * (1 + erf(d / (np.sqrt(2) * sigma)))


def test_mtf_del_borde_de_una_gaussiana_conocida():
    px, sigma = 1.0e-4, 1.5e-4
    img = tilted_edge(sigma=sigma / px)
    f, mtf = xdet.mtf_from_edge_samples(*xdet.slanted_edge_samples(img, px), px, oversample=4, half_width=25 * px)
    expected = np.exp(-2 * (np.pi * sigma * f) ** 2)
    sel = f < 0.5 / px
    sel &= expected > 0.05
    assert np.abs(mtf - expected)[sel].max() < 0.04
    f50 = xdet.frequency_at_mtf(f, mtf)
    assert f50 == pytest.approx(np.sqrt(np.log(2) / 2) / (np.pi * sigma), rel=0.04)


def test_mtf_de_apertura_de_pixel():
    # Borde ideal muestreado por píxeles de lado px: el borde recto tiene MTF sinc de la apertura
    # (promedio en el área del píxel); con submuestreo 8x8 se aproxima el integrado
    n, px, sub = 96, 1.0, 8
    off = (np.arange(sub) + 0.5) / sub - 0.5
    jj, ii = np.indices((n, n))
    img = np.zeros((n, n))
    ang = np.radians(3.0)
    for oj in off:
        for oi in off:
            d = ((ii + oi - n / 2) * np.cos(ang) - (jj + oj - n / 2) * np.sin(ang))
            img += (d > 0)
    img /= sub * sub
    f, mtf = xdet.mtf_from_edge_samples(*xdet.slanted_edge_samples(img, px), px, oversample=4, half_width=20)
    sel = f < 0.8
    assert np.abs(mtf - np.abs(np.sinc(f)))[sel].max() < 0.06


def test_nps_de_ruido_blanco_y_parseval():
    rng = np.random.default_rng(0)
    px, sigma = 0.5e-3, 3.0
    img = sigma * rng.standard_normal((256, 256))
    fz, fx, nps = xdet.noise_power_spectrum(img, px, block=32)
    assert nps.mean() == pytest.approx(sigma**2 * px**2, rel=0.05)
    assert np.std(nps) / nps.mean() < 0.15  # plano
    df = fx[1] - fx[0]
    assert nps.sum() * df * df == pytest.approx(sigma**2, rel=0.05)


def test_nps_de_ruido_correlado_tiene_la_forma_del_filtro():
    rng = np.random.default_rng(1)
    white = rng.standard_normal((300, 300))
    kern = np.array([1.0, 1.0, 1.0]) / 3  # promedio de 3 píxeles en x
    img = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, white)[:, 10:-10]
    fz, fx, nps = xdet.noise_power_spectrum(img, 1.0, block=32)
    h = np.abs(np.fft.fftshift(np.fft.fft(kern, 32))) ** 2  # |H(fx)|²
    profile = nps.mean(axis=0)  # media sobre fz
    assert np.corrcoef(profile, h)[0, 1] > 0.98
    # (el bin central se ignora: la resta de la media lo anula)
    assert profile[18] / profile[24] == pytest.approx(h[18] / h[24], rel=0.25)  # fuga espectral
    f, rad = xdet.radial_average(fx, nps, n_bins=8)
    assert rad[0] > rad[-1]


def test_modelo_de_detector_mtf_y_ruido_electronico():
    model = xdet.DetectorModel(pixel_pitch=0.2e-3, blur_sigma=0.3e-3, electronic_noise=0.0)
    f = np.array([0.0, 1000.0])
    assert model.mtf(f)[0] == 1.0
    assert model.mtf(f)[1] == pytest.approx(np.exp(-2 * (np.pi * 0.3e-3 * 1000) ** 2) * np.sinc(0.2), rel=1e-12)
    # Varianza de las cuentas: Poisson + electrónico
    rng = np.random.default_rng(3)
    expected = np.full((200, 400), 500.0)
    out = xdet.detect_counts(expected, 1e-3, xdet.DetectorModel(electronic_noise=20.0), rng)
    assert out.var() == pytest.approx(500.0 + 400.0, rel=0.04)
    assert out.mean() == pytest.approx(500.0, rel=0.01)
    plain = xdet.detect_counts(expected, 1e-3, None, np.random.default_rng(3))
    assert plain.var() == pytest.approx(500.0, rel=0.04)


def edge_slab(dx=0.25e-3):
    g = Grid2D.from_size(40e-3, 20e-3, dx, x0=-20e-3)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[:, g.x >= 0] = TISSUE_INDEX["water"]
    return Phantom(g, lab)


def test_radiografia_de_un_borde_recupera_la_mtf_del_detector():
    p = edge_slab()
    sp = xray.monoenergetic(60.0)
    sigma = 1.0e-3
    det = xdet.DetectorModel(blur_sigma=sigma)
    base = xray.acquire(p, [0.0], sp, photons=None)
    blurred = xray.acquire(p, [0.0], sp, photons=None, detector=det)
    spacing = base.det[1] - base.det[0]
    # Perfil de transmisión (intensidad) en torno al borde
    esf = blurred.intensity[0]
    sel = np.abs(blurred.det) < 8e-3
    f, mtf = xdet.mtf_from_edge_samples(blurred.det[sel], esf[sel], spacing, oversample=1, half_width=7e-3)
    f0, m0 = xdet.mtf_from_edge_samples(base.det[sel], base.intensity[0][sel], spacing, oversample=1, half_width=7e-3)
    ratio = mtf / m0
    keep = (f < 150.0) & (f > 0)
    assert np.abs(ratio - np.exp(-2 * (np.pi * sigma * f) ** 2))[keep].max() < 0.05


def test_ruido_del_tac_escala_con_los_fotones_y_el_ruido_electronico_lo_aumenta():
    g = Grid2D.from_size(0.1, 0.1, 1.0e-3, x0=-0.05, z0=-0.05)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[rasterize(disk(0.04, 1.0), g) > 0.5] = TISSUE_INDEX["water"]
    p = Phantom(g, lab)
    sp = xray.monoenergetic(60.0)
    X, Z = g.meshgrid()
    roi = np.hypot(X, Z) < 0.02

    def sd(photons, detector=None):
        ct = xray.ct_scan(p, 90, sp, photons, rng=np.random.default_rng(4), detector=detector)
        return float(ct.hu[roi].std())

    s_hi, s_lo = sd(4e4), sd(1e4)
    assert s_lo / s_hi == pytest.approx(2.0, rel=0.2)  # sigma ∝ 1/sqrt(fotones)
    assert sd(1e4, xdet.DetectorModel(electronic_noise=150.0)) > 1.3 * s_lo
    # El HU medio del agua no cambia por el ruido ni el desenfoque suave
    ct = xray.ct_scan(p, 90, sp, 1e4, rng=np.random.default_rng(4), detector=xdet.DetectorModel(blur_sigma=0.5e-3))
    assert abs(ct.hu[roi].mean()) < 25
