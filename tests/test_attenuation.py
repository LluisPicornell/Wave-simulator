"""Atenuación de ley de potencia del GSLS: ajuste por mecanismo, dispersión causal y medida en simulación."""

import warnings

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.core.signals import gaussian_pulse
from wavesim.phantoms import longitudinal_limb
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom
from wavesim.ultrasound import ElasticMedium, ElasticSolver, Receiver, Source
from wavesim.ultrasound import attenuation as att

F0 = 1e6
DB_PER_NP = 20.0 / np.log(10.0)
NARROW = F0 * np.linspace(0.5, 1.8, 60)
WIDE = F0 * np.linspace(0.3, 2.5, 90)


def _law(a0_db, y, f):
    return a0_db * (np.asarray(f) / 1e6) ** y * 100.0 / DB_PER_NP


def _weights(a0_db, y, c, n_mech):
    ts, band = att.relaxation_times(F0, n_mech), att.fit_band(F0, n_mech)
    return att.mechanism_weights(_law(a0_db, y, band)[None], np.array([c]), band, ts, F0)[0], ts


def _max_error(a0_db, y, c, n_mech, f):
    w, ts = _weights(a0_db, y, c, n_mech)
    _, a = att.model_wave(c, w, ts, F0, f)
    return np.abs(a / _law(a0_db, y, f) - 1).max()


@pytest.mark.parametrize("y", [1.0, 1.1, 1.5, 2.0])
def test_ley_de_potencia_con_dos_y_tres_mecanismos(y):
    # Q bajo (tejido blando) y alto (cortical; con y = 2 y tantas pérdidas el error sube al 3-4 %,
    # pero ningún tejido es así: el agua, con y = 2, tiene Q⁻¹ ~ 1e-5)
    for a0, c in ((1.0, 1540.0), (5.0, 3500.0))[: 1 if y > 1.5 else 2]:
        assert _max_error(a0, y, c, 2, NARROW) <= 0.02
        assert _max_error(a0, y, c, 3, NARROW) <= 0.02
        assert _max_error(a0, y, c, 3, WIDE) <= 0.02
        # el método tau con un mecanismo queda muy lejos (referencia de la mejora)
        assert _max_error(a0, y, c, 1, NARROW) > 0.13


def test_pesos_no_negativos_y_velocidad_en_f0():
    for n in (2, 3, 4):
        w, ts = _weights(15.0, 1.0, 2100.0, n)  # hueso trabecular: el de más pérdidas
        assert np.all(w >= 0) and w.sum() > 0
        c, _ = att.model_wave(2100.0, w, ts, F0, F0)
        assert c == pytest.approx(2100.0, rel=1e-12)


def test_un_mecanismo_es_el_metodo_tau():
    """Con L = 1 nada cambia: tau_1 = 1/(2π f0) y el peso por mínimos cuadrados de Q⁻¹."""
    ts, band = att.relaxation_times(F0, 1), att.fit_band(F0, 1)
    assert ts[0] == pytest.approx(1 / (2 * np.pi * F0))
    a = _law(5.0, 1.0, band)
    w = att.mechanism_weights(a[None], np.array([3500.0]), band, ts, F0)
    assert w[0, 0] == att.fit_tau(att.target_qinv(a, 3500.0, band), ts, band)


def test_dispersion_causal_frente_a_kramers_kronig():
    """En la banda, c(f) del modelo sigue la relación de Kramers-Kronig de la ley de potencia."""
    for a0, y, c in ((5.0, 1.0, 3500.0), (2.0, 1.1, 1580.0), (1.0, 1.5, 1540.0)):
        w, ts = _weights(a0, y, c, 3)
        c_model, _ = att.model_wave(c, w, ts, F0, NARROW)
        c_kk = att.power_law_phase_speed(c, _law(a0, y, 1e6), y, F0, NARROW)
        assert np.abs(c_model - c_kk).max() <= 0.0015 * c
        # la dispersión es normal (c crece con f) y del tamaño de la de K-K
        rise, rise_kk = c_model[-1] - c_model[0], c_kk[-1] - c_kk[0]
        assert rise > 0 and rise == pytest.approx(rise_kk, rel=0.15)


def test_paso_de_tiempo_con_tres_mecanismos():
    """El mecanismo de alta frecuencia apenas rigidiza la cortical: el paso baja < 3 % respecto a L = 1."""
    p = longitudinal_limb("tibia", None, width=4e-3, depth=10e-3, dx=0.1e-3)
    dt = {n: ElasticSolver(p, 2e6, n_mechanisms=n, backend="numpy").dt for n in (1, 2, 3)}
    assert dt[3] > 0.97 * dt[1]
    assert dt[2] > 0.92 * dt[1]


def _plane_wave(c, w, ts, ppw=16, dz=12e-3):
    """Onda plana 1D (la fuente cubre todas las columnas: el campo no varía en x)."""
    npml = 10
    dx = c / (F0 * ppw)
    nx, js = 2 * npml + 4, npml + 6
    j1 = js + int(round(3e-3 / dx))
    j2 = j1 + int(round(dz / dx))
    g = Grid2D(nx, j2 + int(round(4e-6 * c / dx)) + npml, dx)
    one = np.ones(g.shape)
    wmap = np.asarray(w)[:, None, None] * one
    med = ElasticMedium(1000 * one, c * one, 0 * one, wmap, 0 * wmap, ts, F0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ElasticSolver(Phantom(g, np.full(g.shape, TISSUE_INDEX["water"], np.int16)), F0, npml=npml, medium=med)
    nt = int(np.ceil((4e-6 + (j2 - js) * dx / c + 4e-6) / s.dt))
    t = s.time_axis(nt)
    s.add_source(Source(np.full(nx, js), np.arange(nx), np.zeros(nx, int), gaussian_pulse(t, F0, 2.0)[None, :]))
    s.add_receiver("r", Receiver(np.array([j1, j2]), np.array([nx // 2] * 2), np.array([0, 1]), 2, "p"))
    tr = s.run(nt, track_peak=False).traces["r"]
    spec = np.fft.rfft(tr, 1 << 15, axis=1)
    fr = np.fft.rfftfreq(1 << 15, s.dt)
    k = np.array([np.argmin(np.abs(fr - f)) for f in NARROW[::6]])
    ratio = spec[1, k] / spec[0, k]
    d = (j2 - j1) * dx
    # la ambigüedad de 2π de la fase se resuelve con c ≈ c0
    phase = (-np.angle(ratio)) % (2 * np.pi)
    phase += 2 * np.pi * np.round((2 * np.pi * NARROW[::6] * d / c - phase) / (2 * np.pi))
    return -np.log(np.abs(ratio)) / d, 2 * np.pi * NARROW[::6] * d / phase


@pytest.mark.parametrize("y", [1.0, 2.0])
def test_atenuacion_medida_en_simulacion(y):
    """alpha(f) y c(f) medidos entre dos receptores de una onda plana con 3 mecanismos."""
    a0, c = 2.0, 1540.0
    w, ts = _weights(a0, y, c, 3)
    a_ref, c_ref = _plane_wave(c, np.zeros(0), np.zeros(0))  # sin pérdidas: dispersión numérica y ventana
    a_sim, c_sim = _plane_wave(c, w, ts)
    f = NARROW[::6]
    alpha = a_sim - a_ref
    assert np.abs(alpha / _law(a0, y, f) - 1).max() <= 0.025
    c_model, a_model = att.model_wave(c, w, ts, F0, f)
    assert np.abs(alpha / a_model - 1).max() <= 0.01
    c_corr = c_sim * c / c_ref
    assert np.abs(c_corr - c_model).max() <= 0.0015 * c
    # la subida de c en la banda (dispersión causal, ~7 m/s con y = 1) también se mide
    if y == 1.0:
        assert c_corr[-1] - c_corr[0] == pytest.approx(c_model[-1] - c_model[0], rel=0.3)
