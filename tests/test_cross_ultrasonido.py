"""Comparación cruzada de wavesim con k-Wave y Devito usando sus resultados guardados (sin instalarlos).

Los datos de ``benchmarks/data/cross_ultrasonido/`` los genera ``benchmarks/cross_ultrasonido.py --save``
con k-wave-python 0.6.2 y Devito 4.8.23; aquí se recalcula sólo wavesim en los mismos montajes. Método,
tablas completas y lectura de las discrepancias en ``docs/validacion/cruzada_ultrasonido.md``.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data" / "cross_ultrasonido"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kw = _load("kwave_comparison")
cu = _load("cross_ultrasonido")
ev = cu.ev


def _npz(name):
    return np.load(DATA / name)


# ----------------------------------------------------------------------------
# k-Wave (fluido)
# ----------------------------------------------------------------------------
def _layers(cfl):
    d = _npz("kwave_A.npz")
    dx = float(d["dx"])
    s = kw.layers_setup(dx)
    t, p, dt, _ = kw.ws_run(s["rho"], s["cp"], dx, s["src"], s["rec"], kw.LAYERS["t_end"], cfl=cfl)
    mw = kw.layers_metrics(t, p, dx, s["j_int"], s["j_src"])
    mk = kw.layers_metrics(d["t"], d["p"].astype(float), dx, s["j_int"], s["j_src"])
    shape = kw.compare(kw.spectrum(t, p), kw.spectrum(d["t"], d["p"].astype(float)))
    return mw, mk, shape, s, dx, dt


def test_k_wave_dos_capas_r_t_y_retardo_de_grupo_del_esquema():
    mw, mk, shape, s, dx, dt = _layers(0.9)
    # R y T: los dos códigos comparten el error de discretizar el salto (+1.3 % en R a 15 ppw)
    assert abs(mw["R"] / mk["R"] - 1) < 0.005 and abs(mw["T"] / mk["T"] - 1) < 0.001
    # con CFL 0.9 la envolvente de wavesim llega antes: es el retardo de grupo de su propio esquema
    zi = (s["j_int"] - 0.5) * dx
    pred = 2 * (zi - s["j_r0"] * dx) * kw.ws_group_delay_error(kw.LAYERS["c1"], dx, dt)
    assert abs((mw["dt_ref"] - mk["dt_ref"]) - pred) < 8e-9
    assert np.all(shape["aligned"] < 0.03)


def test_k_wave_dos_capas_con_cfl_03_coinciden_tiempos_y_forma():
    mw, mk, shape, *_ = _layers(0.3)
    assert abs(mw["dt_ref"] / mk["dt_ref"] - 1) < 0.001 and abs(mw["dt_tr"] / mk["dt_tr"] - 1) < 0.001
    assert np.all(shape["aligned"] < 0.01) and np.all(np.abs(shape["shift"]) < 5e-9)


def test_k_wave_atenuacion_ley_de_potencia_frente_a_k_wave_ajustado():
    d = _npz("kwave_B.npz")
    g = kw.POINT
    s = kw.point_setup(float(d["dx"]))
    t, p, _, _ = kw.ws_run(s["rho"], s["cp"], float(d["dx"]), s["src"], s["rec"], g["t_end"], alpha_db=g["a0"])
    a_w, c_w = kw.point_estimate(t, p, s["r"], g["c0"])
    a_k, c_k = kw.point_estimate(d["t_matched"], d["p_matched"].astype(float), s["r"], g["c0"])
    assert np.all(np.abs(a_w / a_k - 1) < 0.015)
    assert np.all(np.abs(c_w / c_k - 1) < 0.003)
    # cada código sigue su propio modelo continuo: k-Wave "nominal" no da la ley pedida por convención
    c0k, a0k = (float(v) for v in d["kw_matched"])
    for (tt, pp), (c0, a0) in (((d["t_matched"], d["p_matched"]), (c0k, a0k)),
                               ((d["t_nominal"], d["p_nominal"]), (g["c0"], g["a0"]))):
        a_m, c_m = kw.point_estimate(tt, pp.astype(float), s["r"], g["c0"])
        c_mod, a_mod = kw.kwave_model(kw.POINT_FREQS, c0, a0, g["y"])
        assert np.all(np.abs(a_m / a_mod - 1) < 0.005) and np.all(np.abs(c_m / c_mod - 1) < 0.001)
    c_mod, a_mod = kw.kwave_model(kw.POINT_FREQS, g["c0"], g["a0"], g["y"])
    assert np.all(c_mod > 1630) and np.all(a_mod / (g["a0"] * (kw.POINT_FREQS / 1e6) ** g["y"] * 100
                                                      / kw.DB_PER_NP) < 0.93)


@pytest.mark.parametrize("cfl", [0.3, 0.9])
def test_k_wave_medio_tipo_tejido_trazas_del_array(cfl):
    d = _npz("kwave_C.npz")
    dx = float(d["dx"])
    s = kw.tissue_setup(dx)
    t, p, _, _ = kw.ws_run(s["rho"], s["cp"], dx, s["src"], s["rec"], kw.TISSUE["t_end"], cfl=cfl)
    pk = d["p"].astype(float)
    We, Wt = kw.tissue_spectra(t, p, s["t_direct"])
    Ke, Kt = kw.tissue_spectra(d["t"], pk, s["t_direct"])
    ce, ct = kw.compare(We, Ke), kw.compare(Wt, Kt)
    rel = (kw.tissue_arrivals(t, p) - kw.tissue_arrivals(d["t"], pk)) / (kw.tissue_arrivals(d["t"], pk)
                                                                         - kw.pulse_peak_time(kw.F0, kw.BW))
    if cfl == 0.3:
        assert kw.total_error(We, Ke, ce["shift"])[0] < 0.015 and kw.total_error(Wt, Kt, ct["shift"])[0] < 0.006
        assert np.all(np.abs(rel) < 0.001)
    else:
        # dispersión temporal de wavesim a CFL 0.9 (15 ppw): la envolvente llega ~0.6 % antes y deforma el pulso
        assert np.all((rel > -0.008) & (rel < -0.004))
        assert kw.total_error(Wt, Kt, ct["shift"])[0] < 0.08


# ----------------------------------------------------------------------------
# Devito (elástico)
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def beams_12ppw():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {th: cu.beam_at(th, 12.0)[:2] for th in cu.D1_ANGLES}


def test_devito_interfaz_agua_cortical_incidencia_normal_y_oblicua(beams_12ppw):
    d = _npz("devito_D.npz")
    R_an, vs_an = cu.d1_analytic()
    for a, th in enumerate(cu.D1_ANGLES):
        R, vs = beams_12ppw[th]
        R4 = d["R_o4_ppw12"][a]
        e_ws = np.abs(np.abs(R) / abs(R_an[a]) - 1).max()
        e_dv = np.abs(np.abs(R4) / abs(R_an[a]) - 1).max()
        if th == 0:
            # sin cizalla en la interfaz los dos esquemas son el mismo: sólo cambia el borde absorbente
            assert np.abs(np.abs(R) / np.abs(R4) - 1).max() < 5e-4
        else:
            # Devito no tiene la corrección de deslizamiento: más error que wavesim frente a la analítica
            assert e_ws < 0.01 and e_dv > e_ws
            assert np.abs(vs / vs_an[a] - 1).max() < 0.01


def test_devito_orden_4_igual_a_wavesim_sin_correccion_de_interfaz():
    d = _npz("devito_D.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R, vs, _ = cu.beam_at(45.0, 12.0, cu.no_slip_factory)
    a = cu.D1_ANGLES.index(45.0)
    assert np.abs(np.abs(R) / np.abs(d["R_o4_ppw12"][a]) - 1).max() < 0.002
    assert np.abs(vs / d["vs_o4_ppw12"][a] - 1).max() < 0.002


def test_devito_placa_de_cortical_en_agua_s0_y_a0():
    d = _npz("devito_D.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ev.lamb_case(ev.WATER, dx=0.25e-3)
    band = ev.U5_FREQS <= 325e3  # S0 resuelto en todos los códigos y resoluciones
    s_dv, a_dv = d["cS_o4_dx250"], d["cA_o4_dx250"]
    assert np.abs(r["S"]["c"][band] / s_dv[band] - 1).max() < 0.003
    # A0: la corrección de interfaz de wavesim lo frena ~1 % frente a Devito con 12 celdas en el espesor
    da = r["A"]["c"] / a_dv - 1
    assert np.all((da > -0.015) & (da < -0.005))
    for c in (r["S"]["c"][band], s_dv[band]):
        assert np.abs(c / r["S"]["c_ref"][band] - 1).max() < 0.005
    for c in (r["A"]["c"], a_dv):
        assert np.abs(c / r["A"]["c_ref"] - 1).max() < 0.011
