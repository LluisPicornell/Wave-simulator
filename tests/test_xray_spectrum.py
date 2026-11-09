"""Espectro de tubo de rayos X: líneas K, filtración, ánodo y capa hemirreductora (HVL)."""

import numpy as np
import pytest

from wavesim.em import xray
from wavesim.materials import xray as xd

# HVL (mm Al) y energía media de SpekPy 2.5.4 (Spek(kvp, th=12).filter("Al", t)), calculadas
# en la sesión de desarrollo; el paquete no es una dependencia.
SPEKPY = {(50, 2.5): (1.750, 32.22), (70, 2.5): (2.397, 39.40), (80, 2.5): (2.734, 42.90),
          (100, 2.5): (3.458, 49.14), (120, 2.5): (4.219, 54.44), (120, 4.0): (5.305, 57.06)}
# IEC 61267:2005, RQR (kVp, 1.ª HVL en mm Al), tal como los reproduce la presentación del IAEA/TRS-457
RQR = {"RQR3": (50, 1.78), "RQR5": (70, 2.58), "RQR8": (100, 3.97), "RQR9": (120, 5.00)}


def test_espectro_por_defecto_sin_cambios():
    sp = xray.tube_spectrum(80.0, 2.5)
    e = np.arange(10.5, 80.0, 1.0)
    phi = (80.0 - e) / e * np.exp(-xd.linear_attenuation("aluminum", e) * 2.5e-3)
    np.testing.assert_allclose(sp.energies_kev, e)
    np.testing.assert_allclose(sp.weights, phi / phi.sum(), rtol=1e-12)


def test_lineas_k_solo_sobre_el_borde():
    bajo = xray.tube_spectrum(65.0, 0.0, characteristic=True)
    assert bajo.energies_kev.size == xray.tube_spectrum(65.0, 0.0).energies_kev.size
    sp = xray.tube_spectrum(100.0, 1.0, characteristic=True)
    base = xray.tube_spectrum(100.0, 1.0)
    assert sp.energies_kev.size == base.energies_kev.size + 4
    for e, _ in xray.W_K_LINES:
        i = int(np.argmin(abs(sp.energies_kev - e)))
        assert sp.energies_kev[i] == pytest.approx(e)
    i = int(np.argmin(abs(sp.energies_kev - 59.318)))
    assert sp.weights[i] > 1.15 * sp.weights[i - 4]  # Kα1 sobresale del continuo
    assert np.all(np.diff(sp.energies_kev) >= 0)
    assert sp.weights.sum() == pytest.approx(1.0)
    # Kα1 más intensa que Kα2 y que Kβ1
    w = {round(e, 1): sp.weights[np.argmin(abs(sp.energies_kev - e))] for e, _ in xray.W_K_LINES}
    assert w[59.3] > w[58.0] > w[67.2] > w[69.1]


def test_hvl_monoenergetica_es_ln2_sobre_mu():
    hvl = xray.half_value_layer(xray.monoenergetic(60.0))
    assert hvl == pytest.approx(np.log(2) / xd.linear_attenuation("aluminum", 60.0) * 1e3, rel=1e-6)


def test_filtracion_y_anodo_endurecen_el_haz():
    h0 = xray.half_value_layer(xray.tube_spectrum(100.0, 2.5))
    assert xray.half_value_layer(xray.tube_spectrum(100.0, 4.0)) > h0
    assert xray.half_value_layer(xray.tube_spectrum(100.0, 2.5, cu_filter_mm=0.2)) > 1.3 * h0
    assert xray.half_value_layer(xray.tube_spectrum(100.0, 2.5, anode_angle_deg=12.0)) > h0
    # Efecto talón: hacia el lado del ánodo el ángulo de despegue es menor y el haz es más duro
    anode = xray.half_value_layer(xray.tube_spectrum(100.0, 2.5, anode_angle_deg=12.0, takeoff_offset_deg=5.0))
    cathode = xray.half_value_layer(xray.tube_spectrum(100.0, 2.5, anode_angle_deg=12.0, takeoff_offset_deg=-5.0))
    assert anode > cathode


@pytest.mark.parametrize("quality", sorted(RQR))
def test_ajuste_de_filtracion_reproduce_la_hvl_iec(quality):
    kvp, hvl = RQR[quality]
    kw = dict(characteristic=True, anode_angle_deg=12.0)
    t = xray.fit_al_filtration(kvp, hvl, **kw)
    assert 2.0 < t < 5.0  # filtración total de un tubo diagnóstico
    assert xray.half_value_layer(xray.tube_spectrum(kvp, t, **kw)) == pytest.approx(hvl, rel=1e-3)


@pytest.mark.parametrize("case", sorted(SPEKPY))
def test_hvl_y_energia_media_frente_a_spekpy(case):
    kvp, t = case
    hvl_ref, em_ref = SPEKPY[case]
    sp = xray.tube_spectrum(kvp, t, characteristic=True, anode_angle_deg=12.0)
    assert xray.half_value_layer(sp) == pytest.approx(hvl_ref, rel=0.08)
    assert sp.mean_energy == pytest.approx(em_ref, rel=0.04)
