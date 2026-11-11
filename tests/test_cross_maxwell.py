"""Comparación cruzada del FDTD de Maxwell de wavesim con gprMax, sin tener gprMax instalado.

Las trazas de gprMax (4.0.1, mismos polos de Debye) están en ``benchmarks/data/cross_maxwell/``; se generaron con
``benchmarks/cross_maxwell.py`` y aquí se repite sólo la parte de wavesim con las mallas más gruesas. Resultados
completos y lectura en ``docs/validacion/cruzada_maxwell.md``.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


@pytest.fixture(scope="module")
def cm():
    spec = importlib.util.spec_from_file_location("cross_maxwell", BENCH / "cross_maxwell.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cross_maxwell"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_datos_guardados_pequenos(cm):
    files = list(cm.DATA.glob("*.npz"))
    assert len(files) >= 6
    assert sum(f.stat().st_size for f in files) < 2e6


def test_pintor_deja_cada_nodo_con_su_etiqueta(cm):
    """Emula la regla de gprMax (cada caja pinta los nodos de ambos extremos) y comprueba el teselado."""
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 3, size=(9, 12)).astype(np.int16) + 1  # 1 = aire, 2 = piel, 3 = grasa
    lines = cm.painter_boxes(lab, 1.0)
    nodes = np.zeros((lab.shape[0] + 1, lab.shape[1] + 1), int)
    ids = {cm._mat_id(v): v for v in (1, 2, 3)}
    for ln in lines:
        t = ln.split()
        x0, y0, x1, y1 = (int(round(float(v))) for v in (t[1], t[2], t[4], t[5]))
        nodes[y0:y1 + 1, x0:x1 + 1] = ids[t[7]]
    np.testing.assert_array_equal(nodes[:-1, :-1], lab)


def test_losa_wavesim_coincide_con_gprmax_y_con_la_solucion_exacta(cm):
    r = cm.run_slab(None, 0.5e-3)
    ex, w, g = r["exact"], r["t_ws"], r["t_gp"]
    # mismos polos y misma malla: los dos esquemas de Yee dan lo mismo (medido: 0.013 dB, 0.00 grados)
    assert np.abs(cm.db(g / w)).max() < 0.05
    assert np.abs(np.degrees(np.angle(g / w))).max() < 0.1
    # y los dos tienen el error de dispersión numérica de la malla (medido: 0.77 dB y 5.8 grados a 6 GHz)
    for t in (w, g):
        assert np.abs(cm.db(t / ex)).max() < 1.0
        assert np.abs(np.degrees(np.angle(t / ex))).max() < 7.0


def test_cilindro_wavesim_coincide_con_gprmax_y_mie(cm):
    r = cm.run_cylinder(None, 1e-3)
    inside = np.abs(r["ks"] * r["dx"]) <= r["a"]
    for ln in ("z", "x"):
        mie, gp = r[f"mie_{ln}"], r[f"gp_{ln}"]
        for key, tol in (("bb", 0.005), ("nb", 0.015)):  # medido: 0.11 % y 0.57 %
            v = r[f"{key}_{ln}"]
            assert cm.rel_l2(v, gp) < tol
            assert cm.rel_l2(v[inside], mie[inside]) < 0.06  # escalera a 1 mm: ~4 %
            assert cm.rel_l2(v[~inside], mie[~inside]) < 0.02
        assert cm.rel_l2(gp[inside], mie[inside]) < 0.06


def test_pierna_mapas_y_sar_coinciden_con_gprmax(cm):
    r = cm.run_leg(None, 1e-3, broadband=False)
    rows = {(t, q): (med, p95) for t, c, q, med, p95 in cm.leg_table(r)}
    med, p95 = rows[("pierna (sin agua)", "|E|")]
    assert med < 0.015 and p95 < 0.04  # medido: 0.49 % y 1.30 %
    med, p95 = rows[("pierna (sin agua)", "SAR")]
    assert med < 0.03 and p95 < 0.06  # medido: 1.05 % y 2.39 %
    body = r["labels"] != cm.WATER
    pk = lambda k: np.nanmax(np.where(body[:-1, :-1], r[k].astype(float)[:-1, :-1], np.nan))  # noqa: E731
    assert abs(pk("SARc_nb") / pk("SARc_gp") - 1) < 0.03  # medido: -0.7 %
