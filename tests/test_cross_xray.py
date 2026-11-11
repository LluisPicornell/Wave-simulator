"""Comparación cruzada de rayos X / TAC con ASTRA, SpekPy y XCIST, sin tener instaladas las herramientas.

Las referencias son pequeñas y están guardadas en ``benchmarks/data/cross_xray/`` (las genera
``benchmarks/cross_xray.py --save``; método y tablas en ``docs/validacion/cruzada_xray.md``). Las tolerancias son
~1.5-2 veces lo medido, para detectar regresiones sin ser frágiles.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.em import fanbeam as fb
from wavesim.em.projection import detector_axis, fbp, project

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data" / "cross_xray"

_spec = importlib.util.spec_from_file_location("cross_xray", ROOT / "benchmarks" / "cross_xray.py")
cx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cx)


@pytest.fixture(scope="module")
def astra():
    return np.load(DATA / "astra.npz")


def _case(ref, name):
    nx, nz, dx, x0, z0 = ref[f"{name}_grid"]
    g = Grid2D(int(nx), int(nz), float(dx), float(x0), float(z0))
    sid, sdd = ref[f"{name}_fan"]
    geom = fb.FanGeometry.covering(g, float(sid), float(sdd), curved=False)
    mask = np.hypot(*g.meshgrid()) < (0.076 if name == "sl" else 0.04)
    return g, geom, mask, ref[f"{name}_img"].astype(float)


def rmse(a, b, mask):
    return float(np.sqrt(np.mean((a - b)[mask] ** 2)))


@pytest.mark.parametrize("name,tol", [("sl", 0.006), ("leg", 0.002)])
def test_proyector_paralelo_frente_a_astra(astra, name, tol):
    """Mismo sinograma que el núcleo de tira de ASTRA (ángulos, centro y escala coinciden sin transformar)."""
    g, _, _, img = _case(astra, name)
    a = astra[f"{name}_a_par"]
    ref = astra[f"{name}_par_strip"].astype(float)
    w = project(img, g, a, detector_axis(g))[0]
    err = np.linalg.norm(w - ref) / np.linalg.norm(ref)
    assert err < tol
    # La convención importa: invertir el detector da un error mucho mayor (aun en la pierna, casi simétrica)
    assert np.linalg.norm(w[:, ::-1] - ref) / np.linalg.norm(ref) > 10 * err


@pytest.mark.parametrize("name,tol", [("sl", 0.007), ("leg", 0.002)])
def test_proyector_de_abanico_frente_a_astra(astra, name, tol):
    """Abanico plano frente a ``strip_fanflat``; el núcleo de línea de ASTRA se aleja más (3.5 % en Shepp-Logan)."""
    g, geom, _, img = _case(astra, name)
    a = astra[f"{name}_a_fan"]
    w = fb.fan_project(img, g, geom, a)[0]
    ref = astra[f"{name}_fan_strip"].astype(float)
    err = np.linalg.norm(w - ref) / np.linalg.norm(ref)
    assert err < tol
    line = astra[f"{name}_fan_line"].astype(float)
    assert np.linalg.norm(w - line) / np.linalg.norm(line) > 2 * err
    # ASTRA gira 180° y invierte el detector respecto a wavesim: sin esa transformación no coincide
    assert np.linalg.norm(w[:, ::-1] - ref) / np.linalg.norm(ref) > 10 * err


@pytest.mark.parametrize("name,filt,tol", [("sl", "ram-lak", 0.08), ("sl", "shepp-logan", 0.08),
                                           ("leg", "ram-lak", 0.06)])
def test_fbp_paralelo_frente_a_astra(astra, name, filt, tol):
    """La diferencia entre FBP es una fracción pequeña del error de cada uno frente al objeto."""
    g, _, mask, img = _case(astra, name)
    data = astra[f"{name}_fbp_data"].astype(float)
    rec = fbp(data, astra[f"{name}_a_par"], detector_axis(g), g, window=filt)
    ref = astra[f"{name}_fbp_strip_{filt}"].astype(float)
    assert rmse(rec, ref, mask) < tol * rmse(ref, img, mask)
    # Escala (1/m): el valor medio del objeto coincide con ASTRA a 0.1 %
    assert rec[mask].mean() == pytest.approx(ref[mask].mean(), rel=1e-3)


@pytest.mark.parametrize("name,algo,sub,its,tol", [("sl", "sart", 1, 5, 0.2), ("sl", "sirt", 60, 10, 0.05),
                                                   ("leg", "sirt", 60, 10, 0.1)])
def test_sart_y_sirt_frente_a_astra(astra, name, algo, sub, its, tol):
    """Mismas iteraciones y datos: wavesim con ``subset_size`` 1 (SART) o igual al nº de vistas (SIRT)."""
    g, geom, mask, img = _case(astra, name)
    data = astra[f"{name}_fan_data"].astype(float)
    ref = astra[f"{name}_{algo}_strip_free_it{its}"].astype(float)
    rec = fb.os_sart(data, astra[f"{name}_a_fan"], geom, g, n_iter=its, subset_size=sub, nonneg=False)
    assert rmse(rec, ref, mask) < tol
    # y es una fracción pequeña del error frente al objeto
    assert rmse(rec, ref, mask) < 0.15 * rmse(ref, img, mask)


def test_espectros_frente_a_spekpy():
    """Casos fuera del ajuste (otros kVp, Al, Cu y ángulos de ánodo): HVL, energía media y error L1."""
    ref = np.load(DATA / "spekpy.npz")
    errs = []
    for (kvp, al, cu, th, fit, hvl, em), hist in zip(ref["meta"], ref["hist"]):
        n = int(kvp) - 10
        e_hvl, e_em, l1 = cx.spectrum_metrics(cx.wavesim_spectrum(kvp, al, cu, th), hist[:n].astype(float), hvl, em,
                                              kvp)
        assert abs(e_hvl) < 0.09 and abs(e_em) < 0.035 and l1 < 0.25, (kvp, al, cu, th)
        if not fit:
            errs.append(abs(e_hvl))
    assert len(errs) >= 20
    assert np.mean(errs) < 0.04  # medido 2.8 %


def test_dispersion_frente_a_xcist():
    """El modelo de convolución de XCIST (calibrado a un caso) da mucho menos SPR que el Monte Carlo de wavesim.

    Fija la diferencia documentada, no una concordancia: a 20 cm el cociente wavesim/XCIST es ~3 (medido 2.98 con
    2e7 fotones) y crece con el diámetro. Sirve de alarma si cambia el Monte Carlo o la referencia.
    """
    ref = np.load(DATA / "xcist.npz")
    from wavesim.em.xray import XraySpectrum

    spec = XraySpectrum(ref["energies_kev"], ref["weights"])
    x, spr = cx.wavesim_spr(200, spec, n_photons=2_000_000)
    ratio = cx.at(x, spr, 0.0) / float(ref["xcist_200_0"])
    assert 2.2 < ratio < 4.2
    # El perfil de wavesim cae hacia el borde del cilindro (menos camino), más que el de XCIST
    assert cx.at(x, spr, 80.0) / cx.at(x, spr, 0.0) < 0.5
    assert float(ref["xcist_200_80"]) / float(ref["xcist_200_0"]) > 0.6
