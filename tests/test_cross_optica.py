"""Comparación cruzada de la luz en tejido: wavesim frente a MCML (y MCX) sin tener instalada ninguna de las dos.

Las referencias están en ``benchmarks/data/cross_optica/*.npz`` (réplicas independientes de MCML 1.2 de Wang y
Jacques, y de MCXCL/pocl; ver ``benchmarks/cross_optica.py`` para regenerarlas). Aquí se vuelve a ejecutar el Monte
Carlo de wavesim con pocos fotones (réplicas pequeñas, para tener su error estándar) y la difusión, y se compara
con tolerancias de la forma ``max(k·sem, suelo)``. La tabla completa con 1e6 fotones por réplica está en
``docs/validacion/cruzada_optica.md``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cross_optica", RAIZ / "benchmarks" / "cross_optica.py")
cross = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cross)

N_FOT, N_REPL = 20_000, 4  # 8e4 fotones por caso: ~1 % de ruido en los intervalos con >1 % de la energía


def _ref(nombre, herramienta="mcml"):
    return np.load(cross.DATA / f"{herramienta}_{nombre}.npz")


@pytest.fixture(scope="module")
def ws_c():
    layers, n, _ = cross.CASES["c_slab"]
    return cross.ejecutar_wavesim(layers, n, N_FOT, N_REPL)


@pytest.fixture(scope="module")
def ws_b():
    layers, n, _ = cross.CASES["b_slab"]
    return cross.ejecutar_wavesim(layers, n, N_FOT, N_REPL)


@pytest.fixture(scope="module")
def ws_a():
    layers, n, _ = cross.CASES["a_slab"]
    return cross.ejecutar_wavesim(layers, n, N_FOT, N_REPL)


def test_conversion_radial_a_linea_disco():
    """Un disco uniforme de radio 1 tiene por cuerdas ``R_2D(x) = 2 sqrt(1 - x²)``; la integral total es el área."""
    dr, nr, dx = 0.02, 150, 0.05
    ring = (np.arange(nr) + 0.5) * dr < 1.0 - 1e-9
    # los anillos hasta r = 1 (50 de 0.02); el último anillo entra entero: el disco es exacto
    x = np.arange(0, 41) * dx
    r2d = cross.radial_a_linea(ring.astype(float), dr, x, dx)
    assert (r2d[0] + 2 * r2d[1:].sum()) * dx == pytest.approx(np.pi, rel=1e-12)
    # valor medio de la cuerda sobre la columna [x - dx/2, x + dx/2], por cuadratura fina
    for i in (4, 10, 17):
        t = np.linspace(x[i] - dx / 2, x[i] + dx / 2, 20001)
        exacto = np.mean(2 * np.sqrt(np.maximum(1 - t**2, 0.0)))
        assert r2d[i] == pytest.approx(exacto, rel=1e-5)


def test_referencias_balance():
    """Las referencias son coherentes: Rsp + Rd + A + Tt = 1 en cada réplica y caso (MCML imprime 6 cifras)."""
    for nombre in cross.CASES:
        for herr in ("mcml", "mcx"):
            if not (cross.DATA / f"{herr}_{nombre}.npz").exists():
                continue
            tot = _ref(nombre, herr)["tot"]
            assert np.abs(tot.sum(1) - 1.0).max() < (2e-5 if herr == "mcml" else 2e-4), (nombre, herr, tot.sum(1))


def _comparar_totales(ref, ws, tol_suelo=1.5e-3):
    mr, sr = cross.mean_sem(ref["tot"])
    mw, sw = cross.mean_sem(ws["tot"])
    for i, nombre in enumerate(cross.TOT_NAMES):
        tol = max(4.0 * np.hypot(sr[i], sw[i]), tol_suelo if nombre != "specular" else 1e-6)
        assert abs(mw[i] - mr[i]) <= tol, f"{nombre}: wavesim {mw[i]:.5f} MCML {mr[i]:.5f} tol {tol:.5f}"


def _comparar_perfil(ref, ws, eje_mm, bordes, rel=0.08, min_frac=1e-2):
    """Error relativo por intervalo en la zona con señal: la integral de MCML es >= ``min_frac`` de su máximo."""
    br = cross.integral_bins(ref, eje_mm, bordes, cross.DX).mean(0)
    bw = cross.integral_bins(ws, eje_mm, bordes, cross.DX).mean(0)
    fiable = br >= min_frac * br.max()
    assert fiable.sum() >= 4
    err = np.abs(bw[fiable] / br[fiable] - 1.0)
    assert err.max() < rel, f"errores relativos {np.round(err, 3)} en {np.array(bordes)[:-1][fiable]}"


X_MM = np.arange(cross.HALF_NX + 1) * cross.DX * 1e3


def test_losa_indice_desajustado_totales(ws_c):
    """(c) n = 1.4 frente a aire: especular exacta y Rd, absorbida y Tt dentro del error estadístico de MCML."""
    _comparar_totales(_ref("c_slab"), ws_c)
    assert ws_c["tot"][:, 0].mean() == pytest.approx(((1.4 - 1) / (1.4 + 1)) ** 2, rel=1e-12)
    assert ws_c["lateral"].mean() < 1e-5  # la losa de 7 cm de ancho es "infinita" a efectos de MCML


def test_losa_indice_desajustado_perfiles(ws_c):
    ref = _ref("c_slab")
    _comparar_perfil(ref["R2d"], ws_c["R2d"], X_MM, cross.X_BINS_MM)
    _comparar_perfil(ref["T2d"], ws_c["T2d"], X_MM, cross.X_BINS_MM)
    z_mm = (np.arange(ref["phi_int"].shape[1]) + 0.5) * cross.DX * 1e3
    _comparar_perfil(ref["phi_int"], ws_c["phi_int"], z_mm, (0.0, 1.0, 2.0, 4.0, 7.0, 10.05), rel=0.03, min_frac=0.0)


def test_dos_capas_grasa_musculo(ws_b):
    """(b) grasa 3 mm sobre músculo 7 mm: totales, R(x), T(x) y fluencia en profundidad (cruza la interfaz)."""
    ref = _ref("b_slab")
    _comparar_totales(ref, ws_b)
    _comparar_perfil(ref["R2d"], ws_b["R2d"], X_MM, cross.X_BINS_MM)
    _comparar_perfil(ref["T2d"], ws_b["T2d"], X_MM, cross.X_BINS_MM)
    z_mm = (np.arange(ref["phi_int"].shape[1]) + 0.5) * cross.DX * 1e3
    _comparar_perfil(ref["phi_int"], ws_b["phi_int"], z_mm, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.05), rel=0.03,
                     min_frac=0.0)


def test_losa_indice_ajustado_y_difusion(ws_a):
    """(a) n = 1: el Monte Carlo reproduce a MCML; la difusión queda a ~5 % en los totales (peor cerca de la fuente)."""
    ref = _ref("a_slab")
    _comparar_totales(ref, ws_a)
    _comparar_perfil(ref["R2d"], ws_a["R2d"], X_MM, cross.X_BINS_MM)
    layers, n, _ = cross.CASES["a_slab"]
    dif = cross.ejecutar_difusion(layers, n)
    mr = ref["tot"].mean(0)
    # totales de la difusión: ~4 % en R y en absorbida para una losa de 1 cm con dx = 0.1 mm (medido; ver docs)
    assert dif["tot"][1] == pytest.approx(mr[1], rel=0.06)
    assert dif["tot"][2] == pytest.approx(mr[2], rel=0.06)
    assert dif["tot"][:].sum() == pytest.approx(1.0, abs=0.01)  # la difusión conserva la energía (CG a 1e-13)
    # lejos de la fuente (5-12 mm) la difusión es una buena aproximación de R(x)
    lejos = (5.0, 12.0)
    br = cross.integral_bins(ref["R2d"], X_MM, lejos, cross.DX).mean(0)[0]
    bd = cross.integral_bins(dif["R2d"][None], X_MM, lejos, cross.DX)[0, 0]
    assert bd == pytest.approx(br, rel=0.08)
    # bajo la fuente (< 0.5 mm) la difusión subestima R en ~20 % (medido: -21 %): la brecha que motiva el Monte Carlo
    cerca = (0.0, 0.5)
    br = cross.integral_bins(ref["R2d"], X_MM, cerca, cross.DX).mean(0)[0]
    bd = cross.integral_bins(dif["R2d"][None], X_MM, cerca, cross.DX)[0, 0]
    assert bd / br - 1.0 < -0.10


@pytest.mark.parametrize("nombre", ["a_slab", "b_slab", "c_slab", "a_semi", "b_semi", "c_semi"])
def test_mcx_concuerda_con_mcml(nombre):
    """Las dos referencias entre sí (sólo datos guardados): totales dentro de 5 σ combinados o 1.5e-3."""
    if not (cross.DATA / f"mcx_{nombre}.npz").exists():
        pytest.skip("sin referencia MCX guardada")
    mr, sr = cross.mean_sem(_ref(nombre)["tot"])
    mx, sx = cross.mean_sem(_ref(nombre, "mcx")["tot"])
    for i, k in enumerate(cross.TOT_NAMES):
        assert abs(mx[i] - mr[i]) <= max(5 * np.hypot(sr[i], sx[i]), 1.5e-3), (nombre, k, mx[i], mr[i])


def test_difusion_balance_con_perfil_reflejado_y_n_tissue():
    """``reflected_profile`` y ``n_tissue`` (extensiones de ``solve_diffusion`` para esta comparación).

    Balance: reflejada + transmitida + absorbida = fuente (1 W/m); con ``n_tissue`` = fondo no hay reflexión
    interna (A = 1) y sale más luz que con el 1.4 por defecto, que no cambia.
    """
    from wavesim.core.grid import Grid2D
    from wavesim.em import optics
    from wavesim.phantoms import TISSUE_INDEX, Phantom
    dx, nx, nz = 5e-4, 61, 20
    lab = np.full((nz + 2, nx + 2), TISSUE_INDEX["air"], dtype=int)
    lab[1:-1, 1:-1] = TISSUE_INDEX["muscle"]
    ph = Phantom(Grid2D(nx + 2, nz + 2, dx, x0=-(nx // 2 + 1) * dx, z0=-dx / 2), lab)
    src = (0.0, 1e-3)
    d14 = optics.solve_diffusion(ph, 800.0, source=src)
    assert np.allclose(d14.fluence, optics.solve_diffusion(ph, 800.0, source=src, n_tissue=optics.N_TISSUE).fluence)
    d10 = optics.solve_diffusion(ph, 800.0, source=src, n_tissue=1.0)
    for d, tol in ((d14, 1e-6), (d10, 1e-6)):
        sel = slice(1, -1)
        absorbida = float((d.mu_a[1:-1, sel] * d.fluence[1:-1, sel]).sum()) * dx * dx
        sale = (np.nansum(d.reflected_profile[sel]) + np.nansum(d.transmitted_profile[sel])) * dx
        assert absorbida + sale == pytest.approx(1.0, abs=tol + 5e-3)  # la exitancia usa un gradiente de 1 celda
    assert d10.reflected_profile[1:-1].sum() > d14.reflected_profile[1:-1].sum()
