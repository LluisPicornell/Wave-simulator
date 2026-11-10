"""Validación cuantitativa del solver elástico frente a soluciones analíticas (versiones rápidas).

Los casos completos (U3, U4, U5 de ``docs/comparativa_simuladores.md`` §11.1) están en
``benchmarks/elastic_validation.py`` y sus resultados en ``docs/validacion/elastico.md``.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.core.signals import gaussian_pulse
from wavesim.phantoms.phantom import Phantom
from wavesim.ultrasound import ElasticMedium, ElasticSolver, Source
from wavesim.ultrasound import analytic as an

F0 = 1e6
WATER = (1000.0, 1500.0, 0.0)
BONE = (1908.0, 3500.0, 1800.0)

# Los montajes de medida son los del benchmark (con parámetros reducidos), para no duplicarlos
_spec = importlib.util.spec_from_file_location(
    "elastic_validation", Path(__file__).resolve().parents[1] / "benchmarks" / "elastic_validation.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _solver(rho, cp, cs, dx, cfl=0.9, backend="auto", dtype=np.float32, npml=20):
    z = np.zeros_like(rho)
    m = ElasticMedium(rho=rho, cp=cp, cs=cs, tau_p=z, tau_s=z.copy(), tau_sigma=np.zeros(0), f0=F0)
    g = Grid2D(rho.shape[1], rho.shape[0], dx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElasticSolver(Phantom(g, np.zeros(g.shape, np.int16)), F0, npml=npml, cfl=cfl, medium=m,
                             backend=backend, dtype=dtype)


# ----------------------------------------------------------------------------
# Soluciones de referencia
# ----------------------------------------------------------------------------
def test_coeficientes_fluido_solido_conservan_energia_y_coinciden_con_la_forma_cerrada():
    th = np.radians(np.linspace(0, 85, 171))
    c = an.fluid_solid_coefficients(th, *WATER[:2], *BONE)
    closed = an.fluid_solid_reflection_closed(th, *WATER[:2], *BONE)
    assert np.allclose(c.R, closed, atol=1e-12)
    assert np.allclose(np.abs(c.R) ** 2 + c.energy_p + c.energy_s, 1.0, atol=1e-12)
    # incidencia normal: (Z2 - Z1) / (Z2 + Z1); pasado el crítico S, reflexión total
    z1, z2 = 1000 * 1500, 1908 * 3500
    assert c.R[0].real == pytest.approx((z2 - z1) / (z2 + z1), rel=1e-12)
    th_p, th_s = an.critical_angles(1500, 3500, 1800)
    assert np.degrees(th_p) == pytest.approx(25.377, abs=1e-3) and np.degrees(th_s) == pytest.approx(56.443, abs=1e-3)
    assert np.allclose(np.abs(c.R[th > th_s]), 1.0, atol=1e-12)


def test_ondas_de_lamb_libres_y_con_agua_tienden_a_sus_limites():
    cp, cs, rho, d = 3500.0, 1800.0, 1908.0, 3e-3
    c_plate = np.sqrt(4 * cs**2 * (1 - cs**2 / cp**2))
    f = np.array([20e3, 4e6])
    w = 2 * np.pi * f
    s0 = w / an.lamb_fundamental(f, d, cp, cs, "S").real
    a0 = w / an.lamb_fundamental(f, d, cp, cs, "A").real
    c_r = an.rayleigh_velocity(cp, cs)
    assert s0[0] == pytest.approx(c_plate, rel=1e-3)  # S0: velocidad de placa a baja frecuencia
    assert s0[1] == pytest.approx(c_r, rel=1e-3) and a0[1] == pytest.approx(c_r, rel=1e-3)
    # con agua a ambos lados, A0 (ligado, sin fuga) tiende a la onda de Scholte agua-hueso
    fr = np.linspace(50e3, 4e6, 80)
    kw = an.lamb_fundamental(fr, d, cp, cs, "A", rho, 1000.0, 1500.0)
    x = np.linspace(500, 1499.99, 400001)
    q = (x / cs) ** 2
    scholte = (2 - q) ** 2 - 4 * np.sqrt(1 - x**2 / cp**2) * np.sqrt(1 - q) \
        + (1000 / rho) * q**2 * np.sqrt(1 - x**2 / cp**2) / np.sqrt(1 - x**2 / 1500**2)
    c_sch = x[np.argmin(np.abs(scholte))]
    assert 2 * np.pi * fr[-1] / kw[-1].real == pytest.approx(c_sch, rel=1e-3)
    assert abs(kw[-1].imag) < 1e-6 * kw[-1].real


# ----------------------------------------------------------------------------
# Interfaz fluido-sólido a incidencia oblicua (corrección ``_slip_terms``)
# ----------------------------------------------------------------------------
def _beam_reflection(theta_deg, ppw, sigma_lam=3.0, backend="auto"):
    """|R(θ)| medido con un haz gaussiano inclinado (descomposición en ondas planas a lo largo de una línea).

    Dos ejecuciones con el mismo dt (con el semiespacio de cortical y sólo con
    agua): la reflejada es la diferencia, y |R| el cociente de sus DFT en x y t
    en el número de onda horizontal ``kx = w sinθ / c`` (exacto si los dos
    campos se recogen enteros en la línea).
    """
    lam = 1500.0 / F0
    dx = lam / ppw
    th = np.radians(theta_deg)
    npml = 20
    sig = sigma_lam * lam
    j_s = npml + int(round(lam / dx))
    j_r = j_s + int(round(lam / dx))
    j_int = j_r + int(round(lam / dx))
    nz = j_int + int(round(3 * lam / dx)) + npml
    x_c = 3.5 * sig
    nx = int(np.ceil((x_c + 3 * lam * np.tan(th) + 3.5 * sig + 2 * lam) / dx)) + 2 * npml
    x = (np.arange(nx) - npml) * dx
    src = np.nonzero(np.abs(x - x_c) <= 3 * sig)[0]
    delay = (x[src] - x_c) * np.sin(th) / 1500.0
    delay -= delay.min()
    amp = np.exp(-0.5 * ((x[src] - x_c) / sig) ** 2)
    acc = {}
    for name, solid in (("ref", False), ("int", True)):
        rho, cp, cs = np.full((nz, nx), 1000.0), np.full((nz, nx), 1500.0), np.zeros((nz, nx))
        if solid:
            rho[j_int:], cp[j_int:], cs[j_int:] = BONE
        s = _solver(rho, cp, cs, dx, 0.9 if solid else 0.9 * 1500 / 3500, backend)
        nt = int(np.ceil((delay.max() + 3 * lam / (1500 * np.cos(th)) + 10e-6) / s.dt))
        t = s.time_axis(nt)
        s.add_source(Source(np.full(src.size, j_s), src, np.arange(src.size),
                            amp[:, None] * gaussian_pulse(t[None, :] - delay[:, None], F0)))
        a = np.zeros(nx, complex)
        for n in range(nt):
            s.step(n)
            a -= 0.5 * np.exp(2j * np.pi * F0 * t[n]) * (s.sxx[j_r] + s.szz[j_r])
        acc[name] = a
    thetas = np.radians(theta_deg + np.linspace(-1.5, 1.5, 7))
    E = np.exp(-1j * (2 * np.pi * F0 * np.sin(thetas) / 1500.0)[:, None] * x[None, :])
    R = np.abs(E @ (acc["int"] - acc["ref"])) / np.abs(E @ acc["ref"])
    R_an = np.abs(an.fluid_solid_coefficients(thetas, *WATER[:2], *BONE).R)
    return R / R_an - 1


def test_reflexion_oblicua_agua_cortical_a_45_grados():
    # Sin la corrección de ``_slip_terms`` el error era 3.6 % a 12 ppw (y no bajaba de ~1 % al refinar)
    err = _beam_reflection(45.0, ppw=12)
    assert np.abs(err).max() < 0.015


def test_correccion_de_interfaz_conserva_la_energia():
    """Las derivadas de cizalla corregidas siguen siendo adjuntas: <sxz, D_f v> = -<v, D_b sxz>.

    Es la propiedad que hace estable el esquema escalonado; se comprueba con
    campos aleatorios (nulos cerca de los bordes) en una inclusión de hueso con
    contorno escalonado, que activa las reglas en x y en z.
    """
    n = 64
    yy, xx = np.mgrid[:n, :n]
    bone = (yy - 31.3) ** 2 + (xx - 32.6) ** 2 < 15.5**2
    rho, cp, cs = (np.where(bone, b, w) for b, w in zip(BONE, WATER))
    s = _solver(rho, cp, cs, 1e-4, backend="numpy", dtype=np.float64, npml=4)
    rng = np.random.default_rng(3)
    win = np.zeros((n, n))
    win[10:-10, 10:-10] = 1.0
    v, sxz = rng.standard_normal((2, n, n)) * win
    for strain, momentum, axis, fwd in (("exz_a", "sxz_z", 0, True), ("exz_b", "sxz_x", 1, True)):
        a = np.sum(sxz * s._deriv(strain, v, axis, fwd))
        b = np.sum(v * s._deriv(momentum, sxz, axis, not fwd))
        assert abs(a + b) < 1e-12 * abs(a)
    assert {"exz_a", "exz_b", "sxz_z", "sxz_x"} <= set(s._slip)


# ----------------------------------------------------------------------------
# U3: convergencia
# ----------------------------------------------------------------------------
def test_onda_plana_sigue_la_dispersion_del_esquema_y_converge_a_segundo_orden():
    rows = [ev.plane_wave_1d(ev.WATER, "pressure", p) for p in (8, 16, 32)]
    for r in rows:
        # la velocidad medida es la que predice la relación de dispersión discreta (no hay error de implementación)
        assert r["err"] == pytest.approx(r["pred"], abs=2e-6)
        assert r["amp"] == pytest.approx(1.0, abs=1e-4)  # leapfrog sin pérdidas
    # con CFL fija domina el término temporal, de 2.º orden (asintóticamente)
    assert ev.slopes([r["dx"] for r in rows[1:]], [r["err"] for r in rows[1:]])[0] > 1.85
    assert ev.slopes([r["dx"] for r in rows[1:]], [r["l2"] for r in rows[1:]])[0] > 1.8
    assert abs(rows[-1]["err"]) < 6e-4


def test_error_espacial_es_de_cuarto_orden_con_dt_fijo():
    dt = ev.plane_wave_1d(ev.WATER, "pressure", 12, cfl=0.1)["dt"]
    rows = [ev.plane_wave_1d(ev.WATER, "pressure", p, dt=dt) for p in (5, 8)]
    assert ev.slopes([r["dx"] for r in rows], [r["err"] for r in rows])[0] > 3.6


def test_onda_s_en_cortical_sigue_la_dispersion_del_esquema():
    r = ev.plane_wave_1d(ev.BONE, "force_x", 12)
    assert r["err"] == pytest.approx(r["pred"], abs=2e-6) and abs(r["err"]) < 1e-3


def test_reflexion_normal_agua_cortical_converge_a_segundo_orden():
    rows = [ev.layers_1d(ev.WATER, ev.BONE, p) for p in (8, 16)]
    dxs = [r["dx"] for r in rows]
    assert ev.slopes(dxs, [r["eR"] for r in rows])[0] > 1.8
    assert ev.slopes(dxs, [r["eT"] for r in rows])[0] > 1.8
    assert abs(rows[-1]["eR"]) < 3e-3 and abs(rows[-1]["eT"]) < 3e-3
    # la interfaz efectiva está en la fila de nodos vz (media celda entre las filas de tensión)
    assert abs(rows[-1]["delta"]) < 0.01


# ----------------------------------------------------------------------------
# U4: R(θ) agua-cortical
# ----------------------------------------------------------------------------
def test_angulo_critico_p_y_reflexion_total_pasado_el_critico_s():
    th_p, th_s = (np.degrees(a) for a in an.critical_angles(1500.0, 3500.0, 1800.0))
    r = ev.beam_reflection(th_p, ppw=10, dtheta=3.0, n_theta=61, freqs=(1e6,))
    assert ev.critical_angle_from_peak(r)[0] == pytest.approx(th_p, abs=1.0)
    # pasado el crítico S la reflexión es total y su fase, compleja, sigue a la exacta
    r = ev.beam_reflection(62.0, ppw=10, dtheta=2.0, n_theta=9, freqs=(1e6,))
    a = ev.r_theta_analytic(r["theta"])
    assert np.abs(np.abs(r["R"]) - 1).max() < 0.03
    assert np.degrees(np.abs(np.angle(r["R"] / a.R))).max() < 8.0  # ~6° a 10 ppw, ~3° a 15


def test_onda_s_transmitida_y_numba_igual_a_numpy():
    ra, rb = (ev.beam_reflection(40.0, ppw=8, dtheta=2.0, n_theta=9, freqs=(1e6,), backend=b)
              for b in ("auto", "numpy"))
    assert np.abs(ra["R"] - rb["R"]).max() < 1e-4 * np.abs(rb["R"]).max()
    a = ev.r_theta_analytic(ra["theta"])
    assert np.abs(ra["vs"] / a.vs - 1).max() < 0.02  # conversión P -> S en la interfaz
    assert np.abs(np.abs(ra["R"]) / np.abs(a.R) - 1).max() < 0.02


# ----------------------------------------------------------------------------
# U5: ondas de Lamb en una placa de cortical de 3 mm
# ----------------------------------------------------------------------------
def test_lamb_placa_en_agua_sigue_la_dispersion_con_carga_de_fluido():
    freqs = np.array([150e3, 200e3, 250e3, 300e3])
    r = ev.lamb_case(ev.WATER, dx=0.25e-3, freqs=freqs, length=100e-3, t_end=150e-6)
    assert np.abs(r["S"]["c"] / r["S"]["c_ref"] - 1).max() < 0.01
    assert np.abs(r["A"]["c"] / r["A"]["c_ref"] - 1).max() < 0.02
    # el agua frena A0 un ~15 %: la simulación sigue la curva con carga de fluido, no la de la placa libre
    assert np.all(r["A"]["c_free"] / r["A"]["c_ref"] > 1.12)
    assert r["sim"]["finite"] and r["sim"]["final"] < r["sim"]["peak"]


def test_lamb_placa_libre_en_aire_es_estable_y_sigue_rayleigh_lamb():
    freqs = np.array([150e3, 200e3, 250e3, 300e3])
    r = ev.lamb_case(ev.AIR, dx=0.25e-3, freqs=freqs, length=100e-3, t_end=150e-6)
    assert r["sim"]["finite"] and r["sim"]["final"] < r["sim"]["peak"]
    assert np.abs(r["S"]["c"] / r["S"]["c_ref"] - 1).max() < 0.03  # 12 celdas en el espesor
    assert np.abs(r["A"]["c"] / r["A"]["c_ref"] - 1).max() < 0.03
