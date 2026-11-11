"""Validación del modelo completo de electrodo (CEM) y de la reconstrucción iterativa de EIT."""

import numpy as np
import pytest

from wavesim.em import (
    QuasiStaticSolver, Sensitivity, quasistatic, reconstruct_difference, ring_electrodes, sensitivity_matrix,
    simulate_eit,
)
from wavesim.em.eit import (
    EITModel, add_noise, disc_cem_series, disc_point_voltage, gauss_newton, greit_metrics, lcurve_regularization,
)

from test_quasistatic import _disc

F = 50e3
R = 20e-3
DX = 0.5e-3
WIDTH = 2e-3
CURRENT = 1e-3
SIGMA = quasistatic.admittivity("muscle", F)


@pytest.fixture(scope="module")
def disc():
    d = _disc(DX, R)
    el = ring_electrodes(d, 16, WIDTH)
    return d, el, QuasiStaticSolver(d, F)


def _angles(el):
    return np.array([np.arctan2(c[0], -c[1]) for c in el.centers])


def _series_frame0(el, z_c, n_modes=150):
    th = _angles(el)
    w = WIDTH / R
    arcs = [(t - w / 2, t + w / 2) for t in th]
    cur = np.zeros(16, complex)
    cur[0], cur[1] = CURRENT, -CURRENT
    return disc_cem_series(R, SIGMA, z_c, arcs, cur, n_modes=n_modes)[0]


def test_ideal_electrodes_match_point_electrode_solution_of_the_disc(disc):
    d, el, solver = disc
    ideal = simulate_eit(solver, el, current=CURRENT)
    assert ideal.contact is None  # el valor por defecto sigue siendo el electrodo ideal
    th = _angles(el)
    fr = ideal.frames == 0
    ref = np.array([disc_point_voltage(th[0], th[1], th[a], th[b], SIGMA, CURRENT) for a, b in ideal.meas[fr]])
    err = np.abs(ideal.voltages[fr] - ref) / np.abs(ref)
    assert err.max() < 0.02


def test_cem_matches_fourier_series_of_the_disc(disc):
    d, el, solver = disc
    for z_c in (0.01, 0.1):
        cem = simulate_eit(solver, el, current=CURRENT, contact_impedance=z_c)
        assert cem.residual < 1e-9
        u = _series_frame0(el, z_c)
        fr = cem.frames == 0
        ref = u[cem.meas[fr, 0]] - u[cem.meas[fr, 1]]
        err = np.abs(cem.voltages[fr] - ref) / np.abs(ref)
        assert err.max() < 0.02, (z_c, err.max())


def test_cem_series_with_point_like_limit_and_current_balance():
    # La serie respeta la corriente impuesta: con z_c grande la tensión de un electrodo
    # que inyecta I sobre un ancho w es del orden I z_c / w
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    w = 0.1
    cur = np.zeros(8, complex)
    cur[0], cur[4] = 1.0, -1.0
    u, _ = disc_cem_series(R, SIGMA, 10.0, [(t - w / 2, t + w / 2) for t in th], cur, n_modes=60)
    assert abs(u[0] - u[4]) == pytest.approx(2.0 * 10.0 / (w * R), rel=0.02)


def test_contact_drop_scales_with_current_times_zc_over_width(disc):
    d, el, solver = disc
    drops = []
    for z_c in (0.05, 0.1, 0.2):
        cem = simulate_eit(solver, el, current=CURRENT, contact_impedance=z_c)
        m = cem.contact.elec == 0
        drops.append((cem.electrode_potentials[0][0] - cem.fields[0][m].mean()).real)
    slope = np.diff(drops) / np.diff([0.05, 0.1, 0.2])
    # la pendiente de la caída frente a z_c es I / ancho; el resto es el medio camino celda-superficie
    assert np.allclose(slope, CURRENT / WIDTH, rtol=1e-3)
    # y la corriente que sale por los electrodos que no inyectan es nula (derivación sin fugas)
    k = 5
    m = cem.contact.elec == k
    leak = (cem.contact.g[m] * (cem.electrode_potentials[0][k] - cem.fields[0][m])).sum()
    assert abs(leak) < 1e-9 * CURRENT


def test_cem_tends_to_ideal_electrode_when_contact_vanishes(disc):
    d, el, solver = disc
    v = {z: simulate_eit(solver, el, current=CURRENT, contact_impedance=z).voltages for z in (0.0, 1e-6, 1e-2, 1.0)}
    scale = np.abs(v[0.0]).max()
    assert np.abs(v[1e-6] - v[0.0]).max() < 1e-3 * scale  # el límite existe
    assert np.abs(v[1e-2] - v[0.0]).max() < np.abs(v[1.0] - v[0.0]).max()  # y se aproxima de forma monótona
    th = _angles(el)
    fr = np.arange(v[0.0].size) // 13 == 0
    meas = simulate_eit(solver, el, current=CURRENT).meas[fr]
    ref = np.array([disc_point_voltage(th[0], th[1], th[a], th[b], SIGMA, CURRENT) for a, b in meas])
    assert np.abs(v[0.0][fr] - ref).max() / np.abs(ref).max() < 0.05


def test_cem_obeys_reciprocity_and_current_conservation(disc):
    d, el, solver = disc
    cem = simulate_eit(solver, el, current=CURRENT, contact_impedance=0.05)
    u = cem.electrode_potentials
    n = len(el)
    for p, q in ((0, 4), (2, 9), (5, 12)):
        a = u[p, q] - u[p, (q + 1) % n]
        b = u[q, p] - u[q, (p + 1) % n]
        assert abs(a - b) < 1e-8 * abs(a)
    assert np.abs(u.sum(axis=1)).max() < 1e-12  # referencia sum U = 0


def test_cem_jacobian_matches_finite_difference_including_contact_cells():
    d = _disc(1.0e-3, inclusion=(6e-3, 4e-3, 5e-3, "hematoma"))
    el = ring_electrodes(d, 8, 6e-3)
    z_c = 0.02
    base = simulate_eit(QuasiStaticSolver(d, F), el, contact_impedance=z_c)
    sens = sensitivity_matrix(base, factor=5)
    y = d.property_map(lambda k: quasistatic.admittivity(k, F), dtype=complex)
    for x, z in ((18e-3, 0.0), (0.0, -19e-3)):  # bloques que contienen celdas de electrodo
        jc, ic = d.grid.index_of(x, z)
        j0, i0 = (jc // 5) * 5, (ic // 5) * 5
        block = np.zeros(d.grid.shape, bool)
        block[j0:j0 + 5, i0:i0 + 5] = True
        assert (base.contact.elec[block] >= 0).any()
        eps = 1e-4
        pert = simulate_eit(QuasiStaticSolver(d, F, admittivity_map=np.where(block, y * (1 + eps), y)), el,
                            tol=1e-12, contact_impedance=z_c)
        k = int(sens.to_image(np.arange(sens.active.sum()), fill=-1)[j0 // 5, i0 // 5])
        measured = pert.voltages - base.voltages
        assert np.abs(sens.matrix[:, k] * eps - measured).max() < 0.01 * np.abs(measured).max()


# --------------------------------------------------------------------------
# Reconstrucción
# --------------------------------------------------------------------------
INC = (8e-3, 5e-3, 4e-3, "blood")


@pytest.fixture(scope="module")
def inverse_problem():
    """Datos en malla de 0.5 mm con CEM y ruido de 70 dB; modelo de reconstrucción en 1 mm (sin crimen inverso)."""
    z_c = 0.02
    f0, f1 = _disc(0.5e-3, R), _disc(0.5e-3, R, inclusion=INC)
    elf = ring_electrodes(f0, 16, 3e-3)
    v0 = simulate_eit(QuasiStaticSolver(f0, F), elf, CURRENT, contact_impedance=z_c).voltages
    v1 = simulate_eit(QuasiStaticSolver(f1, F), elf, CURRENT, contact_impedance=z_c).voltages
    rng = np.random.default_rng(1)
    v0n, v1n = add_noise(v0, 70.0, rng), add_noise(v1, 70.0, rng)
    rec = _disc(1e-3, R)
    model = EITModel(rec, F, ring_electrodes(rec, 16, 3e-3), factor=3, current=CURRENT, contact_impedance=z_c)
    x, z = rec.grid.meshgrid()
    contrast = quasistatic.admittivity("blood", F).real / quasistatic.admittivity("muscle", F).real - 1.0
    truth = np.where(np.hypot(x - INC[0], z - INC[1]) <= INC[2], contrast, 0.0)
    return model, v0n, v1n, truth


def test_gauss_newton_beats_one_step_in_position_and_amplitude(inverse_problem):
    model, v0, v1, truth = inverse_problem
    g = model.phantom.grid
    _, jac = model.forward(np.zeros(model.n_params), jacobian=True)
    sens = Sensitivity(jac, model._coarse_shape, model.factor, model.active)
    one = np.where(model.tissue, model.to_fine(reconstruct_difference(sens, v1 - v0, 1e-2)), 0.0)
    gn = gauss_newton(model, v1, "noser", 3e-2, n_iter=8, data_ref=v0)
    m1 = greit_metrics(one, truth, model.tissue, g)
    m2 = greit_metrics(gn.image(), truth, model.tissue, g)
    assert gn.cost[-1] < 0.2 * gn.cost[0]  # el ajuste mejora de verdad
    assert m2.position_error < 2e-3
    assert abs(m2.amplitude - 1.0) < abs(m1.amplitude - 1.0)  # el paso lineal subestima el contraste
    assert abs(m2.amplitude - 1.0) < 0.3


@pytest.mark.parametrize("reg", ["tikhonov", "laplaciano", "tv"])
def test_other_regularizers_also_locate_the_inclusion(inverse_problem, reg):
    model, v0, v1, truth = inverse_problem
    gn = gauss_newton(model, v1, reg, 1e-2, n_iter=4, data_ref=v0)
    m = greit_metrics(gn.image(), truth, model.tissue, model.phantom.grid)
    assert m.position_error < 4e-3 and 0.4 < m.amplitude < 1.6


def test_lcurve_picks_interior_corner_and_greit_metrics_are_exact_on_perfect_image(inverse_problem):
    model, v0, v1, truth = inverse_problem
    _, jac = model.forward(np.zeros(model.n_params), jacobian=True)
    jr = np.vstack([jac.real, jac.imag])
    r = np.concatenate([(v1 - v0).real, (v1 - v0).imag])
    lam, curve = lcurve_regularization(jr, r, np.diag(np.diag(jr.T @ jr)))
    assert curve["lams"][0] < lam < curve["lams"][-1]
    m = greit_metrics(truth, truth, model.tissue, model.phantom.grid)
    assert m.position_error < 1e-9 and m.amplitude == pytest.approx(1.0) and m.ringing == 0.0
    assert m.shape_deformation < 0.15  # un disco rasterizado cabe casi en el círculo de igual área
