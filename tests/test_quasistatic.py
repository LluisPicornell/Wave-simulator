"""Validación del solver cuasi-estático (corrientes en tejido a baja frecuencia) y de la EIT."""

import numpy as np

from wavesim.core.grid import Grid2D
from wavesim.em import (
    QuasiStaticSolver, quasistatic, reconstruct_difference, ring_electrodes, sensitivity_matrix, simulate_eit,
    spectrum, surface_electrodes,
)
from wavesim.phantoms import RingFracture, cross_section_limb, longitudinal_limb
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

F = 50e3


def _disc(dx, radius=20e-3, inclusion=None):
    """Disco de músculo en agua de fondo (exterior aislante) con una inclusión circular opcional ``(x, z, r, tejido)``."""
    size = 2 * radius + 10e-3
    g = Grid2D.from_size(size, size, dx, x0=-size / 2, z0=-size / 2)
    X, Z = g.meshgrid()
    labels = np.full(g.shape, TISSUE_INDEX["water"], np.int16)
    labels[np.hypot(X, Z) <= radius] = TISSUE_INDEX["muscle"]
    if inclusion is not None:
        x, z, r, tissue = inclusion
        labels[np.hypot(X - x, Z - z) <= r] = TISSUE_INDEX[tissue]
    return Phantom(g, labels)


def test_plate_between_flat_electrodes_is_linear_and_obeys_ohm():
    dx = 0.5e-3
    g = Grid2D.from_size(40e-3, 20e-3, dx)
    plate = Phantom(g, np.full(g.shape, TISSUE_INDEX["muscle"], np.int16))
    solver = QuasiStaticSolver(plate, F)
    fixed = np.zeros(g.shape, bool)
    fixed[:, 0] = fixed[:, -1] = True
    volts = 1.0
    values = np.where(np.arange(g.nx) == 0, volts, 0.0)[None, :] * np.ones(g.shape)
    r = solver.solve(fixed=fixed, fixed_values=values)
    expected = volts * (1 - np.arange(g.nx) / (g.nx - 1))
    mid = r.phi[g.nz // 2]
    assert np.abs(mid - expected).max() < 0.02 * volts
    # corriente por metro de profundidad = sigma* · V · ancho / largo (entre centros de electrodo)
    current = solver.injected_current(r.phi, fixed & (np.arange(g.nx) == 0)[None, :])
    sigma = quasistatic.admittivity("muscle", F)
    ideal = sigma * volts * g.nz * dx / ((g.nx - 1) * dx)
    assert abs(current - ideal) / abs(ideal) < 0.02
    # la corriente que entra por un lado sale por el otro
    out = solver.injected_current(r.phi, fixed & (np.arange(g.nx) == g.nx - 1)[None, :])
    assert abs(current + out) / abs(current) < 1e-6


def test_disc_solution_converges_with_mesh_refinement():
    results = []
    for dx in (2.0e-3, 1.0e-3, 0.5e-3):
        d = _disc(dx)
        solver = QuasiStaticSolver(d, F)
        el = ring_electrodes(d, 4, 10e-3, start_deg=90.0)  # derecha e izquierda
        fixed = el.mask(0, d.grid.shape) | el.mask(2, d.grid.shape)
        values = np.where(el.mask(0, d.grid.shape), 0.5, -0.5)
        r = solver.solve(fixed=fixed, fixed_values=values)
        j, i = d.grid.index_of(5e-3, 3e-3)
        results.append((solver.injected_current(r.phi, el.mask(0, d.grid.shape)), r.phi[j, i]))
    cur = np.array([abs(c) for c, _ in results])
    pot = np.array([abs(p) for _, p in results])
    # Los electrodos de celdas de borde cambian de forma al refinar (escalones), así que el
    # criterio es que la solución se estabilice a pocos % y no que el error baje monótonamente
    for q in (cur, pot):
        assert abs(q[2] - q[1]) / q[2] < 0.02
        assert abs(q[1] - q[0]) / q[2] < 0.05


def test_reactive_part_matters_in_muscle_but_not_in_air():
    y_muscle = quasistatic.admittivity("muscle", F)
    y_air = quasistatic.admittivity("air", F)
    assert y_muscle.imag / y_muscle.real > 0.03  # corriente de desplazamiento apreciable en el músculo
    assert abs(y_air) < 1e-4 * abs(y_muscle)  # el aire no conduce frente al tejido: es un aislante
    # la parte reactiva cambia de verdad la solución: sin ella la tensión no tendría fase
    d = _disc(1e-3)
    el = ring_electrodes(d, 8, 6e-3)
    complex_run = simulate_eit(QuasiStaticSolver(d, F), el)
    real_map = d.property_map(lambda k: quasistatic.admittivity(k, F).real, dtype=complex)
    real_run = simulate_eit(QuasiStaticSolver(d, F, admittivity_map=real_map), el)
    assert np.abs(real_run.voltages.imag).max() < 1e-9 * np.abs(real_run.voltages).max()
    diff = np.abs(complex_run.voltages - real_run.voltages).max() / np.abs(real_run.voltages).max()
    assert diff > 0.005
    assert complex_run.residual < 1e-9


def test_fracture_changes_boundary_voltages_above_numerical_noise():
    dx = 0.8e-3
    healthy = cross_section_limb("pierna", None, dx=dx)
    broken = cross_section_limb("pierna", RingFracture(), dx=dx)
    assert broken.has_fracture and not healthy.has_fracture
    runs = {}
    for name, ph, tol in (("sana", healthy, 1e-10), ("rota", broken, 1e-10), ("sana_floja", healthy, 1e-7)):
        runs[name] = simulate_eit(QuasiStaticSolver(ph, F), ring_electrodes(ph, 8, 8e-3), tol=tol)
    v0 = runs["sana"].voltages
    change = np.abs(runs["rota"].voltages - v0).max() / np.abs(v0).max()
    noise = np.abs(runs["sana_floja"].voltages - v0).max() / np.abs(v0).max()  # error del solver a tol 1e-7
    assert change > 1e-3
    assert change > 1000 * noise


def test_sensitivity_matches_finite_difference_perturbation():
    d = _disc(1.0e-3, inclusion=(6e-3, 4e-3, 5e-3, "hematoma"))
    el = ring_electrodes(d, 8, 6e-3)
    base = simulate_eit(QuasiStaticSolver(d, F), el)
    sens = sensitivity_matrix(base, factor=5)
    y = d.property_map(lambda k: quasistatic.admittivity(k, F), dtype=complex)
    jc, ic = d.grid.index_of(6e-3, 4e-3)
    block = np.zeros(d.grid.shape, bool)
    j0, i0 = (jc // 5) * 5, (ic // 5) * 5
    block[j0:j0 + 5, i0:i0 + 5] = True
    eps = 1e-4
    pert = simulate_eit(QuasiStaticSolver(d, F, admittivity_map=np.where(block, y * (1 + eps), y)), el, tol=1e-12)
    k = int(sens.to_image(np.arange(sens.active.sum()), fill=-1)[j0 // 5, i0 // 5])
    predicted = sens.matrix[:, k] * eps
    measured = pert.voltages - base.voltages
    assert np.abs(predicted - measured).max() < 0.02 * np.abs(measured).max()


def test_difference_reconstruction_locates_conductive_inclusion():
    base = _disc(1.0e-3)
    inc = _disc(1.0e-3, inclusion=(8e-3, 5e-3, 4e-3, "blood"))
    el = ring_electrodes(base, 16, 4e-3)
    r0 = simulate_eit(QuasiStaticSolver(base, F), el)
    r1 = simulate_eit(QuasiStaticSolver(inc, F), el)
    sens = sensitivity_matrix(r0, factor=4)
    img = sens.to_image(reconstruct_difference(sens, r1.voltages - r0.voltages, reg=1e-2))
    jj, ii = np.unravel_index(np.argmax(img), img.shape)
    x = base.grid.x0 + (ii + 0.5) * sens.factor * base.grid.dx
    z = base.grid.z0 + (jj + 0.5) * sens.factor * base.grid.dx
    assert np.hypot(x - 8e-3, z - 5e-3) < 8e-3


def test_longitudinal_surface_electrodes_and_current_density():
    p = longitudinal_limb("tibia", None, width=40e-3, depth=22e-3, dx=0.5e-3).replace_tissue("water", "air")
    el = surface_electrodes(p, 8, 4e-3)
    assert len(el) == 8 and np.all(np.diff(el.centers[:, 0]) > 0)
    solver = QuasiStaticSolver(p, F)
    res = simulate_eit(solver, el)
    assert res.voltages.size == 8 * 5 and res.residual < 1e-9
    jmag = solver.current_density_magnitude(res.fields[0])
    assert np.isfinite(jmag).all() and jmag[~solver.tissue].max() == 0.0


def test_spectrum_low_frequency_band_uses_quasistatic_solver():
    assert spectrum.band_of(50e3).solver == "em.quasistatic"
