"""Validación de la EIT: modelo completo de electrodo (CEM), reconstrucción Gauss-Newton y caso de la pierna.

Imprime las tablas de ``docs/validacion/eit.md``.

Uso (desde la raíz del repositorio, con ``NUMBA_NUM_THREADS=1`` si hay otros procesos):

    python benchmarks/eit_validation.py analytic   # disco homogéneo frente a las soluciones analíticas  (~3 min)
    python benchmarks/eit_validation.py disc       # inclusión en un disco: un paso frente a Gauss-Newton  (~10 min)
    python benchmarks/eit_validation.py leg        # pierna con fractura, ideal frente a CEM            (~30 min)
    python benchmarks/eit_validation.py            # las tres
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wavesim.core.grid import Grid2D  # noqa: E402
from wavesim.em import (  # noqa: E402
    QuasiStaticSolver, Sensitivity, quasistatic, reconstruct_difference, ring_electrodes, simulate_eit,
)
from wavesim.em.eit import (  # noqa: E402
    EITModel, add_noise, coarsen_phantom, disc_cem_series, disc_point_voltage, gauss_newton, greit_metrics,
)
from wavesim.phantoms import RingFracture, cross_section_limb  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402

F = 50e3
R = 20e-3
CURRENT = 1e-3
SIGMA = quasistatic.admittivity("muscle", F)


def disc(dx, inclusion=None, radius=R):
    size = 2 * radius + 10e-3
    g = Grid2D.from_size(size, size, dx, x0=-size / 2, z0=-size / 2)
    x, z = g.meshgrid()
    labels = np.full(g.shape, TISSUE_INDEX["water"], np.int16)
    labels[np.hypot(x, z) <= radius] = TISSUE_INDEX["muscle"]
    if inclusion is not None:
        cx, cz, r, tissue = inclusion
        labels[np.hypot(x - cx, z - cz) <= r] = TISSUE_INDEX[tissue]
    return Phantom(g, labels)


def angles(el):
    return np.array([np.arctan2(c[0], -c[1]) for c in el.centers])


# --------------------------------------------------------------------------
def analytic():
    print("## A. Disco homogéneo frente a soluciones analíticas (R = 20 mm, músculo a 50 kHz, trama 0, 13 medidas)\n")
    print("Electrodos ideales frente a electrodos puntuales (fórmula cerrada):\n")
    print("| dx (mm) | ancho electrodo (mm) | error máx. por dato (%) | error de la norma (%) |")
    print("|---|---|---|---|")
    cases = ((2e-3, 2e-3), (1e-3, 2e-3), (0.5e-3, 2e-3), (0.5e-3, 1e-3))
    for dx, wd in cases:
        d = disc(dx)
        el = ring_electrodes(d, 16, wd)
        res = simulate_eit(QuasiStaticSolver(d, F), el, CURRENT)
        th = angles(el)
        fr = res.frames == 0
        ref = np.array([disc_point_voltage(th[0], th[1], th[a], th[b], SIGMA, CURRENT) for a, b in res.meas[fr]])
        e = np.abs(res.voltages[fr] - ref)
        print(f"| {dx * 1e3:.2f} | {wd * 1e3:.0f} | {np.max(e / np.abs(ref)) * 100:.2f} | "
              f"{np.linalg.norm(e) / np.linalg.norm(ref) * 100:.2f} |", flush=True)

    print("\nCEM (diferencias finitas) frente a la serie de Fourier del CEM, dx = 0.5 mm, electrodos de 2 mm:\n")
    d = disc(0.5e-3)
    el = ring_electrodes(d, 16, 2e-3)
    solver = QuasiStaticSolver(d, F)
    th = angles(el)
    arcs = [(t - 1e-3 / R, t + 1e-3 / R) for t in th]
    cur = np.zeros(16, complex)
    cur[0], cur[1] = CURRENT, -CURRENT
    print("| z_c (ohm·m²) | error máx. por dato (%) | error de la norma (%) | caída de contacto / (I z_c / w) |")
    print("|---|---|---|---|")
    for z_c in (0.003, 0.03, 0.3):
        cem = simulate_eit(solver, el, CURRENT, contact_impedance=z_c)
        u = disc_cem_series(R, SIGMA, z_c, arcs, cur, n_modes=200)[0]
        fr = cem.frames == 0
        ref = u[cem.meas[fr, 0]] - u[cem.meas[fr, 1]]
        e = np.abs(cem.voltages[fr] - ref)
        m = cem.contact.elec == 0
        drop = (cem.electrode_potentials[0][0] - cem.fields[0][m].mean()).real
        print(f"| {z_c} | {np.max(e / np.abs(ref)) * 100:.2f} | {np.linalg.norm(e) / np.linalg.norm(ref) * 100:.2f} | "
              f"{drop / (CURRENT * z_c / 2e-3):.3f} |", flush=True)

    print("\nConvergencia de la propia serie con el número de modos (z_c = 0.03; U_0 - U_1 en mV):\n")
    print("| modos | 50 | 100 | 200 | 400 | 800 |")
    print("|---|---|---|---|---|---|")
    row = []
    for nm in (50, 100, 200, 400, 800):
        u = disc_cem_series(R, SIGMA, 0.03, arcs, cur, n_modes=nm)[0]
        row.append(f"{abs(u[0] - u[1]) * 1e3:.4f}")
    print("| |" + "|".join(row) + "|", flush=True)

    print("\nLímite z_c -> 0 (dx = 0.5 mm): máx |V(z_c) - V(0)| / máx |V(0)|\n")
    print("| z_c | 1e-6 | 1e-3 | 1e-2 | 1e-1 | 1 |")
    print("|---|---|---|---|---|---|")
    v0 = simulate_eit(solver, el, CURRENT, contact_impedance=0.0).voltages
    row = []
    for z_c in (1e-6, 1e-3, 1e-2, 1e-1, 1.0):
        v = simulate_eit(solver, el, CURRENT, contact_impedance=z_c).voltages
        row.append(f"{np.abs(v - v0).max() / np.abs(v0).max():.2e}")
    print("| |" + "|".join(row) + "|", flush=True)


# --------------------------------------------------------------------------
def _truth(model, inclusion):
    g = model.phantom.grid
    x, z = g.meshgrid()
    contrast = quasistatic.admittivity(inclusion[3], F).real / quasistatic.admittivity("muscle", F).real - 1.0
    return np.where(np.hypot(x - inclusion[0], z - inclusion[1]) <= inclusion[2], contrast, 0.0)


def disc_reconstruction(snr=70.0, n_noise=5, z_c=0.02):
    print(f"## B. Inclusión de sangre (r = 4 mm) en un disco de músculo; datos 0.5 mm, modelo 1 mm, SNR {snr:.0f} dB, "
          f"z_c = {z_c} ohm·m², {n_noise} realizaciones de ruido\n")
    positions = {"centro": (0.0, 0.0), "intermedia": (8e-3, 5e-3), "periferia": (14e-3, 0.0)}
    fine0 = disc(0.5e-3)
    elf = ring_electrodes(fine0, 16, 3e-3)
    v0 = simulate_eit(QuasiStaticSolver(fine0, F), elf, CURRENT, contact_impedance=z_c).voltages
    rec = disc(1e-3)
    model = EITModel(rec, F, ring_electrodes(rec, 16, 3e-3), factor=3, current=CURRENT, contact_impedance=z_c)
    _, jac = model.forward(np.zeros(model.n_params), jacobian=True)
    sens = Sensitivity(jac, model._coarse_shape, model.factor, model.active)
    methods = {
        "un paso (NOSER, 1e-2)": lambda v1, v0n, sig: np.where(
            model.tissue, model.to_fine(reconstruct_difference(sens, v1 - v0n, 1e-2)), 0.0),
        "un paso (NOSER, 1e-1)": lambda v1, v0n, sig: np.where(
            model.tissue, model.to_fine(reconstruct_difference(sens, v1 - v0n, 1e-1)), 0.0),
        "GN dif. NOSER": lambda v1, v0n, sig: gauss_newton(model, v1, "noser", 3e-2, 8, data_ref=v0n).image(),
        "GN dif. Tikhonov": lambda v1, v0n, sig: gauss_newton(model, v1, "tikhonov", 3e-2, 8, data_ref=v0n).image(),
        "GN dif. Laplaciano": lambda v1, v0n, sig: gauss_newton(model, v1, "laplaciano", 3e-2, 8,
                                                                  data_ref=v0n).image(),
        "GN dif. TV": lambda v1, v0n, sig: gauss_newton(model, v1, "tv", 3e-2, 8, data_ref=v0n).image(),
        "GN dif. NOSER, curva L": lambda v1, v0n, sig: gauss_newton(model, v1, "noser", "lcurve", 8,
                                                                     data_ref=v0n).image(),
        "GN absoluto NOSER": lambda v1, v0n, sig: gauss_newton(model, v1, "noser", 3e-2, 8).image(),
    }
    print("| posición | método | error de posición (mm) | amplitud AR | resolución RES | deformación SD | ringing |")
    print("|---|---|---|---|---|---|---|")
    summary = {k: [] for k in methods}
    for pname, (cx, cz) in positions.items():
        inc = (cx, cz, 4e-3, "blood")
        v1 = simulate_eit(QuasiStaticSolver(disc(0.5e-3, inc), F), elf, CURRENT, contact_impedance=z_c).voltages
        truth = _truth(model, inc)
        acc = {k: [] for k in methods}
        for seed in range(n_noise):
            rng = np.random.default_rng(seed)
            v0n, v1n = add_noise(v0, snr, rng), add_noise(v1, snr, rng)
            for k, fn in methods.items():
                m = greit_metrics(fn(v1n, v0n, None), truth, model.tissue, rec.grid)
                acc[k].append([m.position_error * 1e3, m.amplitude, m.resolution, m.shape_deformation, m.ringing])
        for k in methods:
            a = np.array(acc[k])
            summary[k].append(a)
            mean, sd = a.mean(0), a.std(0)
            print(f"| {pname} | {k} | {mean[0]:.2f} ± {sd[0]:.2f} | {mean[1]:.2f} ± {sd[1]:.2f} | {mean[2]:.3f} | "
                  f"{mean[3]:.2f} | {mean[4]:.2f} |", flush=True)
    print("\nPromedio sobre las tres posiciones y las realizaciones:\n")
    print("| método | error de posición (mm) | |1 - AR| | RES | SD | ringing |")
    print("|---|---|---|---|---|---|")
    for k in methods:
        a = np.vstack(summary[k])
        print(f"| {k} | {a[:, 0].mean():.2f} | {np.abs(1 - a[:, 1]).mean():.2f} | {a[:, 2].mean():.3f} | "
              f"{a[:, 3].mean():.2f} | {a[:, 4].mean():.2f} |", flush=True)


# --------------------------------------------------------------------------
def leg(snr=70.0):
    print(f"## C. Pierna con fractura anular (hematoma y edema), 16 electrodos de 8 mm, datos 0.5 mm, modelo 1 mm, "
          f"SNR {snr:.0f} dB\n")
    healthy = cross_section_limb("pierna", None, dx=0.5e-3, seed=3)
    broken = cross_section_limb("pierna", RingFracture(angle_deg=0.0), dx=0.5e-3, seed=3)
    elf = ring_electrodes(healthy, 16, 8e-3)
    rec0, rec1 = coarsen_phantom(healthy, 2), coarsen_phantom(broken, 2)
    elr = ring_electrodes(rec0, 16, 8e-3)
    y0 = rec0.property_map(lambda k: quasistatic.admittivity(k, F), dtype=complex)
    y1 = rec1.property_map(lambda k: quasistatic.admittivity(k, F), dtype=complex)
    changed = rec0.labels != rec1.labels
    truth = np.where(changed & (np.abs(y0) > 0), (y1 / np.where(y0 == 0, 1, y0) - 1).real, 0.0)
    g = rec0.grid
    x, z = g.meshgrid()
    hm = rec1.masks["hematoma"]
    c_true = np.array([x[hm].mean(), z[hm].mean()])
    print(f"Hematoma: centro ({c_true[0] * 1e3:.1f}, {c_true[1] * 1e3:.1f}) mm, "
          f"{broken.masks['hematoma'].sum() * 0.25:.1f} mm²; celdas con cambio de tejido en la malla de "
          f"reconstrucción: {changed.sum()}\n")
    print("| electrodo / contacto | modelo de reconstrucción | método | cambio máx. de una tensión (%) | "
          "error del máximo (mm) | error de posición GREIT (mm) | AR | RES |")
    print("|---|---|---|---|---|---|---|---|")
    runs = {"ideal": None, "CEM z_c = 0.003": 0.003, "CEM z_c = 0.03": 0.03}
    data = {}
    for name, zc in runs.items():
        t = time.time()
        a = simulate_eit(QuasiStaticSolver(healthy, F), elf, CURRENT, contact_impedance=zc).voltages
        b = simulate_eit(QuasiStaticSolver(broken, F), elf, CURRENT, contact_impedance=zc).voltages
        data[name] = (a, b)
        print(f"<!-- {name}: datos en {time.time() - t:.0f} s -->", file=sys.stderr, flush=True)
    pairs = [(n, n) for n in runs] + [("CEM z_c = 0.03", "ideal"), ("CEM z_c = 0.03", "CEM z_c = 0.003")]
    models = {n: EITModel(rec0, F, elr, factor=3, current=CURRENT, contact_impedance=z) for n, z in runs.items()}
    for dname, mname in pairs:
        a, b = data[dname]
        rng = np.random.default_rng(0)
        an, bn = add_noise(a, snr, rng), add_noise(b, snr, rng)
        change = (np.abs(b - a) / np.abs(a)).max() * 100
        model = models[mname]
        _, jac = model.forward(np.zeros(model.n_params), jacobian=True)
        sens = Sensitivity(jac, model._coarse_shape, model.factor, model.active)
        imgs = {
            "un paso": np.where(model.tissue, model.to_fine(reconstruct_difference(sens, bn - an, 1e-2)), 0.0),
            "GN NOSER": gauss_newton(model, bn, "noser", 3e-2, 8, data_ref=an).image(),
            "GN Laplaciano": gauss_newton(model, bn, "laplaciano", 3e-2, 8, data_ref=an).image(),
        }
        for k, img in imgs.items():
            s = np.sign(truth[model.tissue].sum()) or 1.0
            i = np.argmax(np.where(model.tissue, s * img, -np.inf))
            pk = np.array([x.ravel()[i], z.ravel()[i]])
            m = greit_metrics(img, truth, model.tissue, g)
            label = dname if dname == mname else f"{dname} (datos)"
            print(f"| {label} | {mname} | {k} | {change:.2f} | {np.hypot(*(pk - c_true)) * 1e3:.1f} | "
                  f"{m.position_error * 1e3:.1f} | {m.amplitude:.2f} | {m.resolution:.3f} |", flush=True)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("analytic", "all"):
        analytic()
        print()
    if part in ("disc", "all"):
        disc_reconstruction()
        print()
    if part in ("leg", "all"):
        leg()
