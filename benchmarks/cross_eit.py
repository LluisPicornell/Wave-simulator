"""Comparación cruzada de la EIT con pyEIT (FEM): problema directo y reconstrucción de diferencias.

Uso (desde la raíz; pyEIT sólo en un entorno aparte, nunca en ``pyproject.toml``; el entorno necesita además
``numba`` y ``matplotlib`` para importar wavesim):

    python -m venv venv_pyeit && venv_pyeit/bin/pip install pyeit numba matplotlib
    PYTHONPATH=$(pwd) NUMBA_NUM_THREADS=1 venv_pyeit/bin/python benchmarks/cross_eit.py --save     # ~3 min

Secciones (``--only``): ``forward`` (tensiones de borde del disco homogéneo y con inclusión) y ``recon``
(posición del máximo y métricas GREIT de JAC/GREIT de pyEIT frente a un paso y Gauss-Newton de wavesim).
``--save`` escribe ``benchmarks/data/cross_eit/pyeit.npz`` (datos directos de pyEIT, 0.1 MB), que usa
``tests/test_cross_eit.py`` sin tener pyEIT. Las tablas son las de ``docs/validacion/cruzada_eit.md``.

Convenciones (comprobadas contra la solución analítica, no supuestas):
- pyEIT usa el disco unidad, ``sigma = 1`` y corriente 1; en 2D la tensión no depende del radio, así que se escala
  con ``V_wavesim = V_pyEIT * I / sigma`` (aquí ``sigma = 0.5`` S/m, ``I = 1`` mA/m, ``R = 20`` mm).
- Electrodo ``k`` de pyEIT: ángulo 180° - 22.5° k (antihorario desde +x); wavesim: ``start_deg = -90`` y sentido
  horario desde arriba; las coordenadas se pasan con ``(x, z) = (x_p, -y_p)``.
- pyEIT mide ``V[q+1] - V[q]`` y wavesim ``V[q] - V[q+1]``: los datos tienen signo opuesto. La inyección (+ en ``p``,
  - en ``p+1``) y el orden de los 208 datos coinciden.
- pyEIT sólo tiene electrodos puntuales (nodos); su CEM está "en desarrollo" (docstring de ``Forward.solve``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from wavesim.core.grid import Grid2D
from wavesim.em import QuasiStaticSolver, Sensitivity, reconstruct_difference, ring_electrodes, simulate_eit
from wavesim.em.eit import EITModel, add_noise, disc_cem_series, disc_point_voltage, gauss_newton, greit_metrics
from wavesim.em.quasistatic import adjacent_protocol
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

DATA = Path(__file__).parent / "data" / "cross_eit"
R = 20e-3
F = 50e3
SIGMA = 0.5  # S/m, real: pyEIT sólo admite conductividad real en este uso
CURRENT = 1e-3  # A/m
N_EL = 16
INC_R = 4e-3
INC_CONTRAST = 2.0  # sigma_inclusión / sigma_fondo
POSITIONS = {"centro": (0.0, 0.0), "intermedia": (8e-3, 5e-3), "periferia": (14e-3, 0.0)}
THETA = -np.pi / 2 + 2 * np.pi * np.arange(N_EL) / N_EL  # ángulo de cada electrodo (horario desde arriba)
FRAMES, MEAS = adjacent_protocol(N_EL)


def title(s):
    print(f"\n## {s}\n")


def rows(header, data):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in data:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def disc(dx, inc=None):
    """Disco de radio ``R`` con conductividad real ``SIGMA`` e inclusión circular opcional ``(cx, cz)``."""
    size = 2 * R + 10e-3
    g = Grid2D.from_size(size, size, dx, x0=-size / 2, z0=-size / 2)
    x, z = g.meshgrid()
    labels = np.full(g.shape, TISSUE_INDEX["water"], np.int16)
    # Estricto: con ``<= R`` queda una celda aislada en cada extremo del eje, sin caras, que hace singular a NOSER
    labels[np.hypot(x, z) < R - 1e-6] = TISSUE_INDEX["muscle"]
    y = np.where(labels == TISSUE_INDEX["muscle"], SIGMA + 0j, 0.0)
    if inc is not None:
        m = (np.hypot(x - inc[0], z - inc[1]) <= INC_R) & (labels == TISSUE_INDEX["muscle"])
        y[m] = SIGMA * INC_CONTRAST
    return Phantom(g, labels), y


def wavesim_data(dx, width, inc=None, z_c=None):
    ph, y = disc(dx, inc)
    el = ring_electrodes(ph, N_EL, width, start_deg=-90.0)
    return simulate_eit(QuasiStaticSolver(ph, F, admittivity_map=y), el, CURRENT, contact_impedance=z_c).voltages.real


def analytic_point():
    """Tensiones de electrodos puntuales (solución cerrada) con las unidades de wavesim."""
    return np.array([disc_point_voltage(THETA[p], THETA[(p + 1) % N_EL], THETA[a], THETA[b], SIGMA, CURRENT)
                     for p, (a, b) in zip(FRAMES, MEAS)]).real


def analytic_cem(z_c, width):
    """Serie de Fourier del CEM en el disco para las 16 tramas."""
    arcs = [(t - width / (2 * R), t + width / (2 * R)) for t in THETA]
    out = []
    for p in range(N_EL):
        cur = np.zeros(N_EL, complex)
        cur[p], cur[(p + 1) % N_EL] = CURRENT, -CURRENT
        u = disc_cem_series(R, SIGMA, z_c, arcs, cur, n_modes=200)[0]
        out += [u[a] - u[b] for q, (a, b) in zip(FRAMES, MEAS) if q == p]
    return np.array(out).real


def errors(a, ref):
    """(error máximo relativo por dato, error relativo de la norma) en %."""
    emax = 100 * float(np.max(np.abs(a - ref) / np.abs(ref)))
    return emax, 100 * float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def to_wavesim_units(v_pyeit):
    return -np.asarray(v_pyeit) * CURRENT / SIGMA


def to_pyeit_units(v_wavesim):
    return -np.asarray(v_wavesim).real * SIGMA / CURRENT


# --------------------------------------------------------------------------------------
def pyeit_forward(h0, incs=()):
    """Tensiones de pyEIT (en unidades de wavesim) del disco homogéneo y de cada inclusión de ``incs``."""
    import pyeit.eit.protocol as protocol
    from pyeit.eit.fem import EITForward
    from pyeit.mesh import create

    mesh = create(N_EL, h0=h0)
    fwd = EITForward(mesh, protocol.create(N_EL, dist_exc=1, step_meas=1, parser_meas="std"))
    c = mesh.elem_centers
    out = [to_wavesim_units(fwd.solve_eit(perm=1.0))]
    for cx, cz in incs:  # (x, z) de wavesim = (x_p, -y_p)
        perm = np.where(np.hypot(c[:, 0] - cx / R, c[:, 1] + cz / R) <= INC_R / R, INC_CONTRAST, 1.0)
        out.append(to_wavesim_units(fwd.solve_eit(perm=perm)))
    return mesh, out


def section_forward(save):
    try:
        import pyeit  # noqa: F401
    except ImportError:
        print("pyEIT no está instalado: se salta la sección forward (ver la cabecera).")
        return None
    title("Problema directo: disco homogéneo y con inclusión, 16 electrodos, protocolo adyacente (208 datos)")
    ref = analytic_point()
    data = []
    incs = [POSITIONS[k] for k in POSITIONS]
    store = {}
    for h0 in (0.1, 0.05, 0.025, 0.0125):
        t0 = time.time()
        mesh, vs = pyeit_forward(h0, incs if h0 == 0.0125 else ())
        e = errors(vs[0], ref)
        data.append(("pyEIT (FEM, electrodo puntual)", f"h0 = {h0} ({mesh.n_nodes} nodos)", f"{e[0]:.2f}",
                     f"{e[1]:.2f}", f"{time.time() - t0:.0f} s"))
        if h0 == 0.0125:
            store["v0"], store["v1"] = vs[0], np.array(vs[1:])
    for dx, w in ((1e-3, 2e-3), (0.5e-3, 2e-3), (0.5e-3, 1e-3), (0.25e-3, 2e-3), (0.125e-3, 2e-3)):
        t0 = time.time()
        e = errors(wavesim_data(dx, w), ref)
        data.append(("wavesim (electrodo ideal)", f"dx = {dx * 1e3:g} mm, ancho {w * 1e3:g} mm", f"{e[0]:.2f}",
                     f"{e[1]:.2f}", f"{time.time() - t0:.0f} s"))
    print("**Frente a la solución analítica de electrodos puntuales** (`disc_point_voltage`):\n")
    rows(["código", "malla", "error máx. por dato (%)", "error de la norma (%)", "tiempo"], data)

    data = []
    for z_c in (0.003, 0.03):
        ser = analytic_cem(z_c, 2e-3)
        for dx in (0.5e-3, 0.25e-3):
            e = errors(wavesim_data(dx, 2e-3, z_c=z_c), ser)
            data.append((f"{z_c}", f"{dx * 1e3:g}", f"{e[0]:.2f}", f"{e[1]:.2f}"))
    print("**CEM de wavesim frente a la serie de Fourier del CEM** (todas las tramas, electrodos de 2 mm; "
          "pyEIT no tiene CEM, por lo que no hay comparación directa con él):\n")
    rows(["z_c (ohm·m²)", "dx (mm)", "error máx. por dato (%)", "error de la norma (%)"], data)

    data = []
    for dx, w in ((0.5e-3, 2e-3), (0.25e-3, 2e-3)):
        v0 = wavesim_data(dx, w)
        for k, (name, pos) in enumerate(POSITIONS.items()):
            v1 = wavesim_data(dx, w, inc=pos)
            ref1 = store["v1"][k]
            e1, e_d = errors(v1, ref1), 100 * np.linalg.norm((v1 - v0) - (ref1 - store["v0"])) / np.linalg.norm(
                ref1 - store["v0"])
            data.append((f"{dx * 1e3:g}", name, f"{100 * np.max(np.abs(ref1 - store['v0']) / np.abs(store['v0'])):.1f}",
                         f"{e1[0]:.2f}", f"{e1[1]:.2f}", f"{e_d:.2f}"))
    print("**Disco con inclusión** (r = 4 mm, sigma x2; wavesim con electrodos ideales de 2 mm frente a pyEIT con "
          "h0 = 0.0125; no hay solución analítica, pyEIT coincide con la de puntuales a 0.01 % en el disco "
          "homogéneo):\n")
    rows(["dx wavesim (mm)", "posición", "máx. dV/V de la inclusión (%)", "error máx. por dato de V (%)",
          "error de norma de V (%)", "error de norma de la diferencia dV (%)"], data)
    return store


# --------------------------------------------------------------------------------------
def _pyeit_imager(mesh, rec_grid, inverse, lamb, jac_p=0.5):
    """Función ``(v1, v0) -> imagen`` de pyEIT (JAC o GREIT) en la malla de reconstrucción de wavesim.

    La matriz de reconstrucción se calcula una sola vez (es lo caro: jacobiano denso de la malla de pyEIT).
    """
    import matplotlib.tri as mtri
    import pyeit.eit.protocol as protocol
    from pyeit.eit.greit import GREIT
    from pyeit.eit.jac import JAC

    proto = protocol.create(N_EL, dist_exc=1, step_meas=1, parser_meas="std")
    x, z = rec_grid.meshgrid()
    xp, yp = x / R, -z / R
    if inverse == "jac":
        eit = JAC(mesh, proto)
        eit.setup(p=jac_p, lamb=lamb, method="kotre", perm=1.0, jac_normalized=False)
        idx = mtri.Triangulation(mesh.node[:, 0], mesh.node[:, 1], mesh.element).get_trifinder()(xp, yp)

        def image(v1, v0):
            ds = np.real(eit.solve(to_pyeit_units(v1), to_pyeit_units(v0), normalize=False))
            return np.where(idx >= 0, ds[np.maximum(idx, 0)], 0.0)
        return image
    eit = GREIT(mesh, proto)
    eit.setup(method="dist", p=0.5, lamb=lamb, n=32, s=20.0, ratio=0.1, perm=1.0, jac_normalized=False)
    xg, yg, _ = eit.get_grid()
    gx, gy = xg[0, :], yg[:, 0]
    ix = np.clip(np.searchsorted(gx, xp) - 1, 0, gx.size - 2)
    iy = np.clip(np.searchsorted(gy, yp) - 1, 0, gy.size - 2)
    fx = np.clip((xp - gx[ix]) / (gx[ix + 1] - gx[ix]), 0, 1)
    fy = np.clip((yp - gy[iy]) / (gy[iy + 1] - gy[iy]), 0, 1)

    def image(v1, v0):
        ds = np.real(eit.solve(to_pyeit_units(v1), to_pyeit_units(v0), normalize=False))
        img = np.nan_to_num(ds.reshape(xg.shape))  # bilineal; fuera de la malla de pyEIT vale 0
        return ((1 - fy) * ((1 - fx) * img[iy, ix] + fx * img[iy, ix + 1])
                + fy * ((1 - fx) * img[iy + 1, ix] + fx * img[iy + 1, ix + 1]))
    return image


def _metrics(img, truth, model, centre):
    g = model.phantom.grid
    x, z = g.meshgrid()
    i = np.argmax(np.where(model.tissue, img, -np.inf))
    peak = float(np.hypot(x.ravel()[i] - centre[0], z.ravel()[i] - centre[1]))
    m = greit_metrics(img, truth, model.tissue, g)
    return np.array([peak * 1e3, m.position_error * 1e3, m.amplitude, m.resolution, m.shape_deformation, m.ringing])


def section_recon(store):
    try:
        import pyeit  # noqa: F401
        from pyeit.mesh import create
    except ImportError:
        print("pyEIT no está instalado: se salta la sección recon (ver la cabecera).")
        return
    title("Reconstrucción de diferencias: pyEIT (JAC, GREIT) frente a wavesim (un paso, Gauss-Newton)")
    mesh_inv = create(N_EL, h0=0.05)  # malla de reconstrucción de pyEIT (1476 nodos); los datos salen de otra malla
    rec, y0 = disc(1e-3)
    model = EITModel(rec, F, ring_electrodes(rec, N_EL, 3e-3, start_deg=-90.0), factor=3, current=CURRENT, y0=y0)
    _, jac = model.forward(np.zeros(model.n_params), jacobian=True)
    sens = Sensitivity(jac, model._coarse_shape, model.factor, model.active)
    x, z = rec.grid.meshgrid()
    ph0, y0f = disc(0.5e-3)
    el_f = ring_electrodes(ph0, N_EL, 3e-3, start_deg=-90.0)
    v0_w = simulate_eit(QuasiStaticSolver(ph0, F, admittivity_map=y0f), el_f, CURRENT).voltages.real
    methods = {
        "wavesim un paso (NOSER 1e-2)": lambda v1, v0: np.where(
            model.tissue, model.to_fine(reconstruct_difference(sens, v1 - v0, 1e-2)), 0.0),
        "wavesim GN (NOSER 3e-2)": lambda v1, v0: gauss_newton(model, v1, "noser", 3e-2, 8, data_ref=v0).image(),
    }
    for lamb in (1e-4, 1e-3, 1e-2):
        methods[f"pyEIT JAC (kotre p=0.5, lamb={lamb:g})"] = _pyeit_imager(mesh_inv, rec.grid, "jac", lamb)
    for lamb in (1e-3, 1e-2):
        methods[f"pyEIT GREIT (lamb={lamb:g})"] = _pyeit_imager(mesh_inv, rec.grid, "greit", lamb)
    result = {}
    for source in ("wavesim", "pyEIT"):
        desc = ("electrodos ideales de 3 mm, malla 0.5 mm" if source == "wavesim"
                else "electrodo puntual, malla h0 = 0.0125")
        print(f"**Datos de {source}** ({desc}; modelo de reconstrucción de wavesim: malla 1 mm, bloques de 3x3; "
              "sin ruido y con SNR 70 dB, 3 realizaciones):\n")
        data = []
        for k, (pname, (cx, cz)) in enumerate(POSITIONS.items()):
            truth = np.where(np.hypot(x - cx, z - cz) <= INC_R, INC_CONTRAST - 1.0, 0.0)
            if source == "wavesim":
                v0 = v0_w
                v1 = wavesim_data(0.5e-3, 3e-3, inc=(cx, cz))
            else:
                v0, v1 = store["v0"], store["v1"][k]
            for mname, fn in methods.items():
                acc = []
                for seed in range(4):  # 0 = sin ruido
                    if seed == 0:
                        a, b = v0, v1
                    else:
                        rng = np.random.default_rng(seed)
                        a, b = add_noise(v0, 70.0, rng), add_noise(v1, 70.0, rng)
                    acc.append(_metrics(fn(b, a), truth, model, (cx, cz)))
                acc = np.array(acc)
                noisy = acc[1:].mean(0)
                result[(source, pname, mname)] = (acc[0], noisy)
                data.append((pname, mname, f"{acc[0][0]:.1f} / {noisy[0]:.1f}", f"{acc[0][1]:.2f} / {noisy[1]:.2f}",
                             f"{acc[0][2]:.2f} / {noisy[2]:.2f}", f"{acc[0][3]:.3f}", f"{acc[0][4]:.2f}",
                             f"{acc[0][5]:.2f}"))
        rows(["posición", "método", "error del máximo (mm) sin ruido / con ruido",
              "error de posición GREIT (mm)", "AR", "RES", "SD", "ringing"], data)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=["forward", "recon"], default=["forward", "recon"])
    ap.add_argument("--save", action="store_true", help="guarda los datos de pyEIT en benchmarks/data/cross_eit/")
    args = ap.parse_args()
    store = None
    if "forward" in args.only or "recon" in args.only:
        store = section_forward(args.save) if "forward" in args.only else None
        if "recon" in args.only:
            if store is None:  # recon necesita los datos directos de pyEIT
                _, vs = pyeit_forward(0.0125, [POSITIONS[k] for k in POSITIONS])
                store = {"v0": vs[0], "v1": np.array(vs[1:])}
            section_recon(store)
    if args.save and store is not None:
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA / "pyeit.npz", **store)
        print(f"Guardado {DATA / 'pyeit.npz'}")


if __name__ == "__main__":
    main()
