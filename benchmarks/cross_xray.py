"""Comparación cruzada de rayos X / TAC con ASTRA Toolbox, SpekPy y XCIST (CatSim).

Uso (desde la raíz del repositorio; cada herramienta se instala SÓLO en un entorno aparte, nunca en
``pyproject.toml``, y esos entornos necesitan además ``numpy``, ``matplotlib`` y ``numba`` para importar wavesim):

    python -m venv venv_astra && venv_astra/bin/pip install astra-toolbox numpy matplotlib numba
    PYTHONPATH=$(pwd) venv_astra/bin/python benchmarks/cross_xray.py --only astra --save

    python -m venv venv_spekpy && venv_spekpy/bin/pip install spekpy numba matplotlib
    PYTHONPATH=$(pwd) venv_spekpy/bin/python benchmarks/cross_xray.py --only spekpy --save

    # XCIST: el sdist de gecatsim trae los scripts de compilación con finales de línea CRLF y falla con
    # `pip install gecatsim`; hay que descargarlo (`pip download --no-binary :all: gecatsim`), pasar a LF los
    # ficheros de gecatsim/clib_build y gecatsim/reconstruction (sed -i 's/\\r$//') e instalar desde la carpeta.
    PYTHONPATH=$(pwd) venv_gecatsim/bin/python benchmarks/cross_xray.py --only xcist --save

Cada sección es opcional: si la herramienta no está instalada se avisa y se salta. ``--save`` escribe las
referencias pequeñas en ``benchmarks/data/cross_xray/`` (``astra.npz``, ``spekpy.npz``, ``xcist.npz``), que usa
``tests/test_cross_xray.py`` para comparar wavesim sin tener instaladas las herramientas. Las tablas impresas son
las de ``docs/validacion/cruzada_xray.md``. TIGRE no se pudo instalar (necesita ``nvcc``/CUDA); ver el documento.

Secciones (``--only``):
  astra   proyector (paralelo y abanico plano), FBP paralelo, SART y SIRT frente a ASTRA 2.5.0 en CPU;
  spekpy  espectros del tubo en casos que NO entraron en el ajuste (kVp, Al, Cu, ángulo del ánodo);
  xcist   SPR de un cilindro de agua frente al modelo de dispersión por convolución de XCIST.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from wavesim.core.grid import Grid2D
from wavesim.em import fanbeam as fb
from wavesim.em import xray, xray_phantoms as xp
from wavesim.em.projection import detector_axis, fbp, project
from wavesim.materials import xray as xd

DATA = Path(__file__).parent / "data" / "cross_xray"
N_PAR, N_FAN = 120, 60  # vistas del haz paralelo (180°) y del abanico (360°)
SID, SDD = 0.6, 1.1  # distancia fuente-isocentro y fuente-detector del abanico (m)


def title(s):
    print(f"\n## {s}\n")


def rows(header, data):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in data:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def rel(a, b):
    """Diferencia relativa en norma 2 entre ``a`` y la referencia ``b``."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --------------------------------------------------------------------------------------
# ASTRA
# --------------------------------------------------------------------------------------
def astra_cases():
    """Dos objetos: Shepp-Logan (con sinograma analítico exacto) y la sección de pierna a 60 keV."""
    from wavesim.phantoms import cross_section_limb

    g = Grid2D.from_size(0.2, 0.2, 0.003125, x0=-0.1, z0=-0.1)  # 64 x 64
    sl = xp.shepp_logan(0.08, 20.0)
    r = np.hypot(*g.meshgrid())
    cases = {"sl": dict(grid=g, img=xp.rasterize(sl, g), ell=sl, mask=r < 0.08 * 0.95)}
    leg = cross_section_limb("pierna", dx=1.5e-3)
    mu = leg.property_map(lambda k: float(xd.linear_attenuation(k, np.array([60.0]))[0]), physics="xray")
    rl = np.hypot(*leg.grid.meshgrid())
    cases["leg"] = dict(grid=leg.grid, img=mu, ell=None, mask=rl < 0.04)
    return cases


def _a_fp(astra, img, pg, vg, kind):
    pid = astra.create_projector(kind, pg, vg)
    sid, sino = astra.create_sino(img, pid)
    astra.data2d.delete(sid)
    astra.projector.delete(pid)
    return sino


def _a_fbp(astra, sino_px, pg, vg, filt, kind):
    pid = astra.create_projector(kind, pg, vg)
    sid = astra.data2d.create("-sino", pg, sino_px)
    rid = astra.data2d.create("-vol", vg)
    cfg = astra.astra_dict("FBP")
    cfg.update(ProjectorId=pid, ProjectionDataId=sid, ReconstructionDataId=rid, FilterType=filt)
    alg = astra.algorithm.create(cfg)
    astra.algorithm.run(alg)
    out = astra.data2d.get(rid).copy()
    for f, i in ((astra.algorithm, alg), (astra.data2d, sid), (astra.data2d, rid), (astra.projector, pid)):
        f.delete(i)
    return out


def _a_iter(astra, name, sino_px, pg, vg, kind, its, nonneg, per_iter):
    """``name`` = SART o SIRT; ``per_iter`` = pasos de ASTRA por iteración (SART: una vista por paso)."""
    pid = astra.create_projector(kind, pg, vg)
    sid = astra.data2d.create("-sino", pg, sino_px)
    rid = astra.data2d.create("-vol", vg)
    cfg = astra.astra_dict(name)
    cfg.update(ProjectorId=pid, ProjectionDataId=sid, ReconstructionDataId=rid)
    if nonneg:
        cfg["option"] = {"MinConstraint": 0.0}
    alg = astra.algorithm.create(cfg)
    out, done = {}, 0
    for it in its:
        astra.algorithm.run(alg, (it - done) * per_iter)
        done = it
        out[it] = astra.data2d.get(rid).copy()
    for f, i in ((astra.algorithm, alg), (astra.data2d, sid), (astra.data2d, rid), (astra.projector, pid)):
        f.delete(i)
    return out


def section_astra(save):
    try:
        import astra
    except ImportError:
        print("ASTRA no está instalado: se salta la sección astra (ver la cabecera para instalarlo).")
        return
    title(f"ASTRA Toolbox {astra.__version__} (CPU) frente a wavesim")
    store = {}
    res_proj, res_fbp, res_it = [], [], []
    for name, c in astra_cases().items():
        g, img, mask = c["grid"], c["img"], c["mask"]
        dx = g.dx
        vg = astra.create_vol_geom(*g.shape)
        # Convención de unidades: ASTRA trabaja en píxeles (lado 1), así que sus integrales de línea son
        # mu·(longitud en píxeles); se multiplican por dx para pasarlas a mu·m. Las imágenes (1/m) no cambian.
        det = detector_axis(g)
        a_par = np.arange(N_PAR) * 180.0 / N_PAR
        a_fan = np.arange(N_FAN) * 360.0 / N_FAN
        geom = fb.FanGeometry.covering(g, SID, SDD, curved=False)
        pg_par = astra.create_proj_geom("parallel", float(det[1] - det[0]) / dx, det.size, np.radians(a_par))
        # Abanico: el ángulo de ASTRA es el de wavesim + 180° y el eje del detector está invertido.
        pg_fan = astra.create_proj_geom("fanflat", geom.det_spacing / dx, geom.n_det, np.radians(a_fan) + np.pi,
                                        SID / dx, (SDD - SID) / dx)
        to_a = lambda s, dx=dx: s[:, ::-1] / dx  # noqa: E731  sinograma de wavesim -> datos de ASTRA
        from_a = lambda s, dx=dx: s[:, ::-1] * dx  # noqa: E731

        w_par = project(img, g, a_par, det)[0]
        w_fan = fb.fan_project(img, g, geom, a_fan)[0]
        p_lin = _a_fp(astra, img, pg_par, vg, "linear") * dx
        p_str = _a_fp(astra, img, pg_par, vg, "strip") * dx
        f_lin = from_a(_a_fp(astra, img, pg_fan, vg, "line_fanflat"))
        f_str = from_a(_a_fp(astra, img, pg_fan, vg, "strip_fanflat"))
        ex_par = ex_fan = None
        if c["ell"] is not None:
            ex_par = xp.parallel_sinogram(c["ell"], g, a_par, det)
            ex_fan = fb.fan_exact_sinogram(c["ell"], g, geom, a_fan)
        fmt = lambda v: "-" if v is None else f"{100 * v:.2f} %"  # noqa: E731
        res_proj.append((name, "paralelo", fmt(rel(w_par, p_lin)), fmt(rel(w_par, p_str)), fmt(rel(p_lin, p_str)),
                         fmt(None if ex_par is None else rel(w_par, ex_par)),
                         fmt(None if ex_par is None else rel(p_lin, ex_par)),
                         fmt(None if ex_par is None else rel(p_str, ex_par))))
        res_proj.append((name, "abanico plano", fmt(rel(w_fan, f_lin)), fmt(rel(w_fan, f_str)), fmt(rel(f_lin, f_str)),
                         fmt(None if ex_fan is None else rel(w_fan, ex_fan)),
                         fmt(None if ex_fan is None else rel(f_lin, ex_fan)),
                         fmt(None if ex_fan is None else rel(f_str, ex_fan))))

        # FBP paralelo: los mismos datos (analíticos en Shepp-Logan, proyector de wavesim en la pierna)
        data = ex_par if ex_par is not None else w_par
        rm = lambda a, b, mask=mask: float(np.sqrt(np.mean((a - b)[mask] ** 2)))  # noqa: E731
        for filt in ("ram-lak", "shepp-logan"):
            wf = fbp(data, a_par, det, g, window=filt)
            for kind in ("strip", "linear"):
                af = _a_fbp(astra, data / dx, pg_par, vg, filt, kind)
                store[f"{name}_fbp_{kind}_{filt}"] = af
                res_fbp.append((name, filt, kind, f"{rm(wf, af):.3f}", f"{rm(wf, img):.3f}", f"{rm(af, img):.3f}",
                                f"{100 * rm(wf, af) / rm(af, img):.1f} %"))

        # Iterativos con el abanico plano y el mismo sinograma (analítico o de wavesim)
        data_f = ex_fan if ex_fan is not None else w_fan
        for algo, sub, its, per in (("SART", 1, (1, 5, 10), N_FAN), ("SIRT", N_FAN, (10, 30), 1)):
            for kind in ("strip_fanflat", "line_fanflat"):
                for nonneg in (False, True):
                    if kind == "line_fanflat" and (algo == "SIRT" or nonneg):
                        continue  # el contraste ya se ve con SART sin restricción
                    ra = _a_iter(astra, algo, to_a(data_f), pg_fan, vg, kind, its, nonneg, per)
                    for it in its:
                        t0 = time.time()
                        rw = fb.os_sart(data_f, a_fan, geom, g, n_iter=it, subset_size=sub, nonneg=nonneg)
                        dt = time.time() - t0
                        key = f"{name}_{algo.lower()}_{kind.split('_')[0]}_{'nn' if nonneg else 'free'}_it{it}"
                        store[key] = ra[it]
                        res_it.append((name, algo, kind.split("_")[0], "sí" if nonneg else "no", it,
                                       f"{rm(rw, ra[it]):.3f}", f"{rm(rw, img):.3f}", f"{rm(ra[it], img):.3f}",
                                       f"{dt:.1f} s"))
        store.update({f"{name}_img": img, f"{name}_a_par": a_par, f"{name}_a_fan": a_fan,
                      f"{name}_grid": np.array([g.nx, g.nz, g.dx, g.x0, g.z0]), f"{name}_fan": np.array([SID, SDD]),
                      f"{name}_par_linear": p_lin, f"{name}_par_strip": p_str, f"{name}_fan_line": f_lin,
                      f"{name}_fan_strip": f_str, f"{name}_fbp_data": data, f"{name}_fan_data": data_f})

    print("**Proyector** (diferencia relativa en norma 2 entre sinogramas de la misma imagen rasterizada; "
          "`lin`/`str` = núcleos lineal y de tira de ASTRA; `exacto` = integral analítica del fantoma continuo, "
          "sólo Shepp-Logan):\n")
    rows(["objeto", "geometría", "wavesim-ASTRA lin", "wavesim-ASTRA str", "ASTRA lin-str", "wavesim-exacto",
          "ASTRA lin-exacto", "ASTRA str-exacto"], res_proj)
    print("**FBP paralelo** (120 vistas / 180°, mismos datos; RMSE en 1/m dentro de la región del objeto):\n")
    rows(["objeto", "filtro", "núcleo ASTRA", "RMSE wavesim-ASTRA", "RMSE wavesim-objeto", "RMSE ASTRA-objeto",
          "wavesim-ASTRA / ASTRA-objeto"], res_fbp)
    print("**SART / SIRT en abanico plano** (60 vistas / 360°, mismas iteraciones; SART = `subset_size=1`, "
          "SIRT = `subset_size=60`; RMSE en 1/m):\n")
    rows(["objeto", "algoritmo", "núcleo ASTRA", "no neg.", "iter.", "RMSE wavesim-ASTRA", "RMSE wavesim-objeto",
          "RMSE ASTRA-objeto", "t wavesim"], res_it)
    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA / "astra.npz", astra_version=astra.__version__,
                            **{k: np.asarray(v, np.float32 if np.asarray(v).ndim > 1 else float)
                               for k, v in store.items()})
        print(f"Guardado {DATA / 'astra.npz'} ({(DATA / 'astra.npz').stat().st_size / 1e3:.0f} kB)")


# --------------------------------------------------------------------------------------
# SpekPy
# --------------------------------------------------------------------------------------
# (kVp, mm Al, mm Cu, ángulo del ánodo en grados, entró en el ajuste de _DEPTH_FRACTION y la fracción de líneas K)
SPEK_CASES = (
    [(k, 2.5, 0.0, 12.0, True) for k in (50, 70, 80, 100, 120)] + [(120, 4.0, 0.0, 12.0, True)]
    + [(k, 2.5, 0.0, 12.0, False) for k in (60, 90, 110, 140, 150)]  # otros kVp (interpolación y extrapolación)
    + [(100, 1.0, 0.0, 12.0, False), (100, 6.0, 0.0, 12.0, False), (150, 10.0, 0.0, 12.0, False)]  # otras filtraciones
    + [(100, 2.5, 0.1, 12.0, False), (100, 2.5, 0.3, 12.0, False), (120, 2.5, 0.2, 12.0, False),
       (120, 2.5, 0.5, 12.0, False), (120, 2.5, 1.0, 12.0, False), (150, 2.5, 0.5, 12.0, False),
       (150, 2.5, 1.0, 12.0, False)]  # filtración de cobre
    + [(100, 2.5, 0.0, th, False) for th in (10.0, 15.0, 20.0)] + [(120, 2.5, 0.0, th, False) for th in (10.0, 20.0)]
    + [(80, 2.5, 0.0, 20.0, False), (140, 2.5, 0.0, 10.0, False), (140, 2.5, 0.0, 20.0, False)]  # ánodo
)


def wavesim_spectrum(kvp, al, cu, th):
    """Espectro de wavesim con el modelo completo (líneas K y autoabsorción del ánodo)."""
    return xray.tube_spectrum(float(kvp), al, cu_filter_mm=cu, characteristic=True, anode_angle_deg=th)


def rebin_1kev(energies, weights, kvp):
    """Fracción de fotones en las celdas de 1 keV ``[10, 11), ..., [kvp-1, kvp)``, renormalizada a 1.

    Las líneas K se asignan a la celda que las contiene: así el espectro de wavesim (líneas exactas) y el de
    SpekPy (histograma) son comparables celda a celda.
    """
    n = int(kvp) - 10
    idx = np.floor(np.asarray(energies)).astype(int) - 10
    keep = (idx >= 0) & (idx < n)  # wavesim no tiene fotones por debajo de 10 keV: se descartan también en SpekPy
    out = np.bincount(idx[keep], weights=np.asarray(weights)[keep], minlength=n)
    return out / out.sum()


def spectrum_metrics(w_spec, ref_hist, ref_hvl, ref_emean, kvp):
    """(error relativo de HVL, error relativo de energía media, L1 del espectro normalizado) de wavesim."""
    hist = rebin_1kev(w_spec.energies_kev, w_spec.weights, kvp)
    hvl = xray.half_value_layer(w_spec)
    return (hvl / ref_hvl - 1.0, w_spec.mean_energy / ref_emean - 1.0, float(np.abs(hist - ref_hist).sum()))


def section_spekpy(save):
    try:
        import spekpy as sp
    except ImportError:
        print("SpekPy no está instalado: se salta la sección spekpy (ver la cabecera para instalarlo).")
        return
    title(f"SpekPy {sp.__version__} frente al espectro de wavesim")
    meta, hists, data = [], [], []
    for kvp, al, cu, th, fit in SPEK_CASES:
        s = sp.Spek(kvp=kvp, th=th)
        s.filter("Al", al)
        if cu:
            s.filter("Cu", cu)
        k, phi = s.get_spectrum(edges=False)
        # SpekPy da fluencia por keV en celdas de 0.5 keV; se pasa a fotones por celda de 1 keV desde 10 keV
        hist = rebin_1kev(k, phi * 0.5, kvp)
        hvl, em = float(s.get_hvl1()), float(s.get_emean())
        e_hvl, e_em, l1 = spectrum_metrics(wavesim_spectrum(kvp, al, cu, th), hist, hvl, em, kvp)
        meta.append((kvp, al, cu, th, float(fit), hvl, em))
        hists.append(np.pad(hist, (0, 150 - hist.size)))
        data.append(("ajuste" if fit else "predicción", kvp, al, cu, th, f"{hvl:.3f}", f"{em:.1f}",
                     f"{100 * e_hvl:+.1f} %", f"{100 * e_em:+.1f} %", f"{l1:.3f}"))
    rows(["tipo", "kVp", "Al (mm)", "Cu (mm)", "ánodo (°)", "HVL SpekPy (mm Al)", "E media SpekPy (keV)",
          "error HVL", "error E media", "L1 espectro"], data)
    for label, sel in (("ajuste", True), ("predicción", False)):
        vals = np.array([(abs(float(r[7].split()[0])), abs(float(r[8].split()[0])), float(r[9])) for r in data
                         if (r[0] == "ajuste") == sel])
        print(f"{label}: error medio |HVL| {vals[:, 0].mean():.1f} % (máx. {vals[:, 0].max():.1f} %), "
              f"|E media| {vals[:, 1].mean():.1f} % (máx. {vals[:, 1].max():.1f} %), "
              f"L1 medio {vals[:, 2].mean():.3f} (máx. {vals[:, 2].max():.3f}); {len(vals)} casos")
    print()
    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA / "spekpy.npz", spekpy_version=sp.__version__, meta=np.array(meta),
                            hist=np.array(hists, np.float32))
        print(f"Guardado {DATA / 'spekpy.npz'}")


# --------------------------------------------------------------------------------------
# XCIST
# --------------------------------------------------------------------------------------
XC_SID, XC_SDD, XC_ROWS = 540.0, 950.0, 64  # mm; geometría por defecto de gecatsim (filas de 1 mm)
XC_DIAMETERS = (100, 200, 300, 350)  # mm
XC_X_ISO = (0.0, 20.0, 40.0, 60.0, 80.0)  # posiciones laterales (mm en el isocentro) donde se compara el SPR


def xcist_spr(diameters_mm, workdir):
    """SPR por fila central de XCIST para cilindros de agua de longitud 200 mm.

    Devuelve ``{diámetro: SPR(columna)}`` y el espectro (energías keV, pesos) que usa XCIST tras su
    filtración plana, para que wavesim use exactamente el mismo.
    """
    import os

    import gecatsim as xc
    from gecatsim.pyfiles.GetMu import GetMu

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        out = {}
        for d in diameters_mm:
            with open("cyl.ppm", "w") as f:
                f.write("materialList = {'water' };\n\nobject.center(1,:) = [0 0 0];\n"
                        f"object.half_axes(1,:) = [{d / 2} {d / 2} 100.0];\nobject.euler_angs(1,:) = [90 0 0];\n"
                        "object.density(1) = 1.0;\nobject.type(1) = 2;\nobject.material(1) = 1;\n"
                        "object.axial_lims(1,:) = [0 0];\nobject.shape(1) = 0;\nobject.clip{1} = [];\n")
            prep = {}
            for scatter in (False, True):
                ct = xc.CatSim()
                ct.resultsName = f"cyl{d}_{int(scatter)}"
                ct.phantom.callback = "Phantom_Analytic"
                ct.phantom.projectorCallback = "C_Projector_Analytic"
                ct.phantom.filename = os.path.join(workdir, "cyl.ppm")
                ct.protocol.viewsPerRotation = ct.protocol.viewCount = 1
                ct.protocol.stopViewId = 0
                ct.protocol.bowtie = []  # sin bowtie: el haz es el de un abanico uniforme
                ct.physics.enableQuantumNoise = ct.physics.enableElectronicNoise = 0
                ct.scanner.detectorRowsPerMod = ct.scanner.detectorRowCount = XC_ROWS
                if scatter:
                    ct.physics.scatterCallback = "Scatter_ConvolutionModel"
                    ct.physics.scatterKernelCallback = ""
                    ct.physics.scatterScaleFactor = 1
                ct.run_all()
                prep[scatter] = xc.rawread(ct.resultsName + ".prep", [1, XC_ROWS, ct.scanner.detectorColCount],
                                           "float")[-1]
            # prep = -ln(I/I0): la dispersa se suma a I, así que SPR = exp(prep_sin - prep_con) - 1
            out[d] = (np.exp(prep[False] - prep[True]) - 1.0)[XC_ROWS // 2]
        # tras la simulación XCIST deja en este campo la ruta completa del fichero; la primera línea es el nº de celdas
        e, phi = np.loadtxt(ct.protocol.spectrumFilename, delimiter=",", skiprows=1, unpack=True)
        al_mm = ct.protocol.flatFilter[1]
        phi = phi * np.exp(-GetMu("Al", e) * 0.1 * al_mm)
        keep = e >= 10.0
        return out, e[keep], phi[keep] / phi[keep].sum()
    finally:
        os.chdir(cwd)


def wavesim_spr(diameter_mm, spec, n_photons=20_000_000, seed=1):
    """SPR en energía (detector integrador de energía) de wavesim para un cilindro de agua infinito en y.

    Misma geometría de abanico curvo que XCIST y mismo espesor de corte (filas de 1 mm a ``sdd``, proyectadas
    al isocentro). Devuelve la posición lateral de cada elemento en el isocentro (mm) y el SPR.
    """
    from wavesim.em import xray_scatter as xs
    from wavesim.phantoms import TISSUE_INDEX, Phantom

    r = diameter_mm / 2e3
    half = r + 0.01
    g = Grid2D.from_size(2 * half, 2 * half, 0.002, x0=-half, z0=-half)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[np.hypot(*g.meshgrid()) <= r] = TISSUE_INDEX["water"]
    geom = fb.FanGeometry.covering(g, XC_SID * 1e-3, XC_SDD * 1e-3, curved=True)
    res = xs.simulate_scatter(Phantom(g, lab), 0.0, spec, geometry=geom, n_photons=n_photons,
                              slice_thickness=XC_ROWS * XC_SID / XC_SDD * 1e-3, seed=seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        spr = np.where(res.e_prim > 0, res.e_scat / res.e_prim, np.nan)
    return XC_SID * np.sin(geom.gamma), spr


def at(x_mm, spr, x_iso):
    """SPR promediado en una ventana de ±5 mm alrededor de ``|x| = x_iso`` (mm en el isocentro)."""
    sel = np.abs(np.abs(x_mm) - x_iso) <= 5.0
    return float(np.nanmean(spr[sel]))


def section_xcist(save):
    try:
        import gecatsim  # noqa: F401
    except ImportError:
        print("XCIST (gecatsim) no está instalado: se salta la sección xcist (ver la cabecera para instalarlo).")
        return
    import tempfile

    from wavesim.em.xray import XraySpectrum

    title("XCIST (gecatsim 1.6.9, CPU) frente al Monte Carlo de dispersión de wavesim")
    with tempfile.TemporaryDirectory() as tmp:
        xspr, e, w = xcist_spr(XC_DIAMETERS, tmp)
    spec = XraySpectrum(e, w)
    cols = np.arange(next(iter(xspr.values())).size)
    x_xc = XC_SID * np.sin((cols - (cols.size - 1) / 2) / XC_SDD)  # mm en el isocentro (columnas de 1 mm)
    data, store = [], {"energies_kev": e, "weights": w}
    for d in XC_DIAMETERS:
        t0 = time.time()
        x_ws, s_ws = wavesim_spr(d, spec)
        for xi in XC_X_ISO:
            if xi >= d / 2 - 15:
                continue  # cerca del borde del cilindro la primaria cae y el SPR diverge
            a, b = at(x_xc, xspr[d], xi), at(x_ws, s_ws, xi)
            data.append((d, f"{xi:.0f}", f"{100 * a:.1f} %", f"{100 * b:.1f} %", f"{b / a:.2f}"))
            store[f"xcist_{d}_{int(xi)}"] = a
        print(f"(diámetro {d} mm: {time.time() - t0:.0f} s de Monte Carlo)")
    print()
    rows(["diámetro (mm)", "x en el isocentro (mm)", "SPR XCIST", "SPR wavesim", "wavesim / XCIST"], data)
    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA / "xcist.npz", **{k: np.asarray(v) for k, v in store.items()})
        print(f"Guardado {DATA / 'xcist.npz'}")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=["astra", "spekpy", "xcist"], default=["astra", "spekpy", "xcist"])
    ap.add_argument("--save", action="store_true", help="guarda las referencias en benchmarks/data/cross_xray/")
    args = ap.parse_args()
    for sec in args.only:
        {"astra": section_astra, "spekpy": section_spekpy, "xcist": section_xcist}[sec](args.save)


if __name__ == "__main__":
    main()
