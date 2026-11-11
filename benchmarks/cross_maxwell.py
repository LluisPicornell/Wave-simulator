"""Comparación cruzada del FDTD de Maxwell de wavesim con gprMax (FDTD de referencia en CPU).

Uso (desde la raíz del repositorio; NumPy, Numba opcional, y gprMax en un entorno aparte):

    python -m venv venv_gprmax && venv_gprmax/bin/pip install gprMax==4.0.1      # rueda precompilada (Cython + OpenMP)
    NUMBA_NUM_THREADS=1 python benchmarks/cross_maxwell.py --gprmax-python venv_gprmax/bin/python
    NUMBA_NUM_THREADS=1 python benchmarks/cross_maxwell.py --case slab --gprmax-python venv_gprmax/bin/python
    python benchmarks/cross_maxwell.py --case cylinder --no-gprmax    # sólo wavesim; gprMax sale de data/cross_maxwell

``--gprmax-python`` (o la variable ``GPRMAX_PYTHON``) es el intérprete del entorno donde está instalado gprMax; sólo se
usa para lanzarlo como subproceso y para leer sus ficheros HDF5 (h5py), así que wavesim no depende de él. Con
``--no-gprmax`` se reutilizan las trazas guardadas en ``benchmarks/data/cross_maxwell/``, que es lo
que usa ``tests/test_cross_maxwell.py``. Sin gprMax no se puede regenerar esa referencia, pero sí repetir las
comparaciones.

Casos (los mismos materiales en los dos códigos: los polos de Debye que ajusta ``materials.dispersive``; se compara el
solver, no el ajuste):

1. ``slab``: onda plana de banda ancha (0.5-6 GHz) a través de una losa piel 2 / grasa 5 / músculo 15 / cortical 6 mm
   en aire. Coeficiente de transmisión complejo ``T(f)`` (campo detrás con la losa dividido por el de una simulación
   sin ella) de ``BroadbandMaxwellTMSolver``, de gprMax y de la matriz de transferencia exacta con los mismos polos.
2. ``cylinder``: cilindro de músculo (radio 20 mm) en aire iluminado por una onda plana a 2.45 GHz (modo TM). Campo
   ``Ez`` a lo largo de la línea central en la dirección de incidencia (``z``) y en la transversal (``x``) con
   ``MaxwellTMSolver`` (banda estrecha), ``BroadbandMaxwellTMSolver``, gprMax y la serie de Mie 2D de ``em.analytic``.
3. ``leg``: pierna del paquete (``cross_section_limb("pierna")``, sin porosidad, en agua) a 2.45 GHz. Mapas de ``|E|`` y
   SAR con wavesim (banda estrecha y ancha) y con gprMax (monitor SAR en el dominio de la frecuencia), diferencia
   relativa (mediana y percentil 95) y SAR pico.
4. ``timing``: tiempo de ejecución de ambos códigos con 1 hilo en el caso de la pierna.

Convenciones (todas verificadas en este script, no supuestas):

- Polarización: wavesim es TM_y (``Ey`` perpendicular al plano ``x-z``, propagación hacia ``+z``); gprMax, modo 2D
  ``TMz`` (``Ez`` perpendicular al plano ``x-y``, propagación hacia ``+y``). Se traduce ``(x, z) -> (x, y)`` y
  ``Ey -> Ez``; el campo magnético en el plano es el mismo salvo un signo que no afecta a ``|E|`` ni al SAR.
- Fase: los dos usan ``e^{+jwt}``. Se obtienen los fasores con ``X(w) = sum x(t) e^{-jwt} dt`` (``Ez`` de gprMax es un
  seno/ricker real). Sólo se comparan cocientes con una simulación de referencia sin objeto, así que se cancelan el
  origen de tiempos, la normalización de la fuente y la diferencia ``dt/2`` entre ``E`` y ``H``.
- Normalización: ``E`` se da por unidad del campo incidente en el centro del objeto sin el objeto (agua con pérdidas en
  la pierna). Wavesim emite ``e0`` en el plano de la fuente y gprMax inyecta el campo con TF/SF (``#plane_wave_axial``);
  la normalización con la referencia hace irrelevante la posición de la fuente.
- Nodos de Yee: wavesim coloca ``Ey`` en el centro de la celda ``[j, i]``; gprMax coloca ``Ez`` en el nodo ``(i, j)``.
  Con ``#box ... n`` (sin suavizado) el material de un ``Ez`` es el de la caja que lo contiene, y la caja incluye el
  nodo del borde superior: una losa de ``n`` celdas ocupa ``n + 1`` nodos si no se resta una celda a ``p2`` (medido
  con una losa sin pérdidas; ver ``docs/validacion/cruzada_maxwell.md``). Se pinta fila a fila (algoritmo del
  pintor) para que cada nodo ``(i, j)`` tenga exactamente la etiqueta de la celda ``[j, i]`` de wavesim: los dos
  resuelven el mismo problema en el espacio de índices. El monitor SAR de gprMax da ``E`` en el centro de la celda
  (media compleja de los cuatro nodos vecinos); por eso se compara también con la misma media sobre wavesim.
- Losa: gprMax en 3D con una celda unidad (paredes PMC en ``x`` y PEC en ``z``), porque ``#symmetry_boundary`` no admite
  el modo 2D; es la misma onda plana 1D que simula wavesim con una columna de 4 celdas.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import em_validation as ev  # noqa: E402
from wavesim.em import analytic  # noqa: E402
from wavesim.em.fdtd_dispersive import BroadbandMaxwellTMSolver  # noqa: E402
from wavesim.em.fdtd_tm import MaxwellTMSolver  # noqa: E402
from wavesim.materials import dielectric, dispersive  # noqa: E402
from wavesim.materials.acoustic import ACOUSTIC  # noqa: E402
from wavesim.phantoms import cross_section_limb  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402

C0 = dielectric.C0
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "data" / "cross_maxwell"
F0 = 2.45e9
SLAB_BAND = (0.5e9, 6.0e9)
FIELD_BAND = (1.0e9, 4.0e9)  # banda del ajuste de Debye en los casos 2D (alrededor de 2.45 GHz)
NPML_G = 15  # celdas de PML de gprMax (el CPML de wavesim usa 20)
AIR, WATER = TISSUE_INDEX["air"], TISSUE_INDEX["water"]
SLAB = ev.SLAB


# --------------------------------------------------------------------------
# gprMax: entradas, ejecución y lectura
# --------------------------------------------------------------------------
# Lo ejecuta el intérprete del entorno de gprMax (h5py está allí, no en el de wavesim): vuelca los receptores y los
# datasets de los monitores SAR a un .npz que sí lee este script
_CONVERT = """
import sys, h5py, numpy as np
f = h5py.File(sys.argv[1], "r")
out = {"dt": np.float64(f.attrs["dt"])}
for r, grp in f.get("rxs", {}).items():
    for c, ds in grp.items():
        out[r + "_" + c] = ds[()].astype(np.float64)
for sid, grp in f.get("sar", {}).items():
    for name, ds in grp.items():
        if isinstance(ds, h5py.Dataset) and ds.dtype.kind in "fiu":
            out["sar_" + sid + "_" + name] = ds[()]
np.savez(sys.argv[2], **out)
"""


class GprMax:
    """Lanzador de gprMax como subproceso (1 hilo por defecto) y lector de sus resultados."""

    def __init__(self, python: str | os.PathLike, workdir: str | os.PathLike | None = None):
        self.python = str(python)
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="cross_maxwell_"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, deck: str, threads: int = 1) -> tuple[dict, dict]:
        """Ejecuta ``deck`` y devuelve ``(datos, tiempos)``; ``tiempos`` = pared total y bucle FDTD de gprMax."""
        inp = self.workdir / f"{name}.in"
        inp.write_text(deck)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        t0 = time.time()
        res = subprocess.run([self.python, "-m", "gprMax", inp.name, "-o", f"{name}.h5", "--hide-progress-bars"],
                             cwd=self.workdir, env=env, capture_output=True, text=True)
        wall = time.time() - t0
        if res.returncode:
            raise RuntimeError(f"gprMax falló en {name}:\n{res.stdout[-1500:]}\n{res.stderr[-2500:]}")
        m = re.search(r"Time taken: ([0-9.eE+-]+) seconds", res.stdout)
        loop = float(m.group(1)) if m else float("nan")
        npz = self.workdir / f"{name}.npz"
        subprocess.run([self.python, "-I", "-c", _CONVERT, str(self.workdir / f"{name}.h5"), str(npz)], check=True)
        with np.load(npz) as d:
            return {k: d[k] for k in d.files}, {"wall": wall, "loop": loop}


def gprmax_materials(model: dispersive.TissueDebyeModel, keys: list[str], densities: bool = False) -> list[str]:
    """Materiales de gprMax con los polos de Debye del ajuste de wavesim (el aire es el vacío de gprMax)."""
    out = []
    for k in keys:
        if k == "air":
            continue
        m = model.tissues[k]
        out.append(f"#material: {float(m.eps_inf):.12g} {float(m.sigma):.12g} 1 0 {k}")
        poles = [(float(d), float(t)) for d, t in zip(m.delta_eps, model.tau) if d > 0]
        if poles:
            cards = " ".join(f"{d:.12g} {t:.12g}" for d, t in poles)
            out.append(f"#add_dispersion_debye: {len(poles)} {cards} {k}")
        if densities:
            out.append(f"#material_density: {ACOUSTIC[k].density:.12g} {k}")
    return out


def _mat_id(label: int) -> str:
    key = next(k for k, v in TISSUE_INDEX.items() if v == label)
    return "free_space" if key == "air" else key


def painter_boxes(lab: np.ndarray, dx: float, tags: bool = False) -> list[str]:
    """Cajas de gprMax que dejan en el nodo ``Ez`` ``(i, j)`` el material de ``lab[j, i]`` (2D, una celda en z).

    Una caja de gprMax asigna material a los nodos de ``Ez`` de sus dos extremos (el borde superior incluido) pero a
    las celdas ``[i0, i1) x [j, j + 1)`` en los mapas de etiquetas del SAR. Se pintan las tiras de cada fila de
    izquierda a derecha y las filas de arriba abajo: el nodo de más que cubre cada caja lo repinta la siguiente, así que
    el resultado final es exacto. Hay que teselar también el fondo (si no, un nodo de borde queda con el tejido).
    """
    nz, nx = lab.shape
    lines = []
    for j in range(nz):
        row = lab[j]
        cut = np.flatnonzero(np.diff(row)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, nx]):
            mid = _mat_id(int(row[a]))
            tag = f" {mid}" if tags else ""
            lines.append(f"#box: {a * dx:.9g} {j * dx:.9g} 0 {b * dx:.9g} {(j + 1) * dx:.9g} {dx:.9g} {mid} n{tag}")
    return lines


def deck_2d(lab: np.ndarray, dx: float, keys: list[str], model, time_window: float, fc: float,
            rx: list[tuple[int, int]], pml: int = NPML_G, y_source: int = 3, sar_freq: float | None = None,
            sar_tags: list[str] | None = None) -> str:
    """Entrada 2D TMz: dominio ``lab`` (con el anillo de PML ya incluido), onda plana TF/SF y receptores ``Ez``.

    ``lab`` cubre todo el dominio, PML incluida (en gprMax la PML está dentro del dominio). La onda plana (``axial``:
    el perfil de material de la línea de propagación se copia de la columna ``x = 1``, que es todo fondo) ocupa
    todo el ancho, como la lámina de corriente de wavesim, y empieza ``y_source`` celdas dentro de la PML.
    """
    ny, nx = lab.shape
    L = [f"#title: cross_maxwell {nx}x{ny}", f"#domain: {nx * dx:.9g} {ny * dx:.9g} {dx:.9g}",
         f"#dx_dy_dz: {dx:.9g} {dx:.9g} {dx:.9g}", f"#time_window: {time_window:.6g}",
         f"#pml_cells: {pml} {pml} 0 {pml} {pml} 0", "#omp_threads: 1"]
    L += gprmax_materials(model, keys, densities=sar_freq is not None)
    L.append(f"#waveform: ricker 1 {fc:.9g} pulse")
    tags = bool(sar_tags)
    if np.all(lab == lab.flat[0]):  # medio homogéneo (las referencias): una sola caja, o nada si es el vacío
        if lab.flat[0] != AIR:
            mid = _mat_id(int(lab.flat[0]))
            L.append(f"#box: 0 0 0 {nx * dx:.9g} {ny * dx:.9g} {dx:.9g} {mid} n" + (f" {mid}" if tags else ""))
    else:
        L += painter_boxes(lab, dx, tags=tags)
    L.append(f"#plane_wave_axial: 0 {(pml + y_source) * dx:.9g} 0 {nx * dx:.9g} {(ny - pml - y_source) * dx:.9g} "
             f"{dx:.9g} 90 y pulse")
    for i, j in rx:
        L.append(f"#rx: {i * dx:.9g} {j * dx:.9g} 0 r Ez")
    if sar_freq is not None:
        L.append(f"#sar: {sar_freq:.9g} {sar_freq:.9g} 1 pulse 1 8 sar1 " + " ".join(sar_tags or []))
    return "\n".join(L) + "\n"


def dft(x: np.ndarray, dt: float, f) -> np.ndarray:
    """Fasor ``sum x(t) e^{-jwt} dt`` con ``t = n dt`` (convenio ``e^{+jwt}``); ``f`` escalar o vector."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    return np.exp(-2j * np.pi * np.outer(f, np.arange(x.shape[-1]) * dt)) @ x * dt


def db(x):
    return 20.0 * np.log10(np.abs(x))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def avg4(e: np.ndarray) -> np.ndarray:
    """Media de los cuatro nodos que rodean cada celda: ``E`` en su centro, como el SAR de gprMax."""
    return 0.25 * (e[:-1, :-1] + e[1:, :-1] + e[:-1, 1:] + e[1:, 1:])


def warmup() -> None:
    """Compila los kernels de Numba antes de medir tiempos (con la caché de disco apenas tarda)."""
    p = ev.column(1e-3, [AIR] * 60)
    MaxwellTMSolver(p, F0).run(n_periods_steady=1, settle_time=1e-10)
    s = BroadbandMaxwellTMSolver(p, FIELD_BAND, model=dispersive.tissue_debye_model(FIELD_BAND))
    s.run(np.array([F0]), t_max=2e-10)


def _load(name: str) -> dict:
    path = DATA / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"falta {path}: ejecuta sin --no-gprmax (con gprMax instalado) para generarlo")
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _save(name: str, **arrays) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / f"{name}.npz", **arrays)


# --------------------------------------------------------------------------
# 1. Losa
# --------------------------------------------------------------------------
def slab_wavesim(dx: float, freqs: np.ndarray, model) -> tuple[np.ndarray, float]:
    """``T(f)`` de ``BroadbandMaxwellTMSolver`` (cociente con y sin losa) y su tiempo de pared."""
    slab, ref, j_obs, layers = ev.slab_phantoms(dx)
    z_src = slab.grid.z0 + (ev.NPML + 3) * dx
    mon = np.zeros(slab.grid.shape, bool)
    mon[j_obs, 2] = True
    out, t0 = [], time.time()
    for p in (slab, ref):
        s = BroadbandMaxwellTMSolver(p, SLAB_BAND, model=model)
        s.set_plane_wave_source(z=z_src)
        out.append(s.run(freqs, monitor=mon).phasor[:, 0])
    return out[0] / out[1], time.time() - t0


def deck_slab(dx: float, cells: list[int], model, with_slab: bool, t_window: float = 8e-9) -> str:
    """Celda unidad 3D de gprMax (PMC en x, PEC en z, PML en y): la misma onda plana 1D que la columna de wavesim."""
    npml, gap, nx = 15, 10, 4
    ny = 4 * gap + sum(cells) + 2 * npml
    w = nx * dx
    L = [f"#domain: {w:.9g} {ny * dx:.9g} {w:.9g}", f"#dx_dy_dz: {dx:.9g} {dx:.9g} {dx:.9g}",
         f"#time_window: {t_window:.6g}", f"#pml_cells: 0 {npml} 0 0 {npml} 0", "#omp_threads: 1",
         "#symmetry_boundary: x0 pmc", "#symmetry_boundary: xmax pmc", "#symmetry_boundary: z0 pec",
         "#symmetry_boundary: zmax pec"]
    L += gprmax_materials(model, [k for k, _ in SLAB])
    L.append("#waveform: ricker 1 2.5e9 pulse")
    y = npml + 2 * gap
    if with_slab:
        for (k, _), n in zip(SLAB, cells):
            # p2 una celda por debajo: la caja incluye el nodo del borde superior (ver la cabecera)
            L.append(f"#box: 0 {y * dx:.9g} 0 {w:.9g} {(y + n - 1) * dx:.9g} {w:.9g} {k} n")
            y += n
    else:
        y += sum(cells)
    L.append(f"#plane_wave_axial: 0 {(npml + 3) * dx:.9g} 0 {w:.9g} {(ny - npml - 3) * dx:.9g} {w:.9g} 90 y pulse")
    L.append(f"#rx: {w / 2:.9g} {(y + gap // 2) * dx:.9g} {w / 2:.9g} r Ez")
    return "\n".join(L) + "\n"


def run_slab(gp: GprMax | None, dx: float, nf: int = 41) -> dict:
    freqs = np.geomspace(*SLAB_BAND, nf)
    model = dispersive.tissue_debye_model(SLAB_BAND)
    _, _, _, layers = ev.slab_phantoms(dx)
    cells = [int(round(d / dx)) for _, d in layers]
    t_ws, tm_ws = slab_wavesim(dx, freqs, model)
    exact = analytic.insertion_transmission(
        layers, freqs, permittivity=lambda k, ff: model.tissues[k].complex_permittivity(ff))
    gabriel = analytic.insertion_transmission(layers, freqs)
    res = dict(freqs=freqs, dx=dx, t_ws=t_ws, exact=exact, gabriel=gabriel, tm_ws=tm_ws)
    if gp is None:
        old = _load(f"slab_dx{dx * 1e3:g}mm")
        res.update(t_gp=old["t_gp"], tm_gp=old["tm_gp"], tm_gp_wall=old["tm_gp_wall"])
        return res
    d1, tm1 = gp.run("slab_obj", deck_slab(dx, cells, model, True))
    d0, tm0 = gp.run("slab_ref", deck_slab(dx, cells, model, False))
    dt = float(d1["dt"])
    res.update(t_gp=dft(d1["rx1_Ez"], dt, freqs) / dft(d0["rx1_Ez"], dt, freqs),
               tm_gp=tm1["loop"] + tm0["loop"], tm_gp_wall=tm1["wall"] + tm0["wall"])
    _save(f"slab_dx{dx * 1e3:g}mm", **{k: v for k, v in res.items()})
    return res


def report_slab(r: dict) -> None:
    f, ex, w, g = r["freqs"], r["exact"], r["t_ws"], r["t_gp"]
    print(f"\n### Losa piel 2 / grasa 5 / músculo 15 / cortical 6 mm, dx = {r['dx'] * 1e3:g} mm, "
          f"{f[0] / 1e9:g}-{f[-1] / 1e9:g} GHz (polos de Debye comunes)")
    print("| f (GHz) | |T| exacto | wavesim: dB / fase vs exacto | gprMax: dB / fase vs exacto | gprMax - wavesim |")
    print("|---|---|---|---|---|")
    for k in range(0, f.size, 5):
        cell = lambda q: f"{db(q):+.3f} dB / {np.degrees(np.angle(q)):+.2f}°"  # noqa: E731
        print(f"| {f[k] / 1e9:.3g} | {db(ex[k]):.2f} dB | {cell(w[k] / ex[k])} | {cell(g[k] / ex[k])} "
              f"| {cell(g[k] / w[k])} |")
    mx = lambda a: (np.abs(db(a)).max(), np.abs(np.degrees(np.angle(a))).max())  # noqa: E731
    print(f"máx |error| wavesim vs exacto: {mx(w / ex)[0]:.3f} dB, {mx(w / ex)[1]:.2f}°; "
          f"gprMax vs exacto: {mx(g / ex)[0]:.3f} dB, {mx(g / ex)[1]:.2f}°; "
          f"gprMax vs wavesim: {mx(g / w)[0]:.3f} dB, {mx(g / w)[1]:.2f}°. "
          f"Debye frente a Gabriel (exacto): {mx(ex / r['gabriel'])[0]:.3f} dB, {mx(ex / r['gabriel'])[1]:.2f}°")
    print(f"tiempos (1 hilo, losa + referencia): wavesim {r['tm_ws']:.1f} s, gprMax {r['tm_gp']:.1f} s de bucle "
          f"({r['tm_gp_wall']:.1f} s con el arranque)")


# --------------------------------------------------------------------------
# 2. Cilindro de músculo en aire (Mie)
# --------------------------------------------------------------------------
def cylinder_wavesim(dx: float, model, a: float = 0.02, tissue: str = "muscle", f0: float = F0) -> dict:
    """``Ey / E_inc`` (``(nz, nx)``) de wavesim en banda estrecha y ancha, y la geometría de la malla."""
    g, half = ev.cylinder_grid(dx, a, f0)
    labels = ev.disk_labels(g, a, tissue)
    eps_c = complex(model.tissues[tissue].complex_permittivity(f0))
    eps_map = np.where(labels == TISSUE_INDEX[tissue], eps_c, 1.0 + 0j)
    phantom = Phantom(g, labels)
    z_src = g.z0 + (ev.NPML + 2) * dx
    centre = g.index_of(0.0, 0.0)
    settle = 40 / f0 + 3.0 * 2 * g.depth * abs(np.sqrt(eps_c)) / C0  # régimen permanente bien asentado
    s = MaxwellTMSolver(phantom, f0, eps_map=eps_map)
    s.set_plane_wave_source(z=z_src)
    r = s.run(n_periods_steady=8, settle_time=settle)
    ref = MaxwellTMSolver(ev.column(dx, [AIR] * g.nz, g.z0), f0)
    ref.set_plane_wave_source(z=z_src)
    e_nb = r.phasor / ref.run(n_periods_steady=8, settle_time=settle).phasor[centre[0], 2]
    b = BroadbandMaxwellTMSolver(phantom, FIELD_BAND, model=model)
    b.set_plane_wave_source(z=z_src)
    rb = b.run(np.array([f0]))
    mon = np.zeros((g.nz, 4), bool)
    mon[centre[0], 2] = True
    rr = BroadbandMaxwellTMSolver(ev.column(dx, [AIR] * g.nz, g.z0), FIELD_BAND, model=model)
    rr.set_plane_wave_source(z=z_src)
    e_bb = rb.phasor[0] / rr.run(np.array([f0]), monitor=mon).phasor[0, 0]
    return dict(g=g, half=half, labels=labels, eps=eps_c, e_nb=e_nb, e_bb=e_bb, centre=centre,
                tm_nb=r.elapsed_s, tm_bb=rb.elapsed_s)


def cylinder_gprmax(gp: GprMax, w: dict, dx: float, model, tissue: str = "muscle", f0: float = F0,
                    t_window: float = 10e-9) -> dict:
    """Líneas ``Ez / E_inc`` por el centro del cilindro (en ``z`` y en ``x``) de gprMax."""
    n_half = int(np.ceil(w["half"] / dx))
    pad = NPML_G + 4  # el TF/SF debe contener todo el dominio útil de wavesim
    lab = np.pad(w["labels"][ev.NPML:-ev.NPML, ev.NPML:-ev.NPML], pad, constant_values=AIR)
    ic, jc = n_half + pad, n_half + 6 + pad  # nodo del centro del cilindro
    ks = np.arange(-n_half, n_half + 1)
    rx = [(ic, jc + k) for k in ks] + [(ic + k, jc) for k in ks]
    d, tm = gp.run("cyl_obj", deck_2d(lab, dx, [tissue], model, t_window, 2.5e9, rx))
    r, _ = gp.run("cyl_ref", deck_2d(np.full(lab.shape, AIR, np.int16), dx, [tissue], model, t_window, 2.5e9,
                                     [(ic, jc)]))
    dt = float(d["dt"])
    e_inc = dft(r["rx1_Ez"], dt, f0)[0]
    n = ks.size
    line = lambda off: np.array([dft(d[f"rx{off + k + 1}_Ez"], dt, f0)[0] for k in range(n)]) / e_inc  # noqa: E731
    return dict(gp_z=line(0), gp_x=line(n), tm_gp=tm["loop"], tm_gp_wall=tm["wall"],
                tail=float(np.abs(d["rx1_Ez"][-200:]).max() / np.abs(d["rx1_Ez"]).max()))


def run_cylinder(gp: GprMax | None, dx: float) -> dict:
    a, f0, tissue = 0.02, F0, "muscle"
    model = dispersive.tissue_debye_model(FIELD_BAND)
    w = cylinder_wavesim(dx, model, a, tissue, f0)
    n_half = int(np.ceil(w["half"] / dx))
    ks = np.arange(-n_half, n_half + 1)
    zc, xc = w["centre"]
    res = dict(ks=ks, dx=dx, a=a, f0=f0, eps=w["eps"], tm_nb=w["tm_nb"], tm_bb=w["tm_bb"],
               nb_z=w["e_nb"][zc + ks, xc], nb_x=w["e_nb"][zc, xc + ks],
               bb_z=w["e_bb"][zc + ks, xc], bb_x=w["e_bb"][zc, xc + ks],
               mie_z=analytic.mie_cylinder_tm(np.zeros(ks.size), ks * dx, a, w["eps"], f0),
               mie_x=analytic.mie_cylinder_tm(ks * dx, np.zeros(ks.size), a, w["eps"], f0))
    if gp is None:
        old = _load(f"cylinder_dx{dx * 1e3:g}mm")
        res.update({k: old[k] for k in ("gp_z", "gp_x", "tm_gp", "tm_gp_wall")})
        return res
    res.update(cylinder_gprmax(gp, w, dx, model, tissue, f0))
    _save(f"cylinder_dx{dx * 1e3:g}mm", **res)
    return res


def report_cylinder(r: dict) -> None:
    ks, dx, a = r["ks"], r["dx"], r["a"]
    inside = np.abs(ks * dx) <= a
    print(f"\n### Cilindro de músculo (radio {a * 1e3:g} mm) en aire a {r['f0'] / 1e9:g} GHz, dx = {dx * 1e3:g} mm "
          f"(eps = {r['eps']:.2f}); error L2 relativo de Ez a lo largo de la línea central")
    print("| línea | par | dentro | fuera |")
    print("|---|---|---|---|")
    for ln, name in (("z", "incidencia"), ("x", "transversal")):
        c = {"wavesim banda estrecha": r[f"nb_{ln}"], "wavesim banda ancha": r[f"bb_{ln}"], "gprMax": r[f"gp_{ln}"]}
        mie = r[f"mie_{ln}"]
        for k, v in c.items():
            print(f"| {ln} ({name}) | {k} frente a Mie | {rel_l2(v[inside], mie[inside]) * 100:.2f} % "
                  f"| {rel_l2(v[~inside], mie[~inside]) * 100:.2f} % |")
        for k in ("wavesim banda estrecha", "wavesim banda ancha"):
            print(f"| {ln} ({name}) | gprMax frente a {k} | {rel_l2(c['gprMax'][inside], c[k][inside]) * 100:.2f} % "
                  f"| {rel_l2(c['gprMax'][~inside], c[k][~inside]) * 100:.2f} % |")
    print(f"tiempos (1 hilo, objeto): wavesim estrecha {r['tm_nb']:.1f} s, ancha {r['tm_bb']:.1f} s, "
          f"gprMax {r['tm_gp']:.1f} s de bucle ({r['tm_gp_wall']:.1f} s con el arranque)")


# --------------------------------------------------------------------------
# 3. Pierna (SAR)
# --------------------------------------------------------------------------
def leg_wavesim(dx: float, model, f0: float = F0, margin: float = 30e-3, broadband: bool = True) -> dict:
    """``|E|`` y SAR (por V/m incidente en el centro, sin la pierna) de wavesim; nativos y colocados como gprMax."""
    p = cross_section_limb("pierna", dx=dx, margin=margin)
    g = p.grid
    eps_map = dispersive.debye_maps(p, model).complex_permittivity(f0)
    eps_w = complex(model.tissues["water"].complex_permittivity(f0))
    centre = g.index_of(0.0, 0.0)
    col = ev.column(dx, [WATER] * g.nz, g.z0)
    z_src = g.z0 + (ev.NPML + 2) * dx
    s = MaxwellTMSolver(p, f0, eps_map=eps_map)
    s.set_plane_wave_source(z=z_src)
    r = s.run()
    ref = MaxwellTMSolver(col, f0, eps_map=np.full(col.grid.shape, eps_w))
    ref.set_plane_wave_source(z=z_src)
    e_ref = ref.run().phasor[:, 2]
    runs = [("nb", r.phasor, e_ref)]
    tm_bb = float("nan")
    if broadband:
        b = BroadbandMaxwellTMSolver(p, FIELD_BAND, model=model)
        b.set_plane_wave_source(z=z_src)
        rb = b.run(np.array([f0]))
        mon = np.zeros(col.grid.shape, bool)
        mon[:, 2] = True
        rr = BroadbandMaxwellTMSolver(col, FIELD_BAND, model=model)
        rr.set_plane_wave_source(z=z_src)
        runs.append(("bb", rb.phasor[0], rr.run(np.array([f0]), monitor=mon).phasor[0]))
        tm_bb = rb.elapsed_s
    sigma = -eps_map.imag * 2 * np.pi * f0 * dielectric.EPS0
    rho = p.property_map(lambda k: ACOUSTIC[k].density, physics="acoustic")
    out = dict(g=g, labels=p.labels, centre=centre, sigma=sigma, rho=rho, tm_nb=r.elapsed_s, tm_bb=tm_bb)
    for name, e, er in runs:
        j = centre[0]
        nat = e / er[j]
        col_e = avg4(e) / (0.5 * (er[j] + er[j + 1]))  # colocado en el centro de la celda, como gprMax
        out[f"E_{name}"] = np.abs(nat)
        out[f"SAR_{name}"] = sigma * np.abs(nat) ** 2 / (2 * rho)
        full = np.full(g.shape, np.nan)  # la última fila y columna no tienen celda colocada
        out[f"Ec_{name}"] = full.copy()
        out[f"Ec_{name}"][:-1, :-1] = np.abs(col_e)
        out[f"SARc_{name}"] = full.copy()
        out[f"SARc_{name}"][:-1, :-1] = sigma[:-1, :-1] * np.abs(col_e) ** 2 / (2 * rho[:-1, :-1])
    return out


def leg_gprmax(gp: GprMax, w: dict, dx: float, model, f0: float = F0, t_window: float = 14e-9) -> dict:
    """``|E|`` y SAR de gprMax (monitor SAR) por V/m incidente en el centro (celda del centro, sin la pierna)."""
    pad = NPML_G + 4
    labels = w["labels"]
    inner = labels[ev.NPML:-ev.NPML, ev.NPML:-ev.NPML]
    lab = np.pad(inner, pad, constant_values=WATER)
    keys = [k for k in TISSUE_INDEX if TISSUE_INDEX[k] in np.unique(lab)]
    zc, xc = w["centre"]
    ic, jc = xc - ev.NPML + pad, zc - ev.NPML + pad
    d, tm = gp.run("leg_obj", deck_2d(lab, dx, keys, model, t_window, 2.5e9, [(ic, jc)], sar_freq=f0, sar_tags=keys))
    r, _ = gp.run("leg_ref", deck_2d(np.full(lab.shape, WATER, np.int16), dx, ["water"], model, t_window, 2.5e9,
                                     [(ic, jc)], sar_freq=f0, sar_tags=["water"]))
    sig_t = {TISSUE_INDEX[k]: float(model.tissues[k].conductivity(f0)) for k in keys}
    ci = r["sar_sar1_cell_indices"]
    k_ref = int(np.flatnonzero((ci[:, 0] == ic) & (ci[:, 1] == jc))[0])
    e2_ref = 2 * float(r["sar_sar1_absorbed_power_density"][0, k_ref]) / sig_t[WATER]
    ci = d["sar_sar1_cell_indices"]
    shape = inner.shape
    ii, jj = ci[:, 0] - pad, ci[:, 1] - pad
    ok = (ii >= 0) & (ii < shape[1]) & (jj >= 0) & (jj < shape[0])
    apd = np.full(shape, np.nan)
    sar = np.full(shape, np.nan)
    apd[jj[ok], ii[ok]] = d["sar_sar1_absorbed_power_density"][0][ok]
    sar[jj[ok], ii[ok]] = d["sar_sar1_sar"][0][ok]
    sig = np.vectorize(sig_t.get)(inner)
    return dict(Ec_gp=np.sqrt(2 * apd / sig / e2_ref), SARc_gp=sar / e2_ref, tm_gp=tm["loop"], tm_gp_wall=tm["wall"],
                tail=float(np.abs(d["rx1_Ez"][-200:]).max() / np.abs(d["rx1_Ez"]).max()))


def diff_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Mediana y percentil 95 de ``|a - b| / |b|`` en las celdas de ``mask`` donde ``b`` supera el 1 % de su máximo."""
    sel = mask & (b > 0.01 * np.nanmax(np.where(mask, b, np.nan)))
    rel = np.abs(a[sel] - b[sel]) / b[sel]
    return float(np.median(rel)), float(np.percentile(rel, 95))


def run_leg(gp: GprMax | None, dx: float, broadband: bool = True) -> dict:
    model = dispersive.tissue_debye_model(FIELD_BAND)
    w = leg_wavesim(dx, model, broadband=broadband)
    sl = (slice(ev.NPML, -ev.NPML), slice(ev.NPML, -ev.NPML))
    inner = w["labels"][sl]
    res = dict(dx=dx, labels=inner.astype(np.int8), tm_nb=w["tm_nb"], tm_bb=w["tm_bb"])
    for k in [k for k in w if k.startswith(("E_", "Ec_", "SAR_", "SARc_"))]:
        res[k] = w[k][sl].astype(np.float32)
    if gp is None:
        old = _load(f"leg_dx{dx * 1e3:g}mm")
        res.update({k: old[k] for k in ("Ec_gp", "SARc_gp", "tm_gp", "tm_gp_wall")})
        return res
    res.update(leg_gprmax(gp, w, dx, model))
    res = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v)
           for k, v in res.items()}
    # sólo se guardan los mapas de gprMax (los de wavesim se recalculan y ocuparían varios MB)
    keep = ("dx", "labels", "Ec_gp", "SARc_gp", "tm_gp", "tm_gp_wall", "tail")
    _save(f"leg_dx{dx * 1e3:g}mm", **{k: res[k] for k in keep})
    return res


def leg_table(r: dict) -> list[tuple]:
    body = r["labels"] != WATER
    allc = np.ones_like(body)
    rows = []
    for tag, m in (("pierna (sin agua)", body), ("todo el dominio", allc)):
        for code in [c for c in ("nb", "bb") if f"E_{c}" in r]:
            for qty, key in (("|E|", "Ec"), ("SAR", "SARc")):
                a, b = (r[f"{key}_{c}"].astype(float)[:-1, :-1] for c in (code, "gp"))
                med, p95 = diff_stats(a, b, m[:-1, :-1])
                rows.append((tag, code, qty, med, p95))
    return rows


def report_leg(r: dict) -> None:
    dx = r["dx"]
    body = r["labels"] != WATER
    print(f"\n### Pierna (``cross_section_limb('pierna')``, en agua) a {F0 / 1e9:g} GHz, dx = {dx * 1e3:g} mm; "
          "diferencia relativa de gprMax frente a wavesim, mapas colocados en el centro de la celda")
    print("| región | wavesim | magnitud | mediana | percentil 95 |")
    print("|---|---|---|---|---|")
    names = {"nb": "banda estrecha", "bb": "banda ancha"}
    for tag, code, qty, med, p95 in leg_table(r):
        print(f"| {tag} | {names[code]} | {qty} | {med * 100:.2f} % | {p95 * 100:.2f} % |")
    print("\nSAR pico en la pierna (W/kg por (V/m)^2 incidente en el centro), y su celda (fila, columna):")
    for key, name in (("SARc_gp", "gprMax"), ("SARc_nb", "wavesim estrecha"), ("SARc_bb", "wavesim ancha"),
                      ("SAR_nb", "wavesim estrecha (nativo, sin colocar)")):
        if key not in r:
            continue
        a = np.where(body, r[key].astype(float), np.nan)
        a[-1, :] = np.nan
        a[:, -1] = np.nan
        j, i = np.unravel_index(np.nanargmax(a), a.shape)
        print(f"- {name}: {np.nanmax(a):.4g} en ({j}, {i})")
    pk = lambda key: np.nanmax(np.where(body[:-1, :-1], r[key].astype(float)[:-1, :-1], np.nan))  # noqa: E731
    print(f"- cociente de picos wavesim estrecha / gprMax: {pk('SARc_nb') / pk('SARc_gp'):.4f}"
          + (f"; ancha / gprMax: {pk('SARc_bb') / pk('SARc_gp'):.4f}" if "SARc_bb" in r else ""))
    nat = diff_stats(r["E_nb"].astype(float)[:-1, :-1], r["Ec_gp"].astype(float)[:-1, :-1], body[:-1, :-1])
    print(f"Sin colocar (|E| nativo de wavesim en el nodo frente al de gprMax en el centro de la celda, desfase de "
          f"media celda): mediana {nat[0] * 100:.2f} %, p95 {nat[1] * 100:.2f} %")
    print(f"tiempos (1 hilo): wavesim estrecha {r['tm_nb']:.1f} s, ancha {r['tm_bb']:.1f} s, "
          f"gprMax {r['tm_gp']:.1f} s de bucle ({r['tm_gp_wall']:.1f} s con el arranque)")


CASES = {"slab": (run_slab, report_slab, (0.5e-3, 0.25e-3)),
         "cylinder": (run_cylinder, report_cylinder, (1e-3, 0.5e-3)),
         "leg": (run_leg, report_leg, (1e-3, 0.5e-3))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", choices=list(CASES) + ["all"], default="all")
    ap.add_argument("--gprmax-python", default=os.environ.get("GPRMAX_PYTHON"),
                    help="intérprete del entorno con gprMax (o GPRMAX_PYTHON)")
    ap.add_argument("--no-gprmax", action="store_true", help="reutiliza las trazas guardadas en vez de ejecutar gprMax")
    ap.add_argument("--workdir", default=None, help="directorio de trabajo de gprMax (por defecto, uno temporal)")
    ap.add_argument("--quick", action="store_true", help="sólo la malla más gruesa de cada caso")
    args = ap.parse_args()
    gp = None
    if not args.no_gprmax:
        if not args.gprmax_python:
            raise SystemExit("indica --gprmax-python (o GPRMAX_PYTHON), o usa --no-gprmax")
        gp = GprMax(args.gprmax_python, args.workdir)
    warnings.filterwarnings("ignore", message="Sólo")
    warmup()
    t0 = time.time()
    for name, (run, report, dxs) in CASES.items():
        if args.case in (name, "all"):
            for dx in dxs[:1] if args.quick else dxs:
                report(run(gp, dx))
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
