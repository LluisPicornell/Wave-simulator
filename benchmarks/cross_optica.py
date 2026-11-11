"""Comparación cruzada de la luz en tejido: Monte Carlo de wavesim y difusión frente a MCML (Wang & Jacques).

Uso (desde la raíz del repositorio):

    # 1) Sólo comparar con las trazas guardadas en benchmarks/data/cross_optica/ (no hace falta MCML):
    NUMBA_NUM_THREADS=1 PYTHONPATH=$(pwd) python benchmarks/cross_optica.py --photons 500000 --seeds 8  # ~35 min
    NUMBA_NUM_THREADS=1 PYTHONPATH=$(pwd) python benchmarks/cross_optica.py --quick    # 4 x 1.25e5 fotones, ~8 min
    NUMBA_NUM_THREADS=1 PYTHONPATH=$(pwd) python benchmarks/cross_optica.py --cases a_semi c_slab

    # 2) Regenerar las referencias ejecutando MCML (hace falta compilarlo, ver abajo):
    python benchmarks/cross_optica.py --regen --mcml-bin /ruta/a/mcml --jobs 2        # 4 x 4e6 fotones por caso

    # 3) Regenerar las de MCX/MCXCL en un venv aparte (sin CUDA: OpenCL de CPU con pocl):
    python -m venv venv_mcx && venv_mcx/bin/pip install pmcxcl pyopencl pocl-binary-distribution
    # pmcxcl trae su propio cargador OpenCL: se le apunta a la librería pocl del wheel de pyopencl con un .icd propio
    mkdir icd && echo "$PWD/venv_mcx/lib/python3.13/site-packages/pyopencl/.libs/libpocl-21062cec.so" > icd/pocl.icd
    OCL_ICD_VENDORS=$PWD/icd PYTHONPATH=$(pwd) venv_mcx/bin/python benchmarks/cross_optica.py --regen-mcx

MCML se compila desde un espejo de GitHub del código original de Wang y Jacques (v1.2, 1993):

    git clone https://github.com/bipy/MCML.git && cd MCML        # commit bd8fcb2
    patch -p0 < <repo>/benchmarks/data/cross_optica/mcml.patch   # ver abajo
    gcc -O2 -w -o mcml mcmlmain.c mcmlgo.c mcmlio.c mcmlnr.c -lm

El parche (``benchmarks/data/cross_optica/mcml.patch``) sólo toca dos cosas ajenas a la física: (i) ``FindDataLine``
devolvía un puntero a un array local (segfault con gcc moderno): ``static``; (ii) la semilla venía de ``time(NULL)``
(dos runs lanzados en el mismo segundo repiten semilla): ahora se lee de la variable de entorno ``MCML_SEED``.
MCML no es una dependencia del repositorio: sólo ``--regen`` lo usa, y los tests/benchmarks leen las trazas guardadas.

GEOMETRÍA Y CONVERSIÓN 3D -> 2D EXTRUIDO. wavesim discretiza ``x`` y ``z`` y supone el medio invariante en ``y``;
un "haz fino" es una LÍNEA infinita en ``y`` de potencia por metro, y la reflectancia por columna ``R(x)`` es el
flujo saliente por unidad de longitud de ``x`` y por unidad de longitud de línea. MCML es 3D con simetría
cilíndrica: haz fino en el eje, ``R(rho)`` por unidad de área. Como el medio es invariante en ``y`` y el transporte
es lineal, la línea es la superposición de haces finos en cada ``y`` y

    R_2D(x) = integral R_3D( sqrt(x² + y²) ) dy        (misma fórmula para T, fluencia y absorción)

(los fotones de wavesim también se mueven en 3D: tienen ``uy`` y se dispersan en 3D, así que no hay aproximación
2D en el transporte, sólo en el muestreo). La integral se hace EXACTA para el ``R(rho)`` constante a trozos de
MCML (``radial_a_linea``) y promediada sobre el ancho de la columna de wavesim. La fluencia integrada en el plano
perpendicular a ``z`` (``A_z / mu_a``) es idéntica en 2D y 3D sin conversión.

DIFERENCIAS DE CONVENCIÓN (ver docs/validacion/cruzada_optica.md): (1) MCML deposita ``mu_a/mu_t`` en cada colisión
y wavesim usa absorción continua + estimador de longitud de traza: mismo valor esperado; (2) las exitancias de MCML
por cara incluyen fotones fuera de ``nr·dr`` en el último anillo (el rango es mayor que la zona comparada); (3) la
reflexión especular de la entrada se da aparte en ambos; (4) MCML usa ``ran3`` en float (24 bits), wavesim PCG64;
(5) el contorno lateral de wavesim es el borde de la malla (7 cm), MCML es lateralmente infinito: los fotones
laterales de wavesim se dan como columna aparte y son ~1e-6 de lo incidente.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "cross_optica"
LAMBDA_NM = 800.0
DX = 1e-4  # celda de wavesim (y bin z de MCML): 0.1 mm, algo menor que el recorrido libre medio (0.117 mm)
HALF_NX = 350  # media anchura en celdas: la malla es de 701 columnas (7.01 cm) con la fuente en la central
DR_CM, NR = 0.005, 800  # MCML: bins radiales de 50 µm hasta 4 cm (más que la media anchura de wavesim, 3.5 cm)
X_PHI_MM = (0.0, 2.0, 4.0, 8.0, 12.0, 20.0)  # desplazamientos laterales donde se guarda la fluencia 2D
X_BINS_MM = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0)
Z_BINS_MM = (0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 25.0, 50.0)
TOT_NAMES = ("specular", "reflectance", "absorbed", "transmittance")

# nombre -> (capas [(tejido, espesor m)], índice del tejido, descripción)
CASES: dict[str, tuple[list[tuple[str, float]], float, str]] = {
    "a_semi": ([("muscle", 5e-2)], 1.0, "músculo semi-infinito (5 cm), n = 1 dentro y fuera"),
    "a_slab": ([("muscle", 1e-2)], 1.0, "losa de músculo de 1 cm, n = 1"),
    "b_semi": ([("fat", 4e-3), ("muscle", 4.6e-2)], 1.0, "grasa 4 mm sobre músculo semi-infinito, n = 1"),
    "b_slab": ([("fat", 3e-3), ("muscle", 7e-3)], 1.0, "grasa 3 mm + músculo 7 mm (1 cm), n = 1"),
    "c_semi": ([("muscle", 5e-2)], 1.4, "músculo semi-infinito, n = 1.4 frente a aire"),
    "c_slab": ([("muscle", 1e-2)], 1.4, "losa de músculo de 1 cm, n = 1.4 frente a aire"),
}


def tissue_props(key: str, lam_nm: float = LAMBDA_NM) -> tuple[float, float, float]:
    """``(mu_a, mu_s, g)`` en SI (1/m) de ``materials.optical`` a ``lam_nm``."""
    from wavesim.materials import optical
    g = optical.OPTICAL[key].anisotropy
    return (float(optical.mu_a(key, lam_nm)), float(optical.mu_s_reduced(key, lam_nm)) / (1.0 - g), g)


def layer_maps(layers, dx: float = DX):
    """Mapas 1D por fila ``z``: ``(mua, mus, g, nombre de capa)``."""
    mua, mus, gg = [], [], []
    for key, d in layers:
        a, s, g = tissue_props(key)
        n = int(round(d / dx))
        mua += [a] * n
        mus += [s] * n
        gg += [g] * n
    return np.array(mua), np.array(mus), np.array(gg)


def mean_sem(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media y error estándar de la media entre réplicas independientes (eje 0)."""
    a = np.asarray(a, float)
    k = a.shape[0]
    return a.mean(0), (a.std(0, ddof=1) / np.sqrt(k) if k > 1 else np.zeros(a.shape[1:]))


# ----------------------------------------------------------------------------------------------------------
# Conversión 3D (simetría cilíndrica) -> 2D extruido

def _area_cuarto(x: np.ndarray, radio: np.ndarray) -> np.ndarray:
    """``F(x, R) = integral_0^x sqrt(R² - t²) dt``, impar y saturada en ``±pi R²/4``; ``x`` (n,1), ``R`` (m,)."""
    u = np.minimum(np.abs(x), radio)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(radio > 0, u / np.where(radio > 0, radio, 1.0), 0.0)
    return np.sign(x) * 0.5 * (u * np.sqrt(np.maximum(radio**2 - u**2, 0.0)) + radio**2 * np.arcsin(ratio))


def pesos_radial_a_linea(dr: float, nr: int, x_centros: np.ndarray, dx: float) -> np.ndarray:
    """Matriz ``W (n_x, nr)`` tal que ``R_2D[celda] = W @ R_3D[anillo]`` para ``R_3D`` constante en cada anillo.

    ``R_2D`` es la integral en ``y`` de ``R_3D(sqrt(x² + y²))`` promediada sobre el ancho ``dx`` de la columna:
    ``(2/dx) * área({x en la columna, y >= 0, anillo k})`` y el área sale en forma cerrada.
    """
    bordes = np.arange(nr + 1) * dr
    xa, xb = (x_centros - dx / 2)[:, None], (x_centros + dx / 2)[:, None]
    area = _area_cuarto(xb, bordes[None, :]) - _area_cuarto(xa, bordes[None, :])
    return 2.0 * np.diff(area, axis=1) / dx


def radial_a_linea(valores: np.ndarray, dr: float, x_centros: np.ndarray, dx: float) -> np.ndarray:
    """``R_2D(x)`` por columna a partir de ``valores`` por anillo (eje 0 de ``valores``)."""
    return pesos_radial_a_linea(dr, valores.shape[0], x_centros, dx) @ valores


# ----------------------------------------------------------------------------------------------------------
# MCML

def escribir_mci(path: Path, layers, n_tissue: float, n_photons: int, dz_cm: float = 0.01) -> None:
    thick = sum(d for _, d in layers) * 100.0
    lines = ["1.0", "1", "salida.mco A", str(n_photons), f"{dz_cm} {DR_CM}", f"{int(round(thick / dz_cm))} {NR} 1",
             str(len(layers)), "1.0"]
    for key, d in layers:
        a, s, g = tissue_props(key)
        lines.append(f"{n_tissue} {a / 100:.6f} {s / 100:.6f} {g} {d * 100:.6f}")
    lines.append("1.0")
    path.write_text("\n".join(lines) + "\n")


def leer_mco(path: Path, nz: int) -> dict:
    """Lee el fichero de salida de MCML: RAT, A_z, Rd_r, Tt_r y A_rz (A_rz[ir, iz])."""
    txt = path.read_text().splitlines()
    out: dict = {}
    for k, line in enumerate(txt):
        tag = line.split("#")[0].split()
        if not tag or tag[0] not in ("RAT", "A_z", "Rd_r", "Tt_r", "A_rz"):
            continue
        vals = []
        j = k + 1
        while j < len(txt) and (not txt[j].strip() or not txt[j].lstrip()[0].isalpha()):
            if not txt[j].lstrip().startswith("#"):
                vals += [float(v) for v in txt[j].split("#")[0].split()]
            j += 1
        out[tag[0]] = np.array(vals)
    out["A_rz"] = out["A_rz"].reshape(NR, nz)
    return out


def ejecutar_mcml(mcml_bin: str, layers, n_tissue: float, n_photons: int, seed: int) -> dict:
    """Una réplica de MCML; devuelve los totales y los perfiles ya convertidos a 2D extruido (SI)."""
    nz = int(round(sum(d for _, d in layers) / DX))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        escribir_mci(tmp / "e.mci", layers, n_tissue, n_photons)
        r = subprocess.run([mcml_bin, "e.mci"], cwd=tmp, env={**os.environ, "MCML_SEED": str(seed)},
                           capture_output=True, text=True, check=True)
        if not (tmp / "salida.mco").exists():
            raise RuntimeError(f"MCML no escribió la salida: {r.stdout[-300:]} {r.stderr[-300:]}")
        o = leer_mco(tmp / "salida.mco", nz)
    mua_z = layer_maps(layers)[0] / 100.0  # 1/cm
    xs = np.arange(HALF_NX + 1) * DX
    w = pesos_radial_a_linea(DR_CM / 100.0, NR, xs, DX)
    # unidades: Rd_r [1/cm²] -> [1/m²]; fluencia A_rz/mu_a [1/cm²] -> [1/m²]; luego la integral en y da 1/m
    rd2 = w @ (o["Rd_r"] * 1e4)
    tt2 = w @ (o["Tt_r"] * 1e4)
    fluencia3d = o["A_rz"] / mua_z[None, :] * 1e4
    x_phi = np.rint(np.array(X_PHI_MM) * 1e-3 / DX).astype(int)
    phi2d = (w @ fluencia3d)[x_phi]  # (n_x_phi, nz) [1/m]
    phi_int = o["A_z"] / mua_z  # A_z / mu_a [1/cm / (1/cm)]: adimensional, = integral de Phi sobre el plano z
    rat = o["RAT"]  # Rsp, Rd, A, Tt
    return {"tot": rat, "R2d": rd2, "T2d": tt2, "phi_int": phi_int, "phi2d": phi2d}


def _guardar(path: Path, res: list[dict], n_photons: int, **meta) -> None:
    """Réplicas (eje 0) en float32 salvo los totales (float64); las trazas pequeñas viajan con el repositorio."""
    datos = {k: np.array([r[k] for r in res], dtype=np.float64 if k == "tot" else np.float32) for k in res[0]}
    np.savez_compressed(path, **datos, n_photons=n_photons, dx=DX, **meta)


def regenerar(mcml_bin: str, names, n_photons: int, n_seeds: int, jobs: int) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name in names:
        layers, n_t, desc = CASES[name]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(jobs) as ex:
            base = 100 * list(CASES).index(name) + 1
            res = list(ex.map(lambda s, a=(layers, n_t, base): ejecutar_mcml(mcml_bin, a[0], a[1], n_photons, a[2] + s),
                              range(n_seeds)))
        _guardar(DATA / f"mcml_{name}.npz", res, n_photons, dr_cm=DR_CM, x_phi_mm=np.array(X_PHI_MM),
                 mcml_commit="bd8fcb2 (bipy/MCML v1.2)")
        print(f"{name}: {n_seeds} x {n_photons} fotones en {time.perf_counter() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------------------
# MCX / MCXCL (pmcxcl con un dispositivo OpenCL de CPU, pocl)

# voxeles en y con contorno CÍCLICO: la fuente puntual es una línea infinita en y (con 1 voxel MCX pasa a modo 2D)
NY_MCX = 2


def ejecutar_mcx(layers, n_tissue: float, n_photons: int, seed: int) -> dict:
    """Una réplica de MCXCL con la geometría 2D extruida exacta (``y`` cíclico), sin conversión 3D -> 2D.

    ``pmcxcl`` y un dispositivo OpenCL se importan aquí (opcional). Salidas en el mismo formato que
    ``ejecutar_mcml``. Convenciones de MCX: (i) no resta la reflexión especular de la entrada (la fuente parte de la
    superficie), así que todo se multiplica por ``1 - R_esp`` analítica, que es exacto porque la reflexión de entrada
    es independiente del resto de la historia, y ``specular`` se da analítica; (ii) el peso de un fotón detectado es
    ``exp(-sum(mu_a,k L_k))`` con el camino parcial ``L_k`` por medio (MCX no reparte pesos en la interfaz de Fresnel,
    la refleja o transmite al azar), y la absorbida sale de ``energyabs`` de MCX; (iii) la fluencia sale en 1/mm²
    por fotón.
    """
    import pmcxcl
    nz = int(round(sum(d for _, d in layers) / DX))
    nx = 2 * HALF_NX + 1
    vol = np.zeros((nx, NY_MCX, nz), dtype=np.uint8)
    prop = [[0.0, 0.0, 1.0, 1.0]]
    z = 0
    for i, (key, d) in enumerate(layers, start=1):
        a, s, g = tissue_props(key)
        prop.append([a / 1e3, s / 1e3, g, n_tissue])  # 1/m -> 1/mm
        n = int(round(d / DX))
        vol[:, :, z:z + n] = i
        z += n
    mua_mm = np.array([p[0] for p in prop[1:]])
    cfg = dict(nphoton=n_photons, vol=vol, prop=np.array(prop), tstart=0, tend=1e-7, tstep=1e-7,
               srcpos=[HALF_NX + 0.5, NY_MCX / 2, 0.0], srcdir=[0, 0, 1], unitinmm=DX * 1e3, issrcfrom0=1,
               bc="a" + "c" + "_" + "a" + "c" + "_" if n_tissue != 1.0 else "acaaca", isreflect=1,
               savedetflag="dxvp", detpos=[[HALF_NX + 0.5, NY_MCX / 2, 0.0, 400.0]], maxdetphoton=n_photons,
               issave2pt=1, seed=seed, autopilot=1, gpuid=1, outputtype="fluence")
    r = pmcxcl.run(cfg)
    d = r["detp"]  # filas: id, p (una por medio), x (3), v (3)
    nm = len(layers)
    path = d[1:1 + nm]
    xpos, zpos, vz = d[1 + nm], d[3 + nm], d[6 + nm]
    w = np.exp(-(mua_mm[:, None] * path * DX * 1e3).sum(0))  # path está en voxeles
    top = (zpos < 0.5) & (vz < 0)
    bot = (zpos > nz - 0.5) & (vz > 0)
    cell = np.clip(np.floor(xpos).astype(int), 0, nx - 1)
    n_ph = float(n_photons)
    rsp = ((n_tissue - 1.0) / (n_tissue + 1.0)) ** 2
    ent = 1.0 - rsp
    R2d = np.bincount(cell[top], weights=w[top], minlength=nx) / n_ph / DX * ent
    T2d = np.bincount(cell[bot], weights=w[bot], minlength=nx) / n_ph / DX * ent
    # (nx, nz): 1/mm² -> 1/m², por el ancho del periodo en y (una línea de potencia q da q·NY·DX por periodo)
    flu = np.asarray(r["flux"])[:, :, :, 0].mean(1) * 1e6 * (NY_MCX * DX) * ent
    x_phi = np.rint(np.array(X_PHI_MM) * 1e-3 / DX).astype(int)
    absorbed = r["stat"]["energyabs"] / n_ph * ent
    tot = [rsp, R2d.sum() * DX, absorbed, T2d.sum() * DX]
    return {"tot": np.array(tot), "R2d": _plegar(R2d), "T2d": _plegar(T2d), "phi_int": flu.sum(0) * DX,
            "phi2d": _plegar(flu.T)[:, x_phi].T}


def regenerar_mcx(names, n_photons: int, n_seeds: int) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name in names:
        layers, n_t, _ = CASES[name]
        t0 = time.perf_counter()
        res = [ejecutar_mcx(layers, n_t, n_photons, 7000 + 100 * list(CASES).index(name) + s) for s in range(n_seeds)]
        _guardar(DATA / f"mcx_{name}.npz", res, n_photons,
                 herramienta="pmcxcl 0.8.1 + pocl (pthread, CPU), voxel 0.1 mm, ny=2 cíclico")
        print(f"mcx {name}: {n_seeds} x {n_photons} fotones en {time.perf_counter() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------------------
# wavesim: Monte Carlo y difusión

def medio_wavesim(layers, n_tissue: float, dx: float = DX):
    from wavesim.em import montecarlo as mc
    mua, mus, g = layer_maps(layers, dx)
    nx = 2 * HALF_NX + 1
    full = lambda v: np.repeat(v[:, None], nx, axis=1)  # noqa: E731
    nz = mua.size
    return mc.MCMedium(full(mua), full(mus), full(g), np.full((nz, nx), float(n_tissue)), np.ones((nz, nx), bool), dx,
                       1.0, x0=-HALF_NX * dx, z0=dx / 2)


def _plegar(perfil: np.ndarray) -> np.ndarray:
    """Promedia ``x`` y ``-x`` (la fuente está en la columna central): mitad de ruido, ``x >= 0``."""
    c = perfil.shape[-1] // 2
    out = perfil[..., c:].copy()
    out[..., 1:] = 0.5 * (out[..., 1:] + perfil[..., c - 1::-1][..., :c])
    return out


def ejecutar_wavesim(layers, n_tissue: float, n_photons: int, n_seeds: int, dx: float = DX, seed0: int = 0) -> dict:
    """``n_seeds`` réplicas independientes de ``n_photons`` fotones cada una; mismo formato que ``ejecutar_mcml``."""
    from wavesim.em import montecarlo as mc
    m = medio_wavesim(layers, n_tissue, dx)
    out = {k: [] for k in ("tot", "R2d", "T2d", "phi_int", "phi2d", "lateral")}
    for s in range(n_seeds):
        r = mc.simulate(m, n_photons, source="pencil", source_xz=(0.0, 0.0), seed=seed0 + s, backend="auto",
                        periodic_x=False)
        out["tot"].append([r.specular, r.reflectance, r.absorbed_total, r.transmittance])
        out["lateral"].append(r.lateral)
        out["R2d"].append(_plegar(r.reflectance_profile))
        out["T2d"].append(_plegar(r.transmittance_profile))
        out["phi_int"].append(r.fluence.sum(1) * dx)
        x_phi = np.rint(np.array(X_PHI_MM) * 1e-3 / dx).astype(int)
        out["phi2d"].append(_plegar(r.fluence)[:, x_phi].T)  # (n_x_phi, nz)
    return {k: np.array(v) for k, v in out.items()}


def ejecutar_difusion(layers, n_tissue: float, dx: float = DX) -> dict:
    """Difusión de ``optics.solve_diffusion`` con una fuente puntual isótropa a ``1/mu_s'`` bajo la superficie.

    Es la representación estándar de un haz fino en la aproximación de difusión. La fuente se escala por
    ``1 - R_especular`` (la difusión no ve la reflexión de la entrada). El fantoma se rodea de una fila/columna de
    aire para que haya contorno extrapolado en los cuatro lados (el borde de la malla es de flujo nulo).
    """
    from wavesim.core.grid import Grid2D
    from wavesim.em import optics
    from wavesim.materials import optical
    from wavesim.phantoms import TISSUE_INDEX, Phantom
    nx = 2 * int(round(HALF_NX * DX / dx)) + 1
    rows = [(TISSUE_INDEX[k], int(round(d / dx))) for k, d in layers]
    nzt = sum(n for _, n in rows)
    lab = np.full((nzt + 2, nx + 2), TISSUE_INDEX["air"], dtype=int)
    z = 1
    for idx, n in rows:
        lab[z:z + n, 1:-1] = idx
        z += n
    grid = Grid2D(nx + 2, nzt + 2, dx, x0=-(nx // 2 + 1) * dx, z0=-dx / 2)
    ph = Phantom(grid, lab)
    mua0 = float(optical.mu_a(layers[0][0], LAMBDA_NM))
    mus0 = float(optical.mu_s_reduced(layers[0][0], LAMBDA_NM))
    r = optics.solve_diffusion(ph, LAMBDA_NM, source=(0.0, 1.0 / (mua0 + mus0)), n_tissue=n_tissue)
    f_esp = 1.0 - ((n_tissue - 1.0) / (n_tissue + 1.0)) ** 2  # fracción que entra (sin especular)
    sel = slice(1, -1)
    phi = r.fluence[1:-1, sel] * f_esp
    refl = r.reflected_profile[sel] * f_esp
    trans = r.transmitted_profile[sel] * f_esp
    mua_z = layer_maps(layers, dx)[0]
    c = nx // 2
    xs = np.arange(c + 1) * dx
    x_fino = np.arange(HALF_NX + 1) * DX
    x_phi = np.rint(np.array(X_PHI_MM) * 1e-3 / dx).astype(int)
    z_fino = (np.arange(int(round(sum(d for _, d in layers) / DX))) + 0.5) * DX
    z_d = (np.arange(nzt) + 0.5) * dx
    interp = lambda xp, fp, x: np.interp(x, xp, fp)  # noqa: E731
    phi_int_d = phi.sum(1) * dx
    absorbida = float((mua_z[:, None] * phi).sum() * dx * dx)
    tot = [1.0 - f_esp, float(refl.sum() * dx), absorbida, float(np.nansum(trans) * dx)]
    return {
        "tot": np.array(tot),
        "R2d": interp(xs, _plegar(refl), x_fino),
        "T2d": interp(xs, _plegar(trans), x_fino),
        "phi_int": interp(z_d, phi_int_d, z_fino),
        "phi2d": np.array([interp(z_d, _plegar(phi)[:, i], z_fino) for i in x_phi]),
    }


# ----------------------------------------------------------------------------------------------------------
# Comparación

def integral_bins(perfil: np.ndarray, ejes_mm: np.ndarray, bordes_mm, dx: float) -> np.ndarray:
    """Integral del perfil (por réplica, eje -1) en cada intervalo ``[b_i, b_{i+1})`` de ``ejes_mm`` (mm)."""
    out = []
    for a, b in zip(bordes_mm[:-1], bordes_mm[1:]):
        sel = (ejes_mm >= a) & (ejes_mm < b)
        out.append(perfil[..., sel].sum(-1) * dx)
    return np.stack(out, -1)


def tabla_perfil(titulo: str, ref: np.ndarray, ws: np.ndarray, dif: np.ndarray | None, mcx: np.ndarray | None,
                 eje_mm: np.ndarray, bordes, dx: float, umbral_rel: float = 0.05, imprimir: bool = True):
    """Integra en intervalos y compara con MCML (la referencia principal); devuelve filas y errores relativos.

    Fiable = el error estándar relativo de MCML en el intervalo es <= ``umbral_rel`` (los intervalos ruidosos se
    marcan y no entran en los errores relativos resumidos). ``dif`` y ``mcx`` son opcionales.
    """
    br, bw = integral_bins(ref, eje_mm, bordes, dx), integral_bins(ws, eje_mm, bordes, dx)
    mr, sr = mean_sem(br)
    mw, sw = mean_sem(bw)
    md = integral_bins(dif[None], eje_mm, bordes, dx)[0] if dif is not None else None
    mx, sx = mean_sem(integral_bins(mcx, eje_mm, bordes, dx)) if mcx is not None else (None, None)
    filas = []
    for i, (a, b) in enumerate(zip(bordes[:-1], bordes[1:])):
        if mr[i] <= 0:
            continue
        se = np.hypot(sr[i], sw[i])
        fila = dict(rango=f"{a:g}-{b:g}", ref=mr[i], sem=sr[i], ws=mw[i], sem_ws=sw[i], rel_ws=(mw[i] - mr[i]) / mr[i],
                    z=(mw[i] - mr[i]) / se if se > 0 else float("nan"), fiable=sr[i] / mr[i] <= umbral_rel)
        if md is not None:
            fila["dif"], fila["rel_dif"] = md[i], (md[i] - mr[i]) / mr[i]
        if mx is not None:
            se = np.hypot(sr[i], sx[i])
            fila["mcx"], fila["rel_mcx"] = mx[i], (mx[i] - mr[i]) / mr[i]
            fila["z_mcx"] = (mx[i] - mr[i]) / se if se > 0 else float("nan")
        filas.append(fila)
    if imprimir:
        print(f"\n{titulo}\n| intervalo [mm] | MCML ± sem | MCX | wavesim MC | MCX/MCML-1 (z) | wavesim/MCML-1 (z) "
              "| difusión/MCML-1 |"
              "\n|---|---|---|---|---|---|---|")
        for f in filas:
            xs = f"{f['mcx']:.4e}" if "mcx" in f else "-"
            rx = f"{f['rel_mcx']:+.2%} ({f['z_mcx']:+.1f} σ)" if "mcx" in f else "-"
            rd = f"{f['rel_dif']:+.1%}" if "dif" in f else "-"
            print(f"| {f['rango']} | {f['ref']:.4e} ± {f['sem']:.1e} | {xs} | {f['ws']:.4e} | {rx} | "
                  f"{f['rel_ws']:+.2%} ({f['z']:+.1f} σ) | {rd} |" + ("" if f["fiable"] else " ruido"))
    return filas


def resumen(filas, clave: str) -> tuple[float, float]:
    """``(max |error relativo|, rms)`` de ``clave`` en los intervalos fiables."""
    v = np.array([f[clave] for f in filas if f["fiable"] and clave in f])
    return (float(np.abs(v).max()), float(np.sqrt(np.mean(v**2)))) if v.size else (float("nan"), float("nan"))


def comparar(name: str, n_photons: int, n_seeds: int, dx_ws: float = DX, con_difusion: bool = True) -> dict:
    layers, n_t, desc = CASES[name]
    ref = np.load(DATA / f"mcml_{name}.npz")
    mcx = np.load(DATA / f"mcx_{name}.npz") if (DATA / f"mcx_{name}.npz").exists() else None
    t0 = time.perf_counter()
    ws = ejecutar_wavesim(layers, n_t, n_photons, n_seeds, dx_ws)
    t_ws = time.perf_counter() - t0
    dif = ejecutar_difusion(layers, n_t) if con_difusion else None
    print(f"\n### {name}: {desc} (MCML: {ref['tot'].shape[0]} x {int(ref['n_photons']):.0e} fotones; "
          + (f"MCX: {mcx['tot'].shape[0]} x {int(mcx['n_photons']):.0e}; " if mcx is not None else "")
          + f"wavesim: {n_seeds} x {n_photons:.0e} en {t_ws:.0f} s)")
    mr, sr = mean_sem(ref["tot"])
    mw, sw = mean_sem(ws["tot"])
    mxx, sxx = mean_sem(mcx["tot"]) if mcx is not None else (None, None)
    print("\n| total | MCML | MCX | wavesim MC | wavesim - MCML (z) | difusión |\n|---|---|---|---|---|---|")
    for i, nombre in enumerate(TOT_NAMES):
        se = np.hypot(sr[i], sw[i])
        z = (mw[i] - mr[i]) / se if se > 0 else float("nan")
        d = f"{dif['tot'][i]:.5f}" if dif is not None else "-"
        xs = f"{mxx[i]:.5f} ± {sxx[i]:.5f}" if mcx is not None else "-"
        print(f"| {nombre} | {mr[i]:.5f} ± {sr[i]:.5f} | {xs} | {mw[i]:.5f} ± {sw[i]:.5f} | "
              f"{mw[i] - mr[i]:+.5f} ({z:+.1f} σ) | {d} |")
    print(f"| lateral (wavesim) | - | - | {ws['lateral'].mean():.1e} | | |")
    x_mm = np.arange(HALF_NX + 1) * DX * 1e3
    nz = ref["phi_int"].shape[1]
    z_mm = (np.arange(nz) + 0.5) * DX * 1e3
    opt = lambda k: dif[k] if dif is not None else None  # noqa: E731
    mx_ = lambda k: mcx[k] if mcx is not None else None  # noqa: E731
    res = {"tot": dict(mcml=(mr, sr), ws=(mw, sw), mcx=(mxx, sxx), dif=None if dif is None else dif["tot"])}
    res["R"] = tabla_perfil(f"R(x), {name} [integral por intervalo de x: fracción incidente]", ref["R2d"], ws["R2d"],
                            opt("R2d"), mx_("R2d"), x_mm, X_BINS_MM, DX)
    if mr[3] > 1e-5:
        res["T"] = tabla_perfil(f"T(x), {name}", ref["T2d"], ws["T2d"], opt("T2d"), mx_("T2d"), x_mm, X_BINS_MM, DX)
    zb = tuple(b for b in Z_BINS_MM if b < z_mm[-1]) + (z_mm[-1] + DX * 1e3,)
    res["phi"] = tabla_perfil(f"Fluencia integrada en x, Phi_int(z), {name} [integral por intervalo de z]",
                              ref["phi_int"], ws["phi_int"], opt("phi_int"), mx_("phi_int"), z_mm, zb, DX)
    for i in (0, 2, 4):
        res[f"phi_x{i}"] = tabla_perfil(
            f"Fluencia 2D en x = {X_PHI_MM[i]:g} mm, {name} [integral por intervalo de z]", ref["phi2d"][:, i],
            ws["phi2d"][:, i],
            None if dif is None else dif["phi2d"][i], None if mcx is None else mcx["phi2d"][:, i], z_mm, zb, DX)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--regen", action="store_true", help="regenerar las referencias con MCML (necesita --mcml-bin)")
    ap.add_argument("--mcml-bin", default=os.environ.get("MCML_BIN"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--photons", type=int, default=None, help="fotones por réplica (wavesim: 1e6; MCML: 4e6)")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--regen-mcx", action="store_true", help="regenerar las referencias con pmcxcl (ver la cabecera)")
    a = ap.parse_args()
    if a.regen_mcx:
        regenerar_mcx(a.cases, a.photons or 1_000_000, a.seeds)
        return
    if a.regen:
        if not a.mcml_bin:
            raise SystemExit("--regen necesita --mcml-bin (ver la cabecera para compilar MCML)")
        regenerar(a.mcml_bin, a.cases, a.photons or 4_000_000, a.seeds, a.jobs)
        return
    n = a.photons or (125_000 if a.quick else 1_000_000)
    for name in a.cases:
        comparar(name, n, a.seeds)


if __name__ == "__main__":
    main()
