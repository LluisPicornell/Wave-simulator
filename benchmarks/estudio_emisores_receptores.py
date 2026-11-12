"""Estudio: ¿cuántos emisores y receptores hacen falta para detectar fracturas con un dispositivo barato?

Uso (desde la raíz del repositorio; sólo numpy y matplotlib, numba opcional):

    NUMBA_NUM_THREADS=1 PYTHONPATH=. python benchmarks/estudio_emisores_receptores.py --procesos 4
    ... --fase simular        sólo las simulaciones (se guardan en outputs/estudio_er/sim/ y no se repiten)
    ... --fase analizar       estadísticas por fantoma (outputs/estudio_er/stats/), tablas y figuras
    ... --cuerpos normal      sólo un tipo de cuerpo

Coste medido (1 núcleo, Numba): una emisión a 3 MHz ~11 s (cuerpo normal) y ~20 s (obeso); una FMC
de 32 emisiones, 6-11 min; una simulación de baja frecuencia, 20-45 s. Todo el estudio son unas 9 h de
núcleo (~2.5 h con 4 procesos) más ~15 min de análisis.

Diseño (detalles y resultados en docs/estudio_emisores_receptores.md):

1. 3 MHz, reflexión. Sonda de 32 elementos a λ/2 (0.257 mm, apertura 8.2 mm) sobre la cara anterior de la
   tibia (corte longitudinal 2D, malla λ/8 = 60 µm). Por fantoma se simula la captura de matriz completa
   (FMC: cada elemento emite solo y todos reciben) y de ella se reconstruyen todas las configuraciones:
   N_tx × N_rx ∈ {1, 2, 4, 8, 16, 32}² en disposición dispersa (misma apertura) y compacta (mismo paso,
   sonda pequeña desplazada por la superficie), transductor monoestático (con y sin SAFT, de λ/2 y de
   1 mm), A-scan fijo, onda plana de 1/3/7 ángulos y STA completa; DAS y factor de coherencia.
2. Ruido electrónico gaussiano filtrado a la banda, añadido a cada disparo físico (nunca a la FMC antes de
   sumar emisiones), con SNR por canal de 40, 20 y 0 dB respecto al eco cortical de un disparo de un
   elemento en la pierna sana de referencia (cuerpo normal); 3 realizaciones.
3. 0.5 y 1 MHz: transmisión axial (AT) con emisores en los dos extremos y 41 receptores (cada emisor usa
   1-17 de sus 17 lejanos, apilados), y transmisión a través (TT, 0.5 MHz) con 25 receptores puntuales.
4. Detección: negativos = piernas sanas con otra semilla (grosores ±10 %, inclinación del hueso ~N(0, 2°),
   rugosidad y microestructura); positivos = lesiones del ejemplo 11 en varias posiciones y anatomías.
   Estadísticas sin verdad de terreno y AUC con IC por bootstrap de fantomas.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from wavesim.analysis import dilate, resample_mask, residual_db
from wavesim.core.signals import analytic_signal, pulse_duration, pulse_peak_time
from wavesim.phantoms import BODY_TYPES, longitudinal_limb, random_limb_spec
from wavesim.ultrasound import ElasticSolver, LinearArray, das_plane_wave, line_receiver
from wavesim.ultrasound.fmc import (FMCData, acquire_fmc, band_noise, delayed_pairs, fractional_delay, pair_image,
                                    synthesize_rf)
from wavesim.ultrasound.quality import fwhm_1d, point_scatterer_medium, water_phantom

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "estudio_er"
SIM = OUT / "sim"
STATS = OUT / "stats"

_spec = importlib.util.spec_from_file_location("ejemplo11",
                                               ROOT / "examples" / "11_fracturas_internas_baja_frecuencia.py")
EX11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EX11)
CASES = EX11.CASES  # definiciones de las lesiones del ejemplo 11

NPML = 20
# --- 3 MHz (reflexión) ---
F_HI = 3e6
DX_HI = 1450 / F_HI / 8
N_EL = 32
PITCH = 1540 / F_HI / 2
WIDTH_HI = 15e-3
DECIMATE = 2
# --- baja frecuencia ---
DX_LOW = 0.1e-3
F_AT = (0.5e6, 1e6)
F_TT = (0.5e6,)
AT_WIDTH, AT_SRC, AT_RX = 100e-3, 45e-3, 20e-3  # emisores en ±45 mm, receptores cada 1 mm en ±20 mm
TT_WIDTH, TT_RX = 30e-3, 12e-3  # receptores cada 1 mm en ±12 mm

LESIONS_HI = ("control_desplazada", "fisura_superficial", "intracortical", "estres_endostica", "contusion_osea")
LESIONS_LOW = LESIONS_HI + ("cortical_lejana",)
POS_HI = {"normal": (-2e-3, 0.0, 2e-3), "obeso": (-2e-3, 2e-3)}
POS_LOW = {"normal": (-6e-3, 0.0, 6e-3), "obeso": (-6e-3, 6e-3)}
N_NEG = {"tibia": 6, "tibia_metafisis": 4}


# --------------------------------------------------------------------------
# Muestras (fantomas)
# --------------------------------------------------------------------------
def samples(body: str) -> dict:
    """Nombre → (preset, lesión o None, índice de posición, semilla, gemela_de).

    La gemela es la misma pierna (misma semilla y anatomía) sin la lesión: sólo para las
    métricas con verdad de terreno (cuerpo normal, posición central, 3 MHz).
    """
    out = {}
    for preset, n in N_NEG.items():
        base = 1000 if preset == "tibia" else 2000
        for k in range(n):
            out[f"{preset}_sana{k}"] = (preset, None, -1, base + k, None)
    for li, case in enumerate(LESIONS_LOW):
        preset = CASES[case][0]
        for k in range(len(POS_HI[body])):
            out[f"{case}_p{k}"] = (preset, case, k, 3000 + 100 * li + k, None)
    if body == "normal":
        for case in LESIONS_HI:
            out[f"{case}_p1_gemela"] = (CASES[case][0], None, 1, 3000 + 100 * LESIONS_LOW.index(case) + 1,
                                        f"{case}_p1")
    return out


def anatomy(preset: str, seed: int, body: str):
    """Anatomía individual: grosores ±10 % e inclinación del hueso ~N(0, 2°) sobre el tipo de cuerpo."""
    return random_limb_spec(np.random.default_rng(seed), preset, body=BODY_TYPES[body], jitter=0.1)


def lesion(case: str | None, position: float):
    return None if case is None else replace(CASES[case][1], position=position)


def _near_endosteum(spec) -> float:
    return spec.standoff + spec.skin + spec.fat + spec.muscle + spec.cortex


def build(body: str, name: str, technique: str):
    preset, case, k, seed, _ = samples(body)[name]
    spec = anatomy(preset, seed, body)
    tilt = abs(np.tan(np.radians(spec.bone_tilt_deg)))
    if technique == "fmc3":
        pos = POS_HI[body][k] if k >= 0 and case is not None else 0.0
        depth = _near_endosteum(spec) + (8e-3 if spec.trabecular else 6e-3) + tilt * WIDTH_HI / 2 + (NPML + 3) * DX_HI
        return longitudinal_limb(spec, lesion(case, pos), width=WIDTH_HI, dx=DX_HI, depth=depth, seed=seed,
                                 realism="realista")
    pos = POS_LOW[body][k] if k >= 0 and case is not None else 0.0
    if technique.startswith("at"):
        depth = _near_endosteum(spec) + (12e-3 if spec.trabecular else 8e-3) + tilt * AT_WIDTH / 2 + (NPML + 3) * DX_LOW
        return longitudinal_limb(spec, lesion(case, pos), width=AT_WIDTH, dx=DX_LOW, depth=depth, seed=seed,
                                 realism="realista")
    return longitudinal_limb(spec, lesion(case, pos), width=TT_WIDTH, dx=DX_LOW, seed=seed, realism="realista")


def hi_array(grid) -> LinearArray:
    return LinearArray(n_elements=N_EL, pitch=PITCH, z=grid.z0 + (NPML + 3) * grid.dx,
                       x_center=grid.x0 + grid.width / 2 - grid.dx / 2, f0=F_HI)


# --------------------------------------------------------------------------
# Simulaciones
# --------------------------------------------------------------------------
def sim_at(p, f0: float, side: int) -> dict:
    """Transmisión axial: emisor de λ/2 en x = side·45 mm, receptores en la misma línea cada 1 mm en ±20 mm."""
    g = p.grid
    solver = ElasticSolver(p, f0, npml=NPML)
    z = g.z0 + (NPML + 3) * g.dx
    w = min(1500.0 / f0 / 2, 2e-3)
    arr = LinearArray(n_elements=1, pitch=w, element_width=w, z=z, x_center=side * AT_SRC, f0=f0)
    soft = p.meta["bone_top_depth"] - z
    max_off = AT_SRC + AT_RX
    nt = int((pulse_duration(f0) + 2 * soft / 1400 + max_off / 3000 + 3 / f0) / solver.dt)
    t = solver.time_axis(nt)
    solver.add_source(arr.transmit_source(g, t, np.zeros(1)))
    rx = line_receiver(g, z, -AT_RX - 1e-4, AT_RX + 1e-4, spacing=1e-3)
    solver.add_receiver("rx", rx)
    res = solver.run(nt, track_peak=False)
    return {"t": t[::2], "traces": res.traces["rx"][:, ::2].astype(np.float32), "x": g.x[rx.i],
            "src_x": np.array(side * AT_SRC)}


def sim_tt(p, f0: float) -> dict:
    """Transmisión a través: línea de emisores de 1 mm con bordes suavizados (como el ejemplo 11)."""
    g = p.grid
    solver = ElasticSolver(p, f0, npml=NPML)
    n = int((g.width - 2 * (NPML + 6) * g.dx) / 1e-3)
    arr = LinearArray(n_elements=n, pitch=1e-3, element_width=1e-3, z=g.z0 + (NPML + 3) * g.dx,
                      x_center=g.x0 + g.width / 2 - g.dx / 2, f0=f0)
    nt = int((pulse_duration(f0) + g.depth / 1400 + 3 / f0) / solver.dt)
    t = solver.time_axis(nt)
    solver.add_source(arr.transmit_source(g, t, np.zeros(n), EX11._taper(n)))
    rx = line_receiver(g, g.z0 + g.depth - (NPML + 3) * g.dx, -TT_RX - 1e-4, TT_RX + 1e-4, spacing=1e-3)
    solver.add_receiver("rx", rx)
    res = solver.run(nt, track_peak=False)
    return {"t": t[::2], "traces": res.traces["rx"][:, ::2].astype(np.float32), "x": g.x[rx.i]}


def sim_path(body: str, name: str, technique: str) -> Path:
    return SIM / f"{body}__{name}__{technique}.npz"


def techniques_for(body: str, name: str) -> list[str]:
    preset, case, k, seed, twin = samples(body)[name]
    if twin is not None:
        return ["fmc3"]
    out = []
    if case is None or case in LESIONS_HI:
        out.append("fmc3")
    if case != "cortical_lejana":  # la AT sólo recorre la cortical cercana
        out += [f"at{f / 1e6:g}{s}" for f in F_AT for s in "LR"]
    out += [f"tt{f / 1e6:g}" for f in F_TT]
    return out


def run_job(job) -> str:
    body, name, technique = job
    path = sim_path(*job)
    if path.exists():
        return f"(en caché) {path.name}"
    warnings.filterwarnings("ignore", message="Sólo .* puntos por longitud de onda")
    t0 = time.time()
    if technique == "psf":
        return _run_psf(path)
    p = build(body, name, technique)
    if technique == "fmc3":
        fmc = acquire_fmc(p, hi_array(p.grid), F_HI, npml=NPML, decimate=DECIMATE)
        tmp = path.with_suffix(".tmp.npz")
        fmc.save(tmp)
        tmp.rename(path)
    else:
        f0 = float(technique[2:].rstrip("LR")) * 1e6
        data = sim_at(p, f0, -1 if technique.endswith("L") else 1) if technique.startswith("at") else sim_tt(p, f0)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.rename(path)
    return f"{path.name} ({time.time() - t0:.0f} s, malla {p.grid.nz}x{p.grid.nx})"


def _psf_setup():
    """Agua con un dispersor puntual a 6 mm de la sonda (profundidad de la cortical en el cuerpo normal)."""
    from wavesim.core.grid import Grid2D
    g = Grid2D.from_size(WIDTH_HI, 16e-3, DX_HI, x0=-WIDTH_HI / 2)
    arr = hi_array(g)
    return g, arr, (0.0, arr.z + 6e-3)


def _run_psf(path: Path) -> str:
    t0 = time.time()
    g, arr, pt = _psf_setup()
    medium = point_scatterer_medium(g, F_HI, [pt], contrast=3.0)
    fmc = acquire_fmc(water_phantom(g), arr, F_HI, npml=NPML, medium=medium, n_mechanisms=0, decimate=DECIMATE)
    fmc.save(path)
    return f"{path.name} ({time.time() - t0:.0f} s)"


def all_jobs(bodies) -> list[tuple[str, str, str]]:
    jobs = [("agua", "psf", "psf")]
    for body in bodies:
        js = [(body, n, t) for n in samples(body) for t in techniques_for(body, n)]
        # las FMC primero (las más largas) para repartir mejor la carga entre procesos
        js.sort(key=lambda j: (j[2] != "fmc3", j[1]))
        jobs += js
    return jobs


# --------------------------------------------------------------------------
# Reconstrucción de configuraciones a 3 MHz
# --------------------------------------------------------------------------
C0 = 1540.0
PX = C0 / F_HI / 4
X_IMG = np.arange(-3.5e-3, 3.5e-3 + PX / 2, PX)
X_PSF = np.arange(-5.5e-3, 5.5e-3 + PX / 2, PX)
X_STAT = 3.0e-3  # las estadísticas sólo miran |x| <= 3 mm (las lesiones están en x = -2, 0, 2 mm)
F_NUMBER, WINDOW = 0.5, "rect"  # aceptación angular ±45° sin ventana: igual para todas las configuraciones
F_NUMBER_1MM = 1.5  # directividad del elemento de 1 mm a 3 MHz (-6 dB a ~15°)
SNR_DB = (np.inf, 40.0, 20.0, 0.0)  # SNR por canal (inf = sin ruido)
N_REAL = 3
NS = (1, 2, 4, 8, 16, 32)
PW_ANGLES = {1: (0.0,), 3: (-12.0, 0.0, 12.0), 7: tuple(np.linspace(-12.0, 12.0, 7))}
POS_ERR = 0.1e-3  # error de posición (desviación típica) de una sonda movida a mano sin seguimiento preciso
HI_STATS = ("caida", "escalon", "sub_local", "sub_global")
FIG_CFGS = ("mono_lineas", "mono_saft", "mono1mm_lineas", "disp_1x32", "pw1", "pw7", "disp_32x32", "disp_4x4",
            "comp_8x8", "disp_32x32_cf")


def sparse_idx(n: int) -> np.ndarray:
    """Elementos de una disposición dispersa de ``n`` sobre la misma apertura (1 = el central)."""
    return np.array([N_EL // 2]) if n == 1 else np.round(np.linspace(0, N_EL - 1, n)).astype(int)


def compact_tiles(ntx: int, nrx: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Sonda compacta de ``max(ntx, nrx)`` elementos contiguos desplazada a medio bloque: (tx, rx, inicio)."""
    b = max(ntx, nrx)
    starts = list(range(0, N_EL - b + 1, max(b // 2, 1)))
    if starts[-1] != N_EL - b:
        starts.append(N_EL - b)
    return [(s + (b - ntx) // 2 + np.arange(ntx), s + (b - nrx) // 2 + np.arange(nrx), s) for s in starts]


def hi_configs() -> dict:
    """Configuraciones STA: nombre → lista de teselas (tx, rx, centro x o None)."""
    ex = (np.arange(N_EL) - (N_EL - 1) / 2) * PITCH  # relativo al centro de la sonda
    cfgs = {}
    for ntx in NS:
        for nrx in NS:
            cfgs[f"disp_{ntx}x{nrx}"] = [(sparse_idx(ntx), sparse_idx(nrx), None)]
            b = max(ntx, nrx)
            cfgs[f"comp_{ntx}x{nrx}"] = [(tx, rx, ex[s:s + b].mean()) for tx, rx, s in compact_tiles(ntx, nrx)]
    return cfgs


def _composite(images: list[np.ndarray], centers, x_img) -> np.ndarray:
    """Cada columna se toma de la tesela (posición de la sonda pequeña) más cercana."""
    if len(images) == 1:
        return images[0]
    nearest = np.argmin(np.abs(np.asarray(centers)[:, None] - x_img[None, :]), axis=0)
    out = np.empty_like(images[0])
    for k, im in enumerate(images):
        out[:, nearest == k] = im[:, nearest == k]
    return out


def _blank_mask(fmc: FMCData, c0: float = C0) -> np.ndarray:
    xt = fmc.element_x[fmc.tx_elements]
    lim = np.abs(xt[:, None] - fmc.element_x[None, :]) / c0 + fmc.t_peak + 0.6 * fmc.pulse_length
    return fmc.t[None, None, :] < lim[:, :, None]


def _interp(iq: np.ndarray, t: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Muestra ``iq`` (1D) en los instantes ``tau`` (interpolación lineal; 0 fuera)."""
    dt = t[1] - t[0]
    idx = (tau - t[0]) / dt
    i0 = np.floor(idx).astype(int)
    frac = idx - i0
    valid = (i0 >= 0) & (i0 < t.size - 1)
    i0 = np.clip(i0, 0, t.size - 2)
    return np.where(valid, iq[i0] * (1 - frac) + iq[i0 + 1] * frac, 0.0)


class Imager:
    """Todas las imágenes de una FMC: señal sin ruido y ``N_REAL`` imágenes de ruido de valor eficaz 1 por canal.

    Las imágenes son lineales en las trazas, así que la imagen con ruido de nivel σ es
    ``señal + σ·ruido`` (el ruido se genera por disparo físico: por pareja en STA, por
    receptor y ángulo en onda plana, por posición en el monoestático).
    """

    def __init__(self, fmc: FMCData, x_img, z_img, seed: int, c0: float = C0):
        self.fmc, self.x, self.z, self.c0 = fmc, x_img, z_img, c0
        self.rng = np.random.default_rng(seed)
        mask = _blank_mask(fmc, c0)
        self.rf = np.where(mask, 0.0, fmc.rf)
        self.noise_rf = [np.where(mask, 0.0, band_noise(fmc.rf.shape, fmc.dt, F_HI, 1.0, self.rng))
                         for _ in range(N_REAL)]
        self.D = delayed_pairs(self.rf, fmc, x_img, z_img, c0, F_NUMBER, WINDOW)
        self.DN = [delayed_pairs(n, fmc, x_img, z_img, c0, F_NUMBER, WINDOW) for n in self.noise_rf]

    def _cubes(self):
        return [self.D, *self.DN]

    def sta(self, tiles) -> list[np.ndarray]:
        """[señal, ruido_0, ...] de una configuración STA (teselas)."""
        centers = [c for _, _, c in tiles]
        return [_composite([pair_image(D, tx, rx) for tx, rx, _ in tiles], centers, self.x) for D in self._cubes()]

    def channels(self, tiles) -> list[list[np.ndarray]]:
        """Por tesela: [canales de señal, canales de ruido_0, ...], cada uno ``C_j = Σ_k D[k, j]`` (n_rx, nz, nx)."""
        return [[D[np.ix_(tx, rx)].sum(0) for D in self._cubes()] for tx, rx, _ in tiles]

    def sta_cf(self, tiles, parts, sigma: float, r: int) -> np.ndarray:
        """Imagen DAS × factor de coherencia con ruido ``sigma`` (realización ``r``); no es lineal en el ruido."""
        ims = []
        for part in parts:
            C = part[0] + sigma * part[1 + r]
            cf = np.abs(C.sum(0)) ** 2 / np.maximum(C.shape[0] * (np.abs(C) ** 2).sum(0), 1e-30)
            ims.append(C.sum(0) * cf)
        return _composite(ims, [c for _, _, c in tiles], self.x)

    def plane_wave(self, angles) -> list[np.ndarray]:
        fmc = self.fmc
        ex = fmc.element_x
        arr_like = LinearArray(n_elements=ex.size, pitch=PITCH, z=fmc.array_z, x_center=float(ex.mean()), f0=F_HI)
        out = [0.0] * (1 + N_REAL)
        blank = fmc.t_peak + 0.6 * fmc.pulse_length
        for ang in angles:
            delays = arr_like.plane_wave_delays(ang, self.c0)
            mask = fmc.t[None, :] < (delays[:, None] + blank)
            sig = np.where(mask, 0.0, synthesize_rf(fmc, delays))
            out[0] = out[0] + das_plane_wave(sig, fmc.t, ex, fmc.array_z, ang, self.x, self.z, self.c0, fmc.t_peak,
                                             F_NUMBER, WINDOW)
            for r in range(N_REAL):
                nz = np.where(mask, 0.0, band_noise(sig.shape, fmc.dt, F_HI, 1.0, self.rng))
                out[1 + r] = out[1 + r] + das_plane_wave(nz, fmc.t, ex, fmc.array_z, ang, self.x, self.z, self.c0,
                                                         fmc.t_peak, F_NUMBER, WINDOW)
        return out

    def mono_traces(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        """Trazas analíticas (posiciones, nt) de un transductor monoestático de ``width`` elementos y sus centros."""
        starts = np.arange(N_EL - width + 1)
        tr = np.stack([self.rf[s:s + width, s:s + width].sum((0, 1)) for s in starts])
        centers = np.array([self.fmc.element_x[s:s + width].mean() for s in starts])
        return tr, centers

    def mono(self, width: int, saft: bool, f_number: float, pos_err: float = 0.0) -> list[np.ndarray]:
        tr, centers = self.mono_traces(width)
        blank = self.fmc.t < self.fmc.t_peak + 0.6 * self.fmc.pulse_length
        traces = [tr] + [np.where(blank, 0.0, band_noise(tr.shape, self.fmc.dt, F_HI, 1.0, self.rng))
                         for _ in range(N_REAL)]
        used = centers + (self.rng.normal(0.0, pos_err, centers.size) if pos_err else 0.0)
        return [_mono_image(analytic_signal(t, axis=-1), used, self.fmc, self.x, self.z, saft, f_number, self.c0)
                for t in traces]


def _mono_image(iq, centers, fmc, x_img, z_img, saft, f_number, c0):
    X, Z = np.meshgrid(x_img, z_img)
    depth = Z - fmc.array_z
    if not saft:  # una línea A por posición: cada columna, la de la posición más cercana
        nearest = np.argmin(np.abs(centers[:, None] - x_img[None, :]), axis=0)
        tau = fmc.t_peak + 2.0 * (z_img - fmc.array_z) / c0
        return np.stack([_interp(iq[nearest[i]], fmc.t, tau) for i in range(x_img.size)], axis=1)
    out = np.zeros(X.shape, complex)
    half = np.maximum(depth, 1e-6) / (2.0 * f_number)
    for p, xc in enumerate(centers):
        w = np.abs(X - xc) <= half
        out += np.where(w, _interp(iq[p], fmc.t, fmc.t_peak + 2.0 * np.hypot(X - xc, depth) / c0), 0.0)
    return out


def _moving_mean(a: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or a.size < n:
        return a
    return np.convolve(a, np.ones(n) / n, mode="valid")


def image_stats(env: np.ndarray, z_img: np.ndarray, x_img: np.ndarray, array_z: float) -> np.ndarray:
    """Estadísticas de detección sin verdad de terreno (mayor = más sospechoso de fractura).

    Se busca la línea cortical (máximo de cada columna en ±2.5 mm de la profundidad mediana
    del eco más fuerte) y se mide:

    - ``caida``: mediana de la amplitud cortical menos su mínimo local (media de 3 columnas), dB.
    - ``escalon``: mayor salto de la profundidad de la línea entre las medianas de 1 mm a cada
      lado, tras quitar la recta de Theil-Sen (inclinación del hueso), m.
    - ``sub_local``: exceso local (0.5 mm) de energía bajo la cortical (de 0.75 a 4 mm por
      debajo de la línea) relativa a la cortical, dB respecto a su mediana.
    - ``sub_global``: mediana de esa energía relativa, dB (sólo comparable entre personas).
    """
    sel = np.abs(x_img) <= X_STAT
    E = env[:, sel]
    x = x_img[sel]
    dz = z_img[1] - z_img[0]
    j_lo = int(np.searchsorted(z_img, array_z + 3e-3))
    jg = int(np.median(j_lo + np.argmax(E[j_lo:], axis=0)))
    half = int(round(2.5e-3 / dz))
    a, b = max(jg - half, j_lo), min(jg + half + 1, z_img.size)
    jc = a + np.argmax(E[a:b], axis=0)
    cols = np.arange(x.size)
    A = 20 * np.log10(E[jc, cols] + 1e-30)
    caida = float(np.median(A) - _moving_mean(A, 3).min())
    zc = z_img[jc]
    iu = np.triu_indices(x.size, 1)
    slope = float(np.median((zc[iu[1]] - zc[iu[0]]) / (x[iu[1]] - x[iu[0]])))  # Theil-Sen
    r = zc - slope * x
    r = r - np.median(r)
    w = int(round(1e-3 / (x[1] - x[0])))
    escalon = max((abs(np.median(r[i - w:i]) - np.median(r[i:i + w])) for i in range(w, x.size - w + 1)),
                  default=0.0)
    lo, hi = int(round(0.75e-3 / dz)), int(round(4e-3 / dz))
    rel = np.empty(x.size)
    for i in cols:
        band = E[min(jc[i] + lo, z_img.size - 1):min(jc[i] + hi, z_img.size), i]
        rel[i] = 10 * np.log10(np.mean(band ** 2) + 1e-60) - A[i]
    sub_local = float(_moving_mean(rel, 4).max() - np.median(rel))
    return np.array([caida, escalon, sub_local, float(np.median(rel))])


def ascan_stats(iq: np.ndarray, fmc: FMCData) -> np.ndarray:
    """A-scan fijo: ``caida`` = −amplitud del eco más fuerte (dB) y ``sub_global`` = energía 0.75-4 mm por debajo."""
    env = np.abs(iq)
    t = fmc.t
    ok = t > fmc.t_peak + 2 * 3e-3 / C0
    k = int(np.argmax(np.where(ok, env, 0.0)))
    A = 20 * np.log10(env[k] + 1e-30)
    band = (t > t[k] + 2 * 0.75e-3 / C0) & (t < t[k] + 2 * 4e-3 / C0)
    sub = 10 * np.log10(np.mean(env[band] ** 2) + 1e-60) - A
    return np.array([-A, np.nan, np.nan, sub])


def hi_images(im: Imager) -> dict:
    """Configuración → [señal, ruido_0, ...] de todas las configuraciones lineales."""
    out = {name: im.sta(tiles) for name, tiles in hi_configs().items()}
    out["mono_lineas"] = im.mono(1, False, F_NUMBER)
    out["mono_saft"] = im.mono(1, True, F_NUMBER)
    out["mono_saft_err"] = im.mono(1, True, F_NUMBER, POS_ERR)
    out["mono1mm_lineas"] = im.mono(4, False, F_NUMBER_1MM)
    out["mono1mm_saft"] = im.mono(4, True, F_NUMBER_1MM)
    for n, angles in PW_ANGLES.items():
        out[f"pw{n}"] = im.plane_wave(angles)
    return out


def cf_configs() -> list[str]:
    return [f"{lay}_{ntx}x{nrx}" for lay in ("disp", "comp") for ntx in NS for nrx in NS if nrx >= 4]


def ascan_traces(im: Imager) -> dict:
    """A-scan fijo en el centro (λ/2 y 1 mm): [señal, ruido_0, ...] analíticas."""
    out = {}
    for name, width in (("ascan", 1), ("ascan1mm", 4)):
        tr, _ = im.mono_traces(width)
        s = tr[(N_EL - width) // 2]
        blank = im.fmc.t < im.fmc.t_peak + 0.6 * im.fmc.pulse_length
        out[name] = [analytic_signal(s)] + [analytic_signal(np.where(blank, 0.0, band_noise(s.shape, im.fmc.dt, F_HI,
                                                                                            1.0, im.rng)))
                                            for _ in range(N_REAL)]
    return out


def references() -> dict:
    """Niveles de referencia de la SNR por canal (pierna sana 0, cuerpo normal); se calculan una vez."""
    path = OUT / "referencias.json"
    if path.exists():
        return json.loads(path.read_text())
    ref = {}
    fmc = FMCData.load(sim_path("normal", "tibia_sana0", "fmc3"))
    rf = np.where(_blank_mask(fmc), 0.0, fmc.rf)
    ref["fmc3"] = float(np.abs(analytic_signal(rf[N_EL // 2, N_EL // 2])).max())
    # Baja frecuencia: amplitud del primer arribo (la FAS en AT, el pulso transmitido en TT) en el
    # receptor central, que es la señal que se mide; no el máximo de la traza (la onda directa en AT)
    for f in F_AT:
        d = np.load(sim_path("normal", "tibia_sana0", f"at{f / 1e6:g}L"))
        c = int(np.argmin(np.abs(d["x"])))
        t_max = pulse_peak_time(f) + abs(d["x"][c] + AT_SRC) / EX11.C_SKIN - 2.2 / f
        ref[f"at{f / 1e6:g}"] = 10 ** (first_arrival(d["traces"][c], d["t"], t_max, 0.0, f)[1] / 20)
    for f in F_TT:
        d = np.load(sim_path("normal", "tibia_sana0", f"tt{f / 1e6:g}"))
        c = int(np.argmin(np.abs(d["x"])))
        ref[f"tt{f / 1e6:g}"] = 10 ** (first_arrival(d["traces"][c], d["t"], d["t"][-1], 0.0, f)[1] / 20)
    path.write_text(json.dumps(ref, indent=1))
    return ref


def sigmas(ref_amp: float, levels=SNR_DB) -> list[float]:
    return [0.0 if np.isinf(s) else ref_amp * 10 ** (-s / 20) for s in levels]


def _z_axis(fmc: FMCData, bottom: float) -> np.ndarray:
    return np.arange(fmc.array_z + 0.5e-3, bottom, PX)


def analyse_hi(body: str, name: str, ref: dict) -> dict:
    """Estadísticas (niveles × realizaciones × estadísticas) de todas las configuraciones de una FMC."""
    p = build(body, name, "fmc3")
    fmc = FMCData.load(sim_path(body, name, "fmc3"))
    z_img = _z_axis(fmc, p.grid.z0 + p.grid.depth - (NPML + 3) * p.grid.dx)
    seed = samples(body)[name][3]
    im = Imager(fmc, X_IMG, z_img, seed)
    sig = sigmas(ref["fmc3"])
    res, figs = {}, {}
    all_ims = hi_images(im)
    for cfg, ims in all_ims.items():
        st = np.zeros((len(sig), N_REAL, len(HI_STATS)))
        for li, s in enumerate(sig):
            for r in range(N_REAL):
                st[li, r] = image_stats(np.abs(ims[0] + s * ims[1 + r]), z_img, X_IMG, fmc.array_z)
        res[cfg] = st
        if cfg in FIG_CFGS:
            figs[cfg] = np.stack([np.abs(ims[0] + s * ims[1]) for s in sig]).astype(np.float32)
    tiles = hi_configs()
    for cfg in cf_configs():
        parts = im.channels(tiles[cfg])
        st = np.zeros((len(sig), N_REAL, len(HI_STATS)))
        for li, s in enumerate(sig):
            for r in range(N_REAL):
                st[li, r] = image_stats(np.abs(im.sta_cf(tiles[cfg], parts, s, r)), z_img, X_IMG, fmc.array_z)
        res[cfg + "_cf"] = st
        if cfg + "_cf" in FIG_CFGS:
            figs[cfg + "_cf"] = np.stack([np.abs(im.sta_cf(tiles[cfg], parts, s, 0)) for s in sig]).astype(np.float32)
    for cfg, trs in ascan_traces(im).items():
        res[cfg] = np.array([[ascan_stats(trs[0] + s * trs[1 + r], fmc) for r in range(N_REAL)] for s in sig])
    out = {"stats": res, "figs": figs, "z_img": z_img, "x_img": X_IMG, "array_z": fmc.array_z,
           "a_cortical": float(np.abs(analytic_signal(im.rf[N_EL // 2, N_EL // 2])).max())}
    twin = [n for n, v in samples(body).items() if v[4] == name]
    if twin:
        out["gt"] = ground_truth(body, twin[0], p, all_ims, z_img, sig)
    return out


def ground_truth(body, twin, p, a: dict, z_img, sig) -> dict:
    """Métricas con verdad de terreno: cambio por la lesión frente a la gemela sana, y frente al ruido, en la ROI.

    ``cambio`` = 10·log10(Σ(env_fract − env_gemela)² / Σ env_gemela²) en la ROI (sin ruido);
    ``ruido`` = lo mismo entre la gemela con ruido y sin ruido. ``D_ruido = cambio − ruido`` (dB):
    > 0 si el cambio que produce la lesión supera al que produce el ruido electrónico.
    """
    p_twin = build(body, twin, "fmc3")
    fmc_t = FMCData.load(sim_path(body, twin, "fmc3"))
    im_t = Imager(fmc_t, X_IMG, z_img, samples(body)[twin][3] + 1)
    roi = dilate(resample_mask(p, p.labels != p_twin.labels, X_IMG, z_img), int(round(1e-3 / PX)))
    roi &= (np.abs(X_IMG) <= X_STAT)[None, :]
    b = hi_images(im_t)
    out = {}
    for cfg in a:
        fr, tw = np.abs(a[cfg][0])[roi], np.abs(b[cfg][0])[roi]
        change = residual_db(tw, fr)
        noise = [np.mean([residual_db(tw, np.abs(b[cfg][0] + s * b[cfg][1 + r])[roi]) for r in range(N_REAL)])
                 if s > 0 else -np.inf for s in sig]
        out[cfg] = np.array([change, *noise])
    return out


def analyse_psf() -> dict:
    """Resolución lateral (FWHM a -6 dB), lóbulo secundario máximo y ganancia de SNR de cada configuración."""
    fmc = FMCData.load(sim_path("agua", "psf", "psf"))
    _, arr, (_, zp) = _psf_setup()
    z_img = np.arange(zp - 1.5e-3, zp + 1.5e-3, PX / 2)
    im = Imager(fmc, X_PSF, z_img, 7, c0=1500.0)
    out = {}
    for cfg, ims in hi_images(im).items():
        env = np.abs(ims[0])
        j, i = np.unravel_index(np.argmax(env), env.shape)
        prof = env[j]
        fw = fwhm_1d(X_PSF, prof)
        lo = i
        while lo > 0 and prof[lo - 1] < prof[lo]:
            lo -= 1
        hi = i
        while hi < prof.size - 1 and prof[hi + 1] < prof[hi]:
            hi += 1
        side = np.concatenate([prof[:lo], prof[hi + 1:]])
        psl = 20 * np.log10(side.max() / prof[i]) if side.size else -np.inf
        noise_rms = np.sqrt(np.mean([np.mean(np.abs(n) ** 2) for n in ims[1:]]))
        out[cfg] = np.array([fw, psl, 20 * np.log10(prof[i] / noise_rms)])
    g11 = out["disp_1x1"][2]
    for v in out.values():
        v[2] -= g11  # ganancia de SNR respecto a un solo elemento emisor-receptor
    return out


# --------------------------------------------------------------------------
# Baja frecuencia: transmisión axial (AT) y a través (TT)
# --------------------------------------------------------------------------
LOW_SNR_DB = (np.inf, 40.0, 20.0)
AT_NS = (1, 2, 4, 8, 17)
# Receptores útiles en AT: a menos de ~49 mm del emisor el primer arribo por el hueso (FAS) no se
# separa de la onda directa por la piel (medido: FAS detectable en el 0-70 % de las piernas a
# 25-41 mm y en el 100 % desde 49 mm a 0.5 MHz), así que cada emisor usa los 17 receptores lejanos
AT_MIN_OFFSET = 49e-3
TT_NS = (1, 2, 4, 8, 25)
AT_STATS = ("tiempo", "lentitud", "salto", "atenuacion")
# Velocidades de prueba del apilado (velocidad aparente del FAS medida en las piernas sanas: 2500-3600 m/s)
C_STACK = np.arange(2400.0, 4201.0, 200.0)
TT_STATS = ("tof", "energia")


def first_arrival(s: np.ndarray, t: np.ndarray, t_max: float, sigma: float, f0: float) -> tuple[float, float]:
    """Instante del primer arribo (umbral) y su amplitud (dB) antes de ``t_max``; (nan, nan) si no se detecta
    o si no queda 1.5 periodos antes de ``t_max`` (criterio del ejemplo 11).

    Umbral = máx(25 % del máximo del tramo, 4σ del ruido): un detector de umbral fijo
    referido a la señal, como el de los equipos de ultrasonometría axial.
    """
    seg = np.abs(s[t < t_max])
    if seg.size < 4:
        return np.nan, np.nan
    peak = seg.max()
    floor = max(4 * sigma, 10 ** (-50 / 20) * np.abs(s).max())  # como el ejemplo 11: -50 dB del máximo
    if peak < floor:
        return np.nan, np.nan
    thr = max(0.25 * peak, floor)
    k = int(np.argmax(seg >= thr))
    if k == 0:
        return np.nan, np.nan
    frac = (thr - seg[k - 1]) / (seg[k] - seg[k - 1] + 1e-30)
    dt = t[1] - t[0]
    tof = t[k - 1] + frac * dt
    if tof > t_max - 1.5 / f0:  # el pulso no cabe antes de t_max (la onda directa): no separable
        return np.nan, np.nan
    win = (t >= tof) & (t < tof + 1.5 / f0)
    return float(tof), float(20 * np.log10(np.abs(s[win]).max() + 1e-30))


def _fit(x, y):
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.full_like(y, np.nan)
    a, b = np.polyfit(x[ok], y[ok], 1)
    return float(a), y - (a * x + b)


def stacked_arrival(tr: np.ndarray, t: np.ndarray, off: np.ndarray, idx: np.ndarray, sigma: float,
                    f0: float) -> float:
    """Tiempo del primer arribo de las trazas ``idx`` apiladas, referido al receptor más lejano.

    Cada traza se retrasa ``(off_ref − off_j)/c`` para alinear una onda de velocidad aparente
    ``c`` y se promedian: el ruido independiente de n canales baja √n, que es lo que un
    equipo con varios receptores gana frente a uno solo. ``c`` se elige entre ``C_STACK`` por
    la mayor amplitud del primer arribo (apilado inclinado). Con un receptor es su primer arribo.
    """
    tp = pulse_peak_time(f0)
    ref = off[idx].max()
    if idx.size == 1:
        return first_arrival(tr[idx[0]], t, tp + ref / EX11.C_SKIN - 2.2 / f0, sigma, f0)[0]
    best_tof, best_amp = np.nan, -np.inf
    for c in C_STACK:
        lag = (ref - off[idx]) / c
        stack = fractional_delay(tr[idx], lag, t[1] - t[0]).mean(0)
        # la onda directa por la piel de cada receptor, retrasada igual que su traza, marca el límite
        t_max = tp + np.min(off[idx] / EX11.C_SKIN + lag) - 2.2 / f0
        tof, amp = first_arrival(stack, t, t_max, sigma / np.sqrt(idx.size), f0)
        if np.isfinite(tof) and amp > best_amp:
            best_tof, best_amp = tof, amp
    return best_tof


def _at_dir(stack_tof: float, tof, amp, off, idx) -> np.ndarray:
    """(tiempo apilado, lentitud, salto, atenuación) de una dirección con los receptores ``idx``."""
    if idx.size == 1:
        return np.array([stack_tof, np.nan, np.nan, -amp[idx[0]]])
    slope, res = _fit(off[idx], tof[idx])
    aslope, _ = _fit(off[idx], amp[idx])
    jump = np.nanmax(np.abs(np.diff(res))) if idx.size >= 4 and np.isfinite(res).sum() >= 3 else np.nan
    return np.array([stack_tof, slope, jump, -aslope])


def analyse_at(body: str, name: str, f0: float, ref: dict) -> dict:
    key = f"at{f0 / 1e6:g}"
    dL, dR = (np.load(sim_path(body, name, key + s)) for s in "LR")
    t, x = dL["t"], dL["x"]
    rng = np.random.default_rng(samples(body)[name][3] + int(f0))
    tp = pulse_peak_time(f0)
    far = np.nonzero(x + AT_SRC >= AT_MIN_OFFSET - 1e-6)[0]  # receptores lejanos al emisor izquierdo
    subsets = {n: far[-1:] if n == 1 else far[np.round(np.linspace(0, far.size - 1, n)).astype(int)] for n in AT_NS}
    levels = sigmas(ref[key], LOW_SNR_DB)
    out = {f"{key}_{m}_{n}": np.zeros((len(levels), N_REAL, len(AT_STATS))) for m in ("uni", "bi") for n in AT_NS}
    for li, s in enumerate(levels):
        for r in range(N_REAL):
            sides = []  # por emisor: {n: estadísticas con sus n receptores lejanos}
            for d, src, mirror in ((dL, -AT_SRC, False), (dR, AT_SRC, True)):
                tr = d["traces"] + (s * band_noise(d["traces"].shape, t[1] - t[0], f0, 1.0, rng) if s else 0.0)
                off = np.abs(x - src)
                fa = [first_arrival(tr[c], t, tp + off[c] / EX11.C_SKIN - 2.2 / f0, s, f0) for c in range(x.size)]
                tof, amp = np.array([v[0] for v in fa]), np.array([v[1] for v in fa])
                side = {}
                for n, idx in subsets.items():
                    idx = x.size - 1 - idx if mirror else idx  # los receptores simétricos, lejanos a este emisor
                    side[n] = _at_dir(stacked_arrival(tr, t, off, idx, s, f0), tof, amp, off, idx)
                sides.append(side)
            for n in AT_NS:
                out[f"{key}_uni_{n}"][li, r] = sides[0][n]
                # Bidireccional: 2 emisores y 2n receptores; la media compensa la inclinación del hueso
                out[f"{key}_bi_{n}"][li, r] = 0.5 * (sides[0][n] + sides[1][n])
    return out


def analyse_tt(body: str, name: str, f0: float, ref: dict) -> dict:
    key = f"tt{f0 / 1e6:g}"
    d = np.load(sim_path(body, name, key))
    t, x, traces = d["t"], d["x"], d["traces"]
    rng = np.random.default_rng(samples(body)[name][3] + int(f0) + 1)
    out = {}
    span = 2 * TT_RX
    for n in TT_NS:
        if n == 1:
            groups = [np.abs(x) <= 5e-3 + 1e-6]
            centers = np.array([0.0])
        else:
            centers = np.linspace(-TT_RX + span / n / 2, TT_RX - span / n / 2, n) if n < x.size else x
            groups = [np.abs(x - c) <= span / n / 2 + 1e-6 for c in centers]
        el = np.stack([traces[g].mean(0) for g in groups])
        st = np.zeros((len(LOW_SNR_DB), N_REAL, len(TT_STATS)))
        for li, s in enumerate(sigmas(ref[key], LOW_SNR_DB)):
            for r in range(N_REAL):
                tr = el + (s * band_noise(el.shape, t[1] - t[0], f0, 1.0, rng) if s else 0.0)
                tof, amp = np.array([first_arrival(tr[c], t, t[-1], s, f0) for c in range(n)]).T
                if n == 1:
                    st[li, r] = [-tof[0], -amp[0]]  # la reacción endóstica adelanta la llegada
                elif n == 2:
                    st[li, r] = [abs(tof[1] - tof[0]), abs(amp[1] - amp[0])]
                else:
                    _, res = _fit(centers, tof)
                    st[li, r] = [np.nanmax(np.abs(res)) if np.isfinite(res).any() else np.nan,
                                 np.nanmedian(amp) - np.nanmin(amp) if np.isfinite(amp).any() else np.nan]
        out[f"{key}_{n}"] = st
    return out


def stats_path(body: str, name: str) -> Path:
    return STATS / f"{body}__{name}.npz"


def analyse_sample(job) -> str:
    body, name = job
    path = stats_path(body, name)
    if path.exists():
        return f"(en caché) {path.name}"
    warnings.filterwarnings("ignore", message="Sólo .* puntos por longitud de onda")
    t0 = time.time()
    ref = references()
    techs = techniques_for(body, name)
    data = {}
    if "fmc3" in techs:
        hi = analyse_hi(body, name, ref)
        data.update({f"hi/{k}": v for k, v in hi["stats"].items()})
        data.update({f"fig/{k}": v for k, v in hi["figs"].items()})
        data.update({f"gt/{k}": v for k, v in hi.get("gt", {}).items()})
        data.update({"z_img": hi["z_img"], "x_img": hi["x_img"], "array_z": hi["array_z"],
                     "a_cortical": hi["a_cortical"]})
    for f in F_AT:
        if f"at{f / 1e6:g}L" in techs:
            data.update({f"low/{k}": v for k, v in analyse_at(body, name, f, ref).items()})
    for f in F_TT:
        if f"tt{f / 1e6:g}" in techs:
            data.update({f"low/{k}": v for k, v in analyse_tt(body, name, f, ref).items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.rename(path)
    return f"{path.name} ({time.time() - t0:.0f} s)"


# --------------------------------------------------------------------------
# AUC, tablas y figuras
# --------------------------------------------------------------------------
GROUPS = {
    "superficial": ("control_desplazada", "fisura_superficial"),
    "desplazada": ("control_desplazada",),
    "fisura": ("fisura_superficial",),
    "interna": ("intracortical", "estres_endostica"),
    "intracortical": ("intracortical",),
    "estres": ("estres_endostica",),
    "lejana": ("cortical_lejana",),
    "contusion": ("contusion_osea",),
}
N_BOOT = 2000
NAMED = {  # configuración → (descripción, disparos, canales de recepción)
    "ascan": ("A-scan fijo, elemento λ/2", 1, 1),
    "ascan1mm": ("A-scan fijo, elemento 1 mm", 1, 1),
    "mono_lineas": ("Monoestático λ/2 movido (32 pos.), sin SAFT", 32, 1),
    "mono_saft": ("Monoestático λ/2 movido, SAFT", 32, 1),
    "mono_saft_err": ("Ídem, posición con error σ = 0.1 mm", 32, 1),
    "mono1mm_lineas": ("Monoestático 1 mm movido (29 pos.), sin SAFT", 29, 1),
    "mono1mm_saft": ("Monoestático 1 mm movido, SAFT", 29, 1),
    "disp_1x32": ("1 emisor + 32 receptores", 1, 32),
    "disp_32x1": ("32 emisores (uno a uno) + 1 receptor", 32, 1),
    "pw1": ("Onda plana, 1 ángulo", 1, 32),
    "pw3": ("Onda plana, 3 ángulos", 3, 32),
    "pw7": ("Onda plana, 7 ángulos", 7, 32),
    "disp_4x4": ("STA dispersa 4×4", 4, 4),
    "disp_8x8": ("STA dispersa 8×8", 8, 8),
    "comp_8x8": ("STA compacta 8×8 movida", 8, 8),
    "comp_16x16": ("STA compacta 16×16 movida", 16, 16),
    "disp_32x32": ("STA completa 32×32", 32, 32),
    "disp_32x32_cf": ("STA completa 32×32 + CF", 32, 32),
    "disp_8x8_cf": ("STA dispersa 8×8 + CF", 8, 8),
}


def group_members(body: str, group: str) -> tuple[list[str], list[str]]:
    lesions = GROUPS[group]
    preset = CASES[lesions[0]][0]
    s = samples(body)
    pos = [n for n, v in s.items() if v[1] in lesions and v[4] is None]
    neg = [n for n, v in s.items() if v[1] is None and v[4] is None and v[0] == preset]
    return pos, neg


def auc_ci(pos: np.ndarray, neg: np.ndarray, rng) -> tuple[float, float, float]:
    """AUC de Mann-Whitney (mayor = fractura) e IC 95 % por bootstrap de fantomas.

    ``pos``/``neg``: (fantomas, realizaciones de ruido). Las realizaciones de un fantoma no
    son independientes: se remuestrean fantomas enteros (bootstrap por conglomerados). Un
    valor no medible (nan) cuenta como el menos sospechoso.
    """
    p = np.where(np.isfinite(pos), pos, -np.inf)
    n = np.where(np.isfinite(neg), neg, -np.inf)
    gt = (p[:, None, :, None] > n[None, :, None, :]).mean((2, 3))
    eq = (p[:, None, :, None] == n[None, :, None, :]).mean((2, 3))
    M = gt + 0.5 * eq
    ip = rng.integers(0, M.shape[0], (N_BOOT, M.shape[0]))
    ineg = rng.integers(0, M.shape[1], (N_BOOT, M.shape[1]))
    boot = M[ip[:, :, None], ineg[:, None, :]].mean((1, 2))
    return float(M.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load_stats(body: str) -> dict:
    out = {}
    for n, v in samples(body).items():
        if v[4] is None and stats_path(body, n).exists():
            out[n] = dict(np.load(stats_path(body, n)))
    return out


def compute_aucs(bodies) -> dict:
    """{cuerpo: {grupo: {clave: array (niveles, estadísticas, 3) con AUC, IC inferior, IC superior}}}."""
    rng = np.random.default_rng(0)
    res = {}
    for body in bodies:
        data = load_stats(body)
        res[body] = {}
        for group in GROUPS:
            pos, neg = group_members(body, group)
            pos, neg = [n for n in pos if n in data], [n for n in neg if n in data]
            keys = sorted(set.intersection(*(set(data[n]) for n in pos + neg))) if pos and neg else []
            keys = [k for k in keys if k.startswith(("hi/", "low/"))]
            g = {"n_pos": len(pos), "n_neg": len(neg)}
            for k in keys:
                P = np.stack([data[n][k] for n in pos])  # (fantomas, niveles, R, estadísticas)
                N = np.stack([data[n][k] for n in neg])
                g[k] = np.array([[auc_ci(P[:, li, :, si], N[:, li, :, si], rng) for si in range(P.shape[3])]
                                 for li in range(P.shape[1])])
            res[body][group] = g
    return res


def _fmt(a) -> str:
    if a is None or not np.isfinite(a[0]):
        return "—"
    return f"{a[0]:.2f} [{a[1]:.2f}-{a[2]:.2f}]"


def _get(aucs, body, group, key, level, stat):
    g = aucs.get(body, {}).get(group, {})
    return g[key][level, stat] if key in g else None


def _get_nan(aucs, body, group, key, level, stat) -> np.ndarray:
    """Como :func:`_get`, pero (nan, nan, nan) si la configuración no existe (para las figuras)."""
    v = _get(aucs, body, group, key, level, stat)
    return np.full(3, np.nan) if v is None else v


def tables(aucs: dict, psf: dict, gt: dict, bodies) -> str:
    """Tablas en Markdown (se copian al documento)."""
    L = []
    hs = {s: i for i, s in enumerate(HI_STATS)}
    lv = {s: i for i, s in enumerate(SNR_DB)}
    for body in bodies:
        a = aucs[body]
        L.append(f"\n### 3 MHz, cuerpo {body}: configuraciones con nombre propio "
                 f"(positivos/negativos: superficial {a['superficial']['n_pos']}/{a['superficial']['n_neg']}, "
                 f"interna {a['interna']['n_pos']}/{a['interna']['n_neg']}, "
                 f"contusión {a['contusion']['n_pos']}/{a['contusion']['n_neg']})\n")
        for snr in (20.0, 0.0):
            L.append(f"\nSNR por canal {snr:g} dB. AUC [IC 95 %]:\n")
            L.append("| Configuración | Disparos | Canales rx | FWHM lat. (mm) | Lóbulo máx. (dB) | Ganancia SNR (dB) "
                     "| Superficial: caída | Superficial: escalón | Interna: sub. local | Contusión: sub. global |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for key, (desc, ntx, nrx) in NAMED.items():
                ps = psf.get(key.removesuffix("_cf"), [np.nan] * 3) if not key.startswith("ascan") else [np.nan] * 3
                cells = [_fmt(_get(aucs, body, grp, f"hi/{key}", lv[snr], hs[st])) for grp, st in
                         (("superficial", "caida"), ("superficial", "escalon"), ("interna", "sub_local"),
                          ("contusion", "sub_global"))]
                L.append(f"| {desc} | {ntx} | {nrx} | {ps[0] * 1e3:.2f} | {ps[1]:.0f} | {ps[2]:.1f} | "
                         + " | ".join(cells) + " |")
        for group, stat in (("desplazada", "escalon"), ("fisura", "escalon"), ("superficial", "caida")):
            for snr in (40.0, 20.0, 0.0):
                for lay in ("disp", "comp"):
                    L.append(f"\nAUC ({group}, {stat}), {'dispersa' if lay == 'disp' else 'compacta movida'}, "
                             f"SNR {snr:g} dB, cuerpo {body}: filas N_tx, columnas N_rx\n")
                    L.append("| N_tx \\ N_rx | " + " | ".join(str(n) for n in NS) + " |")
                    L.append("|---" * (len(NS) + 1) + "|")
                    for ntx in NS:
                        row = [_get(aucs, body, group, f"hi/{lay}_{ntx}x{nrx}", lv[snr], hs[stat]) for nrx in NS]
                        L.append(f"| {ntx} | " + " | ".join("—" if r is None else f"{r[0]:.2f}" for r in row) + " |")
        L.append(f"\n### 3 MHz, cuerpo {body}: AUC por tipo de lesión (SNR 20 dB; mejor de las 4 estadísticas "
                 "indicado sólo como cota optimista)\n")
        L.append("| Configuración | " + " | ".join(f"{g} ({s})" for g in ("desplazada", "fisura", "intracortical",
                                                                         "estres", "contusion")
                                                  for s in ("caida", "sub_local")) + " |")
        L.append("|---" * 11 + "|")
        for key in ("ascan", "mono_lineas", "mono_saft", "disp_1x32", "pw7", "disp_32x32", "disp_32x32_cf"):
            cells = [_fmt(_get(aucs, body, g, f"hi/{key}", lv[20.0], hs[s])) for g in
                     ("desplazada", "fisura", "intracortical", "estres", "contusion") for s in ("caida", "sub_local")]
            L.append(f"| {NAMED[key][0]} | " + " | ".join(cells) + " |")
    # Baja frecuencia
    llv = {s: i for i, s in enumerate(LOW_SNR_DB)}
    for body in bodies:
        a = aucs[body]
        for snr in (40.0, 20.0):
            L.append(f"\n### Baja frecuencia, cuerpo {body}, SNR por canal {snr:g} dB "
                     f"(AT: estadística 'tiempo' = primer arribo de las señales apiladas, referido al receptor "
                     f"más lejano; "
                     f"TT: 'tof' = adelanto o anomalía lateral del tiempo de vuelo)\n")
            L.append("| Técnica | N rx | Superficial | Interna | Contusión | Lejana |")
            L.append("|---|---|---|---|---|---|")
            for f in F_AT:
                for mode in ("uni", "bi"):
                    for n in AT_NS:
                        key = f"low/at{f / 1e6:g}_{mode}_{n}"
                        if key not in a["superficial"]:
                            continue
                        cells = [_fmt(_get(aucs, body, g, key, llv[snr], 0)) for g in
                                 ("superficial", "interna", "contusion")]
                        desc = "bidireccional (2 tx)" if mode == "bi" else "unidireccional (1 tx)"
                        nrx = 2 * n if mode == "bi" else n
                        L.append(f"| AT {f / 1e6:g} MHz {desc} | {nrx} | " + " | ".join(cells) + " | — |")
            for f in F_TT:
                for n in TT_NS:
                    key = f"low/tt{f / 1e6:g}_{n}"
                    cells = [_fmt(_get(aucs, body, g, key, llv[snr], 0)) for g in
                             ("superficial", "interna", "contusion", "lejana")]
                    L.append(f"| TT {f / 1e6:g} MHz | {n} | " + " | ".join(cells) + " |")
        L.append(f"\n### AT, cuerpo {body}, fracturas superficiales, SNR 40 dB: AUC de cada estadística\n")
        L.append("| Técnica | N rx | " + " | ".join(AT_STATS) + " |")
        L.append("|---" * (2 + len(AT_STATS)) + "|")
        for f in F_AT:
            for n in AT_NS:
                cells = [_fmt(_get(aucs, body, "superficial", f"low/at{f / 1e6:g}_uni_{n}", llv[40.0], si))
                         for si in range(len(AT_STATS))]
                L.append(f"| AT {f / 1e6:g} MHz unidireccional | {n} | " + " | ".join(cells) + " |")
        L.append(f"\n### Baja frecuencia, cuerpo {body}, por lesión (SNR 40 dB; AT 'tiempo', TT 'tof')\n")
        L.append("| Técnica | desplazada | fisura | intracortical | estrés | contusión | lejana |")
        L.append("|---|---|---|---|---|---|---|")
        for key, desc in (("low/at0.5_uni_1", "AT 0.5 MHz, 1 tx + 1 rx"),
                          ("low/at0.5_bi_4", "AT 0.5 MHz bidir., 2 tx + 8 rx"),
                          ("low/at1_uni_1", "AT 1 MHz, 1 tx + 1 rx"), ("low/at1_bi_4", "AT 1 MHz bidir., 2 tx + 8 rx"),
                          ("low/tt0.5_1", "TT 0.5 MHz, 1 rx"), ("low/tt0.5_8", "TT 0.5 MHz, 8 rx")):
            cells = [_fmt(_get(aucs, body, g, key, llv[40.0], 0)) for g in
                     ("desplazada", "fisura", "intracortical", "estres", "contusion", "lejana")]
            L.append(f"| {desc} | " + " | ".join(cells) + " |")
    if gt:
        L.append("\n### Verdad de terreno (cuerpo normal, lesión central frente a su gemela sana): "
                 "cambio en la ROI y D_ruido = cambio − efecto del ruido (dB)\n")
        L.append("| Configuración | " + " | ".join(f"{c}: cambio / D@20 / D@0" for c in LESIONS_HI) + " |")
        L.append("|---" * (len(LESIONS_HI) + 1) + "|")
        for key in ("mono_lineas", "mono_saft", "disp_1x32", "pw1", "pw7", "disp_4x4", "comp_8x8", "disp_32x32"):
            cells = []
            for c in LESIONS_HI:
                v = gt.get(c, {}).get(key)
                cells.append("—" if v is None else f"{v[0]:.1f} / {v[0] - v[3]:+.1f} / {v[0] - v[4]:+.1f}")
            L.append(f"| {NAMED.get(key, (key,))[0]} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def load_gt() -> dict:
    out = {}
    for c in LESIONS_HI:
        path = stats_path("normal", f"{c}_p1")
        if path.exists():
            d = np.load(path)
            out[c] = {k[3:]: d[k] for k in d.files if k.startswith("gt/")}
    return out


def figures(aucs: dict, psf: dict, bodies) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_dir = ROOT / "docs" / "img"
    hs = {s: i for i, s in enumerate(HI_STATS)}
    lv = {s: i for i, s in enumerate(SNR_DB)}

    # 1. Imágenes de ejemplo (cuerpo normal, SNR 20 dB)
    rows = [("tibia_sana0", "sana"), ("control_desplazada_p1", "desplazada 1 mm"),
            ("fisura_superficial_p1", "fisura superficial"), ("estres_endostica_p1", "estrés endóstica")]
    cfgs = ("mono_lineas", "mono_saft", "disp_1x32", "pw1", "pw7", "disp_4x4", "comp_8x8", "disp_32x32")
    titles = {"mono_lineas": "monoestático\nsin SAFT", "mono_saft": "monoestático\nSAFT", "disp_1x32": "1 tx + 32 rx",
              "pw1": "onda plana\n1 ángulo", "pw7": "onda plana\n7 ángulos", "disp_4x4": "dispersa 4×4",
              "comp_8x8": "compacta 8×8\nmovida", "disp_32x32": "STA 32×32"}
    fig, axs = plt.subplots(len(rows), len(cfgs), figsize=(1.9 * len(cfgs), 2.3 * len(rows)), squeeze=False)
    for r, (name, label) in enumerate(rows):
        path = stats_path("normal", name)
        if not path.exists():
            continue
        d = np.load(path)
        z, x = d["z_img"] * 1e3, d["x_img"] * 1e3
        for c, cfg in enumerate(cfgs):
            ax = axs[r, c]
            env = d[f"fig/{cfg}"][lv[20.0]]
            db = 20 * np.log10(env / env.max() + 1e-6)
            ax.imshow(db, vmin=-40, vmax=0, cmap="gray", aspect="auto", extent=(x[0], x[-1], z[-1], z[0]))
            if r == 0:
                ax.set_title(titles[cfg], fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{label}\nz [mm]", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Tibia, cuerpo normal, 3 MHz, SNR por canal 20 dB (dB respecto al máximo, 40 dB de rango)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(img_dir / "estudio_imagenes.png", dpi=90)
    plt.close(fig)

    # 2. Mapas de AUC N_tx × N_rx
    fig, axs = plt.subplots(len(bodies), 4, figsize=(13, 3.2 * len(bodies)), squeeze=False)
    for r, body in enumerate(bodies):
        for c, (lay, snr) in enumerate((("disp", 20.0), ("comp", 20.0), ("disp", 0.0), ("comp", 0.0))):
            ax = axs[r, c]
            M = np.array([[_get_nan(aucs, body, "desplazada", f"hi/{lay}_{a}x{b}", lv[snr], hs["escalon"])[0]
                           for b in NS] for a in NS])
            ax.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis", origin="lower")
            for i in range(len(NS)):
                for j in range(len(NS)):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="w" if M[i, j] < 0.8 else "k")
            ax.set_xticks(range(len(NS)), NS)
            ax.set_yticks(range(len(NS)), NS)
            ax.set_xlabel("N_rx")
            ax.set_ylabel("N_tx")
            ax.set_title(f"{body}, {'dispersa' if lay == 'disp' else 'compacta movida'}, SNR {snr:g} dB", fontsize=9)
    fig.suptitle("AUC de la fractura desplazada 1 mm con el escalón de la línea cortical", fontsize=10)
    fig.tight_layout()
    fig.savefig(img_dir / "estudio_auc_mapa.png", dpi=90)
    plt.close(fig)

    # 3. Resolución, lóbulos y ganancia de SNR
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for lay, style in (("disp", "-o"), ("comp", "--s")):
        for kind, color in (("NxN", "C0"), ("1xN", "C1"), ("Nx1", "C2")):
            keys = [f"{lay}_{n}x{n}" if kind == "NxN" else (f"{lay}_1x{n}" if kind == "1xN" else f"{lay}_{n}x1")
                    for n in NS]
            vals = np.array([psf[k] for k in keys])
            lab = f"{'dispersa' if lay == 'disp' else 'compacta'} {kind.replace('x', '×')}"
            for ax, col, scale in zip(axs, range(3), (1e3, 1, 1)):
                ax.plot(NS, vals[:, col] * scale, style, color=color, label=lab, ms=4)
    for key, mk in (("mono_saft", "*"), ("mono1mm_saft", "P"), ("pw1", "^"), ("pw7", "v")):
        for ax, col, scale in zip(axs, range(3), (1e3, 1, 1)):
            ax.plot([32], [psf[key][col] * scale], mk, color="k", ms=8, label=NAMED[key][0])
    for ax, lab in zip(axs, ("FWHM lateral a -6 dB [mm]", "lóbulo secundario máximo [dB]",
                             "ganancia de SNR frente a 1 elemento [dB]")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS, NS)
        ax.set_xlabel("N (elementos)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axs[0].set_yscale("log")
    axs[2].legend(fontsize=6, ncol=2)
    fig.suptitle("Dispersor puntual en agua a 6 mm, 3 MHz (sonda de 32 elementos λ/2, apertura 8.2 mm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(img_dir / "estudio_resolucion.png", dpi=90)
    plt.close(fig)

    # 4. Baja frecuencia: AUC frente al número de receptores
    llv = {s: i for i, s in enumerate(LOW_SNR_DB)}
    groups = ("superficial", "interna", "contusion")
    fig, axs = plt.subplots(len(bodies), len(groups), figsize=(13, 3.4 * len(bodies)), squeeze=False)
    for r, body in enumerate(bodies):
        for c, group in enumerate(groups):
            ax = axs[r, c]
            series = [(f"AT {f / 1e6:g} MHz {m}", [f"low/at{f / 1e6:g}_{m}_{n}" for n in AT_NS], AT_NS, st)
                      for f, st in zip(F_AT, ("-o", "-s")) for m in ("uni", "bi")]
            series += [(f"TT {f / 1e6:g} MHz", [f"low/tt{f / 1e6:g}_{n}" for n in TT_NS], TT_NS, "--^") for f in F_TT]
            for k, (lab, keys, ns, st) in enumerate(series):
                mult = 2 if "bi" in lab else 1  # bidireccional: n receptores por emisor
                pts = [(mult * n, _get(aucs, body, group, key, llv[40.0], 0)) for n, key in zip(ns, keys)]
                pts = [(n, v) for n, v in pts if v is not None]
                if not pts:
                    continue
                n, v = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                off = 1 + 0.04 * (k - 2)
                ax.errorbar(n * off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=st, ms=4,
                            capsize=2, lw=1, label=lab.replace("uni", "unidir.").replace("bi", "bidir."))
            ax.axhline(0.5, color="k", lw=0.7)
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("receptores")
            ax.set_ylabel("AUC")
            ax.set_title(f"{body}: {group}", fontsize=9)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Baja frecuencia (SNR 40 dB): AT = tiempo del primer arribo (señales apiladas); "
                 "TT = tiempo de vuelo (1 rx) o su anomalía lateral", fontsize=9)
    fig.tight_layout()
    fig.savefig(img_dir / "estudio_baja_frecuencia.png", dpi=90)
    plt.close(fig)

    # 5. AUC frente a la SNR por canal para las configuraciones con nombre (cuerpo normal)
    body = bodies[0]
    keys = ("ascan", "mono_lineas", "mono_saft", "mono1mm_saft", "disp_1x32", "pw1", "pw7", "disp_4x4", "comp_8x8",
            "disp_32x32", "disp_32x32_cf")
    panels = (("superficial", "caida"), ("superficial", "escalon"), ("interna", "sub_local"))
    fig, axs = plt.subplots(1, len(panels), figsize=(14, 4.6))
    snr_x = np.array([60.0 if np.isinf(s) else s for s in SNR_DB])
    for ax, (group, st) in zip(axs, panels):
        for k, key in enumerate(keys):
            v = np.array([_get_nan(aucs, body, group, f"hi/{key}", li, hs[st]) for li in range(4)])
            ax.plot(snr_x + 0.4 * (k - 5), v[:, 0], "-o", ms=3, lw=1, label=NAMED[key][0],
                    color=plt.cm.tab20(k / len(keys)))
        ax.axhline(0.5, color="k", lw=0.7)
        ax.set_xticks(snr_x, ["sin ruido", "40", "20", "0"])
        ax.set_xlabel("SNR por canal [dB]")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"{body}: {group}, estadística '{st}'", fontsize=9)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=7)  # fuera de los paneles
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(img_dir / "estudio_ruido.png", dpi=90)
    plt.close(fig)


def analyse(bodies, procesos: int) -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    references()
    jobs = [(b, n) for b in bodies for n, v in samples(b).items() if v[4] is None
            and all(sim_path(b, n, t).exists() for t in techniques_for(b, n))]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        for k, msg in enumerate(pool.map(analyse_sample, jobs)):
            print(f"[{k + 1}/{len(jobs)}, {time.time() - t0:.0f} s] {msg}", flush=True)
    psf = analyse_psf()
    aucs = compute_aucs(bodies)
    gt = load_gt()
    text = tables(aucs, psf, gt, bodies)
    (OUT / "tablas.md").write_text(text, encoding="utf-8")
    np.save(OUT / "aucs.npy", {"aucs": aucs, "psf": psf, "gt": gt}, allow_pickle=True)
    figures(aucs, psf, bodies)
    print(text)


def simulate(bodies, procesos: int) -> None:
    SIM.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs(bodies)
    print(f"{len(jobs)} simulaciones ({sum(sim_path(*j).exists() for j in jobs)} ya en caché)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=procesos) as pool:
        for k, msg in enumerate(pool.map(run_job, jobs)):
            print(f"[{k + 1}/{len(jobs)}, {(time.time() - t0) / 60:.0f} min] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuerpos", nargs="+", default=["normal", "obeso"])
    parser.add_argument("--procesos", type=int, default=1)
    parser.add_argument("--fase", choices=("simular", "analizar", "todo"), default="todo")
    args = parser.parse_args()
    if args.fase in ("simular", "todo"):
        simulate(args.cuerpos, args.procesos)
    if args.fase in ("analizar", "todo"):
        analyse(args.cuerpos, args.procesos)


if __name__ == "__main__":
    main()
