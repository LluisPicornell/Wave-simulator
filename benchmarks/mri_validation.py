"""Validación del simulador de Bloch espacial de RM (``wavesim.mri.spatial``).

Uso (desde la raíz del repositorio, con un solo hilo para medir tiempos):

    NUMBA_NUM_THREADS=1 python benchmarks/mri_validation.py            # completo, ~3 min
    NUMBA_NUM_THREADS=1 python benchmarks/mri_validation.py --quick    # sin el modo línea a línea, ~1 min

Imprime las tablas de ``docs/validacion/rm.md``. Casos:
  (a) fantoma uniforme sin artefactos frente a ``sequences.signal`` (todas las secuencias y tejidos, 1.5 y 3 T);
  (b) desplazamiento químico medido en la imagen frente a Δf/BW (un pico y seis picos);
  (c) espín-eco frente a un desvío estático y GRE frente a la función característica de la distribución;
  (d) campo ΔB0 de un cilindro infinito perpendicular a B0 frente a la solución exacta;
  (e) pierna (corte transversal y longitudinal): imágenes con y sin artefactos y visibilidad de la fractura;
  (f) tiempos de ejecución a 128x128.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from wavesim.core.grid import Grid2D
from wavesim.materials.mri import FAT_PEAKS, GAMMA_BAR, MRI
from wavesim.mri import SEQUENCES, signal, spatial, tissue_signal
from wavesim.mri import susceptibility as sus
from wavesim.mri.sequences import Sequence
from wavesim.mri.spatial import BlochPhysics, Isochromats
from wavesim.phantoms import Fracture, RingFracture, cross_section_limb, longitudinal_limb
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom

NONE = BlochPhysics.none()


# --------------------------------------------------------------------------
# (a) Fantoma uniforme
# --------------------------------------------------------------------------
def caso_a():
    print("\n## (a) Fantoma uniforme sin artefactos frente a sequences.signal (16x16, error relativo máximo)")
    g = Grid2D.from_size(8e-3, 8e-3, 0.5e-3, x0=-4e-3, z0=-4e-3)
    tissues = [k for k in MRI if MRI[k].pd > 0]
    print("| secuencia | 1.5 T | 3 T | tejidos con señal < 1e-6 (excluidos) |\n|---|---|---|---|")
    worst = 0.0
    for name in SEQUENCES:
        if name == "UTE":
            continue
        row, tiny = [], set()
        for b0 in (1.5, 3.0):
            err = 0.0
            for t in tissues:
                p = Phantom(g, np.full(g.shape, TISSUE_INDEX[t], np.int16))
                ref = tissue_signal(t, name, b0)
                if ref < 1e-6:
                    tiny.add(t)
                    continue
                im = spatial.bloch_scan(p, name, b0, snr=0, texture=0, matrix=(16, 16), physics=NONE)
                err = max(err, float(np.abs(im.magnitude - ref).max() / ref))
            row.append(err)
            worst = max(worst, err)
        print(f"| {name} | {row[0]:.1e} | {row[1]:.1e} | {', '.join(sorted(tiny)) or '-'} |")
    print(f"Peor caso: {worst:.1e} (objetivo < 1e-2)")


# --------------------------------------------------------------------------
# (b) Desplazamiento químico
# --------------------------------------------------------------------------
def _harmonic_centroid(img):
    n = img.shape[1]
    h1 = (img * np.exp(-2j * np.pi * np.arange(n) / n)[None, :]).sum()
    return (-np.angle(h1 / img.sum()) * n / (2 * np.pi)) % n


def _magnitude_centroid(img):
    m = np.abs(img)
    return float((m.sum(0) * np.arange(img.shape[1])).sum() / m.sum())


def _block_image(b0, bw, peaks):
    n, fov = 64, 32e-3
    c = -fov / 2 + (np.arange(2 * n) + 0.5) * fov / (2 * n)
    X, Z = np.meshgrid(c, c)
    blk = (np.abs(X) < 5e-3) & (np.abs(Z) < 5e-3)
    amps = np.array([a for _, a in peaks]) / sum(a for _, a in peaks)
    xs = np.concatenate([X[blk]] * len(peaks))
    zs = np.concatenate([Z[blk]] * len(peaks))
    m0 = np.concatenate([np.full(blk.sum(), a) for a in amps])
    dw = np.concatenate([np.full(blk.sum(), 2 * np.pi * p * 1e-6 * GAMMA_BAR * b0) for p, _ in peaks])
    iso = Isochromats(xs, zs, m0, 0.3, 1e3, 0.0, dw)
    cseq = spatial.cartesian_sequence(Sequence("SE", "SE", 500.0, 30.0), b0, (n, n), (fov, fov), bw)
    return spatial.reconstruct(spatial.acquire(iso, cseq), cseq, (-fov / 2 + fov / n / 2,) * 2)


def caso_b():
    print("\n## (b) Desplazamiento químico de la grasa en la lectura (bloque de 10 mm, 64x64, espín-eco)")
    print("| B0 (T) | BW (Hz/píxel) | modelo | esperado (píxeles) | medido: fase del 1.er armónico | "
          "medido: centroide de la magnitud |\n|---|---|---|---|---|---|")
    for b0 in (1.5, 3.0):
        for bw in (65.0, 130.0, 260.0):
            water = _block_image(b0, bw, ((0.0, 1.0),))
            for model in ("single", "multipeak"):
                fat = _block_image(b0, bw, FAT_PEAKS[model])
                amps = np.array([a for _, a in FAT_PEAKS[model]])
                ppm = float((amps * np.array([p for p, _ in FAT_PEAKS[model]])).sum() / amps.sum())
                exp = ppm * 1e-6 * GAMMA_BAR * b0 / bw
                ph = _harmonic_centroid(fat) - _harmonic_centroid(water)
                mg = _magnitude_centroid(fat) - _magnitude_centroid(water)
                print(f"| {b0} | {bw:g} | {model} | {exp:+.3f} | {ph:+.3f} | {mg:+.3f} |")
    print("(Seis picos en espín-eco: todos se reenfocan en TE y el centroide esperado es la media de los "
          "desplazamientos ponderada por amplitud, -3.01 ppm.)")


# --------------------------------------------------------------------------
# (c) Desvío estático y distribuciones de desvíos
# --------------------------------------------------------------------------
def _offsets(kind, k, width):
    q = (np.arange(k) + 0.5) / k
    if kind == "gauss":
        nodes, w = np.polynomial.hermite_e.hermegauss(k)
        return width * nodes, w / w.sum(), lambda t: np.exp(-0.5 * (2 * np.pi * width * t) ** 2)
    if kind == "uniforme":
        return width * (2 * q - 1), np.full(k, 1.0 / k), lambda t: np.abs(np.sinc(2 * width * t))
    if kind == "lorentz_cuantiles":
        return width * np.tan(np.pi * (q - 0.5)), np.full(k, 1.0 / k), lambda t: np.exp(-2 * np.pi * width * t)
    raise ValueError(kind)


def caso_c():
    print("\n## (c) Espín-eco frente a desvío estático; GRE frente a la distribución de desvíos (error relativo)")
    g = Grid2D.from_size(8e-3, 8e-3, 0.5e-3, x0=-4e-3, z0=-4e-3)
    X, Z = g.meshgrid()
    t1, t2 = 1.0, 0.05
    te_list = (2.0, 5.0, 10.0, 20.0, 40.0)
    cases = [("lorentziana exacta (estados), HWHM 25 Hz", None, 25.0), ("gaussiana, 48 nodos Gauss-Hermite, σ 15 Hz",
                                                                        ("gauss", 48), 15.0),
             ("uniforme ±40 Hz, 64 isocromatos", ("uniforme", 64), 40.0),
             ("lorentziana muestreada, 64 cuantiles", ("lorentz_cuantiles", 64), 25.0),
             ("lorentziana muestreada, 256 cuantiles", ("lorentz_cuantiles", 256), 25.0)]
    print("| distribución | secuencia | " + " | ".join(f"TE = {te:g} ms" for te in te_list) + " |")
    print("|---|---|" + "---|" * len(te_list))
    for label, spec, width in cases:
        if spec is None:
            iso = Isochromats(X.ravel(), Z.ravel(), 1.0, t1, t2, 2 * np.pi * width, 2 * np.pi * 37.0)

            def decay(t, w=width):
                return np.exp(-2 * np.pi * w * t)
        else:
            off, w, decay = _offsets(spec[0], spec[1], width)
            k = off.size
            iso = Isochromats(np.repeat(X.ravel(), k), np.repeat(Z.ravel(), k), np.tile(w, X.size), t1, t2, 0.0,
                              2 * np.pi * (37.0 + np.tile(off, X.size)))
        for kind in ("SE", "GRE"):
            errs = []
            for te in te_list:
                seq = Sequence(kind, kind, tr=200.0, te=te, flip_deg=90.0 if kind == "SE" else 30.0)
                cseq = spatial.cartesian_sequence(seq, 1.5, (16, 16), (8e-3, 8e-3), 1000.0)
                img = spatial.reconstruct(spatial.acquire(iso, cseq), cseq, (-4e-3, -4e-3))
                v = np.abs(img).mean() * img.size / X.size
                ref = float(signal(1.0, t1 * 1e3, t2 * 1e3, t2 * 1e3, seq))
                if kind == "GRE":
                    ref *= float(decay(te * 1e-3))
                errs.append((v - ref) / ref)
            print(f"| {label} | {kind} | " + " | ".join(f"{e:+.1e}" for e in errs) + " |")
    print("(Todos los isocromatos tienen además un desvío estático común de 37 Hz.)")


# --------------------------------------------------------------------------
# (d) Cilindro infinito perpendicular a B0
# --------------------------------------------------------------------------
def caso_d():
    print("\n## (d) Campo ΔB0 de un cilindro infinito frente a la solución exacta (Δχ = 1 ppm, error en ppm "
          "relativo a Δχ/2 = 0.5)")
    print("| malla | radio (celdas) | relleno | B0 | máx. a > 2 celdas del borde | máx. a > 5 celdas | RMS | "
          "media dentro (exacta) |\n|---|---|---|---|---|---|---|---|")
    sub = (np.arange(8) + 0.5) / 8 - 0.5
    for n, a, pad in ((128, 16.0, 1.0), (256, 32.0, 1.0), (256, 32.0, 0.5), (512, 64.0, 1.0)):
        c = np.arange(n) - n / 2 + 0.5
        frac = ((c[None, :, None, None] + sub[None, None, None, :]) ** 2
                + (c[:, None, None, None] + sub[None, None, :, None]) ** 2 <= a**2).mean(axis=(2, 3))
        X, Z = np.meshgrid(c, c)
        r = np.hypot(X, Z)
        for d in ("z", "x", (1.0, 0.0, 1.0), "y"):
            f = sus.dipole_field(frac + sus.CHI_WATER, d, padding=pad)
            ref = sus.cylinder_field(X, Z, a, 1.0, d)
            e = np.abs(f - ref)
            far2, far5 = np.abs(r - a) > 2, np.abs(r - a) > 5
            rms = np.sqrt(np.mean(e[far2] ** 2))
            inside = f[r < a - 2].mean()
            print(f"| {n}² | {a:g} | {pad:g} | {d} | {e[far2].max() / 0.5:.2%} | {e[far5].max() / 0.5:.2%} | "
                  f"{rms / 0.5:.3%} | {inside:+.5f} ({ref[r < a - 2].mean():+.5f}) |")


# --------------------------------------------------------------------------
# (e) La pierna
# --------------------------------------------------------------------------
def pixel_fraction(phantom: Phantom, mask: np.ndarray, im) -> np.ndarray:
    """Fracción de las celdas del fantoma de cada píxel de la imagen que están en ``mask``."""
    g = phantom.grid
    X, Z = g.meshgrid()
    dx, dz = im.x[1] - im.x[0], im.z[1] - im.z[0]
    ix = np.floor((X - (im.x[0] - dx / 2)) / dx).astype(int)
    iz = np.floor((Z - (im.z[0] - dz / 2)) / dz).astype(int)
    nz, nx = im.magnitude.shape
    ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    idx = (iz * nx + ix)[ok]
    tot = np.bincount(idx, minlength=nz * nx)
    hit = np.bincount(idx, weights=mask[ok].astype(float), minlength=nz * nx)
    return (hit / np.maximum(tot, 1)).reshape(nz, nx)


def _crack_profile(p: Phantom, im) -> tuple[int, int, np.ndarray]:
    """Fila de la imagen a media cortical donde la cruza la grieta, columna de la grieta y píxeles de cortical."""
    crack = pixel_fraction(p, p.masks["fracture"], im)
    cort = pixel_fraction(p, p.mask("cortical_bone"), im)
    rows = np.unique(np.nonzero(crack > 0.3)[0])
    run = rows[: int(np.argmax(np.diff(np.append(rows, rows[-1] + 2)) > 1)) + 1]  # primera cortical que cruza
    row = int(run[len(run) // 2])
    col = int(np.round(np.average(np.arange(crack.shape[1]), weights=crack[row])))
    return row, col, (cort[row] + crack[row] > 0.9)


def leg_metrics(p: Phantom, images: dict, sigma: float) -> dict:
    """Contrastes de la fractura en unidades de la desviación del ruido (CNR) y posición aparente de la grieta."""
    ref = next(iter(images.values()))
    fr = {k: pixel_fraction(p, p.mask(k), ref) for k in ("marrow", "muscle", "fat")}
    ed = pixel_fraction(p, p.masks["edema"], ref)
    hem = pixel_fraction(p, p.masks["hematoma"] & ~p.masks["fracture"], ref)
    marrow = (fr["marrow"] > 0.95) & (ed == 0)
    edema = ed > 0.9
    muscle = fr["muscle"] > 0.95
    hema = hem > 0.9
    row, col, bone_px = _crack_profile(p, ref)
    cols = np.arange(ref.magnitude.shape[1])
    near = np.abs(cols - col) <= 2
    far = (np.abs(cols - col) >= 3) & (np.abs(cols - col) * ref.voxel <= 4e-3) & bone_px
    out = {}
    for name, im in images.items():
        m = im.ideal
        prof = m[row]
        k = int(np.argmax(np.where(near, prof, -np.inf)))
        out[name] = {
            "edema": (m[edema].mean() - m[marrow].mean()) / sigma,
            "hematoma": (m[hema].mean() - m[muscle].mean()) / sigma if hema.any() else np.nan,
            "grieta": (prof[k] - np.median(prof[far])) / sigma,
            "desplazamiento": k - col,
            "grasa": m[fr["fat"] > 0.9].mean(),
        }
    return out


def cortex_widths(im, bone_center, threshold) -> tuple[int, int]:
    """Grosor aparente (píxeles) de la cortical a izquierda y derecha en la fila del centro del hueso."""
    j = int(np.argmin(np.abs(im.z - bone_center[1])))
    row = im.ideal[j]
    dark = row < threshold
    widths = []
    for step in (-1, 1):
        i = int(np.argmin(np.abs(im.x - bone_center[0])))
        while 0 < i < row.size - 1 and not dark[i]:
            i += step
        n = 0
        while 0 < i < row.size - 1 and dark[i]:
            n += 1
            i += step
        widths.append(n)
    return widths[0], widths[1]


def leg_table(p: Phantom, configs: dict, seqs, **scan) -> dict:
    """Tabla de CNR de la fractura con cada física; devuelve las imágenes por secuencia."""
    tissue = p.labels != TISSUE_INDEX["air"]
    fat = p.mask("fat")
    f0 = GAMMA_BAR * 1.5 * 1e-6
    for label, ph in configs.items():
        if ph.susceptibility:
            hz = spatial.field_map_hz(p, 1.5, ph)
            crack = p.masks["fracture"]
            out_band = np.mean(np.abs(hz[fat]) > 1.0 * f0)
            print(f"ΔB0 con {label}: tejido p1/p50/p99 = {np.percentile(hz[tissue], 1):+.0f} / "
                  f"{np.percentile(hz[tissue], 50):+.0f} / {np.percentile(hz[tissue], 99):+.0f} Hz, "
                  f"máx. |ΔB0| {np.abs(hz[tissue]).max():.0f} Hz; en la grieta {hz[crack].mean():+.0f} Hz "
                  f"({hz[crack].min():+.0f} .. {hz[crack].max():+.0f}); grasa fuera de la banda de saturación "
                  f"(|ΔB0| > 1 ppm): {out_band:.1%}")
    print("\n| secuencia | física | edema - médula | hematoma - músculo | línea de fractura - cortical "
          "(desplazamiento, píxeles) | grasa subcutánea (señal) |\n|---|---|---|---|---|---|")
    results = {}
    for s in seqs:
        sigma = tissue_signal("muscle", s) / 40.0
        images = {label: spatial.bloch_scan(p, s, snr=0, physics=ph, **scan) for label, ph in configs.items()}
        results[s] = images
        for label, v in leg_metrics(p, images, sigma).items():
            print(f"| {s} | {label} | {v['edema']:+.1f} | {v['hematoma']:+.1f} | {v['grieta']:+.1f} "
                  f"({v['desplazamiento']:+d}) | {v['grasa']:.3f} |")
    return results


def caso_e(seqs=("T1", "STIR", "T2_FS", "GRE", "PD_FS")):
    print("\n## (e) Pierna: corte transversal con fractura de la tibia, 128x128 (vóxel 0.78 mm), BW 130 Hz/píxel")
    print("CNR = diferencia de medias de la imagen sin ruido / σ del ruido con SNR = 40 en el músculo. La línea "
          "de fractura es el pico en ±2 píxeles de la grieta real, en la fila a media cortical, frente a la "
          "mediana de la cortical vecina (hasta 4 mm).")
    p = cross_section_limb("pierna", RingFracture(angle_deg=-10, gap=0.6e-3, edema_length=12e-3), dx=0.2e-3)
    p = p.replace_tissue("water", "air")
    configs = {"sin artefactos": NONE, "sólo desplazamiento químico": BlochPhysics(susceptibility=False),
               "+ χ, B0 ⊥ corte (túnel)": BlochPhysics(b0_direction="y"),
               "+ χ, B0 vertical en el plano": BlochPhysics(b0_direction="z")}
    results = leg_table(p, configs, seqs, matrix=(128, 128))
    bone = p.meta["spec"]["bones"][0]
    print("\nGrosor aparente de la cortical de la tibia en T1 (fila del centro del hueso; real 4 mm = 5.1 "
          "píxeles; umbral: mitad de la señal del músculo):")
    print("| física | izquierda (píxeles) | derecha (píxeles) |\n|---|---|---|")
    for label, im in results["T1"].items():
        left, right = cortex_widths(im, bone["center"], 0.5 * tissue_signal("muscle", "T1"))
        print(f"| {label} | {left} | {right} |")
    print("\nSeñal integrada junto a las interfaces, con susceptibilidad (B0 vertical en el plano) / sin ella, "
          "sin desplazamiento químico:")
    print("| región | PD (espín-eco) | GRE TE 5 ms | GRE TE 15 ms | GRE TE 30 ms |\n|---|---|---|---|---|")
    phz = BlochPhysics(chemical_shift=False, b0_direction="z")
    seq_list = ["PD"] + [Sequence(f"GRE{te}", "GRE", 60.0, float(te), flip_deg=20.0) for te in (5, 15, 30)]
    imgs = [(spatial.bloch_scan(p, s, matrix=(128, 128), snr=0, physics=phz),
             spatial.bloch_scan(p, s, matrix=(128, 128), snr=0, physics=NONE)) for s in seq_list]
    ref = imgs[0][0]
    zz, xx = np.meshgrid(ref.z, ref.x, indexing="ij")
    rb = np.hypot(xx - bone["center"][0], zz - bone["center"][1])
    rr = np.hypot(xx, zz)
    radius = p.meta["spec"]["radius"]
    regions = {"anillo de 3 mm alrededor de la tibia": (rb > bone["outer_radius"]) & (rb < bone["outer_radius"] + 3e-3),
               "6 mm bajo la piel (y 2 mm de aire)": (rr > radius - 6e-3) & (rr < radius + 2e-3)}
    for name, reg in regions.items():
        vals = [a.ideal[reg].sum() / b.ideal[reg].sum() for a, b in imgs]
        print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")


def caso_e_longitudinal(seqs=("T1", "STIR", "GRE")):
    print("\n## (e') Pierna: corte longitudinal de la tibia con fractura transversal, vóxel 0.3 mm, BW 130 Hz/píxel")
    p = longitudinal_limb("tibia", Fracture(), dx=0.1e-3).replace_tissue("water", "air")
    configs = {"sin artefactos": NONE, "sólo desplazamiento químico": BlochPhysics(susceptibility=False),
               "+ χ, B0 a lo largo del hueso (túnel)": BlochPhysics(b0_direction="x"),
               "+ χ, B0 perpendicular al hueso, en el plano": BlochPhysics(b0_direction="z")}
    leg_table(p, configs, seqs, voxel=0.3e-3)


# --------------------------------------------------------------------------
# (f) Tiempos
# --------------------------------------------------------------------------
def caso_f(quick: bool):
    print("\n## (f) Tiempos a 128x128 (pierna transversal, dx 0.2 mm, todos los artefactos, 1 hilo)")
    p = cross_section_limb("pierna", RingFracture(angle_deg=-10, gap=0.6e-3, edema_length=12e-3), dx=0.2e-3)
    p = p.replace_tissue("water", "air")
    print("| secuencia | modo | backend | isocromatos | campo (s) | simulación (s) | total (s) |")
    print("|---|---|---|---|---|---|---|")
    modes = [("steady", "numpy"), ("steady", "numba")]
    if not quick:
        modes += [("events", "numba"), ("events", "numpy")]
    for s in ("T1", "STIR", "GRE"):
        for mode, be in modes:
            if mode == "events" and s != "STIR":
                continue
            try:
                t0 = time.perf_counter()
                im = spatial.bloch_scan(p, s, matrix=(128, 128), physics=BlochPhysics(b0_direction="z"), mode=mode,
                                        backend=be)
                tot = time.perf_counter() - t0
            except ImportError:
                continue
            print(f"| {s} | {mode} | {be} | {im.n_isochromats} | {im.timings['campo']:.2f} | "
                  f"{im.timings['simulacion']:.2f} | {tot:.2f} |")
    if not quick:
        a = spatial.bloch_scan(p, "STIR", matrix=(128, 128), snr=0, physics=BlochPhysics(b0_direction="z"))
        b = spatial.bloch_scan(p, "STIR", matrix=(128, 128), snr=0, physics=BlochPhysics(b0_direction="z"),
                               mode="events")
        print(f"Diferencia máxima entre los modos 'steady' y 'events' (STIR): "
              f"{np.abs(a.kspace - b.kspace).max() / np.abs(a.kspace).max():.1e} del máximo del k-space")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="omite el modo línea a línea (el más lento)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    caso_a()
    caso_b()
    caso_c()
    caso_d()
    caso_e()
    caso_e_longitudinal()
    caso_f(args.quick)
    print(f"\nTiempo total: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
