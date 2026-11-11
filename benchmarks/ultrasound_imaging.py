"""Cadena de imagen de ultrasonido: PSF, speckle, quiste, modos de emisión y formadores de haz.

Uso (desde la raíz del repositorio; sólo numpy, y numba opcional):

    NUMBA_NUM_THREADS=1 python benchmarks/ultrasound_imaging.py                 # todo (~15-20 min con Numba)
    NUMBA_NUM_THREADS=1 python benchmarks/ultrasound_imaging.py --rapido   # STA cada 4 elementos, 2 semillas (~6 min)
    python benchmarks/ultrasound_imaging.py --parte psf                          # psf | speckle | quiste | capas
    NUMBA_NUM_THREADS=1 python benchmarks/ultrasound_imaging.py --parte fractura # tibia, la más cara (~25 min)

Todo ocurre en agua (c = 1500 m/s) a 2 MHz con 10 puntos por longitud de onda, 24×30 mm, sonda de
λ/2 con respuesta de elemento de 2 ciclos con ventana de Hann (``ElementResponse``) y sin TGC. Las
simulaciones FDTD (una por emisión) se hacen una vez por modo y las imágenes de las distintas
ventanas y formadores de haz se reconstruyen de los mismos datos RF.

Partes:

1. ``psf``: tres dispersores puntuales (0, 8/14/20 mm). FWHM axial y lateral a -6 dB de amplitud frente
   a la teoría (diagrama de apertura monocromático y envolvente del pulso de ida y vuelta) para onda plana,
   STA y emisión focalizada (foco fijo a 14 mm), con ventana rectangular y de Hann.
2. ``speckle``: dispersores densos (una anomalía de densidad gaussiana por celda). SNR de la envolvente
   frente al de Rayleigh (1.913) y tamaño de grano frente a la autocorrelación de la PSF.
3. ``quiste``: quiste anecoico de 3.5 mm. Contraste, CNR y gCNR con DAS, CF, DMAS y SLSC y con onda
   plana (1 y 5 ángulos), emisión focalizada y STA.
4. ``capas``: dispersor puntual bajo una capa lenta (1450 m/s sobre 1580 m/s): error de posición con
   1540 m/s fijo frente a la corrección de dos capas.
5. ``fractura``: tibia (3 MHz, 24×22 mm) sana, con fractura desplazada y con fisura; ¿mejora la visibilidad
   (``analysis.fracture_visibility``) con STA o con el factor de coherencia frente a 3 ondas planas con DAS?
   Es la parte cara (~150 simulaciones, ~25 min); ``--rapido`` usa STA cada 4 elementos. No forma parte de
   ``todo``: se pide con ``--parte fractura``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavesim.core.grid import Grid2D  # noqa: E402
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom  # noqa: E402
from wavesim.ultrasound import ElementResponse, LinearArray, TwoLayerSpeed, default_array, simulate_scan  # noqa: E402
from wavesim.ultrasound.beamforming import bmode_channels  # noqa: E402
from wavesim.ultrasound.medium import elastic_medium  # noqa: E402
from wavesim.ultrasound.quality import (RAYLEIGH_SNR, cnr, contrast_db, gcnr, point_scatterer_medium,  # noqa: E402
                                        predicted_grain_fwhm, psf_fwhm, speckle_grain_fwhm, speckle_medium,
                                        speckle_snr, theoretical_axial_fwhm, theoretical_lateral_fwhm,
                                        water_phantom)

C = 1500.0
F0 = 2e6
LAM = C / F0
PX = LAM / 8
WIDTH, DEPTH = 24e-3, 30e-3
MAX_DEPTH = 26e-3
F_NUMBER = 1.5
FOCUS = 14e-3


def setup():
    g = Grid2D.from_size(WIDTH, DEPTH, C / F0 / 10, x0=-WIDTH / 2)
    ph = water_phantom(g)
    a = default_array(ph, F0, 20, C)
    arr = LinearArray(n_elements=a.n_elements, pitch=a.pitch, z=a.z, x_center=a.x_center, f0=F0,
                      response=ElementResponse(F0, "windowed", 2.0, "hann"))
    return g, ph, arr


def acquire(ph, arr, medium, mode, **kw):
    """Simula y devuelve (scan, tiempo total en s). Imagen por defecto de la propia llamada."""
    t0 = time.time()
    sc = simulate_scan(ph, F0, mode, arr, c0=C, medium=medium, n_mechanisms=0, tgc_db_per_cm=0, pixel=PX,
                       max_depth=MAX_DEPTH, f_number=F_NUMBER, **kw)
    return sc, time.time() - t0


def reconstruct(sc, window="hann", beamformer="das", layers=None, transmits=None, rf=None, **kw):
    arr = sc.array
    return bmode_channels(sc.rf if rf is None else rf, sc.transmits if transmits is None else transmits, sc.t,
                          arr.element_x, arr.z, sc.image.x, sc.image.z, c0=C, t_peak=arr.pulse_peak_time,
                          f_number=F_NUMBER, rx_window=window, beamformer=beamformer, layers=layers, **kw)


def fmt(v, n=2):
    return f"{v * 1e3:.{n}f}"


def pct(a, b):
    return f"{100 * (a - b) / b:+.1f} %"


# --------------------------------------------------------------------------
def part_psf(quick):
    g, ph, arr = setup()
    points = [(0.0, 8e-3), (0.0, 14e-3), (0.0, 20e-3)]
    med = point_scatterer_medium(g, F0, points)
    sta_step = 4 if quick else 1
    runs = {
        "onda plana": acquire(ph, arr, med, "plane"),
        f"STA (paso {sta_step})": acquire(ph, arr, med, "sta", sta_step=sta_step),
        "focalizada F#2 @14 mm": acquire(ph, arr, med, "focused", line_step=1, focus_depth=FOCUS, tx_f_number=2.0),
    }
    dt = runs["onda plana"][0].t[1] - runs["onda plana"][0].t[0]
    ax0 = theoretical_axial_fwhm(arr.round_trip_pulse(dt), dt, C)
    ax15 = theoretical_axial_fwhm(arr.round_trip_pulse(dt), dt, C, 1.5)
    print("\n### PSF de un dispersor puntual (FWHM a -6 dB de amplitud, mm)\n")
    print(f"Pulso de ida y vuelta: FWHM axial teórico {fmt(ax0, 3)} mm (nominal), {fmt(ax15, 3)} mm "
          "(con el factor f^1.5 del dispersor 2D). λ = %.3f mm.\n" % (LAM * 1e3))
    print("| Modo | Ventana rx | z (mm) | lateral medido | lateral teoría | error | axial medido | axial teoría "
          "| error | "
          "offset x / z (mm) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, (sc, _) in runs.items():
        for window in ("rect", "hann"):
            img = reconstruct(sc, window)
            for xs, zs in points:
                r = psf_fwhm(img.envelope, img.x, img.z, xs, zs)
                if name.startswith("onda"):
                    th = theoretical_lateral_fwhm(arr.element_x, zs, LAM, F_NUMBER, window, False, xs, arr.z)
                elif name.startswith("STA"):
                    th = theoretical_lateral_fwhm(arr.element_x, zs, LAM, F_NUMBER, window, True, xs, arr.z)
                else:  # el patrón de emisión sólo se calcula con el foco: fuera de él no hay teoría simple
                    th = (theoretical_lateral_fwhm(arr.element_x, zs, LAM, F_NUMBER, window, True, xs, arr.z,
                                                   tx_f_number=2.0, tx_window="hann") if zs == FOCUS else np.nan)
                err = pct(r["lateral"], th) if np.isfinite(th) else "—"
                th_s = fmt(th) if np.isfinite(th) else "—"
                print(f"| {name} | {window} | {zs * 1e3:.0f} | {fmt(r['lateral'])} | {th_s} | {err} | "
                      f"{fmt(r['axial'], 3)} | {fmt(ax0, 3)} | {pct(r['axial'], ax0)} | "
                      f"{r['offset_x'] * 1e3:+.2f} / {r['offset_z'] * 1e3:+.2f} |")
    # Interfaz plana: sin la física 2D del dispersor puntual, la FWHM axial es exactamente la del pulso de ida y vuelta
    from wavesim.ultrasound.quality import _water_medium, fwhm_1d
    step = _water_medium(g, F0)
    step.rho[g.z >= 14e-3, :] *= 1.3
    sc = acquire(ph, arr, step, "plane")[0]
    img = reconstruct(sc, "hann")
    prof, z = img.envelope[:, np.argmin(abs(img.x))], img.z
    sel = np.abs(z - 14e-3) < 3e-3
    ax = fwhm_1d(z[sel], prof[sel])
    print(f"\nInterfaz plana a 14 mm (onda plana, Hann): FWHM axial {fmt(ax, 3)} mm frente a {fmt(ax0, 3)} mm "
          f"del pulso de ida y vuelta ({pct(ax, ax0)}).")
    print("\nCoste (simulaciones FDTD, %d elementos, malla %dx%d):\n" % (arr.n_elements, g.nz, g.nx))
    print("| Modo | Emisiones | Tiempo FDTD total (s) | Tiempo por emisión (s) |")
    print("|---|---|---|---|")
    for name, (sc, _) in runs.items():
        n = len(sc.transmits)
        print(f"| {name} | {n} | {sc.elapsed_s:.0f} | {sc.elapsed_s / n:.1f} |")


# --------------------------------------------------------------------------
def speckle_roi(img, z0=6e-3, z1=12e-3, half_width=6e-3):
    xx, zz = np.meshgrid(img.x, img.z)
    return (np.abs(xx) < half_width) & (zz > z0) & (zz < z1)


def part_speckle(quick):
    g, ph, arr = setup()
    seeds = range(2 if quick else 5)
    xs, zs = 0.0, 9e-3
    psf = acquire(ph, arr, point_scatterer_medium(g, F0, [(xs, zs)]), "plane")[0]
    rows = []
    for window in ("hann", "rect"):
        p = reconstruct(psf, window)
        pred = predicted_grain_fwhm(p.iq, p.x, p.z, xs, zs)
        snrs, grains = [], []
        for seed in seeds:
            sc = acquire(ph, arr, speckle_medium(g, F0, 0.05, seed), "plane")[0]
            im = reconstruct(sc, window)
            roi = speckle_roi(im)
            snrs.append(speckle_snr(im.envelope, roi))
            grains.append(speckle_grain_fwhm(im.envelope, roi, PX, PX))
        rows.append((window, np.array(snrs), np.array(grains), pred))
    print("\n### Speckle totalmente desarrollado (onda plana, %d realizaciones)\n" % len(seeds))
    print(f"Rayleigh: SNR = {RAYLEIGH_SNR:.3f}. ROI: |x| < 6 mm, z de 6 a 12 mm.\n")
    print("| Ventana rx | SNR media ± desv. | error vs 1.913 | grano axial (medido / predicho, mm) | error | "
          "grano lateral (medido / predicho, mm) | error |")
    print("|---|---|---|---|---|---|---|")
    for window, snrs, grains, pred in rows:
        gm = grains.mean(0)
        print(f"| {window} | {snrs.mean():.3f} ± {snrs.std():.3f} | {pct(snrs.mean(), RAYLEIGH_SNR)} | "
              f"{fmt(gm[0], 3)} / {fmt(pred[0], 3)} | {pct(gm[0], pred[0])} | "
              f"{fmt(gm[1], 3)} / {fmt(pred[1], 3)} | {pct(gm[1], pred[1])} |")


# --------------------------------------------------------------------------
def part_cyst(quick):
    g, ph, arr = setup()
    center, radius = (0.0, 15e-3), 3.5e-3
    X, Z = g.meshgrid()
    cyst = (X - center[0]) ** 2 + (Z - center[1]) ** 2 <= radius**2
    med = speckle_medium(g, F0, 0.05, seed=7, anechoic=cyst)
    sta_step = 4 if quick else 1
    acq = {
        "PW 1 ángulo": acquire(ph, arr, med, "plane")[0],
        "PW 5 ángulos (±8°)": acquire(ph, arr, med, "plane", angles=(-8.0, -4.0, 0.0, 4.0, 8.0))[0],
        "focalizada F#2 @15 mm": acquire(ph, arr, med, "focused", line_step=2, focus_depth=15e-3, tx_f_number=2.0)[0],
        f"STA (paso {sta_step})": acquire(ph, arr, med, "sta", sta_step=sta_step)[0],
    }
    img0 = acq["PW 1 ángulo"].image
    d = np.hypot(*np.meshgrid(img0.x - center[0], img0.z - center[1]))
    inside, back = d < radius - 1e-3, (d > radius + 1e-3) & (d < radius + 4e-3)
    print("\n### Quiste anecoico de 3.5 mm de radio a 15 mm (ventana de Hann en recepción)\n")
    print("Contraste = 10·log10(potencia dentro / fuera); CNR y gCNR sobre la envolvente lineal.\n")
    print("| Secuencia | Formador | Contraste (dB) | CNR | gCNR | SNR speckle fondo |")
    print("|---|---|---|---|---|---|")
    for name, sc in acq.items():
        for bf in ("das", "cf", "dmas", "slsc"):
            if bf != "das" and name not in ("PW 1 ángulo", f"STA (paso {sta_step})"):
                continue
            img = reconstruct(sc, "hann", bf)
            e = img.envelope
            print(f"| {name} | {bf} | {contrast_db(e, inside, back):.1f} | {cnr(e, inside, back):.2f} | "
                  f"{gcnr(e, inside, back):.2f} | {speckle_snr(e, back):.2f} |")


# --------------------------------------------------------------------------
def part_layers(quick):
    f0 = F0
    g = Grid2D.from_size(WIDTH, DEPTH, C / f0 / 10, x0=-WIDTH / 2)
    labels = np.full(g.shape, TISSUE_INDEX["muscle"], np.int16)
    labels[g.z < 4e-3, :] = TISSUE_INDEX["water"]
    labels[(g.z >= 4e-3) & (g.z < 15e-3), :] = TISSUE_INDEX["fat"]
    ph = Phantom(g, labels)
    a = default_array(ph, f0, 20, C)
    arr = LinearArray(n_elements=a.n_elements, pitch=a.pitch, z=a.z, x_center=a.x_center, f0=f0,
                      response=ElementResponse(f0, "windowed", 2.0, "hann"))
    xs, zs = 0.0, 21e-3
    med = elastic_medium(ph, f0, n_mechanisms=0, scattering=0.0, attenuation=False)
    j, i = g.index_of(xs, zs)
    med.rho[j, i] *= 4.0
    sc = simulate_scan(ph, f0, "plane", arr, c0=C, medium=med, n_mechanisms=0, tgc_db_per_cm=0, pixel=LAM / 12,
                       max_depth=MAX_DEPTH, f_number=F_NUMBER)
    layers = TwoLayerSpeed(15e-3, 1450.0, 1580.0)
    # Control: mismo dispersor en músculo homogéneo (sin grasa), reconstruido con su velocidad real. Su error de
    # posición es el sesgo numérico del FDTD (dispersión temporal), que no tiene que ver con la capa.
    lab_c = labels.copy()
    lab_c[(g.z >= 4e-3) & (g.z < 15e-3), :] = TISSUE_INDEX["muscle"]
    ph_c = Phantom(g, lab_c)
    med_c = elastic_medium(ph_c, f0, n_mechanisms=0, scattering=0.0, attenuation=False)
    med_c.rho[j, i] *= 4.0
    sc_c = simulate_scan(ph_c, f0, "plane", arr, c0=C, medium=med_c, n_mechanisms=0, tgc_db_per_cm=0, pixel=LAM / 12,
                         max_depth=MAX_DEPTH, f_number=F_NUMBER)
    img_c = bmode_channels(sc_c.rf, sc_c.transmits, sc_c.t, arr.element_x, arr.z, sc_c.image.x, sc_c.image.z,
                           c0=1580.0, t_peak=arr.pulse_peak_time, f_number=F_NUMBER, rx_window="hann")
    rc = psf_fwhm(img_c.envelope, img_c.x, img_c.z, xs, zs, search=2.5e-3)
    bias = rc["offset_z"]
    print("\n### Corrección de la velocidad en dos capas (dispersor en músculo a 21 mm bajo 11 mm de grasa)\n")
    print(f"Control en músculo homogéneo (c = 1580 m/s): desplazamiento axial {bias * 1e3:+.2f} mm (sesgo numérico).\n")
    print("| Reconstrucción | pico (x, z) mm | error de z (mm) | error de z sin el sesgo (mm) | FWHM lateral (mm) "
          "| FWHM axial (mm) | amplitud relativa |")
    print("|---|---|---|---|---|---|---|")
    base = None
    for name, ly, c0 in (("1540 m/s fijo", None, 1540.0), ("1500 m/s fijo", None, 1500.0),
                         ("dos capas (1450 / 1580)", layers, 1540.0)):
        img = bmode_channels(sc.rf, sc.transmits, sc.t, arr.element_x, arr.z, sc.image.x, sc.image.z, c0=c0,
                             t_peak=arr.pulse_peak_time, f_number=F_NUMBER, rx_window="hann", layers=ly)
        r = psf_fwhm(img.envelope, img.x, img.z, xs, zs, search=2.5e-3)
        amp = img.envelope[np.argmin(abs(img.z - r["peak_z"])), np.argmin(abs(img.x - r["peak_x"]))]
        base = base or amp
        print(f"| {name} | ({r['peak_x'] * 1e3:+.2f}, {r['peak_z'] * 1e3:.2f}) | {r['offset_z'] * 1e3:+.2f} | "
              f"{(r['offset_z'] - bias) * 1e3:+.2f} | {fmt(r['lateral'])} | {fmt(r['axial'], 3)} | {amp / base:.2f} |")
    print("\nLa posición verdadera es (0.00, 21.00) mm.")


def part_fracture(quick):
    from dataclasses import replace

    from wavesim.analysis import dilate, fracture_visibility, resample_mask_apparent_depth
    from wavesim.phantoms import Fracture, longitudinal_limb

    f0 = 3e6
    dx = 1450 / f0 / 8
    cases = {
        "fractura desplazada": Fracture(displacement=1.0e-3, gap=0.5e-3),
        "fisura (0.15 mm)": Fracture(kind="fisura", gap=0.15e-3, depth_fraction=0.3, hematoma_thickness=0.3e-3,
                                     hematoma_length=4e-3),
    }
    healthy = longitudinal_limb("tibia", None, width=24e-3, depth=22e-3, dx=dx, seed=1)
    # Misma anatomía sana con otra realización del speckle: da el ruido natural de cada método
    rng = np.random.default_rng(123)
    healthy_b = replace(healthy, heterogeneity=rng.standard_normal(healthy.grid.shape).astype(np.float32))
    phantoms = {"sana": healthy, "sana (otro speckle)": healthy_b}
    for name, fr in cases.items():
        phantoms[name] = longitudinal_limb("tibia", fr, width=24e-3, depth=22e-3, dx=dx, seed=1)
    sta_step = 4 if quick else 2
    methods = {"PW×3 DAS": ("pw", "das"), "PW×3 CF": ("pw", "cf"), "PW×3 DMAS": ("pw", "dmas"),
               "STA DAS": ("sta", "das"), "STA CF": ("sta", "cf")}
    images, scans, cost = {}, {}, {"pw": 0.0, "sta": 0.0}
    for name, p in phantoms.items():
        for seq in ("pw", "sta"):
            t0 = time.time()
            if seq == "pw":
                sc = simulate_scan(p, f0, "plane", angles=(-8.0, 0.0, 8.0))
            else:
                sc = simulate_scan(p, f0, "sta", sta_step=sta_step)
            cost[seq] += time.time() - t0
            scans[name, seq] = sc
            print(f"  [{name}, {seq}: {len(sc.transmits)} emisiones, {time.time() - t0:.0f} s]", file=sys.stderr,
                  flush=True)
        for m, (seq, bf) in methods.items():
            sc = scans[name, seq]
            a = sc.array
            images[name, m] = bmode_channels(sc.rf, sc.transmits, sc.t, a.element_x, a.z, sc.image.x, sc.image.z,
                                             c0=1540.0, t_peak=a.pulse_peak_time, tgc_db_per_cm=0.5 * f0 / 1e6,
                                             f_number=1.5, beamformer=bf)
        scans[name, "pw"].results.clear()
        scans[name, "sta"].results.clear()
    print("\n### Tibia a 3 MHz: visibilidad de la fractura por secuencia y formador de haz\n")
    print(f"STA con un elemento de cada {sta_step} ({scans['sana', 'sta'].array.n_elements // sta_step} emisiones). "
          "ROI = máscara de fractura y hematoma (en profundidad aparente, dilatada 4 px). "
          "«Ruido» = cambio medio entre la sana y la misma sana con otro speckle (dB); «CNR» = cambio medio / ruido.\n")
    print("| Caso | Método | Cambio medio en la ROI (dB) | Ruido (dB) | CNR | Razón de energía |")
    print("|---|---|---|---|---|---|")
    for case in cases:
        p = phantoms[case]
        sc = scans[case, "pw"]
        for m in methods:
            img, ref, ref_b = images[case, m], images["sana", m], images["sana (otro speckle)", m]
            roi = dilate(resample_mask_apparent_depth(p, p.fracture_mask | p.masks["hematoma"], img.x, img.z,
                                                      sc.array.z), 4)
            noise = fracture_visibility(ref.db, ref_b.db, roi)["mean_abs_change"]
            v = fracture_visibility(ref.db, img.db, roi, noise_sigma=noise)
            print(f"| {case} | {m} | {v['mean_abs_change']:.1f} | {noise:.1f} | {v['cnr']:.2f} | "
                  f"{v['change_ratio']:.2f} |")
    print(f"\nTiempo FDTD (4 fantomas + 1 de control): ondas planas {cost['pw']:.0f} s, STA {cost['sta']:.0f} s.")
    _fracture_figure(phantoms, images, list(methods), cases)


def _fracture_figure(phantoms, images, methods, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["sana", *cases]
    fig, axs = plt.subplots(len(names), len(methods), figsize=(3.2 * len(methods), 3.0 * len(names)), squeeze=False)
    for r, case in enumerate(names):
        for c, m in enumerate(methods):
            img = images[case, m]
            axs[r, c].imshow(img.db, extent=img.extent_mm(), cmap="gray", vmin=-50, vmax=0, aspect="equal")
            axs[r, c].set_title(f"{case} — {m}" if r == 0 or c == 0 else m, fontsize=7)
            axs[r, c].tick_params(labelsize=6)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "validacion" / "imagen_ultrasonido_fractura.png"
    fig.savefig(out, dpi=70)
    print(f"Figura: {out}")


PARTS = {"psf": part_psf, "speckle": part_speckle, "quiste": part_cyst, "capas": part_layers,
         "fractura": part_fracture}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parte", choices=sorted(PARTS), help="ejecuta sólo una parte")
    ap.add_argument("--rapido", action="store_true", help="STA cada 4 elementos y menos semillas")
    args = ap.parse_args()
    for name, fn in PARTS.items():
        if args.parte == name or (args.parte is None and name != "fractura"):
            t0 = time.time()
            fn(args.rapido)
            print(f"\n[{name}: {time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
