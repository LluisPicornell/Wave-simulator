"""Fracturas internas del hueso: ¿las detecta el ultrasonido de baja frecuencia cuando la ecografía no las ve?

La ecografía convencional (modo B, 3-5 MHz) ve bien la superficie del hueso
pero casi nada por debajo: la cortical refleja ~45 % de la energía y atenúa
mucho, y deja una sombra acústica. Una fractura que no rompe la superficie
que mira a la sonda (intracortical, de la cortical lejana, contusión ósea
trabecular, fractura por estrés con reacción endóstica) queda escondida.
A frecuencias más bajas (0.25-1 MHz) la onda atraviesa el hueso y su
longitud de onda (3.5-14 mm en la cortical) es comparable al grosor de la
cortical: en principio podría "notar" defectos internos. Este script lo
comprueba con la misma anatomía, sana y fracturada, y tres montajes (corte
longitudinal 2D):

  (a) Modo B convencional a 3 MHz: sonda lineal con una onda plana a 0°
      (una sola emisión para que cada simulación dure menos de un minuto; el
      ejemplo 03 compone tres ángulos).
  (b) Transmisión a través de la extremidad (TT), 0.25-1 MHz: onda plana desde
      la cara anterior y fila de receptores bajo la piel posterior.
  (c) Transmisión axial (AT), 0.25-1 MHz, como en la ultrasonometría ósea
      (Bossy, Talmant y Laugier, JASA 112:297, 2002; Dodd et al., Bone
      40:656, 2007): emisor pequeño sobre la piel y receptores a 15-85 mm a lo
      largo del hueso. Se analiza la primera llegada (FAS), la onda que viaja
      por la cortical y se adelanta a la que va por el tejido blando. Sólo
      muestrea la cortical cercana, así que no se aplica a la lejana.

Cómo se mide la detección. Para cada técnica se compara la señal de la
pierna fracturada con la de la misma pierna sana (misma semilla: idénticas
salvo la fractura) en la zona que importa: la región de la fractura en el
modo B, el pulso principal en los receptores a menos de 5 mm de la fractura
en TT, y la ventana de la FAS en los receptores más allá de la fractura en
AT. "Cambio" = 10·log10(Σ(fracturada − sana)² / Σ sana²). Como referencia
de variabilidad se compara igual una segunda pierna sana con la misma
anatomía nominal pero otra semilla (otra rugosidad de las interfaces y otra
microestructura): es una cota INFERIOR de la variabilidad real (la pierna
contralateral o la de otra persona difieren además en grosores, ±5-20 %).
D = cambio − variabilidad (dB). D ≥ 6 dB (el cambio tiene 4 veces más
energía que la variabilidad mínima): detectable en esta simulación ideal sin
ruido; 0-6 dB: dudoso; D < 0: indistinguible de dos piernas sanas.

Se usa una malla de 0.1 mm para todas las bajas frecuencias (más fina de lo
que pide la longitud de onda) porque debe dibujar grietas de 0.1-0.3 mm: así
el fantoma es idéntico a todas las frecuencias y sólo cambia la onda.

Uso:
    python examples/11_fracturas_internas_baja_frecuencia.py               (~120 simulaciones de < 1 min)
    python examples/11_fracturas_internas_baja_frecuencia.py --procesos 3  (en paralelo)
    python examples/11_fracturas_internas_baja_frecuencia.py --rapido      (~5 min: cuerpo normal, 0.5 MHz)
Cada simulación se guarda en outputs/11_cache/ y no se repite al relanzar.
→ outputs/11_fantomas.png, 11_modo_b.png, 11_transmision_axial.png, 11_resumen.png, 11_tabla.txt
"""

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.analysis import dilate, energy_ratio_db, resample_mask, residual_db, xcorr_delay
from wavesim.core.signals import pulse_duration, pulse_peak_time
from wavesim.phantoms import Fracture, longitudinal_limb, with_body_type
from wavesim.ultrasound import ElasticSolver, LinearArray, line_receiver, simulate_bmode
from wavesim.viz import plot_image, plot_phantom, save

OUT = Path(__file__).resolve().parent.parent / "outputs"
CACHE = OUT / "11_cache"
NPML = 20
SEED_A, SEED_B = 11, 12  # anatomía de referencia y segunda anatomía sana (variabilidad)
F_BMODE = 3e6
DX_BMODE = 1450 / F_BMODE / 8
DX_LOW = 0.1e-3
FREQS = (0.25e6, 0.5e6, 1e6)
TT_WIDTH, TT_RX_HALF = 24e-3, 5e-3
BMODE_WIDTH = 18e-3
AT_WIDTH, AT_SOURCE_X, AT_OFFSETS = 100e-3, -45e-3, (15e-3, 85e-3)
C_SKIN = 1615.0  # onda directa más rápida por el tejido blando (piel)
DETECTABLE_DB = 6.0

# nombre: (preset, fractura, semilla). Todas las grietas están en x = 0.
CASES = {
    "sana_A": ("tibia", None, SEED_A),
    "sana_B": ("tibia", None, SEED_B),
    # Controles positivos: una fractura desplazada 1 mm (la del ejemplo 03, que el modo B ve
    # con claridad) y la misma grieta que la intracortical pero abierta a la superficie
    "control_desplazada": ("tibia", Fracture(kind="desplazada", displacement=1.0e-3, gap=0.5e-3), SEED_A),
    "fisura_superficial": ("tibia", Fracture(kind="fisura", span=(0.0, 0.6), gap=0.2e-3, min_gap_cells=1,
                                             hematoma_thickness=0.3e-3, hematoma_length=4e-3, edema_length=0.0),
                           SEED_A),
    "intracortical": ("tibia", Fracture(kind="intracortical", span=(0.25, 0.85), gap=0.2e-3, min_gap_cells=1,
                                        hematoma_thickness=0.0, edema_length=0.0), SEED_A),
    "cortical_lejana": ("tibia", Fracture(kind="cortical_lejana", span=(2.0, 3.0), gap=0.3e-3, min_gap_cells=1,
                                          hematoma_thickness=1e-3, hematoma_length=8e-3, edema_length=10e-3,
                                          edema_depth=4e-3), SEED_A),
    "estres_endostica": ("tibia", Fracture(kind="estres", span=(0.6, 1.0), gap=0.1e-3, min_gap_cells=1,
                                           hematoma_thickness=0.0, endosteal_reaction=1e-3, reaction_length=15e-3,
                                           edema_length=15e-3, edema_depth=5e-3), SEED_A),
    "meta_sana_A": ("tibia_metafisis", None, SEED_A),
    "meta_sana_B": ("tibia_metafisis", None, SEED_B),
    "contusion_osea": ("tibia_metafisis", Fracture(kind="contusion_osea", gap=0.0, hematoma_thickness=0.0,
                                                   microcracks=12, microcrack_length=1.5e-3, microcrack_gap=0.1e-3,
                                                   min_gap_cells=1, edema_length=16e-3, edema_depth=7e-3,
                                                   edema_fraction=0.25), SEED_A),
}
REFERENCES = {"tibia": ("sana_A", "sana_B"), "tibia_metafisis": ("meta_sana_A", "meta_sana_B")}


# --------------------------------------------------------------------------
# Fantomas y simulaciones
# --------------------------------------------------------------------------
def _depths(preset: str, body: str) -> dict:
    """Profundidades (m) del dominio de cada montaje para este preset y tipo de cuerpo."""
    spec = with_body_type(preset, body)
    layers = [t for _, t in spec.layers()]
    near_endosteum = spec.standoff + sum(layers[:4])
    far_periosteum = spec.standoff + sum(layers[:6])
    pml_b, pml_l = (NPML + 3) * DX_BMODE, (NPML + 3) * DX_LOW
    deep_bmode = near_endosteum + 12e-3 if spec.trabecular else far_periosteum + 4e-3
    # La FAS viaja por la cortical cercana: basta la médula que hay bajo ella
    return {"bmode": deep_bmode + pml_b, "at": near_endosteum + (12e-3 if spec.trabecular else 8e-3) + pml_l,
            "tt": None}


def build(case: str, body: str, technique: str):
    preset, fr, seed = CASES[case]
    dx = DX_BMODE if technique == "bmode" else DX_LOW
    width = {"bmode": BMODE_WIDTH, "tt": TT_WIDTH, "at": AT_WIDTH}[technique]
    return longitudinal_limb(preset, fr, width=width, dx=dx, depth=_depths(preset, body)[technique], seed=seed,
                             body=body)


def _taper(n: int, fraction: float = 0.3) -> np.ndarray:
    """Apodización de Tukey: suaviza los bordes de la apertura para reducir las ondas de borde."""
    w = np.ones(n)
    m = max(int(fraction * n / 2), 1)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(m) + 0.5) / m)
    w[:m], w[-m:] = ramp, ramp[::-1]
    return w


def sim_bmode(p):
    sc = simulate_bmode(p, F_BMODE, angles=(0.0,))
    return {"env": sc.image.envelope, "x": sc.image.x, "z": sc.image.z}


def sim_tt(p, f0):
    g = p.grid
    solver = ElasticSolver(p, f0, npml=NPML)
    # Línea continua de emisores de 1 mm: onda plana con los bordes suavizados
    n = int((g.width - 2 * (NPML + 6) * g.dx) / 1e-3)
    arr = LinearArray(n_elements=n, pitch=1e-3, element_width=1e-3, z=g.z0 + (NPML + 3) * g.dx,
                      x_center=g.x0 + g.width / 2 - g.dx / 2, f0=f0)
    nt = int((pulse_duration(f0) + g.depth / 1400 + 3 / f0) / solver.dt)
    t = solver.time_axis(nt)
    solver.add_source(arr.transmit_source(g, t, np.zeros(n), _taper(n)))
    rx = line_receiver(g, g.z0 + g.depth - (NPML + 3) * g.dx, -2 * TT_RX_HALF, 2 * TT_RX_HALF, spacing=0.5e-3)
    solver.add_receiver("rx", rx)
    res = solver.run(nt, track_peak=False)
    return {"t": t, "traces": res.traces["rx"], "x": g.x[rx.i]}


def sim_at(p, f0):
    g = p.grid
    solver = ElasticSolver(p, f0, npml=NPML)
    z = g.z0 + (NPML + 3) * g.dx
    # Emisor de ~λ/2: radia en un abanico ancho que incluye el ángulo crítico (~26°) de la cortical
    w = min(1500.0 / f0 / 2, 2e-3)
    arr = LinearArray(n_elements=1, pitch=w, element_width=w, z=z, x_center=AT_SOURCE_X, f0=f0)
    soft = p.meta["bone_top_depth"] - z
    nt = int((pulse_duration(f0) + 2 * soft / 1400 + AT_OFFSETS[1] / 3000 + 2 / f0) / solver.dt)
    t = solver.time_axis(nt)
    solver.add_source(arr.transmit_source(g, t, np.zeros(1)))
    rx = line_receiver(g, z, AT_SOURCE_X + AT_OFFSETS[0], AT_SOURCE_X + AT_OFFSETS[1], spacing=1e-3)
    solver.add_receiver("rx", rx)
    res = solver.run(nt, track_peak=False)
    return {"t": t, "traces": res.traces["rx"], "x": g.x[rx.i]}


def _cache_path(body, case, technique, f0) -> Path:
    return CACHE / f"{body}__{case}__{technique}_{f0 / 1e6:g}MHz.npz"


def run_job(job) -> str:
    body, case, technique, f0 = job
    path = _cache_path(*job)
    if path.exists():
        return f"(en caché) {path.name}"
    # A 2-3 MHz la onda S del hueso trabecular (1000 m/s) queda con < 6 puntos por
    # longitud de onda, pero ese hueso la atenúa > 60 dB/cm: no influye en lo medido.
    warnings.filterwarnings("ignore", message="Sólo .* puntos por longitud de onda")
    p = build(case, body, technique)
    t0 = time.time()
    data = sim_bmode(p) if technique == "bmode" else (sim_tt if technique == "tt" else sim_at)(p, f0)
    np.savez_compressed(path, **data)
    return f"{path.name} ({time.time() - t0:.0f} s, malla {p.grid.nz}x{p.grid.nx})"


def jobs_for(bodies, cases, freqs):
    jobs = []
    for body in bodies:
        for case in cases:
            jobs.append((body, case, "bmode", F_BMODE))
            jobs += [(body, case, "tt", f) for f in freqs]
            if case != "cortical_lejana":
                jobs += [(body, case, "at", f) for f in freqs]
    return jobs


# --------------------------------------------------------------------------
# Análisis
# --------------------------------------------------------------------------
def load(body, case, technique, f0):
    return dict(np.load(_cache_path(body, case, technique, f0)))


def tt_windows(ref, f0):
    """Pulso transmitido directo (5 periodos desde su llegada) en los receptores bajo la fractura.

    La ventana se ancla a la primera llegada (10 % del máximo) y no al máximo
    de la traza: a 0.25 MHz la reverberación entre las dos corticales (unos
    20 µs después) apenas se atenúa y puede ser tan fuerte como el pulso.
    """
    sel = np.nonzero(np.abs(ref["x"]) <= TT_RX_HALF)[0]
    dt = ref["t"][1] - ref["t"][0]
    s = np.abs(ref["traces"][sel].mean(axis=0))
    on = int(np.argmax(s > 0.1 * s.max()))
    return [(c, max(on - int(0.5 / f0 / dt), 0), on + int(4.5 / f0 / dt)) for c in sel]


def at_windows(ref, f0):
    """Ventana de la FAS en los receptores más allá de la fractura (x > 3 mm).

    El inicio es el primer instante en que la traza sana supera -50 dB de su
    máximo; la ventana dura 2 periodos y debe terminar antes de que llegue la
    onda directa por el tejido blando. Si no da tiempo (tejido blando grueso o
    frecuencia baja), ese receptor no sirve: la FAS no se separa.
    """
    t, tr = ref["t"], ref["traces"]
    dt = t[1] - t[0]
    out = []
    for c, xr in enumerate(ref["x"]):
        if xr <= 3e-3:
            continue
        s = np.abs(tr[c])
        on = int(np.argmax(s > s.max() * 10 ** (-50 / 20)))
        lo, hi = on - int(0.5 / f0 / dt), on + int(1.5 / f0 / dt)
        t_direct = pulse_peak_time(f0) + (xr - AT_SOURCE_X) / C_SKIN - 2.2 / f0
        if lo > 0 and t[hi] < t_direct:
            out.append((c, lo, hi))
    return out


def compare(ref, test, windows):
    dt = ref["t"][1] - ref["t"][0]
    a = np.concatenate([ref["traces"][c, lo:hi] for c, lo, hi in windows])
    b = np.concatenate([test["traces"][c, lo:hi] for c, lo, hi in windows])
    delays = [xcorr_delay(ref["traces"][c, lo:hi], test["traces"][c, lo:hi], dt) for c, lo, hi in windows]
    return {"dE": energy_ratio_db(a, b), "dt": float(np.median(delays)), "res": residual_db(a, b)}


def bmode_roi(case, body, img):
    """Celdas que la fractura cambia (grieta, hematoma, edema, reacción) más 1 mm alrededor."""
    preset = CASES[case][0]
    ref = build(REFERENCES[preset][0], body, "bmode")
    changed = build(case, body, "bmode").labels != ref.labels
    px = img["x"][1] - img["x"][0]
    return dilate(resample_mask(ref, changed, img["x"], img["z"]), int(round(1e-3 / px)))


def compare_bmode(ref, test, roi):
    a, b = ref["env"][roi], test["env"][roi]
    return {"dE": energy_ratio_db(a, b), "dt": float("nan"), "res": residual_db(a, b)}


def verdict(d: float) -> str:
    return "sí" if d >= DETECTABLE_DB else ("dudoso" if d >= 0 else "no")


def analyse(bodies, fractures, freqs):
    rows = []
    for body in bodies:
        for case in fractures:
            preset = CASES[case][0]
            ref_name, var_name = REFERENCES[preset]
            ref, var, test = (load(body, n, "bmode", F_BMODE) for n in (ref_name, var_name, case))
            roi = bmode_roi(case, body, ref)
            rows.append((body, case, "modo B", F_BMODE, compare_bmode(ref, test, roi), compare_bmode(ref, var, roi)))
            for tech, windows in (("TT", tt_windows), ("AT", at_windows)):
                if tech == "AT" and case == "cortical_lejana":
                    continue
                for f in freqs:
                    ref, var, test = (load(body, n, tech.lower(), f) for n in (ref_name, var_name, case))
                    win = windows(ref, f)
                    if not win:
                        rows.append((body, case, tech, f, None, None))
                        continue
                    rows.append((body, case, tech, f, compare(ref, test, win), compare(ref, var, win)))
    return rows


def format_table(rows) -> str:
    head = (f"{'Cuerpo':<8} {'Fractura':<19} {'Técnica':<7} {'f MHz':>5} {'ΔE dB':>7} {'Δt ns':>7} "
            f"{'Δt var':>7} {'Cambio dB':>9} {'Variab. dB':>10} {'D dB':>6}  Detecta")
    lines = [head, "-" * len(head)]

    def ns(v):
        return "—" if np.isnan(v) else f"{v * 1e9:+.0f}"

    for body, case, tech, f, m, v in rows:
        start = f"{body:<8} {case:<19} {tech:<7} {f / 1e6:>5.2g}"
        if m is None:
            lines.append(f"{start}   la FAS no se separa de la onda directa: no medible")
            continue
        d = m["res"] - v["res"]
        lines.append(f"{start} {m['dE']:>+7.2f} {ns(m['dt']):>7} {ns(v['dt']):>7} {m['res']:>9.1f} {v['res']:>10.1f} "
                     f"{d:>+6.1f}  {verdict(d)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def fig_phantoms(body):
    cases = [c for c in CASES if c not in ("sana_B", "meta_sana_B")]
    fig, axs = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 6.5))
    for ax, case in zip(axs, cases):
        p = build(case, body, "tt")
        plot_phantom(p, ax, case, legend=False)
        ax.set_xlim(-10, 10)
    fig.suptitle(f"Fantomas (cuerpo {body}); la grieta se marca en rojo")
    fig.tight_layout()
    return fig


def fig_bmode(bodies):
    cases = ["sana_A", "control_desplazada", "fisura_superficial", "intracortical", "cortical_lejana",
             "estres_endostica", "meta_sana_A", "contusion_osea"]
    fig, axs = plt.subplots(len(bodies), len(cases), figsize=(3.3 * len(cases), 5.2 * len(bodies)), squeeze=False)
    for r, body in enumerate(bodies):
        for ax, case in zip(axs[r], cases):
            img = load(body, case, "bmode", F_BMODE)
            ref = load(body, REFERENCES[CASES[case][0]][0], "bmode", F_BMODE)
            db = 20 * np.log10(np.maximum(img["env"] / ref["env"].max(), 10 ** (-60 / 20)))
            dxp = img["x"][1] - img["x"][0]
            ext = ((img["x"][0] - dxp / 2) * 1e3, (img["x"][-1] + dxp / 2) * 1e3, (img["z"][-1] + dxp / 2) * 1e3,
                   (img["z"][0] - dxp / 2) * 1e3)
            plot_image(db, ext, ax, f"{body}: {case}", vmin=-60, vmax=0)
            if CASES[case][1] is not None:
                ax.contour(img["x"] * 1e3, img["z"] * 1e3, bmode_roi(case, body, ref).astype(float), levels=[0.5],
                           colors="yellow", linewidths=0.6)
    fig.suptitle("Modo B a 3 MHz (dB respecto al máximo de la pierna sana; amarillo: zona que cambia la fractura)")
    fig.tight_layout()
    return fig


def fig_axial(body, freqs):
    cases = ["fisura_superficial", "intracortical", "estres_endostica"]
    fig, axs = plt.subplots(len(cases), len(freqs), figsize=(4.2 * len(freqs), 3.2 * len(cases)), squeeze=False)
    for r, case in enumerate(cases):
        for ax, f in zip(axs[r], freqs):
            ref, var, test = (load(body, n, "at", f) for n in ("sana_A", "sana_B", case))
            win = at_windows(ref, f)
            ax.set_title(f"{case}, {f / 1e6:g} MHz", fontsize=9)
            if not win:
                ax.text(0.5, 0.5, "FAS no separable", ha="center", transform=ax.transAxes)
                continue
            c, lo, hi = win[len(win) // 2]
            t = ref["t"] * 1e6
            pad = hi - lo
            sl = slice(max(lo - pad // 2, 0), hi + pad // 2)
            for data, label, style in ((ref, "sana A", "k-"), (var, "sana B", "k:"), (test, "fracturada", "r-")):
                ax.plot(t[sl], data["traces"][c, sl], style, lw=1, label=label)
            ax.axvspan(t[lo], t[hi - 1], color="0.9")
            ax.set_xlabel(f"t [µs] (receptor a x = {ref['x'][c] * 1e3:.0f} mm)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Transmisión axial, cuerpo {body}: primera llegada (FAS) más allá de la fractura (ventana en gris)")
    fig.tight_layout()
    return fig


def fig_summary(rows, bodies, fractures):
    fig, axs = plt.subplots(1, len(bodies), figsize=(8 * len(bodies), 6), squeeze=False)
    colors = dict(zip(fractures, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ax, body in zip(axs[0], bodies):
        for case in fractures:
            for tech, style in (("TT", "--o"), ("AT", "-s"), ("modo B", "*")):
                pts = [(f / 1e6, m["res"] - v["res"]) for b, c, te, f, m, v in rows
                       if b == body and c == case and te == tech and m is not None]
                if pts:
                    f, d = np.array(pts).T
                    ax.plot(f, d, style, color=colors[case], ms=12 if tech == "modo B" else 5,
                            label=f"{case} ({tech})")
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(DETECTABLE_DB, color="k", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("frecuencia [MHz]")
        ax.set_ylabel("D = cambio por la fractura − variabilidad entre sanas [dB]")
        ax.set_title(f"Cuerpo {body} (encima de la línea de puntos: detectable)")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuerpos", nargs="+", default=["normal", "obeso"], help="tipos de cuerpo (BODY_TYPES)")
    parser.add_argument("--procesos", type=int, default=1, help="simulaciones en paralelo")
    parser.add_argument("--rapido", action="store_true", help="sólo cuerpo normal, intracortical y 0.5 MHz")
    args = parser.parse_args()
    bodies, cases, freqs = args.cuerpos, list(CASES), FREQS
    if args.rapido:
        bodies, cases, freqs = ["normal"], ["sana_A", "sana_B", "intracortical"], (0.5e6,)
    fractures = [c for c in cases if CASES[c][1] is not None]
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(bodies, cases, freqs)
    print(f"{len(jobs)} simulaciones ({sum(_cache_path(*j).exists() for j in jobs)} ya en caché)", flush=True)
    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        for k, name in enumerate(pool.map(run_job, jobs)):
            print(f"[{k + 1}/{len(jobs)}] {name}", flush=True)

    rows = analyse(bodies, fractures, freqs)
    table = format_table(rows)
    print(table)
    (OUT / "11_tabla.txt").write_text(table + "\n", encoding="utf-8")
    if not args.rapido:
        save(fig_phantoms(bodies[0]), OUT / "11_fantomas.png", dpi=80)
        save(fig_bmode(bodies), OUT / "11_modo_b.png", dpi=70)
        save(fig_axial(bodies[0], freqs), OUT / "11_transmision_axial.png", dpi=80)
    save(fig_summary(rows, bodies, fractures), OUT / "11_resumen.png", dpi=80)
    print("Figuras y tabla en", OUT)


if __name__ == "__main__":
    main()
