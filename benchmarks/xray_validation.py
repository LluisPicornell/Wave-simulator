"""Validación completa del módulo de rayos X / TAC (``wavesim.em.xray*``, ``fanbeam``, ``xray_scatter``).

Uso (desde la raíz del repositorio, con un solo hilo para medir tiempos):

    NUMBA_NUM_THREADS=1 python benchmarks/xray_validation.py            # completo, ~5 min con Numba
    NUMBA_NUM_THREADS=1 python benchmarks/xray_validation.py --quick     # 1/10 de fotones, ~1 min
    python benchmarks/xray_validation.py --only hvl fan                  # sólo algunas secciones

Imprime las tablas de ``docs/validacion/rayos_x.md``. Secciones (``--only``):
  hvl      espectro: HVL frente a las calidades IEC 61267 (RQR) y a valores de SpekPy 2.5.4;
  fan      FBP de abanico frente a haz paralelo y al objeto exacto (disco y Shepp-Logan) y SART frente a FBP;
  scatter  Monte Carlo: primaria frente a Beer-Lambert, Compton y Rayleigh, balance, SPR y efecto en el TAC;
  quality  detector, MTF por el borde y NPS.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from wavesim.core.grid import Grid2D
from wavesim.em import fanbeam as fb
from wavesim.em import xray, xray_detector as xdet, xray_phantoms as xp
from wavesim.em import xray_scatter as xs
from wavesim.em.projection import detector_axis, fbp
from wavesim.materials import xray as xd
from wavesim.materials import xray_processes as xpr
from wavesim.phantoms import TISSUE_INDEX, Phantom

# IEC 61267:2005, calidades RQR: (kVp, 1.ª HVL en mm Al), como las reproduce una presentación del IAEA
# basada en IEC 61267 / TRS-457 (RQR3, 5, 8, 9 comprobadas con una búsqueda; 6, 7 y 10 también)
RQR = {"RQR3": (50, 1.78), "RQR5": (70, 2.58), "RQR6": (80, 3.01), "RQR7": (90, 3.48), "RQR8": (100, 3.97),
       "RQR9": (120, 5.00), "RQR10": (150, 6.57)}
# SpekPy 2.5.4: Spek(kvp, th=12).filter("Al", t) -> (HVL1 mm Al, energía media keV), calculado al desarrollar
SPEKPY = {(50, 2.5): (1.750, 32.22), (70, 2.5): (2.397, 39.40), (80, 2.5): (2.734, 42.90),
          (100, 2.5): (3.458, 49.14), (120, 2.5): (4.219, 54.44), (120, 4.0): (5.305, 57.06)}


def title(s):
    print(f"\n## {s}\n")


def rows(header, data):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in data:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def cylinder_phantom(radius, tissue="water", half=0.14, dx=0.002):
    g = Grid2D.from_size(2 * half, 2 * half, dx, x0=-half, z0=-half)
    X, Z = g.meshgrid()
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[np.hypot(X, Z) <= radius] = TISSUE_INDEX[tissue]
    return Phantom(g, lab), g


# --------------------------------------------------------------------------------------
def section_hvl():
    title("1. Espectro: HVL")
    variants = {"Kramers": {}, "ánodo 12°": dict(anode_angle_deg=12.0),
                "líneas K + ánodo 12°": dict(characteristic=True, anode_angle_deg=12.0)}
    for name, kw in variants.items():
        print(f"**{name}**: filtración total de Al (mm) que da la HVL de cada calidad RQR\n")
        data = []
        for q, (kv, hvl) in RQR.items():
            t = xray.fit_al_filtration(kv, hvl, **kw)
            sp = xray.tube_spectrum(kv, t, **kw)
            data.append((q, kv, hvl, f"{t:.2f}", f"{xray.half_value_layer(sp):.3f}", f"{sp.mean_energy:.1f}"))
        rows(["calidad", "kVp", "HVL norma (mm Al)", "Al ajustado (mm)", "HVL obtenida", "E media (keV)"], data)
    print("**Frente a SpekPy 2.5.4** (ánodo 12°, sin filtración inherente):\n")
    data = []
    for (kv, t), (h, em) in SPEKPY.items():
        cells = [kv, t, f"{h:.3f} / {em:.1f}"]
        for kw in variants.values():
            sp = xray.tube_spectrum(kv, t, **kw)
            hv = xray.half_value_layer(sp)
            cells.append(f"{hv:.3f} ({100 * (hv / h - 1):+.1f} %) / {sp.mean_energy:.1f}")
        data.append(cells)
    rows(["kVp", "Al (mm)", "SpekPy HVL / Em", *variants], data)


# --------------------------------------------------------------------------------------
def section_fan(quick):
    title("2. Haz en abanico y reconstrucción")
    g = Grid2D.from_size(0.2, 0.2, 0.0015625, x0=-0.1, z0=-0.1)  # 128 x 128
    X, Z = g.meshgrid()
    r = np.hypot(X, Z)
    sl = xp.shepp_logan(0.08, 20.0)
    ref_sl = xp.rasterize(sl, g)
    d = xp.disk(0.05, 20.0)
    ref_d = xp.rasterize(d, g)
    det = detector_axis(g)
    a_par = np.arange(360.0) / 2
    m_sl, m_d = r < 0.08 * 0.95, r < 0.045
    rm = lambda a, ref, m: float(np.sqrt(np.mean((a - ref)[m] ** 2)))  # noqa: E731
    out = []
    rec_par_d = fbp(xp.parallel_sinogram(d, g, a_par, det), a_par, det, g)
    rec_par_s = fbp(xp.parallel_sinogram(sl, g, a_par, det), a_par, det, g)
    out.append(("paralelo (360 vistas / 180°)", f"{rm(rec_par_d, ref_d, m_d):.3f}",
                f"{rm(rec_par_s, ref_sl, m_sl):.3f}",
                f"{rec_par_d[r < 0.04].mean():.2f}", f"{rec_par_s[(ref_sl == 4.0) & (r < 0.04)].mean():.2f}"))
    for curved in (True, False):
        geom = fb.FanGeometry.covering(g, 0.6, 1.1, curved)
        ang360 = np.arange(720.0) / 2
        rd = fb.fan_fbp(fb.fan_exact_sinogram(d, g, geom, ang360), ang360, geom, g)
        rs = fb.fan_fbp(fb.fan_exact_sinogram(sl, g, geom, ang360), ang360, geom, g)
        out.append((f"abanico {'curvo' if curved else 'plano'} (720 vistas, sid 0.6 m, sdd 1.1 m)",
                    f"{rm(rd, ref_d, m_d):.3f}", f"{rm(rs, ref_sl, m_sl):.3f}", f"{rd[r < 0.04].mean():.2f}",
                    f"{rs[(ref_sl == 4.0) & (r < 0.04)].mean():.2f}"))
    rows(["método", "RMSE disco (1/m)", "RMSE Shepp-Logan (1/m)", "valor del disco (exacto 20)",
          "valor del cerebro SL (exacto 4)"], out)

    print("**SART frente a FBP de abanico** (Shepp-Logan, rejilla 64x64, RMSE en 1/m dentro del 95 % de la cabeza; "
          "ruido gaussiano relativo al máximo del sinograma):\n")
    g2 = Grid2D.from_size(0.2, 0.2, 0.003125, x0=-0.1, z0=-0.1)
    X2, Z2 = g2.meshgrid()
    m2 = np.hypot(X2, Z2) < 0.08 * 0.95
    ref2 = xp.rasterize(sl, g2)
    geom = fb.FanGeometry.covering(g2, 0.6, 1.1, True)
    rng = np.random.default_rng(1)
    data = []
    for noise in (0.0, 0.05):
        for n in (30, 45, 60, 180):
            a = np.arange(n) * 360.0 / n
            sino = fb.fan_exact_sinogram(sl, g2, geom, a)
            sino = sino + noise * sino.max() * rng.standard_normal(sino.shape) / 3.0
            e_f = rm(fb.fan_fbp(sino, a, geom, g2), ref2, m2)
            t = time.time()
            e_s = rm(fb.os_sart(sino, a, geom, g2, n_iter=10, subset_size=5), ref2, m2)
            e_t = rm(fb.os_sart(sino, a, geom, g2, n_iter=10, subset_size=5, tv_weight=1e-3), ref2, m2)
            data.append((f"{100 * noise / 3:.1f} %", n, f"{e_f:.2f}", f"{e_s:.2f}", f"{e_t:.2f}",
                         f"{(time.time() - t) / 2:.1f} s"))
    rows(["ruido", "vistas", "FBP", "OS-SART (10 it, subconj. 5)", "OS-SART + TV", "tiempo/recon"], data)


# --------------------------------------------------------------------------------------
def section_scatter(quick):
    scale = 0.1 if quick else 1.0
    title("3. Radiación dispersa (Monte Carlo)")
    # (a) primaria frente a Beer-Lambert
    p, g = cylinder_phantom(0.09, half=0.1)
    sp = xray.tube_spectrum(100.0, 3.0)
    geom = fb.FanGeometry.covering(g, 0.6, 1.1, True)
    n = int(2e7 * scale)
    t = time.time()
    r = xs.simulate_scatter(p, 0.0, sp, geom, n_photons=n, slice_thickness=0.01, backend="numba", seed=3)
    dt = time.time() - t
    lw = fb.fan_project((p.labels == TISSUE_INDEX["water"]).astype(float), g, geom, [0.0], step=0.0003)[0, 0]
    lbox = fb.fan_project(np.ones(g.shape), g, geom, [0.0], step=0.0003)[0, 0]
    mu_w, mu_a = (xd.linear_attenuation(k, sp.energies_kev) for k in ("water", "air"))
    expo = np.array([np.sum(sp.weights * np.exp(-mu_w * a - mu_a * (b - a))) for a, b in zip(lw, lbox)])
    c = geom.n_det // 2
    data = []
    for name, s in (("centro (21 elem.)", slice(c - 10, c + 11)), ("borde interior (20)", slice(c + 20, c + 40)),
                    ("fuera del cilindro (20)", slice(c + 55, c + 71))):
        obs, ex = r.prim[s].sum(), expo[s].sum() * r.open_beam
        data.append((name, f"{obs:.0f}", f"{ex:.0f}", f"{(obs - ex) / np.sqrt(ex):+.2f}"))
    print(f"(a) Cilindro de agua de 18 cm, 100 kVp (3 mm Al), {n:.1e} fotones, {dt:.1f} s con Numba "
          f"({n / dt:.2e} fotones/s). Primarios del MC frente a Beer-Lambert sobre el mismo mapa de vóxeles:\n")
    rows(["región", "contados", "esperados", "(obs-esp)/sqrt(esp)"], data)

    # (b) Compton y Rayleigh con medio tenue
    energy = 60.0
    spm = xray.monoenergetic(energy)
    pp, gg = cylinder_phantom(0.05, half=0.1)
    prep = xs.prepare(pp, spm)
    prep.scale = prep.scale * 0.03
    gm = fb.FanGeometry.covering(gg, 0.6, 1.1, True)
    rr = xs.simulate_scatter(pp, 0.0, spm, gm, n_photons=int(3e7 * scale), backend="numba", seed=5, prepared=prep)
    cc = rr.counters
    cosg = np.linspace(-1, 1, 200001)
    k = energy / xpr.MEC2_KEV
    eps = 1 / (1 + k * (1 - cosg))
    xq = energy / xpr.HC_KEV_A * np.sqrt(0.5 * (1 - cosg))
    kn = eps**2 * (eps + 1 / eps - (1 - cosg**2))
    kn_s = kn * np.interp(xq, prep.tables.x_grid, prep.tables.s_over_z[0])
    cdf = prep.tables.ray_cdf[0, int(round((energy - prep.tables.ray_energies_kev[0]) / 5.0))]
    cr = np.linspace(-1, 1, cdf.size)
    wp = xpr.process_attenuation("water", energy)
    data = [
        ("<cos θ> Compton (Klein-Nishina x S)", f"{cc['sum_cos_compton'] / cc['n_compton']:.4f}",
         f"{np.sum(cosg * kn_s) / np.sum(kn_s):.4f}", f"{cc['n_compton']:.0f} colisiones"),
        ("<cos θ> Compton si no hubiera S (KN libre)", "-", f"{np.sum(cosg * kn) / np.sum(kn):.4f}",
         "diferencia = efecto de la ligadura"),
        ("energía media cedida (keV)", f"{cc['e_compton_electron'] / cc['n_compton']:.3f}",
         f"{energy * np.sum((1 - eps) * kn_s) / np.sum(kn_s):.3f}", ""),
        ("<cos θ> Rayleigh", f"{cc['sum_cos_rayleigh'] / cc['n_rayleigh']:.4f}",
         f"{np.sum(cr * np.diff(cdf, prepend=0.0)):.4f}", f"{cc['n_rayleigh']:.0f} colisiones"),
        ("fracción Rayleigh / (Rayleigh + Compton)", f"{cc['n_rayleigh'] / (cc['n_rayleigh'] + cc['n_compton']):.4f}",
         f"{wp['coherent'][0] / (wp['coherent'][0] + wp['incoherent'][0]):.4f}", "(PE excluido)"),
    ]
    print("(b) Distribuciones angulares, agua a 60 keV en medio tenue (casi una sola interacción por fotón):\n")
    rows(["magnitud", "Monte Carlo", "esperado", "nota"], data)

    # (c) balance
    bal = r.balance()
    print(f"(c) Balance del caso (a): fotones {bal['photons']:+.1e}, energía {bal['energy']:+.1e} (relativos). "
          f"Energía: emitida {r.counters['e_emitted']:.4e} keV = fotoeléctrico {r.counters['e_photoelectric']:.4e} + "
          f"electrones Compton {r.counters['e_compton_electron']:.4e} + fuera del detector "
          f"{r.counters['e_outside_detector']:.4e} + detectada {r.e_prim.sum() + r.e_scat.sum():.4e}.\n")

    # (d) SPR de cilindros de agua
    print("(d) SPR (dispersa/primaria) en el centro (11 elementos centrales) de cilindros de agua infinitos en y, "
          "sid 1 m, sdd 1.5 m; espectro con líneas K y ánodo de 12°, 2.5 mm Al:\n")
    data = []
    for kv in (80, 100, 120):
        spk = xray.tube_spectrum(kv, 2.5, characteristic=True, anode_angle_deg=12.0)
        for diam in (0.16, 0.20):
            pc, gc = cylinder_phantom(diam / 2, half=0.14, dx=0.004)
            gd = fb.FanGeometry.covering(gc, 1.0, 1.5, True)
            cells = [kv, f"{100 * diam:.0f} cm"]
            for thick in (0.01, 0.04):
                for grid_deg in (None, 1.5):
                    rs = xs.simulate_scatter(pc, 0.0, spk, gd, n_photons=int(1.2e7 * scale), slice_thickness=thick,
                                             grid_acceptance_deg=grid_deg, backend="numba", seed=11)
                    cc_ = gd.n_det // 2
                    s = slice(cc_ - 5, cc_ + 6)
                    cells.append(f"{rs.scat[s].sum() / rs.prim[s].sum():.3f}")
            data.append(cells)
    rows(["kVp", "diámetro", "SPR corte 1 cm", "1 cm + rejilla ±1.5°", "corte 4 cm", "4 cm + rejilla"], data)

    # (d2) losa gruesa, campo ancho
    gs = Grid2D.from_size(0.5, 0.4, 0.004, x0=-0.25, z0=-0.2)
    gsd = fb.FanGeometry.covering(gs, 1.0, 1.5, True)
    sp125 = xray.tube_spectrum(125.0, 2.5, characteristic=True, anode_angle_deg=12.0)
    data = []
    for thick in (0.10, 0.20, 0.30):
        lab_s = np.where(np.abs(gs.z)[:, None] <= thick / 2, TISSUE_INDEX["water"], TISSUE_INDEX["air"])
        pl = Phantom(gs, (lab_s * np.ones((1, gs.nx), int)).astype(np.int16))
        rs = xs.simulate_scatter(pl, 0.0, sp125, gsd, n_photons=int(1.2e7 * scale), slice_thickness=0.4,
                                 backend="numba", seed=13)
        cc_ = gsd.n_det // 2
        s = slice(cc_ - 5, cc_ + 6)
        data.append((f"{100 * thick:.0f} cm", f"{rs.scat[s].sum() / rs.prim[s].sum():.2f}"))
    print("(d2) Losa de agua (infinita en x e y) de espesor variable, 125 kVp, campo de 40 cm en y a 1 m "
          "(sin rejilla):\n")
    rows(["espesor", "SPR en el eje"], data)

    # (e) TAC
    pe, ge = cylinder_phantom(0.09, "muscle", half=0.1, dx=0.002)
    Xe, Ze = ge.meshgrid()
    pe.labels[np.hypot(Xe - 0.04, Ze) <= 0.012] = TISSUE_INDEX["cortical_bone"]
    pe.labels[np.hypot(Xe + 0.04, Ze) <= 0.012] = TISSUE_INDEX["fat"]
    spe = xray.tube_spectrum(100.0, 3.0)
    ge_g = fb.FanGeometry.covering(ge, 0.6, 1.1, True)
    ct0 = xray.ct_scan(pe, 240, spe, None, geometry=ge_g)
    data = []
    re_ = np.hypot(Xe, Ze)
    for thick in (0.01, 0.04):
        opt = xs.ScatterOptions(n_photons=int(4e6 * scale), n_views=6, slice_thickness=thick, backend="numba")
        ct1 = xray.ct_scan(pe, 240, spe, None, geometry=ge_g, scatter=opt)
        for name, ct in (("sin scatter", ct0), ("con scatter", ct1)):
            ce = ct.hu[re_ < 0.02].mean()
            ed = ct.hu[(re_ > 0.065) & (re_ < 0.083)
                       & (np.hypot(Xe - 0.04, Ze) > 0.02) & (np.hypot(Xe + 0.04, Ze) > 0.02)].mean()
            bone = ct.hu[np.hypot(Xe - 0.04, Ze) < 0.008].mean()
            fat = ct.hu[np.hypot(Xe + 0.04, Ze) < 0.008].mean()
            data.append((f"{100 * thick:.0f} cm", name, f"{ce:.1f}", f"{ed:.1f}", f"{ed - ce:.1f}", f"{bone:.0f}",
                         f"{fat:.0f}", f"{bone - ce:.0f}", f"{ce - fat:.0f}"))
    print("(e) TAC de abanico de un cilindro de músculo de 18 cm con insertos de hueso cortical y grasa "
          "(100 kVp, 3 mm Al, sin ruido, corrección de endurecimiento del agua activada, sin corrección de scatter):\n")
    rows(["corte", "caso", "HU centro", "HU borde", "cupping (borde-centro)", "HU hueso", "HU grasa",
          "contraste hueso-músculo", "contraste músculo-grasa"], data)


# --------------------------------------------------------------------------------------
def section_quality(quick):
    title("4. Detector y calidad de imagen")
    # Radiografía de un borde con desenfoque
    g = Grid2D.from_size(40e-3, 20e-3, 0.25e-3, x0=-20e-3)
    lab = np.full(g.shape, TISSUE_INDEX["air"], np.int16)
    lab[:, g.x >= 0] = TISSUE_INDEX["water"]
    p = Phantom(g, lab)
    sp = xray.monoenergetic(60.0)
    base = xray.acquire(p, [0.0], sp, photons=None)
    sp_ = base.det[1] - base.det[0]
    sel = np.abs(base.det) < 8e-3
    f0, m0 = xdet.mtf_from_edge_samples(base.det[sel], base.intensity[0][sel], sp_, oversample=1, half_width=7e-3)
    data = []
    for sigma in (0.5e-3, 1.0e-3):
        bl = xray.acquire(p, [0.0], sp, photons=None, detector=xdet.DetectorModel(blur_sigma=sigma))
        f, m = xdet.mtf_from_edge_samples(base.det[sel], bl.intensity[0][sel], sp_, oversample=1, half_width=7e-3)
        keep = (f > 0) & (f < 0.5 / (np.pi * sigma))
        ex = np.exp(-2 * (np.pi * sigma * f) ** 2)
        data.append((f"{1e3 * sigma:.1f} mm", f"{np.abs(m / m0 - ex)[keep].max():.3f}",
                     f"{xdet.frequency_at_mtf(f, m / m0):.1f}", f"{np.sqrt(np.log(2) / 2) / (np.pi * sigma):.1f}"))
    print("(a) Radiografía (haz paralelo, 60 keV) de un borde de agua con desenfoque gaussiano del detector; "
          "MTF medida / MTF sin desenfoque frente a la teórica:\n")
    rows(["σ del detector", "error máx. de la MTF (hasta MTF≈0.6)", "f50 medida (ciclos/m)", "f50 teórica"], data)

    # MTF del TAC con detector desenfocado
    gc = Grid2D.from_size(0.1, 0.1, 0.5e-3, x0=-0.05, z0=-0.05)
    Xc, Zc = gc.meshgrid()
    lab = np.full(gc.shape, TISSUE_INDEX["air"], np.int16)
    lab[np.hypot(Xc, Zc) <= 0.03] = TISSUE_INDEX["water"]
    pc = Phantom(gc, lab)
    geo = fb.FanGeometry.covering(gc, 0.5, 1.0, True)
    spc = xray.monoenergetic(60.0)
    sig_d = 0.8e-3
    ct_a = xray.ct_scan(pc, 720, spc, None, geometry=geo)
    ct_b = xray.ct_scan(pc, 720, spc, None, geometry=geo, detector=xdet.DetectorModel(blur_sigma=sig_d))
    out = []
    mt = {}
    for name, ct in (("ideal", ct_a), ("σ_det 0.8 mm", ct_b)):
        d, v = xdet.disk_edge_samples(ct.hu, gc.dx, ((gc.nz - 1) / 2, (gc.nx - 1) / 2), 0.03, 4e-3)
        f, m = xdet.mtf_from_edge_samples(d, v, gc.dx, oversample=4, half_width=4e-3)
        mt[name] = (f, m)
    f, ma = mt["ideal"]
    _, mb = mt["σ_det 0.8 mm"]
    sig_iso = sig_d * 0.5 / 1.0
    for fq in (200.0, 400.0, 600.0, 800.0):
        k = np.argmin(abs(f - fq))
        out.append((f"{fq:.0f}", f"{ma[k]:.3f}", f"{mb[k]:.3f}", f"{mb[k] / ma[k]:.3f}",
                    f"{np.exp(-2 * (np.pi * sig_iso * fq) ** 2):.3f}"))
    print("(b) MTF del TAC de abanico (borde de un cilindro de agua de 6 cm, 720 vistas): el desenfoque del "
          "detector se proyecta al isocentro con el factor sid/sdd = 0.5:\n")
    rows(["f (ciclos/m)", "MTF ideal", "MTF con detector", "cociente", "MTF teórica del detector en el isocentro"], out)

    # NPS del TAC
    rng = np.random.default_rng(0)
    res = []
    for ph in (2e4, 8e4):
        ct = xray.ct_scan(pc, 360, spc, ph, geometry=geo, rng=rng)
        roi = ct.hu[80:120, 80:120]  # centro del cilindro (40x40 píxeles de 0.5 mm)
        f2, _, nps = xdet.noise_power_spectrum(roi, gc.dx, block=20, detrend=True)
        fr, nr = xdet.radial_average(f2, nps, n_bins=8)
        df = f2[1] - f2[0]
        res.append((f"{ph:.0e}", f"{roi.std():.1f}", f"{np.sqrt(nps.sum() * df * df):.1f}",
                    " / ".join(f"{v:.2g}" for v in nr[[1, 3, 5, 7]])))
    print("(c) Ruido del TAC (fotones por elemento y vista). NPS radial en 4 frecuencias (0.19, 0.44, 0.69, 0.94 de la "
          "frecuencia máxima) y comprobación de Parseval (σ medida frente a sqrt(∫NPS)):\n")
    rows(["fotones", "σ de la ROI (HU)", "sqrt(∫NPS) (HU)", "NPS radial (HU² m²)"], res)
    ratio = (float(res[0][1]) / float(res[1][1])) ** 2
    print(f"Cociente de varianzas entre 2e4 y 8e4 fotones: {ratio:.2f} (esperado 4).\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", choices=["hvl", "fan", "scatter", "quality"])
    args = ap.parse_args()
    t0 = time.time()
    todo = args.only or ["hvl", "fan", "scatter", "quality"]
    if "hvl" in todo:
        section_hvl()
    if "fan" in todo:
        section_fan(args.quick)
    if "scatter" in todo:
        section_scatter(args.quick)
    if "quality" in todo:
        section_quality(args.quick)
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
