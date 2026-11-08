"""Fantoma básico frente a realista: la misma tibia en modo B (3 MHz), TAC y RM.

El modo realista (``realism="realista"``) añade a la anatomía de capas:

- porosidad cortical (canales de Havers y, en el anciano osteoporótico,
  cavidades de resorción y trabecularización endocortical);
- rugosidad fina autoafín en periostio y endostio;
- fascia superficial y profunda, septos de los lobulillos grasos, perimisio
  orientado según las fibras y periostio.

Con la misma semilla la anatomía de capas es idéntica: sólo cambia lo que
añade el modo realista. Se comparan dos piernas, cada una básica y
realista: un adulto sano y el tipo de cuerpo "anciano_osteoporotico"
(cortical más fina; en el modo realista, además, porosa y trabecularizada
por dentro). Cifras que se miden en el modo B:

- textura de la grasa subcutánea: nivel medio (dB respecto al máximo de la
  imagen) y SNR de la envolvente (media / desviación típica; 1.91 es el
  speckle totalmente desarrollado de Rayleigh, menos indica ecos
  especulares como los de las fascias);
- eco cortical: máximo de cada columna junto a la superficie del hueso,
  media y dispersión lateral (dB);
- sombra: nivel medio en la médula bajo la cortical cercana respecto al eco
  cortical (dB; más negativo = más sombra).

En TAC, el HU medio y su dispersión en la cortical; en RM (T1 y UTE), la
señal de la cortical respecto al músculo y la textura del músculo.

Uso:  python examples/12_realismo.py   (~1-2 min por ecografía)
→ outputs/12_realismo.png, outputs/12_microestructura.png
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavesim.analysis import resample_mask
from wavesim.em import xray
from wavesim.mri import mri_scan
from wavesim.phantoms import longitudinal_limb
from wavesim.ultrasound import simulate_bmode
from wavesim.viz import plot_image, plot_phantom, save

OUT = Path(__file__).resolve().parent.parent / "outputs"
F0 = 3e6
DX_US = 1450 / F0 / 8
SEED = 1
CASES = {
    "básica": dict(realism=None),
    "realista": dict(realism="realista"),
    "básica, anciano": dict(realism=None, body="anciano_osteoporotico"),
    "realista, anciano": dict(realism="realista", body="anciano_osteoporotico"),
}


def bmode_metrics(p, img) -> dict:
    """Speckle de la grasa, eco cortical y sombra en la imagen de modo B (ver el docstring)."""
    env = img.envelope / img.envelope.max()
    db = 20 * np.log10(np.maximum(env, 1e-5))
    fat = resample_mask(p, p.mask("fat"), img.x, img.z)
    z = img.z[:, None]
    top, skin = p.meta["bone_top_depth"], p.meta["skin_depth"]
    fat_roi = fat & (z > skin + 2.0e-3) & (z < top - 0.8e-3)
    cortex = (z > top - 0.6e-3) & (z < top + 0.6e-3)
    echo = np.where(cortex, db, -np.inf).max(axis=0)
    cort_thick = p.meta["spec"]["cortex"]
    shadow = np.broadcast_to((z > top + cort_thick + 1.0e-3) & (z < top + cort_thick + 6.0e-3), db.shape)
    e = env[fat_roi]
    return {"speckle_db": float(db[fat_roi].mean()), "speckle_snr": float(e.mean() / e.std()),
            "echo_db": float(echo.mean()), "echo_std_db": float(echo.std()),
            "shadow_db": float(db[shadow].mean() - echo.mean())}


def ct_metrics(p, ct) -> dict:
    cort = p.mask("cortical_bone")
    return {"hu_cortex": float(ct.hu[cort].mean()), "hu_cortex_std": float(ct.hu[cort].std())}


def mri_metrics(p, im) -> dict:
    cort = resample_mask(p, p.mask("cortical_bone"), im.x, im.z)
    mus = resample_mask(p, p.mask("muscle") | p.mask("fascia"), im.x, im.z)
    mus &= ~resample_mask(p, p.mask("blood"), im.x, im.z)
    m = im.magnitude
    return {"cortex_vs_muscle": float(m[cort].mean() / m[mus].mean()),
            "muscle_texture": float(m[mus].std() / m[mus].mean())}


def main():
    t0 = time.time()
    rows = {}
    fig, axs = plt.subplots(5, len(CASES), figsize=(6 * len(CASES), 26))
    for col, (name, kw) in enumerate(CASES.items()):
        # Ecografía: 24 × 22 mm a λ/8, tres ondas planas compuestas
        p = longitudinal_limb("tibia", width=24e-3, depth=22e-3, dx=DX_US, seed=SEED, **kw)
        t = time.time()
        sc = simulate_bmode(p, F0, angles=(-8.0, 0.0, 8.0))
        print(f"[{name}] ecografía en {time.time() - t:.0f} s", flush=True)
        r = bmode_metrics(p, sc.image)
        plot_phantom(p, axs[0, col], f"Fantoma {name}", legend=(col == len(CASES) - 1))
        plot_image(sc.image.db, sc.image.extent_mm(), axs[1, col], f"Modo B 3 MHz — {name}", vmin=-50, vmax=0)

        # TAC y RM de la pierna entera (sin recortar en profundidad), rodeada de aire
        q = longitudinal_limb("tibia", width=24e-3, dx=0.1e-3, seed=SEED, **kw).replace_tissue("water", "air")
        ct = xray.ct_scan(q, n_angles=360, photons=2e5)
        r.update(ct_metrics(q, ct))
        plot_image(ct.hu, q.grid.extent_mm(), axs[2, col], f"TAC 120 kVp — {name}", vmin=-200, vmax=2200, label="HU")
        for row, seq in ((3, "T1"), (4, "UTE")):
            im = mri_scan(q, seq, voxel=0.3e-3, snr=40)
            for k, v in mri_metrics(q, im).items():
                r[f"{seq}_{k}"] = v
            plot_image(im.magnitude, im.extent_mm, axs[row, col], f"RM {im.sequence.name} — {name}",
                       vmax=im.magnitude.max() * 0.8)
        if p.porosity is not None:
            r["porosidad_cortical"] = p.meta["microstructure"]["cortical_porosity"]
        rows[name] = r
    fig.tight_layout()
    print("Guardado:", save(fig, OUT / "12_realismo.png", dpi=70))

    _microstructure_figure()

    keys = list(rows["realista"])
    width = max(len(k) for k in keys)
    print("\n" + " " * width + "".join(f"{n[:22]:>24s}" for n in rows))
    for k in keys:
        print(f"{k:<{width}s}" + "".join(f"{rows[n].get(k, float('nan')):24.2f}" for n in rows))
    for name, r, base in (("Adulto sano", rows["realista"], rows["básica"]),
                          ("Anciano osteoporótico", rows["realista, anciano"], rows["básica, anciano"])):
        print(f"{name}, realista frente a básica: grasa {r['speckle_db'] - base['speckle_db']:+.1f} dB, "
              f"eco cortical {r['echo_db'] - base['echo_db']:+.1f} dB, "
              f"sombra {r['shadow_db'] - base['shadow_db']:+.1f} dB, "
              f"HU cortical {r['hu_cortex'] - base['hu_cortex']:+.0f}")
    print(f"Total: {time.time() - t0:.0f} s")


def _microstructure_figure():
    """Detalle de la microestructura: cortical sana y osteoporótica, y red trabecular de la metáfisis."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    kw = dict(width=8e-3, depth=12e-3, dx=20e-6, seed=SEED, realism="realista")
    for ax, (title, extra) in zip(axs[:2], (("Cortical sana (5 %)", {}),
                                            ("Anciano osteoporótico (~18 %)", {"body": "anciano_osteoporotico"}))):
        p = longitudinal_limb("tibia", **kw, **extra)
        plot_phantom(p, ax, title, legend=False)
        ax.set_ylim(p.meta["bone_top_depth"] * 1e3 + 5, p.meta["bone_top_depth"] * 1e3 - 2)
    m = longitudinal_limb("tibia_metafisis", width=12e-3, depth=14e-3, dx=40e-6, seed=SEED, realism="realista")
    info = m.meta["microstructure"]
    plot_phantom(m, axs[2], f"Metáfisis: red trabecular BV/TV = {info['bvtv']:.2f}, "
                            f"Tb.Th = {info['trabecular_thickness'] * 1e3:.2f} mm", legend=True)
    fig.tight_layout()
    print("Guardado:", save(fig, OUT / "12_microestructura.png", dpi=90))


if __name__ == "__main__":
    main()
