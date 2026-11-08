"""Generación de datasets etiquetados (fractura sí/no + máscara) para entrenar redes neuronales.

Cada muestra es un fichero ``.npz`` con:
  - ``image``: imagen 2D float32 de la modalidad.
  - ``mask``: máscara de la fractura en la malla de la imagen (uint8).
  - ``label``: 1 si hay fractura, 0 si no.
  - ``x_mm``, ``z_mm``: coordenadas de los píxeles.
y ``index.json`` resume todas las muestras con sus metadatos (tipo de
fractura, anatomía, semilla), de modo que cada imagen es reproducible.

Las anatomías se aleatorizan (±20 % en grosores, inclinación, rugosidad) y
la mitad de las muestras (por defecto) tiene una fractura aleatoria.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..analysis import resample_mask
from ..em import xray
from ..mri import mri_scan
from ..phantoms import longitudinal_limb, random_fracture, random_limb_spec
from ..phantoms.phantom import Phantom
from ..ultrasound import simulate_bmode

MODALITIES = ("ultrasound", "ct", "mri")


def _ultrasound(ph: Phantom, rng, f0=3e6, angles=(-8.0, 0.0, 8.0)):
    sc = simulate_bmode(ph, f0, angles=angles)
    img = sc.image
    return img.db.astype(np.float32), img.x, img.z


def _ct(ph: Phantom, rng, n_angles=180, photons=1e5):
    ph = ph.replace_tissue("water", "air")
    ct = xray.ct_scan(ph, n_angles=n_angles, photons=photons, rng=rng)
    return ct.hu.astype(np.float32), ph.grid.x, ph.grid.z


def _mri(ph: Phantom, rng, sequence="STIR", voxel=0.4e-3, snr=30.0):
    ph = ph.replace_tissue("water", "air")
    im = mri_scan(ph, sequence, voxel=voxel, snr=snr, rng=rng)
    nz, nx = im.magnitude.shape
    g = ph.grid
    x = g.x0 + (np.arange(nx) + 0.5) * g.width / nx - g.dx / 2
    z = g.z0 + (np.arange(nz) + 0.5) * g.depth / nz - g.dx / 2
    return im.magnitude.astype(np.float32), x, z


_RUNNERS = {"ultrasound": _ultrasound, "ct": _ct, "mri": _mri}
# Resolución del fantoma para cada modalidad (m)
_DX = {"ultrasound": 1450.0 / 3e6 / 8, "ct": 0.25e-3, "mri": 0.2e-3}


def make_phantom(rng: np.random.Generator, preset: str, fracture: bool, dx: float, width: float, depth: float | None):
    spec = random_limb_spec(rng, preset)
    fr = random_fracture(rng) if fracture else None
    seed = int(rng.integers(0, 2**31 - 1))
    return longitudinal_limb(spec, fr, width=width, depth=depth, dx=dx, seed=seed)


def generate_sample(rng: np.random.Generator, modality: str, preset: str = "tibia", fracture: bool | None = None,
                    fracture_prob: float = 0.5, width: float = 24e-3, depth: float | None = 22e-3, **kw) -> dict:
    """Genera una muestra (imagen + etiqueta + máscara) de la modalidad indicada."""
    if modality not in _RUNNERS:
        raise ValueError(f"Modalidad '{modality}' no soportada; usa una de {MODALITIES}")
    if fracture is None:
        fracture = bool(rng.random() < fracture_prob)
    ph = make_phantom(rng, preset, fracture, _DX[modality], width, depth)
    image, x, z = _RUNNERS[modality](ph, rng, **kw)
    mask = resample_mask(ph, ph.fracture_mask, x, z).astype(np.uint8)
    return {"image": image, "mask": mask, "label": int(ph.has_fracture), "x_mm": x * 1e3, "z_mm": z * 1e3,
            "meta": ph.meta}


def generate_dataset(out_dir: str | Path, n: int, modality: str = "ct", preset: str = "tibia", seed: int = 0,
                     fracture_prob: float = 0.5, verbose: bool = True, **kw) -> Path:
    """Escribe ``n`` muestras en ``out_dir`` y un índice ``index.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    index = []
    t0 = time.time()
    for k in range(n):
        s = generate_sample(rng, modality, preset, fracture_prob=fracture_prob, **kw)
        name = f"{modality}_{k:05d}.npz"
        np.savez_compressed(out / name, image=s["image"], mask=s["mask"], label=s["label"], x_mm=s["x_mm"],
                            z_mm=s["z_mm"])
        fr = s["meta"]["fracture"]
        index.append({"file": name, "label": s["label"], "fracture_kind": fr["kind"] if fr else None,
                      "meta": s["meta"]})
        if verbose:
            print(f"[{k + 1}/{n}] {name} fractura={s['label']} ({time.time() - t0:.0f} s)", flush=True)
    with open(out / "index.json", "w", encoding="utf-8") as fh:
        json.dump({"modality": modality, "preset": preset, "seed": seed, "samples": index}, fh, indent=1,
                  ensure_ascii=False, default=float)
    return out
