"""Generación de datasets etiquetados (fractura sí/no + máscara) para entrenar redes neuronales.

Cada muestra es un fichero ``.npz`` con:
  - ``image``: imagen 2D float32 de la modalidad.
  - ``mask``: máscara de la fractura en la malla de la imagen (uint8).
  - ``label``: 1 si hay fractura, 0 si no.
  - ``x_mm``, ``z_mm``: coordenadas de los píxeles.
y ``index.json`` resume todas las muestras con sus metadatos (tipo de
fractura, anatomía, semilla), de modo que cada imagen es reproducible.

Las anatomías se aleatorizan: un tipo de cuerpo (delgado, normal,
sobrepeso, obeso, musculoso, anciano osteoporótico; ver
``wavesim.phantoms.BODY_TYPES``) con variación continua, más ±10 % en cada
grosor y la inclinación del hueso. La rugosidad cambia de realización con la
semilla, pero no de parámetros. Por defecto los fantomas usan el modo realista (hueso poroso, trabéculas,
fascias y periostio; ver ``wavesim.phantoms.realism``). La mitad de las muestras (por defecto)
tiene una fractura aleatoria, que puede ser también interna (intracortical,
de la cortical lejana, contusión ósea o por estrés).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..analysis import resample_mask, resample_mask_apparent_depth
from ..em import xray
from ..mri import mri_scan
from ..phantoms import (FRACTURE_KINDS, BodyType, LimbSpec, RealismSpec, longitudinal_limb, random_fracture,
                        random_limb_spec)
from ..phantoms.phantom import Phantom
from ..ultrasound import simulate_bmode

#: Tipo de cuerpo pedido: ninguno (todos), un nombre, varios nombres o un ``BodyType``.
BodyChoice = str | Sequence[str] | BodyType | None

#: Fracturas que sólo existen bajo la médula: no caben en una imagen que no llegue a la cortical lejana.
_FAR_KINDS = ("cortical_lejana",)
#: Intentos de sortear una fractura que quede dentro de la imagen.
_MAX_FRACTURE_TRIES = 10

MODALITIES = ("ultrasound", "ct", "mri")


# Cada modalidad devuelve (imagen, máscara de la fractura en la malla de la imagen, x, z).
def _ultrasound(ph: Phantom, rng, f0=3e6, angles=(-8.0, 0.0, 8.0), c0=1540.0):
    sc = simulate_bmode(ph, f0, angles=angles, c0=c0)
    img = sc.image
    # La imagen está en profundidad aparente (tiempo × c0): la máscara se lleva a esa escala
    mask = resample_mask_apparent_depth(ph, ph.fracture_mask, img.x, img.z, sc.array.z, c0)
    return img.db.astype(np.float32), mask, img.x, img.z


def _ct(ph: Phantom, rng, n_angles=180, photons=1e5):
    ph = ph.replace_tissue("water", "air")
    ct = xray.ct_scan(ph, n_angles=n_angles, photons=photons, rng=rng)
    g = ph.grid
    return ct.hu.astype(np.float32), resample_mask(ph, ph.fracture_mask, g.x, g.z), g.x, g.z


def _mri(ph: Phantom, rng, sequence="STIR", voxel=0.4e-3, snr=30.0):
    ph = ph.replace_tissue("water", "air")
    im = mri_scan(ph, sequence, voxel=voxel, snr=snr, rng=rng)
    # El fantoma es mucho más fino que el vóxel: se marca cada vóxel que contiene
    # grieta (en vez de muestrear un punto), para que una fisura no desaparezca.
    return im.magnitude.astype(np.float32), _block_any(ph.fracture_mask, *im.magnitude.shape), im.x, im.z


def _block_any(mask: np.ndarray, m_rows: int, m_cols: int) -> np.ndarray:
    """Reduce una máscara fina a (m_rows, m_cols): True si algún píxel fino del bloque lo es.

    El píxel k de la imagen de RM está centrado en el índice fino k·n/m (ver
    ``MRIImage.x``) y abarca ±n/(2m) celdas.
    """
    def bounds(n: int, m: int):
        k = np.arange(m)
        lo = np.clip(np.round((k - 0.5) * n / m).astype(int), 0, n)
        hi = np.clip(np.round((k + 0.5) * n / m).astype(int), 0, n)
        return lo, np.maximum(hi, lo + 1)

    rows = np.stack([mask[a:b].any(axis=0) for a, b in zip(*bounds(mask.shape[0], m_rows), strict=True)])
    return np.stack([rows[:, a:b].any(axis=1) for a, b in zip(*bounds(mask.shape[1], m_cols), strict=True)], axis=1)


_RUNNERS = {"ultrasound": _ultrasound, "ct": _ct, "mri": _mri}
# Resolución del fantoma para cada modalidad (m). La RM se genera mucho más fina
# que su vóxel (0.4 mm) porque el recorte del k-space produce el volumen parcial
# real; el TAC se reconstruye a 0.125 mm (TAC de alta resolución de extremidades).
_DX = {"ultrasound": 1450.0 / 3e6 / 8, "ct": 0.125e-3, "mri": 0.05e-3}


def _kinds_in_view(spec: LimbSpec, depth: float | None, kinds: Sequence[str]) -> tuple[str, ...]:
    """Quita los tipos de fractura que quedarían por debajo de la imagen."""
    far_periosteum = spec.standoff + sum(t for _, t in spec.layers()[:6])
    if depth is None or far_periosteum + 0.5e-3 <= depth:
        return tuple(kinds)
    return tuple(k for k in kinds if k not in _FAR_KINDS)


def auto_depth(spec: LimbSpec, minimum: float = 22e-3, below_cortex: float = 3e-3) -> float:
    """Profundidad de imagen que llega a la mitad de la médula (pasada la cortical cercana).

    Con una profundidad fija, en un muslo (hueso a 40-50 mm) la imagen no
    contendría hueso y todas las muestras saldrían sin fractura.
    """
    layers = spec.layers()
    near_cortex_bottom = spec.standoff + sum(t for _, t in layers[:4])
    return max(minimum, near_cortex_bottom + 0.5 * layers[4][1] + below_cortex)


def make_phantom(rng: np.random.Generator, preset: str, fracture: bool, dx: float, width: float,
                 depth: float | str | None, body: BodyChoice = None,
                 fracture_kinds: Sequence[str] | None = None,
                 realism: str | RealismSpec | None = "realista") -> Phantom:
    """Fantoma longitudinal aleatorio, con un tipo de cuerpo y (si se pide) una fractura.

    ``depth``: profundidad de la imagen (m), ``"auto"`` (ver ``auto_depth``) o
    ``None`` (la extremidad completa).
    ``realism``: microestructura del fantoma (``"realista"`` por defecto: hueso
    poroso, trabéculas, fascias...; ``None`` o ``"basico"`` para el fantoma de
    capas lisas). La calidad ósea sale del tipo de cuerpo sorteado.

    Si se pide fractura, se garantiza que la grieta cae dentro de la imagen:
    las fracturas de la cortical lejana se descartan cuando ``depth`` no
    llega a ella y se vuelve a sortear la fractura si su máscara queda vacía
    (por ejemplo, una contusión profunda en una pierna obesa). Si aun así no
    hay fractura visible, se lanza un error en vez de etiquetar mal la muestra.
    Las grietas se dibujan con al menos una celda de ancho para que las fisuras
    finas no queden punteadas.
    """
    spec = random_limb_spec(rng, preset, body=body)
    if depth == "auto":
        depth = auto_depth(spec)
    if not fracture:
        return longitudinal_limb(spec, None, width=width, depth=depth, dx=dx, seed=int(rng.integers(0, 2**31 - 1)),
                                 realism=realism)
    kinds = _kinds_in_view(spec, depth, fracture_kinds or FRACTURE_KINDS)
    if not kinds:
        raise ValueError(f"Ningún tipo de {fracture_kinds} cabe en una imagen de {depth * 1e3:.0f} mm de profundidad")
    for _ in range(_MAX_FRACTURE_TRIES):
        fr = replace(random_fracture(rng, kinds=kinds), min_gap_cells=1.0)
        ph = longitudinal_limb(spec, fr, width=width, depth=depth, dx=dx, seed=int(rng.integers(0, 2**31 - 1)),
                               realism=realism)
        if ph.has_fracture:
            return ph
    raise ValueError(f"No se consiguió una fractura visible en {_MAX_FRACTURE_TRIES} intentos "
                     f"(preset {preset}, profundidad {depth}): revisa la profundidad de la imagen")


def generate_sample(rng: np.random.Generator, modality: str, preset: str = "tibia", fracture: bool | None = None,
                    fracture_prob: float = 0.5, width: float = 24e-3, depth: float | str | None = "auto",
                    body: BodyChoice = None, fracture_kinds: Sequence[str] | None = None,
                    realism: str | RealismSpec | None = "realista", **kw) -> dict:
    """Genera una muestra (imagen + etiqueta + máscara) de la modalidad indicada.

    ``depth``: profundidad de la imagen (m), ``"auto"`` (hasta la mitad de la
    médula, según la anatomía sorteada) o ``None`` (extremidad completa).
    ``body``: tipo de cuerpo (``None`` = se sortea entre todos los de
    ``BODY_TYPES``; un nombre; una lista de nombres; o un ``BodyType``).
    ``fracture_kinds``: tipos de fractura entre los que sortear (todos por
    defecto; p. ej. ``INTERNAL_FRACTURE_KINDS`` para un dataset de fracturas
    ocultas).
    """
    if modality not in _RUNNERS:
        raise ValueError(f"Modalidad '{modality}' no soportada; usa una de {MODALITIES}")
    if fracture is None:
        fracture = bool(rng.random() < fracture_prob)
    ph = make_phantom(rng, preset, fracture, _DX[modality], width, depth, body, fracture_kinds, realism)
    image, mask, x, z = _RUNNERS[modality](ph, rng, **kw)
    mask = mask.astype(np.uint8)
    return {"image": image, "mask": mask, "label": int(ph.has_fracture), "x_mm": x * 1e3, "z_mm": z * 1e3,
            "meta": ph.meta}


def generate_dataset(out_dir: str | Path, n: int, modality: str = "ct", preset: str = "tibia", seed: int = 0,
                     fracture_prob: float = 0.5, verbose: bool = True, body: BodyChoice = None,
                     fracture_kinds: Sequence[str] | None = None, realism: str | RealismSpec | None = "realista",
                     **kw) -> Path:
    """Escribe ``n`` muestras en ``out_dir`` y un índice ``index.json`` (ver ``generate_sample``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    index = []
    t0 = time.time()
    for k in range(n):
        s = generate_sample(rng, modality, preset, fracture_prob=fracture_prob, body=body,
                            fracture_kinds=fracture_kinds, realism=realism, **kw)
        name = f"{modality}_{k:05d}.npz"
        np.savez_compressed(out / name, image=s["image"], mask=s["mask"], label=s["label"], x_mm=s["x_mm"],
                            z_mm=s["z_mm"])
        fr = s["meta"]["fracture"]
        kind = fr["kind"] if (fr and s["label"]) else None
        index.append({"file": name, "label": s["label"], "fracture_kind": kind,
                      "body_type": s["meta"]["body_type"], "meta": s["meta"]})
        if verbose:
            print(f"[{k + 1}/{n}] {name} fractura={s['label']} cuerpo={s['meta']['body_type']} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    body_desc = asdict(body) if isinstance(body, BodyType) else (body if body is None or isinstance(body, str)
                                                                   else list(body))
    with open(out / "index.json", "w", encoding="utf-8") as fh:
        realism_desc = realism if realism is None or isinstance(realism, str) else asdict(realism)
        json.dump({"modality": modality, "preset": preset, "seed": seed, "body": body_desc, "realism": realism_desc,
                   "fracture_kinds": list(fracture_kinds) if fracture_kinds else None, "samples": index},
                  fh, indent=1, ensure_ascii=False, default=float)
    return out
