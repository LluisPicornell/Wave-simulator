"""Detector de rayos X y herramientas de calidad de imagen (MTF por el borde, espectro de ruido).

Modelo del detector (``DetectorModel``), aplicado a la señal esperada antes del ruido:

- desenfoque del centelleador: gaussiana de desviación ``blur_sigma`` (MTF ``exp(-2π²σ²f²)``);
- apertura del píxel: promedio sobre ``pixel_pitch`` (MTF ``sinc(pitch·f)``);
- ruido electrónico: gaussiano de ``electronic_noise`` fotones equivalentes rms por lectura.

El desenfoque se aplica al valor esperado y luego se añade el ruido de Poisson: es una aproximación
(no reproduce la correlación espacial del ruido que genera un centelleador real).

Medida de calidad de imagen:

- ``mtf_from_edge_samples`` y los generadores de muestras ``slanted_edge_samples`` (borde recto casi
  alineado con los píxeles) y ``disk_edge_samples`` (borde circular, para el TAC): función de
  transferencia de modulación por el método de la función de borde sobremuestreada;
- ``noise_power_spectrum``: espectro de potencia de ruido 2D de una región uniforme, normalizado de
  modo que ``sum(NPS) · df_x · df_z`` es la varianza del ruido.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorModel:
    pixel_pitch: float = 0.0  # apertura del píxel (m); 0 = ideal (sin sinc)
    blur_sigma: float = 0.0  # desenfoque gaussiano del centelleador (m)
    electronic_noise: float = 0.0  # rms en fotones equivalentes

    def mtf(self, freq: np.ndarray) -> np.ndarray:
        """MTF teórica del desenfoque y la apertura (``freq`` en ciclos/m)."""
        f = np.asarray(freq, float)
        return np.exp(-2.0 * (np.pi * self.blur_sigma * f) ** 2) * np.sinc(self.pixel_pitch * f)


def blur_signal(signal: np.ndarray, spacing: float, model: DetectorModel) -> np.ndarray:
    """Aplica la MTF del detector a lo largo del último eje (relleno por reflexión en los bordes)."""
    if model.blur_sigma <= 0 and model.pixel_pitch <= 0:
        return signal
    n = signal.shape[-1]
    pad = n // 2 + 8
    ext = np.pad(signal, [(0, 0)] * (signal.ndim - 1) + [(pad, pad)], mode="edge")
    freq = np.fft.rfftfreq(ext.shape[-1], d=spacing)
    out = np.fft.irfft(np.fft.rfft(ext, axis=-1) * model.mtf(freq), n=ext.shape[-1], axis=-1)
    return out[..., pad:pad + n]


def detect_counts(expected: np.ndarray, spacing: float, model: DetectorModel | None,
                  rng: np.random.Generator) -> np.ndarray:
    """Cuentas medidas a partir de las esperadas: desenfoque, Poisson y ruido electrónico (con suelo en 1)."""
    if model is not None:
        expected = blur_signal(expected, spacing, model)
    counts = rng.poisson(expected).astype(float)
    if model is not None and model.electronic_noise > 0:
        counts = counts + rng.normal(0.0, model.electronic_noise, counts.shape)
    return np.maximum(counts, 1.0)


# --------------------------------------------------------------------------------------
# MTF por el método del borde
# --------------------------------------------------------------------------------------
def slanted_edge_samples(img: np.ndarray, pixel_size: float, axis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Muestras ``(distancia al borde, valor)`` de una imagen con un borde recto casi paralelo a una fila/columna.

    ``axis=1``: el borde es casi vertical (la imagen cambia a lo largo de x, columnas). La posición
    del borde en cada fila es el centroide de su derivada; se ajusta una recta y se calcula la distancia
    firmada de cada píxel (m).
    """
    a = img if axis == 1 else img.T
    nrow, ncol = a.shape
    x = np.arange(ncol, dtype=float)
    deriv = np.abs(np.diff(a, axis=1))
    xc = (deriv * (x[:-1] + 0.5)).sum(axis=1) / np.maximum(deriv.sum(axis=1), 1e-300)
    rows = np.arange(nrow, dtype=float)
    slope, icpt = np.polyfit(rows, xc, 1)
    dist = (x[None, :] - (icpt + slope * rows[:, None])) / np.sqrt(1.0 + slope**2)
    return dist.ravel() * pixel_size, a.ravel()


def disk_edge_samples(img: np.ndarray, pixel_size: float, center: tuple[float, float], radius: float,
                      half_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Muestras ``(r - radio, valor)`` en una corona de ``±half_width`` metros alrededor de un borde circular.

    ``center``: ``(j, i)`` fraccionarios en píxeles. Útil en el TAC, donde un cilindro da un borde en todos
    los ángulos; ``half_width`` debe ser mucho menor que ``radius`` para ignorar la curvatura.
    """
    jj, ii = np.indices(img.shape)
    r = np.hypot(jj - center[0], ii - center[1]) * pixel_size - radius
    sel = np.abs(r) <= half_width
    return r[sel], img[sel]


def mtf_from_edge_samples(dist: np.ndarray, values: np.ndarray, pixel_size: float, oversample: int = 4,
                          half_width: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MTF ``(frecuencias en ciclos/m, valores)`` a partir de muestras de la función de borde (ESF).

    Se promedia la ESF en rejillas de ``pixel_size / oversample``, se deriva (LSF), se aplica una ventana
    de Hann y la FFT, se compensa la respuesta del binning y de la derivada y se normaliza a 1 en frecuencia
    cero. ``half_width`` (m) recorta la ESF a ``±half_width`` del borde.
    """
    step = pixel_size / oversample
    if half_width is None:
        half_width = min(-dist.min(), dist.max())
    sel = np.abs(dist) <= half_width
    d, v = dist[sel], values[sel]
    nbin = int(np.ceil(2 * half_width / step))
    idx = np.clip(((d + half_width) / step).astype(int), 0, nbin - 1)
    cnt = np.bincount(idx, minlength=nbin)
    esf = np.bincount(idx, weights=v, minlength=nbin) / np.maximum(cnt, 1)
    good = cnt > 0
    esf = np.interp(np.arange(nbin), np.nonzero(good)[0], esf[good])
    lsf = np.gradient(esf)
    lsf = lsf * np.hanning(nbin + 2)[1:-1] if lsf.size > 2 else lsf
    mtf = np.abs(np.fft.rfft(lsf, n=4 * nbin))
    freq = np.fft.rfftfreq(4 * nbin, d=step)
    # Compensa el promedio por rejillas (caja de ancho ``step``) y la derivada centrada (sinc(2·f·step))
    mtf = mtf / (np.sinc(freq * step) * np.sinc(2.0 * freq * step))
    return freq, mtf / mtf[0]


def frequency_at_mtf(freq: np.ndarray, mtf: np.ndarray, level: float = 0.5) -> float:
    """Primera frecuencia a la que la MTF cae por debajo de ``level`` (interpolación lineal)."""
    below = np.nonzero(mtf < level)[0]
    if below.size == 0:
        return float("nan")
    k = below[0]
    if k == 0:
        return float(freq[0])
    return float(np.interp(level, [mtf[k], mtf[k - 1]], [freq[k], freq[k - 1]]))


# --------------------------------------------------------------------------------------
# Espectro de potencia de ruido
# --------------------------------------------------------------------------------------
def noise_power_spectrum(img: np.ndarray, pixel_size: float, block: int = 32, detrend: bool = True,
                         step: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NPS 2D de una región uniforme: ``(fz, fx, nps)`` con las frecuencias en ciclos/m (``fftshift``).

    La región se divide en bloques de ``block × block`` (solapados cada ``step``, por defecto sin
    solape); a cada uno se le resta su plano de mejor ajuste (o la media si ``detrend=False``) y se promedia
    ``|FFT|² · dx² / block²``. Con esa normalización ``sum(nps) · df²`` es la varianza del ruido.
    """
    step = step or block
    jj, ii = np.mgrid[:block, :block]
    design = np.column_stack([np.ones(block * block), jj.ravel(), ii.ravel()])
    acc = np.zeros((block, block))
    count = 0
    for j0 in range(0, img.shape[0] - block + 1, step):
        for i0 in range(0, img.shape[1] - block + 1, step):
            tile = img[j0:j0 + block, i0:i0 + block]
            if detrend:
                coef, *_ = np.linalg.lstsq(design, tile.ravel(), rcond=None)
                tile = tile - (design @ coef).reshape(block, block)
            else:
                tile = tile - tile.mean()
            acc += np.abs(np.fft.fft2(tile)) ** 2
            count += 1
    if count == 0:
        raise ValueError("La región es menor que un bloque")
    nps = np.fft.fftshift(acc / count) * pixel_size**2 / block**2
    f = np.fft.fftshift(np.fft.fftfreq(block, d=pixel_size))
    return f, f, nps


def radial_average(f: np.ndarray, nps: np.ndarray, n_bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Promedio radial de un espectro 2D ``(fz, fx)`` centrado: ``(frecuencia radial, valor)``."""
    fz, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.hypot(fz, fx)
    n_bins = n_bins or f.size // 2
    edges = np.linspace(0.0, f.max(), n_bins + 1)
    idx = np.clip(np.digitize(fr.ravel(), edges) - 1, 0, n_bins)
    sel = idx < n_bins
    s = np.bincount(idx[sel], weights=nps.ravel()[sel], minlength=n_bins)
    c = np.bincount(idx[sel], minlength=n_bins)
    return 0.5 * (edges[1:] + edges[:-1]), s / np.maximum(c, 1)
