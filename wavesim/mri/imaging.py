"""Formación de imagen de resonancia: k-space, resolución finita y ruido.

1. Mapa ideal de señal a la resolución del fantoma (ecuación de señal por tejido).
2. Transformada de Fourier → k-space. Se conserva sólo la zona central que
   cubre la matriz de adquisición: eso fija la resolución real y produce los
   artefactos de truncamiento (Gibbs) y de volumen parcial.
3. Ruido gaussiano complejo (térmico) → la magnitud tiene estadística de Rice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..phantoms.phantom import Phantom
from .sequences import Sequence, resolve, tissue_signal


@dataclass
class MRIImage:
    magnitude: np.ndarray
    ideal: np.ndarray  # señal sin ruido a la resolución del fantoma
    sequence: Sequence
    b0: float
    voxel: float  # tamaño de píxel adquirido (m)
    extent_mm: tuple[float, float, float, float]
    snr: float
    x: np.ndarray  # coordenada lateral (m) del centro de cada columna
    z: np.ndarray  # profundidad (m) del centro de cada fila


def mri_scan(phantom: Phantom, sequence: Sequence | str = "T2", b0: float = 1.5, voxel: float = 0.4e-3,
             snr: float = 40.0, texture: float = 0.02, rng: np.random.Generator | None = None) -> MRIImage:
    """Imagen de RM 2D del corte del fantoma.

    ``voxel``: resolución en el plano (m); debe ser ≥ dx del fantoma.
    ``snr``: relación señal/ruido en el músculo.
    ``texture``: variación aleatoria de la señal dentro de cada tejido.
    """
    seq = resolve(sequence, b0)
    rng = rng or np.random.default_rng(0)
    g = phantom.grid
    # La señal de una celda porosa es la suma de la de cada fracción de volumen (volumen parcial)
    sig = {k: tissue_signal(k, seq, b0) for k in phantom.property_tissues()}
    ideal = phantom.property_map(lambda k: sig[k], physics="mri")
    if texture and phantom.heterogeneity is not None:
        ideal = ideal * (1 + texture * np.clip(phantom.heterogeneity, -3, 3))

    # Matriz de adquisición: (nz, nx) recortada al voxel pedido
    mz = max(min(int(round(g.depth / voxel)), g.nz), 8)
    mx = max(min(int(round(g.width / voxel)), g.nx), 8)
    k = np.fft.fftshift(np.fft.fft2(ideal))
    cz, cx = g.nz // 2, g.nx // 2
    kc = k[cz - mz // 2: cz - mz // 2 + mz, cx - mx // 2: cx - mx // 2 + mx]
    img = np.fft.ifft2(np.fft.ifftshift(kc)) * (mz * mx) / (g.nz * g.nx)

    present = phantom.present_tissues()
    ref = sig["muscle"] if "muscle" in present else max(sig[k] for k in present)
    sigma = ref / snr if snr else 0.0
    if sigma:
        img = img + rng.normal(0, sigma, img.shape) + 1j * rng.normal(0, sigma, img.shape)
    # Al recortar el k-space la muestra k de la DFT inversa corresponde a la posición
    # x0 + k·W/n (no a la de los centros de celda del fantoma, x0 + k·dx): los centros
    # reales son éstos, y el borde de la imagen está medio píxel más allá.
    x = g.x0 + np.arange(mx) * g.width / mx
    z = g.z0 + np.arange(mz) * g.depth / mz
    hx, hz = 0.5 * g.width / mx, 0.5 * g.depth / mz
    extent = ((x[0] - hx) * 1e3, (x[-1] + hx) * 1e3, (z[-1] + hz) * 1e3, (z[0] - hz) * 1e3)
    return MRIImage(np.abs(img), ideal, seq, b0, g.width / mx, extent, snr, x, z)
