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
    sig = {k: tissue_signal(k, seq, b0) for k in phantom.present_tissues()}
    ideal = phantom.property_map(lambda k: sig[k])
    if texture and phantom.heterogeneity is not None:
        ideal = ideal * (1 + texture * np.clip(phantom.heterogeneity, -3, 3))

    # Matriz de adquisición: (nz, nx) recortada al voxel pedido
    mz = max(min(int(round(g.depth / voxel)), g.nz), 8)
    mx = max(min(int(round(g.width / voxel)), g.nx), 8)
    k = np.fft.fftshift(np.fft.fft2(ideal))
    cz, cx = g.nz // 2, g.nx // 2
    kc = k[cz - mz // 2: cz - mz // 2 + mz, cx - mx // 2: cx - mx // 2 + mx]
    img = np.fft.ifft2(np.fft.ifftshift(kc)) * (mz * mx) / (g.nz * g.nx)

    ref = sig.get("muscle", max(sig.values()))
    sigma = ref / snr if snr else 0.0
    if sigma:
        img = img + rng.normal(0, sigma, img.shape) + 1j * rng.normal(0, sigma, img.shape)
    return MRIImage(np.abs(img), ideal, seq, b0, g.width / mx, g.extent_mm(), snr)
