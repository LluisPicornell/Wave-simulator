"""Métricas para comparar modalidades: ¿cuánto se nota la fractura en la imagen?"""

from __future__ import annotations

import numpy as np

from .phantoms.phantom import Phantom


def resample_mask(phantom: Phantom, mask: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Lleva una máscara del fantoma a otra malla de píxeles (vecino más cercano)."""
    g = phantom.grid
    ii = np.clip(np.round((x - g.x0) / g.dx).astype(int), 0, g.nx - 1)
    jj = np.clip(np.round((z - g.z0) / g.dx).astype(int), 0, g.nz - 1)
    return mask[np.ix_(jj, ii)]


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Dilatación binaria con un cuadrado de lado 2r+1 (sin SciPy)."""
    out = mask.copy()
    for _ in range(radius_px):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def fracture_visibility(healthy: np.ndarray, fractured: np.ndarray, roi: np.ndarray,
                        noise_sigma: float | None = None) -> dict:
    """Cuánto cambia la imagen por la fractura dentro de la región de interés.

    - ``change_ratio``: energía del cambio en la ROI / energía de la imagen sana en la ROI.
    - ``cnr``: cambio medio en la ROI dividido por el ruido (si se conoce). Como
      regla práctica, un CNR > 3-5 es detectable por un observador.
    """
    diff = fractured - healthy
    e_diff = float(np.sum(diff[roi] ** 2))
    e_ref = float(np.sum(healthy[roi] ** 2)) + 1e-30
    out = {"change_ratio": e_diff / e_ref, "mean_abs_change": float(np.mean(np.abs(diff[roi])))}
    if noise_sigma:
        out["cnr"] = out["mean_abs_change"] / noise_sigma
    return out
