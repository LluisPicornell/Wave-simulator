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


def resample_mask_apparent_depth(phantom: Phantom, mask: np.ndarray, x: np.ndarray, z: np.ndarray,
                                 z_probe: float, c0: float = 1540.0) -> np.ndarray:
    """Lleva una máscara a la malla de una ecografía, en profundidad *aparente*.

    El ecógrafo convierte tiempo en profundidad con una velocidad fija ``c0``:
    un punto a profundidad real ``z`` aparece en ``z_probe + c0·∫ dz'/c(z')``
    (trayecto vertical desde la sonda). Sin esta corrección la máscara queda
    desplazada respecto a los ecos (p. ej. ~0.2 mm a 8 mm de profundidad en agua).
    """
    from .materials.acoustic import ACOUSTIC

    g = phantom.grid
    c = phantom.property_map(lambda k: ACOUSTIC[k].c_p)
    j0 = int(np.clip(round((z_probe - g.z0) / g.dx), 0, g.nz - 1))
    travel = np.cumsum(g.dx / c, axis=0)
    z_app = z_probe + c0 * (travel - travel[j0][None, :])
    rows = np.arange(g.nz)
    ii = np.clip(np.round((x - g.x0) / g.dx).astype(int), 0, g.nx - 1)
    out = np.zeros((z.size, x.size), dtype=mask.dtype)
    for col, i in enumerate(ii):
        # z_app crece monótonamente con la profundidad real: se invierte por interpolación
        j = np.clip(np.round(np.interp(z, z_app[:, i], rows)).astype(int), 0, g.nz - 1)
        out[:, col] = mask[j, i]
    return out


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


# --------------------------------------------------------------------------
# Señales: cambio de una forma de onda respecto a una referencia
# --------------------------------------------------------------------------
def residual_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Energía de la diferencia relativa a la de la referencia (dB): 10·log10(Σ(test−ref)² / Σref²).

    Mide cualquier cambio de la forma de onda (amplitud, retardo o forma). Un
    retardo de una fracción ε de periodo da unos 20·log10(2π·ε) dB; una
    variación de amplitud ε, 20·log10(ε) dB.
    """
    diff = float(np.sum((np.asarray(test) - np.asarray(reference)) ** 2))
    return 10.0 * np.log10(diff / (float(np.sum(np.asarray(reference) ** 2)) + 1e-300) + 1e-30)


def energy_ratio_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Cambio de energía (dB) de ``test`` respecto a ``reference``: 10·log10(Σtest² / Σref²)."""
    return 10.0 * np.log10((float(np.sum(np.asarray(test) ** 2)) + 1e-300) /
                           (float(np.sum(np.asarray(reference) ** 2)) + 1e-300))


def xcorr_delay(reference: np.ndarray, test: np.ndarray, dt: float) -> float:
    """Retardo (s) de ``test`` respecto a ``reference`` (positivo = llega más tarde).

    Máximo de la correlación cruzada con refinamiento parabólico (resolución
    muy inferior a ``dt``). Sólo tiene sentido si las dos señales se parecen
    y el retardo es menor que medio periodo de la portadora.
    """
    a, b = np.asarray(reference, float), np.asarray(test, float)
    n = a.size
    xc = np.fft.irfft(np.fft.rfft(b, 2 * n) * np.conj(np.fft.rfft(a, 2 * n)), 2 * n)
    xc = np.concatenate([xc[-(n - 1):], xc[:n]])  # retardos de -(n-1) a n-1
    k = int(np.argmax(xc))
    shift = 0.0
    if 0 < k < xc.size - 1:
        y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
        den = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return (k - (n - 1) + shift) * dt
