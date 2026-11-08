"""Proyección de rayos paralelos (transformada de Radon) y retroproyección filtrada.

Sirve para cualquier radiación que viaja en línea recta y se atenúa según
Beer-Lambert: rayos X, gamma y fotones balísticos (THz, IR, UV).

Geometría: un ángulo ``theta = 0`` son rayos que bajan en +z (proyección
antero-posterior si la sonda/fuente está arriba); el detector se extiende a lo
largo de x. Ángulos en grados, positivos en sentido horario.
"""

from __future__ import annotations

import numpy as np

from ..core.grid import Grid2D


def _bilinear(img: np.ndarray, jf: np.ndarray, if_: np.ndarray) -> np.ndarray:
    """Interpolación bilineal con valor 0 fuera de la imagen."""
    nz, nx = img.shape
    j0 = np.floor(jf).astype(int)
    i0 = np.floor(if_).astype(int)
    fj, fi = jf - j0, if_ - i0
    out = np.zeros(jf.shape)
    for dj, wj in ((0, 1 - fj), (1, fj)):
        for di, wi in ((0, 1 - fi), (1, fi)):
            jj, ii = j0 + dj, i0 + di
            ok = (jj >= 0) & (jj < nz) & (ii >= 0) & (ii < nx)
            out += np.where(ok, img[np.clip(jj, 0, nz - 1), np.clip(ii, 0, nx - 1)], 0.0) * wj * wi
    return out


def detector_axis(grid: Grid2D, spacing: float | None = None) -> np.ndarray:
    """Posiciones de los detectores (centradas en el centro del dominio)."""
    spacing = spacing or grid.dx
    half = 0.5 * np.hypot(grid.width, grid.depth)
    n = int(np.ceil(2 * half / spacing)) | 1
    return (np.arange(n) - n // 2) * spacing


def project(maps: np.ndarray, grid: Grid2D, angles_deg, det: np.ndarray | None = None,
            step: float | None = None) -> np.ndarray:
    """Integrales de línea de uno o varios mapas.

    ``maps``: (nz, nx) o (n_mapas, nz, nx). Devuelve (n_mapas, n_ángulos, n_det)
    con unidades de mapa × metros.
    """
    maps = maps if maps.ndim == 3 else maps[None]
    det = detector_axis(grid) if det is None else det
    step = step or grid.dx / 1.5
    half = 0.5 * np.hypot(grid.width, grid.depth)
    u = np.arange(-half, half + step / 2, step)
    xc = grid.x0 + (grid.nx - 1) * grid.dx / 2
    zc = grid.z0 + (grid.nz - 1) * grid.dx / 2
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    out = np.zeros((maps.shape[0], angles.size, det.size))
    S, U = np.meshgrid(det, u, indexing="ij")
    for a, th in enumerate(np.radians(angles)):
        dxr, dzr = np.sin(th), np.cos(th)  # dirección del rayo
        nx_, nz_ = np.cos(th), -np.sin(th)  # dirección del detector
        X = xc + S * nx_ + U * dxr
        Z = zc + S * nz_ + U * dzr
        jf = (Z - grid.z0) / grid.dx
        if_ = (X - grid.x0) / grid.dx
        for m in range(maps.shape[0]):
            out[m, a] = _bilinear(maps[m], jf, if_).sum(axis=1) * step
    return out


def ramp_filter(proj: np.ndarray, spacing: float, window: str = "shepp-logan") -> np.ndarray:
    """Filtro rampa (con ventana) aplicado a lo largo del eje del detector."""
    n = proj.shape[-1]
    pad = int(2 ** np.ceil(np.log2(2 * n)))
    freqs = np.fft.fftfreq(pad, d=spacing)
    # Rampa construida desde el núcleo espacial discreto (Kak & Slaney, cap. 3):
    # muestrear |f| directamente deja una componente continua errónea.
    k = np.concatenate([np.arange(0, pad // 2), np.arange(-pad // 2, 0)])
    h = np.zeros(pad)
    h[0] = 1.0 / (4.0 * spacing**2)
    odd = k % 2 == 1
    h[odd] = -1.0 / (np.pi * k[odd] * spacing) ** 2
    filt = np.real(np.fft.fft(h)) * spacing
    if window == "shepp-logan":
        fn = freqs / (2 * freqs.max())
        filt = filt * np.sinc(fn)
    elif window == "hann":
        filt = filt * (0.5 + 0.5 * np.cos(np.pi * freqs / freqs.max()))
    spec = np.fft.fft(proj, n=pad, axis=-1) * filt
    return np.real(np.fft.ifft(spec, axis=-1))[..., :n]


def fbp(sinogram: np.ndarray, angles_deg, det: np.ndarray, grid: Grid2D, window: str = "shepp-logan") -> np.ndarray:
    """Retroproyección filtrada de un sinograma (n_ángulos, n_det) sobre ``grid``.

    Los ángulos deben cubrir 180° de forma uniforme.
    """
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    spacing = det[1] - det[0]
    filtered = ramp_filter(sinogram, spacing, window)
    X, Z = grid.meshgrid()
    xc = grid.x0 + (grid.nx - 1) * grid.dx / 2
    zc = grid.z0 + (grid.nz - 1) * grid.dx / 2
    img = np.zeros(grid.shape)
    for a, th in enumerate(angles):
        s = (X - xc) * np.cos(th) - (Z - zc) * np.sin(th)
        img += np.interp(s, det, filtered[a], left=0.0, right=0.0)
    return img * np.pi / len(angles)
