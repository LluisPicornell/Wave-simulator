"""Geometría de haz en abanico (fuente puntual), FBP de abanico y reconstrucción iterativa.

Convención (la misma que ``projection``): para el ángulo de vista ``theta`` el rayo central
va en la dirección ``d = (sin θ, cos θ)`` en ``(x, z)`` y el eje del detector es
``n = (cos θ, -sin θ)``. La fuente puntual está en ``S = c - sid·d`` (``c`` = isocentro, el centro del
dominio) y el detector a ``sdd`` de la fuente. Con ``curved=True`` el detector es un arco centrado
en la fuente (equiangular, como un TAC clínico) y con ``False`` es plano.

Reconstrucción analítica: FBP de abanico con ponderación de coseno y filtro de rampa (Kak & Slaney,
cap. 3), sin *rebinning*. Elegida porque no remuestrea los datos (el rebinning interpola los ángulos
y las posiciones, lo que suaviza y obliga a una rejilla extra) y porque el mismo núcleo sirve para el
detector curvo y el plano. Exige una vuelta completa (360°) con ángulos uniformes.

Reconstrucción iterativa: SART / OS-SART con no negatividad y, opcionalmente, regularización por
variación total suavizada. El proyector es el muestreo bilineal a lo largo del rayo y su adjunto exacto
(reparto bilineal), así que las iteraciones son coherentes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import Grid2D


@dataclass(frozen=True)
class FanGeometry:
    """Fuente puntual y detector lineal de ``n_det`` elementos.

    ``det_spacing`` es la separación entre elementos medida sobre el detector (longitud de arco a
    distancia ``sdd`` si es curvo; posición sobre el plano si es plano), en metros.
    """

    sid: float
    sdd: float
    n_det: int
    det_spacing: float
    curved: bool = True

    def __post_init__(self):
        if not 0 < self.sid < self.sdd:
            raise ValueError("Se exige 0 < sid < sdd")

    @classmethod
    def covering(cls, grid: Grid2D, sid: float, sdd: float, curved: bool = True, oversample: float = 1.0):
        """Geometría cuyo abanico cubre el círculo que circunscribe la malla; el paso es el de la malla
        proyectado al detector (``oversample`` > 1 da más elementos)."""
        radius = 0.5 * np.hypot(grid.width, grid.depth)
        if radius >= sid:
            raise ValueError("La fuente debe estar fuera del círculo que circunscribe al dominio")
        half_angle = np.arcsin(radius / sid)
        spacing = grid.dx * sdd / sid / oversample
        half_width = sdd * half_angle if curved else sdd * np.tan(half_angle)
        n = int(np.ceil(2 * half_width / spacing)) | 1
        return cls(sid, sdd, n, spacing, curved)

    @property
    def u(self) -> np.ndarray:
        """Posición de cada elemento sobre el detector (m), centrada."""
        return (np.arange(self.n_det) - (self.n_det - 1) / 2) * self.det_spacing

    @property
    def gamma(self) -> np.ndarray:
        """Ángulo de cada elemento respecto al rayo central (rad)."""
        return self.u / self.sdd if self.curved else np.arctan(self.u / self.sdd)

    def view_rays(self, grid: Grid2D, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Posición de la fuente ``(2,)`` y direcciones unitarias ``(n_det, 2)`` de los rayos de una vista."""
        c = _center(grid)
        th = np.radians(theta_deg)
        d = np.array([np.sin(th), np.cos(th)])
        n = np.array([np.cos(th), -np.sin(th)])
        g = self.gamma
        dirs = np.cos(g)[:, None] * d + np.sin(g)[:, None] * n
        return c - self.sid * d, dirs

    def detector_points(self, grid: Grid2D, theta_deg: float) -> np.ndarray:
        """Posición ``(n_det, 2)`` de cada elemento del detector (plano de la sección)."""
        src, dirs = self.view_rays(grid, theta_deg)
        r = np.full(self.n_det, self.sdd) if self.curved else self.sdd / np.cos(self.gamma)
        return src + r[:, None] * dirs


def _center(grid: Grid2D) -> np.ndarray:
    return np.array([grid.x0 + (grid.nx - 1) * grid.dx / 2, grid.z0 + (grid.nz - 1) * grid.dx / 2])


def _radius(grid: Grid2D) -> float:
    return 0.5 * np.hypot(grid.width, grid.depth)


def _samples(grid: Grid2D, src: np.ndarray, dirs: np.ndarray, step: float):
    """Coordenadas fraccionarias de píxel ``(jf, if_)`` de los puntos de muestreo de cada rayo."""
    c, rad = _center(grid), _radius(grid)
    t0 = (c - src) @ dirs.T - rad  # (n_det,)
    ns = int(np.ceil(2 * rad / step)) + 1
    t = t0[:, None] + step * np.arange(ns)[None, :]
    x = src[0] + t * dirs[:, 0:1]
    z = src[1] + t * dirs[:, 1:2]
    return (z - grid.z0) / grid.dx, (x - grid.x0) / grid.dx


def _bilinear(img: np.ndarray, jf: np.ndarray, if_: np.ndarray) -> np.ndarray:
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


def _bilinear_adjoint(vals: np.ndarray, jf: np.ndarray, if_: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Adjunto exacto de ``_bilinear``: reparte ``vals`` en los cuatro píxeles vecinos."""
    nz, nx = shape
    j0 = np.floor(jf).astype(int)
    i0 = np.floor(if_).astype(int)
    fj, fi = jf - j0, if_ - i0
    out = np.zeros(nz * nx)
    for dj, wj in ((0, 1 - fj), (1, fj)):
        for di, wi in ((0, 1 - fi), (1, fi)):
            jj, ii = j0 + dj, i0 + di
            ok = (jj >= 0) & (jj < nz) & (ii >= 0) & (ii < nx)
            idx = (jj * nx + ii)[ok]
            out += np.bincount(idx, weights=(vals * wj * wi)[ok], minlength=nz * nx)
    return out.reshape(shape)


def fan_project(maps: np.ndarray, grid: Grid2D, geom: FanGeometry, angles_deg, step: float | None = None) -> np.ndarray:
    """Integrales de línea de abanico de uno o varios mapas: ``(n_mapas, n_ángulos, n_det)``."""
    maps = maps if maps.ndim == 3 else maps[None]
    step = step or grid.dx / 1.5
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    out = np.zeros((maps.shape[0], angles.size, geom.n_det))
    for a, th in enumerate(angles):
        src, dirs = geom.view_rays(grid, th)
        jf, if_ = _samples(grid, src, dirs, step)
        for m in range(maps.shape[0]):
            out[m, a] = _bilinear(maps[m], jf, if_).sum(axis=1) * step
    return out


def fan_back(sino: np.ndarray, grid: Grid2D, geom: FanGeometry, angles_deg, step: float | None = None) -> np.ndarray:
    """Adjunto de ``fan_project`` para un solo mapa: retroproyecta ``sino (n_ángulos, n_det)``."""
    step = step or grid.dx / 1.5
    img = np.zeros(grid.shape)
    for a, th in enumerate(np.atleast_1d(np.asarray(angles_deg, float))):
        src, dirs = geom.view_rays(grid, th)
        jf, if_ = _samples(grid, src, dirs, step)
        img += _bilinear_adjoint(np.repeat(sino[a][:, None] * step, jf.shape[1], axis=1), jf, if_, grid.shape)
    return img


def fan_exact_sinogram(ellipses, grid: Grid2D, geom: FanGeometry, angles_deg) -> np.ndarray:
    """Sinograma de abanico exacto de un fantoma de elipses (``xray_phantoms``)."""
    from .xray_phantoms import line_integrals

    angles = np.atleast_1d(np.asarray(angles_deg, float))
    out = np.zeros((angles.size, geom.n_det))
    for a, th in enumerate(angles):
        src, dirs = geom.view_rays(grid, th)
        out[a] = line_integrals(ellipses, np.broadcast_to(src, dirs.shape), dirs)
    return out


# --------------------------------------------------------------------------------------
# FBP de abanico
# --------------------------------------------------------------------------------------
def _ramp_kernel(n: int, spacing: float) -> np.ndarray:
    """Núcleo espacial discreto de la rampa para los retardos ``-(n-1)..(n-1)`` (Kak & Slaney, cap. 3)."""
    k = np.arange(-(n - 1), n)
    h = np.zeros(k.size)
    h[k == 0] = 1.0 / (4.0 * spacing**2)
    odd = (k % 2) != 0
    h[odd] = -1.0 / (np.pi * k[odd] * spacing) ** 2
    return h


def _filter_rows(g: np.ndarray, kernel_fn, spacing: float, window: str) -> np.ndarray:
    """Convolución de cada fila de ``g`` con el núcleo ``kernel_fn(k·spacing)``, con ventana en frecuencia."""
    n = g.shape[-1]
    k = np.arange(-(n - 1), n)
    h = _ramp_kernel(n, spacing) * kernel_fn(k * spacing)
    pad = int(2 ** np.ceil(np.log2(3 * n)))
    hp = np.zeros(pad)
    hp[:n] = h[n - 1:]  # retardos 0..n-1
    hp[-(n - 1):] = h[:n - 1]  # retardos -(n-1)..-1
    filt = np.real(np.fft.fft(hp)) * spacing
    freqs = np.fft.fftfreq(pad, d=spacing)
    if window == "shepp-logan":
        filt = filt * np.sinc(freqs / (2 * np.abs(freqs).max()))
    elif window == "hann":
        filt = filt * (0.5 + 0.5 * np.cos(np.pi * freqs / np.abs(freqs).max()))
    elif window not in ("ram-lak", "none"):
        raise ValueError(f"ventana desconocida '{window}'")
    return np.real(np.fft.ifft(np.fft.fft(g, n=pad, axis=-1) * filt, axis=-1))[..., :n]


def fan_fbp(sinogram: np.ndarray, angles_deg, geom: FanGeometry, grid: Grid2D,
            window: str = "shepp-logan") -> np.ndarray:
    """FBP de abanico (360°, ángulos uniformes) de un sinograma ``(n_ángulos, n_det)``.

    Detector curvo equiangular: se pondera por ``sid·cos γ``, se filtra con la rampa multiplicada por
    ``(γ / sin γ)²`` y se retroproyecta con peso ``1 / L²`` (``L`` = distancia fuente-píxel).
    Detector plano: ponderación ``sid / sqrt(sid² + s²)`` en el plano que pasa por el isocentro,
    rampa en ``s`` y retroproyección con peso ``sid² / U²`` (``U`` = distancia a lo largo del eje).
    """
    angles = np.radians(np.asarray(angles_deg, float))
    sid = geom.sid
    X, Z = grid.meshgrid()
    c = _center(grid)
    rx, rz = X - c[0], Z - c[1]
    img = np.zeros(grid.shape)
    if geom.curved:
        dg = geom.det_spacing / geom.sdd
        g = sinogram * (sid * np.cos(geom.gamma))[None, :]

        def modifier(delta):  # (δ / sin δ)²
            out = np.ones_like(delta)
            nz = np.abs(delta) > 1e-12
            out[nz] = (delta[nz] / np.sin(delta[nz])) ** 2
            return out

        q = _filter_rows(g, modifier, dg, window) * 1.0
        pos = geom.gamma
    else:
        ds = geom.det_spacing * sid / geom.sdd  # paso en el plano del isocentro
        s_iso = geom.u * sid / geom.sdd
        g = sinogram * (sid / np.sqrt(sid**2 + s_iso**2))[None, :]
        q = _filter_rows(g, lambda x: np.ones_like(x), ds, window)
        pos = s_iso
    for a, th in enumerate(angles):
        d = np.array([np.sin(th), np.cos(th)])
        n = np.array([np.cos(th), -np.sin(th)])
        lat = rx * n[0] + rz * n[1]
        along = sid + rx * d[0] + rz * d[1]
        if geom.curved:
            gam = np.arctan2(lat, along)
            w = 1.0 / (lat**2 + along**2)
            img += np.interp(gam, pos, q[a], left=0.0, right=0.0) * w
        else:
            s = sid * lat / along
            img += np.interp(s, pos, q[a], left=0.0, right=0.0) * (sid / along) ** 2
    return img * (np.pi / len(angles))  # 1/2 de las 2π vueltas × Δβ = 2π/N


# --------------------------------------------------------------------------------------
# Iterativo: SART / OS-SART con TV opcional
# --------------------------------------------------------------------------------------
def tv_gradient(img: np.ndarray, eps: float) -> np.ndarray:
    """Gradiente de la variación total suavizada ``sum sqrt(|∇u|² + eps²)`` (diferencias hacia delante)."""
    gz = np.diff(img, axis=0, append=img[-1:, :])
    gx = np.diff(img, axis=1, append=img[:, -1:])
    norm = np.sqrt(gz**2 + gx**2 + eps**2)
    pz, px = gz / norm, gx / norm
    div = (np.diff(pz, axis=0, prepend=np.zeros((1, img.shape[1])))
           + np.diff(px, axis=1, prepend=np.zeros((img.shape[0], 1))))
    return -div


def os_sart(sinogram: np.ndarray, angles_deg, geom: FanGeometry, grid: Grid2D, n_iter: int = 10,
            subset_size: int = 1, relax: float = 1.0, x0: np.ndarray | None = None, nonneg: bool = True,
            tv_weight: float = 0.0, tv_steps: int = 5, step: float | None = None) -> np.ndarray:
    """Reconstrucción iterativa SART (``subset_size=1``) u OS-SART con subconjuntos de vistas.

    En cada subconjunto ``S``: ``x += λ · Σ_v Aᵀ_v[(p_v − A_v x) / r_v] / Σ_v c_v`` con ``r = A·1``
    (longitud del rayo en la malla) y ``c = Aᵀ·1``. Los subconjuntos se toman entrelazados en ángulo
    (vistas ``k, k+m, k+2m...``) para que cada uno cubra todo el círculo. ``tv_weight`` > 0 aplica
    tras cada iteración ``tv_steps`` pasos de descenso de la variación total suavizada, en fracción
    del máximo de la imagen por paso.
    """
    angles = np.asarray(angles_deg, float)
    nv = angles.size
    step = step or grid.dx / 1.5
    x = np.zeros(grid.shape) if x0 is None else x0.copy()
    ones = np.ones(grid.shape)
    row = fan_project(ones, grid, geom, angles, step)[0]  # (nv, n_det)
    n_sub = int(np.ceil(nv / subset_size))
    subsets = [np.arange(s, nv, n_sub) for s in range(n_sub)]
    col = [fan_back(np.ones((idx.size, geom.n_det)), grid, geom, angles[idx], step) for idx in subsets]
    for _ in range(n_iter):
        for idx, c_s in zip(subsets, col):
            resid = sinogram[idx] - fan_project(x, grid, geom, angles[idx], step)[0]
            ratio = np.where(row[idx] > 1e-9, resid / np.maximum(row[idx], 1e-9), 0.0)
            upd = fan_back(ratio, grid, geom, angles[idx], step)
            x += relax * upd / np.maximum(c_s, 1e-9)
            if nonneg:
                np.maximum(x, 0.0, out=x)
        if tv_weight > 0:
            scale = max(float(x.max()), 1e-12)
            for _ in range(tv_steps):
                x -= tv_weight * scale * tv_gradient(x / scale, 0.02)
                if nonneg:
                    np.maximum(x, 0.0, out=x)
    return x
