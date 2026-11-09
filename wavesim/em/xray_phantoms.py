"""Fantomas analíticos de elipses (Shepp-Logan) con integrales de línea exactas.

Sirven para validar proyectores y reconstructores sin error de discretización del
objeto: la integral de una elipse a lo largo de un rayo es la longitud de la cuerda
por su valor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import Grid2D


@dataclass(frozen=True)
class Ellipse:
    """Elipse de valor ``value`` (se suma a las demás) centrada en ``(cx, cz)`` (m)."""

    cx: float
    cz: float
    a: float  # semieje a lo largo de x' (m)
    b: float  # semieje a lo largo de z' (m)
    phi_deg: float
    value: float


# Shepp-Logan modificado (Toft 1996): (x, y, a, b, phi, valor) en un cuadrado unidad
_SHEPP_LOGAN = (
    (0.0, 0.0, 0.69, 0.92, 0.0, 1.0),
    (0.0, -0.0184, 0.6624, 0.874, 0.0, -0.8),
    (0.22, 0.0, 0.11, 0.31, -18.0, -0.2),
    (-0.22, 0.0, 0.16, 0.41, 18.0, -0.2),
    (0.0, 0.35, 0.21, 0.25, 0.0, 0.1),
    (0.0, 0.1, 0.046, 0.046, 0.0, 0.1),
    (0.0, -0.1, 0.046, 0.046, 0.0, 0.1),
    (-0.08, -0.605, 0.046, 0.023, 0.0, 0.1),
    (0.0, -0.606, 0.023, 0.023, 0.0, 0.1),
    (0.06, -0.605, 0.023, 0.046, 0.0, 0.1),
)


def shepp_logan(radius: float = 0.1, mu0: float = 20.0, center: tuple[float, float] = (0.0, 0.0)) -> list[Ellipse]:
    """Fantoma de Shepp-Logan modificado; ``radius`` es la escala (m) del cuadrado unidad.

    ``mu0``: valor de la elipse exterior (1/m); las demás son proporcionales (contraste modificado).
    """
    return [Ellipse(center[0] + radius * x, center[1] + radius * y, radius * a, radius * b, phi, mu0 * v)
            for x, y, a, b, phi, v in _SHEPP_LOGAN]


def disk(radius: float, value: float, center: tuple[float, float] = (0.0, 0.0)) -> list[Ellipse]:
    return [Ellipse(center[0], center[1], radius, radius, 0.0, value)]


def rasterize(ellipses: list[Ellipse], grid: Grid2D, supersample: int = 4) -> np.ndarray:
    """Imagen (nz, nx) con el promedio del campo en cada píxel (submuestreo regular)."""
    off = (np.arange(supersample) + 0.5) / supersample - 0.5
    img = np.zeros(grid.shape)
    x, z = grid.x, grid.z
    for ox in off:
        for oz in off:
            X, Z = np.meshgrid(x + ox * grid.dx, z + oz * grid.dx)
            for e in ellipses:
                c, s = np.cos(np.radians(e.phi_deg)), np.sin(np.radians(e.phi_deg))
                u = (X - e.cx) * c + (Z - e.cz) * s
                v = -(X - e.cx) * s + (Z - e.cz) * c
                img += np.where((u / e.a) ** 2 + (v / e.b) ** 2 <= 1.0, e.value, 0.0)
    return img / supersample**2


def line_integrals(ellipses: list[Ellipse], origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Integral exacta del campo a lo largo de rayos infinitos ``origen + t·dir`` (``dir`` unitario).

    ``origins`` y ``dirs`` tienen forma ``(..., 2)`` en ``(x, z)``; devuelve forma ``(...)``.
    """
    o = np.asarray(origins, float)
    d = np.asarray(dirs, float)
    out = np.zeros(np.broadcast_shapes(o.shape, d.shape)[:-1])
    for e in ellipses:
        c, s = np.cos(np.radians(e.phi_deg)), np.sin(np.radians(e.phi_deg))
        px, pz = o[..., 0] - e.cx, o[..., 1] - e.cz
        qx, qz = px * c + pz * s, -px * s + pz * c
        dx, dz = d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c
        A = (dx / e.a) ** 2 + (dz / e.b) ** 2
        B = 2.0 * (qx * dx / e.a**2 + qz * dz / e.b**2)
        C = (qx / e.a) ** 2 + (qz / e.b) ** 2 - 1.0
        disc = B * B - 4.0 * A * C
        out = out + e.value * np.where(disc > 0, np.sqrt(np.maximum(disc, 0.0)) / A, 0.0)
    return out


def parallel_sinogram(ellipses: list[Ellipse], grid: Grid2D, angles_deg, det: np.ndarray) -> np.ndarray:
    """Sinograma exacto de haz paralelo con la misma convención que ``projection.project``."""
    xc = grid.x0 + (grid.nx - 1) * grid.dx / 2
    zc = grid.z0 + (grid.nz - 1) * grid.dx / 2
    th = np.radians(np.atleast_1d(np.asarray(angles_deg, float)))
    d = np.stack([np.sin(th), np.cos(th)], axis=-1)[:, None, :]
    n = np.stack([np.cos(th), -np.sin(th)], axis=-1)[:, None, :]
    origins = np.array([xc, zc]) + det[None, :, None] * n
    return line_integrals(ellipses, origins, np.broadcast_to(d, origins.shape))
