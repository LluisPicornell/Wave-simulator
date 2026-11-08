"""Malla cartesiana 2D compartida por todos los solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """Malla regular 2D de celdas cuadradas de lado ``dx`` (metros).

    Convención usada en todo el paquete: los arrays tienen forma ``(nz, nx)``.
    El eje ``z`` es la profundidad (filas, crece hacia abajo, como en una
    ecografía) y ``x`` la dirección lateral (columnas). El centro de la celda
    ``[j, i]`` está en ``(x0 + i*dx, z0 + j*dx)``.
    """

    nx: int
    nz: int
    dx: float
    x0: float = 0.0
    z0: float = 0.0

    @classmethod
    def from_size(cls, width: float, depth: float, dx: float, x0: float = 0.0, z0: float = 0.0) -> "Grid2D":
        return cls(max(int(round(width / dx)), 1), max(int(round(depth / dx)), 1), dx, x0, z0)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nz, self.nx)

    @property
    def x(self) -> np.ndarray:
        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def z(self) -> np.ndarray:
        return self.z0 + np.arange(self.nz) * self.dx

    @property
    def width(self) -> float:
        return self.nx * self.dx

    @property
    def depth(self) -> float:
        return self.nz * self.dx

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.z)

    def index_of(self, x: float, z: float) -> tuple[int, int]:
        """Índice ``(j, i)`` de la celda más cercana al punto ``(x, z)``."""
        i = int(np.clip(round((x - self.x0) / self.dx), 0, self.nx - 1))
        j = int(np.clip(round((z - self.z0) / self.dx), 0, self.nz - 1))
        return j, i

    def extent_mm(self) -> tuple[float, float, float, float]:
        """Extensión para ``matplotlib.imshow`` en milímetros (izq, der, abajo, arriba)."""
        h = self.dx / 2
        return (
            (self.x0 - h) * 1e3,
            (self.x0 + (self.nx - 1) * self.dx + h) * 1e3,
            (self.z0 + (self.nz - 1) * self.dx + h) * 1e3,
            (self.z0 - h) * 1e3,
        )
