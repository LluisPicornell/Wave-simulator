"""Malla cartesiana 3D (solver elástico 3D y fantomas 3D)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid2D


@dataclass(frozen=True)
class Grid3D:
    """Malla regular 3D de celdas cúbicas de lado ``dx`` (metros).

    Los arrays tienen forma ``(nz, ny, nx)``: z es la profundidad (como en
    ``Grid2D``), x la dirección lateral del corte 2D (en los fantomas, el eje
    del hueso) e y la dirección perpendicular a ese corte. El centro de la celda
    ``[k, j, i]`` está en ``(x0 + i*dx, y0 + j*dx, z0 + k*dx)``. El plano
    ``y = cte`` es un ``Grid2D`` con los mismos índices ``[k, i]``.
    """

    nx: int
    ny: int
    nz: int
    dx: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    @classmethod
    def from_size(cls, length: float, width: float, depth: float, dx: float, x0: float = 0.0, y0: float = 0.0,
                  z0: float = 0.0) -> "Grid3D":
        """Malla de ``length`` (x) × ``width`` (y) × ``depth`` (z)."""
        n = [max(int(round(s / dx)), 1) for s in (length, width, depth)]
        return cls(n[0], n[1], n[2], dx, x0, y0, z0)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def size(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def x(self) -> np.ndarray:
        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def y(self) -> np.ndarray:
        return self.y0 + np.arange(self.ny) * self.dx

    @property
    def z(self) -> np.ndarray:
        return self.z0 + np.arange(self.nz) * self.dx

    def index_of(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Índice ``(k, j, i)`` de la celda más cercana al punto ``(x, y, z)``."""
        i = int(np.clip(round((x - self.x0) / self.dx), 0, self.nx - 1))
        j = int(np.clip(round((y - self.y0) / self.dx), 0, self.ny - 1))
        k = int(np.clip(round((z - self.z0) / self.dx), 0, self.nz - 1))
        return k, j, i

    def plane_xz(self) -> Grid2D:
        """Malla 2D del corte ``y = cte`` (mismos índices ``[k, i]``)."""
        return Grid2D(self.nx, self.nz, self.dx, self.x0, self.z0)

    def cells_in_box(self, x: tuple[float, float], y: tuple[float, float],
                     z: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Índices ``(k, j, i)`` de las celdas cuyo centro cae en la caja (límites incluidos).

        Una dimensión de anchura nula (``(a, a)``) toma la fila de celdas más cercana.
        """
        sel = []
        for (lo, hi), coord, d0 in ((z, self.z, self.z0), (y, self.y, self.y0), (x, self.x, self.x0)):
            idx = np.nonzero((coord >= lo - 1e-9 * self.dx) & (coord <= hi + 1e-9 * self.dx))[0]
            if idx.size == 0:
                idx = np.array([int(np.clip(round((0.5 * (lo + hi) - d0) / self.dx), 0, coord.size - 1))])
            sel.append(idx)
        k, j, i = np.meshgrid(*sel, indexing="ij")
        return k.ravel(), j.ravel(), i.ravel()
