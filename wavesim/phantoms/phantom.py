"""Fantoma digital: mapa de tejidos en una malla 2D más la verdad de terreno."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core.grid import Grid2D
from ..materials.tissues import tissue_keys

#: Índice global de cada tejido en los mapas de etiquetas.
TISSUE_INDEX: dict[str, int] = {k: i for i, k in enumerate(tissue_keys())}


@dataclass
class Phantom:
    """Corte 2D de un cuerpo.

    - ``labels[j, i]``: índice del tejido de la celda (ver ``TISSUE_INDEX``).
    - ``masks``: máscaras de verdad de terreno ("fracture", "hematoma", "edema").
    - ``heterogeneity``: ruido gaussiano blanco de varianza 1 por celda. Cada
      modalidad lo escala para simular la microestructura del tejido (por
      ejemplo, el *speckle* de la ecografía).
    - ``meta``: parámetros con los que se generó (tipo de fractura, posición...).
    """

    grid: Grid2D
    labels: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    heterogeneity: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def fracture_mask(self) -> np.ndarray:
        return self.masks.get("fracture", np.zeros(self.grid.shape, dtype=bool))

    @property
    def has_fracture(self) -> bool:
        return bool(self.fracture_mask.any())

    def mask(self, key: str) -> np.ndarray:
        return self.labels == TISSUE_INDEX[key]

    def present_tissues(self) -> list[str]:
        idx = np.unique(self.labels)
        keys = tissue_keys()
        return [keys[i] for i in idx]

    def property_map(self, getter: Callable[[str], float], dtype=np.float64) -> np.ndarray:
        """Mapa de una propiedad escalar: ``getter(tissue_key) -> valor``."""
        keys = tissue_keys()
        lut = np.zeros(len(keys), dtype=dtype)
        for k in self.present_tissues():
            lut[TISSUE_INDEX[k]] = getter(k)
        return lut[self.labels]

    def tissue_fractions(self) -> dict[str, float]:
        n = self.labels.size
        return {k: float(np.count_nonzero(self.mask(k))) / n for k in self.present_tissues()}

    def replace_tissue(self, old: str, new: str) -> "Phantom":
        """Copia con un tejido sustituido (p. ej. el agua de fondo por aire para rayos X)."""
        labels = self.labels.copy()
        labels[labels == TISSUE_INDEX[old]] = TISSUE_INDEX[new]
        return Phantom(self.grid, labels, dict(self.masks), self.heterogeneity, dict(self.meta))

    def resample(self, dx: float) -> "Phantom":
        """Remuestrea por vecino más cercano (útil para modalidades de menor resolución).

        Para obtener la misma anatomía con más detalle es mejor regenerar el
        fantoma con la misma semilla a otra ``dx``.
        """
        g = self.grid
        new = Grid2D.from_size(g.width, g.depth, dx, g.x0, g.z0)
        jj = np.clip(np.round((new.z - g.z0) / g.dx).astype(int), 0, g.nz - 1)
        ii = np.clip(np.round((new.x - g.x0) / g.dx).astype(int), 0, g.nx - 1)
        sel = np.ix_(jj, ii)
        het = self.heterogeneity[sel] if self.heterogeneity is not None else None
        return Phantom(new, self.labels[sel], {k: m[sel] for k, m in self.masks.items()}, het, dict(self.meta))
