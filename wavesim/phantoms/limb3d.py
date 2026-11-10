"""Fantomas 3D de extremidad para el solver elástico 3D (``ultrasound.elastic3d``).

``Phantom3D`` es el contenedor (etiquetas de tejido por celda en una malla
``Grid3D``); los generadores de segmentos de pierna llegan en otro paso.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core.grid3d import Grid3D
from ..materials.tissues import tissue_keys
from .phantom import TISSUE_INDEX, Phantom


@dataclass
class Phantom3D:
    """Bloque 3D de tejidos: ``labels[k, j, i]`` (z, y, x) con los índices de ``TISSUE_INDEX``.

    ``masks``: verdad de terreno ("fracture", "hematoma"...), arrays booleanos
    de la misma forma. ``meta``: parámetros de construcción; ``meta["bone_axis"]
    = "x"`` indica que el eje del hueso va a lo largo de x (lo necesita la
    cortical anisótropa del solver 3D).
    """

    grid: Grid3D
    labels: np.ndarray
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.labels.shape != self.grid.shape:
            raise ValueError(f"labels tiene forma {self.labels.shape}, la malla {self.grid.shape}")

    @property
    def fracture_mask(self) -> np.ndarray:
        return self.masks.get("fracture", np.zeros(self.grid.shape, dtype=bool))

    @property
    def has_fracture(self) -> bool:
        return bool(self.fracture_mask.any())

    def mask(self, key: str) -> np.ndarray:
        return self.labels == TISSUE_INDEX[key]

    def present_tissues(self) -> list[str]:
        keys = tissue_keys()
        return [keys[i] for i in np.unique(self.labels)]

    def property_map(self, getter: Callable[[str], float], dtype=np.float32) -> np.ndarray:
        """Mapa de una propiedad escalar ``getter(tejido)`` (tabla de búsqueda por etiqueta)."""
        lut = np.zeros(len(tissue_keys()), dtype=dtype)
        for k in self.present_tissues():
            lut[TISSUE_INDEX[k]] = getter(k)
        return lut[self.labels]

    def tissue_fractions(self) -> dict[str, float]:
        n = self.labels.size
        return {k: float(np.count_nonzero(self.mask(k))) / n for k in self.present_tissues()}

    def slice_y(self, y: float = 0.0) -> Phantom:
        """Corte ``y = cte`` como ``Phantom`` 2D (corte longitudinal si el eje del hueso va en x)."""
        g = self.grid
        j = g.index_of(g.x0, y, g.z0)[1]
        meta = dict(self.meta, y_slice=float(g.y[j]))
        if self.meta.get("bone_axis") == "x":
            meta["view"] = "longitudinal"
        het = np.zeros((g.nz, g.nx), np.float32)
        return Phantom(g.plane_xz(), self.labels[:, j, :].copy(), {k: m[:, j, :].copy() for k, m in self.masks.items()},
                       het, meta)


def homogeneous_phantom(grid: Grid3D, tissue: str = "water") -> Phantom3D:
    """Bloque de un solo tejido (útil con un ``ElasticMedium3D`` propio)."""
    return Phantom3D(grid, np.full(grid.shape, TISSUE_INDEX[tissue], np.int16))
