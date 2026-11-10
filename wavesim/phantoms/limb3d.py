"""Fantomas 3D de extremidad para el solver elástico 3D (``ultrasound.elastic3d``).

Dos generadores:

- ``extrude``: repite un fantoma 2D a lo largo de y. Con una fuente uniforme en
  y, el solver 3D reproduce el 2D sobre el mismo corte; sirve para comparar.
- ``leg_segment``: segmento de pierna con la tibia como **tubo**: cilindro
  hueco de cortical (radio externo y grosor de los presets de ``limb.py``) con
  médula dentro, rodeado de capas concéntricas de músculo, grasa y piel, en
  agua, con una fractura transversal opcional (hueco de anchura ``w`` que corta
  el tubo en todo su perímetro o en un arco) y su máscara 3D. El eje del hueso
  va a lo largo de x, como en el corte longitudinal 2D: el plano y = 0 de
  ``leg_segment`` es un corte longitudinal (dos placas de cortical separadas
  por la médula), y ``Phantom3D.slice_y`` lo devuelve como ``Phantom`` 2D.

Los tejidos son los de ``materials`` (``TISSUE_INDEX``) y sus propiedades las de
``ACOUSTIC``. **Sólo hay capas homogéneas**: el modo realista de los fantomas 2D
(porosidad cortical, red trabecular, rugosidad fractal, fascias) es un mapa
2D por celda cuya extrusión no tendría sentido (un canal de Havers extruido en y
sería una lámina), así que ``extrude`` lo rechaza. Tampoco hay rugosidad de
interfaces ni *speckle* en 3D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core.grid3d import Grid3D
from ..materials.tissues import tissue_keys
from .limb import CROSS_PRESETS, PRESETS
from .phantom import TISSUE_INDEX, Phantom

#: Hueso del tubo -> (preset transversal de ``limb.py``, índice del hueso): de ahí salen el radio
#: externo y el grosor de la cortical. La piel y la grasa salen del preset longitudinal homónimo.
TUBE_BONES: dict[str, tuple[str, int, str]] = {
    "tibia": ("pierna", 0, "tibia"),
    "perone": ("pierna", 1, "tibia"),
    "radio": ("antebrazo", 0, "antebrazo"),
    "cubito": ("antebrazo", 1, "antebrazo"),
    "femur": ("muslo", 0, "muslo"),
}


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


def extrude(phantom: Phantom, width: float | None = None, ny: int | None = None) -> Phantom3D:
    """Repite un fantoma 2D (z, x) a lo largo de y, centrado en y = 0.

    ``ny`` celdas, o ``width`` metros (mínimo 4 celdas, lo que necesita el
    estencil). No admite el modo realista (``phantom.porosity``): su
    microestructura es 2D. Se pierde la heterogeneidad (*speckle*): el solver
    3D no la usa. Un corte longitudinal da ``meta["bone_axis"] = "x"``; uno
    transversal, ``"y"`` (la cortical anisótropa sólo admite el eje en x).
    """
    if phantom.porosity is not None:
        raise ValueError("extrude no admite el modo realista (porosidad, trabéculas): es una microestructura 2D "
                         "que extruida sería una colección de láminas. Usa realism=None.")
    g2 = phantom.grid
    if ny is None:
        ny = max(int(round((width if width is not None else 4 * g2.dx) / g2.dx)), 4)
    grid = Grid3D(g2.nx, ny, g2.nz, g2.dx, g2.x0, -0.5 * (ny - 1) * g2.dx, g2.z0)
    labels = np.ascontiguousarray(np.repeat(phantom.labels[:, None, :], ny, axis=1))
    masks = {k: np.repeat(m[:, None, :], ny, axis=1) for k, m in phantom.masks.items()}
    view = phantom.meta.get("view")
    axis = {"longitudinal": "x", "cross_section": "y"}.get(view)
    meta = {"generator": "extrude", "source_meta": dict(phantom.meta), "bone_axis": axis}
    return Phantom3D(grid, labels, masks, meta)


@dataclass
class LegSegmentSpec:
    """Geometría de ``leg_segment`` (metros). ``None`` = valor del preset.

    - ``bone``: clave de ``TUBE_BONES``; da ``outer_radius`` y ``cortex``
      (``limb.CROSS_PRESETS``) y ``skin`` y ``fat`` (``limb.PRESETS``).
    - ``muscle``: grosor del músculo concéntrico alrededor del hueso. Las capas
      concéntricas son una simplificación: en la cara anteromedial la tibia
      está casi bajo la piel y por detrás tiene varios cm de músculo.
    - ``standoff``: agua entre el borde superior del dominio (z = 0) y la piel.
    """

    bone: str = "tibia"
    outer_radius: float | None = None
    cortex: float | None = None
    muscle: float = 2e-3
    fat: float | None = None
    skin: float | None = None
    standoff: float = 2e-3
    medullary: str = "marrow"
    background: str = "water"

    def resolved(self) -> "LegSegmentSpec":
        cross, idx, longi = TUBE_BONES[self.bone]
        b = CROSS_PRESETS[cross].bones[idx]
        lp = PRESETS[longi]
        out = LegSegmentSpec(**self.__dict__)
        out.outer_radius = b.outer_radius if self.outer_radius is None else self.outer_radius
        out.cortex = b.cortex if self.cortex is None else self.cortex
        out.fat = lp.fat if self.fat is None else self.fat
        out.skin = lp.skin if self.skin is None else self.skin
        if not 0 < out.cortex < out.outer_radius:
            raise ValueError("El grosor cortical debe estar entre 0 y el radio externo")
        return out

    @property
    def leg_radius(self) -> float:
        s = self.resolved()
        return s.outer_radius + s.muscle + s.fat + s.skin


@dataclass
class TransverseFracture3D:
    """Fractura transversal del tubo: hueco de anchura ``gap`` perpendicular al eje (x).

    ``position``: x del centro del hueco. ``arc_deg``: arco del perímetro que
    corta (360 = fractura completa; menos = incompleta, centrada en
    ``arc_center_deg``, 0 = el lado de la sonda, -z). ``fill``: tejido que
    ocupa el hueco (hematoma, como en 2D).
    """

    gap: float = 1e-3
    position: float = 0.0
    arc_deg: float = 360.0
    arc_center_deg: float = 0.0
    fill: str = "hematoma"


def leg_segment(
    spec: LegSegmentSpec | str = "tibia",
    fracture: TransverseFracture3D | None = None,
    length: float = 40e-3,
    dx: float = 0.2e-3,
    width: float | None = None,
    depth: float | None = None,
    margin: float = 1e-3,
    bone_length: float | None = None,
) -> Phantom3D:
    """Segmento de pierna con hueso tubular, eje a lo largo de x, centrado en x = 0 e y = 0.

    El eje del tubo está en ``z_c = standoff + radio de la pierna`` (la piel
    toca ``z = standoff`` en y = 0). Por defecto el dominio abarca la pierna
    entera más ``margin`` de agua a los lados y debajo; con ``width`` o
    ``depth`` menores la pierna queda cortada por el borde (la CPML hace
    entonces de tejido que continúa y absorbe). Los tejidos se rasterizan por
    el centro de cada celda, sin rugosidad.

    ``bone_length``: longitud de la cortical, centrada en x = 0 (``None`` =
    todo el dominio). Más allá, el anillo cortical pasa a ser músculo y la
    médula sigue: todo fluido. Sirve para que el sólido no entre en la CPML a
    lo largo de x: un tubo que la atraviesa crece lentamente en simulaciones
    muy largas sin atenuación (``docs/validacion/elastico3d.md``, del orden de
    ×3 en 3·10⁴ pasos), a costa de reflejar las ondas guiadas en los extremos.

    Máscaras: "fracture" (celdas de cortical que ocupa el hueco), "hematoma"
    (todo el hueco) y "cortical" (el tubo sano, para localizarlo).
    """
    sp = (LegSegmentSpec(bone=spec) if isinstance(spec, str) else spec).resolved()
    r_leg = sp.outer_radius + sp.muscle + sp.fat + sp.skin
    if width is None:
        width = 2 * (r_leg + margin)
    if depth is None:
        depth = sp.standoff + 2 * r_leg + margin
    nx = max(int(round(length / dx)), 4)
    ny = max(int(round(width / dx)), 4)
    nz = max(int(round(depth / dx)), 4)
    grid = Grid3D(nx, ny, nz, dx, -0.5 * (nx - 1) * dx, -0.5 * (ny - 1) * dx, 0.0)
    z_c = sp.standoff + r_leg
    # Geometría en el plano transversal (y, z): igual en todo x salvo la fractura
    yy, zz = np.meshgrid(grid.y, grid.z - z_c)  # (nz, ny)
    r = np.hypot(yy, zz)
    plane = np.full((nz, ny), TISSUE_INDEX[sp.background], np.int16)
    for tissue, radius in (("skin", r_leg), ("fat", r_leg - sp.skin), ("muscle", sp.outer_radius + sp.muscle),
                           ("cortical_bone", sp.outer_radius), (sp.medullary, sp.outer_radius - sp.cortex)):
        plane[r <= radius] = TISSUE_INDEX[tissue]
    labels = np.repeat(plane[:, :, None], nx, axis=2)
    if bone_length is not None:
        beyond = np.abs(grid.x) > 0.5 * bone_length
        ring = (labels == TISSUE_INDEX["cortical_bone"]) & beyond[None, None, :]
        labels[ring] = TISSUE_INDEX["muscle"]
    cortical = labels == TISSUE_INDEX["cortical_bone"]
    masks = {"cortical": cortical.copy(), "fracture": np.zeros(grid.shape, bool),
             "hematoma": np.zeros(grid.shape, bool)}
    if fracture is not None and fracture.gap > 0:
        in_gap = np.abs(grid.x - fracture.position) <= 0.5 * fracture.gap
        theta = np.degrees(np.arctan2(yy, -zz))  # 0 = hacia la sonda (-z)
        dtheta = (theta - fracture.arc_center_deg + 180.0) % 360.0 - 180.0
        in_arc = np.abs(dtheta) <= 0.5 * fracture.arc_deg + 1e-9
        tube = (r <= sp.outer_radius) & (r > sp.outer_radius - sp.cortex) & in_arc
        gap = tube[:, :, None] & in_gap[None, None, :]
        labels[gap] = TISSUE_INDEX[fracture.fill]
        masks["fracture"] = gap
        masks["hematoma"] = gap.copy()
    meta = {
        "generator": "leg_segment",
        "bone_axis": "x",
        "spec": dict(sp.__dict__),
        "fracture": dict(fracture.__dict__) if fracture is not None else None,
        "axis_depth": float(z_c),
        "bone_length": bone_length,
        "bone_top_depth": float(z_c - sp.outer_radius),
        "skin_depth": float(sp.standoff),
        "label": int(fracture is not None and fracture.gap > 0),
    }
    return Phantom3D(grid, labels, masks, meta)
