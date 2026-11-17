"""Fantomas 3D de extremidad completos: la base común de todas las modalidades 3D.

``limb_3d`` es el equivalente volumétrico de los dos cortes 2D de ``limb.py``:
la sección transversal es la de ``CROSS_PRESETS`` (tibia anteromedial casi
bajo la piel y peroné lateral y posterior; radio y cúbito; fémur), con piel,
grasa, músculo, cortical y médula, y se prolonga a lo largo del eje del hueso
(x) con una variación suave opcional (ensanchamiento metafisario con hueso
trabecular, inclinación y giro de los huesos). Las fracturas (``Fracture3D``)
son las mismas que en 2D, con los mismos nombres y parámetros, más los
propios del 3D (arco del perímetro que corta la grieta, desplazamiento
lateral del fragmento).

Convenciones (las de ``limb3d.py``): arrays ``(nz, ny, nx)``, x = eje del
hueso, y = lateral, z = profundidad (la sonda arriba, z creciente hacia
dentro). El eje del hueso principal (tibia, radio, fémur) pasa por
``x = 0, y = 0`` de modo que ``slice_y(0)`` es el corte longitudinal por su
centro, como el de ``longitudinal_limb``; el plano ``x = 0`` es una celda
de la malla. El miembro queda a ``standoff`` del borde superior (agua, gel o
aire, parámetro ``background``) y con ``margin`` a los lados y debajo.

La geometría se define en coordenadas continuas (centros y radios de los
huesos, campos de rugosidad 2D sobre cada interfaz con paso propio, planos y
elipsoides de la fractura, posiciones de las microfracturas y de las bolsas
de edema) y se rasteriza por el centro de cada celda: la misma semilla da la
misma anatomía a cualquier ``dx``. La rasterización va por rodajas en x y
evalúa la rugosidad sólo en la banda de cada interfaz, para que una pierna
de 60 × 100 × 100 mm a 0.5 mm (4.8 M celdas) tarde unos segundos.

Simplificaciones respecto a la anatomía real (ver ``docs/validacion/fantoma3d.md``):
huesos de sección circular (la tibia real es triangular), compartimentos
musculares sin fascias ni tabiques, vasos rectos paralelos al hueso, sin
modo realista (porosidad cortical, red trabecular explícita).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..core.grid3d import Grid3D
from .limb import (
    _DISPLACEABLE, _MIN_MEDULLA, BODY_TYPES, CROSS_PRESETS, EDEMA_POCKET_RADIUS, FRACTURE_KINDS, MEDULLARY_TISSUES,
    BodyType, BoneSpec, CrossSectionSpec, Fracture, LimbSpec, random_body_type, random_fracture, with_body_type,
)
from .limb3d import Phantom3D
from .phantom import TISSUE_INDEX
from .roughness import RoughSurface

# --------------------------------------------------------------------------
# Especificaciones
# --------------------------------------------------------------------------
_T = TISSUE_INDEX
_SOFT_IDX = np.array([_T[t] for t in _DISPLACEABLE], dtype=np.int16)
_MED_IDX = np.array([_T[t] for t in MEDULLARY_TISSUES], dtype=np.int16)
_MED_EDEMA_IDX = np.array([_T[t] for t in (*MEDULLARY_TISSUES, "edema")], dtype=np.int16)


@dataclass
class Metaphysis:
    """Ensanchamiento metafisario de un hueso a lo largo de x (como el preset 2D ``tibia_metafisis``).

    El hueso pasa de la diáfisis a la metáfisis con una transición suave de
    longitud ``transition`` centrada en ``position`` (polinomio ``3t² − 2t³``,
    de derivada continua y que vale exactamente 0 y 1 fuera de ella): el radio
    exterior se multiplica por ``flare``, la cortical pasa a ``cortex`` y el
    endostio se tapiza de hueso trabecular de grosor ``trabecular`` (en la
    metáfisis real la cavidad está llena de trabéculas con algo de médula en
    el centro). El hueso crece hacia el centro del miembro: la cara que está
    bajo la piel (la anteromedial de la tibia) no se mueve.
    """

    position: float = 0.0
    direction: int = 1
    transition: float = 20e-3
    flare: float = 1.68
    cortex: float = 1.5e-3
    trabecular: float = 13e-3
    bone_index: int = 0

    def weight(self, x):
        t = np.clip(0.5 + self.direction * (np.asarray(x) - self.position) / self.transition, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)


@dataclass
class Limb3DSpec:
    """Anatomía de un fantoma 3D de extremidad (metros y grados).

    - ``cross``: sección transversal (``CrossSectionSpec`` de ``limb.py``):
      radio de la piel, grosores de piel y grasa, huesos (centro, radio,
      cortical, contenido medular), vasos y rugosidad de cada interfaz. El
      hueso ``bones[0]`` es el principal: su eje pasa por y = 0.
    - ``standoff``: fondo entre el borde superior del dominio (z = 0) y la piel.
    - ``bone_tilt_deg`` / ``bone_yaw_deg``: inclinación de los ejes de los
      huesos respecto al de la piel en el plano x-z (positivo = más profundo
      hacia +x, como ``LimbSpec.bone_tilt_deg``) y en el plano x-y.
    - ``rotation_deg``: giro de huesos y vasos alrededor del eje del miembro
      (0 = la sección de ``cross``; positivo de -z hacia +y). Equivale a girar
      el miembro bajo la sonda.
    - ``metaphysis``: ensanchamiento metafisario opcional (``Metaphysis``).
    """

    name: str = "pierna"
    cross: CrossSectionSpec = field(default_factory=lambda: CROSS_PRESETS["pierna"])
    standoff: float = 5e-3
    bone_tilt_deg: float = 0.0
    bone_yaw_deg: float = 0.0
    rotation_deg: float = 0.0
    metaphysis: Metaphysis | None = None
    body: str = "normal"


#: Presets 3D. "pierna": tibia y peroné (``CROSS_PRESETS["pierna"]``);
#: "pierna_metafisis": la misma pierna con la metáfisis proximal de la tibia
#: hacia x > 0 (cortical de 1.5 mm, radio ×1.68 y 13 mm de trabecular, los
#: grosores del preset 2D ``tibia_metafisis``); "antebrazo": radio y cúbito;
#: "muslo": fémur.
LIMB3D_PRESETS: dict[str, Limb3DSpec] = {
    "pierna": Limb3DSpec("pierna", CROSS_PRESETS["pierna"]),
    "pierna_metafisis": Limb3DSpec("pierna_metafisis", CROSS_PRESETS["pierna"], metaphysis=Metaphysis()),
    "antebrazo": Limb3DSpec("antebrazo", CROSS_PRESETS["antebrazo"]),
    "muslo": Limb3DSpec("muslo", CROSS_PRESETS["muslo"]),
}


@dataclass
class Fracture3D:
    """Fractura de un fantoma 3D: los parámetros de ``limb.Fracture`` más los del 3D.

    Mismo significado que en 2D (``Fracture``), medido en el corte
    longitudinal que pasa por el eje del hueso en la dirección
    ``arc_center_deg`` (0 = el plano y = 0 visto desde la sonda):

    - ``position``, ``angle_deg``: la grieta es un **plano** perpendicular al
      eje del hueso en x = ``position``, girado ``angle_deg`` alrededor del
      eje y (oblicua), con la rugosidad ``roughness`` como campo 2D sobre el
      plano. ``gap``, ``min_gap_cells``: grosor de la lámina.
    - ``depth_fraction`` / ``span``: hasta dónde llega la grieta desde la
      cortical cercana (la del lado ``arc_center_deg``), con la coordenada de
      capas ζ de ``Fracture.span`` (0 periostio cercano, 1 endostio cercano,
      2 endostio lejano, 3 periostio lejano). En 3D un plano limitado en
      profundidad corta un arco del perímetro: es la fractura incompleta.
    - ``displacement``: escalón del fragmento distal (el lado del plano hacia
      +x) en z (positivo = más profundo). ``displacement_y``: además, en y.
    - ``hematoma_*``: manguito de hematoma alrededor del periostio en los
      ángulos donde la grieta sale a la superficie. ``edema_*``,
      ``microcracks``, ``microcrack_*``: edema de médula (elipsoide) y
      microfracturas (discos finos de orientación aleatoria). Reacciones
      ``periosteal_reaction`` / ``endosteal_reaction``: lentes que siguen la
      curvatura del hueso, de extensión ``reaction_length`` a lo largo del
      eje y del perímetro.

    Propios del 3D:

    - ``bone_index``: hueso fracturado (índice en ``cross.bones``). La grieta
      sólo corta ese hueso.
    - ``arc_deg``: arco del perímetro que corta la grieta (360 = todo; menos
      = fisura parcial en arco), centrado en ``arc_center_deg`` o, si la
      grieta está en la cortical lejana, en el lado opuesto.
    - ``arc_center_deg``: dirección del lado "cercano" alrededor del eje del
      hueso (0 = hacia la sonda, -z; positivo hacia +y).
    - ``spiral_pitch``: distinto de 0, la grieta no es un plano sino una
      **hélice** que atraviesa la cortical (fractura espiroidea, la
      "longitudinal/espiroidea" del corte transversal 2D, ``RingFracture``):
      parte de la dirección ``arc_center_deg`` en x = ``position`` y gira una
      vuelta cada ``spiral_pitch`` metros (el signo da el sentido; ``inf`` =
      grieta longitudinal recta), a lo largo de ``crack_length``. ``span`` es
      entonces el tramo de la cortical en fracción de su grosor desde el
      periostio, como en ``RingFracture``. El hematoma sigue la línea de la
      grieta en el periostio. No admite desplazamiento.
    """

    kind: str = "transversal"
    position: float = 0.0
    angle_deg: float = 0.0
    gap: float = 0.4e-3
    displacement: float = 0.0
    depth_fraction: float = 1.0
    roughness: float = 0.08e-3
    hematoma_thickness: float = 1.2e-3
    hematoma_length: float = 10e-3
    edema_length: float = 14e-3
    extra_cracks: tuple[tuple[float, float], ...] = ()
    span: tuple[float, float] | None = None
    min_gap_cells: float = 0.0
    edema_depth: float | None = None
    edema_fraction: float = 1.0
    microcracks: int = 0
    microcrack_length: float = 1.5e-3
    microcrack_gap: float = 0.08e-3
    periosteal_reaction: float = 0.0
    endosteal_reaction: float = 0.0
    reaction_length: float = 15e-3
    bone_index: int = 0
    arc_deg: float = 360.0
    arc_center_deg: float = 0.0
    displacement_y: float = 0.0
    spiral_pitch: float = 0.0
    crack_length: float = 30e-3

    @classmethod
    def from_2d(cls, fracture: Fracture, **extra) -> "Fracture3D":
        """La fractura 2D ``fracture`` como plano 3D (más los parámetros 3D de ``extra``)."""
        return cls(**{**asdict(fracture), **extra})

    far_side = Fracture.far_side
    opens_near_surface = Fracture.opens_near_surface


# --------------------------------------------------------------------------
# Tipos de cuerpo y variabilidad
# --------------------------------------------------------------------------
def limb3d_with_body_type(spec: str | Limb3DSpec, body: str | BodyType) -> Limb3DSpec:
    """Escala la anatomía 3D al tipo de cuerpo ``body`` (``limb.with_body_type`` de la sección transversal).

    Piel y grasa se escalan directamente; el músculo es el compartimento
    entre la grasa y los huesos, y cada hueso conserva su posición relativa
    en él (la tibia sigue bajo la piel y crece o mengua el músculo de
    detrás). La cortical metafisaria se escala como la diafisaria.
    """
    spec = LIMB3D_PRESETS[spec] if isinstance(spec, str) else spec
    b = BODY_TYPES[body] if isinstance(body, str) else body
    meta = spec.metaphysis
    if meta is not None:
        meta = replace(meta, cortex=meta.cortex * b.cortex)
    return replace(spec, cross=with_body_type(spec.cross, b), metaphysis=meta, body=b.name)


def random_limb_3d_spec(rng: np.random.Generator, preset: str = "pierna",
                        body: str | BodyType | Sequence[str] | None = None, jitter: float = 0.1) -> Limb3DSpec:
    """Anatomía 3D aleatoria (equivalente de ``random_limb_spec``).

    Un tipo de cuerpo (``random_body_type``: nombre al azar entre ``body`` y
    variación log-normal de ~10 % de cada factor) más la variación individual
    uniforme ±``jitter`` de la piel, la grasa, el músculo y la cortical de
    cada hueso, la inclinación de los huesos en x-z y x-y (normal, σ = 2°) y
    el giro del miembro (σ = 5°). Las mismas llamadas a ``rng`` en el mismo
    orden: misma semilla, misma anatomía.
    """
    b = body if isinstance(body, BodyType) else random_body_type(rng, [body] if isinstance(body, str) else body)
    spec = limb3d_with_body_type(LIMB3D_PRESETS[preset], b)

    def jit() -> float:
        return float(rng.uniform(1.0 - jitter, 1.0 + jitter))

    k_skin, k_fat, k_mus = jit(), jit(), jit()
    cross = with_body_type(spec.cross, BodyType("jitter", skin=k_skin, fat=k_fat, muscle=k_mus))
    bones = tuple(replace(bn, cortex=min(bn.cortex * jit(), bn.outer_radius - _MIN_MEDULLA)) for bn in cross.bones)
    cross = replace(cross, bones=bones, body=b.name)
    return replace(spec, cross=cross, bone_tilt_deg=float(rng.normal(0, 2)), bone_yaw_deg=float(rng.normal(0, 2)),
                   rotation_deg=float(rng.normal(0, 5)), body=b.name)


#: Tipos de fractura 3D: los del corte longitudinal 2D (``limb.FRACTURE_KINDS``) más la espiroidea (la
#: "longitudinal/espiroidea" del corte transversal 2D).
FRACTURE_KINDS_3D = FRACTURE_KINDS + ("espiroidea",)


def random_fracture_3d(rng: np.random.Generator, kind: str | None = None, x_range: float = 8e-3,
                       kinds: Sequence[str] = FRACTURE_KINDS_3D, bone_index: int = 0) -> Fracture3D:
    """Fractura 3D aleatoria: ``random_fracture`` (mismos rangos y tipos) más lo propio del 3D.

    - fisura: grosor mínimo de una celda (``min_gap_cells = 1``; si no, una
      grieta de 0.05-0.2 mm desaparece en mallas de 0.25 mm o más) y arco de
      60-180° alrededor de la cara de la sonda (σ = 20°).
    - intracortical y estrés: arco de 60-180°; cortical lejana: 90-240°.
    - desplazada: además un desplazamiento lateral (σ = 0.5 mm).
    - espiroidea: hélice de paso ±40-150 mm (o recta, longitudinal, el 20 %
      de las veces) y 20-50 mm de longitud, con los rangos de grosor,
      rugosidad y hematoma de ``random_ring_fracture``.
    """
    if kind == "espiroidea" or (kind is None and "espiroidea" in kinds and rng.random() < 1.0 / len(kinds)):
        u = rng.uniform
        pitch = np.inf if rng.random() < 0.2 else float(rng.choice([-1, 1]) * u(40e-3, 150e-3))
        return Fracture3D(kind="espiroidea", position=float(u(-x_range, x_range)), gap=float(u(0.25e-3, 0.8e-3)),
                          roughness=float(u(0.03e-3, 0.15e-3)), hematoma_thickness=float(u(0.5e-3, 2e-3)),
                          hematoma_length=float(u(5e-3, 12e-3)), edema_length=float(u(6e-3, 20e-3)),
                          arc_center_deg=float(u(-45, 45)), spiral_pitch=pitch, crack_length=float(u(20e-3, 50e-3)),
                          bone_index=bone_index)
    kinds = tuple(k for k in kinds if k != "espiroidea")
    fr = random_fracture(rng, kind, x_range, kinds)
    extra: dict = {"bone_index": bone_index}
    if fr.kind in ("fisura", "intracortical", "estres"):
        extra.update(arc_deg=float(rng.uniform(60, 180)), arc_center_deg=float(rng.normal(0, 20)))
        if fr.kind == "fisura":
            extra["min_gap_cells"] = 1.0
    elif fr.kind == "cortical_lejana":
        extra.update(arc_deg=float(rng.uniform(90, 240)))
    elif fr.kind == "desplazada":
        extra.update(displacement_y=float(rng.normal(0, 0.5e-3)))
    return Fracture3D.from_2d(fr, **extra)


# --------------------------------------------------------------------------
# Geometría continua
# --------------------------------------------------------------------------
_STREAM_IDS = {"skin": 1, "fat": 2, "muscle": 3, "fracture": 4, "edema": 5, "microcracks": 6}


def _stream(seed: int, name: str, index: int = 0) -> np.random.Generator:
    """Subflujo aleatorio propio de cada superficie u objeto: añadir uno no cambia los demás."""
    key = _STREAM_IDS.get(name, 0) if index == 0 else 100 + 2 * index + (name == "in")
    return np.random.default_rng([int(seed), 0x3D11, key])


def _rotate(y: float, z: float, deg: float) -> tuple[float, float]:
    """Gira (y, z) ``deg`` grados alrededor del eje x, de -z hacia +y."""
    a = np.radians(deg)
    w = -z
    return y * np.cos(a) + w * np.sin(a), -(w * np.cos(a) - y * np.sin(a))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class _Geometry:
    """Geometría continua del miembro (independiente de la malla)."""

    def __init__(self, spec: Limb3DSpec, seed: int, length: float, rough: float, fracture: Fracture3D | None):
        c = spec.cross
        self.spec, self.c = spec, c
        self.R = c.radius
        rel = [_rotate(b.center[0], b.center[1], spec.rotation_deg) for b in c.bones]
        self.y_leg = -rel[0][0]  # el eje del hueso principal pasa por y = 0
        self.z_leg = spec.standoff + self.R
        self.bones = c.bones
        self.rel = rel
        self.vessels = [(self.y_leg + vy, self.z_leg + vz, r)
                        for (vy, vz), r in ((_rotate(v[0], v[1], spec.rotation_deg), v[2]) for v in c.vessels)]
        self.tilt = np.tan(np.radians(spec.bone_tilt_deg))
        self.yaw = np.tan(np.radians(spec.bone_yaw_deg))
        self.meta = spec.metaphysis
        # Dirección en que se desplaza el centro de un hueso que se ensancha: hacia
        # el centro del miembro, para que la cara subcutánea no se mueva.
        self.flare_dir = []
        for (ry, rz), b in zip(rel, c.bones):
            d = np.hypot(ry, rz)
            self.flare_dir.append((-ry / d, -rz / d) if d > b.outer_radius else (0.0, 0.0))
        # Campos de rugosidad: la piel, la grasa y el músculo en (x, arco de la piel); periostio y
        # endostio en (x, arco del hueso). Abarcan [-length/2, length/2] más un margen, sin depender de dx.
        a0, la = -0.5 * length - 1e-3, length + 2e-3
        circ = 2 * np.pi * self.R

        def surf(rms_cl, period, rng):
            return RoughSurface(la, period, rms_cl[0] * rough, rms_cl[1], rng, periodic_b=True, a0=a0)

        self.h_skin = surf(c.skin_rough, circ, _stream(seed, "skin"))
        self.h_fat = surf(c.soft_rough, circ, _stream(seed, "fat"))
        self.h_mus = surf(c.soft_rough, circ, _stream(seed, "muscle"))
        self.h_out, self.h_in = [], []
        for i, b in enumerate(c.bones):
            cb = 2 * np.pi * b.outer_radius
            self.h_out.append(surf(c.periosteal_rough, cb, _stream(seed, "out", i + 1)))
            self.h_in.append(surf(c.endosteal_rough, cb, _stream(seed, "in", i + 1)))
        self.fracture = fracture
        self.crack = None
        if fracture is not None:
            self._init_fracture(fracture, seed)

    # ---- huesos ----------------------------------------------------------
    def profile(self, i: int, x):
        """Radio exterior, grosor cortical y tapiz trabecular del hueso ``i`` en ``x``."""
        b = self.bones[i]
        x = np.asarray(x, dtype=np.float64)
        m = self.meta
        if m is None or m.bone_index != i:
            return np.full(x.shape, b.outer_radius), np.full(x.shape, b.cortex), np.zeros(x.shape)
        w = m.weight(x)
        ro = b.outer_radius * (1.0 + (m.flare - 1.0) * w)
        return ro, b.cortex + (m.cortex - b.cortex) * w, m.trabecular * w

    def center(self, i: int, x):
        """Centro (y, z) del hueso ``i`` en ``x`` (sano)."""
        x = np.asarray(x, dtype=np.float64)
        ry, rz = self.rel[i]
        cy = self.y_leg + ry + self.yaw * x
        cz = self.z_leg + rz + self.tilt * x
        m = self.meta
        if m is not None and m.bone_index == i:
            grow = self.profile(i, x)[0] - self.bones[i].outer_radius
            cy = cy + grow * self.flare_dir[i][0]
            cz = cz + grow * self.flare_dir[i][1]
        return cy, cz

    def frame(self, i: int, X, Y, Z, shift=(0.0, 0.0)) -> dict:
        """Coordenadas cilíndricas locales del hueso ``i`` y sus radios (con rugosidad) en cada punto."""
        cy, cz = self.center(i, X)
        dy = Y - (cy + shift[0])
        dz = Z - (cz + shift[1])
        r = np.hypot(dy, dz)
        th = np.arctan2(dy, -dz)
        ro_n, cort, lining = self.profile(i, X)
        arc = th * self.bones[i].outer_radius
        ro = ro_n + self.h_out[i](X, arc)
        ri = np.maximum(ro_n - cort + self.h_in[i](X, arc), 0.3e-3)
        return {"dy": dy, "dz": dz, "r": r, "th": th, "ro": ro, "ri": ri, "Ro": ro_n, "cortex": cort,
                "lining": lining}

    # ---- fractura --------------------------------------------------------
    def _init_fracture(self, fr: Fracture3D, seed: int) -> None:
        i = fr.bone_index
        xc = fr.position
        cy, cz = (float(v) for v in self.center(i, xc))
        ro = float(self.profile(i, xc)[0])
        self.crack_center = (xc, cy, cz)
        self.crack_theta = np.radians(fr.angle_deg)
        crack_len = 4.0 * ro + 10e-3  # como en 2D: 2·(grosor del hueso) + 10 mm
        if fr.spiral_pitch != 0:
            crack_len = max(crack_len, fr.crack_length + 10e-3)
        self.crack = RoughSurface(crack_len, crack_len, fr.roughness, max(fr.gap, 0.3e-3), _stream(seed, "fracture"),
                                  a0=-0.5 * crack_len, b0=-0.5 * crack_len)
        self.shift = (fr.displacement_y, fr.displacement)
        self.displaced = fr.displacement != 0.0 or fr.displacement_y != 0.0
        if self.displaced and fr.spiral_pitch != 0:
            raise ValueError("La fractura espiroidea (spiral_pitch != 0) no admite desplazamiento")

    def crack_distance(self, X, Y, Z, x_offset: float = 0.0, angle_deg: float | None = None):
        """Distancia con signo a la superficie de fractura (> 0: fragmento distal, hacia +x)."""
        xc, yc, zc = self.crack_center
        th = self.crack_theta if angle_deg is None else np.radians(angle_deg)
        xc = xc + x_offset
        along = (X - xc) * np.sin(th) + (Z - zc) * np.cos(th)
        across = (X - xc) * np.cos(th) - (Z - zc) * np.sin(th)
        return across - self.crack(along, Y - yc)


# --------------------------------------------------------------------------
# Rasterización
# --------------------------------------------------------------------------
def _index_range(coords: np.ndarray, lo: float, hi: float) -> slice:
    """Índices de las celdas de una malla regular cuyo centro cae en [lo, hi]."""
    d = coords[1] - coords[0] if coords.size > 1 else 1.0
    i0 = int(np.clip(np.ceil((lo - coords[0]) / d - 1e-9), 0, coords.size))
    i1 = int(np.clip(np.floor((hi - coords[0]) / d + 1e-9) + 1, i0, coords.size))
    return slice(i0, i1)


class _SoftPlane:
    """Clasificación de las celdas del plano (y, z) respecto a las interfaces de las partes blandas.

    Dentro del músculo y fuera de la piel no hace falta evaluar la rugosidad:
    sólo en la banda entre el músculo más interno y la piel más externa.
    """

    def __init__(self, g: _Geometry, ys: np.ndarray, zs: np.ndarray, background: str):
        c = g.c
        Yp, Zp = np.meshgrid(ys - g.y_leg, zs - g.z_leg)
        R2 = np.hypot(Yp, Zp)
        hs, hm = g.h_skin.max_abs, g.h_mus.max_abs
        r_out = g.R + hs
        r_in = g.R - c.skin - c.fat - hm  # el músculo se asigna el último: dentro de r_in siempre gana
        self.base = np.full(R2.shape, _T[background], np.int16)
        self.base[R2 <= r_in] = _T["muscle"]
        band = (R2 > r_in) & (R2 <= r_out)
        self.band = np.flatnonzero(band)
        self.r = R2.ravel()[self.band]
        self.arc = (np.arctan2(Yp, -Zp) * g.R).ravel()[self.band]
        self.bg = _T[background]

    def labels(self, g: _Geometry, xs: np.ndarray) -> np.ndarray:
        c = g.c
        nz, ny = self.base.shape
        out = np.repeat(self.base[:, :, None], xs.size, axis=2)
        if self.band.size:
            X, A = xs[None, :], self.arc[:, None]
            hs = g.h_skin(X, A)
            r = self.r[:, None]
            lab = np.full((self.band.size, xs.size), self.bg, np.int16)
            lab[r <= g.R + hs] = _T["skin"]
            lab[r <= g.R - c.skin + 0.5 * hs + 0.2 * g.h_fat(X, A)] = _T["fat"]
            lab[r <= g.R - c.skin - c.fat + g.h_mus(X, A)] = _T["muscle"]
            out.reshape(nz * ny, xs.size)[self.band] = lab
        return out


def _bone_bbox(g: _Geometry, i: int, xs: np.ndarray, pad: float, shifts=((0.0, 0.0),)):
    cy, cz = g.center(i, xs)
    ro = g.profile(i, xs)[0].max() + g.h_out[i].max_abs + pad
    ylo = min(cy.min() + s[0] for s in shifts) - ro
    yhi = max(cy.max() + s[0] for s in shifts) + ro
    zlo = min(cz.min() + s[1] for s in shifts) - ro
    zhi = max(cz.max() + s[1] for s in shifts) + ro
    return (ylo, yhi), (zlo, zhi)


def _combined_frame(g: _Geometry, i: int, X, Y, Z) -> tuple[dict, dict, np.ndarray | None]:
    """Marco del hueso ``i`` sano y, si es el fracturado y está desplazado, el del fragmento distal."""
    fh = g.frame(i, X, Y, Z)
    fr = g.fracture
    if fr is None or fr.bone_index != i or not g.displaced:
        return fh, fh, None
    fd = g.frame(i, X, Y, Z, g.shift)
    distal = g.crack_distance(X, Y, Z) > 0
    comb = {k: np.where(distal, fd[k], fh[k]) for k in ("dy", "dz", "r", "th", "ro", "ri", "lining")}
    comb.update(Ro=fh["Ro"], cortex=fh["cortex"])
    return fh, comb, distal


def _anatomy(g: _Geometry, plane: _SoftPlane, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Etiquetas del miembro sano (con el fragmento distal desplazado, si lo hay) en la caja (xs, ys, zs)."""
    labels = plane.labels(g, xs)
    X = xs[None, None, :]
    for i, b in enumerate(g.bones):
        shifts = ((0.0, 0.0), g.shift) if g.crack is not None and g.fracture.bone_index == i else ((0.0, 0.0),)
        (ylo, yhi), (zlo, zhi) = _bone_bbox(g, i, xs, 0.0, shifts)
        js, ks = _index_range(ys, ylo, yhi), _index_range(zs, zlo, zhi)
        if js.start == js.stop or ks.start == ks.stop:
            continue
        _, f, _ = _combined_frame(g, i, X, ys[js][None, :, None], zs[ks][:, None, None])
        sub = labels[ks, js, :]
        med = f["r"] <= f["ri"]
        sub[f["r"] <= f["ro"]] = _T["cortical_bone"]
        sub[med] = _T[b.medullary]
        lining = np.broadcast_to(f["lining"], med.shape)
        if lining.any():
            sub[med & (f["r"] > f["ri"] - lining)] = _T["trabecular_bone"]
    for vy, vz, rv in g.vessels:
        js, ks = _index_range(ys, vy - rv, vy + rv), _index_range(zs, vz - rv, vz + rv)
        if js.start == js.stop or ks.start == ks.stop:
            continue
        disk = (ys[js][None, :] - vy) ** 2 + (zs[ks][:, None] - vz) ** 2 <= rv**2
        sub = labels[ks, js, :]
        sub[disk[:, :, None] & np.isin(sub, _SOFT_IDX)] = _T["blood"]
    return labels


# --------------------------------------------------------------------------
# Fractura
# --------------------------------------------------------------------------
def _box_window(coords: np.ndarray, center: float, half: float) -> slice:
    return _index_range(coords, center - half, center + half)


def _sample_ellipsoid(rng, n: int, center, axes, semi, half: bool) -> np.ndarray:
    """``n`` puntos uniformes en un elipsoide (o en su mitad ``+axes[1]`` si ``half``).

    ``axes``: tres vectores unitarios ortogonales; ``semi``: semiejes a lo
    largo de cada uno. En coordenadas continuas: no dependen de la malla.
    """
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-30
    v *= rng.uniform(0.0, 1.0, (n, 1)) ** (1.0 / 3.0)
    if half:
        v[:, 1] = np.abs(v[:, 1])
    return np.asarray(center) + sum(v[:, k:k + 1] * semi[k] * np.asarray(axes[k]) for k in range(3))


def _paint(mask, xs, ys, zs, center, reach: float, inside) -> None:
    """Marca en ``mask`` las celdas de la ventana ``center ± reach`` donde ``inside(dx, dy, dz)``."""
    cx, cy, cz = center
    i_s, j_s, k_s = _box_window(xs, cx, reach), _box_window(ys, cy, reach), _box_window(zs, cz, reach)
    if i_s.start == i_s.stop or j_s.start == j_s.stop or k_s.start == k_s.stop:
        return
    d = (xs[i_s][None, None, :] - cx, ys[j_s][None, :, None] - cy, zs[k_s][:, None, None] - cz)
    mask[k_s, j_s, i_s] |= inside(*d)


class _Zone:
    """Zona de edema (y de microfracturas): elipsoide, entero o la mitad que entra en la médula."""

    def __init__(self, center, axes, semi, half: bool):
        self.center, self.axes, self.semi, self.half = np.asarray(center), axes, semi, half

    def contains(self, X, Y, Z):
        d = (X - self.center[0], Y - self.center[1], Z - self.center[2])
        q = 0.0
        for ax, s in zip(self.axes, self.semi):
            q = q + ((d[0] * ax[0] + d[1] * ax[1] + d[2] * ax[2]) / s) ** 2
        return q <= 1.0

    @property
    def volume(self) -> float:
        v = 4.0 / 3.0 * np.pi * self.semi[0] * self.semi[1] * self.semi[2]
        return 0.5 * v if self.half else v


def _fracture_box(g: _Geometry, grid: Grid3D):
    fr = g.fracture
    i = fr.bone_index
    xc = fr.position
    ro = float(g.profile(i, xc)[0])
    reach = [fr.hematoma_length / 2, fr.edema_length / 2, fr.reaction_length / 2 + 2e-3,
             abs(np.tan(g.crack_theta)) * 2 * ro + fr.gap]
    if fr.spiral_pitch != 0:
        reach.append(fr.crack_length / 2 + fr.hematoma_thickness)
    reach += [abs(off) + abs(np.tan(np.radians(a))) * 2 * ro + fr.gap for off, a in fr.extra_cracks]
    hx = max(reach) + fr.microcrack_length + 2e-3
    xs_all = grid.x
    i_s = _index_range(xs_all, xc - hx, xc + hx)
    xs = xs_all[i_s]
    pad = max(fr.hematoma_thickness, fr.periosteal_reaction + 0.5e-3) + 2e-3
    (ylo, yhi), (zlo, zhi) = _bone_bbox(g, i, xs if xs.size else np.array([xc]), pad, ((0.0, 0.0), g.shift))
    return _index_range(grid.z, zlo, zhi), _index_range(grid.y, ylo, yhi), i_s


def _apply_fracture(g: _Geometry, grid: Grid3D, labels: np.ndarray, masks: dict, seed: int) -> None:
    """Grieta, edema, reacciones, microfracturas y hematoma (el mismo orden que ``_apply_longitudinal_fracture``)."""
    fr = g.fracture
    i = fr.bone_index
    ks, js, i_s = _fracture_box(g, grid)
    if ks.start == ks.stop or js.start == js.stop or i_s.start == i_s.stop:
        return
    xs, ys, zs = grid.x[i_s], grid.y[js], grid.z[ks]
    X, Y, Z = xs[None, None, :], ys[None, :, None], zs[:, None, None]
    out = labels[ks, js, i_s].copy()
    box_masks = {k: masks[k][ks, js, i_s].copy() for k in ("fracture", "hematoma", "edema")}
    xc, yc, zc = g.crack_center
    Ro = float(g.profile(i, xc)[0])
    ri_nom = Ro - float(g.profile(i, xc)[1])

    fh, f, distal = _combined_frame(g, i, X, Y, Z)
    dist = g.crack_distance(X, Y, Z)
    bone_h = fh["r"] <= fh["ro"]
    cort_h = bone_h & (fh["r"] > fh["ri"])
    if distal is not None:
        fd = g.frame(i, X, Y, Z, g.shift)
        bone_any = bone_h | (fd["r"] <= fd["ro"])
        cortical_any = cort_h | ((fd["r"] <= fd["ro"]) & (fd["r"] > fd["ri"]))
    else:
        bone_any, cortical_any = bone_h, cort_h

    a_ref = np.radians(fr.arc_center_deg)
    ref = (np.sin(a_ref), -np.cos(a_ref))  # dirección (y, z) del lado cercano
    far = fr.far_side
    sgn = -1.0 if far else 1.0
    foc = (sgn * ref[0], sgn * ref[1])  # lado donde está la grieta
    th_focus = a_ref + (np.pi if far else 0.0)

    def arc_ok(th):
        return np.abs(_wrap(th - th_focus)) <= 0.5 * np.radians(fr.arc_deg) + 1e-9

    helical = fr.spiral_pitch != 0
    if helical:
        # ángulo de cada punto respecto a la línea de la hélice en su x
        twist = 0.0 if np.isinf(fr.spiral_pitch) else 2 * np.pi / fr.spiral_pitch
        d_helix = _wrap(fh["th"] - th_focus - twist * (X - xc))
        along_helix = np.abs(X - xc) <= 0.5 * fr.crack_length

    # Profundidad de la grieta (``depth_fraction`` o ``span``) desde el lado cercano
    if fr.span is None:
        proj_h = fh["dy"] * ref[0] + fh["dz"] * ref[1]
        away = -(fr.displacement_y * ref[0] + fr.displacement * ref[1])
        depth_ok = Ro - proj_h <= fr.depth_fraction * 2 * Ro + max(away, 0.0)
    else:
        proj = f["dy"] * ref[0] + f["dz"] * ref[1]
        u = (f["ro"] - f["r"]) / np.maximum(f["ro"] - f["ri"], 1e-9)
        zeta = np.where(proj > 0, u, 3.0 - u)
        zeta = np.where(f["r"] <= f["ri"], np.clip(1.5 - 0.5 * proj / np.maximum(f["ri"], 1e-9), 1.0, 2.0), zeta)
        depth_ok = (zeta >= fr.span[0]) & (zeta <= fr.span[1])

    crack = np.zeros(out.shape, dtype=bool)
    if fr.gap > 0:
        half_gap = 0.5 * max(fr.gap, fr.min_gap_cells * grid.dx)
        if helical:
            sel = cort_h & along_helix
            if fr.span is not None:
                u = (fh["ro"] - fh["r"]) / np.maximum(fh["ro"] - fh["ri"], 1e-9)
                sel &= (u >= fr.span[0]) & (u <= fr.span[1])
            crack = (np.abs(fh["r"] * d_helix - g.crack(X - xc, fh["r"] - Ro)) <= half_gap) & sel
        else:
            sel = bone_any & depth_ok & arc_ok(f["th"])
            crack = (np.abs(dist) <= half_gap) & sel
            for off, ang in fr.extra_cracks:
                crack |= (np.abs(g.crack_distance(X, Y, Z, off, ang)) <= half_gap) & sel

    # Edema de médula (antes de abrir la grieta)
    ex = (1.0, 0.0, 0.0)
    if fr.edema_depth is None:
        zone = _Zone((xc, yc, zc), (ex, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), (fr.edema_length / 2, Ro, Ro), False)
    else:
        e_pt = (xc, yc + foc[0] * ri_nom, zc + foc[1] * ri_nom)
        inward = (0.0, -foc[0], -foc[1])
        tangent = (0.0, -foc[1], foc[0])
        zone = _Zone(e_pt, (ex, inward, tangent), (fr.edema_length / 2, fr.edema_depth, fr.edema_length / 2), True)
    zone_mask = np.zeros(out.shape, dtype=bool)
    if fr.edema_length > 0:
        zone_mask = zone.contains(X, Y, Z) & np.isin(out, _MED_IDX)
        edema = zone_mask
        if fr.edema_fraction < 1.0:
            pockets = _edema_pockets(zone, fr.edema_fraction, _stream(seed, "edema"), out.shape, xs, ys, zs)
            edema = zone_mask & ((out != _T["trabecular_bone"]) | pockets)
        out[edema] = _T["edema"]
        box_masks["edema"] |= edema

    # Reacciones perióstica (hueso nuevo + ribete de edema) y endóstica (esclerosis)
    half_len = fr.reaction_length / 2
    dth = _wrap(fh["th"] - th_focus)
    if fr.endosteal_reaction > 0:
        q = ((X - xc) / half_len) ** 2 + ((ri_nom - fh["r"]) / fr.endosteal_reaction) ** 2
        q = q + (ri_nom * dth / half_len) ** 2
        lens = np.isin(out, _MED_EDEMA_IDX) & (q <= 1.0)
        out[lens] = _T["cortical_bone"]
        box_masks["edema"] &= ~lens
    if fr.periosteal_reaction > 0:
        def lens_of(lx, t):
            return ((X - xc) / lx) ** 2 + ((fh["r"] - Ro) / t) ** 2 + (Ro * dth / lx) ** 2 <= 1.0

        lens = np.isin(out, _SOFT_IDX) & lens_of(half_len, fr.periosteal_reaction)
        out[lens] = _T["trabecular_bone"]
        rim = np.isin(out, _SOFT_IDX) & lens_of(half_len + 2e-3, fr.periosteal_reaction + 0.5e-3)
        out[rim] = _T["edema"]
        box_masks["edema"] |= rim

    out[crack] = _T["hematoma"]
    box_masks["fracture"] |= crack & cortical_any
    box_masks["hematoma"] |= crack

    # Microfracturas: discos finos dentro de la médula de la zona de edema
    if fr.microcracks > 0 and zone_mask.any():
        inside = np.isin(out, _MED_EDEMA_IDX) & (f["r"] < f["ri"])
        micro = _microcracks(zone, fr, grid.dx, _stream(seed, "microcracks"), out.shape, xs, ys, zs) & inside
        out[micro] = _T["hematoma"]
        box_masks["fracture"] |= micro
        box_masks["hematoma"] |= micro

    # Hematoma perióstico: manguito en los ángulos donde la grieta sale al periostio
    if fr.gap > 0 and fr.hematoma_thickness > 0 and fr.hematoma_length > 0:
        d_ref = _wrap(fh["th"] - a_ref)
        near_half = np.cos(d_ref) > 0
        if helical:
            reaches = along_helix & (fr.span is None or fr.span[0] <= 0)
        elif fr.span is None:
            away = -(fr.displacement_y * ref[0] + fr.displacement * ref[1])
            reaches = Ro * (1.0 - np.cos(d_ref)) <= fr.depth_fraction * 2 * Ro + max(away, 0.0) + 1e-12
        else:
            reaches = (near_half & (fr.span[0] <= 0)) | (~near_half & (fr.span[1] >= 3))
        radial = ((fh["r"] - Ro) / fr.hematoma_thickness) ** 2
        if helical:
            # a lo largo de la línea de la grieta, con anchura ``hematoma_length`` (como en ``RingFracture``)
            sleeve = radial + (Ro * d_helix / (fr.hematoma_length / 2)) ** 2 <= 1.0
        else:
            sleeve = (((X - xc) / (fr.hematoma_length / 2)) ** 2 + radial <= 1.0) & arc_ok(fh["th"])
        hem = sleeve & reaches & np.isin(out, _SOFT_IDX)
        out[hem] = _T["hematoma"]
        box_masks["hematoma"] |= hem

    labels[ks, js, i_s] = out
    for k, m in box_masks.items():
        masks[k][ks, js, i_s] = m


def _edema_pockets(zone: _Zone, fraction: float, rng, shape, xs, ys, zs) -> np.ndarray:
    """Bolsas esféricas de edema que cubren en promedio ``fraction`` de la zona (Poisson booleano, como en 2D)."""
    out = np.zeros(shape, dtype=bool)
    if fraction <= 0:
        return out
    r = EDEMA_POCKET_RADIUS
    n = int(round(-np.log1p(-min(fraction, 0.99)) * zone.volume / (4.0 / 3.0 * np.pi * r**3)))
    for c in _sample_ellipsoid(rng, n, zone.center, zone.axes, zone.semi, zone.half):
        _paint(out, xs, ys, zs, c, r, lambda a, b, d: a * a + b * b + d * d <= r * r)
    return out


def _microcracks(zone: _Zone, fr: Fracture3D, dx: float, rng, shape, xs, ys, zs) -> np.ndarray:
    """Discos finos (grietas en forma de moneda) de orientación aleatoria con centro en la zona."""
    out = np.zeros(shape, dtype=bool)
    n = fr.microcracks
    centers = _sample_ellipsoid(rng, n, zone.center, zone.axes, zone.semi, zone.half)
    normals = rng.standard_normal((n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-30
    radii = 0.5 * fr.microcrack_length * rng.uniform(0.5, 1.5, n)
    half_w = 0.5 * max(fr.microcrack_gap, fr.min_gap_cells * dx)
    for c, nv, a in zip(centers, normals, radii):
        def disk(px, py, pz, nv=nv, a=a):
            dn = px * nv[0] + py * nv[1] + pz * nv[2]
            return (np.abs(dn) <= half_w) & (px * px + py * py + pz * pz - dn * dn <= a * a)

        _paint(out, xs, ys, zs, c, a + half_w, disk)
    return out


# --------------------------------------------------------------------------
# Generador
# --------------------------------------------------------------------------
def _resolve(preset: str | Limb3DSpec, body: str | BodyType | None) -> Limb3DSpec:
    spec = LIMB3D_PRESETS[preset] if isinstance(preset, str) else preset
    return limb3d_with_body_type(spec, body) if body is not None else spec


def limb_grid(spec: Limb3DSpec, length: float, dx: float, width: float | None = None, depth: float | None = None,
              margin: float = 5e-3) -> Grid3D:
    """Malla por defecto de ``limb_3d``: el miembro entero más ``margin``, con x = 0 e y = 0 en centros de celda."""
    R = spec.cross.radius
    y_leg = -_rotate(*spec.cross.bones[0].center, spec.rotation_deg)[0]
    width = 2 * (R + margin) if width is None else width
    depth = spec.standoff + 2 * R + margin if depth is None else depth
    nx, ny, nz = (max(int(round(s / dx)), 4) for s in (length, width, depth))
    j0 = int(round((0.5 * width - y_leg) / dx))
    return Grid3D(nx, ny, nz, dx, -(nx // 2) * dx, -j0 * dx, 0.0)


def limb_3d(
    preset: str | Limb3DSpec = "pierna",
    fracture: Fracture3D | Fracture | None = None,
    length: float = 60e-3,
    dx: float = 0.5e-3,
    width: float | None = None,
    depth: float | None = None,
    margin: float = 5e-3,
    seed: int = 0,
    body: str | BodyType | None = None,
    background: str | None = None,
    roughness: float | bool = True,
    grid: Grid3D | None = None,
    chunk_cells: int = 2_000_000,
) -> Phantom3D:
    """Fantoma 3D de extremidad con el eje del hueso en x (``Phantom3D``).

    - ``preset``: nombre de ``LIMB3D_PRESETS`` ("pierna", "pierna_metafisis",
      "antebrazo", "muslo") o un ``Limb3DSpec`` (p. ej. de
      ``random_limb_3d_spec``).
    - ``fracture``: ``Fracture3D`` (o una ``Fracture`` 2D, que se convierte
      con ``Fracture3D.from_2d``).
    - ``length``: longitud en x, centrada en x = 0. ``width``, ``depth``: por
      defecto el miembro entero más ``margin`` a los lados y debajo, y el
      ``standoff`` del preset encima. ``grid``: malla propia (sustituye a
      ``length``/``dx``/``width``/``depth``; ``length`` sigue fijando la
      extensión de los campos de rugosidad, que debe cubrir la de la malla).
    - ``body``: tipo de cuerpo (``BODY_TYPES`` o ``BodyType``).
    - ``background``: tejido de fondo ("water" por defecto, "air"...).
    - ``roughness``: escala de la rugosidad de las interfaces (``True`` = la
      del preset, ``False`` o 0 = interfaces lisas). La de la superficie de
      fractura es ``fracture.roughness``.
    - ``chunk_cells``: celdas por rodaja de la rasterización (memoria).

    Máscaras: "fracture", "hematoma" y "edema" (las de 2D). ``meta``: la
    especificación, ``bone_axis = "x"``, el centro de cada hueso en x = 0, la
    profundidad de la piel y del periostio del hueso principal en y = 0,
    ``fracture_center`` (x, y, z) del foco, etc.
    """
    spec = _resolve(preset, body)
    if isinstance(fracture, Fracture):
        fracture = Fracture3D.from_2d(fracture)
    background = background or spec.cross.background
    rough = float(roughness)
    if grid is None:
        grid = limb_grid(spec, length, dx, width, depth, margin)
    g = _Geometry(spec, seed, length, rough, fracture)
    plane = _SoftPlane(g, grid.y, grid.z, background)
    labels = np.empty(grid.shape, dtype=np.int16)
    step = max(1, chunk_cells // max(grid.ny * grid.nz, 1))
    for i0 in range(0, grid.nx, step):
        i1 = min(i0 + step, grid.nx)
        labels[:, :, i0:i1] = _anatomy(g, plane, grid.x[i0:i1], grid.y, grid.z)
    masks = {k: np.zeros(grid.shape, dtype=bool) for k in ("fracture", "hematoma", "edema")}
    if fracture is not None:
        _apply_fracture(g, grid, labels, masks, seed)

    c = spec.cross
    cz0 = float(g.center(0, 0.0)[1])
    ro0 = float(g.profile(0, 0.0)[0])
    meta = {
        "generator": "limb_3d",
        "bone_axis": "x",
        "preset": spec.name,
        "spec": asdict(spec),
        "fracture": asdict(fracture) if fracture is not None else None,
        "seed": seed,
        "label": int(fracture is not None),
        "body_type": spec.body,
        "background": background,
        "roughness": rough,
        "leg_center": (float(g.y_leg), float(g.z_leg)),
        "leg_radius": float(c.radius),
        "bone_centers": [tuple(float(v) for v in g.center(i, 0.0)) for i in range(len(c.bones))],
        "bone_top_depth": cz0 - ro0,
        "bone_bottom_depth": cz0 + ro0,
        "axis_depth": cz0,
        "skin_depth": float(g.z_leg - np.sqrt(max(c.radius**2 - g.y_leg**2, 0.0))),
        "fracture_center": tuple(float(v) for v in g.crack_center) if fracture is not None else None,
    }
    return Phantom3D(grid, labels, masks, meta)


def limb_3d_fractions(*args, supersample: int = 2, **kwargs) -> tuple[Phantom3D, dict[str, np.ndarray]]:
    """``limb_3d`` más la fracción de volumen de cada tejido en cada celda (antialiasing de interfaces).

    Rasteriza la misma anatomía (misma semilla) en una malla ``supersample``
    veces más fina, alineada con la gruesa, y cuenta los tejidos de cada
    bloque de ``supersample³`` subceldas. Devuelve el fantoma grueso (por el
    centro de cada celda, como ``limb_3d``) y ``{tejido: fracción float32}``
    para los tejidos presentes. Coste: el de rasterizar ``supersample³``
    veces más celdas.
    """
    s = int(supersample)
    coarse = limb_3d(*args, **kwargs)
    gc = coarse.grid
    h = gc.dx / s
    off = 0.5 * (s - 1) * h
    fine_grid = Grid3D(gc.nx * s, gc.ny * s, gc.nz * s, h, gc.x0 - off, gc.y0 - off, gc.z0 - off)
    kw = dict(kwargs, grid=fine_grid)
    fine = limb_3d(*args, **kw)
    blocks = fine.labels.reshape(gc.nz, s, gc.ny, s, gc.nx, s)
    fractions = {}
    for k in fine.present_tissues():
        cnt = (blocks == _T[k]).sum(axis=(1, 3, 5), dtype=np.int32)
        fractions[k] = (cnt / float(s**3)).astype(np.float32)
    return coarse, fractions


def equivalent_longitudinal_spec(preset: str | Limb3DSpec = "pierna", body: str | BodyType | None = None) -> LimbSpec:
    """``LimbSpec`` 2D del corte y = 0 de ``limb_3d`` (x = 0, sin rugosidad ni vasos).

    Recorre la vertical que pasa por el eje del hueso principal: sirve para
    comparar ``limb_3d(...).slice_y(0)`` con ``longitudinal_limb`` de la misma
    anatomía (grosores de capa). Supone que la vertical no corta otros huesos
    (cierto en los presets) y sin metáfisis ni inclinación en x = 0.
    """
    spec = _resolve(preset, body)
    c = spec.cross
    b: BoneSpec = c.bones[0]
    ry, rz = _rotate(*b.center, spec.rotation_deg)
    d = abs(ry)
    zl = spec.standoff + c.radius

    def half_chord(rho):
        return np.sqrt(max(rho**2 - d**2, 0.0))

    r_fat, r_mus = c.radius - c.skin, c.radius - c.skin - c.fat
    z_skin, z_fat, z_mus = zl - half_chord(c.radius), zl - half_chord(r_fat), zl - half_chord(r_mus)
    z_top, z_bot = zl + rz - b.outer_radius, zl + rz + b.outer_radius
    z_mus_far, z_fat_far, z_skin_far = zl + half_chord(r_mus), zl + half_chord(r_fat), zl + half_chord(c.radius)
    zero = (0.0, 1e-3)
    return LimbSpec(
        name=f"{spec.name}_y0", standoff=float(z_skin), skin=float(z_fat - z_skin),
        fat=float(min(z_mus, z_top) - z_fat), muscle=float(max(z_top - z_mus, 0.0)), cortex=b.cortex,
        medulla=2 * (b.outer_radius - b.cortex), muscle_far=float(z_mus_far - z_bot),
        fat_far=float(z_fat_far - z_mus_far), skin_far=float(z_skin_far - z_fat_far), medullary=b.medullary,
        background=c.background, skin_rough=zero, soft_rough=zero, periosteal_rough=zero, endosteal_rough=zero,
        vessels=(), body=spec.body,
    )


__all__ = [
    "Metaphysis", "Limb3DSpec", "LIMB3D_PRESETS", "Fracture3D", "FRACTURE_KINDS_3D", "limb3d_with_body_type",
    "random_limb_3d_spec", "random_fracture_3d", "limb_grid", "limb_3d", "limb_3d_fractions",
    "equivalent_longitudinal_spec",
]
