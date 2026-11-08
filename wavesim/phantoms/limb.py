"""Generadores de fantomas de extremidades humanas con o sin fractura.

Dos tipos de corte 2D:

- **Longitudinal** (a lo largo del eje del hueso): capas apiladas en
  profundidad (piel, grasa, músculo, cortical, médula, cortical, músculo,
  grasa, piel). Es el plano en el que se explora una fractura con ecografía:
  la fractura aparece como una interrupción o un escalón de la cortical.
- **Transversal** (sección axial): anillos concéntricos con uno o varios
  huesos (tibia y peroné, radio y cúbito, fémur). Es el plano típico de TAC
  y resonancia.

La geometría se define en coordenadas continuas y se rasteriza a la ``dx``
pedida, de modo que la misma semilla produce la misma anatomía a cualquier
resolución (por ejemplo, fina para ultrasonido y más gruesa para RM).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from ..core.grid import Grid2D
from .phantom import TISSUE_INDEX, Phantom
from .roughness import RoughProfile

BONE_TISSUES = ("cortical_bone", "trabecular_bone", "marrow")
SOFT_TISSUES = ("muscle", "fat", "skin")


# --------------------------------------------------------------------------
# Especificaciones
# --------------------------------------------------------------------------
@dataclass
class Fracture:
    """Fractura en un corte longitudinal.

    ``kind``: etiqueta descriptiva ("transversal", "oblicua", "fisura",
    "conminuta", "incompleta", "desplazada"). La geometría la definen los
    parámetros numéricos:

    - ``position``: coordenada x (m) del centro de la grieta.
    - ``angle_deg``: inclinación de la grieta respecto a la perpendicular al eje
      del hueso (0 = transversal).
    - ``gap``: anchura de la grieta (m), rellena de hematoma.
    - ``displacement``: escalón del fragmento distal (m, positivo = más profundo).
    - ``depth_fraction``: fracción del grosor del hueso que atraviesa la grieta
      desde la cortical cercana (1 = fractura completa).
    - ``roughness``: rugosidad (rms, m) de la superficie de fractura.
    - ``hematoma_thickness`` / ``hematoma_length``: hematoma perióstico.
    - ``edema_length``: extensión del edema de médula alrededor del foco.
    - ``extra_cracks``: grietas adicionales ``(desplazamiento_x, ángulo)`` para
      fracturas conminutas.
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


@dataclass
class LimbSpec:
    """Capas de un corte longitudinal de extremidad (grosores en metros).

    Las rugosidades son pares ``(rms, longitud_de_correlación)``.
    ``vessels`` son vasos sanguíneos ``(x, profundidad_bajo_la_piel, radio)``.
    """

    name: str = "tibia"
    standoff: float = 3e-3  # gel/agua entre la sonda y la piel
    skin: float = 1.5e-3
    fat: float = 3e-3
    muscle: float = 0.0
    cortex: float = 4e-3
    medulla: float = 14e-3
    far_cortex: float | None = None
    muscle_far: float = 22e-3
    fat_far: float | None = None
    skin_far: float | None = None
    medullary: str = "marrow"
    background: str = "water"
    bone_tilt_deg: float = 0.0
    skin_rough: tuple[float, float] = (40e-6, 1.0e-3)
    soft_rough: tuple[float, float] = (0.3e-3, 5e-3)
    periosteal_rough: tuple[float, float] = (25e-6, 0.4e-3)
    endosteal_rough: tuple[float, float] = (0.25e-3, 1.5e-3)
    vessels: tuple[tuple[float, float, float], ...] = ()

    def layers(self) -> list[tuple[str, float]]:
        return [
            ("skin", self.skin),
            ("fat", self.fat),
            ("muscle", self.muscle),
            ("cortical_bone", self.cortex),
            (self.medullary, self.medulla),
            ("cortical_bone", self.cortex if self.far_cortex is None else self.far_cortex),
            ("muscle", self.muscle_far),
            ("fat", self.fat if self.fat_far is None else self.fat_far),
            ("skin", self.skin if self.skin_far is None else self.skin_far),
        ]

    @property
    def thickness(self) -> float:
        return sum(t for _, t in self.layers())


PRESETS: dict[str, LimbSpec] = {
    # Cara anteromedial de la tibia: el hueso está justo bajo la piel.
    "tibia": LimbSpec(name="tibia", vessels=((8e-3, 34e-3, 1.5e-3),)),
    # Radio distal (antebrazo).
    "antebrazo": LimbSpec(
        name="antebrazo", fat=4e-3, muscle=6e-3, cortex=2.5e-3, medulla=7e-3, muscle_far=18e-3,
        vessels=((-6e-3, 6e-3, 1.2e-3),),
    ),
    # Fémur (muslo): hueso profundo, mucho músculo y grasa.
    "muslo": LimbSpec(
        name="muslo", skin=2e-3, fat=10e-3, muscle=30e-3, cortex=6e-3, medulla=14e-3, muscle_far=45e-3,
        vessels=((5e-3, 70e-3, 3e-3),),
    ),
}


# --------------------------------------------------------------------------
# Corte longitudinal
# --------------------------------------------------------------------------
def _interface_roughness(spec: LimbSpec) -> list[tuple[float, float]]:
    s, f, p, e = spec.skin_rough, spec.soft_rough, spec.periosteal_rough, spec.endosteal_rough
    small_skin = (s[0] * 0.5, s[1])
    # interfaces: piel sup., piel/grasa, grasa/músculo, músculo/hueso, cortical/médula,
    # médula/cortical, hueso/músculo, músculo/grasa, grasa/piel, piel inf.
    return [s, small_skin, f, p, e, e, p, f, small_skin, s]


def _stack_labels(Z: np.ndarray, zs: np.ndarray, tissues: list[str], background: str) -> np.ndarray:
    """Etiqueta cada celda según cuántas interfaces tiene por encima."""
    count = np.zeros(Z.shape, dtype=np.int16)
    for zk in zs:
        count += Z >= zk[None, :]
    lut = np.array([TISSUE_INDEX[background]] + [TISSUE_INDEX[t] for t in tissues] + [TISSUE_INDEX[background]],
                   dtype=np.int16)
    return lut[count]


def _monotonic(zs: np.ndarray, thicknesses: list[float]) -> np.ndarray:
    zs = zs.copy()
    # Capas de grosor nominal cero desaparecen por completo
    for k in range(len(thicknesses) - 1, -1, -1):
        if thicknesses[k] <= 0:
            zs[k] = zs[k + 1]
    for k in range(1, len(zs)):
        zs[k] = np.maximum(zs[k], zs[k - 1])
    return zs


def longitudinal_limb(
    preset: str | LimbSpec = "tibia",
    fracture: Fracture | None = None,
    width: float = 40e-3,
    dx: float = 0.1e-3,
    depth: float | None = None,
    seed: int = 0,
    bottom_margin: float = 3e-3,
) -> Phantom:
    """Corte longitudinal de una extremidad, centrado en x = 0.

    El eje del hueso es horizontal (x) y la sonda estaría arriba (z = 0).
    ``depth`` por defecto abarca la extremidad completa más un margen.
    """
    spec = PRESETS[preset] if isinstance(preset, str) else preset
    rng = np.random.default_rng(seed)
    layers = spec.layers()
    tissues = [t for t, _ in layers]
    thick = [t for _, t in layers]
    if depth is None:
        depth = spec.standoff + spec.thickness + bottom_margin
    grid = Grid2D.from_size(width, depth, dx, x0=-width / 2)
    x, z = grid.x, grid.z
    X, Z = np.meshgrid(x, z)

    nominal = spec.standoff + np.concatenate([[0.0], np.cumsum(thick)])
    profiles = [RoughProfile(width, rms, cl, rng) for rms, cl in _interface_roughness(spec)]
    s = x - grid.x0
    zs = np.array([nominal[k] + profiles[k](s) for k in range(len(nominal))])
    bone_if = slice(3, 7)  # interfaces que pertenecen al hueso
    zs[bone_if] += np.tan(np.radians(spec.bone_tilt_deg)) * x[None, :]
    zs = _monotonic(zs, thick)

    labels = _stack_labels(Z, zs, tissues, spec.background)
    masks = {k: np.zeros(grid.shape, dtype=bool) for k in ("fracture", "hematoma", "edema")}

    if fracture is not None:
        labels = _apply_longitudinal_fracture(grid, X, Z, zs, thick, tissues, spec, fracture, labels, masks, rng)

    _add_vessels(labels, X, Z, nominal[0], spec.vessels)
    het = np.random.default_rng(seed + 7919).standard_normal(grid.shape).astype(np.float32)
    meta = {
        "view": "longitudinal",
        "preset": spec.name,
        "spec": asdict(spec),
        "fracture": asdict(fracture) if fracture is not None else None,
        "seed": seed,
        "label": int(fracture is not None),
        "bone_top_depth": float(np.interp(0.0, x, zs[3])),
        "skin_depth": float(np.interp(0.0, x, zs[0])),
    }
    return Phantom(grid, labels, masks, het, meta)


def _apply_longitudinal_fracture(grid, X, Z, zs, thick, tissues, spec, fr: Fracture, labels, masks, rng):
    x = grid.x
    xc = fr.position
    z_top = float(np.interp(xc, x, zs[3]))
    z_bottom = float(np.interp(xc, x, zs[6]))
    zc = 0.5 * (z_top + z_bottom)
    theta = np.radians(fr.angle_deg)
    crack_len = 2.0 * (z_bottom - z_top) + 10e-3
    crack_profile = RoughProfile(crack_len, fr.roughness, max(fr.gap, 0.3e-3), rng)

    def signed_distance(xc_, theta_):
        along = (X - xc_) * np.sin(theta_) + (Z - zc) * np.cos(theta_)
        across = (X - xc_) * np.cos(theta_) - (Z - zc) * np.sin(theta_)
        return across - crack_profile(along + crack_len / 2)

    dist = signed_distance(xc, theta)

    # Escalón: el fragmento distal (dist > 0) se desplaza en profundidad
    if fr.displacement != 0.0:
        zs_d = zs.copy()
        zs_d[3:7] += fr.displacement
        zs_d = _monotonic(zs_d, thick)
        labels_d = _stack_labels(Z, zs_d, tissues, spec.background)
    else:
        labels_d = labels

    bone_idx = [TISSUE_INDEX[t] for t in BONE_TISSUES]
    cort = TISSUE_INDEX["cortical_bone"]
    bone_any = np.isin(labels, bone_idx) | np.isin(labels_d, bone_idx)
    cortical_any = (labels == cort) | (labels_d == cort)
    depth_ok = Z <= z_top + fr.depth_fraction * (z_bottom - z_top) + max(fr.displacement, 0.0)
    out = np.where(dist > 0, labels_d, labels)

    crack = (np.abs(dist) <= fr.gap / 2) & bone_any & depth_ok
    for off, ang in fr.extra_cracks:
        d2 = signed_distance(xc + off, np.radians(ang))
        crack |= (np.abs(d2) <= fr.gap / 2) & bone_any & depth_ok

    # Edema en la médula alrededor del foco (antes de abrir la grieta)
    med = np.isin(out, [TISSUE_INDEX["marrow"], TISSUE_INDEX["trabecular_bone"]])
    z_med = 0.5 * (float(np.interp(xc, x, zs[4])) + float(np.interp(xc, x, zs[5])))
    half_med = max(0.5 * (z_bottom - z_top), 1e-4)
    edema = med & (((X - xc) / (fr.edema_length / 2)) ** 2 + ((Z - z_med) / half_med) ** 2 <= 1.0)
    if fr.edema_length > 0:
        out[edema] = TISSUE_INDEX["edema"]
        masks["edema"] |= edema

    hem_idx = TISSUE_INDEX["hematoma"]
    out[crack] = hem_idx
    masks["fracture"] |= crack & cortical_any
    masks["hematoma"] |= crack

    # Hematoma perióstico sobre la superficie del hueso (y bajo ella si es completa)
    if fr.hematoma_thickness > 0 and fr.hematoma_length > 0:
        soft = np.isin(out, [TISSUE_INDEX[t] for t in ("muscle", "fat")])
        surfaces = [z_top] + ([z_bottom] if fr.depth_fraction >= 1.0 else [])
        for zsurf in surfaces:
            ell = ((X - xc) / (fr.hematoma_length / 2)) ** 2 + ((Z - zsurf) / fr.hematoma_thickness) ** 2 <= 1.0
            hem = ell & soft
            out[hem] = hem_idx
            masks["hematoma"] |= hem
    return out


def _add_vessels(labels, X, Z, skin_z, vessels):
    soft = np.isin(labels, [TISSUE_INDEX["muscle"], TISSUE_INDEX["fat"]])
    for vx, vdepth, r in vessels:
        inside = (X - vx) ** 2 + (Z - (skin_z + vdepth)) ** 2 <= r**2
        labels[inside & soft] = TISSUE_INDEX["blood"]


# --------------------------------------------------------------------------
# Corte transversal
# --------------------------------------------------------------------------
@dataclass
class BoneSpec:
    center: tuple[float, float]  # (x, z) respecto al centro de la extremidad (m)
    outer_radius: float
    cortex: float
    medullary: str = "marrow"


@dataclass
class CrossSectionSpec:
    name: str
    radius: float  # radio exterior de la piel
    skin: float = 1.5e-3
    fat: float = 5e-3
    bones: tuple[BoneSpec, ...] = ()
    vessels: tuple[tuple[float, float, float], ...] = ()  # (x, z, radio) respecto al centro
    background: str = "water"
    skin_rough: tuple[float, float] = (40e-6, 1.0e-3)
    soft_rough: tuple[float, float] = (0.4e-3, 6e-3)
    periosteal_rough: tuple[float, float] = (25e-6, 0.4e-3)
    endosteal_rough: tuple[float, float] = (0.3e-3, 2e-3)


@dataclass
class RingFracture:
    """Fractura en un corte transversal: grieta radial a través de la cortical.

    ``angle_deg`` es la dirección de la grieta desde el centro del hueso
    (0 = hacia arriba, hacia la sonda; positivo en sentido horario).
    """

    kind: str = "longitudinal/espiroidea"
    angle_deg: float = 0.0
    gap: float = 0.4e-3
    roughness: float = 0.08e-3
    hematoma_thickness: float = 1.2e-3
    hematoma_length: float = 8e-3
    edema_length: float = 8e-3
    bone_index: int = 0


CROSS_PRESETS: dict[str, CrossSectionSpec] = {
    "pierna": CrossSectionSpec(
        name="pierna", radius=45e-3, fat=5e-3,
        bones=(BoneSpec((-3e-3, -28e-3), 11e-3, 4e-3), BoneSpec((20e-3, 8e-3), 7e-3, 2e-3)),
        vessels=((2e-3, 5e-3, 2e-3), (-10e-3, 10e-3, 1.5e-3)),
    ),
    "antebrazo": CrossSectionSpec(
        name="antebrazo", radius=30e-3, fat=4e-3,
        bones=(BoneSpec((-12e-3, 0.0), 7e-3, 2.5e-3), BoneSpec((12e-3, 3e-3), 6e-3, 2.5e-3)),
        vessels=((-14e-3, -14e-3, 1.2e-3), (8e-3, -16e-3, 1.2e-3)),
    ),
    "muslo": CrossSectionSpec(
        name="muslo", radius=75e-3, skin=2e-3, fat=12e-3,
        bones=(BoneSpec((0.0, 5e-3), 14e-3, 6e-3),),
        vessels=((-20e-3, 20e-3, 4e-3),),
    ),
}


def cross_section_limb(
    preset: str | CrossSectionSpec = "pierna",
    fracture: RingFracture | None = None,
    dx: float = 0.2e-3,
    margin: float = 5e-3,
    seed: int = 0,
) -> Phantom:
    """Sección axial de una extremidad centrada en (0, 0)."""
    spec = CROSS_PRESETS[preset] if isinstance(preset, str) else preset
    rng = np.random.default_rng(seed)
    size = 2 * (spec.radius + margin)
    grid = Grid2D.from_size(size, size, dx, x0=-size / 2, z0=-size / 2)
    X, Z = grid.meshgrid()
    R = np.hypot(X, Z)
    TH = np.arctan2(X, -Z)  # 0 = hacia arriba (-z)

    circ = 2 * np.pi * spec.radius
    h_skin = RoughProfile(circ, *spec.skin_rough, rng, periodic=True)
    h_fat = RoughProfile(circ, *spec.soft_rough, rng, periodic=True)
    h_mus = RoughProfile(circ, *spec.soft_rough, rng, periodic=True)
    arc = TH * spec.radius
    r_skin = spec.radius + h_skin(arc)
    r_fat = r_skin - spec.skin - 0.5 * h_skin(arc) + 0.2 * h_fat(arc)
    r_mus = spec.radius - spec.skin - spec.fat + h_mus(arc)

    labels = np.full(grid.shape, TISSUE_INDEX[spec.background], dtype=np.int16)
    labels[R <= r_skin] = TISSUE_INDEX["skin"]
    labels[R <= r_fat] = TISSUE_INDEX["fat"]
    labels[R <= r_mus] = TISSUE_INDEX["muscle"]

    for b in spec.bones:
        bx, bz = b.center
        rb = np.hypot(X - bx, Z - bz)
        thb = np.arctan2(X - bx, -(Z - bz))
        cb = 2 * np.pi * b.outer_radius
        h_out = RoughProfile(cb, *spec.periosteal_rough, rng, periodic=True)
        h_in = RoughProfile(cb, *spec.endosteal_rough, rng, periodic=True)
        r_out = b.outer_radius + h_out(thb * b.outer_radius)
        r_in = np.maximum(b.outer_radius - b.cortex + h_in(thb * b.outer_radius), 0.3e-3)
        labels[rb <= r_out] = TISSUE_INDEX["cortical_bone"]
        labels[rb <= r_in] = TISSUE_INDEX[b.medullary]

    soft = np.isin(labels, [TISSUE_INDEX["muscle"], TISSUE_INDEX["fat"]])
    for vx, vz, r in spec.vessels:
        labels[((X - vx) ** 2 + (Z - vz) ** 2 <= r**2) & soft] = TISSUE_INDEX["blood"]

    masks = {k: np.zeros(grid.shape, dtype=bool) for k in ("fracture", "hematoma", "edema")}
    if fracture is not None:
        _apply_ring_fracture(X, Z, spec, fracture, labels, masks, rng)

    het = np.random.default_rng(seed + 7919).standard_normal(grid.shape).astype(np.float32)
    meta = {
        "view": "cross_section",
        "preset": spec.name,
        "spec": asdict(spec),
        "fracture": asdict(fracture) if fracture is not None else None,
        "seed": seed,
        "label": int(fracture is not None),
    }
    return Phantom(grid, labels, masks, het, meta)


def _apply_ring_fracture(X, Z, spec: CrossSectionSpec, fr: RingFracture, labels, masks, rng):
    b = spec.bones[fr.bone_index]
    bx, bz = b.center
    phi = np.radians(fr.angle_deg)
    ux, uz = np.sin(phi), -np.cos(phi)  # dirección de la grieta
    along = (X - bx) * ux + (Z - bz) * uz
    across = (X - bx) * uz - (Z - bz) * ux
    prof = RoughProfile(2 * b.outer_radius, fr.roughness, max(fr.gap, 0.3e-3), rng)
    dist = across - prof(along + b.outer_radius)
    cort = labels == TISSUE_INDEX["cortical_bone"]
    near_bone = np.hypot(X - bx, Z - bz) <= b.outer_radius + 1e-3
    crack = (np.abs(dist) <= fr.gap / 2) & (along > 0) & cort & near_bone

    # Edema en la médula junto al punto donde la grieta atraviesa el endostio
    ex, ez = bx + ux * (b.outer_radius - b.cortex), bz + uz * (b.outer_radius - b.cortex)
    med = np.isin(labels, [TISSUE_INDEX["marrow"], TISSUE_INDEX["trabecular_bone"]])
    edema = med & (np.hypot(X - ex, Z - ez) <= fr.edema_length / 2)
    labels[edema] = TISSUE_INDEX["edema"]
    masks["edema"] |= edema

    labels[crack] = TISSUE_INDEX["hematoma"]
    masks["fracture"] |= crack
    masks["hematoma"] |= crack

    # Hematoma perióstico en la salida de la grieta
    px, pz = bx + ux * b.outer_radius, bz + uz * b.outer_radius
    soft = np.isin(labels, [TISSUE_INDEX["muscle"], TISSUE_INDEX["fat"]])
    # Elipse orientada tangente a la superficie del hueso
    t_ = (X - px) * uz - (Z - pz) * ux
    n_ = (X - px) * ux + (Z - pz) * uz
    hem = soft & ((t_ / (fr.hematoma_length / 2)) ** 2 + (n_ / fr.hematoma_thickness) ** 2 <= 1.0)
    labels[hem] = TISSUE_INDEX["hematoma"]
    masks["hematoma"] |= hem


# --------------------------------------------------------------------------
# Fracturas aleatorias (para datasets)
# --------------------------------------------------------------------------
FRACTURE_KINDS = ("transversal", "oblicua", "fisura", "conminuta", "incompleta", "desplazada")


def random_fracture(rng: np.random.Generator, kind: str | None = None, x_range: float = 8e-3) -> Fracture:
    """Fractura longitudinal aleatoria de un tipo dado (o al azar)."""
    kind = kind or str(rng.choice(FRACTURE_KINDS))
    base = Fracture(
        kind=kind,
        position=float(rng.uniform(-x_range, x_range)),
        angle_deg=float(rng.normal(0, 8)),
        gap=float(rng.uniform(0.25e-3, 0.8e-3)),
        roughness=float(rng.uniform(0.03e-3, 0.15e-3)),
        hematoma_thickness=float(rng.uniform(0.5e-3, 2e-3)),
        hematoma_length=float(rng.uniform(5e-3, 14e-3)),
        edema_length=float(rng.uniform(6e-3, 20e-3)),
    )
    if kind == "oblicua":
        return replace(base, angle_deg=float(rng.choice([-1, 1]) * rng.uniform(25, 50)))
    if kind == "fisura":
        return replace(base, gap=float(rng.uniform(0.05e-3, 0.2e-3)), hematoma_thickness=0.3e-3,
                       hematoma_length=4e-3, depth_fraction=float(rng.uniform(0.15, 0.4)))
    if kind == "incompleta":
        return replace(base, depth_fraction=float(rng.uniform(0.15, 0.6)))
    if kind == "desplazada":
        return replace(base, displacement=float(rng.choice([-1, 1]) * rng.uniform(0.8e-3, 3e-3)))
    if kind == "conminuta":
        extra = tuple((float(rng.uniform(-4e-3, 4e-3)), float(rng.uniform(-50, 50))) for _ in range(int(rng.integers(1, 4))))
        return replace(base, extra_cracks=extra)
    return base


def random_limb_spec(rng: np.random.Generator, preset: str = "tibia") -> LimbSpec:
    """Variación anatómica aleatoria (±20 % en grosores) alrededor de un preset."""
    spec = PRESETS[preset]

    def jit(v: float) -> float:
        return float(v * rng.uniform(0.8, 1.2))

    return replace(
        spec,
        skin=jit(spec.skin), fat=jit(spec.fat), muscle=jit(spec.muscle), cortex=jit(spec.cortex),
        medulla=jit(spec.medulla), muscle_far=jit(spec.muscle_far), bone_tilt_deg=float(rng.normal(0, 2)),
    )
