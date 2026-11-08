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

Además de las fracturas que abren la cortical que mira a la sonda (las que
ve bien la ecografía), hay fracturas **internas** que no rompen esa
superficie: intracorticales, de la cortical lejana, contusión ósea
trabecular y fracturas por estrés (ver ``Fracture``). Los **tipos de cuerpo**
(``BODY_TYPES``) escalan los grosores de piel, grasa, músculo y hueso.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np

from ..core.grid import Grid2D
from .phantom import TISSUE_INDEX, Phantom
from .roughness import RoughProfile

BONE_TISSUES = ("cortical_bone", "trabecular_bone", "marrow")
SOFT_TISSUES = ("muscle", "fat", "skin")
MEDULLARY_TISSUES = ("marrow", "trabecular_bone")

#: Radio de las bolsas de edema intertrabecular de la contusión ósea (m). Del
#: orden de la separación entre trabéculas (0.5-1 mm): el edema ocupa los
#: espacios medulares sin destruir el armazón trabecular.
EDEMA_POCKET_RADIUS = 0.4e-3


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

    Fracturas internas (los valores por defecto reproducen exactamente las
    fracturas anteriores, que siempre empiezan en la cortical cercana):

    - ``span``: tramo ``(ζ0, ζ1)`` que ocupa la grieta en la coordenada de
      capas del hueso ζ, que vale 0 en el periostio cercano, 1 en el endostio
      cercano, 2 en el endostio lejano y 3 en el periostio lejano (lineal
      dentro de cada capa y siguiendo la rugosidad local). Si se da, sustituye
      a ``depth_fraction``. Ejemplos: ``(0.2, 0.8)`` grieta intracortical que
      no llega ni al periostio ni al endostio; ``(0.3, 1.0)`` intracortical
      abierta a la médula; ``(2.0, 3.0)`` sólo la cortical lejana.
    - ``min_gap_cells``: anchura mínima de la grieta en celdas. 0 dibuja la
      anchura física exacta (una grieta más fina que ``dx`` queda punteada o
      desaparece); 1 garantiza una línea continua de al menos una celda, que
      es lo más fino que la malla puede representar (sobrestima la anchura,
      pero conserva el desacoplo de la cizalla a través de la grieta).
    - ``gap = 0`` significa que no hay grieta principal (contusión ósea).
    - ``edema_depth``: si se da, el edema es una semielipse que entra en la
      médula desde el endostio más próximo a la grieta, con esta profundidad
      (m) y anchura ``edema_length``. ``None`` = elipse centrada en la médula.
    - ``edema_fraction``: fracción del hueso trabecular de la zona de edema
      que pasa a edema (bolsas de ``EDEMA_POCKET_RADIUS``). 1 = todo (como
      antes); en una contusión ósea real el armazón trabecular sigue ahí y
      sólo cambia el contenido de los espacios medulares. La médula amarilla
      (sin trabéculas) se convierte siempre entera.
    - ``microcracks``, ``microcrack_length``, ``microcrack_gap``:
      microfracturas trabeculares (segmentos rellenos de sangre) dentro de la
      zona de edema, sólo en la médula: es la "contusión ósea" (*bone
      bruise*) que no rompe la cortical. En una diáfisis con médula amarilla
      no hay trabéculas: la misma geometría representa focos hemorrágicos.
    - ``periosteal_reaction`` / ``endosteal_reaction``: grosor máximo (m) de
      la reacción perióstica (hueso nuevo inmaduro, modelado como trabecular,
      con un ribete de edema perióstico) o endóstica (engrosamiento
      esclerótico hacia la médula, modelado como cortical), típicas de la
      fractura por estrés. ``reaction_length``: su extensión axial. Se ponen
      en el lado (cercano o lejano) donde está la grieta.

    El hematoma perióstico sólo se forma en las superficies que la grieta
    atraviesa: una grieta intracortical no lo tiene.
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

    @property
    def far_side(self) -> bool:
        """True si la grieta está en la cortical lejana (la que no mira a la sonda)."""
        return self.span is not None and 0.5 * (self.span[0] + self.span[1]) > 1.5

    @property
    def opens_near_surface(self) -> bool:
        """True si la grieta llega al periostio cercano (visible para la ecografía de reflexión)."""
        if self.gap <= 0:
            return False
        return self.span[0] <= 0.0 if self.span is not None else True


@dataclass
class LimbSpec:
    """Capas de un corte longitudinal de extremidad (grosores en metros).

    Las rugosidades son pares ``(rms, longitud_de_correlación)``.
    ``vessels`` son vasos sanguíneos ``(x, profundidad_bajo_la_piel, radio)``.
    ``trabecular``: grosor de hueso trabecular que tapiza cada endostio por
    dentro de la médula (0 en la diáfisis; grande en la metáfisis, donde la
    cavidad medular está llena de trabéculas). ``body``: tipo de cuerpo con el
    que se escaló el preset (ver ``with_body_type``).
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
    trabecular: float = 0.0
    body: str = "normal"

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
    # Metáfisis proximal de la tibia (unos 3-5 cm bajo la meseta tibial), donde
    # se producen las contusiones óseas y muchas fracturas ocultas: cortical
    # fina (1-2 mm) y cavidad medular de unos 35 mm (anteroposterior) llena de
    # hueso trabecular, con algo de médula en el centro. Por detrás, poplíteo y
    # gemelos. Grosores aproximados (anatomía de TAC/RM de rodilla).
    "tibia_metafisis": LimbSpec(
        name="tibia_metafisis", fat=4e-3, cortex=1.5e-3, medulla=34e-3, trabecular=13e-3, muscle_far=28e-3,
        vessels=((6e-3, 48e-3, 2e-3),),
    ),
}


# --------------------------------------------------------------------------
# Tipos de cuerpo
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BodyType:
    """Escalas de grosor respecto al adulto de peso normal de los presets.

    - ``skin``, ``fat``, ``muscle``: multiplican el grosor de la piel, la
      grasa subcutánea y el músculo (en el corte transversal, el grosor del
      compartimento muscular alrededor de los huesos).
    - ``cortex``: multiplica el grosor de la cortical.
    - ``bone``: multiplica el diámetro exterior del hueso (periostio). La
      médula ocupa lo que queda: con ``cortex < 1`` y ``bone ≈ 1`` la
      cortical se adelgaza por dentro (resorción endóstica), como en la
      osteoporosis.
    """

    name: str = "normal"
    skin: float = 1.0
    fat: float = 1.0
    muscle: float = 1.0
    cortex: float = 1.0
    bone: float = 1.0


#: Tipos de cuerpo con nombre. Los factores son estimaciones a partir de
#: rangos publicados; la variabilidad entre personas, sexos y zonas es grande
#: y los valores concretos son aproximados (orden de ±30 % en cada factor):
#:
#: - Grasa subcutánea (ecografía estandarizada: Störchle et al., Ultrasound
#:   Med Biol 43:427, 2017; Müller et al., Br J Sports Med 50:45, 2016): en
#:   muslo y pantorrilla, ~1-4 mm en deportistas y personas delgadas, ~4-15 mm
#:   en adultos de peso normal (las mujeres, 1.5-2 veces más que los varones)
#:   y ~15-40 mm en la obesidad. De ahí 0.45 (delgado), 1.7 (sobrepeso, IMC
#:   25-30) y 3 (obeso, IMC 30-40). Sobre la cara anteromedial de la tibia la
#:   grasa es más fina que en el muslo; se aplica el mismo factor.
#: - Músculo (espesor muscular por ecografía: Abe et al., J Sports Sci Med
#:   10:145, 2011, 1507 adultos de 20 a 95 años): el muslo anterior pierde
#:   ~25-35 % entre los 20 y los 70-80 años y la pierna posterior ~10-20 %
#:   (sarcopenia) → 0.7. Deportistas de fuerza o velocidad: +20-40 % → 1.35.
#:   En la obesidad la masa muscular de las piernas es algo mayor (soportan
#:   más peso): +5-15 % (valor incierto) → 1.1.
#: - Piel (ecografía de alta frecuencia, p. ej. Laurent et al., Vaccine
#:   25:6423, 2007): 1-2 mm en las extremidades; la dermis se adelgaza ~20 %
#:   en la vejez y es algo más gruesa con IMC alto (~+10-15 %, incierto).
#: - Hueso (QCT poblacional: Riggs et al., J Bone Miner Res 19:1945, 2004):
#:   entre los 20 y los 90 años la cortical pierde ~25-35 % de grosor en las
#:   mujeres por resorción endóstica, mientras el diámetro exterior crece algo
#:   (~3-5 %, aposición perióstica). Para una osteoporosis se toma 0.6 (algo
#:   más que la media de la edad). Deportistas de impacto (Haapasalo et al.,
#:   Bone 27:351, 2000; Nikander et al., J Bone Miner Res 20:520, 2005): área
#:   cortical +10-25 % → grosor 1.15 y diámetro 1.05.
#:
#: No se modela (las propiedades de los tejidos son fijas): la porosidad
#: cortical de la osteoporosis (del 5 % al 15-30 %, que baja la velocidad del
#: sonido en la cortical), la pérdida de densidad trabecular ni la
#: infiltración grasa del músculo en ancianos y obesos.
BODY_TYPES: dict[str, BodyType] = {
    b.name: b
    for b in [
        BodyType("delgado", skin=0.9, fat=0.45, muscle=0.85, cortex=0.95, bone=0.97),
        BodyType("normal"),
        BodyType("sobrepeso", skin=1.05, fat=1.7, muscle=1.05, cortex=1.0, bone=1.02),
        BodyType("obeso", skin=1.15, fat=3.0, muscle=1.1, cortex=1.05, bone=1.03),
        BodyType("musculoso", skin=1.0, fat=0.6, muscle=1.35, cortex=1.15, bone=1.05),
        BodyType("anciano_osteoporotico", skin=0.8, fat=0.9, muscle=0.7, cortex=0.6, bone=1.03),
    ]
}

#: Médula mínima que se conserva al escalar el hueso (m), para que una
#: cortical muy engrosada no cierre la cavidad medular.
_MIN_MEDULLA = 0.5e-3


def _body(body: str | BodyType) -> BodyType:
    return BODY_TYPES[body] if isinstance(body, str) else body


def with_body_type(spec: str | LimbSpec | CrossSectionSpec, body: str | BodyType) -> LimbSpec | CrossSectionSpec:
    """Escala un corte (longitudinal o transversal) al tipo de cuerpo ``body``.

    Las escalas se aplican sobre el ``spec`` dado, que se supone de un adulto
    normal (como los presets). Los vasos se recolocan para que sigan dentro
    del mismo tejido.
    """
    if isinstance(spec, str):
        spec = PRESETS[spec]
    b = _body(body)
    if isinstance(spec, CrossSectionSpec):
        return _cross_with_body(spec, b)
    near, far = spec.cortex, spec.cortex if spec.far_cortex is None else spec.far_cortex
    # médula = diámetro exterior escalado − corticales escaladas, escrito como incremento
    # para que el tipo "normal" (factores 1) reproduzca el preset bit a bit
    medulla = spec.medulla + (b.bone - 1.0) * (near + spec.medulla + far) + (1.0 - b.cortex) * (near + far)
    medulla = max(medulla, _MIN_MEDULLA)
    near, far = near * b.cortex, far * b.cortex

    def scaled(v: float | None, k: float) -> float | None:
        return None if v is None else v * k

    new = replace(
        spec, skin=spec.skin * b.skin, fat=spec.fat * b.fat, muscle=spec.muscle * b.muscle, cortex=near,
        medulla=medulla, far_cortex=None if spec.far_cortex is None else far,
        muscle_far=spec.muscle_far * b.muscle, fat_far=scaled(spec.fat_far, b.fat),
        skin_far=scaled(spec.skin_far, b.skin), trabecular=min(spec.trabecular, 0.5 * medulla), body=b.name,
    )
    return replace(new, vessels=_remap_vessels(spec, new))


def random_body_type(rng: np.random.Generator, names: Sequence[str] | None = None,
                     spread: float = 0.1) -> BodyType:
    """Tipo de cuerpo al azar entre ``names`` (todos por defecto) con variación continua.

    Cada factor se multiplica por un término log-normal de desviación
    ``spread`` (≈10 %), para que dos "obesos" no tengan exactamente los mismos
    grosores. El nombre se conserva como etiqueta.
    """
    names = tuple(names) if names is not None else tuple(BODY_TYPES)
    b = BODY_TYPES[str(rng.choice(names))]
    k = np.exp(rng.normal(0.0, spread, 5))
    return BodyType(b.name, skin=b.skin * k[0], fat=b.fat * k[1], muscle=b.muscle * k[2], cortex=b.cortex * k[3],
                    bone=b.bone * (1.0 + 0.3 * (k[4] - 1.0)))


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


def _limb_labels(Z: np.ndarray, zs: np.ndarray, tissues: list[str], spec: LimbSpec) -> np.ndarray:
    """Etiquetas de las capas más el hueso trabecular que tapiza los endostios."""
    labels = _stack_labels(Z, zs, tissues, spec.background)
    if spec.trabecular > 0:
        med = labels == TISSUE_INDEX[spec.medullary]
        lining = (Z < zs[4][None, :] + spec.trabecular) | (Z >= zs[5][None, :] - spec.trabecular)
        labels[med & lining] = TISSUE_INDEX["trabecular_bone"]
    return labels


def _layer_coordinate(Z: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Coordenada de capas del hueso ζ (ver ``Fracture.span``) en cada celda."""

    def frac(a, b):
        return (Z - a[None, :]) / np.maximum(b - a, 1e-9)[None, :]

    z4, z5 = zs[4][None, :], zs[5][None, :]
    return np.where(Z < z4, frac(zs[3], zs[4]), np.where(Z < z5, 1.0 + frac(zs[4], zs[5]), 2.0 + frac(zs[5], zs[6])))


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
    body: str | BodyType | None = None,
) -> Phantom:
    """Corte longitudinal de una extremidad, centrado en x = 0.

    El eje del hueso es horizontal (x) y la sonda estaría arriba (z = 0).
    ``depth`` por defecto abarca la extremidad completa más un margen.
    ``body``: tipo de cuerpo (nombre de ``BODY_TYPES`` o ``BodyType``) que
    escala los grosores del preset.
    """
    spec = PRESETS[preset] if isinstance(preset, str) else preset
    if body is not None:
        spec = with_body_type(spec, body)
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

    labels = _limb_labels(Z, zs, tissues, spec)
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
        "body_type": spec.body,
        "bone_top_depth": float(np.interp(0.0, x, zs[3])),
        "bone_bottom_depth": float(np.interp(0.0, x, zs[6])),
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
        labels_d = _limb_labels(Z, zs_d, tissues, spec)
    else:
        zs_d, labels_d = zs, labels

    bone_idx = [TISSUE_INDEX[t] for t in BONE_TISSUES]
    cort = TISSUE_INDEX["cortical_bone"]
    bone_any = np.isin(labels, bone_idx) | np.isin(labels_d, bone_idx)
    cortical_any = (labels == cort) | (labels_d == cort)
    if fr.span is None:
        depth_ok = Z <= z_top + fr.depth_fraction * (z_bottom - z_top) + max(fr.displacement, 0.0)
    else:
        zeta = np.where(dist > 0, _layer_coordinate(Z, zs_d), _layer_coordinate(Z, zs))
        depth_ok = (zeta >= fr.span[0]) & (zeta <= fr.span[1])
    out = np.where(dist > 0, labels_d, labels)

    crack = np.zeros(grid.shape, dtype=bool)
    if fr.gap > 0:
        half_gap = 0.5 * max(fr.gap, fr.min_gap_cells * grid.dx)
        crack = (np.abs(dist) <= half_gap) & bone_any & depth_ok
        for off, ang in fr.extra_cracks:
            d2 = signed_distance(xc + off, np.radians(ang))
            crack |= (np.abs(d2) <= half_gap) & bone_any & depth_ok

    # Edema en la médula alrededor del foco (antes de abrir la grieta). Sin
    # ``edema_depth`` ocupa todo el grosor de la médula; con él, entra en la
    # médula desde el endostio más próximo a la grieta.
    far = fr.far_side
    z_endo = float(np.interp(xc, x, zs[5 if far else 4]))
    if fr.edema_depth is None:
        z_med = 0.5 * (float(np.interp(xc, x, zs[4])) + float(np.interp(xc, x, zs[5])))
        zone_geom = ((xc, z_med), (fr.edema_length / 2, max(0.5 * (z_bottom - z_top), 1e-4)), 0)
    else:
        zone_geom = ((xc, z_endo), (fr.edema_length / 2, fr.edema_depth), -1 if far else 1)
    zone = np.zeros(grid.shape, dtype=bool)
    if fr.edema_length > 0:
        zone = _in_ellipse(X, Z, *zone_geom[:2]) & np.isin(out, [TISSUE_INDEX[t] for t in MEDULLARY_TISSUES])
        edema = zone
        if fr.edema_fraction < 1.0:
            pockets = _edema_pockets(X, Z, *zone_geom, fr.edema_fraction, rng)
            edema = zone & ((out != TISSUE_INDEX["trabecular_bone"]) | pockets)
        out[edema] = TISSUE_INDEX["edema"]
        masks["edema"] |= edema

    _add_reactions(X, Z, out, masks, fr, xc, z_bottom if far else z_top, z_endo)

    hem_idx = TISSUE_INDEX["hematoma"]
    out[crack] = hem_idx
    masks["fracture"] |= crack & cortical_any
    masks["hematoma"] |= crack

    # Microfracturas trabeculares: sólo dentro de la médula, sin tocar la cortical
    if fr.microcracks > 0 and zone.any():
        zeta0 = _layer_coordinate(Z, zs)
        inside = np.isin(out, [TISSUE_INDEX[t] for t in (*MEDULLARY_TISSUES, "edema")]) & (zeta0 > 1) & (zeta0 < 2)
        micro = _microcracks(X, Z, *zone_geom, fr, grid.dx, rng) & inside
        out[micro] = hem_idx
        masks["fracture"] |= micro
        masks["hematoma"] |= micro

    # Hematoma perióstico en las superficies que la grieta atraviesa
    if fr.gap > 0 and fr.hematoma_thickness > 0 and fr.hematoma_length > 0:
        soft = np.isin(out, [TISSUE_INDEX[t] for t in ("muscle", "fat")])
        if fr.span is None:
            surfaces = [z_top] + ([z_bottom] if fr.depth_fraction >= 1.0 else [])
        else:
            surfaces = ([z_top] if fr.span[0] <= 0 else []) + ([z_bottom] if fr.span[1] >= 3 else [])
        for zsurf in surfaces:
            ell = ((X - xc) / (fr.hematoma_length / 2)) ** 2 + ((Z - zsurf) / fr.hematoma_thickness) ** 2 <= 1.0
            hem = ell & soft
            out[hem] = hem_idx
            masks["hematoma"] |= hem
    return out


def _in_ellipse(X, Z, center, semi) -> np.ndarray:
    (cx, cz), (ax, az) = center, semi
    return ((X - cx) / ax) ** 2 + ((Z - cz) / az) ** 2 <= 1.0


def _sample_in_ellipse(rng, n: int, center, semi, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Puntos uniformes en la elipse (``direction`` = 0) o en su mitad z > cz (+1) o z < cz (-1).

    Se muestrea en coordenadas continuas para que la geometría no dependa de la malla.
    """
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    phi = rng.uniform(0.0, np.pi if direction else 2 * np.pi, n)
    (cx, cz), (ax, az) = center, semi
    return cx + ax * r * np.cos(phi), cz + (direction or 1) * az * r * np.sin(phi)


def _edema_pockets(X, Z, center, semi, direction, fraction: float, rng) -> np.ndarray:
    """Bolsas circulares de edema que cubren en promedio ``fraction`` de la zona.

    Modelo booleano de Poisson: con ``n`` discos de radio r repartidos al azar
    en un área A, la fracción cubierta es 1 - exp(-n·π·r²/A); así los solapes
    no reducen la fracción pedida.
    """
    out = np.zeros(X.shape, dtype=bool)
    if fraction <= 0:
        return out
    r = EDEMA_POCKET_RADIUS
    area = np.pi * semi[0] * semi[1] * (0.5 if direction else 1.0)
    n = int(round(-np.log1p(-min(fraction, 0.99)) * area / (np.pi * r**2)))
    for px, pz in zip(*_sample_in_ellipse(rng, n, center, semi, direction)):
        out |= (X - px) ** 2 + (Z - pz) ** 2 <= r**2
    return out


def _microcracks(X, Z, center, semi, direction, fr, dx: float, rng) -> np.ndarray:
    """Segmentos finos de orientación aleatoria con centro dentro de la zona de edema."""
    out = np.zeros(X.shape, dtype=bool)
    half_w = 0.5 * max(fr.microcrack_gap, fr.min_gap_cells * dx)
    cxs, czs = _sample_in_ellipse(rng, fr.microcracks, center, semi, direction)
    angles = rng.uniform(0.0, np.pi, fr.microcracks)
    lengths = fr.microcrack_length * rng.uniform(0.5, 1.5, fr.microcracks)
    for cx, cz, a, length in zip(cxs, czs, angles, lengths):
        along = (X - cx) * np.cos(a) + (Z - cz) * np.sin(a)
        across = (Z - cz) * np.cos(a) - (X - cx) * np.sin(a)
        out |= (np.abs(along) <= length / 2) & (np.abs(across) <= half_w)
    return out


def _add_reactions(X, Z, out, masks, fr, xc: float, z_periosteum: float, z_endosteum: float) -> None:
    """Reacción perióstica (hueso nuevo + edema perióstico) y endóstica (esclerosis)."""
    half_len = fr.reaction_length / 2
    if fr.endosteal_reaction > 0:
        med = np.isin(out, [TISSUE_INDEX[t] for t in (*MEDULLARY_TISSUES, "edema")])
        lens = med & _in_ellipse(X, Z, (xc, z_endosteum), (half_len, fr.endosteal_reaction))
        out[lens] = TISSUE_INDEX["cortical_bone"]
        masks["edema"] &= ~lens
    if fr.periosteal_reaction > 0:
        soft_idx = [TISSUE_INDEX[t] for t in ("muscle", "fat")]
        lens = np.isin(out, soft_idx) & _in_ellipse(X, Z, (xc, z_periosteum), (half_len, fr.periosteal_reaction))
        out[lens] = TISSUE_INDEX["trabecular_bone"]
        # Ribete de edema perióstico de ~0.5 mm alrededor del hueso nuevo
        rim = np.isin(out, soft_idx) & _in_ellipse(X, Z, (xc, z_periosteum),
                                                   (half_len + 2e-3, fr.periosteal_reaction + 0.5e-3))
        out[rim] = TISSUE_INDEX["edema"]
        masks["edema"] |= rim


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
    body: str = "normal"


def _cross_with_body(spec: CrossSectionSpec, b: BodyType) -> CrossSectionSpec:
    """Tipo de cuerpo en el corte transversal.

    El músculo es el compartimento entre la grasa y los huesos: su grosor
    (radio exterior del músculo menos el radio equivalente de los huesos) se
    escala con ``b.muscle``. Cada hueso conserva su posición relativa dentro
    del compartimento, de modo que una tibia subcutánea sigue bajo la piel y
    lo que crece o mengua es el músculo de detrás.
    """
    r_m = spec.radius - spec.skin - spec.fat
    r_core = float(np.sqrt(sum(bn.outer_radius**2 for bn in spec.bones)))
    r_m_new = r_core * b.bone + b.muscle * (r_m - r_core)
    bones = []
    for bn in spec.bones:
        ro = bn.outer_radius * b.bone
        free = r_m - bn.outer_radius
        k = (r_m_new - ro) / free if free > 0 else r_m_new / r_m
        bones.append(replace(bn, center=(bn.center[0] * k, bn.center[1] * k), outer_radius=ro,
                             cortex=min(bn.cortex * b.cortex, ro - _MIN_MEDULLA)))
    s = r_m_new / r_m
    skin, fat = spec.skin * b.skin, spec.fat * b.fat
    return replace(spec, radius=r_m_new + fat + skin, skin=skin, fat=fat, bones=tuple(bones),
                   vessels=tuple((vx * s, vz * s, r) for vx, vz, r in spec.vessels), body=b.name)


@dataclass
class RingFracture:
    """Fractura en un corte transversal: grieta radial a través de la cortical.

    ``angle_deg`` es la dirección de la grieta desde el centro del hueso
    (0 = hacia arriba, hacia la sonda; positivo en sentido horario; 180 = la
    cortical lejana, en la sombra acústica).

    Fracturas internas (los valores por defecto reproducen las anteriores):

    - ``span``: tramo ``(u0, u1)`` de la cortical que ocupa la grieta, en
      fracción del grosor cortical desde el periostio (0) hacia el endostio
      (1). ``None`` = todo el grosor.
    - ``gap = 0``: sin grieta principal (contusión ósea).
    - ``min_gap_cells``, ``edema_fraction``, ``microcracks``,
      ``microcrack_length``, ``microcrack_gap``, ``periosteal_reaction``,
      ``endosteal_reaction``, ``reaction_length``: como en ``Fracture``. El
      edema (y las microfracturas) ocupan un disco de diámetro
      ``edema_length`` junto al endostio, en la dirección de la grieta.
    """

    kind: str = "longitudinal/espiroidea"
    angle_deg: float = 0.0
    gap: float = 0.4e-3
    roughness: float = 0.08e-3
    hematoma_thickness: float = 1.2e-3
    hematoma_length: float = 8e-3
    edema_length: float = 8e-3
    bone_index: int = 0
    span: tuple[float, float] | None = None
    min_gap_cells: float = 0.0
    edema_fraction: float = 1.0
    microcracks: int = 0
    microcrack_length: float = 1.5e-3
    microcrack_gap: float = 0.08e-3
    periosteal_reaction: float = 0.0
    endosteal_reaction: float = 0.0
    reaction_length: float = 12e-3


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
    body: str | BodyType | None = None,
) -> Phantom:
    """Sección axial de una extremidad centrada en (0, 0).

    ``body``: tipo de cuerpo (nombre de ``BODY_TYPES`` o ``BodyType``).
    """
    spec = CROSS_PRESETS[preset] if isinstance(preset, str) else preset
    if body is not None:
        spec = with_body_type(spec, body)
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
        _apply_ring_fracture(X, Z, spec, fracture, labels, masks, rng, dx)

    het = np.random.default_rng(seed + 7919).standard_normal(grid.shape).astype(np.float32)
    meta = {
        "view": "cross_section",
        "preset": spec.name,
        "spec": asdict(spec),
        "fracture": asdict(fracture) if fracture is not None else None,
        "seed": seed,
        "label": int(fracture is not None),
        "body_type": spec.body,
    }
    return Phantom(grid, labels, masks, het, meta)


def _tangent_ellipse(X, Z, point, normal, semi) -> np.ndarray:
    """Elipse con el semieje ``semi[0]`` tangente a la superficie y ``semi[1]`` según la normal."""
    (px, pz), (ux, uz) = point, normal
    t_ = (X - px) * uz - (Z - pz) * ux
    n_ = (X - px) * ux + (Z - pz) * uz
    return (t_ / semi[0]) ** 2 + (n_ / semi[1]) ** 2 <= 1.0


def _apply_ring_fracture(X, Z, spec: CrossSectionSpec, fr: RingFracture, labels, masks, rng, dx: float):
    b = spec.bones[fr.bone_index]
    bx, bz = b.center
    phi = np.radians(fr.angle_deg)
    ux, uz = np.sin(phi), -np.cos(phi)  # dirección de la grieta
    along = (X - bx) * ux + (Z - bz) * uz
    across = (X - bx) * uz - (Z - bz) * ux
    prof = RoughProfile(2 * b.outer_radius, fr.roughness, max(fr.gap, 0.3e-3), rng)
    dist = across - prof(along + b.outer_radius)
    cort = labels == TISSUE_INDEX["cortical_bone"]
    rb = np.hypot(X - bx, Z - bz)
    near_bone = rb <= b.outer_radius + 1e-3
    crack = np.zeros(X.shape, dtype=bool)
    if fr.gap > 0:
        crack = (np.abs(dist) <= 0.5 * max(fr.gap, fr.min_gap_cells * dx)) & (along > 0) & cort & near_bone
        if fr.span is not None:
            u = (b.outer_radius - rb) / b.cortex
            crack &= (u >= fr.span[0]) & (u <= fr.span[1])

    # Edema en la médula junto al punto donde la grieta atraviesa el endostio
    ex, ez = bx + ux * (b.outer_radius - b.cortex), bz + uz * (b.outer_radius - b.cortex)
    med = np.isin(labels, [TISSUE_INDEX[t] for t in MEDULLARY_TISSUES])
    zone = med & (np.hypot(X - ex, Z - ez) <= fr.edema_length / 2)
    edema = zone
    if fr.edema_fraction < 1.0:
        semi = (fr.edema_length / 2, fr.edema_length / 2)
        pockets = _edema_pockets(X, Z, (ex, ez), semi, 0, fr.edema_fraction, rng)
        edema = zone & ((labels != TISSUE_INDEX["trabecular_bone"]) | pockets)
    labels[edema] = TISSUE_INDEX["edema"]
    masks["edema"] |= edema

    px, pz = bx + ux * b.outer_radius, bz + uz * b.outer_radius
    soft_idx = [TISSUE_INDEX["muscle"], TISSUE_INDEX["fat"]]
    if fr.endosteal_reaction > 0:
        med_now = np.isin(labels, [TISSUE_INDEX[t] for t in (*MEDULLARY_TISSUES, "edema")])
        lens = med_now & _tangent_ellipse(X, Z, (ex, ez), (ux, uz), (fr.reaction_length / 2, fr.endosteal_reaction))
        labels[lens] = TISSUE_INDEX["cortical_bone"]
        masks["edema"] &= ~lens
    if fr.periosteal_reaction > 0:
        lens = np.isin(labels, soft_idx) & _tangent_ellipse(X, Z, (px, pz), (ux, uz),
                                                            (fr.reaction_length / 2, fr.periosteal_reaction))
        labels[lens] = TISSUE_INDEX["trabecular_bone"]
        rim = np.isin(labels, soft_idx) & _tangent_ellipse(
            X, Z, (px, pz), (ux, uz), (fr.reaction_length / 2 + 2e-3, fr.periosteal_reaction + 0.5e-3))
        labels[rim] = TISSUE_INDEX["edema"]
        masks["edema"] |= rim

    labels[crack] = TISSUE_INDEX["hematoma"]
    masks["fracture"] |= crack
    masks["hematoma"] |= crack

    if fr.microcracks > 0 and zone.any():
        inside = np.isin(labels, [TISSUE_INDEX[t] for t in (*MEDULLARY_TISSUES, "edema")])
        inside &= rb < b.outer_radius - 0.5 * b.cortex
        semi = (fr.edema_length / 2, fr.edema_length / 2)
        micro = _microcracks(X, Z, (ex, ez), semi, 0, fr, dx, rng) & inside
        labels[micro] = TISSUE_INDEX["hematoma"]
        masks["fracture"] |= micro
        masks["hematoma"] |= micro

    # Hematoma perióstico en la salida de la grieta, si la grieta llega al periostio
    reaches_periosteum = fr.span is None or fr.span[0] <= 0
    if fr.gap > 0 and reaches_periosteum and fr.hematoma_thickness > 0 and fr.hematoma_length > 0:
        soft = np.isin(labels, soft_idx)
        hem = soft & _tangent_ellipse(X, Z, (px, pz), (ux, uz), (fr.hematoma_length / 2, fr.hematoma_thickness))
        labels[hem] = TISSUE_INDEX["hematoma"]
        masks["hematoma"] |= hem


# --------------------------------------------------------------------------
# Fracturas aleatorias (para datasets)
# --------------------------------------------------------------------------
#: Fracturas que abren la cortical cercana (la que mira a la sonda).
CORTICAL_FRACTURE_KINDS = ("transversal", "oblicua", "fisura", "conminuta", "incompleta", "desplazada")
#: Fracturas internas, que no rompen la superficie del hueso que ve la
#: ecografía de reflexión (salvo la variante perióstica de la de estrés, cuya
#: reacción sí asoma a la superficie).
INTERNAL_FRACTURE_KINDS = ("intracortical", "cortical_lejana", "contusion_osea", "estres")
FRACTURE_KINDS = CORTICAL_FRACTURE_KINDS + INTERNAL_FRACTURE_KINDS
RING_FRACTURE_KINDS = ("longitudinal/espiroidea",) + INTERNAL_FRACTURE_KINDS


def random_fracture(rng: np.random.Generator, kind: str | None = None, x_range: float = 8e-3,
                    kinds: Sequence[str] = FRACTURE_KINDS) -> Fracture:
    """Fractura longitudinal aleatoria de un tipo dado (o al azar entre ``kinds``).

    Con ``kinds=CORTICAL_FRACTURE_KINDS`` se obtienen exactamente las mismas
    fracturas que antes de añadir las internas (misma semilla). Rangos de las
    internas:

    - intracortical: grieta de 0.08-0.3 mm desde el 15-45 % del grosor
      cortical hasta el endostio (la mitad de las veces, con edema de médula)
      o hasta el 50-90 % (cerrada, sin edema). Sin hematoma perióstico.
    - cortical_lejana: grieta de 0.15-0.6 mm desde el endostio lejano (o
      desde dentro de la cortical) hasta el periostio lejano, con su hematoma
      y edema junto al endostio lejano.
    - contusion_osea: 6-15 microfracturas de 0.4-3.8 mm y 0.05-0.12 mm de
      ancho en una zona subcortical de 8-25 × 4-10 mm (tamaños típicos de las
      contusiones óseas en RM), con un 15-35 % de edema intertrabecular. Sin
      grieta cortical ni hematoma.
    - estres: grieta capilar de 0.05-0.15 mm, transversal, que nace en el
      periostio y llega al 20-50 % de la cortical (con reacción perióstica de
      0.5-2 mm) o nace en el endostio y llega hasta el 50-80 % del grosor
      desde el periostio (con reacción endóstica de 0.5-2 mm); reacción de
      10-30 mm de longitud y edema de médula (grados 2-4 de Fredericson).
    """
    kind = kind or str(rng.choice(kinds))
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
    if kind in INTERNAL_FRACTURE_KINDS:
        return _internal_fracture(rng, base)
    return base


def _internal_fracture(rng: np.random.Generator, base: Fracture) -> Fracture:
    """Parámetros de las fracturas internas (ver ``random_fracture``)."""
    u = rng.uniform
    if base.kind == "intracortical":
        z0 = float(u(0.15, 0.45))
        opens_medulla = bool(rng.random() < 0.5)
        z1 = 1.0 if opens_medulla else float(u(z0 + 0.25, 0.9))
        return replace(base, span=(z0, z1), gap=float(u(0.08e-3, 0.3e-3)), angle_deg=float(rng.normal(0, 15)),
                       min_gap_cells=1.0, hematoma_thickness=0.0,
                       edema_length=float(u(4e-3, 12e-3)) if opens_medulla else 0.0, edema_depth=float(u(2e-3, 5e-3)))
    if base.kind == "cortical_lejana":
        return replace(base, span=(float(u(2.0, 2.5)), 3.0), gap=float(u(0.15e-3, 0.6e-3)), min_gap_cells=1.0,
                       edema_depth=float(u(3e-3, 6e-3)))
    if base.kind == "contusion_osea":
        return replace(base, gap=0.0, hematoma_thickness=0.0, min_gap_cells=1.0,
                       microcracks=int(rng.integers(6, 16)), microcrack_length=float(u(0.8e-3, 2.5e-3)),
                       microcrack_gap=float(u(0.05e-3, 0.12e-3)), edema_length=float(u(8e-3, 25e-3)),
                       edema_depth=float(u(4e-3, 10e-3)), edema_fraction=float(u(0.15, 0.35)))
    # estrés
    periosteal = bool(rng.random() < 0.5)
    common = {"gap": float(u(0.05e-3, 0.15e-3)), "min_gap_cells": 1.0, "angle_deg": float(rng.normal(0, 5)),
              "hematoma_thickness": 0.0, "reaction_length": float(u(10e-3, 30e-3)),
              "edema_length": float(u(10e-3, 30e-3)), "edema_depth": float(u(3e-3, 8e-3))}
    thickness = float(u(0.5e-3, 2e-3))
    if periosteal:
        return replace(base, span=(0.0, float(u(0.2, 0.5))), periosteal_reaction=thickness, **common)
    return replace(base, span=(float(u(0.5, 0.8)), 1.0), endosteal_reaction=thickness, **common)


def random_ring_fracture(rng: np.random.Generator, kind: str | None = None,
                         kinds: Sequence[str] = RING_FRACTURE_KINDS, bone_index: int = 0) -> RingFracture:
    """Fractura aleatoria para el corte transversal (mismos rangos que ``random_fracture``)."""
    kind = kind or str(rng.choice(kinds))
    u = rng.uniform
    base = RingFracture(kind=kind, angle_deg=float(u(-45, 45)), gap=float(u(0.25e-3, 0.8e-3)),
                        roughness=float(u(0.03e-3, 0.15e-3)), hematoma_thickness=float(u(0.5e-3, 2e-3)),
                        hematoma_length=float(u(5e-3, 12e-3)), edema_length=float(u(4e-3, 12e-3)),
                        bone_index=bone_index)
    if kind == "cortical_lejana":
        return replace(base, angle_deg=180.0 + float(u(-45, 45)), min_gap_cells=1.0)
    if kind == "intracortical":
        u0 = float(u(0.15, 0.45))
        opens_medulla = bool(rng.random() < 0.5)
        return replace(base, span=(u0, 1.0 if opens_medulla else float(u(u0 + 0.25, 0.9))),
                       gap=float(u(0.08e-3, 0.3e-3)), min_gap_cells=1.0, hematoma_thickness=0.0,
                       edema_length=base.edema_length if opens_medulla else 0.0)
    if kind == "contusion_osea":
        return replace(base, gap=0.0, hematoma_thickness=0.0, min_gap_cells=1.0,
                       microcracks=int(rng.integers(4, 11)), microcrack_length=float(u(0.8e-3, 2.5e-3)),
                       microcrack_gap=float(u(0.05e-3, 0.12e-3)), edema_fraction=float(u(0.15, 0.35)))
    if kind == "estres":
        thickness = float(u(0.5e-3, 2e-3))
        common = {"gap": float(u(0.05e-3, 0.15e-3)), "min_gap_cells": 1.0, "hematoma_thickness": 0.0,
                  "reaction_length": float(u(8e-3, 20e-3))}
        if rng.random() < 0.5:
            return replace(base, span=(0.0, float(u(0.2, 0.5))), periosteal_reaction=thickness, **common)
        return replace(base, span=(float(u(0.5, 0.8)), 1.0), endosteal_reaction=thickness, **common)
    return base


def _remap_vessels(old: LimbSpec, new: LimbSpec) -> tuple[tuple[float, float, float], ...]:
    """Recoloca los vasos para que conserven su posición relativa dentro de su capa."""
    c_old = np.concatenate([[0.0], np.cumsum([t for _, t in old.layers()])])
    c_new = np.concatenate([[0.0], np.cumsum([t for _, t in new.layers()])])
    return tuple((vx, float(np.interp(vd, c_old, c_new)), r) for vx, vd, r in old.vessels)


def random_limb_spec(rng: np.random.Generator, preset: str = "tibia",
                     body: str | BodyType | Sequence[str] | None = None, jitter: float = 0.1) -> LimbSpec:
    """Anatomía aleatoria: un tipo de cuerpo más la variación individual de cada capa.

    - ``body``: ``None`` (cualquiera de ``BODY_TYPES``), un nombre, una lista
      de nombres entre los que elegir o un ``BodyType`` concreto. Los tipos
      con nombre llevan además una variación continua (``random_body_type``).
    - ``jitter``: variación uniforme ±``jitter`` de cada grosor por separado
      (asimetrías individuales, sobre la variación del tipo de cuerpo).
    """
    if isinstance(body, BodyType):
        b = body
    else:
        b = random_body_type(rng, [body] if isinstance(body, str) else body)
    spec = with_body_type(PRESETS[preset], b)

    def jit(v: float) -> float:
        return float(v * rng.uniform(1.0 - jitter, 1.0 + jitter))

    medulla = jit(spec.medulla)
    out = replace(
        spec,
        skin=jit(spec.skin), fat=jit(spec.fat), muscle=jit(spec.muscle), cortex=jit(spec.cortex),
        medulla=medulla, muscle_far=jit(spec.muscle_far), bone_tilt_deg=float(rng.normal(0, 2)),
        trabecular=min(spec.trabecular, 0.5 * medulla),
    )
    return replace(out, vessels=_remap_vessels(spec, out))
