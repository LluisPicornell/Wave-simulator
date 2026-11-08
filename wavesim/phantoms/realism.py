"""Realismo de los fantomas: microestructura del hueso y estructuras del tejido blando.

Lo que añade el modo realista (``RealismSpec``) sobre el fantoma de capas:

1. **Porosidad cortical** (``CorticalPorosity``): canales de Havers y
   cavidades de resorción como elipses en coordenadas continuas, con un
   gradiente de porosidad desde el periostio hacia el endostio y, en el
   hueso osteoporótico, una zona endocortical trabecularizada.
2. **Red trabecular** (``TrabecularNetwork``): placas y barras como las
   paredes de una teselación de Voronoi anisótropa, con BV/TV, grosor
   trabecular (Tb.Th), orientación preferente y perforación de placas.
3. **Rugosidad multiescala** (``FractalRoughness``): sobre la ondulación
   gaussiana de siempre, una rugosidad fina autoafín (espectro de von
   Kármán) en periostio, endostio y superficies de fractura.
4. **Tejido blando** (``SoftTissueStructure``): fascia profunda entre la
   grasa y el músculo, fascia superficial dentro de la grasa, septos de los
   lobulillos grasos, perimisio del músculo orientado según las fibras y
   periostio sobre la cortical.

Enfoque multiescala. La geometría se define en coordenadas continuas (la
misma semilla da la misma anatomía a cualquier ``dx``) y se rasteriza
calculando qué fracción de cada celda cubre cada estructura:

- Un detalle que la malla resuelve (su menor dimensión mide al menos
  ``resolve_cells`` celdas) se dibuja explícito: poros y espacios
  trabeculares con porosidad 1 (fluido), matriz y trabéculas con 0.
- Uno más pequeño queda como **fracción de poro por celda** (medio
  efectivo). Cada poro conserva su área, de modo que la porosidad media es
  la misma en los dos casos y la fluctuación de celda a celda tiene la
  varianza que corresponde al tamaño de los poros (es lo que dispersa el
  ultrasonido en la aproximación de Born).
- Las láminas de colágeno (fascias, septos, periostio) son etiquetas: una
  celda pasa a ``fascia`` o ``periosteum`` con una probabilidad igual a la
  fracción que cubre la lámina (*dithering*). A malla fina las láminas salen
  continuas; a malla gruesa, punteadas, pero con la fracción de volumen
  correcta (un septo de 0.1 mm no desaparece en una celda de 0.25 mm).

Las etiquetas del hueso no cambian: un poro cortical sigue etiquetado como
``cortical_bone`` (con porosidad 1) y un espacio trabecular como
``trabecular_bone``. Así las máscaras de fractura, el edema y cualquier
análisis por etiquetas siguen valiendo; las propiedades salen del mapa de
porosidad (``Phantom.porosity`` y ``materials.porous``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.grid import Grid2D
from .roughness import RoughProfile

#: Subflujos aleatorios independientes del generador principal: el modo
#: realista no altera los números aleatorios de la anatomía básica (capas,
#: fractura), así que la tibia básica y la realista son el mismo "paciente".
_STREAMS = {"roughness": 1, "pores": 2, "trabecular": 3, "septa": 4, "dither": 5, "lobules": 6, "fascia": 7}


def stream(seed: int, name: str) -> np.random.Generator:
    """Generador del subflujo ``name`` para la semilla ``seed``."""
    return np.random.default_rng([int(seed), 0x5EA1, _STREAMS[name]])


# --------------------------------------------------------------------------
# Especificaciones
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CorticalPorosity:
    """Porosidad de la cortical (fracciones de volumen; longitudes en m).

    - ``porosity``: porosidad media de la cortical (Ct.Po), incluida la zona
      trabecularizada. Cortical diafisaria adulta sana: 3-8 % (Bousson et
      al., J Bone Miner Res 16:1308, 2001; Cooper et al., Bone 40:957,
      2007); osteoporótica, 10-30 %. Por defecto 5 %, la porosidad de
      referencia de las tablas (``materials.porous``): una cortical sana
      realista tiene así la misma densidad media que la básica.
    - ``endosteal_ratio``: porosidad junto al endostio / junto al periostio
      en la cortical compacta (gradiente lineal). La porosidad crece hacia
      el endostio, sobre todo con la edad (Bousson 2001); 2.5 es una
      estimación.
    - ``canal_diameter``, ``diameter_sigma``: mediana y desviación
      logarítmica del diámetro de los canales (log-normal). Canales de
      Havers de ~40-80 µm en el adulto joven; con la edad la porosidad crece
      sobre todo porque los poros se agrandan, no porque haya más (Thomas,
      Feik & Clement, J Anat 209:219, 2006).
    - ``canal_aspect``: alargamiento de la sección de los canales en el corte
      longitudinal. Los canales corren a lo largo del eje del hueso y el
      plano los corta como trazas de 6-11 diámetros, pero en una simulación
      2D toda estructura se extruye perpendicular al plano: esa traza sería
      una placa de fluido, no un cilindro, y dispersaría unas L/d veces más
      (con trazas de 8 diámetros la cortical sana perdía 11 dB en 4 mm a
      3 MHz). Para una onda que viaja en profundidad, que es la de la
      ecografía, un cilindro a lo largo del eje y uno perpendicular al plano
      dispersan igual, así que se usa una sección casi circular (1.5). En el
      corte transversal los canales son perpendiculares al plano y salen
      casi circulares de verdad (``CROSS_CANAL_ASPECT``).
    - ``trabecularized``: fracción interna del grosor cortical
      trabecularizada (la "zona de transición" endocortical de Zebaze et al.,
      Lancet 375:1729, 2010), con porosidad ``trabecularized_porosity`` en
      cavidades de resorción gigantes de ``cavity_diameter`` (> 385 µm en
      Bousson 2001) y alargamiento ``cavity_aspect`` (el mismo argumento 2D
      que para los canales).
    - ``fill``: tejido que llena los poros. Sangre en los canales
      vasculares del hueso sano; en el osteoporótico se supone médula grasa
      en las cavidades agrandadas (estimación).
    """

    porosity: float = 0.05
    endosteal_ratio: float = 2.5
    canal_diameter: float = 60e-6
    diameter_sigma: float = 0.4
    canal_aspect: float = 1.5
    trabecularized: float = 0.0
    trabecularized_porosity: float = 0.40
    cavity_diameter: float = 0.3e-3
    cavity_aspect: float = 1.5
    fill: str = "blood"

    def compact_porosity(self) -> tuple[float, float]:
        """Porosidad de la cortical compacta junto al periostio y en el borde interno de la compacta.

        Resuelve ``(1-τ)·media(φ_c) + τ·φ_t = porosity`` con un gradiente
        lineal de razón ``endosteal_ratio`` entre ζ = 0 (periostio) y ζ = 1.
        """
        tau, phi_t, r = self.trabecularized, self.trabecularized_porosity, self.endosteal_ratio
        compact = (1.0 - tau) * (1.0 + 0.5 * (r - 1.0) * (1.0 - tau))
        phi_p = (self.porosity - tau * phi_t) / compact
        if phi_p < 0:
            raise ValueError("la zona trabecularizada ya supera la porosidad media pedida")
        return phi_p, phi_p * (1.0 + (r - 1.0) * (1.0 - tau))


@dataclass(frozen=True)
class TrabecularNetwork:
    """Red trabecular explícita (longitudes en m).

    - ``bvtv``: fracción de volumen óseo. Metáfisis tibial adulta ~0.10-0.25
      (Ding et al., J Orthop Res 20:615, 2002; Hildebrand et al., J Bone
      Miner Res 14:1167, 1999). 0.20 por defecto = la BV/TV de referencia de
      las tablas.
    - ``thickness``: grosor trabecular Tb.Th (100-200 µm; Hildebrand 1999).
      La separación sale del modelo de placas de Parfitt,
      ``Tb.Sp = Tb.Th (1/BV/TV - 1)``: 0.6 mm con los valores por defecto
      (rango medido 0.5-1.5 mm).
    - ``anisotropy``: alargamiento de los espacios medulares según la
      dirección preferente (grado de anisotropía; 1.3-2 en la tibia
      proximal). ``orientation_deg``: esa dirección respecto al eje del
      hueso (x en el corte longitudinal); las trabéculas principales de la
      metáfisis transmiten la carga de la meseta a la diáfisis y corren a lo
      largo del eje.
    - ``perforation``: fracción de placas (paredes) perforadas o perdidas,
      que deja extremos libres (barras). Crece con la osteoporosis.
    """

    bvtv: float = 0.20
    thickness: float = 0.15e-3
    anisotropy: float = 1.6
    orientation_deg: float = 0.0
    perforation: float = 0.15

    @property
    def separation(self) -> float:
        """Tb.Sp del modelo de placas (m)."""
        return self.thickness * (1.0 / self.bvtv - 1.0)


@dataclass(frozen=True)
class FractalRoughness:
    """Rugosidad fina autoafín ``(rms [m], ξ [m], H)`` que se suma a la ondulación gaussiana.

    Espectro de von Kármán (``RoughProfile(spectrum="power_law")``): rugosidad
    en todas las escalas por debajo de ξ hasta 20 µm.

    - ``periosteal``: superficie perióstica casi lisa salvo inserciones
      fibrosas y forámenes vasculares; 12 µm rms (estimación).
    - ``endosteal``: lagunas de resorción de ~40-60 µm de profundidad
      (Eriksen, Endocr Rev 7:379, 1986) → 40 µm rms, H = 0.5 (más rugosa a
      escala fina). Crece con la trabecularización (``scaled``).
    - ``fracture``: rms *relativa* a ``Fracture.roughness``; H = 0.8, el
      exponente de rugosidad casi universal de las superficies de fractura
      (Bouchaud 1997; se supone válido para el hueso).
    """

    periosteal: tuple[float, float, float] = (12e-6, 0.3e-3, 0.7)
    endosteal: tuple[float, float, float] = (40e-6, 0.5e-3, 0.5)
    fracture: tuple[float, float, float] = (0.5, 0.5e-3, 0.8)

    def profile(self, kind: str, length: float, rng: np.random.Generator, scale: float = 1.0,
                periodic: bool = False) -> RoughProfile:
        rms, xi, hurst = getattr(self, kind)
        return RoughProfile(length, rms * scale, xi, rng, periodic=periodic, spectrum="power_law", hurst=hurst)


@dataclass(frozen=True)
class SoftTissueStructure:
    """Láminas de colágeno del tejido blando (longitudes en m).

    - ``deep_fascia``: fascia profunda (crural, antebraquial, fascia lata)
      entre la grasa subcutánea y el músculo; ~0.6-1 mm en las extremidades
      según los estudios anatómicos de Stecco y colaboradores (cita sin
      verificar; 0.7 mm es una estimación). Sobre la cara subcutánea de la
      tibia se funde con el periostio.
    - ``superficial_fascia`` a ``superficial_fascia_depth`` (fracción del
      grosor de la grasa desde la piel): la lámina membranosa que separa la
      grasa superficial de la profunda (~0.1-0.3 mm, estimación).
    - ``periosteum``: capa fibrosa sobre la cortical (0.1-0.3 mm en el adulto).
    - ``perimysium_spacing`` / ``perimysium_thickness``: septos entre
      fascículos musculares (fascículos de ~1 mm; septos fibroadiposos de
      decenas de µm a 0.1 mm), que dan la textura estriada de la ecografía.
      ``pennation_deg``: ángulo de las fibras respecto al eje del hueso
      (10-25° en tibial anterior, gemelos y vastos; se usa +θ en el músculo
      del lado de la sonda y -θ en el del otro lado).
    - ``fat_lobule_size`` / ``fat_septum_thickness``: lobulillos grasos de
      varios mm separados por septos fibrosos (retinacula cutis); en el
      corte transversal el perimisio forma también polígonos.
    """

    deep_fascia: float = 0.7e-3
    superficial_fascia: float = 0.2e-3
    superficial_fascia_depth: float = 0.5
    periosteum: float = 0.15e-3
    perimysium_spacing: float = 1.2e-3
    perimysium_thickness: float = 0.1e-3
    pennation_deg: float = 15.0
    fat_lobule_size: float = 4e-3
    fat_septum_thickness: float = 0.1e-3


@dataclass(frozen=True)
class RealismSpec:
    """Qué estructuras realistas se añaden al fantoma (``None`` = ninguna de esa clase).

    ``resolve_cells``: un poro o una trabécula se dibuja explícito si su
    menor dimensión mide al menos ``resolve_cells`` celdas; si no, entra
    como fracción de poro por celda. Con 2 celdas una trabécula de 150 µm se
    dibuja a 3 MHz (dx = 60 µm) y los canales de 60 µm quedan como medio
    efectivo. ``0`` fuerza todo explícito; ``inf``, todo efectivo.
    """

    name: str = "basico"
    cortical: CorticalPorosity | None = None
    trabecular: TrabecularNetwork | None = None
    roughness: FractalRoughness | None = None
    soft: SoftTissueStructure | None = None
    resolve_cells: float = 2.0

    @property
    def is_basic(self) -> bool:
        return all(v is None for v in (self.cortical, self.trabecular, self.roughness, self.soft))

    def scaled(self, porosity: float = 1.0, pore_size: float = 1.0, trabecularization: float = 0.0,
               bvtv: float = 1.0, pore_fill: str | None = None) -> RealismSpec:
        """Ajusta la microestructura a un tipo de cuerpo (ver ``BodyType``).

        - ``porosity`` y ``pore_size`` multiplican la porosidad cortical y el
          diámetro de los canales; ``trabecularization`` se suma a la
          fracción trabecularizada; ``pore_fill`` cambia el relleno.
        - ``bvtv`` multiplica la BV/TV. Con la osteoporosis la red pierde
          sobre todo conectividad y número de trabéculas, y algo de grosor:
          Tb.Th se escala con ``bvtv^(1/3)`` y la fracción intacta de placas
          con ``bvtv^(1/2)`` (estimación).
        - La rugosidad endóstica crece con ``sqrt(porosity)``: es la
          superficie que se reabsorbe.
        """
        out = self
        if self.cortical is not None:
            c = self.cortical
            out = replace(out, cortical=replace(
                c, porosity=c.porosity * porosity, canal_diameter=c.canal_diameter * pore_size,
                trabecularized=min(c.trabecularized + trabecularization, 0.9), fill=pore_fill or c.fill))
        if self.trabecular is not None and bvtv != 1.0:
            t = self.trabecular
            out = replace(out, trabecular=replace(
                t, bvtv=t.bvtv * bvtv, thickness=t.thickness * bvtv ** (1.0 / 3.0),
                perforation=1.0 - (1.0 - t.perforation) * bvtv**0.5))
        if self.roughness is not None and porosity != 1.0:
            rms, xi, h = self.roughness.endosteal
            out = replace(out, roughness=replace(self.roughness, endosteal=(rms * porosity**0.5, xi, h)))
        return out


REALISM_PRESETS: dict[str, RealismSpec] = {
    "basico": RealismSpec("basico"),
    "realista": RealismSpec("realista", CorticalPorosity(), TrabecularNetwork(), FractalRoughness(),
                            SoftTissueStructure()),
}


def resolve_realism(realism: str | RealismSpec | None) -> RealismSpec | None:
    """El ``RealismSpec`` pedido, o ``None`` si es el básico (comportamiento de siempre)."""
    if realism is None:
        return None
    spec = REALISM_PRESETS[realism] if isinstance(realism, str) else realism
    return None if spec.is_basic else spec


# --------------------------------------------------------------------------
# Rasterización con fracción de celda
# --------------------------------------------------------------------------
def band_coverage(dist, half_width: float, dx: float) -> np.ndarray:
    """Fracción de una celda de lado ``dx`` dentro de la banda ``|dist| ≤ half_width``.

    Se aproxima la celda por un segmento de longitud ``dx`` según la normal a
    la banda (exacto para bandas alineadas con la malla).
    """
    d = np.abs(dist)
    lo = np.maximum(d - dx / 2, -half_width)
    hi = np.minimum(d + dx / 2, half_width)
    return np.clip((hi - lo) / dx, 0.0, 1.0)


def rasterize_ellipses(grid: Grid2D, cx, cz, semi_a, semi_b, angle, resolve_cells: float,
                       supersample: int = 4) -> tuple[np.ndarray, int]:
    """Fracción de cada celda cubierta por la unión de elipses ``(cx, cz, a, b, ángulo de a respecto a x)``.

    - Elipse resuelta (``2b ≥ resolve_cells·dx``): celdas binarias (las que
      cubre al menos a medias).
    - No resuelta: su área exacta repartida en las celdas que toca (o entera
      en su celda si es más pequeña que ella).
    - Unión de poros solapados: ``1 - Π(1 - c_i)``, el valor esperado del
      modelo booleano dentro de la celda.

    Devuelve ``(fracción, número de elipses explícitas)``.
    """
    dx, x0, z0 = grid.dx, grid.x0, grid.z0
    log_free = np.zeros(grid.shape)
    sub = (np.arange(supersample) + 0.5) / supersample - 0.5
    n_explicit = 0
    for xc, zc, a, b, th in zip(cx, cz, semi_a, semi_b, angle):
        area = np.pi * a * b
        explicit = 2.0 * b >= resolve_cells * dx
        r = max(a, b)
        if not explicit and 2.0 * r < dx:
            i, j = int(round((xc - x0) / dx)), int(round((zc - z0) / dx))
            if 0 <= i < grid.nx and 0 <= j < grid.nz:
                log_free[j, i] += np.log1p(-min(area / dx**2, 1.0 - 1e-9))
            continue
        i0 = max(int(np.floor((xc - r - x0) / dx + 0.5)), 0)
        i1 = min(int(np.floor((xc + r - x0) / dx + 0.5)) + 1, grid.nx)
        j0 = max(int(np.floor((zc - r - z0) / dx + 0.5)), 0)
        j1 = min(int(np.floor((zc + r - z0) / dx + 0.5)) + 1, grid.nz)
        if i0 >= i1 or j0 >= j1:
            continue
        xs = x0 + (np.arange(i0, i1)[:, None] + sub[None, :]) * dx - xc
        zs = z0 + (np.arange(j0, j1)[:, None] + sub[None, :]) * dx - zc
        ddx, ddz = xs[None, :, None, :], zs[:, None, :, None]
        ca, sa = np.cos(th), np.sin(th)
        u, v = ddx * ca + ddz * sa, -ddx * sa + ddz * ca
        c = ((u / a) ** 2 + (v / b) ** 2 <= 1.0).mean(axis=(2, 3))
        if explicit:
            c = (c >= 0.5).astype(float)
            n_explicit += 1
        elif c.sum() > 0:
            c = np.minimum(c * area / (c.sum() * dx**2), 1.0)
        log_free[j0:j1, i0:i1] += np.log1p(-np.minimum(c, 1.0 - 1e-9))
    return 1.0 - np.exp(log_free), n_explicit


# --------------------------------------------------------------------------
# Poros corticales
# --------------------------------------------------------------------------
@dataclass
class PoreSet:
    """Poros en coordenadas continuas: centros, semiejes (a ≥ b) y orientación del eje a."""

    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    angle: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def extend(self, other: PoreSet) -> None:
        for k in ("x", "z", "a", "b", "angle"):
            setattr(self, k, np.concatenate([getattr(self, k), getattr(other, k)]))

    def shifted(self, dz: float) -> PoreSet:
        return PoreSet(self.x, self.z + dz, self.a, self.b, self.angle)

    def rasterize(self, grid: Grid2D, resolve_cells: float) -> tuple[np.ndarray, int]:
        return rasterize_ellipses(grid, self.x, self.z, self.a, self.b, self.angle, resolve_cells)


#: Número de subcapas en profundidad para el gradiente de porosidad.
_N_BANDS = 12


def _pore_sizes(rng, n: int, median: float, sigma: float, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    d = median * np.exp(sigma * rng.standard_normal(n))
    return 0.5 * d * aspect, 0.5 * d


def _mean_area(median: float, sigma: float, aspect: float) -> float:
    return np.pi * aspect * median**2 * np.exp(2.0 * sigma**2) / 4.0


def _band_targets(spec: CorticalPorosity) -> list[tuple[float, float, str]]:
    """Subcapas ``(ζ0, ζ1, población)`` con su porosidad objetivo; ζ = 0 periostio, 1 endostio.

    En la zona trabecularizada los canales siguen ahí y las cavidades
    completan la porosidad hasta ``trabecularized_porosity``.
    """
    phi_p, _ = spec.compact_porosity()
    r = spec.endosteal_ratio
    out = []
    edges = np.linspace(0.0, 1.0, _N_BANDS + 1)
    for z0, z1 in zip(edges[:-1], edges[1:]):
        out.append((z0, z1, phi_p * (1.0 + (r - 1.0) * min(0.5 * (z0 + z1), 1.0 - spec.trabecularized)), "canal"))
    if spec.trabecularized > 0:
        phi_c_end = phi_p * (1.0 + (r - 1.0) * (1.0 - spec.trabecularized))
        phi_cav = 1.0 - (1.0 - spec.trabecularized_porosity) / (1.0 - phi_c_end)
        out.append((1.0 - spec.trabecularized, 1.0, max(phi_cav, 0.0), "cavity"))
    return out


def sample_layer_pores(spec: CorticalPorosity, rng: np.random.Generator, outer: Callable, inner: Callable,
                       s_range: tuple[float, float], to_xz: Callable, along_angle: float, canal_aspect: float,
                       oriented: bool = True, n_probe: int = 512) -> PoreSet:
    """Poros de una cortical parametrizada por ``s`` (posición a lo largo) y ``ζ`` (profundidad).

    ``outer(s)`` y ``inner(s)`` dan la coordenada de profundidad del periostio
    y del endostio; ``to_xz(s, d)`` pasa de ``(s, profundidad)`` a ``(x, z)`` y
    devuelve también el jacobiano del área (1 en el corte longitudinal, donde
    ``s = x``; el radio ``r`` en el anillo del corte transversal, donde ``s``
    es el ángulo y la profundidad es el radio). Los centros se
    reparten uniformemente en área con aceptación-rechazo, y las subcapas
    extremas se prolongan medio poro hacia fuera para que los poros que
    sobresalen de la cortical no dejen sus bordes con menos porosidad.
    ``oriented``: el eje largo de los poros sigue ``along_angle`` (el eje del
    hueso, ±15°); si no, la orientación es aleatoria.
    """
    s0, s1 = s_range
    probe = np.linspace(s0, s1, n_probe)
    thick = np.abs(inner(probe) - outer(probe))
    pores = PoreSet()
    for z0, z1, phi, kind in _band_targets(spec):
        if phi <= 0:
            continue
        if kind == "canal":
            med, aspect = spec.canal_diameter, canal_aspect
        else:
            med, aspect = spec.cavity_diameter, spec.cavity_aspect
        mean_area = _mean_area(med, spec.diameter_sigma, aspect)
        pad = 0.5 * med / max(float(thick.mean()), 1e-9)
        z0e = z0 - pad if z0 <= 0.0 else z0
        z1e = z1 + pad if z1 >= 1.0 else z1
        _, w_probe = to_xz(probe, outer(probe) + 0.5 * (z0e + z1e) * (inner(probe) - outer(probe)))
        y = thick * w_probe
        area = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(probe))) * (z1e - z0e)
        n = rng.poisson(-np.log1p(-phi) * area / mean_area)
        # cota de la densidad de aceptación (en el anillo crece hacia fuera)
        w_max = 1.5 * float(y.max())
        s, zeta = np.zeros(0), np.zeros(0)
        while s.size < n:
            m = 2 * (n - s.size) + 8
            cs, cz = rng.uniform(s0, s1, m), rng.uniform(z0e, z1e, m)
            depth = outer(cs) + cz * (inner(cs) - outer(cs))
            _, w = to_xz(cs, depth)
            ok = rng.uniform(0.0, w_max, m) <= np.abs(inner(cs) - outer(cs)) * w
            s, zeta = np.concatenate([s, cs[ok]]), np.concatenate([zeta, cz[ok]])
        s, zeta = s[:n], zeta[:n]
        x, z = to_xz(s, outer(s) + zeta * (inner(s) - outer(s)))[0]
        a, b = _pore_sizes(rng, n, med, spec.diameter_sigma, aspect)
        jitter = np.radians(15.0) if oriented else np.pi
        ang = along_angle + rng.uniform(-jitter, jitter, n)
        pores.extend(PoreSet(x, z, a, b, ang))
    return pores


# --------------------------------------------------------------------------
# Paredes de Voronoi (red trabecular, perimisio, lobulillos grasos)
# --------------------------------------------------------------------------
def poisson_seeds(rng: np.random.Generator, bbox: tuple[float, float, float, float],
                  density: float) -> tuple[np.ndarray, np.ndarray]:
    """Semillas de Poisson con ``density`` puntos/m² en ``(x0, x1, z0, z1)``."""
    x0, x1, z0, z1 = bbox
    n = rng.poisson(density * (x1 - x0) * (z1 - z0))
    return rng.uniform(x0, x1, n), rng.uniform(z0, z1, n)


def voronoi_wall_distance(px: np.ndarray, pz: np.ndarray, sx: np.ndarray, sz: np.ndarray, stretch: float = 1.0,
                          angle: float = 0.0, removed: float = 0.0,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Distancia euclídea de cada punto a la pared más próxima de una teselación de Voronoi.

    - ``stretch``: las celdas se alargan este factor según la dirección
      ``angle`` (radianes desde +x): las paredes, y por tanto las trabéculas,
      quedan orientadas preferentemente en esa dirección. Las celdas son las
      de Voronoi en la métrica estirada y su pared con cada vecina es la
      recta bisectriz en esa métrica; la distancia a ella se mide en metros.
    - ``removed``: fracción de paredes eliminadas al azar (perforación). Un
      punto cuyas paredes cercanas se han perdido queda a distancia infinita.

    Se busca la vecina más próxima entre las tres semillas más cercanas,
    por bloques espaciales para no comparar cada punto con todas.
    """
    ca, sa = np.cos(angle), np.sin(angle)
    pu, pv = px * ca + pz * sa, -px * sa + pz * ca
    su, sv = sx * ca + sz * sa, -sx * sa + sz * ca
    keep_key = (rng or np.random.default_rng(0)).random(sx.size)
    out = np.full(px.shape, np.inf)
    if sx.size < 3 or px.size == 0:
        return out
    a2 = stretch**2
    qu, qv = pu / stretch, pv  # métrica estirada
    tu, tv = su / stretch, sv
    area = (tu.max() - tu.min() + 1e-12) * (tv.max() - tv.min() + 1e-12)
    spacing = np.sqrt(area / sx.size)
    tile = 6.0 * spacing
    iu = np.floor((qu - qu.min()) / tile).astype(np.int64)
    iv = np.floor((qv - qv.min()) / tile).astype(np.int64)
    key = iu * (iv.max() + 1) + iv
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    for idx in np.split(order, bounds):
        lo_u, hi_u, lo_v, hi_v = qu[idx].min(), qu[idx].max(), qv[idx].min(), qv[idx].max()
        margin = 3.0 * spacing
        while True:
            sel = np.flatnonzero((tu >= lo_u - margin) & (tu <= hi_u + margin) & (tv >= lo_v - margin) &
                                 (tv <= hi_v + margin))
            if sel.size >= 3 or sel.size == sx.size:
                break
            margin *= 2.0
        d2 = (qu[idx, None] - tu[None, sel]) ** 2 + (qv[idx, None] - tv[None, sel]) ** 2
        near = np.argpartition(d2, 2, axis=1)[:, :3]
        near = np.take_along_axis(near, np.argsort(np.take_along_axis(d2, near, axis=1), axis=1), axis=1)
        n1 = sel[near[:, 0]]
        best = np.full(idx.size, np.inf)
        for col in (1, 2):
            nj = sel[near[:, col]]
            nu, nv = (su[nj] - su[n1]) / a2, sv[nj] - sv[n1]
            mu, mv = 0.5 * (su[nj] + su[n1]), 0.5 * (sv[nj] + sv[n1])
            w = np.abs((pu[idx] - mu) * nu + (pv[idx] - mv) * nv) / np.hypot(nu, nv)
            kept = np.mod(keep_key[n1] + keep_key[nj], 1.0) >= removed
            best = np.where(kept, np.minimum(best, w), best)
        out[idx] = best
    return out


def voronoi_density(wall_length_per_area: float, stretch: float) -> float:
    """Semillas por m² de una teselación de Poisson-Voronoi con esa longitud de pared por área.

    Isótropa: ``L_A = 2 sqrt(λ)``. Estirada un factor ``a``, cada pared se
    alarga según su dirección; en promedio, el perímetro de la elipse de
    semiejes ``(a, 1)`` entre ``2π`` (Ramanujan), y el área crece ``a``.
    """
    a = stretch
    perimeter = np.pi * (3.0 * (a + 1.0) - np.sqrt((3.0 * a + 1.0) * (a + 3.0)))
    return float((wall_length_per_area * np.pi * np.sqrt(a) / perimeter) ** 2)


@dataclass
class NetworkResult:
    porosity: np.ndarray  # fracción de fluido en los puntos pedidos
    bvtv: float
    thickness: float  # 2 × semiancho de las paredes (m)
    explicit: bool
    n_seeds: int


def trabecular_network(px: np.ndarray, pz: np.ndarray, spec: TrabecularNetwork, dx: float,
                       rng: np.random.Generator, resolve_cells: float) -> NetworkResult:
    """Porosidad de la red trabecular en los puntos ``(px, pz)`` (centros de celda de la región).

    Las semillas se reparten con la densidad que, con paredes de Tb.Th y la
    perforación pedida, da la BV/TV pedida en el límite de paredes finas.
    Luego el semiancho ``t`` de las paredes se ajusta para que la BV/TV de
    la región sea exactamente la pedida (los solapes en los nudos la
    bajarían): con la red resuelta, por el cuantil de la distancia a la
    pared; sin resolver, por bisección de la fracción media cubierta por
    las paredes, que da la porosidad por celda del medio efectivo.
    """
    if px.size == 0:
        return NetworkResult(np.zeros(0), 0.0, 0.0, False, 0)
    wall_length = spec.bvtv / (spec.thickness * (1.0 - spec.perforation))
    density = voronoi_density(wall_length, spec.anisotropy)
    pad = 4.0 * spec.anisotropy / np.sqrt(density)
    bbox = (px.min() - pad, px.max() + pad, pz.min() - pad, pz.max() + pad)
    sx, sz = poisson_seeds(rng, bbox, density)
    w = voronoi_wall_distance(px, pz, sx, sz, spec.anisotropy, np.radians(spec.orientation_deg),
                              spec.perforation, rng)
    explicit = spec.thickness >= resolve_cells * dx
    finite = w[np.isfinite(w)]
    if explicit:
        t = float(np.quantile(finite, min(spec.bvtv * w.size / max(finite.size, 1), 1.0)))
        solid = (w <= t).astype(float)
    else:
        lo, hi = 0.0, float(finite.max()) + dx
        for _ in range(50):
            t = 0.5 * (lo + hi)
            if band_coverage(w, t, dx).mean() < spec.bvtv:
                lo = t
            else:
                hi = t
        solid = band_coverage(w, t, dx)
    return NetworkResult(1.0 - solid, float(solid.mean()), 2.0 * t, explicit, int(sx.size))


def wall_coverage(px: np.ndarray, pz: np.ndarray, cell_size: float, thickness: float, dx: float,
                  rng: np.random.Generator, stretch: float = 1.0, angle: float = 0.0,
                  removed: float = 0.0) -> np.ndarray:
    """Fracción de celda cubierta por las paredes de una teselación con celdas de ~``cell_size``."""
    if px.size == 0:
        return np.zeros(0)
    density = 1.0 / cell_size**2
    pad = 4.0 * stretch * cell_size
    sx, sz = poisson_seeds(rng, (px.min() - pad, px.max() + pad, pz.min() - pad, pz.max() + pad), density)
    w = voronoi_wall_distance(px, pz, sx, sz, stretch, angle, removed, rng)
    return band_coverage(w, 0.5 * thickness, dx)


# --------------------------------------------------------------------------
# Septos paralelos (perimisio en el corte longitudinal)
# --------------------------------------------------------------------------
def parallel_septa_coverage(px: np.ndarray, pz: np.ndarray, angle: float, spacing: float, thickness: float,
                            dx: float, rng: np.random.Generator, presence: float = 0.75) -> np.ndarray:
    """Fracción de celda cubierta por septos casi paralelos con dirección ``angle`` (radianes desde +x).

    Separación log-normal alrededor de ``spacing`` (CV ≈ 30 %), ondulación
    suave de cada septo (15 % de la separación, longitud de correlación de
    3 mm) e interrupciones: cada septo existe en una fracción ``presence``
    de su longitud (los fascículos entran y salen del plano de corte).
    """
    out = np.zeros(px.shape)
    if px.size == 0:
        return out
    ca, sa = np.cos(angle), np.sin(angle)
    u, v = px * ca + pz * sa, -px * sa + pz * ca
    u0, length = u.min(), u.max() - u.min() + 1e-6
    v_k = v.min() - spacing * rng.uniform(0.0, 1.0)
    probe = np.linspace(0.0, length, 2048)
    while v_k < v.max() + spacing:
        wav = RoughProfile(length, 0.15 * spacing, 3e-3, rng)
        gate = RoughProfile(length, 1.0, 4e-3, rng)
        threshold = np.quantile(gate(probe), 1.0 - presence)
        near = np.abs(v - v_k) < spacing
        if near.any():
            s = u[near] - u0
            cov = band_coverage(v[near] - v_k - wav(s), 0.5 * thickness, dx) * (gate(s) > threshold)
            out[near] = np.maximum(out[near], cov)
        v_k += spacing * np.exp(0.3 * rng.standard_normal() - 0.045)
    return out


def dither(coverage: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Celdas que se etiquetan con la estructura: probabilidad igual a la fracción cubierta."""
    return coverage > u
