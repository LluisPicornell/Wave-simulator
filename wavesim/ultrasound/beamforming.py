"""Reconstrucción de la imagen de ecografía (modo B) a partir de las señales RF.

Delay-and-sum (DAS) para emisiones de onda plana (Montaldo et al., IEEE
UFFC 56, 489, 2009): para cada píxel se calcula cuándo debería llegar su eco a
cada elemento suponiendo una velocidad del sonido constante ``c0`` (1540 m/s,
como un ecógrafo real). Las diferencias reales de velocidad entre tejidos
(grasa 1450, hueso 3500 m/s) producen las mismas distorsiones que en la
clínica: desenfoque y desplazamiento de ecos bajo el hueso.

Además del DAS de ondas planas hay un camino general (:func:`bmode_channels`)
que trabaja con los datos de canal ya retardados de *cualquier* secuencia de
emisión (onda plana, emisión focalizada línea a línea o apertura sintética, STA)
y permite otros formadores de haz:

- ``das``: suma con apodización (``rx_window``) y apertura dinámica ``f_number``.
- ``cf``: factor de coherencia de Mallart y Fink (JASA 96, 3721, 1994), el cociente
  entre la energía coherente y la total de la apertura, que multiplica al DAS y
  atenúa píxeles dominados por ruido o clutter incoherente.
- ``dmas``: *delay-multiply-and-sum* de Matrone et al. (IEEE TMI 34, 940, 2015),
  versión sobre señal analítica (ver :func:`combine_channels`).
- ``slsc``: coherencia espacial de lag corto de Lediju et al. (IEEE TUFFC 58, 1377,
  2011): imagen de coherencia, no de amplitud.

Las referencias se citan de memoria y no se han verificado en línea.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.signals import analytic_signal
from .transducer import window_function


@dataclass
class BModeImage:
    envelope: np.ndarray  # amplitud lineal (nz, nx)
    x: np.ndarray
    z: np.ndarray
    dynamic_range_db: float = 50.0
    iq: np.ndarray | None = None  # imagen compleja (sin TGC); None en SLSC

    @property
    def db(self) -> np.ndarray:
        env = self.envelope / (self.envelope.max() + 1e-30)
        return 20.0 * np.log10(np.maximum(env, 10 ** (-self.dynamic_range_db / 20)))

    def extent_mm(self) -> tuple[float, float, float, float]:
        dx = self.x[1] - self.x[0] if self.x.size > 1 else 0
        dz = self.z[1] - self.z[0] if self.z.size > 1 else 0
        return ((self.x[0] - dx / 2) * 1e3, (self.x[-1] + dx / 2) * 1e3,
                (self.z[-1] + dz / 2) * 1e3, (self.z[0] - dz / 2) * 1e3)


def das_plane_wave(
    rf: np.ndarray,
    t: np.ndarray,
    element_x: np.ndarray,
    array_z: float,
    angle_deg: float,
    x_img: np.ndarray,
    z_img: np.ndarray,
    c0: float = 1540.0,
    t_peak: float = 0.0,
    f_number: float = 1.5,
    rx_window: str = "hann",
) -> np.ndarray:
    """Imagen compleja (señal analítica) de una emisión de onda plana.

    ``rf``: (n_elementos, nt) muestreada en los instantes ``t`` (uniformes).
    ``t_peak``: instante del centro del pulso emitido (referencia temporal).
    """
    iq = analytic_signal(rf, axis=1)
    dt = t[1] - t[0]
    theta = np.radians(angle_deg)
    s, c = np.sin(theta), np.cos(theta)
    xr = element_x.min() if s >= 0 else element_x.max()
    X, Z = np.meshgrid(x_img, z_img)
    depth = Z - array_z
    t_tx = ((X - xr) * s + depth * c) / c0 + t_peak
    img = np.zeros(X.shape, dtype=complex)
    weight = np.zeros(X.shape)
    nt = rf.shape[1]
    for e, xe in enumerate(element_x):
        lateral = X - xe
        # Apertura dinámica en recepción (f-number) con ventana de Hann
        half_ap = np.maximum(depth, 1e-6) / (2.0 * f_number)
        w = window_function(np.abs(lateral) / half_ap, rx_window)
        if not np.any(w):
            continue
        tau = t_tx + np.hypot(lateral, depth) / c0
        idx = (tau - t[0]) / dt
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        valid = (i0 >= 0) & (i0 < nt - 1)
        i0c = np.clip(i0, 0, nt - 2)
        sample = iq[e, i0c] * (1 - frac) + iq[e, i0c + 1] * frac
        img += np.where(valid, w * sample, 0.0)
        weight += w
    return img / np.maximum(weight, 1e-12)


def bmode(
    rf_list: list[np.ndarray],
    angles_deg: list[float],
    t: np.ndarray,
    element_x: np.ndarray,
    array_z: float,
    x_img: np.ndarray,
    z_img: np.ndarray,
    c0: float = 1540.0,
    t_peak: float = 0.0,
    blank_until: float = 0.0,
    tgc_db_per_cm: float = 0.0,
    f_number: float = 1.5,
    dynamic_range_db: float = 50.0,
    rx_window: str = "hann",
    beamformer: str = "das",
    layers: TwoLayerSpeed | None = None,
) -> BModeImage:
    """Imagen modo B compuesta (coherente) a partir de varias ondas planas.

    ``blank_until``: se anulan las muestras anteriores (pulso directo emitido).
    ``tgc_db_per_cm``: compensación de ganancia con la profundidad (ida y vuelta).
    ``rx_window``: ventana de la apertura de recepción (``transducer.WINDOWS``).
    ``beamformer`` distinto de ``"das"`` o ``layers`` (corrección de la velocidad en dos
    capas) usan el camino general de :func:`bmode_channels`; con los valores por
    defecto el resultado es el DAS de siempre.
    """
    if beamformer != "das" or layers is not None:
        rfs = []
        for rf in rf_list:
            rf = np.array(rf, dtype=float)
            rf[:, t < blank_until] = 0.0
            rfs.append(rf)
        return bmode_channels(rfs, [Transmit("plane", a) for a in angles_deg], t, element_x, array_z, x_img, z_img,
                              c0, t_peak, tgc_db_per_cm, f_number, dynamic_range_db, rx_window, beamformer, layers)
    img = np.zeros((z_img.size, x_img.size), dtype=complex)
    for rf, ang in zip(rf_list, angles_deg, strict=True):
        rf = np.array(rf, dtype=float)
        rf[:, t < blank_until] = 0.0
        img += das_plane_wave(rf, t, element_x, array_z, ang, x_img, z_img, c0, t_peak, f_number, rx_window)
    env = np.abs(img)
    if tgc_db_per_cm:
        depth_cm = np.maximum(z_img - array_z, 0.0) * 100.0
        env *= (10 ** (tgc_db_per_cm * depth_cm / 20.0))[:, None]
    return BModeImage(env, x_img, z_img, dynamic_range_db, img)


# --------------------------------------------------------------------------
# Camino general: datos de canal retardados + formadores de haz alternativos
# --------------------------------------------------------------------------
BEAMFORMERS = ("das", "cf", "dmas", "slsc")


@dataclass
class TwoLayerSpeed:
    """Velocidad del sonido en dos capas planas (p. ej. grasa sobre músculo).

    Para ``z < z_interface`` la velocidad es ``c_top`` y por debajo ``c_bottom``.
    Los tiempos de vuelo se calculan con el principio de Fermat (el trayecto que
    cruza la interfaz en el punto de tiempo mínimo, es decir, con refracción de
    Snell), no con 1540 m/s fijo. La sonda debe estar por encima de la interfaz.
    Una interfaz plana horizontal es un modelo muy simple: no recoge capas
    inclinadas, tres o más capas ni hueso.
    """

    z_interface: float
    c_top: float
    c_bottom: float

    def point_source_time(self, xs: float, zs: float, X: np.ndarray, Z: np.ndarray, n_iter: int = 30) -> np.ndarray:
        """Tiempo de vuelo desde un punto ``(xs, zs)`` (por encima de la interfaz) a cada píxel."""
        h1 = self.z_interface - zs
        t = np.hypot(X - xs, Z - zs) / self.c_top
        below = Z > self.z_interface
        if not below.any():
            return t
        Xb, h2 = X[below], Z[below] - self.z_interface
        lo, hi = np.minimum(xs, Xb), np.maximum(xs, Xb)
        for _ in range(n_iter):  # bisección sobre dt/dxc (creciente en xc)
            xc = 0.5 * (lo + hi)
            d = (xc - xs) / (self.c_top * np.hypot(xc - xs, h1)) + (xc - Xb) / (self.c_bottom * np.hypot(Xb - xc, h2))
            lo, hi = np.where(d < 0, xc, lo), np.where(d < 0, hi, xc)
        xc = 0.5 * (lo + hi)
        t[below] = np.hypot(xc - xs, h1) / self.c_top + np.hypot(Xb - xc, h2) / self.c_bottom
        return t

    def plane_wave_time(self, X: np.ndarray, Z: np.ndarray, angle_deg: float, xr: float, array_z: float) -> np.ndarray:
        """Tiempo de llegada de una onda plana emitida con ``angle_deg`` (frente que toca la sonda en ``xr``)."""
        th = np.radians(angle_deg)
        s1, c1 = np.sin(th), np.cos(th)
        h1 = self.z_interface - array_z
        t = ((X - xr) * s1 + (Z - array_z) * c1) / self.c_top
        below = Z > self.z_interface
        if below.any():
            s2 = np.clip(self.c_bottom / self.c_top * s1, -0.999999, 0.999999)  # Snell
            tan2 = s2 / np.sqrt(1.0 - s2 ** 2)
            Xb, h2 = X[below], Z[below] - self.z_interface
            xc = Xb - h2 * tan2
            t[below] = ((xc - xr) * s1 + h1 * c1) / self.c_top + np.hypot(Xb - xc, h2) / self.c_bottom
        return t


@dataclass
class Transmit:
    """Descripción de una emisión para el beamforming.

    - ``kind="plane"``: onda plana de ángulo ``angle_deg``.
    - ``kind="focused"``: emisión focalizada en ``focus = (x, z)``; ``t_focus`` es el
      tiempo (desde el inicio del pulso) que tarda el frente en llegar al foco
      (la distancia del elemento activo más lejano dividida por ``c0``).
    - ``kind="element"``: un único elemento emite (STA); ``element`` es su índice.
    ``pixel_mask`` limita los píxeles que reconstruye esta emisión (la línea de la
    emisión focalizada); ``weight`` pondera su contribución.
    """

    kind: str = "plane"
    angle_deg: float = 0.0
    focus: tuple[float, float] | None = None
    t_focus: float = 0.0
    element: int | None = None
    pixel_mask: np.ndarray | None = None
    weight: float = 1.0


def _transmit_time(tx: Transmit, X, Z, element_x, array_z, c0, t_peak, layers, tof) -> np.ndarray:
    if tx.kind == "plane":
        xr = element_x.min() if np.sin(np.radians(tx.angle_deg)) >= 0 else element_x.max()
        if layers is not None:
            return layers.plane_wave_time(X, Z, tx.angle_deg, xr, array_z) + t_peak
        th = np.radians(tx.angle_deg)
        return ((X - xr) * np.sin(th) + (Z - array_z) * np.cos(th)) / c0 + t_peak
    if tx.kind == "element":
        return tof[tx.element] + t_peak
    if tx.kind == "focused":
        if layers is not None:
            raise NotImplementedError("La corrección de capas no está implementada para emisión focalizada")
        xf, zf = tx.focus
        # Antes del foco el frente converge (llega antes de t_focus), después diverge
        return t_peak + tx.t_focus + np.sign(Z - zf) * np.hypot(X - xf, Z - zf) / c0
    raise ValueError(f"Tipo de emisión desconocido: {tx.kind!r}")


def channel_cube(iq_list: list[np.ndarray], transmits: list[Transmit], t: np.ndarray, element_x: np.ndarray,
                 array_z: float, x_img: np.ndarray, z_img: np.ndarray, c0: float = 1540.0, t_peak: float = 0.0,
                 f_number: float = 1.5, rx_window: str = "hann",
                 layers: TwoLayerSpeed | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Datos de canal retardados ``C[e, j, i]`` y pesos de apertura ``W[e, j, i]``.

    ``iq_list[k]``: señal analítica ``(n_elementos, nt)`` de la emisión ``transmits[k]``.
    ``C`` suma, sobre todas las emisiones, la muestra de cada elemento en el instante
    del eco de cada píxel (sin apodizar en recepción); la suma coherente de las
    emisiones por canal es lo que da el foco de emisión sintético en STA y la
    composición en onda plana.
    """
    dt = t[1] - t[0]
    nt = t.size
    X, Z = np.meshgrid(x_img, z_img)
    depth = Z - array_z
    n_el = element_x.size
    W = np.empty((n_el, *X.shape))
    tof = np.empty((n_el, *X.shape))
    half_ap = np.maximum(depth, 1e-6) / (2.0 * f_number)
    for e, xe in enumerate(element_x):
        lateral = X - xe
        W[e] = window_function(np.abs(lateral) / half_ap, rx_window)
        if layers is None:
            tof[e] = np.hypot(lateral, depth) / c0
        else:
            tof[e] = layers.point_source_time(xe, array_z, X, Z)
    C = np.zeros((n_el, *X.shape), dtype=complex)
    for iq, tx in zip(iq_list, transmits, strict=True):
        t_tx = _transmit_time(tx, X, Z, element_x, array_z, c0, t_peak, layers, tof)
        gain = tx.weight if tx.pixel_mask is None else tx.weight * tx.pixel_mask
        if tx.kind == "element":  # el elemento emisor también usa la apertura dinámica (F# y ventana)
            gain = gain * W[tx.element]
        for e in range(n_el):
            if not np.any(W[e]):
                continue
            idx = (t_tx + tof[e] - t[0]) / dt
            i0 = np.floor(idx).astype(int)
            frac = idx - i0
            valid = (i0 >= 0) & (i0 < nt - 1)
            i0c = np.clip(i0, 0, nt - 2)
            sample = iq[e, i0c] * (1 - frac) + iq[e, i0c + 1] * frac
            C[e] += np.where(valid, gain * sample, 0.0)
    return C, W


def combine_channels(C: np.ndarray, W: np.ndarray, method: str = "das", cf_power: float = 1.0,
                     max_lag: int | None = None, kernel_px: int = 8) -> np.ndarray:
    """Combina los canales retardados en una imagen (compleja salvo en ``slsc``, que es real).

    - ``das``: ``Σ W·C / Σ W``.
    - ``cf``: DAS × CF^``cf_power``, con ``CF = |Σ C|² / (N Σ |C|²)`` sobre los ``N``
      elementos activos del píxel (``W > 0``); vale 1 con canales en fase y ~1/N
      con ruido incoherente.
    - ``dmas``: con ``ŝ = W·C/√|C|`` (conserva la fase), ``Σ_{i<j} ŝ_i ŝ_j =
      ((Σ ŝ)² − Σ ŝ²)/2``, normalizado por el número de pares. Sobre señal analítica la
      fase se duplica (banda a 2·f0). No es el DMAS de Matrone sobre RF real con
      signo, sino su análogo complejo.
    - ``slsc``: ``Σ_{m=1..max_lag} R(m)``, con ``R(m)`` la correlación normalizada entre
      canales a ``m`` elementos de distancia, dentro de una ventana axial de
      ``kernel_px`` píxeles. Devuelve valores en [-max_lag, max_lag]; no es una amplitud.
    """
    active = W > 0
    n_act = np.maximum(active.sum(0), 1)
    das = (W * C).sum(0) / np.maximum(W.sum(0), 1e-12)
    if method == "das":
        return das
    if method == "cf":
        Ca = np.where(active, C, 0.0)
        coh = np.abs(Ca.sum(0)) ** 2
        tot = n_act * (np.abs(Ca) ** 2).sum(0)
        return das * (coh / np.maximum(tot, 1e-30)) ** cf_power
    if method == "dmas":
        s_hat = W * C / np.sqrt(np.maximum(np.abs(C), 1e-30))
        pairs = np.maximum(n_act * (n_act - 1) / 2.0, 1.0)
        return (s_hat.sum(0) ** 2 - (s_hat ** 2).sum(0)) / 2.0 / pairs
    if method == "slsc":
        return _slsc(C, active, max_lag or max(C.shape[0] // 5, 1), kernel_px)
    raise ValueError(f"Formador de haz desconocido: {method!r} (opciones: {', '.join(BEAMFORMERS)})")


def _moving_sum_z(a: np.ndarray, n: int) -> np.ndarray:
    """Suma móvil de ``n`` muestras a lo largo de la profundidad (eje 1 de un cubo ``(e, z, x)``), centrada."""
    pad = np.concatenate([np.zeros_like(a[:, :1]), np.cumsum(a, axis=1)], axis=1)
    nz = a.shape[1]
    lo = np.clip(np.arange(nz) - n // 2, 0, nz)
    hi = np.clip(np.arange(nz) - n // 2 + n, 0, nz)
    return pad[:, hi] - pad[:, lo]


def _slsc(C: np.ndarray, active: np.ndarray, max_lag: int, kernel_px: int) -> np.ndarray:
    energy = _moving_sum_z(np.where(active, np.abs(C) ** 2, 0.0), kernel_px)
    out = np.zeros(C.shape[1:])
    for m in range(1, min(max_lag, C.shape[0] - 1) + 1):
        both = active[:-m] & active[m:]
        num = _moving_sum_z(np.where(both, (C[:-m] * np.conj(C[m:])).real, 0.0), kernel_px)
        den = np.sqrt(np.maximum(energy[:-m] * energy[m:], 1e-30))
        out += np.where(both, num / den, 0.0).sum(0) / np.maximum(both.sum(0), 1)
    return out


def bmode_channels(
    rf_list: list[np.ndarray],
    transmits: list[Transmit],
    t: np.ndarray,
    element_x: np.ndarray,
    array_z: float,
    x_img: np.ndarray,
    z_img: np.ndarray,
    c0: float = 1540.0,
    t_peak: float = 0.0,
    tgc_db_per_cm: float = 0.0,
    f_number: float = 1.5,
    dynamic_range_db: float = 50.0,
    rx_window: str = "hann",
    beamformer: str = "das",
    layers: TwoLayerSpeed | None = None,
    cf_power: float = 1.0,
    max_lag: int | None = None,
    kernel_px: int = 8,
) -> BModeImage:
    """Imagen modo B a partir de las señales RF de una lista de emisiones cualesquiera."""
    iq_list = [analytic_signal(np.asarray(rf, float), axis=1) for rf in rf_list]
    C, W = channel_cube(iq_list, transmits, t, element_x, array_z, x_img, z_img, c0, t_peak, f_number, rx_window,
                        layers)
    img = combine_channels(C, W, beamformer, cf_power, max_lag, kernel_px)
    env = np.maximum(img, 0.0) if beamformer == "slsc" else np.abs(img)
    if tgc_db_per_cm and beamformer != "slsc":
        depth_cm = np.maximum(z_img - array_z, 0.0) * 100.0
        env = env * (10 ** (tgc_db_per_cm * depth_cm / 20.0))[:, None]
    return BModeImage(env, x_img, z_img, dynamic_range_db, None if beamformer == "slsc" else img)
