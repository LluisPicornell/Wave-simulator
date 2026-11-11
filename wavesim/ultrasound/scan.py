"""Exploración ecográfica completa: simulación + reconstrucción en una llamada."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..phantoms.phantom import Phantom
from .beamforming import BModeImage, Transmit, TwoLayerSpeed, bmode, bmode_channels
from .elastic_fdtd import ElasticSolver, SimulationResult
from .medium import ElasticMedium, elastic_medium
from .transducer import LinearArray, line_receiver


@dataclass
class UltrasoundScan:
    image: BModeImage
    array: LinearArray
    angles: list[float]
    t: np.ndarray
    rf: list[np.ndarray]
    results: list[SimulationResult] = field(default_factory=list)
    transmitted: np.ndarray | None = None  # presión en la fila inferior (1.ª emisión)
    transmits: list[Transmit] = field(default_factory=list)
    elapsed_s: float = 0.0  # tiempo de las simulaciones FDTD

    @property
    def snapshots(self):
        return self.results[0].snapshots

    @property
    def snapshot_times(self):
        return self.results[0].snapshot_times

    @property
    def peak_pressure(self):
        return self.results[0].peak_pressure


def default_array(phantom: Phantom, f0: float, npml: int = 20, c0: float = 1540.0,
                  fill: float = 0.85) -> LinearArray:
    """Sonda que ocupa ``fill`` del ancho útil, con paso λ/2 y bajo la capa absorbente."""
    g = phantom.grid
    pitch = c0 / f0 / 2.0
    usable = g.width - 2 * (npml + 3) * g.dx
    n = max(int(fill * usable / pitch), 8)
    z = g.z0 + (npml + 3) * g.dx
    return LinearArray(n_elements=n, pitch=pitch, z=z, x_center=g.x0 + g.width / 2 - g.dx / 2, f0=f0)


SCAN_MODES = ("plane", "focused", "sta")


def simulate_scan(
    phantom: Phantom,
    f0: float = 3e6,
    mode: str = "plane",
    array: LinearArray | None = None,
    angles: tuple[float, ...] = (0.0,),
    focus_depth: float | None = None,
    tx_f_number: float = 2.0,
    tx_window: str = "hann",
    line_step: int = 2,
    sta_step: int = 1,
    c0: float = 1540.0,
    max_depth: float | None = None,
    npml: int = 20,
    n_mechanisms: int = 2,
    scattering: float = 1.0,
    snapshot_every: int | None = None,
    snapshot_decimate: int = 2,
    snapshot_field: str = "p",
    pixel: float | None = None,
    tgc_db_per_cm: float | None = None,
    record_transmission: bool = False,
    progress: bool = False,
    anisotropic_bone: bool = False,
    f_number: float = 1.5,
    rx_window: str = "hann",
    tx_apodization: np.ndarray | None = None,
    beamformer: str = "das",
    layers: TwoLayerSpeed | None = None,
    medium: ElasticMedium | None = None,
    **bf_kwargs,
) -> UltrasoundScan:
    """Simula una ecografía con la secuencia de emisión ``mode`` y reconstruye el modo B.

    Modos (una simulación FDTD por emisión, que domina el coste):

    - ``"plane"``: una onda plana por ángulo de ``angles`` compuestas coherentemente.
    - ``"focused"``: emisión focalizada en ``focus_depth`` (por defecto, el centro de la
      imagen) con apertura de emisión ``F# = tx_f_number`` y ventana ``tx_window``,
      línea a línea: una emisión cada ``line_step`` elementos, y cada píxel se
      reconstruye con la línea más cercana. Foco fijo: fuera de la zona focal el haz de
      emisión se ensancha (no hay foco dinámico de emisión).
    - ``"sta"``: apertura sintética, un elemento cada ``sta_step`` emite y todos reciben; la
      suma coherente de todas las emisiones da foco de emisión en cada píxel (imagen de
      referencia; con ``sta_step=1`` es la mejor que da la apertura). Es el modo más caro:
      ``n_elementos / sta_step`` simulaciones. Con ``sta_step > 1`` la apertura de emisión
      queda submuestreada (lóbulos de rejilla si el paso supera λ/2).

    ``tx_apodization`` pondera la emisión de onda plana o focalizada (en focalizada
    se multiplica por la ventana de ``tx_window``). ``rx_window``, ``beamformer``
    (``beamforming.BEAMFORMERS``), ``layers`` y ``f_number`` configuran la reconstrucción.
    ``medium``: medio elástico a usar en lugar de construirlo del fantoma (p. ej. con un
    dispersor puntual); el fantoma aporta entonces sólo la malla.
    Con ``mode="plane"`` y los valores por defecto es exactamente :func:`simulate_bmode`.
    Los argumentos adicionales (``cf_power``, ``max_lag``, ``kernel_px``) van al formador de haz.
    """
    if mode not in SCAN_MODES:
        raise ValueError(f"Modo desconocido: {mode!r} (opciones: {', '.join(SCAN_MODES)})")
    g = phantom.grid
    array = array or default_array(phantom, f0, npml, c0)
    bottom = g.z0 + g.depth - (npml + 3) * g.dx
    max_depth = min(max_depth or bottom, bottom)
    if tgc_db_per_cm is None:
        tgc_db_per_cm = 0.5 * f0 / 1e6
    if medium is None:  # se construye una sola vez para todas las emisiones
        medium = elastic_medium(phantom, f0, n_mechanisms, scattering, True, anisotropic_bone)

    depth = max_depth - array.z
    ex = array.element_x
    x_img, z_img = _image_axes(array, max_depth, pixel, c0, f0)
    plan = []  # (emisión, retardos, apodización, instante hasta el que se borra cada elemento)
    if mode == "plane":
        for ang in angles:
            delays = array.plane_wave_delays(ang, c0)
            plan.append((Transmit("plane", ang), delays, tx_apodization, delays))
    elif mode == "focused":
        zf = focus_depth if focus_depth is not None else array.z + 0.5 * depth
        xs_line = ex[::max(line_step, 1)]
        edges = np.concatenate([[-np.inf], 0.5 * (xs_line[1:] + xs_line[:-1]), [np.inf]])
        for k, xl in enumerate(xs_line):
            delays, apod = array.focused_subaperture(xl, zf, tx_f_number, c0, tx_window)
            if tx_apodization is not None:
                apod = apod * tx_apodization
            mask = ((x_img >= edges[k]) & (x_img < edges[k + 1]))[None, :]
            active = apod > 0
            t_focus = float(np.hypot(ex[active] - xl, zf - array.z).max() / c0)
            tx = Transmit("focused", focus=(xl, zf), t_focus=t_focus, pixel_mask=mask)
            plan.append((tx, delays, apod, np.full(ex.size, delays[active].max())))
    else:
        for s in range(0, ex.size, max(sta_step, 1)):
            plan.append((Transmit("element", element=s), np.zeros(ex.size), np.eye(ex.size)[s],
                         np.abs(ex - ex[s]) / c0))

    # Misma duración para todas las emisiones
    max_delay = max(float(p[1].max()) for p in plan)
    if mode == "plane":  # ida hasta max_depth y vuelta al elemento más lejano
        t_end = array.pulse_peak_time + max_delay + (depth + np.hypot(depth, array.aperture)) / 1400.0
    else:
        # Sólo cuentan los píxeles dentro de la apertura dinámica de recepción: |x - xe| <= depth/(2F#)
        t_end = (array.pulse_peak_time + array.pulse_length + max_delay
                 + 2.0 * depth * np.sqrt(1.0 + 0.25 / f_number**2) / 1400.0)

    rf_list, transmits, results, transmitted = [], [], [], None
    blank = array.pulse_peak_time + 0.6 * array.pulse_length
    t_sim = 0.0
    for k, (tx, delays, apod, blank_delay) in enumerate(plan):
        solver = ElasticSolver(phantom, f0, npml=npml, n_mechanisms=n_mechanisms, scattering=scattering,
                               anisotropic_bone=anisotropic_bone, medium=medium)
        nt = int(np.ceil(t_end / solver.dt))
        t = solver.time_axis(nt)
        solver.add_source(array.transmit_source(g, t, delays, apod))
        solver.add_receiver("rf", array.receiver(g))
        if record_transmission and k == 0:
            solver.add_receiver("tx", line_receiver(g, bottom))
        if progress:
            print(f"Emisión {k + 1}/{len(plan)} ({tx.kind}): {nt} pasos, malla {g.nz}x{g.nx}", flush=True)
        res = solver.run(
            nt,
            snapshot_every=snapshot_every if k == 0 else None,
            snapshot_decimate=snapshot_decimate,
            snapshot_field=snapshot_field,
            track_peak=(k == 0),
            progress=progress,
        )
        t_sim += res.elapsed_s
        # Se borra el pulso emitido que cada elemento registra al disparar
        rf = array.receive_filter(res.traces["rf"], solver.dt)
        rf[t[None, :] < (blank_delay[:, None] + blank)] = 0.0
        rf_list.append(rf)
        transmits.append(tx)
        if record_transmission and k == 0:
            transmitted = res.traces["tx"]
        res.traces = {}
        results.append(res)

    if mode == "plane" and beamformer == "das" and layers is None:
        # Camino original (DAS de ondas planas): mismo resultado que antes de añadir las opciones
        img = bmode(rf_list, [tx.angle_deg for tx in transmits], t, ex, array.z, x_img, z_img, c0=c0,
                    t_peak=array.pulse_peak_time, tgc_db_per_cm=tgc_db_per_cm, f_number=f_number,
                    rx_window=rx_window)
    else:
        img = bmode_channels(rf_list, transmits, t, ex, array.z, x_img, z_img, c0=c0, t_peak=array.pulse_peak_time,
                             tgc_db_per_cm=tgc_db_per_cm, f_number=f_number, rx_window=rx_window,
                             beamformer=beamformer, layers=layers, **bf_kwargs)
    angles_out = [tx.angle_deg for tx in transmits] if mode == "plane" else []
    return UltrasoundScan(img, array, angles_out, t, rf_list, results, transmitted, transmits, t_sim)


def _image_axes(array: LinearArray, max_depth: float, pixel: float | None, c0: float, f0: float):
    px = pixel or c0 / f0 / 4.0
    ex = array.element_x
    return np.arange(ex.min(), ex.max() + px / 2, px), np.arange(array.z + 0.5e-3, max_depth, px)


def simulate_bmode(
    phantom: Phantom,
    f0: float = 3e6,
    array: LinearArray | None = None,
    angles: tuple[float, ...] = (0.0,),
    c0: float = 1540.0,
    max_depth: float | None = None,
    npml: int = 20,
    n_mechanisms: int = 2,
    scattering: float = 1.0,
    snapshot_every: int | None = None,
    snapshot_decimate: int = 2,
    snapshot_field: str = "p",
    pixel: float | None = None,
    tgc_db_per_cm: float | None = None,
    record_transmission: bool = False,
    progress: bool = False,
    anisotropic_bone: bool = False,
    **options,
) -> UltrasoundScan:
    """Simula una ecografía de onda plana (una simulación FDTD por ángulo) y reconstruye el modo B.

    ``max_depth``: profundidad máxima de la imagen (por defecto, la del fantoma).
    ``tgc_db_per_cm``: por defecto 0.5 dB/(cm·MHz) · f0 (compensación típica).
    ``anisotropic_bone``: cortical transversalmente isótropa (``ElasticSolver``).
    ``options``: opciones de :func:`simulate_scan` (``rx_window``, ``beamformer``,
    ``layers``, ``tx_apodization``, ``f_number``...). Para emisión focalizada o STA, usar
    directamente :func:`simulate_scan`.
    """
    return simulate_scan(phantom, f0, "plane", array, angles, c0=c0, max_depth=max_depth, npml=npml,
                         n_mechanisms=n_mechanisms, scattering=scattering, snapshot_every=snapshot_every,
                         snapshot_decimate=snapshot_decimate, snapshot_field=snapshot_field, pixel=pixel,
                         tgc_db_per_cm=tgc_db_per_cm, record_transmission=record_transmission, progress=progress,
                         anisotropic_bone=anisotropic_bone, **options)
