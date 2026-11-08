"""Exploración ecográfica completa: simulación + reconstrucción en una llamada."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.signals import pulse_duration
from ..phantoms.phantom import Phantom
from .beamforming import BModeImage, bmode
from .elastic_fdtd import ElasticSolver, SimulationResult
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


def simulate_bmode(
    phantom: Phantom,
    f0: float = 3e6,
    array: LinearArray | None = None,
    angles: tuple[float, ...] = (0.0,),
    c0: float = 1540.0,
    max_depth: float | None = None,
    npml: int = 20,
    n_mechanisms: int = 1,
    scattering: float = 1.0,
    snapshot_every: int | None = None,
    snapshot_decimate: int = 2,
    snapshot_field: str = "p",
    pixel: float | None = None,
    tgc_db_per_cm: float | None = None,
    record_transmission: bool = False,
    progress: bool = False,
) -> UltrasoundScan:
    """Simula una ecografía de onda plana (una simulación FDTD por ángulo) y reconstruye el modo B.

    ``max_depth``: profundidad máxima de la imagen (por defecto, la del fantoma).
    ``tgc_db_per_cm``: por defecto 0.5 dB/(cm·MHz) · f0 (compensación típica).
    """
    g = phantom.grid
    array = array or default_array(phantom, f0, npml, c0)
    bottom = g.z0 + g.depth - (npml + 3) * g.dx
    max_depth = min(max_depth or bottom, bottom)
    if tgc_db_per_cm is None:
        tgc_db_per_cm = 0.5 * f0 / 1e6

    rf_list, results, transmitted = [], [], None
    # Misma duración para todos los ángulos: ida hasta max_depth y vuelta al elemento más lejano
    max_delay = max(array.plane_wave_delays(a, c0).max() for a in angles)
    depth = max_depth - array.z
    t_end = array.pulse_peak_time + max_delay + (depth + np.hypot(depth, array.aperture)) / 1400.0
    for k, ang in enumerate(angles):
        solver = ElasticSolver(phantom, f0, npml=npml, n_mechanisms=n_mechanisms, scattering=scattering)
        delays = array.plane_wave_delays(ang, c0)
        nt = int(np.ceil(t_end / solver.dt))
        t = solver.time_axis(nt)
        solver.add_source(array.transmit_source(g, t, delays))
        solver.add_receiver("rf", array.receiver(g))
        if record_transmission and k == 0:
            solver.add_receiver("tx", line_receiver(g, bottom))
        if progress:
            print(f"Emisión {k + 1}/{len(angles)} (ángulo {ang:+.1f}°): {nt} pasos, malla {g.nz}x{g.nx}")
        res = solver.run(
            nt,
            snapshot_every=snapshot_every if k == 0 else None,
            snapshot_decimate=snapshot_decimate,
            snapshot_field=snapshot_field,
            track_peak=(k == 0),
            progress=progress,
        )
        # Se borra el pulso emitido que cada elemento registra al disparar
        rf = res.traces["rf"]
        blank = array.pulse_peak_time + 0.6 * pulse_duration(array.f0, array.bandwidth)
        rf[t[None, :] < (delays[:, None] + blank)] = 0.0
        rf_list.append(rf)
        if record_transmission and k == 0:
            transmitted = res.traces["tx"]
        res.traces = {}
        results.append(res)

    px = pixel or c0 / f0 / 4.0
    ex = array.element_x
    x_img = np.arange(ex.min(), ex.max() + px / 2, px)
    z_img = np.arange(array.z + 0.5e-3, max_depth, px)
    img = bmode(rf_list, list(angles), t, ex, array.z, x_img, z_img, c0=c0, t_peak=array.pulse_peak_time,
                tgc_db_per_cm=tgc_db_per_cm)
    return UltrasoundScan(img, array, list(angles), t, rf_list, results, transmitted)
