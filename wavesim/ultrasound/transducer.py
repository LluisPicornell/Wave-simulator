"""Sonda lineal de ultrasonido (array de elementos piezoeléctricos)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import Grid2D
from ..core.signals import gaussian_pulse, pulse_peak_time
from .elastic_fdtd import Receiver, Source


@dataclass
class LinearArray:
    """Sonda lineal con ``n_elements`` elementos separados ``pitch`` metros.

    Se coloca horizontalmente a la profundidad ``z`` (dentro del gel de
    acoplamiento) y centrada en ``x_center``. Cada elemento emite un pulso
    gaussiano de frecuencia central ``f0`` y ancho de banda fraccional
    ``bandwidth``, y después escucha los ecos (presión media sobre su anchura).
    """

    n_elements: int = 64
    pitch: float = 0.3e-3
    element_width: float | None = None
    z: float = 1.0e-3
    x_center: float = 0.0
    f0: float = 3e6
    bandwidth: float = 0.6
    amplitude: float = 1e5  # Pa (no influye en la física lineal, sólo en la escala)

    @property
    def element_x(self) -> np.ndarray:
        return self.x_center + (np.arange(self.n_elements) - (self.n_elements - 1) / 2) * self.pitch

    @property
    def aperture(self) -> float:
        return self.n_elements * self.pitch

    @property
    def pulse_peak_time(self) -> float:
        return pulse_peak_time(self.f0, self.bandwidth)

    def element_cells(self, grid: Grid2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Celdas ``(j, i)`` que forman cada elemento y el índice de elemento de cada una."""
        w = self.element_width if self.element_width is not None else 0.85 * self.pitch
        j = grid.index_of(0.0, self.z)[0]
        x = grid.x
        js, is_, ch = [], [], []
        for e, xe in enumerate(self.element_x):
            sel = np.nonzero(np.abs(x - xe) <= w / 2 + 1e-12)[0]
            if sel.size == 0:  # elemento más estrecho que una celda
                sel = np.array([grid.index_of(xe, self.z)[1]])
            js.append(np.full(sel.size, j))
            is_.append(sel)
            ch.append(np.full(sel.size, e))
        return np.concatenate(js), np.concatenate(is_), np.concatenate(ch)

    def plane_wave_delays(self, angle_deg: float, c0: float) -> np.ndarray:
        """Retardos de emisión para una onda plana inclinada ``angle_deg``."""
        s = np.sin(np.radians(angle_deg))
        xr = self.element_x.min() if s >= 0 else self.element_x.max()
        return (self.element_x - xr) * s / c0

    def focused_delays(self, focus_x: float, focus_z: float, c0: float) -> np.ndarray:
        d = np.hypot(self.element_x - focus_x, focus_z - self.z)
        return (d.max() - d) / c0

    def transmit_source(self, grid: Grid2D, t: np.ndarray, delays: np.ndarray,
                        apodization: np.ndarray | None = None) -> Source:
        j, i, ch = self.element_cells(grid)
        apod = np.ones(self.n_elements) if apodization is None else apodization
        sig = np.stack([
            self.amplitude * apod[e] * gaussian_pulse(t - delays[e], self.f0, self.bandwidth)
            for e in range(self.n_elements)
        ])
        return Source(j, i, ch, sig, kind="pressure")

    def receiver(self, grid: Grid2D) -> Receiver:
        j, i, ch = self.element_cells(grid)
        return Receiver(j, i, ch, self.n_elements, "p")


def line_receiver(grid: Grid2D, z: float, x_min: float | None = None, x_max: float | None = None,
                  spacing: float | None = None) -> Receiver:
    """Fila de receptores puntuales a la profundidad ``z`` (p. ej. para medir transmisión)."""
    j = grid.index_of(0.0, z)[0]
    x = grid.x
    lo = x_min if x_min is not None else x[0]
    hi = x_max if x_max is not None else x[-1]
    idx = np.nonzero((x >= lo) & (x <= hi))[0]
    if spacing is not None:
        step = max(int(round(spacing / grid.dx)), 1)
        idx = idx[::step]
    return Receiver(np.full(idx.size, j), idx, np.arange(idx.size), idx.size, "p")
