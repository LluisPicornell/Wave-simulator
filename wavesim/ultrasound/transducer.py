"""Sonda lineal de ultrasonido (array de elementos piezoeléctricos)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.grid import Grid2D
from ..core.signals import gaussian_pulse, pulse_duration, pulse_peak_time
from .elastic_fdtd import Receiver, Source

WINDOWS = ("rect", "hann", "hamming", "tukey")


def window_function(u: np.ndarray, kind: str = "hann") -> np.ndarray:
    """Ventana de apertura en la coordenada normalizada ``u`` ∈ [-1, 1] (0 fuera).

    ``u = 0`` es el centro de la apertura. ``tukey`` es la ventana de Tukey con
    el 50 % del ancho en cosenos (compromiso entre ``rect`` y ``hann``).
    """
    a = np.abs(np.asarray(u, float))
    inside = a <= 1.0
    if kind == "rect":
        w = np.ones_like(a)
    elif kind == "hann":
        w = 0.5 * (1.0 + np.cos(np.pi * a))
    elif kind == "hamming":
        w = 0.54 + 0.46 * np.cos(np.pi * a)
    elif kind == "tukey":
        w = np.where(a < 0.5, 1.0, 0.5 * (1.0 + np.cos(2.0 * np.pi * (a - 0.5))))
    else:
        raise ValueError(f"Ventana desconocida: {kind!r} (opciones: {', '.join(WINDOWS)})")
    return np.where(inside, w, 0.0)


def apodization(n: int, kind: str = "hann") -> np.ndarray:
    """Pesos de apodización para ``n`` elementos (extremos con peso pequeño, no nulo, en ``hann``)."""
    u = (np.arange(n) - (n - 1) / 2) / (n / 2.0)
    return window_function(u, kind)


@dataclass(frozen=True)
class ElementResponse:
    """Respuesta impulsional electro-mecánica del elemento (la misma en emisión y recepción).

    Se modela como una portadora ``sin(2π f0 τ)`` de ``n_cycles`` ciclos con
    ventana (``kind="windowed"``) o con envolvente gaussiana de ancho de banda
    fraccional ``bandwidth`` a -6 dB (``kind="gaussian"``; igual que
    ``gaussian_pulse``). Es impar y centrada en ``τ = 0``: así el filtro de
    recepción, aplicado como convolución centrada, no retrasa el eco.

    En emisión el elemento imprime directamente esta forma (excitación
    impulsiva filtrada por el transductor); en recepción se convoluciona el eco
    con ella. El eco medido es por tanto ``h ∗ (propagación) ∗ h``.
    """

    f0: float = 3e6
    kind: str = "windowed"
    n_cycles: float = 2.0
    window: str = "hann"
    bandwidth: float = 0.6

    @property
    def duration(self) -> float:
        """Duración total (s) de la respuesta."""
        if self.kind == "windowed":
            return self.n_cycles / self.f0
        return pulse_duration(self.f0, self.bandwidth)

    @property
    def peak_time(self) -> float:
        """Instante del centro de la respuesta cuando se usa como pulso emitido."""
        return self.duration / 2.0

    def evaluate(self, tau: np.ndarray) -> np.ndarray:
        """Valor de la respuesta en el tiempo ``tau`` relativo a su centro (0 fuera de su soporte)."""
        tau = np.asarray(tau, float)
        if self.kind == "windowed":
            u = tau / (self.duration / 2.0)
            return window_function(u, self.window) * np.sin(2.0 * np.pi * self.f0 * tau)
        if self.kind == "gaussian":
            half = self.duration / 2.0
            pulse = gaussian_pulse(tau + pulse_peak_time(self.f0, self.bandwidth), self.f0, self.bandwidth)
            return np.where(np.abs(tau) <= half, pulse, 0.0)
        raise ValueError(f"Tipo de respuesta desconocido: {self.kind!r}")

    def impulse(self, dt: float) -> np.ndarray:
        """Respuesta muestreada cada ``dt`` con un número impar de muestras, centrada."""
        half = int(np.ceil(self.duration / 2.0 / dt))
        return self.evaluate(np.arange(-half, half + 1) * dt)

    def spectrum(self, f: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Módulo de la transformada de Fourier de la respuesta (referencia para el ancho de banda)."""
        dt = dt or 1.0 / (40.0 * self.f0)
        h = self.impulse(dt)
        tau = np.arange(h.size) * dt
        return np.abs(np.array([np.sum(h * np.exp(-2j * np.pi * fk * tau)) * dt for fk in np.atleast_1d(f)]))


@dataclass
class LinearArray:
    """Sonda lineal con ``n_elements`` elementos separados ``pitch`` metros.

    Se coloca horizontalmente a la profundidad ``z`` (dentro del gel de
    acoplamiento) y centrada en ``x_center``. Cada elemento emite un pulso
    gaussiano de frecuencia central ``f0`` y ancho de banda fraccional
    ``bandwidth``, y después escucha los ecos (presión media sobre su anchura).

    Opciones de realismo (todas desactivadas por defecto):
      - ``kerf``: separación entre elementos; si se da y no hay ``element_width``,
        el ancho es ``pitch - kerf`` (por defecto el elemento mide 0.85·pitch).
      - ``response``: :class:`ElementResponse` aplicada en emisión y en recepción
        (``receive_filter``). Si se da, sustituye al pulso gaussiano ``f0``/``bandwidth``.
    """

    n_elements: int = 64
    pitch: float = 0.3e-3
    element_width: float | None = None
    z: float = 1.0e-3
    x_center: float = 0.0
    f0: float = 3e6
    bandwidth: float = 0.6
    amplitude: float = 1e5  # Pa (no influye en la física lineal, sólo en la escala)
    kerf: float | None = None
    response: ElementResponse | None = None

    @property
    def width(self) -> float:
        """Ancho efectivo de cada elemento (m)."""
        if self.element_width is not None:
            return self.element_width
        if self.kerf is not None:
            return max(self.pitch - self.kerf, 0.0)
        return 0.85 * self.pitch

    @property
    def element_x(self) -> np.ndarray:
        return self.x_center + (np.arange(self.n_elements) - (self.n_elements - 1) / 2) * self.pitch

    @property
    def aperture(self) -> float:
        return self.n_elements * self.pitch

    @property
    def pulse_peak_time(self) -> float:
        if self.response is not None:
            return self.response.peak_time
        return pulse_peak_time(self.f0, self.bandwidth)

    @property
    def pulse_length(self) -> float:
        """Duración efectiva del pulso emitido (s)."""
        if self.response is not None:
            return self.response.duration
        return pulse_duration(self.f0, self.bandwidth)

    def pulse(self, t: np.ndarray) -> np.ndarray:
        """Pulso emitido por un elemento sin retardo, muestreado en ``t``."""
        if self.response is not None:
            return self.response.evaluate(t - self.response.peak_time)
        return gaussian_pulse(t, self.f0, self.bandwidth)

    def round_trip_pulse(self, dt: float) -> np.ndarray:
        """Pulso de ida y vuelta de un reflector ideal: el emitido, filtrado por la recepción.

        Muestreado cada ``dt``; es la referencia para la resolución axial esperada.
        """
        pad = int(np.ceil(self.pulse_length / dt)) + 2  # holgura: la convolución alarga el pulso
        n = int(np.ceil(2 * self.pulse_peak_time / dt)) + 2 * pad + 1
        p = self.pulse((np.arange(n) - pad) * dt)
        return self.receive_filter(p[None, :], dt)[0]

    def receive_filter(self, rf: np.ndarray, dt: float) -> np.ndarray:
        """Convoluciona el eco con la respuesta del elemento (sin efecto si no hay ``response``).

        Convolución centrada (``h`` es impar y simétrica en soporte), de modo que no se
        introduce retardo. La escala no se normaliza: sólo importa la forma.
        """
        if self.response is None:
            return rf
        h = self.response.impulse(dt)
        n = rf.shape[-1] + h.size
        out = np.fft.irfft(np.fft.rfft(rf, n, axis=-1) * np.fft.rfft(h, n), n, axis=-1)
        k = h.size // 2
        return out[..., k:k + rf.shape[-1]]

    def element_cells(self, grid: Grid2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Celdas ``(j, i)`` que forman cada elemento y el índice de elemento de cada una."""
        w = self.width
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

    def focused_delays(self, focus_x: float, focus_z: float, c0: float,
                       active: np.ndarray | None = None) -> np.ndarray:
        """Retardos de emisión para focalizar en ``(focus_x, focus_z)``.

        ``active`` (máscara booleana) limita la apertura de emisión: el retardo
        se refiere al elemento activo más lejano, y los inactivos quedan con
        retardo 0 (conviene anular su apodización).
        """
        d = np.hypot(self.element_x - focus_x, focus_z - self.z)
        if active is None:
            return (d.max() - d) / c0
        return np.where(active, (d[active].max() - d) / c0, 0.0)

    def focused_subaperture(self, focus_x: float, focus_z: float, f_number: float, c0: float,
                            window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
        """Retardos y apodización de una emisión focalizada con apertura de ``F# = f_number``.

        Se activan los elementos con ``|x_e - focus_x| <= (focus_z - z)/(2·F#)``
        (centrados en la línea) con la ventana ``window``. Devuelve ``(delays, apod)``.
        """
        half = max(focus_z - self.z, 1e-9) / (2.0 * f_number)
        u = (self.element_x - focus_x) / half
        apod = window_function(u, window)
        # Con la ventana muestreada, el primer y último elemento de ``hann`` pueden valer 0
        active = np.abs(u) <= 1.0
        if not active.any():
            active[np.argmin(np.abs(self.element_x - focus_x))] = True
            apod = active.astype(float)
        return self.focused_delays(focus_x, focus_z, c0, active), apod

    def single_element_source(self, grid: Grid2D, t: np.ndarray, element: int) -> Source:
        """Emisión de un único elemento (apertura sintética, STA)."""
        apod = np.zeros(self.n_elements)
        apod[element] = 1.0
        return self.transmit_source(grid, t, np.zeros(self.n_elements), apod)

    def transmit_source(self, grid: Grid2D, t: np.ndarray, delays: np.ndarray,
                        apodization: np.ndarray | None = None) -> Source:
        j, i, ch = self.element_cells(grid)
        apod = np.ones(self.n_elements) if apodization is None else apodization
        sig = np.stack([
            self.amplitude * apod[e] * self.pulse(t - delays[e])
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
