"""Señales de excitación y utilidades de procesado de señal (sin SciPy)."""

from __future__ import annotations

import numpy as np


def gaussian_pulse(t: np.ndarray, f0: float, bandwidth: float = 0.6, t0: float | None = None) -> np.ndarray:
    """Pulso sinusoidal con envolvente gaussiana (típico de un transductor).

    ``bandwidth`` es el ancho de banda fraccional a -6 dB (0.6 = 60 % de f0).
    Si ``t0`` es None, el pico del pulso se coloca en ``pulse_peak_time``.
    """
    sigma_t = _sigma_t(f0, bandwidth)
    if t0 is None:
        t0 = 4.0 * sigma_t
    tau = t - t0
    return np.exp(-0.5 * (tau / sigma_t) ** 2) * np.sin(2.0 * np.pi * f0 * tau)


def pulse_peak_time(f0: float, bandwidth: float = 0.6) -> float:
    """Instante del centro del pulso devuelto por :func:`gaussian_pulse`."""
    return 4.0 * _sigma_t(f0, bandwidth)


def pulse_duration(f0: float, bandwidth: float = 0.6) -> float:
    """Duración total efectiva del pulso (de 0 a 2·t0)."""
    return 8.0 * _sigma_t(f0, bandwidth)


def _sigma_t(f0: float, bandwidth: float) -> float:
    # Ancho a -6 dB del espectro gaussiano: BW*f0 = 2*sqrt(2 ln 2)*sigma_f
    sigma_f = bandwidth * f0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return 1.0 / (2.0 * np.pi * sigma_f)


def ramped_sine(t: np.ndarray, f0: float, ramp_cycles: float = 3.0) -> np.ndarray:
    """Seno continuo con arranque suave (evita el transitorio de encendido)."""
    ramp_t = ramp_cycles / f0
    env = np.where(t < ramp_t, 0.5 * (1.0 - np.cos(np.pi * np.clip(t / ramp_t, 0, 1))), 1.0)
    return env * np.sin(2.0 * np.pi * f0 * t)


def analytic_signal(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Señal analítica (transformada de Hilbert) vía FFT, como ``scipy.signal.hilbert``."""
    x = np.asarray(x)
    n = x.shape[axis]
    spec = np.fft.fft(x, axis=axis)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    shape = [1] * x.ndim
    shape[axis] = n
    return np.fft.ifft(spec * h.reshape(shape), axis=axis)


def envelope(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.abs(analytic_signal(x, axis=axis))


def amplitude_at(signal: np.ndarray, dt: float, f: float) -> complex:
    """Componente de Fourier de ``signal`` a la frecuencia ``f`` (DFT de un solo punto)."""
    t = np.arange(signal.shape[-1]) * dt
    return complex(np.sum(signal * np.exp(-2j * np.pi * f * t)) * dt)
