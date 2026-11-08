"""Perfiles rugosos aleatorios para las interfaces entre tejidos.

Las superficies reales (piel, periostio, superficie de fractura) no son
lisas: su rugosidad dispersa la onda incidente en lugar de reflejarla como un
espejo. Se modelan como procesos gaussianos con desviación típica ``rms`` y
longitud de correlación ``corr_len`` (autocorrelación ``exp(-x²/corr_len²)``),
sintetizados filtrando ruido blanco en el dominio de Fourier.

El perfil se genera con un paso fijo independiente de la malla, de modo que la
misma semilla da la misma anatomía a cualquier resolución.
"""

from __future__ import annotations

import numpy as np


class RoughProfile:
    """Perfil ``h(s)`` evaluable en cualquier coordenada ``s`` (metros)."""

    def __init__(self, length: float, rms: float, corr_len: float, rng: np.random.Generator, periodic: bool = False):
        self.length = length
        self.periodic = periodic
        if rms <= 0 or corr_len <= 0:
            self._s = np.array([0.0, length])
            self._h = np.zeros(2)
            return
        ds = min(corr_len / 8.0, length / 64.0)
        # Sin periodicidad se sintetiza un tramo más largo y se recorta, para
        # que los extremos no queden correlacionados entre sí.
        synth_len = length if periodic else length + 4.0 * corr_len
        n = max(int(np.ceil(synth_len / ds)), 16)
        ds = synth_len / n
        k = 2.0 * np.pi * np.fft.rfftfreq(n, d=ds)
        spectrum = np.exp(-((k * corr_len) ** 2) / 8.0)  # raíz de la PSD gaussiana
        noise = np.fft.rfft(rng.standard_normal(n))
        h = np.fft.irfft(noise * spectrum, n=n)
        h *= rms / (h.std() + 1e-30)
        s = np.arange(n) * ds
        if periodic:
            self._s, self._h = np.append(s, synth_len), np.append(h, h[0])
        else:
            keep = s <= length + ds
            self._s, self._h = s[keep], h[keep]

    def __call__(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if self.periodic:
            s = np.mod(s, self.length)
        return np.interp(s, self._s, self._h)
