"""Perfiles rugosos aleatorios para las interfaces entre tejidos.

Las superficies reales (piel, periostio, superficie de fractura) no son
lisas: su rugosidad dispersa la onda incidente en lugar de reflejarla como un
espejo. Se sintetizan filtrando ruido blanco en el dominio de Fourier con dos
espectros posibles, ambos con desviación típica ``rms``:

- ``"gaussian"`` (el de siempre): autocorrelación ``exp(-x²/corr_len²)``.
  Es una ondulación de una sola escala: el espectro cae como una gaussiana
  por encima de ``1/corr_len`` y no hay rugosidad fina.
- ``"power_law"``: espectro de von Kármán ``P(k) ∝ (1 + k² ξ²)^-(H + 1/2)``
  con ``ξ = corr_len``. Por debajo de ``1/ξ`` es plano y por encima cae como
  ``k^-(1 + 2H)``: es un perfil autoafín (fractal, de dimensión ``2 - H``)
  con rugosidad en todas las escalas entre ``ξ`` y ``min_wavelength``. ``H``
  es el exponente de Hurst (0 < H < 1; menor = más rugoso a escala fina).
  Las superficies de fractura de materiales muy distintos son autoafines con
  ``H ≈ 0.8`` (Bouchaud, J Phys Condens Matter 9:4319, 1997).

El perfil se genera con un paso fijo independiente de la malla, de modo que la
misma semilla da la misma anatomía a cualquier resolución.
"""

from __future__ import annotations

import numpy as np

#: Longitud de onda más corta de los perfiles en ley de potencia (m). Por
#: debajo de ~λ/20 del ultrasonido más agudo que se simula (10 MHz → 150 µm)
#: la rugosidad ya no cambia la reflexión, y alargar el espectro sólo encarece
#: la síntesis.
MIN_WAVELENGTH = 20e-6


class RoughProfile:
    """Perfil ``h(s)`` evaluable en cualquier coordenada ``s`` (metros)."""

    def __init__(self, length: float, rms: float, corr_len: float, rng: np.random.Generator, periodic: bool = False,
                 spectrum: str = "gaussian", hurst: float = 0.8, min_wavelength: float = MIN_WAVELENGTH):
        self.length = length
        self.periodic = periodic
        if spectrum not in ("gaussian", "power_law"):
            raise ValueError(f"espectro desconocido '{spectrum}' (use 'gaussian' o 'power_law')")
        if rms <= 0 or corr_len <= 0:
            self._s = np.array([0.0, length])
            self._h = np.zeros(2)
            return
        ds = min(corr_len / 8.0, length / 64.0)
        if spectrum == "power_law":
            ds = min(ds, min_wavelength / 2.0)
        # Sin periodicidad se sintetiza un tramo más largo y se recorta, para
        # que los extremos no queden correlacionados entre sí.
        synth_len = length if periodic else length + 4.0 * corr_len
        n = max(int(np.ceil(synth_len / ds)), 16)
        ds = synth_len / n
        k = 2.0 * np.pi * np.fft.rfftfreq(n, d=ds)
        if spectrum == "gaussian":
            amplitude = np.exp(-((k * corr_len) ** 2) / 8.0)  # raíz de la PSD gaussiana
        else:
            amplitude = (1.0 + (k * corr_len) ** 2) ** (-(hurst + 0.5) / 2.0)
            amplitude[k > 2.0 * np.pi / min_wavelength] = 0.0
            amplitude[0] = 0.0  # media nula: la posición media la fija la capa
        noise = np.fft.rfft(rng.standard_normal(n))
        h = np.fft.irfft(noise * amplitude, n=n)
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


class SumProfile:
    """Suma de perfiles: ondulación de gran escala más rugosidad fina (rugosidad multiescala)."""

    def __init__(self, *profiles):
        self.profiles = profiles

    def __call__(self, s) -> np.ndarray:
        return sum(p(s) for p in self.profiles)
