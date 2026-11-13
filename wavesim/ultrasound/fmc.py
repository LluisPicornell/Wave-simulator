"""Captura de matriz completa (FMC) y reconstrucción con cualquier subconjunto de emisores y receptores.

En una FMC cada elemento de la sonda emite solo y todos reciben: se obtienen
``n_tx × n_rx`` trazas RF ``S[i, j, t]`` (emisor ``i``, receptor ``j``). Como el
medio es lineal, de una sola FMC se reconstruye a posteriori, sin volver a simular:

- la apertura sintética (STA) con cualquier subconjunto de emisores y receptores
  (:func:`delayed_pairs` + :func:`pair_image`);
- cualquier emisión simultánea de varios elementos con retardos, p. ej. una onda
  plana, sumando las emisiones de un elemento retardadas (:func:`synthesize_rf`);
- un transductor monoestático (la diagonal ``S[i, i]``) o un elemento ancho
  (suma de un bloque de elementos contiguos en emisión y en recepción).

Ruido electrónico. La suma con retardos sólo es exacta sin ruido: en una emisión
simultánea real hay **un** ruido por receptor y disparo, no la suma de los
ruidos de ``n_tx`` disparos. Por eso el ruido se añade después de sintetizar cada
disparo físico (:func:`band_noise`), nunca a la FMC antes de sumar emisiones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..core.signals import analytic_signal
from ..phantoms.phantom import Phantom
from .elastic_fdtd import ElasticSolver
from .medium import ElasticMedium, elastic_medium
from .transducer import LinearArray, window_function


@dataclass
class FMCData:
    """Trazas de una captura de matriz completa.

    ``rf[k, j]``: señal del receptor ``j`` cuando emite el elemento ``tx_elements[k]``,
    muestreada en ``t`` (sin borrar la onda directa entre elementos). ``t_peak``:
    instante del centro del pulso emitido (origen de los tiempos de vuelo).
    """

    rf: np.ndarray
    t: np.ndarray
    tx_elements: np.ndarray
    element_x: np.ndarray
    array_z: float
    t_peak: float
    pulse_length: float
    elapsed_s: float = 0.0

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def save(self, path) -> None:
        np.savez_compressed(path, rf=self.rf, t=self.t, tx_elements=self.tx_elements, element_x=self.element_x,
                            scalars=np.array([self.array_z, self.t_peak, self.pulse_length, self.elapsed_s]))

    @classmethod
    def load(cls, path) -> FMCData:
        d = np.load(path)
        z, tp, pl, el = d["scalars"]
        return cls(d["rf"], d["t"], d["tx_elements"], d["element_x"], float(z), float(tp), float(pl), float(el))


def acquire_fmc(phantom: Phantom, array: LinearArray, f0: float, tx_elements=None, t_end: float | None = None,
                npml: int = 20, medium: ElasticMedium | None = None, n_mechanisms: int = 2,
                scattering: float = 1.0, decimate: int = 1, progress: bool = False) -> FMCData:
    """Simula la FMC: una simulación FDTD por elemento emisor de ``tx_elements`` (por defecto, todos).

    ``t_end``: duración de cada emisión (por defecto, ida y vuelta hasta el fondo del
    dominio y al elemento más lejano a 1400 m/s). ``decimate``: se guarda una muestra
    de cada ``decimate`` (el paso del FDTD suele ser 30-40 veces menor que el periodo;
    con 2-4 sigue habiendo de sobra para la banda del pulso).
    """
    g = phantom.grid
    tx_elements = np.arange(array.n_elements) if tx_elements is None else np.asarray(tx_elements)
    if t_end is None:
        depth = g.z0 + g.depth - npml * g.dx - array.z
        t_end = array.pulse_peak_time + array.pulse_length + (depth + np.hypot(depth, array.aperture)) / 1400.0
    if medium is None:  # el medio se construye una vez y lo comparten todas las emisiones
        medium = elastic_medium(phantom, f0, n_mechanisms, scattering)
    rf, t, elapsed = None, None, 0.0
    t0 = time.time()
    for k, e in enumerate(tx_elements):
        solver = ElasticSolver(phantom, f0, npml=npml, n_mechanisms=n_mechanisms, scattering=scattering,
                               medium=medium)
        nt = int(np.ceil(t_end / solver.dt))
        tt = solver.time_axis(nt)
        apod = np.zeros(array.n_elements)
        apod[e] = 1.0
        solver.add_source(array.transmit_source(g, tt, np.zeros(array.n_elements), apod))
        solver.add_receiver("rf", array.receiver(g))
        res = solver.run(nt, track_peak=False)
        tr = array.receive_filter(res.traces["rf"], solver.dt)[:, ::decimate]
        if rf is None:
            t = tt[::decimate]
            rf = np.zeros((tx_elements.size, array.n_elements, t.size), np.float32)
        rf[k] = tr
        elapsed += res.elapsed_s
        if progress:
            print(f"  FMC emisión {k + 1}/{tx_elements.size} ({time.time() - t0:.0f} s)", flush=True)
    return FMCData(rf, t, tx_elements, array.element_x.copy(), array.z, array.pulse_peak_time, array.pulse_length,
                   elapsed)


def fractional_delay(x: np.ndarray, delay: np.ndarray | float, dt: float) -> np.ndarray:
    """Retrasa ``x`` (último eje) ``delay`` segundos por desfase en frecuencia (exacto para señales de banda limitada).

    ``delay`` se difunde sobre los ejes anteriores. Se rellena con ceros hasta el doble
    de longitud para que el desplazamiento no dé la vuelta.
    """
    n = x.shape[-1]
    nf = 2 * n
    f = np.fft.rfftfreq(nf, dt)
    d = np.asarray(delay, float)[..., None]
    spec = np.fft.rfft(x, nf, axis=-1) * np.exp(-2j * np.pi * f * d)
    return np.fft.irfft(spec, nf, axis=-1)[..., :n]


def synthesize_rf(fmc: FMCData, delays: np.ndarray, apodization: np.ndarray | None = None,
                  rx: np.ndarray | None = None) -> np.ndarray:
    """Señales ``(n_rx, nt)`` de una emisión simultánea de los emisores de la FMC con ``delays`` y ``apodization``.

    ``delays``/``apodization`` se dan por emisor de la FMC (``fmc.tx_elements``); ``rx``
    selecciona receptores. Por linealidad, ``Σ_i a_i S_i(t − τ_i)`` es lo que registraría
    la sonda disparando a la vez esos elementos (sin ruido; ver el docstring del módulo).
    """
    a = np.ones(fmc.tx_elements.size) if apodization is None else np.asarray(apodization, float)
    rf = fmc.rf if rx is None else fmc.rf[:, rx]
    use = np.nonzero(a)[0]
    out = fractional_delay(rf[use].astype(float), np.asarray(delays, float)[use, None], fmc.dt)
    return np.tensordot(a[use], out, axes=(0, 0))


def blank_direct(fmc: FMCData, c0: float = 1540.0, extra: float | None = None) -> np.ndarray:
    """Copia de las trazas con la onda directa entre elementos borrada.

    Se anula, para cada pareja ``(i, j)``, lo anterior a ``|x_i − x_j|/c0 + t_peak + extra``
    (``extra`` = 0.6 duraciones del pulso por defecto, como ``simulate_scan`` en STA).
    """
    extra = 0.6 * fmc.pulse_length if extra is None else extra
    xt = fmc.element_x[fmc.tx_elements]
    lim = np.abs(xt[:, None] - fmc.element_x[None, :]) / c0 + fmc.t_peak + extra
    rf = fmc.rf.copy()
    rf[fmc.t[None, None, :] < lim[:, :, None]] = 0.0
    return rf


def band_noise(shape: tuple[int, ...], dt: float, f0: float, bandwidth: float = 1.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Ruido gaussiano blanco filtrado a la banda del transductor, con valor eficaz 1.

    Filtro gaussiano centrado en ``f0`` con ancho fraccional ``bandwidth`` a -6 dB: el
    ruido de un canal real llega al conversor después del filtro analógico de la
    banda, así que sólo cuenta el que cae en ella. Se normaliza con el valor eficaz
    teórico (no el de la realización) para que el nivel no dependa de la muestra.
    """
    rng = rng or np.random.default_rng()
    n = shape[-1]
    f = np.fft.rfftfreq(n, dt)
    h = np.exp(-np.log(2.0) * (2.0 * (f - f0) / (bandwidth * f0)) ** 2)  # |H| = 1/2 en f0 ± bw·f0/2
    white = rng.standard_normal(shape)
    out = np.fft.irfft(np.fft.rfft(white, axis=-1) * h, n, axis=-1)
    gain = np.sqrt(np.sum(h ** 2) * 2.0 / n)  # potencia que deja pasar el filtro (aprox. sin DC/Nyquist)
    return out / gain


def element_weights(element_x: np.ndarray, array_z: float, x_img: np.ndarray, z_img: np.ndarray,
                    f_number: float = 0.5, window: str = "rect") -> np.ndarray:
    """Peso ``w[e, z, x]`` de cada elemento en cada píxel: apertura angular ``|x − x_e| ≤ (z − z_e)/(2F#)``."""
    X, Z = np.meshgrid(x_img, z_img)
    half = np.maximum(Z - array_z, 1e-6) / (2.0 * f_number)
    return np.stack([window_function(np.abs(X - xe) / half, window) for xe in element_x])


def delayed_pairs(rf: np.ndarray, fmc: FMCData, x_img: np.ndarray, z_img: np.ndarray, c0: float = 1540.0,
                  f_number: float = 0.5, window: str = "rect", tx_positions: np.ndarray | None = None,
                  rx_positions: np.ndarray | None = None) -> np.ndarray:
    """Cubo de pares retardados ``D[k, j, z, x]`` (complejo, ``complex64``) de unas trazas con forma de FMC.

    ``D[k, j]`` es la señal analítica de la pareja emisor ``k`` / receptor ``j`` en el
    instante del eco de cada píxel (``t_peak + τ_k + τ_j``), ya ponderada por la
    apertura angular de los dos elementos. Cualquier imagen STA de un subconjunto
    es entonces una suma de pares (:func:`pair_image`). ``tx_positions`` /
    ``rx_positions`` permiten reconstruir con posiciones distintas de las reales
    (p. ej. con el error de posición de una sonda movida a mano).
    """
    iq = analytic_signal(np.asarray(rf, float), axis=-1)
    t, dt, nt = fmc.t, fmc.dt, fmc.t.size
    xt = fmc.element_x[fmc.tx_elements] if tx_positions is None else np.asarray(tx_positions)
    xr = fmc.element_x if rx_positions is None else np.asarray(rx_positions)
    X, Z = np.meshgrid(x_img, z_img)
    depth = Z - fmc.array_z
    tof_t = np.stack([np.hypot(X - x, depth) / c0 for x in xt])
    tof_r = np.stack([np.hypot(X - x, depth) / c0 for x in xr])
    w_t = element_weights(xt, fmc.array_z, x_img, z_img, f_number, window)
    w_r = element_weights(xr, fmc.array_z, x_img, z_img, f_number, window)
    D = np.zeros((xt.size, xr.size, *X.shape), np.complex64)
    for k in range(xt.size):
        idx = (fmc.t_peak + tof_t[k][None] + tof_r - t[0]) / dt
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        valid = (i0 >= 0) & (i0 < nt - 1)
        i0 = np.clip(i0, 0, nt - 2)
        rows = np.arange(xr.size)[:, None, None]
        s = iq[k][rows, i0] * (1 - frac) + iq[k][rows, i0 + 1] * frac
        D[k] = np.where(valid, s, 0.0) * (w_t[k][None] * w_r)
    return D


def pair_image(D: np.ndarray, tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    """Imagen DAS (compleja) de la STA con los emisores ``tx`` y receptores ``rx`` (índices del cubo)."""
    return D[np.ix_(np.asarray(tx), np.asarray(rx))].sum(axis=(0, 1))


def pair_coherence(D: np.ndarray, tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    """Factor de coherencia de Mallart-Fink sobre los receptores.

    ``CF = |Σ_j C_j|² / (N Σ_j |C_j|²)`` con ``C_j = Σ_k D[k, j]`` (canal ``j`` ya sumado sobre los emisores).
    """
    C = D[np.ix_(np.asarray(tx), np.asarray(rx))].sum(axis=0)
    active = np.abs(C) > 0
    n = np.maximum(active.sum(0), 1)
    return np.abs(C.sum(0)) ** 2 / np.maximum(n * (np.abs(C) ** 2).sum(0), 1e-30)
