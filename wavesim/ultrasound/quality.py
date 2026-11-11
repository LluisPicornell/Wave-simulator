"""Métricas de calidad de la imagen de ultrasonido y medios de prueba (PSF, speckle, quiste).

Tres familias:

- **PSF** de un dispersor puntual: anchos a -6 dB (amplitud mitad) axial y lateral,
  frente a la teoría (``theoretical_axial_fwhm``, ``theoretical_lateral_fwhm``).
- **Speckle totalmente desarrollado** (muchos dispersores por celda de resolución): la
  envolvente sigue una distribución de Rayleigh, con SNR = media/desviación =
  √(π/(4-π)) ≈ 1.913 (``RAYLEIGH_SNR``), y su grano tiene el tamaño de la
  autocorrelación de la PSF (``speckle_grain_fwhm`` frente a ``predicted_grain_fwhm``).
- **Quiste anecoico**: ``cnr``, ``contrast_db`` y ``gcnr`` entre el interior y el fondo.

Los medios (``point_scatterer_medium``, ``speckle_medium``) son agua con anomalías de
densidad celda a celda, pensados para el solver elástico con ``n_mechanisms=0``.
"""

from __future__ import annotations

import numpy as np

from ..core.grid import Grid2D
from ..core.signals import analytic_signal
from ..phantoms.phantom import TISSUE_INDEX, Phantom
from .medium import ElasticMedium, elastic_medium
from .transducer import window_function

RAYLEIGH_SNR = float(np.sqrt(np.pi / (4.0 - np.pi)))  # 1.9131


# --------------------------------------------------------------------------
# Medios de prueba
# --------------------------------------------------------------------------
def water_phantom(grid: Grid2D) -> Phantom:
    """Fantoma de agua en toda la malla."""
    return Phantom(grid, np.full(grid.shape, TISSUE_INDEX["water"], np.int16))


def _water_medium(grid: Grid2D, f0: float) -> ElasticMedium:
    return elastic_medium(water_phantom(grid), f0, n_mechanisms=0, scattering=0.0, attenuation=False)


def point_scatterer_medium(grid: Grid2D, f0: float, points: list[tuple[float, float]],
                           contrast: float = 3.0) -> ElasticMedium:
    """Agua con dispersores puntuales: una celda por punto con la densidad multiplicada por ``1 + contrast``.

    Una celda de ``dx ≈ λ/10`` es mucho menor que la longitud de onda: la señal es la de un
    dispersor puntual, aunque débil (la intensidad escala como ``(contrast·dx²)²``).
    """
    m = _water_medium(grid, f0)
    for x, z in points:
        j, i = grid.index_of(x, z)
        m.rho[j, i] *= 1.0 + contrast
    return m


def speckle_medium(grid: Grid2D, f0: float, strength: float = 0.05, seed: int = 0,
                   anechoic: np.ndarray | None = None) -> ElasticMedium:
    """Agua con dispersores densos: ruido blanco gaussiano de densidad, de desviación ``strength``, en cada celda.

    Con ``dx ≈ λ/10`` hay del orden de cientos de dispersores por celda de resolución, es decir,
    speckle totalmente desarrollado. ``anechoic`` (máscara booleana) deja sin dispersores
    una región (quiste anecoico). ``strength`` pequeño mantiene la dispersión múltiple despreciable.
    """
    m = _water_medium(grid, f0)
    noise = np.random.default_rng(seed).standard_normal(grid.shape)
    if anechoic is not None:
        noise = np.where(anechoic, 0.0, noise)
    m.rho = m.rho * (1.0 + strength * noise)
    return m


def disk_mask(x: np.ndarray, z: np.ndarray, center: tuple[float, float], radius: float) -> np.ndarray:
    """Máscara booleana ``(nz, nx)`` de un disco en coordenadas de los ejes ``x``, ``z``."""
    X, Z = np.meshgrid(x, z)
    return (X - center[0]) ** 2 + (Z - center[1]) ** 2 <= radius**2


# --------------------------------------------------------------------------
# PSF
# --------------------------------------------------------------------------
def fwhm_1d(coord: np.ndarray, profile: np.ndarray, level: float = 0.5) -> float:
    """Ancho (en unidades de ``coord``) a la fracción ``level`` del máximo, alrededor del pico principal.

    Interpolación lineal en los cruces. ``level=0.5`` en amplitud es el ancho a -6 dB. Si el perfil
    no baja de ``level`` en alguno de los lados, devuelve ``nan``.
    """
    p = np.asarray(profile, float)
    k = int(np.argmax(p))
    thr = level * p[k]
    lo = k
    while lo > 0 and p[lo] > thr:
        lo -= 1
    hi = k
    while hi < p.size - 1 and p[hi] > thr:
        hi += 1
    if p[lo] > thr or p[hi] > thr:
        return float("nan")
    x_lo = coord[lo] + (thr - p[lo]) / (p[lo + 1] - p[lo]) * (coord[lo + 1] - coord[lo])
    x_hi = coord[hi - 1] + (p[hi - 1] - thr) / (p[hi - 1] - p[hi]) * (coord[hi] - coord[hi - 1])
    return float(x_hi - x_lo)


def psf_fwhm(envelope: np.ndarray, x: np.ndarray, z: np.ndarray, x0: float, z0: float,
             search: float = 1.5e-3) -> dict:
    """FWHM axial y lateral (a -6 dB de amplitud) de la PSF cerca de ``(x0, z0)``.

    Se localiza el máximo en un entorno de ``search`` (m) y se miden los perfiles que pasan por él.
    Devuelve ``axial``, ``lateral`` (m), la posición del pico y el desplazamiento respecto a ``(x0, z0)``.
    """
    X, Z = np.meshgrid(x, z)
    near = (np.abs(X - x0) <= search) & (np.abs(Z - z0) <= search)
    j, i = np.unravel_index(np.argmax(np.where(near, envelope, -np.inf)), envelope.shape)
    # Perfiles recortados alrededor del pico para que otro máximo de la imagen no se confunda con él
    cz, cx = np.abs(z - z[j]) <= 2.0 * search, np.abs(x - x[i]) <= 2.0 * search
    return {
        "axial": fwhm_1d(z[cz], envelope[cz, i]),
        "lateral": fwhm_1d(x[cx], envelope[j, cx]),
        "peak_x": float(x[i]), "peak_z": float(z[j]),
        "offset_x": float(x[i] - x0), "offset_z": float(z[j] - z0),
    }


def pulse_envelope_fwhm(pulse: np.ndarray, dt: float) -> float:
    """FWHM temporal (s) de la envolvente de un pulso, a la mitad de amplitud."""
    env = np.abs(analytic_signal(pulse))
    return fwhm_1d(np.arange(pulse.size) * dt, env)


def theoretical_axial_fwhm(pulse: np.ndarray, dt: float, c: float = 1540.0, spectral_tilt: float = 0.0) -> float:
    """FWHM axial (m) esperada: ``c/2`` por el ancho de la envolvente del pulso de ida y vuelta.

    ``pulse`` es el pulso de un reflector ideal (``LinearArray.round_trip_pulse``: emisión
    convolucionada con la recepción). La equivalencia tiempo-profundidad es ``Δz = c·Δt/2``.

    ``spectral_tilt``: exponente ``p`` de un factor ``(f/f_pico)^p`` que se aplica al espectro. En este
    simulador 2D un dispersor puntual es una línea: el campo dispersado de una celda pequeña va como
    ``k²`` (Born, contraste de densidad) por el ``k^-1/2`` de la función de Green 2D, es decir
    ``p = 1.5`` con onda plana en emisión (en 3D, y en Field II, sería otro). Con ``p = 0`` es el
    pulso nominal sin más.
    """
    if spectral_tilt:
        n = 1 << int(np.ceil(np.log2(pulse.size * 8)))
        spec = np.fft.rfft(pulse, n)
        f = np.fft.rfftfreq(n, dt)
        f_peak = f[np.argmax(np.abs(spec))]
        pulse = np.fft.irfft(spec * (f / f_peak) ** spectral_tilt, n)
    return 0.5 * c * pulse_envelope_fwhm(pulse, dt)


def theoretical_lateral_fwhm(element_x: np.ndarray, z: float, wavelength: float, f_number: float,
                             window: str = "hann", two_way: bool = False, x_center: float = 0.0,
                             array_z: float = 0.0, tx_f_number: float | None = None,
                             tx_window: str | None = None) -> float:
    """FWHM lateral (m, a -6 dB de amplitud) del diagrama monocromático de la apertura dinámica.

    Se evalúa la suma coherente de los elementos de la apertura de recepción
    (``|x - xe| <= z/(2F#)`` con la ventana ``window``, geometría exacta, sin directividad del
    elemento ni ancho de banda) para un dispersor en ``(x_center, z)``. Para ventana
    rectangular tiende a ``1.206·λ·F#`` (primer cruce de ``sinc`` por 0.5).

    ``two_way=True`` multiplica el diagrama de recepción por el de emisión (el producto es el diagrama
    de ida y vuelta): STA, con la misma apertura en emisión (``≈ 0.886·λ·F#`` en rectangular), o la
    emisión focalizada en ``z`` con ``tx_f_number`` y ``tx_window`` (por defecto, los de recepción).
    Con onda plana en emisión, ``two_way=False``: la emisión no aporta selectividad lateral.
    """
    depth = z - array_z
    k = 2.0 * np.pi / wavelength
    xs = x_center + np.linspace(-3.0, 3.0, 6001) * wavelength * max(f_number, tx_f_number or 0.0)
    r0 = np.hypot(element_x - x_center, depth)

    def pattern(f_n: float, win: str) -> np.ndarray:
        half = depth / (2.0 * f_n)
        out = np.empty(xs.size)
        for n, xp in enumerate(xs):
            w = window_function((xp - element_x) / half, win)
            out[n] = np.abs(np.sum(w * np.exp(1j * k * (np.hypot(element_x - xp, depth) - r0))))
        return out / out.max()

    p = pattern(f_number, window)
    if two_way:
        p = p * pattern(tx_f_number or f_number, tx_window or window)
    return fwhm_1d(xs, p)


# --------------------------------------------------------------------------
# Speckle
# --------------------------------------------------------------------------
def speckle_snr(envelope: np.ndarray, roi: np.ndarray) -> float:
    """SNR del speckle: media / desviación típica de la envolvente lineal en la ROI (Rayleigh: 1.913)."""
    v = np.asarray(envelope, float)[roi]
    return float(v.mean() / v.std())


def _autocorr_2d(a: np.ndarray) -> np.ndarray:
    """Autocorrelación 2D normalizada (cero retardo en el centro) de ``a`` sin media, con corrección de solape."""
    a = a - a.mean()
    nz, nx = a.shape
    f = np.fft.fft2(a, (2 * nz, 2 * nx))
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(f) ** 2).real)[1:, 1:]  # retardos de -(n-1) a n-1
    lag_z, lag_x = np.arange(-nz + 1, nz), np.arange(-nx + 1, nx)
    overlap = (nz - np.abs(lag_z))[:, None] * (nx - np.abs(lag_x))[None, :]
    ac = ac / overlap
    return ac / ac[nz - 1, nx - 1]


def speckle_grain_fwhm(envelope: np.ndarray, roi: np.ndarray, dz: float, dx: float) -> tuple[float, float]:
    """Tamaño del grano de speckle ``(axial, lateral)``: FWHM de la autocovarianza de la intensidad ``|env|²``.

    La ROI debe ser rectangular (se recorta su caja envolvente). Para speckle totalmente desarrollado la
    autocovarianza de la intensidad es ``|ρ|²``, con ``ρ`` la correlación del campo complejo.
    """
    jj, ii = np.nonzero(roi)
    box = np.asarray(envelope, float)[jj.min():jj.max() + 1, ii.min():ii.max() + 1] ** 2
    ac = _autocorr_2d(box)
    nz, nx = box.shape
    # Sólo los retardos hasta la mitad de la caja: más allá el solape es pequeño y la estimación, ruidosa
    lz, lx = np.arange(-nz + 1, nz), np.arange(-nx + 1, nx)
    kz, kx = np.abs(lz) <= nz // 2, np.abs(lx) <= nx // 2
    return (fwhm_1d(lz[kz] * dz, ac[kz, nx - 1]), fwhm_1d(lx[kx] * dx, ac[nz - 1, kx]))


def predicted_grain_fwhm(psf_iq: np.ndarray, x: np.ndarray, z: np.ndarray, x0: float, z0: float,
                         search: float = 1.5e-3,
                         half_window: tuple[float, float] = (3e-3, 6e-3)) -> tuple[float, float]:
    """FWHM ``(axial, lateral)`` de ``|ρ|²`` predicho a partir de la PSF compleja (``ρ`` = autocorrelación del campo).

    ``psf_iq``: imagen compleja de un dispersor puntual en ``(x0, z0)`` (``BModeImage.iq``). Se recorta
    ``±half_window`` metros (axial, lateral) alrededor de su máximo local (a menos de ``search``); debe abarcar la
    PSF entera. Se hace la autocorrelación compleja y se mide el ancho de ``|ρ|²`` en los dos ejes: es la
    predicción teórica del grano de speckle totalmente desarrollado.
    """
    dz, dx = z[1] - z[0], x[1] - x[0]
    X, Z = np.meshgrid(x, z)
    near = (np.abs(X - x0) <= search) & (np.abs(Z - z0) <= search)
    j, i = np.unravel_index(np.argmax(np.where(near, np.abs(psf_iq), -np.inf)), psf_iq.shape)
    hz, hx = int(round(half_window[0] / dz)), int(round(half_window[1] / dx))
    h = psf_iq[max(j - hz, 0):j + hz + 1, max(i - hx, 0):i + hx + 1]
    nz, nx = h.shape
    f = np.fft.fft2(h, (2 * nz, 2 * nx))
    rho = np.fft.fftshift(np.fft.ifft2(np.abs(f) ** 2))
    rho = np.abs(rho) ** 2
    rho = rho / rho.max()
    c0, c1 = nz, nx
    return (fwhm_1d(np.arange(-nz, nz) * dz, rho[:, c1]), fwhm_1d(np.arange(-nx, nx) * dx, rho[c0, :]))


# --------------------------------------------------------------------------
# Quiste anecoico
# --------------------------------------------------------------------------
def cnr(envelope: np.ndarray, inside: np.ndarray, background: np.ndarray) -> float:
    """CNR = |μ_dentro − μ_fondo| / √(σ²_dentro + σ²_fondo) con la envolvente lineal."""
    a, b = np.asarray(envelope, float)[inside], np.asarray(envelope, float)[background]
    return float(abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var()))


def contrast_db(envelope: np.ndarray, inside: np.ndarray, background: np.ndarray) -> float:
    """Contraste (dB): 10·log10(⟨env²⟩_dentro / ⟨env²⟩_fondo) de la potencia media; negativo = más oscuro."""
    e = np.asarray(envelope, float) ** 2
    return float(10.0 * np.log10(e[inside].mean() / e[background].mean()))


def gcnr(envelope: np.ndarray, inside: np.ndarray, background: np.ndarray, bins: int = 100) -> float:
    """gCNR = 1 − solape de los histogramas de la envolvente dentro y fuera (0 = indistinguibles, 1 = separables).

    Definición de Rodriguez-Molares et al. (IEEE TUFFC 67, 2020; cita no verificada en línea). A diferencia del
    CNR no cambia con las transformaciones no lineales (p. ej. la compresión logarítmica).
    """
    a, b = np.asarray(envelope, float)[inside], np.asarray(envelope, float)[background]
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), bins + 1)
    pa, _ = np.histogram(a, edges)
    pb, _ = np.histogram(b, edges)
    return float(1.0 - np.minimum(pa / pa.sum(), pb / pb.sum()).sum())
