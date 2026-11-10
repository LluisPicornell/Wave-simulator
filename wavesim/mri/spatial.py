"""Simulador de Bloch espacial: isocromatos en el corte y secuencia como lista de eventos.

Cada celda del fantoma aporta uno o varios **isocromatos** (agua y grasa de
cada tejido) en su centro, con su PD, T1, T2, T2' y frecuencia propia. Un
vóxel de la imagen contiene varias celdas, de modo que el desfase entre
ellas (por un campo B0 no uniforme o por la mezcla de agua y grasa) se
produce solo al sumar la señal, como en KomaMRI o JEMRIS.

**Secuencia por eventos.** Un TR es una lista de eventos (``RF``, ``Free``,
``ADC``, ``Spoil``, ``SpectralSat``). Entre eventos la evolución se aplica
de forma exacta, sin integrar paso a paso:

- un pulso duro es una rotación instantánea exacta; uno con forma se
  compone segmento a segmento (rotación exacta alrededor del campo efectivo
  ``(B1x, B1y, Δω/γ)`` de cada tramo constante);
- la precesión libre es una rotación alrededor de z por la fase acumulada y
  la relajación, dos exponenciales. Un lóbulo de gradiente sólo cambia la
  fase por su área (rotación en z y relajación conmutan), así que ``Free``
  lleva el momento de gradiente y no su forma;
- la lectura (``ADC``) con gradiente constante: la muestra ``k`` es
  ``sum_j c_j M_j z_j**k`` con ``z_j = exp(-(i w_j + 1/T2) dt)``.

**Desfase microscópico (T2').** La dispersión de frecuencias dentro de cada
celda (estructura no resuelta) se modela con una distribución lorentziana
de semianchura ``1/T2'`` *de forma exacta*, sin muestrearla con
isocromatos: la magnetización de la celda, en función del desvío ``δ`` de
frecuencia, es una suma finita de "estados de desfase"
``M_xy(δ) = sum_τ F_τ exp(-i δ τ)`` (``τ`` = tiempo neto de desfase; un
pulso de 180° lo cambia de signo), y la media lorentziana de cada término es
``F_τ exp(-|τ|/T2')``. Es la teoría de grafos de fase (EPG) con tiempos
arbitrarios; con pulsos duros el resultado es exacto para cualquier ángulo.
Una lorentziana muestreada con isocromatos necesita cientos por celda para
llegar al 1 % (su FID tiene un pico en t = 0; ver ``docs/validacion/rm.md``).

**Estado estacionario exacto.** Con destrucción ideal de la coherencia
transversal al final del TR (``Spoil``), el Mz al principio de un TR es
función afín del anterior, ``Mz' = A Mz + B``; dos simulaciones del TR dan
``A`` y ``B`` y el punto fijo ``B / (1 - A)`` es el estado estacionario
exacto (sin TR de preparación). Como la codificación de fase se aplica
después del último pulso, todas las líneas parten del mismo estado salvo la
fase ``exp(-i k_y y)``: el k-space completo sale de **una** lectura por
isocromato agrupada por filas de la codificación de fase (modo
``"steady"``). El modo ``"events"`` simula TR a TR y línea a línea (con
transitorio si se pide), y sirve para comprobar el primero.

Convenciones: SI (s, T, m, rad/s); ``M_xy = Mx + i My`` precesa como
``exp(-i w t)`` (γ > 0, ``dM/dt = γ M × B``), como ``bloch.py``. Los
tiempos de ``Sequence`` siguen en ms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..materials import porous
from ..materials.mri import FAT_FRACTION, GAMMA_BAR, MRI
from ..phantoms.phantom import TISSUE_INDEX, Phantom
from .imaging import MRIImage
from .sequences import Sequence, resolve, tissue_signal

#: Razón giromagnética del protón (rad/(s·T)).
GAMMA = 2.0 * np.pi * GAMMA_BAR

#: Resolución con la que se identifican los tiempos de desfase de los estados (s).
_TAU_RES = 1e-12

#: Banda (ppm respecto al agua) del pulso de saturación grasa espectral ideal:
#: cubre los picos de -3.8, -3.4 y -2.6 ppm de la grasa, no el agua.
FAT_SAT_BAND_PPM = (-4.4, -2.4)

#: Desplazamiento químico de la grasa que ve el pulso de saturación (ppm).
FAT_PPM = -3.4


def _resolve_backend(backend: str) -> str:
    """``"auto"`` → Numba si está instalado; ``"numba"`` lo exige; ``"numpy"`` nunca lo usa."""
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError(f"backend desconocido '{backend}' (use 'auto', 'numpy' o 'numba')")
    if backend == "numpy":
        return "numpy"
    try:
        import numba  # noqa: F401
    except ImportError:
        if backend == "numba":
            raise ImportError(
                "El backend 'numba' requiere Numba: instálalo con `pip install wavesim[fast]` "
                "o usa backend='numpy'."
            ) from None
        return "numpy"
    return "numba"


# --------------------------------------------------------------------------
# Eventos de la secuencia
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RF:
    """Pulso de radiofrecuencia: ``flip`` (rad) alrededor de un eje del plano transversal a ``phase`` (rad) de x.

    ``duration = 0``: pulso duro (rotación instantánea exacta). Con
    ``duration > 0`` es un pulso de duración finita con forma ``shape``
    (muestras complejas equiespaciadas de B1, en unidades arbitrarias; por
    defecto rectangular), escalado para girar ``flip`` en resonancia; ``gx``,
    ``gz`` (T/m) son gradientes durante el pulso. Cada
    tramo se aplica como rotación exacta alrededor de su campo efectivo
    (incluido el desvío de frecuencia de cada isocromato); la relajación y el
    desfase lorentziano microscópico se aplican medio antes y medio después
    del pulso (error ``O((duración/T2)^2)``).
    """

    flip: float
    phase: float = 0.0
    duration: float = 0.0
    shape: np.ndarray | None = field(default=None, compare=False)
    gx: float = 0.0
    gz: float = 0.0


@dataclass(frozen=True)
class SpectralSat:
    """Pulso espectralmente selectivo ideal: gira ``flip`` sólo los isocromatos con frecuencia en ``[f_lo, f_hi]`` Hz.

    La frecuencia es ``Isochromats.sel_dw`` (desplazamiento químico + ΔB0),
    así que un B0 no uniforme deja grasa sin saturar y satura agua, como en
    la realidad. Se suele seguir de ``Spoil``.
    """

    flip: float
    f_lo: float
    f_hi: float


@dataclass(frozen=True)
class Free:
    """Precesión libre durante ``duration`` (s) con un momento de gradiente ``(area_x, area_z)`` (T·s/m)."""

    duration: float
    area_x: float = 0.0
    area_z: float = 0.0


@dataclass(frozen=True)
class ADC:
    """Lectura de ``n`` muestras separadas ``dwell`` (s) con gradiente constante ``(gx, gz)`` (T/m).

    La muestra ``k`` se toma en ``t0 + k·dwell``; al acabar, el estado está en ``t0 + n·dwell``.
    """

    n: int
    dwell: float
    gx: float = 0.0
    gz: float = 0.0

    @property
    def duration(self) -> float:
        return self.n * self.dwell


@dataclass(frozen=True)
class Spoil:
    """Destrucción ideal de la coherencia: anula la magnetización transversal y los Mz con desfase.

    Es la hipótesis de las ecuaciones de señal de ``sequences.py`` (spoiling
    perfecto); en un equipo la producen gradientes de destrucción y la
    fase de RF variable.
    """


# --------------------------------------------------------------------------
# Isocromatos
# --------------------------------------------------------------------------
@dataclass
class Isochromats:
    """Conjunto de ``N`` isocromatos (todos los campos se convierten a vectores de longitud ``N``).

    - ``x``, ``z``: posición (m); ``m0``: magnetización de equilibrio (puede
      ser negativa: los pesos de la mezcla porosa lineal lo son a veces);
    - ``t1``, ``t2`` (s); ``r2p = 1/T2'`` (1/s) del desfase lorentziano
      microscópico (0 = sin desfase);
    - ``dw``: desvío de frecuencia (rad/s) que se aplica en la precesión;
    - ``sel_dw``: frecuencia que ve un pulso espectralmente selectivo (rad/s;
      por defecto ``dw``);
    - ``coil``: sensibilidad compleja de recepción (B1-); ``b1``: escala de
      la amplitud de transmisión (B1+).
    """

    x: np.ndarray
    z: np.ndarray
    m0: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    r2p: np.ndarray | float = 0.0
    dw: np.ndarray | float = 0.0
    sel_dw: np.ndarray | float | None = None
    coil: np.ndarray | complex = 1.0
    b1: np.ndarray | float = 1.0

    def __post_init__(self):
        self.x = np.ascontiguousarray(np.asarray(self.x, dtype=float).ravel())
        n = self.x.size

        def vec(v, dtype=float):
            # Copia propia y escribible (broadcast_to devuelve una vista de sólo lectura)
            v = np.asarray(v, dtype=dtype)
            return np.array(np.broadcast_to(v.ravel() if v.ndim else v, (n,)), dtype=dtype)

        self.z, self.m0, self.t1, self.t2 = (vec(v) for v in (self.z, self.m0, self.t1, self.t2))
        self.r2p, self.dw, self.b1 = vec(self.r2p), vec(self.dw), vec(self.b1)
        self.sel_dw = self.dw.copy() if self.sel_dw is None else vec(self.sel_dw)
        self.coil = vec(self.coil, complex)

    @property
    def n(self) -> int:
        return self.x.size

    @property
    def uniform_b1(self) -> bool:
        return bool(np.all(self.b1 == 1.0))


class _State:
    """Estados de desfase: ``M_xy(δ) = sum F_τ e^{-iδτ}`` y ``Mz(δ) = sum Z_τ e^{-iδτ}`` (``Z_-τ = conj Z_τ``)."""

    __slots__ = ("tf", "f", "tz", "z")

    def __init__(self, tf, f, tz, z):
        self.tf, self.f, self.tz, self.z = tf, f, tz, z

    @classmethod
    def longitudinal(cls, mz: np.ndarray) -> "_State":
        mz = np.asarray(mz, dtype=complex)
        return cls(np.zeros(0), np.zeros((0, mz.size), complex), np.zeros(1), mz[None, :].copy())

    def copy(self) -> "_State":
        return _State(self.tf.copy(), self.f.copy(), self.tz.copy(), self.z.copy())

    def mz(self) -> np.ndarray:
        """Mz de la celda (media sobre la distribución microscópica: sólo cuenta ``Z_0``)."""
        i0 = np.flatnonzero(_key(self.tz) == 0)
        return self.z[i0[0]].real.copy() if i0.size else np.zeros(self.z.shape[1])


def _key(t) -> np.ndarray:
    return np.round(np.asarray(t, dtype=float) / _TAU_RES).astype(np.int64)


def _rotation(ux, uy, uz, angle) -> np.ndarray:
    """Matriz ``(..., 3, 3)`` de la rotación de Bloch (γ > 0) de ángulo ``angle`` alrededor de ``u``.

    ``dM/dt = γ M × B`` gira M en sentido horario visto desde la punta de B:
    es la rotación de Rodrigues de ángulo ``-angle``. Con ``u = x`` y 90°,
    ``Mz = 1`` pasa a ``My = 1`` (la convención de ``bloch.rotate_x``).
    """
    ux, uy, uz, angle = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (ux, uy, uz, angle)))
    c, s = np.cos(angle), np.sin(angle)
    k = 1.0 - c
    r = np.empty(angle.shape + (3, 3))
    r[..., 0, 0] = c + k * ux * ux
    r[..., 0, 1] = k * ux * uy + s * uz
    r[..., 0, 2] = k * ux * uz - s * uy
    r[..., 1, 0] = k * uy * ux - s * uz
    r[..., 1, 1] = c + k * uy * uy
    r[..., 1, 2] = k * uy * uz + s * ux
    r[..., 2, 0] = k * uz * ux + s * uy
    r[..., 2, 1] = k * uz * uy - s * ux
    r[..., 2, 2] = c + k * uz * uz
    return r


def _coefficients(r: np.ndarray):
    """Acción de la rotación ``r`` sobre ``(M+, M-, Mz)``.

    ``M+' = a M+ + b M- + g Mz`` y ``Mz' = d M+ + d* M- + r22 Mz``.
    """
    p0 = r[..., 0, 0] + 1j * r[..., 1, 0]
    p1 = r[..., 0, 1] + 1j * r[..., 1, 1]
    a = 0.5 * (p0 - 1j * p1)
    b = 0.5 * (p0 + 1j * p1)
    g = r[..., 0, 2] + 1j * r[..., 1, 2]
    d = 0.5 * (r[..., 2, 0] - 1j * r[..., 2, 1])
    return a, b, g, d, r[..., 2, 2]


def _apply_rotation(st: _State, coefs, tol: float) -> None:
    """Aplica una rotación instantánea a todos los estados de desfase.

    ``M-(δ) = conj M+(δ)`` tiene los coeficientes ``conj F_{-τ}``: el pulso
    mezcla el estado ``τ`` con el ``-τ`` (así reenfoca un 180°).
    """
    a, b, g, d, r22 = coefs
    n = st.z.shape[1]
    kf, kz = _key(st.tf), _key(st.tz)
    keys = np.unique(np.concatenate([kf, -kf, kz]))
    m = keys.size
    f_tau = np.zeros((m, n), complex)
    f_neg = np.zeros((m, n), complex)
    z_tau = np.zeros((m, n), complex)
    if kf.size:
        f_tau[np.searchsorted(keys, kf)] = st.f
        f_neg[np.searchsorted(keys, -kf)] = np.conj(st.f)
    z_tau[np.searchsorted(keys, kz)] = st.z
    f_new = a * f_tau + b * f_neg + g * z_tau
    z_new = d * f_tau + np.conj(d) * f_neg + r22 * z_tau
    taus = keys * _TAU_RES
    keep_f = np.abs(f_new).max(axis=1, initial=0.0) > tol
    keep_z = (np.abs(z_new).max(axis=1, initial=0.0) > tol) | (keys == 0)
    st.tf, st.f = taus[keep_f], f_new[keep_f]
    st.tz, st.z = taus[keep_z], z_new[keep_z]


def _hard_coefficients(iso: Isochromats, flip: float, phase: float):
    ux, uy = np.cos(phase), np.sin(phase)
    angle = flip if iso.uniform_b1 else flip * iso.b1
    return _coefficients(_rotation(ux, uy, 0.0, angle))


def _shaped_coefficients(iso: Isochromats, ev: RF):
    shape = np.asarray(ev.shape if ev.shape is not None else [1.0], dtype=complex).ravel()
    dt = ev.duration / shape.size
    area = shape.sum()
    if abs(area) == 0:
        raise ValueError("la forma del pulso tiene área nula: no se puede escalar a un ángulo")
    w1 = ev.flip * shape / (dt * abs(area)) * np.exp(1j * ev.phase) * (np.conj(area) / abs(area))
    wz = iso.dw + GAMMA * (ev.gx * iso.x + ev.gz * iso.z)
    r = np.broadcast_to(np.eye(3), (iso.n, 3, 3)).copy()
    for w in w1:
        bx, by = w.real * iso.b1, w.imag * iso.b1
        norm = np.sqrt(bx * bx + by * by + wz * wz)
        safe = np.where(norm > 0, norm, 1.0)
        rm = _rotation(bx / safe, by / safe, np.where(norm > 0, wz / safe, 1.0), norm * dt)
        r = np.matmul(rm, r)
    return _coefficients(r)


def _free(st: _State, iso: Isochromats, duration: float, area_x: float = 0.0, area_z: float = 0.0,
          precess: bool = True) -> None:
    """Precesión libre exacta: fase ``dw·t + γ (A_x x + A_z z)``, relajación T2 y recuperación T1."""
    if duration < 0:
        raise ValueError(f"duración negativa en la secuencia ({duration * 1e3:.3f} ms)")
    if st.f.shape[0]:
        if precess:
            phase = iso.dw * duration
            if area_x or area_z:
                phase = phase + GAMMA * (area_x * iso.x + area_z * iso.z)
            st.f *= np.exp(-1j * phase - duration / iso.t2)
        elif duration:
            st.f *= np.exp(-duration / iso.t2)
        st.tf = st.tf + duration
    if duration > 0:
        e1 = np.exp(-duration / iso.t1)
        st.z *= e1
        k = _key(st.tz)
        i0 = np.flatnonzero(k == 0)
        if not i0.size:
            st.tz = np.concatenate([st.tz, [0.0]])
            st.z = np.vstack([st.z, np.zeros((1, st.z.shape[1]), complex)])
            i0 = [st.z.shape[0] - 1]
        st.z[i0[0]] += iso.m0 * (1.0 - e1)


def _spoil(st: _State) -> None:
    n = st.z.shape[1]
    i0 = np.flatnonzero(_key(st.tz) == 0)
    st.f, st.tf = np.zeros((0, n), complex), np.zeros(0)
    st.z = st.z[i0[:1]].real.astype(complex) if i0.size else np.zeros((1, n), complex)
    st.tz = np.zeros(1)


def _readout_numpy(w, taus, r2p, zfac, rows, n_rows, n_samples, dwell):
    """``out[r, k] = sum_{j en r} sum_s w[s, j] exp(-r2p_j |τ_s + k dt|) zfac_j**k``."""
    out = np.zeros((n_rows, n_samples), complex)
    zk = np.ones(zfac.size, complex)
    lorentz = bool(np.any(r2p))
    plain = w.sum(axis=0) if not lorentz else None
    for k in range(n_samples):
        if lorentz:
            acc = np.einsum("sj,sj->j", w, np.exp(-np.abs(taus + k * dwell)[:, None] * r2p[None, :]))
        else:
            acc = plain
        v = acc * zk
        if n_rows == 1:
            out[0, k] = v.sum()
        else:
            out[:, k] = np.bincount(rows, v.real, n_rows) + 1j * np.bincount(rows, v.imag, n_rows)
        zk *= zfac
    return out


def _readout(st: _State, iso: Isochromats, adc: ADC, rows: np.ndarray | None, n_rows: int,
             backend: str) -> np.ndarray:
    """Señal de una lectura agrupada por filas (``rows``), sin modificar el estado."""
    n_samples = adc.n
    if not st.f.shape[0]:
        return np.zeros((n_rows, n_samples), complex)
    w = np.ascontiguousarray(st.f * iso.coil[None, :])
    omega = iso.dw + GAMMA * (adc.gx * iso.x + adc.gz * iso.z)
    zfac = np.exp(-(1j * omega + 1.0 / iso.t2) * adc.dwell)
    rows = np.zeros(iso.n, np.int64) if rows is None else np.ascontiguousarray(rows, dtype=np.int64)
    if backend == "numba":
        from . import _numba_kernels as nk

        return nk.readout_rows(w, np.ascontiguousarray(st.tf), iso.r2p, zfac, rows, n_rows, n_samples,
                               float(adc.dwell))
    return _readout_numpy(w, st.tf, iso.r2p, zfac, rows, n_rows, n_samples, adc.dwell)


def _tolerance(iso: Isochromats) -> float:
    return 1e-12 * max(float(np.abs(iso.m0).max(initial=0.0)), 1e-300)


def _apply(st: _State, iso: Isochromats, ev, tol: float, cache: dict) -> None:
    """Aplica un evento que no es una lectura (las lecturas las gestiona quien llama)."""
    if isinstance(ev, Free):
        _free(st, iso, ev.duration, ev.area_x, ev.area_z)
    elif isinstance(ev, RF):
        if ev.duration > 0:
            coefs = cache.get(id(ev))
            if coefs is None:
                coefs = cache[id(ev)] = _shaped_coefficients(iso, ev)
            _free(st, iso, ev.duration / 2, precess=False)
            _apply_rotation(st, coefs, tol)
            _free(st, iso, ev.duration / 2, precess=False)
        else:
            _apply_rotation(st, _hard_coefficients(iso, ev.flip, ev.phase), tol)
    elif isinstance(ev, SpectralSat):
        f = iso.sel_dw / (2 * np.pi)
        sel = (f >= ev.f_lo) & (f <= ev.f_hi)
        a, b, g, d, r22 = _coefficients(_rotation(1.0, 0.0, 0.0, np.where(sel, ev.flip, 0.0)))
        _apply_rotation(st, (a, b, g, d, r22), tol)
    elif isinstance(ev, Spoil):
        _spoil(st)
    elif isinstance(ev, ADC):
        _free(st, iso, ev.duration, ev.gx * ev.duration, ev.gz * ev.duration)
    else:
        raise TypeError(f"evento desconocido: {ev!r}")


def simulate(iso: Isochromats, events, state: _State | None = None, backend: str = "auto"):
    """Simula una lista de eventos. Devuelve ``(señales, estado)``.

    ``señales`` es una lista con un vector complejo por cada ``ADC``: la
    suma de ``coil · M_xy`` de todos los isocromatos (con la media
    lorentziana de cada uno). ``state`` permite encadenar TR; por defecto se
    parte del equilibrio ``Mz = m0``.
    """
    be = _resolve_backend(backend)
    st = state.copy() if state is not None else _State.longitudinal(iso.m0)
    tol = _tolerance(iso)
    cache: dict = {}
    signals = []
    for ev in events:
        if isinstance(ev, ADC):
            signals.append(_readout(st, iso, ev, None, 1, be)[0])
        _apply(st, iso, ev, tol, cache)
    return signals, st


def steady_state_mz(iso: Isochromats, tr_events) -> np.ndarray:
    """Mz exacto al inicio de cada TR en el estado estacionario.

    El TR debe terminar en un estado puramente longitudinal (un ``Spoil``
    seguido, si acaso, de precesión libre): entonces ``Mz_fin = A Mz_ini + B``
    para cada isocromato y el punto fijo es ``B / (1 - A)``.
    """
    tol = _tolerance(iso)
    cache: dict = {}
    ends = []
    for start in (np.zeros(iso.n), np.ones(iso.n)):
        st = _State.longitudinal(start)
        for ev in tr_events:
            _apply(st, iso, ev, tol, cache)
        if st.f.shape[0] or np.any(_key(st.tz) != 0):
            raise ValueError("el TR no acaba en un estado longitudinal: añade Spoil antes del final")
        ends.append(st.mz())
    b = ends[0]
    a = ends[1] - b
    den = 1.0 - a
    ok = np.abs(den) > 1e-14
    return np.where(ok, b / np.where(ok, den, 1.0), iso.m0)


# --------------------------------------------------------------------------
# Secuencias cartesianas 2D
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CartesianSequence:
    """TR de una secuencia cartesiana 2D: ``pre`` + desfasador y codificación de fase + ``adc`` + ``post``.

    ``shape = (n_pe, n_read)``, ``fov = (fov_pe, fov_read)`` (m). La línea
    ``p`` tiene ``k_pe[p] = (p - n_pe/2) 2π/fov_pe`` y la muestra ``n`` de la
    lectura ``k_read = (n - n_read/2) 2π/fov_read``: el centro del k-space
    (``n = n_read/2``) se lee en TE.
    """

    sequence: Sequence
    pre: tuple
    pe_delay: float
    adc: ADC
    post: tuple
    prephase: float
    readout: str
    shape: tuple[int, int]
    fov: tuple[float, float]
    bandwidth: float
    te: float
    tr: float

    @property
    def k_pe(self) -> np.ndarray:
        return (np.arange(self.shape[0]) - self.shape[0] // 2) * (2 * np.pi / self.fov[0])

    @property
    def k_read(self) -> np.ndarray:
        return (np.arange(self.shape[1]) - self.shape[1] // 2) * (2 * np.pi / self.fov[1])

    def pe_event(self, line: int | None) -> Free:
        """Intervalo del desfasador de lectura y de la codificación de fase de la línea ``line`` (None = sin fase)."""
        area_pe = 0.0 if line is None else self.k_pe[line] / GAMMA
        if self.readout == "x":
            return Free(self.pe_delay, self.prephase, area_pe)
        return Free(self.pe_delay, area_pe, self.prephase)

    def events(self, line: int | None = None) -> list:
        return [*self.pre, self.pe_event(line), self.adc, *self.post]


def cartesian_sequence(sequence: Sequence | str, b0: float, shape: tuple[int, int], fov: tuple[float, float],
                       bandwidth: float = 130.0, readout: str = "x") -> CartesianSequence:
    """Construye el TR de una secuencia ``"SE"``, ``"IR"`` o ``"GRE"`` de ``sequences.SEQUENCES``.

    ``bandwidth``: ancho de banda de lectura por píxel (Hz/píxel). Fija el
    tiempo de lectura ``1/bandwidth`` y el gradiente ``bandwidth/(γ̄ Δx)``.

    - SE: 90°x - TE/2 - 180°y - [desfasador + fase] - lectura centrada en TE - spoil.
    - IR: 180° - spoil - TI - SE como arriba (la inversión en t = 0).
    - GRE: α - [desfasador + fase] - lectura centrada en TE - spoil ideal.

    Con ``fat_sat``, un pulso espectral ideal (``SpectralSat``) en la banda
    ``FAT_SAT_BAND_PPM`` con ``cos(flip) = fat_sat_residual`` y spoil, justo
    antes de la excitación: reproduce el residuo de ``tissue_signal``.
    """
    seq = resolve(sequence, b0)
    if readout not in ("x", "z"):
        raise ValueError("readout debe ser 'x' o 'z'")
    n_pe, n_read = (int(v) for v in shape)
    if n_pe < 2 or n_read < 2 or n_pe % 2 or n_read % 2:
        raise ValueError(f"la matriz debe tener tamaños pares >= 2 (se pidió {shape})")
    fov_pe, fov_read = (float(v) for v in fov)
    dwell = 1.0 / (bandwidth * n_read)
    t_acq = n_read * dwell
    g_read = 2 * np.pi / (GAMMA * fov_read * dwell)
    prephase = -g_read * (n_read // 2) * dwell
    tr, te = seq.tr * 1e-3, seq.te * 1e-3
    adc = ADC(n_read, dwell, g_read if readout == "x" else 0.0, g_read if readout == "z" else 0.0)
    pre: list = []
    if seq.fat_sat:
        f0 = GAMMA_BAR * b0 * 1e-6
        pre += [SpectralSat(float(np.arccos(seq.fat_sat_residual)), FAT_SAT_BAND_PPM[0] * f0,
                            FAT_SAT_BAND_PPM[1] * f0), Spoil()]
    exc = np.radians(seq.flip_deg)
    if seq.kind == "GRE":
        pre += [RF(exc)]
        pe_delay, t_end = te - t_acq / 2, te + t_acq / 2
    elif seq.kind == "SE":
        pre += [RF(exc), Free(te / 2), RF(np.pi, np.pi / 2)]
        pe_delay, t_end = te / 2 - t_acq / 2, te + t_acq / 2
    elif seq.kind == "IR":
        ti = seq.ti * 1e-3
        pre += [RF(np.pi), Spoil(), Free(ti), RF(exc), Free(te / 2), RF(np.pi, np.pi / 2)]
        pe_delay, t_end = te / 2 - t_acq / 2, ti + te + t_acq / 2
    else:
        raise ValueError(f"tipo de secuencia desconocido: {seq.kind}")
    if pe_delay < 0:
        need = 1.0 / (2 * te) if seq.kind == "GRE" else 1.0 / te
        raise ValueError(
            f"{seq.name}: la lectura ({t_acq * 1e3:.2f} ms a {bandwidth:g} Hz/píxel) no cabe con TE = {seq.te:g} ms; "
            f"sube el ancho de banda por encima de {need:.0f} Hz/píxel"
            + (" (el UTE necesita lectura radial, no implementada: usa engine='analytic')" if seq.te < 1 else "")
        )
    if tr < t_end:
        raise ValueError(f"{seq.name}: TR = {seq.tr:g} ms es más corto que la lectura ({t_end * 1e3:.1f} ms)")
    post = (Spoil(), Free(tr - t_end))
    return CartesianSequence(seq, tuple(pre), pe_delay, adc, post, prephase, readout, (n_pe, n_read),
                             (fov_pe, fov_read), float(bandwidth), te, tr)


def acquire(iso: Isochromats, cseq: CartesianSequence, mode: str = "steady", backend: str = "auto",
            n_dummy: int | None = None) -> np.ndarray:
    """k-space ``(n_pe, n_read)`` de la secuencia.

    - ``mode="steady"``: estado estacionario exacto y una sola lectura por
      isocromato, agrupada por su coordenada de codificación de fase.
    - ``mode="events"``: TR a TR, línea a línea. ``n_dummy=None`` parte del
      estado estacionario exacto (debe coincidir con ``"steady"``); un entero
      parte del equilibrio y simula esos TR de preparación (transitorio).
    """
    be = _resolve_backend(backend)
    if mode == "steady":
        st = _State.longitudinal(steady_state_mz(iso, cseq.events(None)))
        tol = _tolerance(iso)
        cache: dict = {}
        for ev in (*cseq.pre, cseq.pe_event(None)):
            _apply(st, iso, ev, tol, cache)
        coord = iso.z if cseq.readout == "x" else iso.x
        levels, rows = np.unique(coord, return_inverse=True)
        per_row = _readout(st, iso, cseq.adc, rows.ravel(), levels.size, be)
        return np.exp(-1j * np.outer(cseq.k_pe, levels)) @ per_row
    if mode == "events":
        if n_dummy is None:
            st = _State.longitudinal(steady_state_mz(iso, cseq.events(None)))
        else:
            _, st = simulate(iso, cseq.events(None) * int(n_dummy), backend=be)
        out = np.zeros(cseq.shape, complex)
        for p in range(cseq.shape[0]):
            sig, st = simulate(iso, cseq.events(p), st, be)
            out[p] = sig[0]
        return out
    raise ValueError(f"modo desconocido '{mode}' (use 'steady' o 'events')")


def reconstruct(kspace: np.ndarray, cseq: CartesianSequence, origin: tuple[float, float]) -> np.ndarray:
    """Imagen compleja ``(nz, nx)`` por FFT inversa.

    ``origin = (q0, r0)``: coordenada del centro del primer píxel en la
    dirección de fase y en la de lectura. El factor ``exp(i k·origin)``
    lleva la fase de los gradientes (referida al isocentro x = z = 0) a la
    malla de la imagen.
    """
    q0, r0 = origin
    s = kspace * np.exp(1j * (cseq.k_pe[:, None] * q0 + cseq.k_read[None, :] * r0))
    img = np.fft.ifft2(np.fft.ifftshift(s))
    return img if cseq.readout == "x" else img.T


# --------------------------------------------------------------------------
# Del fantoma a los isocromatos
# --------------------------------------------------------------------------
def tissue_weights(phantom: Phantom) -> dict[str, np.ndarray]:
    """Peso de cada tejido en cada celda (mapas ``(nz, nx)``).

    1 en las celdas de ese tejido; en las porosas, los pesos de la mezcla
    lineal de la RM de ``materials.porous`` (la señal de una celda es la suma
    de la de cada fracción de volumen, como en ``mri_scan``). Pueden salir
    negativos: el contenido de los espacios trabeculares se deduce de la
    tabla restando la matriz.
    """
    keys = phantom.property_tissues()
    w = {k: (phantom.labels == TISSUE_INDEX[k]).astype(float) for k in keys}
    if phantom.porosity is not None:
        for t in porous.POROUS_TISSUES:
            cells = (phantom.labels == TISSUE_INDEX[t]) & np.isfinite(phantom.porosity)
            if not cells.any():
                continue
            phi = phantom.porosity[cells]
            for k in keys:
                w[k][cells] = porous.porous_values(t, phi, lambda q, k=k: float(q == k), "mri", phantom.pore_fill)
    return w


def phantom_isochromats(phantom: Phantom, b0: float, texture: float = 0.02, cell_weight: float = 1.0,
                        chemical_shift: bool = False, fat_peaks: tuple = ((FAT_PPM, 1.0),),
                        field_hz: np.ndarray | None = None, coil: np.ndarray | None = None,
                        t2_prime: bool = True) -> Isochromats:
    """Isocromatos del corte: uno por celda y por componente (agua y cada pico de grasa) de cada tejido.

    - ``cell_weight``: área de la celda entre área del vóxel de la imagen,
      para que la imagen de un tejido uniforme valga su señal por unidad de
      PD (la de ``tissue_signal``).
    - ``texture``: variación de PD con ``phantom.heterogeneity`` (como en ``mri_scan``).
    - ``chemical_shift``: si la grasa precesa a su frecuencia (si no, sólo
      la ve el pulso de saturación grasa).
    - ``field_hz``: mapa ``(nz, nx)`` de ΔB0 en Hz (susceptibilidad).
    - ``coil``: mapa ``(nz, nx)`` complejo de sensibilidad de recepción.
    - ``t2_prime``: desfase lorentziano microscópico con el T2' de la tabla.

    Agua y grasa de un mismo tejido comparten T1 y T2 (los de la tabla, que
    describen el tejido entero); así un tejido uniforme sin desplazamiento
    químico da exactamente ``tissue_signal``.
    """
    g = phantom.grid
    X, Z = g.meshgrid()
    tex = np.ones(g.shape)
    if texture and phantom.heterogeneity is not None:
        tex = 1.0 + texture * np.clip(phantom.heterogeneity, -3, 3)
    f0 = GAMMA_BAR * b0 * 1e-6  # Hz por ppm
    amps = np.array([a for _, a in fat_peaks], dtype=float)
    amps = amps / amps.sum()
    ppms = np.array([p for p, _ in fat_peaks], dtype=float)
    cols = {k: [] for k in ("x", "z", "m0", "t1", "t2", "r2p", "dw", "sel", "coil")}
    for k, wk in tissue_weights(phantom).items():
        pd, t1, t2, _ = MRI[k].at_field(b0)
        cells = wk != 0
        if pd == 0 or not cells.any():
            continue
        r2p = 1e3 / (MRI[k].t2_prime_15 * 1.5 / b0) if t2_prime else 0.0
        m = wk[cells] * pd * cell_weight * tex[cells]
        db = field_hz[cells] if field_hz is not None else 0.0
        ff = FAT_FRACTION.get(k, 0.0)
        species = [(1.0 - ff, 0.0)] + ([(ff * a, p) for a, p in zip(amps, ppms)] if ff > 0 else [])
        for frac, ppm in species:
            if frac == 0:
                continue
            n = m.size
            cols["x"].append(X[cells])
            cols["z"].append(Z[cells])
            cols["m0"].append(frac * m)
            cols["t1"].append(np.full(n, t1 * 1e-3))
            cols["t2"].append(np.full(n, t2 * 1e-3))
            cols["r2p"].append(np.full(n, r2p))
            cs = ppm * f0
            cols["dw"].append(2 * np.pi * (db + (cs if chemical_shift else 0.0)) * np.ones(n))
            cols["sel"].append(2 * np.pi * (db + cs) * np.ones(n))
            cols["coil"].append(coil[cells] if coil is not None else np.ones(n, complex))
    if not cols["x"]:
        raise ValueError("el fantoma no tiene ningún tejido con señal de RM")
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    return Isochromats(cat["x"], cat["z"], cat["m0"], cat["t1"], cat["t2"], cat["r2p"], cat["dw"], cat["sel"],
                       cat["coil"])


# --------------------------------------------------------------------------
# Imagen de un fantoma
# --------------------------------------------------------------------------
@dataclass
class BlochImage(MRIImage):
    """Imagen del simulador de Bloch espacial (``ideal`` = magnitud sin ruido)."""

    complex_image: np.ndarray | None = None
    kspace: np.ndarray | None = None
    bandwidth: float = 0.0
    readout: str = "x"
    n_isochromats: int = 0
    timings: dict = field(default_factory=dict)


def _even(v: float, lo: int, hi: int) -> int:
    n = int(np.clip(round(v / 2.0) * 2, lo, hi))
    return n - (n % 2)


def bloch_scan(phantom: Phantom, sequence: Sequence | str = "T2", b0: float = 1.5, voxel: float = 0.4e-3,
               snr: float = 40.0, texture: float = 0.02, rng: np.random.Generator | None = None, *,
               matrix: tuple[int, int] | None = None, fov: tuple[float, float] | None = None,
               bandwidth: float = 130.0, readout: str = "x", mode: str = "steady",
               backend: str = "auto") -> BlochImage:
    """Imagen de RM 2D del corte con el simulador de Bloch espacial.

    Los cuatro primeros argumentos y ``snr``, ``texture`` y ``rng`` son los
    de ``mri_scan``. ``snr``: señal del músculo (analítica) entre la
    desviación del ruido gaussiano complejo de la imagen, que se añade en el
    k-space (0 = sin ruido). ``matrix = (nz, nx)`` (pares; por defecto el FOV
    entre ``voxel``), ``fov = (fov_z, fov_x)`` en m (por defecto, el fantoma
    entero; centrado en él si es menor, y lo que quede fuera se pliega).
    ``bandwidth``: Hz/píxel de la lectura; ``readout``: su dirección ("x" o
    "z"). Los ejes siguen el convenio de ``mri_scan``: con el FOV por
    defecto los píxeles caen en ``x0 + i·W/nx``.
    """
    t0 = time.perf_counter()
    g = phantom.grid
    seq = resolve(sequence, b0)
    rng = rng or np.random.default_rng(0)
    fov_z, fov_x = fov if fov is not None else (g.depth, g.width)
    if matrix is None:
        matrix = (_even(fov_z / voxel, 8, max(g.nz - g.nz % 2, 8)), _even(fov_x / voxel, 8, max(g.nx - g.nx % 2, 8)))
    nz, nx = matrix
    dz_img, dx_img = fov_z / nz, fov_x / nx
    x_first = g.x0 + (g.width - fov_x) / 2
    z_first = g.z0 + (g.depth - fov_z) / 2
    iso = phantom_isochromats(phantom, b0, texture, g.dx**2 / (dz_img * dx_img))
    t1 = time.perf_counter()
    if readout == "x":
        cseq = cartesian_sequence(seq, b0, (nz, nx), (fov_z, fov_x), bandwidth, "x")
        origin = (z_first, x_first)
    else:
        cseq = cartesian_sequence(seq, b0, (nx, nz), (fov_x, fov_z), bandwidth, "z")
        origin = (x_first, z_first)
    k = acquire(iso, cseq, mode, backend)
    t2 = time.perf_counter()
    ideal = np.abs(reconstruct(k, cseq, origin))
    if snr:
        present = phantom.present_tissues()
        ref = (tissue_signal("muscle", seq, b0) if "muscle" in present
               else max(tissue_signal(t, seq, b0) for t in present))
        # La FFT inversa normalizada reparte el ruido de las N muestras: sigma_img = sigma_k / sqrt(N)
        sigma = ref / snr * np.sqrt(k.size)
        k = k + rng.normal(0, sigma, k.shape) + 1j * rng.normal(0, sigma, k.shape)
    img = reconstruct(k, cseq, origin)
    x = x_first + np.arange(nx) * dx_img
    z = z_first + np.arange(nz) * dz_img
    extent = ((x[0] - dx_img / 2) * 1e3, (x[-1] + dx_img / 2) * 1e3, (z[-1] + dz_img / 2) * 1e3,
              (z[0] - dz_img / 2) * 1e3)
    timings = {"isocromatos": t1 - t0, "simulacion": t2 - t1, "total": time.perf_counter() - t0}
    return BlochImage(np.abs(img), ideal, seq, b0, dx_img, extent, snr, x, z, img, k, float(bandwidth), readout,
                      iso.n, timings)
