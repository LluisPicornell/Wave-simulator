"""FDTD de Maxwell 2D (modo TM) de banda ancha con tejidos dispersivos (Debye multipolo).

El solver de banda estrecha (``em.fdtd_tm``) da el campo a una sola
frecuencia con ``eps'`` y ``sigma`` fijos. Este emite un **pulso** que cubre
una banda ``[f_min, f_max]`` y obtiene de una sola simulación el fasor a
muchas frecuencias, con la permitividad de cada celda dependiente de la
frecuencia:

    eps(w) = eps_inf + sum_k d_eps_k / (1 + j w tau_k) + sigma / (j w eps0)

(ajuste de Debye del Cole-Cole de Gabriel, ``materials.dispersive``). Cada
polo añade una corriente de polarización ``J_k = dP_k/dt`` que cumple la
ecuación diferencial auxiliar (ADE)

    tau_k dJ_k/dt + J_k = eps0 d_eps_k dE/dt

y entra en la ley de Ampère: ``eps0 eps_inf dE/dt + sigma E + sum_k J_k = rot H``.
Ambas se discretizan centradas en ``n + 1/2`` (Taflove y Hagness,
*Computational Electrodynamics*, 3.ª ed., cap. 9; la forma con ``J`` es la
de Okoniewski, Mrozowski y Stuchly, IEEE MGWL 7:121, 1997):

    J_k^{n+1} = kappa_k J_k^n + beta_k (E^{n+1} - E^n)
    kappa_k = (2 tau_k - dt) / (2 tau_k + dt),  beta_k = 2 eps0 d_eps_k / (2 tau_k + dt)

Con ``d_eps_k >= 0``, ``sigma >= 0`` y ``eps_inf >= 1`` es estable con el mismo
paso de tiempo que el vacío. Sin polos se reduce exactamente a la
actualización del solver de banda estrecha. La CPML sólo corrige las
derivadas espaciales, así que no depende del material.

El fasor se acumula durante la simulación con una DFT por bloques (un
producto de matrices cada ``block`` muestras) en las celdas que se pidan.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

from ..materials.acoustic import ACOUSTIC
from ..materials.dispersive import DebyeMedium, TissueDebyeModel, debye_maps, tissue_debye_model
from ..phantoms.phantom import Phantom
from .fdtd_tm import C0, EPS0, ETA0, EMResult, YeeTMGrid


def wideband_pulse_params(f_min: float, f_max: float, edge_db: float = -20.0) -> tuple[float, float, float]:
    """``(f_c, sigma_t, t0)`` del pulso cuyo espectro cae ``edge_db`` en los bordes de la banda."""
    f_c = 0.5 * (f_min + f_max)
    half = 0.5 * (f_max - f_min)
    sigma_f = half / np.sqrt(2.0 * np.log(10.0 ** (-edge_db / 20.0)))
    sigma_t = 1.0 / (2.0 * np.pi * sigma_f)
    return f_c, sigma_t, 5.0 * sigma_t


def wideband_pulse(t: np.ndarray, f_min: float, f_max: float, edge_db: float = -20.0) -> np.ndarray:
    """Seno modulado por una gaussiana centrado en la banda; vale 0 a partir de ``2 t0``.

    El seno (no el coseno) hace que el pulso no tenga componente continua, así
    que no deja carga ni polarización estática al terminar. Su espectro baja
    ``edge_db`` dB en ``f_min`` y ``f_max`` (para una banda de dos décadas,
    unos -35 dB en ``f_min`` por la resta de los dos lóbulos) y es
    despreciable por encima de ~1.5 ``f_max``, donde la malla ya no resuelve
    la onda.
    """
    f_c, sigma_t, t0 = wideband_pulse_params(f_min, f_max, edge_db)
    tau = np.asarray(t, dtype=float) - t0
    out = np.exp(-0.5 * (tau / sigma_t) ** 2) * np.sin(2.0 * np.pi * f_c * tau)
    return np.where(tau <= t0, out, 0.0)


@dataclass
class BroadbandEMResult:
    """Fasores de Ey a ``freqs`` (por unidad de la onda incidente que emite la fuente).

    ``phasor`` tiene forma ``(nf, nz, nx)`` o, con monitor, ``(nf, n_puntos)``
    con las celdas ``points = (j, i)``. Mismo convenio que el solver de banda
    estrecha: ``e^{+jwt}`` y onda incidente de fasor ``j`` en el plano de la
    fuente (la de banda estrecha es ``-sin(wt)``). ``source_spectrum`` es la
    DFT del pulso inyectado.

    Diferencias con ``EMResult`` en el mismo fantoma: la fuente de banda
    estrecha toma el seno en ``t_{n+1}`` (adelanta ``w dt / 2``) y normaliza
    con la impedancia real ``eta0 / sqrt(eps')``, así que en un medio con
    pérdidas emite ``sqrt(eps' / eps)`` V/m; aquí se normaliza con la
    impedancia compleja. Corregidos esos dos factores, en un antebrazo
    realista con la fuente en el agua coinciden al 0.3 %.
    """

    freqs: np.ndarray
    phasor: np.ndarray
    points: tuple[np.ndarray, np.ndarray] | None
    medium: DebyeMedium
    source_spectrum: np.ndarray
    dt: float
    steps: int
    elapsed_s: float
    density: np.ndarray

    def permittivity(self, i: int) -> np.ndarray:
        """Permitividad compleja de cada celda a ``freqs[i]``."""
        return self.medium.complex_permittivity(self.freqs[i])

    def at(self, i: int) -> EMResult:
        """Resultado a ``freqs[i]`` con el mismo formato que el solver de banda estrecha (con SAR)."""
        if self.points is not None:
            raise ValueError("at() necesita el campo completo (simulación sin monitor)")
        f = float(self.freqs[i])
        eps = self.permittivity(i)
        sigma = -eps.imag * 2.0 * np.pi * f * EPS0
        sar = sigma * np.abs(self.phasor[i]) ** 2 / (2.0 * self.density)
        return EMResult(f, self.phasor[i], eps.real, sigma, sar, self.dt, self.steps, self.elapsed_s)


class BroadbandMaxwellTMSolver(YeeTMGrid):
    """FDTD TM con tejidos de Debye multipolo y fuente de pulso de banda ancha.

    - ``band = (f_min, f_max)``: banda del pulso y del ajuste de Debye.
    - ``model``: ajuste de los tejidos (``materials.dispersive.tissue_debye_model``);
      por defecto el de todos los tejidos con ``n_poles`` polos (``None`` = los
      necesarios para un error < 1 %).
    - ``medium``: mapas de Debye por celda que sustituyen a los del fantoma
      (p. ej. promediados en las interfaces con ``em.subpixel``); deben usar los
      polos del modelo o traer los suyos.

    - ``backend``: ``"auto"`` (Numba si está instalado), ``"numpy"`` o ``"numba"``; en float64 dan el
      mismo resultado salvo redondeo.

    La CPML usa ``f_ref = f_min`` para el término CFS: así absorbe toda la banda.
    """

    def __init__(self, phantom: Phantom, band: tuple[float, float], n_poles: int | None = None, npml: int = 20,
                 cfl: float = 0.95, dtype=np.float64, model: TissueDebyeModel | None = None,
                 medium: DebyeMedium | None = None, backend: str = "auto"):
        self.band = (float(band[0]), float(band[1]))
        if medium is None:
            model = model or tissue_debye_model(self.band, n_poles)
            medium = debye_maps(phantom, model)
        self.model, self.medium = model, medium
        g = phantom.grid
        if np.shape(medium.eps_inf) != g.shape:
            raise ValueError(f"los mapas de Debye tienen forma {np.shape(medium.eps_inf)}, la malla {g.shape}")
        if np.min(medium.eps_inf) < 1.0 - 1e-9 or np.min(medium.delta_eps, initial=0.0) < 0 or np.min(medium.sigma) < 0:
            raise ValueError("medio no pasivo: hace falta eps_inf >= 1, d_eps >= 0 y sigma >= 0")
        n_max = float(np.abs(np.sqrt(medium.complex_permittivity(self.band[1]))).max())
        self.points_per_wavelength = C0 / (self.band[1] * n_max * g.dx)
        if self.points_per_wavelength < 10:
            warnings.warn(f"Sólo {self.points_per_wavelength:.1f} puntos por longitud de onda a f_max en el tejido: "
                          "reduce dx.", stacklevel=2)
        self._setup_yee(phantom, npml, cfl, dtype, f_ref=self.band[0], backend=backend)
        dt = self.dt
        # Sólo se integran los polos que tienen amplitud en alguna celda
        active = [k for k in range(medium.n_poles) if np.any(medium.delta_eps[k] > 0)]
        tau = np.asarray(medium.tau, dtype=float)[active]
        d_eps = np.asarray(medium.delta_eps, dtype=float)[active]
        self.kappa = ((2.0 * tau - dt) / (2.0 * tau + dt)).astype(dtype)
        self.beta = (2.0 * EPS0 * d_eps / (2.0 * tau[:, None, None] + dt)).astype(dtype)
        self.g = (0.5 * (1.0 + self.kappa)).astype(dtype)
        eps_dt = EPS0 * np.asarray(medium.eps_inf, dtype=float) / dt
        half_beta = 0.5 * self.beta.sum(axis=0) if len(active) else 0.0
        denom = eps_dt + 0.5 * np.asarray(medium.sigma, dtype=float) + half_beta
        self.ca = ((eps_dt - 0.5 * np.asarray(medium.sigma, dtype=float) + half_beta) / denom).astype(dtype)
        self.cb = (1.0 / denom).astype(dtype)
        self.j_pol = np.zeros((len(active),) + g.shape, dtype)
        self._tmp = np.zeros_like(self.j_pol)
        self._e_old = np.zeros(g.shape, dtype)

    def _set_scale(self, e0: float) -> None:
        # Misma lámina de corriente que en banda estrecha, con la impedancia de alta frecuencia;
        # la onda emitida vale e0 * eta(w) / eta_inf, y run() divide por ese factor.
        m = self.medium
        params = np.vstack([m.eps_inf[self._src_j, self._src_i], m.sigma[self._src_j, self._src_i],
                            m.delta_eps[:, self._src_j, self._src_i]])
        if np.ptp(params, axis=1).max() > 1e-9 * max(float(np.abs(params).max()), 1.0):
            # eta(w)/eta_inf cambia de una celda a otra: cada tramo emitiría un espectro distinto y
            # la normalización (con la primera celda) sólo valdría en ese tramo.
            warnings.warn("La fuente cruza medios distintos: la onda emitida no es uniforme en banda ancha. "
                          "Colócala en una zona homogénea (p. ej. el agua o el aire de fuera del fantoma).",
                          stacklevel=3)
        eps_inf = np.asarray(self.medium.eps_inf)[self._src_j, self._src_i]
        eta = ETA0 / np.sqrt(eps_inf)
        self._src_scale = self.cb[self._src_j, self._src_i] * 2.0 * e0 / (eta * self.grid.dx)

    def incident_factor(self, freqs: np.ndarray) -> np.ndarray:
        """``eta(w) / eta_inf`` en la (primera) celda de la fuente: la onda emitida por unidad de pulso."""
        j, i = int(self._src_j[0]), int(self._src_i[0])
        m = self.medium
        cell = DebyeMedium(float(m.eps_inf[j, i]), float(m.sigma[j, i]), m.delta_eps[:, j, i], m.tau)
        return np.sqrt(cell.eps_inf / cell.complex_permittivity(freqs))

    def run(self, freqs: np.ndarray | None = None, monitor: np.ndarray | None = None, t_max: float | None = None,
            decay: float = 1e-6, edge_db: float = -20.0, block: int = 64,
            progress: bool = False) -> BroadbandEMResult:
        """Emite el pulso y acumula la DFT de Ey hasta que el campo decae.

        - ``freqs``: frecuencias de salida (por defecto 41 log-espaciadas en la banda).
        - ``monitor``: máscara booleana ``(nz, nx)`` de las celdas en las que se
          guarda el fasor (``None`` = todas).
        - Se para cuando, terminado el pulso, el máximo de ``|Ey|`` en las
          celdas del monitor durante un bloque de la DFT baja de ``decay``
          veces el máximo alcanzado, o en ``t_max``. En un medio conductor la
          cola del campo decae como una potencia de ``t`` (difusión de las
          frecuencias por debajo de la banda): en músculo con 0.1-10 GHz,
          ``decay = 1e-4`` dejaba un 2 % de error en beta y alpha a ``f_min``
          por truncar la DFT, y ``1e-6`` un 0.3 %.
        """
        if self._src_j is None:
            self.set_plane_wave_source()
        g = self.grid
        f_lo, f_hi = self.band
        freqs = np.geomspace(f_lo, f_hi, 41) if freqs is None else np.atleast_1d(np.asarray(freqs, dtype=float))
        _, _, t0 = wideband_pulse_params(f_lo, f_hi, edge_db)
        t_pulse = 2.0 * t0
        if t_max is None:
            n_max = float(np.abs(np.sqrt(self.medium.complex_permittivity(f_lo))).max())
            t_max = t_pulse + 10.0 * (g.width + g.depth) * n_max / C0 + 5.0 / f_lo
        nt = int(np.ceil(t_max / self.dt))
        # Muestreo de la DFT: >= 6 muestras por periodo de la frecuencia más alta del pulso (~1.5 f_max)
        stride = max(int(1.0 / (6.0 * 1.5 * max(f_hi, freqs.max()) * self.dt)), 1)
        w = 2.0 * np.pi * freqs
        if monitor is None:
            sel = None
            npts = g.nx * g.nz
        else:
            sel = np.asarray(monitor, dtype=bool)
            npts = int(sel.sum())
        acc_re = np.zeros((freqs.size, npts))
        acc_im = np.zeros((freqs.size, npts))
        buf = np.zeros((block, npts))
        t_buf = np.zeros(block)
        nb = 0
        peak = 0.0
        t0_wall = time.time()
        steps = nt
        has_poles = self.j_pol.shape[0] > 0
        numba = self.backend == "numba"
        for n in range(nt):
            t_src = (n + 0.5) * self.dt
            src = None
            if t_src <= t_pulse:
                src = self._src_scale * wideband_pulse(t_src, f_lo, f_hi, edge_db)
            if numba:
                self._advance_h()
                if has_poles:
                    self._nk.step_e_debye(self.ey, self.hx, self.hz, self.ca, self.cb, self.inv_dx,
                                          self._pk["hx_z"], self._pk["hz_x"], self.j_pol, self.kappa, self.beta,
                                          self.g)
                else:
                    self._nk.step_e(self.ey, self.hx, self.hz, self.ca, self.cb, self.inv_dx,
                                    self._pk["hx_z"], self._pk["hz_x"])
                if src is not None:
                    self.ey[self._src_j, self._src_i] -= src
                    if has_poles:
                        # J_k es lineal en Ey^{n+1}: el kernel no vio la fuente, se añade su parte aquí
                        self.j_pol[:, self._src_j, self._src_i] -= self.beta[:, self._src_j, self._src_i] * src
            else:
                self._step_numpy(has_poles, src)
            if n % stride == 0:
                buf[nb] = self.ey.ravel() if sel is None else self.ey[sel]
                t_buf[nb] = (n + 1) * self.dt
                nb += 1
                if nb == block:
                    self._flush(acc_re, acc_im, buf, t_buf, nb, w, stride)
                    nb = 0
                    m = float(np.abs(buf).max())
                    peak = max(peak, m)
                    if progress:
                        print(f"  paso {n + 1}/{nt}  max|E| = {m:.3e}", flush=True)
                    if t_src > t_pulse and m <= decay * peak:
                        steps = n + 1
                        break
        if nb:
            self._flush(acc_re, acc_im, buf, t_buf, nb, w, stride)
        # Espectro del pulso inyectado (las mismas muestras que la fuente)
        n_src = int(np.ceil(t_pulse / self.dt)) + 1
        t_s = (np.arange(n_src) + 0.5) * self.dt
        drive = wideband_pulse(t_s, f_lo, f_hi, edge_db)
        spectrum = np.exp(-1j * np.outer(w, t_s)) @ drive * self.dt
        # La lámina emite E = -eta J_s / 2 = -pulso * eta / eta_inf. El factor -j deja el mismo
        # convenio de fase que el solver de banda estrecha, cuya onda incidente (un seno) vale j
        # en el plano de la fuente; así BroadbandEMResult.at(i) es comparable con EMResult.
        norm = 1j * spectrum * self.incident_factor(freqs)
        phasor = (acc_re + 1j * acc_im) / norm[:, None]
        if sel is None:
            phasor = phasor.reshape((freqs.size,) + g.shape)
            points = None
        else:
            points = np.nonzero(sel)
        rho = self.phantom.property_map(lambda k: ACOUSTIC[k].density, physics="acoustic")
        return BroadbandEMResult(freqs, phasor, points, self.medium, spectrum, self.dt, steps,
                                 time.time() - t0_wall, rho)

    def _step_numpy(self, has_poles: bool, src: np.ndarray | None) -> None:
        """Un paso con NumPy: Ey con la fuente ``src`` ya escalada (o ``None``) y luego las corrientes J_k."""
        curl = self._curl_h()
        if has_poles:
            curl -= np.tensordot(self.g, self.j_pol, axes=1)
            np.copyto(self._e_old, self.ey)
        self.ey *= self.ca
        self.ey += self.cb * curl
        if src is not None:
            self.ey[self._src_j, self._src_i] -= src
        if has_poles:
            self.j_pol *= self.kappa[:, None, None]
            np.subtract(self.ey, self._e_old, out=self._e_old)
            np.multiply(self.beta, self._e_old, out=self._tmp)
            self.j_pol += self._tmp

    def _flush(self, acc_re, acc_im, buf, t_buf, nb, w, stride):
        # DFT por bloques: dos productos de matrices reales en vez de nf sumas por paso
        ph = np.outer(w, t_buf[:nb])
        h = stride * self.dt
        acc_re += (np.cos(ph) * h) @ buf[:nb]
        acc_im -= (np.sin(ph) * h) @ buf[:nb]
