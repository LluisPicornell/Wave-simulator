"""Solver FDTD elástico 2D (ondas P y S) para ultrasonido en tejidos.

Ecuaciones de la elastodinámica lineal en formulación velocidad-tensión
(Virieux 1986), en malla escalonada:

    rho dvx/dt = d(sxx)/dx + d(sxz)/dz
    rho dvz/dt = d(sxz)/dx + d(szz)/dz
    dsxx/dt = (lambda + 2mu) dvx/dx + lambda dvz/dz   (+ memoria viscoelástica)
    dszz/dt = lambda dvx/dx + (lambda + 2mu) dvz/dz
    dsxz/dt = mu (dvx/dz + dvz/dx)

- Derivadas espaciales de 4.º orden y salto temporal (leapfrog) de 2.º orden.
- En los fluidos (tejidos blandos, agua) ``mu = 0``: sólo hay ondas de
  presión, ``sxx = szz = -p``. En el hueso aparecen ondas de cizalla y
  conversiones P↔S en las interfaces.
- ``mu`` en los nodos ``sxz`` se promedia armónicamente: si algún vecino es
  fluido, la tensión de cizalla es nula (condición física en la interfaz).
- Atenuación viscoelástica GSLS (ver ``attenuation.py``) con memorias
  actualizadas por Crank-Nicolson.
- Bordes absorbentes CPML en los cuatro lados.

Disposición en la malla (índices [j, i] = [z, x]):
    sxx, szz : (i, j)         vx : (i+1/2, j)
    vz       : (i, j+1/2)     sxz: (i+1/2, j+1/2)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..core.pml import CPMLDerivative
from ..phantoms.phantom import Phantom
from . import attenuation as att
from .medium import ElasticMedium, elastic_medium

C1, C2 = 9.0 / 8.0, -1.0 / 24.0


@dataclass
class Source:
    """Fuente aplicada sobre un conjunto de celdas.

    ``signals`` tiene forma ``(n_canales, nt)``; ``channel[k]`` indica qué
    canal alimenta a la celda ``(j[k], i[k])``. ``kind``: "pressure" (presión
    en Pa), "force_x" o "force_z" (velocidad en m/s inyectada como fuerza).
    """

    j: np.ndarray
    i: np.ndarray
    channel: np.ndarray
    signals: np.ndarray
    kind: str = "pressure"


@dataclass
class Receiver:
    """Grupo de celdas cuya presión se promedia por canal en cada paso."""

    j: np.ndarray
    i: np.ndarray
    channel: np.ndarray
    n_channels: int
    quantity: str = "p"  # "p", "vx" o "vz"


@dataclass
class SimulationResult:
    dt: float
    t: np.ndarray
    traces: dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: list[np.ndarray] = field(default_factory=list)
    snapshot_times: list[float] = field(default_factory=list)
    peak_pressure: np.ndarray | None = None
    elapsed_s: float = 0.0


class ElasticSolver:
    """Simulación elástica 2D de un fantoma a la frecuencia central ``f0``.

    Parámetros principales:
      - ``npml``: grosor de la capa absorbente (celdas).
      - ``n_mechanisms``: mecanismos de relajación para la atenuación (0 = sin
        atenuación; 1 es suficiente para pulsos de banda moderada).
      - ``scattering``: escala de la heterogeneidad que produce speckle.
      - ``cfl``: fracción del paso temporal máximo estable.
    """

    def __init__(
        self,
        phantom: Phantom,
        f0: float,
        npml: int = 20,
        n_mechanisms: int = 1,
        scattering: float = 1.0,
        attenuation: bool = True,
        cfl: float = 0.9,
        medium: ElasticMedium | None = None,
        dtype=np.float32,
    ):
        self.phantom = phantom
        self.grid = phantom.grid
        self.f0 = f0
        self.dtype = dtype
        self.medium = medium or elastic_medium(phantom, f0, n_mechanisms, scattering, attenuation)
        m = self.medium
        dx = self.grid.dx
        self.inv_dx = 1.0 / dx
        self.n_mech = m.tau_sigma.size

        # Módulos relajados corregidos por dispersión y no relajados
        fp = att.relaxed_modulus_factor(m.tau_p, m.tau_sigma, f0) if self.n_mech else 1.0
        fs = att.relaxed_modulus_factor(m.tau_s, m.tau_sigma, f0) if self.n_mech else 1.0
        pi_r = m.rho * m.cp**2 * fp
        mu_r = m.rho * m.cs**2 * fs
        L = self.n_mech
        pi_u = pi_r * (1 + L * m.tau_p)
        mu_u = mu_r * (1 + L * m.tau_s)
        cp_u_max = float(np.sqrt(np.max(pi_u / m.rho)))
        self.dt = cfl * dx / (cp_u_max * np.sqrt(2.0) * (C1 + abs(C2)))
        dt = self.dt

        self._check_sampling(dx)

        def f(a):
            return np.ascontiguousarray(a, dtype=dtype)

        # Coeficientes ya multiplicados por dt
        self.dt_pi = f(dt * pi_u)
        self.dt_mu2 = f(dt * 2 * mu_u)
        self.dt_lam = f(dt * (pi_u - 2 * mu_u))
        self.dt_mu_xz = f(dt * _harmonic_xz(mu_u))
        self.dtbx = f(dt * 2.0 / (m.rho + _shift(m.rho, 1)))
        self.dtbz = f(dt * 2.0 / (m.rho + _shift(m.rho, 0)))
        self.has_shear = bool(np.any(mu_u > 0))
        if self.n_mech:
            self.dt_pi_t = f(dt * pi_r * m.tau_p)
            self.dt_mu2_t = f(dt * 2 * mu_r * m.tau_s)
            self.dt_mu_t_xz = f(dt * _harmonic_xz(mu_r * m.tau_s))
            ts = m.tau_sigma
            self.mem_a = (1 - dt / (2 * ts)) / (1 + dt / (2 * ts))
            self.mem_b = (dt / ts) / (1 + dt / (2 * ts))

        shape = self.grid.shape
        z = lambda: np.zeros(shape, dtype)  # noqa: E731
        self.vx, self.vz, self.sxx, self.szz, self.sxz = z(), z(), z(), z(), z()
        self.rxx = [z() for _ in range(L)]
        self.rzz = [z() for _ in range(L)]
        self.rxz = [z() for _ in range(L)]
        # Un buffer por derivada: sus bordes nunca escritos permanecen a cero
        self.d = {k: z() for k in ("sxx_x", "sxz_z", "sxz_x", "szz_z", "exx", "ezz", "exz_a", "exz_b")}
        self.acc, self.acc2, self.acc3, self.div, self.exz = z(), z(), z(), z(), z()
        self.tmp_x = np.zeros((shape[0], shape[1] - 3), dtype)
        self.tmp_z = np.zeros((shape[0] - 3, shape[1]), dtype)

        cp_kw = dict(npml=npml, dx=dx, dt=dt, c_max=cp_u_max, f_ref=f0, dtype=dtype)
        self.pml = {
            "sxx_x": CPMLDerivative(shape, 1, True, **cp_kw),
            "sxz_z": CPMLDerivative(shape, 0, False, **cp_kw),
            "sxz_x": CPMLDerivative(shape, 1, False, **cp_kw),
            "szz_z": CPMLDerivative(shape, 0, True, **cp_kw),
            "exx": CPMLDerivative(shape, 1, False, **cp_kw),
            "ezz": CPMLDerivative(shape, 0, False, **cp_kw),
            "exz_a": CPMLDerivative(shape, 0, True, **cp_kw),
            "exz_b": CPMLDerivative(shape, 1, True, **cp_kw),
        }
        self.npml = npml
        self.sources: list[Source] = []
        self.receivers: dict[str, Receiver] = {}
        self.cp_local = m.cp

    # ------------------------------------------------------------------
    def _check_sampling(self, dx: float) -> None:
        m = self.medium
        speeds = np.concatenate([m.cp.ravel(), m.cs[m.cs > 0].ravel()])
        c_min = float(speeds.min())
        ppw = c_min / (self.f0 * dx)
        self.points_per_wavelength = ppw
        if ppw < 6:
            warnings.warn(
                f"Sólo {ppw:.1f} puntos por longitud de onda a f0 (c_min = {c_min:.0f} m/s). "
                "Habrá dispersión numérica: reduce dx o la frecuencia.",
                stacklevel=3,
            )

    def time_axis(self, nt: int) -> np.ndarray:
        """Instantes en los que se aplican las fuentes y se registran los receptores."""
        return (np.arange(nt) + 1) * self.dt

    def add_source(self, source: Source) -> None:
        self.sources.append(source)

    def add_receiver(self, name: str, receiver: Receiver) -> None:
        self.receivers[name] = receiver

    # ------------------------------------------------------------------
    def _dx(self, f, out, forward: bool):
        o = out[:, 1:-2] if forward else out[:, 2:-1]
        np.subtract(f[:, 2:-1], f[:, 1:-2], out=o)
        o *= C1 * self.inv_dx
        np.subtract(f[:, 3:], f[:, :-3], out=self.tmp_x)
        self.tmp_x *= C2 * self.inv_dx
        o += self.tmp_x

    def _dz(self, f, out, forward: bool):
        o = out[1:-2, :] if forward else out[2:-1, :]
        np.subtract(f[2:-1, :], f[1:-2, :], out=o)
        o *= C1 * self.inv_dx
        np.subtract(f[3:, :], f[:-3, :], out=self.tmp_z)
        self.tmp_z *= C2 * self.inv_dx
        o += self.tmp_z

    def _deriv(self, name, f, axis, forward):
        out = self.d[name]
        (self._dx if axis == 1 else self._dz)(f, out, forward)
        self.pml[name].apply(out)
        return out

    def step(self, n: int) -> None:
        """Avanza un paso: velocidades a t+dt/2 y tensiones a t+dt."""
        d = self._deriv
        acc = self.acc
        # --- velocidades
        a = d("sxx_x", self.sxx, 1, True)
        b = d("sxz_z", self.sxz, 0, False)
        np.add(a, b, out=acc)
        acc *= self.dtbx
        self.vx += acc
        a = d("sxz_x", self.sxz, 1, False)
        b = d("szz_z", self.szz, 0, True)
        np.add(a, b, out=acc)
        acc *= self.dtbz
        self.vz += acc
        for s in self.sources:
            if s.kind in ("force_x", "force_z"):
                # Fuente de fuerza aditiva: emite ~``vals`` m/s de velocidad de partícula
                target = self.vx if s.kind == "force_x" else self.vz
                inc = s.signals[s.channel, n] * (2.0 * self.cp_local[s.j, s.i] * self.dt * self.inv_dx)
                np.add.at(target, (s.j, s.i), inc)

        # --- deformaciones
        exx = d("exx", self.vx, 1, False)
        ezz = d("ezz", self.vz, 0, False)
        exz = self.exz
        np.add(d("exz_a", self.vx, 0, True), d("exz_b", self.vz, 1, True), out=exz)

        # --- tensiones (parte no relajada)
        div, a1, a2 = self.div, self.acc2, self.acc3
        np.add(exx, ezz, out=div)
        np.multiply(self.dt_pi, div, out=a1)
        np.multiply(self.dt_mu2, ezz, out=a2)
        self.sxx += a1
        self.sxx -= a2
        np.multiply(self.dt_mu2, exx, out=a2)
        self.szz += a1
        self.szz -= a2
        if self.has_shear:
            np.multiply(self.dt_mu_xz, exz, out=a2)
            self.sxz += a2

        # --- memorias viscoelásticas
        if self.n_mech:
            f_xx, f_zz = acc, a1
            np.multiply(self.dt_pi_t, div, out=a2)  # parte volumétrica común
            np.multiply(self.dt_mu2_t, ezz, out=f_xx)
            np.subtract(a2, f_xx, out=f_xx)
            np.multiply(self.dt_mu2_t, exx, out=f_zz)
            np.subtract(a2, f_zz, out=f_zz)
            for l in range(self.n_mech):
                self._memory(self.sxx, self.rxx[l], f_xx, l, a2)
                self._memory(self.szz, self.rzz[l], f_zz, l, a2)
            if self.has_shear:
                np.multiply(self.dt_mu_t_xz, exz, out=f_xx)
                for l in range(self.n_mech):
                    self._memory(self.sxz, self.rxz[l], f_xx, l, a2)

        for s in self.sources:
            if s.kind == "pressure":
                vals = s.signals[s.channel, n]
                # Fuente de presión aditiva: emite ~``vals`` Pa hacia cada lado
                inc = vals * (2.0 * self.cp_local[s.j, s.i] * self.dt * self.inv_dx)
                np.subtract.at(self.sxx, (s.j, s.i), inc)
                np.subtract.at(self.szz, (s.j, s.i), inc)

    def _memory(self, stress, r, forcing, l, tmp):
        """sigma += (r_new + r_old)/2 ;  r_new = A r_old - B F  (Crank-Nicolson)."""
        a, b = self.mem_a[l], self.mem_b[l]
        np.multiply(r, 0.5 * (1 + a), out=tmp)
        stress += tmp
        np.multiply(forcing, 0.5 * b, out=tmp)
        stress -= tmp
        r *= a
        np.multiply(forcing, b, out=tmp)
        r -= tmp

    # ------------------------------------------------------------------
    def pressure(self) -> np.ndarray:
        return -0.5 * (self.sxx + self.szz)

    def _sample(self, rec: Receiver) -> np.ndarray:
        if rec.quantity == "p":
            vals = -0.5 * (self.sxx[rec.j, rec.i] + self.szz[rec.j, rec.i])
        elif rec.quantity == "vx":
            vals = self.vx[rec.j, rec.i]
        else:
            vals = self.vz[rec.j, rec.i]
        sums = np.bincount(rec.channel, weights=vals, minlength=rec.n_channels)
        counts = np.bincount(rec.channel, minlength=rec.n_channels)
        return sums / np.maximum(counts, 1)

    def run(
        self,
        nt: int,
        snapshot_every: int | None = None,
        snapshot_field: str = "p",
        snapshot_decimate: int = 1,
        track_peak: bool = True,
        progress: bool = False,
    ) -> SimulationResult:
        """Ejecuta ``nt`` pasos. Devuelve trazas, snapshots y mapa de presión máxima."""
        for s in self.sources:
            if s.signals.shape[1] < nt:
                raise ValueError("Las señales de la fuente son más cortas que la simulación")
        res = SimulationResult(dt=self.dt, t=self.time_axis(nt))
        traces = {k: np.zeros((r.n_channels, nt)) for k, r in self.receivers.items()}
        peak = np.zeros(self.grid.shape, self.dtype) if track_peak else None
        steps_per_period = max(int(1.0 / (self.f0 * self.dt)), 1)
        peak_every = max(steps_per_period // 10, 1)
        t_start = time.time()
        for n in range(nt):
            self.step(n)
            for k, r in self.receivers.items():
                traces[k][:, n] = self._sample(r)
            if track_peak and n % peak_every == 0:
                np.maximum(peak, np.abs(self.pressure()), out=peak)
            if snapshot_every and n % snapshot_every == 0:
                res.snapshots.append(self._snapshot(snapshot_field, snapshot_decimate))
                res.snapshot_times.append(float(res.t[n]))
            if n % 200 == 0 and not np.isfinite(self.vx[::8, ::8]).all():
                raise FloatingPointError(f"La simulación se ha vuelto inestable en el paso {n}")
            if progress and n % max(nt // 10, 1) == 0:
                print(f"  paso {n}/{nt} ({100 * n / nt:.0f} %)", flush=True)
        res.traces = traces
        res.peak_pressure = peak
        res.elapsed_s = time.time() - t_start
        return res

    def _snapshot(self, name: str, dec: int) -> np.ndarray:
        if name == "p":
            f = self.pressure()
        elif name == "vmag":
            f = np.hypot(self.vx, self.vz)
        elif name == "vx":
            f = self.vx
        elif name == "vz":
            f = self.vz
        else:
            raise ValueError(f"Campo desconocido '{name}'")
        return np.array(f[::dec, ::dec], dtype=np.float32)


def _shift(a: np.ndarray, axis: int) -> np.ndarray:
    """Vecino +1 a lo largo del eje (replicando el borde)."""
    out = np.empty_like(a)
    if axis == 1:
        out[:, :-1] = a[:, 1:]
        out[:, -1] = a[:, -1]
    else:
        out[:-1, :] = a[1:, :]
        out[-1, :] = a[-1, :]
    return out


def _harmonic_xz(mu: np.ndarray) -> np.ndarray:
    """Media armónica de los 4 vecinos para los nodos (i+1/2, j+1/2); 0 si alguno es fluido."""
    m1, m2 = mu, _shift(mu, 1)
    m3 = _shift(mu, 0)
    m4 = _shift(m2, 0)
    stack = np.stack([m1, m2, m3, m4])
    with np.errstate(divide="ignore"):
        hm = 4.0 / np.sum(1.0 / stack, axis=0)
    return np.where(np.all(stack > 0, axis=0), hm, 0.0)
