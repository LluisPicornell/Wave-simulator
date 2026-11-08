"""Solver FDTD de Maxwell 2D (modo TMy) para radiofrecuencia, microondas y ondas milimétricas.

Campo eléctrico perpendicular al plano (Ey) y magnético en el plano (Hx, Hz):

    dHx/dt =  (1/mu0) dEy/dz
    dHz/dt = -(1/mu0) dEy/dx
    eps dEy/dt + sigma Ey = dHx/dz - dHz/dx - Jy

Malla de Yee de 2.º orden con bordes CPML. Cada tejido se describe por su
permitividad ``eps'`` y conductividad efectiva ``sigma`` a la frecuencia de
trabajo (modelo de Gabriel), es decir, una simulación de banda estrecha: se
emite una onda continua y se extrae el fasor del campo en régimen
estacionario con una DFT de un solo punto.

Rango práctico: ~10 MHz - 300 GHz. Por debajo, el número de pasos por periodo
se dispara y la física es cuasi-estática; por encima, la longitud de onda en
el tejido exige mallas enormes y la penetración es sub-milimétrica.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np

from ..core.pml import CPMLDerivative
from ..core.signals import ramped_sine
from ..materials import dielectric
from ..materials.acoustic import ACOUSTIC
from ..phantoms.phantom import Phantom

C0, EPS0, MU0 = dielectric.C0, dielectric.EPS0, dielectric.MU0
ETA0 = np.sqrt(MU0 / EPS0)


@dataclass
class EMResult:
    f0: float
    phasor: np.ndarray  # fasor complejo de Ey en régimen estacionario (V/m por V/m incidente)
    eps_r: np.ndarray
    sigma: np.ndarray
    sar: np.ndarray  # tasa de absorción específica (W/kg) para 1 V/m incidente
    dt: float
    steps: int
    elapsed_s: float

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.phasor)


def recommended_dx(f0: float, tissues: list[str], ppw: float = 15.0) -> float:
    """Tamaño de celda para ``ppw`` puntos por longitud de onda en el tejido más denso."""
    n_max = max(abs(np.sqrt(complex(dielectric.complex_permittivity(k, f0)))) for k in tissues)
    return C0 / (f0 * n_max * ppw)


class MaxwellTMSolver:
    def __init__(self, phantom: Phantom, f0: float, npml: int = 20, cfl: float = 0.95, dtype=np.float64):
        self.phantom, self.grid, self.f0 = phantom, phantom.grid, f0
        g = self.grid
        # Permitividad compleja por celda: en las celdas porosas, Maxwell Garnett (materials.porous)
        tissues = phantom.property_tissues()
        eps_c = {k: complex(dielectric.complex_permittivity(k, f0)) for k in tissues}
        eps_map = phantom.property_map(lambda k: eps_c[k], dtype=complex, physics="dielectric")
        self.eps_r = eps_map.real.copy()
        self.sigma = -eps_map.imag * 2.0 * np.pi * f0 * EPS0
        n_max = max(abs(np.sqrt(eps_c[k])) for k in tissues)
        self.points_per_wavelength = C0 / (f0 * n_max * g.dx)
        if self.points_per_wavelength < 10:
            warnings.warn(f"Sólo {self.points_per_wavelength:.1f} puntos por longitud de onda en el tejido: "
                          "reduce dx.", stacklevel=2)
        self.dt = cfl * g.dx / (C0 * np.sqrt(2.0))
        dt = self.dt
        eps = self.eps_r * EPS0
        k = self.sigma * dt / (2 * eps)
        self.ca = ((1 - k) / (1 + k)).astype(dtype)
        self.cb = (dt / eps / (1 + k)).astype(dtype)
        self.ch = dt / MU0
        shape = g.shape
        self.ey = np.zeros(shape, dtype)
        self.hx = np.zeros(shape, dtype)
        self.hz = np.zeros(shape, dtype)
        self.d = {name: np.zeros(shape, dtype) for name in ("ey_z", "ey_x", "hx_z", "hz_x")}
        kw = dict(npml=npml, dx=g.dx, dt=dt, c_max=C0, f_ref=f0, dtype=dtype)
        self.pml = {
            "ey_z": CPMLDerivative(shape, 0, True, **kw),
            "ey_x": CPMLDerivative(shape, 1, True, **kw),
            "hx_z": CPMLDerivative(shape, 0, False, **kw),
            "hz_x": CPMLDerivative(shape, 1, False, **kw),
        }
        self.npml = npml
        self.inv_dx = 1.0 / g.dx
        self._src_j = self._src_i = None
        self._src_scale = None

    def set_plane_wave_source(self, z: float | None = None, e0: float = 1.0) -> None:
        """Lámina de corriente horizontal que emite una onda plana de amplitud ``e0`` hacia abajo (y arriba)."""
        g = self.grid
        z = z if z is not None else g.z0 + (self.npml + 2) * g.dx
        j = g.index_of(0.0, z)[0]
        self._src_j = np.full(g.nx, j)
        self._src_i = np.arange(g.nx)
        self._set_scale(e0)

    def set_point_source(self, x: float, z: float, e0: float = 1.0) -> None:
        """Antena de hilo perpendicular al plano (fuente lineal en 3D)."""
        j, i = self.grid.index_of(x, z)
        self._src_j, self._src_i = np.array([j]), np.array([i])
        self._set_scale(e0)

    def _set_scale(self, e0: float) -> None:
        # Lámina de corriente J_s: emite E = eta J_s / 2 a cada lado → J_s = 2 E0 / eta
        eps = self.eps_r[self._src_j, self._src_i]
        eta = ETA0 / np.sqrt(eps)
        self._src_scale = self.cb[self._src_j, self._src_i] * 2.0 * e0 / (eta * self.grid.dx)

    def _deriv(self, name, f, axis, forward):
        out = self.d[name]
        if axis == 0:
            o = out[:-1, :] if forward else out[1:, :]
            np.subtract(f[1:, :], f[:-1, :], out=o)
        else:
            o = out[:, :-1] if forward else out[:, 1:]
            np.subtract(f[:, 1:], f[:, :-1], out=o)
        o *= self.inv_dx
        self.pml[name].apply(out)
        return out

    def run(self, n_periods_steady: float = 4.0, settle_time: float | None = None, progress: bool = False) -> EMResult:
        """Emite una onda continua hasta el régimen estacionario y extrae el fasor a f0."""
        if self._src_j is None:
            self.set_plane_wave_source()
        g = self.grid
        period = 1.0 / self.f0
        steps_per_period = period / self.dt
        if settle_time is None:
            # Arranque suave + varios tránsitos por el dominio a la velocidad más lenta
            n_max = np.sqrt(self.eps_r.max())
            settle_time = 3 * period + 3.0 * (g.width + g.depth) * n_max / C0
        nt_settle = int(np.ceil(settle_time / self.dt))
        nt_dft = int(np.ceil(n_periods_steady * steps_per_period))
        nt = nt_settle + nt_dft
        if nt > 3_000_000:
            warnings.warn(f"{nt} pasos: la frecuencia es demasiado baja para FDTD; usa el régimen "
                          "cuasi-estático o una malla más gruesa.", stacklevel=2)
        stride = max(int(steps_per_period // 24), 1)
        t = (np.arange(nt) + 1) * self.dt
        drive = ramped_sine(t, self.f0, ramp_cycles=3)
        acc = np.zeros(g.shape, complex)
        n_acc = 0
        t0 = time.time()
        for n in range(nt):
            self.hx += self.ch * self._deriv("ey_z", self.ey, 0, True)
            self.hz -= self.ch * self._deriv("ey_x", self.ey, 1, True)
            curl = self._deriv("hx_z", self.hx, 0, False) - self._deriv("hz_x", self.hz, 1, False)
            self.ey *= self.ca
            self.ey += self.cb * curl
            self.ey[self._src_j, self._src_i] -= self._src_scale * drive[n]
            if n >= nt_settle and (n - nt_settle) % stride == 0:
                acc += self.ey * np.exp(-2j * np.pi * self.f0 * t[n])
                n_acc += 1
            if progress and n % max(nt // 10, 1) == 0:
                print(f"  paso {n}/{nt}", flush=True)
        phasor = 2.0 * acc / max(n_acc, 1)
        rho = self.phantom.property_map(lambda k: ACOUSTIC[k].density, physics="acoustic")
        sar = self.sigma * np.abs(phasor) ** 2 / (2.0 * rho)
        return EMResult(self.f0, phasor, self.eps_r, self.sigma, sar, self.dt, nt, time.time() - t0)


def simulate_em_plane_wave(phantom: Phantom, f0: float, npml: int = 20, progress: bool = False) -> EMResult:
    """Onda plana de 1 V/m que incide desde arriba sobre el fantoma."""
    solver = MaxwellTMSolver(phantom, f0, npml=npml)
    solver.set_plane_wave_source()
    return solver.run(progress=progress)
