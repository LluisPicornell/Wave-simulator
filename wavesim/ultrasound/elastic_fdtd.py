"""Solver FDTD elástico 2D (ondas P y S) para ultrasonido en tejidos.

Ecuaciones de la elastodinámica lineal en formulación velocidad-tensión
(Virieux 1986), en malla escalonada:

    rho dvx/dt = d(sxx)/dx + d(sxz)/dz
    rho dvz/dt = d(sxz)/dx + d(szz)/dz
    dsxx/dt = (lambda + 2mu) dvx/dx + lambda dvz/dz   (+ memoria viscoelástica)
    dszz/dt = lambda dvx/dx + (lambda + 2mu) dvz/dz
    dsxz/dt = mu (dvx/dz + dvz/dx)

En un medio transversalmente isótropo con los ejes de la malla (hueso
cortical, opcional) ``lambda + 2mu`` y ``lambda`` pasan a ser C11, C33 y C13
y ``mu`` es C55; el caso isótropo conserva su propio camino, más barato.

- Derivadas espaciales de 4.º orden y salto temporal (leapfrog) de 2.º orden.
- En los fluidos (tejidos blandos, agua) ``mu = 0``: sólo hay ondas de
  presión, ``sxx = szz = -p``. En el hueso aparecen ondas de cizalla y
  conversiones P↔S en las interfaces.
- ``mu`` en los nodos ``sxz`` se promedia armónicamente: si algún vecino es
  fluido, la tensión de cizalla es nula (condición física en la interfaz).
- Junto a una interfaz fluido-sólido, la deformación de cizalla del sólido no
  lee la velocidad tangencial del fluido (que desliza) y el fluido no recibe
  la cizalla del sólido (ver ``_slip_terms``): sin eso, R(θ) agua-hueso no
  convergía al refinar la malla.
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
from .medium import ElasticMedium, christoffel_speeds, elastic_medium

C1, C2 = 9.0 / 8.0, -1.0 / 24.0

#: Cociente de la M-PML (``ElasticSolver(pml_ratio=...)``, ver ``core.pml.CPMLDerivative``). Con la CPML
#: clásica (0) una placa sólida que atraviesa la capa (la cortical en los fantomas longitudinales) es
#: inestable: sus modos de Lamb retrógrados (S1 entre el ZGV y el corte) se amplifican dentro de la capa
#: (Bécache, Fauqueux & Joly, J Comput Phys 188:399, 2003). El análisis de autovalores de la capa con
#: coeficientes congelados y las simulaciones de 40 000 pasos piden p ≥ 0.3-0.5
#: (``docs/validacion/estabilidad.md``). Precio: la capa refleja más (−53 dB en agua, −28 dB en hueso).
#: Por defecto (``pml_ratio=None``) sólo se activa si hace falta (``_layer_crosses_pml``): así los
#: montajes sin capas sólidas en la CPML (tejido blando, sólidos homogéneos, semiespacios) no cambian.
PML_RATIO = 0.5


def _layer_crosses_pml(solid: np.ndarray, npml: int) -> bool:
    """¿Hay en alguna franja CPML una capa sólida limitada por fluido a los dos lados, a lo largo de la franja?

    Es la geometría de guía de ondas con modos retrógrados (placa, cortical entre tejido blando y médula). Un
    sólido homogéneo o un semiespacio que llega al borde exterior no la tiene y es estable con la CPML clásica.
    """
    if npml <= 0 or not solid.any():
        return False
    strips = [solid[:, :npml], solid[:, -npml:], solid[:npml, :].T, solid[-npml:, :].T]
    for st in strips:
        # a lo largo de cada columna de la franja: sólido con fluido antes y después
        idx = np.arange(st.shape[0])[:, None]
        has = st.any(axis=0)
        if not has.any():
            continue
        first = np.where(st, idx, st.shape[0]).min(axis=0)
        last = np.where(st, idx, -1).max(axis=0)
        if np.any(has & (first > 0) & (last < st.shape[0] - 1)):
            return True
    return False

#: Cada derivada con CPML: (eje, ¿nodo desplazado media celda a lo largo del eje?, ¿y a lo largo del otro?).
#: Las de tensión viven en los nodos de velocidad (vx en (i+1/2, j), vz en (i, j+1/2)) y las de velocidad
#: en los de tensión (sxx, szz en (i, j); sxz en (i+1/2, j+1/2)).
_DERIVATIVE_NODES = {
    "sxx_x": (1, True, False), "sxz_z": (0, False, True), "sxz_x": (1, False, True), "szz_z": (0, True, False),
    "exx": (1, False, False), "ezz": (0, False, False), "exz_a": (0, True, True), "exz_b": (1, True, True),
}


@dataclass
class Source:
    """Fuente aplicada sobre un conjunto de celdas.

    ``signals`` tiene forma ``(n_canales, nt)``; ``channel[k]`` indica qué
    canal alimenta a la celda ``(j[k], i[k])``. ``kind``: "pressure" (presión
    en Pa), "force_x" o "force_z" (fuerza puntual aditiva sobre la velocidad).

    Normalización de las fuentes de fuerza: el incremento se escala con la
    velocidad *P* local (``2 cp dt / dx``), de modo que ``signals`` es la
    velocidad de partícula (m/s) de la onda P emitida en un fluido. En un sólido
    la misma fuerza excita también ondas S, y su amplitud sale ~``cp/cs`` veces
    mayor que ``signals`` (la normalización no usa ``cs``): ``signals`` no es la
    velocidad de partícula de la onda S. Calibrar a mano si se necesita ese valor.
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
        atenuación). Con 1 el error frente a la ley de potencia llega al
        ~14 % en la banda del pulso; con 2, < 2 % en [0.5, 1.8]·f0; con 3,
        < 2 % en [0.3, 2.5]·f0, para exponentes de 1 a 2 (ver ``attenuation``).
      - ``scattering``: escala de la heterogeneidad que produce speckle.
      - ``cfl``: fracción del paso temporal máximo estable.
      - ``backend``: "numpy" (referencia), "numba" (kernels fusionados y
        paralelos de ``_numba_kernels``; error si Numba no está instalado o si
        ``dtype`` no es float32) o "auto" (Numba si está disponible, si no NumPy).
      - ``anisotropic_bone``: cortical transversalmente isótropa (ver
        ``medium.elastic_medium``). Si se pasa ``medium``, manda el medio.
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
        backend: str = "auto",
        anisotropic_bone: bool = False,
        pml_ratio: float | None = None,
    ):  # ``pml_ratio``: cociente de la M-PML; None = ``PML_RATIO`` sólo si una capa sólida cruza la CPML
        self.backend = self._resolve_backend(backend, dtype)
        self.phantom = phantom
        self.grid = phantom.grid
        self.f0 = f0
        self.dtype = dtype
        self.medium = medium or elastic_medium(phantom, f0, n_mechanisms, scattering, attenuation, anisotropic_bone)
        m = self.medium
        dx = self.grid.dx
        self.inv_dx = 1.0 / dx
        self.n_mech = m.tau_sigma.size

        # Módulos relajados corregidos por dispersión y no relajados
        L = self.n_mech
        w_p, w_s = m.weights("p"), m.weights("s")
        fp = att.relaxed_modulus_factor(w_p, m.tau_sigma, f0) if L else 1.0
        fs = att.relaxed_modulus_factor(w_s, m.tau_sigma, f0) if L else 1.0
        self.anisotropic = m.stiffness is not None
        if self.anisotropic:
            # La parte "P" (C11, C33 y C13 + 2 C55) relaja con los pesos de la onda P y C55
            # con los de la S; en un medio isótropo son lambda + 2 mu y mu, como en el camino
            # isótropo, y la memoria de C13 es la de lambda = (lambda + 2 mu) - 2 mu
            a11, a33, a13, a55 = m.stiffness
            p11, p33, p13 = (m.rho * a * fp for a in (a11, a33, a13 + 2 * a55))
            mu_r = m.rho * a55 * fs
        else:
            pi_r = m.rho * m.cp**2 * fp
            mu_r = m.rho * m.cs**2 * fs
        relax_p = 1 + w_p.sum(axis=0)
        mu_u = mu_r * (1 + w_s.sum(axis=0))
        if self.anisotropic:
            c11_u, c33_u = p11 * relax_p, p33 * relax_p
            c13_u = p13 * relax_p - 2 * mu_u
            v2 = _stability_modulus(c11_u, c33_u, c13_u, mu_u) / m.rho
        else:
            pi_u = pi_r * relax_p
            v2 = pi_u / m.rho
        cp_u_max = float(np.sqrt(np.max(v2)))
        self.dt = cfl * dx / (cp_u_max * np.sqrt(2.0) * (C1 + abs(C2)))
        dt = self.dt

        self._check_sampling(dx)

        def f(a):
            return np.ascontiguousarray(a, dtype=dtype)

        # Coeficientes ya multiplicados por dt
        self.dt_mu2 = f(dt * 2 * mu_u)
        self.dt_mu_xz = f(dt * _harmonic_xz(mu_u))
        if self.anisotropic:
            self.dt_c11, self.dt_c33, self.dt_c13 = f(dt * c11_u), f(dt * c33_u), f(dt * c13_u)
        else:
            self.dt_pi = f(dt * pi_u)
        self.dtbx = f(dt * 2.0 / (m.rho + _shift(m.rho, 1)))
        self.dtbz = f(dt * 2.0 / (m.rho + _shift(m.rho, 0)))
        self.has_shear = bool(np.any(mu_u > 0))
        if self.n_mech:
            # Un coeficiente por mecanismo, (L, nz, nx): cada uno con su peso
            if self.anisotropic:
                self.dt_c11_t, self.dt_c33_t = f(dt * p11 * w_p), f(dt * p33 * w_p)
                self.dt_c13_t = f(dt * (p13 * w_p - 2 * mu_r * w_s))
            else:
                self.dt_pi_t = f(dt * pi_r * w_p)
                self.dt_mu2_t = f(dt * 2 * mu_r * w_s)
            self.dt_mu_t_xz = f(np.stack([dt * _harmonic_xz(mu_r * w) for w in w_s]))
            ts = m.tau_sigma
            self.mem_a = (1 - dt / (2 * ts)) / (1 + dt / (2 * ts))
            self.mem_b = (dt / ts) / (1 + dt / (2 * ts))

        shape = self.grid.shape
        z = lambda: np.zeros(shape, dtype)  # noqa: E731
        self.vx, self.vz, self.sxx, self.szz, self.sxz = z(), z(), z(), z(), z()
        if self.backend == "numba":
            # Los kernels necesitan las memorias en una pila (L, nz, nx); las
            # listas son vistas de ella para que el resto del código no cambie
            self._mem = {k: np.zeros((L, *shape), dtype) for k in ("rxx", "rzz", "rxz")}
            self.rxx, self.rzz, self.rxz = (list(self._mem[k]) for k in ("rxx", "rzz", "rxz"))
        else:
            self.rxx = [z() for _ in range(L)]
            self.rzz = [z() for _ in range(L)]
            self.rxz = [z() for _ in range(L)]
        # Un buffer por derivada: sus bordes nunca escritos permanecen a cero
        self.d = {k: z() for k in ("sxx_x", "sxz_z", "sxz_x", "szz_z", "exx", "ezz", "exz_a", "exz_b")}
        self.acc, self.acc2, self.acc3, self.div, self.exz = z(), z(), z(), z(), z()
        self.tmp_x = np.zeros((shape[0], shape[1] - 3), dtype)
        self.tmp_z = np.zeros((shape[0] - 3, shape[1]), dtype)

        if pml_ratio is None:
            pml_ratio = PML_RATIO if _layer_crosses_pml(mu_u > 0, npml) else 0.0
        self.pml_ratio = float(pml_ratio)
        cp_kw = dict(npml=npml, dx=dx, dt=dt, c_max=cp_u_max, f_ref=f0, dtype=dtype, ratio=pml_ratio)
        # (eje, nodo desplazado a lo largo del eje, nodo desplazado a lo largo del otro eje)
        self.pml = {
            name: CPMLDerivative(shape, ax, st, staggered_other=st_o, **cp_kw)
            for name, (ax, st, st_o) in _DERIVATIVE_NODES.items()
        }
        self.npml = npml
        # Fuera de la CPML (sus memorias psi no se tocan) y lejos de los bordes de los buffers
        slip = _slip_terms(mu_u > 0, self.dt_mu_xz > 0, npml + 3) if self.has_shear else {}
        self._slip = {k: (t, S, (W * self.inv_dx).astype(dtype)) for k, (t, S, W) in slip.items()}
        self.sources: list[Source] = []
        self.receivers: dict[str, Receiver] = {}
        self.cp_local = m.cp
        if self.backend == "numba":
            self._init_numba()

    @staticmethod
    def _resolve_backend(backend: str, dtype) -> str:
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
        if np.dtype(dtype) != np.float32:
            # Los kernels están escritos y probados sólo para float32
            if backend == "numba":
                raise ValueError("El backend 'numba' sólo admite dtype=np.float32")
            return "numpy"
        return "numba"

    def _init_numba(self) -> None:
        """Prepara los argumentos de los kernels (se importan aquí: Numba es opcional)."""
        from . import _numba_kernels as nk

        self._nk = nk
        f32 = np.float32
        self._nb_c1 = f32(C1 * self.inv_dx)
        self._nb_c2 = f32(C2 * self.inv_dx)
        self._nb_pml = {k: nk.pack_pml(v) for k, v in self.pml.items()}
        # Marcadores de posición para los coeficientes que el kernel no lee
        flat, stack = np.zeros((1, 1), self.dtype), np.zeros((1, 1, 1), self.dtype)
        if self.n_mech:
            self._nb_mem = (np.asarray(self.mem_a, np.float64), np.asarray(self.mem_b, np.float64))
        else:
            self._nb_mem = (np.zeros(0), np.zeros(0))
        xz_t = self.dt_mu_t_xz if self.n_mech else stack
        if self.anisotropic:
            self._nb_iso = (flat, self.dt_mu2, self.dt_mu_xz, stack, stack, xz_t)
            relax = (self.dt_c11_t, self.dt_c33_t, self.dt_c13_t) if self.n_mech else (stack,) * 3
            self._nb_aniso = (True, self.dt_c11, self.dt_c33, self.dt_c13, *relax)
        else:
            relax = (self.dt_pi_t, self.dt_mu2_t) if self.n_mech else (stack,) * 2
            self._nb_iso = (self.dt_pi, self.dt_mu2, self.dt_mu_xz, *relax, xz_t)
            self._nb_aniso = (False, flat, flat, flat, stack, stack, stack)

    # ------------------------------------------------------------------
    def _check_sampling(self, dx: float) -> None:
        m = self.medium
        speeds = [m.cp.ravel(), m.cs[m.cs > 0].ravel()]
        if m.stiffness is not None:
            # En un medio anisótropo la onda más lenta puede ir en diagonal
            solid = m.stiffness[3] > 0
            for theta in np.linspace(0.0, np.pi / 2, 7):
                speeds += list(christoffel_speeds(m.stiffness[:, solid], theta))
        c_min = float(np.concatenate(speeds).min())
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
        if name in self._slip:
            # junto a una interfaz fluido-sólido (ver ``_slip_terms``)
            t, vals = self._slip_delta(name, f)
            out.reshape(-1)[t] += vals
        return out

    def _slip_delta(self, name, f):
        """Corrección de la derivada ``name`` de ``f``: índices planos de destino (sin repetir) y valores."""
        t, S, W = self._slip[name]
        return t, np.sum(W * f.reshape(-1)[S], axis=1)

    def _slip_correct_numba(self, stage: str) -> None:
        """Lleva el resultado de un kernel Numba al del paso NumPy con los términos de ``_slip_terms``.

        Los kernels usan el estencil de 4.º orden sin corregir. Las
        actualizaciones son lineales en cada derivada y los campos que se
        derivan no cambian dentro del kernel, así que basta sumar después el
        efecto de la corrección en los pocos nodos de la interfaz (fuera de la CPML).
        """
        if stage == "velocity":
            for name, field, coef in (("sxz_z", self.vx, self.dtbx), ("sxz_x", self.vz, self.dtbz)):
                if name in self._slip:
                    t, vals = self._slip_delta(name, self.sxz)
                    field.reshape(-1)[t] += coef.reshape(-1)[t] * vals
            return
        sxz = self.sxz.reshape(-1)
        for name, f in (("exz_a", self.vx), ("exz_b", self.vz)):
            if name not in self._slip:
                continue
            t, vals = self._slip_delta(name, f)
            sxz[t] += self.dt_mu_xz.reshape(-1)[t] * vals
            if self.n_mech:
                # misma corrección en las memorias (Crank-Nicolson, ver ``_memory``), con el peso de cada mecanismo
                for m in range(self.n_mech):
                    forcing = self.dt_mu_t_xz[m].reshape(-1)[t] * vals
                    sxz[t] -= (0.5 * self.mem_b[m]) * forcing
                    self.rxz[m].reshape(-1)[t] -= self.mem_b[m] * forcing

    def _apply_force_sources(self, n: int) -> None:
        for s in self.sources:
            if s.kind in ("force_x", "force_z"):
                # Fuente de fuerza aditiva. La escala usa cp: emite ~``vals`` m/s de velocidad
                # de partícula sólo para la onda P en un fluido; la onda S en un sólido sale
                # ~cp/cs veces mayor (la física no se corrige, sólo se documenta).
                target = self.vx if s.kind == "force_x" else self.vz
                inc = s.signals[s.channel, n] * (2.0 * self.cp_local[s.j, s.i] * self.dt * self.inv_dx)
                np.add.at(target, (s.j, s.i), inc)

    def _apply_pressure_sources(self, n: int) -> None:
        for s in self.sources:
            if s.kind == "pressure":
                vals = s.signals[s.channel, n]
                # Fuente de presión aditiva: emite ~``vals`` Pa hacia cada lado
                inc = vals * (2.0 * self.cp_local[s.j, s.i] * self.dt * self.inv_dx)
                np.subtract.at(self.sxx, (s.j, s.i), inc)
                np.subtract.at(self.szz, (s.j, s.i), inc)

    def _step_numba(self, n: int) -> None:
        """Mismo paso que ``step`` con dos barridos fusionados; fuentes en NumPy."""
        nk, pml = self._nk, self._nb_pml
        nk.velocity_kernel(
            self.vx, self.vz, self.sxx, self.szz, self.sxz, self.dtbx, self.dtbz,
            self._nb_c1, self._nb_c2,
            pml["sxx_x"], pml["sxz_z"], pml["sxz_x"], pml["szz_z"],
        )  # fmt: skip
        if self._slip:
            self._slip_correct_numba("velocity")
        self._apply_force_sources(n)
        nk.stress_kernel(
            self.vx, self.vz, self.sxx, self.szz, self.sxz,
            self._mem["rxx"], self._mem["rzz"], self._mem["rxz"],
            *self._nb_iso, *self._nb_mem,
            self.has_shear, self._nb_c1, self._nb_c2,
            pml["exx"], pml["ezz"], pml["exz_a"], pml["exz_b"], *self._nb_aniso,
        )  # fmt: skip
        if self._slip:
            self._slip_correct_numba("stress")
        self._apply_pressure_sources(n)

    def step(self, n: int) -> None:
        """Avanza un paso: velocidades a t+dt/2 y tensiones a t+dt."""
        if self.backend == "numba":
            self._step_numba(n)
            return
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
        self._apply_force_sources(n)

        # --- deformaciones
        exx = d("exx", self.vx, 1, False)
        ezz = d("ezz", self.vz, 0, False)
        exz = self.exz
        np.add(d("exz_a", self.vx, 0, True), d("exz_b", self.vz, 1, True), out=exz)

        # --- tensiones (parte no relajada)
        div, a1, a2 = self.div, self.acc2, self.acc3
        if self.anisotropic:
            np.multiply(self.dt_c11, exx, out=a1)
            self.sxx += a1
            np.multiply(self.dt_c13, ezz, out=a1)
            self.sxx += a1
            np.multiply(self.dt_c13, exx, out=a1)
            self.szz += a1
            np.multiply(self.dt_c33, ezz, out=a1)
            self.szz += a1
        else:
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

        # --- memorias viscoelásticas (cada mecanismo con su propio peso)
        if self.n_mech:
            f_xx, f_zz = acc, a1
            for m in range(self.n_mech):
                if self.anisotropic:
                    np.multiply(self.dt_c11_t[m], exx, out=f_xx)
                    np.multiply(self.dt_c13_t[m], ezz, out=a2)
                    f_xx += a2
                    np.multiply(self.dt_c13_t[m], exx, out=f_zz)
                    np.multiply(self.dt_c33_t[m], ezz, out=a2)
                    f_zz += a2
                else:
                    np.multiply(self.dt_pi_t[m], div, out=a2)  # parte volumétrica común
                    np.multiply(self.dt_mu2_t[m], ezz, out=f_xx)
                    np.subtract(a2, f_xx, out=f_xx)
                    np.multiply(self.dt_mu2_t[m], exx, out=f_zz)
                    np.subtract(a2, f_zz, out=f_zz)
                self._memory(self.sxx, self.rxx[m], f_xx, m, a2)
                self._memory(self.szz, self.rzz[m], f_zz, m, a2)
                if self.has_shear:
                    np.multiply(self.dt_mu_t_xz[m], exz, out=f_xx)
                    self._memory(self.sxz, self.rxz[m], f_xx, m, a2)

        self._apply_pressure_sources(n)

    def _memory(self, stress, r, forcing, mech, tmp):
        """sigma += (r_new + r_old)/2 ;  r_new = A r_old - B F  (Crank-Nicolson)."""
        a, b = self.mem_a[mech], self.mem_b[mech]
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


def _stability_modulus(c11, c33, c13, c55):
    """Mayor autovalor del operador elástico discreto (en unidades de módulo) para el límite de estabilidad.

    El peor número de onda del esquema escalonado es el de Nyquist en las dos
    direcciones; allí la matriz es ``[[C11 + C55, C13 + C55], [C13 + C55,
    C33 + C55]]`` y su mayor autovalor, dividido por 2, es ``lambda + 2 mu``
    en un medio isótropo (el ``cp`` de la condición de siempre).
    """
    half = 0.5 * (c11 + c33) + c55
    root = np.sqrt(0.25 * (c11 - c33) ** 2 + (c13 + c55) ** 2)
    return 0.5 * (half + root)


def _harmonic_xz(mu: np.ndarray) -> np.ndarray:
    """Media armónica de los 4 vecinos para los nodos (i+1/2, j+1/2); 0 si alguno es fluido."""
    m1, m2 = mu, _shift(mu, 1)
    m3 = _shift(mu, 0)
    m4 = _shift(m2, 0)
    stack = np.stack([m1, m2, m3, m4])
    with np.errstate(divide="ignore"):
        hm = 4.0 / np.sum(1.0 / stack, axis=0)
    return np.where(np.all(stack > 0, axis=0), hm, 0.0)


def _slip_terms(solid: np.ndarray, shear: np.ndarray, margin: int) -> dict[str, tuple[np.ndarray, ...]]:
    """Términos que corrigen las derivadas de cizalla junto a una interfaz fluido-sólido.

    En la deformación de cizalla de un nodo ``sxz`` del sólido, el estencil de
    4.º orden lee con peso C2 la velocidad tangencial de un nodo situado al otro
    lado de la interfaz. El fluido no viscoso desliza, así que esa velocidad no
    es la prolongación de la del sólido y el salto entra con peso C2/dx: un
    error O(1) que no baja al refinar la malla (R(θ) agua-hueso quedaba ~1 %
    alto a 45° tanto con 25 como con 35 puntos por longitud de onda, y 2.7 % a
    50° con 25; ``docs/validacion/elastico.md``). Ese punto exterior
    se sustituye por el nodo interior del mismo lado (extrapolación de orden
    cero: el estencil sigue anulando las constantes).

    Para que el esquema siga conservando la energía, y por tanto siendo
    estable, el cambio traspuesto se aplica a la ecuación del movimiento: el
    nodo de velocidad del fluido deja de recibir esa tensión de cizalla (que en
    un fluido es nula) y la recibe el nodo interior del sólido. Pasar a 2.º
    orden sólo las derivadas, sin esta simetría, rompía la conservación: un
    fantoma con fractura sin atenuación crecía exponencialmente tras ~10⁴ pasos.

    ``solid``: celdas con ``mu > 0``; ``shear``: nodos ``sxz`` con cizalla. Se
    excluye la franja de ``margin`` celdas del borde (la CPML y los bordes de
    los buffers de derivadas). Devuelve, para cada derivada (``exz_a`` = dvx/dz
    y ``exz_b`` = dvz/dx en nodos ``sxz``; ``sxz_z`` y ``sxz_x`` en nodos de
    velocidad), ``(t, S, W)`` en índices planos: ``deriv.flat[t] += sum(W * f.flat[S], axis=1) / dx``.
    Cada destino ``t`` aparece una vez y sus términos ocupan una fila de ``S`` y
    ``W`` (relleno con peso 0): así la corrección es una suma densa y rápida.
    """
    nz, nx = solid.shape
    fluid = ~solid
    j, i = np.nonzero(shear)
    keep = (j >= margin) & (j < nz - margin) & (i >= margin) & (i < nx - margin)
    j, i = j[keep], i[keep]
    terms = {k: [] for k in ("exz_a", "exz_b", "sxz_z", "sxz_x")}

    def add(strain, momentum, sel, out_off, in_off):
        """Sustituye el punto exterior ``out_off`` por el interior ``in_off`` (desplazamientos (dj, di) del nodo)."""
        jj, ii = j[sel], i[sel]
        (oj, oi), (nj, ni) = out_off, in_off
        # deformación en el nodo sxz: -C2 f[exterior] -> -C2 f[interior] (o +C2, según el lado)
        sign = 1.0 if oj + oi < 0 else -1.0
        w = np.full(jj.size, sign * C2)
        terms[strain] += [(jj, ii, jj + oj, ii + oi, w), (jj, ii, jj + nj, ii + ni, -w)]
        # su traspuesta en la ecuación del movimiento (D_b = -D_f^T)
        terms[momentum] += [(jj + oj, ii + oi, jj, ii, -w), (jj + nj, ii + ni, jj, ii, w)]

    def cell(dj, di):
        return fluid[np.clip(j + dj, 0, nz - 1), np.clip(i + di, 0, nx - 1)]

    # vx[j-1] y vx[j+2] están entre las celdas (fila, i) y (fila, i+1); vz[i-1] y vz[i+2], entre (j, col) y (j+1, col)
    add("exz_a", "sxz_z", cell(-1, 0) | cell(-1, 1), (-1, 0), (0, 0))
    add("exz_a", "sxz_z", cell(2, 0) | cell(2, 1), (2, 0), (1, 0))
    add("exz_b", "sxz_x", cell(0, -1) | cell(1, -1), (0, -1), (0, 0))
    add("exz_b", "sxz_x", cell(0, 2) | cell(1, 2), (0, 2), (0, 1))
    out = {}
    for name, parts in terms.items():
        tj, ti, sj, si, w = (np.concatenate(c) for c in zip(*parts))
        if not tj.size:
            continue
        t, inv = np.unique(tj * nx + ti, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=t.size)
        col = np.arange(inv.size) - np.repeat(np.cumsum(counts) - counts, counts)
        S = np.zeros((t.size, counts.max()), np.int64)
        W = np.zeros((t.size, counts.max()))
        S[inv[order], col] = (sj * nx + si)[order]
        W[inv[order], col] = w[order]
        out[name] = (t, S, W)
    return out

