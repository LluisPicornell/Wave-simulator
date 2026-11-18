"""Solver FDTD de Maxwell 3D vectorial (radio y microondas a través de la extremidad).

Es el equivalente 3D de los solvers 2D TM (``em.fdtd_tm`` y
``em.fdtd_dispersive``): las seis componentes en la malla de Yee de 2.º orden,
CPML en las seis caras (los coeficientes de ``core.pml.cpml_coefficients``,
la memoria de ``ultrasound.elastic3d.CPML3D``) y los tejidos de un
``Phantom3D``:

    dH/dt = -(1/mu0) rot E
    eps dE/dt + sigma E (+ sum_k J_k) = rot H

- **Banda estrecha** (``Maxwell3DSolver``): ``eps'`` y ``sigma`` de cada tejido
  a ``f0`` (Gabriel), onda continua y fasor por DFT corriente, como ``fdtd_tm``.
- **Banda ancha** (``BroadbandMaxwell3DSolver``): los mismos polos de Debye
  comunes (``materials.dispersive``) y el mismo ADE que ``fdtd_dispersive``,
  un pulso y la DFT a muchas frecuencias de una sola simulación.

Disposición (arrays ``(nz, ny, nx)``, índice ``[k, j, i]``; x = eje del hueso,
y = lateral, z = profundidad):

    Ex: (i+1/2, j, k)    Ey: (i, j+1/2, k)    Ez: (i, j, k+1/2)
    Hx: (i, j+1/2, k+1/2)    Hy: (i+1/2, j, k+1/2)    Hz: (i+1/2, j+1/2, k)

El plano ``y = cte`` contiene Ey, Hx y Hz en las posiciones del modo TM 2D
(Ey en el centro de la celda, Hx desplazado en z, Hz en x): con campos
uniformes en y el solver reproduce ``MaxwellTMSolver``.

**Materiales en los nodos.** Las etiquetas del fantoma están en los centros de
celda; cada componente de E está a media celda a lo largo de su eje y toma la
media aritmética de ``eps*`` (``eps'`` y ``sigma``, o ``eps_inf``, ``sigma`` y
``d_eps_k``) de las dos celdas que separa (como gprMax). La media es lineal,
así que con polos comunes sigue siendo un Debye pasivo. En una capa plana
normal a z (la losa) las componentes Ex y Ey caen en los centros de celda en z
y la interfaz queda a medio camino entre dos nodos, como en 2D. En banda
estrecha, ``subcell_node_permittivity`` da un promedio sub-celda a partir de
una geometría analítica (``node_eps``); ver ``docs/validacion/maxwell3d.md``.

**Fuentes.**

- ``set_plane_wave``: onda plana de incidencia normal (+z) por campo
  total/campo dispersado (TF/SF) con la polarización pedida (E a lo largo de x
  o de y, o «paralela/perpendicular» al eje del hueso de ``meta["bone_axis"]``).
  El campo incidente sale de una malla de Yee 1D auxiliar con el medio de fondo
  y el mismo ``dt``: a incidencia normal el esquema 3D de una onda plana *es*
  ese esquema 1D, así que el campo dispersado fuera de la caja es del orden
  del redondeo. Por defecto la caja ocupa todo x e y (sólo hay caras en z): la
  onda es lateralmente infinita y la CPML lateral sólo ve el campo dispersado.
  Los fasores se normalizan con el fasor incidente en ``z_ref``.
- ``set_current_sheet``: lámina de corriente blanda (la fuente del 2D, con su
  misma normalización: sirve para reproducir el 2D bit a bit).
- ``set_dipole``: dipolo eléctrico puntual blando (momento ``I l`` en A·m) en
  un nodo de E; el fasor sale por A·m de momento.

**Resultados.** ``EM3DResult``: los tres fasores de E llevados a los centros de
celda (media de los dos nodos vecinos a lo largo de su eje), ``|E|``,
``SAR = sigma |E|² / (2 rho)`` (densidad de ``materials.acoustic``), la
potencia absorbida, cortes por planos y el SAR promediado en cubos de 1 g o
10 g (``averaged_sar``).

**Backends**: ``"numpy"`` (referencia), ``"numba"`` (``_numba_fdtd3d``: dos
barridos fusionados por paso, paralelos por planos z) o ``"auto"``. Ambos en
float32 (por defecto: mitad de memoria) o float64.

Memoria (float32): 6 campos + 6 coeficientes = 48 B/celda, más la CPML
(``12 · 2 · npml / n`` campos por eje) y la DFT (3 fasores complex128 = 48
B/celda en banda estrecha); cada polo de Debye añade 24 B/celda (J y beta por
componente). ``memory_bytes()`` da la cifra del solver.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..core.grid3d import Grid3D
from ..core.pml import cpml_coefficients
from ..core.signals import ramped_sine
from ..materials import dielectric
from ..materials.acoustic import ACOUSTIC
from ..materials.dispersive import DebyeMedium, TissueDebyeModel, tissue_debye_model
from ..ultrasound.elastic3d import CPML3D
from .fdtd_dispersive import wideband_pulse, wideband_pulse_params

C0, EPS0, MU0 = dielectric.C0, dielectric.EPS0, dielectric.MU0
ETA0 = np.sqrt(MU0 / EPS0)

#: Ejes de los arrays: 0 = z, 1 = y, 2 = x.
AXES = {"x": 2, "y": 1, "z": 0}
AXIS_NAME = {2: "x", 1: "y", 0: "z"}
COMPONENTS = ("x", "y", "z")

#: Avance de H: componente -> ((campo E, eje, signo), ...) con ``H += dt/mu0 (s1 dE1/da1 + s2 dE2/da2)``.
H_TERMS = {
    "hx": (("ey", 0, 1.0), ("ez", 1, -1.0)),
    "hy": (("ez", 2, 1.0), ("ex", 0, -1.0)),
    "hz": (("ex", 1, 1.0), ("ey", 2, -1.0)),
}
#: Rotacional de H en cada componente de E: ``(rot H)_c = s1 dH1/da1 + s2 dH2/da2``.
E_TERMS = {
    "ex": (("hz", 1, 1.0), ("hy", 0, -1.0)),
    "ey": (("hx", 0, 1.0), ("hz", 2, -1.0)),
    "ez": (("hy", 2, 1.0), ("hx", 1, -1.0)),
}
#: Orden de las tuplas de CPML que reciben los kernels.
H_PML = ("ey_z", "ez_y", "ez_x", "ex_z", "ex_y", "ey_x")
E_PML = ("hz_y", "hy_z", "hx_z", "hz_x", "hy_x", "hx_y")


# ----------------------------------------------------------------------------
# Materiales
# ----------------------------------------------------------------------------
def node_average(cell_map: np.ndarray, axis: int) -> np.ndarray:
    """Media de cada celda y la siguiente a lo largo de ``axis`` (las dos que separa un nodo i+1/2).

    La última fila no tiene vecina y conserva su valor. Admite mapas con ejes
    delanteros (p. ej. ``(K, nz, ny, nx)``): ``axis`` cuenta desde el final.
    """
    a = np.asarray(cell_map)
    ax = a.ndim - 3 + axis
    n = a.shape[ax]
    out = a.astype(np.result_type(a.dtype, np.float64), copy=True)
    if n > 1:
        lo = [slice(None)] * a.ndim
        hi = [slice(None)] * a.ndim
        lo[ax], hi[ax] = slice(0, n - 1), slice(1, n)
        out[tuple(lo)] = 0.5 * (a[tuple(lo)] + a[tuple(hi)])
    return out


def subcell_node_permittivity(grid: Grid3D, eps_at, factor: int = 5) -> np.ndarray:
    """Promedio sub-celda de la permitividad en los nodos de Ex, Ey, Ez: ``(3, nz, ny, nx)`` complejo.

    ``eps_at(x, y, z)`` da la permitividad compleja en puntos arbitrarios (la
    geometría exacta, p. ej. una esfera). Cada nodo de la componente ``a``
    promedia su celda dual (la celda de lado ``dx`` centrada en el nodo) con
    ``factor³`` puntos y toma la diagonal del tensor de Meep (Farjadpour et
    al., Opt. Lett. 31:2972, 2006): ``eps_aa = n_a² / <1/eps> + (1 - n_a²) <eps>``,
    con ``n`` la normal de la interfaz estimada por el primer momento de
    ``eps'`` en la celda. Con una interfaz plana normal (paralela) a ``a`` da
    la media armónica (aritmética), que es la exacta. Sólo banda estrecha:
    la media armónica de dos Debye no es un Debye.
    """
    off = ((np.arange(factor) + 0.5) / factor - 0.5) * grid.dx
    oz, oy, ox = (o.ravel() for o in np.meshgrid(off, off, off, indexing="ij"))
    out = np.empty((3,) + grid.shape, complex)
    for c, comp in enumerate(COMPONENTS):
        shift = np.zeros(3)
        shift[AXES[comp]] = 0.5 * grid.dx  # (z, y, x)
        yy, xx = np.meshgrid(grid.y + shift[1], grid.x + shift[2], indexing="ij")
        for k, z in enumerate(grid.z + shift[0]):
            e = np.asarray(eps_at(xx[..., None] + ox, yy[..., None] + oy, z + oz), dtype=complex)
            mean = e.mean(axis=-1)
            harm = 1.0 / (1.0 / e).mean(axis=-1)
            w = e.real - e.real.mean(axis=-1, keepdims=True)
            nvec = np.stack([(w * o).sum(axis=-1) for o in (oz, oy, ox)])
            norm = np.sqrt((nvec ** 2).sum(axis=0))
            na2 = np.where(norm > 0, (nvec[AXES[comp]] / np.where(norm > 0, norm, 1.0)) ** 2, 0.0)
            out[c, k] = na2 * harm + (1.0 - na2) * mean
    return out


def debye_maps_3d(phantom, model: TissueDebyeModel) -> DebyeMedium:
    """Mapas ``(nz, ny, nx)`` de ``eps_inf`` y ``sigma`` y ``(K, nz, ny, nx)`` de ``d_eps`` de un ``Phantom3D``.

    Equivale a ``materials.dispersive.debye_maps`` para fantomas sin porosidad
    (los 3D son de capas homogéneas).
    """
    from ..phantoms.phantom import TISSUE_INDEX  # import tardío: phantoms importa materials

    present = phantom.present_tissues()
    missing = [k for k in present if k not in model.tissues]
    if missing:
        raise KeyError(f"el modelo de Debye no incluye: {', '.join(missing)}")
    lut = np.zeros((len(TISSUE_INDEX), model.n_poles + 2))
    for k in present:
        m = model.tissues[k]
        lut[TISSUE_INDEX[k]] = np.concatenate([[m.eps_inf, m.sigma], m.delta_eps])
    params = np.moveaxis(lut[phantom.labels], -1, 0)
    return DebyeMedium(params[0].copy(), params[1].copy(), params[2:].copy(), model.tau.copy())


def _density(phantom) -> np.ndarray:
    return phantom.property_map(lambda k: ACOUSTIC[k].density, dtype=np.float64)


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError(f"backend desconocido '{backend}' (use 'auto', 'numpy' o 'numba')")
    if backend == "numpy":
        return "numpy"
    try:
        import numba  # noqa: F401
    except ImportError:
        if backend == "numba":
            raise ImportError("El backend 'numba' requiere Numba: instálalo con `pip install wavesim[fast]` "
                              "o usa backend='numpy'.") from None
        return "numpy"
    return "numba"


def polarization_axis(polarization: str, meta: dict | None = None) -> str:
    """Eje (``"x"`` o ``"y"``) del campo eléctrico incidente.

    ``"x"``/``"y"`` directamente, o ``"parallel"``/``"perpendicular"`` (E
    paralelo o perpendicular al eje del hueso, ``meta["bone_axis"]``, que debe
    ser horizontal: ``"x"`` o ``"y"``).
    """
    if polarization in ("x", "y"):
        return polarization
    if polarization not in ("parallel", "perpendicular"):
        raise ValueError(f"polarización desconocida '{polarization}' (x, y, parallel o perpendicular)")
    axis = (meta or {}).get("bone_axis")
    if axis not in ("x", "y"):
        raise ValueError(f"'{polarization}' necesita meta['bone_axis'] en el plano horizontal ('x' o 'y'); "
                         f"es {axis!r}")
    if polarization == "parallel":
        return axis
    return "y" if axis == "x" else "x"


# ----------------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------------
def center_fields(ex: np.ndarray, ey: np.ndarray, ez: np.ndarray, extended=(False, False, False)) -> np.ndarray:
    """Lleva los fasores de los nodos de Yee a los centros de celda: ``(3, nz, ny, nx)``.

    Cada componente es la media de sus dos nodos vecinos a lo largo de su eje
    (en la primera celda, el nodo de su derecha). ``extended[axis]``: los arrays
    traen una fila más al principio de ese eje (la del nodo ``i - 1/2`` de la
    primera celda de una caja que no empieza en el borde de la malla); esa fila
    se usa para centrar y se descarta.
    """
    out = []
    for comp, f in zip(COMPONENTS, (ex, ey, ez)):
        own = AXES[comp]
        g = f
        for ax in range(3):
            if extended[ax] and ax != own:
                g = np.take(g, np.arange(1, g.shape[ax]), axis=ax)
        n = g.shape[own]
        if extended[own]:
            c = 0.5 * (np.take(g, np.arange(0, n - 1), axis=own) + np.take(g, np.arange(1, n), axis=own))
        else:
            c = g.copy()
            if n > 1:
                c_inner = 0.5 * (np.take(g, np.arange(0, n - 1), axis=own) + np.take(g, np.arange(1, n), axis=own))
                sl = [slice(None)] * 3
                sl[own] = slice(1, n)
                c[tuple(sl)] = c_inner
        out.append(c)
    return np.stack(out)


@dataclass
class EM3DResult:
    """Campo eléctrico complejo a ``f0`` en los centros de celda.

    ``e``: ``(3, nz, ny, nx)`` con Ex, Ey, Ez (convenio ``e^{+jwt}``). Unidades
    según ``normalization``: ``"incident"`` (V/m por V/m de onda incidente en
    ``z_ref``, onda plana TF/SF), ``"sheet"`` (como el 2D: por V/m que emite la
    lámina) o ``"moment"`` (V/m por A·m de momento del dipolo).
    ``incident``: fasor del campo incidente en cada z de la malla (con la misma
    normalización; ``NaN`` fuera de la malla auxiliar), sólo con onda plana.
    ``power_density``: potencia disipada (W/m³) calculada en los nodos de Yee con
    la ``sigma`` de cada nodo (``1/2 sum_c sigma_c |E_c|²``) y llevada a las
    celdas: es la que de verdad disipa el esquema, también con promedio
    sub-celda. ``sar`` usa la ``sigma`` y el ``|E|`` de la celda, como el 2D.
    """

    f0: float
    e: np.ndarray
    grid: Grid3D
    eps_r: np.ndarray
    sigma: np.ndarray
    density: np.ndarray
    labels: np.ndarray
    normalization: str
    dt: float
    steps: int
    elapsed_s: float
    incident: np.ndarray | None = None
    polarization: str | None = None
    power_density: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def magnitude(self) -> np.ndarray:
        """``|E| = sqrt(|Ex|² + |Ey|² + |Ez|²)`` (amplitud de pico del fasor)."""
        return np.sqrt(self.e_squared)

    @property
    def e_squared(self) -> np.ndarray:
        if "e2" not in self._cache:
            self._cache["e2"] = np.sum(self.e.real ** 2 + self.e.imag ** 2, axis=0)
        return self._cache["e2"]

    @property
    def sar(self) -> np.ndarray:
        """Tasa de absorción específica (W/kg) para la normalización del resultado (1 V/m incidente, ...)."""
        return self.sigma * self.e_squared / (2.0 * self.density)

    def absorbed_power(self, mask: np.ndarray | None = None) -> float:
        """Potencia absorbida (W) en las celdas de ``mask`` (todas por defecto).

        Con ``power_density`` (banda estrecha) suma la disipada en los nodos;
        sin ella, ``sum 1/2 sigma |E|² dV`` con los valores de celda.
        """
        pd = self.power_density if self.power_density is not None else 0.5 * self.sigma * self.e_squared
        p = pd * self.grid.dx ** 3
        return float(p.sum() if mask is None else p[mask].sum())

    def tissue_mask(self, exclude: tuple[str, ...] = ("water", "air")) -> np.ndarray:
        """Celdas de tejido (sin el agua de acoplamiento ni el aire)."""
        from ..phantoms.phantom import TISSUE_INDEX

        return ~np.isin(self.labels, [TISSUE_INDEX[k] for k in exclude])

    def averaged_sar(self, mass: float = 10e-3, exclude: tuple[str, ...] = ("water", "air")) -> np.ndarray:
        """SAR promediado en cubos de ``mass`` kg de tejido centrados en cada celda (ver ``averaged_sar``)."""
        return averaged_sar(self.sar, self.density, self.grid.dx, self.tissue_mask(exclude), mass)

    def plane(self, axis: str, value: float) -> np.ndarray:
        """Fasores ``(3, ...)`` del plano de celdas ``axis = value`` más cercano (p. ej. ``plane("y", 0.0)``)."""
        g = self.grid
        ax = AXES[axis]
        coord = {"x": g.x, "y": g.y, "z": g.z}[axis]
        idx = int(np.argmin(np.abs(coord - value)))
        return np.take(self.e, idx, axis=ax + 1)


@dataclass
class BroadbandEM3DResult:
    """Fasores de E a ``freqs`` en la caja ``box`` de celdas: ``e`` tiene forma ``(nf, 3, nz', ny', nx')``.

    Misma normalización que ``EM3DResult`` (el pulso se divide frecuencia a
    frecuencia por el fasor incidente o por el espectro del momento del dipolo).
    """

    freqs: np.ndarray
    e: np.ndarray
    box: tuple[slice, slice, slice]
    grid: Grid3D
    medium: DebyeMedium
    density: np.ndarray
    labels: np.ndarray
    normalization: str
    dt: float
    steps: int
    elapsed_s: float
    incident: np.ndarray | None = None
    polarization: str | None = None

    def at(self, i: int) -> EM3DResult:
        """Resultado a ``freqs[i]`` con el formato de banda estrecha (en la malla de la caja)."""
        f = float(self.freqs[i])
        m = self.medium
        cell = DebyeMedium(m.eps_inf[self.box], m.sigma[self.box], m.delta_eps[(slice(None),) + self.box], m.tau)
        eps = cell.complex_permittivity(f)
        sigma = -eps.imag * 2.0 * np.pi * f * EPS0
        inc = None if self.incident is None else self.incident[i]
        return EM3DResult(f, self.e[i], self.grid, eps.real, sigma, self.density[self.box], self.labels[self.box],
                          self.normalization, self.dt, self.steps, self.elapsed_s, inc, self.polarization)


def _box_sum(a: np.ndarray, n: int) -> np.ndarray:
    """Suma de ``a`` en cubos de ``n`` (impar) celdas centrados en cada celda (ceros fuera de la malla)."""
    h = n // 2
    out = a.astype(np.float64)
    for ax in range(3):
        pad = [(0, 0)] * 3
        pad[ax] = (h + 1, h)
        c = np.cumsum(np.pad(out, pad), axis=ax)
        m = out.shape[ax]
        out = np.take(c, np.arange(n, n + m), axis=ax) - np.take(c, np.arange(0, m), axis=ax)
    return out


def averaged_sar(sar: np.ndarray, density: np.ndarray, dx: float, tissue: np.ndarray,
                 mass: float = 10e-3) -> np.ndarray:
    """SAR promediado en masa (W/kg) en un cubo de ``mass`` kg de tejido centrado en cada celda de tejido.

    El lado del cubo es el de ``mass`` con la densidad media del tejido
    (redondeado a un número impar de celdas) y el promedio es
    ``sum(SAR rho) / sum(rho)`` sobre las celdas de tejido del cubo. Es una
    versión simplificada de IEEE/IEC 62704-1: junto a la superficie no se
    desplaza ni se agranda el cubo, así que allí promedia menos masa. Fuera
    del tejido vale 0.
    """
    tissue = np.asarray(tissue, dtype=bool)
    if not tissue.any():
        return np.zeros(sar.shape)
    rho_mean = float(density[tissue].mean())
    n = max(int(round((mass / rho_mean) ** (1.0 / 3.0) / dx)), 1)
    n += 1 - n % 2
    w = np.where(tissue, density, 0.0)
    num = _box_sum(sar * w, n)
    den = _box_sum(w, n)
    out = np.zeros(sar.shape)
    ok = tissue & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


# ----------------------------------------------------------------------------
# Malla auxiliar 1D de la onda incidente (TF/SF)
# ----------------------------------------------------------------------------
class _IncidentLine:
    """Malla de Yee 1D a lo largo de z con el medio de fondo: campo incidente de la TF/SF.

    ``e[m]`` está en el nodo entero ``k = m + offset`` y ``h[m]`` en ``k + 1/2``.
    La recurrencia es la de una onda plana ``(Ex, Hy)`` en la malla 3D; para
    ``(Ey, Hx)`` es la misma con ``Hx = -h``. CPML en los dos extremos (``pad``
    celdas fuera de la zona útil) y fuente blanda en ``src``.
    """

    def __init__(self, k_lo: int, k_hi: int, pad: int, dx: float, dt: float, f_ref: float, ca: float, cb: float,
                 beta: np.ndarray, kappa: np.ndarray, g: np.ndarray):
        self.offset = k_lo - pad
        n = k_hi - k_lo + 1 + 2 * pad
        self.n = n
        self.e = np.zeros(n)
        self.h = np.zeros(n)
        self.inv_dx = 1.0 / dx
        self.ch = dt / MU0
        self.ca, self.cb = ca, cb
        self.beta, self.kappa, self.g = beta, kappa, g
        self.j = np.zeros((beta.size, n))
        pos = np.arange(n, dtype=float)
        self.a_h, self.b_h = cpml_coefficients(pos + 0.5, n, pad, dx, dt, C0, f_ref)
        self.a_e, self.b_e = cpml_coefficients(pos, n, pad, dx, dt, C0, f_ref)
        self.psi_h = np.zeros(n)
        self.psi_e = np.zeros(n)
        self.src = pad + 1

    def k_index(self, k):
        return np.asarray(k) - self.offset

    def step_h(self) -> None:
        d = np.zeros(self.n)
        d[:-1] = (self.e[1:] - self.e[:-1]) * self.inv_dx
        self.psi_h = self.b_h * self.psi_h + self.a_h * d
        d += self.psi_h
        self.h -= self.ch * d

    def step_e(self, src: float) -> None:
        d = np.zeros(self.n)
        d[1:] = (self.h[1:] - self.h[:-1]) * self.inv_dx
        self.psi_e = self.b_e * self.psi_e + self.a_e * d
        d += self.psi_e
        curl = -d
        old = self.e.copy()
        if self.beta.size:
            curl -= self.g @ self.j
        self.e = self.ca * self.e + self.cb * curl
        self.e[self.src] -= src
        if self.beta.size:
            self.j = self.kappa[:, None] * self.j + self.beta[:, None] * (self.e - old)[None, :]


@dataclass
class _Correction:
    """Una corrección TF/SF: ``field[slab] += coef * incidente(zidx)``."""

    field: str
    slab: tuple
    coef: np.ndarray
    kind: str  # "e" o "h": qué campo incidente se lee
    zidx: np.ndarray | int


# ----------------------------------------------------------------------------
# Malla de Yee 3D común
# ----------------------------------------------------------------------------
class YeeGrid3D:
    """Malla de Yee 3D con CPML, fuentes y paso de tiempo; las subclases ponen el material.

    Las subclases definen ``self.ca`` y ``self.cb`` (diccionarios por
    componente ``"ex"``, ``"ey"``, ``"ez"``) y, con polos de Debye, ``self.jpol``,
    ``self.beta`` (``(K, nz, ny, nx)`` por componente), ``self.kappa`` y ``self.g``.
    """

    def _setup_yee(self, phantom, npml, cfl: float, dt: float | None, dtype, f_ref: float, backend: str) -> None:
        self.backend = _resolve_backend(backend)
        if np.dtype(dtype) not in (np.float32, np.float64):
            raise ValueError("dtype debe ser np.float32 o np.float64")
        self.dtype = np.dtype(dtype)
        self.phantom, self.grid = phantom, phantom.grid
        g = self.grid
        shape = g.shape
        self.npml = (int(npml),) * 3 if np.isscalar(npml) else tuple(int(n) for n in npml)
        for ax, n in enumerate(shape):
            if self.npml[ax] and n <= 2 * self.npml[ax] + 2:
                raise ValueError(f"El eje {AXIS_NAME[ax]} tiene {n} celdas: no caben dos CPML de {self.npml[ax]}")
        # Límite de Courant con las dimensiones que de verdad tienen variación (ny = 1: esquema 2D)
        n_dims = sum(1 for n in shape if n > 1)
        dt_limit = g.dx / (C0 * np.sqrt(max(n_dims, 1)))
        if dt is not None and dt > dt_limit * (1 + 1e-12):
            raise ValueError(f"dt = {dt:.4e} s supera el límite de Courant ({dt_limit:.4e} s)")
        self.dt = float(dt) if dt is not None else cfl * dt_limit
        self.ch = self.dt / MU0
        self.inv_dx = 1.0 / g.dx
        z = lambda: np.zeros(shape, self.dtype)  # noqa: E731
        self.f = {name: z() for name in ("ex", "ey", "ez", "hx", "hy", "hz")}
        kw = dict(dx=g.dx, dt=self.dt, c_max=C0, f_ref=f_ref, dtype=self.dtype)
        self.pml = {}
        for terms, staggered in ((H_TERMS, True), (E_TERMS, False)):
            for comps in terms.values():
                for src, ax, _ in comps:
                    self.pml[f"{src}_{AXIS_NAME[ax]}"] = CPML3D(shape, ax, staggered, self.npml[ax], **kw)
        self.f_ref = f_ref
        self._sources: list[tuple[str, tuple, np.ndarray]] = []
        self._source_kind: str | None = None
        self._aux: _IncidentLine | None = None
        self._corr: list[_Correction] = []
        self.polarization: str | None = None
        self.z_ref_index: int | None = None
        if self.backend == "numba":
            from . import _numba_fdtd3d as nk

            self._nk = nk
            self._pk = {name: cp.pack() for name, cp in self.pml.items()}
            self._s_ch = self.dtype.type(self.ch)
            self._s_inv_dx = self.dtype.type(self.inv_dx)
        else:
            self._buf = (z(), z())

    # ------------------------------------------------------------------ medio
    def _cell_medium(self) -> np.ndarray:
        """Parámetros del medio por celda ``(P, nz, ny, nx)`` (comprueba que la TF/SF está en un fondo uniforme)."""
        raise NotImplementedError

    def _background(self, k: int, j: int, i: int) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """``(ca, cb, beta, kappa, g)`` del medio de la celda ``(k, j, i)`` para la malla 1D incidente."""
        raise NotImplementedError

    def _eta(self, comp: str, sl: tuple) -> np.ndarray:
        raise NotImplementedError

    @property
    def has_poles(self) -> bool:
        return bool(getattr(self, "jpol", None)) and self.kappa.size > 0

    # --------------------------------------------------------------- fuentes
    def _check_free(self) -> None:
        if self._source_kind is not None:
            raise ValueError(f"ya hay una fuente ({self._source_kind}); crea otro solver para cambiarla")

    def set_plane_wave(self, polarization: str = "parallel", z_top: float | None = None,
                       z_bottom: float | None = None, x_range: tuple[float, float] | None = None,
                       y_range: tuple[float, float] | None = None, z_ref: float | None = None,
                       e0: float = 1.0) -> None:
        """Onda plana que incide hacia +z por campo total/campo dispersado.

        - ``polarization``: ``"x"``, ``"y"``, ``"parallel"`` o ``"perpendicular"``
          (al eje del hueso, ver ``polarization_axis``).
        - ``z_top``, ``z_bottom``: caras de la caja de campo total (por defecto
          3 celdas dentro de la CPML de z). ``x_range``, ``y_range``: caras
          laterales; ``None`` = sin caras (la caja ocupa todo ese eje, también
          la CPML, y la onda es lateralmente infinita).
        - ``z_ref``: plano en el que el fasor incidente vale 1 (por defecto ``z_top``).

        Las caras deben quedar en un medio homogéneo (el fondo: agua, aire).
        """
        self._check_free()
        g = self.grid
        pol = polarization_axis(polarization, getattr(self.phantom, "meta", None))
        nz = g.shape[0]
        npz = self.npml[0]
        k0 = npz + 3 if z_top is None else g.index_of(g.x0, g.y0, z_top)[0]
        k1 = nz - npz - 4 if z_bottom is None else g.index_of(g.x0, g.y0, z_bottom)[0]
        lo, hi = {0: k0, 1: None, 2: None}, {0: k1, 1: None, 2: None}
        for ax, rng in ((2, x_range), (1, y_range)):
            if rng is not None:
                coord = g.x if ax == 2 else g.y
                lo[ax] = int(np.argmin(np.abs(coord - rng[0])))
                hi[ax] = int(np.argmin(np.abs(coord - rng[1])))
        for ax in range(3):
            if lo[ax] is None:
                continue
            n, p = g.shape[ax], self.npml[ax]
            if not (max(p, 2) + 1 <= lo[ax] < hi[ax] <= n - max(p, 2) - 2):
                raise ValueError(f"la caja TF/SF en {AXIS_NAME[ax]} ({lo[ax]}-{hi[ax]}) debe quedar fuera de la "
                                 f"CPML ({p} celdas) y dentro de la malla ({n} celdas)")
        self._check_uniform_shell(lo, hi)
        ca, cb, beta, kappa, gk = self._background(k0, g.ny // 2 if lo[1] is None else lo[1],
                                                   g.nx // 2 if lo[2] is None else lo[2])
        pad = max(self.npml[0], 10)
        self._aux = _IncidentLine(k0 - 3, k1 + 3, pad, g.dx, self.dt, self.f_ref, ca, cb, beta, kappa, gk)
        eps_hf = self._background_eps_inf(k0)
        self._aux_scale = cb * 2.0 * e0 / (ETA0 / np.sqrt(eps_hf) * g.dx)
        self.polarization = pol
        self.z_ref_index = k0 if z_ref is None else g.index_of(g.x0, g.y0, z_ref)[0]
        if not (k0 - 2 <= self.z_ref_index <= k1 + 2):
            raise ValueError("z_ref debe estar entre las caras de la caja TF/SF")
        self._source_kind = "plane_wave"
        self._build_corrections(lo, hi, pol)

    def _background_eps_inf(self, k0: int) -> float:
        raise NotImplementedError

    def _check_uniform_shell(self, lo: dict, hi: dict) -> None:
        """Error si las celdas a ±2 de las caras de la caja TF/SF no son de un mismo medio."""
        g = self.grid
        med = self._cell_medium()
        outer, inner = [], []
        for ax in range(3):
            n = g.shape[ax]
            if lo[ax] is None:
                outer.append(slice(0, n))
                inner.append(slice(0, n))
            else:
                outer.append(slice(max(lo[ax] - 2, 0), min(hi[ax] + 3, n)))
                inner.append(slice(lo[ax] + 2, hi[ax] - 1))
        shell = np.zeros(g.shape, bool)
        shell[tuple(outer)] = True
        shell[tuple(inner)] = False
        vals = med[:, shell]
        ref = vals[:, :1]
        scale = max(float(np.abs(ref).max()), 1.0)
        if vals.size and float(np.abs(vals - ref).max()) > 1e-9 * scale:
            raise ValueError("Las caras de la caja TF/SF cruzan medios distintos: colócalas en el fondo "
                             "homogéneo (agua o aire) o usa x_range/y_range = None.")

    def _build_corrections(self, lo: dict, hi: dict, pol: str) -> None:
        # Componentes incidentes no nulas y su signo respecto a la (e, h) de la malla 1D
        inc = {"ex": ("e", 1.0), "hy": ("h", 1.0)} if pol == "x" else {"ey": ("e", 1.0), "hx": ("h", -1.0)}
        dx = self.grid.dx
        shape = self.grid.shape
        corr = []
        for target, terms in list(H_TERMS.items()) + list(E_TERMS.items()):
            is_e = target[0] == "e"
            own = AXES[target[1]]
            for src, a, s in terms:
                if src not in inc or lo[a] is None:
                    continue
                kind, sign = inc[src]
                ranges = []
                for b in range(3):
                    if b == a:
                        ranges.append(None)
                    elif lo[b] is None:
                        ranges.append(slice(0, shape[b]))
                    elif (b == own) == (not is_e):
                        # Nodo entero a lo largo de b: H en su propio eje, E en los ajenos
                        ranges.append(slice(lo[b], hi[b] + 1))
                    else:
                        ranges.append(slice(lo[b], hi[b]))
                for face, f_sign in (("lo", -1.0), ("hi", 1.0)):
                    if is_e:
                        idx = lo[a] if face == "lo" else hi[a]
                        z_src = (lo[0] - 1 if face == "lo" else hi[0]) if a == 0 else None
                    else:
                        idx = lo[a] - 1 if face == "lo" else hi[a]
                        z_src = (lo[0] if face == "lo" else hi[0]) if a == 0 else None
                    slab = tuple(slice(idx, idx + 1) if b == a else ranges[b] for b in range(3))
                    base = f_sign * s * sign / dx
                    coef = base * (self.cb[target][slab].astype(np.float64) if is_e else self.ch)
                    if z_src is None:
                        zr = slab[0]
                        zidx = np.arange(zr.start, zr.stop)[:, None, None]
                    else:
                        zidx = z_src
                    corr.append(_Correction(target, slab, np.asarray(coef), kind, zidx))
        self._corr = corr

    def _apply_corrections(self, kind_target: str) -> None:
        aux = self._aux
        for c in self._corr:
            if c.field[0] != kind_target:
                continue
            arr = aux.e if c.kind == "e" else aux.h
            delta = c.coef * arr[aux.k_index(c.zidx)]
            if kind_target == "h":
                self.f[c.field][c.slab] += delta
            else:
                self._add_e(c.field, c.slab, delta)

    def _add_e(self, comp: str, sl: tuple, delta) -> None:
        """``E[sl] += delta`` después del barrido de E; con polos, ``J_k`` es lineal en ``E^{n+1}`` y se corrige."""
        self.f[comp][sl] += delta
        if self.has_poles:
            self.jpol[comp][(slice(None),) + sl] += self.beta[comp][(slice(None),) + sl] * delta

    def set_current_sheet(self, z: float | None = None, polarization: str = "y", e0: float = 1.0) -> None:
        """Lámina de corriente blanda que emite una onda plana de ``e0`` V/m hacia +z y -z (la fuente del 2D).

        Misma normalización que ``MaxwellTMSolver.set_plane_wave_source`` (con la
        impedancia real de cada celda): con E a lo largo de y en un fantoma
        extruido da el mismo campo que el solver 2D. Sólo banda estrecha.
        """
        self._check_free()
        g = self.grid
        pol = polarization_axis(polarization, getattr(self.phantom, "meta", None))
        k = g.index_of(g.x0, g.y0, z if z is not None else g.z0 + (self.npml[0] + 2) * g.dx)[0]
        comp = "e" + pol
        sl = (slice(k, k + 1), slice(None), slice(None))
        eta = self._eta(comp, sl)
        scale = self.cb[comp][sl].astype(np.float64) * 2.0 * e0 / (eta * g.dx)
        self._sources = [(comp, sl, scale)]
        self.polarization = pol
        self._source_kind = "sheet"

    def set_dipole(self, x: float, y: float, z: float, axis: str = "z", moment: float = 1.0) -> None:
        """Dipolo eléctrico puntual blando de momento ``moment`` (A·m) orientado según ``axis``.

        Es una densidad de corriente ``J = I l / dx³`` en el nodo de E más
        cercano (``E -= cb J``). El fasor del resultado va por A·m de momento.
        """
        self._check_free()
        g = self.grid
        own = AXES[axis]
        k, j, i = g.index_of(x, y, z)
        idx = [k, j, i]
        # El nodo de E está a media celda a lo largo de su eje: el de la izquierda de la celda más cercana
        coord = (z, y, x)[own]
        origin = (g.z0, g.y0, g.x0)[own]
        idx[own] = int(np.clip(np.floor((coord - origin) / g.dx), 0, g.shape[own] - 1))
        comp = "e" + axis
        sl = tuple(slice(n, n + 1) for n in idx)
        scale = self.cb[comp][sl].astype(np.float64) * moment / g.dx ** 3
        self._sources = [(comp, sl, scale)]
        self.polarization = axis
        self._source_kind = "dipole"

    # ------------------------------------------------------------------ paso
    def _deriv(self, src: str, axis: int, forward: bool, out: np.ndarray) -> np.ndarray:
        f = self.f[src]
        n = f.shape[axis]
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        o = [slice(None)] * 3
        e = [slice(None)] * 3
        a[axis], b[axis] = slice(1, n), slice(0, n - 1)
        if forward:
            o[axis], e[axis] = slice(0, n - 1), slice(n - 1, n)
        else:
            o[axis], e[axis] = slice(1, n), slice(0, 1)
        dst = out[tuple(o)]
        np.subtract(f[tuple(a)], f[tuple(b)], out=dst)
        dst *= self.inv_dx
        out[tuple(e)] = 0.0
        self.pml[f"{src}_{AXIS_NAME[axis]}"].apply(out)
        return out

    def _step_h(self) -> None:
        if self.backend == "numba":
            f = self.f
            self._nk.step_h(f["ex"], f["ey"], f["ez"], f["hx"], f["hy"], f["hz"], self._s_ch, self._s_inv_dx,
                            *(self._pk[n] for n in H_PML))
            return
        b1, b2 = self._buf
        for h, ((s1, a1, _), (s2, a2, _)) in H_TERMS.items():
            d1 = self._deriv(s1, a1, True, b1)
            d2 = self._deriv(s2, a2, True, b2)
            np.subtract(d1, d2, out=d1)
            d1 *= self.ch
            self.f[h] += d1

    def _step_e(self) -> None:
        f = self.f
        if self.backend == "numba":
            args = (f["ex"], f["ey"], f["ez"], f["hx"], f["hy"], f["hz"], self.ca["ex"], self.cb["ex"],
                    self.ca["ey"], self.cb["ey"], self.ca["ez"], self.cb["ez"], self._s_inv_dx,
                    *(self._pk[n] for n in E_PML))
            if self.has_poles:
                self._nk.step_e_debye(*args, self.jpol["ex"], self.jpol["ey"], self.jpol["ez"], self.beta["ex"],
                                      self.beta["ey"], self.beta["ez"], self.kappa, self.g)
            else:
                self._nk.step_e(*args)
            return
        b1, b2 = self._buf
        poles = self.has_poles
        for e, ((s1, a1, _), (s2, a2, _)) in E_TERMS.items():
            d1 = self._deriv(s1, a1, False, b1)
            d2 = self._deriv(s2, a2, False, b2)
            np.subtract(d1, d2, out=d1)
            if poles:
                d1 -= np.tensordot(self.g, self.jpol[e], axes=1)
                np.copyto(b2, f[e])
            f[e] *= self.ca[e]
            np.multiply(self.cb[e], d1, out=d1)
            f[e] += d1
            if poles:
                jp = self.jpol[e]
                jp *= self.kappa[:, None, None, None]
                np.subtract(f[e], b2, out=b2)
                jp += self.beta[e] * b2

    def step(self, drive: float) -> None:
        """Un paso completo: H (con TF/SF), malla 1D, E (con TF/SF y fuente ``drive``)."""
        self._step_h()
        if self._aux is not None:
            self._apply_corrections("h")
            self._aux.step_h()
        self._step_e()
        if self._aux is not None:
            self._apply_corrections("e")
            self._aux.step_e(self._aux_scale * drive)
        for comp, sl, scale in self._sources:
            self._add_e(comp, sl, -scale * drive)

    # ------------------------------------------------------------ utilidades
    def energy(self) -> float:
        """Energía electromagnética ``sum (eps |E|² + mu0 |H|²)/2 dV`` (J) con la ``eps'`` de alta frecuencia."""
        dv = self.grid.dx ** 3
        we = sum(float(np.sum(self._eps_node(c) * self.f[c].astype(np.float64) ** 2)) for c in ("ex", "ey", "ez"))
        wh = sum(float(np.sum(self.f[c].astype(np.float64) ** 2)) for c in ("hx", "hy", "hz"))
        return 0.5 * dv * (EPS0 * we + MU0 * wh)

    def _eps_node(self, comp: str) -> np.ndarray:
        raise NotImplementedError

    def memory_bytes(self) -> dict[str, int]:
        """Memoria del solver por bloques (bytes): campos, material, CPML, polos y fuente."""
        def size(obj) -> int:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            if isinstance(obj, dict):
                return sum(size(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return sum(size(v) for v in obj)
            return 0

        out = {
            "fields": size(self.f),
            "material": size(self.ca) + size(self.cb),
            "cpml": sum(cp.psi_l.nbytes + cp.psi_r.nbytes for cp in self.pml.values()),
            "poles": size(getattr(self, "jpol", {})) + size(getattr(self, "beta", {})),
            "numpy_buffers": size(getattr(self, "_buf", ())),
            "tfsf": sum(size(c.coef) for c in self._corr),
        }
        out["total"] = sum(out.values())
        return out

    def _norm_name(self) -> str:
        return {"plane_wave": "incident", "sheet": "sheet", "dipole": "moment"}[self._source_kind]

    def _normalization(self, aux_phasor, moment_phasor):
        """``(divisor por frecuencia, fasor incidente en cada z o None)``."""
        nf = moment_phasor.size
        if self._source_kind == "plane_wave":
            ref = aux_phasor[:, self._aux.k_index(self.z_ref_index)]
            inc = np.full((nf, self.grid.nz), np.nan + 0j)
            ks = np.arange(self.grid.nz)
            m = self._aux.k_index(ks)
            ok = (m >= 0) & (m < self._aux.n)
            inc[:, ok] = aux_phasor[:, m[ok]] / ref[:, None]
            return ref, inc
        if self._source_kind == "dipole":
            return moment_phasor, None
        return np.ones(nf, complex), None

    def _accumulators(self, box, nf: int):
        """Acumuladores de la DFT en la caja ampliada una celda por abajo (para centrar)."""
        ext = tuple(slice(max(s.start - 1, 0), s.stop) for s in box)
        extended = tuple(s.start > 0 for s in box)
        shape = tuple(s.stop - s.start for s in ext)
        return ext, extended, {c: np.zeros((nf,) + shape, complex) for c in ("ex", "ey", "ez")}

    def _accumulate(self, acc: dict, ext: tuple, ph: np.ndarray) -> None:
        for c, a in acc.items():
            view = self.f[c][ext]
            if self.backend == "numba":
                self._nk.accumulate_dft(a, view, ph.real.copy(), ph.imag.copy())
            else:
                for m in range(ph.size):
                    a[m] += view * ph[m]

    def _normalize_box(self, box):
        g = self.grid
        if box is None:
            return tuple(slice(0, n) for n in g.shape)
        out = []
        for s, n in zip(box, g.shape):
            start, stop, step = s.indices(n)
            if step != 1:
                raise ValueError("la caja del monitor debe tener paso 1")
            out.append(slice(start, stop))
        return tuple(out)

    def _box_grid(self, box) -> Grid3D:
        g = self.grid
        kz, jy, ix = (s.start for s in box)
        return Grid3D(box[2].stop - ix, box[1].stop - jy, box[0].stop - kz, g.dx, g.x0 + ix * g.dx,
                      g.y0 + jy * g.dx, g.z0 + kz * g.dx)


# ----------------------------------------------------------------------------
# Banda estrecha
# ----------------------------------------------------------------------------
class Maxwell3DSolver(YeeGrid3D):
    """FDTD 3D de banda estrecha: ``eps'`` y ``sigma`` de cada celda a ``f0``.

    - ``npml``: grosor de la CPML (entero o ``(nz, ny, nx)``; 0 = sin capa en ese eje).
    - ``dt``: paso impuesto (p. ej. el de una simulación 2D); error si supera el
      límite de Courant de las dimensiones con más de una celda.
    - ``eps_map``: permitividad relativa compleja ``eps' - j eps''`` por celda
      que sustituye a la del fantoma (el fantoma sigue dando la densidad).
    - ``permittivity(tejido, f)``: modelo de los tejidos (Gabriel por defecto;
      p. ej. el ajuste de Debye para compararlo con la banda ancha).
    - ``node_eps``: ``(3, nz, ny, nx)`` permitividades complejas en los nodos de
      Ex, Ey, Ez que sustituyen a la media de las dos celdas (p. ej. el promedio
      sub-celda de ``subcell_node_permittivity``).
    """

    def __init__(self, phantom, f0: float, npml=12, cfl: float = 0.95, dtype=np.float32,
                 eps_map: np.ndarray | None = None, permittivity=None, backend: str = "auto",
                 dt: float | None = None, node_eps: np.ndarray | None = None):
        self.f0 = float(f0)
        g = phantom.grid
        get = permittivity or (lambda k, f: complex(dielectric.complex_permittivity(k, f)))
        if eps_map is None:
            eps_map = phantom.property_map(lambda k: complex(get(k, f0)), dtype=complex)
        else:
            eps_map = np.asarray(eps_map, dtype=complex)
            if eps_map.shape != g.shape:
                raise ValueError(f"eps_map tiene forma {eps_map.shape}, la malla {g.shape}")
        self.eps_cell = eps_map
        n_max = float(np.abs(np.sqrt(eps_map)).max())
        self.points_per_wavelength = C0 / (f0 * n_max * g.dx)
        if self.points_per_wavelength < 10:
            warnings.warn(f"Sólo {self.points_per_wavelength:.1f} puntos por longitud de onda en el tejido: "
                          "reduce dx.", stacklevel=2)
        self._setup_yee(phantom, npml, cfl, dt, dtype, f_ref=f0, backend=backend)
        w = 2.0 * np.pi * f0
        self.ca, self.cb, self._eps_r_node, self._sigma_node = {}, {}, {}, {}
        if node_eps is not None and np.shape(node_eps) != (3,) + g.shape:
            raise ValueError(f"node_eps debe tener forma {(3,) + g.shape}")
        for c, comp in enumerate(COMPONENTS):
            node = node_average(eps_map, AXES[comp]) if node_eps is None else np.asarray(node_eps[c], complex)
            eps = node.real * EPS0
            sigma = -node.imag * w * EPS0
            k = sigma * self.dt / (2 * eps)
            self.ca["e" + comp] = ((1 - k) / (1 + k)).astype(self.dtype)
            self.cb["e" + comp] = (self.dt / eps / (1 + k)).astype(self.dtype)
            self._eps_r_node["e" + comp] = node.real.astype(np.float32)
            self._sigma_node["e" + comp] = sigma.astype(np.float32)

    def _cell_medium(self) -> np.ndarray:
        return np.stack([self.eps_cell.real, self.eps_cell.imag])

    def _background(self, k, j, i):
        eps_c = self.eps_cell[k, j, i]
        eps = eps_c.real * EPS0
        sigma = -eps_c.imag * 2 * np.pi * self.f0 * EPS0
        kk = sigma * self.dt / (2 * eps)
        # Mismo redondeo que los coeficientes de la malla 3D
        ca = float(self.dtype.type((1 - kk) / (1 + kk)))
        cb = float(self.dtype.type(self.dt / eps / (1 + kk)))
        e = np.zeros(0)
        return ca, cb, e, e, e

    def _background_eps_inf(self, k0: int) -> float:
        g = self.grid
        return float(self.eps_cell[k0, g.ny // 2, g.nx // 2].real)

    def _eta(self, comp, sl):
        return ETA0 / np.sqrt(self._eps_r_node[comp][sl].astype(np.float64))

    def _eps_node(self, comp):
        return self._eps_r_node[comp]

    def run(self, n_periods_steady: float = 4.0, settle_time: float | None = None, exact_dft: bool = True,
            progress: bool = False) -> EM3DResult:
        """Onda continua hasta el régimen estacionario y fasor a ``f0`` por DFT corriente.

        Mismo protocolo que ``MaxwellTMSolver.run``: arranque suave de 3
        periodos, ``settle_time`` (por defecto 3 periodos más tres tránsitos de
        ``Lz + max(Lx, Ly)`` a la velocidad más lenta) y ``n_periods_steady``
        periodos de DFT con ~24 muestras por periodo.

        ``exact_dft``: la ventana de la DFT no abarca un número entero de
        periodos (el periodo no es múltiplo de ``dt``), así que ``2/N sum x
        e^{-jwt}`` arrastra una fracción de la frecuencia negativa (``P*``); con
        ``Q = sum e^{-2jwt}`` el fasor exacto de una sinusoide es ``2 (N S - Q
        S*) / (N² - |Q|²)``. En vacío deja la amplitud de la onda plana
        uniforme a 1e-6 en vez de un rizado del ~2 %. ``False`` reproduce el
        estimador del 2D (``MaxwellTMSolver``).
        """
        if self._source_kind is None:
            self.set_plane_wave()
        g = self.grid
        period = 1.0 / self.f0
        steps_per_period = period / self.dt
        if settle_time is None:
            n_max = np.sqrt(self.eps_cell.real.max())
            lz, lx, ly = g.nz * g.dx, g.nx * g.dx, g.ny * g.dx
            settle_time = 3 * period + 3.0 * (lz + max(lx, ly)) * n_max / C0
        nt_settle = int(np.ceil(settle_time / self.dt))
        nt_dft = int(np.ceil(n_periods_steady * steps_per_period))
        nt = nt_settle + nt_dft
        stride = max(int(steps_per_period // 24), 1)
        t = (np.arange(nt) + 1) * self.dt
        # La lámina toma el seno en t_{n+1} como el 2D; el dipolo y la TF/SF, la corriente centrada en n + 1/2
        t_drive = t if self._source_kind == "sheet" else t - 0.5 * self.dt
        drive = ramped_sine(t_drive, self.f0, ramp_cycles=3)
        box = self._normalize_box(None)
        ext, extended, acc = self._accumulators(box, 1)
        aux_acc = np.zeros(self._aux.n, complex) if self._aux is not None else None
        n_acc, q_acc = 0, 0j
        t0 = time.time()
        for n in range(nt):
            self.step(drive[n])
            if n >= nt_settle and (n - nt_settle) % stride == 0:
                ph = np.array([np.exp(-2j * np.pi * self.f0 * t[n])])
                self._accumulate(acc, ext, ph)
                if aux_acc is not None:
                    aux_acc += self._aux.e * ph[0]
                n_acc += 1
                q_acc += ph[0] ** 2
            if progress and n % max(nt // 10, 1) == 0:
                print(f"  paso {n}/{nt}", flush=True)
        elapsed = time.time() - t0

        def phasor(sums):
            if not exact_dft:
                return sums * (2.0 / max(n_acc, 1))
            return 2.0 * (n_acc * sums - q_acc * np.conj(sums)) / (n_acc ** 2 - abs(q_acc) ** 2)

        norm, incident = self._normalization(None if aux_acc is None else phasor(aux_acc)[None, :],
                                             np.array([-1j]))
        nodes = {c: phasor(acc[c][0]) / norm[0] for c in ("ex", "ey", "ez")}
        del acc
        e = center_fields(*(nodes[c] for c in ("ex", "ey", "ez")), extended=extended)
        pd = center_fields(*(0.5 * self._sigma_node[c] * np.abs(nodes[c]) ** 2 for c in ("ex", "ey", "ez")),
                           extended=extended).sum(axis=0)
        del nodes
        w = 2.0 * np.pi * self.f0
        sigma = -self.eps_cell.imag * w * EPS0
        return EM3DResult(self.f0, e, g, self.eps_cell.real.copy(), sigma, _density(self.phantom),
                          self.phantom.labels, self._norm_name(), self.dt, nt, elapsed,
                          None if incident is None else incident[0], self.polarization, pd)

# ----------------------------------------------------------------------------
# Banda ancha (Debye multipolo)
# ----------------------------------------------------------------------------
class BroadbandMaxwell3DSolver(YeeGrid3D):
    """FDTD 3D con tejidos de Debye multipolo (ADE) y pulso de banda ancha.

    Mismos argumentos que ``BroadbandMaxwellTMSolver`` (``band``, ``n_poles``,
    ``model``, ``medium``) más los de ``Maxwell3DSolver`` (``npml``, ``dt``,
    ``dtype``, ``backend``). ``medium`` son mapas de Debye por celda ``(nz, ny,
    nx)``; en los nodos se promedian como ``eps*``. La CPML usa ``f_ref =
    f_min``. Fuentes: onda plana TF/SF o dipolo (la lámina de corriente es sólo
    de banda estrecha).
    """

    def __init__(self, phantom, band: tuple[float, float], n_poles: int | None = None, npml=12,
                 cfl: float = 0.95, dtype=np.float32, model: TissueDebyeModel | None = None,
                 medium: DebyeMedium | None = None, backend: str = "auto", dt: float | None = None):
        self.band = (float(band[0]), float(band[1]))
        if medium is None:
            model = model or tissue_debye_model(self.band, n_poles)
            medium = debye_maps_3d(phantom, model)
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
        self._setup_yee(phantom, npml, cfl, dt, dtype, f_ref=self.band[0], backend=backend)
        dt = self.dt
        active = [k for k in range(medium.n_poles) if np.any(medium.delta_eps[k] > 0)]
        tau = np.asarray(medium.tau, dtype=float)[active]
        self._active = active
        self.kappa = ((2.0 * tau - dt) / (2.0 * tau + dt)).astype(self.dtype)
        self.g = (0.5 * (1.0 + self.kappa.astype(np.float64))).astype(self.dtype)
        self.ca, self.cb, self.beta, self.jpol, self._eps_inf_node = {}, {}, {}, {}, {}
        for comp in COMPONENTS:
            ax = AXES[comp]
            eps_inf = node_average(medium.eps_inf, ax)
            sigma = node_average(medium.sigma, ax)
            d_eps = node_average(np.asarray(medium.delta_eps)[active], ax) if active else np.zeros((0,) + g.shape)
            beta = 2.0 * EPS0 * d_eps / (2.0 * tau[:, None, None, None] + dt)
            eps_dt = EPS0 * eps_inf / dt
            half_beta = 0.5 * beta.sum(axis=0) if active else 0.0
            denom = eps_dt + 0.5 * sigma + half_beta
            name = "e" + comp
            self.ca[name] = ((eps_dt - 0.5 * sigma + half_beta) / denom).astype(self.dtype)
            self.cb[name] = (1.0 / denom).astype(self.dtype)
            self.beta[name] = beta.astype(self.dtype)
            self.jpol[name] = np.zeros((len(active),) + g.shape, self.dtype)
            self._eps_inf_node[name] = eps_inf.astype(np.float32)

    def _cell_params(self, sl) -> np.ndarray:
        m = self.medium
        return np.concatenate([np.asarray(m.eps_inf)[sl][None], np.asarray(m.sigma)[sl][None],
                               np.asarray(m.delta_eps)[(slice(None),) + sl]])

    def _cell_medium(self) -> np.ndarray:
        return self._cell_params((slice(None),) * 3)

    def _background(self, k, j, i):
        p = self._cell_params((k, j, i)).astype(np.float64)
        dt = self.dt
        tau = np.asarray(self.medium.tau, dtype=float)[self._active]
        d_eps = p[2:][self._active]
        beta = 2.0 * EPS0 * d_eps / (2.0 * tau + dt)
        eps_dt = EPS0 * p[0] / dt
        denom = eps_dt + 0.5 * p[1] + 0.5 * beta.sum()
        ca = float(self.dtype.type((eps_dt - 0.5 * p[1] + 0.5 * beta.sum()) / denom))
        cb = float(self.dtype.type(1.0 / denom))
        return (ca, cb, beta.astype(self.dtype).astype(np.float64), self.kappa.astype(np.float64),
                self.g.astype(np.float64))

    def _background_eps_inf(self, k0: int) -> float:
        g = self.grid
        return float(np.asarray(self.medium.eps_inf)[k0, g.ny // 2, g.nx // 2])

    def set_current_sheet(self, *args, **kwargs):
        """No disponible: la lámina del 2D se normaliza con la impedancia a una sola frecuencia."""
        raise ValueError("la lámina de corriente sólo existe en banda estrecha: usa set_plane_wave o set_dipole")

    def _eps_node(self, comp):
        return self._eps_inf_node[comp]

    def run(self, freqs: np.ndarray | None = None, box: tuple[slice, slice, slice] | None = None,
            t_max: float | None = None, decay: float = 1e-6, edge_db: float = -20.0, check_every: int = 64,
            progress: bool = False) -> BroadbandEM3DResult:
        """Emite el pulso y acumula la DFT de E en ``box`` hasta que el campo decae.

        - ``freqs``: frecuencias de salida (por defecto 11 log-espaciadas en la banda).
        - ``box``: ``(slice_z, slice_y, slice_x)`` de celdas en las que se guarda
          el fasor (``None`` = toda la malla; ojo con la memoria: 3 fasores
          complejos por frecuencia y celda).
        - Parada como en 2D: terminado el pulso, cuando el máximo de ``|E|`` en la
          caja durante ``check_every`` muestras baja de ``decay`` veces el máximo
          alcanzado, o en ``t_max``.
        """
        if self._source_kind is None:
            self.set_plane_wave()
        g = self.grid
        f_lo, f_hi = self.band
        freqs = np.geomspace(f_lo, f_hi, 11) if freqs is None else np.atleast_1d(np.asarray(freqs, dtype=float))
        _, _, t0p = wideband_pulse_params(f_lo, f_hi, edge_db)
        t_pulse = 2.0 * t0p
        if t_max is None:
            n_max = float(np.abs(np.sqrt(self.medium.complex_permittivity(f_lo))).max())
            lz, lx, ly = g.nz * g.dx, g.nx * g.dx, g.ny * g.dx
            t_max = t_pulse + 10.0 * (lz + max(lx, ly)) * n_max / C0 + 5.0 / f_lo
        nt = int(np.ceil(t_max / self.dt))
        stride = max(int(1.0 / (6.0 * 1.5 * max(f_hi, freqs.max()) * self.dt)), 1)
        w = 2.0 * np.pi * freqs
        box = self._normalize_box(box)
        ext, extended, acc = self._accumulators(box, freqs.size)
        aux_acc = np.zeros((freqs.size, self._aux.n), complex) if self._aux is not None else None
        peak, block_max, n_samples = 0.0, 0.0, 0
        steps = nt
        t_wall = time.time()
        for n in range(nt):
            t_src = (n + 0.5) * self.dt
            drive = float(wideband_pulse(t_src, f_lo, f_hi, edge_db)) if t_src <= t_pulse else 0.0
            self.step(drive)
            if n % stride == 0:
                ph = np.exp(-1j * w * (n + 1) * self.dt) * (stride * self.dt)
                self._accumulate(acc, ext, ph)
                if aux_acc is not None:
                    aux_acc += ph[:, None] * self._aux.e[None, :]
                n_samples += 1
                if n_samples % 8 == 0 or n_samples % check_every == 0:
                    block_max = max(block_max, self._box_max(ext))
                if n_samples % check_every == 0:
                    peak = max(peak, block_max)
                    if progress:
                        print(f"  paso {n + 1}/{nt}  max|E| = {block_max:.3e}", flush=True)
                    if t_src > t_pulse and block_max <= decay * peak:
                        steps = n + 1
                        break
                    block_max = 0.0
        elapsed = time.time() - t_wall
        n_src = int(np.ceil(t_pulse / self.dt)) + 1
        t_s = (np.arange(n_src) + 0.5) * self.dt
        spectrum = np.exp(-1j * np.outer(w, t_s)) @ wideband_pulse(t_s, f_lo, f_hi, edge_db) * self.dt
        norm, incident = self._normalization(aux_acc, spectrum)
        e = np.stack([center_fields(*(acc[c][m] / norm[m] for c in ("ex", "ey", "ez")), extended=extended)
                      for m in range(freqs.size)])
        return BroadbandEM3DResult(freqs, e, box, self._box_grid(box), self.medium, _density(self.phantom),
                                   self.phantom.labels, self._norm_name(), self.dt, steps, elapsed,
                                   incident, self.polarization)

    def _box_max(self, ext) -> float:
        if self.backend == "numba":
            return max(float(self._nk.max_abs(self.f[c][ext])) for c in ("ex", "ey", "ez"))
        return max(float(np.abs(self.f[c][ext]).max()) for c in ("ex", "ey", "ez"))


def simulate_em3d_plane_wave(phantom, f0: float, polarization: str = "parallel", npml=12,
                             backend: str = "auto", progress: bool = False) -> EM3DResult:
    """Onda plana de 1 V/m (en la cara superior de la caja TF/SF) que incide desde arriba sobre el fantoma."""
    solver = Maxwell3DSolver(phantom, f0, npml=npml, backend=backend)
    solver.set_plane_wave(polarization)
    return solver.run(progress=progress)
