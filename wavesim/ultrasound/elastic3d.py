"""Solver FDTD elástico 3D (ondas P y S) en CPU para dominios pequeños.

Es el esquema de ``elastic_fdtd.py`` llevado a 3D (Virieux 1986; Graves,
BSSA 86:1091, 1996): velocidad-tensión en malla escalonada, derivadas de
4.º orden en espacio, leapfrog de 2.º orden en tiempo, CPML en las seis caras
y atenuación GSLS con un peso por mecanismo (``attenuation.py``):

    rho dv_i/dt = d(s_ij)/dx_j
    ds_ij/dt    = C_ijkl d(v_k)/dx_l      (+ memorias viscoelásticas)

Para qué sirve: en un corte 2D la tibia es una placa (dos, en el corte
longitudinal) y no un tubo, y una fuente puntual 2D es una línea infinita en
3D. La difracción, el decaimiento geométrico y las ondas guiadas de un tubo
sólo salen en 3D (Bossy, Talmant & Laugier, JASA 115:2314, 2004).

- Fluidos (tejidos blandos, agua) con ``mu = 0``; ``mu`` en los nodos de
  cizalla es la media armónica de las cuatro celdas que los rodean (nula si
  alguna es fluida), como en 2D.
- Junto a una interfaz fluido-sólido, la deformación de cizalla del sólido no
  lee la velocidad tangencial del fluido (que desliza) y el fluido no recibe
  la cizalla del sólido (``_slip_terms``, el mismo principio que la corrección
  del solver 2D): sin ella, el estencil de 4.º orden comete un error O(1) en la
  interfaz que no baja al refinar.
- Cortical transversalmente isótropa opcional con el eje del hueso a lo largo
  de x (``ElasticMedium3D.stiffness``, de ``materials.acoustic.CORTICAL_STIFFNESS``).
- Fuentes de presión o de fuerza y receptores por celdas, como en 2D.
- Backends: "numpy" (referencia, lento y con más memoria temporal), "numba"
  (``_numba_kernels3d``: dos barridos fusionados por paso, paralelos por planos
  z) o "auto", con el mismo criterio que ``ElasticSolver``. Sólo float32 con Numba.

Disposición (arrays ``(nz, ny, nx)``, índices ``[k, j, i] = [z, y, x]``):

    sxx, syy, szz : (i, j, k)
    vx : (i+1/2, j, k)        vy : (i, j+1/2, k)        vz : (i, j, k+1/2)
    sxy: (i+1/2, j+1/2, k)    sxz: (i+1/2, j, k+1/2)    syz: (i, j+1/2, k+1/2)

El plano ``y = cte`` contiene ``sxx, szz, sxz, vx, vz`` en las posiciones del
esquema 2D: con campos uniformes en y (fuente uniforme a lo largo de todo y en un
medio extruido) el solver reproduce ``ElasticSolver`` salvo redondeo.

Memoria (float32, medido con ``memory_bytes``): ~120 bytes por celda con un
mecanismo de relajación (9 campos, 3 densidades inversas, 5 módulos, 6
memorias y 5 coeficientes de memoria), más el medio (~20 B/celda) y la CPML
(depende de la forma del dominio). Ver ``docs/validacion/elastico3d.md``.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from ..core.pml import cpml_coefficients
from ..materials.acoustic import ACOUSTIC, CORTICAL_STIFFNESS
from ..phantoms.limb3d import Phantom3D
from ..phantoms.phantom import TISSUE_INDEX
from . import attenuation as att
from .elastic_fdtd import C1, C2, ElasticSolver
from .medium import MIN_SHEAR_PPW, christoffel_speeds

#: Derivadas de la etapa de velocidades: nombre -> (campo, eje, hacia delante, componente que actualiza).
#: Ejes de los arrays: 0 = z, 1 = y, 2 = x.
VELOCITY_DERIVS = {
    "sxx_x": ("sxx", 2, True, "vx"), "sxy_y": ("sxy", 1, False, "vx"), "sxz_z": ("sxz", 0, False, "vx"),
    "sxy_x": ("sxy", 2, False, "vy"), "syy_y": ("syy", 1, True, "vy"), "syz_z": ("syz", 0, False, "vy"),
    "sxz_x": ("sxz", 2, False, "vz"), "syz_y": ("syz", 1, False, "vz"), "szz_z": ("szz", 0, True, "vz"),
}
#: Derivadas de la etapa de tensiones: nombre -> (campo, eje, hacia delante).
STRESS_DERIVS = {
    "exx": ("vx", 2, False), "eyy": ("vy", 1, False), "ezz": ("vz", 0, False),
    "exy_a": ("vx", 1, True), "exy_b": ("vy", 2, True),
    "exz_a": ("vx", 0, True), "exz_b": ("vz", 2, True),
    "eyz_a": ("vy", 0, True), "eyz_b": ("vz", 1, True),
}
#: Componentes de cizalla: (tensión, derivadas a + b de su deformación, ejes del plano).
SHEAR = (("sxy", "exy_a", "exy_b", (1, 2)), ("sxz", "exz_a", "exz_b", (0, 2)), ("syz", "eyz_a", "eyz_b", (0, 1)))


# ----------------------------------------------------------------------------
# Medio
# ----------------------------------------------------------------------------
@dataclass
class ElasticMedium3D:
    """Propiedades por celda ``(nz, ny, nx)`` (float32) para ``ElasticSolver3D``.

    Igual que ``medium.ElasticMedium``: densidad, velocidades P/S y pesos de
    los mecanismos GSLS (``tau_p``, ``tau_s``: ``(L, nz, ny, nx)`` o ``(nz, ny,
    nx)`` con el mismo peso en todos los mecanismos).

    ``stiffness`` (opcional): medio transversalmente isótropo con el eje de
    simetría a lo largo de x, ``(5, nz, ny, nx)`` con ``(C_A, C_T, C_TA, G_A,
    G_T) / rho`` (m²/s²): ``C11 = C_A``, ``C22 = C33 = C_T``, ``C12 = C13 =
    C_TA``, ``C23 = C_T - 2 G_T``, ``C55 = C66 = G_A`` (planos que contienen el
    eje) y ``C44 = G_T`` (plano transversal). En una celda isótropa vale
    ``(cp², cp², cp² - 2cs², cs², cs²)``. ``None`` = isótropo.
    """

    rho: np.ndarray
    cp: np.ndarray
    cs: np.ndarray
    tau_p: np.ndarray
    tau_s: np.ndarray
    tau_sigma: np.ndarray
    f0: float
    stiffness: np.ndarray | None = None

    def weights(self, wave: str) -> np.ndarray:
        """Pesos ``(L, nz, ny, nx)`` de los mecanismos para la onda ``"p"`` o ``"s"``."""
        w = np.asarray(self.tau_p if wave == "p" else self.tau_s)
        if w.ndim == 3:
            return np.broadcast_to(w, (self.tau_sigma.size, *w.shape))
        return w

    def nbytes(self) -> int:
        arrays = [self.rho, self.cp, self.cs, self.tau_p, self.tau_s]
        if self.stiffness is not None:
            arrays.append(self.stiffness)
        return int(sum(a.nbytes for a in arrays))


def isotropic_stiffness_3d(cp, cs) -> np.ndarray:
    """``(C_A, C_T, C_TA, G_A, G_T) / rho`` de un medio isótropo."""
    p2, s2 = np.broadcast_arrays(np.asarray(cp, dtype=float) ** 2, np.asarray(cs, dtype=float) ** 2)
    return np.stack([p2, p2, p2 - 2.0 * s2, s2, s2])


def elastic_medium_3d(phantom: Phantom3D, f0: float, n_mechanisms: int = 2, attenuation: bool = True,
                      anisotropic_bone: bool = False, dtype=np.float32) -> ElasticMedium3D:
    """Medio del solver 3D a partir de un ``Phantom3D`` (tablas de ``materials.acoustic``).

    Los pesos GSLS se ajustan por tejido exactamente como en
    ``medium.elastic_medium`` (filas tejido × onda ajustadas a la vez). No hay
    *speckle* ni porosidad: el fantoma 3D es de capas homogéneas.

    ``anisotropic_bone``: la cortical toma ``CORTICAL_STIFFNESS`` con el eje del
    hueso a lo largo de x (necesita ``phantom.meta["bone_axis"] == "x"``, que
    ponen los generadores de ``phantoms.limb3d``). Las velocidades resultantes
    son 4024 m/s (P axial), 3221 m/s (P transversal), 1690 y 1457 m/s (S). La
    atenuación (Q) es la de la tabla en todas las direcciones.
    """
    n_mech = n_mechanisms if attenuation else 0
    tau_sigma = att.relaxation_times(f0, n_mech)
    band = att.fit_band(f0, n_mech)
    rho = phantom.property_map(lambda k: ACOUSTIC[k].density, dtype)
    cp = phantom.property_map(lambda k: ACOUSTIC[k].c_p, dtype)
    cs = phantom.property_map(lambda k: ACOUSTIC[k].c_s, dtype)
    keys = phantom.present_tissues()
    rows = [(k, shear) for k in keys for shear in (False, True)]
    alpha = np.array([[ACOUSTIC[k].alpha_np_per_m(f, shear=shear) for f in band] for k, shear in rows])
    speed = np.array([ACOUSTIC[k].c_s if shear else ACOUSTIC[k].c_p for k, shear in rows])
    fitted = att.mechanism_weights(alpha, speed, band, tau_sigma, f0).reshape(len(keys), 2, n_mech)
    table = {k: fitted[n] for n, k in enumerate(keys)}
    tau_p = np.stack([phantom.property_map(lambda k, m=m: table[k][0, m], dtype) for m in range(n_mech)]) \
        if n_mech else np.zeros((0, *rho.shape), dtype)
    tau_s = np.stack([phantom.property_map(lambda k, m=m: table[k][1, m], dtype) for m in range(n_mech)]) \
        if n_mech else np.zeros((0, *rho.shape), dtype)
    stiffness = None
    if anisotropic_bone:
        if phantom.meta.get("bone_axis") != "x":
            raise ValueError("anisotropic_bone necesita phantom.meta['bone_axis'] = 'x' (eje del hueso a lo largo "
                             f"de x); es {phantom.meta.get('bone_axis')!r}")
        cort = ACOUSTIC["cortical_bone"]
        ti = CORTICAL_STIFFNESS
        aniso = np.array([ti.c_a, ti.c_t, ti.c_ta, ti.g_a, ti.g_t]) / cort.density
        iso = isotropic_stiffness_3d(cp, cs)
        cells = phantom.labels == TISSUE_INDEX["cortical_bone"]
        if cells.any():
            # cp y cs (los de la tabla isótropa) siguen normalizando las fuentes, como en 2D
            stiffness = iso.astype(dtype)
            stiffness[:, cells] = aniso[:, None]
    return ElasticMedium3D(rho, cp, cs, tau_p, tau_s, tau_sigma, f0, stiffness)


# ----------------------------------------------------------------------------
# Fuentes, receptores y resultados
# ----------------------------------------------------------------------------
@dataclass
class Source3D:
    """Fuente sobre un conjunto de celdas ``(k, j, i)``.

    ``signals``: ``(n_canales, nt)``; ``channel[n]`` es el canal de la celda n.
    ``kind``: "pressure" (Pa), "force_x", "force_y" o "force_z". La
    normalización es la de ``elastic_fdtd.Source`` (incremento por paso
    ``2 cp dt / dx``): un plano de celdas emite una onda plana de ~``signals``
    Pa (presión) o m/s (fuerza, onda P en un fluido) hacia cada lado. Una
    celda aislada en un fluido es un monopolo de caudal ``Q(t) = 2 dx² s(t) /
    (rho c)`` (ver ``point_source_pressure``).
    """

    k: np.ndarray
    j: np.ndarray
    i: np.ndarray
    channel: np.ndarray
    signals: np.ndarray
    kind: str = "pressure"


@dataclass
class Receiver3D:
    """Grupo de celdas cuya magnitud se promedia por canal en cada paso.

    ``quantity``: "p" (``-(sxx + syy + szz)/3``), "vx", "vy" o "vz".
    """

    k: np.ndarray
    j: np.ndarray
    i: np.ndarray
    channel: np.ndarray
    n_channels: int
    quantity: str = "p"


@dataclass
class SimulationResult3D:
    dt: float
    t: np.ndarray
    traces: dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: list[np.ndarray] = field(default_factory=list)
    snapshot_times: list[float] = field(default_factory=list)
    peak_pressure: np.ndarray | None = None
    elapsed_s: float = 0.0
    steps_per_second: float = 0.0


def point_source_pressure(r: np.ndarray, t: np.ndarray, signal_derivative, dx: float, c: float,
                          dt: float = 0.0) -> np.ndarray:
    """Presión analítica de una fuente de presión de una celda en un fluido infinito (función de Green 3D).

    Con la normalización de ``Source3D`` la celda inyecta un caudal ``Q(t) =
    2 dx² s(t) / (rho c)``; el monopolo da ``p = rho Q'(t - r/c) / (4 pi r) =
    dx² s'(t - r/c) / (2 pi r c)``, sin término de campo cercano en la
    presión. ``dt``: el incremento del paso n usa ``s(t_{n+1})`` para el
    intervalo ``[t_n, t_{n+1}]``, lo que adelanta la fuente medio paso.
    """
    return dx**2 / (2.0 * np.pi * r * c) * signal_derivative(t + 0.5 * dt - r / c)


# ----------------------------------------------------------------------------
# CPML 3D
# ----------------------------------------------------------------------------
class CPML3D:
    """Memoria CPML de una derivada (un campo y un eje) en las dos franjas de ese eje.

    Igual que ``core.pml.CPMLDerivative`` (mismos coeficientes,
    ``cpml_coefficients``) para arrays 3D.
    """

    def __init__(self, shape, axis: int, staggered: bool, npml: int, dx: float, dt: float, c_max: float,
                 f_ref: float, dtype=np.float32):
        self.axis = axis
        n = shape[axis]
        pos = np.arange(n, dtype=float) + (0.5 if staggered else 0.0)
        a, b = cpml_coefficients(pos, n, npml, dx, dt, c_max, f_ref)
        active = np.nonzero(a != 0.0)[0]
        left, right = active[active < n // 2], active[active >= n // 2]
        self.nl = int(left.max() + 1) if left.size else 0
        self.nr_start = int(right.min()) if right.size else n
        bshape = [1, 1, 1]
        bshape[axis] = -1
        self.a_l = a[: self.nl].astype(dtype).reshape(bshape)
        self.b_l = b[: self.nl].astype(dtype).reshape(bshape)
        self.a_r = a[self.nr_start:].astype(dtype).reshape(bshape)
        self.b_r = b[self.nr_start:].astype(dtype).reshape(bshape)
        sl, sr = list(shape), list(shape)
        sl[axis], sr[axis] = self.nl, n - self.nr_start
        self.psi_l = np.zeros(sl, dtype)
        self.psi_r = np.zeros(sr, dtype)

    def _slices(self):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[self.axis] = slice(0, self.nl)
        right[self.axis] = slice(self.nr_start, None)
        return tuple(left), tuple(right)

    def apply(self, deriv: np.ndarray) -> None:
        """Corrige ``deriv`` (array completo) in situ dentro de las franjas."""
        left, right = self._slices()
        for psi, a, b, d in ((self.psi_l, self.a_l, self.b_l, deriv[left]),
                             (self.psi_r, self.a_r, self.b_r, deriv[right])):
            if d.size:
                psi *= b
                psi += a * d
                d += psi

    def pack(self) -> tuple:
        """Tupla para los kernels Numba: coeficientes 1D a lo largo del eje y ``psi`` in situ."""
        return (self.nl, self.nr_start, self.a_l.ravel(), self.b_l.ravel(), self.a_r.ravel(), self.b_r.ravel(),
                self.psi_l, self.psi_r)


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
class ElasticSolver3D:
    """Simulación elástica 3D de un ``Phantom3D`` a la frecuencia central ``f0``.

    Parámetros como en ``ElasticSolver`` (sin ``scattering``: el fantoma 3D no
    tiene microestructura), más:

      - ``npml``: grosor de la CPML en celdas, entero o ``(nz, ny, nx)`` por eje
        (0 = sin capa en ese eje).
      - ``dt``: paso de tiempo impuesto (p. ej. el de una simulación 2D para
        compararla); error si supera el límite de estabilidad 3D.
      - ``interface_correction``: corrección de la cizalla junto a interfaces
        fluido-sólido (``_slip_terms``). Desactivarla sólo sirve para comparar
        con un solver que no la tenga.
      - ``track_peak`` de ``run`` está desactivado por defecto: el mapa de
        presión máxima ocupa un array 3D más.
    """

    def __init__(
        self,
        phantom: Phantom3D,
        f0: float,
        npml: int | tuple[int, int, int] = 10,
        n_mechanisms: int = 2,
        attenuation: bool = True,
        cfl: float = 0.9,
        medium: ElasticMedium3D | None = None,
        dtype=np.float32,
        backend: str = "auto",
        anisotropic_bone: bool = False,
        dt: float | None = None,
        interface_correction: bool = True,
    ):
        self.backend = ElasticSolver._resolve_backend(backend, dtype)
        self.phantom = phantom
        self.grid = g = phantom.grid
        if min(g.shape) < 4:
            raise ValueError(f"La malla necesita al menos 4 celdas por eje (es {g.shape})")
        self.f0 = f0
        self.dtype = dtype
        self.medium = m = medium or elastic_medium_3d(phantom, f0, n_mechanisms, attenuation, anisotropic_bone)
        if m.rho.shape != g.shape:
            raise ValueError(f"El medio tiene forma {m.rho.shape} y la malla {g.shape}")
        dx = g.dx
        self.inv_dx = 1.0 / dx
        self.n_mech = L = m.tau_sigma.size
        self.npml = (npml,) * 3 if np.isscalar(npml) else tuple(int(n) for n in npml)
        self.anisotropic = m.stiffness is not None
        shape = g.shape

        def f(a):
            return np.ascontiguousarray(a, dtype=dtype)

        # --- módulos relajados (corregidos por dispersión en f0) y no relajados, celda a celda
        w_p, w_s = m.weights("p"), m.weights("s")
        rho = m.rho.astype(np.float64)
        fp = _planewise_factor(w_p, m.tau_sigma, f0)
        fs = _planewise_factor(w_s, m.tau_sigma, f0)
        relax_p = 1.0 + w_p.sum(axis=0, dtype=np.float64)
        relax_s = 1.0 + w_s.sum(axis=0, dtype=np.float64)
        if self.anisotropic:
            # La parte "P" (C11, C22 = C33 y C12 + 2 C66 = C13 + 2 C55, C23 + 2 C44 = C_T) relaja con los
            # pesos de la onda P y las cizallas con los de la S: en una celda isótropa es lambda + 2 mu y mu,
            # como en el camino isótropo (el mismo criterio que el solver 2D)
            ca, ct, cta, ga, gt = (rho * m.stiffness[q].astype(np.float64) for q in range(5))
            p11, pt, p12 = ca * fp, ct * fp, (cta + 2 * ga) * fp
            mua_r, mut_r = ga * fs, gt * fs
            mua_u, mut_u = mua_r * relax_s, mut_r * relax_s
            c11_u, ct_u = p11 * relax_p, pt * relax_p
            c12_u, c23_u = p12 * relax_p - 2 * mua_u, pt * relax_p - 2 * mut_u
            v2_max = _max_stability_modulus(c11_u, ct_u, c12_u, c23_u, mua_u, mut_u, rho)
            shear_u = {"sxy": mua_u, "sxz": mua_u, "syz": mut_u}
            shear_r = {"sxy": mua_r, "sxz": mua_r, "syz": mut_r}
            solid = mua_u > 0
        else:
            pi_r = rho * m.cp.astype(np.float64) ** 2 * fp
            mu_r = rho * m.cs.astype(np.float64) ** 2 * fs
            pi_u, mu_u = pi_r * relax_p, mu_r * relax_s
            v2_max = float(np.max(pi_u / rho))
            shear_u = dict.fromkeys(("sxy", "sxz", "syz"), mu_u)
            shear_r = dict.fromkeys(("sxy", "sxz", "syz"), mu_r)
            solid = mu_u > 0
        self.c_max = cp_u_max = float(np.sqrt(v2_max))
        # El peor número de onda del esquema escalonado es el de Nyquist en las tres direcciones
        dt_limit = dx / (cp_u_max * np.sqrt(3.0) * (C1 + abs(C2)))
        if dt is not None and dt > dt_limit * (1 + 1e-9):
            raise ValueError(f"dt = {dt:.3e} s supera el límite de estabilidad 3D ({dt_limit:.3e} s)")
        self.dt = float(dt) if dt is not None else cfl * dt_limit
        dt = self.dt
        self._check_sampling(dx)

        # --- coeficientes ya multiplicados por dt
        self.dtb = {c: f(dt * 2.0 / (rho + _shift(rho, ax))) for c, ax in (("vx", 2), ("vy", 1), ("vz", 0))}
        self.dt_mu = {s: f(dt * _harmonic(shear_u[s], axes)) for s, _, _, axes in SHEAR}
        self.has_shear = bool(solid.any())
        if self.anisotropic:
            self.dt_c11, self.dt_ct, self.dt_c12, self.dt_c23 = (f(dt * a) for a in (c11_u, ct_u, c12_u, c23_u))
        else:
            self.dt_pi, self.dt_mu2 = f(dt * pi_u), f(dt * 2 * mu_u)
        if L:
            ts = m.tau_sigma
            self.mem_a = (1 - dt / (2 * ts)) / (1 + dt / (2 * ts))
            self.mem_b = (dt / ts) / (1 + dt / (2 * ts))
            if self.anisotropic:
                self.dt_c11_t, self.dt_ct_t = f(dt * p11 * w_p), f(dt * pt * w_p)
                self.dt_c12_t = f(dt * (p12 * w_p - 2 * mua_r * w_s))
                self.dt_c23_t = f(dt * (pt * w_p - 2 * mut_r * w_s))
            else:
                self.dt_pi_t, self.dt_mu2_t = f(dt * pi_r * w_p), f(dt * 2 * mu_r * w_s)
            self.dt_mu_t = {s: f(np.stack([dt * _harmonic(shear_r[s] * w, axes) for w in w_s]))
                            for s, _, _, axes in SHEAR}
        else:
            self.mem_a = self.mem_b = np.zeros(0)
        del rho, fp, fs, relax_p, relax_s, shear_u, shear_r

        z = lambda: np.zeros(shape, dtype)  # noqa: E731
        self.vx, self.vy, self.vz = z(), z(), z()
        self.sxx, self.syy, self.szz = z(), z(), z()
        self.sxy, self.sxz, self.syz = z(), z(), z()
        self.mem = {s: np.zeros((L, *shape), dtype) for s in ("sxx", "syy", "szz", "sxy", "sxz", "syz")}

        cp_kw = dict(dx=dx, dt=dt, c_max=cp_u_max, f_ref=f0, dtype=dtype)
        self.pml = {}
        for name, (_, axis, fwd, _) in VELOCITY_DERIVS.items():
            self.pml[name] = CPML3D(shape, axis, fwd, self.npml[axis], **cp_kw)
        for name, (_, axis, fwd) in STRESS_DERIVS.items():
            self.pml[name] = CPML3D(shape, axis, fwd, self.npml[axis], **cp_kw)

        self._slip = {}
        if interface_correction and self.has_shear:
            nodes = {s: self.dt_mu[s] > 0 for s, _, _, _ in SHEAR}
            self._slip = {k: (t, S, (W * self.inv_dx).astype(dtype))
                          for k, (t, S, W) in _slip_terms(solid, nodes, self.npml).items()}
        self.sources: list[tuple[Source3D, np.ndarray]] = []
        self.receivers: dict[str, Receiver3D] = {}
        if self.backend == "numba":
            self._init_numba()
        else:
            self._buf = {k: z() for k in ("d", "acc", "e1", "e2", "e3", "div", "a", "b", "f")}
            self._tmp = np.zeros(g.size, dtype)

    # ------------------------------------------------------------------
    def _check_sampling(self, dx: float) -> None:
        m = self.medium
        speeds = [m.cp.ravel(), m.cs[m.cs > 0].ravel()]
        if m.stiffness is not None:
            solid = m.stiffness[3] > 0
            st = m.stiffness[:, solid].astype(np.float64)
            if st.size:
                # Christoffel en el plano que contiene el eje (x-z) y la S del plano transversal
                plane = np.unique(np.stack([st[0], st[1], st[2], st[3]]), axis=1)
                for theta in np.linspace(0.0, np.pi / 2, 7):
                    speeds += list(christoffel_speeds(plane, theta))
                speeds.append(np.sqrt(np.unique(st[4])))
        c_min = float(np.concatenate([np.atleast_1d(s) for s in speeds]).min())
        self.points_per_wavelength = ppw = c_min / (self.f0 * dx)
        if ppw < MIN_SHEAR_PPW:
            warnings.warn(f"Sólo {ppw:.1f} puntos por longitud de onda a f0 (c_min = {c_min:.0f} m/s). "
                          "Habrá dispersión numérica: reduce dx o la frecuencia.", stacklevel=3)

    def _init_numba(self) -> None:
        """Argumentos de los kernels (se importan aquí: Numba es opcional)."""
        from . import _numba_kernels3d as nk

        self._nk = nk
        f32 = np.float32
        self._nb_c = (f32(C1 * self.inv_dx), f32(C2 * self.inv_dx))
        self._nb_pml = {k: v.pack() for k, v in self.pml.items()}
        flat, stack = np.zeros((1, 1, 1), self.dtype), np.zeros((1, 1, 1, 1), self.dtype)
        mem_t = (self.dt_mu_t["sxy"], self.dt_mu_t["sxz"], self.dt_mu_t["syz"]) if self.n_mech else (stack,) * 3
        if self.anisotropic:
            normal = (flat, flat, stack, stack)
            relax = (self.dt_c11_t, self.dt_ct_t, self.dt_c12_t, self.dt_c23_t) if self.n_mech else (stack,) * 4
            aniso = (True, self.dt_c11, self.dt_ct, self.dt_c12, self.dt_c23, *relax)
        else:
            normal = (self.dt_pi, self.dt_mu2, *((self.dt_pi_t, self.dt_mu2_t) if self.n_mech else (stack,) * 2))
            aniso = (False, flat, flat, flat, flat, stack, stack, stack, stack)
        self._nb_stress = (*normal, self.dt_mu["sxy"], self.dt_mu["sxz"], self.dt_mu["syz"], *mem_t, *aniso)
        self._nb_mem_ab = (np.asarray(self.mem_a, np.float64), np.asarray(self.mem_b, np.float64))

    # ------------------------------------------------------------------
    def time_axis(self, nt: int) -> np.ndarray:
        """Instantes en los que se aplican las fuentes y se registran los receptores."""
        return (np.arange(nt) + 1) * self.dt

    def add_source(self, source: Source3D) -> None:
        if source.kind not in ("pressure", "force_x", "force_y", "force_z"):
            raise ValueError(f"Tipo de fuente desconocido '{source.kind}'")
        # Escala 2 cp dt / dx de cada celda (ver ``Source3D``)
        scale = 2.0 * self.medium.cp[source.k, source.j, source.i].astype(np.float64) * self.dt * self.inv_dx
        self.sources.append((source, scale))

    def add_receiver(self, name: str, receiver: Receiver3D) -> None:
        if receiver.quantity not in ("p", "vx", "vy", "vz"):
            raise ValueError(f"Magnitud desconocida '{receiver.quantity}'")
        self.receivers[name] = receiver

    def memory_bytes(self) -> dict[str, int]:
        """Memoria de los arrays del solver (campos, coeficientes, memorias, CPML) y del medio, en bytes."""
        seen: set[int] = set()

        def size(obj) -> int:
            if isinstance(obj, np.ndarray):
                base = obj if obj.base is None else obj.base
                if id(base) in seen or not isinstance(base, np.ndarray):
                    return 0
                seen.add(id(base))
                return base.nbytes
            if isinstance(obj, dict):
                return sum(size(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return sum(size(v) for v in obj)
            if isinstance(obj, CPML3D):
                return size(obj.psi_l) + size(obj.psi_r)
            return 0

        pml = sum(size(p) for p in self.pml.values())
        solver = sum(size(v) for k, v in self.__dict__.items() if k not in ("medium", "phantom", "pml"))
        return {"solver": solver, "pml": pml, "medium": self.medium.nbytes(), "labels": self.phantom.labels.nbytes}

    # ------------------------------------------------------------------
    def _derivative(self, name: str, f: np.ndarray, out: np.ndarray, axis: int, forward: bool) -> np.ndarray:
        _stencil(f, axis, forward, out, C1 * self.inv_dx, C2 * self.inv_dx, self._tmp)
        self.pml[name].apply(out)
        if name in self._slip:
            t, S, W = self._slip[name]
            out.reshape(-1)[t] += np.sum(W * f.reshape(-1)[S], axis=1)
        return out

    def _apply_sources(self, n: int, pressure: bool) -> None:
        for s, scale in self.sources:
            if (s.kind == "pressure") != pressure:
                continue
            inc = s.signals[s.channel, n] * scale
            idx = (s.k, s.j, s.i)
            if pressure:
                for st in (self.sxx, self.syy, self.szz):
                    np.subtract.at(st, idx, inc)
            else:
                np.add.at(getattr(self, "v" + s.kind[-1]), idx, inc)

    def step(self, n: int) -> None:
        """Avanza un paso: velocidades a t+dt/2 y tensiones a t+dt."""
        if self.backend == "numba":
            self._step_numba(n)
        else:
            self._step_numpy(n)

    def _step_numpy(self, n: int) -> None:
        b = self._buf
        acc, d = b["acc"], b["d"]
        # --- velocidades
        for comp in ("vx", "vy", "vz"):
            first = True
            for name, (src, axis, fwd, target) in VELOCITY_DERIVS.items():
                if target != comp or (not self.has_shear and src in ("sxy", "sxz", "syz")):
                    continue
                if first:
                    self._derivative(name, getattr(self, src), acc, axis, fwd)
                    first = False
                else:
                    acc += self._derivative(name, getattr(self, src), d, axis, fwd)
            acc *= self.dtb[comp]
            v = getattr(self, comp)
            v += acc
        self._apply_sources(n, pressure=False)

        # --- deformaciones normales y tensiones normales (parte no relajada)
        e1 = self._derivative("exx", self.vx, b["e1"], 2, False)
        e2 = self._derivative("eyy", self.vy, b["e2"], 1, False)
        e3 = self._derivative("ezz", self.vz, b["e3"], 0, False)
        a1, a2, fo, div = b["a"], b["b"], b["f"], b["div"]
        normals = ((self.sxx, (e1, e2, e3)), (self.syy, (e2, e1, e3)), (self.szz, (e3, e1, e2)))
        if self.anisotropic:
            # sxx = C11 exx + C12 (eyy + ezz); syy = C12 exx + C_T eyy + C23 ezz; szz = C12 exx + C23 eyy + C_T ezz,
            # sumados siempre en el orden exx, eyy, ezz: en un fluido los cuatro coeficientes son iguales y las
            # tres tensiones deben recibir el mismo incremento bit a bit. Con otro orden el redondeo float32 deja
            # una tensión desviadora en el fluido, que no tiene fuerza recuperadora y se acumula paso a paso
            for st, coefs in self._aniso_rows():
                for coef, e in zip(coefs, (e1, e2, e3)):
                    np.multiply(coef, e, out=a1)
                    st += a1
        else:
            np.add(e1, e2, out=div)
            div += e3
            np.multiply(self.dt_pi, div, out=a1)
            for st, (_, ea, eb) in normals:
                np.add(ea, eb, out=a2)
                a2 *= self.dt_mu2
                st += a1
                st -= a2
        for m in range(self.n_mech):
            if self.anisotropic:
                for (st, coefs), key in zip(self._aniso_rows(m), ("sxx", "syy", "szz")):
                    np.multiply(coefs[0], e1, out=fo)
                    for coef, e in zip(coefs[1:], (e2, e3)):
                        np.multiply(coef, e, out=a2)
                        fo += a2
                    self._memory(st, self.mem[key][m], fo, m, a2)
            else:
                np.multiply(self.dt_pi_t[m], div, out=a1)  # parte volumétrica común
                for (st, (_, ea, eb)), key in zip(normals, ("sxx", "syy", "szz")):
                    np.add(ea, eb, out=fo)
                    fo *= self.dt_mu2_t[m]
                    np.subtract(a1, fo, out=fo)
                    self._memory(st, self.mem[key][m], fo, m, a2)

        # --- cizalla
        if self.has_shear:
            exy = b["e1"]  # las deformaciones normales ya no se usan
            for st, da, db, _ in SHEAR:
                fa, axa, fwa = STRESS_DERIVS[da]
                fb, axb, fwb = STRESS_DERIVS[db]
                self._derivative(da, getattr(self, fa), exy, axa, fwa)
                exy += self._derivative(db, getattr(self, fb), d, axb, fwb)
                stress = getattr(self, st)
                np.multiply(self.dt_mu[st], exy, out=a2)
                stress += a2
                for m in range(self.n_mech):
                    np.multiply(self.dt_mu_t[st][m], exy, out=fo)
                    self._memory(stress, self.mem[st][m], fo, m, a2)
        self._apply_sources(n, pressure=True)

    def _aniso_rows(self, mech: int | None = None):
        """Tensiones normales y sus coeficientes de (exx, eyy, ezz); con ``mech``, los de esa memoria."""
        if mech is None:
            c11, ct, c12, c23 = self.dt_c11, self.dt_ct, self.dt_c12, self.dt_c23
        else:
            c11, ct, c12, c23 = (a[mech] for a in (self.dt_c11_t, self.dt_ct_t, self.dt_c12_t, self.dt_c23_t))
        return ((self.sxx, (c11, c12, c12)), (self.syy, (c12, ct, c23)), (self.szz, (c12, c23, ct)))

    def _memory(self, stress, r, forcing, mech, tmp):
        """sigma += (r_new + r_old)/2 ;  r_new = A r_old - B F  (Crank-Nicolson, como en 2D)."""
        a, b = self.mem_a[mech], self.mem_b[mech]
        np.multiply(r, 0.5 * (1 + a), out=tmp)
        stress += tmp
        np.multiply(forcing, 0.5 * b, out=tmp)
        stress -= tmp
        r *= a
        np.multiply(forcing, b, out=tmp)
        r -= tmp

    def _step_numba(self, n: int) -> None:
        """Dos barridos fusionados (``_numba_kernels3d``) más las correcciones de interfaz; fuentes en NumPy."""
        nk, p, (c1, c2) = self._nk, self._nb_pml, self._nb_c
        stresses = (self.sxx, self.syy, self.szz, self.sxy, self.sxz, self.syz)
        nk.velocity_kernel(
            self.vx, self.vy, self.vz, *stresses, self.dtb["vx"], self.dtb["vy"], self.dtb["vz"],
            c1, c2, self.has_shear, *(p[k] for k in VELOCITY_DERIVS),
        )  # fmt: skip
        if self._slip:
            for name, (src, _, _, comp) in VELOCITY_DERIVS.items():
                if name in self._slip:
                    t, S, W = self._slip[name]
                    nk.slip_velocity(_flat(getattr(self, comp)), _flat(self.dtb[comp]), _flat(getattr(self, src)),
                                     t, S, W)
        self._apply_sources(n, pressure=False)
        m = self.mem
        nk.stress_kernel(
            self.vx, self.vy, self.vz, *stresses,
            m["sxx"], m["syy"], m["szz"], m["sxy"], m["sxz"], m["syz"],
            *self._nb_stress, *self._nb_mem_ab, self.has_shear, c1, c2, *(p[k] for k in STRESS_DERIVS),
        )  # fmt: skip
        if self._slip:
            L = self.n_mech
            empty = np.zeros((0, 1), self.dtype)
            for st, da, db, _ in SHEAR:
                coef_t = self.dt_mu_t[st].reshape(L, -1) if L else empty
                r = m[st].reshape(L, -1) if L else empty
                for name in (da, db):
                    if name in self._slip:
                        t, S, W = self._slip[name]
                        nk.slip_stress(_flat(getattr(self, st)), _flat(self.dt_mu[st]), coef_t, r, self._nb_mem_ab[1],
                                       _flat(getattr(self, STRESS_DERIVS[name][0])), t, S, W)
        self._apply_sources(n, pressure=True)

    # ------------------------------------------------------------------
    def pressure(self, k=slice(None), j=slice(None), i=slice(None)) -> np.ndarray:
        """Presión ``-(sxx + syy + szz)/3`` (en todo el bloque o en los índices dados)."""
        return -(self.sxx[k, j, i] + self.syy[k, j, i] + self.szz[k, j, i]) / 3.0

    def _sample(self, rec: Receiver3D) -> np.ndarray:
        idx = (rec.k, rec.j, rec.i)
        vals = self.pressure(*idx) if rec.quantity == "p" else getattr(self, rec.quantity)[idx]
        sums = np.bincount(rec.channel, weights=vals, minlength=rec.n_channels)
        counts = np.bincount(rec.channel, minlength=rec.n_channels)
        return sums / np.maximum(counts, 1)

    def run(self, nt: int, snapshot_every: int | None = None, snapshot_field: str = "p",
            snapshot_plane: tuple[str, int] | None = None, track_peak: bool = False,
            progress: bool = False) -> SimulationResult3D:
        """Ejecuta ``nt`` pasos. Devuelve trazas, snapshots de un plano y (opcional) la presión máxima 3D.

        ``snapshot_plane``: ``("x" | "y" | "z", índice)``; por defecto el plano
        ``y`` central. ``snapshot_field``: "p", "vx", "vy", "vz" o "vmag".
        """
        for s, _ in self.sources:
            if s.signals.shape[1] < nt:
                raise ValueError("Las señales de la fuente son más cortas que la simulación")
        res = SimulationResult3D(dt=self.dt, t=self.time_axis(nt))
        traces = {k: np.zeros((r.n_channels, nt)) for k, r in self.receivers.items()}
        peak = np.zeros(self.grid.shape, self.dtype) if track_peak else None
        peak_every = max(int(1.0 / (self.f0 * self.dt)) // 10, 1)
        plane = snapshot_plane or ("y", self.grid.ny // 2)
        t_start = time.time()
        for n in range(nt):
            self.step(n)
            for k, r in self.receivers.items():
                traces[k][:, n] = self._sample(r)
            if track_peak and n % peak_every == 0:
                np.maximum(peak, np.abs(self.pressure()), out=peak)
            if snapshot_every and n % snapshot_every == 0:
                res.snapshots.append(self._snapshot(snapshot_field, plane))
                res.snapshot_times.append(float(res.t[n]))
            if n % 200 == 0 and not np.isfinite(self.vz[::8, ::8, ::8]).all():
                raise FloatingPointError(f"La simulación se ha vuelto inestable en el paso {n}")
            if progress and n % max(nt // 10, 1) == 0:
                print(f"  paso {n}/{nt} ({100 * n / nt:.0f} %)", flush=True)
        res.traces = traces
        res.peak_pressure = peak
        res.elapsed_s = time.time() - t_start
        res.steps_per_second = nt / max(res.elapsed_s, 1e-12)
        return res

    def _snapshot(self, name: str, plane: tuple[str, int]) -> np.ndarray:
        axis, index = plane
        sel = [slice(None)] * 3
        sel[{"z": 0, "y": 1, "x": 2}[axis]] = index
        sel = tuple(sel)
        if name == "p":
            f = self.pressure(*sel)
        elif name == "vmag":
            f = np.sqrt(self.vx[sel] ** 2 + self.vy[sel] ** 2 + self.vz[sel] ** 2)
        elif name in ("vx", "vy", "vz"):
            f = getattr(self, name)[sel]
        else:
            raise ValueError(f"Campo desconocido '{name}'")
        return np.array(f, dtype=np.float32)


# ----------------------------------------------------------------------------
# Auxiliares
# ----------------------------------------------------------------------------
def _flat(a: np.ndarray) -> np.ndarray:
    """Vista 1D de un array contiguo (los kernels de interfaz trabajan con índices planos)."""
    return a.reshape(-1)


def _stencil(f, axis: int, forward: bool, out, c1: float, c2: float, tmp_flat) -> None:
    """Derivada de 4.º orden a lo largo de ``axis`` en ``out`` (como ``ElasticSolver._dx``/``_dz``).

    Hacia delante (resultado en +1/2) se escriben los índices ``[1, n-3]``;
    hacia atrás, ``[2, n-2]``. El resto se pone a cero: son los bordes que el
    solver 2D nunca escribe, y los kernels Numba hacen lo mismo.
    """
    n = f.shape[axis]

    def s(a, b):
        idx = [slice(None)] * 3
        idx[axis] = slice(a, b)
        return tuple(idx)

    o = out[s(1, n - 2)] if forward else out[s(2, n - 1)]
    np.subtract(f[s(2, n - 1)], f[s(1, n - 2)], out=o)
    o *= c1
    t = tmp_flat[: o.size].reshape(o.shape)
    np.subtract(f[s(3, n)], f[s(0, n - 3)], out=t)
    t *= c2
    o += t
    for z in ((0, n - 2, n - 1) if forward else (0, 1, n - 1)):
        out[s(z, z + 1)] = 0


def _shift(a: np.ndarray, axis: int) -> np.ndarray:
    """Vecino +1 a lo largo del eje (replicando el borde)."""
    out = np.empty_like(a)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    src[axis], dst[axis] = slice(1, None), slice(None, -1)
    out[tuple(dst)] = a[tuple(src)]
    last = [slice(None)] * 3
    last[axis] = slice(-1, None)
    out[tuple(last)] = a[tuple(last)]
    return out


def _harmonic(mu: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Media armónica de las 4 celdas que rodean un nodo de cizalla del plano ``axes``; 0 si alguna es fluida."""
    a, b = axes
    m2 = _shift(mu, a)
    cells = (mu, m2, _shift(mu, b), _shift(m2, b))
    with np.errstate(divide="ignore"):
        inv = sum(1.0 / c for c in cells)
        hm = 4.0 / inv
    solid = np.ones(mu.shape, bool)
    for c in cells:
        solid &= c > 0
    return np.where(solid, hm, 0.0)


def _planewise_factor(w: np.ndarray, tau_sigma: np.ndarray, f0: float) -> np.ndarray | float:
    """``attenuation.relaxed_modulus_factor`` plano a plano (evita arrays complejos del tamaño de la malla)."""
    if tau_sigma.size == 0:
        return 1.0
    out = np.empty(w.shape[1:], np.float64)
    for k in range(w.shape[1]):
        out[k] = att.relaxed_modulus_factor(w[:, k].astype(np.float64), tau_sigma, f0)
    return out


def _max_stability_modulus(c11, ct, c12, c23, mua, mut, rho) -> float:
    """Mayor ``lambda_max(Gamma)/(3 rho)`` del operador discreto en el número de onda de Nyquist en las 3 direcciones.

    Allí la matriz de Christoffel del esquema escalonado es la de ``s = (1, 1,
    1)``; en un medio isótropo su mayor autovalor es ``3 (lambda + 2 mu)`` y la
    condición se reduce a la habitual con ``cp``. Para un medio ortótropo con los
    ejes de la malla, el máximo sobre el cubo de Brillouin está en un vértice y
    todos los vértices son equivalentes (cambios de signo).
    """
    best = 0.0
    for k in range(rho.shape[0]):
        g = np.empty((rho[k].size, 3, 3))
        a11, at, a12, a23 = c11[k].ravel(), ct[k].ravel(), c12[k].ravel(), c23[k].ravel()
        ga, gt = mua[k].ravel(), mut[k].ravel()
        g[:, 0, 0] = a11 + 2 * ga
        g[:, 1, 1] = g[:, 2, 2] = at + ga + gt
        g[:, 0, 1] = g[:, 1, 0] = g[:, 0, 2] = g[:, 2, 0] = a12 + ga
        g[:, 1, 2] = g[:, 2, 1] = a23 + gt
        lam = np.linalg.eigvalsh(g)[:, -1] / (3.0 * rho[k].ravel())
        best = max(best, float(lam.max()))
    return best


def _slip_terms(solid: np.ndarray, shear: dict[str, np.ndarray],
                npml: tuple[int, int, int]) -> dict[str, tuple[np.ndarray, ...]]:
    """Términos que corrigen las derivadas de cizalla junto a una interfaz fluido-sólido (versión 3D).

    Es el principio de la corrección del solver 2D (commit f23a3cf,
    ``elastic_fdtd._slip_terms``) aplicado a las tres componentes de cizalla.
    En la deformación ``d v_c / d s`` de un nodo de cizalla del sólido, el
    estencil de 4.º orden a lo largo de ``s`` lee con peso C2 la velocidad
    ``v_c`` en ``s - 1`` y en ``s + 2``. Si ese nodo de velocidad toca una celda
    fluida (está entre las celdas ``c`` y ``c + 1`` de esa fila), es la velocidad
    tangencial de un fluido que desliza y no la prolongación de la del sólido:
    el salto entra con peso C2/dx, un error O(1) que no baja al refinar. Ese
    punto exterior se sustituye por el interior del mismo lado (``s`` y ``s +
    1``), y el cambio traspuesto se aplica a la ecuación del movimiento (``D_b =
    -D_f^T``): el nodo de velocidad del fluido deja de recibir esa tensión de
    cizalla y la recibe el interior. Así los operadores siguen siendo adjuntos y
    el esquema conserva la energía (``tests/test_elastic3d.py`` lo comprueba).

    ``solid``: celdas con ``mu > 0``; ``shear``: nodos con cizalla de cada
    componente ("sxy", "sxz", "syz"). Como en 2D, se excluye la franja de
    ``npml + 3`` celdas de cada borde (la CPML, cuyas memorias ``psi`` no se
    tocan, y los bordes de las derivadas que no se escriben); en un eje sin
    CPML sólo hace falta si es el del estencil. Así, en un medio extruido en y
    sin CPML en y, los términos de cada plano y son los del 2D y el resultado
    coincide con el del solver 2D corregido. Devuelve, por derivada, ``(t, S,
    W)`` en índices planos: ``deriv.flat[t] += sum(W * f.flat[S], axis=1) /
    dx``, con cada destino una sola vez (relleno con peso 0).
    """
    shape = solid.shape
    fluid = ~solid
    # componente -> [(deformación, ecuación del movimiento, eje s del estencil, eje c de la velocidad)]
    rules = {
        "sxy": (("exy_a", "sxy_y", 1, 2), ("exy_b", "sxy_x", 2, 1)),
        "sxz": (("exz_a", "sxz_z", 0, 2), ("exz_b", "sxz_x", 2, 0)),
        "syz": (("eyz_a", "syz_z", 0, 1), ("eyz_b", "syz_y", 1, 0)),
    }
    terms: dict[str, list] = {}
    for comp, mask in shear.items():
        nodes = np.array(np.nonzero(mask))
        for strain, momentum, s, c in rules[comp]:
            keep = np.ones(nodes.shape[1], bool)
            for a in range(3):
                margin = npml[a] + 3 if (npml[a] > 0 or a == s) else 0
                keep &= (nodes[a] >= margin) & (nodes[a] < shape[a] - margin)
            pos = nodes[:, keep]
            for o, inner in ((-1, 0), (2, 1)):
                ext = pos.copy()
                ext[s] += o
                nxt = ext.copy()
                nxt[c] += 1
                nxt[c] = np.minimum(nxt[c], shape[c] - 1)
                sel = fluid[tuple(ext)] | fluid[tuple(nxt)]
                if not sel.any():
                    continue
                inn = pos[:, sel].copy()
                inn[s] += inner
                t = np.ravel_multi_index(tuple(pos[:, sel]), shape)
                e = np.ravel_multi_index(tuple(ext[:, sel]), shape)
                n_ = np.ravel_multi_index(tuple(inn), shape)
                # -C2 f[s-1] -> -C2 f[s] ;  +C2 f[s+2] -> +C2 f[s+1]
                w = np.full(t.size, C2 if o < 0 else -C2)
                terms.setdefault(strain, []).extend([(t, e, w), (t, n_, -w)])
                terms.setdefault(momentum, []).extend([(e, t, -w), (n_, t, w)])
    out = {}
    for name, parts in terms.items():
        tgt, src, w = (np.concatenate(p) for p in zip(*parts))
        t, inv = np.unique(tgt, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=t.size)
        col = np.arange(inv.size) - np.repeat(np.cumsum(counts) - counts, counts)
        S = np.zeros((t.size, counts.max()), np.int64)
        W = np.zeros((t.size, counts.max()))
        S[inv[order], col] = src[order]
        W[inv[order], col] = w[order]
        out[name] = (t, S, W)
    return out
