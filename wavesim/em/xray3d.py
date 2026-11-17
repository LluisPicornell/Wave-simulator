"""Rayos X y TAC en 3D: proyector de haz cónico/paralelo, radiografía digital 2D y TAC de haz cónico (FDK).

Todo lee un ``Phantom3D`` (o un mapa de atenuación 3D) con la convención de los fantomas 3D: arrays
``(nz, ny, nx)``, x = eje del hueso, y = lateral, z = profundidad (la fuente está arriba, en -z, para la
vista antero-posterior). Reutiliza el espectro del tubo, la descomposición agua + hueso, el detector
(``xray_detector``) y el Monte Carlo de scatter de los módulos 2D.

**Geometría** (``Geometry3D``). Órbita circular alrededor de un eje (por defecto x, el del hueso, como un TAC
de extremidad) con la fuente a ``sid`` del isocentro (el centro del dominio) y un detector plano a ``sdd`` de
la fuente, de ``n_u × n_v`` píxeles de paso ``du × dv`` medido en el plano del detector. Para la vista
``theta`` el rayo central es ``d = cos θ a + sin θ b`` (con ``(a, b) = (z, y)`` si el eje es x), el eje
horizontal del detector ``u = -sin θ a + cos θ b`` y el vertical ``v`` = eje de giro. Con ``theta = 0`` es la
vista antero-posterior (AP: u = y, v = x, el hueso queda vertical en la imagen) y con ``theta = 90°`` la
lateral. ``parallel=True`` da haz paralelo (la imagen no se magnifica y ``sdd`` se ignora). La magnificación
en el isocentro es ``sdd / sid``.

**Proyector.** Dos métodos, ambos con su adjunto:

- ``"siddon"`` (por defecto): longitudes exactas de cada rayo en cada celda del volumen constante a trozos
  (Siddon; recorrido celda a celda al estilo Amanatides-Woo). No tiene error de muestreo: el único error
  respecto a la integral del objeto continuo es el de discretizar el objeto en celdas. Backend Numba
  (``backend="numba"``, el recorrido en paralelo con los hilos de Numba) o NumPy (cruces de planos
  ordenados, vectorizado por lotes; mismo resultado, unas 20-50 veces más lento).
- ``"sampling"``: muestreo del volumen con interpolación trilineal a lo largo del rayo (regla del punto
  medio con paso ``step`` celdas). Suaviza el objeto (la trilineal es un filtro paso bajo de media celda)
  y tiene un error de cuadratura que crece con ``step``. Sólo NumPy. Se incluye para comparar
  (``docs/validacion/rayos_x3d.md``).

Para el TAC y la radiografía se usa Siddon: exacto en las cuerdas y sin sesgo de suavizado.

**Reconstrucción.** ``fdk`` (Feldkamp-Davis-Kress: ponderación por coseno, rampa a lo largo de las filas del
detector, retroproyección dirigida por vóxel con peso ``(sid/(sid+r))²``) reconstruye una órbita circular
completa; es exacta en el plano central y aproximada fuera de él (artefacto de haz cónico, crece con el
ángulo de cono). ``mode="stack"`` reconstruye cada corte con la FBP de abanico sin corrección de cono (el
apilado de cortes de abanico) y ``mode="parallel"`` es el apilado de cortes de haz paralelo (con
``Geometry3D(parallel=True)``). ``os_sart3d`` es la alternativa iterativa con variación total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.grid3d import Grid3D
from ..materials import xray as xd
from ..phantoms.limb3d import Phantom3D
from .projection import ramp_filter
from .xray import Projections, XraySpectrum, basis_coefficients, effective_water_mu, tube_spectrum, water_linearize
from .xray_detector import DetectorModel, blur_signal

# (a, b, v) por eje de giro: rayo central d = cos θ a + sin θ b, detector u = -sin θ a + cos θ b, v = eje
_AXES = {"x": ((0, 0, 1), (0, 1, 0), (1, 0, 0)), "y": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
         "z": ((1, 0, 0), (0, 1, 0), (0, 0, 1))}


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


def center(grid: Grid3D) -> np.ndarray:
    """Isocentro (x, y, z): el centro del dominio."""
    return np.array([grid.x0 + 0.5 * (grid.nx - 1) * grid.dx, grid.y0 + 0.5 * (grid.ny - 1) * grid.dx,
                     grid.z0 + 0.5 * (grid.nz - 1) * grid.dx])


# --------------------------------------------------------------------------------------
# Geometría
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry3D:
    """Fuente puntual (o haz paralelo) y detector plano ``n_u × n_v`` con órbita circular (ver el módulo)."""

    sid: float
    sdd: float
    n_u: int
    n_v: int
    du: float
    dv: float
    axis: str = "x"
    parallel: bool = False

    def __post_init__(self):
        if self.axis not in _AXES:
            raise ValueError("axis debe ser 'x', 'y' o 'z'")
        if not self.parallel and not 0 < self.sid < self.sdd:
            raise ValueError("Se exige 0 < sid < sdd")

    @classmethod
    def covering(cls, grid: Grid3D, sid: float, sdd: float | None = None, axis: str = "x", parallel: bool = False,
                 oversample: float = 1.0, margin: float = 1.02) -> "Geometry3D":
        """Detector que cubre todo el dominio en cualquier vista, con el paso de la malla proyectado.

        ``oversample`` > 1 reduce el paso (más píxeles). La fuente debe quedar fuera del cilindro que
        circunscribe al dominio en el plano perpendicular al eje.
        """
        sdd = sid if sdd is None else sdd
        ext = {"x": grid.nx, "y": grid.ny, "z": grid.nz}
        ext_perp = [ext[k] * grid.dx for k in "xyz" if k != axis]
        radius = 0.5 * float(np.hypot(*ext_perp))
        half_v = 0.5 * ext[axis] * grid.dx
        if not parallel and radius >= sid:
            raise ValueError("La fuente debe estar fuera del cilindro que circunscribe al dominio")
        mag = 1.0 if parallel else sdd / sid
        du = grid.dx * mag / oversample
        if parallel:
            half_u, half_vv = radius, half_v
        else:
            # Semi-anchura máxima proyectada: el punto más cercano a la fuente se magnifica más (peor caso)
            half_u = sdd * radius / np.sqrt(sid**2 - radius**2)
            half_vv = sdd * half_v / (sid - radius)
        n_u = int(np.ceil(2 * half_u * margin / du)) | 1
        n_v = int(np.ceil(2 * half_vv * margin / du)) | 1
        return cls(sid, sdd, n_u, n_v, du, du, axis, parallel)

    @property
    def magnification(self) -> float:
        return 1.0 if self.parallel else self.sdd / self.sid

    @property
    def u(self) -> np.ndarray:
        """Posición de cada columna del detector (m), centrada."""
        return (np.arange(self.n_u) - 0.5 * (self.n_u - 1)) * self.du

    @property
    def v(self) -> np.ndarray:
        return (np.arange(self.n_v) - 0.5 * (self.n_v - 1)) * self.dv

    def view_vectors(self, theta_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rayo central ``d`` y ejes ``u``, ``v`` del detector (vectores unitarios (x, y, z))."""
        a, b, v = (np.array(t, float) for t in _AXES[self.axis])
        th = np.radians(theta_deg)
        return np.cos(th) * a + np.sin(th) * b, -np.sin(th) * a + np.cos(th) * b, v

    def source(self, grid: Grid3D, theta_deg: float) -> np.ndarray:
        return center(grid) - self.sid * self.view_vectors(theta_deg)[0]

    def rays(self, grid: Grid3D, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """Origen y dirección unitaria ``(n_v·n_u, 3)`` de los rayos de una vista (orden fila-columna)."""
        d, u, v = self.view_vectors(theta_deg)
        c = center(grid)
        uu, vv = np.meshgrid(self.u, self.v)
        if self.parallel:
            org = c + uu.reshape(-1, 1) * u + vv.reshape(-1, 1) * v - self.sid * d
            return org, np.broadcast_to(d, org.shape).copy()
        src = c - self.sid * d
        pts = c + (self.sdd - self.sid) * d + uu.reshape(-1, 1) * u + vv.reshape(-1, 1) * v
        vec = pts - src
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        return np.broadcast_to(src, vec.shape).copy(), vec


# --------------------------------------------------------------------------------------
# Proyector y adjunto
# --------------------------------------------------------------------------------------
def _to_index(grid: Grid3D, org: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((org - np.array([grid.x0, grid.y0, grid.z0])) / grid.dx)


def _clip_numpy(o: np.ndarray, d: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``[tmin, tmax]`` de cada rayo en la caja de celdas (``tmax <= tmin`` si no la cruza)."""
    nz, ny, nx = shape
    hi = np.array([nx, ny, nz]) - 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-0.5 - o) / d, (hi - o) / d
    par = np.abs(d) < 1e-12
    inside = (o >= -0.5) & (o <= hi)
    lo = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    up = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    return lo.max(axis=1), up.min(axis=1)


def _siddon_segments(o: np.ndarray, d: np.ndarray, shape: tuple[int, int, int]):
    """Celdas planas y longitudes de cada rayo con cruces de planos ordenados (lote ``(n, 3)``).

    Devuelve ``idx (n, m)`` y ``seg (n, m)`` (longitud 0 en los tramos vacíos o fuera de la caja).
    """
    nz, ny, nx = shape
    tmin, tmax = _clip_numpy(o, d, shape)
    ok = tmax > tmin
    tmin, tmax = np.where(ok, tmin, 0.0), np.where(ok, tmax, 0.0)
    planes = []
    for a, n in enumerate((nx, ny, nz)):
        b = np.arange(n + 1) - 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (b[None, :] - o[:, a:a + 1]) / d[:, a:a + 1]
        planes.append(np.where(np.isfinite(t), t, tmin[:, None]))
    t = np.clip(np.concatenate(planes + [tmin[:, None], tmax[:, None]], axis=1), tmin[:, None], tmax[:, None])
    t.sort(axis=1)
    seg = np.diff(t, axis=1)
    mid = 0.5 * (t[:, 1:] + t[:, :-1])
    p = o[:, None, :] + mid[:, :, None] * d[:, None, :]
    ci = np.clip(np.floor(p + 0.5).astype(np.int64), 0, [nx - 1, ny - 1, nz - 1])
    idx = (ci[..., 2] * ny + ci[..., 1]) * nx + ci[..., 0]
    return idx, seg


def _sampling_points(o: np.ndarray, d: np.ndarray, shape: tuple[int, int, int], step: float):
    """Puntos de muestreo (regla del punto medio dentro de la caja) y pesos ``(n, ns)`` de cada rayo."""
    tmin, tmax = _clip_numpy(o, d, shape)
    length = np.maximum(tmax - tmin, 0.0)
    tmin = np.where(length > 0, tmin, 0.0)
    ns_r = np.maximum(np.ceil(length / step).astype(int), 1)
    ns = int(ns_r.max())
    k = np.arange(ns)[None, :]
    h = (length / ns_r)[:, None]
    t = tmin[:, None] + (k + 0.5) * h
    valid = (k < ns_r[:, None]) & (length[:, None] > 0)
    return o[:, None, :] + t[:, :, None] * d[:, None, :], np.where(valid, h, 0.0)


def _trilinear_corners(p: np.ndarray, shape: tuple[int, int, int]):
    """Índices planos y pesos de los 8 vecinos de cada punto (repetición de borde fuera del centro de celda)."""
    nz, ny, nx = shape
    lim = np.array([nx - 1, ny - 1, nz - 1])
    p = np.clip(p, 0.0, lim)
    i0 = np.minimum(np.floor(p).astype(np.int64), lim - 1) if min(shape) > 1 else np.zeros(p.shape, np.int64)
    f = p - i0
    out = []
    for dk in (0, 1):
        for dj in (0, 1):
            for di in (0, 1):
                w = ((f[..., 0] if di else 1 - f[..., 0]) * (f[..., 1] if dj else 1 - f[..., 1])
                     * (f[..., 2] if dk else 1 - f[..., 2]))
                ii = np.minimum(i0[..., 0] + di, nx - 1)
                jj = np.minimum(i0[..., 1] + dj, ny - 1)
                kk = np.minimum(i0[..., 2] + dk, nz - 1)
                out.append(((kk * ny + jj) * nx + ii, w))
    return out


def _chunks(n: int, per: int):
    for s in range(0, n, per):
        yield slice(s, min(s + per, n))


def _forward_view(vols: np.ndarray, o: np.ndarray, d: np.ndarray, method: str, be: str, step: float) -> np.ndarray:
    """Integrales de línea (en celdas) de los mapas ``vols (m, nz, ny, nx)`` para un lote de rayos ``(n, 3)``."""
    m = vols.shape[0]
    shape = vols.shape[1:]
    if method == "siddon" and be == "numba":
        from ._numba_xray3d import forward

        out = np.zeros((m, o.shape[0]))
        forward(vols, o, np.ascontiguousarray(d), out)
        return out
    flat = vols.reshape(m, -1)
    out = np.zeros((m, o.shape[0]))
    if method == "siddon":
        per = max(int(4e6 // (sum(shape) + 5)), 1)
        for sl in _chunks(o.shape[0], per):
            idx, seg = _siddon_segments(o[sl], d[sl], shape)
            for q in range(m):
                out[q, sl] = np.einsum("rs,rs->r", flat[q][idx], seg)
        return out
    per = max(int(2e6 // (max(shape) / step + 2)), 1)
    for sl in _chunks(o.shape[0], per):
        p, h = _sampling_points(o[sl], d[sl], shape, step)
        for q in range(m):
            val = np.zeros(h.shape)
            for idx, w in _trilinear_corners(p, shape):
                val += flat[q][idx] * w
            out[q, sl] = (val * h).sum(axis=1)
    return out


def _adjoint_view(vals: np.ndarray, o: np.ndarray, d: np.ndarray, shape: tuple[int, int, int], method: str, be: str,
                  step: float, acc: np.ndarray) -> None:
    """Suma en ``acc`` (plano) la retroproyección exacta de ``vals`` (n,) para un lote de rayos."""
    nz, ny, nx = shape
    if method == "siddon" and be == "numba":
        from ._numba_xray3d import adjoint

        adjoint(np.ascontiguousarray(vals, dtype=np.float64), o, np.ascontiguousarray(d), acc, nx, ny, nz)
        return
    if method == "siddon":
        per = max(int(4e6 // (sum(shape) + 5)), 1)
        for sl in _chunks(o.shape[0], per):
            idx, seg = _siddon_segments(o[sl], d[sl], shape)
            acc += np.bincount(idx.ravel(), weights=(seg * vals[sl, None]).ravel(), minlength=acc.size)
        return
    per = max(int(2e6 // (max(shape) / step + 2)), 1)
    for sl in _chunks(o.shape[0], per):
        p, h = _sampling_points(o[sl], d[sl], shape, step)
        for idx, w in _trilinear_corners(p, shape):
            acc += np.bincount(idx.ravel(), weights=(w * h * vals[sl, None]).ravel(), minlength=acc.size)


def _check_method(method: str) -> None:
    if method not in ("siddon", "sampling"):
        raise ValueError(f"método desconocido '{method}' (use 'siddon' o 'sampling')")


def project(volume: np.ndarray, grid: Grid3D, geometry: Geometry3D, angles_deg, method: str = "siddon",
            backend: str = "auto", step: float = 0.5) -> np.ndarray:
    """Integrales de línea de un mapa 3D ``(nz, ny, nx)`` (o de varios ``(m, nz, ny, nx)``).

    Devuelve ``(n_ángulos, n_v, n_u)`` (``(m, ...)`` si hay varios mapas) en unidades de mapa × metros.
    ``step``: paso de muestreo en celdas del método ``"sampling"``.
    """
    _check_method(method)
    be = _resolve_backend(backend) if method == "siddon" else "numpy"
    vols = np.ascontiguousarray(volume, dtype=np.float64)
    single = vols.ndim == 3
    vols = vols[None] if single else vols
    if vols.shape[1:] != grid.shape:
        raise ValueError(f"el volumen tiene forma {vols.shape[1:]}, la malla {grid.shape}")
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    out = np.zeros((vols.shape[0], angles.size, geometry.n_v, geometry.n_u))
    for a, th in enumerate(angles):
        org, dirs = geometry.rays(grid, th)
        r = _forward_view(vols, _to_index(grid, org), dirs, method, be, step)
        out[:, a] = r.reshape(vols.shape[0], geometry.n_v, geometry.n_u) * grid.dx
    return out[0] if single else out


def backproject(sino: np.ndarray, grid: Grid3D, geometry: Geometry3D, angles_deg, method: str = "siddon",
                backend: str = "auto", step: float = 0.5) -> np.ndarray:
    """Operador adjunto de ``project``: ``Aᵀ y`` para ``y (n_ángulos, n_v, n_u)``, forma ``(nz, ny, nx)``.

    Con ``Siddon`` es la transpuesta exacta (``<A x, y> = <x, Aᵀ y>`` hasta el redondeo); con ``"sampling"``
    también (reparte con los mismos pesos trilineales).
    """
    _check_method(method)
    be = _resolve_backend(backend) if method == "siddon" else "numpy"
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    sino = np.asarray(sino, float)
    acc = np.zeros(grid.size)
    for a, th in enumerate(angles):
        org, dirs = geometry.rays(grid, th)
        _adjoint_view(sino[a].reshape(-1), _to_index(grid, org), dirs, grid.shape, method, be, step, acc)
    return (acc * grid.dx).reshape(grid.shape)


# --------------------------------------------------------------------------------------
# Fantomas analíticos de elipsoides
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Ellipsoid:
    """Elipsoide de valor ``value`` (centro y semiejes en metros respecto al centro del dominio, giro ``phi``
    en el plano (x, y) en grados)."""

    center: tuple[float, float, float]
    semi_axes: tuple[float, float, float]
    value: float
    phi_deg: float = 0.0


def ellipsoid_volume(grid: Grid3D, ellipsoids, supersample: int = 1) -> np.ndarray:
    """Rasteriza elipsoides (los valores se suman). ``supersample`` = n da ocupación parcial con n³ puntos por celda.

    Con ``supersample=1`` es la rasterización por el centro de celda (borde "escalera"); con 3 el error de
    volumen parcial es mucho menor, que es lo que importa al comparar con cuerdas analíticas.
    """
    vol = np.zeros(grid.shape)
    c0 = center(grid)
    s = int(supersample)
    off = (np.arange(s) + 0.5) / s - 0.5
    for e in ellipsoids:
        ctr = c0 + np.array(e.center)
        a = np.array(e.semi_axes)
        ph = np.radians(e.phi_deg)
        reach = float(a.max()) + grid.dx
        ix = np.nonzero(np.abs(grid.x - ctr[0]) <= reach)[0]
        iy = np.nonzero(np.abs(grid.y - ctr[1]) <= reach)[0]
        iz = np.nonzero(np.abs(grid.z - ctr[2]) <= reach)[0]
        if ix.size == 0 or iy.size == 0 or iz.size == 0:
            continue
        frac = np.zeros((iz.size, iy.size, ix.size))
        for oz in off:
            zz = (grid.z[iz] + oz * grid.dx - ctr[2])[:, None, None]
            for oy in off:
                yy = (grid.y[iy] + oy * grid.dx - ctr[1])[None, :, None]
                for ox in off:
                    xx = (grid.x[ix] + ox * grid.dx - ctr[0])[None, None, :]
                    xr = xx * np.cos(ph) + yy * np.sin(ph)
                    yr = -xx * np.sin(ph) + yy * np.cos(ph)
                    frac += ((xr / a[0]) ** 2 + (yr / a[1]) ** 2 + (zz / a[2]) ** 2 <= 1.0)
        vol[np.ix_(iz, iy, ix)] += e.value * frac / s**3
    return vol


def shepp_logan_3d(grid: Grid3D, scale: float = 1.0, mu: float = 1.0) -> tuple[np.ndarray, list[Ellipsoid]]:
    """Fantoma de elipsoides del tipo Shepp-Logan modificado, a escala ``scale`` (semiejes relativos al lado menor).

    **No es el fantoma 3D canónico** (cuyos parámetros no se pudieron verificar): son 10 elipsoides de estilo
    Shepp-Logan 3D con contrastes del 2D modificado (cráneo 1, interior -0.8 → 0.2, ventrículos, lesiones),
    definidos aquí. Devuelve ``(volumen, lista de elipsoides)``; el valor máximo es ``mu`` (por 1.0).
    """
    half = 0.5 * min(grid.nx, grid.ny, grid.nz) * grid.dx * scale * 0.95
    sp = [  # (centro, semiejes, valor, giro): en unidades de ``half``, contrastes del Shepp-Logan modificado
        ((0, 0, 0), (0.69, 0.92, 0.90), 1.0, 0),
        ((0, -0.0184, 0), (0.6624, 0.874, 0.88), -0.8, 0),
        ((0.22, 0, 0), (0.41, 0.16, 0.21), -0.2, -18),
        ((-0.22, 0, 0), (0.31, 0.11, 0.22), -0.2, 18),
        ((0, 0.35, -0.15), (0.21, 0.25, 0.35), 0.1, 0),
        ((0, 0.1, 0.25), (0.046, 0.046, 0.046), 0.1, 0),
        ((0, -0.1, 0.25), (0.046, 0.046, 0.046), 0.1, 0),
        ((-0.08, -0.605, 0), (0.046, 0.023, 0.02), 0.1, 0),
        ((0, -0.606, 0), (0.023, 0.023, 0.02), 0.1, 0),
        ((0.06, -0.605, 0), (0.023, 0.046, 0.02), 0.1, 0),
    ]
    # El 2D usa x horizontal e y vertical; aquí el eje largo (y de la lista) cae sobre x del dominio
    ell = [Ellipsoid((c[1] * half, c[0] * half, c[2] * half), (a[1] * half, a[0] * half, a[2] * half), mu * v, -p)
           for c, a, v, p in sp]
    return ellipsoid_volume(grid, ell, supersample=2), ell


def ray_point_distance(src: np.ndarray, dirs: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Distancia de cada rayo ``(src, dirs unitarios (n, 3))`` a un punto: para las cuerdas analíticas."""
    w = point - src
    return np.linalg.norm(np.cross(dirs, w), axis=-1)


# --------------------------------------------------------------------------------------
# Detector plano 2D
# --------------------------------------------------------------------------------------
#: Atenuación másica aproximada del CsI (cm²/g) a 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200 keV. NO está
#: verificada (no se pudo consultar la base NIST): se reconstruyó con el salto K del I (33.2 keV) y la
#: dependencia ~E^-2.8; el error esperado es de ±20 %. Si importa, pasa ``efficiency`` propia.
_CSI_E = np.array([10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200], float)
_CSI_MU_RHO = np.array([200.0, 80.0, 36.0, 10.4, 26.0, 13.9, 8.4, 3.9, 2.1, 0.8, 0.5])
_CSI_DENSITY = 4.51  # g/cm³


@dataclass(frozen=True)
class FlatPanel:
    """Detector de panel plano: eficiencia cuántica por energía, MTF/apertura y ruido electrónico.

    - ``csi_thickness``: espesor del centelleador (m). La eficiencia es ``1 - exp(-μ_CsI(E)·packing·t)``;
      ``None`` = detector ideal (eficiencia 1 a todas las energías, como el modelo 2D).
    - ``packing``: fracción de densidad del CsI estructurado (agujas): 0.75 típico, no verificado.
    - ``energy_integrating``: la señal de cada fotón absorbido es proporcional a su energía (panel
      integrador, el habitual en radiografía). Si es ``False``, cuenta fotones. No se modela la dispersión
      de la luz entre agujas ni el factor de Swank: el ruido es el de Poisson de los fotones absorbidos.
    - ``model``: desenfoque, apertura del píxel y ruido electrónico de ``xray_detector.DetectorModel``,
      aplicados en los dos ejes del detector (separables).
    - ``efficiency``: función ``E(keV) -> η`` que sustituye a la del CsI.
    """

    model: DetectorModel = DetectorModel()
    csi_thickness: float | None = 0.5e-3
    packing: float = 0.75
    energy_integrating: bool = True
    efficiency: Callable[[np.ndarray], np.ndarray] | None = None

    def quantum_efficiency(self, energies_kev: np.ndarray) -> np.ndarray:
        e = np.asarray(energies_kev, float)
        if self.efficiency is not None:
            return np.asarray(self.efficiency(e), float)
        if self.csi_thickness is None:
            return np.ones_like(e)
        mu_rho = np.exp(np.interp(np.log(np.clip(e, _CSI_E[0], _CSI_E[-1])), np.log(_CSI_E), np.log(_CSI_MU_RHO)))
        return 1.0 - np.exp(-mu_rho * _CSI_DENSITY * self.packing * 100.0 * self.csi_thickness)

    def signal_weights(self, spectrum: XraySpectrum) -> tuple[np.ndarray, float]:
        """Pesos de señal por energía (normalizados a 1) y fracción de fotones absorbidos."""
        eta = self.quantum_efficiency(spectrum.energies_kev)
        absorbed = spectrum.weights * eta
        w = absorbed * (spectrum.energies_kev if self.energy_integrating else 1.0)
        return w / w.sum(), float(absorbed.sum())


def _as_panel(detector) -> FlatPanel:
    if detector is None:
        return FlatPanel(csi_thickness=None, energy_integrating=False)  # ideal: igual que el modelo 2D
    if isinstance(detector, DetectorModel):
        return FlatPanel(model=detector, csi_thickness=None, energy_integrating=False)
    return detector


def _blur2d(img: np.ndarray, du: float, dv: float, model: DetectorModel) -> np.ndarray:
    """MTF del detector separable en u (último eje) y v (penúltimo) de ``(..., n_v, n_u)``."""
    out = blur_signal(img, du, model)
    return np.swapaxes(blur_signal(np.swapaxes(out, -1, -2), dv, model), -1, -2)


# --------------------------------------------------------------------------------------
# Adquisición: radiografía y proyecciones de TAC
# --------------------------------------------------------------------------------------
@dataclass
class Projections3D:
    """Proyecciones 2D de una o varias vistas.

    ``line_integrals = -ln(I/I0)`` medido (con ruido), ``intensity = I/I0``; ``primary`` y ``scatter``
    sólo si se pidió scatter (señal esperada, sin ruido). Formas ``(n_ángulos, n_v, n_u)``.
    """

    angles_deg: np.ndarray
    geometry: Geometry3D
    line_integrals: np.ndarray
    intensity: np.ndarray
    spectrum: XraySpectrum
    photons: float | None
    primary: np.ndarray | None = None
    scatter: np.ndarray | None = None

    def as_projections(self) -> Projections:
        """Vista compatible con las utilidades 2D (``xray.typical_path``...): aplana vistas y filas."""
        li = self.line_integrals.reshape(-1, self.geometry.n_u)
        return Projections(self.angles_deg, self.geometry.u, li, self.intensity.reshape(li.shape), self.spectrum)


def acquire3d(phantom: Phantom3D, angles_deg, geometry: Geometry3D, spectrum: XraySpectrum | None = None,
              photons: float | None = 1e5, detector: FlatPanel | DetectorModel | None = None,
              rng: np.random.Generator | None = None, scatter=None, method: str = "siddon",
              backend: str = "auto") -> Projections3D:
    """Proyecciones policromáticas con detector, ruido de Poisson y (opcional) scatter del ``Phantom3D``.

    Cada tejido se descompone en agua + hueso a las energías del espectro (``xray.basis_coefficients``):
    dos proyecciones por vista dan el haz policromático exacto de esa descomposición. ``photons``: fotones
    incidentes por píxel del detector sin objeto (None = sin ruido); el ruido es el de Poisson de los
    fotones absorbidos y es uniforme en el detector (no se incluye la caída del flujo en los bordes ``cos³``
    de un haz cónico). ``scatter``: ``xray3d_scatter.Scatter3DOptions`` (suma la radiación dispersa del
    Monte Carlo 3D a la intensidad primaria antes del ruido).
    """
    spectrum = spectrum or tube_spectrum(60.0, 2.5)
    panel = _as_panel(detector)
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    coef = {k: basis_coefficients(k, spectrum) for k in phantom.present_tissues()}
    maps = np.stack([phantom.property_map(lambda k, i=i: coef[k][i], dtype=np.float64) for i in (0, 1)])
    lw, lb = project(maps, phantom.grid, geometry, angles, method, backend)
    e = spectrum.energies_kev
    mu_w, mu_b = xd.linear_attenuation("water", e), xd.linear_attenuation("cortical_bone", e)
    sig_w, absorbed = panel.signal_weights(spectrum)
    primary = np.zeros(lw.shape)
    for q in range(e.size):  # un bucle por energía evita el tensor (vistas, filas, columnas, energías)
        primary += sig_w[q] * np.exp(-(lw * mu_w[q] + lb * mu_b[q]))
    intensity, scattered = primary, None
    if scatter is not None:
        from .xray3d_scatter import scatter_views

        scattered = scatter_views(phantom, angles, geometry, spectrum, scatter, panel)
        intensity = primary + scattered
    if photons is not None:
        rng = rng or np.random.default_rng(0)
        n0 = photons * absorbed
        expected = intensity * n0
        if panel.model.blur_sigma > 0 or panel.model.pixel_pitch > 0:
            expected = _blur2d(expected, geometry.du, geometry.dv, panel.model)
        counts = rng.poisson(np.maximum(expected, 0.0)).astype(float)
        if panel.model.electronic_noise > 0:
            counts += rng.normal(0.0, panel.model.electronic_noise, counts.shape)
        intensity = np.maximum(counts, 1.0) / n0
    elif panel.model.blur_sigma > 0 or panel.model.pixel_pitch > 0:
        intensity = _blur2d(intensity, geometry.du, geometry.dv, panel.model)
    return Projections3D(angles, geometry, -np.log(intensity), intensity, spectrum, photons,
                         primary if scatter is not None else None, scattered)


_VIEWS = {"ap": 0.0, "lateral": 90.0}


def radiograph(phantom: Phantom3D, view: str | float = "ap", spectrum: XraySpectrum | None = None,
               geometry: Geometry3D | None = None, sid: float = 1.0, sdd: float = 1.1, **kw) -> Projections3D:
    """Radiografía digital 2D de la extremidad: vista ``"ap"`` (antero-posterior), ``"lateral"`` o un ángulo.

    Por defecto 60 kVp y 2.5 mm Al (extremidades), fuente a ``sid`` del eje del hueso y panel a ``sdd`` de la
    fuente (magnificación ``sdd/sid`` en el eje), con un detector que cubre todo el dominio y el paso de
    la malla proyectado. ``kw`` pasa a ``acquire3d`` (``photons``, ``detector``, ``scatter``, ``rng``...).
    La imagen ``line_integrals[0]`` tiene el eje del hueso en vertical (filas = x, columnas = y).
    """
    ang = _VIEWS[view] if isinstance(view, str) else float(view)
    geometry = geometry or Geometry3D.covering(phantom.grid, sid, sdd, oversample=2.0)
    return acquire3d(phantom, [ang], geometry, spectrum or tube_spectrum(60.0, 2.5), **kw)


# --------------------------------------------------------------------------------------
# Reconstrucción FDK
# --------------------------------------------------------------------------------------
def _backproject_numpy(q: np.ndarray, vol: np.ndarray, grid: Grid3D, geom: Geometry3D, theta_deg: float,
                       weight: float, vmag_fixed: float) -> None:
    d, u, v = geom.view_vectors(theta_deg)
    c = center(grid)
    nv, nu = q.shape
    xs, ys = grid.x - c[0], grid.y - c[1]
    for k0 in range(0, grid.nz, 8):
        k1 = min(k0 + 8, grid.nz)
        zz = (grid.z[k0:k1] - c[2])[:, None, None]
        px, py = xs[None, None, :], ys[None, :, None]
        r = px * d[0] + py * d[1] + zz * d[2]
        t = px * u[0] + py * u[1] + zz * u[2]
        w = px * v[0] + py * v[1] + zz * v[2]
        if geom.parallel:
            mag, f = np.ones_like(r), 1.0
        else:
            den = geom.sid + r
            mag, f = geom.sdd / den, (geom.sid / den) ** 2
        fu = t * mag / geom.du + 0.5 * (nu - 1)
        fv = w * (vmag_fixed if vmag_fixed > 0 else mag) / geom.dv + 0.5 * (nv - 1)
        iu, iv = np.floor(fu).astype(int), np.floor(fv).astype(int)
        ok = (iu >= 0) & (iu < nu - 1) & (iv >= 0) & (iv < nv - 1)
        iu, iv = np.clip(iu, 0, nu - 2), np.clip(iv, 0, nv - 2)
        au, av = fu - iu, fv - iv
        val = ((1 - av) * ((1 - au) * q[iv, iu] + au * q[iv, iu + 1])
               + av * ((1 - au) * q[iv + 1, iu] + au * q[iv + 1, iu + 1]))
        vol[k0:k1] += weight * np.where(ok, f * val, 0.0)


def fdk(sino: np.ndarray, angles_deg, geometry: Geometry3D, grid: Grid3D, window: str = "shepp-logan",
        mode: str = "fdk", backend: str = "auto") -> np.ndarray:
    """Reconstrucción de haz cónico de Feldkamp-Davis-Kress de ``sino (n_ángulos, n_v, n_u)`` (integrales de línea).

    Los ángulos deben cubrir 360° (órbita circular completa) con paso uniforme; con ``geometry.parallel``
    basta con 180°. Devuelve el mapa ``mu`` (1/m) en ``grid``. ``mode``:

    - ``"fdk"``: FDK completo (peso de coseno ``sdd/√(sdd² + u² + v²)`` y rayo oblicuo en v).
    - ``"stack"``: cada corte axial con la FBP de abanico (peso ``sdd/√(sdd² + u²)`` y fila fija por la
      magnificación del isocentro), es decir, el apilado de cortes de abanico sin corrección de cono.
    - ``"parallel"``: apilado de cortes de haz paralelo (exige ``geometry.parallel``).
    """
    if mode not in ("fdk", "stack", "parallel"):
        raise ValueError("mode debe ser 'fdk', 'stack' o 'parallel'")
    if (mode == "parallel") != geometry.parallel:
        raise ValueError("mode='parallel' exige Geometry3D(parallel=True) y viceversa")
    be = _resolve_backend(backend)
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    sino = np.asarray(sino, float)
    uu, vv = np.meshgrid(geometry.u, geometry.v)
    if geometry.parallel:
        g, ds = sino, geometry.du
    else:
        # Los pesos se calculan con las coordenadas del detector; el paso de la rampa es el del isocentro
        denom = geometry.sdd**2 + uu**2 + (vv**2 if mode == "fdk" else 0.0)
        g, ds = sino * (geometry.sdd / np.sqrt(denom))[None], geometry.du * geometry.sid / geometry.sdd
    q_all = ramp_filter(g, ds, window)
    vmag = geometry.sdd / geometry.sid if mode == "stack" else 0.0
    vol = np.zeros(grid.shape)
    weight = np.pi / angles.size
    if be == "numba":
        from ._numba_xray3d import backproject_fdk

        c = center(grid)
    for a, th in enumerate(angles):
        if be == "numba":
            d, u, v = geometry.view_vectors(th)
            backproject_fdk(np.ascontiguousarray(q_all[a]), vol, grid.x0, grid.y0, grid.z0, grid.dx, c, d, u, v,
                            geometry.sid, geometry.sdd, geometry.du, geometry.dv, not geometry.parallel, weight,
                            vmag)
        else:
            _backproject_numpy(q_all[a], vol, grid, geometry, th, weight, vmag)
    return vol


# --------------------------------------------------------------------------------------
# OS-SART 3D con variación total
# --------------------------------------------------------------------------------------
def tv_gradient_3d(vol: np.ndarray, eps: float) -> np.ndarray:
    """Gradiente de la variación total suavizada ``sum sqrt(|∇u|² + eps²)`` en 3D (diferencias hacia delante)."""
    g = [np.diff(vol, axis=a, append=np.take(vol, [-1], axis=a)) for a in range(3)]
    norm = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2 + eps**2)
    div = np.zeros_like(vol)
    for a in range(3):
        p = g[a] / norm
        div += np.diff(p, axis=a, prepend=np.zeros_like(np.take(p, [0], axis=a)))
    return -div


def os_sart3d(sino: np.ndarray, angles_deg, geometry: Geometry3D, grid: Grid3D, n_iter: int = 5, subset_size: int = 8,
              relax: float = 1.0, nonneg: bool = True, tv_weight: float = 0.0, tv_steps: int = 3,
              x0: np.ndarray | None = None, backend: str = "auto") -> np.ndarray:
    """OS-SART 3D con proyector de Siddon y su adjunto exacto, no negatividad y TV suavizada opcional.

    ``x += λ · Aᵀ[(p − A x)/r] / c`` en cada subconjunto entrelazado de vistas, con ``r = A·1`` (longitud
    del rayo en el volumen) y ``c = Aᵀ·1``. ``tv_weight`` > 0 aplica tras cada iteración ``tv_steps`` pasos
    de descenso de la TV suavizada (fracción del máximo por paso).
    """
    angles = np.asarray(angles_deg, float)
    nv = angles.size
    n_sub = max(int(np.ceil(nv / subset_size)), 1)
    subsets = [np.arange(s, nv, n_sub) for s in range(n_sub)]
    x = np.zeros(grid.shape) if x0 is None else x0.astype(float).copy()
    ones = np.ones(grid.shape)
    row = project(ones, grid, geometry, angles, backend=backend)
    col = [backproject(np.ones((idx.size, geometry.n_v, geometry.n_u)), grid, geometry, angles[idx], backend=backend)
           for idx in subsets]
    for _ in range(n_iter):
        for idx, c_s in zip(subsets, col):
            resid = sino[idx] - project(x, grid, geometry, angles[idx], backend=backend)
            ratio = np.where(row[idx] > 1e-9, resid / np.maximum(row[idx], 1e-9), 0.0)
            x += relax * backproject(ratio, grid, geometry, angles[idx], backend=backend) / np.maximum(c_s, 1e-9)
            if nonneg:
                np.maximum(x, 0.0, out=x)
        if tv_weight > 0:
            scale = max(float(x.max()), 1e-12)
            for _ in range(tv_steps):
                x -= tv_weight * scale * tv_gradient_3d(x / scale, 0.02)
                if nonneg:
                    np.maximum(x, 0.0, out=x)
    return x


# --------------------------------------------------------------------------------------
# TAC 3D completo
# --------------------------------------------------------------------------------------
@dataclass
class CTVolume:
    mu: np.ndarray  # 1/m
    hu: np.ndarray
    projections: Projections3D


def ct_scan3d(phantom: Phantom3D, n_angles: int = 360, geometry: Geometry3D | None = None,
              spectrum: XraySpectrum | None = None, photons: float | None = 1e5,
              beam_hardening_correction: bool = True, method: str = "fdk",
              detector: FlatPanel | DetectorModel | None = None,
              rng: np.random.Generator | None = None, scatter=None, backend: str = "auto",
              sid: float | None = None, sdd: float | None = None, **recon_kw) -> CTVolume:
    """TAC de haz cónico del ``Phantom3D`` en la malla del fantoma (órbita circular de 360° alrededor de x).

    ``method``: ``"fdk"`` / ``"stack"`` (``fdk(mode=...)``), ``"sart"`` (``os_sart3d``; ``recon_kw``:
    ``n_iter``, ``subset_size``, ``tv_weight``...) o ``"parallel"`` (haz paralelo, apilado de cortes).
    Como ``xray.ct_scan``, linealiza para agua (endurecimiento del haz) si ``beam_hardening_correction``.
    Por defecto 120 kVp y 6 mm Al y una geometría de ``sid`` = 3 × diagonal transversal y ``sdd`` = 1.4 ``sid``
    (cono de pocos grados).
    """
    spectrum = spectrum or tube_spectrum(120.0, 6.0)
    parallel = method == "parallel"
    if geometry is None:
        g = phantom.grid
        radius = 0.5 * float(np.hypot(g.ny, g.nz)) * g.dx
        sid = sid or 4.0 * radius
        geometry = Geometry3D.covering(g, sid, sdd or 1.4 * sid, parallel=parallel)
    if method not in ("fdk", "stack", "sart", "parallel"):
        raise ValueError("method debe ser 'fdk', 'stack', 'sart' o 'parallel'")
    angles = np.arange(n_angles) * (180.0 if parallel else 360.0) / n_angles
    proj = acquire3d(phantom, angles, geometry, spectrum, photons, detector, rng, scatter, backend=backend)
    e_ref = float(np.clip(spectrum.mean_energy, xd.ENERGIES_KEV[0], xd.ENERGIES_KEV[-1]))
    mu_ref = float(xd.linear_attenuation("water", e_ref))
    sino = proj.line_integrals
    if beam_hardening_correction and spectrum.energies_kev.size > 1:
        # La curva de linealización es la del espectro *detectado* (eficiencia y ponderación en energía)
        sig_w, _ = _as_panel(detector).signal_weights(spectrum)
        sino = water_linearize(sino, XraySpectrum(spectrum.energies_kev, sig_w), mu_ref)
        mu_w = mu_ref
    else:
        mu_w = effective_water_mu(spectrum, 0.05) if spectrum.energies_kev.size > 1 else mu_ref
    if method == "sart":
        mu = os_sart3d(sino, angles, geometry, phantom.grid, backend=backend, **recon_kw)
    else:
        mu = fdk(sino, angles, geometry, phantom.grid, mode=method, backend=backend, **recon_kw)
    return CTVolume(mu, xd.hounsfield(mu, mu_w, 0.0), proj)


def mu_map(phantom: Phantom3D, energy_kev: float) -> np.ndarray:
    """Mapa 3D de ``mu`` (1/m) del fantoma a una energía (para usar con ``project`` / ``backproject``)."""
    return phantom.property_map(lambda k: float(xd.linear_attenuation(k, energy_kev)), dtype=np.float64)


__all__ = ["Geometry3D", "FlatPanel", "Projections3D", "CTVolume", "Ellipsoid", "project", "backproject",
           "acquire3d", "radiograph", "fdk", "os_sart3d", "ct_scan3d", "ellipsoid_volume", "shepp_logan_3d",
           "ray_point_distance", "mu_map", "center"]
