"""EIT: soluciones analíticas del disco, reconstrucción iterativa (Gauss-Newton) y métricas de calidad (GREIT).

Este módulo complementa a ``quasistatic`` (problema directo en diferencias
finitas, CEM y jacobiano):

- ``disc_point_voltage`` y ``disc_cem_series``: soluciones de referencia del
  disco homogéneo (electrodos puntuales ideales y modelo completo de electrodo),
  con las que se valida el problema directo.
- ``EITModel`` + ``gauss_newton``: reconstrucción absoluta o de diferencias con
  jacobiano recalculado en cada iteración y regularización Tikhonov,
  Laplaciana, NOSER o de variación total (difusividad retardada).
- ``lcurve_regularization`` / ``discrepancy_regularization``: elección del
  parámetro de regularización.
- ``greit_metrics``: error de posición, resolución, deformación, amplitud y
  *ringing* de Adler et al. (2009).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..phantoms.phantom import Phantom
from .quasistatic import (
    Electrodes, QuasiStaticSolver, sensitivity_matrix, simulate_eit,
)


# --------------------------------------------------------------------------
# Soluciones analíticas del disco homogéneo
# --------------------------------------------------------------------------
def disc_point_voltage(theta_in: float, theta_out: float, theta_a: float, theta_b: float, sigma: complex,
                       current: float = 1.0) -> complex:
    """Tensión ``V(a) - V(b)`` en el borde de un disco homogéneo con dos electrodos puntuales ideales.

    La corriente ``current`` (A por metro de profundidad) entra en ``theta_in``
    y sale en ``theta_out``. La función de Green de Neumann del disco, con la
    fuente en el borde, es ``u = -(I / (pi sigma)) ln|z - z1|`` + const; en el
    borde ``|e^{it} - e^{is}| = 2 |sin((t - s)/2)|``. No depende del radio.
    """
    def u(t):
        return -(current / (np.pi * sigma)) * (np.log(abs(np.sin((t - theta_in) / 2.0)))
                                               - np.log(abs(np.sin((t - theta_out) / 2.0))))
    return u(theta_a) - u(theta_b)


def disc_cem_series(radius: float, sigma: complex, z_c: float | complex, electrode_arcs: list[tuple[float, float]],
                    currents: np.ndarray, n_modes: int = 150, n_quad: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Solución en serie del CEM en un disco homogéneo: ``(U_l, coeficientes del potencial de borde)``.

    Derivación (Galerkin en la base trigonométrica; ``sigma`` conductividad del
    disco, ``z_c`` en ohm·m², radio ``R``, ``ds = R dt``):

    El potencial armónico en el disco es ``u = a0 + sum_n r^n (a_n cos nt + b_n sin nt)``, de modo
    que el flujo normal en el borde es ``sigma du/dr = sigma sum_n n R^{n-1} (a_n cos nt + b_n sin nt)``.
    El CEM impone ``sigma du/dr = (U_l - u)/z_c`` bajo el electrodo ``l`` (arco ``[t1_l, t2_l]``) y 0 fuera,
    más ``(1/z_c) int_{E_l} (U_l - u) ds = I_l``. Se multiplica la condición de flujo por cada
    función de la base ``phi_k`` y se integra sobre el borde: por ortogonalidad

        sigma n R^n pi c_k = (R / z_c) sum_l [ U_l int_{E_l} phi_k dt - sum_j c_j int_{E_l} phi_k phi_j dt ]

    (para ``k = 0`` el lado izquierdo es 0). Es un sistema lineal en ``(c_j, U_l)`` al que se añaden las
    ``L`` ecuaciones de corriente por electrodo y el gauge ``sum U_l = 0``. Las integrales de las bases sobre
    cada arco se hacen con cuadratura de Gauss-Legendre por tramos (exacta para polinomios trigonométricos de
    grado < ``n_quad`` por tramo; se subdivide cada arco). Se resuelve el sistema cuadrado resultante. Las
    esquinas del electrodo tienen una singularidad integrable, de modo que la serie truncada converge
    sólo como ``1/n_modes``.

    ``currents`` suma 0. Devuelve ``U`` (L,) y ``c`` (1 + 2 n_modes,) con el orden ``[a0, a_1..a_N, b_1..b_N]``.
    """
    n_el = len(electrode_arcs)
    nm = n_modes
    # Base: [1, cos t..., sin t...] evaluada en nodos de cuadratura de cada arco
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    n_pieces = max(1, nm // n_quad + 1)
    rows = 1 + 2 * nm
    # integrales A_l[k, j] = int_{E_l} phi_k phi_j dt  y  B_l[k] = int_{E_l} phi_k dt
    a_l = np.zeros((n_el, rows, rows))
    b_l = np.zeros((n_el, rows))
    for k, (t1, t2) in enumerate(electrode_arcs):
        edges = np.linspace(t1, t2, n_pieces + 1)
        tq = ((edges[:-1] + edges[1:])[:, None] / 2.0 + (edges[1:] - edges[:-1])[:, None] / 2.0 * xg).ravel()
        wq = ((edges[1:] - edges[:-1])[:, None] / 2.0 * wg).ravel()
        n = np.arange(1, nm + 1)
        phi = np.vstack([np.ones_like(tq), np.cos(np.outer(n, tq)), np.sin(np.outer(n, tq))])
        b_l[k] = phi @ wq
        a_l[k] = (phi * wq) @ phi.T
    n = np.arange(1, nm + 1)
    # Se trabaja con los coeficientes del potencial *en el borde* (a_n R^n): evita el desbordamiento de R^n
    diag_lhs = np.concatenate([[0.0], np.tile(sigma * n * np.pi, 2)])
    # Incógnitas: c (rows) + U (n_el). La fila k = 0 de Galerkin (suma de corrientes = 0) es combinación de las
    # ecuaciones de corriente por electrodo: se omite y con c_0 = 0 queda un sistema cuadrado.
    mat = np.zeros((rows + n_el, rows + n_el), dtype=complex)
    rhs = np.zeros(rows + n_el, dtype=complex)
    mat[:rows, :rows] = np.diag(diag_lhs) + (radius / z_c) * a_l.sum(axis=0)
    for k in range(n_el):
        mat[:rows, rows + k] = -(radius / z_c) * b_l[k]
        mat[rows + k, :rows] = (radius / z_c) * b_l[k]
        mat[rows + k, rows + k] = -(radius / z_c) * (electrode_arcs[k][1] - electrode_arcs[k][0])
        rhs[rows + k] = -currents[k]
    # Se fija c_0 = 0 (media nula del potencial de borde): es el gauge; al final se refiere U a sum U = 0
    keep = np.arange(rows + n_el) != 0
    mat2 = mat[np.ix_(keep, keep)]
    rhs2 = rhs[keep]
    sol = np.zeros(rows + n_el, dtype=complex)
    sol[keep] = np.linalg.solve(mat2, rhs2)
    u = sol[rows:]
    return u - u.mean(), sol[:rows]


# --------------------------------------------------------------------------
# Modelo directo parametrizado y utilidades
# --------------------------------------------------------------------------
def coarsen_phantom(phantom: Phantom, k: int) -> Phantom:
    """Fantoma submuestreado ``k`` veces (cada ``k``-ésima celda): la malla de reconstrucción.

    Sirve para no cometer el "crimen inverso": los datos se generan en la
    malla fina y se invierten en una más gruesa, con los electrodos y la piel
    en escalera distintos. El centro de la celda ``[j, i]`` gruesa coincide con
    el de la celda ``[k j, k i]`` fina.
    """
    from ..core.grid import Grid2D

    g = phantom.grid
    sub = (slice(None, None, k), slice(None, None, k))
    ng = Grid2D(-(-g.nx // k), -(-g.nz // k), g.dx * k, g.x0, g.z0)
    return Phantom(ng, phantom.labels[sub].copy(), {key: m[sub].copy() for key, m in phantom.masks.items()},
                   None if phantom.heterogeneity is None else phantom.heterogeneity[sub].copy(), dict(phantom.meta),
                   None if phantom.porosity is None else phantom.porosity[sub].copy(), phantom.pore_fill)


def add_noise(voltages: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido gaussiano complejo con ``SNR = 20 log10(rms|V| / sigma)`` (``sigma`` total, repartida en Re e Im)."""
    sigma = np.sqrt(np.mean(np.abs(voltages) ** 2)) * 10.0 ** (-snr_db / 20.0)
    noise = rng.standard_normal(voltages.shape) + 1j * rng.standard_normal(voltages.shape)
    return voltages + sigma / np.sqrt(2.0) * noise


@dataclass
class EITModel:
    """Modelo directo para reconstruir: malla, electrodos, contacto y parametrización por bloques.

    Los parámetros ``p`` (uno por celda gruesa con tejido, de ``factor`` x
    ``factor`` celdas finas) son el logaritmo del factor que multiplica la
    admitividad de fondo ``y0``: ``y = y0 exp(p)``. Así la admitividad
    permanece positiva y el cambio relativo ``exp(p) - 1`` es la imagen que se
    compara con la verdad. ``p = 0`` es el fondo (por defecto el fantoma tal
    cual; para imagen absoluta puede ser un modelo homogéneo).
    """

    phantom: Phantom
    f: float
    electrodes: Electrodes
    factor: int = 3
    current: float = 1.0
    contact_impedance: float | complex | np.ndarray | None = None
    y0: np.ndarray | None = None  # mapa de admitividad de fondo; None = el del fantoma
    tol: float = 1e-9

    def __post_init__(self):
        base = QuasiStaticSolver(self.phantom, self.f, admittivity_map=self.y0)
        self.y0 = base.y.copy()
        self.tissue = base.tissue
        g = self.phantom.grid
        self._coarse_shape = (-(-g.nz // self.factor), -(-g.nx // self.factor))
        nzc, nxc = self._coarse_shape
        pad = np.zeros((nzc * self.factor, nxc * self.factor))
        pad[:g.nz, :g.nx] = self.tissue
        self.active = pad.reshape(nzc, self.factor, nxc, self.factor).sum(axis=(1, 3)) > 0

    @property
    def n_params(self) -> int:
        return int(self.active.sum())

    def to_fine(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Vector de parámetros -> mapa en la malla fina (bloques constantes)."""
        img = np.full(self._coarse_shape, fill, dtype=values.dtype)
        img[self.active] = values
        big = np.kron(img, np.ones((self.factor, self.factor)))
        g = self.phantom.grid
        return big[:g.nz, :g.nx]

    def forward(self, p: np.ndarray, jacobian: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """Tensiones (y jacobiano ``dV/dp`` complejo si se pide) con la admitividad ``y0 exp(p)``."""
        y = self.y0 * np.exp(self.to_fine(np.asarray(p, dtype=float)))
        solver = QuasiStaticSolver(self.phantom, self.f, admittivity_map=y)
        eit = simulate_eit(solver, self.electrodes, self.current, tol=self.tol,
                           contact_impedance=self.contact_impedance)
        jac = sensitivity_matrix(eit, self.factor).matrix if jacobian else None
        return eit.voltages, jac

    def relative_change_image(self, p: np.ndarray) -> np.ndarray:
        """Imagen ``exp(p) - 1`` (cambio relativo de admitividad) en la malla fina."""
        return np.where(self.tissue, self.to_fine(np.exp(np.asarray(p, dtype=float)) - 1.0), 0.0)


def _edge_operator(active: np.ndarray) -> np.ndarray:
    """Matriz de diferencias ``D`` (aristas x parámetros) entre celdas gruesas activas vecinas."""
    idx = -np.ones(active.shape, dtype=int)
    idx[active] = np.arange(int(active.sum()))
    rows = []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        ok = (a >= 0) & (b >= 0)
        rows.append(np.stack([a[ok], b[ok]], axis=1))
    edges = np.vstack(rows)
    d = np.zeros((edges.shape[0], int(active.sum())))
    d[np.arange(edges.shape[0]), edges[:, 0]] = -1.0
    d[np.arange(edges.shape[0]), edges[:, 1]] = 1.0
    return d


def _stack(jac: np.ndarray) -> np.ndarray:
    return np.vstack([jac.real, jac.imag])


def _split(v: np.ndarray) -> np.ndarray:
    return np.concatenate([v.real, v.imag])


REGULARIZERS = ("tikhonov", "laplaciano", "noser", "tv")


class _Regularizer:
    """Penalización ``pen(p - p_ref)`` con su aproximación cuadrática ``R(p)`` (Hessiano aproximado).

    ``tikhonov``: ``|p|²``; ``laplaciano``: ``|L p|²`` con ``L`` el Laplaciano
    de grafo de las celdas activas (EIDORS ``laplace``); ``noser``: ``p' diag(J'J) p``
    (Cheney et al. 1990) con el jacobiano de la primera iteración; ``tv``:
    ``sum_e sqrt((D p)_e² + beta²)`` por difusividad retardada (Vogel). Todas
    se escalan una vez para que ``lam = 1`` iguale la media de la diagonal de
    ``J'J``: así ``lam`` es adimensional y comparable entre regularizadores.
    """

    def __init__(self, kind: str, active: np.ndarray, hess: np.ndarray, tv_beta: float = 1e-2):
        if kind not in REGULARIZERS:
            raise ValueError(f"regularizador desconocido: {kind} (opciones: {REGULARIZERS})")
        self.kind, self.beta = kind, tv_beta
        n = hess.shape[0]
        self.d = _edge_operator(active) if kind in ("laplaciano", "tv") else None
        if kind == "tikhonov":
            base = np.eye(n)
        elif kind == "noser":
            base = np.diag(np.diag(hess))
        elif kind == "laplaciano":
            lap = self.d.T @ self.d
            base = lap.T @ lap
        else:
            base = self.d.T @ self.d
        self._base = base
        self.scale = np.mean(np.diag(hess)) / np.mean(np.diag(base))
        if kind == "tv":
            self.scale *= tv_beta  # con p = 0 el Hessiano de TV coincide con el de un suavizado de primer orden

    def matrix(self, dp: np.ndarray) -> np.ndarray:
        if self.kind != "tv":
            return self.scale * self._base
        w = 1.0 / np.sqrt((self.d @ dp) ** 2 + self.beta**2)
        return self.scale * (self.d.T * w) @ self.d

    def penalty(self, dp: np.ndarray) -> float:
        if self.kind == "tv":
            return float(self.scale * np.sum(np.sqrt((self.d @ dp) ** 2 + self.beta**2)))
        return float(dp @ (self.scale * self._base) @ dp)


def lcurve_regularization(jac: np.ndarray, resid: np.ndarray, reg: np.ndarray,
                          lams: np.ndarray | None = None) -> tuple[float, dict]:
    """Parámetro de la curva L: el de máxima curvatura de ``(log |r - J d|, log sqrt(d' R d))``.

    ``jac`` y ``resid`` reales (ya apilados); ``reg`` es la matriz ``R``. La
    curvatura se calcula por diferencias finitas sobre ``log lam``. Es el
    criterio de Hansen; en datos de EIT con ruido la esquina es suave, por lo
    que se devuelve también la curva para poder inspeccionarla.
    """
    lams = np.logspace(-8, 2, 41) if lams is None else np.asarray(lams, dtype=float)
    hess = jac.T @ jac
    rhs = jac.T @ resid
    rn, sn = [], []
    for lam in lams:
        d = np.linalg.solve(hess + lam * reg, rhs)
        rn.append(np.linalg.norm(resid - jac @ d))
        sn.append(np.sqrt(max(d @ reg @ d, 1e-300)))
    x, y = np.log(np.array(rn)), np.log(np.array(sn))
    t = np.log(lams)
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
    curv = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-300)
    k = int(np.argmax(np.abs(curv[1:-1]))) + 1
    return float(lams[k]), {"lams": lams, "residual": np.array(rn), "seminorm": np.array(sn), "curvature": curv}


def discrepancy_regularization(jac: np.ndarray, resid: np.ndarray, reg: np.ndarray, noise_norm: float,
                               tau: float = 1.0, lams: np.ndarray | None = None) -> float:
    """Principio de discrepancia de Morozov: el mayor ``lam`` con ``|r - J d| <= tau · noise_norm``.

    ``noise_norm`` es la norma esperada del ruido en el vector apilado (Re, Im)
    (``sigma sqrt(n_datos_reales)``). Si ni el menor ``lam`` alcanza el nivel de
    ruido se devuelve ése (el ajuste no puede ser mejor).
    """
    lams = np.logspace(-10, 2, 49) if lams is None else np.asarray(lams, dtype=float)
    hess = jac.T @ jac
    rhs = jac.T @ resid
    best = float(lams[0])
    for lam in lams:
        d = np.linalg.solve(hess + lam * reg, rhs)
        if np.linalg.norm(resid - jac @ d) <= tau * noise_norm:
            best = float(lam)
        else:
            break
    return best


@dataclass
class GNResult:
    p: np.ndarray  # parámetros finales (log del factor de admitividad)
    cost: list[float]  # coste (residuo² + penalización) tras cada iteración, el primero es el inicial
    residual: list[float]  # |r| ponderado tras cada iteración
    lam: list[float]  # parámetro de regularización usado en cada iteración
    iterations: int
    voltages: np.ndarray  # tensiones del modelo con ``p``
    lcurve: dict | None = None
    model: EITModel | None = None

    def image(self) -> np.ndarray:
        """Cambio relativo de admitividad ``exp(p) - 1`` en la malla fina."""
        return self.model.relative_change_image(self.p)


def gauss_newton(model: EITModel, data: np.ndarray, reg: str = "noser", lam: float | str = 1e-2,
                 n_iter: int = 8, p0: np.ndarray | None = None, data_ref: np.ndarray | None = None,
                 noise_std: float | None = None, tv_beta: float = 1e-2, tau: float = 1.05,
                 tol: float = 1e-3, p_ref: np.ndarray | None = None) -> GNResult:
    """Reconstrucción por Gauss-Newton regularizado con el jacobiano recalculado en cada iteración.

    Minimiza ``|W (data - F(p))|² + lam · pen(p - p_ref)`` con ``F`` el modelo
    directo (``EITModel.forward``), ``W = 1/noise_std`` (por defecto
    ``1/rms|data|``, de modo que ``lam`` sea adimensional) y los datos
    complejos apilados en parte real e imaginaria. Cada paso resuelve
    ``(J'J + lam R) dp = J' r - lam R (p - p_ref)`` y recorta el paso a la mitad
    (hasta 5 veces) mientras no baje el coste.

    - Imagen absoluta: ``data`` son las tensiones medidas; ``p0 = 0`` es el fondo del modelo.
    - Imagen de diferencias: pasar ``data_ref`` (la medida de referencia); el objetivo pasa a ser
      ``F(p0) + (data - data_ref)``, que cancela buena parte del error de modelo (posición de
      electrodos, geometría) igual que en EIDORS.
    - ``lam`` numérico, ``"lcurve"`` (esquina de la curva L del primer paso, fijo después) o
      ``"discrepancy"`` (Morozov en cada iteración; requiere ``noise_std``; para si ``|r| <= tau · ruido``).
    """
    p = np.zeros(model.n_params) if p0 is None else np.asarray(p0, dtype=float).copy()
    p_ref = np.zeros_like(p) if p_ref is None else np.asarray(p_ref, dtype=float)
    v, jac = model.forward(p, jacobian=True)
    target = np.asarray(data, dtype=complex) if data_ref is None else v + (np.asarray(data) - np.asarray(data_ref))
    w = 1.0 / (noise_std if noise_std is not None else np.sqrt(np.mean(np.abs(target) ** 2)))
    jr = w * _stack(jac)
    regularizer = _Regularizer(reg, model.active, jr.T @ jr, tv_beta)
    noise_norm = None if noise_std is None else np.sqrt(2.0 * target.size)  # |W n| con W = 1/sigma
    lam_cur = float(lam) if not isinstance(lam, str) else 1e-2
    lcurve = None

    def cost_of(vv, pp):
        r_ = w * _split(target - vv)
        return float(r_ @ r_ + lam_cur * regularizer.penalty(pp - p_ref)), float(np.linalg.norm(r_))

    cost, resid, lams = [], [], []
    for it in range(n_iter + 1):
        r = w * _split(target - v)
        rmat = regularizer.matrix(p - p_ref)
        if isinstance(lam, str) and (it == 0 or lam == "discrepancy"):
            if lam == "lcurve":
                lam_cur, lcurve = lcurve_regularization(jr, r, rmat)
            elif lam == "discrepancy":
                if noise_norm is None:
                    raise ValueError("lam='discrepancy' necesita noise_std")
                lam_cur = discrepancy_regularization(jr, r, rmat, noise_norm, tau)
            else:
                raise ValueError(f"lam desconocido: {lam}")
        c, rn = cost_of(v, p)
        cost.append(c)
        resid.append(rn)
        lams.append(lam_cur)
        if it == n_iter or (lam == "discrepancy" and rn <= tau * noise_norm):
            break
        step = np.linalg.solve(jr.T @ jr + lam_cur * rmat, jr.T @ r - lam_cur * rmat @ (p - p_ref))
        alpha, accepted = 1.0, False
        for _ in range(6):
            p_new = p + alpha * step
            v_new, _ = model.forward(p_new)
            c_new, _ = cost_of(v_new, p_new)
            if c_new < c:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        p = p_new
        v, jac = model.forward(p, jacobian=True)
        jr = w * _stack(jac)
        if (c - c_new) < tol * c:
            c, rn = cost_of(v, p)
            cost.append(c)
            resid.append(rn)
            lams.append(lam_cur)
            break
    return GNResult(p, cost, resid, lams, len(cost) - 1, v, lcurve, model)


# --------------------------------------------------------------------------
# Métricas de calidad de imagen (GREIT, Adler et al. 2009)
# --------------------------------------------------------------------------
@dataclass
class ImageMetrics:
    position_error: float  # m: distancia entre el centro de gravedad de la imagen y el del objetivo
    position_error_rel: float  # relativo al radio del dominio
    resolution: float  # RES = sqrt(A_Q / A_dominio)
    shape_deformation: float  # SD: fracción de Q fuera del círculo de igual área centrado en el CG
    amplitude: float  # AR: integral de la imagen / integral del objetivo
    ringing: float  # RNG: |suma de la parte negativa fuera de Q| / suma en Q


def greit_metrics(image: np.ndarray, truth: np.ndarray, domain: np.ndarray, grid) -> ImageMetrics:
    """Métricas de GREIT para un objetivo único, sobre la malla fina.

    ``image`` es el cambio relativo reconstruido, ``truth`` el verdadero (cero fuera del
    objetivo), ``domain`` el tejido y ``grid`` la malla (``dx``, ``meshgrid``). La
    imagen se orienta con el signo del contraste del objetivo para que "positivo" sea
    "en la dirección del objetivo". ``Q`` es el conjunto de píxeles con al menos el 25 %
    del máximo (la región de un cuarto de amplitud de GREIT).
    """
    area = grid.dx**2
    x, z = grid.meshgrid()
    sign = np.sign(truth[domain].sum()) or 1.0
    img = np.where(domain, sign * image, 0.0)
    true = np.where(domain, sign * truth, 0.0)
    q = (img >= 0.25 * img.max()) & domain
    wq = img[q]
    cg = np.array([(x[q] * wq).sum(), (z[q] * wq).sum()]) / wq.sum()
    obj = true > 0
    c_true = np.array([(x[obj] * true[obj]).sum(), (z[obj] * true[obj]).sum()]) / true[obj].sum()
    pe = float(np.hypot(*(cg - c_true)))
    r_dom = np.sqrt(domain.sum() * area / np.pi)
    a_q = q.sum() * area
    res = float(np.sqrt(a_q / (domain.sum() * area)))
    circle = np.hypot(x - cg[0], z - cg[1]) <= np.sqrt(a_q / np.pi)
    sd = float((q & ~circle).sum() * area / a_q)
    ar = float(img[domain].sum() / true[domain].sum())
    neg = img[domain & ~q & (img < 0)].sum()
    return ImageMetrics(pe, pe / r_dom, res, sd, ar, float(abs(neg) / wq.sum()))
