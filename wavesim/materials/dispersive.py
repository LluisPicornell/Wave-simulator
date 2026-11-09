"""Modelo de Debye multipolo de los tejidos para simulaciones de banda ancha.

El FDTD de banda estrecha (``em.fdtd_tm``) usa ``eps'`` y ``sigma`` a una sola
frecuencia. Para propagar un pulso hace falta una permitividad que dependa de
la frecuencia y que el FDTD pueda integrar en el tiempo. El Cole-Cole de
Gabriel no sirve directamente (su exponente fraccionario no tiene ecuación
diferencial auxiliar sencilla), así que en una banda ``[f_min, f_max]`` se
ajusta a

    eps(w) = eps_inf + sum_k d_eps_k / (1 + j w tau_k) + sigma / (j w eps0)

con ``eps_inf >= 1``, ``d_eps_k >= 0`` y ``sigma >= 0`` (convención
``e^{+jwt}`` de ``materials.dielectric``). Con esas restricciones el medio es
pasivo y causal y el FDTD con ecuaciones diferenciales auxiliares (ADE) es
estable con el mismo paso de tiempo que en el vacío.

Los tiempos de relajación ``tau_k`` son **comunes** a todos los tejidos de la
malla: sólo cambian por celda las amplitudes. Así el FDTD guarda un mapa por
polo en vez de uno por tejido y polo, y cualquier combinación lineal de
tejidos (el promedio sub-celda de las interfaces, la interpolación entre
niveles de porosidad) sigue siendo un modelo de Debye con los mismos polos y
amplitudes no negativas. Con los ``tau_k`` fijos el ajuste es lineal en
``(eps_inf - 1, sigma, d_eps_k)`` y se resuelve con mínimos cuadrados no
negativos (NNLS, Lawson y Hanson); los ``tau_k`` se optimizan para todos los
tejidos a la vez con proyección de variables (Levenberg-Marquardt sobre
``log tau``, con el NNLS dentro). El residuo es el error relativo de ``eps'``
y de la conductividad efectiva ``sigma_eff = w eps0 eps''``, que incluye la
conductividad iónica.

Celdas porosas (``Phantom.porosity``): la mezcla de Maxwell Garnett de
``materials.porous`` no es un Debye, así que se ajusta en una tabla de
porosidades (paso 0.01) y cada celda interpola linealmente las amplitudes
entre los dos niveles vecinos (la mezcla es suave en la porosidad: el error
de interpolación es de ~1e-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import dielectric, porous
from .tissues import tissue_keys

EPS0 = dielectric.EPS0

#: Puntos (log-espaciados) de la banda en los que se ajusta el modelo.
FIT_POINTS = 60

#: Error relativo máximo (``eps'`` y ``sigma``) con el que se elige el número de polos.
DEFAULT_TOLERANCE = 0.01

#: Paso de la tabla de porosidades de las celdas porosas.
POROSITY_STEP = 0.01


# --------------------------------------------------------------------------
# Mínimos cuadrados no negativos
# --------------------------------------------------------------------------
def nnls(a: np.ndarray, b: np.ndarray, max_iter: int | None = None) -> np.ndarray:
    """``argmin ||a x - b||`` con ``x >= 0`` (algoritmo de conjunto activo de Lawson y Hanson, 1974).

    NumPy no lo trae y SciPy no es una dependencia. Para los problemas de aquí
    (decenas de columnas) converge en unas pocas iteraciones.
    """
    m, n = a.shape
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    max_iter = max_iter or 3 * n + 10
    w = a.T @ (b - a @ x)
    tol = 1e-12 * max(1.0, float(np.abs(a).max()) * float(np.abs(b).max()))
    for _ in range(max_iter):
        if passive.all() or w[~passive].max() <= tol:
            break
        passive[np.argmax(np.where(passive, -np.inf, w))] = True
        while True:
            z = np.zeros(n)
            z[passive] = np.linalg.lstsq(a[:, passive], b, rcond=None)[0]
            if (z[passive] > 0).all():
                x = z
                break
            # Retroceder hasta el primer coeficiente que se hace cero y sacarlo del conjunto
            neg = passive & (z <= 0)
            step = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + step * (z - x)
            passive &= x > 1e-15 * max(1.0, float(np.abs(x).max()))
            x[~passive] = 0.0
            if not passive.any():
                break
        w = a.T @ (b - a @ x)
    return x


# --------------------------------------------------------------------------
# Medio de Debye (escalar o por celda)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DebyeMedium:
    """``eps_inf + sum_k d_eps_k / (1 + j w tau_k) + sigma / (j w eps0)``.

    ``eps_inf`` y ``sigma`` son escalares o mapas ``(nz, nx)``; ``delta_eps``
    tiene forma ``(K,)`` o ``(K, nz, nx)``; ``tau`` (s) es común, forma ``(K,)``.
    """

    eps_inf: np.ndarray | float
    sigma: np.ndarray | float
    delta_eps: np.ndarray
    tau: np.ndarray

    @property
    def n_poles(self) -> int:
        return len(self.tau)

    def complex_permittivity(self, f_hz) -> np.ndarray:
        """Permitividad relativa compleja. Forma ``f.shape + forma del medio``."""
        f = np.asarray(f_hz, dtype=float)
        w = 2.0 * np.pi * f.reshape(f.shape + (1,) * np.ndim(self.eps_inf))
        eps = np.asarray(self.eps_inf, dtype=complex) + self.sigma / (1j * w * EPS0)
        for k, tau in enumerate(self.tau):
            eps = eps + self.delta_eps[k] / (1.0 + 1j * w * tau)
        return eps

    def conductivity(self, f_hz) -> np.ndarray:
        """Conductividad efectiva ``w eps0 eps''`` (S/m), con la iónica incluida."""
        f = np.asarray(f_hz, dtype=float)
        w = 2.0 * np.pi * f.reshape(f.shape + (1,) * np.ndim(self.eps_inf))
        return -self.complex_permittivity(f).imag * w * EPS0

    def static_permittivity(self) -> np.ndarray | float:
        """``eps_inf + sum d_eps``: permitividad real a frecuencia cero (sin la conductividad)."""
        return self.eps_inf + np.sum(self.delta_eps, axis=0)


# --------------------------------------------------------------------------
# Ajuste
# --------------------------------------------------------------------------
def fit_frequencies(band: tuple[float, float], n: int = FIT_POINTS) -> np.ndarray:
    return np.geomspace(band[0], band[1], n)


def _target_parts(eps_target: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = 2.0 * np.pi * f
    return np.real(eps_target), -np.imag(eps_target) * w * EPS0


def _design(f: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columnas ``[eps_inf - 1, sigma, d_eps_k]`` de ``eps'`` y de ``sigma_eff``."""
    w = 2.0 * np.pi * f
    wt = w[:, None] * tau[None, :]
    n = f.size
    a_eps = np.hstack([np.ones((n, 1)), np.zeros((n, 1)), 1.0 / (1.0 + wt**2)])
    a_sig = np.hstack([np.zeros((n, 1)), np.ones((n, 1)), EPS0 * w[:, None] * wt / (1.0 + wt**2)])
    return a_eps, a_sig


def _solve_amplitudes(eps_t: np.ndarray, sig_t: np.ndarray, a_eps: np.ndarray, a_sig: np.ndarray):
    """NNLS en error relativo. Devuelve ``(x, residuo)`` con ``x = [eps_inf - 1, sigma, d_eps...]``."""
    rows_e = a_eps / eps_t[:, None]
    b_e = (eps_t - 1.0) / eps_t
    lossy = sig_t > 0
    rows_s = a_sig[lossy] / sig_t[lossy, None]
    b_s = np.ones(int(lossy.sum()))
    a = np.vstack([rows_e, rows_s])
    b = np.concatenate([b_e, b_s])
    # Escalar columnas para que el NNLS no dependa de las unidades (S/m frente a adimensional)
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1.0
    x = nnls(a / scale, b) / scale
    return x, a @ x - b


def fit_amplitudes(eps_target: np.ndarray, f: np.ndarray, tau: np.ndarray) -> DebyeMedium:
    """Amplitudes no negativas para los polos ``tau`` que mejor reproducen ``eps_target(f)``."""
    eps_t, sig_t = _target_parts(np.asarray(eps_target), np.asarray(f, dtype=float))
    x, _ = _solve_amplitudes(eps_t, sig_t, *_design(f, tau))
    return DebyeMedium(1.0 + x[0], x[1], x[2:].copy(), np.asarray(tau, dtype=float))


def optimize_poles(targets: list[np.ndarray], f: np.ndarray, n_poles: int, max_iter: int = 60) -> np.ndarray:
    """Tiempos de relajación comunes que minimizan el error cuadrático relativo de todos los ``targets``.

    Proyección de variables: para unos ``tau`` dados, las amplitudes salen del
    NNLS y el residuo sólo depende de ``u = log tau``, que se optimiza con
    Levenberg-Marquardt (jacobiano por diferencias finitas). Se parte de polos
    log-espaciados entre ``f_min/3`` y ``3 f_max``: los polos de Cole-Cole
    fuera de la banda también curvan ``eps'`` y ``sigma`` dentro de ella.
    """
    f = np.asarray(f, dtype=float)
    parts = [_target_parts(np.asarray(t), f) for t in targets]
    parts = [(e, s) for e, s in parts if not (np.allclose(e, 1.0) and np.allclose(s, 0.0))]
    u = np.log(1.0 / (2.0 * np.pi * np.geomspace(f[0] / 3.0, f[-1] * 3.0, n_poles)))
    if not parts:
        return np.sort(np.exp(u))[::-1]

    def residual(u_):
        a_eps, a_sig = _design(f, np.exp(u_))
        return np.concatenate([_solve_amplitudes(e, s, a_eps, a_sig)[1] for e, s in parts])

    # Los polos no pueden alejarse mucho de la banda: fuera no los restringe el ajuste
    lo, hi = np.log(1.0 / (2.0 * np.pi * f[-1] * 30.0)), np.log(1.0 / (2.0 * np.pi * f[0] / 30.0))
    r = residual(u)
    lam, h = 1e-2, 1e-4
    for _ in range(max_iter):
        jac = np.empty((r.size, n_poles))
        for i in range(n_poles):
            du = u.copy()
            du[i] += h
            jac[:, i] = (residual(du) - r) / h
        g, hess = jac.T @ r, jac.T @ jac
        improved = False
        while lam < 1e8:
            step = np.linalg.solve(hess + lam * np.diag(np.diag(hess) + 1e-12), -g)
            u_new = np.clip(u + step, lo, hi)
            r_new = residual(u_new)
            if r_new @ r_new < r @ r:
                improved = True
                u, r, lam = u_new, r_new, max(lam / 3.0, 1e-7)
                break
            lam *= 4.0
        if not improved or np.abs(step).max() < 1e-6:
            break
    return np.sort(np.exp(u))[::-1]


def fit_errors(medium: DebyeMedium, eps_target: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """Error relativo máximo ``(eps', sigma)`` del modelo frente a ``eps_target`` en ``f``."""
    eps_t, sig_t = _target_parts(np.asarray(eps_target), np.asarray(f, dtype=float))
    model = medium.complex_permittivity(f)
    eps_m, sig_m = _target_parts(model, f)
    err_e = float(np.max(np.abs(eps_m / eps_t - 1.0)))
    lossy = sig_t > 0
    err_s = float(np.max(np.abs(sig_m[lossy] / sig_t[lossy] - 1.0))) if lossy.any() else float(np.max(np.abs(sig_m)))
    return err_e, err_s


# --------------------------------------------------------------------------
# Modelo de los tejidos en una banda
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TissueDebyeModel:
    """Polos comunes y ajuste de cada tejido en ``band``.

    ``errors[k] = (err_eps, err_sigma)``: error relativo máximo frente a
    Gabriel en una rejilla fina de la banda (400 puntos), no sólo en los
    puntos del ajuste.
    """

    band: tuple[float, float]
    tau: np.ndarray
    tissues: dict[str, DebyeMedium]
    errors: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def n_poles(self) -> int:
        return len(self.tau)

    @property
    def max_error(self) -> float:
        return max((max(e) for e in self.errors.values()), default=0.0)

    def medium(self, key: str) -> DebyeMedium:
        return self.tissues[key]

    def fit(self, eps_target: np.ndarray, f: np.ndarray | None = None) -> DebyeMedium:
        """Ajusta otro material (p. ej. una mezcla) con los polos de este modelo."""
        f = fit_frequencies(self.band) if f is None else f
        return fit_amplitudes(eps_target, f, self.tau)


#: Mezclas porosas que entran en la optimización de los polos: la de Maxwell
#: Garnett con poros conductores (sangre) crea una relajación interfacial
#: (Maxwell-Wagner) en la banda de las microondas que los polos de los tejidos
#: puros no cubren (sin ellas, la cortical con un 80 % de poros quedaba con un
#: 1.8 % de error).
POROUS_FIT_LEVELS: tuple[tuple[str, str, float], ...] = (
    ("cortical_bone", "blood", 0.5), ("cortical_bone", "blood", 0.8),
    ("cortical_bone", "marrow", 0.5), ("cortical_bone", "marrow", 0.8),
)


def porous_permittivity(tissue: str, phi, f: np.ndarray, fill: str = porous.CORTICAL_REFERENCE_FILL) -> np.ndarray:
    """Permitividad de Maxwell Garnett de un tejido poroso, forma ``(len(phi), len(f))``."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    eps = {k: dielectric.complex_permittivity(k, f) for k in porous.dependencies(fill)}
    out = np.empty((phi.size, f.size), dtype=complex)
    # porous_values trabaja con escalares por tejido: se recorre la banda frecuencia a frecuencia
    for i in range(f.size):
        out[:, i] = porous.porous_values(tissue, phi, lambda k, i=i: eps[k][i], "dielectric", fill)
    return out


@lru_cache(maxsize=32)
def _tissue_model(band: tuple[float, float], tissues: tuple[str, ...], n_poles: int,
                  porous_levels: tuple[tuple[str, str, float], ...]) -> TissueDebyeModel:
    f = fit_frequencies(band)
    targets = {k: dielectric.complex_permittivity(k, f) for k in tissues}
    extra = [porous_permittivity(t, phi, f, fill)[0] for t, fill, phi in porous_levels]
    tau = optimize_poles(list(targets.values()) + extra, f, n_poles)
    fits = {k: fit_amplitudes(t, f, tau) for k, t in targets.items()}
    f_check = np.geomspace(band[0], band[1], 400)
    errors = {k: fit_errors(m, dielectric.complex_permittivity(k, f_check), f_check) for k, m in fits.items()}
    return TissueDebyeModel(band, tau, fits, errors)


def tissue_debye_model(band: tuple[float, float], n_poles: int | None = None, tissues: list[str] | None = None,
                       tolerance: float = DEFAULT_TOLERANCE, max_poles: int = 8,
                       include_porous: bool = True) -> TissueDebyeModel:
    """Modelo de Debye de los tejidos en la banda ``(f_min, f_max)`` (Hz).

    Por defecto ajusta **todos** los tejidos del registro con los mismos polos
    (el modelo no depende del fantoma y se calcula una vez por banda) y tiene
    en cuenta las mezclas porosas de ``POROUS_FIT_LEVELS`` al elegir los polos.
    Con ``n_poles=None`` usa el menor número de polos (de 2 a ``max_poles``)
    cuyo error relativo máximo en ``eps'`` y ``sigma`` no supera ``tolerance``.
    """
    f_min, f_max = float(band[0]), float(band[1])
    if not (0.0 < f_min < f_max):
        raise ValueError(f"banda no válida: {band}")
    if f_max > dielectric.GABRIEL_MAX_FREQ:
        raise ValueError("el modelo de Debye se ajusta al Cole-Cole de Gabriel, válido hasta 100 GHz")
    keys = tuple(tissues) if tissues is not None else tuple(tissue_keys())
    levels = POROUS_FIT_LEVELS if include_porous else ()
    if n_poles is not None:
        return _tissue_model((f_min, f_max), keys, int(n_poles), levels)
    model = None
    for k in range(2, max_poles + 1):
        model = _tissue_model((f_min, f_max), keys, k, levels)
        if model.max_error <= tolerance:
            break
    return model


# --------------------------------------------------------------------------
# Mapas por celda de un fantoma
# --------------------------------------------------------------------------
def porous_level_table(model: TissueDebyeModel, tissue: str, fill: str = porous.CORTICAL_REFERENCE_FILL,
                       step: float = POROSITY_STEP) -> tuple[np.ndarray, np.ndarray]:
    """``(phi, params)``: ajuste de la mezcla de Maxwell Garnett en una tabla de porosidades.

    ``params[i] = [eps_inf, sigma, d_eps_1..K]`` de la celda con porosidad ``phi[i]``.
    """
    f = fit_frequencies(model.band)
    phi = np.linspace(0.0, 1.0, int(round(1.0 / step)) + 1)
    target = porous_permittivity(tissue, phi, f, fill)
    params = np.empty((phi.size, model.n_poles + 2))
    for j in range(phi.size):
        m = fit_amplitudes(target[j], f, model.tau)
        params[j] = np.concatenate([[m.eps_inf, m.sigma], m.delta_eps])
    return phi, params


def debye_maps(phantom, model: TissueDebyeModel) -> DebyeMedium:
    """Mapas ``(nz, nx)`` de ``eps_inf`` y ``sigma`` y ``(K, nz, nx)`` de ``d_eps`` del fantoma.

    Sigue la misma semántica que ``Phantom.property_map(physics="dielectric")``:
    cada celda toma el ajuste de su tejido y, si el fantoma tiene mapa de
    porosidad, las celdas porosas con porosidad finita toman el ajuste de la
    mezcla de Maxwell Garnett (interpolado en la tabla de porosidades).
    """
    from ..phantoms.phantom import TISSUE_INDEX  # import tardío: phantoms importa materials

    missing = [k for k in phantom.property_tissues() if k not in model.tissues]
    if missing:
        raise KeyError(f"el modelo de Debye no incluye: {', '.join(missing)}")
    n_par = model.n_poles + 2
    lut = np.zeros((len(TISSUE_INDEX), n_par))
    for k in phantom.present_tissues():
        m = model.tissues[k]
        lut[TISSUE_INDEX[k]] = np.concatenate([[m.eps_inf, m.sigma], m.delta_eps])
    params = lut[phantom.labels]  # (nz, nx, n_par)
    if phantom.porosity is not None:
        for k in porous.POROUS_TISSUES:
            cells = (phantom.labels == TISSUE_INDEX[k]) & np.isfinite(phantom.porosity)
            if not cells.any():
                continue
            phi_tab, par_tab = porous_level_table(model, k, phantom.pore_fill)
            phi = np.clip(phantom.porosity[cells], 0.0, 1.0)
            params[cells] = np.stack([np.interp(phi, phi_tab, par_tab[:, p]) for p in range(n_par)], axis=-1)
    params = np.moveaxis(params, -1, 0)
    return DebyeMedium(params[0].copy(), params[1].copy(), params[2:].copy(), model.tau.copy())
