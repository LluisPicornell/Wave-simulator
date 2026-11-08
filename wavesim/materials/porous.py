"""Medio efectivo de los tejidos porosos: hueso cortical y hueso trabecular.

Un fantoma puede llevar un mapa de **porosidad** (``Phantom.porosity``): la
fracción de volumen de cada celda ocupada por fluido. Sólo se aplica a las
celdas de ``POROUS_TISSUES``; en el resto se ignora. Cada modalidad combina
entonces la matriz ósea y el fluido de los poros con la regla de mezcla que
corresponde a su física:

- **lineal** (exacta para la densidad, ``mu`` de rayos X, absorción óptica y
  la señal de RM, que suma la magnetización de cada fracción de volumen);
- **Maxwell Garnett 2D** para la permitividad compleja: inclusiones
  circulares aisladas en una matriz. Una mezcla lineal (en paralelo) haría
  que el 5 % de sangre de los canales dominara la conductividad de la
  cortical a baja frecuencia (la sangre conduce 35 veces más), cuando los
  poros aislados apenas la cambian; Maxwell Garnett lo reproduce y es lo que
  dan los propios solvers si los poros se dibujan explícitos;
- módulos elásticos: cota superior de **Hashin-Shtrikman 2D** para la
  cortical (poros aislados en deformación plana) y una ley de potencia en la
  fracción de sólido para la red trabecular (ver ``acoustic_effective``).

Referencias de cada tabla. Las tablas de cada física describen el tejido
entero, no la matriz: la cortical que se midió ya tenía sus canales
vasculares. Para no contarlos dos veces, la matriz (sin poros) se obtiene
invirtiendo la mezcla en la porosidad de referencia de esa tabla
(``CORTICAL_REFERENCE_POROSITY``), con los poros llenos de sangre. Así, una
celda con esa porosidad reproduce exactamente la tabla, y el modo básico
(sin mapa de porosidad) no cambia:

- acústica, rayos X y dieléctrica: muestras de cortical diafisaria adulta
  sana, con un 3-8 % de porosidad (Bousson et al., J Bone Miner Res 16:1308,
  2001; Cooper et al., Bone 40:957, 2007) → 5 %;
- RM: el T2* de 0.3-0.5 ms de la tabla es el del agua ligada a la matriz
  (Du & Bydder 2013); el agua de los poros no está incluida → 0 %;
- óptica: la composición de la tabla sólo tiene un 1 % de sangre, menos que
  el 5 % de poros de referencia → 0 %.

El hueso trabecular de las tablas es un medio efectivo con BV/TV ≈ 20 %
(``TRABECULAR_REFERENCE_BVTV``, la misma fracción que su mezcla de rayos X).
Sus trabéculas son de la misma matriz que la cortical, y el contenido de los
espacios se deduce de cada tabla: es la fracción que, mezclada con un 20 %
de matriz, reproduce la tabla. Por eso no es la médula amarilla de la
diáfisis en todas las físicas: la tabla dieléctrica de Gabriel (hueso
esponjoso con médula roja) da un relleno de ε' ≈ 22 a 1 GHz y la de rayos X
uno casi igual a la médula amarilla. Así cada física sigue siendo coherente
con su propia tabla.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .acoustic import ACOUSTIC

POROUS_TISSUES = ("cortical_bone", "trabecular_bone")

#: Relleno con el que se invierten las tablas de la cortical (canales vasculares).
CORTICAL_REFERENCE_FILL = "blood"

#: Porosidad cortical que ya incluye la tabla de cada física (ver el docstring).
CORTICAL_REFERENCE_POROSITY: dict[str, float] = {
    "acoustic": 0.05, "xray": 0.05, "dielectric": 0.05, "optical": 0.0, "mri": 0.0,
}

#: Regla de mezcla de cada física.
MIXING_RULE: dict[str, str] = {
    "acoustic": "linear", "xray": "linear", "dielectric": "maxwell_garnett", "optical": "linear", "mri": "linear",
}

#: BV/TV del hueso trabecular de las tablas.
TRABECULAR_REFERENCE_BVTV = 0.20

#: Fluido de los espacios trabeculares para los módulos elásticos.
TRABECULAR_FILL = "marrow"

PHYSICS = tuple(MIXING_RULE)


# --------------------------------------------------------------------------
# Reglas de mezcla escalares
# --------------------------------------------------------------------------
def maxwell_garnett_2d(host, inclusion, fraction):
    """Maxwell Garnett 2D (inclusiones circulares) para una propiedad tipo permitividad.

    ``(e - h) / (e + h) = f (i - h) / (i + h)``. Da exactamente ``host`` con
    ``fraction = 0`` e ``inclusion`` con ``fraction = 1``. Vale para valores
    complejos y para cualquier magnitud proporcional a la permitividad (la
    admitividad ``j w eps0 eps`` incluida).
    """
    beta = (inclusion - host) / (inclusion + host)
    return host * (1.0 + fraction * beta) / (1.0 - fraction * beta)


def _maxwell_garnett_host(effective, inclusion, fraction):
    """Matriz ``h`` tal que ``maxwell_garnett_2d(h, inclusion, fraction) = effective``.

    Es la raíz de ``(1-f) h² + (1+f)(i-e) h - (1-f) e i = 0`` más próxima a
    ``e`` (la que tiende a ``e`` cuando no hay inclusiones).
    """
    f, e, i = fraction, complex(effective), complex(inclusion)
    a, b, c = 1.0 - f, (1.0 + f) * (i - e), -(1.0 - f) * e * i
    disc = np.sqrt(b * b - 4.0 * a * c)
    roots = ((-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a))
    h = min(roots, key=lambda r: abs(r - e))
    return h if np.iscomplexobj(effective) else h.real


def mix(rule: str, host, inclusion, fraction):
    """Propiedad de una mezcla con una fracción de volumen ``fraction`` de ``inclusion``."""
    if rule == "linear":
        return (1.0 - fraction) * host + fraction * inclusion
    if rule == "maxwell_garnett":
        return maxwell_garnett_2d(host, inclusion, fraction)
    raise ValueError(f"regla de mezcla desconocida '{rule}'")


def invert_host(rule: str, effective, inclusion, fraction):
    """Matriz que, con una fracción ``fraction`` de ``inclusion``, da ``effective``."""
    if fraction <= 0.0:
        return effective
    if rule == "linear":
        return (effective - fraction * inclusion) / (1.0 - fraction)
    if rule == "maxwell_garnett":
        return _maxwell_garnett_host(effective, inclusion, fraction)
    raise ValueError(f"regla de mezcla desconocida '{rule}'")


# --------------------------------------------------------------------------
# Propiedades escalares por celda
# --------------------------------------------------------------------------
def dependencies(fill: str) -> tuple[str, ...]:
    """Tejidos cuyas propiedades hacen falta para el medio efectivo con relleno cortical ``fill``."""
    return tuple(dict.fromkeys(("cortical_bone", "trabecular_bone", CORTICAL_REFERENCE_FILL, fill)))


def cortical_matrix(getter: Callable[[str], float], physics: str):
    """Valor de la matriz ósea sin poros en la física ``physics``."""
    rule = MIXING_RULE[physics]
    return invert_host(rule, getter("cortical_bone"), getter(CORTICAL_REFERENCE_FILL),
                       CORTICAL_REFERENCE_POROSITY[physics])


def trabecular_space(getter: Callable[[str], float], physics: str):
    """Contenido de los espacios trabeculares deducido de la tabla (ver el docstring del módulo)."""
    rule = MIXING_RULE[physics]
    return invert_host(rule, getter("trabecular_bone"), cortical_matrix(getter, physics),
                       TRABECULAR_REFERENCE_BVTV)


def porous_values(tissue: str, porosity: np.ndarray, getter: Callable[[str], float], physics: str,
                  fill: str = CORTICAL_REFERENCE_FILL) -> np.ndarray:
    """Propiedad efectiva de las celdas de ``tissue`` con fracción de fluido ``porosity``.

    ``getter(clave_de_tejido)`` da el valor de la tabla. En la cortical los
    poros se llenan de ``fill``; en el trabecular, del contenido deducido de
    la tabla.
    """
    rule = MIXING_RULE[physics]
    phi = np.asarray(porosity, dtype=float)
    matrix = cortical_matrix(getter, physics)
    if tissue == "cortical_bone":
        return mix(rule, matrix, getter(fill), phi)
    if tissue == "trabecular_bone":
        # La matriz es la inclusión (fracción de sólido 1 - phi) dentro del fluido.
        return mix(rule, trabecular_space(getter, physics), matrix, 1.0 - phi)
    raise KeyError(f"'{tissue}' no es un tejido poroso ({', '.join(POROUS_TISSUES)})")


# --------------------------------------------------------------------------
# Acústica y elasticidad
# --------------------------------------------------------------------------
def hashin_shtrikman_2d(k1, mu1, k2, mu2, f2):
    """Cota superior de Hashin-Shtrikman en 2D (deformación plana) con la fase 1 más rígida.

    ``k`` es el módulo de compresión en el plano (``lambda + mu``), no el 3D.
    Para una fase 2 fluida (``mu2 = 0``) con poros aislados coincide con la
    estimación de Mori-Tanaka, y es exacta en los límites ``f2 = 0`` y
    ``f2 = 1``. Hashin, J Mech Phys Solids 13:119 (1965).
    """
    f2 = np.asarray(f2, dtype=float)
    f1 = 1.0 - f2
    k = k1 + f2 / (1.0 / (k2 - k1) + f1 / (k1 + mu1))
    mu = mu1 + f2 / (1.0 / (mu2 - mu1) + f1 * (k1 + 2.0 * mu1) / (2.0 * mu1 * (k1 + mu1)))
    return k, mu


@dataclass(frozen=True)
class ElasticMatrix:
    """Matriz ósea sin poros: densidad, módulos 2D y atenuación (Np/m) en ``f``."""

    rho: float
    k: float
    mu: float
    alpha_p: np.ndarray
    alpha_s: np.ndarray

    @property
    def c_p(self) -> float:
        return float(np.sqrt((self.k + self.mu) / self.rho))

    @property
    def c_s(self) -> float:
        return float(np.sqrt(self.mu / self.rho))


def _moduli(key: str) -> tuple[float, float, float]:
    """``(rho, k, mu)`` del tejido en 2D: ``M = rho c_p²``, ``mu = rho c_s²``, ``k = M - mu``."""
    a = ACOUSTIC[key]
    m, mu = a.density * a.c_p**2, a.density * a.c_s**2
    return a.density, m - mu, mu


def _alpha(key: str, f: np.ndarray, shear: bool = False) -> np.ndarray:
    return np.array([ACOUSTIC[key].alpha_np_per_m(fk, shear=shear) for fk in np.atleast_1d(f)])


def elastic_matrix(f_hz) -> ElasticMatrix:
    """Matriz cortical invertida en la porosidad de referencia (llena de sangre).

    Los módulos se obtienen por punto fijo sobre Hashin-Shtrikman 2D; la
    densidad y la atenuación, invirtiendo la mezcla lineal.
    """
    phi_r = CORTICAL_REFERENCE_POROSITY["acoustic"]
    ref = CORTICAL_REFERENCE_FILL
    rho_c, k_c, mu_c = _moduli("cortical_bone")
    rho_f, k_f, _ = _moduli(ref)
    k_m, mu_m = k_c, mu_c
    for _ in range(60):
        k_e, mu_e = hashin_shtrikman_2d(k_m, mu_m, k_f, 0.0, phi_r)
        k_m, mu_m = k_m * k_c / float(k_e), mu_m * mu_c / float(mu_e)
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    alpha_p = invert_host("linear", _alpha("cortical_bone", f), _alpha(ref, f), phi_r)
    alpha_s = _alpha("cortical_bone", f, shear=True) / (1.0 - phi_r)
    return ElasticMatrix(invert_host("linear", rho_c, rho_f, phi_r), k_m, mu_m, alpha_p, alpha_s)


@dataclass(frozen=True)
class AcousticEffective:
    """Propiedades acústicas efectivas en función de la porosidad ``phi`` (vector).

    ``alpha_p`` y ``alpha_s`` tienen forma ``(len(phi), len(f))`` (Np/m).
    """

    phi: np.ndarray
    rho: np.ndarray
    c_p: np.ndarray
    c_s: np.ndarray
    alpha_p: np.ndarray
    alpha_s: np.ndarray


def trabecular_exponents(matrix: ElasticMatrix) -> tuple[float, float]:
    """``(n_M, n_mu)`` tales que la ley de potencia reproduce la tabla con BV/TV de referencia."""
    _, k_t, mu_t = _moduli("trabecular_bone")
    rho_f, k_f, _ = _moduli(TRABECULAR_FILL)
    m_t, m_f, m_m = k_t + mu_t, k_f, matrix.k + matrix.mu
    b = np.log(TRABECULAR_REFERENCE_BVTV)
    return float(np.log((m_t - m_f) / (m_m - m_f)) / b), float(np.log(mu_t / matrix.mu) / b)


def acoustic_effective(tissue: str, phi, f_hz, fill: str = CORTICAL_REFERENCE_FILL) -> AcousticEffective:
    """Densidad, velocidades y atenuación efectivas de un tejido poroso.

    - **Cortical**: matriz (``elastic_matrix``) con poros llenos de ``fill``.
      Densidad lineal; módulos de Hashin-Shtrikman 2D; atenuación de
      absorción lineal en volumen. La dispersión por los poros no se añade a
      mano: la produce el propio solver, porque los poros resueltos están en
      el mapa y los no resueltos dejan una porosidad que fluctúa de celda a
      celda con la misma varianza (en la aproximación de Born, un poro más
      pequeño que la celda dispersa igual que su volumen repartido en ella).
    - **Trabecular**: módulos ``M = M_f + (M_m - M_f) s^n`` y ``mu = mu_m s^n'``
      con ``s = 1 - phi`` la fracción de sólido; los exponentes se calibran
      para pasar por la tabla con BV/TV = 20 % (salen ~1.2 y ~1.1; las leyes
      módulo-densidad medidas tienen exponentes de 1.5-2.2 según la zona,
      Morgan, Bayraktar & Keaveny, J Biomech 36:897, 2003, pero aquí se
      prefiere reproducir la tabla del modo básico). La atenuación de la
      tabla (15 dB/(cm·MHz)) es sobre todo dispersión por la red: su exceso
      sobre la absorción de matriz y médula se reparte con un peso
      ``phi (1 - phi)``, la varianza de un medio de dos fases (Born), que vale
      1 en la BV/TV de referencia y 0 en las celdas de trabécula o de médula
      puras, donde la dispersión ya la produce la red dibujada.
    """
    phi = np.clip(np.atleast_1d(np.asarray(phi, dtype=float)), 0.0, 1.0)
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    mat = elastic_matrix(f)
    if tissue == "cortical_bone":
        rho_f, k_f, _ = _moduli(fill)
        rho = (1.0 - phi) * mat.rho + phi * rho_f
        k, mu = hashin_shtrikman_2d(mat.k, mat.mu, k_f, 0.0, phi)
        alpha_p = (1.0 - phi)[:, None] * mat.alpha_p[None, :] + phi[:, None] * _alpha(fill, f)[None, :]
        alpha_s = (1.0 - phi)[:, None] * mat.alpha_s[None, :]
    elif tissue == "trabecular_bone":
        s = 1.0 - phi
        b_r = TRABECULAR_REFERENCE_BVTV
        rho_t, _, _ = _moduli("trabecular_bone")
        rho_space = invert_host("linear", rho_t, mat.rho, b_r)
        rho = s * mat.rho + phi * rho_space
        n_m, n_mu = trabecular_exponents(mat)
        _, k_f, _ = _moduli(TRABECULAR_FILL)
        m = k_f + (mat.k + mat.mu - k_f) * s**n_m
        mu = mat.mu * s**n_mu
        k = m - mu
        weight = (phi * s / (b_r * (1.0 - b_r)))[:, None]
        a_f = _alpha(TRABECULAR_FILL, f)
        excess_p = _alpha("trabecular_bone", f) - (b_r * mat.alpha_p + (1.0 - b_r) * a_f)
        excess_s = _alpha("trabecular_bone", f, shear=True) - b_r * mat.alpha_s
        alpha_p = s[:, None] * mat.alpha_p[None, :] + phi[:, None] * a_f[None, :] + weight * excess_p[None, :]
        alpha_s = s[:, None] * mat.alpha_s[None, :] + weight * excess_s[None, :]
    else:
        raise KeyError(f"'{tissue}' no es un tejido poroso ({', '.join(POROUS_TISSUES)})")
    c_p = np.sqrt((k + mu) / rho)
    c_s = np.sqrt(np.maximum(mu, 0.0) / rho)
    return AcousticEffective(phi, rho, c_p, c_s, alpha_p, alpha_s)
