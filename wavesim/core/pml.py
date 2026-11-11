"""Bordes absorbentes CPML (Convolutional Perfectly Matched Layer).

Sin bordes absorbentes, las ondas que llegan al final de la malla rebotarían
como si hubiera una pared, contaminando la simulación. La CPML (Roden & Gedney
2000; Komatitsch & Martin 2007) corrige cada derivada espacial dentro de una
franja de ``npml`` celdas en el contorno:

    d~f/dx = df/dx + psi,     psi <- b * psi + a * df/dx

con un perfil de amortiguamiento polinómico ``d(x)`` y un término CFS ``alpha``
que mejora la absorción de ondas rasantes y evanescentes. Se usa ``kappa = 1``.

El mismo código sirve para el solver elástico (ultrasonido) y para Maxwell.
"""

from __future__ import annotations

import numpy as np


def cpml_profiles(
    positions: np.ndarray,
    n: int,
    npml: int,
    dx: float,
    c_max: float,
    f_ref: float,
    reflection: float = 1e-5,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Perfiles ``(d, alpha)`` (1/s) de la CPML en las posiciones dadas (en celdas).

    ``positions`` son índices (enteros o semienteros para nodos desplazados) a
    lo largo de un eje de ``n`` celdas; las celdas ``0..npml-1`` y las
    ``npml`` últimas forman la capa absorbente.
    """
    if npml <= 0:
        z = np.zeros_like(positions, dtype=float)
        return z, z.copy()
    thickness = npml * dx
    d0 = -(order + 1) * c_max * np.log(reflection) / (2.0 * thickness)
    alpha_max = np.pi * f_ref
    # Penetración normalizada (0 en el borde interior de la capa, 1 en el exterior)
    pen_left = np.clip((npml - positions) / npml, 0.0, 1.0)
    pen_right = np.clip((positions - (n - 1 - npml)) / npml, 0.0, 1.0)
    pen = np.maximum(pen_left, pen_right)
    return d0 * pen**order, np.where(pen > 0, alpha_max * (1.0 - pen), 0.0)


def _recursion(d: np.ndarray, alpha: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """``(a, b)`` de la convolución recursiva ``psi <- b psi + a df`` para un amortiguamiento ``d`` y un ``alpha``."""
    b = np.exp(-(d + alpha) * dt)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(d > 0, d * (b - 1.0) / (d + alpha), 0.0)
    return a, b


def cpml_coefficients(
    positions: np.ndarray,
    n: int,
    npml: int,
    dx: float,
    dt: float,
    c_max: float,
    f_ref: float,
    reflection: float = 1e-5,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Coeficientes ``(a, b)`` de la CPML en las posiciones dadas (ver ``cpml_profiles``)."""
    if npml <= 0:
        z = np.zeros_like(positions, dtype=float)
        return z, np.ones_like(positions, dtype=float)
    return _recursion(*cpml_profiles(positions, n, npml, dx, c_max, f_ref, reflection, order), dt)


def _bands(active: np.ndarray, n: int) -> tuple[int, int]:
    """Fin de la franja izquierda y comienzo de la derecha de un eje de ``n`` celdas (máscara de celdas activas)."""
    idx = np.nonzero(active)[0]
    left, right = idx[idx < n // 2], idx[idx >= n // 2]
    return (int(left.max() + 1) if left.size else 0), (int(right.min()) if right.size else n)


class CPMLDerivative:
    """Memoria CPML de una derivada concreta (un campo y un eje).

    Guarda ``psi`` sólo en las franjas absorbentes para no gastar memoria ni
    tiempo en el interior de la malla: las dos del propio eje (``*_l``,
    ``*_r``) y, con ``ratio > 0``, las de la otra dirección (``*_cl``, ``*_cr``).

    ``ratio`` > 0 convierte la capa en una M-PML (*multiaxial PML*,
    Meza-Fajardo & Papageorgiou, Bull Seismol Soc Am 98:1811, 2008): dentro
    de la capa normal a la otra dirección esta derivada también se amortigua,
    con ``ratio`` veces el perfil ``d`` de esa capa (y su ``alpha``). En las
    esquinas el amortiguamiento es la suma ``d_propio + ratio · d_otro``.
    ``staggered_other`` indica si el nodo de la derivada está desplazado media
    celda a lo largo de la otra dirección. Con ``ratio = 0`` es la CPML de
    siempre.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        axis: int,
        staggered: bool,
        npml: int,
        dx: float,
        dt: float,
        c_max: float,
        f_ref: float,
        dtype=np.float32,
        reflection: float = 1e-5,
        ratio: float = 0.0,
        staggered_other: bool = False,
    ):
        self.axis = axis
        self.ratio = float(ratio)
        n, other = shape[axis], shape[1 - axis]
        pos = np.arange(n, dtype=float) + (0.5 if staggered else 0.0)
        a, b = cpml_coefficients(pos, n, npml, dx, dt, c_max, f_ref, reflection)
        self.nl, self.nr_start = _bands(a != 0.0, n)
        nl, nr = self.nl, self.nr_start

        def shaped(v: np.ndarray) -> np.ndarray:
            v = v.astype(dtype)
            return v[None, :] if axis == 1 else v[:, None]

        def oriented(own: np.ndarray, cross: np.ndarray) -> np.ndarray:
            """Array 2D (filas = z) a partir de un perfil a lo largo del eje propio y otro a lo largo del otro."""
            return own[None, :] + cross[:, None] if axis == 1 else own[:, None] + cross[None, :]

        if self.ratio > 0 and npml > 0:
            pos_o = np.arange(other, dtype=float) + (0.5 if staggered_other else 0.0)
            d_own, al_own = cpml_profiles(pos, n, npml, dx, c_max, f_ref, reflection)
            d_oth, al_oth = cpml_profiles(pos_o, other, npml, dx, c_max, f_ref, reflection)
            # Franjas propias: d_propio + ratio·d_otro (las esquinas cambian) con el alpha propio
            zero = np.zeros(other)
            d2 = oriented(d_own, self.ratio * d_oth)
            al2 = oriented(al_own, zero)
            a2, b2 = _recursion(d2, al2, dt)
            sl = (slice(None), slice(0, nl)) if axis == 1 else (slice(0, nl), slice(None))
            sr = (slice(None), slice(nr, n)) if axis == 1 else (slice(nr, n), slice(None))
            self.a_l, self.b_l = a2[sl].astype(dtype), b2[sl].astype(dtype)
            self.a_r, self.b_r = a2[sr].astype(dtype), b2[sr].astype(dtype)
            # Franjas de la otra dirección (fuera de las propias): sólo ratio·d_otro, con el alpha de esa capa
            ac, bc = _recursion(self.ratio * d_oth, al_oth, dt)
            self.ml, self.mr_start = _bands(ac != 0.0, other)
            ml, mr = self.ml, self.mr_start

            def shaped_o(v: np.ndarray) -> np.ndarray:
                v = v.astype(dtype)
                return v[:, None] if axis == 1 else v[None, :]

            self.a_cl, self.b_cl = shaped_o(ac[:ml]), shaped_o(bc[:ml])
            self.a_cr, self.b_cr = shaped_o(ac[mr:]), shaped_o(bc[mr:])
            inner = max(nr - nl, 0)
            if axis == 1:
                self.psi_cl, self.psi_cr = np.zeros((ml, inner), dtype), np.zeros((other - mr, inner), dtype)
            else:
                self.psi_cl, self.psi_cr = np.zeros((inner, ml), dtype), np.zeros((inner, other - mr), dtype)
        else:
            self.a_l, self.b_l = shaped(a[:nl]), shaped(b[:nl])
            self.a_r, self.b_r = shaped(a[nr:]), shaped(b[nr:])
            self.ml, self.mr_start = 0, other
        if axis == 1:
            self.psi_l = np.zeros((other, nl), dtype)
            self.psi_r = np.zeros((other, n - nr), dtype)
        else:
            self.psi_l = np.zeros((nl, other), dtype)
            self.psi_r = np.zeros((n - nr, other), dtype)

    @staticmethod
    def _update(psi: np.ndarray, a: np.ndarray, b: np.ndarray, d: np.ndarray) -> None:
        psi *= b
        psi += a * d
        d += psi

    def apply(self, deriv: np.ndarray) -> None:
        """Corrige ``deriv`` (array completo) in situ dentro de las franjas."""
        nl, nr = self.nl, self.nr_start
        if self.axis == 1:
            dl, dr = deriv[:, :nl], deriv[:, nr:]
        else:
            dl, dr = deriv[:nl, :], deriv[nr:, :]
        if nl:
            self._update(self.psi_l, self.a_l, self.b_l, dl)
        if dr.size:
            self._update(self.psi_r, self.a_r, self.b_r, dr)
        if self.ml == 0 and self.mr_start == deriv.shape[1 - self.axis]:
            return
        ml, mr = self.ml, self.mr_start
        if self.axis == 1:
            cl, cr = deriv[:ml, nl:nr], deriv[mr:, nl:nr]
        else:
            cl, cr = deriv[nl:nr, :ml], deriv[nl:nr, mr:]
        if cl.size:
            self._update(self.psi_cl, self.a_cl, self.b_cl, cl)
        if cr.size:
            self._update(self.psi_cr, self.a_cr, self.b_cr, cr)
