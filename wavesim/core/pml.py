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
    """Coeficientes ``(a, b)`` de la CPML en las posiciones dadas (en celdas).

    ``positions`` son índices (enteros o semienteros para nodos desplazados) a
    lo largo de un eje de ``n`` celdas; las celdas ``0..npml-1`` y las
    ``npml`` últimas forman la capa absorbente.
    """
    if npml <= 0:
        z = np.zeros_like(positions, dtype=float)
        return z, np.ones_like(positions, dtype=float)
    thickness = npml * dx
    d0 = -(order + 1) * c_max * np.log(reflection) / (2.0 * thickness)
    alpha_max = np.pi * f_ref
    # Penetración normalizada (0 en el borde interior de la capa, 1 en el exterior)
    pen_left = np.clip((npml - positions) / npml, 0.0, 1.0)
    pen_right = np.clip((positions - (n - 1 - npml)) / npml, 0.0, 1.0)
    pen = np.maximum(pen_left, pen_right)
    d = d0 * pen**order
    alpha = np.where(pen > 0, alpha_max * (1.0 - pen), 0.0)
    b = np.exp(-(d + alpha) * dt)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(d > 0, d * (b - 1.0) / (d + alpha), 0.0)
    return a, b


class CPMLDerivative:
    """Memoria CPML de una derivada concreta (un campo y un eje).

    Guarda ``psi`` sólo en las dos franjas absorbentes del eje para no gastar
    memoria ni tiempo en el interior de la malla.
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
    ):
        self.axis = axis
        n = shape[axis]
        pos = np.arange(n, dtype=float) + (0.5 if staggered else 0.0)
        a, b = cpml_coefficients(pos, n, npml, dx, dt, c_max, f_ref, reflection)
        active = np.nonzero(a != 0.0)[0]
        left = active[active < n // 2]
        right = active[active >= n // 2]
        self.nl = int(left.max() + 1) if left.size else 0
        self.nr_start = int(right.min()) if right.size else n
        other = shape[1 - axis]

        def shaped(v: np.ndarray) -> np.ndarray:
            v = v.astype(dtype)
            return v[None, :] if axis == 1 else v[:, None]

        self.a_l, self.b_l = shaped(a[: self.nl]), shaped(b[: self.nl])
        self.a_r, self.b_r = shaped(a[self.nr_start :]), shaped(b[self.nr_start :])
        nr = n - self.nr_start
        if axis == 1:
            self.psi_l = np.zeros((other, self.nl), dtype)
            self.psi_r = np.zeros((other, nr), dtype)
        else:
            self.psi_l = np.zeros((self.nl, other), dtype)
            self.psi_r = np.zeros((nr, other), dtype)

    def apply(self, deriv: np.ndarray) -> None:
        """Corrige ``deriv`` (array completo) in situ dentro de las franjas."""
        if self.axis == 1:
            dl = deriv[:, : self.nl]
            dr = deriv[:, self.nr_start :]
        else:
            dl = deriv[: self.nl, :]
            dr = deriv[self.nr_start :, :]
        if self.nl:
            self.psi_l *= self.b_l
            self.psi_l += self.a_l * dl
            dl += self.psi_l
        if dr.size:
            self.psi_r *= self.b_r
            self.psi_r += self.a_r * dr
            dr += self.psi_r
