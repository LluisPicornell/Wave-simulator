"""Inhomogeneidad de B0 inducida por la susceptibilidad magnética de los tejidos.

Un tejido de susceptibilidad volumétrica ``χ`` (SI, ~1e-6) se imanta en el
campo ``B0`` y perturba la componente de B a lo largo de ``B0``. Para
``|χ| << 1`` el campo relativo que ven los núcleos, corregido por la esfera
de Lorentz, es la convolución de ``χ`` con el núcleo dipolar; en Fourier
(Salomir et al., Concepts Magn Reson B 19B:26, 2003; Marques & Bowtell,
Concepts Magn Reson B 25B:65, 2005):

    ΔB(k) / B0 = D(k) χ(k),      D(k) = 1/3 - (k·b)² / |k|²

con ``b`` el vector unitario de ``B0``. El 1/3 es la esfera de Lorentz: un
medio uniforme e infinito no desplaza la frecuencia respecto de sí mismo.

**Corte 2D.** El fantoma es un corte (x, z) de un objeto que se supone
**invariante en la dirección perpendicular al corte (y)**. Su transformada
3D es ``χ(kx, kz) δ(ky)``: sólo cuenta el núcleo en ``ky = 0``, y la
convolución 3D se reduce **exactamente** a la 2D con

    D(kx, kz) = 1/3 - (kx bx + kz bz)² / (kx² + kz²).

Lo aproximado es la geometría (un hueso infinitamente largo), no el campo.
Según la orientación de B0:

- ``b = y`` (B0 perpendicular al corte; el corte transversal de una
  extremidad dentro del túnel de un equipo convencional): ``D = 1/3``, cada
  tejido sólo se desplaza ``(χ - χ_ref)/3`` y no hay gradientes; es el
  conocido "un cilindro paralelo a B0 no perturba el exterior".
- ``b`` en el plano (B0 a lo largo del hueso en un corte longitudinal, o un
  imán abierto de campo vertical con el corte transversal): el campo de un
  cilindro perpendicular a B0 es ``-Δχ/6`` dentro y
  ``(Δχ/2) (a/r)² cos 2φ`` fuera (``φ`` medido desde B0).

El componente ``k = 0`` no está definido (depende de la forma del objeto
en el infinito); se toma la media angular de ``D``,
``1/3 - (bx² + bz²)/2``, que es la que da el campo exacto del cilindro
(una frontera circular en el infinito). Una constante global sólo
desplaza la frecuencia central, que el equipo ajusta.
"""

from __future__ import annotations

import numpy as np

from ..materials.mri import CHI_WATER, SUSCEPTIBILITY_PPM
from ..phantoms.phantom import Phantom


def b0_unit_vector(direction) -> tuple[float, float, float]:
    """``(bx, by, bz)`` de ``"x"``, ``"y"`` (perpendicular al corte), ``"z"`` o un vector de 3 componentes."""
    named = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    if isinstance(direction, str):
        if direction not in named:
            raise ValueError(f"dirección de B0 desconocida '{direction}' (use 'x', 'y', 'z' o un vector)")
        return named[direction]
    v = np.asarray(direction, dtype=float)
    if v.shape != (3,) or not np.linalg.norm(v):
        raise ValueError("la dirección de B0 debe ser un vector (bx, by, bz) no nulo")
    v = v / np.linalg.norm(v)
    return float(v[0]), float(v[1]), float(v[2])


def susceptibility_map(phantom: Phantom) -> np.ndarray:
    """Mapa ``(nz, nx)`` de susceptibilidad SI (ppm), con la mezcla lineal en las celdas porosas."""
    return phantom.property_map(lambda k: SUSCEPTIBILITY_PPM[k], physics="mri")


def dipole_field(chi_ppm: np.ndarray, direction="z", reference: float = CHI_WATER,
                 padding: float = 1.0) -> np.ndarray:
    """Campo relativo ``ΔB/B0`` (ppm) del mapa de susceptibilidad ``chi_ppm`` (ppm).

    ``reference``: susceptibilidad del medio que rodea al objeto hasta el
    infinito (el campo se refiere a él). ``padding``: fracción del tamaño
    que se añade a cada lado repitiendo el borde (``np.pad(mode="edge")``)
    antes de la FFT, para que las réplicas periódicas queden lejos; repetir
    el borde prolonga lo que sale del corte (las capas de un corte
    longitudinal siguen fuera del FOV, como en la pierna real). El tamaño
    de celda no interviene: el núcleo sólo depende de la dirección de k.

    Precisión (``docs/validacion/rm.md``): con B0 a lo largo de un eje de la
    malla, el campo de un cilindro coincide con el exacto a < 0.7 % de
    ``Δχ/2`` a más de 2 celdas del borde; con B0 oblicuo a la malla la
    escalera del borde deja estrías (hasta un 5 % a 2-4 celdas, 2 % a 10).
    """
    bx, by, bz = b0_unit_vector(direction)
    d = np.asarray(chi_ppm, dtype=float) - reference
    nz, nx = d.shape
    pz, px = int(np.ceil(padding * nz)), int(np.ceil(padding * nx))
    dp = np.pad(d, ((pz, pz), (px, px)), mode="edge")
    kz = np.fft.fftfreq(dp.shape[0])
    kx = np.fft.rfftfreq(dp.shape[1])
    KZ, KX = np.meshgrid(kz, kx, indexing="ij")
    k2 = KX**2 + KZ**2
    kb = KX * bx + KZ * bz
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = 1.0 / 3.0 - kb**2 / k2
    kernel[0, 0] = 1.0 / 3.0 - (bx**2 + bz**2) / 2.0
    field = np.fft.irfft2(kernel * np.fft.rfft2(dp), s=dp.shape)
    return field[pz:pz + nz, px:px + nx]


def cylinder_field(X: np.ndarray, Z: np.ndarray, radius: float, delta_chi: float, direction="z") -> np.ndarray:
    """Solución exacta ``ΔB/B0`` (unidades de ``delta_chi``) de un cilindro infinito a lo largo de y centrado en 0.

    B0 en el plano (``"x"``, ``"z"`` o un vector con ``by = 0``): ``-Δχ/6``
    dentro y ``(Δχ/2)(a/r)² cos 2φ`` fuera, con ``φ`` medido desde B0.
    B0 según y: ``Δχ/3`` dentro y 0 fuera.
    """
    bx, by, bz = b0_unit_vector(direction)
    r2 = X**2 + Z**2
    inside = r2 <= radius**2
    if by == 1.0:
        return np.where(inside, delta_chi / 3.0, 0.0)
    if abs(by) > 1e-12:
        raise ValueError("cylinder_field sólo admite B0 en el plano o perpendicular a él")
    with np.errstate(invalid="ignore", divide="ignore"):
        proj = X * bx + Z * bz  # componente a lo largo de B0
        cos2 = np.where(r2 > 0, (2 * proj**2 - r2) / np.where(r2 > 0, r2, 1.0), 1.0)
        out = 0.5 * delta_chi * radius**2 / np.where(r2 > 0, r2, 1.0) * cos2
    return np.where(inside, -delta_chi / 6.0, out)
