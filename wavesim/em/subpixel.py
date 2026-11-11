"""Promedio sub-celda de la permitividad en las interfaces (como el *subpixel smoothing* de Meep).

Con un tejido por celda, una interfaz curva queda en escalera y el error del
campo cerca de ella es de primer orden en ``dx``. En modo TM el campo
eléctrico (Ey, perpendicular al plano) es siempre **tangente** a cualquier
interfaz del plano, y para la componente tangente la permitividad efectiva
correcta de una celda partida es la media aritmética de las permitividades
ponderada por área (Farjadpour et al., Opt. Lett. 31:2972, 2006; es el
promedio que usa Meep para E paralelo a la interfaz). Para la componente
normal haría falta la media armónica, pero no aparece en TM.

La media es lineal en ``eps*(w)``, así que vale para el solver de banda
estrecha (``eps' - j eps''`` complejo a ``f0``) y, como los polos de Debye
son comunes a todos los tejidos, para el de banda ancha promediando
directamente ``eps_inf``, ``sigma`` y ``d_eps_k``: el resultado es
exactamente la media de ``eps*(w)`` a todas las frecuencias.

Las fracciones de área salen de un fantoma auxiliar ``factor`` veces más fino
(``factor`` impar, alineado con ``fine_grid``): cada celda gruesa es la media
de sus ``factor x factor`` celdas finas.
"""

from __future__ import annotations

import numpy as np

from ..core.grid import Grid2D
from ..materials import dielectric
from ..materials.dispersive import DebyeMedium, TissueDebyeModel, debye_maps
from ..phantoms.phantom import Phantom


def fine_grid(grid: Grid2D, factor: int) -> Grid2D:
    """Malla ``factor`` veces más fina cuyas celdas subdividen exactamente las de ``grid``."""
    if factor < 1 or factor % 2 == 0:
        raise ValueError("factor debe ser un entero impar (la celda fina central coincide con el nodo de Ey)")
    h = grid.dx / factor
    off = (factor - 1) / 2 * h
    return Grid2D(grid.nx * factor, grid.nz * factor, h, grid.x0 - off, grid.z0 - off)


def block_average(fine: np.ndarray, factor: int) -> np.ndarray:
    """Media de cada bloque ``factor x factor`` de las dos últimas dimensiones."""
    *lead, nz, nx = fine.shape
    return fine.reshape(*lead, nz // factor, factor, nx // factor, factor).mean(axis=(-3, -1))


def coarse_phantom(fine: Phantom, factor: int) -> Phantom:
    """Fantoma grueso con el tejido de la celda fina central de cada bloque (para la densidad del SAR)."""
    c = (factor - 1) // 2
    g = fine.grid
    grid = Grid2D(g.nx // factor, g.nz // factor, g.dx * factor, g.x0 + c * g.dx, g.z0 + c * g.dx)
    sel = (slice(c, None, factor), slice(c, None, factor))
    por = fine.porosity[sel] if fine.porosity is not None else None
    return Phantom(grid, fine.labels[sel], {k: m[sel] for k, m in fine.masks.items()}, None, dict(fine.meta),
                   por, fine.pore_fill)


def smoothed_permittivity(fine: Phantom, factor: int, f0: float) -> np.ndarray:
    """Permitividad compleja ``eps' - j eps''`` promediada por área, en la malla gruesa (``eps_map``)."""
    tissues = fine.property_tissues()
    eps_c = {k: complex(dielectric.complex_permittivity(k, f0)) for k in tissues}
    return block_average(fine.property_map(lambda k: eps_c[k], dtype=complex, physics="dielectric"), factor)


def smoothed_debye_medium(fine: Phantom, factor: int, model: TissueDebyeModel) -> DebyeMedium:
    """Mapas de Debye promediados por área en la malla gruesa (``medium`` del solver de banda ancha)."""
    m = debye_maps(fine, model)
    return DebyeMedium(block_average(m.eps_inf, factor), block_average(m.sigma, factor),
                       block_average(m.delta_eps, factor), m.tau)
