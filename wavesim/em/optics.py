"""Luz (UV, visible, infrarrojo) en tejido: difusión y fotones balísticos.

En el visible y el infrarrojo cercano los tejidos dispersan la luz cientos de
veces por milímetro (``mu_s' >> mu_a``): los fotones pierden su dirección y
la luz se propaga como calor difundiéndose. Se resuelve la aproximación de
difusión estacionaria:

    -div(D grad(Phi)) + mu_a Phi = S,     D = 1 / (3 (mu_a + mu_s'))

con diferencias finitas y gradiente conjugado precondicionado. En la frontera
con el exterior (agua o aire de fondo) se impone el contorno de flujo nulo
*extrapolado* (Robin): ``Phi = 0`` a una distancia ``z_b = 2 A D`` fuera de la
superficie, donde ``A`` sale del salto de índice de refracción tejido/exterior.
Imponer ``Phi = 0`` en la celda vecina, como se hacía antes, hacía depender la
transmisión del tamaño de celda ``dx``; con ``z_b`` físico el resultado converge.
La transmisión se da como flujo saliente (exitancia) por la cara inferior.

Para imagen nítida sólo valen los fotones balísticos (no dispersados), que
se atenúan con ``mu_a + mu_s``: a través de una extremidad su número es
prácticamente cero. Eso explica por qué la luz no permite "ver" el hueso a
través del brazo aunque una linterna lo ilumine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import optical
from ..phantoms.phantom import TISSUE_INDEX, Phantom

NON_TISSUE = ("water", "air")

N_TISSUE = 1.4  # índice de refracción típico de un tejido blando en el visible / NIR
N_OUTER = {"air": 1.0, "water": 1.33}


@dataclass
class DiffusionResult:
    wavelength_nm: float
    fluence: np.ndarray  # W/m² por W/m² de irradiancia incidente
    mu_a: np.ndarray
    mu_s_reduced: np.ndarray
    transmitted_profile: np.ndarray  # exitancia (flujo saliente) por la cara inferior de cada columna
    iterations: int
    residual: float
    floor: float  # por debajo de este valor la fluencia no es numéricamente fiable
    transmission_floor: float  # lo mismo para la exitancia transmitida

    @property
    def transmission(self) -> float:
        """Exitancia media por el borde inferior (relativa a la irradiancia incidente).

        Es el flujo neto que sale del tejido, ``D dPhi/dz`` en la superficie, y no la
        fluencia de la última celda: ésta depende de ``dx`` y sobrestima lo que sale.
        """
        p = self.transmitted_profile
        return float(np.mean(p[np.isfinite(p)])) if np.isfinite(p).any() else 0.0

    @property
    def transmission_reliable(self) -> bool:
        """False si la luz transmitida es tan débil que queda bajo la precisión del solver."""
        return self.transmission > self.transmission_floor

    def transmission_text(self) -> str:
        return f"{self.transmission:.1e}" if self.transmission_reliable else f"< {self.transmission_floor:.0e}"


def optical_maps(phantom: Phantom, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """``(mu_a, mu_s')`` por celda; en las celdas porosas, mezcla lineal por volumen."""
    mua = phantom.property_map(lambda k: float(optical.mu_a(k, wavelength_nm)), physics="optical")
    mus = phantom.property_map(lambda k: float(optical.mu_s_reduced(k, wavelength_nm)), physics="optical")
    return mua, mus


def boundary_parameter(n_outer: float, n_tissue: float = N_TISSUE) -> float:
    """Parámetro ``A = (1 + R_eff) / (1 - R_eff)`` del contorno extrapolado.

    ``R_eff`` es la reflectividad interna efectiva de la interfaz tejido/exterior,
    con el ajuste polinómico de Groenhuis et al. (1983) en ``n = n_tejido / n_exterior``.
    ``A = 1`` si no hay salto de índice.
    """
    n = n_tissue / n_outer
    r_eff = -1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n
    return (1.0 + r_eff) / (1.0 - r_eff)


def _outer_index(phantom: Phantom) -> float:
    """Índice de refracción del medio de fondo: el más abundante entre agua y aire (aire si no hay)."""
    counts = {k: int(np.count_nonzero(phantom.labels == TISSUE_INDEX[k])) for k in NON_TISSUE}
    key = max(counts, key=counts.get)
    return N_OUTER[key] if counts[key] > 0 else N_OUTER["air"]


def _face_diffusion(D: np.ndarray, tissue: np.ndarray, axis: int, dx: float, A: float) -> np.ndarray:
    a = D[:, :-1] if axis == 1 else D[:-1, :]
    b = D[:, 1:] if axis == 1 else D[1:, :]
    ta = tissue[:, :-1] if axis == 1 else tissue[:-1, :]
    tb = tissue[:, 1:] if axis == 1 else tissue[1:, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = 2 * a * b / (a + b)
    face = np.where(ta & tb, hm, 0.0)
    # Vecino fuera: Phi = 0 a z_b = 2 A D de la cara. Entre el centro de la celda y ese
    # punto hay dx/2 + z_b, de modo que la conductancia es D dx / (dx/2 + z_b).
    face = np.where(ta & ~tb, a * dx / (dx / 2 + 2 * A * a), face)
    face = np.where(~ta & tb, b * dx / (dx / 2 + 2 * A * b), face)
    return face


def solve_diffusion(phantom: Phantom, wavelength_nm: float, source: str | tuple[float, float] = "top",
                    tol: float = 1e-13, max_iter: int = 50_000) -> DiffusionResult:
    """Fluencia de luz en el fantoma para iluminación desde arriba (``"top"``) o una fibra en ``(x, z)``.

    El exterior (agua o aire) se toma del fondo del fantoma para calcular el
    contorno extrapolado. Lanza ``ValueError`` si la fibra no está dentro del tejido.
    """
    g = phantom.grid
    dx2 = g.dx**2
    mua, mus = optical_maps(phantom, wavelength_nm)
    tissue = ~np.isin(phantom.labels, [TISSUE_INDEX[k] for k in NON_TISSUE])
    D = np.where(tissue, 1.0 / (3.0 * (mua + mus + 1e-12)), 0.0)
    A = boundary_parameter(_outer_index(phantom))
    fx = _face_diffusion(D, tissue, 1, g.dx, A) / dx2
    fz = _face_diffusion(D, tissue, 0, g.dx, A) / dx2
    diag = mua.copy()
    diag[:, :-1] += fx
    diag[:, 1:] += fx
    diag[:-1, :] += fz
    diag[1:, :] += fz
    diag = np.where(tissue, diag, 1.0)

    def apply(phi):
        out = mua * phi
        flx = fx * (phi[:, 1:] - phi[:, :-1])
        out[:, :-1] -= flx
        out[:, 1:] += flx
        flz = fz * (phi[1:, :] - phi[:-1, :])
        out[:-1, :] -= flz
        out[1:, :] += flz
        return np.where(tissue, out, 0.0)

    b = np.zeros(g.shape)
    if source == "top":
        # Haz ancho colimado: fuente isótropa a un recorrido libre medio de transporte bajo la piel
        has = tissue.any(axis=0)
        first = np.argmax(tissue, axis=0)
        for i in np.nonzero(has)[0]:
            j0 = first[i]
            depth_cells = int(round(1.0 / max(mus[j0, i] + mua[j0, i], 1e-9) / g.dx))
            j = min(j0 + depth_cells, g.nz - 1)
            while j > j0 and not tissue[j, i]:
                j -= 1
            b[j, i] = 1.0 / g.dx
    else:
        j, i = g.index_of(*source)
        if not tissue[j, i]:
            raise ValueError(f"la fuente puntual {tuple(source)} cae fuera del tejido (en agua o aire)")
        b[j, i] = 1.0 / dx2

    # Gradiente conjugado con precondicionador de Jacobi
    phi = np.zeros(g.shape)
    r = b - apply(phi)
    zv = r / diag
    p = zv.copy()
    rz = np.sum(r * zv)
    bnorm = np.linalg.norm(b) + 1e-300
    it = 0
    res = 1.0
    for it in range(1, max_iter + 1):
        ap = apply(p)
        alpha = rz / np.sum(p * ap)
        phi += alpha * p
        r -= alpha * ap
        res = np.linalg.norm(r) / bnorm
        if not (np.isfinite(alpha) and np.isfinite(res)):
            raise RuntimeError(f"el gradiente conjugado divergió en la iteración {it} (valor no finito)")
        if res < tol:
            break
        zv = r / diag
        rz_new = np.sum(r * zv)
        p = zv + (rz_new / rz) * p
        rz = rz_new

    # El error del gradiente conjugado es ~tol relativo al máximo: lo que quede por
    # debajo (incluidos valores negativos) es ruido numérico.
    floor = 10.0 * tol * float(phi.max())
    phi = np.maximum(phi, 0.0)
    last = g.nz - 1 - np.argmax(tissue[::-1, :], axis=0)
    has = tissue.any(axis=0)
    cols = np.arange(g.nx)
    # Exitancia = D Phi_c / (dx/2 + z_b): lo mismo que Phi_frontera / (2A) con Phi_frontera
    # extrapolado linealmente desde el centro de la última celda de tejido.
    d_last = D[last, cols]
    gain = d_last / (g.dx / 2 + 2 * A * d_last)
    prof = np.where(has, phi[last, cols] * gain, np.nan)
    exit_floor = floor * (float(np.max(gain[has])) if has.any() else 1.0)
    return DiffusionResult(wavelength_nm, phi, mua, mus, prof, it, float(res), floor, exit_floor)


def ballistic_transmission(phantom: Phantom, wavelength_nm: float, x: float = 0.0) -> float:
    """Fracción de fotones que cruzan la columna ``x`` de arriba abajo sin dispersarse."""
    _, i = phantom.grid.index_of(x, 0.0)
    mut = {k: 0.0 if k in NON_TISSUE else float(optical.mu_total(k, wavelength_nm)) for k in phantom.property_tissues()}
    col = phantom.property_map(lambda k: mut[k], physics="optical")[:, i]
    return float(np.exp(-np.sum(col) * phantom.grid.dx))
