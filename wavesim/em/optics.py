"""Luz (UV, visible, infrarrojo) en tejido: difusión y fotones balísticos.

En el visible y el infrarrojo cercano los tejidos dispersan la luz cientos de
veces por milímetro (``mu_s' >> mu_a``): los fotones pierden su dirección y
la luz se propaga como calor difundiéndose. Se resuelve la aproximación de
difusión estacionaria:

    -div(D grad(Phi)) + mu_a Phi = S,     D = 1 / (3 (mu_a + mu_s'))

con diferencias finitas y gradiente conjugado precondicionado. Fuera del
tejido (agua o aire de fondo) ``Phi = 0`` (aproximación de contorno de flujo
nulo extrapolado a una celda).

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


@dataclass
class DiffusionResult:
    wavelength_nm: float
    fluence: np.ndarray  # W/m² por W/m² de irradiancia incidente
    mu_a: np.ndarray
    mu_s_reduced: np.ndarray
    transmitted_profile: np.ndarray  # fluencia en la última celda de tejido de cada columna
    iterations: int
    residual: float
    floor: float  # por debajo de este valor la fluencia no es numéricamente fiable

    @property
    def transmission(self) -> float:
        """Fluencia media transmitida al lado opuesto (relativa a la irradiancia incidente)."""
        p = self.transmitted_profile
        return float(np.mean(p[np.isfinite(p)])) if np.isfinite(p).any() else 0.0

    @property
    def transmission_reliable(self) -> bool:
        """False si la luz transmitida es tan débil que queda bajo la precisión del solver."""
        return self.transmission > self.floor

    def transmission_text(self) -> str:
        return f"{self.transmission:.1e}" if self.transmission_reliable else f"< {self.floor:.0e}"


def optical_maps(phantom: Phantom, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
    mua = phantom.property_map(lambda k: float(optical.mu_a(k, wavelength_nm)))
    mus = phantom.property_map(lambda k: float(optical.mu_s_reduced(k, wavelength_nm)))
    return mua, mus


def _face_diffusion(D: np.ndarray, tissue: np.ndarray, axis: int) -> np.ndarray:
    a = D[:, :-1] if axis == 1 else D[:-1, :]
    b = D[:, 1:] if axis == 1 else D[1:, :]
    ta = tissue[:, :-1] if axis == 1 else tissue[:-1, :]
    tb = tissue[:, 1:] if axis == 1 else tissue[1:, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = 2 * a * b / (a + b)
    face = np.where(ta & tb, hm, 0.0)
    face = np.where(ta & ~tb, a, face)  # vecino fuera: Phi = 0 en la celda vecina
    face = np.where(~ta & tb, b, face)
    return face


def solve_diffusion(phantom: Phantom, wavelength_nm: float, source: str | tuple[float, float] = "top",
                    tol: float = 1e-13, max_iter: int = 50_000) -> DiffusionResult:
    """Fluencia de luz en el fantoma para iluminación desde arriba (``"top"``) o una fibra en ``(x, z)``."""
    g = phantom.grid
    dx2 = g.dx**2
    mua, mus = optical_maps(phantom, wavelength_nm)
    tissue = ~np.isin(phantom.labels, [TISSUE_INDEX[k] for k in NON_TISSUE])
    D = np.where(tissue, 1.0 / (3.0 * (mua + mus + 1e-12)), 0.0)
    fx = _face_diffusion(D, tissue, 1) / dx2
    fz = _face_diffusion(D, tissue, 0) / dx2
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
    prof = np.where(has, phi[last, np.arange(g.nx)], np.nan)
    return DiffusionResult(wavelength_nm, phi, mua, mus, prof, it, float(res), floor)


def ballistic_transmission(phantom: Phantom, wavelength_nm: float, x: float = 0.0) -> float:
    """Fracción de fotones que cruzan la columna ``x`` de arriba abajo sin dispersarse."""
    _, i = phantom.grid.index_of(x, 0.0)
    keys = [k for k in phantom.present_tissues() if k not in NON_TISSUE]
    mut = {k: float(optical.mu_total(k, wavelength_nm)) for k in keys}
    col = phantom.labels[:, i]
    total = 0.0
    for k in keys:
        total += mut[k] * np.count_nonzero(col == TISSUE_INDEX[k]) * phantom.grid.dx
    return float(np.exp(-total))
