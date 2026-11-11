"""Multigrid de agregación para operadores de 5 puntos con coeficientes muy contrastados (precondicionador de CG).

Los problemas de ``em.quasistatic`` (admitividad de piel, grasa, músculo y hueso, que difieren en 2-3 órdenes de
magnitud) y de ``em.optics`` (difusión) dan matrices simétricas ``A = D - F`` de 5 puntos con conductancias de cara
``F`` positivas. Jacobi converge en miles de iteraciones porque las componentes suaves del error apenas se
amortiguan; un ciclo en V las elimina en las mallas gruesas.

Construcción (algebraica, sin hipótesis geométricas sobre los coeficientes):

- Los agregados son bloques 2x2 de celdas; la interpolación ``P`` es constante a trozos.
- La matriz gruesa es la de Galerkin, ``A_c = P^T A P``. Para un operador de 5 puntos eso es de nuevo un operador
  de 5 puntos: la conductancia entre dos agregados vecinos es la **suma** de las caras finas que los separan (el
  paralelo de las dos conductancias), la diagonal es la suma de las diagonales finas menos el doble de las caras
  interiores. Así los saltos de coeficiente se propagan tal cual a los niveles gruesos, sin promediar a mano, y los
  contornos irregulares (celdas fuera del dominio, electrodos fijos) también.
- Suavizador: Gauss-Seidel rojo-negro (rojo, negro antes de bajar; negro, rojo al subir), que hace simétrico el
  ciclo: es un precondicionador válido para CG/COCG. Con una interpolación constante a trozos la corrección gruesa
  se queda corta; se multiplica por ``over`` (1.8) como en la agregación sin suavizar.
- En el nivel más grueso (<= ``coarse_cells`` celdas) se aplica la pseudoinversa densa (el operador de Neumann es
  singular, con el vector constante en el núcleo).

Necesita Numba (``wavesim[fast]``); ``QuasiStaticSolver`` y ``solve_diffusion`` usan Jacobi sin él.
"""

from __future__ import annotations

import numpy as np

from . import _numba_mg as nk


class Multigrid:
    """Ciclo en V como precondicionador: ``apply(r)`` devuelve una aproximación de ``A^-1 r``.

    ``fx`` y ``fz`` (formas ``(nz, nx - 1)`` y ``(nz - 1, nx)``) son las conductancias entre celdas **activas** (nulas
    si alguna no lo es); ``diag`` es la diagonal completa de ``A`` (puede incluir caras hacia celdas fijas o
    el contorno extrapolado, que no aparecen fuera de la diagonal). Las celdas con ``diag == 0`` quedan fuera.
    """

    def __init__(self, fx: np.ndarray, fz: np.ndarray, diag: np.ndarray, coarse_cells: int = 64,
                 over: float = 1.8, n_pre: int = 2, n_post: int = 2, gamma: int = 1,
                 shift: float = 1e-8):
        dtype = np.result_type(fx, fz, diag, float)
        nz, nx = diag.shape
        fxp = np.zeros((nz, nx), dtype)
        fzp = np.zeros((nz, nx), dtype)
        fxp[:, :-1], fzp[:-1, :] = fx, fz
        # Un operador de Neumann puro es singular (el vector constante está en su núcleo) y Gauss-Seidel y la
        # corrección gruesa amplifican la componente constante que el redondeo deja en el residuo: se regulariza
        # el precondicionador (no el sistema) con un pequeño término a tierra relativo a la diagonal.
        d = np.asarray(diag, dtype) * (1.0 + shift)
        self.levels = []
        while True:
            invd = np.zeros_like(d)
            np.divide(1.0, d, out=invd, where=d != 0)
            self.levels.append((fxp, fzp, d, invd))
            if d.size <= coarse_cells or min(d.shape) < 4:
                break
            fxp, fzp, d = self._coarsen(fxp, fzp, d)
        self.over, self.n_pre, self.n_post, self.gamma = over, n_pre, n_post, gamma
        # Pseudoinversa densa del nivel más grueso
        fxc, fzc, dc, _ = self.levels[-1]
        n = dc.size
        idx = np.arange(n).reshape(dc.shape)
        mat = np.diag(dc.ravel()).astype(dtype)
        mat[idx[:, :-1].ravel(), idx[:, 1:].ravel()] -= fxc[:, :-1].ravel()
        mat[idx[:, 1:].ravel(), idx[:, :-1].ravel()] -= fxc[:, :-1].ravel()
        mat[idx[:-1, :].ravel(), idx[1:, :].ravel()] -= fzc[:-1, :].ravel()
        mat[idx[1:, :].ravel(), idx[:-1, :].ravel()] -= fzc[:-1, :].ravel()
        self._coarse_inv = np.linalg.pinv(mat)

    @staticmethod
    def _coarsen(fx, fz, d):
        """Operador de Galerkin sobre agregados 2x2: caras en paralelo, diagonal menos el doble de las interiores."""
        nz, nx = d.shape
        mz, mx = -(-nz // 2), -(-nx // 2)
        pad = ((0, 2 * mz - nz), (0, 2 * mx - nx))
        fx, fz, dp = np.pad(fx, pad), np.pad(fz, pad), np.pad(d, pad)

        def blocks(a):
            return a.reshape(mz, 2, mx, 2)

        # Caras interiores: horizontales entre columnas (2I, 2I+1) y verticales entre filas (2J, 2J+1)
        inner = blocks(fx)[:, :, :, 0].sum(axis=1) + blocks(fz)[:, 0, :, :].sum(axis=2)
        dc = blocks(dp).sum(axis=(1, 3)) - 2.0 * inner
        # Caras entre agregados: columna impar (2I+1 -> 2I+2) y fila impar (2J+1 -> 2J+2), sumando las 2 filas/columnas
        fxc = blocks(fx)[:, :, :, 1].sum(axis=1)
        fzc = blocks(fz)[:, 1, :, :].sum(axis=2)
        fxc[:, -1] = 0.0
        fzc[-1, :] = 0.0
        return fxc, fzc, dc

    def _cycle(self, k: int, b: np.ndarray, x: np.ndarray) -> None:
        """Ciclo desde el nivel ``k`` con el segundo miembro ``b``; deja la aproximación en ``x``."""
        fx, fz, d, invd = self.levels[k]
        if k == len(self.levels) - 1:
            nk.dense_apply(x.reshape(-1), self._coarse_inv, b.reshape(-1))
            return
        x[:] = 0.0
        for _ in range(self.n_pre):
            nk.gs_color(x, b, fx, fz, invd, 0)
            nk.gs_color(x, b, fx, fz, invd, 1)
        r = np.empty_like(x)
        nk.residual(r, b, x, fx, fz, d)
        fxc, fzc, dc, _ = self.levels[k + 1]
        bc = np.empty_like(dc)
        nk.restrict_sum(bc, r)
        xc = np.empty_like(dc)
        self._cycle(k + 1, bc, xc)
        for _ in range(self.gamma - 1):
            # Ciclo W: segunda visita al nivel grueso sobre el residuo que dejó la primera
            nk.residual(bc, bc, xc, fxc, fzc, dc)
            dx = np.empty_like(dc)
            self._cycle(k + 1, bc, dx)
            xc += dx
        nk.prolong_add(x, xc, self.over)
        for _ in range(self.n_post):
            nk.gs_color(x, b, fx, fz, invd, 1)
            nk.gs_color(x, b, fx, fz, invd, 0)

    def apply(self, r: np.ndarray) -> np.ndarray:
        """Aproximación de ``A^-1 r`` con un ciclo en V."""
        r = np.ascontiguousarray(r, dtype=self.levels[0][2].dtype)
        out = np.empty_like(r)
        self._cycle(0, r, out)
        return out
