"""Kernels Numba del FDTD de Maxwell TM (backend opcional de ``MaxwellTMSolver`` y del solver de banda ancha).

El backend NumPy da muchas pasadas por la malla por paso (una por derivada,
por producto y por suma, más la CPML y los polos de Debye) y está limitado por
el ancho de banda de memoria. Aquí cada paso son dos barridos fusionados,
paralelizados por filas:

  1. ``step_h``: derivadas de Ey con CPML y avance de Hx, Hz.
  2. ``step_e`` / ``step_e_debye``: derivadas de H con CPML, rotacional y
     actualización de Ey (con las corrientes de polarización J_k de los polos
     de Debye en la variante de banda ancha).

Cada celda sólo escribe en sí misma (campo, corrientes y psi de la CPML en su
posición) y lee campos que ese barrido no modifica: no hay carreras con
``prange`` sobre filas. Para que el resultado coincida con NumPy salvo
redondeo (float64) no se usa ``fastmath`` y las derivadas sólo existen en las
celdas que escribe el backend NumPy (hacia delante ``[0, n-2]``, hacia atrás
``[1, n-1]``); fuera valen cero, y allí la memoria CPML permanece en cero.

La fuente se aplica fuera de los kernels (en Python, igual que en NumPy):
sobre las celdas de la fuente se corrige después ``Ey`` y, en banda ancha, la
corriente de polarización, que es lineal en ``Ey^{n+1}``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..ultrasound._numba_kernels import pack_pml  # noqa: F401  (se reexporta para el solver)


@njit(cache=True, parallel=True)
def step_h(ey, hx, hz, inv_dx, ch, pz, px):
    """Avanza Hx y Hz medio paso con ``dEy/dz`` y ``dEy/dx`` (diferencias hacia delante) y CPML.

    Los bucles se separan por tramos (CPML / interior) para que el interior, que es casi toda la
    malla, quede sin ramas y Numba lo vectorice.
    """
    nz, nx = ey.shape
    for jp in prange(nz):
        j = np.int64(jp)  # el índice de prange es uint64: restar un int64 daría float64
        if j < nz - 1:
            if j < pz[0] or j >= pz[1]:
                # Franja CPML en z: la fila entera lleva memoria psi
                left = j < pz[0]
                k = j if left else j - pz[1]
                a_row = pz[2][k] if left else pz[4][k]
                b_row = pz[3][k] if left else pz[5][k]
                psi_row = pz[6][k] if left else pz[7][k]
                for i in range(nx):
                    d = (ey[j + 1, i] - ey[j, i]) * inv_dx
                    psi = b_row[i] * psi_row[i] + a_row[i] * d
                    psi_row[i] = psi
                    d += psi
                    hx[j, i] += ch * d
            else:
                for i in range(nx):
                    hx[j, i] += ch * ((ey[j + 1, i] - ey[j, i]) * inv_dx)
        # dEy/dx: tramo CPML izquierdo, interior y tramo CPML derecho
        nl = min(px[0], nx - 1)
        for i in range(nl):
            d = (ey[j, i + 1] - ey[j, i]) * inv_dx
            psi = px[3][j, i] * px[6][j, i] + px[2][j, i] * d
            px[6][j, i] = psi
            d += psi
            hz[j, i] -= ch * d
        r0 = max(px[1], nl)
        for i in range(nl, min(r0, nx - 1)):
            hz[j, i] -= ch * ((ey[j, i + 1] - ey[j, i]) * inv_dx)
        for i in range(r0, nx - 1):
            k = i - px[1]
            d = (ey[j, i + 1] - ey[j, i]) * inv_dx
            psi = px[5][j, k] * px[7][j, k] + px[4][j, k] * d
            px[7][j, k] = psi
            d += psi
            hz[j, i] -= ch * d


@njit(cache=True, inline="always")
def _curl_row(hx, hz, j, inv_dx, pz, px, c):
    """``dHx/dz - dHz/dx`` (diferencias hacia atrás, con CPML) de la fila ``j`` de nodos de Ey, en ``c``."""
    nz, nx = hx.shape
    if j > 0:
        if j < pz[0] or j >= pz[1]:
            left = j < pz[0]
            k = j if left else j - pz[1]
            a_row = pz[2][k] if left else pz[4][k]
            b_row = pz[3][k] if left else pz[5][k]
            psi_row = pz[6][k] if left else pz[7][k]
            for i in range(nx):
                d = (hx[j, i] - hx[j - 1, i]) * inv_dx
                psi = b_row[i] * psi_row[i] + a_row[i] * d
                psi_row[i] = psi
                c[i] = d + psi
        else:
            for i in range(nx):
                c[i] = (hx[j, i] - hx[j - 1, i]) * inv_dx
    else:
        for i in range(nx):
            c[i] = 0.0
    # dHz/dx existe a partir de i = 1 (en i = 0 vale cero)
    nl = min(px[0], nx)
    for i in range(1, nl):
        d = (hz[j, i] - hz[j, i - 1]) * inv_dx
        psi = px[3][j, i] * px[6][j, i] + px[2][j, i] * d
        px[6][j, i] = psi
        c[i] -= d + psi
    r0 = max(px[1], nl, 1)
    for i in range(max(nl, 1), min(r0, nx)):
        c[i] -= (hz[j, i] - hz[j, i - 1]) * inv_dx
    for i in range(r0, nx):
        k = i - px[1]
        d = (hz[j, i] - hz[j, i - 1]) * inv_dx
        psi = px[5][j, k] * px[7][j, k] + px[4][j, k] * d
        px[7][j, k] = psi
        c[i] -= d + psi


@njit(cache=True, parallel=True)
def step_e(ey, hx, hz, ca, cb, inv_dx, pz, px):
    """Actualiza Ey con el rotacional de H (medio de banda estrecha: sólo ``ca`` y ``cb``)."""
    nz, nx = ey.shape
    for jp in prange(nz):
        j = np.int64(jp)  # el índice de prange es uint64: restar un int64 daría float64
        c = np.empty(nx)
        _curl_row(hx, hz, j, inv_dx, pz, px, c)
        for i in range(nx):
            ey[j, i] = ey[j, i] * ca[j, i] + cb[j, i] * c[i]


@njit(cache=True, parallel=True)
def step_e_debye(ey, hx, hz, ca, cb, inv_dx, pz, px, jpol, kappa, beta, gk):
    """Actualiza Ey y las corrientes de polarización J_k de los polos de Debye (ADE).

    ``curl -= sum_k g_k J_k``; ``Ey <- ca Ey + cb curl``; ``J_k <- kappa_k J_k + beta_k (Ey^{n+1} - Ey^n)``.
    """
    nz, nx = ey.shape
    npol = jpol.shape[0]
    for jp in prange(nz):
        j = np.int64(jp)  # el índice de prange es uint64: restar un int64 daría float64
        c = np.empty(nx)
        de = np.empty(nx)
        _curl_row(hx, hz, j, inv_dx, pz, px, c)
        for k in range(npol):
            for i in range(nx):
                c[i] -= gk[k] * jpol[k, j, i]
        for i in range(nx):
            old = ey[j, i]
            new = old * ca[j, i] + cb[j, i] * c[i]
            ey[j, i] = new
            de[i] = new - old
        for k in range(npol):
            for i in range(nx):
                jpol[k, j, i] = jpol[k, j, i] * kappa[k] + beta[k, j, i] * de[i]


@njit(cache=True, parallel=True)
def accumulate_dft(acc, ey, fr, fi):
    """``acc += Ey * (fr + i fi)``: una muestra de la DFT de un solo punto."""
    nz, nx = ey.shape
    for jp in prange(nz):
        j = np.int64(jp)  # el índice de prange es uint64: restar un int64 daría float64
        for i in range(nx):
            acc[j, i] += ey[j, i] * complex(fr, fi)
