"""Kernel de lectura del simulador de Bloch espacial compilado con Numba (opcional).

Se importa sólo si se pide (o hay) Numba: ``spatial._resolve_backend``.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def readout_rows(w, taus, r2p, zfac, rows, n_rows, n_samples, dwell):
    """Igual que ``spatial._readout_numpy``: suma por filas de la señal de cada isocromato.

    ``out[rows[j], k] += sum_s w[s, j] exp(-r2p[j] |taus[s] + k dwell|) zfac[j]**k``.
    Un solo hilo: la suma por filas en paralelo necesitaría acumuladores
    privados y la máquina de referencia se comparte con otros procesos.
    """
    n_states, n = w.shape
    out = np.zeros((n_rows, n_samples), dtype=np.complex128)
    for j in range(n):
        zj = zfac[j]
        rj = r2p[j]
        row = rows[j]
        for s in range(n_states):
            f = w[s, j]
            if f == 0:
                continue
            tau = taus[s]
            zk = 1.0 + 0.0j
            for k in range(n_samples):
                out[row, k] += f * zk * np.exp(-rj * abs(tau + k * dwell))
                zk *= zj
    return out
