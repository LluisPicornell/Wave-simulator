"""Simulador de Bloch mínimo (isocromatos sin desfase, pulsos instantáneos).

La resonancia no "atraviesa" el cuerpo como una onda: los protones del agua y
la grasa, alineados por el campo B0, se excitan con pulsos de radiofrecuencia
(64 MHz a 1.5 T) y emiten una señal que decae con T2 y se recupera con T1.
Las ecuaciones de Bloch describen esa magnetización:

    dM/dt = gamma M × B  - (Mx, My, 0)/T2 - (0, 0, Mz - M0)/T1

Aquí se usan para validar las ecuaciones de señal analíticas de
``sequences.py`` (ver ``tests/test_mri.py``).
"""

from __future__ import annotations

import numpy as np


def rotate_x(m: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.stack([m[0], c * m[1] + s * m[2], -s * m[1] + c * m[2]])


def rotate_y(m: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.stack([c * m[0] - s * m[2], m[1], s * m[0] + c * m[2]])


def relax(m: np.ndarray, t: float, t1, t2, m0) -> np.ndarray:
    e1, e2 = np.exp(-t / t1), np.exp(-t / t2)
    return np.stack([m[0] * e2, m[1] * e2, m0 + (m[2] - m0) * e1])


def spin_echo_steady_state(t1, t2, tr: float, te: float, m0=1.0, n_tr: int = 30) -> np.ndarray:
    """|Mxy| en el eco tras ``n_tr`` repeticiones de 90x - TE/2 - 180y - TE/2 (con spoiling al final de TR)."""
    t1, t2, m0 = np.broadcast_arrays(np.asarray(t1, float), np.asarray(t2, float), np.asarray(m0, float))
    m = np.stack([np.zeros_like(t1), np.zeros_like(t1), m0.copy()])
    signal = np.zeros_like(t1)
    for _ in range(n_tr):
        m = rotate_x(m, np.pi / 2)
        m = relax(m, te / 2, t1, t2, m0)
        m = rotate_y(m, np.pi)
        m = relax(m, te / 2, t1, t2, m0)
        signal = np.hypot(m[0], m[1])
        m = relax(m, tr - te, t1, t2, m0)
        m[0] = 0.0  # spoiling perfecto
        m[1] = 0.0
    return signal


def inversion_recovery_steady_state(t1, t2, tr: float, te: float, ti: float, m0=1.0, n_tr: int = 30) -> np.ndarray:
    """|Mxy| de 180 - TI - 90 - TE/2 - 180 - TE/2 - eco (spoiling al final de TR)."""
    t1, t2, m0 = np.broadcast_arrays(np.asarray(t1, float), np.asarray(t2, float), np.asarray(m0, float))
    m = np.stack([np.zeros_like(t1), np.zeros_like(t1), m0.copy()])
    signal = np.zeros_like(t1)
    for _ in range(n_tr):
        m = rotate_x(m, np.pi)
        m = relax(m, ti, t1, t2, m0)
        m = rotate_x(m, np.pi / 2)
        m = relax(m, te / 2, t1, t2, m0)
        m = rotate_y(m, np.pi)
        m = relax(m, te / 2, t1, t2, m0)
        signal = np.hypot(m[0], m[1])
        m = relax(m, tr - ti - te, t1, t2, m0)
        m[0] = 0.0
        m[1] = 0.0
    return signal
