"""Reconstrucción de la imagen de ecografía (modo B) a partir de las señales RF.

Delay-and-sum (DAS) para emisiones de onda plana (Montaldo et al., IEEE
UFFC 56, 489, 2009): para cada píxel se calcula cuándo debería llegar su eco a
cada elemento suponiendo una velocidad del sonido constante ``c0`` (1540 m/s,
como un ecógrafo real). Las diferencias reales de velocidad entre tejidos
(grasa 1450, hueso 3500 m/s) producen las mismas distorsiones que en la
clínica: desenfoque y desplazamiento de ecos bajo el hueso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.signals import analytic_signal


@dataclass
class BModeImage:
    envelope: np.ndarray  # amplitud lineal (nz, nx)
    x: np.ndarray
    z: np.ndarray
    dynamic_range_db: float = 50.0

    @property
    def db(self) -> np.ndarray:
        env = self.envelope / (self.envelope.max() + 1e-30)
        return 20.0 * np.log10(np.maximum(env, 10 ** (-self.dynamic_range_db / 20)))

    def extent_mm(self) -> tuple[float, float, float, float]:
        dx = self.x[1] - self.x[0] if self.x.size > 1 else 0
        dz = self.z[1] - self.z[0] if self.z.size > 1 else 0
        return ((self.x[0] - dx / 2) * 1e3, (self.x[-1] + dx / 2) * 1e3,
                (self.z[-1] + dz / 2) * 1e3, (self.z[0] - dz / 2) * 1e3)


def das_plane_wave(
    rf: np.ndarray,
    t: np.ndarray,
    element_x: np.ndarray,
    array_z: float,
    angle_deg: float,
    x_img: np.ndarray,
    z_img: np.ndarray,
    c0: float = 1540.0,
    t_peak: float = 0.0,
    f_number: float = 1.5,
) -> np.ndarray:
    """Imagen compleja (señal analítica) de una emisión de onda plana.

    ``rf``: (n_elementos, nt) muestreada en los instantes ``t`` (uniformes).
    ``t_peak``: instante del centro del pulso emitido (referencia temporal).
    """
    iq = analytic_signal(rf, axis=1)
    dt = t[1] - t[0]
    theta = np.radians(angle_deg)
    s, c = np.sin(theta), np.cos(theta)
    xr = element_x.min() if s >= 0 else element_x.max()
    X, Z = np.meshgrid(x_img, z_img)
    depth = Z - array_z
    t_tx = ((X - xr) * s + depth * c) / c0 + t_peak
    img = np.zeros(X.shape, dtype=complex)
    weight = np.zeros(X.shape)
    nt = rf.shape[1]
    for e, xe in enumerate(element_x):
        lateral = X - xe
        # Apertura dinámica en recepción (f-number) con ventana de Hann
        half_ap = np.maximum(depth, 1e-6) / (2.0 * f_number)
        u = np.abs(lateral) / half_ap
        w = np.where(u <= 1.0, 0.5 * (1 + np.cos(np.pi * u)), 0.0)
        if not np.any(w):
            continue
        tau = t_tx + np.hypot(lateral, depth) / c0
        idx = (tau - t[0]) / dt
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        valid = (i0 >= 0) & (i0 < nt - 1)
        i0c = np.clip(i0, 0, nt - 2)
        sample = iq[e, i0c] * (1 - frac) + iq[e, i0c + 1] * frac
        img += np.where(valid, w * sample, 0.0)
        weight += w
    return img / np.maximum(weight, 1e-12)


def bmode(
    rf_list: list[np.ndarray],
    angles_deg: list[float],
    t: np.ndarray,
    element_x: np.ndarray,
    array_z: float,
    x_img: np.ndarray,
    z_img: np.ndarray,
    c0: float = 1540.0,
    t_peak: float = 0.0,
    blank_until: float = 0.0,
    tgc_db_per_cm: float = 0.0,
    f_number: float = 1.5,
    dynamic_range_db: float = 50.0,
) -> BModeImage:
    """Imagen modo B compuesta (coherente) a partir de varias ondas planas.

    ``blank_until``: se anulan las muestras anteriores (pulso directo emitido).
    ``tgc_db_per_cm``: compensación de ganancia con la profundidad (ida y vuelta).
    """
    img = np.zeros((z_img.size, x_img.size), dtype=complex)
    for rf, ang in zip(rf_list, angles_deg, strict=True):
        rf = np.array(rf, dtype=float)
        rf[:, t < blank_until] = 0.0
        img += das_plane_wave(rf, t, element_x, array_z, ang, x_img, z_img, c0, t_peak, f_number)
    env = np.abs(img)
    if tgc_db_per_cm:
        depth_cm = np.maximum(z_img - array_z, 0.0) * 100.0
        env *= (10 ** (tgc_db_per_cm * depth_cm / 20.0))[:, None]
    return BModeImage(env, x_img, z_img, dynamic_range_db)
