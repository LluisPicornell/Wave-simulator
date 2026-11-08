"""Ondas electromagnéticas en todo el espectro: Maxwell FDTD, óptica, rayos X/gamma."""

from . import spectrum, xray, optics, projection, quasistatic
from .fdtd_tm import EMResult, MaxwellTMSolver, recommended_dx, simulate_em_plane_wave
from .quasistatic import (
    EITResult, Electrodes, QuasiStaticSolver, Sensitivity, adjacent_protocol, reconstruct_difference,
    ring_electrodes, sensitivity_matrix, simulate_eit, surface_electrodes,
)

__all__ = [
    "spectrum", "xray", "optics", "projection", "quasistatic", "EMResult", "MaxwellTMSolver", "recommended_dx",
    "simulate_em_plane_wave", "EITResult", "Electrodes", "QuasiStaticSolver", "Sensitivity", "adjacent_protocol",
    "reconstruct_difference", "ring_electrodes", "sensitivity_matrix", "simulate_eit", "surface_electrodes",
]
