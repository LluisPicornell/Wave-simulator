"""Ondas electromagnéticas en todo el espectro: Maxwell FDTD, óptica, rayos X/gamma."""

from . import (
    spectrum, xray, optics, montecarlo, projection, quasistatic, analytic, bessel, subpixel, fanbeam,
    xray_phantoms, xray_scatter, xray_detector,
)
from .fdtd_tm import EMResult, MaxwellTMSolver, recommended_dx, simulate_em_plane_wave
from .fdtd_dispersive import BroadbandEMResult, BroadbandMaxwellTMSolver, wideband_pulse
from .quasistatic import (
    EITResult, Electrodes, QuasiStaticSolver, Sensitivity, adjacent_protocol, reconstruct_difference,
    ring_electrodes, sensitivity_matrix, simulate_eit, surface_electrodes,
)

__all__ = [
    "spectrum", "xray", "optics", "montecarlo", "projection", "quasistatic", "analytic", "bessel", "subpixel",
    "fanbeam", "xray_phantoms", "xray_scatter", "xray_detector",
    "EMResult", "MaxwellTMSolver", "recommended_dx", "BroadbandEMResult", "BroadbandMaxwellTMSolver",
    "wideband_pulse",
    "simulate_em_plane_wave", "EITResult", "Electrodes", "QuasiStaticSolver", "Sensitivity", "adjacent_protocol",
    "reconstruct_difference", "ring_electrodes", "sensitivity_matrix", "simulate_eit", "surface_electrodes",
]
