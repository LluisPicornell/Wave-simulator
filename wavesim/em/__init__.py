"""Ondas electromagnéticas en todo el espectro: Maxwell FDTD, óptica, rayos X/gamma."""

from . import spectrum, xray, optics, projection
from .fdtd_tm import EMResult, MaxwellTMSolver, recommended_dx, simulate_em_plane_wave

__all__ = [
    "spectrum", "xray", "optics", "projection", "EMResult", "MaxwellTMSolver", "recommended_dx",
    "simulate_em_plane_wave",
]
