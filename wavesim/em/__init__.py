"""Ondas electromagnéticas en todo el espectro: Maxwell FDTD, óptica, rayos X/gamma."""

from . import (
    fdtd3d, spectrum, xray, optics, montecarlo, projection, quasistatic, analytic, bessel, subpixel, eit, fanbeam,
    xray_phantoms, xray_scatter, xray_detector, xray3d, xray3d_scatter,
)
from .fdtd_tm import EMResult, MaxwellTMSolver, recommended_dx, simulate_em_plane_wave
from .fdtd_dispersive import BroadbandEMResult, BroadbandMaxwellTMSolver, wideband_pulse
from .fdtd3d import (
    BroadbandEM3DResult, BroadbandMaxwell3DSolver, EM3DResult, Maxwell3DSolver, averaged_sar,
    simulate_em3d_plane_wave,
)
from .quasistatic import (
    ContactModel, EITResult, Electrodes, QuasiStaticSolver, Sensitivity, adjacent_protocol, reconstruct_difference,
    build_contact, ring_electrodes, sensitivity_matrix, simulate_eit, surface_electrodes,
)

__all__ = [
    "spectrum", "xray", "optics", "montecarlo", "projection", "quasistatic", "analytic", "bessel", "subpixel", "eit",
    "fanbeam", "xray_phantoms", "xray_scatter", "xray_detector", "xray3d", "xray3d_scatter",
    "EMResult", "MaxwellTMSolver", "recommended_dx", "BroadbandEMResult", "BroadbandMaxwellTMSolver",
    "wideband_pulse", "fdtd3d", "EM3DResult", "Maxwell3DSolver", "BroadbandEM3DResult", "BroadbandMaxwell3DSolver",
    "averaged_sar", "simulate_em3d_plane_wave",
    "simulate_em_plane_wave", "ContactModel", "build_contact", "EITResult", "Electrodes", "QuasiStaticSolver",
    "Sensitivity", "adjacent_protocol",
    "reconstruct_difference", "ring_electrodes", "sensitivity_matrix", "simulate_eit", "surface_electrodes",
]
