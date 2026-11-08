"""Biblioteca de tejidos humanos y sus propiedades físicas por modalidad."""

from .tissues import TISSUES, TissueInfo, tissue_keys
from .acoustic import ACOUSTIC, AcousticProps, reflection_coefficient, intensity_transmission
from . import dielectric, xray, optical, mri

__all__ = [
    "TISSUES", "TissueInfo", "tissue_keys", "ACOUSTIC", "AcousticProps",
    "reflection_coefficient", "intensity_transmission", "dielectric", "xray", "optical", "mri",
]
