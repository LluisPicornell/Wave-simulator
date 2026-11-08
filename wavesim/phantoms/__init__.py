"""Fantomas digitales de extremidades (con y sin fractura)."""

from .phantom import Phantom, TISSUE_INDEX
from .limb import (
    Fracture, LimbSpec, PRESETS, longitudinal_limb,
    RingFracture, BoneSpec, CrossSectionSpec, CROSS_PRESETS, cross_section_limb,
    FRACTURE_KINDS, random_fracture, random_limb_spec,
)

__all__ = [
    "Phantom", "TISSUE_INDEX", "Fracture", "LimbSpec", "PRESETS", "longitudinal_limb",
    "RingFracture", "BoneSpec", "CrossSectionSpec", "CROSS_PRESETS", "cross_section_limb",
    "FRACTURE_KINDS", "random_fracture", "random_limb_spec",
]
