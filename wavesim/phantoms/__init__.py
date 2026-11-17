"""Fantomas digitales de extremidades (con y sin fractura)."""

from .phantom import Phantom, TISSUE_INDEX
from .limb import (
    Fracture, LimbSpec, PRESETS, longitudinal_limb,
    RingFracture, BoneSpec, CrossSectionSpec, CROSS_PRESETS, cross_section_limb,
    FRACTURE_KINDS, CORTICAL_FRACTURE_KINDS, INTERNAL_FRACTURE_KINDS, RING_FRACTURE_KINDS,
    random_fracture, random_ring_fracture, random_limb_spec,
    BodyType, BODY_TYPES, with_body_type, random_body_type,
)
from .limb3d import Phantom3D, extrude, homogeneous_phantom, leg_segment
from .limb3d_full import (
    Metaphysis, Limb3DSpec, LIMB3D_PRESETS, Fracture3D, FRACTURE_KINDS_3D, limb3d_with_body_type,
    random_limb_3d_spec, random_fracture_3d, limb_grid, limb_3d, limb_3d_fractions, equivalent_longitudinal_spec,
)
from .realism import (
    RealismSpec, REALISM_PRESETS, CorticalPorosity, TrabecularNetwork, FractalRoughness, SoftTissueStructure,
)

__all__ = [
    "Phantom", "TISSUE_INDEX", "Fracture", "LimbSpec", "PRESETS", "longitudinal_limb",
    "RingFracture", "BoneSpec", "CrossSectionSpec", "CROSS_PRESETS", "cross_section_limb",
    "FRACTURE_KINDS", "CORTICAL_FRACTURE_KINDS", "INTERNAL_FRACTURE_KINDS", "RING_FRACTURE_KINDS",
    "random_fracture", "random_ring_fracture", "random_limb_spec",
    "BodyType", "BODY_TYPES", "with_body_type", "random_body_type",
    "RealismSpec", "REALISM_PRESETS", "CorticalPorosity", "TrabecularNetwork", "FractalRoughness",
    "SoftTissueStructure",
    "Phantom3D", "extrude", "homogeneous_phantom", "leg_segment",
    "Metaphysis", "Limb3DSpec", "LIMB3D_PRESETS", "Fracture3D", "FRACTURE_KINDS_3D", "limb3d_with_body_type",
    "random_limb_3d_spec", "random_fracture_3d", "limb_grid", "limb_3d", "limb_3d_fractions",
    "equivalent_longitudinal_spec",
]
