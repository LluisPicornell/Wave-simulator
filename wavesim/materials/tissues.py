"""Registro central de tejidos.

Cada tejido tiene una clave (``"cortical_bone"``...), un nombre legible y un
color para las figuras. Sus propiedades físicas viven en un módulo por física
(``acoustic``, ``dielectric``, ``xray``, ``optical``, ``mri``), cada uno con sus
propias fuentes bibliográficas. ``tests/test_materials.py`` comprueba que todas
las tablas cubren todos los tejidos de este registro.

Para añadir un tejido nuevo: añádelo aquí y en cada tabla de propiedades.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TissueInfo:
    key: str
    name: str
    color: str


TISSUES: dict[str, TissueInfo] = {
    t.key: t
    for t in [
        TissueInfo("water", "agua / gel de acoplamiento", "#9ecae1"),
        TissueInfo("air", "aire", "#ffffff"),
        TissueInfo("skin", "piel", "#e7a98b"),
        TissueInfo("fat", "grasa subcutánea", "#f6e27f"),
        TissueInfo("muscle", "músculo", "#c0392b"),
        TissueInfo("blood", "sangre (vaso)", "#7b0d1e"),
        TissueInfo("cortical_bone", "hueso cortical", "#f4f1e8"),
        TissueInfo("trabecular_bone", "hueso trabecular", "#c9b99a"),
        TissueInfo("marrow", "médula ósea amarilla", "#e9c46a"),
        TissueInfo("hematoma", "hematoma (sangre en la fractura)", "#5e2a84"),
        TissueInfo("edema", "médula con edema óseo", "#2a9d8f"),
    ]
}


def tissue_keys() -> list[str]:
    return list(TISSUES)


def check_known(key: str) -> None:
    if key not in TISSUES:
        raise KeyError(f"Tejido desconocido '{key}'. Disponibles: {', '.join(TISSUES)}")
