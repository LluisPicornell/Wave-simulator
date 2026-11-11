"""Secuencias de resonancia y sus ecuaciones de señal (derivadas de Bloch).

Suponen pulsos ideales y destrucción completa de la magnetización transversal
al final de cada TR (spoiling), lo que en ``bloch.py`` se verifica numéricamente.
Tiempos en milisegundos.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..materials.mri import FAT_FRACTION, MRI

__all__ = ["FAT_FRACTION", "SEQUENCES", "Sequence", "resolve", "signal", "tissue_signal"]


@dataclass(frozen=True)
class Sequence:
    """``kind``: "SE" (espín-eco), "GRE" (eco de gradiente spoiled) o "IR" (inversión-recuperación)."""

    name: str
    kind: str
    tr: float
    te: float
    ti: float | None = None
    flip_deg: float = 90.0
    fat_sat: bool = False
    fat_sat_residual: float = 0.05


SEQUENCES: dict[str, Sequence] = {
    "T1": Sequence("T1 (espín-eco)", "SE", tr=500.0, te=15.0),
    "T2": Sequence("T2 (espín-eco)", "SE", tr=4000.0, te=100.0),
    "PD": Sequence("Densidad protónica", "SE", tr=3000.0, te=30.0),
    "PD_FS": Sequence("DP con saturación grasa", "SE", tr=3000.0, te=30.0, fat_sat=True),
    "T2_FS": Sequence("T2 con saturación grasa", "SE", tr=4000.0, te=80.0, fat_sat=True),
    # TI = None: se calcula para anular la grasa al campo de trabajo
    "STIR": Sequence("STIR (supresión grasa por inversión)", "IR", tr=4000.0, te=40.0, ti=None),
    "GRE": Sequence("Eco de gradiente T2*", "GRE", tr=30.0, te=5.0, flip_deg=20.0),
    # Eco ultracorto: el hueso cortical (T2* < 1 ms) deja de ser negro
    "UTE": Sequence("UTE (eco ultracorto)", "GRE", tr=10.0, te=0.05, flip_deg=10.0),
}


def resolve(seq: Sequence | str, b0: float) -> Sequence:
    seq = SEQUENCES[seq] if isinstance(seq, str) else seq
    if seq.kind == "IR" and seq.ti is None:
        t1_fat = MRI["fat"].at_field(b0)[1]
        seq = replace(seq, ti=float(t1_fat * np.log(2.0)))
    return seq


def signal(pd, t1, t2, t2s, seq: Sequence):
    """Magnitud de la señal (unidades arbitrarias, PD del agua = 1)."""
    pd, t1, t2, t2s = (np.asarray(v, dtype=float) for v in (pd, t1, t2, t2s))
    if seq.kind == "SE":
        long = 1 - 2 * np.exp(-(seq.tr - seq.te / 2) / t1) + np.exp(-seq.tr / t1)
        return pd * long * np.exp(-seq.te / t2)
    if seq.kind == "IR":
        long = 1 - 2 * np.exp(-seq.ti / t1) + 2 * np.exp(-(seq.tr - seq.te / 2) / t1) - np.exp(-seq.tr / t1)
        return pd * np.abs(long) * np.exp(-seq.te / t2)
    if seq.kind == "GRE":
        a = np.radians(seq.flip_deg)
        e1 = np.exp(-seq.tr / t1)
        return pd * np.sin(a) * (1 - e1) / (1 - np.cos(a) * e1) * np.exp(-seq.te / t2s)
    raise ValueError(f"Tipo de secuencia desconocido: {seq.kind}")


def tissue_signal(tissue: str, seq: Sequence | str, b0: float = 1.5) -> float:
    seq = resolve(seq, b0)
    pd, t1, t2, t2s = MRI[tissue].at_field(b0)
    s = float(signal(pd, t1, t2, t2s, seq))
    if seq.fat_sat:
        ff = FAT_FRACTION.get(tissue, 0.0)
        s *= 1 - ff * (1 - seq.fat_sat_residual)
    return s
