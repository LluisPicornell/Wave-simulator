"""Datasets etiquetados para entrenar detectores de fracturas."""

from .generate import MODALITIES, generate_dataset, generate_sample, make_phantom

__all__ = ["MODALITIES", "generate_dataset", "generate_sample", "make_phantom"]
