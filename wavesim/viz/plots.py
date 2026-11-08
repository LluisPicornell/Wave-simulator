"""Figuras: fantomas, campos de onda, animaciones e imágenes reconstruidas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin pantalla: se guardan ficheros
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..materials.tissues import TISSUES, tissue_keys  # noqa: E402

_KEYS = tissue_keys()
TISSUE_CMAP = ListedColormap([TISSUES[k].color for k in _KEYS])


def plot_phantom(phantom, ax=None, title: str | None = None, show_fracture: bool = True, legend: bool = True):
    """Mapa de tejidos con leyenda y, opcionalmente, el contorno de la fractura."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ext = phantom.grid.extent_mm()
    ax.imshow(phantom.labels, cmap=TISSUE_CMAP, vmin=-0.5, vmax=len(_KEYS) - 0.5, extent=ext,
              interpolation="nearest")
    if show_fracture and phantom.has_fracture:
        g = phantom.grid
        ax.contour(g.x * 1e3, g.z * 1e3, phantom.fracture_mask.astype(float), levels=[0.5], colors="red",
                   linewidths=0.8)
    if legend:
        present = phantom.present_tissues()
        handles = [Patch(facecolor=TISSUES[k].color, edgecolor="k", lw=0.3, label=TISSUES[k].name) for k in present]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("profundidad z [mm]")
    if title:
        ax.set_title(title)
    return ax


def plot_wavefield(phantom, field, ax=None, title=None, clip: float = 0.5, cmap="seismic", alpha_tissue=0.35):
    """Campo de onda (p. ej. presión) superpuesto a la anatomía en gris."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ext = phantom.grid.extent_mm()
    ax.imshow(phantom.labels, cmap="gray", extent=ext, interpolation="nearest", alpha=alpha_tissue)
    vmax = clip * (np.abs(field).max() + 1e-30)
    # Los snapshots pueden estar submuestreados: se pintan con la misma extensión física
    ax.imshow(field, cmap=cmap, vmin=-vmax, vmax=vmax, extent=ext, alpha=0.75, interpolation="bilinear")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("profundidad z [mm]")
    if title:
        ax.set_title(title)
    return ax


def animate_wavefield(phantom, frames, times, path, fps: int = 15, clip: float = 0.3, title: str = "",
                      log_compress: bool = False):
    """Guarda una animación (GIF o MP4 según la extensión) de la propagación."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ext = phantom.grid.extent_mm()
    ax.imshow(phantom.labels, cmap="gray", extent=ext, interpolation="nearest", alpha=0.35)
    # Escala global robusta: percentil alto de todos los fotogramas
    vmax = clip * float(np.percentile(np.abs(np.asarray(frames)), 99.9) + 1e-30)
    data0 = _compress(frames[0], vmax) if log_compress else frames[0]
    lim = 1.0 if log_compress else vmax
    im = ax.imshow(data0, cmap="seismic", vmin=-lim, vmax=lim, extent=ext, alpha=0.8, interpolation="bilinear")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("profundidad z [mm]")
    ttl = ax.set_title("")

    def update(k):
        im.set_data(_compress(frames[k], vmax) if log_compress else frames[k])
        ttl.set_text(f"{title}  t = {times[k] * 1e6:6.2f} µs")
        return im, ttl

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    writer = animation.PillowWriter(fps=fps) if path.suffix.lower() == ".gif" else animation.FFMpegWriter(fps=fps)
    anim.save(path, writer=writer, dpi=90)
    plt.close(fig)
    return path


def _compress(f, vmax, dyn_db: float = 40.0):
    """Compresión logarítmica con signo, para ver a la vez ecos fuertes y débiles."""
    a = np.abs(f) / vmax
    floor = 10 ** (-dyn_db / 20)
    return np.sign(f) * np.clip(np.log10(np.maximum(a, floor) / floor) / np.log10(1 / floor), 0, 1)


def plot_image(img, extent_mm, ax=None, title=None, cmap="gray", vmin=None, vmax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(img, cmap=cmap, extent=extent_mm, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    if title:
        ax.set_title(title)
    if label:
        plt.colorbar(im, ax=ax, label=label, fraction=0.046)
    return ax


def save(fig_or_ax, path, dpi: int = 130):
    fig = fig_or_ax if isinstance(fig_or_ax, plt.Figure) else fig_or_ax.figure
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
