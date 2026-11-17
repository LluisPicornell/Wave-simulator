"""Radiación dispersa 3D por Monte Carlo de fotones (generalización 3D de ``xray_scatter``).

El Monte Carlo 2D tiene el objeto **extruido sin límite en y**: sobrestima el scatter de un objeto finito.
Aquí el material es el volumen ``Phantom3D`` completo (celdas ``mat[k, j, i]``, tejido por etiqueta), los fotones
nacen de la fuente puntual (o del haz paralelo) hacia un punto uniforme del detector plano y se transportan
por *delta tracking* (``_numba_xray3d.run_photons3d``) con las mismas secciones eficaces que el 2D
(``materials.xray_processes``: fotoeléctrico, Compton con ``S(x, Z)`` y Rayleigh). Los fotones dispersados
viajan en línea recta hasta el plano del detector (sin rejilla antidispersora ni escape de fluorescencia) y se
acumulan en **bins gruesos** del detector (el scatter es una señal de baja frecuencia): ``n_bins`` bins por
eje, interpolados linealmente a los píxeles tras un suavizado gaussiano de ``smooth_sigma`` bins.

Normalización: el flujo de la fuente es uniforme sobre el detector, así que ``n_emitidos / n_bins²`` es el
número de fotones de la medida sin objeto por bin y ``scatter_rel = señal dispersa / señal abierta`` está en
las unidades de ``I/I0`` de ``xray3d.acquire3d``. La señal de cada fotón detectado lleva el peso del panel
(``FlatPanel.signal_weights``: eficiencia y, si integra energía, la energía).

Para una vista de ``n_photons`` el ruido del scatter relativo por bin es ~``1/sqrt(n_scat_bin)``. Requiere
Numba (un fotón cada vez en Python puro no es viable); en un TAC se calculan ``n_views`` vistas y se
interpola en el ángulo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import xray_processes as xpr
from ..phantoms.limb3d import Phantom3D
from ..phantoms.phantom import TISSUE_INDEX

@dataclass
class Scatter3DOptions:
    """Parámetros del scatter 3D de ``xray3d.acquire3d``."""

    n_photons: int = 3_000_000
    n_views: int = 8  # sólo en TAC: vistas con Monte Carlo (las demás se interpolan en el ángulo)
    n_bins: int = 20  # bins del detector por eje
    smooth_sigma: float = 1.0  # suavizado gaussiano en bins
    seed: int = 0


@dataclass
class ScatterView3D:
    """Resultado de una vista: mapas de bins ``(n_bins_v, n_bins_u)`` y contadores."""

    prim_n: np.ndarray
    scat_n: np.ndarray
    prim_w: np.ndarray
    scat_w: np.ndarray
    counters: dict
    open_beam_w: float  # señal por bin sin objeto

    @property
    def scatter_rel(self) -> np.ndarray:
        return self.scat_w / self.open_beam_w

    @property
    def primary_rel(self) -> np.ndarray:
        return self.prim_w / self.open_beam_w

    def balance(self) -> float:
        """Cierre del balance de fotones (diferencia relativa; debe ser ~0)."""
        c = self.counters
        n_out = c["n_photoelectric"] + c["n_below_cutoff"] + c["n_outside_detector"] + self.prim_n.sum() \
            + self.scat_n.sum()
        return float((n_out - c["n_emitted"]) / c["n_emitted"])


_ACC = ("n_emitted", "w_emitted", "n_photoelectric", "n_below_cutoff", "n_outside_detector", "n_real", "n_virtual",
        "n_compton", "n_rayleigh")


def _prepare(phantom: Phantom3D, spectrum):
    e = spectrum.energies_kev
    if e.min() < 10.0 or e.max() > 200.0:
        raise ValueError("El Monte Carlo de scatter se valida entre 10 y 200 keV")
    keys = sorted(set(phantom.present_tissues()), key=lambda k: TISSUE_INDEX[k])
    tables = xpr.build_tables(keys, e_max_kev=float(e.max()))
    lut = np.zeros(max(TISSUE_INDEX.values()) + 1, dtype=np.int32)
    for i, k in enumerate(keys):
        lut[TISSUE_INDEX[k]] = i
    mat = np.ascontiguousarray(lut[phantom.labels])
    mu_max = tables.mu_total[np.unique(mat)].max(axis=0)
    cdf = np.cumsum(spectrum.weights)
    return mat, tables, mu_max, np.asarray(e, float), cdf / cdf[-1]


def simulate_view(phantom: Phantom3D, theta_deg: float, geometry, spectrum, panel=None, n_photons: int = 3_000_000,
                  n_bins: int = 20, seed: int = 0, prepared=None) -> ScatterView3D:
    """Una vista: ``n_photons`` fotones; primarios y dispersados por bin del detector (ver el módulo)."""
    try:
        from ._numba_xray3d import NACC3, run_photons3d
    except ImportError:
        raise ImportError("El scatter 3D requiere Numba: instálalo con `pip install wavesim[fast]`.") from None
    from .xray3d import FlatPanel, center

    panel = panel or FlatPanel(csi_thickness=None, energy_integrating=False)
    g = phantom.grid
    mat, tb, mu_max, spec_e, spec_cdf = prepared or _prepare(phantom, spectrum)
    d, u, v = geometry.view_vectors(theta_deg)
    nbu = max(min(int(n_bins), geometry.n_u), 1)
    nbv = max(min(int(n_bins), geometry.n_v), 1)
    ext = {"x": g.nx, "y": g.ny, "z": g.nz}
    rad = 0.5 * float(np.hypot(*[ext[k] * g.dx for k in "xyz" if k != geometry.axis])) * (1.0 + 1e-9)
    gv = np.concatenate([center(g), d, u, v, [geometry.sid, geometry.sdd, 0.5 * geometry.n_u * geometry.du,
                                              0.5 * geometry.n_v * geometry.dv, nbu, nbv, rad,
                                              1.0 if geometry.parallel else 0.0]])
    e_grid = tb.energies_kev
    eta = panel.quantum_efficiency(e_grid)
    det_w = np.ascontiguousarray(eta * (e_grid if panel.energy_integrating else 1.0))
    prim_n, scat_n, prim_w, scat_w = (np.zeros((nbv, nbu)) for _ in range(4))
    acc = np.zeros(NACC3)
    x_step = float(tb.x_grid[1] - tb.x_grid[0])
    run_photons3d(int(seed), int(n_photons), mat, np.array([g.x0, g.y0, g.z0, g.dx]), float(e_grid[0]), tb.mu_pe,
                  tb.mu_coh, tb.mu_inc, mu_max, x_step, tb.s_over_z, float(tb.ray_energies_kev[0]),
                  float(tb.ray_energies_kev[1] - tb.ray_energies_kev[0]), tb.ray_cdf, spec_e, spec_cdf, det_w, gv,
                  prim_n, scat_n, prim_w, scat_w, acc)
    counters = dict(zip(_ACC, acc))
    return ScatterView3D(prim_n, scat_n, prim_w, scat_w, counters, counters["w_emitted"] / (nbu * nbv))


def _smooth_matrix(n: int, sigma: float) -> np.ndarray:
    """Matriz de suavizado gaussiano de ``n`` bins con filas normalizadas (los bordes no pierden señal)."""
    d = np.arange(n)[:, None] - np.arange(n)[None, :]
    w = np.exp(-0.5 * (d / sigma) ** 2)
    return w / w.sum(axis=1, keepdims=True)


def _to_pixels(bins: np.ndarray, n_v: int, n_u: int, sigma: float) -> np.ndarray:
    """Suaviza los bins y los interpola linealmente (con extremos constantes) a ``(n_v, n_u)`` píxeles."""
    if sigma > 0:
        bins = _smooth_matrix(bins.shape[0], sigma) @ bins @ _smooth_matrix(bins.shape[1], sigma).T
    nbv, nbu = bins.shape
    pu = (np.arange(n_u) + 0.5) / n_u * nbu - 0.5
    pv = (np.arange(n_v) + 0.5) / n_v * nbv - 0.5
    rows = np.array([np.interp(pu, np.arange(nbu), r) for r in bins])  # (nbv, n_u)
    return np.array([np.interp(pv, np.arange(nbv), rows[:, i]) for i in range(n_u)]).T


def scatter_views(phantom: Phantom3D, angles_deg, geometry, spectrum, options: Scatter3DOptions | None = None,
                  panel=None) -> np.ndarray:
    """Scatter relativo ``S/I0`` de cada ángulo: ``(n_ángulos, n_v, n_u)``.

    Si hay más ángulos que ``options.n_views`` se simulan ``n_views`` vistas equiespaciadas entre el primer
    y el último ángulo y se interpola linealmente en el ángulo (el scatter varía despacio con la vista).
    """
    opt = options or Scatter3DOptions()
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    prepared = _prepare(phantom, spectrum)
    exact = angles.size <= opt.n_views
    sim_angles = angles if exact else np.linspace(angles.min(), angles.max(), opt.n_views)
    maps = np.array([_to_pixels(simulate_view(phantom, th, geometry, spectrum, panel, opt.n_photons, opt.n_bins,
                                              opt.seed + k, prepared).scatter_rel,
                                geometry.n_v, geometry.n_u, opt.smooth_sigma)
                     for k, th in enumerate(sim_angles)])
    if exact:
        return maps
    flat = maps.reshape(sim_angles.size, -1)
    k = np.clip(np.searchsorted(sim_angles, angles) - 1, 0, sim_angles.size - 2)
    w = ((angles - sim_angles[k]) / (sim_angles[k + 1] - sim_angles[k]))[:, None]
    return (flat[k] * (1 - w) + flat[k + 1] * w).reshape(angles.size, geometry.n_v, geometry.n_u)
