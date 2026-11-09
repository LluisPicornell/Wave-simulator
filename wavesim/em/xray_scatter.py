"""Radiación dispersa en rayos X por Monte Carlo de fotones (fotoeléctrico, Compton y Rayleigh).

Los fotones se mueven en 3D por un objeto que es el corte 2D de la malla **extruido sin límite en y**
(un cilindro infinito para un fantoma circular). La fuente se colima a un grosor de corte
``slice_thickness`` en el isocentro y el detector tiene una altura ``det_height`` (por defecto la del
haz colimado proyectada); los fotones dispersados fuera de esa altura no se cuentan. Como el objeto es
infinito en y, el scatter es mayor que el de un objeto finito del mismo diámetro.

Procesos (``materials.xray_processes`` documenta de dónde sale cada sección eficaz):

- fotoeléctrico: el fotón se absorbe (sin fluorescencia ni transporte de electrones);
- Compton: Klein-Nishina con la función de dispersión incoherente ``S(x, Z)`` del material, por
  rechazo (el electrón se deposita en el sitio);
- Rayleigh: ``(1 + cos²θ) F(x, Z)²`` tabulado por material y energía.

Se registran por elemento del detector el número de fotones primarios (sin interacción alguna) y
dispersados (con al menos una), con su energía, y los contadores para el balance de fotones y energía.

Normalización: el flujo de la fuente es uniforme sobre los ``n_det`` elementos, así que
``n_emitidos / n_det`` es el número de fotones de la medida sin objeto (*open beam*) por elemento y
``scatter_rel = dispersos / open_beam`` está en las mismas unidades que ``I/I0`` de ``xray.acquire``.

Backends: ``"numpy"`` (vectorizado por lotes) y ``"numba"`` (``_numba_xray``, un fotón cada vez, es mucho
más rápido); ``"auto"`` usa Numba si está instalado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.grid import Grid2D
from ..materials import xray as xd
from ..materials import xray_processes as xpr
from ..phantoms.phantom import TISSUE_INDEX, Phantom
from .fanbeam import FanGeometry
from .projection import detector_axis

ACC_NAMES = ("n_emitted", "e_emitted", "n_photoelectric", "e_photoelectric", "e_compton_electron", "n_below_cutoff",
             "e_below_cutoff", "n_outside_detector", "e_outside_detector", "n_grid_rejected", "e_grid_rejected",
             "n_real_collisions", "n_virtual_collisions", "n_compton", "n_rayleigh", "sum_cos_compton",
             "sum_cos_rayleigh")


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError(f"backend desconocido '{backend}' (use 'auto', 'numpy' o 'numba')")
    if backend == "numpy":
        return "numpy"
    try:
        import numba  # noqa: F401
    except ImportError:
        if backend == "numba":
            raise ImportError("El backend 'numba' requiere Numba: instálalo con `pip install wavesim[fast]` "
                              "o usa backend='numpy'.") from None
        return "numpy"
    return "numba"


@dataclass
class ScatterResult:
    prim: np.ndarray  # fotones primarios detectados por elemento
    scat: np.ndarray  # fotones dispersados detectados
    scat_compton: np.ndarray  # dispersados con al menos un Compton
    scat_rayleigh: np.ndarray  # dispersados sólo por Rayleigh
    e_prim: np.ndarray  # energía (keV) detectada de los primarios
    e_scat: np.ndarray
    counters: dict
    n_emitted: int
    backend: str
    n_det: int

    @property
    def open_beam(self) -> float:
        """Fotones por elemento sin objeto (emisión uniforme)."""
        return self.n_emitted / self.n_det

    @property
    def primary_rel(self) -> np.ndarray:
        return self.prim / self.open_beam

    @property
    def scatter_rel(self) -> np.ndarray:
        return self.scat / self.open_beam

    @property
    def spr(self) -> np.ndarray:
        """Cociente dispersa / primaria por elemento (NaN donde no hay primarios)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.prim > 0, self.scat / self.prim, np.nan)

    def balance(self) -> dict:
        """Cierre del balance de fotones y de energía (diferencia relativa, debe ser ~1e-12)."""
        c = self.counters
        n_out = (c["n_photoelectric"] + c["n_below_cutoff"] + c["n_outside_detector"] + c["n_grid_rejected"]
                 + self.prim.sum() + self.scat.sum())
        e_out = (c["e_photoelectric"] + c["e_compton_electron"] + c["e_below_cutoff"] + c["e_outside_detector"]
                 + c["e_grid_rejected"] + self.e_prim.sum() + self.e_scat.sum())
        return {"photons": (n_out - c["n_emitted"]) / c["n_emitted"],
                "energy": (e_out - c["e_emitted"]) / c["e_emitted"]}


@dataclass
class _Prepared:
    mat: np.ndarray
    scale: np.ndarray
    tables: xpr.ProcessTables
    mu_max: np.ndarray
    spec_e: np.ndarray
    spec_cdf: np.ndarray
    grid: Grid2D
    cache_key: tuple = field(default=())


def prepare(phantom: Phantom, spectrum) -> _Prepared:
    """Convierte el fantoma y el espectro a las tablas del Monte Carlo.

    El material de cada celda es su tejido (etiqueta); en las celdas porosas ``scale`` multiplica los
    coeficientes por el cociente entre la atenuación efectiva de la mezcla (regla de ``property_map`` con
    ``physics="xray"``) y la del tejido, a la energía media del espectro.
    """
    e = spectrum.energies_kev
    if e.min() < 10.0 or e.max() > 200.0:
        raise ValueError("El Monte Carlo de scatter se valida entre 10 y 200 keV")
    keys = sorted(set(phantom.present_tissues()), key=lambda k: TISSUE_INDEX[k])
    tables = xpr.build_tables(keys, e_max_kev=float(e.max()))
    lut = np.zeros(max(TISSUE_INDEX.values()) + 1, dtype=np.int32)
    for i, k in enumerate(keys):
        lut[TISSUE_INDEX[k]] = i
    mat = lut[phantom.labels]
    e_ref = float(np.clip(spectrum.mean_energy, 10.0, 200.0))
    mu_ref = phantom.property_map(lambda k: float(xd.linear_attenuation(k, e_ref)), physics="xray")
    base = np.array([float(xd.linear_attenuation(k, e_ref)) for k in keys])[mat]
    scale = mu_ref / base
    mu_tot = tables.mu_total
    mu_max = np.max(mu_tot[np.unique(mat)] * np.array([scale[mat == i].max() for i in np.unique(mat)])[:, None],
                    axis=0)
    cdf = np.cumsum(spectrum.weights)
    return _Prepared(mat, np.ascontiguousarray(scale), tables, mu_max, np.asarray(e, float), cdf / cdf[-1],
                     phantom.grid)


def _geometry_vector(grid: Grid2D, theta_deg: float, geometry: FanGeometry | None, det: np.ndarray | None,
                     slice_thickness: float, det_height: float | None, grid_acceptance_deg: float | None):
    th = np.radians(theta_deg)
    d = (np.sin(th), np.cos(th))
    n = (np.cos(th), -np.sin(th))
    rad = 0.5 * np.hypot(grid.width, grid.depth)
    if geometry is None:
        spacing = float(det[1] - det[0])
        n_det, kind, sid, sdd = det.size, 2, rad, 2 * rad
        half_h = 0.5 * (det_height or slice_thickness)
    else:
        spacing, n_det, kind = geometry.det_spacing, geometry.n_det, 0 if geometry.curved else 1
        sid, sdd = geometry.sid, geometry.sdd
        half_h = 0.5 * (det_height or slice_thickness * sdd / sid)
    cos_acc = np.cos(np.radians(grid_acceptance_deg)) if grid_acceptance_deg else 0.0
    gf = np.array([d[0], d[1], n[0], n[1], sid, sdd, spacing, 0.5 * slice_thickness, half_h, cos_acc, rad])
    c = (grid.x0 + (grid.nx - 1) * grid.dx / 2, grid.z0 + (grid.nz - 1) * grid.dx / 2)
    return gf, np.array([grid.x0, grid.z0, grid.dx, c[0], c[1]]), kind, n_det


def simulate_scatter(phantom: Phantom, theta_deg: float, spectrum, geometry: FanGeometry | None = None,
                     det: np.ndarray | None = None, n_photons: int = 1_000_000, slice_thickness: float = 0.01,
                     det_height: float | None = None, grid_acceptance_deg: float | None = None,
                     backend: str = "auto", seed: int = 0, prepared: _Prepared | None = None) -> ScatterResult:
    """Una vista: transporta ``n_photons`` fotones y devuelve primarios y dispersados por elemento.

    ``geometry=None`` es haz paralelo con el detector ``det`` (por defecto ``detector_axis(grid)``).
    ``grid_acceptance_deg``: semiángulo de aceptación de una rejilla antidispersora enfocada
    (los dispersados que llegan con más desviación respecto al rayo fuente-elemento se descartan).
    """
    be = _resolve_backend(backend)
    grid = phantom.grid
    if geometry is None and det is None:
        det = detector_axis(grid)
    prep = prepared or prepare(phantom, spectrum)
    gf, grid_f, kind, n_det = _geometry_vector(grid, theta_deg, geometry, det, slice_thickness, det_height,
                                               grid_acceptance_deg)
    outs = [np.zeros(n_det) for _ in range(6)]
    acc = np.zeros(len(ACC_NAMES))
    tb = prep.tables
    if be == "numba":
        from ._numba_xray import run_photons

        x_step = float(tb.x_grid[1] - tb.x_grid[0])
        run_photons(int(seed), int(n_photons), prep.mat, prep.scale, grid_f, float(tb.energies_kev[0]), tb.mu_pe,
                    tb.mu_coh, tb.mu_inc, prep.mu_max, x_step, tb.s_over_z, float(tb.ray_energies_kev[0]),
                    float(tb.ray_energies_kev[1] - tb.ray_energies_kev[0]), tb.ray_cdf, prep.spec_e, prep.spec_cdf,
                    gf, kind, n_det, outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], acc)
    else:
        _run_numpy(np.random.default_rng(seed), int(n_photons), prep, grid_f, gf, kind, n_det, outs, acc)
    return ScatterResult(outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], dict(zip(ACC_NAMES, acc)),
                         int(n_photons), be, n_det)


# --------------------------------------------------------------------------------------
# Backend NumPy: mismo algoritmo, vectorizado por lotes
# --------------------------------------------------------------------------------------
def _interp(table: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Interpolación lineal de ``table`` (última dimensión) en el índice fraccionario ``f``."""
    i = np.clip(f.astype(int), 0, table.shape[-1] - 2)
    w = f - i
    return table[..., i] * (1 - w) + table[..., i + 1] * w


def _rotate(u: np.ndarray, cost: np.ndarray, phi: np.ndarray) -> np.ndarray:
    ux, uy, uz = u
    sint = np.sqrt(np.maximum(1 - cost**2, 0.0))
    cp, sp = np.cos(phi), np.sin(phi)
    polar = np.abs(uz) > 0.99999
    tmp = np.sqrt(np.maximum(1 - uz**2, 1e-30))
    nx = np.where(polar, sint * cp, sint * (ux * uz * cp - uy * sp) / tmp + ux * cost)
    ny = np.where(polar, sint * sp, sint * (uy * uz * cp + ux * sp) / tmp + uy * cost)
    nz = np.where(polar, np.sign(uz) * cost, -sint * cp * tmp + uz * cost)
    n = np.sqrt(nx**2 + ny**2 + nz**2)
    return np.array([nx / n, ny / n, nz / n])


def _exit_t(x, z, ux, uz, box):
    bx0, bx1, bz0, bz1 = box
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(ux > 1e-14, (bx1 - x) / ux, np.where(ux < -1e-14, (bx0 - x) / ux, np.inf))
        tz = np.where(uz > 1e-14, (bz1 - z) / uz, np.where(uz < -1e-14, (bz0 - z) / uz, np.inf))
    return np.minimum(tx, tz)


def _run_numpy(rng, nphot, prep, grid_f, gf, kind, n_det, outs, acc, batch=40000):
    tb = prep.tables
    nz, nx = prep.mat.shape
    x0, z0, dx, cx, cz = grid_f
    box = (x0 - 0.5 * dx, x0 + (nx - 0.5) * dx, z0 - 0.5 * dx, z0 + (nz - 0.5) * dx)
    dxr, dzr, nxr, nzr, sid, sdd, spacing, half_t, half_h, cos_acc, rad = gf
    if kind == 2:
        sx, sz, sdd_eff = cx - rad * dxr, cz - rad * dzr, 2 * rad
    else:
        sx, sz, sdd_eff = cx - sid * dxr, cz - sid * dzr, sdd
    e0 = tb.energies_kev[0]
    x_step = tb.x_grid[1] - tb.x_grid[0]
    ray_de = tb.ray_energies_kev[1] - tb.ray_energies_kev[0]
    ncos = tb.ray_cdf.shape[2]
    for start in range(0, nphot, batch):
        n = min(batch, nphot - start)
        en = prep.spec_e[np.minimum(np.searchsorted(prep.spec_cdf, rng.random(n)), prep.spec_e.size - 1)]
        acc[0] += n
        acc[1] += en.sum()
        y_iso = (2 * rng.random(n) - 1) * half_t
        u_pos = (rng.random(n) - 0.5) * n_det * spacing
        if kind == 2:
            x, z, y = sx + u_pos * nxr, sz + u_pos * nzr, y_iso.copy()
            u = np.array([np.full(n, dxr), np.zeros(n), np.full(n, dzr)])
        else:
            gam = u_pos / sdd if kind == 0 else np.arctan(u_pos / sdd)
            tanphi = y_iso * np.cos(gam) / sid
            nrm = np.sqrt(1 + tanphi**2)
            u = np.array([(np.cos(gam) * dxr + np.sin(gam) * nxr) / nrm, tanphi / nrm,
                          (np.cos(gam) * dzr + np.sin(gam) * nzr) / nrm])
            x, z, y = np.full(n, sx), np.full(n, sz), np.zeros(n)
        n_int = np.zeros(n, dtype=int)
        had_c = np.zeros(n, dtype=bool)
        alive = np.ones(n, dtype=bool)
        # Hasta la caja (slab); lo que no la toca queda sin interacciones
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = np.where(np.abs(u[0]) > 1e-14, (box[0] - x) / u[0], -np.inf)
            tb_ = np.where(np.abs(u[0]) > 1e-14, (box[1] - x) / u[0], np.inf)
            tc = np.where(np.abs(u[2]) > 1e-14, (box[2] - z) / u[2], -np.inf)
            td = np.where(np.abs(u[2]) > 1e-14, (box[3] - z) / u[2], np.inf)
        tmin = np.maximum(np.maximum(np.minimum(ta, tb_), np.minimum(tc, td)), 0.0)
        tmax = np.minimum(np.maximum(ta, tb_), np.maximum(tc, td))
        inside = tmax > tmin
        move = (tmin + 1e-9 * dx) * inside
        x, y, z = x + move * u[0], y + move * u[1], z + move * u[2]
        act = np.nonzero(inside)[0]
        exited = np.nonzero(~inside)[0]
        parts = [exited]
        while act.size:
            f = en[act] - e0
            s = -np.log(1 - rng.random(act.size)) / _interp(prep.mu_max, f)
            t_exit = _exit_t(x[act], z[act], u[0][act], u[2][act], box)
            out = s >= t_exit
            step = np.where(out, t_exit, s)
            for k, arr in enumerate((x, y, z)):
                arr[act] += step * u[(0, 1, 2)[k]][act]
            if out.any():
                parts.append(act[out])
            act, f, out = act[~out], f[~out], out[~out]
            if act.size == 0:
                break
            i = ((x[act] - box[0]) / dx).astype(int)
            j = ((z[act] - box[2]) / dx).astype(int)
            ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz)
            if not ok.all():
                parts.append(act[~ok])
                act, f, i, j = act[ok], f[ok], i[ok], j[ok]
            m = prep.mat[j, i]
            sc = prep.scale[j, i]
            a_pe = sc * _interp_rows(tb.mu_pe, m, f)
            a_coh = sc * _interp_rows(tb.mu_coh, m, f)
            a_inc = sc * _interp_rows(tb.mu_inc, m, f)
            a_tot = a_pe + a_coh + a_inc
            mmax = _interp(prep.mu_max, f)
            real = rng.random(act.size) * mmax <= a_tot
            acc[12] += (~real).sum()
            virtual = act[~real]  # siguen su camino sin interactuar
            act, f, m, a_pe, a_coh, a_tot = act[real], f[real], m[real], a_pe[real], a_coh[real], a_tot[real]
            if act.size == 0:
                act = virtual
                continue
            acc[11] += act.size
            n_int[act] += 1
            r = rng.random(act.size) * a_tot
            is_pe = r < a_pe
            acc[2] += is_pe.sum()
            acc[3] += en[act[is_pe]].sum()
            alive[act[is_pe]] = False
            is_ray = ~is_pe & (r < a_pe + a_coh)
            is_comp = ~is_pe & ~is_ray
            if is_ray.any():
                idx = act[is_ray]
                ir = np.clip(((en[idx] - tb.ray_energies_kev[0]) / ray_de + 0.5).astype(int), 0,
                             tb.ray_cdf.shape[1] - 1)
                rows = tb.ray_cdf[m[is_ray], ir]
                q = rng.random(idx.size)
                pos = (rows < q[:, None]).sum(axis=1)
                lo = np.clip(pos - 1, 0, ncos - 2)
                c0 = rows[np.arange(idx.size), lo]
                c1 = rows[np.arange(idx.size), lo + 1]
                frac = np.where(c1 > c0, (q - c0) / np.where(c1 > c0, c1 - c0, 1.0), 0.0)
                cost = -1 + 2 * (lo + frac) / (ncos - 1)
                acc[14] += idx.size
                acc[16] += cost.sum()
                u[:, idx] = _rotate(u[:, idx], cost, 2 * np.pi * rng.random(idx.size))
            if is_comp.any():
                idx = act[is_comp]
                had_c[idx] = True
                e_c = en[idx]
                mc = m[is_comp]
                kk = e_c / xpr.MEC2_KEV
                emin = 1 / (1 + 2 * kk)
                a1, a2 = -np.log(emin), 0.5 * (1 - emin**2)
                eps = np.empty(idx.size)
                cost = np.empty(idx.size)
                todo = np.arange(idx.size)
                while todo.size:
                    kt, et = kk[todo], emin[todo]
                    pick = rng.random(todo.size) * (a1[todo] + a2[todo]) < a1[todo]
                    ee = np.where(pick, et ** rng.random(todo.size),
                                  np.sqrt(et**2 + (1 - et**2) * rng.random(todo.size)))
                    ct = 1 - (1 / ee - 1) / kt
                    tacc = 1 - ee * (1 - ct**2) / (1 + ee**2)
                    xq = e_c[todo] / xpr.HC_KEV_A * np.sqrt(np.maximum(0.5 * (1 - ct), 0.0))
                    sz_ = _interp_rows(tb.s_over_z, mc[todo], xq / x_step)
                    good = rng.random(todo.size) < tacc * sz_
                    eps[todo[good]] = ee[good]
                    cost[todo[good]] = ct[good]
                    todo = todo[~good]
                acc[4] += (e_c * (1 - eps)).sum()
                acc[13] += idx.size
                acc[15] += cost.sum()
                en[idx] = e_c * eps
                u[:, idx] = _rotate(u[:, idx], cost, 2 * np.pi * rng.random(idx.size))
                low = en[idx] < e0 + 1e-9
                if low.any():
                    acc[5] += low.sum()
                    acc[6] += en[idx[low]].sum()
                    alive[idx[low]] = False
            act = np.concatenate([virtual, act[alive[act]]])
        _detect(np.concatenate(parts), alive, x, y, z, u, en, n_int, had_c, (sx, sz, dxr, dzr, nxr, nzr, spacing, sdd,
                                                                         sdd_eff, half_h, cos_acc), kind, n_det, outs,
                acc)


def _interp_rows(table: np.ndarray, rows: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``table[rows[k]]`` evaluada por interpolación lineal en ``f[k]`` (rejilla uniforme)."""
    i = np.clip(f.astype(int), 0, table.shape[1] - 2)
    w = f - i
    return table[rows, i] * (1 - w) + table[rows, i + 1] * w


def _detect(idx, alive, x, y, z, u, en, n_int, had_c, g, kind, n_det, outs, acc):
    sx, sz, dxr, dzr, nxr, nzr, spacing, sdd, sdd_eff, half_h, cos_acc = g
    idx = idx[alive[idx]]
    ux, uy, uz = u[0][idx], u[1][idx], u[2][idx]
    a, b, e, yy = x[idx] - sx, z[idx] - sz, en[idx], y[idx]
    h2 = ux**2 + uz**2
    hit = np.zeros(idx.size, dtype=bool)
    t = np.zeros(idx.size)
    j = np.zeros(idx.size, dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == 0:
            bb = a * ux + b * uz
            disc = bb**2 - h2 * (a**2 + b**2 - sdd**2)
            t = np.where((disc >= 0) & (h2 > 1e-24), (-bb + np.sqrt(np.maximum(disc, 0))) / h2, -1.0)
            ha, hb = a + t * ux, b + t * uz
            along = ha * dxr + hb * dzr
            hit = (t >= 0) & (along > 0)
            gam = np.arctan2(ha * nxr + hb * nzr, along)
            j = np.floor(gam / (spacing / sdd) + 0.5 * n_det).astype(int)
        else:
            den = ux * dxr + uz * dzr
            t = np.where(den > 1e-12, (sdd_eff - (a * dxr + b * dzr)) / den, -1.0)
            hit = t >= 0
            lat = (a + t * ux) * nxr + (b + t * uz) * nzr
            j = np.floor(lat / spacing + 0.5 * n_det).astype(int)
    yh = yy + t * uy
    hit &= (j >= 0) & (j < n_det) & (np.abs(yh) <= half_h)
    miss = ~hit
    acc[7] += miss.sum()
    acc[8] += e[miss].sum()
    sc = n_int[idx] > 0
    if cos_acc > 0.0:
        if kind == 2:
            ca = ux * dxr + uz * dzr
        else:
            vx, vy, vz = a + t * ux, yh, b + t * uz
            ca = (ux * vx + uy * vy + uz * vz) / np.sqrt(vx**2 + vy**2 + vz**2)
        rej = hit & sc & (ca < cos_acc)
        acc[9] += rej.sum()
        acc[10] += e[rej].sum()
        hit &= ~rej
    p, s_ = hit & ~sc, hit & sc
    outs[0] += np.bincount(j[p], minlength=n_det)
    outs[4] += np.bincount(j[p], weights=e[p], minlength=n_det)
    outs[1] += np.bincount(j[s_], minlength=n_det)
    outs[5] += np.bincount(j[s_], weights=e[s_], minlength=n_det)
    sc_c = s_ & had_c[idx]
    outs[2] += np.bincount(j[sc_c], minlength=n_det)
    outs[3] += np.bincount(j[s_ & ~had_c[idx]], minlength=n_det)


# --------------------------------------------------------------------------------------
# Scatter para un TAC completo: algunas vistas, suavizado y interpolación en ángulo
# --------------------------------------------------------------------------------------
@dataclass
class ScatterOptions:
    """Parámetros del scatter de ``xray.acquire`` / ``xray.ct_scan``.

    El scatter varía despacio con el ángulo y con la posición en el detector, así que se calcula en
    ``n_views`` vistas equiespaciadas (``n_photons`` fotones cada una), se suaviza con una gaussiana
    de ``smooth_sigma`` elementos y se interpola linealmente en el ángulo.
    """

    n_photons: int = 2_000_000
    n_views: int = 12
    slice_thickness: float = 0.01
    det_height: float | None = None
    smooth_sigma: float | None = None
    grid_acceptance_deg: float | None = None
    backend: str = "auto"
    seed: int = 0


def smooth_profile(values: np.ndarray, sigma: float) -> np.ndarray:
    """Suavizado gaussiano a lo largo del último eje, con normalización de los bordes."""
    if sigma <= 0:
        return values
    half = int(np.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    ones = np.convolve(np.ones(values.shape[-1]), k, mode="same")
    flat = values.reshape(-1, values.shape[-1])
    out = np.array([np.convolve(row, k, mode="same") for row in flat]) / ones
    return out.reshape(values.shape)


def scatter_sinogram(phantom: Phantom, angles_deg, spectrum, geometry: FanGeometry | None = None,
                     det: np.ndarray | None = None, options: ScatterOptions | None = None):
    """Scatter relativo ``S/I0`` en cada ángulo y elemento: devuelve ``(n_ángulos, n_det)`` y la lista de
    ``ScatterResult`` de las vistas calculadas."""
    opt = options or ScatterOptions()
    if geometry is None and det is None:
        det = detector_axis(phantom.grid)
    angles = np.atleast_1d(np.asarray(angles_deg, float))
    period = 360.0 if geometry is not None else 180.0
    views = np.arange(opt.n_views) * period / opt.n_views
    prep = prepare(phantom, spectrum)
    results = [simulate_scatter(phantom, v, spectrum, geometry, det, opt.n_photons, opt.slice_thickness,
                                opt.det_height, opt.grid_acceptance_deg, opt.backend, opt.seed + k, prep)
               for k, v in enumerate(views)]
    n_det = results[0].n_det
    sigma = opt.smooth_sigma if opt.smooth_sigma is not None else n_det / 30.0
    prof = smooth_profile(np.array([r.scatter_rel for r in results]), sigma)
    # Cierre del periodo: en haz paralelo la vista a 180° es la de 0° con el detector invertido
    closing = prof[0] if geometry is not None else prof[0][::-1]
    prof = np.vstack([prof, closing[None]])
    pos = (angles % period) / period * opt.n_views
    k0 = np.minimum(pos.astype(int), opt.n_views - 1)
    w = (pos - k0)[:, None]
    return prof[k0] * (1 - w) + prof[k0 + 1] * w, results
