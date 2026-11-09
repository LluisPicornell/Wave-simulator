"""Transporte de luz por Monte Carlo de fotones (tipo MCML / MCX) en el fantoma 2D.

La aproximación de difusión de ``optics.solve_diffusion`` falla cerca de las
fuentes y de las fronteras (donde la luz aún no es isótropa) y no da
resolución temporal. Aquí se sigue fotón a fotón la ecuación de transporte,
siguiendo el algoritmo estándar de Wang, Jacques & Zheng, *MCML - Monte Carlo
modeling of light transport in multi-layered tissues*, Comput. Methods
Programs Biomed. 47, 131 (1995):

  - paquetes de fotones con peso ``w``; paso libre exponencial entre
    dispersiones (``s = -ln(u)/mu_s``, ver ABSORCIÓN);
  - dispersión con la función de fase de Henyey-Greenstein de anisotropía ``g``;
  - reflexión / refracción de Fresnel (luz no polarizada) en cada cambio de
    índice de refracción: entre el tejido y el exterior y entre celdas de
    tejidos con distinto ``n``;
  - ruleta rusa para pesos < 1e-4 (supervivencia 1/10, como MCML);
  - semilla reproducible.

GEOMETRÍA. El fantoma es 2D ``(x, z)`` y se trata como invariante en ``y``
(extrusión infinita): los fotones se mueven en 3D (tienen componente ``uy``
y se dispersan en 3D), pero la malla sólo discretiza ``x`` y ``z``. Es la
misma hipótesis del resto del paquete para un corte; a cambio la fluencia y las
exitancias son las de un medio 3D que fuera infinito y homogéneo en ``y``.
Para recorrer las celdas se usa el trazado de rayos por celdas de MCX: en cada
pared se re-evalúa ``mu_t`` y el índice de refracción de la celda nueva.
Las celdas de agua y aire se tratan como EXTERIOR: transparente, sin
dispersión ni camino de vuelta (un fotón que cruza hacia ellas escapa, salvo
la reflexión de Fresnel); el borde de la malla también es exterior (salvo con
``periodic_x=True``, para losas infinitas). El índice del exterior es el del
fondo (``optics._outer_index``). Con paredes alineadas a la malla, la
superficie curva de un fantoma queda en escalera: la normal de Fresnel es
siempre un eje, una aproximación que mejora con ``dx``.

ABSORCIÓN. Se usa absorción continua a lo largo del segmento en vez de
depositar ``mu_a/mu_t`` en cada interacción: en un segmento de longitud
``l`` el peso baja ``w -> w exp(-mu_a l)``. Es la misma esperanza (la
deposición discreta de MCML es un estimador de ella) pero con menos varianza,
permite tratar igual los fotones que cruzan celdas sin interactuar, y da la
fluencia por el estimador de longitud de traza, ``w l (1 - e^{-a})/a`` con
``a = mu_a l``, que además es válido con ``mu_a = 0``. El peso absorbido por
celda es la fluencia por ``mu_a`` salvo redondeo (la prueba de balance
``reflectancia + transmitancia + laterales + absorbida + especular = 1``
lo comprueba).

NORMALIZACIÓN (todo por fotón incidente). Con ``N`` fotones y celdas de lado
``dx``, la fluencia es ``Phi = W_s sum(w l) / (N dx²)``, donde ``W_s`` es el
ancho de la fuente. Para el haz ancho ``"top"`` (irradiancia uniforme sobre
todas las columnas con tejido), ``W_s`` = ancho de esas columnas y ``Phi`` es
adimensional: W/m² de fluencia por W/m² de irradiancia incidente, igual que
``optics.solve_diffusion``. Para una fuente fina (``"pencil"`` o
``"isotropic"``, que en el medio extruido son una línea de potencia por
metro de ``y``) ``W_s = 1 m`` y ``Phi`` está en m⁻¹ (W/m² por W/m de línea).
Las exitancias por columna se normalizan igual: ``W_col W_s / (N dx)``,
de modo que su media sobre las columnas del haz ancho es la reflectancia
(o transmitancia) total. La reflexión especular de la entrada (``n_exterior``
-> ``n_tejido``) se contabiliza aparte (``specular``) y el fotón entra con el
peso restante, como en MCML.

EXACTITUD ESTADÍSTICA. Las ``N`` historias se reparten en ``n_batches`` tandas
independientes; el error estándar de cada magnitud es la desviación entre
tandas dividida por ``sqrt(n_batches)``.

BACKENDS. ``backend="numpy"`` vectoriza muchos fotones a la vez; ``"numba"``
usa un kernel escalar compilado (``_numba_mc.py``, un solo hilo); ``"auto"``
elige Numba si está instalado (mismo patrón que ``ElasticSolver``). Los dos
usan generadores distintos y son iguales sólo en sentido estadístico.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..materials import optical
from ..phantoms.phantom import TISSUE_INDEX, Phantom
from . import optics

C0 = 299_792_458.0
WTH = 1e-4  # ruleta rusa por debajo de este peso
CHANCE = 0.1
_FLUSH = 4_000_000  # elementos acumulados antes de volcar la fluencia con bincount


@dataclass
class MCMedium:
    """Medio de transporte: mapas ``(nz, nx)`` en unidades SI (coeficientes en 1/m)."""

    mua: np.ndarray
    mus: np.ndarray  # coeficiente de dispersión mu_s (no reducido)
    g: np.ndarray
    n: np.ndarray  # índice de refracción
    tissue: np.ndarray  # True donde hay medio dispersivo; False = exterior transparente
    dx: float
    n_outer: float = 1.0
    x0: float = 0.0  # coordenada x del centro de la celda i = 0
    z0: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.mua.shape


def slab_medium(mua: float, mus: float, g: float, thickness: float, n: float = 1.0, n_outer: float = 1.0,
                dx: float | None = None, nx: int = 4) -> MCMedium:
    """Losa homogénea infinita en ``x`` e ``y``: úsese con ``periodic_x=True``. Unidades SI (1/m, m)."""
    dx = dx or thickness / 40.0
    nz = max(int(round(thickness / dx)), 1)
    dx = thickness / nz
    full = lambda v: np.full((nz, nx), float(v))  # noqa: E731
    return MCMedium(full(mua), full(mus), full(g), full(n), np.ones((nz, nx), bool), dx, n_outer)


def medium_from_phantom(phantom: Phantom, wavelength_nm: float, n_tissue: float = optics.N_TISSUE,
                        n_by_tissue: dict[str, float] | None = None) -> MCMedium:
    """Medio óptico de un fantoma a una longitud de onda.

    ``mu_a`` y ``mu_s' `` salen de ``materials.optical`` (con la mezcla por volumen de las celdas porosas);
    ``mu_s = mu_s' / (1 - g)`` con la ``g`` de cada tejido (``OpticalComposition.anisotropy``). Los mapas de
    ``mu_s`` y ``g`` se mezclan linealmente en celdas porosas. El índice de refracción es ``n_tissue`` en
    todo el tejido, salvo los de ``n_by_tissue`` (no hay una tabla verificada por tejido en el paquete).
    """
    g = phantom.grid
    mua, _ = optics.optical_maps(phantom, wavelength_nm)
    gmap = phantom.property_map(lambda k: optical.OPTICAL[k].anisotropy, physics="optical")
    mus = phantom.property_map(
        lambda k: float(optical.mu_s_reduced(k, wavelength_nm)) / (1.0 - optical.OPTICAL[k].anisotropy),
        physics="optical")
    over = n_by_tissue or {}
    nmap = phantom.property_map(lambda k: over.get(k, n_tissue), physics=None)
    tissue = ~np.isin(phantom.labels, [TISSUE_INDEX[k] for k in optics.NON_TISSUE])
    return MCMedium(mua, mus, gmap, nmap, tissue, g.dx, optics._outer_index(phantom), g.x0, g.z0)


@dataclass
class MonteCarloResult:
    """Resultado de ``simulate``. Todo por fotón incidente (ver la normalización en el módulo)."""

    fluence: np.ndarray  # (nz, nx): Phi, adimensional con haz ancho; m⁻¹ con fuente fina
    fluence_sem: np.ndarray  # error estándar de ``fluence`` entre tandas
    absorbed: np.ndarray  # fracción del peso incidente absorbida en cada celda
    reflectance: float  # difusa total, fotones que salen hacia -z (cara de entrada)
    transmittance: float  # difusa total, fotones que salen hacia +z (cara opuesta)
    lateral: float  # salen por los lados (+/-x)
    specular: float  # reflexión de Fresnel a la entrada (no entra en ``reflectance``)
    ballistic_transmittance: float  # parte de ``transmittance`` de fotones que nunca se dispersaron
    absorbed_total: float
    sem: dict[str, float]  # error estándar de las magnitudes escalares (mismos nombres que los campos)
    reflectance_profile: np.ndarray  # (nx,) exitancia por la cara de entrada, por columna
    transmittance_profile: np.ndarray  # (nx,) por la cara opuesta
    lateral_profile: np.ndarray  # (2, nz) izquierda / derecha, por fila
    n_photons: int
    source_width: float
    backend: str
    seconds: float
    batch_values: dict[str, np.ndarray] = field(default_factory=dict)
    time_edges: np.ndarray | None = None  # (nt + 1,) s
    fluence_t: np.ndarray | None = None  # (nt, nz, nx): fluencia acumulada en cada intervalo de tiempo
    reflectance_t: np.ndarray | None = None  # (nt,) fracción incidente que sale por -z en cada intervalo
    transmittance_t: np.ndarray | None = None  # (nt,) ídem por +z
    late: float = 0.0  # fracción que sale después de ``t_max`` (con resolución temporal)

    @property
    def balance(self) -> float:
        """Suma de todos los destinos del peso; debe ser 1 salvo redondeo."""
        return (self.reflectance + self.transmittance + self.lateral + self.specular + self.absorbed_total)

    @property
    def photons_per_second(self) -> float:
        return self.n_photons / self.seconds if self.seconds > 0 else float("inf")


# ----------------------------------------------------------------------------------------------------------
# Física común

def fresnel(n1, n2, cos_i):
    """Reflectancia de Fresnel no polarizada y coseno transmitido; ``cos_i`` >= 0. Admite arrays."""
    n1, n2 = np.broadcast_arrays(np.asarray(n1, float), np.asarray(n2, float))
    ci = np.maximum(np.asarray(cos_i, float), 1e-12)
    st = n1 / n2 * np.sqrt(np.maximum(1.0 - ci * ci, 0.0))
    tir = st >= 1.0
    ct = np.sqrt(np.maximum(1.0 - np.where(tir, 0.0, st) ** 2, 0.0))
    rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
    rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
    r = np.where(tir, 1.0, 0.5 * (rs * rs + rp * rp))
    return np.where(n1 == n2, 0.0, r), np.where(tir, 0.0, np.where(n1 == n2, ci, ct))


def henyey_greenstein_cos(g, u):
    """Coseno del ángulo de dispersión de Henyey-Greenstein por inversión de la CDF (``u`` uniforme en [0, 1))."""
    g = np.asarray(g, float)
    g_safe = np.where(np.abs(g) < 1e-6, 1.0, g)
    q = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
    ct = np.where(np.abs(g) < 1e-6, 2.0 * u - 1.0, (1.0 + g * g - q * q) / (2.0 * g_safe))
    return np.clip(ct, -1.0, 1.0)


class _Sinks:
    """Acumuladores de una tanda (cada tanda tiene los suyos para poder estimar el error entre tandas)."""

    def __init__(self, shape, nt, nx, nz, ntof):
        self.nz, self.nx, self.nt = nz, nx, nt
        self.flu = np.zeros(shape)
        self.absorbed = np.zeros(shape)
        self.flu_t = np.zeros((nt, *shape))
        self.prof_top = np.zeros(nx)
        self.prof_bot = np.zeros(nx)
        self.prof_side = np.zeros((2, nz))
        self.tof_top = np.zeros(ntof)
        self.tof_bot = np.zeros(ntof)
        self.acc = np.zeros(2)  # [especular, transmitido balístico]


# ----------------------------------------------------------------------------------------------------------
# Backend NumPy (muchos fotones a la vez)

def _flush(buf, size):
    if not buf:
        return np.zeros(size)
    idx = np.concatenate([i for i, _ in buf])
    val = np.concatenate([v for _, v in buf])
    buf.clear()
    return np.bincount(idx, weights=val, minlength=size)


def _run_batch_numpy(m: MCMedium, nphot, rng, kind, src_iz, src_ix, src_x, src_z, dt_bin, periodic, sk: _Sinks):
    nz, nx = m.shape
    dx = m.dx
    ncell = nz * nx
    mua, mus, gm, nm = m.mua.ravel(), m.mus.ravel(), m.g.ravel(), m.n.ravel()
    tissue = m.tissue
    nt = sk.nt
    ntof = sk.tof_top.size
    wdx = nx * dx

    # ---- lanzamiento
    uy = np.zeros(nphot)
    ux = np.zeros(nphot)
    if kind == 2:
        ix = np.full(nphot, min(max(int(src_x / dx), 0), nx - 1))
        iz = np.full(nphot, min(max(int(src_z / dx), 0), nz - 1))
        x = np.full(nphot, float(src_x))
        z = np.full(nphot, float(src_z))
        uz = 2.0 * rng.random(nphot) - 1.0
        ph = 2.0 * np.pi * rng.random(nphot)
        s = np.sqrt(np.maximum(1.0 - uz * uz, 0.0))
        ux, uy = s * np.cos(ph), s * np.sin(ph)
        w = np.ones(nphot)
    else:
        if kind == 0:
            k = (rng.random(nphot) * src_ix.size).astype(int)
            ix, iz = src_ix[k], src_iz[k]
            x = (ix + rng.random(nphot)) * dx
        else:
            ix, iz = np.full(nphot, src_ix[0]), np.full(nphot, src_iz[0])
            x = np.full(nphot, float(src_x))
        z = iz * dx
        uz = np.ones(nphot)
        rsp, _ = fresnel(m.n_outer, m.n[iz, ix], np.ones(nphot))
        sk.acc[0] += rsp.sum()
        w = 1.0 - rsp
    tau = -np.log1p(-rng.random(nphot))
    t = np.zeros(nphot)
    scat = np.zeros(nphot, bool)  # ya se dispersó alguna vez

    flu_buf, abs_buf, flut_buf = [], [], []

    def drain(buf, size, dest):
        v = _flush(buf, size)
        dest += v.reshape(dest.shape)

    while ix.size:
        c = iz * nx + ix
        mu_a, mu_s_c, n_c = mua[c], mus[c], nm[c]
        # paso libre con mu_s: la absorción es continua (ver el docstring del módulo)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_ev = np.where(mu_s_c > 0, tau / mu_s_c, 1e300)
            dxb = np.where(ux > 0, ((ix + 1) * dx - x) / ux, np.where(ux < 0, (ix * dx - x) / ux, 1e300))
            dzb = np.where(uz > 0, ((iz + 1) * dx - z) / uz, np.where(uz < 0, (iz * dx - z) / uz, 1e300))
        axis_x = dxb < dzb
        db = np.maximum(np.where(axis_x, dxb, dzb), 0.0)
        hit = db < s_ev
        ell = np.where(hit, db, s_ev)

        # absorción continua + fluencia por longitud de traza
        a = mu_a * ell
        big = a > 1e-8
        f = np.where(big, -np.expm1(-a), a)
        dw = w * f
        trk = w * ell * np.where(big, f / np.where(big, a, 1.0), 1.0 - 0.5 * a)
        flu_buf.append((c, trk))
        abs_buf.append((c, dw))
        if nt:
            it = ((t + 0.5 * n_c * ell / C0) / dt_bin).astype(np.int64)
            ok = it < nt
            flut_buf.append((it[ok] * ncell + c[ok], trk[ok]))
        if sum(len(i) for i, _ in flu_buf) >= _FLUSH:
            drain(flu_buf, ncell, sk.flu)
            drain(abs_buf, ncell, sk.absorbed)
            if nt:
                drain(flut_buf, nt * ncell, sk.flu_t)
        w = w - dw
        t = t + n_c * ell / C0
        x = x + ux * ell
        z = z + uz * ell

        # ---- dispersión (fotones que no tocan pared)
        sc = ~hit
        if sc.any():
            idx = np.nonzero(sc)[0]
            ct = henyey_greenstein_cos(gm[c[idx]], rng.random(idx.size))
            st = np.sqrt(1.0 - ct * ct)
            ph = 2.0 * np.pi * rng.random(idx.size)
            cp, sp = np.cos(ph), np.sin(ph)
            uxi, uyi, uzi = ux[idx], uy[idx], uz[idx]
            polar = np.abs(uzi) > 0.99999
            tmp = np.sqrt(np.maximum(1.0 - uzi * uzi, 1e-300))
            nux = np.where(polar, st * cp, st * (uxi * uzi * cp - uyi * sp) / tmp + uxi * ct)
            nuy = np.where(polar, st * sp, st * (uyi * uzi * cp + uxi * sp) / tmp + uyi * ct)
            nuz = np.where(polar, np.where(uzi > 0, ct, -ct), -st * cp * tmp + uzi * ct)
            ux[idx], uy[idx], uz[idx] = nux, nuy, nuz
            tau[idx] = -np.log1p(-rng.random(idx.size))
            scat[idx] = True

        # ---- pared: Fresnel y cruce
        if hit.any():
            idx = np.nonzero(hit)[0]
            tau[idx] -= mu_s_c[idx] * ell[idx]
            ax = axis_x[idx]
            uax = np.where(ax, ux[idx], uz[idx])
            sgn = np.where(uax > 0, 1, -1)
            ci = np.abs(uax)
            ixi, izi = ix[idx], iz[idx]
            nix = np.where(ax, ixi + sgn, ixi)
            niz = np.where(ax, izi, izi + sgn)
            # situar la coordenada normal exactamente en la pared
            x[idx] = np.where(ax, np.where(sgn > 0, (ixi + 1) * dx, ixi * dx), x[idx])
            z[idx] = np.where(ax, z[idx], np.where(sgn > 0, (izi + 1) * dx, izi * dx))
            wrapped = np.zeros(idx.size, bool)
            if periodic:
                wl = ax & (nix < 0)
                wr = ax & (nix >= nx)
                nix = np.where(wl, nx - 1, np.where(wr, 0, nix))
                wrapped = wl | wr
            outside = (nix < 0) | (nix >= nx) | (niz < 0) | (niz >= nz)
            inb = ~outside
            tn = np.zeros(idx.size, bool)
            tn[inb] = tissue[niz[inb], nix[inb]]
            outside = ~tn
            n1 = n_c[idx]
            n2 = np.where(outside, m.n_outer, nm[np.where(inb, niz * nx + nix, 0)])
            rf, ctr = fresnel(n1, n2, ci)
            wi = w[idx]
            # a) salida al exterior: sale w (1 - R) y el resto se refleja
            ex = outside
            we = np.where(ex, wi * (1.0 - rf), 0.0)
            side = ex & ax
            if side.any():
                for j in (0, 1):
                    mj = side & ((sgn < 0) if j == 0 else (sgn > 0))
                    if mj.any():
                        sk.prof_side[j] += np.bincount(izi[mj], weights=we[mj], minlength=nz)
            top, bot = ex & ~ax & (sgn < 0), ex & ~ax & (sgn > 0)
            ti = np.minimum((t[idx] / dt_bin).astype(np.int64), ntof - 1) if ntof > 1 else np.zeros(idx.size, int)
            for mk, prof, tof in ((top, sk.prof_top, sk.tof_top), (bot, sk.prof_bot, sk.tof_bot)):
                if mk.any():
                    prof += np.bincount(ixi[mk], weights=we[mk], minlength=nx)
                    tof += np.bincount(ti[mk], weights=we[mk], minlength=ntof)
            sk.acc[1] += we[bot & ~scat[idx]].sum()
            # b) interior: reflexión estocástica con probabilidad R
            refl = ex | (rng.random(idx.size) < rf)
            w[idx] = np.where(ex, wi * rf, wi)
            flip = np.where(refl, -1.0, 1.0)
            # dirección tras reflejar o refractar
            tr = ~refl
            r = np.where(n2 != 0, n1 / n2, 1.0)
            ref_t = tr & (n1 != n2)
            uxi, uyi, uzi = ux[idx], uy[idx], uz[idx]
            nux = np.where(ax, uxi * flip, np.where(ref_t, uxi * r, uxi))
            nuy = np.where(ref_t, uyi * r, uyi)
            nuz = np.where(ax, np.where(ref_t, uzi * r, uzi), uzi * flip)
            nux = np.where(ref_t & ax, sgn * ctr, nux)
            nuz = np.where(ref_t & ~ax, sgn * ctr, nuz)
            ux[idx], uy[idx], uz[idx] = nux, nuy, nuz
            mv = tr
            ix[idx] = np.where(mv, nix, ixi)
            iz[idx] = np.where(mv, niz, izi)
            if wrapped.any():
                wm = wrapped & mv
                x[idx[wm]] = np.where(nix[wm] == 0, 0.0, wdx)

        # ---- ruleta rusa y compactación
        low = w < WTH
        if low.any():
            li = np.nonzero(low)[0]
            surv = (rng.random(li.size) < CHANCE) & (w[li] > 0)
            w[li[surv]] /= CHANCE
            dead = np.zeros(w.size, bool)
            dead[li[~surv]] = True
            keep = ~dead
            if not keep.all():
                state = (ix, iz, x, z, ux, uy, uz, w, tau, t, scat)
                ix, iz, x, z, ux, uy, uz, w, tau, t, scat = (v[keep] for v in state)

    drain(flu_buf, ncell, sk.flu)
    drain(abs_buf, ncell, sk.absorbed)
    if nt:
        drain(flut_buf, nt * ncell, sk.flu_t)


# ----------------------------------------------------------------------------------------------------------
# Resolución de backend y fuente

def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError(f"backend desconocido '{backend}' (use 'auto', 'numpy' o 'numba')")
    if backend == "numpy":
        return "numpy"
    try:
        import numba  # noqa: F401
    except ImportError:
        if backend == "numba":
            raise ImportError(
                "El backend 'numba' requiere Numba: instálalo con `pip install wavesim[fast]` o usa backend='numpy'."
            ) from None
        return "numpy"
    return "numba"


def _source(m: MCMedium, source: str, source_xz):
    """``(kind, iz, ix, x, z, ancho)``; coordenadas locales: el origen es la pared izquierda de la celda 0."""
    nz, nx = m.shape
    xo, zo = m.x0 - m.dx / 2, m.z0 - m.dx / 2
    if source not in ("top", "pencil", "isotropic"):
        raise ValueError(f"fuente desconocida '{source}' (use 'top', 'pencil' o 'isotropic')")
    if source == "top":
        cols = np.nonzero(m.tissue.any(axis=0))[0]
        if cols.size == 0:
            raise ValueError("el medio no tiene tejido")
        return 0, np.argmax(m.tissue[:, cols], axis=0), cols, 0.0, 0.0, cols.size * m.dx
    if source_xz is None:
        raise ValueError(f"la fuente '{source}' necesita source_xz")
    xs, zs = source_xz[0] - xo, (source_xz[1] - zo) if len(source_xz) > 1 else 0.0
    ix = int(np.clip(xs // m.dx, 0, nx - 1))
    if source == "pencil":
        col = m.tissue[:, ix]
        if not col.any():
            raise ValueError("el haz fino cae en una columna sin tejido")
        return 1, np.array([int(np.argmax(col))]), np.array([ix]), float(xs), 0.0, 1.0
    iz = int(np.clip(zs // m.dx, 0, nz - 1))
    if not m.tissue[iz, ix]:
        raise ValueError(f"la fuente isótropa {tuple(source_xz)} cae fuera del tejido (en agua o aire)")
    return 2, np.array([iz]), np.array([ix]), float(xs), float(zs), 1.0


def simulate(medium: MCMedium, n_photons: int = 100_000, source: str = "top", source_xz=None, seed: int = 0,
             backend: str = "auto", n_batches: int = 20, periodic_x: bool = False, time_bins: int = 0,
             t_max: float | None = None) -> MonteCarloResult:
    """Lanza ``n_photons`` fotones en el medio y devuelve fluencia, reflectancia y transmitancia.

    ``source``: ``"top"`` haz ancho colimado y uniforme, en incidencia normal sobre la primera celda de tejido
    de cada columna; ``"pencil"`` haz fino normal en ``x = source_xz[0]``; ``"isotropic"`` fuente puntual
    isótropa en ``(x, z) = source_xz`` (en el medio extruido, una línea). ``periodic_x`` cierra la malla en x
    (losa infinita). ``time_bins`` > 0 añade resolución temporal: ``fluence_t`` y los histogramas de tiempo de
    vuelo de la luz que sale (``reflectance_t`` / ``transmittance_t``), con ``time_bins`` intervalos iguales en
    ``[0, t_max]`` (s); lo que sale o se registra después va a ``late`` (en la fluencia, se pierde). El tiempo
    es camino óptico, ``sum(n l) / c``. Los resultados son reproducibles con ``seed`` y ``backend`` fijos
    (y el número de tandas).
    """
    if n_photons < 1:
        raise ValueError("n_photons debe ser >= 1")
    if time_bins and not t_max:
        raise ValueError("con time_bins hay que dar t_max")
    be = _resolve_backend(backend)
    n_batches = max(1, min(n_batches, n_photons))
    kind, src_iz, src_ix, src_x, src_z, width = _source(medium, source, source_xz)
    nz, nx = medium.shape
    nt = int(time_bins)
    dt_bin = float(t_max) / nt if nt else 1.0
    ntof = nt + 1 if nt else 1
    per = np.full(n_batches, n_photons // n_batches)
    per[: n_photons % n_batches] += 1

    t0 = time.perf_counter()
    if be == "numba":
        from . import _numba_mc
        mua, mus, gm, nm = (np.ascontiguousarray(a, dtype=np.float64) for a in (medium.mua, medium.mus, medium.g,
                                                                               medium.n))
    children = np.random.SeedSequence(seed).spawn(n_batches)
    batches = []
    for b in range(n_batches):
        sk = _Sinks(medium.shape, nt, nx, nz, ntof)
        if be == "numba":
            bseed = int(children[b].generate_state(1)[0] % (2**31 - 1))
            _numba_mc.run_batch(bseed, int(per[b]), mua, mus, gm, nm, np.ascontiguousarray(medium.tissue),
                                medium.dx, float(medium.n_outer), bool(periodic_x), kind, src_iz.astype(np.int64),
                                src_ix.astype(np.int64), float(src_x), float(src_z), dt_bin, sk.flu, sk.absorbed,
                                sk.flu_t, sk.prof_top, sk.prof_bot, sk.prof_side, sk.tof_top, sk.tof_bot, sk.acc)
        else:
            rng = np.random.default_rng(children[b])
            _run_batch_numpy(medium, int(per[b]), rng, kind, src_iz, src_ix, src_x, src_z, dt_bin, periodic_x, sk)
        batches.append(sk)
    seconds = time.perf_counter() - t0

    # ---- normalización y estadística entre tandas
    cell = width / medium.dx**2
    colw = width / medium.dx

    def per_batch(getter, scale):
        return np.array([getter(sk) * scale / per[b] for b, sk in enumerate(batches)])

    flu_b = per_batch(lambda s: s.flu, cell)
    n_b = per.astype(float)
    wts = n_b / n_b.sum()
    flu = (flu_b * wts[:, None, None]).sum(0)
    var = (wts[:, None, None] * (flu_b - flu) ** 2).sum(0) * n_batches / max(n_batches - 1, 1)
    flu_sem = np.sqrt(var / n_batches)

    def stat(vals):
        v = np.asarray(vals)
        mean = float(np.sum(v * wts))
        sem = float(np.sqrt(np.sum(wts * (v - mean) ** 2) * n_batches / max(n_batches - 1, 1) / n_batches))
        return mean, sem

    refl_v = per_batch(lambda s: s.prof_top.sum(), 1.0)
    trans_v = per_batch(lambda s: s.prof_bot.sum(), 1.0)
    lat_v = per_batch(lambda s: s.prof_side.sum(), 1.0)
    spec_v = per_batch(lambda s: s.acc[0], 1.0)
    ball_v = per_batch(lambda s: s.acc[1], 1.0)
    abs_v = per_batch(lambda s: s.absorbed.sum(), 1.0)
    vals = {"reflectance": refl_v, "transmittance": trans_v, "lateral": lat_v, "specular": spec_v,
            "absorbed_total": abs_v, "ballistic_transmittance": ball_v}
    stats = {k: stat(v) for k, v in vals.items()}

    def total(getter, scale):
        return sum(getter(sk) for sk in batches) * scale / n_photons

    res = MonteCarloResult(
        fluence=flu, fluence_sem=flu_sem, absorbed=total(lambda s: s.absorbed, 1.0),
        reflectance=stats["reflectance"][0], transmittance=stats["transmittance"][0], lateral=stats["lateral"][0],
        specular=stats["specular"][0], ballistic_transmittance=stats["ballistic_transmittance"][0],
        absorbed_total=stats["absorbed_total"][0],
        sem={k: s for k, (_, s) in stats.items()},
        reflectance_profile=total(lambda s: s.prof_top, colw), transmittance_profile=total(lambda s: s.prof_bot, colw),
        lateral_profile=total(lambda s: s.prof_side, colw), n_photons=int(n_photons), source_width=float(width),
        backend=be, seconds=seconds, batch_values=vals,
    )
    if nt:
        res.time_edges = np.linspace(0.0, float(t_max), nt + 1)
        res.fluence_t = total(lambda s: s.flu_t, cell)
        res.reflectance_t = total(lambda s: s.tof_top[:nt], 1.0)
        res.transmittance_t = total(lambda s: s.tof_bot[:nt], 1.0)
        res.late = total(lambda s: s.tof_top[nt] + s.tof_bot[nt], 1.0)
    return res

