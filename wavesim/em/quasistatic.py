"""Corrientes en tejido a baja frecuencia (< 10 MHz): régimen cuasi-estático y EIT.

Cuando la longitud de onda es mucho mayor que el cuerpo (a 1 MHz, λ ≈ 3 m en
músculo frente a unos 10 cm de pierna) los campos no se propagan: el campo
eléctrico es el gradiente de un potencial y las corrientes de conducción y de
desplazamiento se reparten instantáneamente según la admitividad de cada tejido.
Con la convención ``eps = eps' - j eps''`` de ``materials.dielectric``:

    sigma* = sigma + j w eps0 eps_r = j w eps0 * eps          (S/m)
    div(sigma* grad(phi)) = 0

El cuerpo está rodeado de aire (o agua de fondo, que aquí también se trata
como medio exterior), que es aislante frente al tejido: el flujo hacia las
celdas que no son tejido es nulo (Neumann). Las corrientes entran por
electrodos pegados a la piel.

La discretización es de volúmenes finitos de 5 puntos. La conductancia de
cada cara entre dos celdas de tejido es la media armónica de sus
admitividades (la que conserva la corriente en una interfaz). Como el
problema es 2D, las corrientes son por metro de profundidad (A/m) y las
conductancias de cara son directamente ``sigma*`` (S/m).

El sistema es complejo simétrico (no hermítico), así que se resuelve con
COCG (gradiente conjugado con el producto bilineal sin conjugar) y
precondicionador de Jacobi, sin formar la matriz, igual que ``optics``.

Tomografía de impedancia eléctrica (EIT): se inyecta corriente entre pares de
electrodos adyacentes y se mide la tensión entre los demás pares adyacentes.
Por reciprocidad (Geselowitz) la sensibilidad de una medida a la admitividad
de una región es el producto de los gradientes del campo de inyección y del
campo de la medida; esa es la matriz que se usa para reconstruir imágenes de
diferencias (p. ej. sano frente a fracturado) en un solo paso linealizado.

Modelo completo de electrodo (CEM, Cheng et al. 1989; Somersalo et al. 1992),
opcional con ``simulate_eit(..., contact_impedance=z_c)``: cada electrodo es
una condición de contorno tipo Robin ``sigma* dphi/dn = (U_l - phi) / z_c``
con su tensión ``U_l`` como incógnita y su corriente total impuesta. La
derivación y la conversión de unidades están en ``docs/validacion/eit.md``;
la reconstrucción iterativa (Gauss-Newton) está en ``wavesim.em.eit``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import dielectric
from ..phantoms.phantom import TISSUE_INDEX, Phantom

NON_TISSUE = ("water", "air")


def admittivity(tissue: str, f: float) -> complex:
    """Admitividad ``sigma* = j w eps0 eps`` (S/m) del tejido a la frecuencia ``f``.

    Con ``eps = eps' - j eps''`` el producto da ``w eps0 eps'' + j w eps0 eps'``:
    parte real = conductividad total (iónica incluida) y parte imaginaria =
    corriente de desplazamiento (reactiva).
    """
    return 1j * 2.0 * np.pi * f * dielectric.EPS0 * complex(dielectric.complex_permittivity(tissue, f))


def tissue_mask(phantom: Phantom) -> np.ndarray:
    """Celdas por las que puede circular corriente (todo menos el medio exterior)."""
    return ~np.isin(phantom.labels, [TISSUE_INDEX[k] for k in NON_TISSUE])


@dataclass
class PotentialResult:
    phi: np.ndarray  # potencial complejo (V), forma (nz, nx); referencia de media nula si no hay Dirichlet
    iterations: int
    residual: float  # residuo relativo final


class QuasiStaticSolver:
    """Resuelve ``div(sigma* grad(phi)) = -s`` en un fantoma a la frecuencia ``f``.

    ``admittivity_map`` permite sustituir el mapa complejo de admitividades (por
    ejemplo para perturbarlo y comprobar la matriz de sensibilidad).

    ``shunt`` (S/m, forma ``(nz, nx)``) es una conductancia extra a tierra por
    celda: la usa el modelo completo de electrodo para el contacto (término de
    Robin ``g (phi - U)``, con ``U`` en el lado derecho). Con ella el operador
    deja de ser singular y ``solve`` no necesita referencia de media nula.
    """

    def __init__(self, phantom: Phantom, f: float, admittivity_map: np.ndarray | None = None,
                 shunt: np.ndarray | None = None):
        self.phantom = phantom
        self.grid = phantom.grid
        self.f = float(f)
        self.tissue = tissue_mask(phantom)
        if admittivity_map is None:
            # Maxwell Garnett en las celdas porosas (es homogéneo en la admitividad, que es ∝ eps)
            admittivity_map = phantom.property_map(lambda k: admittivity(k, self.f), dtype=complex,
                                                   physics="dielectric")
        self.y = np.where(self.tissue, admittivity_map, 0.0)
        tis = self.tissue
        # Media armónica en las caras tejido-tejido; cero si algún lado es exterior (flujo nulo)
        self._ya_x, self._yb_x = self.y[:, :-1], self.y[:, 1:]
        self._ya_z, self._yb_z = self.y[:-1, :], self.y[1:, :]
        self.face_x = self._face(self._ya_x, self._yb_x, tis[:, :-1] & tis[:, 1:])
        self.face_z = self._face(self._ya_z, self._yb_z, tis[:-1, :] & tis[1:, :])
        diag = np.zeros(self.grid.shape, dtype=complex)
        diag[:, :-1] += self.face_x
        diag[:, 1:] += self.face_x
        diag[:-1, :] += self.face_z
        diag[1:, :] += self.face_z
        self._shunt = None if shunt is None else np.where(tis, shunt, 0.0).astype(complex)
        if self._shunt is not None:
            diag += self._shunt
        # Celdas aisladas (sin caras) no aportan ecuación: diagonal 1 para no dividir por cero
        self._diag = np.where(diag != 0, diag, 1.0)

    @staticmethod
    def _face(ya, yb, both):
        with np.errstate(divide="ignore", invalid="ignore"):
            hm = 2.0 * ya * yb / (ya + yb)
        return np.where(both, hm, 0.0)

    def apply(self, phi: np.ndarray) -> np.ndarray:
        """Corriente neta (A/m) que sale de cada celda hacia sus vecinas: ``A phi``."""
        out = np.zeros(phi.shape, dtype=np.result_type(phi, complex))
        flx = self.face_x * (phi[:, 1:] - phi[:, :-1])
        out[:, :-1] -= flx
        out[:, 1:] += flx
        flz = self.face_z * (phi[1:, :] - phi[:-1, :])
        out[:-1, :] -= flz
        out[1:, :] += flz
        if self._shunt is not None:
            out += self._shunt * phi
        return out

    def solve(self, source: np.ndarray | None = None, fixed: np.ndarray | None = None,
              fixed_values: complex | np.ndarray = 0.0, tol: float = 1e-10, max_iter: int = 20_000) -> PotentialResult:
        """Potencial para una corriente inyectada ``source`` (A/m por celda) y/o celdas a potencial fijo.

        Sin celdas fijas (``fixed``) el problema es de Neumann puro y el
        potencial está definido salvo una constante: se exige que la corriente
        total inyectada sea nula y se fija la referencia con media nula en el
        tejido. Con ``fixed`` (booleano) el potencial queda determinado.
        """
        shape = self.grid.shape
        b = np.zeros(shape, dtype=complex) if source is None else np.asarray(source, dtype=complex)
        b = np.where(self.tissue, b, 0.0)
        neumann = fixed is None and self._shunt is None
        if fixed is None:
            free = self.tissue
            phi0 = np.zeros(shape, dtype=complex)
            if neumann and abs(b.sum()) > 1e-9 * max(np.abs(b).sum(), 1e-300):
                raise ValueError("Con Neumann puro la corriente total inyectada debe ser nula.")
        else:
            fixed = np.asarray(fixed, dtype=bool) & self.tissue
            free = self.tissue & ~fixed
            phi0 = np.where(fixed, fixed_values, 0.0).astype(complex)

        # Se eliminan las celdas fijas: se resuelve sólo en las libres con el término de contorno en el lado derecho
        r = np.where(free, b - self.apply(phi0), 0.0)
        bnorm = np.linalg.norm(r)
        x = np.zeros(shape, dtype=complex)
        it, res = 0, 0.0
        if bnorm > 0.0:
            z = r / self._diag
            p = z.copy()
            rz = np.sum(r * z)  # producto bilineal (sin conjugar): el sistema es simétrico, no hermítico
            res = 1.0
            for it in range(1, max_iter + 1):
                ap = np.where(free, self.apply(p), 0.0)
                alpha = rz / np.sum(p * ap)
                x += alpha * p
                r -= alpha * ap
                res = np.linalg.norm(r) / bnorm
                if res < tol:
                    break
                z = r / self._diag
                rz_new = np.sum(r * z)
                p = z + (rz_new / rz) * p
                rz = rz_new
        phi = x + phi0
        if neumann:
            phi = np.where(self.tissue, phi - phi[self.tissue].mean(), 0.0)
        return PotentialResult(phi, it, float(res))

    def injected_current(self, phi: np.ndarray, cells: np.ndarray) -> complex:
        """Corriente neta (A/m) que entra en la red por las celdas ``cells`` (booleano)."""
        return complex(self.apply(phi)[cells].sum())

    def current_density(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Densidad de corriente ``J = -sigma* grad(phi)`` (A/m²) ``(Jx, Jz)`` en el centro de las celdas.

        Se promedian las caras conductoras de cada celda; las que dan al
        exterior no cuentan (por ahí no hay flujo).
        """
        dx = self.grid.dx
        jx_f = self.face_x * (phi[:, :-1] - phi[:, 1:]) / dx
        jz_f = self.face_z * (phi[:-1, :] - phi[1:, :]) / dx
        jx = np.zeros(self.grid.shape, dtype=complex)
        jz = np.zeros(self.grid.shape, dtype=complex)
        nx = np.zeros(self.grid.shape)
        nz = np.zeros(self.grid.shape)
        jx[:, :-1] += jx_f
        jx[:, 1:] += jx_f
        jz[:-1, :] += jz_f
        jz[1:, :] += jz_f
        nx[:, :-1] += self.face_x != 0
        nx[:, 1:] += self.face_x != 0
        nz[:-1, :] += self.face_z != 0
        nz[1:, :] += self.face_z != 0
        return jx / np.maximum(nx, 1), jz / np.maximum(nz, 1)

    def current_density_magnitude(self, phi: np.ndarray) -> np.ndarray:
        """|J| (A/m²) = sqrt(|Jx|² + |Jz|²) del fasor."""
        jx, jz = self.current_density(phi)
        return np.sqrt(np.abs(jx) ** 2 + np.abs(jz) ** 2)


# --------------------------------------------------------------------------
# Electrodos
# --------------------------------------------------------------------------
@dataclass
class Electrodes:
    """Electrodos como conjuntos de celdas de piel.

    ``cells[k]`` son los índices planos (sobre ``(nz, nx)``) del electrodo k y
    ``centers[k]`` su posición ``(x, z)`` media en metros.
    """

    cells: list[np.ndarray]
    centers: np.ndarray
    widths: np.ndarray | None = None  # ancho nominal (m) de cada electrodo; lo necesita el CEM

    def __len__(self) -> int:
        return len(self.cells)

    def mask(self, k: int, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(int(np.prod(shape)), dtype=bool)
        m[self.cells[k]] = True
        return m.reshape(shape)


def _make_electrodes(phantom: Phantom, groups: list[np.ndarray], width: float | None = None) -> Electrodes:
    g = phantom.grid
    centers = []
    for flat in groups:
        j, i = np.unravel_index(flat, g.shape)
        centers.append((g.x0 + i.mean() * g.dx, g.z0 + j.mean() * g.dx))
    widths = None if width is None else np.full(len(groups), float(width))
    return Electrodes(groups, np.array(centers), widths)


def ring_electrodes(phantom: Phantom, n: int, width: float, start_deg: float = 0.0) -> Electrodes:
    """``n`` electrodos de ancho ``width`` (m) equiespaciados en la piel de un corte transversal.

    Se toman las celdas de tejido que tocan el exterior y se reparten por
    ángulo desde el centroide del tejido (0 = arriba, sentido horario, como en
    ``cross_section_limb``). Se usa el ángulo, no la distancia a lo largo de la
    piel, porque la piel es rugosa y así los electrodos siguen su contorno.
    """
    g = phantom.grid
    tis = tissue_mask(phantom)
    padded = np.pad(tis, 1)
    outside = ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    boundary = tis & outside
    X, Z = g.meshgrid()
    cx, cz = X[tis].mean(), Z[tis].mean()
    flat = np.flatnonzero(boundary)
    theta = np.arctan2(X.ravel()[flat] - cx, -(Z.ravel()[flat] - cz))
    r_mean = np.hypot(X.ravel()[flat] - cx, Z.ravel()[flat] - cz).mean()
    half = width / (2.0 * r_mean)
    groups = []
    for k in range(n):
        t0 = np.radians(start_deg) + 2.0 * np.pi * k / n
        d = np.abs((theta - t0 + np.pi) % (2.0 * np.pi) - np.pi)
        sel = d <= half
        groups.append(flat[sel] if sel.any() else flat[[np.argmin(d)]])
    return _make_electrodes(phantom, groups, width)


def surface_electrodes(phantom: Phantom, n: int, width: float, margin: float | None = None) -> Electrodes:
    """``n`` electrodos de ancho ``width`` (m) equiespaciados en la superficie superior de un corte longitudinal.

    Cada electrodo ocupa la primera celda de tejido de las columnas que
    cubre. Los extremos se separan ``margin`` (por defecto media anchura) de
    los bordes laterales del tejido.
    """
    g = phantom.grid
    tis = tissue_mask(phantom)
    cols = np.flatnonzero(tis.any(axis=0))
    first = np.argmax(tis, axis=0)
    margin = width / 2.0 if margin is None else margin
    xs = g.x[cols]
    centers = np.linspace(xs.min() + margin, xs.max() - margin, n)
    groups = []
    for xc in centers:
        sel = cols[np.abs(g.x[cols] - xc) <= width / 2.0]
        if sel.size == 0:
            sel = cols[[np.argmin(np.abs(g.x[cols] - xc))]]
        groups.append(first[sel] * g.nx + sel)
    return _make_electrodes(phantom, groups, width)


# --------------------------------------------------------------------------
# Protocolo adyacente y problema directo de EIT
# --------------------------------------------------------------------------
def adjacent_protocol(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patrón adyacente estándar: ``(inyección, trama, medida)`` de cada dato.

    La trama ``p`` inyecta entre los electrodos ``p`` y ``p+1`` y se mide entre
    cada par adyacente ``(q, q+1)`` que no comparta electrodo con la
    inyección (la tensión junto a la fuente depende del contacto, no del
    interior). Hay ``n (n - 3)`` datos. Devuelve ``frames`` (n_datos,) y
    ``meas`` (n_datos, 2) con los electrodos de cada medida.
    """
    frames, meas = [], []
    for p in range(n):
        for q in range(n):
            if q in (p, (p + 1) % n, (p - 1) % n):
                continue
            frames.append(p)
            meas.append((q, (q + 1) % n))
    return np.array(frames), np.array(meas)


@dataclass
class ContactModel:
    """Contacto del modelo completo de electrodo (CEM) discretizado sobre las celdas de los electrodos.

    ``g`` es la conductancia de contacto de cada celda (S por metro de
    profundidad), ``dg`` su derivada respecto al cambio relativo de la
    admitividad de esa celda (entra en el jacobiano) y ``elec`` el electrodo al
    que pertenece (-1 fuera de los electrodos).
    """

    z_c: np.ndarray  # (n,) impedancia de contacto de cada electrodo (ohm·m²)
    g: np.ndarray  # (nz, nx)
    dg: np.ndarray  # (nz, nx)
    elec: np.ndarray  # (nz, nx) int


def build_contact(solver: QuasiStaticSolver, electrodes: Electrodes, z_c: float | complex | np.ndarray) -> ContactModel:
    """Conductancias de contacto del CEM.

    Unidades: la impedancia de contacto ``z_c`` se da en ohm·m² (resistencia
    por área de piel, la unidad en que se tabula para electrodos reales). En el
    corte 2D se calcula por metro de profundidad: un tramo de electrodo de
    longitud ``l`` tiene área ``l · 1 m`` y conductancia ``l / z_c`` (S/m de
    profundidad), la misma unidad que las admitividades de cara ``sigma*``.

    El ancho nominal ``w_l`` del electrodo (no la longitud en escalera de sus
    celdas, que depende de la malla) se reparte a partes iguales entre sus
    celdas. A ``z_c`` se suma el camino del centro de la celda a la superficie,
    ``(dx/2)/sigma*``: así ``z_c -> 0`` da un electrodo ideal en la superficie y
    no un electrodo medio celda dentro del tejido (error de primer orden).
    """
    n = len(electrodes)
    z = np.broadcast_to(np.asarray(z_c, dtype=complex), (n,)).copy()
    shape = solver.grid.shape
    h = 0.5 * solver.grid.dx
    g = np.zeros(shape, dtype=complex)
    dg = np.zeros(shape, dtype=complex)
    elec = np.full(shape, -1, dtype=int)
    for k, cells in enumerate(electrodes.cells):
        width = electrodes.widths[k] if electrodes.widths is not None else cells.size * solver.grid.dx
        length = width / cells.size
        j, i = np.unravel_index(cells, shape)
        y = solver.y[j, i]
        gc = length / (z[k] + h / y)
        g[j, i] = gc
        # g = l/(z + h/y); y dg/dy = g² (h/y) / l
        dg[j, i] = gc**2 * (h / y) / length
        elec[j, i] = k
    return ContactModel(z, g, dg, elec)


@dataclass
class EITResult:
    solver: QuasiStaticSolver
    electrodes: Electrodes
    current: float
    fields: np.ndarray  # (n_tramas, nz, nx) potencial de cada inyección (V)
    electrode_potentials: np.ndarray  # (n_tramas, n_electrodos) potencial medio de cada electrodo
    frames: np.ndarray  # (n_datos,) trama de cada dato
    meas: np.ndarray  # (n_datos, 2) electrodos entre los que se mide
    voltages: np.ndarray  # (n_datos,) tensiones diferenciales complejas (V)
    iterations: int  # máximo de iteraciones de COCG entre tramas
    residual: float  # peor residuo relativo entre tramas
    contact: ContactModel | None = None  # None = electrodo ideal; si no, CEM (electrode_potentials = U_l)


def _simulate_cem(solver: QuasiStaticSolver, electrodes: Electrodes, current: float, z_c, tol: float) -> EITResult:
    """CEM por complemento de Schur sobre las tensiones de electrodo.

    Con ``U`` fijas, ``(A + G) phi = G U`` es un problema de Dirichlet-Robin
    bien planteado. Se resuelve un campo ``psi_l`` por electrodo (``U_l = 1``,
    el resto 0) y la matriz de transferencia ``S[k, l]`` = corriente que entra
    por ``k`` con ``U = e_l`` (simétrica, filas de suma nula, es la matriz de
    "tensión a corriente" del CEM). Para cualquier patrón de corrientes ``I``,
    ``S U = I`` (con ``sum U = 0``) da las tensiones y el campo es ``sum U_l psi_l``.
    Cuesta lo mismo que el modo ideal (un COCG por electrodo) y sirve para
    cualquier protocolo.
    """
    n = len(electrodes)
    shape = solver.grid.shape
    contact = build_contact(solver, electrodes, z_c)
    cem = QuasiStaticSolver(solver.phantom, solver.f, admittivity_map=solver.y, shunt=contact.g)
    psi = np.zeros((n,) + shape, dtype=complex)
    smat = np.zeros((n, n), dtype=complex)
    its, worst = 0, 0.0
    for lidx in range(n):
        r = cem.solve(np.where(contact.elec == lidx, contact.g, 0.0), tol=tol)
        psi[lidx] = r.phi
        its, worst = max(its, r.iterations), max(worst, r.residual)
        for k in range(n):
            m = contact.elec == k
            smat[k, lidx] = (contact.g[m].sum() if k == lidx else 0.0) - (contact.g[m] * r.phi[m]).sum()
    # Referencia de tensión: sum U = 0 (se añade el vector 1 a S, que es singular con núcleo 1)
    sreg = smat + np.abs(np.diag(smat)).mean() / n * np.ones((n, n))
    pattern = np.zeros((n, n), dtype=complex)
    pattern[np.arange(n), np.arange(n)] = current
    pattern[np.arange(n), (np.arange(n) + 1) % n] = -current
    pots = np.linalg.solve(sreg, pattern.T).T
    fields = np.einsum("pl,lzx->pzx", pots, psi)
    frames, meas = adjacent_protocol(n)
    volts = pots[frames, meas[:, 0]] - pots[frames, meas[:, 1]]
    return EITResult(solver, electrodes, current, fields, pots, frames, meas, volts, its, worst, contact)


def simulate_eit(solver: QuasiStaticSolver, electrodes: Electrodes, current: float = 1.0,
                 tol: float = 1e-10, contact_impedance: float | complex | np.ndarray | None = None) -> EITResult:
    """Problema directo: tensiones diferenciales del patrón adyacente.

    ``current`` es la amplitud inyectada (A por metro de profundidad). Por
    defecto (``contact_impedance=None``) cada electrodo reparte su corriente
    por igual entre sus celdas y su potencial es el medio de ellas (electrodo
    ideal, sin impedancia de contacto, sin derivación).

    Con ``contact_impedance`` (ohm·m², escalar o uno por electrodo, puede ser
    complejo) se usa el modelo completo de electrodo: cada electrodo es
    equipotencial en el metal, con una caída ``J z_c`` en el contacto, y la
    corriente se reparte según la resistencia del contacto y del tejido
    (derivación o *shunting*). Los electrodos necesitan ancho nominal
    (``ring_electrodes`` y ``surface_electrodes`` lo guardan) y
    ``electrode_potentials`` pasa a ser la tensión de metal ``U_l``.
    """
    if contact_impedance is not None:
        return _simulate_cem(solver, electrodes, current, contact_impedance, tol)
    n = len(electrodes)
    shape = solver.grid.shape
    fields = np.zeros((n,) + shape, dtype=complex)
    pots = np.zeros((n, n), dtype=complex)
    its, worst = 0, 0.0
    for p in range(n):
        b = np.zeros(int(np.prod(shape)), dtype=complex)
        src, snk = electrodes.cells[p], electrodes.cells[(p + 1) % n]
        b[src] += current / src.size
        b[snk] -= current / snk.size
        r = solver.solve(b.reshape(shape), tol=tol)
        fields[p] = r.phi
        its, worst = max(its, r.iterations), max(worst, r.residual)
        flat = r.phi.ravel()
        pots[p] = [flat[c].mean() for c in electrodes.cells]
    frames, meas = adjacent_protocol(n)
    volts = pots[frames, meas[:, 0]] - pots[frames, meas[:, 1]]
    return EITResult(solver, electrodes, current, fields, pots, frames, meas, volts, its, worst)


# --------------------------------------------------------------------------
# Sensibilidad y reconstrucción de diferencias
# --------------------------------------------------------------------------
@dataclass
class Sensitivity:
    matrix: np.ndarray  # (n_datos, n_parametros) complejo: dV / (cambio relativo de admitividad)
    coarse_shape: tuple[int, int]  # malla gruesa (nzc, nxc)
    factor: int  # celdas finas por celda gruesa en cada eje
    active: np.ndarray  # (nzc, nxc) booleano: celdas gruesas con tejido (los parámetros)

    def to_image(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Coloca el vector de parámetros en la malla gruesa."""
        img = np.full(self.coarse_shape, fill, dtype=values.dtype)
        img[self.active] = values
        return img

    def upsample(self, img: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
        """Repite la imagen gruesa sobre la malla fina (para superponerla al fantoma)."""
        big = np.kron(img, np.ones((self.factor, self.factor)))
        return big[:grid_shape[0], :grid_shape[1]]


def _block_sum(a: np.ndarray, factor: int, coarse_shape: tuple[int, int]) -> np.ndarray:
    nzc, nxc = coarse_shape
    pad = np.zeros((nzc * factor, nxc * factor), dtype=a.dtype)
    pad[:a.shape[0], :a.shape[1]] = a
    return pad.reshape(nzc, factor, nxc, factor).sum(axis=(1, 3))


def sensitivity_matrix(eit: EITResult, factor: int = 6) -> Sensitivity:
    """Jacobiano de las tensiones respecto al cambio relativo de admitividad de cada celda gruesa.

    Si la admitividad de las celdas finas de un bloque se multiplica por
    ``1 + d``, la tensión de la medida ``m`` (inyección ``p``) cambia
    ``dV = sum_caras -dg/dd * (grad phi_p)(grad phi_q) * d``, donde ``phi_q`` es
    el campo de inyectar la corriente unitaria entre los electrodos de la
    medida (reciprocidad: es el mismo conjunto de campos que ya se calculó
    para inyectar). Se evita así resolver un problema adjunto por dato.

    Es un modelo linealizado alrededor de la configuración ``eit``
    (típicamente la sana), válido para cambios pequeños.
    """
    s = eit.solver
    g = s.grid
    coarse_shape = (-(-g.nz // factor), -(-g.nx // factor))
    # d(g_cara)/d(d_a) = g * y_b / (y_a + y_b) con y_a, y_b de las celdas de la cara
    with np.errstate(divide="ignore", invalid="ignore"):
        wxa = np.where(s.face_x != 0, s.face_x * s._yb_x / (s._ya_x + s._yb_x), 0.0)
        wxb = np.where(s.face_x != 0, s.face_x * s._ya_x / (s._ya_x + s._yb_x), 0.0)
        wza = np.where(s.face_z != 0, s.face_z * s._yb_z / (s._ya_z + s._yb_z), 0.0)
        wzb = np.where(s.face_z != 0, s.face_z * s._ya_z / (s._ya_z + s._yb_z), 0.0)
    phi = eit.fields / eit.current  # campo de la medida con corriente unitaria
    gx = phi[:, :, :-1] - phi[:, :, 1:]
    gz = phi[:, :-1, :] - phi[:, 1:, :]
    gx_inj = eit.fields[:, :, :-1] - eit.fields[:, :, 1:]
    gz_inj = eit.fields[:, :-1, :] - eit.fields[:, 1:, :]

    active = _block_sum(s.tissue.astype(float), factor, coarse_shape) > 0
    n_meas = eit.frames.size
    mat = np.zeros((n_meas, int(active.sum())), dtype=complex)
    for m in range(n_meas):
        p, q = eit.frames[m], eit.meas[m, 0]  # la medida (q, q+1) equivale a la trama q
        px = gx_inj[p] * gx[q]
        pz = gz_inj[p] * gz[q]
        cell = np.zeros(g.shape, dtype=complex)
        cell[:, :-1] += wxa * px
        cell[:, 1:] += wxb * px
        cell[:-1, :] += wza * pz
        cell[1:, :] += wzb * pz
        if eit.contact is not None:
            # El contacto también depende de la admitividad de la celda de piel: mismo producto de caídas
            c = eit.contact
            e = np.where(c.elec >= 0, c.elec, 0)
            dp = np.where(c.elec >= 0, eit.fields[p] - eit.electrode_potentials[p][e], 0.0)
            dq = np.where(c.elec >= 0, eit.fields[q] - eit.electrode_potentials[q][e], 0.0)
            cell += c.dg * dp * dq / eit.current
        mat[m] = -_block_sum(cell, factor, coarse_shape)[active]
    return Sensitivity(mat, coarse_shape, factor, active)


def reconstruct_difference(sens: Sensitivity, delta_v: np.ndarray, reg: float = 1e-2) -> np.ndarray:
    """Reconstrucción de diferencias de un paso con regularización de Tikhonov.

    Resuelve ``min |J d - dV|² + reg · d' diag(J'J) d`` (tipo NOSER): escalar
    la regularización por la diagonal de ``J'J`` compensa que las celdas
    centrales son mucho menos sensibles que las de la periferia; con una
    identidad el resultado se pegaría a los electrodos. Los datos son
    complejos y el cambio de admitividad se toma real, así que se apilan
    parte real e imaginaria. Devuelve el cambio relativo de admitividad por
    celda gruesa activa (el vector que ``Sensitivity.to_image`` coloca en la malla).
    """
    jac = np.vstack([sens.matrix.real, sens.matrix.imag])
    data = np.concatenate([np.real(delta_v), np.imag(delta_v)])
    hess = jac.T @ jac
    return np.linalg.solve(hess + reg * np.diag(np.diag(hess)), jac.T @ data)
