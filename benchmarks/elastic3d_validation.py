"""Validación del solver elástico 3D (``wavesim.ultrasound.elastic3d``) y caso de transmisión axial en un tubo.

Uso (desde la raíz del repositorio; numpy y, para que no tarde horas, numba):

    NUMBA_NUM_THREADS=1 python benchmarks/elastic3d_validation.py                # todo (~1 h con 1 hilo)
    python benchmarks/elastic3d_validation.py --parte velocidades                # ondas planas P y S (~1 min)
    python benchmarks/elastic3d_validation.py --parte extrusion                  # 3D extruido frente a 2D (~1 min)
    python benchmarks/elastic3d_validation.py --parte green                      # fuente puntual en agua (~3 min)
    python benchmarks/elastic3d_validation.py --parte reflexion                  # incidencia normal (~1 min)
    python benchmarks/elastic3d_validation.py --parte oblicua                    # R(45°) en los 3 planos (~4 min)
    python benchmarks/elastic3d_validation.py --parte estabilidad --pasos 30000  # tubo en la CPML (~20 min)
    python benchmarks/elastic3d_validation.py --parte rendimiento --hilos 1 4    # memoria y pasos/s (~3 min)
    python benchmarks/elastic3d_validation.py --parte aplicacion                 # tubo frente a placa 2D (~25 min)

Partes:

1. ``velocidades``: ondas planas (campos uniformes en dos ejes) en agua y en
   cortical isótropa a lo largo de cada eje y con cada polarización, y en la
   cortical transversalmente isótropa (eje en x). Velocidad por correlación
   entre dos receptores.
2. ``extrusion``: tibia 2D con fractura desplazada extruida en y, fuente
   uniforme en y, mismo dt: diferencia de las trazas con el solver 2D.
3. ``green``: fuente de presión de una celda en agua frente a
   ``p = dx² s'(t - r/c) / (2 pi r c)`` (amplitud, decaimiento 1/r, retardo y
   forma de onda) a 10 y 15 puntos por longitud de onda.
4. ``reflexion``: incidencia normal agua-cortical (cociente espectral en f0)
   frente a (Z2 - Z1)/(Z2 + Z1), isótropa y anisótropa.
5. ``oblicua``: |R(45°)| agua-cortical con un haz gaussiano inclinado en cada
   uno de los planos x-z, y-z y x-y (cada uno ejercita una componente de
   cizalla de la corrección de interfaz), con y sin la corrección, frente a la
   fórmula cerrada del semiespacio sólido.
6. ``estabilidad``: energía a largo plazo de un tubo con fractura que atraviesa
   la CPML en x (o que termina antes, ``bone_length``), con y sin atenuación,
   isótropo y anisótropo.
7. ``rendimiento``: memoria por celda y pasos por segundo del caso objetivo
   (40 × 24 × 24 mm a 0.5 y 1 MHz) con 1 y N hilos.
8. ``aplicacion``: transmisión axial sobre la tibia tubular (3D) frente al
   corte longitudinal 2D (la placa) a 0.5 y 1 MHz: velocidad del primer arribo
   (FAS) en función de la distancia y efecto de una fractura transversal
   completa o incompleta en su amplitud.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavesim.core.grid3d import Grid3D  # noqa: E402
from wavesim.core.signals import amplitude_at, gaussian_pulse  # noqa: E402
from wavesim.materials.acoustic import ACOUSTIC, CORTICAL_STIFFNESS  # noqa: E402
from wavesim.phantoms import Fracture, longitudinal_limb  # noqa: E402
from wavesim.phantoms.limb3d import (  # noqa: E402
    LegSegmentSpec, TransverseFracture3D, extrude, homogeneous_phantom, leg_segment,
)
from wavesim.ultrasound import ElasticSolver, Receiver, Source  # noqa: E402
from wavesim.ultrasound.elastic3d import (  # noqa: E402
    ElasticMedium3D, ElasticSolver3D, Receiver3D, Source3D, isotropic_stiffness_3d, point_source_pressure,
)

F0 = 1e6
WATER = (1000.0, 1500.0, 0.0)
BONE = (1908.0, 3500.0, 1800.0)
RHO_B = ACOUSTIC["cortical_bone"].density
TI = np.array([CORTICAL_STIFFNESS.c_a, CORTICAL_STIFFNESS.c_t, CORTICAL_STIFFNESS.c_ta, CORTICAL_STIFFNESS.g_a,
               CORTICAL_STIFFNESS.g_t]) / RHO_B


def medium(shape, rho, cp, cs, stiffness=None):
    full = lambda v: np.broadcast_to(np.asarray(v, np.float32), shape).copy()  # noqa: E731
    z = np.zeros((0, *shape), np.float32)
    return ElasticMedium3D(full(rho), full(cp), full(cs), z, z.copy(), np.zeros(0), F0, stiffness)


def solver(grid, med, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElasticSolver3D(homogeneous_phantom(grid), F0, medium=med, **kw)


def lag(a, b, dt, up=16):
    """Retardo de ``b`` respecto a ``a`` (correlación cruzada sobremuestreada y ajuste parabólico)."""
    n = a.size
    spec = np.fft.rfft(b, 2 * n) * np.conj(np.fft.rfft(a, 2 * n))
    xc = np.fft.irfft(spec, 2 * n * up)
    k = int(np.argmax(xc))
    y0, y1, y2 = xc[k - 1], xc[k], xc[(k + 1) % xc.size]
    k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    if k > xc.size / 2:
        k -= xc.size
    return k * dt / up


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


# ----------------------------------------------------------------------------
# 1. Velocidades
# ----------------------------------------------------------------------------
def plane_wave_speed(axis, kind, quantity, rho, cp, cs, stiffness=None, ppw=10, c_ref=1500.0):
    dx = c_ref / F0 / ppw
    n = [4, 4, 4]
    n[axis] = int(round(40e-3 / dx)) + 24
    grid = Grid3D(n[2], n[1], n[0], dx)
    npml = [0, 0, 0]
    npml[axis] = 12
    st = None if stiffness is None else np.broadcast_to(np.asarray(stiffness, np.float32)[:, None, None, None],
                                                        (5, *grid.shape)).copy()
    s = solver(grid, medium(grid.shape, rho, cp, cs, st), npml=tuple(npml))
    nt = int(30e-6 / s.dt)
    t = s.time_axis(nt)

    def plane(index):
        idx = [np.arange(4)] * 3
        idx[axis] = np.array([index])
        return tuple(a.ravel() for a in np.meshgrid(*idx, indexing="ij"))

    k, j, i = plane(16)
    s.add_source(Source3D(k, j, i, np.zeros(k.size, int), gaussian_pulse(t, F0)[None, :], kind))
    ia, ib = 16 + int(round(10e-3 / dx)), 16 + int(round(30e-3 / dx))
    for name, index in (("a", ia), ("b", ib)):
        k, j, i = plane(index)
        s.add_receiver(name, Receiver3D(k, j, i, np.zeros(k.size, int), 1, quantity))
    r = s.run(nt)
    return (ib - ia) * dx / lag(r.traces["a"][0], r.traces["b"][0], s.dt)


def part_velocidades():
    print("## Velocidades de ondas planas\n")
    axes = {0: "z", 1: "y", 2: "x"}
    aniso = (RHO_B, np.sqrt(TI[0]), np.sqrt(TI[3]), TI)
    cases = [
        ("P agua", 0, "pressure", "p", WATER, None, 10, 1500.0, 1500.0),
        ("P agua", 1, "pressure", "p", WATER, None, 10, 1500.0, 1500.0),
        ("P agua", 2, "pressure", "p", WATER, None, 10, 1500.0, 1500.0),
        ("P cortical", 0, "pressure", "p", BONE, None, 10, 3500.0, 3500.0),
        ("S cortical (vx)", 0, "force_x", "vx", BONE, None, 10, 1800.0, 1800.0),
        ("S cortical (vy)", 0, "force_y", "vy", BONE, None, 10, 1800.0, 1800.0),
        ("S cortical (vz)", 2, "force_z", "vz", BONE, None, 10, 1800.0, 1800.0),
        ("S cortical (vx)", 1, "force_x", "vx", BONE, None, 10, 1800.0, 1800.0),
        ("P axial, anisótropa", 2, "pressure", "p", aniso[:3], TI, 10, np.sqrt(TI[0]), np.sqrt(TI[0])),
        ("P transversal, anisótropa", 0, "pressure", "p", aniso[:3], TI, 10, np.sqrt(TI[1]), np.sqrt(TI[1])),
        ("S plano con el eje (G_A)", 0, "force_x", "vx", aniso[:3], TI, 10, np.sqrt(TI[3]), np.sqrt(TI[3])),
        ("S plano transversal (G_T)", 0, "force_y", "vy", aniso[:3], TI, 10, np.sqrt(TI[4]), np.sqrt(TI[4])),
    ]
    rows = []
    for name, axis, kind, q, props, st, ppw, c_ref, expected in cases:
        v = plane_wave_speed(axis, kind, q, *props, stiffness=st, ppw=ppw, c_ref=c_ref)
        rows.append((name, axes[axis], ppw, f"{expected:.0f}", f"{v:.1f}", f"{100 * (v / expected - 1):+.2f} %"))
    table(("onda", "dirección", "ppw", "teórica (m/s)", "medida (m/s)", "error"), rows)


# ----------------------------------------------------------------------------
# 2. Extrusión
# ----------------------------------------------------------------------------
def part_extrusion(nt=1500):
    print("## Medio extruido con fuente uniforme en y frente al solver 2D\n")
    f0 = 2e6
    ph = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=10e-3, depth=16e-3, dx=1450 / f0 / 8)
    g = ph.grid
    s2 = ElasticSolver(ph, f0, npml=12, cfl=0.6, n_mechanisms=2, scattering=0.0)
    has_slip = hasattr(s2, "_slip")
    i = np.arange(g.nx // 3, g.nx // 3 + 3)
    j = np.full(i.size, 14)
    sig = gaussian_pulse(s2.time_axis(nt), f0)[None, :]
    s2.add_source(Source(j, i, np.zeros(i.size, int), sig, "pressure"))
    rj = np.array([20, 20, 20, g.nz // 2, g.nz // 2, g.nz - 20])
    ri = np.array([g.nx // 4, g.nx // 2, 3 * g.nx // 4, g.nx // 3, 2 * g.nx // 3, g.nx // 2])
    for q in ("p", "vx", "vz"):
        s2.add_receiver(q, Receiver(rj, ri, np.arange(rj.size), rj.size, q))
    r2 = s2.run(nt, track_peak=False)
    rows = []
    for corr in ((has_slip,) if has_slip else (False, True)):
        p3 = extrude(ph, ny=6)
        s3 = ElasticSolver3D(p3, f0, npml=(12, 0, 12), n_mechanisms=2, dt=s2.dt, interface_correction=corr)
        ny = p3.grid.ny
        s3.add_source(Source3D(np.repeat(j, ny), np.tile(np.arange(ny), i.size), np.repeat(i, ny),
                               np.zeros(i.size * ny, int), sig, "pressure"))
        for q in ("p", "vx", "vz"):
            s3.add_receiver(q, Receiver3D(rj, np.full(rj.size, ny // 2), ri, np.arange(rj.size), rj.size, q))
        r3 = s3.run(nt)
        errs = []
        # la presión 2D (-(sxx + szz)/2) y la 3D (-(sxx + syy + szz)/3) sólo coinciden en un fluido
        fluid = s2.medium.cs[rj, ri] == 0
        for q in ("p", "vx", "vz"):
            a, b = r2.traces[q], r3.traces[q]
            sel = fluid if q == "p" else np.ones(rj.size, bool)
            errs.append(np.max(np.abs(a - b).max(axis=1)[sel] / np.abs(a).max(axis=1)[sel]))
        label = f"3D {'con' if corr else 'sin'} corrección frente a 2D {'con' if has_slip else 'sin'} ella"
        rows.append((label, nt, *(f"{e:.1e}" for e in errs)))
    table(("caso", "pasos", "máx |Δp|/máx|p|", "Δvx", "Δvz"), rows)
    print("(diferencia máxima entre 6 receptores, sólo los que están en fluido para la presión; cada traza normalizada "
          "por su máximo)\n")


# ----------------------------------------------------------------------------
# 3. Función de Green
# ----------------------------------------------------------------------------
def part_green():
    print("## Fuente puntual en agua frente a la función de Green 3D\n")
    c = 1500.0
    h = 1e-12
    deriv = lambda tt: (gaussian_pulse(tt + h, F0) - gaussian_pulse(tt - h, F0)) / (2 * h)  # noqa: E731
    rows, slopes = [], []
    for ppw, n in ((10, 96), (15, 132)):
        dx = c / F0 / ppw
        grid = Grid3D(n, n, n, dx)
        s = solver(grid, medium(grid.shape, *WATER), npml=10)
        centre = 10 + 6
        radii = np.unique(np.round(np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * ppw).astype(int))
        radii = radii[centre + radii < n - 16]
        nt = int((radii.max() * dx / c + 7e-6) / s.dt)
        t = s.time_axis(nt)
        s.add_source(Source3D(np.array([centre]), np.array([n // 2]), np.array([n // 2]), np.array([0]),
                              gaussian_pulse(t, F0)[None, :]))
        s.add_receiver("r", Receiver3D(centre + radii, np.full(radii.size, n // 2), np.full(radii.size, n // 2),
                                       np.arange(radii.size), radii.size))
        tr = s.run(nt).traces["r"]
        amps = []
        for q, rc in enumerate(radii):
            ref = point_source_pressure(rc * dx, t, deriv, dx, c, s.dt)
            err = np.sqrt(np.mean((tr[q] - ref) ** 2)) / np.sqrt(np.mean(ref**2))
            dly = lag(ref, tr[q], s.dt)
            # forma de onda tras quitar el retardo de la dispersión numérica
            ref_al = point_source_pressure(rc * dx, t - dly, deriv, dx, c, s.dt)
            err_al = np.sqrt(np.mean((tr[q] - ref_al) ** 2)) / np.sqrt(np.mean(ref_al**2))
            amp = np.abs(tr[q]).max() / np.abs(ref).max()
            amps.append(np.abs(tr[q]).max())
            c_eff = rc * dx / (rc * dx / c + dly)
            rows.append((ppw, f"{rc / ppw:.1f}", f"{amp:.4f}", f"{dly * 1e9:+.1f}", f"{100 * (c_eff / c - 1):+.2f} %",
                         f"{100 * err:.1f} %", f"{100 * err_al:.2f} %"))
        slopes.append((ppw, np.polyfit(np.log(radii), np.log(amps), 1)[0]))
    table(("ppw", "r/λ", "amplitud / analítica", "retardo (ns)", "velocidad aparente", "error rms",
           "error rms sin retardo"), rows)
    for ppw, sl in slopes:
        print(f"- {ppw} ppw: exponente del decaimiento de la amplitud con r = {sl:.4f} (analítico: -1)")
    print()


# ----------------------------------------------------------------------------
# 4. Reflexión normal
# ----------------------------------------------------------------------------
def normal_reflection(ppw, axis=0, aniso=False):
    dx = 1500.0 / F0 / ppw
    n = [4, 4, 4]
    n[axis] = int(round(40e-3 / dx))
    grid = Grid3D(n[2], n[1], n[0], dx)
    coord = (grid.z, grid.y, grid.x)[axis]
    shape = [1, 1, 1]
    shape[axis] = -1
    bone = np.broadcast_to((coord >= 24e-3).reshape(shape), grid.shape)
    rho, cp, cs = (np.where(bone, b, w) for b, w in zip(BONE, WATER))
    st, z_bone = None, BONE[0] * BONE[1]
    if aniso:
        st = isotropic_stiffness_3d(cp, cs).astype(np.float32)
        st[:, bone] = TI[:, None]
        rho = np.where(bone, RHO_B, WATER[0])
        z_bone = np.sqrt(RHO_B * RHO_B * (TI[0] if axis == 2 else TI[1]))
    npml = [0, 0, 0]
    npml[axis] = 20
    s = solver(grid, medium(grid.shape, rho, cp, cs, st), npml=tuple(npml))
    nt = int(34e-6 / s.dt)
    t = s.time_axis(nt)

    def plane(pos):
        idx = [np.arange(4)] * 3
        idx[axis] = np.array([int(round(pos / dx))])
        return tuple(a.ravel() for a in np.meshgrid(*idx, indexing="ij"))

    k, j, i = plane(5e-3)
    s.add_source(Source3D(k, j, i, np.zeros(k.size, int), gaussian_pulse(t, F0)[None, :]))
    k, j, i = plane(12e-3)
    s.add_receiver("r", Receiver3D(k, j, i, np.zeros(k.size, int), 1))
    tr = s.run(nt).traces["r"][0]
    t_split = (24e-3 - 12e-3) / 1500 + 6e-6
    inc, refl = np.where(t < t_split, tr, 0.0), np.where(t >= t_split, tr, 0.0)
    z_w = WATER[0] * WATER[1]
    r_an = (z_bone - z_w) / (z_bone + z_w)
    return abs(amplitude_at(refl, s.dt, F0) / amplitude_at(inc, s.dt, F0)), r_an


def part_reflexion():
    print("## Reflexión a incidencia normal agua-cortical (|R| en f0)\n")
    rows = []
    for name, axis, aniso, ppws in (("isótropa", 0, False, (8, 12, 16, 24)),
                                     ("anisótropa, transversal (z)", 0, True, (12,)),
                                     ("anisótropa, axial (x)", 2, True, (12,))):
        for ppw in ppws:
            r, r_an = normal_reflection(ppw, axis, aniso)
            rows.append((name, ppw, f"{r_an:.4f}", f"{r:.4f}", f"{100 * (r / r_an - 1):+.2f} %"))
    table(("cortical", "ppw (agua)", "R analítico", "R medido", "error"), rows)


# ----------------------------------------------------------------------------
# 5. Incidencia oblicua
# ----------------------------------------------------------------------------
def fluid_solid_r(theta, rho1, c1, rho2, cp, cs):
    """|R(θ)| de una onda plana en un fluido sobre un semiespacio sólido (fórmula cerrada, Brekhovskikh)."""
    s = np.sin(theta)
    cos = lambda x: np.sqrt(1 - x**2 + 0j)  # noqa: E731
    sp, ss = cp * s / c1, cs * s / c1
    z1 = rho1 * c1 / np.cos(theta)
    zp, zs = rho2 * cp / cos(sp), rho2 * cs / cos(ss)
    th_s2 = 2 * np.arcsin(ss + 0j)
    num = zp * np.cos(th_s2) ** 2 + zs * np.sin(th_s2) ** 2
    return np.abs((num - z1) / (num + z1))


def beam_reflection(plane, theta_deg, ppw, sigma_lam=3.0):
    """|R| con un haz gaussiano inclinado (descomposición en ondas planas a lo largo de una línea, como en 2D).

    ``plane``: (eje normal a la interfaz, eje tangencial); el tercero es
    uniforme (4 celdas, sin CPML). Devuelve el error relativo de |R| en 7
    ángulos alrededor de ``theta_deg`` sin y con la corrección de interfaz
    (la simulación de referencia, sólo agua, es común).
    """
    n_ax, t_ax = plane
    u_ax = 3 - n_ax - t_ax
    lam = 1500.0 / F0
    dx = lam / ppw
    th = np.radians(theta_deg)
    npml = 20
    sig = sigma_lam * lam
    j_s = npml + int(round(lam / dx))
    j_r = j_s + int(round(lam / dx))
    j_int = j_r + int(round(lam / dx))
    nn = j_int + int(round(3 * lam / dx)) + npml
    x_c = 3.5 * sig
    nt_ = int(np.ceil((x_c + 3 * lam * np.tan(th) + 3.5 * sig + 2 * lam) / dx)) + 2 * npml
    x = (np.arange(nt_) - npml) * dx
    src = np.nonzero(np.abs(x - x_c) <= 3 * sig)[0]
    delay = (x[src] - x_c) * np.sin(th) / 1500.0
    delay -= delay.min()
    amp = np.exp(-0.5 * ((x[src] - x_c) / sig) ** 2)
    dims = [0, 0, 0]
    dims[n_ax], dims[t_ax], dims[u_ax] = nn, nt_, 4
    grid = Grid3D(dims[2], dims[1], dims[0], dx)
    npmls = [0, 0, 0]
    npmls[n_ax] = npmls[t_ax] = npml
    acc = {}
    for name, solid, corr in (("ref", False, True), ("sin", True, False), ("con", True, True)):
        idx_n = np.arange(nn).reshape([-1 if a == n_ax else 1 for a in range(3)])
        bone = np.broadcast_to(idx_n >= j_int, grid.shape) & solid
        rho, cp, cs = (np.where(bone, b, w) for b, w in zip(BONE, WATER))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ElasticSolver3D(homogeneous_phantom(grid), F0, npml=tuple(npmls),
                                medium=medium(grid.shape, rho, cp, cs), cfl=0.9 if solid else 0.9 * 1500 / 3500,
                                interface_correction=corr)
        nt = int(np.ceil((delay.max() + 3 * lam / (1500 * np.cos(th)) + 10e-6) / s.dt))
        t = s.time_axis(nt)
        cells = [None, None, None]
        cells[n_ax], cells[t_ax], cells[u_ax] = np.array([j_s]), src, np.arange(4)
        kk, jj, ii = (a.ravel() for a in np.meshgrid(*cells, indexing="ij"))
        # un canal por posición a lo largo del eje tangencial
        chan = np.broadcast_to(np.arange(src.size).reshape([-1 if a == t_ax else 1 for a in range(3)]),
                               [len(c) for c in cells]).ravel()
        s.add_source(Source3D(kk, jj, ii, chan, amp[:, None] * gaussian_pulse(t[None, :] - delay[:, None], F0)))
        line = [slice(None)] * 3
        line[n_ax], line[u_ax] = j_r, 1
        line = tuple(line)
        a = np.zeros(nt_, complex)
        for n in range(nt):
            s.step(n)
            a += np.exp(2j * np.pi * F0 * t[n]) * s.pressure(*line)
        acc[name] = a
    thetas = np.radians(theta_deg + np.linspace(-1.5, 1.5, 7))
    E = np.exp(-1j * (2 * np.pi * F0 * np.sin(thetas) / 1500.0)[:, None] * x[None, :])
    r_an = fluid_solid_r(thetas, *WATER[:2], *BONE)
    return {k: np.abs(E @ (acc[k] - acc["ref"])) / np.abs(E @ acc["ref"]) / r_an - 1 for k in ("sin", "con")}


def part_oblicua(ppw=12):
    print(f"## Reflexión oblicua agua-cortical a 45° ({ppw} ppw en agua)\n")
    planes = {"x-z (sxz)": (0, 2), "y-z (syz)": (0, 1), "x-y (sxy)": (1, 2)}
    r45 = fluid_solid_r(np.radians(45.0), *WATER[:2], *BONE)
    rows = []
    for name, plane in planes.items():
        errs = beam_reflection(plane, 45.0, ppw)
        for key, err in errs.items():
            rows.append((name, key, f"{100 * np.abs(err).max():.2f} %", f"{100 * err[3]:+.2f} %"))
    print(f"|R(45°)| analítico = {r45:.4f}\n")
    table(("plano de incidencia", "corrección", "máx |error| en 43.5-46.5°", "error a 45°"), rows)


# ----------------------------------------------------------------------------
# 6. Estabilidad a largo plazo
# ----------------------------------------------------------------------------
def long_run(nt, attenuation, aniso, short, npml=6):
    spec = LegSegmentSpec(outer_radius=2.5e-3, cortex=1e-3, muscle=0.5e-3, fat=0.5e-3, skin=0.5e-3, standoff=1e-3)
    dx, length = 0.3e-3, 8e-3
    ph = leg_segment(spec, TransverseFracture3D(gap=0.6e-3, position=0.4e-3, arc_deg=200.0), length=length, dx=dx,
                     margin=0.9e-3, bone_length=(length - 2 * (npml + 3) * dx) if short else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = ElasticSolver3D(ph, 1e6, npml=npml, attenuation=attenuation, anisotropic_bone=aniso)
    t = s.time_axis(nt)
    k, j, i = ph.grid.index_of(0.0, 0.0, 0.6e-3)
    s.add_source(Source3D(np.array([k]), np.array([j]), np.array([i]), np.array([0]), gaussian_pulse(t, 1e6)[None, :]))
    hist = []
    for n in range(nt):
        s.step(n)
        if n % 500 == 0:
            hist.append(float(np.sqrt(np.mean(s.vx**2 + s.vy**2 + s.vz**2))))
            if not np.isfinite(hist[-1]):
                break
    hist = np.array(hist)
    return hist


def part_estabilidad(nt=30000):
    print(f"## Estabilidad a largo plazo ({nt} pasos, tubo con fractura incompleta)\n")
    rows = []
    for aniso in (False, True):
        for att, short in ((False, False), (False, True), (True, False)):
            h = long_run(nt, att, aniso, short)
            k5 = min(10, h.size - 1)  # paso 5000: la onda ya ha salido del dominio
            late = h[k5:]
            ratio = late[-1] / late.min()
            rows.append(("anisótropa" if aniso else "isótropa", "sí" if att else "no",
                         "termina antes" if short else "entra en la CPML", f"{h.max():.1e}", f"{h[k5]:.1e}",
                         f"{h[-1]:.1e}", f"{ratio:.1f}"))
    table(("cortical", "atenuación", "tubo en x", "rms(v) máx", "rms(v) paso 5000", "rms(v) final",
           "final / mín. tardío"), rows)


# ----------------------------------------------------------------------------
# 7. Rendimiento
# ----------------------------------------------------------------------------
def part_rendimiento(threads=(1,), steps=20):
    try:
        import numba
    except ImportError:
        numba = None
    load = os.getloadavg()
    print(f"## Memoria y velocidad del caso objetivo (40 × 24 × 24 mm, tibia cortada por el dominio)\n\n"
          f"Carga de la máquina al empezar: {load[0]:.1f} (1 min), {os.cpu_count()} núcleos.\n")
    rows = []
    for f0, dx in ((0.5e6, 0.3e-3), (1e6, 0.2e-3), (1e6, 0.15e-3)):
        ph = leg_segment("tibia", length=40e-3, dx=dx, width=24e-3, depth=24e-3)
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = ElasticSolver3D(ph, f0, npml=10, n_mechanisms=1)
        t_init = time.time() - t0
        mem = s.memory_bytes()
        total = sum(mem.values())
        n = ph.grid.size
        k, j, i = ph.grid.index_of(-15e-3, 0.0, 1e-3)
        tt = s.time_axis(steps + 5)
        s.add_source(Source3D(np.array([k]), np.array([j]), np.array([i]), np.array([0]),
                              gaussian_pulse(tt, f0)[None, :]))
        for it in range(3):
            s.step(it)
        nt_30us = int(30e-6 / s.dt)
        for th in threads:
            if numba is not None:
                numba.set_num_threads(min(th, numba.config.NUMBA_NUM_THREADS))
            t0 = time.time()
            for it in range(steps):
                s.step(3 + it)
            sps = steps / (time.time() - t0)
            rows.append((f"{f0 / 1e6:.1f}", f"{dx * 1e3:.2f}", "×".join(str(v) for v in ph.grid.shape[::-1]),
                         f"{n / 1e6:.2f}", f"{total / 2**20:.0f}", f"{total / n:.0f}",
                         f"{mem['pml'] / n:.0f}", th, f"{sps:.2f}", f"{sps * n / 1e6:.1f}",
                         f"{nt_30us}", f"{nt_30us / sps / 60:.1f}", f"{t_init:.1f}"))
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"- {f0 / 1e6:.1f} MHz, dx = {dx * 1e3:.2f} mm: el pico de memoria del proceso sube "
              f"{(rss - rss0) / 1024:.0f} MiB (construcción incluida)")
        del s
    print()
    table(("f0 (MHz)", "dx (mm)", "nx×ny×nz", "Mceldas", "memoria (MiB)", "B/celda", "de ellos CPML", "hilos",
           "pasos/s", "Mceldas·paso/s", "pasos para 30 µs", "min para 30 µs", "init (s)"), rows)


# ----------------------------------------------------------------------------
# 8. Transmisión axial: tubo frente a placa
# ----------------------------------------------------------------------------
C_DIRECT = 1615.0  # onda directa más rápida (por la piel)


def fas_pick(t, p, d, f0, bw, frac=0.25):
    """Primer arribo (FAS): instante en que |p| alcanza ``frac`` de su máximo en una ventana previa a la directa.

    La ventana acaba 3 sigma antes de que llegue la onda directa por la piel
    (``C_DIRECT``), donde su envolvente es < 1 % de su máximo. Si el máximo de
    la ventana está a menos de medio periodo de su final, el FAS no está
    separado (o está truncado) y se devuelve NaN.
    """
    sigma = 2.355 / (2 * np.pi * bw * f0)
    t_end = 4 * sigma + d / C_DIRECT - 3 * sigma
    w = t < t_end
    a = np.abs(p[w])
    if a.size < 3:
        return np.nan, np.nan
    k_max = int(np.argmax(a))
    if t[w][k_max] > t_end - 0.5 / f0:
        return np.nan, np.nan
    thr = frac * a[k_max]
    k = int(np.argmax(a >= thr))
    return t[k - 1] + (thr - a[k - 1]) / (a[k] - a[k - 1]) * (t[k] - t[k - 1]), a[k_max]


def window_amplitude(t, p, t_on, f0):
    """Amplitud del primer arribo de una traza con fractura: máx |p| en ~1.5 ciclos desde el FAS sano."""
    w = (t >= t_on - 0.25 / f0) & (t <= t_on + 1.25 / f0)
    return np.abs(p[w]).max()


def axial_transmission(f0, dx, fracture, bw=1.0, length=44e-3, npml=10):
    """Trazas 3D (tubo) y 2D (corte y = 0) de una emisión de transmisión axial."""
    pml = npml * dx
    # Tibia de los presets con la piel y la grasa del tipo de cuerpo "delgado" (x0.9, x0.45) y sin músculo
    # encima (cara anteromedial, subcutánea): con capas más gruesas el FAS no se separa de la directa a < 2 cm
    spec = LegSegmentSpec(bone="tibia", muscle=0.0, skin=1.35e-3, fat=1.35e-3, standoff=pml + 0.6e-3)
    r_leg = spec.leg_radius
    ph = leg_segment(spec, fracture, length=length, dx=dx, width=2 * r_leg + 2 * pml,
                     depth=spec.standoff + 2 * r_leg + pml)
    g = ph.grid
    xs, zs = -17e-3, spec.standoff - 0.3e-3
    d = np.arange(8e-3, 35.01e-3, 1e-3)
    T = 26e-6 if f0 >= 1e6 else 34e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s3 = ElasticSolver3D(ph, f0, npml=npml, n_mechanisms=1)
    nt = int(T / s3.dt)
    t = s3.time_axis(nt)
    # emisor: tira de 3 mm a lo largo de y (perpendicular al eje del hueso), una celda en x
    k, j, i = g.cells_in_box((xs, xs), (-1.5e-3, 1.5e-3), (zs, zs))
    s3.add_source(Source3D(k, j, i, np.zeros(k.size, int), gaussian_pulse(t, f0, bw)[None, :]))
    k0, j0, _ = g.index_of(0.0, 0.0, zs)
    ir = np.array([g.index_of(xs + q, 0.0, zs)[2] for q in d])
    s3.add_receiver("p", Receiver3D(np.full(d.size, k0), np.full(d.size, j0), ir, np.arange(d.size), d.size))
    r3 = s3.run(nt)
    ph2 = ph.slice_y(0.0)
    s2 = ElasticSolver(ph2, f0, npml=npml, n_mechanisms=1, scattering=0.0)
    nt2 = int(T / s2.dt)
    t2 = s2.time_axis(nt2)
    s2.add_source(Source(np.array([k0]), np.array([g.index_of(xs, 0.0, zs)[2]]), np.array([0]),
                         gaussian_pulse(t2, f0, bw)[None, :]))
    s2.add_receiver("p", Receiver(np.full(d.size, k0), ir, np.arange(d.size), d.size))
    r2 = s2.run(nt2, track_peak=False)
    return d, {"3D": (r3.t, r3.traces["p"]), "2D": (r2.t, r2.traces["p"])}, r3.elapsed_s, g.size


def fracture_effect(t_fas, a_fas, ok, tf, pf, d, f0):
    """Pérdida (dB) y retardo (ns) del primer arribo con fractura respecto al sano, por receptor."""
    loss, delay = np.full(d.size, np.nan), np.full(d.size, np.nan)
    for q in np.nonzero(ok)[0]:
        loss[q] = 20 * np.log10(window_amplitude(tf, pf[q], t_fas[q], f0) / a_fas[q])
        w = (tf >= t_fas[q] - 0.5 / f0) & (tf <= t_fas[q] + 1.25 / f0)
        a = np.abs(pf[q][w])
        delay[q] = (tf[w][int(np.argmax(a >= 0.25 * a.max()))] - t_fas[q]) * 1e9
    return loss, delay


def part_aplicacion(freqs=((0.5e6, 0.3e-3),), bw=1.0, x_frac=8e-3, gap=1e-3, save=None):
    print("## Transmisión axial: tibia tubular (3D) frente a su corte longitudinal (2D)\n")
    print(f"Emisor en x = -17 mm, receptores a d = 8-35 mm, fractura transversal de {gap * 1e3:.0f} mm en "
          f"x = {x_frac * 1e3:.0f} mm (d = {(x_frac + 17e-3) * 1e3:.0f} mm), pulso gaussiano de ancho de banda "
          f"{bw:.0%}.\n")
    cases = {"sana": None, "completa": TransverseFracture3D(gap=gap, position=x_frac),
             "incompleta 120°": TransverseFracture3D(gap=gap, position=x_frac, arc_deg=120.0)}
    d_f = x_frac + 17e-3
    for f0, dx in freqs:
        traces = {}
        for name, fr in cases.items():
            d, tr, el, ncell = axial_transmission(f0, dx, fr, bw)
            traces[name] = tr
            print(f"- {f0 / 1e6:.1f} MHz, {name}: 3D de {ncell / 1e6:.2f} Mceldas en {el / 60:.1f} min", flush=True)
        if save:
            np.savez(Path(save) / f"elastic3d_aplicacion_{f0 / 1e6:.1f}MHz.npz", d=d, f0=f0, bw=bw,
                     **{f"{c}_{dim}_{k}": v for c, tr in traces.items() for dim, tv in tr.items()
                        for k, v in zip(("t", "p"), tv)})
        print()
        rows_v, rows_a, per_d = [], [], {}
        for dim in ("3D", "2D"):
            t, p = traces["sana"][dim]
            picks = np.array([fas_pick(t, p[q], d[q], f0, bw) for q in range(d.size)])
            t_fas, a_fas = picks[:, 0], picks[:, 1]
            ok = np.isfinite(t_fas)
            first = d[ok].min()
            for lo, hi in ((first, first + 7e-3), (first + 4e-3, first + 11e-3), (d[-1] - 7e-3, d[-1]),
                           (first, d[-1])):
                sel = ok & (d >= lo - 1e-9) & (d <= hi + 1e-9)
                if sel.sum() >= 4:
                    v = 1.0 / np.polyfit(d[sel], t_fas[sel], 1)[0]
                    rows_v.append((f"{f0 / 1e6:.1f}", dim, f"{lo * 1e3:.0f}-{hi * 1e3:.0f}", f"{v:.0f}"))
            n_exp = np.polyfit(np.log(d[ok]), np.log(a_fas[ok]), 1)[0]
            rows_v.append((f"{f0 / 1e6:.1f}", dim, "exponente n de A_FAS ∝ d^n (sano)", f"{n_exp:.2f}"))
            before, after = ok & (d < d_f), ok & (d > d_f + 1e-3)
            for name in ("completa", "incompleta 120°"):
                loss, delay = fracture_effect(t_fas, a_fas, ok, *traces[name][dim], d, f0)
                per_d[(dim, name)] = loss
                rows_a.append((f"{f0 / 1e6:.1f}", dim, name,
                               f"{np.nanmean(loss[before]):+.1f}" if before.any() else "—",
                               f"{np.nanmean(loss[after]):+.1f}", f"{np.nanmin(loss[after]):+.1f}",
                               f"{np.nanmean(delay[after]):+.0f}"))
        print(f"### {f0 / 1e6:.1f} MHz: velocidad del FAS (hueso sano)\n")
        table(("f0 (MHz)", "modelo", "tramo de d (mm)", "v_FAS (m/s)"), rows_v)
        print(f"### {f0 / 1e6:.1f} MHz: primer arribo con fractura respecto al hueso sano\n")
        table(("f0 (MHz)", "modelo", "fractura", "ΔA antes (dB)", "ΔA después, media (dB)", "ΔA después, peor (dB)",
               "retardo después (ns)"), rows_a)
        cols = np.nonzero(np.isfinite(per_d[("2D", "completa")]) | np.isfinite(per_d[("3D", "completa")]))[0]
        rows = [(f"{dim}, {name}", *(f"{per_d[(dim, name)][q]:+.1f}" for q in cols))
                for dim in ("3D", "2D") for name in ("completa", "incompleta 120°")]
        table(("ΔA (dB) a d (mm) =", *(f"{d[q] * 1e3:.0f}" for q in cols)), rows)


PARTS = {
    "velocidades": part_velocidades,
    "extrusion": part_extrusion,
    "green": part_green,
    "reflexion": part_reflexion,
    "oblicua": part_oblicua,
    "estabilidad": part_estabilidad,
    "rendimiento": part_rendimiento,
    "aplicacion": part_aplicacion,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parte", choices=list(PARTS), action="append")
    ap.add_argument("--pasos", type=int, default=30000, help="pasos de la parte de estabilidad")
    ap.add_argument("--hilos", type=int, nargs="+", default=[1], help="hilos Numba de la parte de rendimiento")
    ap.add_argument("--guardar", help="carpeta donde guardar las trazas de la parte de aplicación (.npz)")
    args = ap.parse_args()
    for name in args.parte or list(PARTS):
        t0 = time.time()
        if name == "estabilidad":
            part_estabilidad(args.pasos)
        elif name == "rendimiento":
            part_rendimiento(tuple(args.hilos))
        elif name == "aplicacion":
            part_aplicacion(save=args.guardar)
        else:
            PARTS[name]()
        print(f"({name}: {time.time() - t0:.0f} s)\n", flush=True)


if __name__ == "__main__":
    main()
