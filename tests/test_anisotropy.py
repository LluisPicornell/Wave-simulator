"""Hueso cortical transversalmente isótropo: rigideces, orientación por corte y velocidades medidas."""

import warnings

import numpy as np
import pytest

from wavesim.core.grid import Grid2D
from wavesim.core.signals import gaussian_pulse
from wavesim.materials.acoustic import ACOUSTIC, CORTICAL_STIFFNESS
from wavesim.phantoms import Fracture, cross_section_limb, longitudinal_limb
from wavesim.phantoms.phantom import TISSUE_INDEX, Phantom
from wavesim.ultrasound import ElasticMedium, ElasticSolver, Receiver, Source, elastic_medium
from wavesim.ultrasound import attenuation as att
from wavesim.ultrasound.medium import christoffel_speeds, isotropic_stiffness

F0 = 1e6
NPML = 10
RHO = ACOUSTIC["cortical_bone"].density
#: Rigideces específicas (C/rho) del corte longitudinal: x = eje del hueso
TI = np.array(CORTICAL_STIFFNESS.plane("longitudinal")) / RHO


def _solver(g, stiffness, n_mech=0):
    one = np.ones(g.shape)
    z = np.zeros((n_mech, *g.shape))
    med = ElasticMedium(RHO * one, np.sqrt(stiffness[0]) * one, np.sqrt(stiffness[3]) * one, z, z.copy(),
                        att.relaxation_times(F0, n_mech), F0, stiffness[:, None, None] * one)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ElasticSolver(Phantom(g, np.full(g.shape, TISSUE_INDEX["water"], np.int16)), F0, npml=NPML,
                             medium=med)


def _lag(a, b, dt):
    """Retardo de ``b`` respecto a ``a`` por correlación cruzada con ajuste parabólico."""
    n = a.size
    xc = np.fft.irfft(np.fft.rfft(b, 2 * n) * np.conj(np.fft.rfft(a, 2 * n)))
    k = int(np.argmax(xc[:n]))
    y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
    return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * dt


def _plane_wave_speed(axis: str, kind: str, dx=0.1e-3, d1=3e-3, d2=13e-3):
    """Velocidad de una onda plana a lo largo de ``axis`` ("x" o "z"): la fuente ocupa toda una fila o columna."""
    long_n = int(round((d2 + 6e-3) / dx)) + 2 * NPML + 6
    short_n = 2 * NPML + 4
    g = Grid2D(long_n, short_n, dx) if axis == "x" else Grid2D(short_n, long_n, dx)
    s = _solver(g, TI)
    nt = int(np.ceil(10e-6 / s.dt))
    t = s.time_axis(nt)
    k0 = NPML + 4
    k1, k2 = k0 + int(round(d1 / dx)), k0 + int(round(d2 / dx))
    across = np.arange(short_n)
    sig = gaussian_pulse(t, F0, 1.0)[None, :]
    q = {"pressure": "p", "force_x": "vx", "force_z": "vz"}[kind]
    if axis == "x":
        s.add_source(Source(across, np.full(short_n, k0), np.zeros(short_n, int), sig, kind))
        s.add_receiver("r", Receiver(np.array([short_n // 2] * 2), np.array([k1, k2]), np.array([0, 1]), 2, q))
    else:
        s.add_source(Source(np.full(short_n, k0), across, np.zeros(short_n, int), sig, kind))
        s.add_receiver("r", Receiver(np.array([k1, k2]), np.array([short_n // 2] * 2), np.array([0, 1]), 2, q))
    tr = s.run(nt, track_peak=False).traces["r"]
    return (k2 - k1) * dx / _lag(tr[0], tr[1], s.dt)


@pytest.mark.parametrize("axis, kind, idx", [("x", "pressure", 0), ("z", "pressure", 1), ("x", "force_z", 3),
                                             ("z", "force_x", 3)])
def test_velocidades_p_y_s_a_lo_largo_de_los_ejes(axis, kind, idx):
    """P a lo largo de x (eje del hueso) y z (radial), S en ambas: sqrt(C11/ρ), sqrt(C33/ρ), sqrt(C55/ρ)."""
    assert _plane_wave_speed(axis, kind) == pytest.approx(np.sqrt(TI[idx]), rel=0.01)


def test_qp_a_45_grados_frente_a_christoffel():
    """Onda plana con la normal a 45°: fuente de línea en la diagonal i + j = cte."""
    dx = 0.1e-3
    n = 300
    g = Grid2D(n, n, dx)
    s = _solver(g, TI)
    nt = int(np.ceil(9e-6 / s.dt))
    t = s.time_axis(nt)
    c = n // 2
    i = np.arange(n)
    j = 2 * c - 60 - i  # la línea queda 30·√2 celdas por detrás del centro
    keep = (j >= 0) & (j < n)
    s.add_source(Source(j[keep], i[keep], np.zeros(keep.sum(), int), gaussian_pulse(t, F0, 1.5)[None, :]))
    k = np.array([-10, 40])  # receptores a lo largo de la normal (1, 1)/√2
    s.add_receiver("r", Receiver(c + k, c + k, np.array([0, 1]), 2, "p"))
    tr = s.run(nt, track_peak=False).traces["r"]
    v_qp, v_qs = christoffel_speeds(TI, np.pi / 4)
    d = (k[1] - k[0]) * dx * np.sqrt(2)
    # ventana de la onda qP en cada receptor (la qSV llega bastante después)
    t0 = 4 * 0.6 / (2 * np.pi * 1.5 * F0 / 2.355)
    for m, kk in enumerate(k):
        arrival = t0 + (kk + 30) * dx * np.sqrt(2) / v_qp
        tr[m] *= np.abs(t - arrival) < 0.5 * ((kk + 30) * dx * np.sqrt(2) / v_qs - (kk + 30) * dx * np.sqrt(2) / v_qp
                                               + 2e-6)
    assert d / _lag(tr[0], tr[1], s.dt) == pytest.approx(float(v_qp), rel=0.02)


def test_camino_anisotropo_con_rigideces_isotropas_reproduce_el_isotropo():
    """El caso isótropo por el camino anisótropo da lo mismo que el camino rápido (a redondeo de float32)."""
    ph = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=6e-3, depth=10e-3, dx=1450 / 2e6 / 8)
    med = elastic_medium(ph, 2e6, n_mechanisms=2)
    out = {}
    for name, stiff in (("iso", None), ("aniso", isotropic_stiffness(med.cp, med.cs))):
        m = ElasticMedium(med.rho, med.cp, med.cs, med.tau_p, med.tau_s, med.tau_sigma, med.f0, stiff)
        s = ElasticSolver(ph, 2e6, npml=10, medium=m, backend="numpy")
        assert s.anisotropic == (stiff is not None)
        t = s.time_axis(200)
        cols = np.arange(5, ph.grid.nx - 5)
        s.add_source(Source(np.full(cols.size, 14), cols, np.zeros(cols.size, int), gaussian_pulse(t, 2e6)[None]))
        s.run(200, track_peak=False)
        out[name] = (s.dt, s.sxx.copy(), s.sxz.copy(), s.vz.copy())
    assert out["iso"][0] == pytest.approx(out["aniso"][0], rel=1e-12)
    for a, b in zip(out["iso"][1:], out["aniso"][1:]):
        assert np.abs(a - b).max() <= 1e-4 * np.abs(a).max()


def test_orientacion_segun_el_corte_y_porosidad():
    f0 = 3e6
    p = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=6e-3, depth=12e-3, dx=0.06e-3,
                          realism="realista")
    cort = p.mask("cortical_bone")
    iso = elastic_medium(p, f0)
    med = elastic_medium(p, f0, anisotropic_bone=True)
    assert iso.stiffness is None and med.stiffness is not None
    c11, c33, c13, c55 = med.stiffness
    # fuera de la cortical, isótropo e igual que siempre
    assert np.allclose(med.stiffness[:, ~cort], isotropic_stiffness(iso.cp, iso.cs)[:, ~cort])
    # cerca de la porosidad de referencia (la de la tabla), cerca de las rigideces publicadas
    ref = cort & (np.abs(p.porosity - 0.05) < 0.01)
    assert ref.any()
    for k in range(4):
        assert np.allclose(med.stiffness[k][ref] * med.rho[ref], CORTICAL_STIFFNESS.plane("longitudinal")[k],
                           rtol=0.06)
    # sin mapa de porosidad, exactamente las publicadas
    b = longitudinal_limb("tibia", None, width=6e-3, depth=12e-3, dx=0.06e-3)
    mb = elastic_medium(b, f0, anisotropic_bone=True)
    assert np.allclose(mb.stiffness[:, b.mask("cortical_bone")].T, TI)
    # estable en todas las celdas: tensor definido positivo, poros resueltos isótropos
    solid = c55 > 0
    assert np.all(c11 > 0) and np.all(c55 >= 0) and np.all((c11 * c33 > c13**2)[solid])
    assert np.allclose(c13[~solid], c11[~solid]) and np.allclose(c33[~solid], c11[~solid])
    pores = cort & (iso.cs == 0)  # poros resueltos (la banda de borde usa la porosidad media)
    assert pores.any() and np.allclose(c11[pores], c33[pores]) and np.all(c55[pores] == 0)
    assert np.all(c11[cort & (p.porosity < 0.2)] > c33[cort & (p.porosity < 0.2)])  # eje del hueso = x

    # corte transversal: plano de isotropía, el medio sigue siendo isótropo con las constantes transversales
    q = cross_section_limb("pierna", dx=0.2e-3)
    mq = elastic_medium(q, f0, anisotropic_bone=True)
    cq = q.mask("cortical_bone")
    assert mq.stiffness is None
    assert np.allclose(mq.cp[cq], np.sqrt(CORTICAL_STIFFNESS.c_t / RHO))
    assert np.allclose(mq.cs[cq], np.sqrt(CORTICAL_STIFFNESS.g_t / RHO))

    with pytest.raises(ValueError):
        elastic_medium(Phantom(p.grid, p.labels), f0, anisotropic_bone=True)  # sin meta["view"]
