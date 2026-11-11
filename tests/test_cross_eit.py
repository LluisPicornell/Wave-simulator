"""Comparación cruzada de la EIT con pyEIT (FEM) sin tener pyEIT instalado.

Los datos directos de pyEIT (malla h0 = 0.0125, 23276 nodos) están en ``benchmarks/data/cross_eit/pyeit.npz``
(``benchmarks/cross_eit.py --save``; método y tablas en ``docs/validacion/cruzada_eit.md``).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cross_eit", ROOT / "benchmarks" / "cross_eit.py")
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)


@pytest.fixture(scope="module")
def ref():
    return np.load(ROOT / "benchmarks" / "data" / "cross_eit" / "pyeit.npz")


def test_pyeit_guardado_coincide_con_la_solucion_analitica(ref):
    """La referencia es fiable: pyEIT converge a los electrodos puntuales del disco (medido 0.01 %)."""
    assert ce.errors(ref["v0"], ce.analytic_point())[0] < 0.05


def test_wavesim_ideal_frente_a_pyeit_en_el_disco_homogeneo(ref):
    """Mismas unidades, signo y orden de los 208 datos; la diferencia es la del electrodo de 2 mm frente al puntual."""
    v = ce.wavesim_data(0.25e-3, 2e-3)
    emax, enorm = ce.errors(v, ref["v0"])
    assert emax < 2.0 and enorm < 1.5  # medido 1.33 % y 1.02 %
    # La convención de signo importa: con el signo contrario el error es del 200 %
    assert ce.errors(-v, ref["v0"])[1] > 150


def test_diferencia_con_inclusion_frente_a_pyeit(ref):
    """El dato que se usa para imagen (V1 - V0) coincide con pyEIT al nivel de la discretización."""
    v0 = ce.wavesim_data(0.5e-3, 2e-3)
    v1 = ce.wavesim_data(0.5e-3, 2e-3, inc=ce.POSITIONS["intermedia"])
    d_ref = ref["v1"][1] - ref["v0"]
    err = np.linalg.norm((v1 - v0) - d_ref) / np.linalg.norm(d_ref)
    assert err < 0.06


def test_reconstruccion_de_wavesim_con_los_datos_de_pyeit(ref):
    """Gauss-Newton de wavesim alimentado con datos de pyEIT (electrodo puntual) localiza la inclusión."""
    from wavesim.em import Sensitivity, ring_electrodes, reconstruct_difference
    from wavesim.em.eit import EITModel

    rec, y0 = ce.disc(1e-3)
    model = EITModel(rec, ce.F, ring_electrodes(rec, ce.N_EL, 3e-3, start_deg=-90.0), factor=3, current=ce.CURRENT,
                     y0=y0)
    _, jac = model.forward(np.zeros(model.n_params), jacobian=True)
    sens = Sensitivity(jac, model._coarse_shape, model.factor, model.active)
    x, z = rec.grid.meshgrid()
    for k, (name, (cx, cz)) in enumerate(ce.POSITIONS.items()):
        img = np.where(model.tissue, model.to_fine(reconstruct_difference(sens, ref["v1"][k] - ref["v0"], 1e-2)), 0.0)
        i = np.argmax(img)
        assert np.hypot(x.ravel()[i] - cx, z.ravel()[i] - cz) < 4.5e-3, name
