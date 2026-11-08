"""Mapas de propiedades elásticas a partir de un fantoma."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials.acoustic import ACOUSTIC
from ..phantoms.phantom import Phantom
from . import attenuation as att

#: Intensidad relativa de los dispersores sub-longitud de onda (fluctuación
#: rms de densidad). Es lo que genera el moteado (*speckle*) de la ecografía:
#: el músculo y la grasa son "ecogénicos"; la sangre, el agua y el hematoma
#: fresco casi anecoicos. Valores de orden de magnitud, ajustables.
SCATTER_STRENGTH: dict[str, float] = {
    "water": 0.0,
    "air": 0.0,
    "skin": 0.03,
    "fat": 0.02,
    "muscle": 0.025,
    "blood": 0.002,
    "cortical_bone": 0.02,
    "trabecular_bone": 0.05,
    "marrow": 0.015,
    "hematoma": 0.005,
    "edema": 0.012,
}


@dataclass
class ElasticMedium:
    """Propiedades por celda: densidad, velocidades P/S y parámetros tau del GSLS."""

    rho: np.ndarray
    cp: np.ndarray
    cs: np.ndarray
    tau_p: np.ndarray
    tau_s: np.ndarray
    tau_sigma: np.ndarray
    f0: float

    @property
    def is_solid(self) -> np.ndarray:
        return self.cs > 0


def elastic_medium(phantom: Phantom, f0: float, n_mechanisms: int = 1, scattering: float = 1.0,
                   attenuation: bool = True) -> ElasticMedium:
    """Construye el medio para el solver elástico a la frecuencia central ``f0``.

    ``scattering`` escala ``SCATTER_STRENGTH`` (0 = tejidos perfectamente
    homogéneos, sin speckle).
    """
    tau_sigma = att.relaxation_times(f0, n_mechanisms if attenuation else 0)
    band = att.fit_band(f0)
    rho = phantom.property_map(lambda k: ACOUSTIC[k].density)
    cp = phantom.property_map(lambda k: ACOUSTIC[k].c_p)
    cs = phantom.property_map(lambda k: ACOUSTIC[k].c_s)

    def tau_for(k: str, shear: bool) -> float:
        p = ACOUSTIC[k]
        c = p.c_s if shear else p.c_p
        if c <= 0 or tau_sigma.size == 0:
            return 0.0
        alpha = np.array([p.alpha_np_per_m(f, shear=shear) for f in band])
        return att.fit_tau(att.target_qinv(alpha, c, band), tau_sigma, band)

    tau_p = phantom.property_map(lambda k: tau_for(k, False))
    tau_s = phantom.property_map(lambda k: tau_for(k, True))

    if scattering > 0 and phantom.heterogeneity is not None:
        strength = phantom.property_map(lambda k: SCATTER_STRENGTH.get(k, 0.0)) * scattering
        rho = rho * (1.0 + strength * np.clip(phantom.heterogeneity, -3, 3))
    return ElasticMedium(rho, cp, cs, tau_p, tau_s, tau_sigma, f0)
