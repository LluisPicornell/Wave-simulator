"""Mapas de propiedades elásticas a partir de un fantoma."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import porous
from ..materials.acoustic import ACOUSTIC
from ..phantoms.phantom import TISSUE_INDEX, Phantom
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
    # colágeno denso en láminas: lo más ecogénico de las partes blandas (estimación)
    "fascia": 0.04,
    "periosteum": 0.03,
}

#: Tamaño de celda de referencia (m) de ``SCATTER_STRENGTH``: la ``dx`` de los
#: ejemplos a 3 MHz (λ/8 en agua), de modo que sus resultados no cambian.
DX_REF = 60e-6

#: Puntos por longitud de onda por debajo de los cuales la onda S de una celda
#: porosa se suprime (el mismo umbral con el que avisa ``ElasticSolver``). Una
#: celda con poco sólido tiene una velocidad de cizalla tan baja que la malla
#: no la resuelve; esa onda, además, se atenúa en una fracción de milímetro.
MIN_SHEAR_PPW = 6.0

#: Muestras de porosidad en las que se tabulan las propiedades efectivas.
_PHI_SAMPLES = np.linspace(0.0, 1.0, 129)

#: Banda de borde (celdas) en la que la microestructura ósea se sustituye por
#: su medio efectivo uniforme. Poros o trabéculas de una o dos celdas dentro de
#: la capa absorbente CPML (20 celdas por defecto) la vuelven inestable: un
#: poro en la esquina de la CPML hacía crecer la presión ~100 veces. La banda
#: queda fuera de la apertura de la sonda y por debajo de la imagen, así que
#: no cambia lo que se ve; con una CPML más gruesa hay que ampliarla.
MICROSTRUCTURE_EDGE = 24


@dataclass
class ElasticMedium:
    """Propiedades por celda: densidad, velocidades P/S y pesos de los mecanismos del GSLS.

    ``tau_p`` y ``tau_s`` son los pesos ``y_l`` de los mecanismos de
    relajación (``attenuation``) para las ondas P y S: arrays ``(L, nz, nx)``,
    uno por mecanismo, o ``(nz, nx)`` con el mismo peso para todos (método tau).
    """

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

    def weights(self, wave: str) -> np.ndarray:
        """Pesos ``(L, nz, nx)`` de los mecanismos para la onda ``"p"`` o ``"s"``."""
        w = np.asarray(self.tau_p if wave == "p" else self.tau_s)
        if w.ndim == 2:
            return np.broadcast_to(w, (self.tau_sigma.size, *w.shape))
        return w


def elastic_medium(phantom: Phantom, f0: float, n_mechanisms: int = 1, scattering: float = 1.0,
                   attenuation: bool = True) -> ElasticMedium:
    """Construye el medio para el solver elástico a la frecuencia central ``f0``.

    ``scattering`` escala ``SCATTER_STRENGTH`` (0 = tejidos perfectamente
    homogéneos, sin speckle). La amplitud se normaliza por el tamaño de celda
    (ver ``DX_REF``): el nivel del speckle no depende de la resolución de la malla.

    Si el fantoma tiene mapa de porosidad, las celdas de hueso toman las
    propiedades del medio efectivo (``materials.porous``): la dispersión por
    los poros la produce el propio solver a partir de ese mapa.
    """
    n_mech = n_mechanisms if attenuation else 0
    tau_sigma = att.relaxation_times(f0, n_mech)
    band = att.fit_band(f0, n_mech)
    rho = phantom.property_map(lambda k: ACOUSTIC[k].density)
    cp = phantom.property_map(lambda k: ACOUSTIC[k].c_p)
    cs = phantom.property_map(lambda k: ACOUSTIC[k].c_s)

    # Pesos de cada tejido presente, ajustados todos a la vez: filas (tejido, onda)
    keys = phantom.present_tissues()
    rows = [(k, shear) for k in keys for shear in (False, True)]
    alpha = np.array([[ACOUSTIC[k].alpha_np_per_m(f, shear=shear) for f in band] for k, shear in rows])
    speed = np.array([ACOUSTIC[k].c_s if shear else ACOUSTIC[k].c_p for k, shear in rows])
    fitted = att.mechanism_weights(alpha, speed, band, tau_sigma, f0).reshape(len(keys), 2, n_mech)
    table = {k: fitted[n] for n, k in enumerate(keys)}
    tau_p = np.zeros((n_mech, *rho.shape))
    tau_s = np.zeros((n_mech, *rho.shape))
    for m in range(n_mech):
        tau_p[m] = phantom.property_map({k: w[0, m] for k, w in table.items()}.__getitem__)
        tau_s[m] = phantom.property_map({k: w[1, m] for k, w in table.items()}.__getitem__)
    if phantom.porosity is not None:
        _porous_cells(phantom, f0, band, tau_sigma, rho, cp, cs, tau_p, tau_s)

    if scattering > 0 and phantom.heterogeneity is not None:
        # La heterogeneidad es ruido blanco *por celda*: la varianza de densidad por unidad
        # de área escala como s²·dx², y en 2D la intensidad retrodispersada va como s²·dx²
        # (una celda más pequeña aporta menos dispersor por celda pero hay más celdas por
        # longitud de onda). Sin corrección, afinar la malla bajaba el speckle ~6 dB por
        # cada mitad de dx. Escalar s con DX_REF/dx deja la intensidad constante.
        strength = phantom.property_map(lambda k: SCATTER_STRENGTH.get(k, 0.0), physics="acoustic") * scattering
        strength = strength * (DX_REF / phantom.grid.dx)
        rho = rho * (1.0 + strength * np.clip(phantom.heterogeneity, -3, 3))
    return ElasticMedium(rho, cp, cs, tau_p, tau_s, tau_sigma, f0)


def _porous_cells(phantom: Phantom, f0: float, band: np.ndarray, tau_sigma: np.ndarray, rho, cp, cs, tau_p,
                  tau_s) -> None:
    """Sustituye en las celdas porosas las propiedades de la tabla por las del medio efectivo.

    Las propiedades (``materials.porous.acoustic_effective``) y los pesos de
    sus mecanismos GSLS se tabulan en función de la porosidad y se interpolan en
    cada celda. La onda S se suprime donde su velocidad cae por debajo de
    ``MIN_SHEAR_PPW`` puntos por longitud de onda. En la banda de borde
    (``MICROSTRUCTURE_EDGE``) la porosidad es la media del tejido.
    """
    cs_min = MIN_SHEAR_PPW * f0 * phantom.grid.dx
    edge = np.ones(phantom.grid.shape, dtype=bool)
    e = MICROSTRUCTURE_EDGE
    edge[e:-e, e:-e] = False
    for key in porous.POROUS_TISSUES:
        cells = (phantom.labels == TISSUE_INDEX[key]) & np.isfinite(phantom.porosity)
        if not cells.any():
            continue
        eff = porous.acoustic_effective(key, _PHI_SAMPLES, band, phantom.pore_fill)
        phi_map = np.where(cells & edge, float(phantom.porosity[cells].mean()), phantom.porosity)
        phi = phi_map[cells]
        shear = np.interp(phi, _PHI_SAMPLES, eff.c_s)
        solid = shear >= cs_min
        rho[cells] = np.interp(phi, _PHI_SAMPLES, eff.rho)
        cp[cells] = np.interp(phi, _PHI_SAMPLES, eff.c_p)
        cs[cells] = np.where(solid, shear, 0.0)
        w_p = att.mechanism_weights(eff.alpha_p, eff.c_p, band, tau_sigma, f0)
        w_s = att.mechanism_weights(eff.alpha_s, eff.c_s, band, tau_sigma, f0)
        for m in range(tau_sigma.size):
            tau_p[m, cells] = np.interp(phi, _PHI_SAMPLES, w_p[:, m])
            tau_s[m, cells] = np.where(solid, np.interp(phi, _PHI_SAMPLES, w_s[:, m]), 0.0)
