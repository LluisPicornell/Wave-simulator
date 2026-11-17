"""Tiempo y memoria de la rasterización de los fantomas 3D (``phantoms.limb3d_full.limb_3d``).

Uso (desde la raíz del repositorio)::

    PYTHONPATH=. python benchmarks/phantom3d_performance.py            # 1, 0.5 y 0.25 mm
    PYTHONPATH=. python benchmarks/phantom3d_performance.py 0.5        # sólo una dx (mm)
    PYTHONPATH=. python benchmarks/phantom3d_performance.py coherencia # cortes frente al 2D y dx

Cada caso corre en un proceso aparte para medir su pico de memoria residente
(``ru_maxrss``, que incluye el intérprete y NumPy, unos 60-80 MB). Pierna de
60 × 100 × 100 mm (x × y × z): el preset "pierna" con su margen por defecto.
Casos: pierna sana, con fractura transversal desplazada, con contusión ósea
en la metáfisis (miles de bolsas de edema y microfracturas) y, a 1 y 0.5 mm,
las fracciones de volumen con supermuestreo 2 (``limb_3d_fractions``).
"""

from __future__ import annotations

import json
import subprocess
import sys

CASES = {
    "sana": "limb_3d('pierna', dx=dx)",
    "desplazada": "limb_3d('pierna', Fracture3D(kind='desplazada', displacement=2e-3, angle_deg=15), dx=dx)",
    "contusion_metafisis": ("limb_3d('pierna_metafisis', "
                            "random_fracture_3d(np.random.default_rng(1), 'contusion_osea'), dx=dx)"),
    "fracciones_x2": "limb_3d_fractions('pierna', dx=dx, supersample=2)[0]",
}

CHILD = """
import json, resource, time
import numpy as np
from wavesim.phantoms import Fracture3D, limb_3d, limb_3d_fractions, random_fracture_3d
dx = {dx}
t = time.perf_counter()
ph = {expr}
el = time.perf_counter() - t
print(json.dumps({{"time": el, "cells": ph.grid.size, "shape": ph.grid.shape,
                  "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}}))
"""


def run(dx_mm: float, case: str) -> dict:
    code = CHILD.format(dx=dx_mm * 1e-3, expr=CASES[case])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def coherence(dx: float = 0.5e-3) -> None:
    """Cortes y = 0 y x = 0 de ``limb_3d`` sin rugosidad frente a ``longitudinal_limb`` y ``cross_section_limb``."""
    from dataclasses import replace

    import numpy as np

    from wavesim.phantoms import (CROSS_PRESETS, Fracture3D, cross_section_limb, equivalent_longitudinal_spec,
                                  limb_3d, longitudinal_limb)
    from wavesim.phantoms.phantom import TISSUE_INDEX

    def runs(col):
        out = []
        for i, lab in enumerate(col):
            if not out or out[-1][0] != lab:
                out.append((lab, i))
        return out

    zero = (0.0, 1e-3)
    print(f"dx = {dx * 1e3} mm")
    print(f"{'preset':<10} {'cuerpo':<22} {'capas y=0':>9} {'máx. desv. (celdas)':>19} {'acuerdo x=0':>11} "
          f"{'máx. dif. área (sin vasos)':>26}")
    for preset in ("pierna", "antebrazo", "muslo"):
        for body in ("normal", "delgado", "obeso", "musculoso", "anciano_osteoporotico"):
            ph = limb_3d(preset, length=2e-3, dx=dx, roughness=False, body=body)
            l2 = longitudinal_limb(equivalent_longitudinal_spec(preset, body), width=2e-3, dx=dx,
                                   depth=ph.grid.nz * dx)
            r3, r2 = runs(ph.slice_y(0).labels[:, ph.grid.nx // 2]), runs(l2.labels[:, l2.grid.nx // 2])
            same = [a[0] for a in r3] == [b[0] for a, b in zip(r3, r2)] and len(r3) == len(r2)
            dev = max(abs(a[1] - b[1]) for a, b in zip(r3, r2))
            cs = replace(CROSS_PRESETS[preset], skin_rough=zero, soft_rough=zero, periosteal_rough=zero,
                         endosteal_rough=zero)
            c2 = cross_section_limb(cs, dx=dx, body=body)
            sx = ph.slice_x(0.0)
            yl, zl = ph.meta["leg_center"]
            Y, Z = np.meshgrid(sx.grid.x - yl, sx.grid.z - zl)
            ii = np.round((Y - c2.grid.x0) / dx).astype(int)
            jj = np.round((Z - c2.grid.z0) / dx).astype(int)
            leg = np.hypot(Y, Z) <= ph.meta["leg_radius"]
            agree = (c2.labels[jj[leg], ii[leg]] == sx.labels[leg]).mean()
            darea = max(abs((sx.labels == TISSUE_INDEX[t]).sum() / (c2.labels == TISSUE_INDEX[t]).sum() - 1)
                        for t in c2.present_tissues() if t not in ("water", "blood"))  # vasos: ~40 celdas
            print(f"{preset:<10} {body:<22} {('igual' if same else 'DISTINTA'):>9} {dev:19d} {100 * agree:10.2f}% "
                  f"{100 * darea:25.1f}%")
    fr = Fracture3D(kind="desplazada", position=1e-3, angle_deg=20.0, displacement=1.5e-3, roughness=0.3e-3)
    a = limb_3d("pierna", fr, length=8e-3, dx=1e-3, seed=7)
    b = limb_3d("pierna", fr, length=8e-3, dx=0.5e-3, seed=7)
    ga, gb = a.grid, b.grid
    sel = [np.round((c - c0) / gb.dx).astype(int) for c, c0 in ((ga.z, gb.z0), (ga.y, gb.y0), (ga.x, gb.x0))]
    ok = [(s_ >= 0) & (s_ < n) for s_, n in zip(sel, gb.shape)]
    sub = b.labels[np.ix_(*(s_[o] for s_, o in zip(sel, ok)))]
    print(f"misma semilla a 1 y 0.5 mm (rugosa, fractura desplazada): {100 * (sub == a.labels[np.ix_(*ok)]).mean():.3f}"
          " % de celdas comunes con la misma etiqueta")


def main() -> None:
    if sys.argv[1:] == ["coherencia"]:
        coherence()
        return
    dxs = [float(a) for a in sys.argv[1:]] or [1.0, 0.5, 0.25]
    print(f"{'dx (mm)':>8} {'caso':<20} {'forma (z, y, x)':<18} {'Mceldas':>8} {'tiempo (s)':>10} "
          f"{'ns/celda':>9} {'pico RSS (MB)':>13}")
    for dx in dxs:
        for case in CASES:
            if case == "fracciones_x2" and dx < 0.5:
                continue  # 8 × 38 M celdas: fuera del presupuesto de memoria de este banco
            r = run(dx, case)
            print(f"{dx:8.2f} {case:<20} {str(tuple(r['shape'])):<18} {r['cells'] / 1e6:8.2f} {r['time']:10.2f} "
                  f"{1e9 * r['time'] / r['cells']:9.0f} {r['rss_mb']:13.0f}")


if __name__ == "__main__":
    main()
