# Comparación cruzada de la EIT con pyEIT

Script: `benchmarks/cross_eit.py` (~5 min con Numba y un hilo, en un entorno aparte con `pip install pyeit numba matplotlib`).
Datos de pyEIT guardados: `benchmarks/data/cross_eit/pyeit.npz` (5 kB). Test rápido sin pyEIT: `tests/test_cross_eit.py`
(4 tests, ~11 s). Versión: **pyEIT 1.2.4** (instaló con pip sin problemas, CPU, FEM con SciPy). EIDORS/Octave no se probó
porque pyEIT sí instaló.

**Qué hace pyEIT y qué no.** Su `Forward.solve` sólo tiene el electrodo simple (nodo de la malla con corriente
unitaria): el docstring dice que el CEM "is under development". Por tanto **no hay comparación del CEM con pyEIT**;
el CEM de wavesim se compara sólo con su serie de Fourier analítica. La comparación con pyEIT es contra electrodos
puntuales.

**Convenciones** (comprobadas contra la solución analítica; ver cabecera del script):
disco unidad en pyEIT y R = 20 mm en wavesim con sigma real 0.5 S/m e I = 1 mA/m (en 2D la tensión escala como
`I/sigma` y no depende del radio); electrodo k de pyEIT a 180° - 22.5° k, que en wavesim es `start_deg = -90` con
`(x, z) = (x_p, -y_p)`; **signo opuesto** (pyEIT mide `V[q+1] - V[q]`, wavesim `V[q] - V[q+1]`); mismo orden de los
208 datos (13 por trama); las inclusiones (r = 4 mm, sigma x2) están en los mismos puntos físicos.

## 1. Problema directo (16 electrodos, protocolo adyacente, 208 datos)

Frente a la solución analítica de electrodos puntuales (`disc_point_voltage`):

| código | malla | error máx. por dato (%) | error de la norma (%) |
|---|---|---|---|
| pyEIT | h0 = 0.1 (376 nodos) | 1.27 | 0.70 |
| pyEIT | h0 = 0.05 (1476) | 0.20 | 0.12 |
| pyEIT | h0 = 0.025 (5845) | 0.06 | 0.03 |
| pyEIT | h0 = 0.0125 (23276) | 0.01 | 0.00 |
| wavesim ideal | dx = 1 mm, electrodo 2 mm | 7.29 | 3.29 |
| wavesim ideal | dx = 0.5 mm, 2 mm | 4.34 | 2.19 |
| wavesim ideal | dx = 0.5 mm, 1 mm | 10.85 | 5.01 |
| wavesim ideal | dx = 0.25 mm, 2 mm | 1.33 | 1.02 |
| wavesim ideal | dx = 0.125 mm, 2 mm | 1.38 | 1.05 |

pyEIT converge a la solución analítica (referencia fiable). wavesim se estanca en ~1.0-1.4 % al refinar (0.25 y 0.125
mm dan lo mismo): ese resto no es de discretización sino de modelo, un electrodo de 2 mm (corriente repartida) frente a
uno puntual; con 0.5 mm la escalera del borde añade 3-4 % y el error depende del ancho en celdas. La tabla de
`eit.md` (1.33 % a 0.5 mm) era el caso de una sola trama; con las 16 tramas el máximo es 4.3 %.

CEM de wavesim frente a la serie del CEM (todas las tramas, electrodos de 2 mm):

| z_c (ohm·m²) | dx (mm) | error máx. (%) | error de norma (%) |
|---|---|---|---|
| 0.003 | 0.5 | 4.02 | 2.02 |
| 0.003 | 0.25 | 0.79 | 0.36 |
| 0.03 | 0.5 | 3.94 | 2.02 |
| 0.03 | 0.25 | 0.72 | 0.30 |

Disco con inclusión (wavesim ideal, 2 mm, frente a pyEIT h0 = 0.0125):

| dx (mm) | posición | error máx. de V (%) | error de norma de V (%) | error de norma de dV (%) |
|---|---|---|---|---|
| 0.5 | centro / intermedia / periferia | 4.34 / 4.58 / 4.72 | 2.18 / 2.20 / 2.24 | 2.26 / 2.68 / 2.42 |
| 0.25 | centro / intermedia / periferia | 1.33 / 1.37 / 1.85 | 1.01 / 1.02 / 1.04 | 0.58 / 0.53 / 0.45 |

El dato que se usa para reconstruir (diferencia dV) coincide mejor (0.5 % a 0.25 mm) porque el error de modelo del
electrodo se cancela casi por completo.

## 2. Reconstrucción de diferencias

Inclusión de r = 4 mm, sigma x2, en tres posiciones; pyEIT en su malla de reconstrucción h0 = 0.05 (JAC "kotre" p = 0.5 y
GREIT "dist"); wavesim con su modelo de malla 1 mm y bloques de 3x3 mm, electrodos ideales (un paso NOSER 1e-2 y
Gauss-Newton NOSER 3e-2). Imágenes de pyEIT interpoladas a la malla de wavesim y evaluadas con `greit_metrics`. "Error del
máximo" = distancia del píxel máximo al centro verdadero. Con ruido (SNR 70 dB, 3 realizaciones) los números cambian en el
segundo decimal, por lo que se muestran sin ruido. Datos de wavesim (electrodos ideales de 3 mm, malla 0.5 mm):

| posición | método | error del máximo (mm) | error de posición GREIT (mm) | AR | RES | SD | ringing |
|---|---|---|---|---|---|---|---|
| intermedia | wavesim un paso | 3.0 | 0.54 | 0.63 | 0.208 | 0.19 | 0.30 |
| intermedia | wavesim GN | 3.0 | 0.92 | 0.89 | 0.208 | 0.19 | 0.23 |
| intermedia | pyEIT JAC lamb 1e-4 | 0.0 | 0.22 | 0.67 | 0.257 | 0.06 | 0.24 |
| intermedia | pyEIT JAC lamb 1e-3 | 0.0 | 0.34 | 0.66 | 0.286 | 0.05 | 0.20 |
| intermedia | pyEIT GREIT lamb 1e-3 | 1.0 | 0.17 | 0.60 | 0.412 | 0.01 | 0.08 |
| periferia | wavesim un paso | 3.2 | 0.54 | 0.67 | 0.244 | 0.11 | 0.38 |
| periferia | wavesim GN | 3.2 | 0.65 | 1.02 | 0.240 | 0.14 | 0.24 |
| periferia | pyEIT JAC lamb 1e-3 | 1.0 | 0.20 | 0.68 | 0.272 | 0.04 | 0.21 |
| periferia | pyEIT GREIT lamb 1e-3 | 1.0 | 0.45 | 0.67 | 0.353 | 0.10 | 0.05 |
| centro | wavesim un paso / GN | 1.4 / 1.4 | 0.00 / 0.00 | 0.64 / 0.81 | 0.255 | 0.05 | 0.17 / 0.13 |
| centro | pyEIT JAC lamb 1e-3 | 0.0 | 0.07 | 0.65 | 0.345 | 0.04 | 0.16 |
| centro | pyEIT GREIT lamb 1e-3 | 0.0 | 0.00 | 0.55 | 0.495 | 0.01 | 0.07 |

(La tabla completa, con los tres lamb de JAC, dos de GREIT y los datos generados por pyEIT, la imprime el script; con
datos de pyEIT los números son prácticamente los mismos: error del máximo idéntico, GREIT-posición ±0.3 mm.)

**Lectura honesta.**
- Posición: todos localizan la inclusión a < 1 mm de error de centroide GREIT (wavesim 0.0-0.9, pyEIT 0.0-1.0). El píxel
  máximo de wavesim cae a 1.4-3.2 mm porque sus bloques son de 3 mm (el máximo es el píxel central del bloque, un
  artefacto de la parametrización, no de la física); pyEIT, con elementos de ~1 mm, da 0-1 mm.
- Resolución: la RES de wavesim (0.21-0.26) es mejor (menor) que la de JAC (0.26-0.41) y que la de GREIT (0.35-0.59), con
  más deformación (SD 0.05-0.21 frente a 0.01-0.10) y más *ringing* (0.13-0.38 frente a 0.05-0.24): la clásica relación
  de compromiso, no una ventaja neta. La amplitud AR de wavesim GN (0.8-1.0) es la más cercana a 1 porque es no lineal;
  JAC y GREIT dan 0.5-0.7, el nivel esperado de un método lineal regularizado.
- Los lamb de pyEIT y las regularizaciones de wavesim no son equivalentes: la comparación es de familias de métodos con
  un parámetro razonable, no de óptimos.
- Hallazgo en wavesim, no en pyEIT: con un disco cuyo radio coincide exactamente con un punto de la malla (`<= R`) queda
  una celda aislada, sin caras, en cada extremo del eje; su columna del jacobiano es ~1e-20 y NOSER (regularización
  proporcional a la diagonal de J'J) resuelve un sistema singular y el resultado explota (1e12). Se evitó en el
  benchmark con `< R - 1e-6`; no se ha cambiado `eit.py`, y conviene que `reconstruct_difference` proteja las columnas sin
  sensibilidad.
- Limitaciones: electrodos ideales (no CEM) por no haber CEM en pyEIT; 3 posiciones de una sola inclusión; ruido sólo
  70 dB; sin EIDORS.
