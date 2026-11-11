# Comparación cruzada del solver elástico 2D con k-Wave y Devito

Scripts: `benchmarks/cross_ultrasonido.py` (orquesta, solver Devito, guarda los
datos) y `benchmarks/kwave_comparison.py` (casos A, B y C). Datos de
referencia: `benchmarks/data/cross_ultrasonido/` (384 kB). Test rápido sin
ninguna de las herramientas: `tests/test_cross_ultrasonido.py` (8 tests, ~50 s).

wavesim es la versión actual: 2 mecanismos de atenuación por defecto,
corrección de deslizamiento en la interfaz fluido-sólido (`_slip_terms`) y
M-PML automática (no se activa en ninguno de estos montajes: ninguna capa
sólida cruza la CPML).

## 1. Herramientas y entorno

| Herramienta | Versión | Instalación | Uso |
|---|---|---|---|
| k-wave-python | 0.6.2 (la última en PyPI el 11-11-2026) | `venv --system-site-packages` + `pip install k-wave-python scipy` (NumPy 2.2.6 en ese entorno) | `kspaceFirstOrder(..., backend="python")`, CPU, float64 |
| Devito | 4.8.23 | `venv --system-site-packages` + `pip install devito` (NumPy 2.4.3), gcc del sistema, sin OpenMP | operador velocidad-tensión propio (abajo) |

- **k-wave-python 0.6.2 no trae solver elástico.** No hay módulo
  `pstd_elastic_2d` ni equivalente, y `kWaveSimulation.py` (línea 544) aborta
  con `assert not calling_func_name.startswith(("pstdElastic",
  "kspaceElastic")), "Elastic simulation is not supported."`. Por eso el caso
  D (interfaz agua-cortical y placa de Lamb) se hizo con Devito y k-Wave queda
  limitado a fluidos (A, B, C). No hizo falta SPECFEM2D.
- Se usó el backend `python` de k-Wave: el binario C++ OMP del wheel ya se
  descartó en la comparación anterior (necesita `libhdf5_serial_hl.so.100` y
  era más lento). Los tiempos de k-Wave de este documento son de ese backend
  NumPy y no miden el rendimiento de k-Wave en C++/CUDA.
- Todo en una CPU de 4 núcleos compartida con otros 3 procesos (Numba con 1
  hilo, Devito sin OpenMP): los tiempos son orientativos.

### Solver Devito (`DevitoElastic`)

Esquema de Virieux en malla escalonada, escrito con las primitivas de Devito
(`VectorTimeFunction`, `TensorTimeFunction`, `div`, `grad`) como en sus
ejemplos `examples/seismic/elastic`, con orden espacial 4 (el mismo estencil
9/8, −1/24 que wavesim; se comprobó en las ecuaciones evaluadas) y 8. Para que
la comparación sea de esquema y no de montaje, recibe lo mismo que wavesim:

- los mismos mapas `rho`, `cp`, `cs` y el mismo `dt` (CFL 0.9 del límite de
  wavesim; con orden 8 sigue siendo estable, CFL efectiva 0.97);
- la misma discretización del medio: densidad media aritmética en los nodos
  de velocidad y `mu` media armónica de 4 celdas en `sxz` (nula si alguna es
  fluida), en `Function` escalonadas propias (Devito, por defecto,
  interpolaría `mu` aritméticamente, lo que dejaría cizalla en la interfaz);
- las mismas fuentes (de presión, en el mismo instante; las de fuerza, medio
  paso más tarde, indiferente para medir velocidades de fase) y las mismas
  filas de medida y post-proceso (`elastic_validation.beam_reflection` y
  `lamb_case`, que ahora aceptan `factory=`).

Diferencias que quedan, a propósito: **no tiene la corrección de
deslizamiento** de la interfaz fluido-sólido y usa una **esponja**
(atenuación exponencial de todos los campos, perfil cuadrático) en lugar de la
CPML. La esponja tiene 3 longitudes de onda (de la onda más rápida del borde a
f0): con una franja fija de 60 celdas, a 20-30 ppw su eco cambiaba |R| a 45°
hasta un 14 % y S0 hasta un 1 %; con 3 λ el orden 4 coincide con wavesim sin
corrección a ≤ 0.01 % (abajo). Para aislar la corrección se añadió un tercer
código, **wavesim sin corrección** (`_slip = {}`).

## 2. Resultados

Tolerancias orientativas del encargo: R/T ±1 %, tiempos ±0.5 %, forma de onda
tras alinear ≤ 3 %, velocidades ±1 %.

### A. Dos capas fluidas (1500/1000 | 1800/1100), onda plana normal

Analítico: R = 0.1379, T = 1.1379. R y T por cociente de envolventes,
tiempos por máximo de envolvente; forma de onda comparada en frecuencia
(0-4 MHz, sin interpolación) con el desfase continuo que mejor alinea.

| dx (mm) | ppw | R ws / kW (err vs analítico) | T ws / kW | ws/kW − 1 en R, T | Δ(t_ref − t_inc) ws − kW (predicho) | L2 alineado rec. 1 / rec. 2 |
|---|---|---|---|---|---|---|
| 0.20 | 7.5 | 0.1467 / 0.1464 (+6.4 / +6.2 %) | 1.1321 / 1.1353 | +0.22 %, −0.28 % | −63.9 ns, −0.69 % (−68.3) | 5.3 / 5.6 % |
| 0.15 | 10 | 0.1424 / 0.1425 (+3.2 / +3.3 %) | 1.1343 / 1.1363 | −0.06 %, −0.18 % | −68.0 ns, −0.75 % (−61.0) | 3.8 / 4.8 % |
| 0.10 | 15 | 0.1397 / 0.1399 (+1.3 / +1.4 %) | 1.1372 / 1.1372 | −0.12 %, +0.00 % | −41.5 ns, −0.45 % (−35.4) | 1.9 / 2.8 % |
| 0.05 | 30 | 0.1384 / 0.1384 (+0.31 / +0.34 %) | 1.1377 / 1.1377 | −0.03 %, +0.01 % | −11.7 ns, −0.13 % (−10.1) | 0.54 / 0.81 % |

wavesim con CFL 0.3 (la de k-Wave), a 0.10 mm: Δ(t_ref − t_inc) = −1.2 ns
(−0.01 %), L2 alineado 0.25 / 0.24 %; a 0.05 mm, −1.7 ns y 0.14 / 0.14 %.

### B. Fuente puntual con atenuación (2 dB/cm/MHz^1.1, 1580 m/s), dx = 0.10 mm

α y c entre receptores a 5 y 15 mm. "Su modelo": la relación de dispersión
continua de cada código (k-Wave: `kwave_model`, resuelta desde la ecuación
de estado de `kspace_solver.py`; wavesim: `attenuation.model_wave` con sus
pesos GSLS). "k-Wave ajustado": `sound_speed` = 1523.07 m/s y `alpha_coeff`
= 2.1678 para que su modelo dé 1580 m/s y 2 dB/cm a 1 MHz.

| f (MHz) | 0.70 | 0.85 | 1.00 | 1.15 | 1.30 |
|---|---|---|---|---|---|
| α k-Wave nominal / su modelo − 1 | −0.29 % | −0.27 % | −0.26 % | −0.24 % | −0.34 % |
| α k-Wave nominal / ley pedida − 1 | −7.5 % | −7.6 % | −7.7 % | −7.8 % | −7.9 % |
| c k-Wave nominal, medida / su modelo (m/s) | 1634.7 / 1634.6 | 1635.9 / 1635.7 | 1637.0 / 1636.6 | 1637.9 / 1637.4 | 1638.8 / 1638.0 |
| α k-Wave ajustado / ley pedida − 1 | −0.04 % | −0.13 % | −0.24 % | −0.39 % | −0.53 % |
| α wavesim 2 mec. / su modelo − 1 | −0.26 % | −0.21 % | −0.19 % | −0.17 % | −0.12 % |
| α wavesim 2 mec. / ley pedida − 1 | +1.09 % | +0.32 % | −0.71 % | −1.37 % | −1.50 % |
| **α wavesim 2 mec. / k-Wave ajustado − 1** | **+1.13 %** | **+0.46 %** | **−0.47 %** | **−0.99 %** | **−0.97 %** |
| **c wavesim 2 mec. / k-Wave ajustado − 1** | **+0.05 %** | **+0.10 %** | **+0.14 %** | **+0.18 %** | **+0.22 %** |
| c wavesim medida − su modelo, predicho por la dispersión del esquema | +0.07 % (+0.09) | +0.11 % | +0.16 % (+0.18) | +0.22 % | +0.28 % (+0.30) |
| α wavesim 1 mec. / k-Wave ajustado − 1 | +5.3 % | +8.2 % | +7.6 % | +4.9 % | +1.0 % |
| α wavesim 3 mec. / k-Wave ajustado − 1 | −1.4 % | −1.6 % | −1.1 % | −0.2 % | +0.7 % |

Tiempos: k-Wave 97-109 s por ejecución, wavesim 3.7 / 5.2 / 6.3 s con 1 / 2 / 3 mecanismos.

### C. Medio tipo tejido, fluido (dx = 0.10 y 0.05 mm)

32 × 26 mm: agua, piel de 1.8 mm con superficie en cuenco (5 → 6.2 mm),
grasa con fondo ondulado y rugoso (13 ± 1.2 mm, más ±0.3 mm de 2.3 mm de
periodo), músculo (propiedades de `materials.acoustic`, sin pérdidas). Fuente
puntual en el agua; 32 receptores de eco a 1.5 mm (sin la onda directa) y 32
de transmisión a 23 mm. Error L2 del array entero; desfase y llegada, mediana
y rango de las 32 trazas.

| dx, CFL de wavesim | L2 crudo eco / trans. | L2 alineado eco / trans. | desfase que alinea (trans.) | 1.ª llegada ws − kW (rel. al vuelo) | amplitud |
|---|---|---|---|---|---|
| 0.10 mm, 0.9 | 21.2 / 27.7 % | 8.1 / 6.4 % | +40.6 ns [+37.7, +45.7] | −0.63 % [−0.67, −0.61] | 0.998 |
| 0.10 mm, 0.3 | 1.4 / 0.9 % | 0.93 / 0.34 % | −1.0 ns | +0.010 % [−0.03, +0.04] | 0.999 |
| 0.05 mm, 0.9 | 6.7 / 8.5 % | 2.5 / 1.8 % | +12.5 ns | −0.19 % | 1.000 |
| 0.05 mm, 0.3 | 0.8 / 0.5 % | 0.46 / 0.21 % | −0.8 ns | −0.006 % | 1.000 |

Tiempos a 0.10 mm: k-Wave 36 s, wavesim 1.3 s (CFL 0.9) y 4.4 s (CFL 0.3).

### D1. Interfaz agua | cortical (1000/1500 | 1908/3500/1800), haces gaussianos

|R| y vS/vi en el centro del haz a 1 MHz; entre corchetes, máx\|err\|
frente a la analítica (Zoeppritz fluido-sólido, `analytic.fluid_solid_coefficients`)
en 0.8-1.2 MHz. Críticos: P 25.4°, S 56.4° (a 30° y 45° la P transmitida es
evanescente y toda la transmisión es S).

| ppw | código | 0° | 30° | 45° | vS/vi 30° | vS/vi 45° |
|---|---|---|---|---|---|---|
| | analítico | 0.6332 | 0.4151 | 0.4399 | 0.6256 | 0.6861 |
| 12 | wavesim | 0.6357 [0.61 %] | 0.4166 [0.62 %] | 0.4421 [0.69 %] | 0.6243 [0.81 %] | 0.6854 [0.16 %] |
| 12 | wavesim sin corrección | 0.6357 [0.61 %] | 0.4200 [1.18 %] | 0.4519 [3.16 %] | 0.6232 | 0.6817 |
| 12 | Devito orden 4 | 0.6357 [0.61 %] | 0.4198 [1.14 %] | 0.4518 [3.15 %] | 0.6240 [1.08 %] | 0.6816 [0.77 %] |
| 12 | Devito orden 8 | 0.6356 [0.56 %] | 0.4204 [1.26 %] | 0.4552 [4.04 %] | 0.6241 [1.09 %] | 0.6805 [0.93 %] |
| 20 | wavesim | 0.6340 [0.19 %] | 0.4173 [0.68 %] | 0.4407 [0.24 %] | 0.6226 [0.99 %] | 0.6859 [0.04 %] |
| 20 | Devito orden 4 | 0.6340 [0.19 %] | 0.4190 [0.94 %] | 0.4475 [2.02 %] | 0.6229 [1.05 %] | 0.6833 [0.47 %] |
| 20 | Devito orden 8 | 0.6340 [0.18 %] | 0.4194 [1.03 %] | 0.4496 [2.60 %] | 0.6228 [1.06 %] | 0.6825 [0.61 %] |
| 30 | wavesim | 0.6335 [0.08 %] | 0.4175 [0.71 %] | 0.4403 [0.11 %] | 0.6215 [1.06 %] | 0.6860 [0.03 %] |
| 30 | wavesim sin corrección | 0.6335 [0.08 %] | 0.4186 [0.83 %] | 0.4452 [1.40 %] | 0.6211 | 0.6841 |
| 30 | Devito orden 4 | 0.6335 [0.08 %] | 0.4186 [0.83 %] | 0.4451 [1.40 %] | 0.6216 [1.14 %] | 0.6842 [0.33 %] |
| 30 | Devito orden 8 | 0.6335 [0.08 %] | 0.4188 [0.90 %] | 0.4466 [1.79 %] | 0.6215 [1.15 %] | 0.6837 [0.42 %] |

Devito / wavesim − 1 en |R| a 1 MHz: 0° ≤ 0.02 % a todas las resoluciones;
30° +0.78 / +0.42 / +0.26 % (orden 4; 12 / 20 / 30 ppw); 45° +2.20 / +1.54 /
+1.10 %. **Devito orden 4 frente a wavesim sin corrección: ≤ 0.01 % en |R| y
≤ 0.1° en fase en todos los casos.** Tiempos (3 haces, 30 ppw): wavesim 67 s,
Devito 145-155 s.

### D2. Placa de cortical de 3 mm en agua: S0 y A0 (100-500 kHz)

Velocidad de fase por f-k frente a Rayleigh-Lamb con carga de agua (parte
real de la raíz compleja, `analytic.lamb_fundamental`). S0 sólo hasta
350 kHz (más arriba su fuga, Im k = 77-312 1/m, hace ambiguo el máximo f-k en
todos los códigos); A0 en las 17 frecuencias. máx\|err\| y, entre paréntesis,
error medio:

| dx (celdas en el espesor) | código | S0 | A0 |
|---|---|---|---|
| 0.25 mm (12) | wavesim | 0.42 % (+0.02 %) | 1.02 % (−0.66 %) |
| | wavesim sin corrección | 1.59 % (+0.11 %) | 0.56 % (+0.35 %) |
| | Devito orden 4 | 1.59 % (+0.10 %) | 0.56 % (+0.35 %) |
| | Devito orden 8 | 1.71 % (+0.11 %) | 0.99 % (+0.84 %) |
| 0.15 mm (20) | wavesim | 0.42 % (+0.03 %) | 0.39 % (−0.24 %) |
| | wavesim sin corrección | 0.44 % (+0.02 %) | 0.55 % (+0.45 %) |
| | Devito orden 4 | 0.44 % (+0.02 %) | 0.55 % (+0.45 %) |
| | Devito orden 8 | 0.45 % (+0.02 %) | 0.92 % (+0.77 %) |
| 0.10 mm (30) | wavesim | 0.47 % (+0.06 %) | 0.16 % (−0.09 %) |
| | wavesim sin corrección | 0.46 % (+0.05 %) | 0.51 % (+0.40 %) |
| | Devito orden 4 | 0.46 % (+0.04 %) | 0.51 % (+0.40 %) |
| | Devito orden 8 | 0.46 % (+0.04 %) | 0.79 % (+0.62 %) |

Los máximos de S0 están en 350 kHz (con 12 celdas, +1.6 % sin corrección y en
Devito frente a +0.4 % en wavesim); por debajo, todos ≤ 0.33 %.

Devito orden 4 / wavesim − 1: S0 −0.01 % de media (máx 0.07 % a 0.15 y
0.10 mm); A0 +1.01 / +0.69 / +0.49 % de media a 0.25 / 0.15 / 0.10 mm. Devito
orden 4 frente a wavesim sin corrección: A0 igual a ≤ 0.01 %. Todos los
campos estables (máx\|v\| de la segunda mitad 0.3-0.6 veces el de la primera).

## 3. Resumen frente a las tolerancias

| Magnitud | Medido | Tolerancia | ¿Cumple? |
|---|---|---|---|
| A: R, T wavesim / k-Wave (0.10 mm) | −0.12 %, 0.00 % | ±1 % | sí (a todas las dx: ≤ 0.28 %) |
| A: R frente a la analítica, ambos códigos | +1.3 % (15 ppw), +0.3 % (30 ppw) | ±1 % | sólo desde ~20 ppw (los dos igual) |
| A: tiempos, CFL 0.9 / 0.3 (0.10 mm) | −0.45 % / −0.01 % | ±0.5 % | sí (al límite con CFL 0.9; no a 7.5-10 ppw) |
| A: forma de onda tras alinear, CFL 0.9 / 0.3 | 1.9-2.8 % / 0.24 % | ≤ 3 % | sí desde 15 ppw |
| B: α wavesim 2 mec. / k-Wave ajustado | −1.0 … +1.1 % | ±1 % | casi (máx 1.13 % a 0.7 MHz) |
| B: c wavesim / k-Wave ajustado | +0.05 … +0.22 % | ±1 % | sí |
| C: L2 alineado, CFL 0.9 (0.10 / 0.05 mm) | 6-8 % / 1.8-2.5 % | ≤ 3 % | **no** a 15 ppw; sí a 30 ppw o con CFL 0.3 (0.3-0.9 %) |
| C: 1.ª llegada, CFL 0.9 / 0.3 | −0.63 % / +0.01 % | ±0.5 % | **no** con CFL 0.9 a 15 ppw; sí con 0.3 o a 30 ppw (−0.19 %) |
| D1: \|R\| wavesim / analítica, 0° y 45° | ≤ 0.7 % (12 ppw), ≤ 0.24 % (≥ 20 ppw) | ±1 % | sí |
| D1: \|R\| wavesim / analítica, 30° | 0.6-0.7 % (no converge) | ±1 % | sí |
| D1: \|R\| wavesim / Devito orden 4, 45° | 2.2 / 1.5 / 1.1 % | ±1 % | **no**: es la corrección de interfaz (abajo) |
| D1: vS/vi wavesim / analítica | ≤ 1.06 % (30°), ≤ 0.16 % (45°) | ±1 % | casi |
| D2: S0 y A0 wavesim / analítica | ≤ 0.47 % (≥ 20 celdas) | ±1 % | sí (A0 1.02 % con 12 celdas) |
| D2: A0 wavesim / Devito orden 4 | −1.0 / −0.7 / −0.5 % | ±1 % | sí, al límite con 12 celdas |

## 4. Lectura: de dónde sale cada discrepancia

- **R en dos capas (A): error común de discretizar el salto, no de
  wavesim.** Los dos códigos dan el mismo R a cada resolución (≤ 0.3 % entre
  sí) y convergen juntos a la analítica (+6.3 → +0.3 % de 7.5 a 30 ppw, ~2.º
  orden), aunque k-Wave es espectral en espacio: lo que comparten es el medio
  muestreado en la malla con la densidad promediada en los nodos de velocidad.
  `elastico.md` §2(b) lo analiza en 1D (cociente espectral, sin dependencia
  de la CFL).
- **Tiempos y forma de onda (A, C): dispersión temporal del leapfrog de
  wavesim a CFL 0.9.** k-Wave corrige la dispersión temporal (corrección
  k-space, exacta en un medio homogéneo con `c = c_ref`) y usa CFL 0.3. El
  adelanto de la envolvente de wavesim coincide con el retardo de grupo que
  predice la relación de dispersión de su propio esquema
  (`analytic.discrete_wavenumber`): −35.4 frente a −41.5 ns a 0.10 mm, −10.1
  frente a −11.7 a 0.05 mm (la velocidad de grupo se adelanta ~3 veces más
  que la de fase, `(ω dt)²/8` frente a `/24`). Con CFL 0.3 el desfase
  desaparece (−1 a −2 ns) y el error de forma tras alinear baja de 2-8 % a
  0.2-0.9 %. El error "tras alinear" no llega a cero con CFL 0.9 porque la
  dispersión no es un retardo puro: deforma el pulso (fase y envolvente se
  desplazan distinto: +40 ns frente a −96 ns en C). Es decir, el 6-8 % de C a
  15 ppw es el precio de la CFL por defecto de wavesim, no del medio
  heterogéneo, de las interfaces escalonadas ni de la CPML: con la misma malla
  y CFL 0.3, las 64 trazas coinciden con k-Wave a 0.3-0.9 % (eco con la
  interfaz rugosa incluido) y las amplitudes a 0.1 %.
- **Atenuación (B): el −7.7 % de k-Wave de la comparación anterior es una
  convención suya, ahora verificada.** En k-Wave `sound_speed` es la velocidad a
  frecuencia cero de su modelo (`1/c = 1/c0 + a0 tan(πy/2) ω^(y−1)`, con
  tan(π·1.1/2) < 0), y la absorción efectiva es ~`α0 ω^y (c0/c)^(y+1)`. El
  modelo exacto de su ecuación de estado reproduce sus medidas a 0.3 % en α y
  a 0.05 % en c (1634.7-1638.8 m/s). Con los parámetros ajustados k-Wave
  sigue la ley pedida a ≤ 0.5 % y la diferencia con wavesim (−1.0 a +1.1 %)
  es la del ajuste GSLS de 2 mecanismos a la ley de potencia (+1.1 a −1.5 %
  frente a la ley). La c de wavesim queda +0.05 a +0.22 % por encima por la
  dispersión numérica (predicha +0.09 a +0.30 %). El −0.2/−0.3 % común de
  "medida / su modelo" en todos los códigos es del estimador (campo cercano
  de la fuente lineal 2D, corrección 1/√r asintótica).
- **R(θ) oblicuo (D1) y A0 (D2): la corrección de deslizamiento de wavesim.**
  Devito orden 4 y wavesim sin corrección dan lo mismo a ≤ 0.01 % en todos
  los casos: las dos implementaciones del esquema estándar son idénticas
  (mismo estencil, medio y dt; la esponja y la CPML no influyen). La
  diferencia wavesim − Devito es, entera, la corrección de `_slip_terms`:
  sin ella el error de |R| a 45° no converge como el de un esquema de 4.º
  orden (3.2 → 2.0 → 1.4 % de 12 a 30 ppw, ~1/ppw) y A0 queda +0.4-0.5 %
  rápida con 20 y 30 celdas en el espesor; con ella, 0.69 → 0.11 % a 45° y
  A0 −0.66 → −0.09 % de media. Subir a orden 8 en Devito **empeora**
  (45°: 4.0 / 2.6 / 1.8 %; A0 +0.6-0.9 %): el estencil más ancho lee más
  nodos del otro lado de la interfaz. A incidencia normal (0°) no hay
  cizalla en la interfaz y los cuatro códigos coinciden a 0.02 %.
- **Lo que no se explica del todo:** |R| a 30° (+0.6-0.7 %) y vS/vi a 30°
  (−0.2 a −0.65 % a 1 MHz, máx ~1 %) no bajan al refinar en ningún código; a
  30° el haz está a 4.6° del crítico P y `elastico.md` §3 ya atribuye los
  errores de esa zona al montaje finito (onda lateral). Como aparece igual en
  Devito, no es del solver. A0 con 12 celdas en el espesor (−1.0 % en
  wavesim) es dispersión de malla: baja a −0.1 % con 30 celdas.

## 5. Qué no cubre

- k-Wave elástico (`pstdElastic2D`) no existe en k-wave-python 0.6.2; no se
  probó el k-Wave de MATLAB. La parte elástica se comparó con un segundo FDTD
  (Devito) que comparte el tipo de esquema: valida la implementación y aísla
  la corrección de interfaz, pero no es un método numérico distinto (como
  SPECFEM2D o un pseudoespectral elástico).
- Sin atenuación en los casos elásticos ni en C; B es homogéneo.
- Medio de C sin hueso ni speckle; una sola geometría.
