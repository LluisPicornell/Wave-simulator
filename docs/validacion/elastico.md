# Validación del solver elástico frente a soluciones analíticas (U3, U4, U5)

Casos U3, U4 y U5 del plan de `docs/comparativa_simuladores.md` §11.1: la
parte del ultrasonido que k-wave-python no cubre (ondas S, conversión de
modo, ondas guiadas) y que se puede comprobar sin software externo.

- Script: `benchmarks/elastic_validation.py` (`--case u3|u4|u5|backends`,
  `--backend numpy`, `--quick`). Todo: 6.5 min con 1 hilo de Numba en una CPU
  compartida (U3 ~1 min, U4 ~4.5 min, U5 ~1 min).
- Tests rápidos: `tests/test_elastic_validation.py` (12 tests, ~21 s).
- Referencias: `wavesim/ultrasound/analytic.py` (dispersión del propio
  esquema, coeficientes fluido-sólido, Rayleigh-Lamb con y sin fluido).
- Medios isótropos y **sin atenuación** (`ElasticMedium` construido a mano,
  `tau = 0`), con las propiedades de `materials/acoustic.py`: agua 1000 kg/m³,
  1500 m/s; cortical 1908 kg/m³, cp = 3500 m/s, cs = 1800 m/s; aire 1.2 kg/m³,
  343 m/s. CFL 0.9 salvo que se diga otra cosa; CPML de 20 celdas.

## Resumen

| Caso | Magnitud | Medido | Referencia | Objetivo | ¿Cumple? |
|---|---|---|---|---|---|
| U3a | Velocidad de fase, agua, frente a la dispersión del esquema | coincide a ≤ 1e-6 (6-32 ppw) | relación de dispersión discreta | — | sí |
| U3a | Pendiente log-log del error de c (CFL fija) | 1.84 → 1.92 → 1.96 (12→32 ppw) | 2 (temporal) | ≥ 2 | asintóticamente (→ 2 por abajo) |
| U3a | Ídem con dt fijo (error espacial) | 3.96, 3.99, 4.12 | 4 | — | sí |
| U3b | Pendiente del error de R agua-cortical | 1.94-2.83 (6-32 ppw) | | ≥ 2 | sí (≈ 2) |
| U3b | Error de R / T agua-cortical a 16 ppw | +0.21 % / −0.17 % | 0.6332 / 1.6332 | | |
| U4 | \|R(θ)\| lejos de los críticos (0-22°) | ≤ 0.7 % | Zoeppritz fluido-sólido | ±3 % | sí |
| U4 | \|R(θ)\|, 28-53° | ≤ 3.9 % (media +0.2 %) | | ±3 % | no en toda la zona: ≤ 1.2 % en los ángulos de la tabla, 3.9 % en sus bordes |
| U4 | \|R(θ)\|, 59-70° (reflexión total) | ≤ 5 % (a 1 MHz ≤ 1.8 %) | 1 | ±3 % | no en todo el rango |
| U4 | Ángulo crítico P (máximo de \|R\|) | 25.24-25.48° | 25.38° | ±1° | sí (±0.15°) |
| U4 | Amplitud de la S transmitida, 0-53° | ≤ 3.4 % (a 1 MHz ≤ 0.7 % en 30-50°) | | | |
| U5 | S0 / A0 placa libre, dx 0.10 mm | ≤ 1.3 % / ≤ 1.7 % | Rayleigh-Lamb | ±2 % | sí |
| U5 | S0 / A0 placa en agua, dx 0.15 mm | ≤ 0.4 % / ≤ 0.4 % | Rayleigh-Lamb con fluido | ±2 % | sí (S0 hasta 350 kHz) |
| — | Numba frente a NumPy | R idéntico a 7 cifras; 6e-8 en U4 | | | sí |

Por el camino se encontró y corrigió un error real del solver en las
interfaces fluido-sólido (sección 1), y dos inestabilidades que **no** se
corrigen aquí (sección 6).

## 1. Corrección del solver: la interfaz fluido-sólido a incidencia oblicua

**Síntoma.** En U4, el error de \|R(θ)\| agua-cortical no bajaba al refinar
la malla: a 45° era +2.4 % con 15 puntos por longitud de onda (ppw), +1.1 %
con 25 y +1.2 % con 35; a 50°, +4.4 % y +2.7 %. A incidencia normal (U3b) sí
convergía a 2.º orden.

**Causa.** La deformación de cizalla de un nodo `sxz` del sólido se calcula
con el estencil de 4.º orden, que toma con peso C2 = −1/24 la velocidad
tangencial `vx` de una fila situada al otro lado de la interfaz. El fluido no
viscoso desliza, así que esa velocidad no es la prolongación de la del sólido:
el salto entra como C2/dx, un error O(1) que no desaparece al refinar. A
incidencia normal `vx = 0` y no se ve.

**Arreglo** (`_slip_terms` en `elastic_fdtd.py`, una función auxiliar y una
corrección aplicada en `_deriv` para NumPy y tras cada kernel para Numba): ese
punto exterior se sustituye por el nodo interior del sólido y el cambio
traspuesto se aplica a la ecuación del movimiento (el fluido deja de recibir
la cizalla del sólido). Así los operadores de deformación y de fuerza siguen
siendo adjuntos (D_b = −D_fᵀ, comprobado en un test) y el esquema conserva la
energía. Una primera versión que sólo pasaba esas derivadas a 2.º orden
mejoraba R igual pero rompía esa simetría: un fantoma de tibia con fractura
sin atenuación crecía exponencialmente (×25 cada 2500 pasos a partir del paso
~10⁴). La versión final es estable 40 000 pasos.

**Efecto medido** (haz a 45°, |R| en 43.5-46.5°, `tests/test_elastic_validation.py`):

| ppw | sin corrección | con corrección |
|---|---|---|
| 10 | 4.6 % | 1.5 % |
| 12 | 3.6 % | 0.8 % |
| 20 | 2.3 % | 0.4 % |

**Impacto en las simulaciones por defecto** (rama integrada con la atenuación
por mecanismo y la cortical anisótropa):

- B-mode de tibia con fractura (`simulate_bmode`, 3 MHz, 3 ángulos, malla
  367×400): eco cortical (máximo por columna) −0.00 dB de media, rango
  [−0.43, +0.03] dB; sombra 2-8 mm bajo la cortical +0.01 dB; tejido blando
  sin cambio; píxeles por encima de −30 dB: p95 de \|Δ\| 0.33 dB; diferencia
  L2 de la envolvente 0.6 %.
- Estabilidad, 40 000 pasos, fantoma de tibia con fractura: estable con y sin
  corrección, con 0 y 1 mecanismos. NumPy y Numba coinciden a ~1e-6 del pico
  de `sxz` (400 pasos) con 0, 2 y 3 mecanismos, isótropo y con
  `anisotropic_bone=True` (la cizalla es C55), en el fantoma básico y en el
  realista poroso.
- Coste: +7-11 % por paso con Numba (1 hilo, 2790-11 900 nodos corregidos en
  la malla de 367×400); en NumPy, dentro del ruido de medida (±10 %, CPU
  compartida).
- Tests existentes: todos pasan.

## 2. U3: convergencia

### (a) Onda plana en medio homogéneo

Fuente de presión en una fila entera y medio uniforme en x: el problema 2D es
exactamente 1D (también dentro de la CPML lateral), así que basta una franja
de 6 columnas y se puede refinar mucho. Velocidad de fase en f0 = 1 MHz por la
fase de la DFT entre dos receptores separados 20 λ; error L2 de la forma de
onda frente a la propagación exacta.

Agua, CFL 0.9:

| ppw | dx (µm) | c/c − 1 medido | predicho por el esquema | error L2 | \|b\|/\|a\| |
|---|---|---|---|---|---|
| 6 | 250.0 | +8.546e-3 | +8.546e-3 | 1.02 | 1.00008 |
| 8 | 187.5 | +5.996e-3 | +5.996e-3 | 0.80 | 1.00003 |
| 12 | 125.0 | +3.064e-3 | +3.063e-3 | 0.45 | 0.99999 |
| 16 | 93.8 | +1.805e-3 | +1.805e-3 | 0.27 | 1.00001 |
| 24 | 62.5 | +8.284e-4 | +8.284e-4 | 0.13 | 1.00003 |
| 32 | 46.9 | +4.712e-4 | +4.712e-4 | 0.073 | 0.99998 |

Pendientes locales: c 1.23, 1.66, 1.84, 1.92, 1.96; L2 0.83, 1.44, 1.74, 1.87, 1.94.

Con dt fijo (1.89 ns, CFL 0.1 a 32 ppw), sólo error espacial: −5.40e-3,
−1.73e-3, −3.42e-4, −1.04e-4 a 6, 8, 12 y 16 ppw (pendientes 3.96, 3.99,
4.12). A 10 ppw, al variar la CFL: −4.7e-4 (0.2), +2.5e-4 (0.4), +1.5e-3 (0.6),
+3.2e-3 (0.8), +4.2e-3 (0.9), +5.2e-3 (0.99). En la cortical, la P sigue la
misma tabla y la S (dt fijado por cp, CFL efectiva 0.46) tiene errores
4-15 veces menores (+3.9e-4 a 16 ppw); siempre igual a la predicción.

**Lectura.** La velocidad medida coincide con la que predice la relación de
dispersión del propio esquema (`analytic.discrete_wavenumber`) a 1e-6: no hay
error de implementación, todo el error es dispersión numérica y se puede
calcular de antemano. Para CFL y kdx pequeños,
`c_num/c − 1 ≈ (ω dt)²/24 − 3 (k dx)⁴/640`: con CFL 0.9 domina el término
temporal (2.º orden, la onda se adelanta) y el espacial (4.º orden, la
frena) lo compensa en parte en mallas gruesas; por eso la pendiente sube
hacia 2 por abajo (1.96 entre 24 y 32 ppw) en lugar de superarla. La
amplitud se conserva a 1e-4 (el leapfrog no disipa). El error L2 de 7 % a
32 ppw es la fase acumulada en 20 λ (2π·20·4.7e-4 = 0.06 rad). La CFL que
minimiza el error a 10 ppw está entre 0.2 y 0.4, donde los dos términos se
cancelan; el solver usa 0.9 por coste.

### (b) Reflexión y transmisión a incidencia normal

El medio 2 ocupa las filas `j >= j_int`; la interfaz queda en `(j_int − 1/2) dx`,
sobre la fila de nodos `vz` (densidad media aritmética). Dos ejecuciones con
el mismo dt (con interfaz y sólo con el medio 1); |R| y |T| en f0 por cociente
espectral (en 1D sin pérdidas no dependen de la dispersión); T es la tracción
normal `−szz` en el sólido. `delta` es el desplazamiento efectivo de la
interfaz (de la fase de R corregida con el k discreto).

| ppw | fluido-fluido 1500/1000 \| 1800/1100: err R | err T | agua \| cortical: err R | err T | delta/dx |
|---|---|---|---|---|---|
| 6 | +9.9 % | −0.86 % | +2.22 % | −2.77 % | −0.004 |
| 8 | +4.9 % | −0.31 % | +1.05 % | −1.11 % | −0.003 |
| 12 | +1.96 % | −0.083 % | +0.39 % | −0.35 % | −0.001 |
| 16 | +1.08 % | −0.035 % | +0.21 % | −0.17 % | −0.001 |
| 24 | +0.48 % | −0.012 % | +0.095 % | −0.065 % | −0.000 |
| 32 | +0.24 % | −0.006 % | +0.042 % | −0.035 % | −0.000 |

Pendientes de err R: 2.47, 2.25, 2.08, 1.98, 2.43 (fluido-fluido) y 2.61,
2.43, 2.17, 1.94, 2.83 (agua-cortical); de err T, 2.2-3.5.

Diagnóstico con el mismo R = 0.1379 (16 ppw): sólo salto de c (1500 | 1980)
err R +2.0 %; sólo salto de rho (1000 | 1320) −1.24 %. Agua-cortical a 12 ppw
con CFL 0.3, 0.6 y 0.9: err R +0.388, +0.389 y +0.390 %.

**Lectura: qué domina el error de R.**

- **No es la dispersión numérica**: |R| por cociente espectral no depende de
  ella y da lo mismo que el cociente de envolventes de §10.2 (salvo a 6-8 ppw,
  donde el pulso se deforma); tampoco depende de la CFL (0.388-0.390 %).
- **No es la "media celda" de la interfaz escalonada**: la interfaz efectiva
  está en la fila `vz`, a menos de 0.004 dx de `(j_int − 1/2) dx`. Un error de
  posición cambiaría la fase, no |R|.
- **Es la discretización del salto en sí**: el estencil de 4.º orden atraviesa
  la discontinuidad de la derivada (p y vz son continuos pero con un quiebro),
  y el esquema pasa a ser de 2.º orden en la interfaz. El salto sólo de c
  (módulo, nodos de tensión) da R por exceso y el salto sólo de densidad
  (promedio aritmético en el nodo vz), por defecto; con los dos a la vez se
  compensan en parte. En valor absoluto el error es pequeño (0.0015 a 16 ppw
  en ambos casos); en relativo pesa más cuando R es pequeño, como en el caso
  fluido-fluido de §10.2, cuyo 6 % a 8 ppw (también en k-Wave) es compatible
  con este efecto (aquí, en 1D, +4.9 % espectral y +5.4 % por envolvente a
  8 ppw).
- T converge más deprisa en relativo porque T = 1 + R es grande.

## 3. U4: R(θ) en la interfaz agua-cortical

**Montaje.** Haces gaussianos (σ = 4 λ, truncados a ±3.5σ) emitidos por una
fila de fuentes de presión con retardo lineal, centrados en 0, 10, 20, 30, 38,
46, 52, 60, 68 y 76°, más uno en el crítico P con barrido fino; 1.5 λ de
agua entre las filas de receptores y la interfaz y 6 λ de cortical. En cada
frecuencia (0.8-1.2 MHz) y cada número de onda horizontal `kx = ω sinθ/c` (DFT
a lo largo de toda la fila) se separan la onda descendente y la ascendente con
dos filas de receptores a 3 celdas y el kz exacto del esquema
(`analytic.discrete_kz`): `R = ascendente/descendente` en el plano de la
interfaz, válido para cualquier haz. Restar una ejecución sin hueso (lo
primero que se probó) dejaba un rizado de ±3 % en R(θ) por el eco de la CPML
superior, que a incidencia rasante es del 1-3 %. La S transmitida se separa
de la P con un ajuste de ondas planas descendentes en seis filas del sólido y
se da como velocidad de partícula relativa a la de la onda incidente. Para
cada θ se usa el haz con más energía en ese ángulo.

**Referencia.** Sistema 3×3 de las condiciones de contorno (continuidad de
`u_z`, `σzz = −p`, `σxz = 0`), que coincide con la forma cerrada de
Brekhovskikh a 1e-12 y conserva la energía (\|R\|² + E_P + E_S = 1).
Críticos: P 25.38°, S 56.44°; Rayleigh con fuga cerca de 63.6°.

Errores por zona (0.8-1.2 MHz):

| zona | 15 ppw: máx\|err \|R\|\| | err medio | máx\|Δfase\| | máx\|err vS\| | 25 ppw: máx\|err\| | err medio | máx\|Δfase\| | máx\|err vS\| |
|---|---|---|---|---|---|---|---|---|
| 0° - 22.4° | 0.68 % | +0.20 % | 0.35° | 3.2 % | 0.73 % | +0.07 % | 0.30° | 3.4 % |
| θp ± 3° | 15.7 % | +0.16 % | 11.6° | — | 15.9 % | +0.11 % | 11.7° | — |
| 28.4° - 53.4° | 3.9 % | +0.23 % | 2.1° | 2.3 % | 4.0 % | +0.05 % | 2.2° | 2.8 % |
| θs ± 3° | 17.2 % | +1.1 % | 9.6° | — | 17.6 % | +0.5 % | 10.0° | — |
| 59.4° - 70° | 5.0 % | +0.21 % | 4.3° | — | 5.2 % | +0.21 % | 2.9° | — |
| 70° - 80° | 9.4 % | +0.02 % | 5.8° | — | 9.7 % | +0.14 % | 8.2° | — |

A 1 MHz (25 ppw), medido / exacto:

| θ | \|R\| | fase (°) | vS/vi |
|---|---|---|---|
| 0° | 0.6337 / 0.6332 | 0.0 / 0.0 | — |
| 20° | 0.6278 / 0.6279 | 0.1 / 0.0 | 0.287 / 0.287 |
| 27° | 0.6139 / 0.6003 | −29.2 / −29.5 | 0.539 / 0.545 |
| 30° | 0.4174 / 0.4151 | −12.5 / −12.7 | 0.622 / 0.626 |
| 40° | 0.4539 / 0.4538 | −2.1 / −2.1 | 0.646 / 0.646 |
| 45° | 0.4404 / 0.4399 | −10.0 / −9.9 | 0.686 / 0.686 |
| 50° | 0.4238 / 0.4288 | −26.1 / −26.6 | 0.762 / 0.763 |
| 54° | 0.5007 / 0.5072 | −48.6 / −50.4 | 0.880 / 0.892 |
| 60° | 1.0032 / 1.0000 | −145.7 / −145.9 | |
| 65° | 1.0063 / 1.0000 | 172.0 / 171.5 | |
| 70° | 1.0182 / 1.0000 | 154.7 / 155.5 | |
| 75° | 1.0691 / 1.0000 | 155.2 / 152.7 | |

Ángulo crítico P (máximo de \|R\|) a 0.8, 0.9, 1.0, 1.1 y 1.2 MHz: 25.24,
25.33, 25.40, 25.45 y 25.48° (25 ppw; 15 ppw: lo mismo ±0.01°).

**Lectura.**

- Se cumple ±3 % en \|R\| hasta 22° (≤ 0.7 %) y, en el centro de los haces,
  en 28-53° (≤ 1.2 % en los ángulos de la tabla a 1 MHz) y pasado el crítico
  S hasta 70° (≤ 1.8 %). La fase compleja pasado el crítico P sigue a la exacta a 0.2-2.5°
  (salto de 2π de la onda de Rayleigh con fuga incluido). La conversión P→S
  (vS) está a ≤ 0.7 % entre 30 y 50°.
- El crítico P se sitúa a ±0.15° (objetivo ±1°). Su ligera deriva con la
  frecuencia (0.24° entre 0.8 y 1.2 MHz) es la dispersión numérica del
  agua y del hueso.
- **Lo que no se cumple**: el máximo de 3.9-5 % en 28-53° y 59-70° (en los
  bordes de cada zona, junto a los críticos o en los flancos de los haces) y
  los errores de 9-17 % a ±3° de los críticos y por encima de 70°. **No son
  del solver**: no cambian entre 15 y 25 ppw. Son del montaje finito: cerca
  de los críticos la onda lateral y la de Rayleigh con fuga llevan energía a lo
  largo de la interfaz hasta fuera de la fila de medida, y a incidencia
  rasante el haz es muy ancho sobre la fila. Una onda plana infinita los
  eliminaría; aquí haría falta un dominio mucho mayor. El error que sí
  dependía de la malla (sección 1: 1-3 % a 45-50° con 25-35 ppw) baja con la
  corrección a +0.1 % a 45° y 25 ppw.

## 4. U5: ondas de Lamb en una placa de cortical de 3 mm

**Montaje.** Placa de 200 mm, libre (en 1 mm de aire a cada lado) o en agua
(6 mm a cada lado), excitada con una fuerza horizontal y otra vertical en toda
su columna (pulso gaussiano de 300 kHz y ancho de banda 1.2); registro de
`vx` y `vz` en dos filas simétricas respecto al plano medio a lo largo de
180 mm durante 260 µs. Las combinaciones pares separan las familias S y A. La
velocidad de fase se obtiene por la transformada f-k (máximo cerca de la
curva teórica) afinada con la pendiente de la fase del modo filtrado en f-k
(banda ±25 % alrededor del máximo, k > 0): el máximo f-k solo está limitado por
la resolución 2π/L, que a 100 kHz es el 20 % del k de S0. Se da por resuelto un
modo si su máximo f-k es al menos el 30 % del mayor del espectro a esa
frecuencia.

**Referencia.** Raíces de Rayleigh-Lamb (simétricas y antisimétricas) de la
placa libre y, en agua, de la ecuación con carga de fluido (raíces complejas:
parte real = velocidad de fase, parte imaginaria = fuga), seguidas por
continuación desde la placa libre. Comprobaciones: S0 tiende a la velocidad de
placa (3087 m/s) a baja frecuencia, S0 y A0 libres a la de Rayleigh
(1675 m/s) a alta, y A0 en agua a la onda de Scholte agua-hueso (1359.6 m/s,
calculada aparte).

Velocidad de fase (m/s), medida / referencia:

| f (kHz) | S0 libre (0.10 mm) | A0 libre (0.10 mm) | S0 en agua | A0 en agua | A0 libre (teoría) |
|---|---|---|---|---|---|
| 100 | 3077.5 / 3076.4 | 1096.1 / 1094.9 | 3083.3 / 3079.3 | 945.5 / 945.9 | 1094.9 |
| 150 | 3065.5 / 3061.3 | 1238.1 / 1248.7 | 3066.3 / 3068.5 | 1083.8 / 1086.1 | 1248.7 |
| 200 | 3030.2 / 3037.5 | 1342.8 / 1351.6 | 3056.5 / 3052.4 | 1175.2 / 1176.1 | 1351.6 |
| 250 | 2997.6 / 3000.9 | 1420.3 / 1424.3 | 3029.8 / 3029.8 | 1233.6 / 1235.9 | 1424.3 |
| 300 | 2937.6 / 2943.6 | 1470.8 / 1477.5 | 2998.4 / 2999.9 | 1272.6 / 1276.3 | 1477.5 |
| 350 | 2836.9 / 2848.2 | 1511.6 / 1517.5 | 2975.6 / 2963.2 | 1300.9 / 1303.6 | 1517.5 |
| 400 | 2672.0 / 2675.3 | 1543.9 / 1548.2 | no resuelto (Im k 107 1/m) | 1318.0 / 1322.2 | 1548.2 |
| 450 | 2412.0 / 2380.8 | 1572.8 / 1572.0 | no resuelto | 1331.0 / 1334.7 | 1572.0 |
| 500 | no resuelto | 1597.7 / 1590.9 | no resuelto | 1338.0 / 1343.1 | 1590.9 |

Máximo \|error\| donde está resuelto (17 frecuencias de 100 a 500 kHz):

| configuración | S0 | A0 |
|---|---|---|
| libre, dx 0.15 mm (20 celdas en el espesor) | 8.7 % (≤ 0.5 % hasta 325 kHz) | 1.5 % |
| libre, dx 0.10 mm (30 celdas) | 1.3 % (15/17) | 1.7 % |
| en agua, dx 0.15 mm | 0.42 % (11/17, hasta 350 kHz) | 0.39 % |

**Lectura.**

- Se cumple ±2 % en S0 y A0 donde están resueltos, con 30 celdas en el
  espesor para la placa libre y con 20 en agua.
- **La placa libre (en aire) es la más exigente**: con 20 celdas, S0 se
  desvía hasta un 8.7 % a 400-500 kHz, donde su curva cae deprisa
  (fd = 1.2-1.5 MHz·mm) y el modo se concentra cerca de las caras (por la
  pendiente de la curva, equivale aproximadamente a una placa una celda más
  delgada; estimación). Con 30 celdas baja a ≤ 1.3 %: es error de
  discretización de la superficie libre, representada por un fluido de
  densidad casi nula (en la fila `vz` de la cara la densidad es la media
  hueso-aire, la mitad de la del hueso). En agua la misma malla da ≤ 0.4 %.
- El agua frena A0 un 13-16 % (946 frente a 1095 m/s a 100 kHz; 1338 frente a
  1591 a 500 kHz): A0 queda por debajo de la velocidad del agua, no radia y
  tiende a la onda de Scholte. La simulación sigue la curva con carga de
  fluido, no la de la placa libre; es el efecto de las condiciones de
  contorno del tejido blando que menciona la comparativa (Fotiadis y
  colaboradores) y que importa para la transmisión axial.
- S0 en agua tiene fuga creciente (Im k de 2 a 312 1/m entre 100 y 475 kHz):
  por encima de 350 kHz se atenúa en pocos milímetros y no se puede medir con
  este registro.
- **Estabilidad.** La placa libre en aire es estable (el campo no crece en
  la segunda mitad del registro: 0.16-0.24 del máximo de la primera), siempre
  que no entre en la CPML (sección 6).

## 5. Backends

Con Numba (`backend="auto"`) y con `backend="numpy"`: R y T de U3b
agua-cortical a 12 ppw idénticos a 7 cifras (0.6356345 y 1.6274419); en U4, a
46° y 12 ppw, \|R_numba − R_numpy\| / \|R\| = 6e-8. Un test lo comprueba en U4.

## 6. Problemas encontrados que no se corrigen aquí

- **CPML con ondas guiadas.** Una placa de cortical que entra en la CPML
  lateral se vuelve inestable: crecimiento exponencial dentro de la CPML a
  partir de ~3000 pasos (×300 cada 1500 pasos), con aire o con agua
  alrededor, con o sin la corrección de la sección 1, y también con CFL 0.6.
  Por eso en U5 la placa termina 5 mm antes de la CPML. Los fantomas de
  extremidad, cuya cortical sí cruza la CPML, se mantuvieron estables en las
  pruebas de 40 000 pasos (hueso grueso rodeado de tejido blando), pero una
  simulación larga de transmisión axial en una cortical fina podría
  encontrárselo.
- **Fantoma realista poroso sin atenuación.** La tibia `realism="realista"`
  del tipo `anciano_osteoporotico` crece exponencialmente con
  `n_mechanisms=0` (×10 cada 2500 pasos desde el paso ~5000) y, con el valor
  por defecto `n_mechanisms=1`, crece despacio (×2-3 en 40 000 pasos). Ocurre
  igual sin la corrección de la sección 1 y con CFL 0.75; el máximo está en el
  foco de fractura (hematoma junto a hueso poroso con poros de una celda). No
  se investigó la causa; en las duraciones de un B-mode (unos miles de pasos)
  no se nota.

## 7. Límites de esta validación

- Todo es 2D, isótropo y sin atenuación; la cortical anisótropa y la
  atenuación por mecanismo sólo se cubren con la igualdad NumPy/Numba de la
  sección 1, no contra soluciones analíticas.
- U4 sólo cubre una interfaz plana alineada con la malla; una interfaz
  inclinada (escalonada) añade el error de la escalera, no medido aquí.
- La onda de Rayleigh en superficie libre (opcional en el plan) no se hizo; el
  comportamiento de la placa libre con 20 celdas sugiere que necesitaría
  ~30 celdas por longitud de onda para llegar al 1 %.
