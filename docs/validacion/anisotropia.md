# Hueso cortical transversalmente isótropo

Script: `benchmarks/anisotropic_bone.py`. Tests rápidos:
`tests/test_anisotropy.py` y `tests/test_numba_backend.py`.

## Qué cambió

- **Solver** (`elastic_fdtd.py`, `_numba_kernels.py`): rigideces por celda
  C11, C33, C13, C55 con los ejes de la malla (x lateral, z profundidad):
  `sxx = C11 exx + C13 ezz`, `szz = C13 exx + C33 ezz`, `sxz = C55 exz`.
  El medio isótropo conserva su camino (mismos coeficientes y mismo orden de
  operaciones): con la opción desactivada el resultado es idéntico bit a bit
  al de antes y el paso cuesta lo mismo. En el kernel Numba la rama se elige
  por fila.
- **Viscoelasticidad coherente**: la parte "P" del tensor (C11, C33 y
  C13 + 2·C55) relaja con los pesos de la onda P y C55 con los de la S. En el
  límite isótropo son λ + 2μ y μ, y la memoria de C13 es la de
  λ = (λ + 2μ) − 2μ, exactamente la del camino isótropo. El Q es el de la
  tabla en todas las direcciones (la atenuación por unidad de longitud baja
  algo en la dirección rápida).
- **Estabilidad**: el paso se calcula con el mayor autovalor del operador en
  el número de onda de Nyquist en las dos direcciones,
  `[[C11 + C55, C13 + C55], [C13 + C55, C33 + C55]]` (en el caso isótropo da
  la condición de siempre, `2(λ + 2μ)`), con los módulos no relajados.
- **Medio** (`medium.py`): `elastic_medium(..., anisotropic_bone=False)`,
  también en `ElasticSolver` y `simulate_bmode`. Por defecto **desactivada**.

## Datos de la cortical

`materials.acoustic.CORTICAL_STIFFNESS`, de las medidas ultrasónicas de Rho
(Ultrasonics 34:777, 1996) en 96 muestras de tibia de 8 cadáveres, tal como
las da el resumen de Hoffmeister et al. (Med Biol Eng Comput 38:333, 2000):

| Constante | Publicado (GPa) | Usado (GPa) | Estado |
|---|---|---|---|
| c11 (radial), c22 (circunferencial) | 19.5, 20.1 | C_T = 19.8 (media) | verificado en el resumen |
| c33 (axial) | 30.9 | C_A = 30.9 | verificado |
| c44, c55 | 5.72, 5.17 | G_A = 5.445 (media) | verificado |
| c66 | 4.05 | G_T = 4.05 | verificado |
| c23 | 12.5 | C_TA = 12.5 | verificado |
| c13 | — | se supone = c23 | **no verificado** (no está en el resumen) |
| c12 | — | C_T − 2·G_T = 11.7 | derivado (isotropía transversa) |
| densidad | — | 1908 kg/m³ (tabla) | **no verificada** |

Comprobación: este tensor da módulos de Young de 21.0 GPa a lo largo del eje y
11.8 GPa perpendicular; el resumen da 20.9 y 11.8 GPa. No se usaron otros
conjuntos porque no se pudieron verificar sus cifras desde esta sesión (las
webs de las revistas, PubMed y HAL no eran accesibles): Ashman et al. (J
Biomech 17:349, 1984), Bossy, Talmant & Laugier (JASA 115:2314, 2004) y Lang
(1970) quedan sin comprobar; las de Dong & Guo (J Biomech 37:1281, 2004) son
cuasi-estáticas (menores que las ultrasónicas).

Velocidades resultantes (ρ = 1908 kg/m³): P axial 4024 m/s, P radial 3221
m/s, S en planos con el eje 1689 m/s, S en el plano transversal 1457 m/s
(la cortical isótropa de la tabla: 3500 y 1800 m/s).

**Orientación por corte** (`phantom.meta["view"]`, que ponen
`longitudinal_limb` y `cross_section_limb`):

- `"longitudinal"`: eje del hueso a lo largo de x, z radial:
  C11 = C_A, C33 = C_T, C13 = C_TA, C55 = G_A. Se ignora la inclinación del
  hueso (`bone_tilt_deg`, ~N(0, 2°) en los datasets).
- `"cross_section"`: el plano es el de isotropía transversa, el medio sigue
  siendo isótropo (C11 = C33 = C_T, C55 = G_T) y se usa el camino rápido con
  otras `cp`, `cs`.
- Otro fantoma sin `meta["view"]`: error explícito.

**Porosidad (modo realista).** Cada celda porosa ya tiene su medio efectivo
isótropo (Hashin-Shtrikman 2D, `materials.porous`). Se le suma la desviación
anisótropa de la tabla (tensor publicado menos la cortical isótropa)
multiplicada por μ_efectiva/μ_tabla. Así, en la porosidad de referencia (5 %,
la de la tabla) la celda toma las rigideces publicadas; un poro resuelto
(fluido) queda isótropo; la matriz sin poros es algo más rígida y anisótropa.
Como en el caso isótropo, la onda S de una celda porosa se suprime si su
velocidad más lenta (en cualquier dirección) baja de 6 puntos por longitud de
onda. Es una aproximación: los canales de Havers, alineados con el eje,
deberían reducir más la rigidez transversal que la axial, pero Cai et al. (J R
Soc Interface 16:20180911, 2019) encuentran que la porosidad explica casi toda
la variación y la forma y organización de los poros muy poco. Se comprobó que
el tensor es definido positivo en todas las celdas de una tibia realista.

La trabecular sigue isótropa (su anisotropía la da la red dibujada).

## Validación

### Velocidades de fase medidas frente a Christoffel

Medio homogéneo con las rigideces del corte longitudinal, f0 = 1 MHz,
dx = 0.1 mm (17 puntos por longitud de onda de la S más lenta). Onda plana con
la normal a θ del eje x: fuente de línea perpendicular a la normal (a 0° y 90°
la fuente ocupa una columna o fila y la simulación es 1D); velocidad por
correlación cruzada entre dos receptores a 8 y 18 mm de la línea, con ventana
en la llegada de cada onda. qSV con una fuerza perpendicular a la normal.

| θ | qP medida (m/s) | Christoffel | error | qSV medida | Christoffel | error |
|---|---|---|---|---|---|---|
| 0° (eje, sqrt(C11/ρ), sqrt(C55/ρ)) | 4027.2 | 4024.3 | +0.07 % | 1689.8 | 1689.3 | +0.03 % |
| 15° | 3971.9 | 3968.3 | +0.09 % | 1706.8 | 1706.3 | +0.03 % |
| 30° | 3817.1 | 3814.4 | +0.07 % | 1745.1 | 1744.9 | +0.01 % |
| 45° | 3607.1 | 3604.5 | +0.07 % | 1775.3 | 1774.2 | +0.07 % |
| 60° | 3404.7 | 3402.3 | +0.07 % | 1764.4 | 1763.5 | +0.05 % |
| 75° | 3269.4 | 3266.8 | +0.08 % | 1717.8 | 1717.3 | +0.03 % |
| 90° (radial, sqrt(C33/ρ), sqrt(C55/ρ)) | 3223.7 | 3221.4 | +0.07 % | 1689.8 | 1689.3 | +0.03 % |

Objetivos: ≤ 1 % en los ejes y ≤ 2 % para la qP a 45° → cumplidos (≤ 0.1 %).
En los tests rápidos (`tests/test_anisotropy.py`) se repiten los ejes (P y S,
≤ 1 %) y la qP a 45° (≤ 2 %), y que el camino anisótropo con rigideces
isótropas reproduce el isótropo (diferencia ≤ 1e-4 relativa, redondeo de
float32).

### Estabilidad y backends

`simulate_bmode`-equivalente de una emisión (0°) sobre una tibia con fractura
desplazada 1 mm en modo realista (poros, red trabecular), 24×22 mm a 3 MHz
(dx = 0.06 mm), Numba con 1 hilo:

| Cortical | L | Backend | Pasos | dt (ns) | Tiempo | Finita | Pico de presión | RF Numba vs NumPy |
|---|---|---|---|---|---|---|---|---|
| isótropa | 1 | Numba | 3880 | 8.610 | 9.2 s | sí | 1.85e5 Pa | |
| isótropa | 1 | NumPy | 3880 | 8.610 | 19.7 s | sí | 1.85e5 Pa | 0 (idénticas) |
| isótropa | 3 | Numba | 3963 | 8.430 | 15.0 s | sí | 1.87e5 Pa | |
| anisótropa | 1 | Numba | 3998 | 8.355 | 10.5 s | sí | 1.94e5 Pa | |
| anisótropa | 1 | NumPy | 3998 | 8.355 | 20.0 s | sí | 1.94e5 Pa | 0 (idénticas) |
| anisótropa | 3 | Numba | 4084 | 8.181 | 17.1 s | sí | 1.96e5 Pa | |

Corte transversal de pierna con fractura (modo realista, 1.5 MHz, dx = 0.12
mm, 833×833 celdas): estable con la cortical isótropa (9151 pasos, 96 s) y
con la del plano de isotropía (8412 pasos, 89 s: la cortical transversal es
más lenta, 3221 frente a 3500 m/s, y el paso es mayor).

La anisotropía cambia la RF de la tibia un 9 % (norma L2, L = 1).

### Coste

Tiempo por paso en la tibia anterior (Numba, 1 hilo, mínimo de 5 bloques):
iso 2.38 ms / aniso 3.01 ms con L = 1 (×1.27) y 3.74 / 4.45 ms con L = 3
(×1.19); el paso de tiempo es un 3 % menor (la cortical axial, 4024 m/s, es
más rápida que la isótropa). El camino isótropo no se encarece: con la
opción desactivada se midió 2.33 ms/paso frente a 2.28-2.34 ms antes del
cambio (L = 1) y 3.71-3.96 frente a 3.81-3.88 ms (L = 3), dentro del ruido
de la máquina compartida, y el resultado es idéntico bit a bit.

## Límites

- Sólo ejes alineados con la malla (sin C15/C35): la inclinación del hueso
  y la de los fragmentos no giran el tensor.
- c13 y la densidad de las muestras no están verificados (ver la tabla).
- El Q es el mismo en todas las direcciones; la atenuación de la cortical
  también es anisótropa en la realidad (no se modela).
- La regla de mezcla con la porosidad es aproximada (ver arriba).
- Activarla por defecto cambiaría todas las imágenes con hueso (velocidad
  radial de la cortical 3221 en lugar de 3500 m/s, axial 4024 m/s): lo
  decide el coordinador.
