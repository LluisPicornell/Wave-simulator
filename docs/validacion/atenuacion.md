# Atenuación de ley de potencia en el solver elástico

Script: `benchmarks/attenuation_power_law.py` (modelo, simulación y coste;
`--disenar` repite la optimización de los tiempos de relajación). Tests
rápidos: `tests/test_attenuation.py` y `tests/test_numba_backend.py`.

## Qué cambió

El GSLS (`wavesim/ultrasound/attenuation.py`) usaba el "método tau": un único
peso por celda compartido por todos los mecanismos y tiempos de relajación
fijos repartidos en f0/2.5-2.5·f0. Ahora:

- **Un peso por mecanismo** y tejido (o celda porosa): `ElasticMedium.tau_p`
  y `tau_s` son arrays `(L, nz, nx)`; un array `(nz, nx)` se sigue aceptando
  como método tau. Los dos backends (NumPy y Numba) usan coeficientes
  `(L, nz, nx)`.
- **Ajuste minimax** del error relativo de alpha(f) con pesos no negativos
  (NNLS exacto por enumeración de conjuntos activos y repesado de Lawson), y
  Q⁻¹ objetivo calculado con la velocidad de fase del propio modelo.
- **Tiempos de relajación optimizados** (`MECHANISM_DESIGN`) a la vez para
  y = 1.0 (Q alto y bajo), 1.1, 1.2, 1.5 y 2.0, con una penalización por el
  aumento del módulo no relajado de la cortical, que fija el paso de tiempo.
  La banda de ajuste es [0.5, 1.8]·f0 con 2 mecanismos y [0.3, 2.5]·f0 con 3.
- **Con 1 mecanismo nada cambia** (mismo tau_1 = 1/(2π f0) y el mismo ajuste
  por mínimos cuadrados): comprobado bit a bit en un fantoma básico y en uno
  realista con poros, con ambos backends. Un ajuste minimax con un mecanismo
  bajaría el error máximo del 13.8 al 11.2 %, pero subiría el de f0 del +7 al
  +11 % y el cuadrático ponderado por el espectro del pulso del 6 al 9 %: no
  es una mejora clara y no se aplicó.
- Kernel Numba: en las celdas sin cizalla (fluidos, la mayoría de la malla) no
  se leen los coeficientes de cizalla de las memorias; el resultado es el
  mismo y el paso con 2-3 mecanismos cuesta lo mismo o menos que antes.

## 1. Modelo: error máximo de alpha(f) frente a alpha0·f^y

Medio genérico c = 1540 m/s, alpha(1 MHz) = 1 dB/cm. Banda estrecha
[0.5, 1.8]·f0 / ancha [0.3, 2.5]·f0, en %.

| Método | L | y = 1.0 | y = 1.1 | y = 1.5 | y = 2.0 |
|---|---|---|---|---|---|
| tau (anterior) | 1 | 14.2 / 40.8 | 14.3 / 32.8 | 29.1 / 51.3 | 96.8 / 126 |
| tau (anterior) | 2 | 0.8 / 12.5 | 7.2 / 12.9 | 48.1 / 68.8 | 128 / 236 |
| tau (anterior) | 3 | 6.2 / 23.9 | 9.1 / 20.9 | 40.1 / 47.0 | 116 / 192 |
| nuevo | 1 | = tau L = 1 | | | |
| nuevo | 2 | **1.84** / 20.2 | **1.36** / 16.8 | **1.84** / 7.8 | **1.91** / 5.4 |
| nuevo | 3 | **1.61 / 1.6** | **1.57 / 1.6** | **1.59 / 1.6** | **0.86 / 0.9** |
| nuevo | 4 | 0.33 / 0.3 | 0.21 / 0.2 | 0.25 / 0.3 | 0.34 / 0.4 |

Objetivo: ≤ 2 % en la banda estrecha con 2-3 mecanismos → cumplido (peor
caso 1.91 % con L = 2, 1.61 % con L = 3); con L = 3 también ≤ 1.7 % en la banda
ancha. Tejidos de la tabla con L = 3: cortical P/S 1.57 %, trabecular 1.68 %,
músculo 1.61 %, sangre (y = 1.2) 1.67 %, agua (y = 2) 0.73 %, en ambas bandas.

Límite: con y = 2 **y** pérdidas muy altas (Q⁻¹(f0) ≈ 0.06, como la cortical
pero con exponente 2, que no tiene ningún tejido de la tabla) el error sube a
3.7 % (L = 2) y 2.9 % (L = 3).

**Paso de tiempo.** Con pesos no negativos el esquema es estable con el paso
del módulo no relajado. La cortical P fija el paso: respecto a L = 1, el paso
es 0.932 con L = 2 (el mecanismo a 9.2·f0 que necesita y = 2 rigidiza la
cortical un 11 %) y 0.982 con L = 3.

**Dispersión causal.** La velocidad de fase en f0 es la de la tabla (exacta).
Diferencia máxima entre c(f) del modelo y la relación de Kramers-Kronig de la
ley de potencia (Szabo 1994 / Waters 2005, `power_law_phase_speed`):

| Caso | L | Subida de c en la banda estrecha, modelo vs K-K (m/s) | máx \|c − c_KK\| estrecha / ancha (m/s) |
|---|---|---|---|
| cortical P (5 dB/cm/MHz, 3500 m/s) | 1 | +135.7 vs +91.3 | 25.9 / 27.0 |
| | 2 | +92.1 vs +91.3 | 9.0 / 15.7 |
| | 3 | +96.2 vs +91.3 | 4.8 (0.14 %) / 12.5 (0.36 %) |
| y = 1.1 (2 dB/cm, 1580 m/s) | 3 | +7.8 vs +7.4 | 0.4 / 1.1 |
| y = 1.5 (1 dB/cm, 1540 m/s) | 3 | +3.0 vs +2.8 | 0.3 / 0.7 |
| y = 2.0 (1 dB/cm, 1540 m/s) | 3 | +0.7 vs 0 | 0.5 / 1.2 |

Fuera de la banda la diferencia crece porque el modelo sólo sigue la ley en
la banda, y la relación de Kramers-Kronig depende de alpha en todas las
frecuencias.

## 2. Simulación: onda plana en medio homogéneo

Malla de 26 columnas con la fuente en todas: el campo no varía en x y la
simulación es 1D exacta (sin difracción). alpha(f) y c(f) salen del cociente
espectral entre dos receptores separados 15 mm, con pulso de ancho de banda
200 % y 20 puntos por longitud de onda a f0 = 1 MHz. Una simulación idéntica
sin pérdidas sirve de referencia: se resta su atenuación aparente (ventana
finita) y su velocidad corrige la dispersión numérica de la malla. Errores
máximos, banda estrecha | ancha:

| Caso | L | alpha vs ley (%) | alpha vs modelo (%) | c vs modelo (m/s) | c vs K-K (m/s) |
|---|---|---|---|---|---|
| P, y = 1.0 | 1 | 14.07 \| 39.6 | 0.63 \| 1.25 | 1.2 \| 1.2 | 2.0 \| 2.3 |
| | 2 | 2.06 \| 19.1 | 0.33 \| 1.10 | 0.7 \| 1.5 | 0.8 \| 2.9 |
| | 3 | 1.78 \| 1.78 | 0.34 \| 0.94 | 0.5 \| 1.9 | 0.5 \| 2.7 |
| P, y = 1.1 | 3 | 1.70 \| 1.87 | 0.30 \| 0.97 | 1.3 \| 1.3 | 1.2 \| 1.8 |
| P, y = 1.5 | 2 | 2.16 \| 8.4 | 0.36 \| 1.15 | 1.2 \| 1.2 | 1.2 \| 1.4 |
| | 3 | 1.73 \| 2.45 | 0.36 \| 0.87 | 1.2 \| 1.2 | 1.1 \| 1.1 |
| P, y = 2.0 | 1 | 97.4 \| 131 | 0.63 \| 1.22 | 0.6 \| 1.8 | 6.5 \| 10.7 |
| | 2 | 1.99 \| 5.05 | 0.51 \| 1.35 | 1.2 \| 1.2 | 1.8 \| 4.2 |
| | 3 | 1.08 \| 1.55 | 0.58 \| 1.40 | 1.2 \| 2.0 | 1.0 \| 1.0 |
| cortical P (3500 m/s) | 3 | 1.71 \| 2.19 | 0.32 \| 1.15 | 1.7 \| 2.2 | 3.6 \| 11.4 |
| cortical S (1800 m/s, sólo estrecha) | 3 | 1.66 | 0.12 | 0.2 | 2.6 |

(Tabla completa con L = 1, 2, 3 en todos los casos: salida del script.)

- La simulación reproduce el modelo a ≤ 0.6 % en la banda estrecha y ≤ 1.4 %
  en la ancha; la velocidad de fase, a ≤ 2 m/s (0.13 %) del modelo.
- Frente a la ley nominal: con L = 3, ≤ 1.8 % en la banda estrecha y ≤ 2.5 %
  en la ancha; con L = 2, ≤ 2.2 % en la estrecha. Con L = 1, 14 % (y = 1) a
  97 % (y = 2).
- El resto del error simulación-modelo crece con la frecuencia y con
  (y − 1): es la discretización temporal. Con Crank-Nicolson y salto de
  rana el esquema reproduce el modelo continuo a la frecuencia deformada
  ω̃ = (2/dt)·tan(ω dt/2), que cambia Q⁻¹ en ≈ (y − 1)·(ω dt)²/12 (+1.5 % a
  2.5·f0 con y = 2 y 20 puntos/λ en agua; menos en un fantoma con hueso,
  donde dt lo fija la cortical). No se precompensa.

Caso B de `benchmarks/kwave_comparison.py` (fuente puntual, c = 1580 m/s,
2 dB/(cm·MHz^1.1), sólo la parte de wavesim; k-wave-python no estaba
instalado en esta sesión). Error de alpha frente a la ley nominal:

| f (MHz) | 0.70 | 0.85 | 1.00 | 1.15 | 1.30 |
|---|---|---|---|---|---|
| L = 1 (igual que §10.3 de la comparativa) | +5.2 % | +8.1 % | +7.3 % | +4.5 % | +0.4 % |
| L = 2 (nuevo; antes +4.9 … −2.1 %) | +1.1 % | +0.4 % | −0.7 % | −1.4 % | −1.5 % |
| L = 3 (nuevo) | −1.4 % | −1.7 % | −1.4 % | −0.7 % | +0.2 % |

## 3. Coste

`simulate_bmode`, una emisión, tibia con fractura de 24×22 mm a 3 MHz
(dx = 0.06 mm, 367×400 celdas), Numba con 1 hilo (máquina compartida: ±10 %):

| L | Tiempo | Pasos | dt (ns) | ms/paso |
|---|---|---|---|---|
| 1 | 8.4 s (1.00×) | 3687 | 9.061 | 2.21 |
| 2 | 12.4 s (1.49×) | 3957 | 8.442 | 3.11 |
| 3 | 14.3 s (1.72×) | 3753 | 8.901 | 3.78 |

Por paso, con el mismo número de mecanismos, el kernel nuevo cuesta lo mismo o
algo menos que el anterior (L = 2: 3.67 frente a 3.75 ms; L = 3: 4.31 frente a
4.64 ms, medidos uno detrás de otro). L = 2 da más pasos que L = 3 por su paso
de tiempo menor.

## Recomendación (no aplicada)

- **`n_mechanisms=2` como valor por defecto**: el pulso de la ecografía (60 %
  de ancho de banda, −20 dB entre ~0.45 y ~1.55·f0) cae en la banda donde el
  error es < 2 % para cualquier exponente, a 1.5 veces el coste de L = 1
  (cuyo error es del 14 % en la cortical y el músculo).
- **`n_mechanisms=3`** para pulsos de banda ancha o espectroscopía (transmisión
  a baja frecuencia, estudios de dispersión): < 2 % en [0.3, 2.5]·f0, a 1.7
  veces el coste de L = 1.
- Cambiar el valor por defecto cambia todas las imágenes: la atenuación media
  en la banda del pulso baja unos puntos (L = 1 sobreestima ~7 % en f0).
