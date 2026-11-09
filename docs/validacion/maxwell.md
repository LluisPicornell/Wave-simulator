# Validación del solver de Maxwell (FDTD 2D, modo TM)

Cierra tres brechas de `wavesim.em` frente a gprMax, Meep y openEMS
(`docs/comparativa_simuladores.md`, §3): **tejidos dispersivos de banda
ancha**, **validación analítica del campo** (capas planas y cilindro) y
**promedio sub-celda** de las interfaces. Todas las cifras de este documento
salen de `benchmarks/em_validation.py` (NumPy, 1 hilo, CPU compartida con
otros procesos; los tiempos son orientativos) y las versiones reducidas están
en `tests/test_em_dispersive.py`.

El solver de banda estrecha (`MaxwellTMSolver`) sigue siendo el de siempre:
la refactorización (malla de Yee común en `YeeTMGrid`) da resultados
idénticos bit a bit con los parámetros por defecto (comprobado sobre un
antebrazo a 915 MHz con la versión anterior del fichero).

## 1. Qué hay nuevo

| Pieza | Módulo | Qué hace |
|---|---|---|
| Ajuste de Debye | `materials/dispersive.py` | Ajusta el Cole-Cole de Gabriel (con la conductividad iónica) a `eps_inf + sum d_eps_k/(1 + j w tau_k) + sigma/(j w eps0)` en la banda pedida; polos comunes a todos los tejidos; mapas por celda (también porosas) |
| FDTD de banda ancha | `em/fdtd_dispersive.py` | `BroadbandMaxwellTMSolver`: ecuaciones diferenciales auxiliares (ADE) de Debye, pulso de banda ancha y DFT a muchas frecuencias en una sola simulación |
| Funciones de Bessel | `em/bessel.py` | `J_n(z)` para `z` complejo, `Y_n(x)` y `H_n^(2)(x)` de orden entero, sin SciPy |
| Soluciones exactas | `em/analytic.py` | Matriz de transferencia general (`r`, `t` complejos) y serie de Mie 2D TM del cilindro |
| Promedio sub-celda | `em/subpixel.py` | Media de la permitividad por área en las celdas partidas por una interfaz |

Uso mínimo:

```python
from wavesim.em import BroadbandMaxwellTMSolver
from wavesim.phantoms import cross_section_limb

p = cross_section_limb("antebrazo", dx=0.5e-3, margin=30e-3, realism="realista")
s = BroadbandMaxwellTMSolver(p, band=(0.5e9, 5e9))   # ajuste de Debye automático (polos comunes)
s.set_plane_wave_source()                            # en el agua, por encima del fantoma
r = s.run(freqs=[0.915e9, 2.45e9, 4e9])              # un solo pulso
em_245 = r.at(1)                                     # EMResult a 2.45 GHz (campo, eps', sigma, SAR)
```

(240x240 celdas, 4 polos, 74 384 pasos, ~100 s en un núcleo.)

## 2. Método

### 2.1 Ajuste de Debye multipolo

Con los tiempos de relajación `tau_k` fijos, `eps'(w)` y la conductividad
efectiva `sigma_eff(w) = w eps0 eps''` son **lineales** en
`x = (eps_inf - 1, sigma, d_eps_1..K)`. El ajuste minimiza el error
*relativo* de `eps'` y de `sigma_eff` en 60 frecuencias log-espaciadas de la
banda con mínimos cuadrados no negativos (NNLS de Lawson y Hanson,
implementado en NumPy). `x >= 0` garantiza un medio pasivo (`eps_inf >= 1`,
amplitudes y conductividad no negativas), que es lo que hace estable el ADE.

Los `tau_k` se optimizan una sola vez para **todos** los tejidos del
registro a la vez (proyección de variables: Levenberg-Marquardt sobre
`log tau` con el NNLS dentro), partiendo de polos log-espaciados entre
`f_min/3` y `3 f_max`. Que los polos sean comunes tiene tres ventajas: el
FDTD guarda un mapa por polo (no uno por tejido y polo); cualquier mezcla
lineal de celdas (el promedio sub-celda, la interpolación entre porosidades)
sigue siendo un Debye con amplitudes no negativas; y el modelo no depende del
fantoma (se calcula una vez por banda, 1-5 s). El número de polos por
defecto es el menor que deja un error < 1 % en todos los tejidos.

En la optimización de los polos entran también cuatro mezclas porosas
(cortical con un 50 y un 80 % de poros de sangre o de médula): los poros
conductores en una matriz poco conductora crean una relajación interfacial
de Maxwell-Wagner en la banda de las microondas que los polos de los tejidos
puros no cubrían (sin ellas, la cortical con un 80 % de poros quedaba con un
1.8 % de error).

No hizo falta ningún polo de Lorentz: entre 10 MHz y 20 GHz todos los
tejidos de Gabriel son relajaciones (Cole-Cole con `alpha <= 0.25`), sin
resonancias, y bastan 4-8 Debye.

**Celdas porosas.** Con mapa de porosidad, cada celda de cortical o
trabecular toma la mezcla de Maxwell Garnett de `materials.porous` (la misma
que usa el modo de banda estrecha). Esa mezcla se ajusta con los polos
comunes en una tabla de porosidades con paso 0.01 y cada celda interpola
linealmente las amplitudes entre los dos niveles vecinos. Celdas con
porosidad `NaN` toman el ajuste de la tabla del tejido, como en
`Phantom.property_map`.

### 2.2 ADE en el FDTD

Cada polo añade una corriente de polarización `J_k = dP_k/dt` con
`tau_k dJ_k/dt + J_k = eps0 d_eps_k dE/dt`. Discretizada en `n + 1/2`
(Taflove y Hagness, *Computational Electrodynamics*, 3.ª ed., cap. 9;
forma con `J` de Okoniewski, Mrozowski y Stuchly, IEEE MGWL 7:121, 1997):

    J_k^{n+1} = kappa_k J_k^n + beta_k (E^{n+1} - E^n)
    kappa_k = (2 tau_k - dt)/(2 tau_k + dt),   beta_k = 2 eps0 d_eps_k/(2 tau_k + dt)
    E^{n+1} = ca E^n + cb (rot H - sum_k (1 + kappa_k)/2 J_k^n)

con `ca`, `cb` que incluyen `eps0 eps_inf/dt`, `sigma/2` y `sum beta_k/2`.
Sin polos las ecuaciones son exactamente las de banda estrecha (un test
compara el pulso con la onda continua en una losa con `eps` y `sigma`
constantes: coinciden a 0.2 % de mediana). La CPML sólo toca las derivadas espaciales, así que no
depende del material; su término CFS se fija con `f_ref = f_min` para que
absorba toda la banda. El paso de tiempo es el mismo que en el vacío
(CFL 0.95 del 2D).

**Fuente.** Lámina de corriente con un seno modulado por una gaussiana
centrada en la banda, cuyo espectro baja 20 dB en `f_min` y `f_max` (unos
-35 dB en 0.1 GHz para la banda 0.1-10 GHz) y es despreciable por encima de
~1.5 `f_max`. El seno no tiene componente continua: no deja carga ni
polarización estática.

**Fasores.** La DFT se acumula durante la simulación en las celdas del
monitor (todas o una máscara) por bloques de 64 muestras, con dos productos
de matrices reales por bloque y una muestra cada `stride` pasos (≥ 6 por
periodo de 1.5 `f_max`). El fasor se divide por el espectro del pulso
inyectado y por `eta(w)/eta_inf` de la celda de la fuente, así que vale
«V/m por V/m incidente en el plano de la fuente» a cada frecuencia, con el
mismo convenio de fase que el solver de banda estrecha. Por eso la fuente
debe estar en una zona homogénea (agua o aire); si cruza tejidos distintos,
el solver avisa.

**Parada.** Cuando, terminado el pulso, el máximo de `|Ey|` del monitor en un
bloque baja de `1e-6` veces el máximo alcanzado. En un medio conductor la
cola del campo decae como una potencia de `t` (las frecuencias por debajo de
la banda se difunden); con `1e-4` el error de truncar la DFT llegaba al 2 %
en beta y alpha a 0.1 GHz en músculo, con `1e-6` queda en 0.3 %.

### 2.3 Soluciones exactas

- **Capas planas**: matriz característica de cada capa
  `[[cos kd, j eta sin kd], [j sin kd/eta, cos kd]]` sobre `(E, H)`, medios
  de entrada y salida arbitrarios. El FDTD mide el cociente del campo detrás
  de la losa con y sin ella, que es `t e^{j k_0 D}`
  (`analytic.insertion_transmission`). Con el mismo medio a los dos lados
  `|t|^2` coincide con `spectrum._coherent_transmission` a 1e-10 (test).
- **Cilindro (Mie 2D, TM)**: `E_inc = e^{-jkz}`; dentro
  `sum eps_n (-j)^n c_n J_n(k_c rho) cos(n phi)`, fuera
  `E_inc + sum eps_n (-j)^n a_n H_n^(2)(k rho) cos(n phi)`, con `a_n` y `c_n` de
  la continuidad de `E_y` y de `dE_y/drho` en `rho = a` (`c_n` por el
  wronskiano, sin dividir por `J_n(k_c a)`). Comprobado: continuidad del
  campo en la superficie a 1e-9 y campo incidente exacto (1e-15) sin
  contraste.
- **Bessel**: `J_n(z) = (1/2pi) ∫ e^{j(z sin t - n t)} dt` con la regla del
  trapecio (espectralmente exacta por ser periódica; todos los órdenes a la
  vez con una FFT) para `n <= |z|`, y recurrencia hacia atrás de Miller,
  normalizada con esos órdenes, para `n > |z|` (el trapecio sólo tiene
  precisión absoluta y ahí `J_n` es diminuta). `Y_0`, `Y_1` con la
  representación integral DLMF 10.9.7 (Gauss-Legendre por tramos) y
  recurrencia hacia arriba.

### 2.4 Promedio sub-celda

En TM el campo eléctrico es perpendicular al plano y por tanto **tangente**
a toda interfaz; para la componente tangente la permitividad efectiva de una
celda partida es la media aritmética ponderada por área (Farjadpour et al.,
Opt. Lett. 31:2972, 2006; es el promedio de Meep para E paralelo). La media
de `eps*(w)` complejo vale para la banda estrecha y, como los polos son
comunes, para la ancha promediando `eps_inf`, `sigma` y `d_eps_k`. Las
fracciones de área salen de un fantoma auxiliar `factor` veces más fino
(impar, alineado con `subpixel.fine_grid`); aquí, 9.

## 3. Resultados

### 3.1 Ajuste de Debye frente a Gabriel (0.1-10 GHz)

6 polos comunes, `f_k = 1/(2 pi tau_k)` = 0.0385, 0.212, 0.849, 3.32,
9.00 y 20.9 GHz (ajuste en 2.1 s). Error relativo máximo en 400 frecuencias
de la banda (no sólo en las 60 del ajuste). Objetivo: ≤ 2-3 %.

| Tejido | `eps_inf` | `sigma` (S/m) | error máx. `eps'` | error máx. `sigma` |
|---|---|---|---|---|
| agua | 1.72 | 0.005 | 0.66 % | 0.35 % |
| piel | 5.14 | 0.201 | 0.32 % | 0.20 % |
| grasa | 3.00 | 0.030 | 0.25 % | 0.08 % |
| músculo | 9.59 | 0.612 | 0.30 % | 0.14 % |
| sangre / hematoma | 7.96 | 1.089 | 0.07 % | 0.08 % |
| hueso cortical | 3.39 | 0.042 | 0.16 % | 0.10 % |
| hueso trabecular | 4.32 | 0.121 | 0.18 % | 0.14 % |
| médula | 3.01 | 0.010 | 0.25 % | 0.27 % |
| edema | 4.98 | 0.442 | 0.09 % | 0.08 % |
| fascia / periostio | 4.74 | 0.406 | 0.19 % | 0.09 % |
| aire | 1.00 | 0 | 0 | 0 |

`sigma` es la conductividad estática del modelo (la iónica de Gabriel más
la cola de los polos de baja frecuencia que caen por debajo de la banda);
`eps_inf` no es la de Gabriel porque el polo de 20.9 GHz absorbe parte de
la relajación del agua. Fuera de la banda el modelo no se ajusta a Gabriel.

Celdas porosas: tabla de porosidad interpolada frente a la mezcla de Maxwell
Garnett exacta, en 42 porosidades (40 aleatorias en [0, 1], la de referencia
0.05 y 0.8):

| Tejido | relleno de los poros | error máx. `eps'` | error máx. `sigma` |
|---|---|---|---|
| cortical | sangre | 0.28 % | 0.67 % |
| cortical | médula | 0.25 % | 0.26 % |
| trabecular | (el de la tabla) | 0.19 % | 0.14 % |

Otras bandas (número de polos elegido automáticamente para < 1 %):

| Banda | polos | error máx. (todos los tejidos) | tiempo de ajuste |
|---|---|---|---|
| 0.1-3 GHz | 5 | 0.27 % | 1.0 s |
| 0.5-5 GHz | 4 | 0.22 % | 0.5 s |
| 1-20 GHz | 5 | 0.42 % | 1.1 s |
| 0.01-10 GHz | 8 | 0.66 % | 4.7 s |

En la pierna realista del test (`realism="realista"`: cortical porosa,
fascias, dx = 1 mm), los mapas de Debye reproducen el `eps' - j eps''` celda
a celda del modo de banda estrecha con una diferencia máxima de 0.07-0.15 %
a 0.15, 0.915 y 2.45 GHz y de 0.35 % a 8 GHz (el test exige < 2 %).

### 3.2 Propagación en medio homogéneo: beta(f) y alpha(f)

Un solo pulso de 0.1-10 GHz en músculo y en grasa homogéneos, 20 puntos por
longitud de onda a 10 GHz (dx = 0.219 mm en músculo, 0.690 mm en grasa).
Constante de fase y atenuación ajustadas a lo largo de z a 21 frecuencias,
frente a las del Cole-Cole de Gabriel (no frente al Debye: el error incluye
el del ajuste). Objetivo: ≤ 2 % en beta y ≤ 5 % en alpha.

| f (GHz) | músculo: beta | músculo: alpha | grasa: beta | grasa: alpha |
|---|---|---|---|---|
| 0.1 | -0.04 % | +0.31 % | -0.09 % | -0.29 % |
| 0.251 | +0.02 % | -0.03 % | -0.00 % | -0.04 % |
| 0.631 | +0.02 % | -0.04 % | +0.02 % | -0.01 % |
| 1 | +0.03 % | -0.02 % | +0.02 % | +0.01 % |
| 2.51 | +0.04 % | +0.10 % | +0.04 % | +0.10 % |
| 3.98 | +0.06 % | +0.15 % | +0.07 % | +0.18 % |
| 6.31 | +0.14 % | +0.46 % | +0.13 % | +0.48 % |
| 10 | +0.19 % | +1.64 % | +0.23 % | +1.38 % |
| **máximo (21 frecuencias)** | **0.19 %** | **1.64 %** | **0.23 %** | **1.38 %** |

Coste: músculo 36 587 pasos en 5.9 s, grasa 13 882 pasos en 1.4 s (malla
1D de 4 columnas). El error crece con la frecuencia por la dispersión
numérica de la malla (20 puntos por longitud de onda a 10 GHz); a 0.1 GHz
el error residual (0.3 %) es el de truncar la DFT: con un umbral de parada
de `1e-12` en vez de `1e-6` (112 589 pasos en vez de 36 587) baja a 0.01 %.

### 3.3 Losa multicapa a incidencia normal

Piel 2 mm / grasa 5 mm / músculo 15 mm / cortical 6 mm en aire. Coeficiente
de transmisión complejo (campo detrás de la losa entre el de una simulación
sin losa) frente a la matriz de transferencia exacta con Gabriel. Objetivo:
≤ 0.5 dB.

Banda estrecha (onda continua, `MaxwellTMSolver` por defecto):

| f | dx | \|T\| FDTD | \|T\| exacto | error \|T\| | error de fase |
|---|---|---|---|---|---|
| 128 MHz | 0.5 mm | -11.009 dB | -11.100 dB | +0.091 dB | -0.03° |
| 915 MHz | 0.5 mm | -9.198 dB | -9.227 dB | +0.029 dB | +0.08° |
| 2.45 GHz | 0.5 mm | -12.425 dB | -12.386 dB | -0.039 dB | -0.85° |
| 128 MHz | 0.25 mm | -11.096 dB | -11.100 dB | +0.004 dB | -0.73° |
| 915 MHz | 0.25 mm | -9.285 dB | -9.227 dB | -0.059 dB | +0.62° |
| 2.45 GHz | 0.25 mm | -12.281 dB | -12.386 dB | +0.106 dB | -0.40° |

Todas por debajo de 0.11 dB y 0.9°. Que el error no baje al refinar indica
que lo domina el régimen permanente de la onda continua (tiempo de asentado
por defecto), no la discretización.

Banda ancha 0.1-10 GHz, un pulso por malla (25 frecuencias). «Debye» compara
con la matriz de transferencia calculada con el propio ajuste, lo que separa
el error del FDTD del del ajuste:

| f (GHz) | \|T\| Gabriel | dx = 0.25 mm: error | fase | dx = 0.125 mm: error | fase | 0.125 mm frente a Debye |
|---|---|---|---|---|---|---|
| 0.1 | -10.81 dB | -0.001 dB | +0.01° | -0.001 dB | +0.01° | -0.000 dB |
| 0.316 | -12.57 dB | -0.002 dB | +0.00° | -0.002 dB | +0.00° | -0.000 dB |
| 1 | -8.59 dB | -0.001 dB | -0.07° | -0.000 dB | -0.06° | -0.000 dB |
| 1.78 | -10.29 dB | -0.001 dB | -0.12° | +0.002 dB | -0.07° | -0.001 dB |
| 3.16 | -16.77 dB | -0.035 dB | -0.35° | -0.011 dB | -0.14° | -0.008 dB |
| 5.62 | -34.19 dB | -0.137 dB | -1.14° | -0.020 dB | -0.22° | -0.039 dB |
| 10 | -59.74 dB | -0.882 dB | -3.28° | -0.314 dB | +0.79° | -0.184 dB |
| **máximo** | | **0.88 dB** | **3.3°** | **0.31 dB** | **0.8°** | **0.18 dB** |

Con dx = 0.25 mm (18 puntos por longitud de onda en el músculo a 10 GHz) el
objetivo de 0.5 dB se cumple hasta ~8 GHz (-0.48 dB a 8.25 GHz); en el
extremo, con 60 dB de pérdida total, 0.88 dB es un 1.5 % de error en la
atenuación acumulada. Con dx = 0.125 mm se cumple en toda la banda (0.31 dB).
Frente al propio Debye el error del FDTD baja de 0.75 a 0.18 dB al dividir
dx por dos: segundo orden. Tiempo: 5.6 s y 9.6 s (dos simulaciones 1D).

### 3.4 Funciones de Bessel

| Comprobación | Resultado |
|---|---|
| 14 valores tabulados (J_0, J_1, J_2, J_5, Y_0, Y_1, Y_2, Y_5 en x = 1, 5, 10; 10 decimales) | diferencia máx. 4.9e-11 (redondeo de la tabla) |
| Wronskiano `J_{n+1} Y_n - J_n Y_{n+1} = 2/(pi x)`, n = 0-40, x = 0.01-150 | error relativo máx. 9.5e-15 |
| `J_0(i) = I_0(1)`, `J_1(i) = i I_1(1)` | 1e-9 (los decimales de la tabla) |
| Serie de potencias de `J_12(z)` para z = 0.5+0.2j y 7.5-0.9j | 1e-11 relativo |

Fuente de los valores tabulados: Abramowitz y Stegun, *Handbook of
Mathematical Functions* (NBS AMS 55, 1964), tablas 9.1 y 9.2 (J_n, Y_n) y
9.8 (I_0, I_1). **[NV]**: la tabla no se pudo abrir desde esta sesión (las
webs con copias de las tablas no eran accesibles); los valores se escribieron
de memoria y el buscador sólo confirmó `Y_0(1) = 0.088257` y
`J_0(1) ≈ 0.7652`. Que 14 valores independientes coincidan a 5e-11 con una
implementación que no los usa, junto con el wronskiano a 1e-14, hace muy
improbable un error en ellos.

### 3.5 Cilindro dieléctrico en onda plana (Mie 2D, TM)

Onda plana TM sobre un cilindro de músculo y otro de hueso cortical (radio
20 mm) en aire a 2.45 GHz, solver de banda estrecha. Error L2 relativo del
campo complejo frente a la serie de Mie dentro del cilindro y fuera de él,
en un cuadro de lado `2 (a + lambda0/4)` (el campo se divide por el
incidente en el centro, medido en una simulación 1D sin cilindro). Puntos
por longitud de onda **dentro del cilindro**. «Sub-celda» = permitividad
promediada por área (§2.4, factor 9). Régimen permanente: 40 periodos más
tres tránsitos y 8 periodos de DFT. Objetivo: ≤ 3-5 % con 15-20 puntos por
longitud de onda.

Músculo (`eps = 52.7 - 12.8j`, `|k_c| a = 7.6`):

| puntos/lambda | dx | malla | escalera: dentro | escalera: fuera | sub-celda: dentro | sub-celda: fuera |
|---|---|---|---|---|---|---|
| 10 | 1.661 mm | 109x103 | 7.68 % | 1.55 % | 6.95 % | 0.79 % |
| 15 | 1.108 mm | 139x133 | 5.99 % | 0.78 % | **3.26 %** | 0.71 % |
| 20 | 0.831 mm | 169x163 | 1.73 % | 0.21 % | **1.74 %** | 0.20 % |
| 30 | 0.554 mm | 231x225 | 0.68 % | 0.43 % | 0.75 % | 0.47 % |
| 40 | 0.415 mm | 291x285 | 0.78 % | 0.09 % | 0.43 % | 0.08 % |

Hueso cortical (`eps = 11.4 - 2.9j`, `|k_c| a = 3.5`):

| puntos/lambda | dx | malla | escalera: dentro | escalera: fuera | sub-celda: dentro | sub-celda: fuera |
|---|---|---|---|---|---|---|
| 10 | 3.571 mm | 77x71 | 9.39 % | 2.64 % | 7.18 % | 2.25 % |
| 15 | 2.381 mm | 91x85 | 7.71 % | 2.32 % | **2.52 %** | 0.92 % |
| 20 | 1.785 mm | 105x99 | 10.73 % | 3.26 % | **1.74 %** | 0.82 % |
| 30 | 1.190 mm | 133x127 | 0.83 % | 0.76 % | 1.29 % | 0.83 % |
| 40 | 0.893 mm | 161x155 | 2.15 % | 0.98 % | 0.77 % | 0.76 % |

Pendientes log-log del error dentro frente a dx (ajuste a los 5 puntos):
músculo 1.9 en escalera y 2.0 con sub-celda; hueso 1.5 y 1.5. La escalera
no converge de forma monótona: su error depende de cómo cae la circunferencia
en la malla (el radio de 20 mm son 11.2 celdas en el hueso a 20 puntos/lambda
y da un 10.7 %, peor que a 15). Con el promedio sub-celda el error baja de
forma regular y cumple el objetivo con 15 puntos por longitud de onda (3.3 %
y 2.5 %) y con 20 (1.7 %). Sin él, 15-20 puntos no bastan siempre (6.0 %,
7.7 %, 10.7 %).

Fuera del cilindro en el hueso el error se queda en ~0.8 % a partir de 15
puntos/lambda con los dos tratamientos. No es el régimen permanente (igual
con 120 periodos de asentado), ni el grosor de la CPML (igual con 40
celdas), ni la distancia del cuadro a la CPML (0.71 % con 30 mm más de
margen); es mayor cerca del cilindro (1.3 % a menos de 10 mm de la
superficie) y no se ha identificado su origen.

Barrido del radio (músculo, 2.45 GHz, 15 puntos/lambda, error dentro):

| radio | 12 mm | 14 mm | 16 mm | 18 mm | 20 mm | 22 mm | media |
|---|---|---|---|---|---|---|---|
| escalera | 6.65 % | 2.19 % | 6.44 % | 6.38 % | 5.99 % | 5.89 % | 5.59 % |
| sub-celda | 2.64 % | 3.64 % | 2.48 % | 3.84 % | 3.26 % | 3.40 % | 3.21 % |

El promedio reduce el error medio un 43 % y su dispersión (de 2.2-6.7 % a
2.5-3.8 %); en un radio concreto (14 mm) la escalera cae justo bien y es
mejor. Lo que queda con sub-celda a 15 puntos/lambda es sobre todo
dispersión numérica dentro del cilindro, que es de segundo orden.

Cilindro de músculo (radio 15 mm) con el **solver de banda ancha**, un solo
pulso de 0.5-3 GHz, 20 puntos/lambda a 3 GHz (dx = 0.682 mm, 165x159
celdas, 30 697 pasos, 21 s), en escalera, frente a Mie con la permitividad
de Gabriel y con la del ajuste de Debye a cada frecuencia:

| f (GHz) | Mie con Gabriel: dentro | fuera | Mie con el Debye: dentro | fuera |
|---|---|---|---|---|
| 0.5 | 0.23 % | 0.19 % | 0.28 % | 0.24 % |
| 0.715 | 0.26 % | 0.13 % | 0.26 % | 0.13 % |
| 1.02 | 0.72 % | 0.34 % | 0.74 % | 0.36 % |
| 1.47 | 0.85 % | 0.19 % | 0.88 % | 0.19 % |
| 2.1 | 2.11 % | 0.24 % | 2.11 % | 0.24 % |
| 3 | 3.61 % | 0.30 % | 3.61 % | 0.29 % |

El error crece con la frecuencia porque baja la resolución, como en banda
estrecha; que sea igual con Gabriel y con el Debye confirma que el ajuste no
aporta error apreciable.

### 3.6 Fantoma realista: banda ancha frente a banda estrecha

Antebrazo transversal `realism="realista"` (cortical porosa, fascias,
periostio), dx = 0.8 mm, fuente en el agua. A 2.45 GHz el fasor del pulso de
1-4 GHz y el de la onda continua difieren en una mediana de 0.02 dB (máximo
0.3 % del campo máximo) y el SAR máximo en un 0.7 %, una vez corregidos el
desfase de medio paso y la normalización con impedancia real de la fuente de
banda estrecha (§4). El test hace lo mismo a dx = 0.9 mm.

## 4. Límites y riesgos

- **2D TM.** Sigue sin haber 3D ni modo TE; las antenas reales, la
  adaptación de impedancia y el SAR en cuerpo entero siguen siendo terreno de
  openEMS, gprMax o Sim4Life.
- **Fuente de onda plana en banda ancha**: tiene que estar en una zona
  homogénea (el solver avisa si no). El de banda estrecha no lo exige porque
  normaliza celda a celda a una sola frecuencia.
- **Coste**: cada polo añade un mapa y ~4 operaciones por celda y paso
  (NumPy, sin Numba): en una malla de 200x200 el paso cuesta ~1.0 ms con 4-6
  polos frente a 0.43 ms en banda estrecha (~2.3 veces), y
  en tejidos conductores la cola difusiva obliga a simular más tiempo cuanto
  más baja es `f_min` (decenas de miles de pasos para 0.1 GHz). Con agua
  alrededor (poco absorbente y de permitividad alta) el campo tarda en bajar
  a `1e-6` y la simulación suele acabar en el tope `t_max` (en el ejemplo del
  §1, 74 384 pasos). Para una sola frecuencia la onda continua sigue siendo
  más barata.
- **Fuera de la banda** el modelo de Debye no reproduce a Gabriel (no se
  ajusta allí); el pulso apenas tiene energía fuera, pero los fasores sólo
  valen dentro de `band`.
- **Banda estrecha y fuentes con pérdidas**: el solver de banda estrecha
  normaliza la fuente con la impedancia real, así que en un medio con
  pérdidas emite `sqrt(eps'/eps)` V/m (en el agua a 2.45 GHz, 0.35 % en
  módulo y 3.4° de fase) y toma el seno medio paso tarde (`w dt/2` de fase). No se
  ha cambiado para no alterar resultados existentes; el de banda ancha
  normaliza con la impedancia compleja. Corregidos esos dos factores, en el
  antebrazo realista con la fuente en el agua los dos solvers coinciden a
  2.45 GHz con una diferencia mediana de 0.02 dB (test: < 0.5 % mediana,
  < 2 % máxima).
- **Régimen permanente de la onda continua**: el tiempo de asentado por
  defecto de `MaxwellTMSolver.run` deja ~0.5 % de error fuera del cilindro;
  el benchmark de Mie usa uno más largo (40 periodos más tres tránsitos y 8
  periodos de DFT).
- **Promedio sub-celda**: necesita un fantoma auxiliar más fino alineado con
  la malla (`subpixel.fine_grid`). Los generadores de fantomas no aceptan una
  malla dada, así que con ellos sólo es inmediato en geometrías propias (como
  el cilindro); integrarlo en `phantoms` queda pendiente.

## 5. Reproducir

```bash
NUMBA_NUM_THREADS=1 python benchmarks/em_validation.py            # todo
python benchmarks/em_validation.py --case fit|propagation|slab|mie
python -m pytest -q tests/test_em_dispersive.py                   # versión rápida (~40 s)
```
