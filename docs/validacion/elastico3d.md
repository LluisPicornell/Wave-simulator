# Solver elástico 3D y tibia tubular

Script: `benchmarks/elastic3d_validation.py` (partes `velocidades`,
`extrusion`, `green`, `reflexion`, `oblicua`, `estabilidad`, `rendimiento` y
`aplicacion`). Tests rápidos: `tests/test_elastic3d.py` (22 tests, ~25 s con
Numba y 1 hilo).

## Por qué

La comparativa (§2, "3D y anisotropía") señala como principal limitación del
ultrasonido de wavesim frente a SimSonic, Sim4Life y k-Wave que sólo es 2D: en
un corte la tibia es una placa (dos, en el corte longitudinal) y no un tubo,
una fuente puntual es una línea infinita y la difracción y las ondas guiadas
no son las de un tubo (Bossy, Talmant & Laugier, JASA 115:2314, 2004, usan un
FDTD 3D precisamente por eso).

## Qué se añadió

- `wavesim/ultrasound/elastic3d.py`: `ElasticSolver3D`, `ElasticMedium3D`,
  `elastic_medium_3d`, `Source3D`, `Receiver3D`, `point_source_pressure`.
  Velocidad-tensión en malla escalonada 3D (vx, vy, vz, sxx, syy, szz, sxy,
  sxz, syz; Graves 1996), 4.º orden en espacio, leapfrog, CPML en las seis
  caras (coeficientes de `core/pml.py::cpml_coefficients`; grosor por eje),
  fluidos con μ = 0 y media armónica de μ en los nodos de cizalla, atenuación
  GSLS con pesos por mecanismo (`attenuation.py`, ajuste por tejido idéntico
  al 2D), cortical transversalmente isótropa opcional con el eje del hueso en
  x (`CORTICAL_STIFFNESS`: C11 = C_A, C22 = C33 = C_T, C12 = C13 = C_TA,
  C23 = C_T − 2 G_T, C44 = G_T, C55 = C66 = G_A), fuentes de presión y de
  fuerza (fx, fy, fz) y receptores por celdas con la normalización del 2D.
  El paso de tiempo usa el mayor autovalor de la matriz de Christoffel
  discreta en el número de onda de Nyquist en las tres direcciones (en un
  medio isótropo, la condición habitual con √3). Se puede imponer `dt` (para
  comparar con el 2D).
- `wavesim/ultrasound/_numba_kernels3d.py`: dos barridos fusionados por paso
  (velocidades; deformaciones, tensiones y memorias), paralelos por planos z
  con `prange`, en float32, más dos kernels para la corrección de interfaz.
  Backend NumPy de referencia (más lento y con 9 buffers temporales) y la
  misma selección `backend="auto" | "numpy" | "numba"` que el 2D (reutiliza
  `ElasticSolver._resolve_backend`). Sin Numba funciona con NumPy.
- `wavesim/core/grid3d.py` (`Grid3D`, arrays `(nz, ny, nx)`) y
  `wavesim/phantoms/limb3d.py` (`Phantom3D`, `extrude`, `leg_segment`,
  `LegSegmentSpec`, `TransverseFracture3D`), descritos abajo.
- No se tocó el solver 2D.

### Corrección de interfaz fluido-sólido en 3D

Es el principio del commit f23a3cf del solver 2D: en la deformación de
cizalla ∂v_c/∂s de un nodo del sólido, el estencil de 4.º orden lee con peso
C2 la velocidad v_c en s − 1 y s + 2; si ese nodo de velocidad toca una celda
fluida (que desliza), se sustituye por el nodo interior (s y s + 1), y el
cambio traspuesto se aplica a la ecuación del movimiento (D_b = −D_fᵀ) para
que el esquema siga conservando la energía. En 3D hay seis reglas (∂vx/∂y y
∂vy/∂x en sxy, ∂vx/∂z y ∂vz/∂x en sxz, ∂vy/∂z y ∂vz/∂y en syz). Como en 2D,
no se aplica en la franja de `npml + 3` celdas de cada borde (la CPML y los
bordes que no se escriben); en un eje sin CPML sólo hace falta si es el del
estencil, de modo que un medio extruido en y sin CPML en y recibe en cada
plano exactamente los términos del 2D. Con Numba se aplica después de cada
barrido (las actualizaciones son lineales en cada derivada).

Comprobaciones: la adjunción ⟨s, D_f v⟩ = −⟨v, D_b s⟩ se cumple a 1e-12 en
las seis reglas sobre una esfera de hueso con contorno escalonado (test); un
medio extruido con la corrección reproduce el solver 2D **corregido**
(aa3145e, rama del commit f23a3cf) a ~1e-6 en 1200 pasos (presión, vx, vz
en 5 receptores; medido con una copia de ese solver fuera del repositorio:
cuando se integre, `test_medio_extruido_con_fuente_uniforme_en_y_reproduce_el_2d`
lo comprueba solo); y la reflexión oblicua de la tabla de abajo.

### Un defecto de redondeo encontrado por el camino

En el camino anisótropo, las tres tensiones normales se sumaban en órdenes
distintos (sxx = C11 exx + C12 (eyy + ezz), syy = … ). En un fluido los
cuatro coeficientes son iguales, pero el redondeo float32 distinto dejaba una
pequeña tensión desviadora, que en un fluido no tiene fuerza recuperadora y
se acumulaba linealmente: frente a float64, el error crecía hasta 1e-3 del
pico tras 640 pasos. Ahora se suman siempre en el orden exx, eyy, ezz y el
error es el del camino isótropo (2e-14 absoluto, constante). El 2D no lo
tiene (sus dos tensiones normales ya se suman en el mismo orden).

## Fantomas 3D (`wavesim/phantoms/limb3d.py`)

- `extrude(phantom2d, ny=… | width=…)`: repite el corte a lo largo de y.
  Rechaza el modo realista (porosidad, trabéculas): es una microestructura
  2D que extruida serían láminas. Se pierde el *speckle* (el 3D no lo usa).
- `leg_segment(spec, fracture, length, dx, width, depth, bone_length)`:
  cilindro hueco de cortical con médula, eje en x; radio externo y grosor de
  `limb.CROSS_PRESETS` (tibia: 11 y 4 mm), piel y grasa de `limb.PRESETS`
  (1.5 y 3 mm) y un músculo concéntrico (2 mm por defecto), en agua. Fractura
  transversal (`TransverseFracture3D`): hueco de anchura `gap` lleno de
  hematoma, completo o en un arco (`arc_deg`, centrado hacia la sonda), con
  máscaras "fracture", "hematoma" y "cortical". `slice_y(0)` da el corte
  longitudinal 2D equivalente (dos placas de cortical separadas por la
  médula). Las capas concéntricas son una simplificación (en la cara
  anteromedial la tibia está casi bajo la piel; por detrás hay varios cm de
  músculo). Sin rugosidad ni modo realista.

## Validación

Medidas con Numba (float32), 1 hilo.

**Velocidades** (ondas planas, 10 ppw de cada onda): P en agua a lo largo de
x, y, z 1504.1 m/s (+0.28 %); P cortical 3509.6 (+0.28 %); S cortical con
las cuatro polarizaciones/direcciones probadas 1799.9 (−0.01 %); anisótropa:
P axial 4040.7 frente a 4024 (+0.41 %), P transversal 3228.6 frente a 3221
(+0.22 %), S (G_A) 1689.0 (−0.02 %), S (G_T) 1456.4 frente a 1457 (−0.04 %).
Tolerancia pedida ≤ 1 %: cumplida.

**Equivalencia con el 2D** (tibia con fractura desplazada extruida en y,
fuente uniforme en y, mismo dt, 1500 pasos, 2 mecanismos): diferencia máxima
de las trazas normalizadas 2.4e-6 (p, receptores en fluido), 5.0e-6 (vx),
1.8e-6 (vz) sin corrección frente al 2D de esta rama (sin ella); con la
corrección frente al 2D corregido (aa3145e), ~1e-6. Con la corrección frente
al 2D sin ella la diferencia es del 10 % (es lo que corrige). Pedido ≤ 2-3 %.

**Fuente puntual en agua** frente a p = dx² s'(t − r/c)/(2π r c):

| ppw | r/λ | amplitud/analítica | velocidad aparente | error rms | error rms sin el retardo |
|---|---|---|---|---|---|
| 10 | 1-5 | 0.986-1.003 | +0.29 % | 2.1-10.1 % | 0.6-1.9 % |
| 15 | 1-5 | 0.995-1.001 | +0.16 % | 1.1-5.6 % | 0.3-1.4 % |

Exponente del decaimiento: −1.0036 (10 ppw) y −1.0002 (15 ppw). El error de
forma de onda crece con r por la dispersión numérica (la onda va un 0.3 %
rápida a 10 ppw).

**Reflexión normal agua-cortical** (|R| en f0): isótropa 0.6398 / 0.6357 /
0.6345 / 0.6337 a 8 / 12 / 16 / 24 ppw frente a 0.6332 (+1.04, +0.40,
+0.21, +0.09 %: segundo orden); anisótropa a 12 ppw +0.39 % tanto en la
dirección transversal (R = 0.6077) como en la axial (0.6731).

**Reflexión oblicua a 45°** (haz gaussiano, 12 ppw) en los planos x-z, y-z y
x-y (cada uno usa una componente de cizalla distinta): error máximo de |R|
en 43.5-46.5° del 3.55 % sin la corrección de interfaz y 0.79 % con ella,
idéntico en los tres planos (el 2D corregido da 3.6 → 0.8 %).

**Estabilidad** (tubo pequeño con fractura incompleta, 30 000 pasos):

| cortical | atenuación | tubo en x | rms(v) final / mínimo tardío |
|---|---|---|---|
| isótropa | no | entra en la CPML | 2.5 (crece despacio) |
| isótropa | no | termina antes (`bone_length`) | 1.1 (decae) |
| isótropa | sí | entra en la CPML | 1.3 (decae) |
| anisótropa | no | entra en la CPML | 3·10⁶ (**inestable**) |
| anisótropa | no | termina antes | 1.0 (decae) |
| anisótropa | sí | entra en la CPML | 1.2 (decae) |

Como el 2D, una cortical sólida que atraviesa la CPML crece sin atenuación;
anisótropa, de forma exponencial (duplica cada ~1500 pasos; la inestabilidad
conocida de la PML en medios anisótropos). Con la atenuación por defecto no
se observa en 3·10⁴ pasos. Recomendación: con cortical anisótropa o sin
atenuación, usar `leg_segment(..., bone_length=...)` para que el hueso
termine antes de la capa.


## Memoria y tiempo

Caso objetivo 40 × 24 × 24 mm (tibia cortada por el dominio), 1 mecanismo,
CPML de 10 celdas, máquina de 4 núcleos compartida con otros procesos:

| f0 | dx | Mceldas | memoria | pasos/s 1 hilo | pasos/s 4 hilos | 30 µs con 1 hilo |
|---|---|---|---|---|---|---|
| 0.5 MHz | 0.30 mm | 0.85 | 124 MiB | 30.8 | 79.8 | 0.4 min |
| 1 MHz | 0.20 mm | 2.88 | 405 MiB | 9.6 | 6.6 (*) | 2.1 min |
| 1 MHz | 0.15 mm | 6.84 | 941 MiB | 4.0 | 5.6 (*) | 6.8 min |

Memoria: ~145-155 B por celda en total, es decir **~150 MB por millón de
celdas** (112 B/celda del solver: 9 campos, 3 densidades inversas, 5 módulos,
6 memorias y 5 coeficientes de memoria; 20 B del medio; 8-16 B de la CPML; los
términos de interfaz, el resto). El pico de la construcción añade ~60 %
(temporales en float64). Velocidad con 1 hilo: ~27 Mceldas·paso/s. (*) Con
4 hilos la escala fue irregular (2.6× a 0.5 MHz, peor que 1 hilo a 1 MHz):
la máquina no estaba libre; con 2 hilos y carga baja se midió 1.7×. La
corrección de interfaz cuesta ~20 % del paso en un tubo.


## Caso de aplicación: transmisión axial en la tibia tubular

Versión reducida (sólo 0.5 MHz; la de 1 MHz quedó sin terminar por
presupuesto). Tibia de los presets (R = 11 mm, cortical 4 mm) con piel y
grasa del tipo "delgado" (1.35 mm cada una) y sin músculo encima (cara
anteromedial; con capas más gruesas el primer arribo no se separa de la onda
directa antes de ~2 cm), agua 0.6 mm. Emisor: tira de 3 mm en y a x = −17 mm;
receptores puntuales a d = 8-35 mm; pulso de ancho de banda 100 %; fractura
transversal de 1 mm (hematoma) a d = 25 mm, completa o en un arco de 120°
hacia la sonda. 2D: el corte y = 0 del mismo fantoma (placa) con fuente
puntual (línea infinita en 3D). FAS: cruce del 25 % del máximo en una
ventana que acaba 3σ antes de la onda directa (sólo válido desde d ≈ 20-22 mm).
Cada 3D: 1.84 Mceldas, 2-3 min con 2 hilos.

| | 3D (tubo) | 2D (placa) |
|---|---|---|
| v_FAS sano, d = 22(20)-35 mm | 3518 m/s (3525-3527 por tramos) | 3508 m/s (3461-3574) |
| decaimiento A_FAS ∝ d^n | n = −2.97 | n = −1.87 |
| ΔA tras la fractura completa, media / peor | −2.8 / −7.0 dB | −3.5 / −7.2 dB |
| retardo tras la fractura | +296 ns | +285 ns |
| fractura incompleta 120° | igual que completa | igual |

Lectura honesta:
- A 0.5 MHz y 4 mm de cortical (e/λ ≈ 0.57) la velocidad del FAS del tubo y
  de la placa coinciden (diferencia < 0.5 %, dentro de la dispersión de la
  medida, ±50 m/s) y es la de la onda P del hueso. Es coherente con que el
  FAS sea una onda lateral local; no pude verificar en esta sesión si Bossy
  et al. (2004) afirman que tubos y placas dan la misma velocidad **[NV]**.
  Sí verifiqué (resumen) que la anisotropía reduce a la mitad el rango de
  velocidades y que la dependencia con el grosor aparece por debajo de
  0.5 λ (anisótropa) o λ (isótropa); el caso de aquí está por encima de 0.5 λ,
  así que no se espera dependencia del grosor. No se corrió el tubo
  anisótropo.
- Lo que sí cambia en 3D es la amplitud: el FAS decae como d^−3 (con la
  atenuación del hueso; una onda de cabeza 3D sin atenuación va como ~d^−2)
  frente a d^−1.9 en el corte 2D, porque la fuente 2D es una línea infinita.
  Comparar amplitudes absolutas entre 2D y medidas reales exige el 3D.
- La fractura de 1 mm baja el FAS de 3 a 7 dB y lo retrasa ~0.3 µs pasada la
  fractura, con un patrón de interferencia (ganancias de ~1 dB justo
  después); es la tendencia publicada (Dodd et al., Bone 40:656, 2007, y
  Humphrey/Dodd 2008: pérdida del primer arribo a través de la fractura,
  más sensible con huecos pequeños, verificado en resúmenes; cifras en dB
  **[NV]**). 3D y 2D dan pérdidas parecidas.
- La fractura incompleta (120°) da el mismo FAS que la completa: el primer
  arribo sólo ve la cortical bajo la sonda; las ondas que rodean el tubo
  llegan después (las trazas difieren desde t ≈ 18 µs, hasta un 3 % del
  máximo). Es un efecto que sólo puede estudiarse en 3D, pero el FAS no lo
  distingue.


## Limitaciones y pendientes

- Rendimiento CPU: ~27 Mceldas·paso/s por hilo; dominios de pocos cm a
  ≤ 1 MHz. A 3 MHz (B-mode) un volumen útil necesitaría GPU.
- Cortical anisótropa dentro de la CPML: inestable sin atenuación (ver
  arriba); usar `bone_length`.
- Sin speckle, porosidad, rugosidad ni modo realista en 3D; capas
  concéntricas.
- Caso de aplicación sólo a 0.5 MHz, isótropo y con un criterio de FAS
  concreto; falta 1 MHz y el tubo anisótropo (el benchmark admite
  `part_aplicacion(freqs=((0.5e6, 0.3e-3), (1e6, 0.2e-3)))`).
- El escalado con 4 hilos no se pudo medir con la máquina libre.

