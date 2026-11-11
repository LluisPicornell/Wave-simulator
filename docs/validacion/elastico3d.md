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

PENDIENTE

## Memoria y tiempo

PENDIENTE

## Caso de aplicación: transmisión axial en la tibia tubular

PENDIENTE

## Limitaciones y pendientes

PENDIENTE
