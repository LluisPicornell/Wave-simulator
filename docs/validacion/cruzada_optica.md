# Comparación cruzada de la luz en tejido: wavesim frente a MCML y MCX

Compara el Monte Carlo de fotones (`wavesim/em/montecarlo.py`) y la difusión (`wavesim/em/optics.py`) con dos
códigos de referencia ejecutados en esta máquina, a 800 nm con las propiedades de `materials/optical.py`.
Script: `benchmarks/cross_optica.py`. Test sin las herramientas: `tests/test_cross_optica.py`, que lee las trazas
de `benchmarks/data/cross_optica/*.npz` (0.5 MB).

## Herramientas

| herramienta | versión / origen | estado |
|---|---|---|
| **MCML** (Wang y Jacques, C) | espejo de GitHub `bipy/MCML`, commit `bd8fcb2`, "Version 1.2, 1993"; gcc `-O2`. Es un espejo, no verificado byte a byte contra OMLC | ejecutado |
| **MCXCL** | `pmcxcl 0.8.1` (pip, venv aparte) con **pocl** (dispositivo OpenCL `pthread` de CPU, vía `pyopencl` + `pocl-binary-distribution`) | ejecutado |

`pmcx` (CUDA) no se probó. `pmcxcl` encuentra pocl apuntando `OCL_ICD_VENDORS` a un `.icd` propio con la ruta absoluta
de `libpocl` (cómo, en la cabecera del script); no hizo falta `apt`.

MCML tiene un parche mínimo (`benchmarks/data/cross_optica/mcml.patch`), ajeno a la física: (i) `FindDataLine`
devolvía un puntero a un array local (segfault con gcc moderno), ahora `static`; (ii) la semilla venía de `time(NULL)`
(runs simultáneos repetían semilla), ahora de `MCML_SEED`. Limitaciones heredadas de MCML: `ran3` en `float` y
semilla de 15 bits (se usan 4 réplicas con semillas distintas).

## Casos y geometría

Músculo a 800 nm: μa = 47.83 m⁻¹, μs' = 851.6 m⁻¹, g = 0.9 (μs = 8516 m⁻¹). Grasa: μa = 29.90, μs' = 1341.7 m⁻¹, g = 0.9.
Haz fino en incidencia normal. Celda de wavesim y bin z de MCML: 0.1 mm (el recorrido libre medio es 0.117 mm).

| caso | geometría | n |
|---|---|---|
| a_semi / a_slab | músculo 5 cm ("semi-infinito", T < 1e-8) / 1 cm | 1 dentro y fuera |
| b_semi / b_slab | grasa 4 mm + músculo 46 mm / grasa 3 mm + músculo 7 mm | 1 |
| c_semi / c_slab | músculo 5 cm / 1 cm | 1.4 frente a aire |

**Conversión 3D → 2D extruido.** wavesim es invariante en `y`: un haz fino es una línea infinita y `R(x)` es flujo por
unidad de longitud de `x` y de línea. Los fotones sí se mueven en 3D. Por linealidad la línea es la superposición de
haces finos, así que `R_2D(x) = ∫ R_3D(√(x²+y²)) dy`, y lo mismo con T y fluencia. Con el `R(ρ)` constante a trozos de
MCML la integral es exacta (área de cada anillo dentro de la columna: `pesos_radial_a_linea`, comprobada con un disco
uniforme en el test) y se promedia en el ancho de la columna de wavesim. La fluencia integrada en `x`, `A_z/μa`, no
necesita conversión. **Comprobación independiente de la conversión**: MCX se ejecuta con la geometría 2D extruida
exacta (2 voxeles en `y` con contorno cíclico; con 1 voxel MCX pasa a modo 2D con dispersión plana) y no usa
conversión; coincide con MCML+conversión (tabla de abajo).

Convenciones: MCML deposita μa/μt en cada colisión; wavesim usa absorción continua con estimador de longitud de
traza (misma esperanza). MCX no resta la reflexión especular de entrada: sus salidas se multiplican por 1 − R_esp
analítica, exacto porque esa reflexión es independiente de la historia posterior. Origen de z: superficie. La
fuente de wavesim es `pencil` en la columna central, la malla mide 7.01 cm (los fotones laterales son ~1e-6) y los
perfiles se pliegan en ±x. La **difusión** representa el haz fino con una fuente puntual isótropa a 1/μs' bajo la
superficie (con su peso × (1 − R_esp)); no tiene fuente de haz fino propia.

Réplicas: MCML 4 × 4·10⁶ fotones por caso, MCX 4 × 10⁶ (losas) o 4 × 5·10⁵ (semi), wavesim 8 × 5·10⁵ (Numba, 1 hilo).
Errores estándar: desviación entre réplicas / √K (con K = 4 y 8 el propio error es poco fiable: |z| ≲ 3 es ruido).

## Totales (fracción incidente; media ± error estándar)

| caso | magnitud | MCML | MCX | wavesim MC | wavesim − MCML | difusión |
|---|---|---|---|---|---|---|
| a_semi | R difusa | 0.50149 ± 0.00011 | 0.50264 ± 0.00017 | 0.50209 ± 0.00027 | +0.00060 (+2.1 σ) | 0.52205 |
| a_semi | absorbida | 0.49851 ± 0.00011 | 0.49736 ± 0.00017 | 0.49790 ± 0.00027 | −0.00060 | 0.47795 |
| a_slab | R difusa | 0.50119 ± 0.00008 | 0.50208 ± 0.00020 | 0.50178 ± 0.00010 | +0.00059 (+4.8 σ) | 0.52166 |
| a_slab | absorbida | 0.47463 ± 0.00007 | 0.47400 ± 0.00020 | 0.47420 ± 0.00009 | −0.00044 (−3.8 σ) | 0.45381 |
| a_slab | T | 0.02418 ± 0.00001 | 0.02392 ± 0.00007 | 0.02403 ± 0.00002 | −0.00016 (−5.8 σ) | 0.02453 |
| b_semi | R difusa | 0.63747 ± 0.00007 | 0.63781 ± 0.00018 | 0.63787 ± 0.00012 | +0.00040 (+2.8 σ) | 0.64768 |
| b_slab | R difusa | 0.62861 ± 0.00007 | 0.62886 ± 0.00030 | 0.62936 ± 0.00025 | +0.00075 (+2.9 σ) | 0.63842 |
| b_slab | absorbida | 0.35052 ± 0.00008 | 0.35026 ± 0.00024 | 0.34979 ± 0.00022 | −0.00073 (−3.1 σ) | 0.34063 |
| b_slab | T | 0.02087 ± 0.00003 | 0.02087 ± 0.00006 | 0.02084 ± 0.00004 | −0.00003 | 0.02095 |
| c_semi | R difusa | 0.34063 ± 0.00003 | 0.34112 ± 0.00010 | 0.34108 ± 0.00014 | +0.00045 (+3.1 σ) | 0.34475 |
| c_semi | absorbida | 0.63160 ± 0.00003 | 0.63110 ± 0.00010 | 0.63114 ± 0.00014 | −0.00046 (−3.1 σ) | 0.62748 |
| c_slab | R difusa | 0.34058 ± 0.00005 | 0.34084 ± 0.00015 | 0.34088 ± 0.00019 | +0.00030 (+1.6 σ) | 0.34470 |
| c_slab | absorbida | 0.60972 ± 0.00006 | 0.60971 ± 0.00013 | 0.60956 ± 0.00018 | −0.00016 | 0.60651 |
| c_slab | T | 0.02192 ± 0.00001 | 0.02167 ± 0.00003 | 0.02178 ± 0.00004 | −0.00015 (−4.1 σ) | 0.02100 |

La reflexión especular (0.02778 con n = 1.4) coincide a 1e-8 en los tres. Los tres códigos de transporte concuerdan
en R, A y T a ≤ 0.0008 absolutos (≤ 0.15 % de R); wavesim y MCX quedan sistemáticamente ~0.0005 por encima de MCML en R
y por debajo en A, en los seis casos: MCML es la discrepante en el mismo sentido (por eso se ve ~3 σ), con una
diferencia de 0.1 % cuya causa no he aislado (candidatos: `ran3` en float, deposición discreta en MCML).
La difusión se desvía +2 a +4 % en R en n = 1 y +1 % con n = 1.4.

## Perfiles: error relativo respecto a MCML en la zona con señal fiable

Zona fiable = intervalos (de x en mm: 0-0.5, 0.5-1, 1-2, 2-3, 3-5, 5-8, 8-12, 12-18, 18-25, 25-35; de z: 1-10 mm
según el caso) donde el error estándar relativo de MCML ≤ 5 % (todos los intervalos lo cumplen). Cada celda:
máximo / rms del |error relativo| en %.

| caso | perfil | wavesim MC | MCX | difusión |
|---|---|---|---|---|
| a_semi | R(x) | 2.2 / 0.9 | 11.8 / 3.8 | 24.4 / 13.1 |
| a_semi | Φ integrada en z | 1.0 / 0.5 | 2.4 / 1.2 | 31.6 / 11.9 |
| a_slab | R(x) | 2.0 / 0.8 | 1.9 / 0.7 | 24.4 / 13.0 |
| a_slab | T(x) | 6.3 / 2.6 | 7.5 / 2.8 | 6.9 / 2.6 |
| a_slab | Φ integrada | 0.5 / 0.3 | 0.6 / 0.4 | 31.6 / 14.8 |
| b_semi | R(x) | 7.5 / 2.4 | 10.5 / 3.6 | 18.3 / 9.1 |
| b_semi | Φ integrada | 0.8 / 0.4 | 0.5 / 0.3 | 17.3 / 6.8 |
| b_slab | R(x) | 8.0 / 2.7 | 3.1 / 1.2 | 18.4 / 9.1 |
| b_slab | T(x) | 12.1 / 3.9 | 1.1 / 0.5 | 4.1 / 1.6 |
| b_slab | Φ integrada | 0.5 / 0.3 | 0.2 / 0.2 | 17.6 / 8.1 |
| c_semi | R(x) | 1.7 / 0.8 | 2.3 / 1.0 | 24.6 / 10.5 |
| c_semi | Φ integrada | 2.2 / 0.9 | 1.5 / 0.7 | 22.4 / 9.6 |
| c_slab | R(x) | 0.7 / 0.5 | 4.8 / 1.6 | 24.5 / 10.3 |
| c_slab | T(x) | 4.2 / 1.5 | 3.1 / 1.6 | 6.3 / 4.5 |
| c_slab | Φ integrada | 0.6 / 0.4 | 0.4 / 0.3 | 22.4 / 12.2 |

Todos los errores de wavesim por encima de 3 % (hasta 12 %) están en colas lejanas (18-35 mm en x, 25-50 mm en z), con
señal ≤ 1e-5 de lo incidente, es decir ruido de wavesim con 4·10⁶ fotones (|z| ≤ 3.1); en el resto de intervalos el
error es ≤ 2.5 %. El error de MCX en R(x) de
`a_semi` (11.8 %) es también de cola (25-35 mm, 5·10⁵ fotones). Las tablas completas (también fluencia 2D en x = 0,
4 y 12 mm) las imprime el script (`python benchmarks/cross_optica.py --photons 500000 --seeds 8`).

## Difusión

- Error grande **bajo la fuente**: R en 0-0.5 mm entre −18 y −25 %, fluencia en el eje (x = 0, z < 1 mm) −70 a −89 %,
  Φ integrada en z < 1 mm −17 a −32 %. Es la brecha que justifica el Monte Carlo. Sobreestima +12 a +20 % en 0.5-3 mm.
- Lejos de la fuente (≥ 5 mm de x o ≥ 2 mm de profundidad) queda en 1-7 %; las colas (> 18 mm) vuelven a quedar −2 a −9 % por debajo.
- T(x) de la losa: −2 a −9 % (≤ 3 % hasta 12 mm).
- **No está convergida en la malla**: con a_semi, R total = 0.550, 0.531, 0.522 para dx = 0.4, 0.2, 0.1 mm (convergencia de orden 1;
  extrapolando, ≈ 0.513, +2 % sobre MCML). Parte del error es la cuantización de la profundidad de la fuente a la
  celda más cercana a 1/μs'. Los valores de las tablas son con dx = 0.1 mm.
- Para esta comparación se extendió `optics.solve_diffusion`: `n_tissue` (índice del tejido en el contorno
  extrapolado; antes fijo en 1.4, por defecto sigue valiendo 1.4 y los resultados existentes no cambian), necesario para los casos
  de índice ajustado, y `DiffusionResult.reflected_profile` (exitancia por la cara de entrada, que no existía). Ambos
  tienen test (balance de energía, el valor por defecto no cambia).

## Lectura honesta

- No hay solución analítica para estos casos (ni el haz fino a 800 nm ni las dos capas), así que MCML y MCX son las
  referencias; se comprueban entre sí y con wavesim, y concuerdan a ≤ 0.15 % en totales y ≤ 2 % por intervalo (salvo colas).
- La parte de diferencia total (~0.1 % de R) con MCML no la he atribuido; MCML es la que se separa de las otras dos.
- MCML es un espejo de GitHub, no el fichero de OMLC; sólo se aplicaron los dos parches descritos.
- Se ha probado un solo escenario óptico (800 nm, g = 0.9, haz normal): no cubre g bajos, otras λ, fuentes isótropas
  ni resolución temporal.
- Los tests rápidos usan 8·10⁴ fotones con tolerancias de varios sigmas y del 3-8 %; son una red de seguridad,
  no reproducen la tabla de arriba.
