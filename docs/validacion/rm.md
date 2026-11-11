# Validación del simulador de Bloch espacial de RM

Cierra la brecha principal de `wavesim.mri` frente a KomaMRI.jl, JEMRIS y
MRiLab (`docs/comparativa_simuladores.md`, §6). Antes la RM eran ecuaciones de
señal analíticas por tejido, un k-space recortado y ruido de Rice; el Bloch de
isocromatos (`bloch.py`) sólo servía para comprobar esas ecuaciones. Ahora hay
un **simulador de Bloch espacial** (`wavesim/mri/spatial.py`) que produce los
artefactos de verdad: desplazamiento químico de la grasa, distorsión y pérdida
de señal por la susceptibilidad de los tejidos, desfase intravóxel (T2*),
sensibilidad de bobina no uniforme y ruido en el k-space.

`mri_scan` sigue dando exactamente lo mismo por defecto (`engine="analytic"`);
el nuevo motor se elige con `mri_scan(..., engine="bloch")` o con
`spatial.bloch_scan`. Todas las cifras de este documento salen de
`benchmarks/mri_validation.py` (1 hilo, CPU compartida con otros procesos: los
tiempos son orientativos); las versiones reducidas están en
`tests/test_mri_bloch.py` (56 tests, ~1.5 s).

## 1. Qué hay nuevo

| Pieza | Módulo | Qué hace |
|---|---|---|
| Motor por eventos | `mri/spatial.py` | Isocromatos (`Isochromats`) y secuencia como lista de eventos (`RF` duro o con forma, `Free` con momento de gradiente, `ADC`, `Spoil`, `SpectralSat`); `simulate` aplica rotaciones y relajación exactas entre eventos |
| T2' lorentziano exacto | `mri/spatial.py` | Estados de desfase (grafos de fase con tiempos arbitrarios) en vez de muestrear la lorentziana con isocromatos |
| Secuencias cartesianas | `mri/spatial.py` | `cartesian_sequence`: SE (T1, T2, DP y DP/T2 con saturación grasa), IR (STIR) y GRE con TR, TE, TI, ángulo, FOV, matriz, ancho de banda por píxel y dirección de lectura; `acquire` (modos `"steady"` y `"events"`) y `reconstruct` (FFT) |
| Fantoma → isocromatos | `mri/spatial.py` | `phantom_isochromats`: agua y grasa de cada tejido en cada celda, mezcla porosa lineal de `materials/porous.py`, textura, campo B0 y bobina |
| Campo B0 de la susceptibilidad | `mri/susceptibility.py` | Mapa de χ por tejido y núcleo dipolar de Fourier en 2D; solución exacta del cilindro |
| Datos | `materials/mri.py` | `GAMMA_BAR`, `FAT_FRACTION` (antes en `sequences.py`, que la reexporta), `FAT_PEAKS` (1 o 6 picos), `SUSCEPTIBILITY_PPM`, `blood_susceptibility` |
| Kernel Numba | `mri/_numba_kernels.py` | Lectura fusionada (opcional, `backend="auto"/"numpy"/"numba"`) |
| Artefactos | `mri/spatial.py` | `BlochPhysics` (desplazamiento químico, modelo de grasa, susceptibilidad y orientación de B0, T2', bobina) y `SurfaceCoil` |

Uso mínimo:

```python
from wavesim.mri import mri_scan
from wavesim.mri.spatial import BlochPhysics, SurfaceCoil
from wavesim.phantoms import RingFracture, cross_section_limb

p = cross_section_limb("pierna", RingFracture(angle_deg=-10), dx=0.2e-3).replace_tissue("water", "air")
im = mri_scan(p, "STIR", engine="bloch", matrix=(128, 128), bandwidth=130.0,
              physics=BlochPhysics(b0_direction="z", fat_model="multipeak"))
im.magnitude, im.field_hz, im.fat_shift_px, im.timings   # imagen, ΔB0 (Hz), -1.67 píxeles, ~1 s
```

`examples/14_rm_artefactos.py` guarda en `outputs/14_rm_artefactos.png` la
pierna en T1, STIR, GRE y DP-FS sin y con artefactos, la diferencia en STIR y el
mapa de ΔB0.

## 2. Método

### 2.1 Isocromatos y eventos

Cada celda del fantoma aporta un isocromato en su centro por cada componente de
cada tejido presente en ella: el agua (fracción `1 - FAT_FRACTION`) y la grasa
(un pico a -3.4 ppm o seis picos). Las celdas porosas (cortical y trabecular con
mapa de porosidad) se descomponen con los pesos de la mezcla lineal de
`materials.porous`, que es la que ya usaba `mri_scan` para la RM; los pesos
pueden ser negativos (el contenido de los espacios trabeculares se deduce
restando la matriz), lo que es legítimo porque la señal es lineal en `M0`. La
magnetización de equilibrio se escala por `área de celda / área de vóxel`, de
modo que un tejido uniforme da su señal por unidad de PD. Un vóxel de 128×128
sobre la pierna (0.78 mm) contiene unas 15 celdas de 0.2 mm: el desfase entre
ellas por un campo no uniforme o por la mezcla agua-grasa sale solo al sumar.

La secuencia es una lista de eventos y la evolución entre ellos es exacta:

- **Pulso duro**: rotación instantánea alrededor de un eje transversal
  (`dM/dt = γ M × B`, γ > 0: 90°x lleva `Mz` a `+My`, como `bloch.rotate_x`).
- **Pulso con forma o de duración finita**: rotación exacta de cada tramo
  constante alrededor de su campo efectivo `(B1x, B1y, Δω/γ)`, que incluye el
  desvío de cada isocromato (grasa, ΔB0, gradientes durante el pulso). Se
  comprueba con la fórmula de Rabi de un pulso rectangular fuera de
  resonancia (1e-12). La relajación y el desfase microscópico durante el
  pulso se aplican medio antes y medio después (error `O((d/T2)²)`).
- **Precesión libre**: fase `Δω t + γ (A_x x + A_z z)` y dos exponenciales. Un
  lóbulo de gradiente sólo entra por su área: la rotación en z y la
  relajación conmutan, así que el estado final no depende de la forma del
  lóbulo y no hace falta integrar paso a paso.
- **Lectura** con gradiente constante: la muestra `k` es `Σ c_j M_j z_j^k`
  con `z_j = exp(-(i ω_j + 1/T2) dt)`.
- **Spoil**: destrucción ideal de la coherencia (la hipótesis de las
  ecuaciones de `sequences.py`); **SpectralSat**: pulso espectral ideal que
  gira sólo los isocromatos cuya frecuencia (desplazamiento químico + ΔB0)
  cae en la banda.

### 2.2 T2' lorentziano exacto (estados de desfase)

La dispersión microscópica de frecuencias de cada celda (estructura no
resuelta) tiene en la tabla una forma lorentziana de semianchura `1/T2'`. Muestrearla
con isocromatos converge muy mal: su FID `exp(-|t|/T2')` tiene un pico en
`t = 0` que una suma finita de cosenos no reproduce, y con 64 o 256 cuantiles
el error en GRE es del 3-6 % a TE cortos y crece sin límite a TE largos
(tabla (c)). En vez de eso, la magnetización de cada isocromato en función del
desvío microscópico `δ` se guarda como suma finita de estados,
`M_xy(δ) = Σ_τ F_τ e^{-iδτ}` y `Mz(δ) = Σ_τ Z_τ e^{-iδτ}`: la precesión
desplaza `τ` de los estados transversales, un pulso mezcla el estado `τ` con el
`-τ` (`M- = conj M+` tiene coeficientes `conj F_{-τ}`; así reenfoca un 180°) y
la media lorentziana de cada término es exactamente `F_τ e^{-|τ|/T2'}`. Es la
teoría de grafos de fase extendidos con tiempos arbitrarios. Con pulsos
ideales sólo hay 1-3 estados vivos; con B1 no uniforme aparecen los ecos
estimulados, también exactos.

### 2.3 Estado estacionario exacto y k-space en una pasada

Con `Spoil` al final del TR, el Mz al principio de un TR es función afín del
anterior, `Mz' = A Mz + B`. Dos simulaciones del TR (con `Mz = 0` y `Mz = 1`)
dan `A` y `B`, y el punto fijo `B/(1 - A)` es el estado estacionario exacto,
sin TR de preparación. Además, la codificación de fase se aplica después del
último pulso, así que todas las líneas parten del mismo estado salvo la fase
`exp(-i k_pe q_j)`: la lectura se calcula **una vez** por isocromato y se suma
por filas de la coordenada de codificación de fase; el k-space completo es el
producto de la matriz `exp(-i k_pe q_r)` (líneas × filas) por esas sumas. Es
exacto (no una aproximación de convergencia) y unas 50 veces más rápido que
simular línea a línea a 128×128, que sigue disponible (`mode="events"`, con transitorio si
se pide `n_dummy`). Los dos modos coinciden a 1e-16 con B1+ no uniforme y
desvíos arbitrarios (test) y a 1.9e-14 en la pierna de 128×128.

### 2.4 Secuencias y reconstrucción

- **SE**: 90°x - TE/2 - 180°y - [desfasador de lectura + codificación de fase] - lectura centrada en TE - spoil.
- **IR (STIR)**: 180° - spoil - TI - SE como arriba; TI anula la grasa (`resolve`).
- **GRE**: α - [desfasador + fase] - lectura centrada en TE - spoil ideal.
- **Saturación grasa** (DP-FS, T2-FS): `SpectralSat` en la banda -4.4 .. -2.4 ppm
  con `cos(flip) = fat_sat_residual` y spoil justo antes de la excitación
  (reproduce el residuo del 5 % de `tissue_signal`).

El ancho de banda por píxel `BW` fija el tiempo de lectura `1/BW`, el gradiente
`BW/(γ̄ Δx)` y el desplazamiento de un desvío `Δf`: `Δf/BW` píxeles. Si la
lectura no cabe (TE < 1/BW en SE, TE < 1/(2 BW) en GRE, o el UTE, que necesita
lectura radial) se lanza un error que dice el BW mínimo. La reconstrucción es la
FFT inversa con el factor `exp(i k·origen)` que lleva la fase de los gradientes
(referida al isocentro) a la malla de la imagen; los ejes `x`, `z` son los de
`mri_scan`.

### 2.5 Desplazamiento químico

La grasa precesa a `ppm · γ̄ B0` respecto del agua: -217 Hz a 1.5 T con un pico
a -3.4 ppm (metileno). El modelo de seis picos es el del ISMRM Fat-Water
Toolbox (-3.80, -3.40, -2.60, -1.94, -0.39 y +0.60 ppm con amplitudes 0.087,
0.693, 0.128, 0.004, 0.039 y 0.048; comprobado en el código público de Hernando
et al. en GitHub), que procede del espectro in vivo de Hamilton et al. (NMR
Biomed 24:784, 2011; la correspondencia exacta con la tabla del artículo no se
pudo verificar: PubMed no era accesible). Con gradiente de lectura positivo la
grasa aparece desplazada hacia `x` menores. Agua y grasa de un tejido comparten
el T1 y el T2 de la tabla (que describen el tejido entero), así que en SE un
tejido uniforme da `tissue_signal`; en GRE la interferencia agua-grasa a cada
TE es física real que la ecuación analítica no tiene.

### 2.6 Susceptibilidad y campo B0

Tabla `SUSCEPTIBILITY_PPM` (χ volumétrica SI):

| tejido | χ (ppm) | fuente |
|---|---|---|
| agua | -9.05 | Schenck, Med Phys 23:815 (1996) |
| aire | +0.36 | Schenck 1996 (paramagnético por el O2; +0.358 ppm a 20 °C) |
| hueso cortical | -11.3 | Hopkins y Wehrli, MRM 37:494 (1997): -0.90e-6 cgs en costilla bovina (×4π) |
| lípido | -8.44 | +0.61 ppm sobre el agua (Karampinos et al., MRM 2012) |
| grasa, médula, edema, piel, músculo, fascia, periostio | mezcla agua/lípido con `FAT_FRACTION` | las diferencias entre tejidos blandos acuosos (0.01-0.1 ppm) se desprecian |
| hueso trabecular | -9.08 | 20 % de matriz cortical + 80 % de médula (la BV/TV de `porous`) |
| sangre de los vasos | -8.87 | Spees et al., MRM 45:533 (2001), Y = 0.8, Hct = 0.42 |
| hematoma | -9.08 | hiperagudo (oxihemoglobina, Y = 0.95), coherente con su T2 largo; agudo (Y = 0): -7.75 |

El campo relativo con la corrección de la esfera de Lorentz es la convolución
de χ con el núcleo dipolar (Salomir et al., Concepts Magn Reson B 19B:26, 2003;
Marques y Bowtell, Concepts Magn Reson B 25B:65, 2005):
`ΔB(k)/B0 = (1/3 - (k·b)²/|k|²) χ(k)`. **Fórmula 2D**: el fantoma es un corte
(x, z) de un objeto que se supone **invariante en la dirección perpendicular
al corte (y)**; su transformada 3D es `χ(kx, kz) δ(ky)` y la convolución 3D se
reduce exactamente a la 2D con `D(kx, kz) = 1/3 - (kx bx + kz bz)²/(kx² + kz²)`.
Lo aproximado es la geometría (hueso infinitamente largo), no el campo. El
término `k = 0` toma la media angular de `D` (`1/3 - (bx² + bz²)/2`), la que da
el campo exacto del cilindro; una constante sólo mueve la frecuencia central,
que el equipo ajusta (se resta la mediana de ΔB0 en los tejidos acuosos). Antes
de la FFT se repite el borde (`padding = 1`, 3× el tamaño), lo que prolonga las
capas de un corte longitudinal fuera del FOV.

**Orientación de B0** (`BlochPhysics.b0_direction`):

- `"y"` (perpendicular al corte): `D = 1/3`, sólo el desplazamiento de Lorentz
  `Δχ/3` de cada tejido, sin gradientes. Es el corte transversal de una
  extremidad dentro del túnel de un equipo convencional (B0 a lo largo de la
  pierna).
- `"x"` o `"z"` (en el plano): el corte longitudinal en el túnel (B0 a lo largo
  del hueso, `"x"`), o el corte transversal en un imán abierto de campo
  vertical / con la extremidad perpendicular al túnel (`"z"`).
- `"auto"` (por defecto) sigue la geometría del túnel: `"y"` en
  `cross_section_limb`, `"x"` en `longitudinal_limb`.

### 2.7 Bobina y ruido

`SurfaceCoil(center, radius)` multiplica la señal de recepción por
`(R²/(R² + d²))^{3/2}` (la ley del campo en el eje de una espira aplicada con
la distancia `d`: un modelo simple de la caída con la profundidad, no
Biot-Savart); también se acepta un mapa complejo o una función. El ruido es
gaussiano complejo en el k-space con `σ_k = σ_img √N`, de modo que la imagen
tiene `σ_img = señal analítica del músculo / snr` por canal (la definición de
`mri_scan`); la magnitud tiene así estadística de Rice.

## 3. Resultados

### (a) Fantoma uniforme sin artefactos frente a `sequences.signal`

Fantoma de un solo tejido que llena el FOV (16×16), `BlochPhysics.none()`
(sin desplazamiento químico ni B0; con el T2' de la tabla). Error relativo
máximo sobre los 12 tejidos con señal (la cortical, con T2 = 0.5 ms, tiene
señal < 1e-6 salvo en GRE y se excluye):

| secuencia | 1.5 T | 3 T |
|---|---|---|
| T1 | 5.7e-15 | 5.3e-15 |
| T2 | 5.8e-15 | 5.3e-15 |
| PD | 5.9e-15 | 5.7e-15 |
| PD_FS | 5.9e-15 | 5.2e-15 |
| T2_FS | 6.3e-15 | 6.2e-15 |
| STIR | 2.7e-11 | 9.2e-13 |
| GRE | 8.3e-15 | 1.2e-14 |

Objetivo < 1 %: se cumple con 9 órdenes de margen (el STIR de la grasa, casi
anulada, es el peor caso relativo). Las celdas porosas reproducen la mezcla
lineal de `property_map(..., physics="mri")` a 1e-9 (test).

### (b) Desplazamiento químico medido en la imagen frente a Δf/BW

Bloque de 10 mm de agua y bloque igual de grasa pura, SE (TR 500, TE 30 ms),
64×64. Desplazamiento de la grasa respecto del agua a lo largo de la lectura,
medido de dos formas: la fase del primer armónico de la imagen compleja
(exacta para un objeto simétrico de banda limitada) y el centroide de la
magnitud (lo que se ve):

| B0 (T) | BW (Hz/píxel) | modelo | esperado | fase del 1.er armónico | centroide de la magnitud |
|---|---|---|---|---|---|
| 1.5 | 65 | 1 pico | -3.341 | -3.341 | -3.284 |
| 1.5 | 130 | 1 pico | -1.670 | -1.670 | -1.646 |
| 1.5 | 260 | 1 pico | -0.835 | -0.835 | -0.809 |
| 3.0 | 65 | 1 pico | -6.681 | -6.681 | -6.576 |
| 3.0 | 130 | 1 pico | -3.341 | -3.341 | -3.284 |
| 3.0 | 260 | 1 pico | -1.670 | -1.670 | -1.646 |
| 1.5 | 65 | 6 picos | -2.964 | -2.969 | -3.020 |
| 1.5 | 130 | 6 picos | -1.482 | -1.483 | -1.532 |
| 1.5 | 260 | 6 picos | -0.741 | -0.741 | -0.728 |
| 3.0 | 65 | 6 picos | -5.928 | -5.964 | -5.919 |

Con un pico el error es < 0.001 píxeles (fase) y < 0.11 píxeles (magnitud),
dentro de ±0.2. Con seis picos en espín-eco todos se reenfocan en TE y el
centroide esperado es la media de los desplazamientos ponderada por amplitud
(-3.01 ppm); se mide a ≤ 0.04 píxeles (fase) y ≤ 0.06 (magnitud). Los tests
repiten el caso de un pico a 1.5 T/130, 3 T/260 y 1.5 T/80 Hz/píxel.

### (c) Espín-eco frente a un desvío estático; GRE frente a la distribución

Fantoma uniforme (T1 = 1 s, T2 = 50 ms) con un desvío estático común de 37 Hz
y una distribución intravóxel de desvíos; SE (TR 200 ms) y GRE (TR 200 ms,
α = 30°), BW 1000 Hz/píxel. Error relativo frente a `signal()` (SE) y frente a
`signal() × |FID de la distribución en TE|` (GRE):

| distribución | sec. | TE 2 ms | TE 5 ms | TE 10 ms | TE 20 ms | TE 40 ms |
|---|---|---|---|---|---|---|
| lorentziana exacta (estados), HWHM 25 Hz → `exp(-TE/T2')` | SE | 0 | 1.7e-16 | 9.8e-16 | -3.8e-16 | -8.3e-16 |
| | GRE | 5.1e-16 | 6.5e-16 | 2.6e-16 | 9.6e-16 | 4.1e-16 |
| gaussiana σ 15 Hz, 48 nodos de Gauss-Hermite → `exp(-(2πσTE)²/2)` | SE | 3.9e-15 | 6.9e-16 | -5.9e-16 | 3.8e-15 | -1.7e-15 |
| | GRE | -7.6e-16 | 1.8e-15 | 1.7e-16 | -3.3e-15 | -2.8e-13 |
| uniforme ±40 Hz, 64 isocromatos → `sinc(2·40·TE)` | SE | -1.3e-14 | -1.2e-14 | -8.8e-15 | -1.7e-14 | -2.9e-15 |
| | GRE | 1.0e-05 | 6.4e-05 | 2.6e-04 | 1.0e-03 | 4.1e-03 |
| lorentziana muestreada, 64 cuantiles | SE | -1.3e-14 | -1.2e-14 | -8.8e-15 | -1.7e-14 | -3.1e-15 |
| | GRE | +3.2e-02 | +6.4e-02 | +3.7e-02 | +0.90 | +21 |
| lorentziana muestreada, 256 cuantiles | GRE | +1.3e-03 | +2.2e-02 | -3.2e-02 | -0.35 | +4.0 |

El espín-eco no depende ni del desvío ni de su dispersión (1e-14). El GRE
sigue la función característica de la distribución a precisión de máquina
cuando la distribución se representa exactamente (estados de desfase para la
lorentziana, cuadratura de Gauss-Hermite para la gaussiana); con isocromatos
equiespaciados la uniforme converge (error 1e-5 a 4e-3, de discretización), y
la lorentziana muestreada por cuantiles da errores del 3-6 % y se re-enfasa a
TE largos: por eso el T2' de la tabla se modela con estados.

### (d) Campo de un cilindro infinito perpendicular a B0

Cilindro con fracción de área exacta en cada celda (supermuestreo 8×8), Δχ = 1
ppm. Error frente a la solución exacta (`-Δχ/6` dentro, `(Δχ/2)(a/r)² cos 2φ`
fuera), en % de `Δχ/2` (el valor en la superficie), fuera de la escalera del
borde:

| malla | radio (celdas) | relleno | B0 | máx. a > 2 celdas | máx. a > 5 celdas | RMS | media dentro |
|---|---|---|---|---|---|---|---|
| 128² | 16 | 1 | z | 0.64 % | 0.14 % | 0.045 % | -0.16667 (exacta -0.16667) |
| 128² | 16 | 1 | x | 0.64 % | 0.14 % | 0.045 % | -0.16667 |
| 256² | 32 | 1 | z | 0.69 % | 0.18 % | 0.033 % | -0.16667 |
| 256² | 32 | 0.5 | z | 0.70 % | 0.23 % | 0.101 % | -0.16667 |
| 512² | 64 | 1 | z | 0.66 % | 0.19 % | 0.027 % | -0.16667 |
| 128² | 16 | 1 | 45° en el plano | 5.50 % | 3.25 % | 0.40 % | -0.16667 |
| 512² | 64 | 1 | 45° en el plano | 5.10 % | 2.63 % | 0.20 % | -0.16667 |
| todas | | | y (⊥ corte) | 0.00 % | 0.00 % | 0.000 % | +0.33333 (exacta Δχ/3) |

Con B0 a lo largo de un eje de la malla (los casos que usa la pierna) el error
es < 0.7 %, dentro del objetivo de 2-3 %. Con B0 oblicuo a la malla el error
cuadrático medio sigue siendo bajo (0.2-0.4 %), pero la escalera del borde
deja estrías radiales que decaen despacio: 5 % a 2-4 celdas del borde y 2-3 %
a más de 5, sin mejorar al refinar (el error está ligado a la distancia en
celdas al borde). No se corrigió: queda como limitación documentada.

### (e) La pierna: qué cambia en la visibilidad de la fractura

Corte transversal `cross_section_limb("pierna", RingFracture(angle_deg=-10,
gap=0.6 mm, edema_length=12 mm), dx=0.2 mm)` con el agua sustituida por aire,
128×128 (vóxel 0.78 mm), BW 130 Hz/píxel (grasa: -1.67 píxeles), 1.5 T. CNR =
diferencia de medias de la imagen sin ruido entre la σ del ruido con SNR 40 en
el músculo. La línea de fractura es el pico de la imagen en ±2 píxeles de la
grieta real, en la fila a media cortical, frente a la mediana de la cortical
vecina (hasta 4 mm); entre paréntesis, dónde aparece el pico respecto de la
grieta real.

Campo: con B0 ⊥ corte el tejido está entre -48 y +12 Hz (sólo desplazamientos
de Lorentz por tejido, nada en la grieta); con B0 vertical en el plano, entre
-40 y +63 Hz (percentiles 1-99; máx. 169 Hz junto a la tibia) y +79 Hz de media
en la grieta (+8 .. +112 Hz). Ninguna celda de grasa sale de la banda de la
saturación espectral (|ΔB0| > 1 ppm): dentro de una extremidad redonda el campo
de la piel es casi uniforme.

| secuencia | física | edema - médula | hematoma - músculo | línea de fractura - cortical | grasa subcutánea |
|---|---|---|---|---|---|
| T1 | sin artefactos | -42.2 | +20.4 | +35.7 (0) | 0.525 |
| T1 | sólo desplazamiento químico | -26.6 | +20.1 | +36.3 (0) | 0.461 |
| T1 | + χ, B0 ⊥ corte (túnel) | -27.8 | +20.1 | +36.0 (0) | 0.466 |
| T1 | + χ, B0 vertical en el plano | -25.6 | +17.2 | +37.9 (+1) | 0.461 |
| STIR | sin artefactos | +84.8 | +50.3 | +46.9 (0) | 0.004 |
| STIR | sólo desplazamiento químico | +80.7 | +50.3 | +49.5 (0) | 0.005 |
| STIR | + χ, B0 ⊥ corte | +80.8 | +50.2 | +49.6 (0) | 0.005 |
| STIR | + χ, B0 vertical | +81.2 | +45.4 | +51.5 (+1) | 0.005 |
| T2_FS | sin artefactos | +103.7 | +168.2 | +112.0 (0) | 0.022 |
| T2_FS | + χ, B0 vertical | +105.0 | +156.8 | +115.2 (+1) | 0.021 |
| GRE | sin artefactos | -33.7 | +9.4 | +30.5 (0) | 0.151 |
| GRE | sólo desplazamiento químico | -21.4 | +10.4 | +31.0 (0) | 0.131 |
| GRE | + χ, B0 vertical | -20.6 | +7.5 | +33.3 (+1) | 0.131 |
| PD_FS | sin artefactos | +39.5 | +42.9 | +44.6 (0) | 0.053 |
| PD_FS | + χ, B0 vertical | +40.3 | +38.4 | +46.1 (+1) | 0.049 |

Grosor aparente de la cortical de la tibia en T1, en la fila del centro del
hueso (real 4 mm = 5.1 píxeles; umbral: media señal del músculo): 4 y 5
píxeles (izquierda y derecha) sin artefactos; 2 y 7 con desplazamiento químico.

Señal integrada con susceptibilidad (B0 vertical) entre la señal sin ella, sin
desplazamiento químico (la región incluye la zona a la que la distorsión mueve
la señal):

| región | DP (espín-eco) | GRE TE 5 ms | GRE TE 15 ms | GRE TE 30 ms |
|---|---|---|---|---|
| anillo de 3 mm alrededor de la tibia | 0.999 | 0.994 | 0.959 | 0.928 |
| 6 mm bajo la piel (y 2 mm de aire) | 0.998 | 0.986 | 0.969 | 0.950 |

Corte longitudinal (`longitudinal_limb("tibia", Fracture())`, dx 0.1 mm, vóxel
0.3 mm): con B0 a lo largo del hueso (túnel) la grieta transversal es una
lámina perpendicular a B0 y su hematoma está a -111 Hz de media (-146 .. 0);
con B0 perpendicular al hueso, a +112 Hz.

| secuencia | física | edema - médula | hematoma - músculo | línea de fractura - cortical |
|---|---|---|---|---|
| T1 | sin artefactos | -42.3 | +22.8 | +50.9 (0) |
| T1 | + χ, B0 a lo largo del hueso | -39.3 | +29.0 | +46.2 (-1) |
| STIR | sin artefactos | +84.0 | +46.6 | +70.4 (0) |
| STIR | + χ, B0 a lo largo del hueso | +82.9 | +53.3 | +63.8 (-1) |
| GRE | sin artefactos | -33.6 | +11.4 | +43.4 (0) |
| GRE | + χ, B0 a lo largo del hueso | -31.7 | +15.8 | +37.6 (-1) |

**Lectura honesta.**

1. **El edema en STIR y en T2/DP con saturación grasa no cambia** (CNR 81-85 y
   104-105 en STIR y T2-FS con o sin artefactos): la supresión grasa elimina la
   grasa que se desplaza, y el campo dentro del hueso es casi uniforme. Es la
   secuencia que conviene para detectar la fractura y el simulador lo confirma.
2. **El desplazamiento químico sí degrada el T1 y el GRE**: el contraste medio
   edema-médula baja de -42 a -27 (T1) y de -34 a -21 (GRE), porque la grasa de
   la médula se corre 1.7 píxeles y deja una media luna oscura de agua en un
   borde de la cavidad (que se confunde con edema) y se monta sobre la cortical
   en el otro. La cortical aparece de 2 píxeles en un lado y de 7 en el otro
   (real: 5): una cortical adelgazada o una interrupción fina del lado de la
   media luna se podrían enmascarar o simular. En el corte longitudinal la
   lectura va a lo largo de las capas y el efecto es menor (-42 → -39).
3. **La línea de fractura sigue visible en todas las secuencias**: la
   susceptibilidad la desplaza ±1 píxel (±80-110 Hz en la grieta frente a 130
   Hz/píxel) y en el corte longitudinal con B0 a lo largo del hueso le quita un
   9-13 % de contraste (T1, STIR, GRE), más en GRE por el desfase dentro del
   vóxel de la grieta. Con una grieta de 0.4-0.6 mm en vóxeles de 0.3-0.8 mm el
   contraste ya está limitado por el volumen parcial, no por los artefactos.
4. **El hematoma perióstico pierde un 5-10 % de contraste con B0 en el plano**
   (STIR 50 → 45, T2-FS 168 → 157), por la distorsión en el borde de la tibia.
   En el corte longitudinal el contraste "sube" (23 → 29 en T1) porque la grasa
   de la médula desplazada invade el hematoma intramedular: es un artefacto, no
   mejor visibilidad.
5. **La pérdida de señal en GRE junto al hueso es pequeña en esta geometría**:
   0.6 % a TE 5 ms (el GRE del paquete), 4 % a 15 ms y 7 % a 30 ms en un anillo
   de 3 mm alrededor de la tibia, frente a < 0.1 % en espín-eco. Un cilindro
   largo de hueso cortical en músculo crea poco campo (Δχ = -2.3 ppm, 72 Hz en la
   superficie a 1.5 T); el "blooming" clínico junto al hueso viene sobre todo del
   hueso trabecular, del metal y de la geometría 3D, que este modelo 2D no
   tiene. Con aire es mucho mayor (test: cilindro de aire de 4 mm en músculo, la
   señal GRE a TE 15 ms cae por debajo del 90 % y la SE no cambia).
6. **Con B0 perpendicular al corte (corte transversal en un túnel) la
   susceptibilidad casi no hace nada**: un cilindro paralelo a B0 no perturba su
   exterior; sólo quedan los desplazamientos de Lorentz por tejido (< 0.4
   píxeles). En 2D no se puede representar la interrupción del hueso a lo largo
   de B0 que sí produciría campo en una fractura real vista en axial.

### (f) Tiempos

Pierna transversal 128×128, dx 0.2 mm (309 735 isocromatos), todos los
artefactos con B0 vertical, un hilo:

| secuencia | modo | backend | campo B0 (s) | simulación (s) | total (s) |
|---|---|---|---|---|---|
| T1 | steady | numpy | 0.06 | 0.83 | 0.94 |
| T1 | steady | numba | 0.14 | 0.51 | 0.70 |
| STIR | steady | numpy | 0.13 | 1.01 | 1.19 |
| STIR | steady | numba | 0.16 | 0.74 | 0.97 |
| GRE | steady | numpy | 0.09 | 0.83 | 0.98 |
| GRE | steady | numba | 0.19 | 0.55 | 0.83 |
| STIR | events (línea a línea) | numba | 0.14 | 48.5 | 48.7 |
| STIR | events (línea a línea) | numpy | 0.07 | 58.7 | 58.8 |

(De una ejecución a otra los tiempos varían un 20-30 % por la carga de la
máquina.) Los dos modos dan el mismo k-space (diferencia máxima 1.9e-14 del
máximo). Numba sólo compila la lectura; el estado estacionario, los pulsos, la
gestión de los estados de desfase de cada TR y el campo B0 son NumPy con los
dos backends, por eso la ganancia es de 1.2-1.8×. Los 56 tests de
`tests/test_mri_bloch.py` tardan ~1.5 s; el benchmark completo, ~3.5 min.

## 4. Limitaciones y diferencias con KomaMRI y JEMRIS

- **2D**: no hay perfil de corte ni selección de corte, ni flujo, movimiento o
  difusión; el campo B0 supone el objeto infinito en la dirección
  perpendicular al corte (§2.6, punto 6 de la lectura).
- **Spoiling ideal** (la hipótesis de las ecuaciones analíticas): no se simulan
  los gradientes de destrucción ni la fase de RF variable, ni por tanto los
  ecos estimulados residuales que dejan en un equipo real.
- **Sólo trayectorias cartesianas**: sin radial ni espiral (el UTE sigue en el
  motor analítico); no se leen secuencias Pulseq.
- **Pulsos con forma**: relajación y T2' microscópico con separación simétrica
  (no exactos dentro del pulso); el resto del motor es exacto.
- **Grasa**: agua y grasa de un tejido comparten T1 y T2; la saturación
  espectral es un pulso ideal de banda rectangular.
- **Susceptibilidad**: hueso medido en costilla bovina; tejidos blandos
  ≈ agua; corrección de Lorentz isótropa (discutible en tejidos anisótropos);
  error de discretización con B0 oblicuo a la malla (tabla (d)).
- **Bobina**: modelo de la espira en su eje, no Biot-Savart.
- **No se ha comparado numéricamente con KomaMRI ni JEMRIS** (Julia y C++ no
  están en el entorno): la validación es frente a soluciones exactas. Un
  siguiente paso sería exportar el fantoma y la secuencia a KomaMRI y comparar
  la señal por vóxel, como propone el plan de `comparativa_simuladores.md`.

## 5. Referencias y estado de verificación

- Schenck, Med Phys 23:815 (1996): χ del agua (-9.05 ppm) y del aire; la cifra
  del agua y el +0.358 ppm del aire se confirmaron en resúmenes del ISMRM que
  los citan, no en el artículo (no accesible).
- Hopkins y Wehrli, MRM 37:494 (1997): -0.90e-6 (cgs) para hueso bovino; la
  conversión a -11.3 ppm SI es aritmética propia (×4π) y coincide con la cita
  de una revisión posterior.
- Karampinos et al., MRM (2012): Δχ grasa-agua 0.61 ppm (confirmado en su
  resumen del ISMRM 2011 y en citas).
- Spees et al., MRM 45:533 (2001): fórmula de la χ de la sangre (confirmada en
  artículos que la citan).
- Hamilton et al., NMR Biomed 24:784 (2011) y el ISMRM Fat-Water Toolbox:
  tabla de seis picos (verificada en el código del toolbox; la atribución a la
  tabla exacta del artículo, no verificada).
- Salomir et al., Concepts Magn Reson B 19B:26 (2003); Marques y Bowtell,
  Concepts Magn Reson B 25B:65 (2005): núcleo dipolar de Fourier (citas
  confirmadas por búsqueda).
- CODATA 2018 (tabla de NIST): γ/2π del protón = 42.577 478 518 MHz/T
  (confirmada por búsqueda).
