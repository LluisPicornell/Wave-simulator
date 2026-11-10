# Validación de la EIT: modelo completo de electrodo y reconstrucción Gauss-Newton

Código: `wavesim/em/quasistatic.py` (problema directo, CEM, jacobiano), `wavesim/em/eit.py`
(soluciones analíticas del disco, `EITModel`, `gauss_newton`, regularizadores, curva L, discrepancia, métricas
GREIT). Tests rápidos: `tests/test_eit_cem.py`. Tablas: `python benchmarks/eit_validation.py [analytic|disc|leg]`
(~15 min, ~10 min y ~30 min de CPU en una máquina cargada; el caso de la pierna es lento porque la piel, con
conductividad 10⁻³ veces la del músculo, empeora mucho el condicionamiento de COCG en la malla de 0.5 mm).

Con los parámetros por defecto nada cambia: `simulate_eit` sigue usando electrodos ideales
(`contact_impedance=None`) y la suite anterior pasa sin modificar.

## 1. Modelo completo de electrodo (CEM)

**Formulación** (Cheng et al. 1989; Somersalo, Cheney e Isaacson 1992). Bajo el electrodo `l`,
`sigma* dphi/dn = (U_l - phi) / z_c`; fuera de los electrodos, flujo nulo; cada electrodo impone su corriente
total `I_l = (1/z_c) ∫_{E_l} (U_l - phi) ds`. Las incógnitas son `phi` y las tensiones `U_l`.

**Unidades.** `z_c` se da en ohm·m² (resistencia × área de piel, como se tabulan los contactos). El corte es 2D
con corrientes por metro de profundidad: un tramo de electrodo de longitud `l` tiene área `l · 1 m` y conductancia
`l / z_c` (S por metro de profundidad), la misma unidad que las admitividades de cara. Se usa el ancho nominal
del electrodo (repartido a partes iguales entre sus celdas), no la longitud en escalera de las celdas de piel. A
`z_c` se suma el camino del centro de la celda a la superficie, `(dx/2)/sigma*`: así `z_c → 0` es un electrodo
ideal en la superficie. Consecuencia: la caída medida entre el metal y el potencial medio de las celdas es
`I (z_c + dx/(2 sigma*)) / w`, no `I z_c / w`; lo que sí escala exactamente como `I/w` es su pendiente respecto a
`z_c` (test `test_contact_drop_scales_with_current_times_zc_over_width`).

**Solución numérica.** Con `U` fijas, `(A + G) phi = G U` es un problema de Robin bien planteado (COCG). Se
resuelve un campo `psi_l` por electrodo y la matriz de transferencia `S` (simétrica, filas de suma nula); para
cualquier patrón de corrientes, `S U = I` con `sum U = 0` y `phi = sum U_l psi_l`. Coste: igual que el modo ideal
(un COCG por electrodo). El jacobiano reutiliza los campos de inyección (reciprocidad) y añade el término de las
celdas de contacto `-dg/dd (phi_p - U_p)(phi_q - U_q)`; coincide con la diferencia finita al 7·10⁻⁵ (test, bloques que
tocan electrodos).

**Serie del CEM en el disco** (`disc_cem_series`; derivación completa en su docstring). Con `u = a0 + sum r^n (a_n cos nt + b_n sin nt)`,
flujo `sigma du/dr = sigma sum n R^{n-1}(...)`, se multiplica la condición de contorno por cada función de la base y se
integra sobre el borde (`ds = R dt`). Por ortogonalidad: `sigma n pi c_k = (R/z_c) sum_l [U_l ∫_{E_l} phi_k − sum_j c_j ∫_{E_l} phi_k phi_j]`,
con `c_k = a_n R^n` (coeficientes del potencial de borde, para evitar desbordar `R^n`), más `L` ecuaciones de corriente
por electrodo y `c_0 = 0` como referencia. Las integrales sobre arcos se hacen con Gauss-Legendre por tramos; el
sistema es cuadrado. Las esquinas del electrodo dan una singularidad integrable y la serie converge como `1/N`
(tabla abajo).

### Resultados (disco homogéneo R = 20 mm, músculo a 50 kHz, `I` = 1 mA/m, 16 electrodos, trama 0, 13 medidas)

Electrodos ideales frente a electrodos puntuales (fórmula cerrada `V = (I/pi sigma*) [...]` con la función de Green de Neumann):

| dx (mm) | ancho (mm) | error máx. por dato (%) | error de norma (%) |
|---|---|---|---|
| 2.00 | 2 | 6.29 | 5.33 |
| 1.00 | 2 | 5.31 | 2.55 |
| 0.50 | 2 | **1.33** | 0.97 |
| 0.50 | 1 | 4.39 | 2.09 |

Se cumple el objetivo (< 2 %) sólo con dx = 0.5 mm y electrodos de 2 mm (4 celdas); con electrodos de una celda domina
la escalera del borde y el modelo de electrodo (corriente uniforme, no puntual), no la discretización del interior.

CEM (diferencias finitas, dx = 0.5 mm, electrodos de 2 mm) frente a la serie de Fourier (200 modos):

| z_c (ohm·m²) | error máx. (%) | error de norma (%) | pendiente de la caída / (I/w) |
|---|---|---|---|
| 0.003 | 1.39 | 0.67 | – (caída = 1.235 I z_c/w por el término dx/2σ) |
| 0.03 | 1.48 | 0.67 | caída 1.024 I z_c/w |
| 0.3 | 1.49 | 0.67 | caída 1.002 I z_c/w |

Convergencia de la serie (z_c = 0.03; |U₀ − U₁| en mV): 35.1207 (50 modos), 35.1547 (100), 35.1681 (200), 35.1719 (400),
35.1728 (800): convergida al 10⁻⁴ con 200 modos.

Límite `z_c → 0`: `max|V(z_c) − V(0)| / max|V(0)|` = 1.0·10⁻⁵ (1e-6), 6.6·10⁻³ (1e-3), 1.6·10⁻² (1e-2), 1.9·10⁻² (1e-1),
2.0·10⁻² (1): el límite existe y se alcanza de forma monótona. Con `z_c` pequeño, `V(0)` frente al electrodo puntual
analítico queda al 5 % (test). Que el contacto cambie las tensiones sólo un 2 % entre 0 y 1 ohm·m² en este disco
indica que, con electrodos de 2 mm y músculo, la tensión de las medidas lejanas depende poco del contacto; en cambio la caída en
los electrodos que inyectan sí (I z_c/w).

## 2. Reconstrucción Gauss-Newton

`EITModel`: parámetros por bloques de 3×3 celdas, `y = y0 exp(p)`; el jacobiano se recalcula en cada iteración con
`sensitivity_matrix` (y su término de contacto). `gauss_newton` minimiza `|W(d − F(p))|² + λ pen(p − p_ref)` con
búsqueda de línea; regularizadores Tikhonov, Laplaciano de grafo (`L'L`, como `laplace` de EIDORS), NOSER y variación
total (difusividad retardada, β = 10⁻²); `λ` adimensional (1 = igual a la media de la diagonal de `J'J`), por valor, por
curva L (máxima curvatura, primer paso) o por discrepancia de Morozov (si el error de modelo impide llegar al nivel de
ruido, como aquí, recurre a la curva L). Modo de diferencias: el objetivo es `F(p0) + (d − d_ref)`.

Protocolo de validación (sin crimen inverso): datos en malla de 0.5 mm con CEM (z_c = 0.02 ohm·m², electrodos de 3 mm) y
ruido gaussiano complejo de SNR 70 dB (rms|V|); reconstrucción en una malla de 1 mm con otros electrodos en escalera.
Inclusión de sangre (r = 4 mm, σ×2) en un disco de músculo R = 20 mm; 3 posiciones × 5 realizaciones de ruido.
Métricas GREIT (Adler et al. 2009): posición (centro de gravedad de la región ≥ 25 % del máximo), AR, RES, SD, ringing.

Promedio sobre posiciones y realizaciones:

| método | error de posición (mm) | \|1 − AR\| | RES | SD | ringing |
|---|---|---|---|---|---|
| un paso (NOSER, λ=1e-2) | 3.20 | 1.20 | 0.140 | 0.08 | 0.72 |
| un paso (NOSER, λ=1e-1) | 2.38 | 0.68 | 0.188 | 0.06 | 0.23 |
| GN diferencias NOSER (3e-2) | 2.31 | 0.49 | 0.163 | 0.09 | 0.23 |
| GN diferencias Tikhonov | 0.33 | 0.24 | 0.310 | 0.10 | 0.23 |
| GN diferencias Laplaciano | 0.18 | 0.15 | 0.280 | 0.11 | 0.24 |
| GN diferencias TV | **0.08** | **0.10** | 0.240 | 0.11 | 0.10 |
| GN diferencias NOSER, curva L | 2.23 | 0.48 | 0.151 | 0.05 | 0.24 |
| GN absoluto NOSER | 3.16 | 0.59 | 0.148 | 0.24 | 0.30 |

Lectura. Frente al paso lineal, Gauss-Newton mejora el error de posición y, sobre todo, la amplitud (AR en el centro:
0.41–0.63 del paso lineal frente a 0.76–0.84 de GN; en posición intermedia 0.57–0.60 frente a 0.88–0.91). Los promedios
de NOSER (incluidos GN) los dominan los fallos en la **periferia** (inclusión a 14 mm del centro, rozando el borde): con
NOSER el máximo cae en un píxel de la piel (6.08 mm de error, AR 2–3.6), sea paso lineal o GN. Tikhonov, Laplaciano y TV no
fallan ahí (0.04–0.58 mm). Con NOSER y λ = 1e-2 el GN oscilaba entre iteraciones (un píxel caliente), por lo que el valor
por defecto es 3e-2. El criterio de discrepancia no es aplicable aquí (el error de modelo entre mallas es mayor que el
ruido); la curva L da λ ≈ 1.8e-2 (adimensional) y resultados equivalentes a λ = 3e-2. La imagen absoluta (arranque
homogéneo) funciona pero es la peor en posición y deformación (SD 0.24): depende del error de modelo (geometría y
electrodos). No se ha implementado GREIT (matriz de reconstrucción entrenada); queda pendiente. Las diferencias entre
regularizadores se midieron con una sola configuración de ruido (70 dB) y λ fijado a mano (3e-2), sin ajustar cada uno.

## 3. Caso de la pierna (`cross_section_limb`, fractura anular con hematoma y edema)

Datos en 0.5 mm (200×200), 16 electrodos de 8 mm, SNR 70 dB; reconstrucción en 1 mm (fantoma submuestreado), mismo
protocolo de diferencias, λ = 3e-2 (NOSER) o 3e-2 (Laplaciano), 8 iteraciones. Hematoma real: centro (−3.2, −38.9) mm,
12.2 mm² (≈ 4 mm de diámetro). Cambio máximo de una tensión por la fractura: 41 %.

| datos (electrodo) | modelo | método | error del píxel máximo (mm) | AR | RES |
|---|---|---|---|---|---|
| ideal | ideal | un paso | 7.2 | 0.00 | 0.152 |
| ideal | ideal | GN NOSER | 8.4 | 0.69 | 0.084 |
| ideal | ideal | GN Laplaciano | 2.0 | 0.08 | 0.188 |
| CEM 0.003 | CEM 0.003 | un paso | 7.2 | 0.00 | 0.141 |
| CEM 0.003 | CEM 0.003 | GN NOSER | 8.4 | 0.91 | 0.084 |
| CEM 0.003 | CEM 0.003 | GN Laplaciano | 4.3 | 0.08 | 0.180 |
| CEM 0.03 | CEM 0.03 | un paso | 7.2 | 0.00 | 0.141 |
| CEM 0.03 | CEM 0.03 | GN NOSER | 8.4 | 0.91 | 0.084 |
| CEM 0.03 | CEM 0.03 | GN Laplaciano | 4.3 | 0.08 | 0.184 |
| CEM 0.03 | ideal (desajuste) | un paso | 27.4 | 0.01 | 0.114 |
| CEM 0.03 | ideal (desajuste) | GN NOSER | 27.4 | 0.87 | 0.085 |
| CEM 0.03 | ideal (desajuste) | GN Laplaciano | 4.3 | 0.08 | 0.187 |

Lectura honesta:

- **Contacto.** Con z_c = 0.003 y 0.03 ohm·m² (0.03–0.3 kΩ·cm²) la imagen no cambia entre sí, y apenas respecto del
  electrodo ideal: explicación probable (no medida aparte): la capa de piel del fantoma (σ ≈ 3·10⁻⁴ S/m a 50 kHz, espesor de
  ~mm) ya pone en serie una resistencia por área del orden de ohm·m², muy superior a z_c. El orden de magnitud de z_c para Ag/AgCl con gel (decenas a cientos de
  ohm·cm²) **no pude verificarlo**: la búsqueda web no devolvió un valor en ohm·cm² y la fuente (PMC6302411, EIT cerebral a 50 kHz) no
  era accesible desde el entorno. Los valores usados son una hipótesis para comparar, no un dato citado. Lo que sí se midió:
  un modelo de reconstrucción ideal con datos CEM falla en el máximo del paso lineal y del NOSER (27 mm, un artefacto de borde), mientras
  que el Laplaciano resiste (4.3 mm).
- **Localización del hematoma.** El paso lineal da 7.2 mm de error del píxel máximo (el ejemplo `10_impedancia_eit.py`
  daba ~6 mm con otra malla); GN con NOSER no mejora (8.4 mm: el máximo cae en el borde). GN con Laplaciano da 2.0–4.3 mm, mejor,
  pero con una sola configuración de ruido y sin replicar, así que no es una cifra firme. Con el hematoma de ~4 mm y celdas de
  reconstrucción de 3 mm, el límite físico es de unos pocos mm.
- **Métricas GREIT en este caso.** El «objetivo» (todas las celdas que cambian de tejido: hematoma, edema, hueco de la
  fractura) tiene contrastes de signos mezclados y el AR (0.00–0.91) y el error GREIT que imprime el benchmark no son
  interpretables como en el disco; por eso la tabla sólo recoge el error del píxel máximo, el AR y RES.

## Limitaciones y pendientes

- Sólo electrodos de contacto con ancho nominal y patrón adyacente; sin 3D ni FEM.
- No hay GREIT entrenado; sí TV (difusividad retardada).
- El GN se probó con λ fijo (3e-2), 70 dB y un tamaño de inclusión; la elección automática del parámetro es la curva L y la
  discrepancia sólo aplica si el error de modelo es menor que el ruido.
- Caso de la pierna: una realización de ruido y una fractura; no se repite con otras semillas.
