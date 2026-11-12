# ¿Cuántos emisores y receptores para detectar fracturas? Estudio con simulación

Pregunta: para un dispositivo portátil y barato de detección rápida de fracturas, ¿mejora la
detección tener uno o varios emisores y uno o varios receptores, cuántos y en qué disposición?

Código: `benchmarks/estudio_emisores_receptores.py` (simulación y análisis, con caché en
`outputs/estudio_er/`), `wavesim/ultrasound/fmc.py` (captura de matriz completa y reconstrucción
de subconjuntos; tests en `tests/test_fmc.py`).

<!-- RESULTADOS -->

## Método

### Fantomas y tarea de detección

Corte longitudinal 2D de la tibia (`longitudinal_limb(..., realism="realista")`: porosidad cortical,
rugosidad multiescala, fascias). Cada pierna es una anatomía distinta (`random_limb_spec`): grosores de
piel, grasa, cortical y médula ±10 %, inclinación del hueso ~N(0°, 2°), y semilla propia de rugosidad y
microestructura. Dos tipos de cuerpo (normal y obeso: grasa ×3).

- **Negativos**: 6 tibias diafisarias sanas y 4 metáfisis sanas por cuerpo.
- **Positivos**: las lesiones del ejemplo 11 (`CASES`), cada una en otra anatomía y otra posición:
  fractura desplazada 1 mm (control), fisura superficial, grieta intracortical, fractura por estrés con
  reacción endóstica, grieta de la cortical lejana y contusión ósea (en la metáfisis). Posiciones a 3 MHz:
  x = −2, 0, +2 mm (obeso: −2, +2); a baja frecuencia: −6, 0, +6 mm (obeso: −6, +6).
- Grupos para el AUC: **superficial** = desplazada + fisura (abren la cortical que mira a la sonda),
  **interna** = intracortical + estrés endóstica, **contusión** (frente a las metáfisis sanas),
  **lejana** (sólo transmisión a través).

Tamaño de muestra por cuerpo y grupo: normal 6 positivos (grupos de dos lesiones) o 3 (una lesión)
frente a 6 negativos (4 en la contusión); obeso 4 o 2 frente a 6 (4). Con 6 frente a 6, un AUC de
0.5 tiene un IC por *bootstrap* de aproximadamente ±0.3, y sólo los AUC por encima de ~0.85 tienen el
IC entero por encima de 0.5. **Las diferencias entre configuraciones menores de ~0.2 no son
concluyentes con esta muestra.**

### 3 MHz: captura de matriz completa

Sonda de 32 elementos a λ/2 (0.257 mm, apertura 8.2 mm, anchura de elemento 0.85·paso) en el gel sobre
la cara anterior, malla de 60 µm (λ/8 en la grasa), dominio de 15 mm de ancho hasta 6 mm bajo el
endostio cercano (8 mm en la metáfisis). Una simulación FDTD por elemento emisor (32 por fantoma, ~5 min
de núcleo cada FMC en el cuerpo normal, ~9 en el obeso). De cada FMC se reconstruyen, sin volver a
simular:

| Familia | Qué es | Disparos | Canales rx |
|---|---|---|---|
| Dispersa N_tx × N_rx | elementos repartidos por toda la apertura (N ∈ 1, 2, 4, 8, 16, 32): misma apertura, menos elementos | N_tx | N_rx |
| Compacta N_tx × N_rx | bloque contiguo de max(N_tx, N_rx) elementos (sonda pequeña) desplazado medio bloque cada vez por la superficie; cada columna de la imagen sale de la posición más cercana | N_tx por posición | N_rx |
| Monoestático | un solo transductor emisor-receptor (λ/2, o 1 mm = 4 elementos sumados) movido a 32 (29) posiciones: líneas A apiladas, o SAFT (suma con retardos de todas las posiciones); SAFT con error de posición σ = 0.1 mm | 1 por posición | 1 |
| A-scan fijo | el transductor monoestático quieto en el centro (sin imagen) | 1 | 1 |
| Onda plana | los 32 elementos disparan a la vez con retardos (síntesis por linealidad: Σ S_i(t − τ_i)); 1, 3 (±12°) o 7 ángulos | 1-7 | 32 |
| STA completa | 32 × 32 | 32 | 32 |

Formación de haz: DAS a 1540 m/s con aceptación angular ±45° (F# = 0.5) y sin ventana, igual para todas
(F# = 1.5 para el elemento de 1 mm, por su directividad); factor de coherencia (CF) sobre los receptores.
La síntesis por linealidad está verificada: la onda plana sintetizada a partir de la FMC coincide con la
simulada directamente con un error relativo de 2·10⁻⁷ (0°) y 7·10⁻⁵ (12°), y la FMC es recíproca
(S_ij = S_ji) a 7·10⁻⁶ (`tests/test_fmc.py`).

### Ruido electrónico

Ruido gaussiano blanco filtrado a la banda (filtro gaussiano centrado en f0 con −6 dB en f0 ± f0/2),
de valor eficaz σ fijo por canal, añadido **a cada disparo físico**: a cada pareja emisor-receptor en
STA, a cada receptor y ángulo en onda plana (no a las 32 trazas que se suman para sintetizarla: eso
inflaría el ruido √32), a cada posición del monoestático. El nivel se define como SNR por canal =
amplitud de la envolvente del eco cortical en un disparo de un elemento λ/2 central, recibido por ese
mismo elemento, en la pierna sana de referencia del cuerpo normal, frente a σ: 40 dB (electrónica
buena), 20 dB y 0 dB (electrónica barata con un elemento pequeño). El mismo σ absoluto se aplica al
cuerpo obeso, donde el eco llega más débil. Tres realizaciones de ruido por fantoma (el *bootstrap*
remuestrea fantomas enteros, no realizaciones). A baja frecuencia la referencia es la amplitud del
primer arribo en el receptor central de la pierna sana de referencia (40 y 20 dB).

### Estadísticas de detección (sin verdad de terreno)

Ninguna usa la máscara de la lesión. Se calculan en |x| ≤ 3 mm y "mayor = más sospechoso":

- **caída**: se sigue la línea cortical (máximo de cada columna en ±2.5 mm de la profundidad mediana
  del eco más fuerte); caída = mediana de su amplitud menos su mínimo local (media de 3 columnas), dB.
- **escalón**: mayor salto de la profundidad de la línea entre las medianas de 1 mm a cada lado, tras
  quitar la recta de Theil-Sen (la inclinación del hueso).
- **sub. local**: exceso local de energía de 0.75 a 4 mm bajo la línea (relativa a la cortical) frente
  a su mediana.
- **sub. global**: mediana de esa energía relativa (sólo comparable entre personas).
- A-scan fijo: −amplitud del eco más fuerte (caída) y energía bajo él (sub. global).
- Transmisión axial (AT): tiempo del primer arribo (FAS) por umbral (25 % del máximo antes de la onda
  directa y ≥ 4σ; el pulso debe caber 1.5 periodos antes de la onda directa, como en el ejemplo 11).
  - **tiempo** (la estadística principal): con 1 receptor, su tiempo de vuelo; con varios, el primer
    arribo de sus señales **apiladas** (cada traza se retrasa para alinear una onda de velocidad
    aparente c y se promedian; c se elige entre 2400 y 4200 m/s por la mayor amplitud), referido al
    receptor más lejano. El ruido independiente de n canales baja √n: es lo que un equipo con varios
    receptores gana frente a uno solo.
  - **lentitud** (pendiente tiempo-distancia por regresión, inversa de la velocidad aparente),
    **salto** (mayor salto de los residuos, ≥ 4 receptores) y **atenuación** (pendiente de la amplitud).
  - Bidireccional = media de los dos emisores (compensa la inclinación del hueso).
  - La estadística *tiempo* apilada se añadió **después** de ver los primeros resultados (la pendiente
    no ve una fractura situada entre el emisor y los receptores, porque sólo retrasa todas las llegadas
    por igual). No se ajustó con las etiquetas, pero es una elección a posteriori y hay que leerla así.
- Transmisión a través (TT): con 1 receptor (10 mm), el adelanto del primer arribo y la pérdida de
  energía; con 2, su diferencia; con ≥ 4, la mayor desviación del tiempo de vuelo respecto a una recta a
  lo largo del hueso (anomalía local, independiente del grosor total).

Las estadísticas de caída/escalón se asignan de antemano a las lesiones superficiales y las
subcorticales a las internas y la contusión; las tablas completas tienen las cuatro.

AUC de Mann-Whitney sobre todas las parejas (positivo, negativo) y realizaciones de ruido; IC 95 % por
*bootstrap* de fantomas (2000 remuestreos, percentiles). Con separación perfecta el *bootstrap* da
[1.00-1.00], que es optimista: con 6 frente a 6 la probabilidad de separar perfectamente por azar es
1/924, pero la de que el AUC poblacional sea < 0.8 no es despreciable.

Métricas con verdad de terreno, para explicar el porqué: resolución lateral (FWHM a −6 dB), lóbulo
secundario máximo y ganancia de SNR de cada configuración sobre un dispersor puntual en agua a 6 mm;
y, para la lesión central de cada tipo en el cuerpo normal, el cambio de la imagen en la región de la
lesión respecto a la misma pierna sin lesión (gemela), y D_ruido = ese cambio − el que produce el ruido.

### Baja frecuencia

- **AT** a 0.5 y 1 MHz: emisor de λ/2 sobre la piel en x = −45 mm y otro en +45 mm (dos simulaciones),
  41 receptores cada 1 mm en ±20 mm (distancias 25-65 mm). A menos de ~49 mm del emisor el FAS no se
  separa de la onda directa por la piel (medido: detectable en el 0-70 % de las piernas a 25-41 mm y en
  el 100 % desde 49 mm a 0.5 MHz), así que cada emisor usa sus 17 receptores lejanos (49-65 mm): de
  ellos se toman 1 (el más lejano), 2, 4, 8 o 17 repartidos. Unidireccional = 1 emisor + n receptores;
  bidireccional = 2 emisores + 2n receptores (cada emisor con los n lejanos a él).
- **TT** a 0.5 MHz: línea emisora de ~25 mm (como el ejemplo 11) en la cara anterior y 25 receptores
  puntuales cada 1 mm en ±12 mm bajo la piel posterior; receptores de 10 mm (1), 12 (2), 6 (4), 3 (8)
  o 1 mm (25).
