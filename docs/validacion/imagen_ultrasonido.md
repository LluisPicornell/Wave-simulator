# Cadena de imagen de ultrasonido: validación

Código: `wavesim/ultrasound/{transducer,beamforming,scan,quality}.py`. Tests rápidos (~12 s):
`tests/test_ultrasound_imaging.py`. Benchmark: `benchmarks/ultrasound_imaging.py`
(`NUMBA_NUM_THREADS=1`; todo salvo la tibia ~10 min, `--parte fractura` aparte).

## Qué se añadió (todo opcional; con los valores por defecto `simulate_bmode` da la misma imagen, comprobado bit a bit)

- Respuesta del elemento (`ElementResponse`, aplicada en emisión y como convolución centrada en recepción), kerf/ancho, ventanas de apodización.
- `simulate_scan`: onda plana, emisión focalizada línea a línea (foco fijo) y STA (un FDTD por emisión).
- Formadores de haz sobre datos de canal: DAS, factor de coherencia (CF), DMAS complejo, SLSC; `TwoLayerSpeed` (Snell/Fermat) para plano y STA.
- Métricas (`quality.py`): FWHM, teoría de apertura y de pulso, SNR de speckle, grano, CNR, contraste, gCNR.
- Referencias (Mallart-Fink 1994, Matrone 2015, Lediju 2011, Rodriguez-Molares 2020) citadas de memoria, **no verificadas en línea**.

## Método

Agua (c = 1500), 2 MHz, 10 puntos/λ, 24×30 mm, 46 elementos a λ/2, pulso de 2 ciclos con ventana de Hann, sin TGC.
Dispersor puntual = una celda con densidad ×4; speckle = ruido de densidad (σ = 5 %) por celda.
Teoría lateral: diagrama monocromático de la apertura dinámica (rect → 1.206·λ·F#; ida y vuelta en STA y focalizada).
Teoría axial: envolvente del pulso de ida y vuelta (h∗h).

## PSF (FWHM a -6 dB de amplitud, mm)

| Modo | Ventana | Lateral medido / teoría | Error | Axial medido (z = 8/14/20) | Error vs h∗h (0.478) |
|---|---|---|---|---|---|
| Onda plana | rect | 1.30 / 1.39 (z = 14) | -6 % | 0.527/0.533/0.549 | +10/+11/+15 % |
| Onda plana | Hann | 2.13 / 2.29 | -7 % | 0.523/0.530/0.546 | +9/+11/+14 % |
| STA completa | rect | 0.92 / 1.01 | -9 % | 0.511/0.548/0.561 | +7/+15/+17 % |
| STA completa | Hann | 1.50 / 1.65 | -9 % | 0.536/0.542/0.555 | +12/+13/+16 % |
| Focalizada F#2 @14 | Hann | 1.67 / 1.86 | -10 % | 0.447 (en el foco) | -6 % |
| Interfaz plana, PW | Hann | — | — | 0.487 | **+1.9 %** |

Lateral: dentro de 12 % en todas las filas (≤ 10 % salvo STA-Hann a 8 mm: -11.8 %); el medido sale ~7 % más estrecho porque el
dispersor 2D (campo ∝ k^1.5) eleva la frecuencia efectiva (centroide 2.16 frente a 2.0 MHz en una medida de control).
Axial: **el dispersor puntual 2D da +9 a +17 % sobre h∗h** (física 2D del dispersor y dispersión temporal del FDTD; con el factor f^1.5 el teórico
sube a 0.515 mm y el error baja a +2…+9 %). Con una interfaz plana, que no tiene esa física, el error es +1.9 %: la cadena de
formación de imagen no ensancha el pulso. **El objetivo ≤ 10 % axial para el dispersor puntual no se cumple a 2 MHz** (sí a 1 MHz en los tests: +6 %).
Sesgo de profundidad del FDTD: -0.06 mm a 8 mm, -0.24 mm a 20 mm (~1 %, dispersión temporal).
STA completa ≈ 0.7× el lóbulo lateral de onda plana, como predice el diagrama sinc².

## Speckle (onda plana, 5 realizaciones)

| Ventana | SNR (Rayleigh 1.913) | Grano axial medido/predicho (mm) | Grano lateral medido/predicho (mm) |
|---|---|---|---|
| Hann | 1.849 ± 0.099 (-3.4 %) | 0.495 / 0.477 (+3.9 %) | 1.381 / 1.485 (-7.0 %) |
| rect | 1.789 ± 0.047 (-6.5 %) | 0.511 / 0.405 (+26 %) | 0.914 / 0.574 (+59 %) |

La predicción del grano con ventana rectangular no es fiable (colas del sinc recortadas en la imagen finita); con Hann coincide a ≤ 7 %.

## Quiste anecoico (3.5 mm de radio, Hann)

| Secuencia | Formador | Contraste (dB) | CNR | gCNR |
|---|---|---|---|---|
| PW 1 ángulo | DAS | -19.7 | 1.77 | 0.91 |
| PW 1 ángulo | CF | -48.9 | 0.68 | 0.76 |
| PW 1 ángulo | DMAS | -21.0 | 0.94 | 0.81 |
| PW 1 ángulo | SLSC | -31.5 | 0.78 | 0.70 |
| PW 5 ángulos | DAS | -25.3 | 1.86 | 0.96 |
| Focalizada F#2 | DAS | -25.8 | 1.75 | 0.96 |
| STA | DAS | -27.7 | 1.95 | 0.97 |
| STA | CF | -55.3 | 1.20 | 0.94 |
| STA | DMAS | -36.9 | 1.45 | 0.96 |
| STA | SLSC | -25.0 | 2.66 | 0.94 |

CF y DMAS oscurecen mucho más el quiste (contraste), pero **no mejoran CNR ni gCNR**: aumentan la varianza del fondo. Sólo SLSC en STA sube el CNR (su imagen no es de amplitud).

## Corrección de dos capas (grasa 1450 sobre 1580, dispersor a 21 mm)

| Reconstrucción | Error z (mm) | Error z sin sesgo numérico (-0.09) |
|---|---|---|
| 1540 fijo | +0.41 | +0.50 |
| 1500 fijo | -0.15 | -0.06 |
| Dos capas | -0.27 | -0.19 |

Con una capa de 11 mm la mejora sobre 1540 es real pero modesta (0.50 → 0.19 mm) y un c0 fijo bien elegido (1500) hace igual; no se ha verificado con capas inclinadas ni más de dos.

## Coste (malla 400×320, 1 hilo): ~3 s por emisión; onda plana 3 s, STA de 46 elementos 138 s, focalizada de 46 líneas 153 s.

