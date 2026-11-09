# Validación del transporte de luz por Monte Carlo

Módulo: `wavesim/em/montecarlo.py` (backend NumPy vectorizado y kernel Numba opcional en
`wavesim/em/_numba_mc.py`). Benchmark que genera estas tablas:

```
NUMBA_NUM_THREADS=1 python benchmarks/optics_validation.py          # completo, ~10 min
NUMBA_NUM_THREADS=1 python benchmarks/optics_validation.py --quick  # ~1.5 min
```

Tests rápidos (umbrales de la forma `k·sem` con semilla fija): `python -m pytest tests/test_montecarlo.py`.

## Modelo

Algoritmo MCML/MCX (Wang, Jacques & Zheng 1995): paquetes con peso, paso libre exponencial con `mu_s`,
Henyey-Greenstein con `g`, Fresnel (no polarizado) en cada cambio de índice, ruleta rusa (w < 1e-4,
supervivencia 1/10) y semilla reproducible. El fantoma 2D (x, z) se extruye infinitamente en y: los fotones
se mueven en 3D, la malla sólo discretiza (x, z).

- **Absorción continua** (no `mu_a/mu_t` por interacción): el peso decae `exp(-mu_a l)` en cada segmento. Misma
  esperanza que MCML con menos varianza; la fluencia sale del estimador de longitud de traza, válido con
  `mu_a = 0`. Por eso el paso libre se muestrea con `mu_s` (con `mu_t` la atenuación balística sería errónea;
  fue un error real durante el desarrollo, detectado por el test de atenuación).
- **Agua y aire = exterior** transparente: cruzarlos es salir (con Fresnel). Índice del tejido 1.4
  (`optics.N_TISSUE`), del exterior el de fondo. No hay tabla verificada de `n` por tejido; se puede pasar
  `n_by_tissue`.
- **g y mu_s**: `OpticalComposition.anisotropy` ya existía (g = 0.9 para todos los tejidos). No se ha
  cambiado (alteraría `mu_total`/`ballistic_transmission`); `mu_s = mu_s'/(1-g)` reproduce la tabla actual de
  `mu_s'` por construcción. Valores de g por tejido de la literatura (típicamente 0.7-0.95) NO se han añadido:
  no he podido verificarlos tejido a tejido.
- **Normalización**: por fotón incidente. Haz ancho `"top"`: fluencia adimensional (W/m² por W/m² incidentes,
  igual que `solve_diffusion`); fuente fina (`"pencil"`, `"isotropic"`): m⁻¹ (por W/m de línea en y).
  Exitancias por columna: su media sobre el haz es la reflectancia/transmitancia total. La reflexión
  especular de entrada se da aparte (`specular`).
- Error estándar: 20-40 tandas independientes, `std/sqrt(K)`.
- Resolución temporal opcional (`time_bins`, `t_max`): `fluence_t`, `reflectance_t`, `transmittance_t` con
  tiempo = camino óptico `sum(n l)/c`.

## (a) Losa de van de Hulst, índice ajustado (n = 1)

mu_a = 10 cm⁻¹, mu_s = 90 cm⁻¹, g = 0.75, d = 0.02 cm (tau = 2, albedo 0.9). Referencia: van de Hulst (1980),
tabla 35, R = 0.09739, T = 0.66096 (verificada por búsqueda: la reproducen un modelo MC
publicado y los resultados de MCML).

| magnitud | MC (± e.e.) | referencia | diferencia |
|---|---|---|---|
| R difusa (Numba, 4·10⁶ fotones) | 0.09758 ± 0.00012 | 0.09739 | +1.6 σ |
| T total (Numba, 4·10⁶) | 0.66084 ± 0.00014 | 0.66096 | -0.9 σ |
| R difusa (NumPy, 10⁶, 2 corridas) | 0.09759, 0.09744 (± 0.00022) | 0.09739 | +0.9 σ, +0.2 σ |
| T total (NumPy, 10⁶, 2 corridas) | 0.66070, 0.66088 (± 0.00026) | 0.66096 | -1.0 σ, -0.3 σ |

Balance `R + T + A + laterales + especular = 1` hasta 1e-7 (el resto es la ruleta rusa).

## (b) Desajuste de índice (n = 1.5 en la losa, 1 fuera)

- **Caso exacto verificable**: losa sin dispersión (mu_a = 4 mm⁻¹, d = 1 mm). En incidencia normal las
  reflexiones múltiples tienen serie cerrada, `T = (1-R)² e^{-μd} / (1 - R² e^{-2μd})`, `R = 0.04`:

| magnitud | MC | exacto | diferencia |
|---|---|---|---|
| T | 0.6182114 | 0.6182114 | 4e-8 |
| R difusa | 0.0165761 | 0.0165760 | 1e-7 |
| R especular | 0.0400000 | 0.0400000 | 1e-15 |

- **Losa dispersiva con n = 1.5 (mismos mu_a, mu_s, g, d que (a))**: R especular 0.04000, R difusa
  0.08676 ± 0.00011, T 0.49297 ± 0.00021, absorbida 0.38027 ± 0.00019. **Sin referencia verificada**: no
  encontré tablas publicadas (Prahl, MCML) con esta combinación exacta (las de Prahl que mencionan las
  búsquedas usan losa 1.4 entre vidrios de 1.5, o semi-infinito isótropo) y no pude comprobar cifras de memoria.
  Sólo se comprueba balance, Fresnel de entrada y el sentido físico (menos T y más A que con índice ajustado).

## (c) Atenuación balística

Transmitancia de fotones que nunca se dispersaron (incluye reflexiones múltiples de Fresnel) frente a
`(1-R)² e^{-μ_t d}/(1 - R² e^{-2 μ_t d})`, 10⁶ fotones:

| n | mu_a, mu_s [m⁻¹], d | MC | referencia | diferencia |
|---|---|---|---|---|
| 1.0 | 300, 500, 2 mm | 0.201652 ± 0.000290 | 0.201897 | -0.8 σ |
| 1.0 | 50, 2000, 2 mm | 0.016559 ± 0.000114 | 0.016573 | -0.1 σ |
| 1.5 | 300, 500, 2 mm | 0.186350 ± 0.000231 | 0.186080 | +1.2 σ |
| 1.5 | 50, 2000, 2 mm | 0.015365 ± 0.000117 | 0.015273 | +0.8 σ |

## (d) Frente a la difusión (`solve_diffusion`), 800 nm, losa semi-infinita de 100 mm, dx = 0.5 mm

Pendiente de log(fluencia) promediada en 40 columnas centrales, relativa a `mu_eff = sqrt(3 mu_a (mu_a+mu_s'))`
(5·10⁵ fotones; la variación entre ventanas es ruido estadístico de la cola profunda):

| tejido | mu_a/mu_s' | ventana [mm] | MC / mu_eff - 1 | difusión / mu_eff - 1 |
|---|---|---|---|---|
| grasa | 0.022 | 8-18 | -0.4 % | -0.1 % |
| grasa | 0.022 | 10-25 | -1.2 % | -0.1 % |
| grasa | 0.022 | 15-30 | +1.3 % | -0.1 % |
| músculo | 0.056 | 8-18 | -0.9 % | -0.1 % |
| músculo | 0.056 | 10-25 | -0.7 % | -0.1 % |
| músculo | 0.056 | 15-30 | -0.7 % | -0.1 % |

Todas dentro de la tolerancia del 5 %. **Cerca de la cara iluminada** (grasa, haz ancho) la difusión no
reproduce el MC: fluencia MC / difusión = 1.27 a 0.25 mm de la superficie, 0.91 a 0.75 mm y 0.92-0.94 a
1-10 mm (la difusión sobrestima ~7 % en el interior; hipótesis no comprobada: el modelo de fuente de la difusión,
un punto isótropo a un recorrido libre de transporte bajo la superficie, frente al haz colimado real). La fluencia en la propia piel, que es lo que importa para la
dosimetría, está un 27 % por encima de la difusión. Reflectancia difusa MC 0.478 ± 0.001, absorbida 0.468
(la difusión no entrega la reflectancia).

**Cerca de una fuente puntual** (fibra isótropa a 10 mm de profundidad en grasa, 5·10⁵ fotones): MC / difusión
= 1.013 (0.5 mm), 0.974 (1 mm), 0.980 (2 mm), 0.986 (3 mm), 0.994 (5 mm), 0.978 (8 mm), 1.009 (12 mm),
con errores estándar del MC de 0.2-1.3 %. Con esta malla (0.5 mm, algo mayor que el recorrido libre de transporte, 0.8 mm) la
difusión y el MC coinciden en 1-3 % cerca de la fuente; no se ha probado con medios de dispersión más débil
ni con malla más fina, donde cabe esperar más discrepancia.

## (e) La pierna (`cross_section_limb("pierna")`) a 800 nm, dx = 1 mm, 2·10⁶ fotones en 5 corridas

Transmisión por la cara inferior, media de la exitancia sobre las columnas con tejido (como
`DiffusionResult.transmission`):

| columnas (espesor de tejido) | MC (± e.e. entre corridas) | difusión | MC / difusión |
|---|---|---|---|
| todas (91) | 2.48e-3 ± 2.8e-5 | 7.15e-4 | 3.47 |
| 0-5 mm (2) | 1.13e-1 ± 1.2e-3 | 3.25e-2 | 3.48 |
| 5-20 mm (2) | 4.2e-5 ± 6.3e-6 | 4.35e-5 | 0.97 |
| 20-40 mm (6) | 2.6e-6 ± 3.8e-7 | 3.65e-6 | 0.70 |
| > 40 mm (81) | 2.3e-8 ± 1.2e-8 | 2.15e-8 | 1.06 |

Lectura honesta: la transmisión media difiere en un factor 3.5, **pero ese factor lo domina casi por completo
una columna del borde de 2 mm de espesor** (la "escalera" de la sección circular), donde la difusión no es
aplicable (espesor < 2 recorridos libres de transporte) y donde el MC aporta ~1.1e-1 de las 91 columnas. En las
columnas de 5-20 mm y > 40 mm MC y difusión coinciden dentro de la incertidumbre estadística (0.97 y 1.06,
esta última con ~50 % de error por 5·10⁻⁸ de transmisión); en las de 20-40 mm el MC está 30 % por debajo
(~2 σ de corridas). En la parte gruesa la transmisión (~10⁻⁸) está al límite de lo medible con 2·10⁶ fotones. La luz
que sale por los lados (20 %) y la reflectada (33 %) dominan el balance; la absorbida es el 44 %.
Limitaciones: la superficie curva en escalera hace que la normal de Fresnel sea siempre un eje, y con
dx = 1 mm la grasa/piel (1-5 mm) se resuelven con pocas celdas; no se ha estudiado la convergencia en dx del MC.

## (f) Rendimiento (un hilo, `NUMBA_NUM_THREADS=1`, CPU compartida)

| caso | NumPy | Numba |
|---|---|---|
| losa van de Hulst (40 celdas, periódica) | 30 000 fotones/s | 540 000 fotones/s |
| pierna 800 nm (100x100, dx = 1 mm) | 98 fotones/s | 10 600 fotones/s |

El backend NumPy es muy lento en medios donde las historias son largas (pocas absorciones): el número de
iteraciones vectoriales lo fija el fotón más longevo. Numba es ~18x (losa) a ~100x (pierna) más rápido. Las
medidas se hicieron con otros agentes usando la CPU: son cotas inferiores aproximadas.

## Tiempos

Benchmark completo: 611 s (Numba, 1 hilo); `--quick`: 99 s. Tests: ver informe.
