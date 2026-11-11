# Comparación cruzada de rayos X / TAC (ASTRA, SpekPy, XCIST)

Script: `benchmarks/cross_xray.py` (secciones `astra`, `spekpy`, `xcist`; las dos primeras tardan ~1 min cada una,
la tercera ~2 min con Numba y un hilo). Referencias guardadas en `benchmarks/data/cross_xray/*.npz` (0.75 MB) y test
rápido sin las herramientas instaladas: `tests/test_cross_xray.py` (12 tests, ~9 s). Cada herramienta se instaló sólo en
un entorno aparte (`venv_<nombre>` en el directorio de trabajo); ninguna está en `pyproject.toml`.

| herramienta | versión | resultado de la instalación |
|---|---|---|
| ASTRA Toolbox | 2.5.0 (`pip install astra-toolbox`) | funciona en CPU (`astra.use_cuda() == False`) |
| SpekPy | 2.5.4 (`pip install spekpy`) | funciona |
| XCIST (gecatsim) | 1.6.9 | `pip install gecatsim` **falla** (los scripts de compilación del sdist tienen finales de línea CRLF: `BuildLinux64: line 2: $'\r': command not found`); funciona tras descargar el sdist, convertir a LF los ficheros de `clib_build` y `reconstruction` e instalar desde la carpeta |
| TIGRE | pytigre | **no instala**: `OSError: The CUDA path could not be located in $PATH or $CUDAHOME` (necesita `nvcc`; esta máquina no tiene GPU ni CUDA). No se ha comparado |

## 1. ASTRA Toolbox 2.5.0 (CPU)

**Casos.** Malla de 64×64 con `dx` = 3.125 mm y Shepp-Logan (`xray_phantoms.shepp_logan(0.08, 20)`, valor
máximo 20 1/m; hay sinograma analítico exacto), y la sección transversal de pierna (`cross_section_limb("pierna",
dx=1.5e-3)`, 66×66, mapa de atenuación a 60 keV; sin solución analítica). Paralelo: 120 vistas en 180°, detector de
`detector_axis` (paso `dx`). Abanico **plano**: 60 vistas en 360°, `sid` 0.6 m, `sdd` 1.1 m, `FanGeometry.covering`.

**Convenciones** (todas medidas comparando sinogramas, no supuestas):

- *Ángulos, paralelo*: el ángulo de ASTRA es el de wavesim (mismo signo, sin desfase) y el eje del detector tiene el
  mismo sentido. Con el detector invertido el error sube a ~100 %.
- *Ángulos, abanico*: el ángulo de ASTRA es el de wavesim **+180°** y el eje del detector está **invertido**
  (`astra_angulo = theta + pi`, sinograma de wavesim `[:, ::-1]`). En paralelo el desfase de 180° no se nota (rayos
  antiparalelos); en abanico sí, porque cambia de qué lado está la fuente.
- *Centro de rotación*: el centro geométrico de la malla (entre los píxeles centrales si `n` es par), en ambos.
- *Escala de píxel*: ASTRA trabaja con píxel de lado 1: sus integrales de línea son valor × longitud en píxeles, hay
  que multiplicar por `dx` para tener valor × metros. Las imágenes (1/m) no cambian de escala; el valor medio de la
  FBP coincide a 0.1 %.
- *Núcleo del proyector*: el de wavesim (muestreo bilineal a lo largo del rayo, paso `dx/1.5`) se parece al de tira
  (`strip`, área de intersección) de ASTRA, no al lineal (`linear`/`line_fanflat`); se muestran ambos.
- *Filtro de la FBP*: `ram-lak` y `shepp-logan` con el mismo nombre en ambos (en wavesim `window="ram-lak"` es la
  rampa sin ventana).
- *Detector curvo*: ASTRA sólo tiene detector plano (`fanflat`); el abanico curvo equiangular de wavesim no tiene
  contraparte y no se compara.
- *Iteraciones*: `SART` de ASTRA cuenta pasos de una vista (una iteración de wavesim son `n_vistas` pasos de ASTRA);
  `os_sart(subset_size=1)` es SART y `subset_size=n_vistas` es SIRT (mismas fórmulas, normalización por suma de fila
  y de columna), que es lo que se compara.

**Proyector** (diferencia relativa en norma 2 entre sinogramas de la misma imagen rasterizada; `exacto` = integral
analítica del fantoma continuo, que incluye el error de pixelar el objeto y por eso es ~7 % para cualquiera):

| objeto | geometría | wavesim-ASTRA tira | wavesim-ASTRA lineal | ASTRA lineal-tira | wavesim-exacto | ASTRA tira-exacto |
|---|---|---|---|---|---|---|
| Shepp-Logan | paralelo | **0.36 %** | 0.93 % | 0.91 % | 7.17 % | 7.18 % |
| Shepp-Logan | abanico plano | **0.40 %** | 3.46 % | 3.47 % | 6.87 % | 6.86 % |
| pierna | paralelo | **0.09 %** | 0.21 % | 0.20 % | - | - |
| pierna | abanico plano | **0.10 %** | 0.65 % | 0.65 % | - | - |

Frente al núcleo de tira, wavesim difiere menos de 0.4 % (0.1 % en la pierna, más suave); las diferencias entre los
dos núcleos de ASTRA son mayores (hasta 3.5 %) que las de wavesim con el de tira.

**FBP paralelo** (mismos datos: sinograma analítico en Shepp-Logan, sinograma de wavesim en la pierna; RMSE en 1/m
dentro del objeto):

| objeto | filtro | núcleo ASTRA | RMSE wavesim-ASTRA | RMSE wavesim-objeto | RMSE ASTRA-objeto | wavesim-ASTRA / ASTRA-objeto |
|---|---|---|---|---|---|---|
| Shepp-Logan | ram-lak | tira | 0.055 | 1.286 | 1.284 | 4.3 % |
| Shepp-Logan | shepp-logan | tira | 0.044 | 1.353 | 1.348 | 3.2 % |
| pierna | ram-lak | tira | 0.076 | 2.915 | 2.910 | 2.6 % |
| pierna | shepp-logan | tira | 0.064 | 3.266 | 3.258 | 2.0 % |
| Shepp-Logan | ram-lak | lineal | 0.340 | 1.286 | 1.333 | 25.5 % |
| pierna | ram-lak | lineal | 0.435 | 2.915 | 2.790 | 15.6 % |

La retroproyección de la FBP de ASTRA usa el adjunto del proyector elegido; con el de tira coincide con wavesim al
2-4 % del error frente al objeto, con el lineal se separa 12-25 %. Los errores frente al objeto son prácticamente
iguales en las dos herramientas (la diferencia de RMSE respecto al objeto es < 4 %).

**FBP de abanico**: no comparable con ASTRA en CPU: `FBP currently only supports parallel projection geometries`
(la FBP de abanico de ASTRA es `FBP_CUDA`, que exige GPU). La FBP de abanico de wavesim sólo se valida frente al
objeto y frente al haz paralelo (`docs/validacion/rayos_x.md`).

**SART y SIRT en abanico plano** (60 vistas, mismas iteraciones y mismo sinograma; RMSE en 1/m; núcleo de tira):

| objeto | algoritmo | iter. | RMSE wavesim-ASTRA | RMSE wavesim-objeto | RMSE ASTRA-objeto |
|---|---|---|---|---|---|
| Shepp-Logan | SART | 1 | 0.059 | 1.483 | 1.481 |
| Shepp-Logan | SART | 5 | 0.137 | 1.589 | 1.587 |
| Shepp-Logan | SART | 10 | 0.189 | 1.744 | 1.743 |
| Shepp-Logan | SART (no neg.) | 5 | 0.098 | 1.118 | 1.118 |
| Shepp-Logan | SIRT | 10 | 0.024 | 2.698 | 2.698 |
| Shepp-Logan | SIRT | 30 | 0.050 | 1.865 | 1.864 |
| pierna | SART | 5 | 0.181 | 2.626 | 2.626 |
| pierna | SART | 10 | 0.215 | 2.477 | 2.480 |
| pierna | SIRT | 10 | 0.061 | 6.590 | 6.590 |
| pierna | SIRT | 30 | 0.119 | 4.815 | 4.815 |

Con el núcleo de tira, wavesim y ASTRA dan reconstrucciones que difieren como máximo un 11 % del error frente al objeto
(SART) y menos del 3 % (SIRT); la diferencia crece despacio con las iteraciones porque el SART es sensible a las
diferencias del operador. Con el núcleo de línea de ASTRA la diferencia es del orden del propio error (0.65-1.2
frente a 1.5-1.7 en Shepp-Logan) y no es una discrepancia del algoritmo sino del operador. La restricción de no negatividad se
comporta igual (`MinConstraint=0` de ASTRA).

**Lectura.** El proyector, la FBP paralela, el SART y el SIRT de wavesim coinciden con ASTRA al nivel de las
diferencias entre núcleos de proyección de la propia ASTRA, y se ve que el operador de wavesim se parece al de área
(tira). No hay una discrepancia que exija corregir wavesim. Limitaciones: ASTRA no cubre el detector curvo ni la FBP
de abanico en CPU, y TIGRE no se pudo ejecutar, así que el abanico curvo y su FBP no tienen referencia cruzada; los
dos fantomas son pequeños (64-66 píxeles) y sólo hay un par de geometrías.

## 2. SpekPy 2.5.4: espectros no usados en el ajuste

`wavesim.em.xray.tube_spectrum(kvp, Al, cu_filter_mm=Cu, characteristic=True, anode_angle_deg=th)` frente a
`Spek(kvp, th).filter("Al", t).filter("Cu", t)` (HVL1 en mm Al, energía media en keV, espectro de fluencia
agrupado en celdas de 1 keV de 10 keV a `kvp`, normalizado a 1; los fotones de SpekPy por debajo de 10 keV se
descartan porque wavesim no los tiene). "L1" es $\sum_i |p_i^{wavesim}-p_i^{SpekPy}|$ (0 = idéntico, 2 = disjuntos). Los
seis primeros casos son los del ajuste de `_DEPTH_FRACTION` y la fracción de líneas K (de `rayos_x.md`); los
demás no intervinieron en él.

| tipo | kVp | Al (mm) | Cu (mm) | ánodo (°) | HVL SpekPy | E media SpekPy | error HVL | error E media | L1 |
|---|---|---|---|---|---|---|---|---|---|
| ajuste | 50 | 2.5 | 0 | 12 | 1.750 | 32.2 | -6.6 % | -2.2 % | 0.080 |
| ajuste | 70 | 2.5 | 0 | 12 | 2.397 | 39.4 | -5.2 % | -2.0 % | 0.056 |
| ajuste | 80 | 2.5 | 0 | 12 | 2.734 | 42.9 | -3.3 % | -1.0 % | 0.050 |
| ajuste | 100 | 2.5 | 0 | 12 | 3.458 | 49.1 | +1.0 % | +0.3 % | 0.100 |
| ajuste | 120 | 2.5 | 0 | 12 | 4.219 | 54.4 | +4.3 % | +0.7 % | 0.144 |
| ajuste | 120 | 4.0 | 0 | 12 | 5.305 | 57.1 | +3.4 % | +0.3 % | 0.154 |
| predicción | 60 | 2.5 | 0 | 12 | 2.082 | 36.0 | -6.0 % | -2.2 % | 0.067 |
| predicción | 90 | 2.5 | 0 | 12 | 3.092 | 46.2 | -1.1 % | -0.2 % | 0.067 |
| predicción | 110 | 2.5 | 0 | 12 | 3.843 | 51.9 | +2.6 % | +0.5 % | 0.120 |
| predicción | 140 | 2.5 | 0 | 12 | 4.989 | 59.1 | +6.6 % | +0.7 % | 0.178 |
| predicción | 150 | 2.5 | 0 | 12 | 5.372 | 61.4 | +7.4 % | +0.7 % | 0.194 |
| predicción | 100 | 1.0 | 0 | 12 | 2.068 | 45.1 | +0.4 % | +0.5 % | 0.084 |
| predicción | 100 | 6.0 | 0 | 12 | 5.416 | 54.3 | +0.9 % | -0.1 % | 0.123 |
| predicción | 150 | 10.0 | 0 | 12 | 9.059 | 70.7 | +0.8 % | -1.3 % | 0.211 |
| predicción | 100 | 2.5 | 0.1 | 12 | 5.460 | 54.4 | +0.2 % | -0.2 % | 0.122 |
| predicción | 100 | 2.5 | 0.3 | 12 | 7.585 | 60.0 | -0.7 % | -0.8 % | 0.145 |
| predicción | 120 | 2.5 | 0.2 | 12 | 7.728 | 63.2 | +0.5 % | -0.8 % | 0.172 |
| predicción | 120 | 2.5 | 0.5 | 12 | 9.767 | 69.4 | -1.5 % | -1.7 % | 0.177 |
| predicción | 120 | 2.5 | 1.0 | 12 | 11.282 | 75.5 | -2.5 % | -2.1 % | 0.160 |
| predicción | 150 | 2.5 | 0.5 | 12 | 11.034 | 77.5 | -2.0 % | -2.3 % | 0.200 |
| predicción | 150 | 2.5 | 1.0 | 12 | 12.608 | 85.1 | -2.8 % | -2.5 % | 0.164 |
| predicción | 100 | 2.5 | 0 | 10 | 3.576 | 49.5 | +0.8 % | +0.2 % | 0.105 |
| predicción | 100 | 2.5 | 0 | 15 | 3.330 | 48.7 | +1.2 % | +0.4 % | 0.093 |
| predicción | 100 | 2.5 | 0 | 20 | 3.189 | 48.2 | +1.6 % | +0.6 % | 0.085 |
| predicción | 120 | 2.5 | 0 | 10 | 4.379 | 54.9 | +4.1 % | +0.5 % | 0.152 |
| predicción | 120 | 2.5 | 0 | 20 | 3.847 | 53.3 | +4.8 % | +1.1 % | 0.122 |
| predicción | 80 | 2.5 | 0 | 20 | 2.559 | 42.2 | -2.4 % | -0.6 % | 0.037 |
| predicción | 140 | 2.5 | 0 | 10 | 5.189 | 59.7 | +6.3 % | +0.5 % | 0.187 |
| predicción | 140 | 2.5 | 0 | 20 | 4.512 | 57.7 | +7.4 % | +1.3 % | 0.151 |

Resumen: ajuste, error medio $|\Delta$HVL$|$ 4.0 % (máx. 6.6 %), energía media 1.1 % (máx. 2.2 %), L1 medio 0.097. **Predicción
(23 casos)**: HVL 2.8 % (máx. 7.4 %), energía media 0.9 % (máx. 2.5 %), L1 medio 0.135 (máx. 0.211).

**Lectura.** Fuera del ajuste el modelo no es peor que dentro: la HVL queda a menos del 3 % en la mayoría de los casos
con otras filtraciones, con cobre (la tabla de Cu era una estimación recordada y aun así sale a -2.8...+0.5 %) y con otros
ángulos de ánodo (la tendencia con el ángulo se reproduce: de 10° a 20° la HVL de SpekPy baja 10.8 % y el error de wavesim
sólo cambia de +0.8 a +1.6 %). Los puntos débiles son sistemáticos en el kVp: **+6.3...+7.4 % a 140-150 kVp** (fuera del
rango 50-120 kVp del ajuste: el modelo de líneas K y de profundidad del ánodo se calibró a 100 kVp) y -6 % a 50-60 kVp. El
objetivo orientativo de ±5 % de `rayos_x.md` no se cumple ahí. El error L1 del espectro (0.04-0.21, es decir, 2-10 % de
los fotones en la celda equivocada) es mayor a kVp altos y con filtración fuerte: no es sólo la HVL lo que falla, la
forma del espectro filtrado difiere (la causa no se ha investigado). Es una medida de predicción real, y se mantiene como límite de validez del espectro (50-120 kVp, ±6 %).

## 3. XCIST 1.6.9 (CPU): SPR de un cilindro de agua

XCIST sí corre en CPU. Se simuló una vista de un cilindro de agua de 200 mm de longitud (analítico), `sid` 540 mm, `sdd`
950 mm, detector curvo de 64 filas × 1 mm (36.4 mm en el isocentro), espectro `tungsten_tar7.0_120_filt.dat` con 3 mm de
Al, sin bowtie y sin ruido, con y sin `Scatter_ConvolutionModel`. La SPR es `exp(prep_sin - prep_con) - 1` en la fila
central. En wavesim, `simulate_scatter` con el mismo abanico curvo, el **mismo espectro** (leído del fichero de XCIST y
filtrado igual), `slice_thickness` = 36.4 mm y 2·10⁷ fotones; SPR en **energía** (E_dispersa / E_primaria detectadas), como un
detector integrador de energía. Promedio en una ventana de ±5 mm.

| diámetro (mm) | x isocentro (mm) | SPR XCIST | SPR wavesim | wavesim / XCIST |
|---|---|---|---|---|
| 100 | 0 | 4.8 % | 6.3 % | 1.31 |
| 200 | 0 | 8.9 % | 26.5 % | 2.98 |
| 200 | 40 | 8.3 % | 21.4 % | 2.59 |
| 200 | 80 | 5.8 % | 7.6 % | 1.31 |
| 300 | 0 | 12.5 % | 76.4 % | 6.13 |
| 300 | 80 | 10.9 % | 42.1 % | 3.85 |
| 350 | 0 | 14.2 % | 117.3 % | 8.29 |
| 350 | 80 | 12.9 % | 74.3 % | 5.75 |

**Lectura honesta.** Las dos herramientas **no concuerdan** (factor 1.3 a 8) y XCIST no es una referencia física
independiente: su modelo de dispersión es un núcleo de convolución genérico (su código dice "simplified kernel based
scatter model, will have roughly 15% SPR for 35-cm water phantom, 120 kVp, 40-mm beam width"), con una constante 0.025
fija y un factor de escala "1 apropiado para 64 mm de detector y 20 cm de agua". Es decir, está calibrado a un caso y
da SPR casi independiente del diámetro (4.8 → 14 % de 10 a 35 cm), mientras que el transporte físico da una SPR que crece
con el diámetro mucho más rápido (la primaria cae como $e^{-\mu d}$ y la dispersa casi no). Dos puntos de la literatura
orientan sobre cuál es más verosímil, **sin verificar** (cifras de resúmenes encontrados por búsqueda web, no leídos
en el artículo): un ajuste Monte Carlo de CBCT da SPR 0.44 con campo de 2 cm y 3.7 con 26 cm para un cilindro de 30.6 cm
([Bootsma et al., Med. Phys. 2015](https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.4903260)); wavesim da 0.76 para 30 cm
con 3.6 cm de haz (el orden de magnitud y la tendencia son los de la literatura; XCIST con 0.12 queda lejos). Por lo
tanto el resultado indica que el modelo de XCIST subestima la dispersión de pacientes grandes, no que el Monte Carlo de
wavesim esté validado por XCIST. Limitaciones de la comparación: el cilindro de wavesim es infinito en y (XCIST: 200
mm), el detector de wavesim es ideal (sin eficiencia de absorción ni filtro de grafito), el cociente central a 35 cm
tiene ruido estadístico del 10-20 % (pocos primarios), y no se ha medido el *cupping* (XCIST sólo da un SPR de una
vista).
