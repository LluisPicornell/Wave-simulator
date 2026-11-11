# Comparación cruzada del solver de Maxwell con gprMax

Compara `wavesim/em/fdtd_tm.py` (banda estrecha, `MaxwellTMSolver`) y `wavesim/em/fdtd_dispersive.py` (banda ancha con
polos de Debye, `BroadbandMaxwellTMSolver`) con un FDTD de referencia ejecutado en CPU. Todo sale de
`benchmarks/cross_maxwell.py`; las trazas de gprMax están en `benchmarks/data/cross_maxwell/` (0.6 MB) y
`tests/test_cross_maxwell.py` las compara con wavesim sin tener gprMax instalado.

## Herramientas

| Herramienta | Resultado |
|---|---|
| **gprMax 4.0.1** | Usada. Rueda precompilada de PyPI (Cython + OpenMP) instalada en un entorno aparte (`venv_gprmax`): `pip install gprMax==4.0.1`. No se compiló desde el árbol de GitHub (`git ls-remote` a `gprMax/gprMax` funciona; no hizo falta). Modo 2D `TMz`, TF/SF de onda plana (`#plane_wave_axial`), materiales Debye multipolo y monitor SAR en frecuencia. |
| Meep | **No instalable aquí.** `pip install meep` descarga `meep 1.0.6`, un gestor de tareas sin relación (autor Håård, bitbucket); `pip download pymeep` no encuentra ninguna distribución, y no hay conda. No se ejecutó. |
| openEMS | No se intentó: gprMax cubrió los tres casos. |

Los mismos materiales en los dos códigos: los polos de `materials.dispersive.tissue_debye_model` (banda 0.5-6 GHz en la
losa, 1-4 GHz en los casos 2D), traducidos a `#material` + `#add_dispersion_debye`. Se compara el solver, no el ajuste
(el ajuste frente a Gabriel difiere 0.015 dB y 0.54° en la losa). Wavesim en banda estrecha recibe la permitividad
compleja del ajuste a 2.45 GHz (`eps_map`), no la de Gabriel.

## Convenciones (verificadas)

- **Polarización**: wavesim es TM con `Ey` perpendicular al plano `x-z`; gprMax, `TMz` con `Ez` perpendicular a `x-y`
  (`psi = 90`). Correspondencia `(x, z) -> (x, y)`, `Ey -> Ez`. Propagación hacia `+z`/`+y` en ambos.
- **Fase**: los dos usan `e^{+jwt}`; los fasores salen de `sum x(t) e^{-jwt} dt`. Sólo se comparan cocientes con una
  simulación de referencia sin objeto, que cancelan el origen de tiempos, la normalización de la fuente y el desfase
  `dt/2` entre `E` y `H`. Con T de la losa, `T = E(con losa) / E(sin losa)` en el mismo punto.
- **Normalización**: `E` por unidad del campo incidente en el centro del objeto sin el objeto (en agua con pérdidas en
  la pierna). Así es irrelevante dónde esté la fuente y que gprMax inyecte con TF/SF y wavesim con una lámina de corriente.
- **Nodos de Yee** (lo más delicado). Wavesim: `Ey` en el centro de la celda `[j, i]`. gprMax: `Ez` en el nodo `(i, j)`.
  Con `#box ... n`, la caja **incluye el nodo del borde superior**: con una losa sin pérdidas de 20 celdas, una caja
  de `p1` a `p1 + 20 dx` se comporta como una losa de 21 celdas (error de fase 0.43° a 0.5 GHz frente a la solución
  exacta de 20) y con `p2` una celda menos da error 0.00°. Mi primera comparación de la losa dio por eso -12.5° a 6 GHz
  en gprMax y 5.8° en wavesim; era la geometría, no el solver. Para el 2D se pintan las tiras fila a fila
  (`painter_boxes`, con test) para que el nodo `(i, j)` tenga exactamente la etiqueta de la celda `[j, i]`: los dos
  códigos resuelven el mismo problema en el espacio de índices y se comparan índice a índice; la interfaz queda en el
  punto medio entre dos nodos en los dos. Las etiquetas del monitor SAR de gprMax (celdas) coinciden con las de
  wavesim celda a celda (comprobado: 0 diferencias).
- **SAR de gprMax**: da `E` en el centro de la celda (media compleja de los cuatro nodos) y la absorción `P = 1/2 sigma_eff
  |E|^2` con la conductividad efectiva de los polos (se comprobó que `P` es proporcional al `|E|^2` de wavesim, con
  cociente constante en las 1251 celdas del cilindro). Para comparar se promedia igual `Ey` de wavesim (2x2) y se usa
  la etiqueta de la celda `[j, i]`. Sin esa colocación hay medio paso de malla de desfase entre los dos
  esquemas: se da aparte.
- **Losa en 3D**: `#symmetry_boundary` no admite el modo 2D de gprMax (`cannot currently be used in 2D mode`). Sin
  paredes la TF/SF de ancho finito da un haz, no una onda plana (`|T| ~ 0.7-0.95` en vez de `0.01-0.3`). Se usa una
  celda unidad 3D (4x N x 4 celdas, paredes PMC en `x` y PEC en `z`, PML en `y`) con `#plane_wave_axial` a todo el
  ancho. La forma `#plane_wave_vector` con paredes de simetría diverge (NaN).
- **TF/SF en 2D**: el dominio útil debe quedar dentro de la caja TF/SF (con la línea hasta la última fila, la mitad
  inferior estaba fuera y daba un 22 % de error); wavesim emite desde una lámina y no tiene ese límite. La TF/SF ocupa
  todo el ancho (incluida la PML lateral) como la lámina de wavesim. Amplitud 1 con `ricker 1 f`.
- PML: 15 celdas (CFS-PML) en gprMax, 20 (CPML) en wavesim. Paso: gprMax usa su límite CFL (`dt = dx / (c sqrt 2)` en 2D,
  `dx / (c sqrt 3)` en la celda 3D); wavesim, 0.95 de ese límite en 2D.

## 1. Losa piel 2 / grasa 5 / músculo 15 / cortical 6 mm, 0.5-6 GHz

`|T(f)|` y fase del campo transmitido (con losa / sin losa). Error de cada código frente a la matriz de transferencia
exacta con los mismos polos (`analytic.insertion_transmission`); en la última columna, gprMax frente a wavesim.

dx = 0.5 mm:

| f (GHz) | \|T\| exacto | wavesim | gprMax | gprMax - wavesim |
|---|---|---|---|---|
| 0.5 | -12.84 dB | -0.002 dB / 0.00° | -0.002 dB / 0.00° | 0.000 dB / 0.00° |
| 1.27 | -9.58 dB | -0.012 dB / -0.06° | -0.011 dB / -0.06° | +0.001 dB / 0.00° |
| 2.36 | -12.09 dB | -0.048 dB / -0.50° | -0.046 dB / -0.50° | +0.002 dB / 0.00° |
| 3.22 | -17.20 dB | -0.136 dB / -1.19° | -0.133 dB / -1.19° | +0.003 dB / 0.00° |
| 4.4 | -25.53 dB | -0.323 dB / -2.73° | -0.317 dB / -2.73° | +0.007 dB / 0.00° |
| 6 | -36.67 dB | -0.773 dB / -5.77° | -0.760 dB / -5.77° | +0.013 dB / 0.00° |

Máximos en toda la banda (41 frecuencias): wavesim 0.773 dB y 5.77°, gprMax 0.760 dB y 5.77°, **gprMax frente a wavesim
0.013 dB y 0.00°**. Con dx = 0.25 mm: 0.189 dB / 1.43° y 0.186 dB / 1.43° (error de dispersión numérica del esquema de
Yee, de orden 2: se divide por 4 al duplicar la resolución), y 0.003 dB / 0.00° entre los dos.

Lectura: en este caso los dos esquemas son el mismo salvo redondeo; el error frente a la solución exacta es el de la malla
(a 6 GHz hay 14 puntos por longitud de onda en el músculo), no del solver ni del ADE de Debye.

## 2. Cilindro de músculo (radio 20 mm) en aire, 2.45 GHz

Error L2 relativo de `Ez` a lo largo de la línea central en la dirección de incidencia (`z`) y en la transversal (`x`),
dentro y fuera del cilindro. Mie: `em.analytic.mie_cylinder_tm` con la permitividad del ajuste (52.74 - 12.75j).

| dx | par | z dentro | z fuera | x dentro | x fuera |
|---|---|---|---|---|---|
| 1 mm | wavesim estrecha frente a Mie | 3.40 % | 0.98 % | 4.26 % | 0.69 % |
| 1 mm | wavesim ancha frente a Mie | 3.49 % | 0.63 % | 4.14 % | 0.75 % |
| 1 mm | gprMax frente a Mie | 3.44 % | 0.62 % | 4.11 % | 0.74 % |
| 1 mm | gprMax frente a wavesim estrecha | 0.57 % | 0.54 % | 0.53 % | 0.38 % |
| 1 mm | gprMax frente a wavesim ancha | 0.11 % | 0.01 % | 0.11 % | 0.01 % |
| 0.5 mm | wavesim estrecha frente a Mie | 2.23 % | 0.79 % | 2.16 % | 0.39 % |
| 0.5 mm | wavesim ancha frente a Mie | 2.17 % | 0.40 % | 2.01 % | 0.30 % |
| 0.5 mm | gprMax frente a Mie | 2.16 % | 0.40 % | 2.01 % | 0.30 % |
| 0.5 mm | gprMax frente a wavesim estrecha | 0.52 % | 0.50 % | 0.48 % | 0.35 % |
| 0.5 mm | gprMax frente a wavesim ancha | 0.03 % | 0.00 % | 0.03 % | 0.00 % |

Lectura: el error frente a Mie (3-4 % a 1 mm, 2 % a 0.5 mm) es de la escalera del disco y es el mismo en los tres
códigos. gprMax y el solver de banda ancha coinciden a 0.03-0.11 % (decrece con dx). El de banda estrecha se separa un
0.5 % de los otros dos de forma constante con la malla (también fuera del cilindro, en la línea hasta el borde): no es de
la geometría ni de los materiales (mismo `eps_map` del ajuste). La causa no está determinada; sospecho del
transitorio de la onda continua (arranque suave de 3 ciclos y 8 periodos de DFT) pero no lo he verificado.

## 3. Pierna (`cross_section_limb("pierna")`, sin porosidad, en agua) a 2.45 GHz

Diferencia relativa de gprMax frente a wavesim por celda, `|a - b| / |b|` en las celdas de la pierna (sin agua) donde
`b` supera el 1 % de su máximo; mapas colocados en el centro de la celda. SAR por (V/m)^2 incidente en el centro.

| dx | wavesim | \|E\| mediana / p95 | SAR mediana / p95 | SAR pico (W/kg): gprMax / wavesim | cociente |
|---|---|---|---|---|---|
| 1 mm | estrecha | 0.49 % / 1.30 % | 1.05 % / 2.39 % | 0.01945 / 0.01932 | 0.993 |
| 1 mm | ancha | 0.16 % / 0.37 % | 0.55 % / 0.75 % | 0.01945 / 0.01960 | 1.007 |
| 0.5 mm | estrecha | 0.62 % / 0.88 % | 1.24 % / 1.76 % | 0.01624 / 0.01642 | 1.011 |
| 0.5 mm | ancha | 0.04 % / 0.09 % | 0.13 % / 0.18 % | 0.01624 / 0.01627 | 1.002 |

El SAR pico está en la misma celda en los tres códigos (16, 53 con 1 mm; 43, 111 con 0.5 mm, en la piel del lado
iluminado). Sobre todo el dominio (incluida el agua) las cifras son parecidas (p. ej. 0.5 mm, ancha: 0.06 % / 0.13 % en
|E|). Sin colocar (|E| de wavesim en el nodo frente al de gprMax en el centro de la celda) la diferencia es de
**2.3 % mediana y 8.5 % p95 a 0.5 mm (4.3 % y 16.9 % a 1 mm)**: es el medio paso de malla entre los dos esquemas, no un
error; el SAR pico nativo de wavesim está un 1.6 % por debajo del colocado.

El pico de SAR depende de la malla (0.0195 a 1 mm, 0.0162 a 0.5 mm) porque cae en la piel rugosa de 1.5 mm; los códigos
coinciden entre sí en cada malla, no con una solución exacta (no la hay para la pierna).

## 4. Tiempos (1 hilo, `NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`)

Máquina de 4 núcleos compartida con otros procesos (carga media de hasta 18 durante las medidas), así que los tiempos
son orientativos. Sólo el objeto (la referencia aparte), bucle FDTD, sin el arranque de Python ni la compilación
(Numba con caché y calentamiento previo; gprMax: línea `Time taken`). gprMax usa 1 pulso y da toda la banda; wavesim
estrecha, sólo 2.45 GHz; wavesim ancha, 1 frecuencia pedida (el coste no depende de cuántas).

| Caso | wavesim estrecha | wavesim ancha | gprMax |
|---|---|---|---|
| Cilindro, dx = 1 mm (143x149 celdas de wavesim) | 2.0 s | 4.8 s | 5.1 s |
| Cilindro, dx = 0.5 mm | 16.6 s | 22.8 s | 23.9 s |
| Pierna, dx = 1 mm | 1.8 s | 11.7 s | 9.0 s |
| Pierna, dx = 0.5 mm (260x260 celdas) | 20.1 s | 84.1 s | 45.5 s |
| Losa (objeto + referencia), 0.5 mm | - | 3.6 s | 5.3 s |

En la pierna, la banda estrecha es 2.3 veces más rápida que gprMax y la ancha 1.8 veces más lenta, porque wavesim espera
a que el campo decaiga a 1e-6 en un medio de baja pérdida a baja frecuencia (la banda empieza en 1 GHz), mientras que el
tiempo de gprMax es una ventana fija (14 ns). No es una comparación de coste por celda y paso con la misma ventana.

## Cómo repetirlo

```
python -m venv venv_gprmax && venv_gprmax/bin/pip install gprMax==4.0.1
NUMBA_NUM_THREADS=1 python benchmarks/cross_maxwell.py --gprmax-python venv_gprmax/bin/python   # ~8 min, regenera los datos
python benchmarks/cross_maxwell.py --no-gprmax                                                  # sólo wavesim, con las trazas guardadas
```

## Límites

- Una sola malla de referencia por caso 2D (dos resoluciones); un solo fantoma de pierna (semilla 0, sin porosidad).
- El monitor SAR de gprMax exige `>= 8` celdas por longitud de onda (se pasó 8); en el agua a 2.45 GHz hay 27 a 0.5 mm.
- La causa del 0.5 % de diferencia del solver de banda estrecha sigue sin determinar.
- La rueda 4.0.1 de PyPI puede no ser idéntica a `master` de GitHub (no se comparó).
