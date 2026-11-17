# Validación de rayos X y TAC en 3D

Resultados de `benchmarks/xray3d_validation.py` (completo: unos 2.5 minutos con Numba y un hilo,
`NUMBA_NUM_THREADS=1`; secciones sueltas con `--only chords adjoint extruded poly crack fdk sart scatter leg`).
Código: `wavesim/em/xray3d.py` (geometría, proyector y adjunto, radiografía, FDK, OS-SART, fantomas de
elipsoides), `_numba_xray3d.py` (kernels Numba: Siddon, retroproyección FDK y Monte Carlo 3D) y
`xray3d_scatter.py` (scatter 3D). Tests rápidos (21, unos 3 s): `tests/test_xray3d.py`. Los módulos 2D
(`xray`, `fanbeam`, `projection`, `xray_scatter`...) no cambian; el 3D reutiliza de ellos el espectro del tubo
(`tube_spectrum`: continuo de Kramers, líneas K, filtros), la descomposición agua + hueso
(`basis_coefficients`), el detector (`DetectorModel`: desenfoque, apertura del píxel, ruido electrónico), la
rampa (`projection.ramp_filter`) y las secciones eficaces del Monte Carlo (`materials.xray_processes`).

**Qué no se pudo verificar.** Valen las advertencias de `rayos_x.md` sobre las tablas de atenuación. Además:

- La atenuación másica del **CsI** del panel (`FlatPanel`) no sale de ninguna tabla: está reconstruida con el
  salto K del yodo (33.2 keV) y una dependencia ~E^-2.8 (error esperado ±20 %); la densidad efectiva del CsI
  estructurado (0.75 de la del cristal) es un valor típico no verificado. Si importa, pasa `efficiency=`.
- El fantoma de elipsoides `shepp_logan_3d` **no es el Shepp-Logan 3D canónico** (no se pudo comprobar sus
  parámetros): son 10 elipsoides de estilo Shepp-Logan con los contrastes del 2D modificado, definidos en el código.
- No se contrastó con ningún escáner ni simulador externo: toda la validación es analítica o interna.

## 1. Geometría y convenciones

Arrays `(nz, ny, nx)`, x = eje del hueso, y = lateral, z = profundidad (fuente arriba, en -z). La órbita es
circular alrededor de un eje (por defecto x, como un TAC de extremidad): para la vista `theta` el rayo central
es `d = cos θ ẑ + sin θ ŷ`, el eje horizontal del detector `u = −sin θ ẑ + cos θ ŷ` y el vertical `v = x̂`. Con
`theta = 0` es la vista antero-posterior (AP: columnas = y, filas = x, el hueso queda vertical) y con 90° la
lateral. Fuente a `sid` del isocentro (centro del dominio), detector plano a `sdd` de la fuente, `n_u × n_v`
píxeles de paso `du × dv` medido sobre el detector; magnificación en el isocentro `sdd/sid`.
`Geometry3D.covering` dimensiona el detector para cubrir el dominio en cualquier vista. `parallel=True` da haz
paralelo. Con eje `"y"` y `parallel=True` un volumen extruido en y reproduce la convención del 2D
(`projection.project`: ángulo 0 = rayos hacia +z, detector a lo largo de x).

## 2. Proyector: Siddon exacto frente a muestreo trilineal

Se implementan los dos, cada uno con su adjunto:

| | Siddon (por defecto) | muestreo trilineal |
|---|---|---|
| modelo | volumen constante a trozos; longitud exacta de cada rayo en cada celda | interpolación trilineal, regla del punto medio con paso 0.5 celdas |
| error | ninguno del proyector (sólo el de discretizar el objeto) | suaviza media celda y cuadratura O(paso²) |
| adjunto | transpuesta exacta | transpuesta exacta (mismos pesos) |
| backends | Numba (en paralelo) y NumPy | NumPy |
| coste por vista, 128³, 187×141 rayos, 1 hilo | 0.04 s (Numba), 1.06 s (NumPy) | 2.3 s |

**Elección: Siddon.** Es exacto para lo que representa el volumen (una caja homogénea da la cuerda de la caja
con error 6.5·10⁻¹⁶ m en los dos backends, cono, 5 ángulos), no suaviza los bordes (la grieta de 1 celda no se
emborrona) y es el más rápido. El recorrido es celda a celda al estilo Amanatides-Woo, que da los mismos
tramos que Siddon sin ordenar los cruces de planos.

**Error medido** frente a la cuerda analítica de una esfera (R = 15 mm) y un cilindro (R = 12 mm, eje x), con las
celdas rasterizadas con ocupación parcial (3³ submuestras), en los rayos con ρ < 0.8 R (a una vista de 37°;
los rayos rasantes cerca del borde ρ → R tienen un error de cuerda inevitable de orden dx):

| objeto | dx (mm) | geometría | Siddon rms (mm) | muestreo rms (mm) |
|---|---|---|---|---|
| esfera | 1.0 | paralelo, paso = dx | 0.548 | 0.067 |
| esfera | 1.0 | paralelo, paso = dx/1.37 | 0.340 | 0.069 |
| esfera | 1.0 | cónico | 0.116 | 0.072 |
| esfera | 0.5 | paralelo, paso = dx | 0.278 | 0.033 |
| esfera | 0.5 | paralelo, paso = dx/1.37 | 0.163 | 0.034 |
| esfera | 0.5 | cónico | 0.057 | 0.035 |
| cilindro | 1.0 | paralelo, paso = dx | 0.134 | 0.095 |
| cilindro | 1.0 | cónico | 0.122 | 0.092 |
| cilindro | 0.5 | paralelo, paso = dx | 0.059 | 0.040 |
| cilindro | 0.5 | cónico | 0.065 | 0.043 |

Lectura honesta: **el error de Siddon es el del objeto en celdas, no el del algoritmo**, y es mayor que el del
muestreo trilineal en la esfera con haz paralelo alineado con la malla (1.8 % del diámetro con dx = 1 mm, 0.9 % con
0.5 mm: baja ∝ dx). Con rayos exactamente en los planos de centros de celda (detector paralelo con `du = dx`),
la cuerda que ve Siddon es la media de la celda en el espesor de la celda, y cerca del borde de la esfera esa
media se aleja de la cuerda local (la cuerda varía como raíz); la interpolación trilineal la reconstruye mejor
porque suaviza justo ese efecto. Con haz cónico, cuyos rayos no caen en las líneas de la malla, Siddon (0.116 mm)
queda cerca del muestreo (0.072 mm). Para la radiografía y el TAC se mantiene Siddon: el error de ~0.1 mm rms
a dx = 1 mm es ≪ el tamaño de píxel del detector, y a cambio no hay sesgo de suavizado en los bordes finos
(la grieta) ni coste extra. El muestreo queda como `method="sampling"`.

## 3. Adjunción ⟨A x, y⟩ = ⟨x, Aᵀ y⟩

Volumen y datos aleatorios, malla 24×20×18, 5 vistas (0°, 20°, 75°, 130°, 250°):

| método | backend | geometría | error relativo |
|---|---|---|---|
| Siddon | Numba | cónico / paralelo | 0 / 3.3·10⁻¹⁶ |
| Siddon | NumPy | cónico / paralelo | 0 / 0 |
| muestreo | NumPy | cónico / paralelo | 0 / 0 |

Ambos backends de Siddon coinciden entre sí a 10⁻¹² (test `test_numpy_y_numba_dan_la_misma_proyeccion`). Los
tests de pytest exigen < 10⁻¹⁰.

## 4. Coherencia con el 2D (volumen extruido, haz paralelo)

Pierna longitudinal 2D (`longitudinal_limb`, dx = 0.5 mm, 60 kVp) extruida con `extrude`, geometría paralela con
eje y y el mismo paso de detector (0.37 dx, para que ningún rayo caiga justo en una cara de celda), frente a
`xray.acquire` 2D. Error relativo L2 de `-ln I` en el interior del detector y máximo absoluto:

| ángulo | Siddon: L2 rel. | Siddon: máx | muestreo: L2 rel. | muestreo: máx |
|---|---|---|---|---|
| 0° | 0.0212 | 0.480 | 0.0014 | 0.007 |
| 30° | 0.0042 | 0.030 | 0.0012 | 0.013 |
| 60° | 0.0045 | 0.043 | 0.0009 | 0.007 |
| 90° | 0.0682 | 0.915 | 0.0009 | 0.012 |
| 123° | 0.0042 | 0.028 | 0.0009 | 0.008 |

El muestreo trilineal es el mismo esquema que el bilineal del 2D y coincide a 0.1 %. Siddon coincide a 0.4 % en
ángulos oblicuos y se aparta a 0° y 90°, donde el rayo corre paralelo a las capas: allí el 2D interpola
bilinealmente entre dos filas de celdas (capas de piel y cortical de 1-2 celdas emborronadas) y Siddon da el
valor exacto de la celda. La diferencia es del proyector 2D, no del 3D. TAC: el FDK de haz paralelo (`mode="parallel"`)
del corte central de un fantoma extruido da el mismo mapa que `projection.fbp` 2D con error rms relativo 1.4 %
(diferencias del proyector y de interpolar la retroproyección).

## 5. Radiografía policromática

`acquire3d` proyecta con Siddon los dos mapas de la descomposición agua + hueso de cada tejido, forma
`Σ_i q_i exp(−L_w μ_w,i − L_b μ_b,i)` con los pesos del panel `q_i` (fracción de fotones del espectro × eficiencia
cuántica del CsI × energía si el panel integra energía), añade scatter si se pide, aplica desenfoque y apertura
del píxel en u y v, ruido de Poisson sobre los fotones absorbidos y ruido electrónico. Vistas `"ap"` y `"lateral"`.

Losa homogénea frente a Beer-Lambert integrado en energía (`-ln Σ q_i exp(−μ_i t)` con el μ tabulado de cada tejido
a cada energía); **ojo, la referencia usa el mismo espectro, pesos de panel y tablas**, así que valida la
geometría, la integración en energía y el detector, no las tablas:

| losa | 60 kVp / 2.5 mm Al, ideal | 60 kVp, CsI integrador | 120 kVp / 6 mm Al, ideal |
|---|---|---|---|
| agua, 40 mm | +4·10⁻¹⁶ | +7·10⁻¹⁶ | −2·10⁻¹⁶ |
| cortical, 10 mm | +4·10⁻¹⁶ | −4·10⁻¹⁶ | +1·10⁻¹⁵ |
| músculo, 40 mm | +4.3·10⁻⁴ | +6.9·10⁻⁴ | +5·10⁻⁵ |
| grasa, 40 mm | −4.1·10⁻⁴ | −8.4·10⁻⁴ | −2·10⁻⁵ |

(error de `-ln T`). Agua y hueso son exactos (son la base); músculo y grasa, descompuestos en agua + hueso, se
apartan menos de 10⁻³ en `-ln T` (0.08 % de transmisión). Endurecimiento del haz (80 kVp, 2.5 mm Al, agua): la
atenuación efectiva baja de 33.4 /m (5 mm) a 31.8 (20 mm) y 30.2 (40 mm).

**Ruido de Poisson.** Con 5·10³ fotones por píxel sobre agua uniforme, la media de las cuentas es `N0·T` (2 %) y la
varianza es igual a la media dentro del 15 % (1600 píxeles; test `test_ruido_de_poisson_en_la_radiografia`, también
con el panel de CsI sobre los fotones absorbidos). **Limitaciones del ruido**: no hay factor de Swank ni correlación
espacial del ruido (el desenfoque se aplica a la señal esperada antes del Poisson) ni la caída ~cos³ del flujo en los
bordes del haz cónico (el flujo abierto es uniforme por píxel).

## 6. Visibilidad de una grieta frente a su orientación

Bloque cortical cúbico de 20 mm con una grieta plana de 0.5 mm (agua) en el plano y = 0; haz paralelo, 60 keV
monoenergético, señal = máximo de `−ln T(sin grieta) − (−ln T(con grieta))`. `φ` es el ángulo entre el haz y el plano
de la grieta. Modelo geométrico: `(μ_b − μ_w)·min(w/sin φ, Z/cos φ)` (camino del rayo dentro del hueco).

| φ | señal simulada | modelo | CNR (10⁵ fotones/px, píxel de la grieta) |
|---|---|---|---|
| 0° | 0.7970 | 0.7970 | 158 |
| 1° | 0.7972 | 0.7972 | 158 |
| 2° | 0.5709 | 0.5709 | 107 |
| 5° | 0.2286 | 0.2286 | 40 |
| 10° | 0.1147 | 0.1147 | 19 |
| 20° | 0.0583 | 0.0583 | 9.3 |
| 45° | 0.0282 | 0.0282 | 4.5 |
| 90° | 0.0199 | 0.0199 | 3.3 |

La grieta sólo se ve bien con el haz paralelo a su plano: pasa de 0.80 a 0.02 (40 veces) al girar de 0° a 90°, y
coincide con el modelo (el modelo es la geometría exacta; el acuerdo prueba el proyector, no la física). A 90°
queda `w·(μ_b − μ_w)` = 0.020 de `−ln T`, es decir, una grieta de 0.5 mm normal al haz es casi invisible y una
fractura oblicua desaparece al alejarse del haz tangente: por eso una sola radiografía AP puede no mostrarla y se
piden dos proyecciones ortogonales (o un TAC).

## 7. TAC de haz cónico: FDK

Fantoma de elipsoides tipo Shepp-Logan 3D (64³, dx = 1 mm, 360 vistas, `sdd = 1.4·sid`). Para no cometer el
«crimen inverso», los datos se proyectan sobre una malla del doble de resolución (0.5 mm) y se reconstruyen en la de
1 mm. Error rms relativo al fantoma (rasterizado a 1 mm) dentro del interior del cráneo. «Semiángulo de cono» =
`atan(½ n_v dv / sdd)`. Plano central: 7 cortes x = n/2 ± 3; exteriores: los cortes con |x − centro| > n/4.

| geometría | sid (mm) | semiángulo de cono | error plano central | error planos exteriores | t proyección (s) | t FDK (s) |
|---|---|---|---|---|---|---|
| paralelo (apilado) | – | 0° | 0.082 | 0.019 | 2.0 | 0.3 |
| FDK | 1000 | 2.0° | 0.069 | 0.015 | 4.6 | 0.8 |
| FDK | 400 | 5.4° | 0.068 | 0.054 | 4.7 | 0.8 |
| FDK | 200 | 12.0° | 0.068 | 0.085 | 4.9 | 0.8 |
| FDK | 120 | 23.6° | 0.067 | 0.075 | 5.5 | 0.9 |
| apilado de abanico | 1000 / 400 / 200 / 120 | 2 / 5.4 / 12 / 23.6° | 0.069 | 0.016 / 0.054 / 0.077 / 0.066 | | 0.6-0.8 |

El error del plano central es ~7 % en todas las geometrías (lo domina la resolución: las lesiones del fantoma son de
2-6 celdas) y es **independiente del ángulo de cono**, como corresponde a un FDK exacto en el plano central. El de los
planos exteriores crece de 1.5 % (2°) a 5.4 % (5.4°) y 8.5 % (12°): el artefacto de cono. El 23.6° baja
algo (7.5 %), no tengo una explicación verificada (cambian a la vez la magnificación y el recorte del dominio por el
detector). **En este fantoma el FDK no mejora al apilado de abanico** (0.085 frente a 0.077 a 12°): la diferencia
entre ambos sólo se aprecia en objetos con estructura a lo largo del eje, como el cilindro de abajo.

Esfera homogénea (R = 20 mm, μ = 50 /m), valor reconstruido / μ:

| geometría | centro | polo (0.8 R, a lo largo de x) | error rms dentro |
|---|---|---|---|
| paralelo | 1.004 | 1.000 | 0.039 |
| FDK sid 400 | 1.002 | 1.038 | 0.035 |
| FDK sid 200 | 1.002 | 1.044 | 0.035 |
| FDK sid 120 | 1.002 | 0.971 | 0.034 |

El centro sale con 0.2-0.4 % de error; fuera del plano central el FDK sesga el valor 3-4 % (artefacto de cono: ±4 %
a 0.8 R; el signo cambia a 24° de semiángulo).

Cilindro de R = 15 mm a lo largo del eje de giro (invariante en x), valor en el eje / μ:

| geometría | modo | x = 0 | x = 15 mm | x = 24 mm |
|---|---|---|---|---|
| sid 400 | FDK / apilado | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.001 |
| sid 200 | FDK / apilado | 0.999 / 0.999 | 0.999 / 1.002 | 0.999 / 1.007 |
| sid 120 | FDK / apilado | 1.000 / 1.000 | 1.000 / 1.009 | 1.000 / 1.021 |

FDK es exacto en un objeto invariante a lo largo del eje (el peso de coseno compensa la inclinación del rayo), y el
apilado se sesga hasta un 2 % a 24 mm con el cono de sid 120.

**OS-SART 3D** (`os_sart3d`; Siddon y su adjunto, no negatividad, TV suavizada opcional; 48³, dx = 1.3 mm, sid 400,
4 iteraciones de 6 subconjuntos; error rms relativo en el interior del cráneo):

| vistas | fotones/px | FDK | OS-SART | OS-SART + TV (0.003) |
|---|---|---|---|---|
| 120 | sin ruido | 0.012 | 0.020 | 0.016 |
| 36 | sin ruido | 0.014 | 0.018 | 0.018 |
| 36 | 300 | 0.068 | 0.060 | **0.027** |

Sin ruido el FDK es mejor (el iterativo se queda en 4 iteraciones); con pocas vistas y ruido fuerte, la TV reduce el
error 2.5 veces. El peso de TV se eligió mirando 0, 0.001, 0.003 y 0.01 sobre esos mismos datos (sin conjunto de
validación aparte): 0.001 y 0.01 dan 0.045 y 0.029.

## 8. Scatter 3D (Monte Carlo)

El MC 2D (`xray_scatter`) trata el objeto como un corte extruido sin límite en y. `xray3d_scatter` generaliza el
kernel (`run_photons3d`, *delta tracking* en el volumen `Phantom3D` completo, los mismos procesos y secciones
eficaces: fotoeléctrico, Compton con S(x,Z), Rayleigh) a un detector plano 2D con fuente cónica o haz paralelo. El
scatter se acumula en bins gruesos del detector (20×20 por defecto), se suaviza (σ = 1 bin) y se interpola a los
píxeles; en un TAC se simulan `n_views` vistas (8) y se interpola en el ángulo. Requiere Numba. No hay rejilla
antidispersora ni fluorescencia ni retrodispersión desde el detector.

| comprobación | resultado |
|---|---|
| balance de fotones (cubo de agua de 40 mm, 4·10⁶ fotones) | 0 (exacto) |
| primaria MC (4 bins centrales) / Beer-Lambert | 0.2513 / 0.2511 |
| losa de agua de 40 mm × 120 × 240 mm, rendija de 10 mm, haz paralelo, 8·10⁶ fotones: scatter relativo por bin, MC 2D / MC 3D | bin 0: 0.0103 / 0.0102; 2: 0.0141 / 0.0141; 4: 0.0147 / 0.0150; 6: 0.0152 / 0.0152; 11: 0.0103 / 0.0103 |
| SPR global de esa losa, 2D / 3D | 0.0545 / 0.0546 (diferencia máxima por bin 2.2 %) |

La losa finita en y (240 mm) con rendija reproduce el 2D extruido a ~2 % (el ruido MC por bin es ~1-2 %). Con el
fantoma de pierna (60 kVp, sid 1 m, sdd 1.1 m, 6·10⁶ fotones): SPR (dispersa/primaria) 0.12 sobre el hueso
(T < 0.2) y 0.034 sobre el tejido blando en la vista AP (0.119 y 0.036 en la lateral); el contraste de la fractura
en `-ln I` baja de 1.701 a 1.543 (−9 %). Los SPR son bajos porque la pierna del fantoma es fina (35 mm de
diámetro) y no hay carga de tejido alrededor; no se comparan con la literatura (cifras no verificadas).
Coste: 4·10⁶ fotones/s con un hilo.

## 9. Caso de aplicación: pierna 3D (`docs/img/rayos_x3d.png`)

`leg_segment("tibia", TransverseFracture3D(gap=1.2e-3))` (tibia como tubo con médula, músculo, grasa, piel, hueco
transversal de 1.2 mm con hematoma). Figura: radiografías AP y lateral con ruido (60 kVp, 2.5 mm Al, panel de CsI,
píxel de 0.25 mm, 2·10⁴ fotones/px), perfil de la fractura con y sin scatter, y cortes axiales y longitudinal del
TAC cónico (128³, dx = 0.4 mm, 360 vistas, 120 kVp con 6 mm de Al, sid 0.5 m, sdd 0.9 m, panel 187×141, 2·10⁵
fotones/px, linealización del endurecimiento para agua, FDK).

HU medidos en el TAC (erosionados 1-2 celdas de los bordes, lejos de la fractura) frente al modelo (μ del fantoma a
la energía media del espectro): agua −11.5 (modelo 0; σ = 39 HU), músculo 49.9 (44.9), grasa −127 (−103.8),
cortical 2104 (2063). El sesgo de −11 a −23 HU en agua y grasa es el del endurecimiento del haz residual
(linealizado para agua, no para el resto) y del cono.

| etapa | tamaño | tiempo (1 hilo Numba) | memoria máxima del proceso |
|---|---|---|---|
| radiografía AP o lateral | 76×74×120 celdas (0.67 M), 253×217 px | 0.1-0.3 s | 175 MB |
| radiografía AP con scatter MC (4·10⁶ fotones) | ídem | 2.2 s | |
| TAC cónico completo (proyectar 360 vistas + ruido + FDK) | 128³ (2.1 M), 187×141 px | 43 s | 1.3 GB |
| FDK sólo (64³ / 360 vistas) | | 0.8 s | |
| apilado de abanico sobre las mismas proyecciones (128³) | | 4.9 s | |

Objetivos cumplidos: radiografía en menos de 1 s (2 s con scatter) y TAC 128³ en menos de 1 minuto con un hilo.
Memoria ≈ 0.6 kB/celda en el TAC (volumen y proyecciones en float64: las proyecciones 360×141×187 son 76 MB cada
mapa y hay varios temporales); por debajo de los 3 GB del límite.

## 10. Limitaciones

- La proyección usa la descomposición agua + hueso (error < 10⁻³ en `-ln T` en tejidos blandos); no hay materiales
  con borde K intermedio (contraste yodado, metal).
- El detector es un modelo de panel simple (sin factor de Swank, ruido espacialmente correlacionado ni fluorescencia
  del centelleador) y el flujo abierto es uniforme por píxel (sin cos³ ni talón del ánodo, que `tube_spectrum` sí
  admite pero no varía con la posición en el detector).
- FDK exige una órbita circular completa de 360° y el objeto completamente dentro del detector (sin truncamiento): no
  hay corrección de truncamiento, ni Parker (escaneo corto), ni corrección de scatter en el TAC (si se pide scatter, se
  suma sin corregir y produce *cupping*).
- El scatter 3D requiere Numba y se acumula en bins gruesos (no resuelve estructura de scatter de alta frecuencia);
  `scatter` está validado entre 10 y 200 keV.
- El eje de giro del TAC es x (el del hueso) por defecto; `axis="y"|"z"` está soportado por la geometría pero la
  validación 3D se hizo con x y, para el extruido, con y.
- El proyector rasteriza cada celda con un solo tejido por etiqueta: la microestructura (porosidad, trabéculas) no
  existe en `Phantom3D`.
