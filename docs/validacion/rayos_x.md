# Validación de rayos X / TAC

Resultados de `benchmarks/xray_validation.py` (completo: unos 4.5 minutos con Numba y un hilo; las secciones
`hvl` y `fan` tardan unos 20 s). Código: `wavesim/em/xray.py` (espectro, `acquire`, `ct_scan`),
`fanbeam.py` (abanico, FBP, SART), `xray_phantoms.py` (Shepp-Logan exacto), `xray_scatter.py` y
`_numba_xray.py` (Monte Carlo), `xray_detector.py` (detector, MTF, NPS) y
`materials/xray_processes.py` (secciones eficaces por proceso). Tests rápidos: `tests/test_xray_*.py`.

Los valores por defecto de `ct_scan`, `radiograph` y `acquire` no cambian: todo lo nuevo son opciones
(`characteristic`, `anode_angle_deg`, `cu_filter_mm`, `geometry`, `method`, `scatter`, `detector`).

**Qué no se pudo verificar.** La sesión no tenía acceso a la base NIST (XCOM, tablas de coeficientes) ni a
las páginas de la norma; la lista de abajo dice qué cifras salen de memoria.

- Tablas de Cu y de $\mu_{en}/\rho$ del aire (NIST 126, recordadas) y de W (estimación aproximada, error
  esperado ~10 %): no verificadas. Líneas K del W (energías 59.318, 57.982, 67.245 y 69.067 keV; intensidades
  relativas 1, 0.58, 0.22, 0.08) y borde K 69.525 keV: recordadas del *X-Ray Data Booklet*, no verificadas.
- Las HVL de las calidades RQR sí salen de una búsqueda web (presentación del IAEA que reproduce IEC 61267 y
  TRS-457), pero la edición vigente (IEC 61267:2025/2026) puede haberlas cambiado y no se pudo consultar.
- SpekPy 2.5.4 se instaló con pip y se ejecutó en la sesión (valores reales, no recordados).
- Los valores de SPR de la literatura que aparecen abajo vienen de un snippet de búsqueda y no se pudieron
  contrastar con el artículo original.

## 1. Espectro del tubo

Modelo: Kramers $(E_0-E)/E$ en fotones, con (opcional) las cuatro líneas K, la autoabsorción en el ánodo de
W y filtración de Al y Cu.

- Líneas K: sólo si `kvp > 69.525`. Los fotones de línea por electrón crecen como $(U-1)^{1.67}$, con
  $U = kVp/69.525$; la constante está anclada a que valgan el 4 % de los fotones del continuo a 100 kVp. Es
  una calibración heurística (el 4 % y la profundidad del ánodo se ajustaron a la vez a SpekPy), no un modelo
  de ionización.
- Ánodo: los fotones de energía $E$ nacen a profundidad uniforme entre 0 y $0.27\,[R_{KO}(E_0)-R_{KO}(E)]$
  ($R_{KO}$ = alcance de Kanaya-Okayama) y salen con ángulo de despegue `anode_angle_deg - takeoff_offset_deg`;
  `takeoff_offset_deg > 0` reproduce el efecto talón (haz más duro hacia el lado del ánodo).
- HVL: bisección sobre el kerma en aire $\sum w_i E_i (\mu_{en}/\rho)_{aire,i}\,e^{-\mu_{Al,i} t}$.

**Calidades IEC 61267 (RQR).** Se ajusta sólo el espesor total de Al hasta obtener la HVL de la norma (error de
la HVL resultante < 0.1 % por construcción); la validación es si el espesor ajustado es físicamente razonable
para un tubo diagnóstico (2.5-5 mm) y cómo cambia con el modelo:

| calidad | kVp | HVL norma (mm Al) | Al ajustado, Kramers | Al ajustado, ánodo 12° | Al ajustado, líneas K + ánodo 12° |
|---|---|---|---|---|---|
| RQR3 | 50 | 1.78 | 3.20 | 2.91 | 2.91 |
| RQR5 | 70 | 2.58 | 3.65 | 3.12 | 3.12 |
| RQR6 | 80 | 3.01 | 3.93 | 3.26 | 3.14 |
| RQR7 | 90 | 3.48 | 4.25 | 3.44 | 3.15 |
| RQR8 | 100 | 3.97 | 4.58 | 3.61 | 3.17 |
| RQR9 | 120 | 5.00 | 5.21 | 3.93 | 3.27 |
| RQR10 | 150 | 6.57 | 6.03 | 4.25 | 3.53 |

Sin ánodo el modelo necesita más filtración que la real a bajo kV (Kramers sin autofiltración es demasiado
blando) y menos a 150 kV; con ánodo y líneas K los espesores quedan entre 2.9 y 3.5 mm. No se pudo verificar la
filtración total que usa un laboratorio para cada RQR, así que esta tabla no es una validación frente a la
norma más allá de la HVL.

**Frente a SpekPy 2.5.4** (`Spek(kvp, th=12).filter("Al", t)`, HVL1 en mm Al / energía media en keV; entre
paréntesis, error relativo de la HVL; sin filtración inherente). Objetivo orientativo ±5 %:

| kVp | Al (mm) | SpekPy | Kramers | ánodo 12° | líneas K + ánodo 12° |
|---|---|---|---|---|---|
| 50 | 2.5 | 1.750 / 32.2 | 1.528 (-12.7 %) / 31.0 | 1.635 (-6.6 %) / 31.5 | 1.635 (-6.6 %) / 31.5 |
| 70 | 2.5 | 2.397 / 39.4 | 1.996 (-16.7 %) / 37.5 | 2.272 (-5.2 %) / 38.6 | 2.272 (-5.2 %) / 38.6 |
| 80 | 2.5 | 2.734 / 42.9 | 2.199 (-19.6 %) / 40.4 | 2.591 (-5.2 %) / 41.9 | 2.643 (-3.3 %) / 42.5 |
| 100 | 2.5 | 3.458 / 49.1 | 2.584 (-25.3 %) / 45.7 | 3.260 (-5.7 %) / 48.0 | 3.492 (+1.0 %) / 49.3 |
| 120 | 2.5 | 4.219 / 54.4 | 2.964 (-29.7 %) / 50.6 | 3.993 (-5.4 %) / 53.8 | 4.399 (+4.3 %) / 54.8 |
| 120 | 4.0 | 5.305 / 57.1 | 4.201 (-20.8 %) / 54.3 | 5.045 (-4.9 %) / 56.6 | 5.483 (+3.3 %) / 57.2 |

Con líneas K y ánodo se cumple ±5 % de 80 a 120 kVp, pero a 50 y 70 kVp el error es -6.6 % y -5.2 % (no se
alcanza el objetivo). Los dos parámetros libres (fracción de líneas, profundidad del ánodo) se ajustaron a estos mismos
seis casos de SpekPy, de modo que la comparación mide la calidad del ajuste, no una predicción independiente.
La energía media coincide con SpekPy en menos de 3.5 %.

## 2. Haz en abanico y reconstrucción

Geometría: fuente puntual a `sid` del isocentro, detector a `sdd` de la fuente, curvo equiangular o plano.
Proyector: muestreo bilineal a lo largo del rayo; `fan_back` es su adjunto exacto (test: $\langle Ax,y\rangle =
\langle x,A^Ty\rangle$ a 1e-10). FBP de abanico (ponderación de coseno, rampa con la corrección $(\gamma/\sin\gamma)^2$
en el detector curvo, retroproyección con $1/L^2$), elegida frente al *rebinning* porque no remuestrea los datos y
sirve igual para curvo y plano. Exige 360° con ángulos uniformes (sin *short scan* ni pesos de Parker).

Rejilla 128x128 de 1.56 mm, sinogramas analíticos exactos de elipses (no se genera con el proyector que reconstruye), RMSE en 1/m dentro del objeto:

| método | RMSE disco | RMSE Shepp-Logan | valor del disco (exacto 20) | cerebro SL (exacto 4) |
|---|---|---|---|---|
| paralelo (360 vistas / 180°) | 0.016 | 0.793 | 19.99 | 3.98 |
| abanico curvo (720 vistas) | 0.015 | 0.782 | 20.00 | 3.98 |
| abanico plano (720 vistas) | 0.014 | 0.759 | 20.00 | 3.98 |

El abanico iguala al haz paralelo (el RMSE del Shepp-Logan, 0.76-0.79 de 20, está dominado por el
suavizado de los bordes por la ventana y la malla, igual en los tres).

**SART/OS-SART frente a FBP** (Shepp-Logan, rejilla 64x64, 10 iteraciones, subconjuntos de 5 vistas entrelazadas,
no negatividad; el ruido es gaussiano aditivo con desviación típica relativa al máximo del sinograma; TV con
peso 1e-3):

| ruido | vistas | FBP | OS-SART | OS-SART + TV |
|---|---|---|---|---|
| 0 % | 30 | 1.98 | 1.47 | 1.57 |
| 0 % | 45 | 1.56 | 1.13 | 1.26 |
| 0 % | 60 | 1.39 | 0.95 | 1.02 |
| 0 % | 180 | 1.34 | 1.19 | 0.94 |
| 1.7 % | 30 | 2.03 | 1.51 | 1.59 |
| 1.7 % | 45 | 1.60 | 1.17 | 1.27 |
| 1.7 % | 60 | 1.43 | 1.01 | 1.05 |
| 1.7 % | 180 | 1.36 | 1.24 | 0.95 |

OS-SART gana a FBP con 30-60 vistas en un 25-32 % de RMSE, también con ruido. Con 180 vistas ya no gana
(1.19 frente a 1.34 sin TV; la TV es lo que ayuda ahí). Con pocas vistas la TV de peso 1e-3 empeora un poco la
reconstrucción sin ruido (la TV sirve sobre todo a ruido alto o muchas vistas). Hace falta ajustar el peso
a cada problema; el valor 1e-3 es heurístico. SART de una sola vista por subconjunto con relajación 1 fue peor
que FBP en varios casos (1.65 con 30 vistas y 1.93 con 180), por eso el recomendado es OS-SART con subconjuntos.

## 3. Radiación dispersa

Física en `materials/xray_processes.py` (el módulo documenta el origen de cada magnitud):

- Total: el de `linear_attenuation`. Los tres procesos suman exactamente ese total.
- Compton: Klein-Nishina por la función de dispersión incoherente $S(x,Z)$ de cada átomo, por elementos de la composición
  ICRU-44. Sin ensanchamiento Doppler.
- Rayleigh: $(1+\cos^2\theta)F^2(x,Z)$ con $F$ de orbitales de Slater (analítico, aproximado).
- Fotoeléctrico: el resto hasta el total (con suelo del 0.1 %). Sin fluorescencia ni transporte de electrones.

Control del modelo (agua; $\mu/\rho$ en cm²/g, XCOM recordado, no verificado): a 100 keV el modelo da incoherente
0.1629, coherente 0.0054 y fotoeléctrico 0.0025, frente a total 0.1707; el fotoeléctrico residual cae como
E^-3 y la suma Compton+Rayleigh queda por debajo del total en todos los tejidos entre 10 y 200 keV (test). No se
pudo comprobar contra XCOM; el factor de forma hidrogenoide-Slater es aproximado y la fracción coherente puede
errar decenas por ciento (<= ~1 % del total).

Transporte 3D (delta tracking) por el corte extruido sin límite en $y$, fuente colimada a un grosor de corte y
detector de altura igual al haz proyectado. La luz dispersa se registra por elemento: primaria, dispersa
(Compton o sólo Rayleigh) y energía.

**(a) Primaria frente a Beer-Lambert** (cilindro de agua de 18 cm, 100 kVp con 3 mm Al, 2e7 fotones, 9.5 s con
Numba, 2.1e6 fotones/s). Esperado: Beer-Lambert exacto sobre el mismo mapa de vóxeles (la discretización del cilindro
cambia la cuerda hasta 1.8 cm en algunos rayos respecto al círculo ideal):

| región | contados | esperados | (obs-esp)/sqrt(esp) |
|---|---|---|---|
| centro (21 elementos) | 42473 | 42878 | -1.96 |
| borde interior (20) | 117930 | 117289 | +1.87 |
| fuera del cilindro (20) | 2237178 | 2237281 | -0.07 |

**(b) Distribuciones angulares** (agua, 60 keV, medio tenue para una sola interacción; 3e7 fotones):

| magnitud | Monte Carlo | esperado |
|---|---|---|
| ⟨cos θ⟩ Compton (KN × S) | 0.0574 | 0.0583 (≈ 1.4 σ; 443443 colisiones) |
| ⟨cos θ⟩ Compton, KN libre (sin S) | - | 0.0786 (S la baja un 26 %) |
| energía media cedida (keV) | 5.712 | 5.739 |
| ⟨cos θ⟩ Rayleigh | 0.9009 | 0.9021 |
| Rayleigh / (Rayleigh + Compton) | 0.0736 | 0.0731 |

**(c) Balance** (caso a): fotones, 0 de diferencia; energía, -3.3e-14 relativo (emitida = fotoeléctrico +
electrones Compton + fuera del detector + detectada).

**(d) SPR en el eje** (11 elementos centrales) de cilindros de agua infinitos en $y$; `sid` 1 m, `sdd` 1.5 m; espectro con
líneas K y ánodo 12° con 2.5 mm Al; 1.2e7 fotones por celda; la rejilla antidispersora enfocada acepta
±1.5°:

| kVp | diámetro | corte 1 cm | 1 cm + rejilla | corte 4 cm | 4 cm + rejilla |
|---|---|---|---|---|---|
| 80 | 16 cm | 0.039 | 0.006 | 0.155 | 0.013 |
| 80 | 20 cm | 0.072 | 0.010 | 0.262 | 0.019 |
| 100 | 16 cm | 0.037 | 0.006 | 0.136 | 0.012 |
| 100 | 20 cm | 0.062 | 0.009 | 0.226 | 0.017 |
| 120 | 16 cm | 0.035 | 0.006 | 0.128 | 0.013 |
| 120 | 20 cm | 0.057 | 0.008 | 0.211 | 0.016 |

Losa de agua de campo ancho (infinita en $x$ e $y$, campo de 40 cm en $y$ a 1 m, sin rejilla), 125 kVp:

| espesor | SPR en el eje |
|---|---|
| 10 cm | 0.78 |
| 20 cm | 2.50 |
| 30 cm | 5.43 |

Comparación con la literatura: una búsqueda web devolvió, para un haz cónico abierto de 125 kVp sobre agua de 10, 20 y
30 cm, SPR de 0.94, 3.0 y 7.6 (no se pudo abrir el artículo, así que ni la cita ni el tamaño de campo están
confirmados). Nuestros 0.78, 2.50 y 5.43 son un 17-29 % menores; el orden de magnitud y el crecimiento con el espesor
coinciden, pero no hay una validación cuantitativa. Para cortes colimados de TAC no se encontró ningún valor
verificable (un snippet daba SPR 4 y otro 0.59-3.00 con otras definiciones y rejillas, que no son comparables), así que los SPR de
la tabla anterior (0.06 para 20 cm y corte de 1 cm a 100 kVp) son sólo lo obtenido. Tendencia: el SPR baja
ligeramente con el kVp y sube con el diámetro y con la apertura del corte, como se espera.

**(e) Efecto en el TAC** (abanico, cilindro de músculo de 18 cm con insertos de hueso cortical y grasa, 100 kVp con 3 mm Al,
sin ruido; con corrección de endurecimiento del agua, sin corrección de scatter):

| corte | caso | HU centro | HU borde | *cupping* (borde-centro) | HU hueso | HU grasa | contraste hueso-músculo | contraste músculo-grasa |
|---|---|---|---|---|---|---|---|---|
| 1 cm | sin scatter | 45.1 | 47.1 | 2.0 | 1947 | -117 | 1902 | 162 |
| 1 cm | con scatter | 17.4 | 38.6 | 21.3 | 1759 | -128 | 1742 | 146 |
| 4 cm | sin scatter | 45.1 | 47.1 | 2.0 | 1947 | -117 | 1902 | 162 |
| 4 cm | con scatter | -37.4 | 17.0 | 54.5 | 1377 | -157 | 1414 | 119 |

El scatter produce *cupping* (+21 y +54 HU) y reduce el contraste hueso-músculo un 8 % (corte de 1 cm) y un 26 %
(4 cm), y el músculo-grasa un 10 % y un 27 %.

## 4. Detector y calidad de imagen

`DetectorModel`: desenfoque gaussiano, apertura de píxel (sinc) y ruido electrónico, aplicado al valor esperado
antes del Poisson (no reproduce la correlación espacial del ruido en el centelleador).

MTF por el método del borde (borde inclinado o circular), con compensación de la respuesta del binning y de la
derivada. Test con un borde gaussiano de $\sigma$ conocido: error máximo < 0.04 hasta MTF > 0.05; con una apertura de
píxel cuadrada el error contra $|\mathrm{sinc}|$ es < 0.06.

**(a) Radiografía de un borde** con desenfoque del detector $\sigma$ (MTF medida / MTF sin desenfoque frente a la teórica):

| σ del detector | error máx. de la MTF (hasta MTF≈0.6) | f50 medida (ciclos/m) | f50 teórica |
|---|---|---|---|
| 0.5 mm | 0.009 | 380.6 | 374.8 |
| 1.0 mm | 0.027 | 197.2 | 187.4 |

**(b) MTF del TAC** de abanico (cilindro de agua de 6 cm, 720 vistas; $\sigma_{det}$ = 0.8 mm en el detector, 0.4 mm en el isocentro por
`sid/sdd` = 0.5):

| f (ciclos/m) | MTF ideal | MTF con detector | cociente | teórica del detector en el isocentro |
|---|---|---|---|---|
| 200 | 0.894 | 0.813 | 0.910 | 0.881 |
| 400 | 0.584 | 0.371 | 0.635 | 0.603 |
| 600 | 0.311 | 0.118 | 0.380 | 0.321 |
| 800 | 0.110 | 0.027 | 0.245 | 0.132 |

El cociente coincide con la teoría dentro del 3-5 % hasta MTF ideal ≈ 0.3; a 800 ciclos/m la MTF ideal es 0.11 y el
cociente es poco fiable (ruido de la MTF). La pequeña diferencia es esperable: la ampliación del detector varía con la posición del
borde respecto al isocentro (el borde del cilindro está a 3 cm del eje).

**(c) Ruido del TAC** (cilindro de agua de 6 cm, ROI central de 20 mm; fotones por elemento y vista):

| fotones | σ de la ROI (HU) | √∫NPS (HU) | NPS radial a 0.19 / 0.44 / 0.69 / 0.94 de la frecuencia máxima (HU² m²) |
|---|---|---|---|
| 2e4 | 33.6 | 33.5 | 2.8e-4 / 5.6e-4 / 3.7e-4 / 2.5e-4 |
| 8e4 | 16.3 | 16.3 | 6.1e-5 / 1.0e-4 / 1.2e-4 / 6.2e-5 |

Parseval se cumple al 0.3 % y el cociente de varianzas con 4 veces más fotones es 4.25 (esperado 4). La forma del
NPS (sube con la frecuencia, ramp, y cae por la ventana de Shepp-Logan y el *interpolation* de la retroproyección)
es la típica del FBP; no hay una referencia verificada con la que compararla cuantitativamente.

## Pendientes y limitaciones

- El Monte Carlo trata el objeto como infinito en $y$ y no incluye fluorescencia, transporte de electrones,
  ensanchamiento Doppler ni interferencia molecular en Rayleigh. Las secciones coherente/incoherente son modelos
  (no XCOM) y no pueden contrastarse con NIST desde esta sesión.
- El scatter de un TAC completo se calcula en pocas vistas y se interpola y suaviza (`ScatterOptions`): válido
  para componentes de baja frecuencia, no resuelve estructura fina del scatter.
- Falta una corrección de scatter en la reconstrucción (sólo se simula el efecto), *short scan* con pesos de
  Parker y haz cónico 3D.
- El tiempo del benchmark completo y de los tests se recoge en el informe final de la sesión.
