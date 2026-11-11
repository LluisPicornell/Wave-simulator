# wavesim frente a los simuladores de referencia

Comparación de `wavesim` con los simuladores usados hoy en investigación y en
la industria, modalidad por modalidad, más un plan de validación y una primera
comparación numérica con k-Wave. La descripción de los otros simuladores es la
de la primera redacción; lo que se ha actualizado es el estado de wavesim tras
una serie de mejoras dirigidas a cerrar las brechas que este documento
señalaba, cada una con sus cifras medidas en [`validacion/`](validacion/).

**Lectura honesta del conjunto:** se han cerrado o reducido varias brechas
*de física y de método* (atenuación, interfaz fluido-sólido, ondas guiadas,
Maxwell dispersivo, Monte Carlo óptico, scatter de rayos X, Bloch espacial,
EIT con CEM y Gauss-Newton, ultrasonido 3D de dominio pequeño). **No** se han
cerrado las de escala y de validación externa: wavesim sigue sin GPU, sin 3D de
cuerpo entero, sin no linealidad acústica, sin física completa de partículas,
y sin validación revisada por pares. La **comparación cruzada directa** sí se
ha hecho ya en cinco modalidades, ejecutando aquí en CPU k-Wave, Devito,
gprMax, MCML, MCX, ASTRA, SpekPy, XCIST y pyEIT (resumen en §10.0); la RM
(KomaMRI, JEMRIS) y el modelo completo de electrodo de EIT siguen sin ella.

## Resumen: brecha → estado → evidencia

| Brecha señalada | Estado | Evidencia medida |
|---|---|---|
| Atenuación por ley de potencia (US) | **Cerrada** en la banda del pulso; `n_mechanisms=2` por defecto | Error vs α₀·f^y ≤ 2 % en [0.5, 1.8]·f₀ (antes 14-97 % con 1 mecanismo); coste ×1.5. [atenuacion](validacion/atenuacion.md) |
| Interfaz fluido-sólido oblicua (US) | **Corregida** (error real del solver) | \|R\| a 45°: 4.6 → 1.5 % (10 ppw), 2.3 → 0.4 % (20 ppw). [elastico](validacion/elastico.md) |
| Ondas S, conversión de modo y ondas guiadas (U3-U5) | **Verificadas** frente a analítico, con excepciones | R(θ) ≤ 0.7 % hasta 22°, crítico P ±0.15°; Lamb S0/A0 ≤ 2 %. Fuera de tolerancia: \|R\| 4-5 % en bordes de zona y 9-17 % cerca de los críticos. [elastico](validacion/elastico.md) |
| Cortical anisótropa (US) | **Cerrada, opcional** (`anisotropic_bone`, desactivada); c13 no verificado | qP/qSV vs Christoffel ≤ 0.1 %. [anisotropia](validacion/anisotropia.md) |
| Inestabilidad de la CPML con ondas guiadas (US) | **Corregida** con M-PML automática | Sin crecimiento en 40 000 pasos; reflexión −28 dB (amplitud) en hueso, −53 dB en agua. [estabilidad](validacion/estabilidad.md) |
| Sólo 2D (US) | **Parcialmente cerrada**: solver 3D de dominio pequeño | Velocidades ≤ 0.4 %; 2D extruido = 3D a ~1e-6; ~27 Mceldas·paso/s por hilo; sin M-PML, sin speckle ni porosidad. [elastico3d](validacion/elastico3d.md) |
| Sonda realista, STA, formadores de haz (US) | **Cerrada** en la cadena; sin sondas reales ni no linealidad | Lateral ≤ 12 % de la teoría; **axial +9 a +17 % (no cumple 10 %)**; SNR de speckle 1.85 vs 1.91. [imagen_ultrasonido](validacion/imagen_ultrasonido.md) |
| Maxwell de banda estrecha (EM) | **Cerrada**: Debye multipolo (ADE), banda ancha, Numba | Debye ≤ 0.7 %; β ≤ 0.23 %, α ≤ 1.6 %; losa ≤ 0.31 dB; Mie 1.7-3.3 %. Sigue 2D TM, sin antenas. [maxwell](validacion/maxwell.md) |
| Monte Carlo óptico (óptica) | **Cerrada** (índice único por tejido) | van de Hulst ≤ 1.6 σ; Fresnel 1e-7; balístico ≤ 1.2 σ; la difusión da ×1.27 menos fluencia cerca de la piel. [optica](validacion/optica.md) |
| Scatter, líneas K, abanico, iterativo, detector (rayos X) | **Parcialmente cerrada** | Abanico = paralelo; OS-SART −25-32 % de RMSE con 30-60 vistas; HVL vs SpekPy ±5 % a 80-120 kVp pero −6.6/−5.2 % a 50/70 kVp; sin cono 3D ni corrección de scatter. [rayos_x](validacion/rayos_x.md) |
| RM: Bloch espacial y artefactos | **Cerrada** en 2D cartesiano | Señal vs analítico 1e-11 o mejor; desplazamiento químico exacto; cilindro de susceptibilidad ≤ 0.7 % (B0 alineado). Sin comparar con KomaMRI/JEMRIS. [rm](validacion/rm.md) |
| EIT: CEM, Gauss-Newton, solver rápido | **Cerrada** en 2D; sin FEM, 3D ni GREIT | CEM vs Fourier ≤ 1.5 %; GN-TV 0.08 mm en el disco; multigrid 922 → 13-16 iteraciones. [eit](validacion/eit.md), [rendimiento_em](validacion/rendimiento_em.md) |
| GPU y escala | **Abierta** | Sólo CPU (Numba). |
| 3D de cuerpo entero, fantomas segmentados | **Abierta** | Anatomía paramétrica 2D (+ tibia tubular 3D). |
| No linealidad acústica, terapia | **Abierta** | Acústica lineal. |
| Física completa de partículas (GATE, MC-GPU) | **Abierta** | Scatter MC sin electrones ni fluorescencia. |
| Comparación cruzada directa | **Hecha en 5 modalidades** (k-Wave, Devito, gprMax, MCML, MCX, ASTRA, SpekPy, XCIST, pyEIT); abierta en RM, CEM de EIT, FBP de abanico y Field II/SimSonic | §10.0 y `validacion/cruzada_*.md`. |
| Validación externa revisada por pares | **Abierta** | Toda la validación es interna. |

## 0. Cómo leer este documento (y qué no se pudo verificar)

- **Estado de verificación.** Cada dato lleva una marca implícita:
  - Sin marca: confirmado en esta sesión (resultado de búsqueda, página de
    GitHub/PyPI, inspección del paquete instalado, o ejecución propia).
  - **[NV]**: no verificado en esta sesión. Procede de conocimiento previo del
    autor y debe comprobarse antes de citarlo.
  - **[?]**: dato que no se pudo confirmar (típicamente licencias).
- **Acceso a la web.** Desde el entorno donde se escribió esto sólo eran
  accesibles GitHub y PyPI (y el buscador). Las webs oficiales de k-Wave
  (k-wave.org), SimSonic (simsonic.fr), Field II (field-ii.dk), Meep
  (readthedocs), arXiv, PubMed, doi.org, NCBI, Wikipedia, astra-toolbox.com y
  openems.de **no eran accesibles** (fallo de DNS o conexión rechazada por el
  proxy). Los datos de esos proyectos proceden de fragmentos del buscador y de
  páginas espejo; por eso hay varias licencias en **[?]**.
- **Versión de wavesim comparada.** El análisis se hizo sobre una versión
  anterior. Las filas y apartados de wavesim se han reescrito para el estado
  actual (atenuación por mecanismo, interfaz corregida, M-PML, anisotropía
  opcional, solver 3D, cadena de imagen, Maxwell de banda ancha, Monte Carlo
  óptico, rayos X ampliados, Bloch espacial y EIT con CEM y Gauss-Newton). Las
  descripciones de los demás simuladores **no** se han vuelto a verificar.
- **Cifras de wavesim.** Sólo las de `docs/validacion/*.md` o medidas al
  escribir esta revisión (indicadas); las referencias bibliográficas que esos
  documentos marcan como no verificadas siguen sin verificar.
- **Fecha.** Noviembre de 2026. Versiones y licencias pueden haber cambiado.

## 1. Qué es wavesim (resumen honesto)

Paquete Python + NumPy (Numba opcional en CPU, sin GPU), casi todo 2D, con un
solo conjunto de fantomas de extremidad con fractura compartido por todas las
modalidades:

| Modalidad | Qué resuelve realmente | Límites principales |
|---|---|---|
| Ultrasonido | FDTD elástico velocidad-tensión, 4.º orden en espacio, leapfrog; CPML con M-PML automática si una capa sólida la cruza; atenuación GSLS con un peso por mecanismo (2 mecanismos por defecto, error < 2 % frente a la ley α₀·f^y en la banda del pulso); corrección de la interfaz fluido-sólido; speckle por fluctuación de densidad, rugosidad de interfaces; cortical transversalmente isótropa opcional; sonda con respuesta de elemento, onda plana, focalizada y STA, DAS/CF/DMAS/SLSC | 2D, CPU, float32, acústica lineal, tejidos blandos como fluido (`mu = 0`), sonda ideal; FWHM axial de un dispersor puntual +9-17 % sobre la teoría; reflexión de la M-PML −28 dB en el hueso |
| Ultrasonido 3D | `ElasticSolver3D` (9 campos, CPML, GSLS, cortical anisótropa opcional, corrección de interfaz) y tibia tubular | Dominios de pocos cm a ≲ 1 MHz (~27 Mceldas·paso/s por hilo); CPML clásica (cortical anisótropa sin atenuación inestable en la capa); sin speckle ni porosidad |
| Maxwell | FDTD de Yee 2D modo TM, CPML; banda estrecha o **banda ancha** con Debye multipolo (ADE), promedio sub-celda, Numba; SAR | 2D TM, sin antenas ni fuentes realistas; Debye sólo vale dentro de la banda ajustada |
| Óptica | Difusión estacionaria (FD + gradiente conjugado), atenuación balística y **Monte Carlo de fotones** (MCML/MCX en 2D extruido, Fresnel, resolución temporal, Numba) | n único por tejido y g = 0.9 para todos (no verificado); superficie curva en escalera; sin fluorescencia |
| Rayos X/TAC | Beer-Lambert NIST/ICRU-44, espectro de tubo con líneas K, ánodo y filtros, haz paralelo y en abanico (FBP, OS-SART con TV), scatter Monte Carlo, detector (MTF, NPS) | 2D; scatter sin electrones ni fluorescencia; sin corrección de scatter, short scan ni cono 3D; HVL −5 a −7 % frente a SpekPy a 50-70 kVp |
| RM | Ecuaciones de señal analíticas (rápidas, por defecto) o **Bloch espacial** (isocromatos, desplazamiento químico, B0 por susceptibilidad, T2*, bobina, ruido en k-space) | 2D cartesiano, spoiling ideal, sin flujo ni movimiento, sin comparar con KomaMRI/JEMRIS |
| EIT | Cuasi-estático div(σ*∇φ) = 0 con la admitividad de Gabriel, COCG con multigrid, electrodos ideales o **CEM**, matriz de sensibilidad, reconstrucción de un paso y **Gauss-Newton** (Tikhonov, Laplaciano, NOSER, TV), métricas GREIT | 2D con diferencias finitas (sin FEM ni 3D), sin GREIT entrenado; impedancia de contacto sin valor de referencia verificado |

Qué lo distingue: **una sola anatomía paramétrica 2D (tibia, antebrazo,
muslo; cortes longitudinales y transversales), con máscara de verdad de
terreno de la fractura, hematoma y edema, reproducible por semilla a
cualquier resolución, vista por cinco físicas distintas, y un generador de
datasets etiquetados.** Ningún simulador de las tablas siguientes ofrece eso
en un solo paquete (ver §9). No lo distingue la escala, la velocidad absoluta
ni la validación externa, donde los demás están por delante.

## 2. Ultrasonido

| Herramienta | Método | 2D/3D | GPU | Atenuación / dispersión | Validación publicada | Licencia | Lenguaje | Uso principal |
|---|---|---|---|---|---|---|---|---|
| **wavesim** | FDTD elástico P+S, 4.º orden, CPML (M-PML si un sólido la cruza); solver 3D de dominio pequeño; cadena de imagen (STA, DAS/CF/DMAS/SLSC) | 2D (+ 3D reducido) | No (Numba en CPU) | GSLS por mecanismo (2 por defecto): α(f) a ≤ 2 % de la ley α₀·f^y y c(f) a ≤ 2 m/s del modelo; cortical anisótropa opcional | Validación interna frente a analítico en `docs/validacion/`: convergencia de 2.º orden, R(θ), Lamb S0/A0 ≤ 2 %, Christoffel ≤ 0.1 %; contra k-Wave sólo el caso de dos capas fluidas (§10) | Ver `pyproject.toml` (no declarada en esta copia) | Python/NumPy | Visibilidad de fracturas, datasets |
| **k-Wave** (MATLAB) | Pseudoespectral con corrección k-space; solver fluido 1D/2D/3D/axisimétrico; solver elástico `pstdElastic2D/3D` (PSTD) | 1D/2D/3D | Sí (binarios C++/CUDA; el elástico en 2016 sólo vía `gpuArray` de MATLAB según el foro, dato antiguo) | Ley de potencia con Laplaciano fraccionario (absorción y dispersión causal); el elástico usa Kelvin-Voigt (∝ f² a baja frecuencia) | Treeby & Cox 2010 (JBO y JASA); Treeby et al. 2014 (elástico); participó en benchmarks multi-código (ver Aubry 2022, participación concreta [NV]) | LGPL (3.0 en el wheel de Python; versión para MATLAB [?]) | MATLAB, C++, CUDA | Fotoacústica, ultrasonido terapéutico, imagen, transcraneal |
| **k-wave-python** 0.6.2 | Mismo k-Wave; backend `python` (NumPy/CuPy) y backend `cpp` (binarios OMP/CUDA) | 1D/2D/3D, axisimétrico | Sí (CuPy o CUDA con capacidad de cómputo 7.5 o superior según README de v0.6.x) | Igual que k-Wave (fluido) | 47 pruebas de paridad con MATLAB según notas de v0.6.1 | LGPL-3.0 (verificado en el paquete instalado) | Python | Idem, desde Python |
| **SimSonic** | FDTD elástico (Bossy, Laugier et al.), 2D y 3D | 2D/3D | [?] | Anisotropía y microporosidad modeladas en cortical; detalle de la atenuación [NV] | Bossy et al. 2004 (JASA, axial transmission en cortical 3D) y 2005 (PMB, hueso trabecular desde µCT sincrotrón); usado en decenas de estudios de ultrasonometría ósea | [?] (la web oficial no era accesible) | [NV] (C/C++) | **Ultrasonometría cuantitativa de hueso**, transmisión axial, hueso trabecular |
| **Field II** | Respuesta impulsional espacial (Tupholme-Stepanishen), acústica lineal | 3D (campo) | No | Atenuación simple [NV]; sin ondas S ni medios heterogéneos [NV] | Jensen & Svendsen 1992; Jensen 1996; referencia estándar de imagen en modo B | Gratis con obligación de citar (guía de usuario 3.30) | MATLAB (núcleo C) | Diseño de sondas, PSF, B-mode con dispersores puntuales, Doppler |
| **FOCUS** | Método de campo cercano rápido (FNM) + espectro angular | 3D | [NV] | Atenuación en el dominio de la frecuencia [NV] | Comparado con Field II (McGough y colaboradores; Honer & McGough 2025, Sensors) | [?] | MATLAB / C++ | Campos de transductores, terapia térmica, plane-wave y Doppler |
| **j-Wave** | Pseudoespectral (puerto de k-Wave a JAX), diferenciable; dominios del tiempo y de Helmholtz | 1D/2D/3D | Sí (JAX CUDA, TPU) | Absorción tipo k-Wave [NV] | Stanziola et al., SoftwareX 22 (2023) 101338, comparado con k-Wave | LGPL-3.0 | Python/JAX | Problemas inversos con autodiferenciación, aprendizaje |
| **Stride / Devito** | FDTD generado por Devito, con gradientes | 2D/3D | Sí (CPU y GPU) | Fluidos con atenuación [NV] | Cueto et al., Comput. Methods Programs Biomed. 2022 | [?] | Python (Devito) | Tomografía ultrasónica computacional (FWI) |
| **Fullwave 2.5** | FDTD 8.º orden espacial / 4.º temporal, multi-GPU | 2D (y 3D [NV]) | Multi-GPU | Atenuación ley de potencia **heterogénea** (alpha0 y gamma variables en el espacio) | Sode & Pinton (PyPI/preprint) | [?] | Python/CUDA | Imagen y propagación en tejido heterogéneo |
| **Sim4Life (acústica, ZMT/IT'IS)** | FDTD multi-GPU en mallas rectilíneas adaptativas, lineal y no lineal; mapas de hueso desde CT | 3D | Multi-GPU | Atenuación en medios heterogéneos: la documentación consultada no lo detalla [?] | Verificado y validado, especialmente en ultrasonido focalizado transcraneal (según el benchmark de Aubry 2022 [NV: confirmar] y material del fabricante) | Comercial; precios y tipos de licencia no confirmados | C++/CUDA [NV] | Dosimetría y ultrasonido focalizado con la Virtual Population |
| **COMSOL (Acústica)** | FEM; `Elastic Waves, Time Explicit` y `Pressure Acoustics, Time Explicit` con dG-FEM (formas de 4.º orden por defecto) | 2D (deformación plana generalizada) y 3D | [NV] | Disipación disponible en las interfaces dG de fluidos (desde 5.5) | Documentación del fabricante | Comercial [?] | C++ (GUI) | Geometrías complejas, acoplamiento fluido-sólido-estructura, CAD |

Trabajos específicos de **fractura y ultrasonometría ósea** (no son
simuladores sino usos de los anteriores):

- Transmisión axial en cortical 3D con FDTD: Bossy, Talmant & Laugier,
  JASA 115 (2004) 2314: la profundidad de cortical que contribuye a la onda
  lateral es ~1-1.5 mm entre 0.5 y 2 MHz; la anisotropía reduce a la mitad el
  rango de velocidades.
- Guías de onda: revisiones de Moilanen (IEEE TUFFC 55, 2008) y Tran, Le & Ta
  (IEEE TUFFC 69, 2022).
- Fractura con FDTD 2D: Li et al. 2017 (100 kHz, razón S0/A0 para ancho y
  ángulo de la fractura); Dodd et al. y Humphrey et al. (200 kHz, pérdida de
  transmisión por fractura en imitaciones de cortical, simulación ajustada a
  medidas in vitro); Fotiadis y colaboradores: las condiciones de contorno
  con tejido blando cambian la dispersión de las ondas guiadas respecto a
  placa libre.
- Datos sintéticos: una tesis de Rutgers entrena segmentación en hueso con
  ecografía simulada a gran escala; un trabajo de IEEE IUS 2019 (Tokio) usa
  k-Wave para generar RF de entrenamiento (agua, hueso y alambre, sin tejido
  blando); IEEE DataPort aloja RF simulada de cortical (simulador no
  confirmado). No se encontró un estudio que entrene un clasificador de
  fracturas sólo con fracturas simuladas y lo valide en pacientes.

### Qué hacen mejor ellos

Brechas de la primera redacción, con su estado actual:

- **3D y anisotropía: parcialmente cerrada.** wavesim tiene ahora un solver
  3D (`ElasticSolver3D`, tibia tubular) y cortical transversalmente isótropa
  opcional (tensor de Rho 1996; c13 y densidad no verificados). Velocidades
  P/S a ≤ 0.4 % y 2D extruido = 3D a ~1e-6 ([elastico3d](validacion/elastico3d.md),
  [anisotropia](validacion/anisotropia.md)). **Sigue por detrás de SimSonic,
  Sim4Life y COMSOL**: ~27 Mceldas·paso/s por hilo (dominios de pocos cm a
  ≲ 1 MHz), CPML clásica (una cortical anisótropa sin atenuación que entra en la
  capa es inestable), sin speckle, porosidad ni rugosidad, y sólo a 0.5 MHz en
  el caso de aplicación. Un B-mode 3D a 3 MHz necesitaría GPU. Los ejes de la
  cortical siguen alineados con la malla (sin inclinación).
- **Rapidez y escala: abierta.** Sim4Life, k-Wave (CUDA), j-Wave, Stride y
  Fullwave corren en GPU; wavesim sigue en CPU. Medido en esta revisión: una
  tibia de 24×22 mm a 3 MHz con 3 ángulos tarda 21 s con Numba y 4 hilos y
  2 mecanismos (16 s con 1; la primera incluye la compilación). A 7-10 MHz,
  minutos o más.
- **Menos puntos por longitud de onda: abierta.** El pseudoespectral de k-Wave
  permite mallas más gruesas [NV: valor concreto]. En el FDTD de wavesim el error de
  velocidad de fase es de segundo orden por el paso temporal (+0.3 % a 16 ppw
  con CFL 0.9) y el de R es de segundo orden en la interfaz (+0.2 % a 16 ppw)
  ([elastico](validacion/elastico.md)).
- **Atenuación más flexible: cerrada en lo esencial, no en la flexibilidad.**
  Con un peso por mecanismo y tiempos de relajación optimizados, 2 mecanismos
  dan ≤ 2 % frente a α₀·f^y en [0.5, 1.8]·f₀ para y de 1 a 2 (antes 14-97 % con
  1) a ×1.5 de coste, y 3 dan ≤ 1.8 % en [0.3, 2.5]·f₀ a ×1.7
  ([atenuacion](validacion/atenuacion.md)); `n_mechanisms=2` es ahora el valor
  por defecto. **Siguen por delante** k-Wave (exponente arbitrario en toda la
  banda, con Laplaciano fraccionario) y Fullwave (α₀ y γ variables en el
  espacio): wavesim sólo ajusta una banda y el exponente es el del tejido.
- **Ondas S, conversión de modo y ondas guiadas: verificadas, no comparadas.**
  U3-U5 frente a soluciones analíticas (ver §11): R(θ) agua-cortical ≤ 0.7 %
  hasta 22°, ángulo crítico P a ±0.15°, Lamb S0/A0 ≤ 2 % de Rayleigh-Lamb, también
  con carga de agua. **Fuera de tolerancia:** |R| de 3.9-5 % en los bordes de
  28-53° y 59-70° y de 9-17 % a ±3° de los ángulos críticos (por el montaje finito, no
  cambia con la malla); S0 de la placa libre con 20 celdas en el espesor llega a
  8.7 % a 400-500 kHz (con 30 celdas, 1.3 %). Se encontró y corrigió un error
  real del solver en la interfaz fluido-sólido a incidencia oblicua (4.6 → 1.5 % a
  10 ppw; 2.3 → 0.4 % a 20 ppw). No se ha comparado con SimSonic ni con k-Wave
  elástico.
- **Estabilidad con ondas guiadas: corregida con un coste.** La cortical que
  cruza la CPML crecía exponencialmente (la inestabilidad clásica de las PML con
  ondas retrógradas); la M-PML automática la elimina en 40 000 pasos, pero
  refleja −28 dB (amplitud, 4 %) en el hueso y −53 dB en agua. En el B-mode de
  tibia la envolvente cambia 1.3 % (L2). Queda sin investigar un fantoma
  osteoporótico realista sin atenuación que crece despacio, y el 3D no tiene
  M-PML ([estabilidad](validacion/estabilidad.md)).
- **Imagen: cadena cerrada, calidad medida.** Sonda con respuesta de elemento,
  onda plana, focalizada y STA, DAS/CF/DMAS/SLSC, métricas de PSF y speckle
  ([imagen_ultrasonido](validacion/imagen_ultrasonido.md)). FWHM lateral dentro
  del 12 % de la teoría, SNR de speckle 1.85 ± 0.10 frente a 1.91 y gCNR de
  0.91-0.97 en un quiste. **No cumple** el objetivo ≤ 10 % de FWHM axial para un
  dispersor puntual a 2 MHz (+9 a +17 %; con una interfaz plana, +1.9 %). Sin ruido
  electrónico ni observador, ninguna variante mejora de forma clara la visibilidad
  de la fisura de 0.15 mm. No hay sondas reales (curvadas, matriciales,
  geometría de un fabricante) ni Field II.
- **No linealidad, terapia, transductores reales: abierta.** k-Wave, Field II y
  FOCUS; wavesim es de acústica lineal.
- **Hueso trabecular desde µCT** (SimSonic): abierta; wavesim lo trata como
  medio efectivo (con red trabecular dibujada en el modo realista).
- **Validación publicada y revisada por pares: abierta.** k-Wave, Field II,
  SimSonic y Sim4Life tienen años de comparaciones externas; la de wavesim es
  interna, frente a soluciones analíticas, más el cruce antiguo con k-Wave de §10.

### Qué aporta wavesim

- La misma anatomía con **máscara de fractura/hematoma/edema** alimenta
  cinco físicas (ver §9) y un generador de datasets.
- Solver elástico con conversión de modo en el hueso **y** fluidos en un
  paquete de decenas de ficheros Python legibles, sin MATLAB ni binarios,
  ahora con atenuación de ley de potencia a ≤ 2 % y validación analítica de
  ondas S y guiadas (que k-wave-python 0.6.2 no cubre).
- Rugosidad de interfaces y fracturas paramétricas (transversal, oblicua,
  fisura, conminuta, incompleta, desplazada) reproducibles por semilla.
- Atenuación viscoelástica por tejido y speckle por microestructura
  integrados en el mismo flujo.

### Cuándo recomendar otra herramienta

| Si necesitas... | Usa |
|---|---|
| Ultrasonido 3D con hueso cortical anisótropo a 1 MHz o más, transmisión axial, ondas guiadas en un tubo (wavesim 3D llega a dominios pequeños a ≲ 1 MHz) | SimSonic (si lo consigues bajo su licencia [?]), Sim4Life o COMSOL |
| Física de ondas S en un medio mixto fluido-sólido a gran escala con GPU | Sim4Life o COMSOL (dG-FEM), o k-Wave elástico en MATLAB (CPU) |
| Propagación fluida 2D/3D con atenuación de ley de potencia rápida y de referencia | k-Wave / k-wave-python |
| Optimizar o invertir (FWI, tomografía ultrasónica, redes diferenciables) | j-Wave o Stride |
| Diseño de sondas, PSF, B-mode de dispersores puntuales con rapidez extrema | Field II o FOCUS |
| Cumplimiento regulatorio o dosimetría con fantomas humanos 3D | Sim4Life + Virtual Population |

## 3. Electromagnetismo (RF, microondas, ondas milimétricas)

| Herramienta | Método | 2D/3D | GPU | Dispersión | Validación publicada | Licencia | Lenguaje | Uso principal |
|---|---|---|---|---|---|---|---|---|
| **wavesim `em.fdtd_tm`** | FDTD de Yee 2.º orden, TM, CPML; banda estrecha o banda ancha (ADE de Debye multipolo), sub-celda, Numba | 2D | No | Cole-Cole de Gabriel ajustado por 6 polos de Debye comunes (≤ 0.7 % en 0.1-10 GHz) | Frente a soluciones exactas: β ≤ 0.23 %, α ≤ 1.6 %, losa ≤ 0.31 dB, Mie 2D 1.7-3.3 % ([maxwell](validacion/maxwell.md)) | (ver arriba) | Python/NumPy (+ Numba) | Penetración, campo y SAR en el corte |
| **openEMS** | FDTD por circuito equivalente (Liebig et al., Int. J. Numer. Model.), cartesianas y cilíndricas | 3D (cilíndricas incluidas) | No confirmado | Lorentz/Drude [NV] | Validado con antenas y estructuras de microondas (Liebig et al.); no biomédico de forma específica | GPL v3 o posterior (CSXCAD, LGPL) | C++ con interfaces Matlab/Octave y Python | Antenas, circuitos de RF |
| **Meep** | FDTD de Yee, subpíxel, no lineal (Padé), cilíndrico | 1D/2D/3D, cilíndrico | No confirmado | ε(ω), μ(ω) con pérdidas o ganancia | Oskooi et al., Comput. Phys. Commun. 181 (2010) 687 | GPL (2.0 o posterior según GitHub) | Python, Scheme, C++ | Fotónica; 2D TM listo para uso general |
| **gprMax** | FDTD, 2D y 3D, material dispersivo multipolo, CPML | 2D/3D | Sí (CUDA, OpenCL, Metal), MPI | Multipolo (Debye, Drude, Lorentz) | Warren et al., Comput. Phys. Commun. 209 (2016) y 237 (2019, motor GPU hasta 30 veces más rápido que CPU OpenMP) | GPL v3 o posterior | Python + Cython | Georradar; también bioelectromagnetismo y SAR (según README) |
| **Sim4Life / SEMCAD X** | FDTD con solver GPU, fantomas Virtual Population, SAR validado | 3D | Sí (CUDA) | Sí [NV] | Cálculo de SAR validado (documentación del fabricante) | Comercial; **precios y términos no confirmados**; SEMCAD X Matterhorn comparte código con Sim4Life | C++/CUDA | Dosimetría (SAR), exposición, dispositivos implantables |
| **CST Studio Suite / COMSOL RF** | FIT/FDTD/FEM, etc. | 3D | [NV] | Sí [NV] | [NV] | Comerciales [?] | | Antenas, dispositivos |

Detección de fracturas con microondas (no son simuladores, son usos):
Santos, Fernandes & Costa (IEEE OJAP 2022; Vivaldi, 8.3-11.1 GHz, migración de
Kirchhoff, fisuras transversales de 1 mm de ancho y 13 mm de profundidad con
2 mm de piel), Santos et al. (IEEE Access 2023), Nouri Moqadam & Kazemi
(Sci. Rep. 2023) y Bosco et al. (Curr. Med. Imaging 2024). El solver usado
no figura en los fragmentos consultados.

**Qué hacen mejor ellos.** Brechas de la primera redacción:

- **Materiales dispersivos de banda ancha: cerrada.** `BroadbandMaxwellTMSolver`
  ajusta el Cole-Cole de Gabriel con polos de Debye comunes (error máximo de
  ε′ y σ ≤ 0.7 % en 0.1-10 GHz) y obtiene muchas frecuencias de un solo pulso.
  En músculo y grasa homogéneos, β a ≤ 0.23 % y α a ≤ 1.6 % hasta 10 GHz;
  losa piel/grasa/músculo/cortical a ≤ 0.31 dB en toda la banda con
  dx = 0.125 mm (0.88 dB a 10 GHz con 0.25 mm: 60 dB de pérdida total); frente a
  la serie de Mie 2D, 1.7-3.3 % con 15-20 puntos por longitud de onda y
  promedio sub-celda (sin él, 6-11 %). Kernels Numba 2-5 veces más rápidos que
  NumPy ([maxwell](validacion/maxwell.md),
  [rendimiento_em](validacion/rendimiento_em.md)). Frente a gprMax/Meep esto
  iguala el *método*, no la escala ni la verificación cruzada: no se ha
  ejecutado ninguno de los dos.
- **Abiertas:** 3D y modo TE; antenas reales, adaptación de impedancia y
  arrays (openEMS, Meep, gprMax); mallas no uniformes; SAR en cuerpo entero con
  fantomas 3D (Sim4Life); GPU. La fuente de onda plana de banda ancha debe estar
  en una zona homogénea, y quedan ~0.8 % de error de campo fuera del cilindro
  de hueso sin origen identificado.

**Qué aporta wavesim.** La misma geometría de fractura con edema y hematoma
alimenta el Maxwell, con las propiedades de Gabriel del mismo diccionario de
tejidos que usa el ultrasonido.

**Cuándo recomendar otra:** antenas o arrays de microondas para imagen
(openEMS, Meep, gprMax), SAR normativo (Sim4Life), 3D.

## 4. Óptica (UV, visible, IR, THz)

| Herramienta | Método | 2D/3D | GPU | Atenuación/dispersión | Validación publicada | Licencia | Lenguaje | Uso |
|---|---|---|---|---|---|---|---|---|
| **wavesim `em.optics`, `em.montecarlo`** | Difusión estacionaria (FD + CG), balístico y Monte Carlo de fotones (MCML/MCX, Numba) | 2D extruido (fotones en 3D) | No (Numba en CPU) | μa, μs' por tejido y longitud de onda (orden de magnitud en IR medio y UV); n = 1.4 y g = 0.9 únicos | MC frente a van de Hulst (R, T a ≤ 1.6 σ), Fresnel exacto (1e-7), balístico (≤ 1.2 σ) ([optica](validacion/optica.md)) | (ver arriba) | Python/NumPy (+ Numba) | Penetración de la luz en la extremidad |
| **MCX** | Monte Carlo de fotones en voxels 3D (Fang & Boas 2009) | 3D | Sí (CUDA); MCXCL para OpenCL | μa, μs, g, n por voxel | Fang & Boas, Opt. Express 17 (2009) 20178 | GPL v3 | C/CUDA; Python (`pmcx`), MATLAB | Estándar de facto para Monte Carlo GPU |
| **MCML** | Monte Carlo en capas planas infinitas (Wang, Jacques & Zheng 1995) | Capas 1D (simetría cilíndrica) | No | μa, μs, g, n por capa | Wang et al., Comput. Methods Programs Biomed. 47 (1995) 131 [NV: cita] | [?] | C | Referencia histórica |
| **NIRFAST** | FEM, difusión (Dehghani et al., 2009) | 2D/3D | [NV] | Múltiples longitudes de onda | Dehghani et al., Commun. Numer. Methods Eng. 25 (2009) 711 | [?] | MATLAB | Tomografía óptica difusa |
| **Toast++** | FEM, difusión, solvers inversos (Schweiger & Arridge) | 2D/3D | No | Dominio del tiempo, frecuencia y estacionario [NV] | J. Biomed. Opt. 19 (2014) 040801 | GPL (según su página en SourceForge) | C++ con MATLAB/Python | Reconstrucción de μa y μs' |

**Qué hacen mejor ellos.** Brechas de la primera redacción:

- **Monte Carlo: cerrada en 2D extruido.** wavesim tiene ahora un Monte Carlo
  estilo MCML/MCX con Henyey-Greenstein, Fresnel, ruleta rusa y resolución
  temporal opcional, con 540 000 fotones/s en la losa (Numba, 1 hilo). Frente a
  la tabla 35 de van de Hulst, R = 0.09758 ± 0.00012 (0.09739) y T = 0.66084 ±
  0.00014 (0.66096); losa de índice 1.5 sin dispersión, 1e-7 del exacto; atenuación
  balística ≤ 1.2 σ. Con él se midió lo que la difusión no ve: la fluencia en la
  piel iluminada es ×1.27 la de la difusión a 0.25 mm y la difusión sobreestima
  ~7 % en el interior de la grasa; lejos de las fronteras coinciden en μ_eff a
  ≤ 1.3 % ([optica](validacion/optica.md)).
- **Abiertas:** MCX sigue siendo el estándar (3D, GPU, fluorescencia); el
  fantoma de wavesim es 2D extruido y de superficie en escalera, con un índice de
  refracción (1.4) y un g (0.9) únicos y **sin valores de g por tejido
  verificados**. La transmisión de la pierna es ~10⁻⁸ en las columnas gruesas, al
  límite de lo medible con 2·10⁶ fotones. No se ha comparado con MCX ni MCML
  ejecutados. NIRFAST/Toast++ siguen aportando reconstrucción inversa y 3D.
  En una extremidad de ~5 cm la señal es completamente difusa.

**Qué aporta wavesim.** El mismo corte con fractura, en el mismo "paciente"
que la ecografía, y la conclusión cuantitativa (pérdida en dB) de que la luz
no forma imagen del hueso a través de la extremidad.

**Cuándo recomendar otra:** cualquier estudio cuantitativo de fluencia en
3D o con fuentes cercanas al tejido (MCX), tomografía óptica difusa con
inversión (NIRFAST o Toast++).

## 5. Rayos X y TAC

| Herramienta | Método | 2D/3D | GPU | Física incluida | Validación publicada | Licencia | Lenguaje | Uso |
|---|---|---|---|---|---|---|---|---|
| **wavesim `em.xray`, `fanbeam`, `xray_scatter`, `xray_detector`** | Beer-Lambert polienergético (espectro con líneas K, ánodo, filtros), FBP paralelo y en abanico, OS-SART, scatter Monte Carlo, detector | 2D, paralelo o abanico | No (Numba en CPU) | Endurecimiento, ruido de Poisson, líneas K, Compton/Rayleigh en el scatter (sin electrones ni fluorescencia), MTF y NPS | Beer-Lambert y FBP exactos, HU, NIST < 1.5 %, abanico = paralelo, HVL frente a SpekPy ±5 % a 80-120 kVp ([rayos_x](validacion/rayos_x.md)) | (ver arriba) | Python/NumPy (+ Numba) | Radiografía y TAC del corte |
| **GATE (Geant4)** | Monte Carlo de partículas | 3D | Variantes [NV] | Física completa de fotones y electrones | Jan et al., Phys. Med. Biol. 49 (2004) 4543 (PET/SPECT) y años de validaciones; nacido para TAC y PET/SPECT | LGPL en la versión pública inicial; términos actuales [?] | C++ con macros | PET, SPECT, TAC, radioterapia |
| **TOPAS** | Envoltura de Geant4 (Perl et al., Med. Phys. 39, 2012) | 3D | No | Física completa, orientado a radioterapia/protones | Perl et al. 2012 | Gratis para investigación "en intención" según el artículo; términos exactos [?] | C++ con ficheros de parámetros | Radioterapia con protones |
| **MC-GPU** | Monte Carlo de rayos X en GPU, voxels; secciones eficaces PENELOPE 2006 (sin transporte de electrones) | 3D | Sí (CUDA) | Dispersión Compton, Rayleigh, fotoeléctrico | Badal & Badano, Med. Phys. 36 (2009) 4878 [NV: cita]; 20-40 veces más rápido que 1 núcleo | [?] (distribuido por la FDA) | C/CUDA | Radiografía y TAC con dispersión realista |
| **ASTRA** | Proyectores y reconstructores en GPU | 2D/3D | Sí | Sólo geometría de proyección | van Aarle et al., Ultramicroscopy 157 (2015) y Opt. Express 24 (2016) | GPL v3 | C++/CUDA; MATLAB y Python | Reconstrucción de TAC/tomografía |
| **TIGRE** | Reconstrucción CBCT en GPU (Bath/CERN) | 3D | Sí (CUDA) | Sólo geometría | Biguri et al., Biomed. Phys. Eng. Express 2 (2016) 055010 | BSD | MATLAB/Python con CUDA | CBCT |
| **XCIST / CatSim** | Simulador analítico de TAC con fantomas y reconstrucción | 3D | [NV] | Espectro, scatter, ruido, detector (según PMB 2022); usa secciones eficaces de Geant4 | "XCIST: an open access x-ray/CT simulation toolkit", Phys. Med. Biol. 2022 | [?] | Python + C/C++ (`gecatsim`) | Simulación de TAC realista |

**Qué hacen mejor ellos.** Brechas de la primera redacción:

- **Scatter: parcialmente cerrada.** Monte Carlo de fotones con Klein-Nishina
  y función de dispersión incoherente, Rayleigh con factor de forma y
  fotoeléctrico por diferencia. En un cilindro de músculo de 18 cm a 100 kVp, el scatter
  da *cupping* de +21 HU (corte de 1 cm) y +54 HU (4 cm) y baja el contraste
  hueso-músculo un 8 % y un 26 %. La SPR en agua (0.78, 2.50, 5.43 a 10, 20, 30 cm)
  es un 17-29 % menor que la de una cifra de la literatura que no se pudo
  verificar: **no hay validación cuantitativa del scatter**. Sin fluorescencia,
  transporte de electrones ni Doppler; el scatter de un TAC se calcula en pocas
  vistas y se interpola; no hay corrección de scatter en la reconstrucción.
- **Espectro y líneas K: parcialmente cerrada.** Frente a SpekPy 2.5.4
  (ejecutado en la sesión), la HVL cae dentro de ±5 % de 80 a 120 kVp pero a
  **−6.6 % y −5.2 % a 50 y 70 kVp (fuera del objetivo)**; los dos parámetros libres se
  ajustaron a esos mismos casos, así que no es una predicción independiente. Las
  tablas de Cu y W y las energías de las líneas K están de memoria, sin
  verificar.
- **Geometrías reales y reconstrucción iterativa: parcialmente cerrada.**
  Abanico curvo y plano con FBP exacta (RMSE del disco 0.014-0.016, igual al haz
  paralelo) y OS-SART con TV: con 30-60 vistas el RMSE baja un 25-32 % respecto de
  FBP, pero con 180 vistas ya no gana. **Siguen abiertos** el haz cónico 3D, el
  *short scan* con pesos de Parker y los proyectores en GPU de ASTRA/TIGRE.
- **Detector y calidad de imagen: cerrada en lo básico.** MTF por el método
  del borde (error ≤ 0.027 hasta MTF ≈ 0.6) y NPS del TAC (Parseval al 0.3 %); no hay
  referencia cuantitativa verificada para la forma del NPS.
- **Física completa de partículas y dosimetría: abierta.** GATE, MC-GPU y TOPAS.

**Qué aporta wavesim.** TAC y radiografía del mismo corte que la ecografía y
la RM, y rapidez (segundos para el TAC analítico).

**Cuándo recomendar otra:** dosimetría o imagen con scatter fiable (MC-GPU,
GATE), CBCT/geometrías reales en 3D (XCIST, TIGRE, ASTRA).

## 6. Resonancia magnética

| Herramienta | Método | 2D/3D | GPU | Física | Validación publicada | Licencia | Lenguaje | Uso |
|---|---|---|---|---|---|---|---|---|
| **wavesim `mri`** | Dos motores: señal analítica por tejido (rápida, por defecto) y Bloch espacial de isocromatos con eventos y estados de desfase | 2D cartesiano | No | T1, T2, PD, T2', desplazamiento químico (1 o 6 picos), campo B0 por susceptibilidad, bobina de superficie, ruido en k-space; SE/IR/GRE | Bloch espacial = ecuaciones de señal a 1e-11; desplazamiento químico exacto; cilindro de susceptibilidad ≤ 0.7 % con B0 alineado ([rm](validacion/rm.md)); no comparado con KomaMRI/JEMRIS | (ver arriba) | Python/NumPy (+ Numba) | Contraste y artefactos de la fractura (edema, hematoma) |
| **JEMRIS** | Bloch con sólo CPU multihilo; secuencias autocontenidas | 3D | No | Bloch completo con campos B0/B1/gradiente, pulsos reales | Stöcker et al., Magn. Reson. Med. 64 (2010) 186 | GPL-2.0 o posterior (catálogo Helmholtz) | C++ (MATLAB/Python) | Secuencias, artefactos |
| **MRiLab** | Bloch en GPU, GUI | 3D | Sí | Bloch completo | Liu et al., Magn. Reson. Med. 2017 | [?] | MATLAB/C++/CUDA | Docencia, diseño de secuencias |
| **KomaMRI.jl** | Bloch en CPU y GPU, secuencias Pulseq | 3D | Sí | Bloch completo, movimiento | Castillo-Passi et al., Magn. Reson. Med. (2023): error medio absoluto del 0.1 % frente a JEMRIS | MIT | Julia | Simulación general |
| **MRXCAT** | Generador de fantomas basado en XCAT con modelo de señal, bobinas y ruido | 2D+t / 3D | No | Señal cardíaca con movimiento | Wissmann et al., J. Cardiovasc. Magn. Reson. 16 (2014) 63 | [?] ("disponible online" según el artículo) | MATLAB [NV] | RM cardíaca |

**Qué hacen mejor ellos.** Brechas de la primera redacción:

- **Bloch espacial y artefactos: cerrada en 2D cartesiano.** `engine="bloch"`
  da desplazamiento químico de la grasa (1.7 píxeles con 130 Hz/píxel a 1.5 T,
  idéntico al esperado), distorsión y pérdida de señal por el campo B0 de la
  susceptibilidad (cilindro infinito a ≤ 0.7 % con B0 alineado con la malla;
  2.6-5.5 % con B0 oblicuo, por la escalera), desfase intravóxel y sensibilidad de
  bobina. Una imagen de pierna de 128×128 tarda ~1 s en modo estacionario
  y ~50 s línea a línea ([rm](validacion/rm.md)).
- **Abiertas:** 3D, perfil de corte y pulsos selectivos reales, flujo,
  movimiento y difusión, trayectorias radial y espiral, spoiling con gradientes
  reales y ecos estimulados residuales, secuencias Pulseq, GPU. **No se ha
  comparado numéricamente con KomaMRI ni con JEMRIS** (la validación es frente a
  soluciones exactas); el UTE sigue en el motor analítico. La χ de los tejidos
  blandos se aproxima por la del agua y la del hueso viene de una costilla bovina.

**Qué aporta wavesim.** Valores de T1/T2/PD de fractura (edema, hematoma)
sobre la misma anatomía que el resto, a segundos por imagen.

**Cuándo recomendar otra:** flujo, movimiento, 3D o secuencias arbitrarias
(KomaMRI, JEMRIS), diseño de secuencias o fantomas cardíacos (MRXCAT).

## 7. Impedancia eléctrica (EIT)

| Herramienta | Método | 2D/3D | Licencia | Lenguaje | Uso |
|---|---|---|---|---|---|
| **EIDORS** | FEM + inversión; Adler & Lionheart, Physiol. Meas. 27 (2006) S25 | 2D/3D | GPL v2 o v3 | MATLAB/Octave | Estándar académico en EIT (pulmón, cerebro) |
| **pyEIT** | FEM + reconstrucción (back-projection, GREIT, etc.), SoftwareX 2018 | 2D/3D | Apache 2.0 según el artículo; un listado de Anaconda dice BSD-3: comprobar el `LICENSE` del repositorio | Python | EIT en Python |

Para fracturas: la bibliografía hallada es experimental (impedancia en
conejo, placa inteligente, estudios antiguos de Aberdeen con EIT de miembro);
no se encontró ningún simulador específico de EIT para detección de
fracturas.

**Frente a `wavesim.em.quasistatic`.** wavesim incorpora ahora el modelo
completo de electrodo (CEM) con impedancia de contacto, la reconstrucción
Gauss-Newton (Tikhonov, Laplaciano, NOSER, variación total), las métricas GREIT
y un precondicionador multigrid ([eit](validacion/eit.md)). Frente a la serie
de Fourier de un disco, el CEM da errores de 1.4-1.5 % (norma 0.67 %); con
electrodos ideales sólo se llega a < 2 % con dx = 0.5 mm y electrodos de 4
celdas. Con una inclusión de sangre de 4 mm de radio y 70 dB, el error de
posición baja de 2.4-3.2 mm (un paso, NOSER) a 0.08 mm (TV), 0.18 (Laplaciano) y 0.33
(Tikhonov); con NOSER falla en la periferia. En la pierna con fractura, el
un paso da 7.2 mm de error del píxel máximo y GN-Laplaciano 2.0-4.3 mm (**una sola
realización de ruido**, no es una cifra firme); un modelo ideal con datos CEM
falla en 27 mm salvo con el Laplaciano. El multigrid pasa de 922 a 13-16
iteraciones (16 inyecciones: ~255 s → 0.7 s).

**Siguen abiertos:** FEM (geometrías curvas), 3D, GREIT entrenado, patrones
de inyección no adyacentes y electrodos de forma arbitraria. El valor típico de
z_c de un electrodo real no se pudo verificar y, con la piel del fantoma, el
contacto apenas cambia la imagen. EIDORS y pyEIT siguen siendo los estándares
validados: para reconstrucción cuantitativa conviene exportar el fantoma y
usarlos; wavesim sirve para estimar si el hematoma y el edema cambian de forma
medible las tensiones de borde.

## 8. Fantomas anatómicos

| Fantoma | Tipo | Dimensión | Contenido | Acceso/licencia |
|---|---|---|---|---|
| **wavesim `phantoms`** | Paramétrico, geométrico, por semilla; modo realista (hueso poroso, red trabecular, fascias) | 2D (+ tibia tubular 3D y fractura transversal 3D) | Tibia, antebrazo, muslo; fracturas (6 tipos) con máscara; hematoma, edema | Parte del paquete |
| **XCAT (Segars et al.)** | Superficies NURBS 4D, cuerpo entero | 3D+t | Anatomía y fisiología (corazón, respiración); series por edad y tamaño | Distribuido por Duke; **términos [?]** |
| **Virtual Population (IT'IS)** | Modelos voxelizados segmentados desde RM, computables y posables | 3D | Cuerpo entero, incluidas extremidades y huesos | Académica y comercial bajo acuerdos distintos; modelos computables de alta resolución integrados con Sim4Life; **precios [?]** |
| **MIDA (Iacono et al. 2015)** | Segmentación multimodal 500 µm | 3D | **Sólo cabeza y cuello** (no sirve para extremidades) | Gratis (salvo gastos de gestión), DOI 10.13099/ViP-MIDA-V1.0 |

**Qué hacen mejor ellos.** Anatomía real, 3D, variabilidad poblacional
(XCAT, VP) y validación frente a pacientes. wavesim es una idealización
paramétrica (capas y anillos con rugosidad), útil para datasets
controlados pero no sustituye a un fantoma segmentado. **Qué aporta wavesim.**
Fracturas con verdad de terreno píxel a píxel, que no traen ni XCAT ni VP
[NV: comprobar si XCAT incluye lesiones óseas].

**Cuándo recomendar otro:** cualquier estudio que necesite anatomía 3D real
o variabilidad entre sujetos (VP, XCAT).

## 9. Multimodalidad y datasets: qué aporta wavesim que no tienen los demás

- Ninguno de los simuladores de las tablas cubre ultrasonido elástico,
  Maxwell, difusión óptica, rayos X/TAC y RM en un único paquete que lea
  la misma anatomía. En la práctica se combinan MATLAB (k-Wave), C++
  (openEMS), Julia (Koma), CUDA (MCX, ASTRA)... con formatos de geometría
  distintos.
- La máscara de fractura/hematoma/edema (verdad de terreno) es una
  contribución propia del proyecto; en la búsqueda no apareció ningún
  simulador abierto que la ofrezca para varias modalidades.
- El generador de datasets etiquetados (`wavesim/datasets/`) y la
  reproducibilidad por semilla a cualquier resolución.
- Coste cero de entrada: `pip install -e .` con sólo NumPy y Matplotlib.

Límite de esta afirmación: "no apareció" en una búsqueda web limitada, no
una revisión sistemática.

## 10. Comparación numérica con k-Wave

Script: `benchmarks/kwave_comparison.py` (cabecera con instrucciones de
instalación; `k-wave-python` **no** está en `pyproject.toml`).

> **Estado de esta sección.** §10.0 resume la comparación cruzada actual
> (noviembre de 2026). §10.1-10.4 son la primera comparación con k-Wave, hecha
> con una versión anterior de wavesim; se conservan como histórico y sus casos A
> y B están repetidos con el wavesim actual en
> [cruzada_ultrasonido](validacion/cruzada_ultrasonido.md).

### 10.0 Comparación cruzada con simuladores de referencia

Cada herramienta se instaló en un entorno aparte (no es dependencia del
proyecto) y se ejecutó en CPU con los mismos casos que wavesim. Los resultados
de referencia están guardados en `benchmarks/data/cross_*/` y los tests
`tests/test_cross_*.py` comparan wavesim con ellos sin la herramienta instalada.

| Modalidad | Referencia | Resultado medido | Detalle |
|---|---|---|---|
| Ultrasonido, fluido | k-Wave (k-wave-python 0.6.2) | R y T de dos capas a ≤ 0.28 % entre códigos; α a ±1.1 % y c(f) a ≤ 0.22 % (el −7.7 % de §10.3 era la convención de `sound_speed` de k-Wave). Tejido heterogéneo: forma de onda a 0.3-0.9 % con CFL 0.3, pero **6-8 % con la CFL 0.9 por defecto** (dispersión temporal de wavesim; no cumple el 3 %) | [cruzada_ultrasonido](validacion/cruzada_ultrasonido.md) |
| Ultrasonido, elástico | Devito 4.8.23 (mismo esquema, orden 4) | Devito coincide a ≤ 0.01 % con wavesim *sin* la corrección de interfaz; con ella, \|R\| a 45° queda a 0.11-0.69 % de la analítica frente a 1.4-3.2 % de Devito. Lamb S0/A0 ≤ 0.47 %. k-wave-python no tiene solver elástico | ídem |
| Maxwell | gprMax 4.0.1 (mismos polos de Debye) | Losa 0.5-6 GHz: 0.013 dB y 0.00°. Pierna a 0.5 mm, banda ancha: \|E\| 0.04 % y SAR 0.13 % de mediana; banda estrecha: 0.62 % y 1.24 %; pico de SAR en la misma celda (cociente 0.993-1.011). La banda estrecha se separa ~0.5 % sin causa identificada. Meep no instaló | [cruzada_maxwell](validacion/cruzada_maxwell.md) |
| Óptica | MCML 1.2 y MCX (pmcxcl 0.8.1, CPU) | R, A y T totales a ≤ 0.0008 de MCML (≤ 0.15 % de R); perfiles R(x), T(x) y fluencia a ≤ 2.5 % (MCX frente a MCML: ≤ 4.8 %). La difusión se aparta hasta −25 % bajo la fuente. Un solo escenario (800 nm, g = 0.9) | [cruzada_optica](validacion/cruzada_optica.md) |
| Rayos X, geometría | ASTRA 2.5.0 (CPU) | Proyector a 0.09-0.40 %; FBP paralela y SIRT a 1-4 % del error frente al objeto. Sin referencia para el abanico curvo ni su FBP (ASTRA sólo en GPU; TIGRE exige CUDA) | [cruzada_rayos_x](validacion/cruzada_rayos_x.md) |
| Rayos X, espectro | SpekPy 2.5.4, 23 casos fuera del ajuste | HVL: error medio 2.8 %, máx. 7.4 % (**+6-7 % a 140-150 kVp, −6 % a 50-60 kVp**); energía media ≤ 2.5 % | ídem |
| Rayos X, scatter | XCIST 1.6.9 | **No concuerdan**: SPR wavesim/XCIST = 1.3 (10 cm), 3.0 (20 cm), 8.3 (35 cm). XCIST usa un núcleo de convolución calibrado, no un Monte Carlo: no decide cuál acierta | ídem |
| EIT | pyEIT 1.2.4 (FEM) | dV con inclusión a 0.5 % (dx = 0.25 mm); tensiones absolutas de wavesim a 1.3 % de la analítica puntual (electrodo de tamaño finito); posición reconstruida 0-0.9 mm en ambos. pyEIT no tiene CEM | [cruzada_eit](validacion/cruzada_eit.md) |

Lectura: donde hay una referencia independiente del mismo tipo (k-Wave, gprMax,
MCML/MCX, ASTRA, pyEIT), wavesim coincide con ella al nivel del error de malla
o estadístico. Los puntos débiles que la comparación deja a la vista son la
dispersión temporal del ultrasonido con la CFL por defecto, el espectro de
rayos X fuera de 70-120 kVp y el scatter, que sigue sin una referencia física
independiente (haría falta GATE o MC-GPU). La comparación con pyEIT destapó un
fallo (reconstrucción de un paso singular con una celda aislada), ya corregido.
**Sin comparación cruzada todavía:** RM (KomaMRI, JEMRIS), ultrasonido frente a
SimSonic, Field II o k-Wave elástico de MATLAB, solver 3D, CEM de EIT.

### 10.1 Entorno y método

- `pip install k-wave-python scipy` funcionó (k-wave-python 0.6.2, LGPL-3.0,
  incluye binarios OMP y CUDA en el wheel). Se instaló sólo en el entorno de
  ejecución.
- **Backend C++ no utilizable aquí**: el binario OMP necesitaba
  `libhdf5_serial_hl.so.100`, que no estaba en el sistema; se instaló
  `libhdf5-hl-100t64` con `apt` y entonces ejecutaba, pero con un caso
  diminuto (266×200 celdas, 333 pasos) tardó 133 s (el backend `python` lo
  hizo en 8 s). No se investigó la causa. Todos los resultados siguientes
  usan el backend `python` de k-wave-python (NumPy puro, float64, CPU de 4
  núcleos).
- Pulso: Gaussiano con f0 = 1 MHz y ancho de banda fraccional 0.6, igual en
  ambos (`wavesim.core.signals.gaussian_pulse`). PML de 20 celdas fuera de
  la región activa en ambos códigos. wavesim usa CFL 0.9 de su límite y
  k-Wave 0.3.
- Como la amplitud absoluta de una fuente aditiva no se define igual en los
  dos códigos, las amplitudes se normalizan por la onda incidente medida por
  el mismo código. Los tiempos se miden con el máximo de la envolvente
  (señal analítica) y ajuste parabólico.
- wavesim se ejecutó como fluido (`cs = 0`); es decir, el solver elástico en
  el caso en que se reduce a acústica. **Este benchmark no ejercita ni las
  ondas S ni la conversión de modo**, que es lo que k-wave-python (0.6.2) no
  puede dar: no se encontró un solver elástico en el paquete instalado.

### 10.2 Caso A: onda plana en dos capas fluidas

Capa 1: c = 1500 m/s, ρ = 1000 kg/m³. Capa 2: c = 1800 m/s, ρ = 1100 kg/m³.
Analítico: R = 0.1379, T = 1.1379 (presión). Fuente de línea ancha de 40 mm
(plano) a 2 mm, receptores a 4 mm (incidente + reflejada) y a 15 mm
(transmitida), interfaz a 11 mm (queda a media celda por debajo de la fila
`j_int`; se tiene en cuenta en los tiempos esperados).

| dx (mm) | Puntos/λ (capa 1) | R wavesim | R k-Wave | T wavesim | T k-Wave | Err. R wavesim / k-Wave (vs analítico) | Err. T wavesim / k-Wave |
|---|---|---|---|---|---|---|---|
| 0.20 | 8 | 0.1467 | 0.1464 | 1.1321 | 1.1353 | 6.39 % / 6.15 % | 0.51 % / 0.23 % |
| 0.15 | 10 | 0.1424 | 0.1425 | 1.1343 | 1.1363 | 3.24 % / 3.29 % | 0.32 % / 0.14 % |
| 0.10 | 15 | 0.1397 | 0.1399 | 1.1372 | 1.1372 | 1.29 % / 1.41 % | 0.06 % / 0.06 % |

Tiempos de llegada (diferencias entre pulsos, independientes del origen de
tiempos) y forma de onda a 0.10 mm:

| Magnitud | Analítico | wavesim | k-Wave | wavesim − k-Wave |
|---|---|---|---|---|
| t(reflejada) − t(incidente) | 9.267 µs | 9.231 µs | 9.273 µs | −41.5 ns (−0.45 %) |
| t(transmitida) − t(incidente) | 6.883 µs | 6.856 µs | 6.886 µs | −30.7 ns (−0.45 %) |
| Cociente de amplitud incidente | | | | 1.001 |
| Error L2 relativo, receptor 1 (crudo / tras alinear y escalar) | | | | 6.4 % / 1.9 % |
| Error L2 relativo, receptor 2 (crudo / tras alinear y escalar) | | | | 13.8 % / 2.9 % |

A 0.20 y 0.15 mm el error de forma tras alinear es 5.1-5.5 % y 3.8-4.9 %.

Lectura:

- **Los dos códigos coinciden entre sí** en R y T a 0.3 % o mejor (R a 0.0003
  absoluto a 0.20 y 0.15 mm), y ambos convergen hacia el valor analítico al
  refinar la malla. T converge a 0.06 % con 15 puntos por longitud de onda.
- **El error de R no es de wavesim**: ambos códigos dan el mismo valor a cada
  resolución y baja 6.2 %, 3.3 %, 1.3 % (orden de convergencia observado ≈ 2.2
  entre dx = 0.20 y 0.10 mm). Se comprobó con wavesim solo que ensanchar la fuente a 120 y 240 mm
  no lo cambia (0.13970 a dx = 0.10 mm), así que no es difracción por la
  apertura finita. La causa exacta (interfaz escalonada y promediado de
  densidad) no se investigó.
- **El error "crudo" de forma de onda (6-14 %) está dominado por un desfase
  temporal de 30-40 ns** entre los códigos (wavesim llega antes: 0.45 % del
  tiempo de vuelo). A 1 MHz, 40 ns son 14° de fase, así que 2π·0.04 ≈ 0.25
  rad explican el error. Tras alinear y ajustar la amplitud queda 1.9-2.9 %.
  No se determinó cuánto del desfase se debe a la convención del instante
  de la fuente/receptor y cuánto a la posición efectiva de la interfaz
  escalonada (media celda = 50 µm = 33 ns a dx = 0.10 mm ya es del orden del
  desfase) o a la dispersión numérica; el desfase no converge de forma
  monótona (64, 68 y 41 ns al refinar), por lo que probablemente pesa la
  convención de posición de la interfaz.
- **Tiempo de cálculo** (4 núcleos): a dx = 0.10 mm wavesim 1.7 s frente a
  28.6 s de k-wave-python (backend Python, 1080 pasos de CFL 0.3 contra 594
  de wavesim). No es una medida del rendimiento de k-Wave en su binario
  C++/CUDA y no debe leerse como "wavesim es más rápido que k-Wave".

### 10.3 Caso B: fuente puntual en medio homogéneo con atenuación

c = 1580 m/s, ρ = 1090 kg/m³, α = 2 dB/(cm·MHz^1.1) (exponente 1.1 porque la
dispersión causal de k-Wave diverge con y = 1). Fuente puntual (en 2D, fuente
de línea) en el centro de 50×50 mm, receptores a 5, 10 y 15 mm. La
atenuación se estima de la razón espectral entre 5 y 15 mm corrigiendo la
expansión cilíndrica 1/√r; la velocidad de fase de la diferencia de fase.

| f (MHz) | 0.70 | 0.85 | 1.00 | 1.15 | 1.30 |
|---|---|---|---|---|---|
| α nominal (dB/cm) | 1.351 | 1.673 | 2.000 | 2.332 | 2.669 |
| α k-Wave medida | 1.252 | 1.546 | 1.846 | 2.153 | 2.458 |
| α wavesim 1 mecanismo | 1.421 | 1.808 | 2.146 | 2.437 | 2.681 |
| Error vs nominal, k-Wave | −7.3 % | −7.5 % | −7.7 % | −7.7 % | −7.9 % |
| Error vs nominal, wavesim 1 mec. (igual que antes) | +5.2 % | +8.1 % | +7.3 % | +4.5 % | +0.4 % |
| Error vs nominal, wavesim 2 mec. (**nuevo**, `atenuacion.md`; antes +4.9 … −2.1 %) | +1.1 % | +0.4 % | −0.7 % | −1.4 % | −1.5 % |
| Error vs nominal, wavesim 3 mec. (**nuevo**) | −1.4 % | −1.7 % | −1.4 % | −0.7 % | +0.2 % |
| c fase k-Wave (m/s) | 1633 | 1636 | 1638 | 1637 | 1638 |
| c fase wavesim 1 mec. (m/s) | 1578 | 1580 | 1583 | 1585 | 1587 |

Las filas de k-Wave y de 1 mecanismo son las medidas originales; las de 2 y 3
mecanismos son de wavesim solo (el nuevo ajuste con tiempos de relajación
optimizados y un peso por mecanismo), sin k-Wave en esa sesión. Los valores
absolutos de α y las velocidades de fase con 2 mecanismos de la tabla original
se han retirado porque el ajuste cambió.

Tiempos (medida original): k-Wave 91.5 s, wavesim 6.3 s (1 mec.) y 7.9 s (2 mec.).
Con el ajuste nuevo, 2 mecanismos cuestan ×1.49 y 3 mecanismos ×1.72 respecto de 1
en una tibia de 24×22 mm a 3 MHz (`atenuacion.md`).

Lectura:

- **Con un mecanismo wavesim reproduce la ley nominal con errores de +0.4 a
  +8 %.** Con el ajuste nuevo, dos mecanismos dan errores de −1.5 a +1.1 % en
  este caso (antes −2.1 a +4.9 %) y tres de −1.7 a +0.2 %. Esta es la
  evidencia que llevó a poner `n_mechanisms=2` por defecto (el error de la
  banda del pulso es ≤ 2 % para cualquier exponente de 1 a 2). Es una
  comparación de wavesim con la ley nominal, no con k-Wave.
- **k-Wave tampoco clava el valor nominal en este montaje**: α medida −7.7 %
  y velocidad de fase 3.6 % por encima del valor `sound_speed` (1633-1638
  frente a 1580 m/s). En k-Wave el `sound_speed` con absorción con
  dispersión es una velocidad de referencia, no la velocidad de fase a f0,
  así que la discrepancia de velocidad es probablemente una convención y
  no un error (comprobable en su documentación, que no era accesible).
  La diferencia de atenuación de −7.7 % se parece a (1580/1637)² = 0.93;
  es una coincidencia sugerente, no verificada. Mientras no se aclare, la
  comparación honesta es **cada código frente al valor nominal**, no
  wavesim frente a k-Wave: la diferencia relativa wavesim/k-Wave (+6 a
  +17 %) mezcla ambos sesgos.
- wavesim fija la velocidad de fase en f0 al valor de la tabla (1583 m/s a
  1 MHz, +0.2 %) y la dispersión causal asociada al GSLS es pequeña en la
  banda (1578-1587 m/s).
- Limitaciones del experimento: una sola resolución (0.10 mm), un solo
  exponente, distancia corta (10 mm de camino) con un estimador de
  cociente espectral que acumula errores de fase y de expansión; no hay
  barra de error.

### 10.4 Qué no cubre

No se comparó con k-Wave: ondas S, conversión de modo, hueso, interfaces
rugosas, speckle, B-mode ni ninguna otra modalidad (las ondas S y guiadas se han
verificado después frente a soluciones analíticas, no frente a k-Wave).
k-wave-python 0.6.2 no contiene el solver elástico, así que la comparación en
el hueso exige MATLAB (k-Wave `pstdElastic2D`) o SimSonic. Tampoco se ha
ejecutado ningún otro simulador (SimSonic, Field II, gprMax, Meep, MCX, GATE,
KomaMRI, EIDORS...); la única otra referencia externa ejecutada es SpekPy 2.5.4
para los espectros de rayos X.

Para repetirlo: `python benchmarks/kwave_comparison.py` (≈ 3-4 min en 4
núcleos con el backend Python).

## 11. Plan para validar wavesim contra los simuladores de referencia

Criterios de aceptación propuestos. Son umbrales de partida, a ajustar
cuando se vea el ruido de cada caso.

Estado: **hecho** = ejecutado con su criterio cumplido; **parcial** = ejecutado
con alguna cifra fuera de tolerancia o sin la referencia externa prevista;
**pendiente** = no ejecutado. «Referencia» es la propuesta original; cuando se
hizo sin el simulador externo se indica.

### 11.1 Ultrasonido

| # | Caso | Referencia | Umbral propuesto | Estado y resultado |
|---|---|---|---|---|
| U1 | Dos capas fluidas, onda plana | Analítico y k-Wave | ≤ 3 % a 15 ppw en R y forma, ≤ 0.5 % en T | **Hecho** (§10.2, versión anterior de wavesim): R 1.3 %, T 0.06 %, forma 1.9-2.9 %. No repetido con k-Wave |
| U2 | Fuente puntual con atenuación | Ley nominal; k-Wave | ≤ 5 % con 2 mecanismos | **Hecho** (§10.3 y [atenuacion](validacion/atenuacion.md)): 2 mecanismos −1.5 a +1.1 %; onda plana 1D ≤ 2.2 % en la banda estrecha |
| U3 | Convergencia: dx y CFL; orden observado | Analítico | Pendiente ≥ 2 | **Hecho** ([elastico](validacion/elastico.md)): c 1.84 → 1.96, R agua-cortical 1.94-2.83; velocidad igual a la del esquema a 1e-6 |
| U4 | Semiespacio agua-cortical: R(θ), T(θ) de P y S, críticos | Zoeppritz/Snell (sin SimSonic ni k-Wave elástico) | ≤ 3 % en amplitud, ≤ 1° | **Parcial**: \|R\| ≤ 0.7 % hasta 22°; 28-53° ≤ 3.9 % (≤ 1.2 % en los ángulos de la tabla); 59-70° ≤ 5 %; ±3° de los críticos 9-17 %; crítico P a ±0.15°. Tras corregir un error real del solver en la interfaz oblicua |
| U5 | Placa libre y sumergida (3 mm): Lamb S0/A0, 100-500 kHz | Rayleigh-Lamb (sin SimSonic ni COMSOL) | ≤ 2 % | **Hecho**: libre ≤ 1.3 % / 1.7 % con 30 celdas (con 20: S0 8.7 %); en agua ≤ 0.42 % / 0.39 % (S0 hasta 350 kHz) |
| U6 | Transmisión axial con tejido blando: primer arribo, onda lateral | Bossy et al. 2004; Fotiadis et al. | Reproduce los intervalos publicados | **Parcial** ([elastico3d](validacion/elastico3d.md)): 3D a 0.5 MHz, FAS de 3518 m/s (tubo) frente a 3508 (placa); decae como d^−2.97 (3D) frente a d^−1.87 (2D). Sin comparar con Bossy (no verificado si afirman lo mismo); sin 1 MHz ni tubo anisótropo |
| U7 | Fractura transversal con hueco w (0.1-2 mm), 200 kHz | Dodd, Humphrey, Li | Misma tendencia; < 3 dB | **Parcial**: una sola fractura de 1 mm a 0.5 MHz, −3 a −7 dB y +0.3 µs de retardo, tendencia coherente con la publicada (cifras en dB no verificadas); sin barrido de w ni razón S0/A0 |
| U8 | Benchmark transcraneal BM1-BM2 | Aubry et al. 2022 | < 10 % y < 1 mm | **Pendiente** |
| U9 | B-mode de dispersor puntual y de speckle | Field II y k-Wave | FWHM ≤ 10 %, SNR ≈ 1.91 | **Parcial** ([imagen_ultrasonido](validacion/imagen_ultrasonido.md)): FWHM lateral ≤ 12 % de la teoría; axial **+9 a +17 %** (no cumple; interfaz plana +1.9 %); SNR de speckle 1.85 ± 0.10 (Hann). Sin Field II: la referencia es la teoría de apertura |

### 11.2 Resto de modalidades

| Modalidad | Caso | Referencia | Estado y resultado |
|---|---|---|---|
| Maxwell | Losa multicapa (piel/grasa/músculo/hueso) a 128 MHz y 2.45 GHz | Matriz de transferencia; Meep y openEMS | **Hecho** frente a la matriz de transferencia ([maxwell](validacion/maxwell.md)): ≤ 0.11 dB y 0.9° (banda estrecha), ≤ 0.31 dB en 0.1-10 GHz (banda ancha, dx 0.125 mm). **Pendiente** Meep/openEMS |
| Maxwell | Cilindro dieléctrico en onda plana | Mie 2D | **Hecho**: 1.7-3.3 % dentro del cilindro con 15-20 ppw y sub-celda (sin sub-celda 6-11 %); ~0.8 % fuera en el hueso sin explicar |
| Óptica | Losa homogénea y de dos capas | MCX y MCML | **Parcial** ([optica](validacion/optica.md)): frente a van de Hulst (tabla 35), Fresnel exacto y balístico, no frente a MCX ni MCML ejecutados; la losa dispersiva con n = 1.5 no tiene referencia verificada |
| Rayos X | Cilindros agua/hueso/aire a 80 kVp | NIST XCOM, MC-GPU o GATE; ASTRA/TIGRE; MTF | **Parcial** ([rayos_x](validacion/rayos_x.md)): HU y NIST < 1.5 %, FBP exacta, abanico = paralelo, MTF y NPS; espectro frente a SpekPy (±5 % a 80-120 kVp, −6.6/−5.2 % a 50/70 kVp). **Pendiente**: scatter frente a MC-GPU/GATE, FBP frente a ASTRA |
| RM | T1/T2/PD con SE, IR, GRE | KomaMRI.jl y JEMRIS | **Parcial** ([rm](validacion/rm.md)): frente a las ecuaciones de señal (1e-11), al desplazamiento químico esperado y al cilindro de susceptibilidad exacto; **pendiente** KomaMRI/JEMRIS (Julia y C++ no disponibles) |
| Fantomas | Espesores, fracciones y radios óseos | VP/XCAT o literatura | **Pendiente** |
| EIT | Disco homogéneo con inclusión | Analítico y EIDORS/pyEIT | **Parcial** ([eit](validacion/eit.md)): CEM frente a la serie de Fourier ≤ 1.5 % y electrodos puntuales ≤ 1.3 %; GN con métricas GREIT. **Pendiente** EIDORS/pyEIT |

### 11.3 Orden de trabajo (actualizado)

1. **Hecho:** el arnés y U3, U4 (con excepciones) y U5, que se comprueban
   analíticamente sin software externo; Maxwell frente a la matriz de
   transferencia y Mie; óptica frente a van de Hulst; rayos X, RM y EIT frente a
   soluciones exactas o internas.
2. **Pendiente, lo que más falta:** ejecutar de verdad otro simulador y comparar
   señal a señal: SimSonic (U6-U7, fractura en hueso largo), k-Wave elástico
   (MATLAB), Field II (U9), MCX, gprMax o Meep, KomaMRI, EIDORS. Hasta entonces toda la
   validación es interna.
3. Barrido de la anchura de la fractura (U7), 1 MHz y tubo anisótropo en U6,
   benchmark transcraneal U8 y fantomas frente a VP/XCAT.
4. Registrar cada caso como test en `tests/` cuando tenga umbral estable (ya
   hecho para los casos rápidos de cada documento de `docs/validacion/`) y
   guardar las trazas de referencia, no los binarios.

## 12. Referencias

Enlaces consultados en esta sesión (resultados del buscador o páginas
accesibles). Ver §0 para las webs oficiales que no se pudieron abrir.

**Ultrasonido**
- k-Wave, ejemplo y documentación de ondas elásticas: <https://www.k-wave.org/documentation/example_ewp_layered_medium.php>
- Treeby, Jaroš, Rohrbach & Cox, "Modelling elastic wave propagation using the k-Wave MATLAB toolbox", IEEE IUS 2014: <https://discovery.ucl.ac.uk/10113895/1/2014_Treeby_IEEE_kWaveElastic.pdf>
- Treeby & Cox, "Modeling power law absorption and dispersion for acoustic propagation using the fractional Laplacian", JASA 127(5), 2741 (2010): <https://typeset.io/papers/modeling-power-law-absorption-and-dispersion-for-acoustic-ym58pcq9p6>
- k-wave-python: <https://github.com/waltsims/k-wave-python> y notas de versión <https://github.com/waltsims/k-wave-python/releases/tag/v0.6.1>
- j-Wave (Stanziola, Arridge, Cox, Treeby; SoftwareX 22, 101338, 2023): <https://arxiv.org/pdf/2207.01499>, <https://pypi.org/project/jwave>
- Stride (Cueto et al., 2022): <https://arxiv.org/abs/2110.03345>
- Field II (guía de usuario 3.30): <https://field-ii.dk/documents/users_guide.pdf> (no accesible; citada por el buscador), <https://home.healthtech.dtu.dk/jaj/aui_2023/lectures/lecture_1_field_II/hand_outs.pdf>
- FOCUS: <https://www.egr.msu.edu/~fultras-web/background/fnm.php>, Honer & McGough 2025: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12299044/>
- Sim4Life (acústica): <https://sim4life.swiss/dive-deep-powerful-solvers>
- Aubry et al., benchmark transcraneal, JASA 152 (2022) 1003: <https://arxiv.org/pdf/2202.04552>, <https://discovery.ucl.ac.uk/10154839/2/Gelat_10.0013426.pdf>, datos: <https://zenodo.org/record/6020543>
- Fullwave 2.5 (Sode & Pinton): <https://pypi.org/project/fullwave25/1.0.8/>
- COMSOL, Elastic Waves Time Explicit: <https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_modeling.05.017.html>
- Bossy, Talmant & Laugier, JASA 115 (2004) 2314: <https://hal.archives-ouvertes.fr/hal-00109676>
- Bossy et al., PMB 50 (2005) 5545 y usos de SimSonic: <https://arxiv.org/pdf/1805.03083>, <https://pmc.ncbi.nlm.nih.gov/articles/PMC6909975>
- Fractura, ondas guiadas: <https://pmc.ncbi.nlm.nih.gov/articles/PMC5274688>, <https://researchportal.bath.ac.uk/en/publications/numerical-modelling-and-in-vitro-studies-of-ultrasound-signal-los/>, <https://medlab.cc.uoi.gr/?p=6603>
- Moilanen 2008 y Tran et al. 2022 (revisiones): <https://vufind-test.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c434t-7fad14e9e681b5b352e9fe1e0a06ee7e68c521202cbb0353e26dc37433d1793f3>
- Ecografía simulada para entrenar redes: <https://rucore.libraries.rutgers.edu/rutgers-lib/64047/>, <https://ewh.ieee.org/conf/ius/2019/media/files/1327.pdf>

**Electromagnetismo**
- openEMS: <https://docs.openems.de/intro.html>, <https://github.com/thliebig/openEMS-Project/>
- Meep (Oskooi et al., 2010): <https://math.mit.edu/~stevenj/papers/OskooiRo10.pdf>
- gprMax: <https://github.com/gprMax/gprMax>, GPU: <https://repository.uwl.ac.uk/id/eprint/5755/>
- SEMCAD X / Sim4Life: <https://speag.swiss/products/semcad>, <https://itis.swiss/tools-and-systems/sim4life>
- Gabriel 1996 (IT'IS): <https://itis.swiss/virtual-population/tissue-properties/database/dielectric-properties>
- Microondas y fracturas: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10319743/>, <https://repositorio.iscte-iul.pt/bitstream/10071/26303/1/article_89899.pdf>

**Óptica**
- MCX: <https://pypi.org/project/pmcx/>, <https://github.com/fangq/mcxcl>
- Toast++: <https://www.spiedigitallibrary.org/journals/Journal-of-Biomedical-Optics/volume-19/issue-4/040801/The-Toast-software-suite-for-forward-and-inverse-modeling-in/10.1117/1.JBO.19.4.040801.pdf>
- NIRFAST: <https://research.birmingham.ac.uk/en/publications/near-infrared-optical-tomography-using-nirfast-algorithm-for-nume>

**Rayos X**
- GATE: <https://arxiv.org/pdf/physics/0408109>
- TOPAS: <https://pubmed.ncbi.nlm.nih.gov/23127075/>
- MC-GPU: <https://github.com/mghro/MCGPU>, <https://www.fda.gov/about-fda/cdrh-offices/monte-carlo-simulation-x-ray-transport-gpu-cuda-mc-cpu>
- ASTRA: <https://astra-toolbox.readthedocs.io/>, <https://pypi.org/project/astra-toolbox/>
- TIGRE: <https://github.com/Daveelvt/TIGRE>
- XCIST: <https://github.com/xcist/main>, <https://pypi.org/project/gecatsim>, <https://www.pubmed.gov/36096127/>

**RM**
- KomaMRI.jl: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10952765/>, <https://arxiv.org/pdf/2301.02702>
- JEMRIS: <https://helmholtz.software/software/jemris>
- MRiLab: <https://mrilab.sourceforge.net/news.html>
- MRXCAT: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4422262>

**EIT y fantomas**
- EIDORS: <https://eprints.maths.manchester.ac.uk/1045/>
- pyEIT: <https://doaj.org/article/6e5388e9c4674f6ba3305a7d33099e58>
- EIT y fracturas: <https://doi.org/10.1007/978-981-287-928-8_4>, <https://doaj.org/article/fa95ef52eeb6479b98da4dd2bd1aa9fb>
- XCAT: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4603876>
- Virtual Population: <https://itis.swiss/virtual-population/virtual-population/faq/>
- MIDA: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4406723/>, <https://itis.swiss/virtual-population/regional-human-models/mida-model>
