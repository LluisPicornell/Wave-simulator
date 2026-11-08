# wavesim frente a los simuladores de referencia

Comparación de `wavesim` con los simuladores usados hoy en investigación y en
la industria, modalidad por modalidad, más un plan de validación y una primera
comparación numérica con k-Wave.

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
- **Estado real de wavesim en esta copia del repositorio.** El encargo
  mencionaba una EIT cuasi-estática y aceleración Numba. **No existen en esta
  copia**: `wavesim/em/` contiene `fdtd_tm.py` (Maxwell TM), `optics.py`
  (difusión y balístico), `projection.py` (Radon/FBP), `spectrum.py` y
  `xray.py`; no hay ninguna referencia a EIT ni a Numba en `wavesim/`, y el
  README lo lista como "próximos pasos". Aquí se compara lo que existe. Para
  EIT sólo se describen los simuladores de referencia y cómo se validaría un
  futuro módulo.
- **Fecha.** Noviembre de 2026. Versiones y licencias pueden haber cambiado.

## 1. Qué es wavesim (resumen honesto)

Paquete Python + NumPy (sin GPU, sin Numba), 2D, con un solo conjunto de
fantomas de extremidad con fractura compartido por todas las modalidades:

| Modalidad | Qué resuelve realmente | Límites principales |
|---|---|---|
| Ultrasonido | FDTD elástico velocidad-tensión, 4.º orden en espacio, leapfrog de 2.º orden, CPML, atenuación GSLS con "método tau" (1 mecanismo por defecto; el README da ±12 % de ajuste a la ley ∝ f en la banda del pulso), speckle por fluctuación de densidad, rugosidad de interfaces | 2D, CPU, float32, hueso isótropo, acústica lineal, tejidos blandos como fluido (`mu = 0`) |
| Maxwell | FDTD de Yee 2D modo TM, CPML, onda continua y fasor por DFT, propiedades Cole-Cole (Gabriel 1996) a la frecuencia central, SAR | banda estrecha (sin dispersión dentro del pulso), 2D TM, sin antenas ni fuentes realistas |
| Óptica | Difusión estacionaria (diferencias finitas + gradiente conjugado) y atenuación balística | sin Monte Carlo, sin resolución temporal, contorno extrapolado simple |
| Rayos X/TAC | Beer-Lambert con base agua + hueso (NIST/ICRU-44), espectro de tubo de Kramers, endurecimiento, ruido de Poisson, FBP | sin scatter, sin líneas características del W, haz paralelo, 2D |
| RM | Ecuaciones de señal analíticas (T1, T2, DP, STIR, GRE, UTE) validadas contra un Bloch de isocromatos; k-space recortado y ruido de Rice | no es un simulador de Bloch espacial: sin B0/B1 no uniformes, sin gradientes reales, sin artefactos de desfase |
| EIT | no existe | |

Qué lo distingue: **una sola anatomía paramétrica 2D (tibia, antebrazo,
muslo; cortes longitudinales y transversales), con máscara de verdad de
terreno de la fractura, hematoma y edema, reproducible por semilla a
cualquier resolución, vista por cinco físicas distintas, y un generador de
datasets etiquetados.** Ningún simulador de las tablas siguientes ofrece eso
en un solo paquete (ver §9).

## 2. Ultrasonido

| Herramienta | Método | 2D/3D | GPU | Atenuación / dispersión | Validación publicada | Licencia | Lenguaje | Uso principal |
|---|---|---|---|---|---|---|---|---|
| **wavesim** | FDTD elástico P+S, 4.º orden, CPML | 2D | No | GSLS (ley ∝ f, Q casi constante); dispersión causal consistente con el GSLS | Tests propios contra soluciones analíticas (velocidad P/S < 1 %, reflexión 0.539 vs 0.538, atenuación < 5 %); §10 de este documento contra k-Wave | Ver `pyproject.toml` (no declarada en esta copia) | Python/NumPy | Visibilidad de fracturas, datasets |
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

- **3D y anisotropía.** SimSonic (y Sim4Life, COMSOL) trabajan en 3D con
  cortical anisótropa y porosa; en wavesim el corte 2D no reproduce la
  difracción 3D ni las ondas guiadas de un tubo (ver Bossy 2004).
- **Rapidez y escala.** Sim4Life, k-Wave (CUDA), j-Wave, Stride y Fullwave
  corren en GPU; wavesim a 3 MHz y 24×22 mm tarda ~1 min en CPU, y a 7-10 MHz
  minutos o más.
- **Menos puntos por longitud de onda.** El método pseudoespectral de k-Wave
  permite mallas más gruesas que un FDTD de 4.º orden para la misma
  dispersión numérica [NV: valor concreto]. En §10, con 8 puntos por longitud
  de onda ambos códigos dan errores comparables.
- **Atenuación más flexible.** k-Wave usa una ley de potencia con exponente
  arbitrario; Fullwave admite alpha0 y gamma variables en el espacio; wavesim
  usa 1-3 mecanismos de relajación con ajuste global (error de ±12 % con 1
  mecanismo; ver §10: con 2 mecanismos baja a ±5 %).
- **No linealidad, trayectorias focalizadas, terapia y transductores
  reales.** k-Wave, Field II, FOCUS: wavesim sólo tiene sonda lineal ideal y
  acústica lineal.
- **Hueso trabecular desde µCT** (SimSonic): wavesim lo trata como medio
  efectivo.
- **Validación publicada y revisada por pares.** k-Wave, Field II, SimSonic y
  Sim4Life tienen años de comparaciones externas; la de wavesim es interna
  (y, desde este documento, un primer cruce con k-Wave).

### Qué aporta wavesim

- La misma anatomía con **máscara de fractura/hematoma/edema** alimenta
  cinco físicas (ver §9) y un generador de datasets.
- Solver elástico con conversión de modo en el hueso **y** fluidos en un
  paquete de decenas de ficheros Python legibles, sin MATLAB ni binarios.
- Rugosidad de interfaces y fracturas paramétricas (transversal, oblicua,
  fisura, conminuta, incompleta, desplazada) reproducibles por semilla.
- Atenuación viscoelástica por tejido y speckle por microestructura
  integrados en el mismo flujo.

### Cuándo recomendar otra herramienta

| Si necesitas... | Usa |
|---|---|
| Ultrasonido 3D con hueso cortical anisótropo, transmisión axial, ondas guiadas en un tubo | SimSonic (si lo consigues bajo su licencia [?]), Sim4Life o COMSOL |
| Física de ondas S en un medio mixto fluido-sólido a gran escala con GPU | Sim4Life o COMSOL (dG-FEM), o k-Wave elástico en MATLAB (CPU) |
| Propagación fluida 2D/3D con atenuación de ley de potencia rápida y de referencia | k-Wave / k-wave-python |
| Optimizar o invertir (FWI, tomografía ultrasónica, redes diferenciables) | j-Wave o Stride |
| Diseño de sondas, PSF, B-mode de dispersores puntuales con rapidez extrema | Field II o FOCUS |
| Cumplimiento regulatorio o dosimetría con fantomas humanos 3D | Sim4Life + Virtual Population |

## 3. Electromagnetismo (RF, microondas, ondas milimétricas)

| Herramienta | Método | 2D/3D | GPU | Dispersión | Validación publicada | Licencia | Lenguaje | Uso principal |
|---|---|---|---|---|---|---|---|---|
| **wavesim `em.fdtd_tm`** | FDTD de Yee 2.º orden, TM, CPML, banda estrecha | 2D | No | No (propiedades a f0, Cole-Cole de Gabriel) | Tests: constante de fase y atenuación en músculo y grasa (< 2 % y < 5 %) | (ver arriba) | Python/NumPy | Penetración, campo y SAR en el corte |
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

**Qué hacen mejor ellos.** 3D, antenas reales y adaptación de impedancia,
materiales dispersivos de banda ancha, mallas no uniformes, SAR en cuerpo
entero con fantomas 3D (Sim4Life), GPU (gprMax, Sim4Life). En wavesim una
onda plana sin antena en 2D TM da la penetración pero no la señal medida en
una antena.

**Qué aporta wavesim.** La misma geometría de fractura con edema y hematoma
alimenta el Maxwell, con las propiedades de Gabriel del mismo diccionario de
tejidos que usa el ultrasonido.

**Cuándo recomendar otra:** antenas o arrays de microondas para imagen
(openEMS, Meep, gprMax), SAR normativo (Sim4Life), banda ancha con tejidos
dispersivos (gprMax con multipolo, Meep).

## 4. Óptica (UV, visible, IR, THz)

| Herramienta | Método | 2D/3D | GPU | Atenuación/dispersión | Validación publicada | Licencia | Lenguaje | Uso |
|---|---|---|---|---|---|---|---|---|
| **wavesim `em.optics`** | Difusión estacionaria (FD + CG) y balístico | 2D | No | μa, μs' por tejido y longitud de onda (orden de magnitud en IR medio y UV) | Test: decaimiento coincide con μ_eff analítico | (ver arriba) | Python/NumPy | Penetración de la luz en la extremidad |
| **MCX** | Monte Carlo de fotones en voxels 3D (Fang & Boas 2009) | 3D | Sí (CUDA); MCXCL para OpenCL | μa, μs, g, n por voxel | Fang & Boas, Opt. Express 17 (2009) 20178 | GPL v3 | C/CUDA; Python (`pmcx`), MATLAB | Estándar de facto para Monte Carlo GPU |
| **MCML** | Monte Carlo en capas planas infinitas (Wang, Jacques & Zheng 1995) | Capas 1D (simetría cilíndrica) | No | μa, μs, g, n por capa | Wang et al., Comput. Methods Programs Biomed. 47 (1995) 131 [NV: cita] | [?] | C | Referencia histórica |
| **NIRFAST** | FEM, difusión (Dehghani et al., 2009) | 2D/3D | [NV] | Múltiples longitudes de onda | Dehghani et al., Commun. Numer. Methods Eng. 25 (2009) 711 | [?] | MATLAB | Tomografía óptica difusa |
| **Toast++** | FEM, difusión, solvers inversos (Schweiger & Arridge) | 2D/3D | No | Dominio del tiempo, frecuencia y estacionario [NV] | J. Biomed. Opt. 19 (2014) 040801 | GPL (según su página en SourceForge) | C++ con MATLAB/Python | Reconstrucción de μa y μs' |

**Qué hacen mejor ellos.** Monte Carlo reproduce el transporte exacto
(incluida la zona de transición cerca de fuentes y fronteras donde la
difusión falla), 3D, resolución temporal, fluorescencia. NIRFAST/Toast++
incluyen reconstrucción inversa. wavesim sólo da fluencia estacionaria en
2D, y en una extremidad de ~5 cm la señal es completamente difusa.

**Qué aporta wavesim.** El mismo corte con fractura, en el mismo "paciente"
que la ecografía, y la conclusión cuantitativa (pérdida en dB) de que la luz
no forma imagen del hueso a través de la extremidad.

**Cuándo recomendar otra:** cualquier estudio cuantitativo de fluencia en
3D o con fuentes cercanas al tejido (MCX), tomografía óptica difusa con
inversión (NIRFAST o Toast++).

## 5. Rayos X y TAC

| Herramienta | Método | 2D/3D | GPU | Física incluida | Validación publicada | Licencia | Lenguaje | Uso |
|---|---|---|---|---|---|---|---|---|
| **wavesim `em.xray`** | Beer-Lambert polienergético en base agua + hueso, FBP | 2D, haz paralelo | No | Endurecimiento, ruido de Poisson; sin scatter ni líneas K | Test: Beer-Lambert exacto, FBP exacta en disco, HU de músculo, grasa y hueso a 60 keV, NIST < 1.5 % | (ver arriba) | Python/NumPy | Radiografía y TAC del corte |
| **GATE (Geant4)** | Monte Carlo de partículas | 3D | Variantes [NV] | Física completa de fotones y electrones | Jan et al., Phys. Med. Biol. 49 (2004) 4543 (PET/SPECT) y años de validaciones; nacido para TAC y PET/SPECT | LGPL en la versión pública inicial; términos actuales [?] | C++ con macros | PET, SPECT, TAC, radioterapia |
| **TOPAS** | Envoltura de Geant4 (Perl et al., Med. Phys. 39, 2012) | 3D | No | Física completa, orientado a radioterapia/protones | Perl et al. 2012 | Gratis para investigación "en intención" según el artículo; términos exactos [?] | C++ con ficheros de parámetros | Radioterapia con protones |
| **MC-GPU** | Monte Carlo de rayos X en GPU, voxels; secciones eficaces PENELOPE 2006 (sin transporte de electrones) | 3D | Sí (CUDA) | Dispersión Compton, Rayleigh, fotoeléctrico | Badal & Badano, Med. Phys. 36 (2009) 4878 [NV: cita]; 20-40 veces más rápido que 1 núcleo | [?] (distribuido por la FDA) | C/CUDA | Radiografía y TAC con dispersión realista |
| **ASTRA** | Proyectores y reconstructores en GPU | 2D/3D | Sí | Sólo geometría de proyección | van Aarle et al., Ultramicroscopy 157 (2015) y Opt. Express 24 (2016) | GPL v3 | C++/CUDA; MATLAB y Python | Reconstrucción de TAC/tomografía |
| **TIGRE** | Reconstrucción CBCT en GPU (Bath/CERN) | 3D | Sí (CUDA) | Sólo geometría | Biguri et al., Biomed. Phys. Eng. Express 2 (2016) 055010 | BSD | MATLAB/Python con CUDA | CBCT |
| **XCIST / CatSim** | Simulador analítico de TAC con fantomas y reconstrucción | 3D | [NV] | Espectro, scatter, ruido, detector (según PMB 2022); usa secciones eficaces de Geant4 | "XCIST: an open access x-ray/CT simulation toolkit", Phys. Med. Biol. 2022 | [?] | Python + C/C++ (`gecatsim`) | Simulación de TAC realista |

**Qué hacen mejor ellos.** Scatter y física completa (GATE, MC-GPU), 3D y
geometrías reales abanico/cono con detectores (XCIST), reconstrucción
iterativa y cono (ASTRA, TIGRE). wavesim no puede dar el contraste real a
bajo kV, donde el scatter importa.

**Qué aporta wavesim.** TAC y radiografía del mismo corte que la ecografía y
la RM, y rapidez (segundos).

**Cuándo recomendar otra:** dosimetría o imagen con scatter (MC-GPU, GATE),
CBCT/geometrías reales (XCIST, TIGRE, ASTRA).

## 6. Resonancia magnética

| Herramienta | Método | 2D/3D | GPU | Física | Validación publicada | Licencia | Lenguaje | Uso |
|---|---|---|---|---|---|---|---|---|
| **wavesim `mri`** | Ecuaciones de señal analíticas por tejido + k-space recortado + ruido de Rice; Bloch de isocromatos sólo para validar | 2D | No | T1, T2, PD, secuencias T1/T2/DP-FS/STIR/GRE/UTE | Ecuaciones coinciden con Bloch (1e-6) | (ver arriba) | Python/NumPy | Contraste de la fractura (edema, hematoma) |
| **JEMRIS** | Bloch con sólo CPU multihilo; secuencias autocontenidas | 3D | No | Bloch completo con campos B0/B1/gradiente, pulsos reales | Stöcker et al., Magn. Reson. Med. 64 (2010) 186 | GPL-2.0 o posterior (catálogo Helmholtz) | C++ (MATLAB/Python) | Secuencias, artefactos |
| **MRiLab** | Bloch en GPU, GUI | 3D | Sí | Bloch completo | Liu et al., Magn. Reson. Med. 2017 | [?] | MATLAB/C++/CUDA | Docencia, diseño de secuencias |
| **KomaMRI.jl** | Bloch en CPU y GPU, secuencias Pulseq | 3D | Sí | Bloch completo, movimiento | Castillo-Passi et al., Magn. Reson. Med. (2023): error medio absoluto del 0.1 % frente a JEMRIS | MIT | Julia | Simulación general |
| **MRXCAT** | Generador de fantomas basado en XCAT con modelo de señal, bobinas y ruido | 2D+t / 3D | No | Señal cardíaca con movimiento | Wissmann et al., J. Cardiovasc. Magn. Reson. 16 (2014) 63 | [?] ("disponible online" según el artículo) | MATLAB [NV] | RM cardíaca |

**Qué hacen mejor ellos.** Bloch espacial real: heterogeneidad de B0/B1,
gradientes, pulsos selectivos, flujo y movimiento, trayectorias no
cartesianas, artefactos de susceptibilidad en la interfaz hueso-aire; GPU.
wavesim es "señal en estado estacionario ideal + k-space + ruido".

**Qué aporta wavesim.** Valores de T1/T2/PD de fractura (edema, hematoma)
sobre la misma anatomía que el resto, a segundos por imagen.

**Cuándo recomendar otra:** artefactos reales (KomaMRI, JEMRIS), diseño de
secuencias o fantomas cardíacos (MRXCAT).

## 7. Impedancia eléctrica (EIT) – no implementado en wavesim

| Herramienta | Método | 2D/3D | Licencia | Lenguaje | Uso |
|---|---|---|---|---|---|
| **EIDORS** | FEM + inversión; Adler & Lionheart, Physiol. Meas. 27 (2006) S25 | 2D/3D | GPL v2 o v3 | MATLAB/Octave | Estándar académico en EIT (pulmón, cerebro) |
| **pyEIT** | FEM + reconstrucción (back-projection, GREIT, etc.), SoftwareX 2018 | 2D/3D | Apache 2.0 según el artículo; un listado de Anaconda dice BSD-3: comprobar el `LICENSE` del repositorio | Python | EIT en Python |

Para fracturas: la bibliografía hallada es experimental (impedancia en
conejo, placa inteligente, estudios antiguos de Aberdeen con EIT de miembro);
no se encontró ningún simulador específico de EIT para detección de
fracturas.

## 8. Fantomas anatómicos

| Fantoma | Tipo | Dimensión | Contenido | Acceso/licencia |
|---|---|---|---|---|
| **wavesim `phantoms`** | Paramétrico, geométrico, por semilla | 2D | Tibia, antebrazo, muslo; fracturas (6 tipos) con máscara; hematoma, edema | Parte del paquete |
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
| α wavesim 2 mecanismos | 1.417 | 1.718 | 2.015 | 2.314 | 2.613 |
| Error vs nominal, k-Wave | −7.3 % | −7.5 % | −7.7 % | −7.7 % | −7.9 % |
| Error vs nominal, wavesim 1 mec. | +5.2 % | +8.1 % | +7.3 % | +4.5 % | +0.4 % |
| Error vs nominal, wavesim 2 mec. | +4.9 % | +2.7 % | +0.8 % | −0.8 % | −2.1 % |
| c fase k-Wave (m/s) | 1633 | 1636 | 1638 | 1637 | 1638 |
| c fase wavesim 1 mec. (m/s) | 1578 | 1580 | 1583 | 1585 | 1587 |
| c fase wavesim 2 mec. (m/s) | 1579 | 1580 | 1583 | 1585 | 1587 |

Tiempos: k-Wave 91.5 s, wavesim 6.3 s (1 mec.) y 7.9 s (2 mec.).

Lectura:

- **Con un mecanismo (el valor por defecto) wavesim reproduce la ley
  nominal con errores de +0.4 a +8 %**, coherente con el "±12 %" del README.
  Con dos mecanismos el error baja a ±5 % (±3 % salvo en el borde de banda
  de 0.7 MHz). Esta es la mejor evidencia numérica de que `n_mechanisms=2`
  merece ser el valor para estudios cuantitativos.
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

No se comparó: ondas S, conversión de modo, hueso, interfaces rugosas,
speckle, B-mode ni ninguna otra modalidad. k-wave-python 0.6.2 no contiene el
solver elástico, así que la comparación en el hueso exige MATLAB (k-Wave
`pstdElastic2D`) o SimSonic.

Para repetirlo: `python benchmarks/kwave_comparison.py` (≈ 3-4 min en 4
núcleos con el backend Python).

## 11. Plan para validar wavesim contra los simuladores de referencia

Criterios de aceptación propuestos. Son umbrales de partida, a ajustar
cuando se vea el ruido de cada caso.

### 11.1 Ultrasonido

| # | Caso | Referencia | Métricas | Umbral propuesto |
|---|---|---|---|---|
| U1 | Dos capas fluidas, onda plana (hecho, §10.2) | Analítico y k-Wave | R, T; Δt de llegada; error L2 tras alinear | ≤ 3 % a 15 puntos/λ en R y forma, ≤ 0.5 % en T (cumplido: R 1.3 %, T 0.06 %, forma 1.9-2.9 %) |
| U2 | Fuente puntual con atenuación (hecho, §10.3) | Ley nominal; k-Wave | α(f) y c(f) | ≤ 5 % con 2 mecanismos |
| U3 | Convergencia: variar dx y CFL; orden observado | Analítico | Pendiente log-log del error | ≥ 2 |
| U4 | Semiespacio elástico (agua-cortical): R(θ) y T(θ) de P y S, ángulo crítico | Fórmulas de Zoeppritz/Snell, SimSonic, k-Wave `pstdElastic2D` (MATLAB) | Amplitudes, ángulos críticos, conversión P↔S | ≤ 3 % en amplitud, ≤ 1° |
| U5 | Placa libre y sumergida en agua (cortical de 2-4 mm): curvas de dispersión S0/A0, 100-500 kHz | Ecuaciones de Rayleigh-Lamb (raíces), SimSonic o COMSOL | Velocidad de grupo/fase de S0 y A0, f-k | ≤ 2 % |
| U6 | Transmisión axial en cortical con capa de tejido blando (primer arribo, velocidad de la onda lateral) | Bossy et al. 2004; Fotiadis et al. | SOS; comparar la tendencia con la profundidad de cortical (1-1.5 mm) | Reproduce los intervalos publicados |
| U7 | Fractura transversal con hueco de ancho w (0.1-2 mm), 200 kHz | Dodd et al. y Humphrey et al. (pérdida de transmisión, FTL) y Li et al. 2017 (razón S0/A0) | FTL(dB) frente a w; razón S0/A0 | Misma tendencia; diferencia < 3 dB |
| U8 | Benchmark transcraneal BM1-BM2 (hueso planar en agua), con `cs = 0` para igualar el modelo compresional | Aubry et al. 2022 (datos en Zenodo) | Presión focal y posición | < 10 % y < 1 mm (el nivel de acuerdo entre códigos publicado) |
| U9 | B-mode de dispersor puntual y de región con speckle | Field II (FWHM axial/lateral) y k-Wave | FWHM, CNR, SNR de speckle (Rayleigh ≈ 1.91) | FWHM ≤ 10 % |

### 11.2 Resto de modalidades

| Modalidad | Caso | Referencia | Métricas |
|---|---|---|---|
| Maxwell | Losa plana multicapa (piel/grasa/músculo/hueso) a 128 MHz y 2.45 GHz | Cálculo analítico por matriz de transferencia; Meep 2D (TM) y openEMS | Módulo y fase del campo a lo largo de una línea, coeficiente de transmisión (dB), SAR pico relativo |
| Maxwell | Cilindro dieléctrico (músculo) en onda plana | Solución de Mie 2D | Error del campo interior y del scattering |
| Óptica | Losa homogénea y de dos capas (μa, μs', g, n de músculo/grasa) | MCX (3D, volumen ancho) y MCML | Perfil de fluencia en profundidad (en escala logarítmica), transmitancia difusa, ignorando la zona de 1-2 caminos libres medios de transporte alrededor de fuentes y fronteras |
| Rayos X | Cilindros de agua/hueso/aire con espectro de 80 kVp | NIST XCOM (analítico), MC-GPU o GATE para scatter | HU de cada tejido, cupping, perfil de ruido, RMSE de la FBP con ASTRA/TIGRE, MTF |
| RM | Mismas T1/T2/PD con SE, IR, GRE | KomaMRI.jl y JEMRIS | Señal por voxel (error relativo < 1 %), PSF, estadística de Rice del ruido |
| Fantomas | Espesores, fracciones de tejido y radios óseos del corte | Medidas de VP/XCAT (o literatura) | Distribución de espesores; no pretende igualar un sujeto |
| EIT (futuro) | Disco homogéneo con inclusión | Solución analítica y EIDORS/pyEIT | Tensiones de contorno, condicionamiento |

### 11.3 Orden de trabajo sugerido

1. Fijar el arnés (el script actual) y añadir U3, U4 y U5: son lo que
   diferencia a wavesim de k-wave-python (el elástico) y se pueden
   comprobar analíticamente sin software externo.
2. Obtener SimSonic (o su equivalente) para U6-U7: son los casos relevantes
   para el objetivo del proyecto (fractura en hueso largo).
3. Maxwell, óptica y rayos X con soluciones analíticas y herramientas libres
   (Meep, MCX, ASTRA): no requieren licencias comerciales.
4. Registrar cada caso como test en `tests/` cuando tenga umbral estable, y
   guardar las trazas de referencia (no los binarios) para que la
   validación no dependa de instalar el simulador externo.

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
