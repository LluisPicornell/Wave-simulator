# Validación del solver de Maxwell 3D (FDTD vectorial)

`wavesim/em/fdtd3d.py` (con los kernels Numba de `wavesim/em/_numba_fdtd3d.py`) es el equivalente 3D de los
solvers 2D TM (`fdtd_tm.py`, `fdtd_dispersive.py`): las seis componentes del campo en la malla de Yee, CPML en las
seis caras, tejidos de un `Phantom3D` en banda estrecha (Gabriel a `f0`) o en banda ancha (los mismos polos de Debye
comunes y el mismo ADE que el 2D). Todas las cifras de este documento salen de `benchmarks/em3d_validation.py`
(Numba, 1 hilo, float32 salvo que se diga, CPU compartida con otros dos procesos: los tiempos son orientativos); las
versiones reducidas están en `tests/test_fdtd3d.py`.

## 1. Qué hay

| Pieza | Dónde | Qué hace |
|---|---|---|
| `Maxwell3DSolver` | `em/fdtd3d.py` | Banda estrecha: `eps'` y `sigma` por celda a `f0`, onda continua, fasor por DFT corriente |
| `BroadbandMaxwell3DSolver` | `em/fdtd3d.py` | Debye multipolo (ADE), pulso de banda ancha, fasores a muchas frecuencias en una caja de celdas |
| Fuentes | `set_plane_wave`, `set_current_sheet`, `set_dipole` | Onda plana TF/SF de incidencia normal (E ∥ x, E ∥ y, o paralelo/perpendicular al eje del hueso de `meta["bone_axis"]`), lámina de corriente del 2D, dipolo puntual blando |
| Resultados | `EM3DResult`, `BroadbandEM3DResult` | `e` (3 componentes en los centros de celda), `magnitude`, `sar`, `absorbed_power`, `averaged_sar` (1 g, 10 g), `plane`, `incident` |
| Promedio sub-celda | `subcell_node_permittivity` | Diagonal del tensor de Meep en la celda dual de cada nodo de E (banda estrecha) |
| Serie de Mie 3D | `em/analytic.py` (`mie_sphere_efficiencies`, `mie_sphere_field`) | Esfera con pérdidas: eficiencias y campo dentro y fuera |
| Funciones esféricas de Bessel | `em/bessel.py` | `j_n(z)` complejo, `y_n(x)`, derivada de Riccati, sin SciPy |

Uso mínimo:

```python
from wavesim.core.grid3d import Grid3D
from wavesim.em import Maxwell3DSolver
from wavesim.phantoms import LIMB3D_PRESETS, Fracture3D, limb_3d, limb_grid

g0 = limb_grid(LIMB3D_PRESETS["pierna"], 80e-3, 1e-3)
pad = 16                                       # agua de más arriba y abajo para las caras de la TF/SF (CPML de 10)
grid = Grid3D(g0.nx, g0.ny, g0.nz + 2 * pad, g0.dx, g0.x0, g0.y0, g0.z0 - pad * g0.dx)
p = limb_3d("pierna", Fracture3D(gap=2e-3), length=80e-3, dx=1e-3, grid=grid, roughness=False)
s = Maxwell3DSolver(p, 2.45e9, npml=10)        # float32, Numba si está instalado
s.set_plane_wave("perpendicular", z_ref=p.meta["skin_depth"])   # E ⊥ eje del hueso, 1 V/m en la piel
r = s.run()                                     # ~3 min en un núcleo (1.1 M celdas)
r.sar, r.averaged_sar(10e-3).max(), r.plane("y", 0.0)
```

## 2. Método

### 2.1 Malla, materiales y CPML

Arrays `(nz, ny, nx)` con x = eje del hueso, y = lateral, z = profundidad (convenio común del 3D). Nodos de Yee:
`Ex (i+½, j, k)`, `Ey (i, j+½, k)`, `Ez (i, j, k+½)`, `Hx (i, j+½, k+½)`, `Hy (i+½, j, k+½)`, `Hz (i+½, j+½, k)`. El
plano `y = cte` contiene `Ey`, `Hx`, `Hz` en las posiciones del modo TM 2D (`Ey` en el centro de la celda): con campos
uniformes en y el esquema es exactamente el 2D. Leapfrog de 2.º orden, `dt = 0.95 dx / (c sqrt(d))` con `d` el número
de ejes con más de una celda (con `ny = 1` el límite es el 2D). Cada componente de E está a media celda a lo largo de
su eje y toma la media aritmética de `eps*` de las dos celdas que separa (`eps'` y `sigma`; en banda ancha
`eps_inf`, `sigma` y `d_eps_k`: la media de Debye con polos comunes sigue siendo un Debye pasivo). La CPML es la del
2D (`core.pml.cpml_coefficients`, perfil cuadrático, `R = 1e-5`, término CFS con `f_ref = f0` o `f_min`), con la
memoria de `ultrasound.elastic3d.CPML3D` (sólo en las franjas); grosor por eje (0 = sin capa).

### 2.2 Onda plana por campo total/campo dispersado

El campo incidente sale de una malla de Yee 1D a lo largo de z con el medio de fondo (con pérdidas o con los polos de
Debye), el mismo `dx` y el mismo `dt`, CPML en sus extremos y una fuente blanda por encima de la caja. Las
correcciones se generan para cada término del rotacional que cruza una cara de la caja (doce casos genéricos: cara
inferior/superior × componente × derivada); con incidencia normal sólo intervienen `Ex, Hy` (E ∥ x) o `Ey, Hx = -h`
(E ∥ y). Como a incidencia normal el esquema 3D de una onda plana *es* el esquema 1D, el campo dispersado fuera de la
caja en vacío es redondeo (3e-16 medido). Por defecto la caja ocupa todo x e y (sólo hay caras en z): la onda es
lateralmente infinita, la CPML lateral sólo ve el campo dispersado (la incidente no varía a lo largo de x ni de y) y un
cuerpo que atraviesa los bordes laterales (la pierna a lo largo de x, una losa) se comporta como infinito. El solver
exige que las celdas a ±2 de cada cara sean de un mismo medio. Los fasores se dividen por el fasor incidente en `z_ref`
(1 V/m de onda incidente allí); `incident` da el fasor incidente en cada z.

Con polos de Debye, `J_k^{n+1}` es lineal en `E^{n+1}`: las correcciones de la TF/SF y las fuentes, que se aplican
después del barrido de E, corrigen también `J_k` (`beta_k δE`), como la fuente del 2D de banda ancha.

### 2.3 Fasor de la onda continua

Mismo protocolo que `MaxwellTMSolver.run` (arranque suave de 3 periodos, asentado, DFT con ~24 muestras por periodo).
La ventana de la DFT no abarca un número entero de periodos (el periodo no es múltiplo de `dt`) y `2/N sum x e^{-jwt}`
arrastra una fracción del conjugado. Con `Q = sum e^{-2jwt}` el fasor exacto de una sinusoide estacionaria es
`P = 2 (N S - Q S*) / (N² - |Q|²)`: en vacío la amplitud de la onda plana pasa de un rizado de ±1.7 % (estimador del
2D) a 1.3e-5. `run(exact_dft=False)` reproduce el estimador del 2D (para compararse con él).

### 2.4 Fuentes y resultados

- **Lámina de corriente** (`set_current_sheet`): la del 2D, con su misma normalización (impedancia real de la celda).
- **Dipolo** (`set_dipole`): `J = I l / dx³` en un nodo de E, corriente centrada en `n + ½`; el fasor va por A·m.
- **Campo en las celdas**: cada componente es la media de sus dos nodos a lo largo de su eje.
- **SAR** = `sigma |E|² / (2 rho)` con la `sigma` y la densidad (`materials.acoustic`) de la celda, como el 2D.
- **Potencia absorbida**: `1/2 sum_c sigma_c |E_c|²` en los nodos de Yee con la `sigma` de cada nodo (la que disipa el
  esquema), llevada a las celdas (`power_density`). Con promedio sub-celda es la única coherente.
- **SAR promediado** (`averaged_sar`): cubo de 1 g o 10 g de tejido (lado con la densidad media del tejido, número
  impar de celdas) centrado en cada celda de tejido, media `sum(SAR rho) / sum(rho)` sólo sobre el tejido (sin el agua
  de acoplamiento ni el aire). Es una simplificación de IEEE/IEC 62704-1: junto a la superficie el cubo no se desplaza
  ni crece, así que allí promedia menos masa.

### 2.5 Promedio sub-celda

`subcell_node_permittivity(grid, eps_at, factor)` muestrea la geometría exacta (`eps_at(x, y, z)`) con `factor³`
puntos en la celda dual de cada nodo y toma la diagonal del tensor de Farjadpour et al. (Opt. Lett. 31:2972, 2006; el
de Meep): `eps_aa = n_a² / <1/eps> + (1 - n_a²) <eps>`, con la normal `n` estimada por el primer momento de `eps'` en
la celda. Exacto para interfaces planas alineadas con la malla (test). Sólo banda estrecha (la media armónica de dos
Debye no es un Debye) y sólo con una geometría analítica: los generadores de `Phantom3D` no dan una función de la
posición, así que en la pierna se usa la escalera.

### 2.6 Serie de Mie

`em.analytic` calcula `a_n, b_n, c_n, d_n` de Bohren y Huffman (*Absorption and Scattering of Light by Small
Particles*, 1983, eq. 4.52-4.53 con `mu1 = mu`) con el convenio `e^{-iwt}` del libro y conjuga al final (el fasor de
wavesim es el conjugado; `m_BH = conj(sqrt(eps))`). `c_n` y `d_n` usan el wronskiano `psi xi' - xi psi' = i` para no
dividir por `j_n(m x)`. `j_n(z)` complejo por la fracción continua hacia atrás de `j_n / j_{n-1}` (la recurrencia
hacia atrás es estable para la solución mínima) escalada con `j_0` o `j_1` en forma cerrada; `y_n(x)` por recurrencia
hacia arriba. El campo interior es `sum E_n (c_n M_o1n - i d_n N_e1n)` con `j_n(m k r)`, el exterior la onda
incidente más `sum E_n (i a_n N_e1n - b_n M_o1n)` con `h_n^(1)`; `n_max = |m| x + 4 (|m| x)^{1/3} + 10`.

## 3. Resultados

### 3.1 Medio homogéneo y dipolo

Lámina de corriente en músculo y grasa homogéneos, 20 puntos por longitud de onda, float64; beta y alpha ajustadas a
lo largo de z frente al Cole-Cole de Gabriel (`dielectric.wave_numbers`). Objetivo: ≤ 1 % en beta, ≤ 3 % en alpha.

| tejido | f | E ∥ x: beta | alpha | E ∥ y: beta | alpha |
|---|---|---|---|---|---|
| músculo | 915 MHz | +0.379 % | +1.238 % | +0.379 % | +1.238 % |
| músculo | 2.45 GHz | +0.395 % | +1.243 % | +0.395 % | +1.243 % |
| grasa | 915 MHz | +0.382 % | +1.185 % | +0.382 % | +1.185 % |
| grasa | 2.45 GHz | +0.385 % | +1.183 % | +0.385 % | +1.183 % |

Es la dispersión numérica de Yee a 20 puntos/lambda (2.º orden), igual en las dos polarizaciones.

Dipolo puntual (`set_dipole`, eje z) frente al campo exacto del dipolo de Hertz (con el campo cercano), `Ez` en el
plano ecuatorial a lo largo de x, 20 puntos/lambda en el medio, malla 81³, CPML de 12 celdas, float64:

| medio | dx | error L2 (r = 4-24 celdas) | error L2 (r = 8-24 celdas) | error máx. de \|Ez\| | tiempo |
|---|---|---|---|---|---|
| aire, 2.45 GHz | 6.12 mm | 4.21 % | 0.98 % | 6.13 % | 8 s |
| músculo, 2.45 GHz | 0.83 mm | 5.21 % | 1.18 % | 7.07 % | 55 s |

El error se concentra a pocas celdas de la fuente (fuente de una celda, campo cercano en `1/r³`); desde 8 celdas la
onda esférica, con pérdidas, sigue a la exacta al ~1 %.

### 3.2 Losa multicapa a incidencia normal

Piel 2 / grasa 5 / músculo 15 / cortical 6 mm en aire, onda plana TF/SF (malla 2×2×nz sin CPML lateral: la losa es
lateralmente infinita), float64, 40 periodos de asentado y 8 de DFT. `T` = campo 10 mm detrás de la losa / incidente
en ese punto (= `analytic.insertion_transmission`); `R` = campo por encima de la caja (sólo la onda reflejada) /
incidente. Columna 2D: `MaxwellTMSolver` (cociente con y sin losa, el método de `em_validation.py`). Objetivo ≤ 0.5 dB.

| f | dx | \|T\| exacto | 3D: error \|T\| | fase | 2D TM: error \|T\| | fase | \|R\| exacto | 3D: \|R\| | tiempo 3D |
|---|---|---|---|---|---|---|---|---|---|
| 915 MHz | 0.5 mm | -9.227 dB | -0.004 dB | -0.02° | -0.067 dB | +0.19° | 0.6385 | 0.6385 | 11 s |
| 2.45 GHz | 0.5 mm | -12.386 dB | -0.052 dB | -0.57° | -0.053 dB | -0.73° | 0.5682 | 0.5694 | 4 s |
| 915 MHz | 0.25 mm | -9.227 dB | -0.001 dB | -0.01° | -0.026 dB | +0.01° | 0.6385 | 0.6385 | 46 s |
| 2.45 GHz | 0.25 mm | -12.386 dB | -0.013 dB | -0.14° | -0.001 dB | -0.28° | 0.5682 | 0.5685 | 15 s |

E ∥ x y E ∥ y dan exactamente lo mismo (a incidencia normal el problema es 1D). El error 3D baja ×4 al dividir `dx`
por dos (2.º orden). El 2D lleva además el error de su estimador de la DFT (§2.3): a 915 MHz, -0.067 dB frente a
-0.004 dB del 3D con la misma malla.

### 3.3 Caso invariante en y frente al 2D TM

Cilindro de músculo (radio 20 mm) con núcleo de cortical (8 mm) en aire, extruido en y, E ∥ y, la lámina de corriente
y el estimador de la DFT del 2D (`exact_dft=False`) en los dos. Diferencia de `Ey` en el plano central frente a
`MaxwellTMSolver` (fuera de las CPML), relativa al máximo:

| f | dx | variante | diferencia máx. | mediana | SAR máx. 3D/2D - 1 | tiempo 2D | tiempo 3D |
|---|---|---|---|---|---|---|---|
| 915 MHz | 1.0 mm | ny = 1, dt del 2D, float64 | 0 | 0 | 2e-16 | 0.8 s | 1.5 s |
| 915 MHz | 1.0 mm | ny = 16, CPML en y, dt 3D, float32 | 1.2e-3 | 5.1e-4 | 3e-5 | 0.8 s | 21 s |
| 2.45 GHz | 0.5 mm | ny = 1, dt del 2D, float64 | 0 | 0 | 2e-16 | 3.8 s | 7.3 s |
| 2.45 GHz | 0.5 mm | ny = 16, CPML en y, dt 3D, float32 | 1.3e-3 | 2.8e-4 | 8e-5 | 3.8 s | 83 s |

Con `ny = 1` y el `dt` del 2D el 3D es el 2D bit a bit (también con Numba). Un 3D completo (dt del límite 3D,
float32, CPML en y) queda a 0.13 % como máximo, por debajo del ~1 % pedido: es la dispersión temporal (el `dt` 3D es
un 18 % menor) más el redondeo de float32.

### 3.4 Esfera dieléctrica con pérdidas frente a Mie

**La serie de Mie** (§2.6), sin el FDTD:

| Comprobación | Resultado |
|---|---|
| m = 1.55, x = 5.2128 (ejemplo de BHMIE de Bohren y Huffman) | Q_ext = Q_sca = 3.1054255; referencia 3.1054255 (salida de `miepython` para ese caso en su documentación en línea; el apéndice del libro no se pudo abrir) |
| Rayleigh, m = 1.5 + 0.1i: Q_abs / [4x Im((m²-1)/(m²+2))] - 1 | 5.6e-5 (x = 0.01), 1.4e-3 (0.05), 5.6e-3 (0.1): la corrección O(x²) esperada |
| Sin contraste (eps = 1) | la onda incidente exacta (4e-16) |
| Continuidad en r = a (músculo, 2.45 GHz, a = 10 mm) | E tangencial y D normal continuos a 1e-8 (los puntos están a ±1e-9 a) |
| Absorción por integral de volumen del campo interior (Gauss) frente a Q_abs de los coeficientes | 4e-16 relativo |
| `j_n` esféricas: forma cerrada, serie de potencias (z complejo, n = 12), wronskiano con `y_n` (n ≤ 30) | 2e-15 a 1e-13 |

**FDTD frente a Mie.** Onda plana TF/SF con E ∥ x en aire a 2.45 GHz, caja en todo x e y, CPML de 10 celdas,
float32, 25 periodos de asentado; `z_ref` = centro de la esfera. Q_abs = potencia disipada en los nodos / intensidad
incidente / (pi a²). Error L2 relativo del campo complejo (tres componentes, centros de celda): «dentro» en
`r <= a - dx`, «fuera» en `a + dx < r < a + lambda0/4` (sin las celdas de la superficie, donde la componente normal
es discontinua y la media de dos nodos no tiene sentido). Puntos por longitud de onda dentro de la esfera.

Cortical (a = 15 mm, eps = 11.38 - 2.89j, |m| x = 2.64, Q_abs = 1.3229):

| ppw | dx | malla | volumen escalera / exacto | Q_abs escalera | Q_abs sub-celda | escalera: dentro | fuera | sub-celda: dentro | fuera | tiempo (2 casos) |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 3.57 mm | 55³ | 0.982 | +10.3 % | +0.49 % | 15.6 % | 10.9 % | 10.4 % | 5.6 % | 8 s |
| 15 | 2.38 mm | 67³ | 0.974 | +2.7 % | +0.95 % | 9.3 % | 6.7 % | 6.1 % | 3.9 % | 21 s |
| 20 | 1.79 mm | 81³ | 1.013 | +10.0 % | -0.01 % | 7.5 % | 6.4 % | 5.2 % | 3.1 % | 42 s |
| 30 | 1.19 mm | 105³ | 1.000 | +4.3 % | +0.04 % | 4.5 % | 4.0 % | 3.4 % | 2.2 % | 143 s |

Músculo (a = 10 mm, eps = 52.73 - 12.76j, |m| x = 3.78, Q_abs = 0.7760):

| ppw | dx | malla | volumen escalera / exacto | Q_abs escalera | Q_abs sub-celda | escalera: dentro | fuera | sub-celda: dentro | fuera | tiempo (2 casos) |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1.66 mm | 77³ | 1.013 | +15.9 % | +19.3 % | 13.8 % | 7.3 % | 10.1 % | 5.3 % | 41 s |
| 15 | 1.11 mm | 103³ | 0.996 | +9.2 % | +12.2 % | 7.2 % | 4.7 % | 6.5 % | 3.5 % | 125 s |
| 20 | 0.83 mm | 127³ | 0.979 | +6.2 % | +9.1 % | 4.7 % | 3.0 % | 5.0 % | 2.8 % | 341 s |

Lectura:

- **Cortical** (contraste moderado): en escalera la absorción sigue al volumen rasterizado y no converge de forma
  monótona (+10 % a 20 ppw con un 1.3 % de volumen de más). Con el promedio sub-celda Q_abs queda a < 1 % desde
  10 ppw (0.04 % a 30 ppw) y el error del campo baja de forma regular hasta 3.4 % dentro y 2.2 % fuera a 30 ppw. El
  objetivo del 2D (3-5 % con 15-20 ppw) se cumple con sub-celda fuera desde 15 ppw y dentro desde 20 ppw (5.2 %).
- **Músculo** (eps' = 53, 12 celdas de radio a 20 ppw): **no se alcanza** esa precisión. Q_abs converge despacio
  (+6 % en escalera, +9 % con sub-celda a 20 ppw) y el campo queda al 5 % dentro y al 3 % fuera. Aquí el promedio
  sub-celda no mejora la absorción: la diagonal del tensor (sin los términos cruzados) con este contraste parece
  sobrestimar la disipación en las celdas de la superficie; no se ha investigado más. 30 ppw en el músculo (~16 M
  celdas, ~25 min por caso en un núcleo) no se ha corrido.
- Asentado: con 50 periodos en vez de 25 la cortical a 15 ppw da el mismo Q_abs (1.3431 frente a 1.3430).

### 3.5 Banda ancha: un pulso frente a varias ondas continuas

Losa de §3.2 a dx = 0.5 mm, E ∥ y, un pulso de 0.5-4 GHz (4 polos de Debye comunes, `tissue_debye_model`), 41 883
pasos en 17 s; ondas continuas a cada frecuencia con la permitividad del mismo ajuste (`permittivity=`), para comparar
el solver y no el ajuste. Error frente a la matriz de transferencia con el ajuste de Debye.

| f | \|T\| exacto (Debye) | pulso: error \|T\| | fase | onda continua: error \|T\| | fase | \|T_pulso / T_continua - 1\| | tiempo onda continua |
|---|---|---|---|---|---|---|---|
| 0.5 GHz | -12.841 dB | -0.002 dB | +0.00° | -0.002 dB | +0.00° | 1.7e-5 | 25 s |
| 0.915 GHz | -9.225 dB | -0.004 dB | -0.02° | -0.004 dB | -0.02° | 3.4e-6 | 14 s |
| 1.5 GHz | -10.559 dB | -0.014 dB | -0.10° | -0.014 dB | -0.10° | 3.1e-6 | 9 s |
| 2.45 GHz | -12.384 dB | -0.052 dB | -0.57° | -0.052 dB | -0.57° | 1.8e-5 | 6 s |
| 3 GHz | -15.553 dB | -0.115 dB | -1.01° | -0.115 dB | -1.01° | 3.9e-5 | 5 s |
| 4 GHz | -22.765 dB | -0.244 dB | -2.06° | -0.243 dB | -2.06° | 1.0e-4 | 3 s |

El pulso reproduce las seis ondas continuas a ≤ 1e-4 (el test exige 1 %); el error común frente a la solución exacta
es el de la malla (a 4 GHz, 14 puntos/lambda en el músculo).

Esfera de cortical (a = 15 mm) con un pulso de 1-3 GHz (escalera, dx = 1.475 mm = 20 ppw a 3 GHz, 91³ celdas, 4148
pasos, 83 s, fasores en toda la malla a 5 frecuencias: 989 MiB de pico) frente a Mie:

| f | dentro (Mie con Gabriel) | fuera | dentro (Mie con el Debye) | fuera |
|---|---|---|---|---|
| 1 GHz | 7.51 % | 2.55 % | 7.57 % | 2.55 % |
| 1.5 GHz | 7.25 % | 2.97 % | 7.25 % | 2.97 % |
| 2 GHz | 6.64 % | 3.79 % | 6.64 % | 3.79 % |
| 2.45 GHz | 5.83 % | 4.93 % | 5.83 % | 4.93 % |
| 3 GHz | 6.26 % | 6.04 % | 6.24 % | 6.06 % |

Son errores de escalera (del orden de la tabla de §3.4 a resolución parecida); que sean iguales con Gabriel y con el
Debye confirma que el ajuste no aporta error apreciable.

### 3.6 Rendimiento

Malla 120³ de músculo, CPML de 10 celdas, onda plana TF/SF; Numba ya compilado (la primera compilación cuesta ~30 s y
queda en caché). Máquina de 4 núcleos **compartida con otros dos procesos de cálculo** (carga ~3 durante la medida):
las cifras con 4 hilos no son representativas, y las de 1 hilo varían ±40 % entre ejecuciones (111 Mceldas·paso/s en
float32 en una medida anterior con más carga).

| modo | backend | dtype | hilos | Mceldas·paso/s | memoria del solver (B/celda) |
|---|---|---|---|---|---|
| banda estrecha | NumPy | float32 | 1 | 42 | 64 |
| banda estrecha | NumPy | float64 | 1 | 27 | 129 |
| banda estrecha | Numba | float32 | 1 | 181 | 56 |
| banda estrecha | Numba | float32 | 4 | 80 | 56 |
| banda estrecha | Numba | float64 | 1 | 54 | 113 |
| banda estrecha | Numba | float64 | 4 | 57 | 113 |
| Debye, 4 polos | Numba | float32 | 1 | 53 | 152 |
| Debye, 4 polos | Numba | float32 | 4 | 43 | 152 |

Memoria del solver (`memory_bytes`): campos y coeficientes (48 B/celda en float32) más la CPML (en 120³ con 10
celdas, ~8 B/celda) más 24 B/celda por polo de Debye. La DFT de banda estrecha añade 48 B/celda (tres fasores
complex128) y el resultado otros ~60 B/celda (campo en las celdas, potencia, |E|²). En la pierna a 2.45 GHz (1.06 M
celdas) el proceso entero llegó a 772 MiB con tres soluciones 3D y seis 2D en memoria; la esfera de músculo de 127³
(2 M celdas), a 1.7 GiB. Una malla de ~3 M celdas en banda estrecha cabe en ~1 GB. En banda ancha con fasores en toda
la malla son 48 B/celda por frecuencia: usa `box` para guardar sólo una región.

### 3.7 Aplicación: pierna 3D a 915 MHz y 2.45 GHz

`limb_3d("pierna")` (tibia anterior a 6 mm de la piel, peroné, 5-6 cm de músculo detrás, dos vasos), interfaces lisas
(`roughness=False`, para que la comparación con el 2D no dependa de la rugosidad), fractura transversal de 2 mm en la
tibia en x = 0, en agua; 16 celdas más de agua arriba y abajo para las caras de la TF/SF. Onda plana desde arriba con E
∥ al hueso (x) o ⊥ (y), normalizada a 1 V/m de onda incidente (en el agua) en el plano de la piel. 915 MHz con dx = 2
mm (50×50×82 celdas, 8478 pasos, 18 s por caso) y 2.45 GHz con dx = 1 mm (80×100×132 = 1.06 M celdas, 12 715 pasos,
~90 s por caso; 14 puntos/lambda en el agua y 17 en el músculo). Todo el caso (seis 3D y doce 2D) tarda 325 s y el
proceso llega a 772 MiB.

2D TM de referencia: el corte transversal `slice_x(-25 mm)` (lejos de la fractura) con E ∥ hueso, que es el
problema exacto para un miembro uniforme a lo largo de x, y el corte longitudinal `slice_y(0)` con E ⊥ hueso, que
convierte la pierna en capas planas infinitas en y (el modelo longitudinal 2D de wavesim). La normalización del 2D es
la misma (incidente en la piel medido en una simulación sólo con agua).

| f | caso | SAR pico (W/kg por (V/m)²) | SAR 1 g | SAR 10 g | potencia en el hueso | potencia a > 2 cm de la piel | \|E\| 2 mm dentro de la cortical (x = -25 mm, y = 0) |
|---|---|---|---|---|---|---|---|
| 915 MHz | 3D, E ∥ hueso, con fractura | 8.62e-4 | 3.67e-4 | 2.79e-4 | 2.2 % | 63.0 % | 1.069 |
| 915 MHz | 3D, E ⊥ hueso, con fractura | 1.47e-3 | 4.26e-4 | 2.87e-4 | 8.2 % | 58.6 % | 1.755 |
| 915 MHz | 3D, E ⊥ hueso, sin fractura | 1.05e-3 | 4.23e-4 | 2.81e-4 | 6.2 % | 59.2 % | 1.820 |
| 915 MHz | 2D TM transversal (E ∥ hueso) | 8.68e-4 | - | - | 1.6 % | 63.0 % | 1.067 |
| 915 MHz | 2D TM longitudinal (E ⊥, capas planas) | 1.78e-3 | - | - | 31.5 % | 44.4 % | 1.438 |
| 2.45 GHz | 3D, E ∥ hueso, con fractura | 1.51e-3 | 4.69e-4 | 3.82e-4 | 7.3 % | 21.8 % | 1.046 |
| 2.45 GHz | 3D, E ⊥ hueso, con fractura | 1.61e-3 | 5.16e-4 | 3.83e-4 | 10.9 % | 26.0 % | 1.292 |
| 2.45 GHz | 3D, E ⊥ hueso, sin fractura | 1.52e-3 | 5.16e-4 | 3.82e-4 | 8.7 % | 27.6 % | 1.169 |
| 2.45 GHz | 2D TM transversal (E ∥ hueso) | 1.50e-3 | - | - | 5.1 % | 21.6 % | 1.079 |
| 2.45 GHz | 2D TM longitudinal (E ⊥, capas planas) | 2.13e-3 | - | - | 58.7 % | 34.3 % | 1.948 |

(Fracciones de la potencia absorbida en el tejido. En 3D «hueso» incluye cortical, médula, hematoma y edema de los
dos huesos en toda la longitud; en el 2D transversal, el corte sin fractura.)

3D frente a 2D en el mismo plano:

| f | comparación | \|E\| en la vertical y = 0 (piel a 2 mm dentro de la cortical): diferencia máx. | SAR máx. del plano 3D/2D | contraste de la fractura en \|E\| (centro del hueco, 2 mm dentro de la cortical) 3D | 2D |
|---|---|---|---|---|---|
| 915 MHz | E ∥: 3D en x = -26 mm frente a 2D transversal | 0.4 % | 0.986 | - | - |
| 915 MHz | E ⊥: 3D en y = 0 frente a 2D longitudinal | 19.6 % | 0.645 | -33.5 % | +2.1 % |
| 2.45 GHz | E ∥: 3D en x = -25 mm frente a 2D transversal | 2.2 % | 0.997 | - | - |
| 2.45 GHz | E ⊥: 3D en y = 0 frente a 2D longitudinal | 35.7 % | 0.656 | -25.9 % | -30.7 % |

Lectura (¿cuánto se equivoca el 2D?):

- **E ∥ hueso, lejos de la fractura**: el 2D TM transversal es el problema correcto y el 3D lo confirma (|E| a 0.4-2 %,
  SAR máximo del plano a 1-1.5 %). Para esta polarización el 2D transversal basta fuera de la fractura.
- **E ⊥ hueso**: el 2D TM no tiene esta polarización en el corte transversal, y el corte longitudinal la modela con
  la pierna como capas planas. Eso sobrestima el SAR máximo del plano ×1.5, el campo en la cortical un 20-36 % y,
  sobre todo, la fracción de la potencia que se absorbe en el hueso (31 % y 59 % frente a 8 % y 11 % en 3D): en capas
  planas la cortical es una losa infinita que intercepta toda la onda, mientras que la tibia real es un tubo de 22 mm
  en un miembro de 90 mm. El contraste de la fractura tampoco es fiable en 2D: a 915 MHz tiene el signo contrario
  (+2 % frente a -34 %) y a 2.45 GHz difiere 5 puntos (-31 % frente a -26 %).
- **Polarización**: en 3D, E ⊥ hueso da un SAR pico un 70 % (915 MHz) y un 7 % (2.45 GHz) mayor que E ∥ y lleva
  3.7 y 1.5 veces más potencia al hueso; los SAR promediados en 10 g apenas cambian (≤ 3 %) porque los domina el
  músculo. La fractura (hematoma en el hueco) sube el SAR pico un 40 % a 915 MHz con E ⊥, pero no el promediado.
- Penetración: a 915 MHz el 59-63 % de la potencia se absorbe a más de 2 cm de la piel; a 2.45 GHz, el 22-28 %.

## 4. Límites y riesgos

- **Sólo incidencia normal** en la TF/SF (a lo largo de +z): no hay onda plana oblicua (haría falta interpolar la malla
  1D y corregir su dispersión). Las caras deben estar en un fondo homogéneo (se comprueba); con la pierna de
  `limb_3d`, que sólo trae 5 mm de agua encima, hay que ampliar la malla (`grid=`, §1).
- **Promedio sub-celda**: sólo en banda estrecha y con una geometría analítica (`eps_at`); los generadores de
  `Phantom3D` no la ofrecen (`limb_3d_fractions` da fracciones de volumen, pero no la normal de la interfaz), así que
  la pierna va en escalera. En la esfera de músculo (contraste alto) no mejoró la absorción (§3.4).
- **Esfera de músculo**: el error de absorción a 20 ppw (+6 % / +9 %) y del campo (5 % dentro) es mayor que el
  objetivo de 3-5 %; con contrastes así hacen falta ≥ 30 ppw dentro (no se ha corrido).
- **SAR promediado simplificado** (cubo centrado sólo sobre el tejido, sin el desplazamiento de IEC 62704-1 en la
  superficie): sirve para comparar casos entre sí, no para certificar.
- **Lámina de corriente y dipolo**: la lámina normaliza con la impedancia real como el 2D (en un medio con pérdidas
  emite `sqrt(eps'/eps)` V/m) y sólo existe en banda estrecha; el dipolo es de una celda (error del 4-5 % a menos de
  8 celdas).
- **Rendimiento con varios hilos** sin medir en limpio (máquina compartida); el paralelismo es por planos z como en el
  elástico 3D. Las correcciones TF/SF y la DFT del fasor incidente se hacen en Python en cada paso (despreciable en
  mallas grandes; en mallas de pocas celdas, como las losas, domina el tiempo).
- Sin modo de banda ancha con lámina de corriente, sin materiales magnéticos ni anisótropos, sin fuentes de antena
  realistas (puerto, impedancia de entrada).

## 5. Reproducir

```bash
NUMBA_NUM_THREADS=1 python benchmarks/em3d_validation.py                    # todo (~30 min en un núcleo)
python benchmarks/em3d_validation.py --case plane|slab|yinv|mie|broadband|speed|leg
NUMBA_NUM_THREADS=2 python -m pytest -q tests/test_fdtd3d.py                # 17 tests, ~40 s
```

Tiempos medidos (1 hilo, máquina compartida): `plane` 73 s, `slab` 163 s, `yinv` 117 s, `mie` 806 s, `broadband`
171 s, `speed` 36 s, `leg` 325 s.
