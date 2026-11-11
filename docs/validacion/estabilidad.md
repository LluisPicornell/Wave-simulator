# Estabilidad a tiempos largos del solver elástico

Dos inestabilidades que encontró la validación (`elastico.md` §6) y su arreglo.
Tests: `tests/test_estabilidad.py` (5 tests, ~20 s, NumPy y Numba; fallan con
el código anterior).

## Diagnóstico

**Placa de cortical en la CPML lateral.** Placa de 3 mm (agua o aire alrededor,
dx 0.15 mm, 300 kHz, CPML de 20 celdas). Energía total (cinética + elástica):
×46 cada 500 pasos desde el paso ~1500; el 99 % de la energía dentro de la
placa en la parte exterior de la CPML. Medido:

| configuración | resultado (CPML clásica) |
|---|---|
| hueso homogéneo, agua homogénea, aire homogéneo | estable |
| placa 3 mm en agua / en aire | crece (×7 / ×17 en amplitud cada 500 pasos) |
| placa 9 mm / 1.5 mm en agua | crece; frecuencia del modo 100 / 294 / 602 kHz para 9 / 3 / 1.5 mm (f·espesor ≈ 0.9 MHz·mm: es un modo de placa) |
| capa fluida (cs = 0) en agua | estable |
| sin la corrección de interfaz `_slip_terms`, o CFL 0.6 | crece igual |
| alpha (CFS) ×3 / constante en la capa | crece más despacio / sigue creciendo (aire, ×10¹⁰ en 40 000 pasos) |

Análisis de autovalores del sistema semidiscreto (placa invariante en x,
modo de Fourier en x, CPML con `d` y `alpha` congelados, operadores en z del
solver; guion de trabajo, no incluido): sin CPML todos los autovalores son
imaginarios puros; con cualquier `d > 0` la placa tiene autovalores con
Re λ ≈ 0.013-0.03·d, y el más inestable está en k ≈ 330 rad/m, 557 kHz: la
rama retrógrada de S1 (entre el ZGV, 537 kHz, y el corte, 583 kHz, calculados
con Rayleigh-Lamb). Es la inestabilidad clásica de las PML con ondas
retrógradas (velocidad de grupo y de fase opuestas; Bécache, Fauqueux & Joly,
J Comput Phys 188:399, 2003; para guías elásticas, Skelton, Adams & Craster,
Wave Motion 44:573, 2007). El CFS la frena pero no la quita. Con fluido
alrededor y sin cizalla no hay ramas retrógradas: por eso sólo la cortical.

**Tibia realista `anciano_osteoporotico`.** Mismo mecanismo: en el fantoma
longitudinal la cortical (1.7 mm con el escalado 0.6) cruza la CPML lateral
entre hematoma/grasa y médula. Con `n_mechanisms=0` (malla 154×88, 2 MHz) la
energía crecía ×5.6 cada 1000 pasos; el 70 % estaba en la CPML izquierda,
dentro de la cortical. No hizo falta tocar el GSLS ni las celdas porosas.

## Arreglo: M-PML

`CPMLDerivative(ratio=p)` (`core/pml.py`, kernels Numba): dentro de la capa
normal a una dirección se amortiguan también las derivadas paralelas con
`p·d` (M-PML, Meza-Fajardo & Papageorgiou, Bull Seismol Soc Am 98:1811,
2008), en forma convolucional con el mismo `alpha`. `ElasticSolver` usa por
defecto (`pml_ratio=None`) `PML_RATIO = 0.5` sólo si en alguna franja CPML hay
una capa sólida con fluido a ambos lados (`_layer_crosses_pml`: placas,
cortical de los fantomas de extremidad); si no, la CPML de antes, idéntica bit
a bit (tejido blando, sólido homogéneo, semiespacio: estables y sin coste).
Maxwell y el solver 3D no cambian (el 3D conserva la CPML clásica: el test que
compara 2D y 3D fija `pml_ratio=0`).

Elección de p (autovalores, placa en agua, máx Re λ en 1/s):

| p | d = 3e5 | 1e6 | 3e6-2e7 |
|---|---|---|---|
| 0 | +8.9e3 | +2.5e4 | +7e4 … +2.6e5 |
| 0.2 | +4.0e3 | +2.7e3 | ≤ 0 |
| 0.3 | +2.3e3 | ≤ 0 | ≤ 0 |
| 0.5 | +54 | ≤ 0 | ≤ 0 |
| 1.0 | ≤ 0 | ≤ 0 | ≤ 0 |

FDTD, 40 000 pasos: p = 0.2 aún crece despacio (×2.6 en la segunda mitad,
placas de 1.5 y 3 mm en agua y 3 mm en aire); p ≥ 0.3 (agua) y p = 0.5 (agua y
aire, 1.5, 3 y 9 mm) no crecen. Tibia osteoporótica con p = 0.5, 30 000 pasos
sin atenuación: la energía cae de 5e-2 a 9e-9 (antes 1.6e13 a los 20 000).

## Coste

La M-PML no es perfectamente adaptada. Reflexión de una fuente puntual (dominio
pequeño frente a uno 3× mayor, banda de 10 celdas junto a la capa, 10 ppw):

| medio | CPML (p = 0) | p = 0.2 | p = 0.5 |
|---|---|---|---|
| agua, amplitud | −80.5 dB | −61 dB | −53 dB |
| cortical (fuerza), amplitud | −80.5 dB | −36 dB | −28 dB |

En agua (tejido blando) el eco de la capa sigue por debajo del speckle; en el
hueso es mayor (4 % en amplitud, sobre todo onda S rasante). Los montajes
cuasi-1D cuyo dominio lateral es todo CPML (franjas de pocas columnas) deben
usar `pml_ratio=0`: la M-PML amortiguaría la propia onda plana.

## Pendiente

- Un p dependiente del material (sólo cerca de sólidos) evitaría el coste en
  el tejido blando; no se probó.
- B-mode de tibia básica con fractura (3 MHz, 0°, 16 × 14 mm, dx 60 µm; la
  M-PML se activa porque la cortical cruza la CPML): diferencia L2 de la
  envolvente 1.3 %; píxeles por encima de −30 dB: p95 de |Δ| 0.69 dB, máx
  2.5 dB (bordes laterales). El test de absorción de la CPML (agua) no cambia.
- El solver 3D (`elastic3d.py`) sigue con la CPML clásica y tendría la misma
  inestabilidad con una placa que cruce su capa.
