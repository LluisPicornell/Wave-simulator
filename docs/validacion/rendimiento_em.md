# Rendimiento de los solvers electromagnéticos

Tiempos medidos con `NUMBA_NUM_THREADS=1` en una máquina de 4 núcleos compartida con otros procesos (la carga
añade ruido de ±30 %). Se reproducen con `benchmarks/rendimiento_em.py --case fdtd`. La física no cambia: los
tests de `tests/test_em_performance.py` comprueban la igualdad de resultados.

## 1. Maxwell FDTD: NumPy frente a Numba (float64, 1 hilo, Numba ya compilado)

`backend="auto"|"numpy"|"numba"` en `MaxwellTMSolver` y `BroadbandMaxwellTMSolver`; kernels fusionados con CPML y
polos de Debye en `wavesim/em/_numba_fdtd.py`.

| caso | NumPy (s) | Numba (s) | aceleración | discrepancia relativa |
|---|---|---|---|---|
| onda plana en músculo 2.45 GHz (`test_em.py`) | 0.8 | 0.2 | 3.5x | 0 |
| losa banda estrecha 2.45 GHz, dx 0.25 mm | 1.4 | 0.3 | 4.7x | 0 |
| losa banda ancha 0.1-10 GHz, dx 0.25 mm | 4.8 | 1.2 | 4.0x | 1.6e-13 |
| Mie banda estrecha 2.45 GHz (a = 20 mm, 15 ppw) | 6.9 | 2.7 | 2.6x | 0 |
| Mie banda ancha 0.5-3 GHz | 28.0 | 13.7 | 2.0x | 5.8e-26 |

La aceleración baja en mallas grandes porque el coste pasa a ser el de la DFT por bloques y el de los monitores,
que siguen en NumPy. No se midió con 4 hilos (la máquina no estaba libre). Compilar la primera vez cuesta ~5 s
(`cache=True` lo evita después).

## 2. EIT (COCG cuasi-estático): pierna de 16 electrodos, 200x200 celdas (dx 0.5 mm), tol 1e-10

Dónde se iba el tiempo: 922 iteraciones de COCG con Jacobi, y el 90 % del tiempo estaba en `np.linalg.norm`
(`dot` de OpenBLAS multihilo con la CPU compartida: ~10 ms por norma frente a ~0.1 ms de una suma de NumPy). Se
sustituye por `norm2` y se añade un multigrid de agregación 2x2 (Galerkin, Gauss-Seidel rojo-negro, ciclo en V con
sobrecorrección 1.8 y 2+2 suavizados, `wavesim/em/multigrid.py`) como precondicionador; la jerarquía se construye una
vez por solver y las 16 inyecciones la reutilizan.

| configuración | iteraciones | tiempo por inyección | 16 inyecciones |
|---|---|---|---|
| antes (Jacobi, `np.linalg.norm`) | 922 | 16 s | ~255 s |
| Jacobi con `norm2` (OpenBLAS a 1 hilo) | 922 | ~1.1 s | ~18 s |
| multigrid (nuevo por defecto con Numba) | 13-16 | 0.044 s | 0.7 s |

La solución coincide con la de Jacobi en 3e-12 (relativo). Robustez probada en pierna 0.5 y 1 mm, con fractura,
antebrazo, 50 kHz-1 MHz, electrodo ideal y CEM: 12-16 iteraciones siempre. Hace falta regularizar el nivel grueso del
caso de Neumann puro (singular) con un término a tierra de 1e-8 de la diagonal; sin él, sobrecorrección >1 diverge.

No se ha medido el benchmark `eit_validation.py leg` completo (30 min antes) ni el ejemplo 10 de punta a punta;
la mejora por solve anterior es la medida disponible. La difusión óptica (`optics.py`) no se ha tocado.
