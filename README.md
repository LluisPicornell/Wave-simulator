# Wave-simulator

**English** | [Español](README.es.md)

A simulator of waves travelling through human tissue. It is used to study
whether ultrasound, microwaves, light, X-rays or magnetic resonance can see
inside a limb (specifically, **bone fractures**) and how easily. It can also
generate labelled synthetic images to train a neural network that detects
fractures.

Almost everything is 2D (a slice of the limb), written in Python + NumPy, and
runs on the CPU (Numba optional). There is also a 3D elastic solver for small
domains.

The code, comments and detailed documents under `docs/` are in Spanish; this
README summarises them in English.

## What it simulates and with which physics

Each kind of wave needs different physics. Using the wrong one gives
meaningless results (for example, solving X-rays as a wave on a grid would
need about 10¹⁰ points per millimetre).

| Modality | Physics | Module |
|---|---|---|
| **Ultrasound** (0.5-15 MHz) | Full elastodynamics: P and S waves, mode conversion in bone, reflection and refraction by impedance, speckle from microstructure, surface roughness. 4th-order FDTD. GSLS viscoelastic attenuation with one weight per mechanism (2 mechanisms by default: error < 2 % against the α₀·f^y law within the pulse band). Fluid-solid interface correction at oblique incidence. CPML boundaries, with automatic M-PML when a solid layer (the cortex) crosses them. Optional transversely isotropic cortex (`anisotropic_bone=True`, off by default). | `wavesim.ultrasound` |
| **Ultrasound imaging** | Probe with element response and apodisation; plane-wave, focused and synthetic-aperture (STA) transmission; DAS, coherence factor, DMAS and SLSC beamformers; two-layer speed correction; PSF, speckle, CNR and gCNR metrics; full matrix capture (FMC) and reconstruction with any subset of transmitters and receivers. | `wavesim.ultrasound` (`transducer`, `scan`, `beamforming`, `quality`, `fmc`) |
| **3D ultrasound** | Same scheme (velocity-stress, 4th order, CPML, GSLS, optional anisotropic cortex, interface correction) on a 3D grid, with a tubular tibia phantom. For domains of a few cm at ≲ 1 MHz. | `wavesim.ultrasound.elastic3d`, `wavesim.phantoms.limb3d` |
| **Radio and microwaves** (10 MHz-300 GHz) | TM-mode Maxwell FDTD with Gabriel 1996 Cole-Cole tissue models: narrowband (properties at f₀) or **broadband** with a multipole Debye fit (ADE) in a single pulse; optional sub-cell averaging; Numba kernels. Gives the field, penetration and SAR. | `wavesim.em.fdtd_tm`, `fdtd_dispersive` |
| **Low frequency** (< 10 MHz) | Quasi-static regime div(σ* ∇φ) = 0 with Gabriel's complex admittivity, ideal electrodes or the **complete electrode model (CEM)**, one-step difference reconstruction or **Gauss-Newton** (Tikhonov, Laplacian, NOSER, TV), multigrid preconditioner (EIT). | `wavesim.em.quasistatic`, `eit`, `multigrid` |
| **Terahertz, IR, visible, UV** | Light transport: diffusion, ballistic attenuation and **photon Monte Carlo** (MCML/MCX style, Fresnel, optional time resolution, Numba). | `wavesim.em.optics`, `montecarlo` |
| **X-rays and gamma** (10 keV-10 MeV) | Beer-Lambert with NIST/ICRU-44 data, tube spectrum with K lines, heel effect and Al/Cu filters; parallel or fan beam (FBP and OS-SART with TV); Monte Carlo scatter (Compton with S(x,Z), Rayleigh); detector with MTF and noise; beam hardening, Poisson noise. | `wavesim.em.xray`, `fanbeam`, `xray_scatter`, `xray_detector` |
| **Magnetic resonance** | Two engines: analytic signal equations (default, instantaneous) or **spatial Bloch** (`engine="bloch"`) with isochromats: fat chemical shift, susceptibility B0 field, T2*, surface coil and k-space noise. T1, T2, PD-FS, STIR, GRE and UTE sequences (UTE analytic only). | `wavesim.mri` |

### Tissues

Skin, fat, muscle, blood, cortical bone, trabecular bone, yellow marrow,
haematoma, oedematous marrow, water/gel and air. Each one has acoustic,
elastic, dielectric, X-ray, optical and MR properties (`wavesim/materials/`).

### Fracture phantoms

`wavesim.phantoms` generates longitudinal slices (tibia, forearm, thigh) and
cross-sections (leg with tibia and fibula, forearm with radius and ulna,
thigh). They include:

- rough interfaces (skin, periosteum, endosteum) and internal heterogeneity;
- transverse, oblique, hairline, comminuted, incomplete and displaced
  fractures (cortical step);
- **internal** fractures that do not open the cortex facing the probe:
  intracortical cracks, fractures of the far cortex, trabecular bone bruise
  (microfractures and oedema, `tibia_metafisis` preset) and stress fractures
  with periosteal or endosteal reaction;
- crack with a rough surface, periosteal haematoma and marrow oedema;
- **body types** (`BODY_TYPES`: thin, normal, overweight, obese, muscular,
  osteoporotic elderly) that scale skin, fat, muscle and bone within published
  ranges; the datasets draw them at random;
- ground-truth mask of the fracture (for segmentation);
- an optional **realistic mode** (`realism="realista"`, see below): porous
  bone, trabecular network, fractal roughness, fasciae and periosteum.

The same seed produces the same anatomy at any resolution, so modalities can
be compared on the same "patient".

## Installation

```bash
pip install -e ".[dev]"         # numpy, matplotlib (+ pytest for the tests)
pip install -e ".[dev,fast]"    # plus Numba: ultrasound ~1.6-3x faster
```

**Optional fast backend (Numba).** The elastic solver defaults to
`backend="auto"`: fused, parallel Numba kernels if Numba is installed, NumPy
otherwise. Both give the same result (the tests compare them).
`backend="numpy"` forces the reference implementation; `backend="numba"`
fails with a clear error if Numba is missing. Float32 only. The first run
compiles the kernels (a few seconds, then cached) and the number of threads
is set with `NUMBA_NUM_THREADS`.

## Quick start

```python
from wavesim.phantoms import longitudinal_limb, Fracture
from wavesim.ultrasound import simulate_bmode
from wavesim.em import xray
from wavesim.mri import mri_scan

# Tibia with a fracture displaced by 1 mm
p = longitudinal_limb("tibia", Fracture(displacement=1e-3), width=24e-3, depth=22e-3, dx=0.06e-3)

echo = simulate_bmode(p, f0=3e6, angles=(-8, 0, 8))          # ultrasound (1-2 min)
ct = xray.ct_scan(p.replace_tissue("water", "air"))           # CT (seconds)
mr = mri_scan(p.replace_tissue("water", "air"), "STIR")       # MRI (instantaneous)
```

## Examples

Each script saves its figures to `outputs/`:

| Script | What it shows |
|---|---|
| `examples/01_fantomas.py` | Anatomies and every fracture type |
| `examples/02_propagacion_ultrasonido.py` | Animation of the pulse crossing the fractured tibia and peak-pressure map (penetration and acoustic shadow) |
| `examples/03_ecografia_fractura.py` | B-mode ultrasound: healthy, displaced fracture and hairline crack |
| `examples/04_espectro_penetracion.py` | Penetration across the whole spectrum and loss table for crossing the forearm |
| `examples/05_microondas.py` | Electric field and SAR from 128 MHz to 10 GHz, and the change caused by the fracture |
| `examples/06_rayos_x_tac.py` | Transverse and longitudinal CT, and radiograph |
| `examples/07_resonancia.py` | The same leg with six MRI sequences |
| `examples/08_luz.py` | Light from 450 to 1450 nm through the forearm |
| `examples/09_generar_dataset.py` | Labelled dataset to train a neural network |
| `examples/10_impedancia_eit.py` | EIT of a leg at 50 kHz with 16 electrodes: potential and current density, healthy vs fractured, change in boundary voltages and difference reconstruction |
| `examples/11_fracturas_internas_baja_frecuencia.py` | Does low-frequency ultrasound (0.25-1 MHz) see the internal fractures that imaging misses? Through-transmission across the leg and axial transmission along the cortex, in normal and obese bodies (`--procesos N` to parallelise; ~30 min) |
| `examples/12_realismo.py` | Basic vs realistic phantom (porous bone, trabeculae, fasciae) in ultrasound, CT and MRI, in a healthy adult and an osteoporotic elderly person |
| `examples/13_luz_montecarlo.py` | Light in the forearm at 800 nm: photon Monte Carlo vs diffusion (diffusion fails near the illuminated skin and at boundaries) |
| `examples/14_rm_artefactos.py` | MRI of the leg with the spatial Bloch simulator: fat chemical shift and susceptibility B0 field, without and with artefacts |

### Main result: what gets through a ~47 mm forearm?

Output of `04_espectro_penetracion.py`: intensity loss when crossing the
whole forearm at normal incidence, surrounded by water. For radio and
microwaves the reflections are computed coherently (transfer matrix: layers
thinner than λ interfere); for ultrasound they are summed per interface (the
broadband pulse averages out the interference).

| Wave | Loss | Comment |
|---|---|---|
| Ultrasound 1 MHz / 5 MHz | 15 / 42 dB | Gets through, but bone reflects and attenuates strongly; this is why ultrasound images see the bone surface and leave a shadow below it |
| MRI RF 64 MHz | 3 dB | Penetrates (which is why MRI works), but λ in tissue ≈ 50 cm: gives no resolution on its own |
| Microwaves 915 MHz / 2.45 GHz | 9 / 21 dB | Penetrate a few cm; λ ≈ 1.8 cm in muscle at 2.45 GHz: very poor resolution |
| mm-waves 60 GHz | 650 dB | Absorbed in the skin |
| Terahertz 1 THz | ~1900 dB | Water absorbs it within ~50 µm |
| Near infrared 800 nm | 79 dB (diffuse) | Some light gets through, but fully diffuse: no photon arrives undeviated |
| Green light 530 nm | 460 dB | Absorbed by haemoglobin |
| UV 300 nm | > 3000 dB | Does not get past the epidermis |
| X-rays 60 keV | 5 dB | Gets through with bone/soft-tissue contrast: the standard tool for fractures |
| Gamma 662 keV | 2 dB | Gets through with almost no contrast |

### Ultrasound image of the fracture

![B-mode ultrasound](docs/img/03_ecografia.png)

3 MHz linear probe (λ ≈ 0.5 mm) over the tibia. The displaced fracture (1 mm
step) is clearly visible: a mean change of 10.1 dB in the region. The 0.15 mm
hairline crack is barely distinguishable (4.7 dB) because it is thinner than
λ/3, as in clinical practice. The marrow oedema is not visible: it lies in the
acoustic shadow of the bone.

![Pulse propagation](docs/img/02_propagacion.png)

Pulse propagation (frames from `02_propagacion.gif`): reflection at the bone
surface, splitting of the wavefront at the fracture step, faster transmission
along the cortex and scattering by the microstructure.

### X-rays, CT and MRI

![CT and radiograph](docs/img/06_rayos_x.png)

![MRI](docs/img/07_resonancia.png)

MRI sees the fracture mainly through the oedema and haematoma (bright in STIR
and fat-saturated PD). With microwaves (`05_microondas.py`), the fracture with
its oedema and haematoma changes the field by a median of 1.5 dB at 2.45 GHz,
but with centimetre resolution.

![Penetration across the spectrum](docs/img/04_penetracion.png)

### Internal fractures: do low frequencies see them?

![Detection of internal fractures](docs/img/11_resumen.png)

`11_fracturas_internas_baja_frecuencia.py` compares ultrasound imaging
(3 MHz) with two low-frequency ultrasound techniques (0.25, 0.5 and 1 MHz):
through-transmission across the leg (TT) and axial transmission along the
cortex (AT, the technique used in bone ultrasonometry). To avoid confusing the
fracture with anatomical differences, the index D subtracts the difference
between two healthy legs from the change caused by the fracture
(D ≥ 6 dB = detectable).

| Lesion | Imaging at 3 MHz | Low frequency |
|---|---|---|
| Fracture displaced 1 mm (control) | yes (D ≈ +13 dB) | yes with AT (+9 to +14 dB) |
| Intracortical crack or far-cortex crack | no | no (D between −1 and −21 dB) |
| Stress fracture with endosteal reaction | no | **yes with TT** (+7 to +9 dB): the signal arrives ~0.45 µs earlier |
| Bone bruise (trabecular oedema) | no | doubtful with TT (+4 to +6 dB) |

Figures for the normal body. In the obese body, AT at 0.25 MHz gives
D ≈ +8 dB for the superficial crack and the intracortical crack, but only at
that frequency (between +0.4 and −8 dB at 0.5 and 1 MHz) and with a single
pair of healthy reference legs: not enough to count as detection.

Conclusion: no technique distinguishes the internal **crack** itself, because
it is 0.1-0.3 mm wide, much smaller than the wavelength (6 mm at 0.25 MHz),
and scatters almost nothing. What low frequencies do see is the **bone's
reaction**: the endosteal thickening of a stress fracture changes the time of
flight through the whole bone. In practice, thickness differences between
people also change that time, so the measurement would need to be compared
with the other leg or swept along the bone. In an obese body the wave guided
by the cortex (AT) no longer separates from the direct wave at low frequency.
Limits: 2D slice (the real tibia is a tube), isotropic, non-porous bone,
noise-free simulation.

### How many transmitters and receivers does a portable device need?

[docs/estudio_emisores_receptores.md](docs/estudio_emisores_receptores.md)
(`benchmarks/estudio_emisores_receptores.py`, 281 simulations with electronic
noise and different healthy legs as negatives) compares everything from a
single transducer up to a 32 × 32 probe. In 3 MHz imaging, detection of a
displaced fracture grows with transmitters × receivers (AUC 0.38 with 1 × 1,
0.93 with 8 × 8), and firing all elements at once gives the best return per
channel. In axial transmission at 0.5-1 MHz, one transmitter + one receiver is
enough with good electronics (AUC 0.84-0.94); more receivers only add
robustness to noise, and the technique fails in the obese body. Small sample
(6 healthy legs vs 2-3 of each lesion): these are trends, not sensitivity
figures.

### Realistic mode: porous bone, trabeculae and fasciae

![Basic vs realistic phantom](docs/img/12_realismo.png)

`longitudinal_limb(..., realism="realista")` and `cross_section_limb(..., realism="realista")`
add to the layered phantom:

- **cortical porosity**: 5 % Haversian canals in the healthy adult and ~18 %,
  with cavities and endocortical trabecularisation, in `anciano_osteoporotico`;
- an explicit **trabecular network** (BV/TV, thickness, orientation and
  perforation);
- **fine self-affine roughness** (power-law spectrum) on periosteum, endosteum
  and fractures;
- **fasciae, fat septa, perimysium and periosteum**.

What the grid resolves is drawn explicitly; the rest is kept as a pore
fraction per cell, with the same mean density. Each physics uses its own
mixing rule: linear (density, X-rays, optics and MRI), Maxwell Garnett
(dielectric) and Hashin-Shtrikman (elasticity). By default (`realism=None`)
the phantom is the usual one; the datasets use the realistic mode.

In `12_realismo.py`:
- **ultrasound**: the fasciae produce echoes in the fat (+17.6 dB) and the
  porous cortex replaces clean reverberations with diffuse speckle;
- **CT**: in the elderly person the cortex drops by ~300 HU and its standard
  deviation goes from 147 to 694 HU;
- **MRI**: the porous cortex of the elderly person brightens in T1 (from 0.18
  to 0.63 relative to muscle).

The fascia and periosteum properties are estimates. Bone is isotropic by
default; the transversely isotropic cortex (Rho 1996 tensor, see
`docs/validacion/anisotropia.md`) is enabled with `anisotropic_bone=True`.

### Electrical impedance (low frequency)

At 50 kHz the current crosses the leg without difficulty, but the resolution
is ~1 cm. In `10_impedancia_eit.py` the fracture haematoma changes some
boundary voltages by up to ~40 % (~13 % of the maximum voltage), and the
one-step reconstruction locates it with an error of ~6 mm (7.2 mm in the
validation; Gauss-Newton with a Laplacian prior brings it down to 2-4 mm, with
a single noise realisation). It can detect and roughly locate, not see the
crack. The complete electrode model (`contact_impedance`) includes the contact
impedance, but with the phantom's skin the contact barely changes the image;
its typical value is unverified. See `docs/validacion/eit.md`.

## Layout

```
wavesim/
  core/         2D/3D grids, CPML/M-PML absorbing boundaries, signals
  materials/    properties per tissue and per physics (with their sources)
  phantoms/     limb generators, roughness, fractures, 3D tibia
  ultrasound/   2D/3D elastic solvers, GSLS attenuation, probe, STA, FMC,
                beamformers, image quality metrics
  em/           Maxwell FDTD (narrow- and broadband), EIT (CEM, Gauss-Newton,
                multigrid), optics (diffusion and Monte Carlo), X-rays
                (spectrum, fan beam, SART, scatter, detector), spectrum map
  mri/          analytic sequences, spatial Bloch, susceptibility
  viz/          figures and animations
  datasets/     labelled data generation
  analysis.py   fracture visibility metrics
examples/       demonstration scripts
benchmarks/     quantitative and performance validation (writes docs/validacion/)
tests/          physical validation (pytest)
docs/           comparison with other simulators and validation results
```

## Validation

`pytest` (467 tests, about 4 min with 4 Numba threads) compares each solver
with analytic solutions and checks the phantoms and datasets. The scripts in
`benchmarks/` repeat the comparisons at higher resolution and write their
tables to [`docs/validacion/`](docs/validacion/) (one document per topic, with
the measured figures, the references that could not be verified and the
risks). Summary of what was measured:

- **Elastic ultrasound** ([elastico](docs/validacion/elastico.md)): phase
  velocity equal to the scheme's own (1e-6); second-order convergence in R
  and T; water-cortex |R(θ)| ≤ 0.7 % up to 22° and critical angle within
  ±0.15°; S0/A0 Lamb waves within ≤ 2 % of Rayleigh-Lamb (free plate and in
  water). Out of tolerance: |R| of 4-5 % at the 28-53° and 59-70° edges and
  9-17 % near the critical angles (a limitation of the finite set-up; it does
  not change with the grid).
- **Attenuation** ([atenuacion](docs/validacion/atenuacion.md)): error against
  the α₀·f^y law ≤ 2 % in [0.5, 1.8]·f₀ with 2 mechanisms (y from 1 to 2);
  14 % to 97 % with 1 mechanism. Cost of 2 mechanisms: ×1.5.
- **Anisotropy** ([anisotropia](docs/validacion/anisotropia.md)): qP and qSV
  velocities against Christoffel within ≤ 0.1 %.
- **Stability** ([estabilidad](docs/validacion/estabilidad.md)): a cortex
  crossing the CPML no longer grows over 40 000 steps (automatic M-PML); the
  price is a −28 dB (amplitude) reflection in bone.
- **3D** ([elastico3d](docs/validacion/elastico3d.md)): velocities ≤ 0.4 %,
  equivalence with extruded 2D to ~1e-6 and oblique reflection within 0.8 %.
- **Ultrasound imaging** ([imagen_ultrasonido](docs/validacion/imagen_ultrasonido.md)):
  lateral FWHM within 12 % of theory; axial FWHM of a point scatterer
  **+9 to +17 %** (misses the 10 % target); speckle SNR 1.85 vs 1.91.
- **Maxwell** ([maxwell](docs/validacion/maxwell.md)): Debye fit ≤ 0.7 %
  against Gabriel (0.1-10 GHz); β and α in muscle and fat within ≤ 0.23 % and
  ≤ 1.6 %; multilayer slab within ≤ 0.11 dB (narrowband) and ≤ 0.31 dB
  (broadband, dx 0.125 mm); cylinder against 2D Mie within 1.7-3.3 % with
  15-20 points per wavelength and sub-cell averaging.
- **Optics** ([optica](docs/validacion/optica.md)): Monte Carlo against van de
  Hulst (R and T within ≤ 1.6 σ), exact Fresnel (1e-7) and ballistic
  attenuation (≤ 1.2 σ); against diffusion, ±1.3 % in μ_eff and 27 % more
  fluence in the illuminated skin.
- **X-rays** ([rayos_x](docs/validacion/rayos_x.md)): fan beam equal to
  parallel beam in FBP; OS-SART beats FBP by 25-32 % with 30-60 views;
  spectrum against SpekPy within ±5 % from 80 to 120 kVp, but −6.6 % and
  −5.2 % HVL at 50 and 70 kVp.
- **MRI** ([rm](docs/validacion/rm.md)): the spatial Bloch engine reproduces
  the signal equations to 1e-11 or better; chemical shift and the field of a
  cylinder (≤ 0.7 % with B0 aligned with the grid) against the exact solution.
- **EIT** ([eit](docs/validacion/eit.md)): CEM against the Fourier series
  within ≤ 1.5 %; Gauss-Newton locates an inclusion in a disc within
  0.08-0.33 mm (Laplacian, TV, Tikhonov); multigrid takes the solver from 922
  to 13-16 iterations ([rendimiento_em](docs/validacion/rendimiento_em.md)).
- **Materials**: the dielectric model reproduces the Gabriel/IT'IS reference
  values at 1 GHz (< 1 %); the X-ray decomposition reproduces the NIST muscle
  table (< 1.5 %).

Besides the analytic solutions, every modality except MRI has been compared
with at least one reference simulator (see the next section).

## Comparison with other simulators

[docs/comparativa_simuladores.md](docs/comparativa_simuladores.md) compares
wavesim with the reference simulators of each modality: k-Wave, SimSonic,
Field II, openEMS, Meep, gprMax, MCX, GATE, ASTRA, KomaMRI, JEMRIS, EIDORS and
the XCAT/Virtual Population phantoms. It explains what each one does better,
when to use another tool instead of wavesim and how to validate against them,
with a table of closed and open gaps.

**Direct cross-validation** (each tool run here on the CPU, in a separate
environment; they are not project dependencies; details in
`docs/validacion/cruzada_*.md`):

| Modality | Reference | Measured agreement |
|---|---|---|
| Ultrasound | k-Wave, Devito | R and T ≤ 0.28 %; attenuation ±1.1 %; water-bone \|R\| at 45° within 0.1-0.7 % of the analytic value |
| Maxwell | gprMax | slab 0.013 dB; leg: field 0.04-0.6 %, SAR 0.13-1.2 % |
| Light | MCML, MCX | totals ≤ 0.15 %; profiles ≤ 2.5 % |
| X-rays | ASTRA, SpekPy, XCIST | projector ≤ 0.4 %; HVL 2.8 % on average; scatter does not agree with XCIST |
| EIT | pyEIT | voltage change 0.5 %; position 0-0.9 mm |

The `tests/test_cross_*.py` tests compare wavesim with stored results from
those tools, without needing to install them.

## Data accuracy and limitations

- **Sources**: the tables were transcribed from the literature (the online
  databases were not reachable from the development environment). The
  dielectric and X-ray data are checked against reference values (see the
  tests). The **optical** data (especially mid-infrared and UV) are order of
  magnitude. The **MRI** data for oedema, haematoma and skin are
  representative. Each module in `materials/` states its source and
  uncertainty. Several figures in the validation documents come from memory
  and are marked as unverified (Cu and W tables, optical g, cortical c13 and
  density, tissue χ, electrode contact impedance).
- **Attenuation**: with 2 mechanisms (default) the error is < 2 % in the pulse
  band (±0.5-1.8 f₀); for very broadband pulses use `n_mechanisms=3`. With
  1 mechanism, the previous default, it reaches ~14 %. With `y = 2` and very
  high losses the error rises to 3.7 % (2 mechanisms).
- **2D**: scattering and diffraction by a 3D object are not identical to those
  of a slice (a point source is an infinite line: amplitude decays as d^-1.9
  instead of d^-3 in axial transmission). The 3D solver handles this, but only
  for small domains (~27 Mcell·step/s per thread), without speckle, porosity
  or roughness, and **with the classic CPML**: an anisotropic cortex entering
  the layer without attenuation is unstable (use `bone_length`). The
  "radiograph" of a 2D slice is a 1D profile; the useful 2D X-ray image is CT.
- **Ultrasound**: linear acoustics (no nonlinearity); soft tissue as a fluid;
  ideal probe except for the element response. The axial FWHM of a point
  scatterer is 9-17 % larger than theory and no beamforming variant clearly
  improves the visibility of the 0.15 mm crack (without electronic noise or an
  observer). The interface correction and the M-PML have costs: the M-PML
  reflects −28 dB (amplitude) in bone. The instability of the realistic
  osteoporotic tibia (its cortex crossed the CPML) is fixed by the M-PML
  ([estabilidad](docs/validacion/estabilidad.md)). Cortical anisotropy is only
  grid-aligned and the c13 value is unverified.
- **Maxwell**: 2D TM, no antennas or real sources; the broadband plane-wave
  source must sit in a homogeneous region; the field error outside the bone
  cylinder stays at ~0.8 % with no identified cause.
- **Optics**: Monte Carlo with a single refractive index per tissue (1.4) and
  the same g for all (0.9): no verified per-tissue g values; staircased curved
  surface; the leg's transmission at 800 nm is 10⁻⁸ in the thickest columns,
  at the limit of what can be measured.
- **X-rays**: the scatter is Monte Carlo with an object infinite in y, without
  fluorescence or Doppler, and there is no scatter correction in the
  reconstruction; no short scan or 3D cone beam; the spectrum's HVL deviates
  from SpekPy (−5 to −7 %) at 50-70 kVp; the two free parameters were fitted
  to the same SpekPy cases.
- **MRI**: 2D (no slice profile, flow or motion), ideal spoiling, Cartesian
  trajectories only, soft-tissue χ ≈ water, 2.6-5.5 % error in the B0 field
  when B0 is oblique to the grid. Not compared with KomaMRI or JEMRIS.
- **EIT**: 2D with finite differences (no FEM or 3D); trained GREIT not
  implemented; the Gauss-Newton results come from a single noise realisation
  (70 dB) and one inclusion size.
- **Performance**: a 3 MHz, 24×22 mm ultrasound image with 3 angles takes
  ~20 s with Numba and 4 threads (2 mechanisms); ~1 min with NumPy. At
  7-10 MHz the grid grows a lot (several minutes or more). No GPU.
- **Partial cross-validation**: done with k-Wave, Devito, gprMax, MCML, MCX,
  ASTRA, SpekPy, XCIST and pyEIT (see above); missing for MRI (KomaMRI,
  JEMRIS), the 3D solver and against SimSonic or Field II. It exposed three
  weak points: the ultrasound waveform differs by 6-8 % from k-Wave at the
  default CFL of 0.9 (0.3-0.9 % at CFL 0.3), the X-ray spectrum's HVL deviates
  by +6-7 % at 140-150 kVp, and the scatter does not agree with XCIST (which is
  not a physical reference). No peer-reviewed validation.

## Possible next steps

- Complete the cross-validation: KomaMRI or JEMRIS for MRI, GATE or MC-GPU for
  scatter, SimSonic or elastic k-Wave for bone.
- Reduce the ultrasound temporal dispersion (lower default CFL or dispersion
  correction) and recalibrate the X-ray spectrum at high kVp.
- GPU (CuPy, JAX or Numba-CUDA) for 3D and for high-frequency B-mode images.
- M-PML in the 3D solver and a material-dependent p in 2D.
- Scatter correction in the reconstruction, short scan and cone beam.
- Radial/spiral trajectories and flow in MRI; GREIT and FEM in EIT; antennas
  and 3D in Maxwell.
- Healing callus and other anatomies; training the neural network (PyTorch)
  on the generated datasets.
