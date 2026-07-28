# Vertical Plastic Profile App

## Citations

### Rouse-profile validation

- Valero, D., Belay, B. S., Moreno-Rodenas, A., Kramer, M. and Franca, M. J. (2022). *Water Research*, 226, 119078. DOI: [10.1016/j.watres.2022.119078](https://doi.org/10.1016/j.watres.2022.119078).
- Lofty, J., Valero, D., Moreno-Rodenas, A., Belay, B. S., Wilson, C., Ouro, P. and Franca, M. J. (2024). *Water Research*, 254, 121306. DOI: [10.1016/j.watres.2024.121306](https://doi.org/10.1016/j.watres.2024.121306).
- Born, M. P., Brüll, C., Schaefer, D., Hillebrand, G. and Schüttrumpf, H. (2023). *Environmental Science & Technology*, 57(14), 5569–5579. DOI: [10.1021/acs.est.2c06885](https://doi.org/10.1021/acs.est.2c06885).

### Buoyant and sinking velocity equations

- Dietrich, W. E. (1982). *Water Resources Research*, 18(6), 1615–1626. DOI: [10.1029/WR018i006p01615](https://doi.org/10.1029/WR018i006p01615).
- Goral, K. D. et al. (2023). *Environmental Research*, 228, 115783. DOI: [10.1016/j.envres.2023.115783](https://doi.org/10.1016/j.envres.2023.115783).
- Yu, Z., Yang, G. and Zhang, W. (2022). *Marine Pollution Bulletin*, 176, 113449. DOI: [10.1016/j.marpolbul.2022.113449](https://doi.org/10.1016/j.marpolbul.2022.113449).
- Lofty, J., Valero, D. and Franca, M. J. (2026). *Settling and Rising Dynamics of River Litter*. EarthArXiv.

## Introduction

The River Plastic Vertical Profiler is a scientific exploration tool for estimating how microplastics and macroplastics may be distributed vertically through a river water column. Its primary workflow evaluates sampling bias: it predicts what fraction of the vertically integrated plastic concentration is captured by a selected sampling layer, then uses that fraction to estimate depth-averaged concentration and, when discharge is supplied, plastic load.

The tool:

1. Generates a reproducible synthetic microplastic population from user-selected particle size, shape, and polymer assumptions.
2. Uses measured macroplastic buoyant and sinking data when grouped classes or individual litter items are selected.
3. Calculates vertical velocities with the Dietrich (1982), Goral et al. (2023), and Yu et al. (2022) equations.
4. Converts vertical velocities to Rouse numbers and estimates direction-aware vertical concentration profiles.
5. Quantifies the captured and missed fractions associated with a selected sampling interval.
6. Corrects a measured concentration and optionally combines it with river discharge to estimate load.
7. Exports generated particles and result tables for checking and further analysis.

### App pages

- **Sampling correction** is the main workflow. Define flow conditions, plastics, sampling interval, measured concentration, and optional discharge; then inspect and export the predicted correction.
- **Buoyant and sinking velocities** compares the velocity distributions predicted by Dietrich, Goral, and Yu.
- **About & Methods** contains the scientific basis, equations, assumptions, limitations, and interpretation guidance.

## **1. Purpose**

This application estimates how plastics may be distributed vertically within a river water column using a modified Rouse-profile approach.

The app is designed to help users:

- **Explore** likely vertical concentration profiles
- **Compare** different particle types
- **Investigate** the effect of particle size, density, shape, and polymer composition
- **Estimate** the fraction of plastics captured within a chosen sampling interval
- **Estimate** depth-averaged concentrations
- **Estimate** river plastic loads when discharge is known

The app is intended as a first-order scientific tool rather than an exact prediction of real rivers. Results should always be interpreted alongside field observations and professional judgement.

---

## **2. Synthetic microplastic generation**

Microplastic particles are generated synthetically rather than being read directly from a measured microplastic dataset.

Each synthetic particle is assigned:

- **Particle size**
- **Particle shape**
- **Polymer type**
- **Polymer density**

These properties are then used to estimate buoyant or sinking velocity.

The synthetic population represents a statistical sample of particles that satisfy the user-defined inputs. It should not be interpreted as an exact representation of the particles present within a real river.

### **2.1 Particle size**

Users define minimum and maximum particle sizes.

Three size distributions are available.

**Log-uniform**

Each order of magnitude contains approximately equal numbers of particles. This is useful when particle sizes span a wide range and no single characteristic size is known.

**Uniform**

Every particle size between the minimum and maximum values has an equal probability of occurring.

**Truncated lognormal**

Most particles cluster around a characteristic size, while fewer particles occur at very small and very large sizes. The distribution is truncated so that all generated particles remain inside the selected size limits.

### **2.2 Particle shape**

Particles are assigned as either **fibres** or **fragments** using the percentages selected by the user.

For example:

- **Fibres = 70%**
- **Fragments = 30%**

means approximately 70% of generated particles are fibres and 30% are fragments.

The sliders are constrained so the total is always **100%**.

### **2.3 Polymer composition**

The polymer sliders specify the composition of the synthetic population.

For example:

- **PE = 40%**
- **PET = 30%**
- **PP = 30%**

means approximately 40% polyethylene, 30% polyethylene terephthalate, and 30% polypropylene.

Each polymer has an accepted density range. For every generated particle, density is randomly sampled from the density range of its assigned polymer before velocity calculations begin.

---

## **3. Buoyant and sinking velocity calculations**

The app estimates buoyant or sinking velocity for every synthetic microplastic particle using three published models:

- **Dietrich (1982)**
- **Goral et al. (2023)**
- **Yu et al. (2022)**

Each equation predicts terminal vertical velocity from particle properties.

The app calculates all three predictions independently. The mean predicted vertical velocity is then used to calculate the particle Rouse number.

**Positive velocity** means the particle sinks towards the bed.

**Negative velocity** means the particle is buoyant and moves towards the surface.

---

## **4. Macroplastic data**

Macroplastics are not generated synthetically.

Instead, macroplastic particles are taken from the supplied macroplastic dataset.

Users may select:

- **Grouped material classes**
- **Individual litter items**

The grouped classes are:

- **Foams (very buoyant):** EPS items
- **Plastics (buoyant):** hard and soft polyolefins, plus multilayer items with negative measured vertical velocity
- **Plastics (settling):** PET and PS, plus multilayer items with positive measured vertical velocity
- **Glass & metal (strongly settling):** glass and metal items

Paper and textile records are not labelled as plastics in these grouped
classes. They remain available through individual litter-item selection.
Density ranges are not used in the displayed class labels.

For grouped material classes, the app combines the selected records as
**Total** and also reports the selected classes separately.

For individual litter items, the user enters an already calculated measured
concentration for every selected item, in particles/m³ or g/m³. The app
corrects each item independently using that item's vertical-velocity records.
This is appropriate when, for example, bags, bottles, and film fragments have
different measured concentrations and different vertical behaviour.

Measured vertical velocities are converted into Rouse numbers using the same Rouse-number calculation used for synthetic microplastics.

---

## **5. Shear velocity**

Users may specify shear velocity directly or calculate it from hydraulic radius and slope. These inputs are grouped as **Flow conditions** throughout the app.

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
u* = √(g R S)
</div>

where:

- **u\*** = shear velocity
- **g** = gravitational acceleration
- **R** = hydraulic radius
- **S** = slope

For wide channels, hydraulic radius is commonly approximated by flow depth.

---

## **6. Rouse number**

The Rouse number describes the balance between particle sinking or buoyant motion and turbulent vertical mixing.

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
β = w / (κ u*)
</div>

where:

- **β** = Rouse number
- **w** = buoyant or sinking velocity
- **κ** = von Kármán constant, taken as 0.41
- **u\*** = shear velocity

Interpretation:

- **β > 0**: particles tend to sink towards the bed
- **β ≈ 0**: particles are approximately well mixed
- **β < 0**: particles are buoyant and tend to move towards the surface

Large positive values produce profiles concentrated near the bed. Negative values produce profiles concentrated near the water surface.

---

## **7. Vertical concentration profiles**

The app uses a direction-aware Rouse profile.

Sinking particles and buoyant particles are treated differently because they concentrate near opposite boundaries of the water column.

### **7.1 Sinking particles**

For sinking particles, where **β ≥ 0**, the app uses a bed-referenced Rouse profile.

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
C(z) / C(a<sub>bed</sub>) = [ ((H − z) / z) / ((H − a<sub>bed</sub>) / a<sub>bed</sub>) ]<sup>β</sup>
</div>

where:

- **C(z)** = concentration at height **z**
- **C(a<sub>bed</sub>)** = concentration at the bed reference height
- **H** = flow depth
- **z** = height above the bed
- **a<sub>bed</sub>** = reference height above the bed
- **β** = Rouse number

This profile gives high concentration near the bed and lower concentration towards the surface.

### **7.2 Buoyant particles**

For buoyant particles, where **β < 0**, the app uses a mirrored surface-referenced profile.

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
C(z) / C(H − a<sub>surf</sub>) = [ (z / (H − z)) / ((H − a<sub>surf</sub>) / a<sub>surf</sub>) ]<sup>|β|</sup>
</div>

where:

- **a<sub>surf</sub>** = reference distance below the water surface
- **H − a<sub>surf</sub>** = surface-referenced concentration point
- **|β|** = absolute value of the Rouse number

This profile gives low concentration near the bed and high concentration near the water surface.

### **7.3 Reference levels**

The app uses two reference offsets:

- **a<sub>bed</sub>/H** for sinking particles
- **a<sub>surf</sub>/H** for buoyant particles

These avoid calculating the profile exactly at the bed or surface, where the equations become singular.

---

## **8. Profile normalisation**

Each profile is normalised by its maximum predicted concentration.

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
C<sub>norm</sub>(z) = C(z) / C<sub>max</sub>
</div>

This means the plotted profile ranges from 0 to 1.

The plotted profiles therefore show the **relative shape** of the vertical distribution, not the absolute concentration.

---

## **9. Uncertainty and percentile bands**

Each selected particle group contains many particles.

The app calculates one profile per particle and then summarises the group.

The plotted line is the **median profile**.

The shaded band is the selected percentile range, by default:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
P25 to P75
</div>

This shows how much profile shape varies across the generated or selected particle population.

The variability comes from differences in:

- Particle size
- Shape
- Polymer density
- Buoyant or sinking velocity
- Rouse number
- Macroplastic material or item type

On the Sampling correction page, synthetic microplastics are automatically
separated into **buoyant** and **sinking** groups on the figure when both
behaviours are present. The captured-fraction table reports
**Microplastics: total**, **Microplastics: buoyant**, and
**Microplastics: settling**. This makes the opposing vertical distributions
easier to see without requiring any additional user inputs. The highlighted
numerical correction, corrected-concentration table, and load table use only
the total microplastic population. The table's median-captured value matches
the highlighted median capture. Capture, missed fraction, corrected
concentration, and load are displayed as median [P25–P75].

---

## **10. Sampling-interval estimation**

The sampling tool estimates how much of the vertically distributed plastic population lies within a selected sampling interval. The interval is expressed as **z/H**, where 0 is the river bed and 1 is the water surface.

On the Sampling correction page, users choose either **Microplastics** or
**Macroplastics**. The graph, highlights, correction tables, load estimate,
and downloads use only the selected plastic type; microplastic and
macroplastic results are not combined. Flow edits are applied together when
the user clicks **Apply flow values**.

The interval may represent:

- Nets
- Pumps
- Bottles
- Integrated samplers
- Any other method that samples only part of the water column

The selected interval is shown on the profile plot using dashed lines and a shaded band.

---

## **11. Sampling correction**

The app compares the vertically integrated concentration in the selected sampling interval with the vertically integrated concentration over the full modelled profile.

The captured fraction is:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
F<sub>capture</sub> = sampled integrated concentration / total integrated concentration
</div>

More explicitly:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
F<sub>capture</sub> = ∫<sub>z1</sub><sup>z2</sup> C(z) dz / ∫<sub>total</sub> C(z) dz
</div>

where:

- **z1** = lower boundary of the sampling interval
- **z2** = upper boundary of the sampling interval
- **C(z)** = predicted concentration profile

The app calculates this captured fraction separately for every modelled
particle. For the sampling correction, it then takes the median:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
F<sub>median</sub> = median(F<sub>capture,1</sub>, F<sub>capture,2</sub>, ..., F<sub>capture,N</sub>)
</div>

The central correction is calculated from **F<sub>median</sub>**. To show
particle-to-particle variability, the app also calculates the correction,
corrected concentration, and load for every valid particle and reports each
result as median [P25–P75]. Using the median consistently makes the plotted
central profile, highlighted capture, tables, concentration correction, and
load describe the central particle response.

**P25–P75 describes variability among the modelled particles. It is not a
confidence interval and does not represent uncertainty in the field
measurement.**

The missed fraction is:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
F<sub>missed</sub> = 1 − F<sub>capture</sub>
</div>

The correction factor is:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
Correction factor = f<sub>depth sampled</sub> / F<sub>median</sub>
</div>

where **f<sub>depth sampled</sub> = |z2/H − z1/H|**.

The corrected depth-averaged concentration is:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
C<sub>depth-avg</sub> = C<sub>measured</sub> × f<sub>depth sampled</sub> / F<sub>median</sub>
</div>

For individually selected macroplastic items, this correction is applied to
each item separately:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
C<sub>depth-avg,j</sub> = C<sub>measured,j</sub> × f<sub>depth sampled</sub> / F<sub>j</sub>
</div>

The total macroplastic concentration is then the sum of the separately
corrected item concentrations:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
C<sub>macro,total</sub> = Σ C<sub>depth-avg,j</sub>
</div>

The total P25–P75 interval is estimated by repeatedly drawing one valid
correction result from each selected item and summing those draws. Therefore,
an item with a measured concentration of zero contributes zero to the total,
while more abundant items contribute proportionally more.

Example:

- **Measured concentration = 15 particles m⁻³**
- **Sampled depth fraction = 0.20**
- **Median captured fraction = 0.22**
- **Correction factor = 0.20 / 0.22 = 0.91**
- **Estimated depth-averaged concentration = 13.6 particles m⁻³**

The correction becomes very sensitive when **F<sub>median</sub>** is small.

Very small captured fractions should therefore be interpreted cautiously.

---

## **12. Plastic load estimation**

If river discharge is supplied, the app estimates plastic transport rate.

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
Load = C<sub>depth-avg</sub> × Q
</div>

where:

- **Load** = plastic transport rate
- **C<sub>depth-avg</sub>** = corrected depth-averaged concentration
- **Q** = river discharge

If concentration is in **particles m⁻³** and discharge is in **m³ s⁻¹**, then load is reported as **particles s⁻¹**.

If concentration is in mass units, load is reported as **mg s⁻¹** or **g s⁻¹**.

---

## **13. Assumptions**

The method assumes:

- Flow is approximately steady
- Turbulent vertical mixing is represented by shear velocity
- Buoyant or sinking velocities are representative
- The river is reasonably laterally mixed
- The vertical profile is time averaged
- Rouse theory is applicable to the selected particle classes

---

## **14. Limitations**

The method is less reliable for:

- Very buoyant particles strongly affected by surface tension
- Very large macroplastic items
- Very thin sampling intervals
- Strongly unsteady flow
- Strongly stratified flow
- Particles affected by vegetation, wind, biofouling, aggregation, or surface films

Macroplastic profiles should be interpreted carefully. For large items, the profile may be better understood as a probability of occurrence with depth rather than a smooth concentration field.

---

## **15. Recommended interpretation**

A careful way to report results is:

> Based on the predicted Rouse profile, the selected sampling interval is estimated to contain X percent of the vertically integrated concentration for this particle class. The measured concentration was therefore corrected using a factor of Y to estimate a depth-averaged water-column concentration.

Avoid saying:

> The app proves the true river concentration is X.

Better wording is:

> The app estimates the depth-averaged concentration as X, conditional on the assumed Rouse profile and input parameters.
