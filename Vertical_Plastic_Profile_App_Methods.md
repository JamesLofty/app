# Vertical Plastic Profile App

## **1. Purpose**

This application estimates how plastics may be distributed vertically within a river water column using a modified Rouse-profile approach.

The app is designed to help users:

- **Explore** likely vertical concentration profiles
- **Compare** different particle types
- **Investigate** the effect of particle size, density, shape, and polymer composition
- **Estimate** the fraction of plastics captured within a chosen sampling depth
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

These properties are then used to estimate settling or rising velocity.

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

## **3. Settling and rising velocity calculations**

The app estimates settling or rising velocity for every synthetic microplastic particle using three published models:

- **Dietrich (1982)**
- **Goral et al. (2023)**
- **Yu et al. (2022)**

Each equation predicts terminal vertical velocity from particle properties.

The app calculates all three predictions independently. The mean predicted vertical velocity is then used to calculate the particle Rouse number.

**Positive velocity** means the particle settles towards the bed.

**Negative velocity** means the particle rises towards the surface.

---

## **4. Macroplastic data**

Macroplastics are not generated synthetically.

Instead, macroplastic particles are taken from the supplied macroplastic dataset.

Users may select:

- **Grouped material classes**
- **Individual litter items**

Measured vertical velocities are converted into Rouse numbers using the same Rouse-number calculation used for synthetic microplastics.

---

## **5. Shear velocity**

Users may specify shear velocity directly or calculate it from hydraulic radius and energy slope.

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
u* = √(g R S)
</div>

where:

- **u\*** = shear velocity
- **g** = gravitational acceleration
- **R** = hydraulic radius
- **S** = energy slope

For wide channels, hydraulic radius is commonly approximated by flow depth.

---

## **6. Rouse number**

The Rouse number describes the balance between particle settling or rising and turbulent vertical mixing.

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
β = w / (κ u*)
</div>

where:

- **β** = Rouse number
- **w** = settling or rising velocity
- **κ** = von Kármán constant, taken as 0.41
- **u\*** = shear velocity

Interpretation:

- **β > 0**: particles tend to settle towards the bed
- **β ≈ 0**: particles are approximately well mixed
- **β < 0**: particles tend to rise towards the surface

Large positive values produce profiles concentrated near the bed. Negative values produce profiles concentrated near the water surface.

---

## **7. Vertical concentration profiles**

The app uses a direction-aware Rouse profile.

Settling particles and buoyant particles are treated differently because they concentrate near opposite boundaries of the water column.

### **7.1 Settling particles**

For settling particles, where **β ≥ 0**, the app uses a bed-referenced Rouse profile.

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

- **a<sub>bed</sub>/H** for settling particles
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
- Settling or rising velocity
- Rouse number
- Macroplastic material or item type

---

## **10. Sampling depth estimation**

The sampling depth tool estimates how much of the vertically distributed plastic population lies within a selected sampling interval.

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

The missed fraction is:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
F<sub>missed</sub> = 1 − F<sub>capture</sub>
</div>

The correction factor is:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
Correction factor = 1 / F<sub>capture</sub>
</div>

The corrected depth-averaged concentration is:

<div style="text-align:center; font-weight:600; margin:0.6rem 0;">
C<sub>depth-avg</sub> = C<sub>measured</sub> / F<sub>capture</sub>
</div>

Example:

- **Measured concentration = 15 particles m⁻³**
- **Captured fraction = 0.22**
- **Correction factor = 4.55**
- **Estimated depth-averaged concentration = 68 particles m⁻³**

The correction becomes very sensitive when **F<sub>capture</sub>** is small.

For example:

- **F<sub>capture</sub> = 0.50** gives correction factor **2**
- **F<sub>capture</sub> = 0.25** gives correction factor **4**
- **F<sub>capture</sub> = 0.10** gives correction factor **10**
- **F<sub>capture</sub> = 0.05** gives correction factor **20**

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
- Settling or rising velocities are representative
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