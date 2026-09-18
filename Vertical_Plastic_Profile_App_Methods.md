## Introduction

RIVER-PLAST is a tool for estimating depth-average concentrations and loads of suspended microplastics and macroplastics from a single depth sample.

The tool uses the Rouse profile to calculate the proportion of microplastics or macroplastics captured and missed from a collected sample and corrects the measured concentration to a depth-average concentration. If river discharge is provided, the tool calculates total plastic load.

## Methods

### Background

RIVER-PLAST adapts the Rouse profile, a model originally developed to describe the vertical distribution of suspended sediment in turbulent flow (Rouse 1937). For settling particles, the profile is:

<div class="equation"><span class="fraction"><span><i>C</i>(<i>z</i>)</span><span><i>C</i>(<i>a</i><sub>bed</sub>)</span></span> = [<span class="fraction"><span>(<i>H</i>−<i>z</i>)/<i>z</i></span><span>(<i>H</i>−<i>a</i><sub>bed</sub>)/<i>a</i><sub>bed</sub></span></span>]<sup>β</sup></div>

where *C(z)* is concentration at height *z* above the bed, *H* is water depth, and <i>a</i><sub>bed</sub> is a near-bed reference height. For buoyant particles, the app uses a mirrored surface-referenced profile, using a reference depth from the surface (<i>a</i><sub>surf</sub>), so that concentration increases towards the surface.

a<sub>bed</sub> and a<sub>surf</sub> define the lower and upper limits of the modelled suspended water column where the Rouse profile is accplicable, excluding the near-bed and near-surface zones, respectively. Both are set to 0.05 of water depth by default, as practical Rouse-profile reference offsets based on classical sediment-transport approximations. Users can change a<sub>bed</sub> and a<sub>surf</sub> in the Advanced tab.

The Rouse number β determines the shape of the predicted profile, which represents the balance between vertical particle motion, driven by gravity and buoyancy, and turbulent mixing, which scales with the shear velocity u*:

<div class="equation">β = <span class="fraction"><span><i>w</i></span><span>κ<i>u</i><sub>*</sub></span></span></div>

where *w* is particle settling or rising velocity, κ = 0.41 is the von Kármán constant

For negatively buoyant (sinking) plastics, the settling velocity is taken as a positive value, yielding β > 0. For positively buoyant (rising) plastics, the rising velocity is taken as a negative value, yielding β < 0.

Larger positive β values mean that particles are concentrated towards the bed. Larger negative β values mean that particles are concentrated towards the surface. Values of β close to zero indicate a uniform concentration through the water column.

### Flow conditions

Users enter river width (*w*), average depth (*d*), slope (*S*), and discharge (*Q*). The app estimates hydraulic radius for a rectangular channel as:

<div class="equation"><i>R</i> = <span class="fraction"><span><i>w d</i></span><span><i>w</i> + 2<i>d</i></span></span></div>

Shear velocity is calculated as:

<div class="equation"><i>u</i><sub>*</sub> = √(<i>g R S</i>)</div>

where *g* is gravitational acceleration and *S* is slope. Users can instead enter a direct shear velocity, which overrides the calculated value.

### Particle vertical velocity

A population of 5,000 microplastics is generated from the selected size range, fibre/fragment proportion, and polymer mixture. Polymer density is sampled from the app's predefined density ranges based on Kooi and Koelmans (2019). Terminal vertical velocity is calculated for every generated particle using the Dietrich (1982), Goral et al. (2023), and Yu et al. (2022) equations. The mean of the three predictions is used to calculate the Rouse number.

Macroplastic mode uses measured vertical velocities from the supplied macroplastic dataset, provided by Lofty et al. (2026). Users can select individual litter items or grouped material classes. Each velocity record within a selected item or class has equal weight in the profile calculation.

### Modelled profiles

The plotted profiles are normalised to their own maximum concentration, so they show predicted vertical shape rather than absolute concentration at each depth. The central line is the median profile and the shaded range shows the selected percentile range across modelled microplastic particles or macroplastic records.

### Sampling correction and load

The captured fraction is the integrated modelled concentration within the selected relative sampling depth, divided by the integrated concentration over the full modelled water column:

<div class="equation"><i>F</i><sub>capture</sub> = <span class="fraction"><span>∫<sub>sample</sub> <i>C</i>(<i>z</i>) d<i>z</i></span><span>∫<sub>water column</sub> <i>C</i>(<i>z</i>) d<i>z</i></span></span></div>

The app applies the correction to each particle, then reports the median and selected percentile range. The depth-average concentration is:

<div class="equation"><i>C</i><sub>depth-average</sub> = <i>C</i><sub>measured</sub> <span class="fraction"><span><i>f</i><sub>sampled depth</sub></span><span><i>F</i><sub>capture</sub></span></span></div>

For individually selected macroplastic items, the correction is calculated separately for each item and then summed. The app withholds an estimated concentration or load where the median captured fraction is below 5%. When discharge is supplied, load is calculated as:

<div class="equation">Load = <i>C</i><sub>depth-average</sub> <i>Q</i></div>

Percentile ranges describe variation among the modelled particles or selected records. They are not confidence intervals for the field measurement.

### Interpretation and limitations

RIVER-PLAST provides a first-order estimate conditional on the selected particle population, flow conditions, and Rouse-profile assumptions. Results are most useful for approximately steady, laterally mixed flow and freely suspended particles.

Biofouling, surface tension effects, wind, vegetation, aggregation, and surface films are not represented. Interpret results cautiously for very buoyant particles, large macroplastic items, thin sampling layers, and unsteady or stratified flow. 

The model assumes that the sampled particle class is sufficiently available throughout the water column. For example, finding one plastic item at a given depth does not necessarily mean that many items occur below it, because the supply of individual macroplastic items may be limited.

## References

- Dietrich, W. E. (1982). *Water Resources Research*, 18(6), 1615–1626. DOI: [10.1029/WR018i006p01615](https://doi.org/10.1029/WR018i006p01615).
- Goral, K. D. et al. (2023). *Environmental Research*, 228, 115783. DOI: [10.1016/j.envres.2023.115783](https://doi.org/10.1016/j.envres.2023.115783).
- Kooi, M. and Koelmans, A. A. (2019). Simplifying microplastic via continuous probability distributions for size, shape, and density. *Environmental Science & Technology Letters*, 6(9), 551–557. DOI: [10.1021/acs.estlett.9b00379](https://doi.org/10.1021/acs.estlett.9b00379).
- Lofty, J., Valero, D., Moreno-Rodenas, A., Belay, B. S., Wilson, C., Ouro, P. and Franca, M. J. (2024). *Water Research*, 254, 121306. DOI: [10.1016/j.watres.2024.121306](https://doi.org/10.1016/j.watres.2024.121306).
- Lofty, J., Valero, D. and Franca, M. J. (2026). *Settling and Rising Dynamics of River Litter*. EarthArXiv.
- Rouse, H. (1937). *Modern Conceptions of the Mechanics of Fluid Turbulence*. *Transactions of the American Society of Civil Engineers*, 102(1), 463–505. DOI: [10.1061/TACEAT.0004872](https://ascelibrary.org/doi/10.1061/TACEAT.0004872).
- Valero, D., Belay, B. S., Moreno-Rodenas, A., Kramer, M. and Franca, M. J. (2022). *Water Research*, 226, 119078. DOI: [10.1016/j.watres.2022.119078](https://doi.org/10.1016/j.watres.2022.119078).
- Yu, Z., Yang, G. and Zhang, W. (2022). *Marine Pollution Bulletin*, 176, 113449. DOI: [10.1016/j.marpolbul.2022.113449](https://doi.org/10.1016/j.marpolbul.2022.113449).
