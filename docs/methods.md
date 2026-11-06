## About RIVER-PLAST

RIVER-PLAST is a tool to estimate where microplastics and macrolitter (including plastic and non-plastic items) are distributed through the water column and to convert concentrations measured at single or multiple depths into concentrations representative of the whole water column and plastic load.

The tool has two workflows:

**Sample design** supports field work planning. It predicts vertical profiles for selected flow conditions and plastic populations, allowing users to compare proposed sampling depths and the fractions expected to be captured or missed.

**Sample correction** supports analysis after field work. Users upload river flow, sampling-depth, sampled-volume, and plastic-count data. The tool estimates captured (or missed) fractions, depth-average concentrations, and plastic load for each sample.

RIVER-PLAST provides first-order estimates of vertical concentration profiles for suspended plastics. It assumes particles are in equilibrium with the supplied flow conditions and does not represent biofouling, surface tension effects, wind, vegetation, aggregation, or spatial variation in particle supply.

## Methods

### Vertical distribution model

RIVER-PLAST adapts the Rouse profile, originally developed to describe the vertical distribution of suspended sediment in turbulent flow. For sinking particles, the concentration profile is:

<div class="equation"><span class="fraction"><span><i>C</i>(<i>z</i>)</span><span><i>C</i>(<i>a</i><sub>bed</sub>)</span></span> = [<span class="fraction"><span>(<i>H</i>−<i>z</i>)/<i>z</i></span><span>(<i>H</i>−<i>a</i><sub>bed</sub>)/<i>a</i><sub>bed</sub></span></span>]<sup>β</sup></div>

where *C(z)* is concentration at height *z* above the bed, *H* is water depth, and <i>a</i><sub>bed</sub> is a near-bed reference height. For buoyant particles, RIVER-PLAST uses a mirrored surface-referenced profile, with reference depth <i>a</i><sub>surf</sub>, so concentration increases towards the surface.

<i>a</i><sub>bed</sub> and <i>a</i><sub>surf</sub> define the lower and upper limits of the modelled suspended water column. They exclude the near-bed and near-surface zones, where the Rouse profile is not applicable. Both are set to 0.05 of water depth by default and can be changed in the Advanced tab.

The Rouse number, β, governs the shape of the predicted profile:

<div class="equation">β = <span class="fraction"><span><i>w</i></span><span>κ<i>u</i><sub>*</sub></span></span></div>

where *w* is settling or rising velocity, κ = 0.41 is the von Kármán constant, and <i>u</i><sub>*</sub> is shear velocity. 

A β value close to 0 produces a more uniform vertical profile. As β departs from 0, plastics become increasingly concentrated towards either the riverbed or surface, depending on their buoyancy. Sinking plastics have β > 0, whereas rising plastics have β < 0.

### Flow conditions

Users enter river width (*W*), average depth (*H*), slope (*S*), and discharge (*Q*). The app estimates hydraulic radius for a rectangular channel as:

<div class="equation"><i>R</i> = <span class="fraction"><span><i>W H</i></span><span><i>W</i> + 2<i>H</i></span></span></div>

Shear velocity is calculated as:

<div class="equation"><i>u</i><sub>*</sub> = √(<i>g R S</i>)</div>

where *g* is gravitational acceleration. A directly entered shear velocity overrides the calculated value.

### Microplastics

Particle size, shape, and polymer type are used to estimate settling or rising velocity. Together with the selected flow conditions, this velocity is used to calculate the Rouse number and predict each particle’s vertical distribution through the water column. The predicted profiles are then used to estimate the fraction captured or missed by the selected sampling depth interval.

RIVER-PLAST generates a population of 5,000 microplastics from the selected size range, fibre/fragment proportion, and polymer mixture. 

The typical microplastic population uses a log-uniform 20–5,000 µm size range, a 50:50 fibre-to-fragment ratio, and the default polymer mixture. It is based on distribution data from [Kooi et al. (2019)](https://doi.org/10.1021/acs.estlett.9b00379).

Polymer densities are based on the [MaTCH web tool](https://hannahhapich.shinyapps.io/match/) ([Hapich et al., 2024](https://doi.org/10.1021/acs.est.4c02406)). Terminal vertical velocity is calculated for each generated particle using the equations of [Dietrich (1982)](https://doi.org/10.1029/WR018i006p01615), [Goral et al. (2023)](https://doi.org/10.1016/j.envres.2023.115783), and [Yu et al. (2022)](https://doi.org/10.1016/j.marpolbul.2022.113449). The mean of the three velocity estimates is used to calculate the Rouse number.

### Macrolitter

The macrolitter options represent the 25 most common River-OSPAR litter items observed in rivers, accounting for approximately 80% of total litter ([Lofty et al., 2026](https://doi.org/10.1021/acs.est.5c13126)). The top-five option provides a quick starting selection for macrolitter. Individual items and grouped litter classes can also be selected.

Settling and rising velocity data are taken from [Lofty et al. (2026)](https://doi.org/10.31223/X5TF4V). These velocities, together with the selected flow conditions, are used to predict each item’s vertical distribution and the fraction captured or missed by the selected sampling depth interval. Each velocity record within a selected item or class has equal weight in the profile calculation.

### Predicted profiles

Each plotted line is the median predicted profile across the selected microplastic particles or macrolitter velocity records.

The plotted profiles are normalised to their own maximum concentration. They therefore show predicted vertical shape, rather than absolute concentration at each depth. On the x-axis, 1 is the maximum predicted concentration, 0.5 is half of that maximum, and 0.1 is 10% of that maximum.

### Sampling fractions

For each particle or macrolitter velocity record, the captured fraction is the integrated modelled concentration within the chosen sampled depth interval, divided by the integrated concentration over the full modelled water column:

<div class="equation"><i>F</i><sub>capture</sub> = <span class="fraction"><span>∫<sub>sample</sub> <i>C</i>(<i>z</i>) d<i>z</i></span><span>∫<sub>water column</sub> <i>C</i>(<i>z</i>) d<i>z</i></span></span></div>

The missed fraction is 1 − <i>F</i><sub>capture</sub>. Where more than one sampling depth is used, RIVER-PLAST integrates each interval separately. Gaps between intervals are not treated as sampled, and overlapping intervals are counted once.

### Concentration correction and load

For a **single sampled depth**, the estimated depth-average concentration is:

<div class="equation"><i>C</i><sub>depth-average</sub> = <i>C</i><sub>measured</sub> <span class="fraction"><span><i>f</i><sub>sampled depth</sub></span><span><i>F</i><sub>capture</sub></span></span></div>

where <i>f</i><sub>sampled depth</sub> is the fraction of the water depth covered by the sample. This compares the fraction of the water column sampled with the modelled fraction of the plastic population captured.

For example, if you sampled 20% of the depth (*f* = 0.2) but captured 60% of the plastics (*F* = 0.6), your measured concentration is higher than the whole-water-column average, so the app reduces it.

If you sampled 20% of the depth but captured only 5% of the plastics, the app increases the measured concentration because most plastics were predicted to be outside the sampled layer.

For **multiple-depth sampling**, concentrations are corrected using a volume-weighted depth-stratified calculation. For each sampling interval *i*, the app uses its sampled volume *V*<sub>i</sub>, sampled-depth fraction *f*<sub>i</sub>, and modelled capture fraction *F*<sub>i</sub>:

<div class="equation"><i>C</i><sub>depth-average</sub> = <span class="fraction"><span>Σ<i>N</i><sub>i</sub></span><span>Σ[<i>V</i><sub>i</sub> <i>F</i><sub>i</sub>/<i>f</i><sub>i</sub>]</span></span></div>

where *N*<sub>i</sub> is the count measured in interval *i*.

For example, if two samples were taken simultaneously between 20% and 40%, and 70% and 90%, of the water column, each sampling 1 m³ of water:

Near the surface, at 0.7–0.9: *f* = 0.2, *F* = 0.6, and 30 particles in 1 m³.

Near the bed, at 0.2–0.4: *f* = 0.2, *F* = 0.1, and 5 particles in 1 m³.

The app calculates a profile-adjusted effective sampled volume: (1 × 0.6 / 0.2) + (1 × 0.1 / 0.2) = 3.5 m³

It then divides the total count by this effective volume: 35 particles / 3.5 m³ = 10 particles/m³

The estimated depth-average concentration is therefore 10 particles/m³. This approach uses each depth sample separately, rather than treating the full range between the shallowest and deepest samples as continuously sampled.

The correction is calculated for each modelled particle or macrolitter velocity record. RIVER-PLAST reports the median and 25th–75th percentile range. The app withholds estimated concentration and load where the median captured fraction is below 5%.

When discharge is supplied, load is calculated as:

<div class="equation">Load = <i>C</i><sub>depth-average</sub> <i>Q</i></div>

### Sampling correction workbook

The example workbook shows the required structure needed for input into the app.

The Flow sheet contains river conditions for each sample. 

The Sample sheet contains one or more depth intervals for each `sample_id`, with lower and upper relative depths and sampled volume. A `sampling_depth_id` can be included to identify each separate depth interval.

The `microplastics` and `macroplastic` sheets contain the corresponding records for each `sample_id` and, where used, `sampling_depth_id`. Data from OpenSpecy or MaTCH can be transferred into the required RIVER-PLAST workbook format.

The `advanced` sheet contains the `a_bed` and `a_surf` reference offsets used for the uploaded workbook.

When a workbook is uploaded, RIVER-PLAST produces one set of profiles and results for each `sample_id`. 
The results table and exported workbook report captured and missed fractions, measured concentration, depth-average concentration, and load for the total and available fractions.

### Interpretation and limitations

RIVER-PLAST is most suitable for approximately steady, laterally mixed flow and freely suspended particles. The model also assumes that the selected particle class is sufficiently available throughout the water column. Finding a single litter item at one depth does not necessarily mean that many similar items occur elsewhere in the water column.

## References

Dietrich, W. E. (1982). *Water Resources Research*, *18*(6), 1615–1626. DOI: [10.1029/WR018i006p01615](https://doi.org/10.1029/WR018i006p01615).

Goral, K. D., Guler, H. G., Larsen, B. E., Carstensen, S., Christensen, E. D., Kerpen, N. B., Schlurmann, T. and Fuhrman, D. R. (2023). *Environmental Research*, *228*, 115783. DOI: [10.1016/j.envres.2023.115783](https://doi.org/10.1016/j.envres.2023.115783).

Hapich, H., Cowger, W. and Gray, A. (2024). *Environmental Science & Technology*. DOI: [10.1021/acs.est.4c02406](https://doi.org/10.1021/acs.est.4c02406).

Kooi, M. et al. (2019). *Environmental Science & Technology Letters*. DOI: [10.1021/acs.estlett.9b00379](https://doi.org/10.1021/acs.estlett.9b00379).

Lofty, J., Valero, D. and Franca, M. J. (2026). River-OSPAR litter observations. DOI: [10.1021/acs.est.5c13126](https://doi.org/10.1021/acs.est.5c13126).

Lofty, J., Valero, D. and Franca, M. J. (2026). *Settling and Rising Dynamics of River Litter*. EarthArXiv. DOI: [10.31223/X5TF4V](https://doi.org/10.31223/X5TF4V).

Rouse, H. (1939). *Annual Meeting of the American Society of Civil Engineers, New York City. Preprint, United States Department of Agriculture.*

Yu, Z., Yang, G. and Zhang, W. (2022). *Marine Pollution Bulletin*, *176*, 113449. DOI: [10.1016/j.marpolbul.2022.113449](https://doi.org/10.1016/j.marpolbul.2022.113449).
