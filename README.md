# Synthetic microplastic Rouse-profile app

Run:

```bash
shiny run --reload app16_synthetic_microplastics.py
```

Synthetic microplastic mode now:

- always generates 20,000 particles
- uses random seed 42
- allows size distributions between 20 and 5000 µm
- supports log-uniform, uniform, and truncated lognormal size distributions
- constrains polymer-percentage sliders so the total cannot exceed 100%
