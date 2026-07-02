
# -*- coding: utf-8 -*-
"""
Synthetic microplastic generator and settling-velocity equations.

This module is designed to be imported by the Shiny app. It generates a
synthetic microplastic particle dataset using user-selected polymer fractions,
density ranges, particle-type fractions, and size ranges. It then calculates
settling/rising velocities using Goral, Yu, and Dietrich equations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# FLUID PROPERTIES
# ============================================================
rho_water = 1000.0        # kg/m3
nu = 1.004e-6             # m2/s, water at ~20 C
g = 9.81                  # m/s2


# ============================================================
# DEFAULT POLYMER MIX AND DENSITY RANGES
# Densities are in g/cm3 here, then converted to kg/m3 in the generator.
# ============================================================
DEFAULT_POLYMER_PERCENTAGES = {
    "PE": 25.0,
    "PET": 16.5,
    "PA": 12.0,
    "PP": 14.0,
    "PS": 8.5,
    "PVA": 6.0,
    "PVC": 2.0,
}

POLYMER_DENSITY_RANGES_G_CM3 = {
    "PE": (0.89, 0.98),
    "PET": (0.96, 1.45),
    "PA": (1.02, 1.16),
    "PP": (0.83, 0.92),
    "PS": (1.04, 1.10),
    "PVA": (1.19, 1.31),
    "PVC": (1.10, 1.58),
}


# ============================================================
# SETTLING VELOCITY EQUATIONS
# ============================================================
def compute_velocity_goral(d, rho_p, csf, particle_type):
    d = np.asarray(d, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    csf = np.clip(np.asarray(csf, dtype=float), 1e-6, 1.0)
    particle_type = np.asarray(particle_type)

    delta_rho = np.abs(rho_p - rho_water)
    sign = np.where(rho_p >= rho_water, 1.0, -1.0)
    velocity_goral = np.zeros_like(d)

    for i in range(len(d)):
        if d[i] <= 0 or delta_rho[i] == 0:
            velocity_goral[i] = 0.0
            continue

        psi = csf[i]
        w = delta_rho[i] * g * d[i] ** 2 / (18.0 * rho_water * nu)

        for _ in range(100):
            Re = max(abs(w) * d[i] / nu, 1e-12)

            if particle_type[i] == "fiber":
                Cd = max(19.0 * Re ** (-0.6), 0.86)
            else:
                Cd = (1.0 + 3.2 / np.sqrt(Re) + 32.0 / Re) * min(
                    0.44 * psi ** (-2.0), 1.0
                )

            w_new = np.sqrt(
                (4.0 * delta_rho[i] * g * d[i]) / (3.0 * rho_water * Cd)
            )

            if abs(w_new - w) < 1e-12:
                w = w_new
                break

            w = w_new

        velocity_goral[i] = sign[i] * w

    return velocity_goral


def compute_velocity_yu(d, rho_p, csf):
    d = np.asarray(d, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    csf = np.clip(np.asarray(csf, dtype=float), 1e-6, 1.0)

    phi = csf
    delta_rho = np.abs(rho_p - rho_water)
    sign = np.where(rho_p >= rho_water, 1.0, -1.0)

    velocity_yu = np.zeros_like(d)
    valid = (d > 0) & (delta_rho > 0)

    d_star = np.zeros_like(d)
    d_star[valid] = d[valid] * (
        (delta_rho[valid] * g) / (rho_water * nu ** 2)
    ) ** (1.0 / 3.0)

    d_star_safe = np.clip(d_star, 1e-12, None)

    Cd_s = (
        432.0 / d_star_safe ** 3
        * (1.0 + 0.022 * d_star_safe ** 3) ** 0.54
        + 0.47 * (1.0 - np.exp(-0.15 * d_star_safe ** 0.45))
    )

    Cd = Cd_s / (phi * csf * d_star_safe ** (-0.25 + 0.03 + 0.33)) ** 0.25

    velocity_yu[valid] = (
        (nu * g * (delta_rho[valid] / rho_water)) ** (1.0 / 3.0)
        * np.sqrt((4.0 * d_star_safe[valid]) / (3.0 * Cd[valid]))
    )

    return sign * velocity_yu


def compute_velocity_dietrich(d, rho_p, csf, powers_roundness=3.5):
    d = np.asarray(d, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    csf = np.clip(np.asarray(csf, dtype=float), 1e-6, 1.0)

    delta_rho = np.abs(rho_p - rho_water)
    sign = np.where(rho_p >= rho_water, 1.0, -1.0)

    velocity_dietrich = np.zeros_like(d)
    valid = (d > 0) & (delta_rho > 0)

    D_star = np.zeros_like(d)
    D_star[valid] = (
        delta_rho[valid] * g * d[valid] ** 3 / (rho_water * nu ** 2)
    )

    D_star_safe = np.clip(D_star, 1e-12, None)
    logD = np.log10(D_star_safe)

    R1 = (
        -3.76715
        + 1.92944 * logD
        - 0.09815 * logD ** 2
        - 0.00575 * logD ** 3
        + 0.00056 * logD ** 4
    )

    # Dietrich expression can become invalid for some extreme CSF values.
    # The clipping keeps the log argument positive.
    log_arg = 1.0 - ((1.0 - csf) / 0.85)
    log_arg = np.clip(log_arg, 1e-12, None)

    R2 = (
        np.log10(log_arg)
        - (1.0 - csf) ** 2.3 * np.tanh(logD - 4.6)
        + 0.3 * (0.5 - csf) * (1.0 - csf) ** 2 * (logD - 4.6)
    )

    P = powers_roundness

    R3_base = 0.65 - ((csf / 2.83) * np.tanh(logD - 4.6))
    R3_base = np.clip(R3_base, 1e-12, None)

    R3 = R3_base ** (1.0 + ((3.5 - P) / 2.5))
    W_star = R3 * 10.0 ** (R1 + R2)

    velocity_dietrich[valid] = (
        W_star[valid] * delta_rho[valid] * g * nu / rho_water
    ) ** (1.0 / 3.0)

    return sign * velocity_dietrich


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
def _normalise_percentages(percentages: dict[str, float]) -> tuple[list[str], np.ndarray]:
    polymers = list(percentages.keys())
    weights = np.array([max(float(percentages[p]), 0.0) for p in polymers], dtype=float)

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(weights)

    weights = weights / weights.sum()
    return polymers, weights


def _sample_sizes_from_ranges(
    rng: np.random.Generator,
    n: int,
    size_ranges_um: list[tuple[float, float]],
    distribution: str = "loguniform",
) -> np.ndarray:
    """Sample particle sizes in metres.

    Size limits are clipped to 20–5000 µm for the synthetic generator.

    Supported distributions:
        loguniform
            Equal probability per log-size interval. Useful when particles span
            several orders of magnitude.
        uniform
            Equal probability per µm. This creates more large particles.
        lognormal
            A truncated lognormal distribution centred on the geometric midpoint
            of the selected size range.
    """
    clean_ranges = []
    for lo, hi in size_ranges_um:
        lo = max(float(lo), 20.0)
        hi = min(float(hi), 5000.0)
        if hi > lo:
            clean_ranges.append((lo, hi))

    if not clean_ranges:
        clean_ranges = [(20.0, 5000.0)]

    distribution = str(distribution).lower().strip()

    # Choose ranges in proportion to their width under the selected sampling scale.
    if distribution in {"loguniform", "lognormal"}:
        range_widths = np.array(
            [max(np.log10(hi) - np.log10(lo), 1e-12) for lo, hi in clean_ranges],
            dtype=float,
        )
    else:
        range_widths = np.array([max(hi - lo, 1e-12) for lo, hi in clean_ranges], dtype=float)

    range_weights = range_widths / range_widths.sum()
    range_idx = rng.choice(len(clean_ranges), size=n, p=range_weights)
    sizes_um = np.empty(n, dtype=float)

    for i, (lo, hi) in enumerate(clean_ranges):
        mask = range_idx == i
        m = int(mask.sum())
        if m == 0:
            continue

        if distribution == "uniform":
            sizes_um[mask] = rng.uniform(lo, hi, m)

        elif distribution == "lognormal":
            # Truncated lognormal centred on the geometric midpoint.
            mu = np.log(np.sqrt(lo * hi))
            sigma = 0.85
            vals = rng.lognormal(mean=mu, sigma=sigma, size=m)

            # Resample out-of-range values a few times, then clip any leftovers.
            for _ in range(10):
                bad = (vals < lo) | (vals > hi)
                if not bad.any():
                    break
                vals[bad] = rng.lognormal(mean=mu, sigma=sigma, size=int(bad.sum()))

            sizes_um[mask] = np.clip(vals, lo, hi)

        else:
            sizes_um[mask] = 10 ** rng.uniform(np.log10(lo), np.log10(hi), m)

    return sizes_um * 1e-6


def _sample_csf(
    rng: np.random.Generator,
    particle_type: np.ndarray,
) -> np.ndarray:
    """Sample Corey shape factor.

    Fibres are assigned low CSF values. Fragments are assigned higher and more
    variable CSF values, using a simple approximation to the Kooi-style bimodal
    shape distribution.
    """
    csf = np.empty(len(particle_type), dtype=float)

    fiber_mask = particle_type == "fiber"
    fragment_mask = ~fiber_mask

    csf[fiber_mask] = rng.normal(loc=0.075, scale=0.030, size=fiber_mask.sum())
    csf[fragment_mask] = rng.normal(loc=0.44, scale=0.19, size=fragment_mask.sum())

    return np.clip(csf, 0.01, 1.0)


def generate_synthetic_microplastics(
    n_particles: int,
    size_ranges_um: list[tuple[float, float]],
    polymer_percentages: dict[str, float] | None = None,
    fiber_percent: float = 50.0,
    seed: int | None = 42,
    size_distribution: str = "loguniform",
) -> pd.DataFrame:
    """Generate synthetic microplastic particles and settling velocities.

    Returned columns are compatible with the Rouse-profile app:

        size
        size_um
        density
        density_g_cm3
        polymer
        particle_type
        CSF
        velocity_dietrich
        velocity_goral
        velocity_yu
    """
    n = int(max(n_particles, 1))
    rng = np.random.default_rng(seed)

    if polymer_percentages is None:
        polymer_percentages = DEFAULT_POLYMER_PERCENTAGES.copy()

    polymers, polymer_weights = _normalise_percentages(polymer_percentages)
    sampled_polymers = rng.choice(polymers, size=n, p=polymer_weights)

    density_g_cm3 = np.empty(n, dtype=float)
    for polymer in polymers:
        mask = sampled_polymers == polymer
        lo, hi = POLYMER_DENSITY_RANGES_G_CM3[polymer]
        density_g_cm3[mask] = rng.uniform(lo, hi, mask.sum())

    density = density_g_cm3 * 1000.0

    fiber_p = np.clip(float(fiber_percent), 0.0, 100.0) / 100.0
    particle_type = np.where(rng.random(n) < fiber_p, "fiber", "fragment")

    csf = _sample_csf(rng, particle_type)
    size = _sample_sizes_from_ranges(
        rng=rng,
        n=n,
        size_ranges_um=size_ranges_um,
        distribution=size_distribution,
    )

    df = pd.DataFrame(
        {
            "size": size,
            "size_um": size * 1e6,
            "density": density,
            "density_g_cm3": density_g_cm3,
            "polymer": sampled_polymers,
            "particle_type": particle_type,
            "CSF": csf,
        }
    )

    df["velocity_goral"] = compute_velocity_goral(
        d=df["size"].to_numpy(),
        rho_p=df["density"].to_numpy(),
        csf=df["CSF"].to_numpy(),
        particle_type=df["particle_type"].to_numpy(),
    )

    df["velocity_yu"] = compute_velocity_yu(
        d=df["size"].to_numpy(),
        rho_p=df["density"].to_numpy(),
        csf=df["CSF"].to_numpy(),
    )

    df["velocity_dietrich"] = compute_velocity_dietrich(
        d=df["size"].to_numpy(),
        rho_p=df["density"].to_numpy(),
        csf=df["CSF"].to_numpy(),
        powers_roundness=3.5,
    )

    return df
