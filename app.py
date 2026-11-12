#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:13:23 2026

@author: jameslofty
"""



# -*- coding: utf-8 -*-
"""
Shiny for Python app: vertical Rouse concentration profiles

Run:
    shiny run --reload app.py

Required data file:
    data/macroplastic_settling_data.xlsx

This version:
    - Keeps only the vertical Rouse-profile figure.
    - Uses a normal linear x-axis from 0 to 1.
    - Lets the user define up to three microplastic size ranges.
    - Places all user controls in collapsible sidebar sections.
    - Lets macrolitter be shown either by grouped category or by individual litter item.
    - Keeps the main panel focused on the plot, optional sampling estimate, and dataset summary.
    - Includes an optional sampling-depth estimator using a z/H interval for captured/missed vertical fraction.
    - Includes an optional sampling-correction calculator for measured concentration and discharge.
    - Uses separate reference offsets:
        * a_bed/H for bed-referenced settling profiles.
        * a_surf/H for surface-referenced buoyant profiles.
    - Profiles are max-normalised so the plotted concentration range is 0 to 1.
"""
# os.system("python -m shiny run --port 8001 app13.py")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import base64
from io import BytesIO
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DOCS_DIR = APP_DIR / "docs"
EXAMPLES_DIR = APP_DIR / "examples"

from codes.synthetic_particle_generator import (
    DEFAULT_POLYMER_PERCENTAGES,
    generate_synthetic_microplastics,
)


# ============================================================
# LOAD DATA
# ============================================================
macro = pd.read_excel(DATA_DIR / "macroplastic_settling_data.xlsx")


# ============================================================
# CONSTANTS
# ============================================================
kappa = 0.41
g = 9.81
PLOT_FONT_STANDARD = 9
PLOT_FONT_LARGE = 11

velocity_cols = [
    "velocity_dietrich",
    "velocity_goral",
    "velocity_yu",
]

macro_group_labels = {
    "foam_very_buoyant": "Foams (very buoyant, density 0.02–0.08 g cm⁻³)",
    "plastic_buoyant": "Plastics (buoyant, density 0.8–1 g cm⁻³)",
    "plastic_settling": "Plastics (settling, density 0.8–1 g cm⁻³)",
    "glass_metal_very_settling": "Glass & metal (strongly settling, density 2.5–4.3 g cm⁻³)",
}


def standardise_polymer_name(value) -> str | None:
    """Map common short and descriptive polymer names to app polymer groups."""
    name = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "pe": "PE", "polyethylene": "PE", "polyethylenes": "PE",
        "pp": "PP", "polypropylene": "PP", "polypropylenes": "PP",
        "pet": "PET", "polyethyleneterephthalate": "PET",
        "pa": "PA", "polyamide": "PA", "polyamides": "PA", "nylon": "PA",
        "ps": "PS", "polystyrene": "PS", "polystyrenes": "PS",
        "pva": "PVA", "polyvinylalcohol": "PVA", "polyvinylalcohols": "PVA",
        "pvc": "PVC", "polyvinylchloride": "PVC", "polyvinylchlorides": "PVC",
    }
    if name in aliases:
        return aliases[name]
    if "methylstyrene" in name or "styrene" in name:
        return "PS"
    if "vinylalcohol" in name:
        return "PVA"
    if "vinylchloride" in name or "vinylhalide" in name:
        return "PVC"
    if "terephthalate" in name:
        return "PET"
    if "polyamide" in name or "nylon" in name:
        return "PA"
    if "polyurethane" in name or "polyacrylonitrile" in name:
        return "PA"
    if "polyacrylate" in name or "methacrylate" in name or "methacrylamide" in name or "polycarbonate" in name:
        return "PVA"
    if "ethersulfone" in name or "ptfe" in name or "tetrafluoroethylene" in name:
        return "PVC"
    if "vinylester" in name:
        return "PET"
    if "vinylether" in name or "vinylketone" in name:
        return "PE"
    if any(token in name for token in ("polysiloxane", "epdm", "polydiene", "butadiene", "isoprene", "ethylenepropylene", "sbr")):
        return "PE"
    if name == "abs":
        return "PS"
    return None


# ============================================================
# STATIC PREP
# ============================================================
macro["Material_grouped"] = pd.Series(pd.NA, index=macro.index, dtype="object")
macro.loc[macro["Material"] == "EPS", "Material_grouped"] = "foam_very_buoyant"
macro.loc[
    macro["Material"].isin(["PO hard", "PO soft"])
    | ((macro["Material"] == "Multilayer") & (macro["vz_mean"] < 0)),
    "Material_grouped",
] = "plastic_buoyant"
macro.loc[
    macro["Material"].isin(["PET", "PS"])
    | ((macro["Material"] == "Multilayer") & (macro["vz_mean"] >= 0)),
    "Material_grouped",
] = "plastic_settling"
macro.loc[
    macro["Material"].isin(["Glass", "Metal"]),
    "Material_grouped",
] = "glass_metal_very_settling"

if "Common name" in macro.columns:
    macro_common_names = sorted(
        macro["Common name"].dropna().astype(str).unique().tolist()
    )
else:
    macro_common_names = []

top_five_macroplastic_items = [
    next(name for name in macro_common_names if name.replace("\u00a0", " ") == item)
    for item in [
        "Soft plastic pieces/films 0.5-2.5 cm",
        "Soft plastic pieces/films 2.5-50 cm",
        "Candy, snack and crisps packaging",
        "Hard plastic pieces 2.5-50 cm",
        "Polystyrene foam food packaging",
    ]
]

macro_common_name_to_index = {
    name: index for index, name in enumerate(macro_common_names)
}

macro_ospar_name_map = {}
if "OSPAR ID" in macro.columns and "Common name" in macro.columns:
    for ospar_id, common_name in macro[["OSPAR ID", "Common name"]].dropna().drop_duplicates().itertuples(index=False):
        try:
            macro_ospar_name_map[str(int(float(ospar_id)))] = str(common_name)
        except (TypeError, ValueError):
            continue

MIN_RELIABLE_CAPTURE = 0.05
LOW_CAPTURE_WARNING = "No estimate (<5% captured)"
MACRO_LOW_CAPTURE_WARNING = (
    "No estimate (<5% captured for one or more categories)"
)

def macro_item_concentration_input_id(common_name: str) -> str:
    """Return a stable Shiny input id for one macrolitter item."""
    return f"samp_macro_item_concentration_{macro_common_name_to_index[common_name]}"


def macro_group_concentration_input_id(group_key: str) -> str:
    """Return a stable Shiny input id for one grouped macrolitter class."""
    return f"samp_macro_group_concentration_{group_key}"


# ============================================================
# HELPERS
# ============================================================
def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def display_unit(unit: object) -> str:
    """Format concentration units for display without changing stored values."""
    unit = str(unit).strip()
    return {
        "particles/m3": "particles/m³",
        "items/m3": "items/m³",
        "mg/m3": "mg/m³",
        "g/m3": "g/m³",
    }.get(unit, unit)


def fmt_sig(x, sig=3):
    """Format with significant figures."""
    if pd.isna(x) or not np.isfinite(x):
        return ""
    return f"{x:.{sig}g}"


def fmt_interval(median, low, high, sig=3):
    """Format median [low - high]."""
    return (
        f"{fmt_sig(median, sig)} "
        f"[{fmt_sig(low, sig)} - {fmt_sig(high, sig)}]"
    )


def calculate_shear_velocity_from_slope_radius(hydraulic_radius: float, slope: float) -> float:
    """
    Open-channel estimate:

        u* = sqrt(g R S)

    For a wide channel, hydraulic radius R can be approximated by flow depth H.
    """
    return float(np.sqrt(g * hydraulic_radius * slope))


def calculate_micro_rouse_mean(u_star: float, micro_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate mean microplastic Rouse number across the three velocity equations.

        beta = w / (kappa u*)

    The app passes a generated microplastic dataframe with the required
    velocity columns.
    """
    df = micro_df

    beta_arrays = []

    for col in velocity_cols:
        if col not in df.columns:
            continue

        w = df[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        beta_arrays.append(w / (kappa * u_star))

    if len(beta_arrays) == 0:
        return np.full(len(df), np.nan)

    return np.nanmean(np.vstack(beta_arrays), axis=0)


def calculate_macro_rouse(u_star: float) -> np.ndarray:
    """
    Calculate macrolitter Rouse number.

    Preserves your earlier conversion:
        macro_w = vz_mean / 100
    """
    w = (macro["vz_mean"] / 100).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    return w / (kappa * u_star)


def beta_values_for_micro_range(
    min_um: float,
    max_um: float,
    u_star: float,
    micro_df: pd.DataFrame,
) -> np.ndarray:
    """Return beta values for one user-selected microplastic size range.

    Size inputs are in micrometres. The underlying dataset stores size in metres,
    so a precomputed size_um column is used for direct filtering.
    """
    df = micro_df

    if max_um <= min_um or "size_um" not in df.columns:
        return np.array([])

    beta = calculate_micro_rouse_mean(u_star, micro_df=df)
    mask = (df["size_um"] >= min_um) & (df["size_um"] <= max_um)

    return finite(beta[mask.to_numpy()])


def beta_values_for_macro_group(group_key: str, u_star: float) -> np.ndarray:
    """Return beta values for one macrolitter buoyancy/density group."""
    if group_key not in macro_group_labels:
        return np.array([])

    beta = calculate_macro_rouse(u_star)
    mask = macro["Material_grouped"] == group_key

    return finite(beta[mask.to_numpy()])


def beta_values_for_macro_item(common_name: str, u_star: float) -> np.ndarray:
    """Return beta values for one macrolitter item using the Common name column."""
    if "Common name" not in macro.columns:
        return np.array([])

    beta = calculate_macro_rouse(u_star)
    mask = macro["Common name"].astype(str) == str(common_name)

    return finite(beta[mask.to_numpy()])


def normalise_0_1(c: np.ndarray) -> np.ndarray:
    """
    Max-normalise a profile to 0-1.

    This is not mass-normalisation. It is visual/profile normalisation:
        max(C_norm) = 1
    """
    c = np.asarray(c, dtype=float)
    c[~np.isfinite(c)] = np.nan

    max_c = np.nanmax(c)

    if not np.isfinite(max_c) or max_c <= 0:
        return np.full_like(c, np.nan)

    return c / max_c


def rouse_profile_from_beta(
    beta: float,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    n: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a direction-aware, max-normalised Rouse profile.

    Sinking particles, beta >= 0:
        Use a bed-referenced Rouse profile with reference height a_bed above the bed.

        C(z) / C(a_bed) =
        [ ((H - z) / z) / ((H - a_bed) / a_bed) ] ** beta

    Buoyant particles, beta < 0:
        Use a surface-referenced mirrored profile with reference distance a_surf below
        the surface, i.e. z_ref = H - a_surf.

        C(z) / C(H - a_surf) =
        [ (z / (H - z)) / ((H - a_surf) / a_surf) ] ** abs(beta)

    The output is max-normalised, so plotted values are always 0 to 1.
    """
    a_bed = a_bed_frac * H
    a_surf = a_surf_frac * H

    if a_bed <= 0 or a_surf <= 0 or (a_bed + a_surf) >= H:
        return np.array([]), np.array([])

    # Avoid singularities exactly at the bed and surface.
    z = np.linspace(a_bed, H - a_surf, n)
    z_rel = z / H

    if beta >= 0:
        # Sinking profile: high near bed, low toward surface.
        c = (((H - z) / z) / ((H - a_bed) / a_bed)) ** beta
    else:
        # Buoyant profile: high near surface, low toward bed.
        p = abs(beta)
        c = ((z / (H - z)) / ((H - a_surf) / a_surf)) ** p

    c = np.asarray(c, dtype=float)
    c[~np.isfinite(c)] = np.nan

    # Limit pathological overflow before normalisation.
    c = np.clip(c, 0, 1e12)

    c_norm = normalise_0_1(c)

    return z_rel, c_norm


def rouse_profile_matrix(
    beta_values: np.ndarray,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    n: int = 250,
    z_rel: np.ndarray | None = None,
    normalise: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Rouse profiles for fast interactive calculations."""
    beta_values = finite(beta_values)
    if (
        len(beta_values) == 0
        or a_bed_frac <= 0
        or a_surf_frac <= 0
        or (a_bed_frac + a_surf_frac) >= 1
    ):
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    if z_rel is None:
        z_rel = np.linspace(a_bed_frac, 1.0 - a_surf_frac, n)
    else:
        z_rel = np.asarray(z_rel, dtype=float)

    z_rel = np.clip(z_rel, a_bed_frac, 1.0 - a_surf_frac)
    profiles = np.empty((len(beta_values), len(z_rel)), dtype=float)
    sinking = beta_values >= 0

    if np.any(sinking):
        base = ((1.0 - z_rel) / z_rel) / (
            (1.0 - a_bed_frac) / a_bed_frac
        )
        exponent = beta_values[sinking, None] * np.log(base[None, :])
        profiles[sinking] = np.exp(np.clip(exponent, -745.0, np.log(1e12)))

    if np.any(~sinking):
        base = (z_rel / (1.0 - z_rel)) / (
            (1.0 - a_surf_frac) / a_surf_frac
        )
        exponent = np.abs(beta_values[~sinking, None]) * np.log(base[None, :])
        profiles[~sinking] = np.exp(np.clip(exponent, -745.0, np.log(1e12)))

    profiles[~np.isfinite(profiles)] = np.nan
    if normalise:
        maxima = np.nanmax(profiles, axis=1, keepdims=True)
        profiles = np.divide(
            profiles,
            maxima,
            out=np.full_like(profiles, np.nan),
            where=np.isfinite(maxima) & (maxima > 0),
        )

    return z_rel, profiles


def group_profile_summary(
    beta_values: np.ndarray,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    iqr_lower: float = 25,
    iqr_upper: float = 75,
    n: int = 250,
) -> pd.DataFrame:
    """Return median and user-selected percentile profiles for a group."""
    beta_values = finite(beta_values)

    lower = float(min(iqr_lower, iqr_upper))
    upper = float(max(iqr_lower, iqr_upper))

    if len(beta_values) == 0:
        return pd.DataFrame(columns=["z_rel", "median", "q_low", "q_high"])

    z_rel, profiles = rouse_profile_matrix(
        beta_values=beta_values,
        H=H,
        a_bed_frac=a_bed_frac,
        a_surf_frac=a_surf_frac,
        n=n,
        normalise=True,
    )
    if profiles.size == 0:
        return pd.DataFrame(columns=["z_rel", "median", "q_low", "q_high"])

    return pd.DataFrame(
        {
            "z_rel": z_rel,
            "median": np.nanmedian(profiles, axis=0),
            "q_low": np.nanpercentile(profiles, lower, axis=0),
            "q_high": np.nanpercentile(profiles, upper, axis=0),
        }
    )


def sampling_fraction_from_profile(
    z_rel: np.ndarray,
    c: np.ndarray,
    net_z_min: float,
    net_z_max: float,
) -> tuple[float, float]:
    """Estimate captured and missed fractions from one concentration profile."""
    z_rel = np.asarray(z_rel, dtype=float)
    c = np.asarray(c, dtype=float)

    valid = np.isfinite(z_rel) & np.isfinite(c)
    z_rel = z_rel[valid]
    c = c[valid]

    if len(z_rel) < 2:
        return np.nan, np.nan

    order = np.argsort(z_rel)
    z_rel = z_rel[order]
    c = c[order]

    total = float(np.trapezoid(c, z_rel))
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan

    profile_min = float(np.nanmin(z_rel))
    profile_max = float(np.nanmax(z_rel))

    z_min = max(profile_min, min(float(net_z_min), float(net_z_max)))
    z_max = min(profile_max, max(float(net_z_min), float(net_z_max)))

    if z_max <= z_min:
        return 0.0, 1.0

    sampled = (z_rel >= z_min) & (z_rel <= z_max)

    # Include exact boundary points by interpolation so the fraction is not
    # sensitive to the plotting grid resolution.
    boundary_z = np.array([z_min, z_max], dtype=float)
    boundary_c = np.interp(boundary_z, z_rel, c)

    sample_z = np.concatenate([boundary_z[:1], z_rel[sampled], boundary_z[1:]])
    sample_c = np.concatenate([boundary_c[:1], c[sampled], boundary_c[1:]])

    order = np.argsort(sample_z)
    sample_z = sample_z[order]
    sample_c = sample_c[order]

    captured = float(np.trapezoid(sample_c, sample_z) / total)
    captured = min(max(captured, 0.0), 1.0)
    missed = 1.0 - captured

    return captured, missed


def sampling_fraction_from_summary(
    summary: pd.DataFrame,
    net_z_min: float,
    net_z_max: float,
) -> tuple[float, float]:
    """Estimate captured and missed fractions from the median profile."""
    if summary.empty:
        return np.nan, np.nan

    return sampling_fraction_from_profile(
        z_rel=summary["z_rel"].to_numpy(dtype=float),
        c=summary["median"].to_numpy(dtype=float),
        net_z_min=net_z_min,
        net_z_max=net_z_max,
    )


def sampling_fraction_distribution_from_beta(
    beta_values: np.ndarray,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    net_z_min: float | None = None,
    net_z_max: float | None = None,
    net_z_intervals: list[tuple[float, float]] | None = None,
    n: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    """Return captured/missed fractions for every beta value in a group.

    This is used for uncertainty summaries. Instead of calculating only the
    median profile and then integrating that one curve, this function integrates
    each particle/profile first. The table can then report:

        median [lower percentile - upper percentile]

    for capture fraction, missed fraction, correction factor, corrected
    concentration, and load.
    """
    beta_values = finite(beta_values)
    z_rel, profiles = rouse_profile_matrix(
        beta_values=beta_values,
        H=H,
        a_bed_frac=a_bed_frac,
        a_surf_frac=a_surf_frac,
        n=n,
        normalise=False,
    )
    if profiles.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    totals = np.trapezoid(profiles, z_rel, axis=1)
    intervals = net_z_intervals or [(net_z_min, net_z_max)]
    clipped_intervals = []
    for lower, upper in intervals:
        if lower is None or upper is None:
            continue
        z_min = max(float(z_rel[0]), min(float(lower), float(upper)))
        z_max = min(float(z_rel[-1]), max(float(lower), float(upper)))
        if z_max > z_min:
            clipped_intervals.append((z_min, z_max))

    clipped_intervals.sort()
    merged_intervals: list[tuple[float, float]] = []
    for z_min, z_max in clipped_intervals:
        if merged_intervals and z_min <= merged_intervals[-1][1]:
            merged_intervals[-1] = (merged_intervals[-1][0], max(z_max, merged_intervals[-1][1]))
        else:
            merged_intervals.append((z_min, z_max))

    sampled = np.zeros(len(beta_values), dtype=float)
    for z_min, z_max in merged_intervals:
        interior = z_rel[(z_rel > z_min) & (z_rel < z_max)]
        sample_z = np.concatenate(([z_min], interior, [z_max]))
        _, sample_profiles = rouse_profile_matrix(
            beta_values=beta_values,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            z_rel=sample_z,
            normalise=False,
        )
        sampled += np.trapezoid(sample_profiles, sample_z, axis=1)

    captured = np.divide(
        sampled,
        totals,
        out=np.full_like(sampled, np.nan),
        where=np.isfinite(totals) & (totals > 0),
    )
    captured = np.clip(captured, 0.0, 1.0)

    valid = np.isfinite(captured)
    captured = captured[valid]
    return captured, 1.0 - captured


def observation_factor_distribution_from_beta(
    beta_values: np.ndarray,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    net_z_intervals: list[tuple[float, float]],
    net_sample_volumes: list[float] | None = None,
) -> np.ndarray:
    """Return the expected sampled concentration relative to the depth average.

    Each sampled layer is weighted by its sampled volume and its represented
    depth fraction. This gives the standard depth-stratified correction and
    reduces to sampled_fraction / captured_fraction for one depth sample.
    """
    intervals = list(net_z_intervals)
    if not intervals:
        return np.array([], dtype=float)
    volumes = np.asarray(
        net_sample_volumes if net_sample_volumes is not None else [1.0] * len(intervals),
        dtype=float,
    )
    if len(volumes) != len(intervals) or not np.isfinite(volumes).all() or (volumes <= 0).any():
        volumes = np.ones(len(intervals), dtype=float)

    factors = []
    for (lower, upper), volume in zip(intervals, volumes):
        depth_fraction = abs(float(upper) - float(lower))
        if depth_fraction <= 0:
            continue
        captured, _ = sampling_fraction_distribution_from_beta(
            beta_values=beta_values,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            net_z_intervals=[(lower, upper)],
        )
        if len(captured) > 0:
            factors.append((float(volume), depth_fraction, captured))

    if not factors:
        return np.array([], dtype=float)
    total_volume = sum(volume for volume, _, _ in factors)
    return sum(volume * captured / depth_fraction for volume, depth_fraction, captured in factors) / total_volume


def format_median_iqr(
    values: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
    percent: bool = False,
) -> str:
    """
    Format values as:

        median [lower - upper]

    using 3 significant figures.
    """

    values = finite(values)

    if len(values) == 0:
        return "NA"

    lower = float(min(lower_percentile, upper_percentile))
    upper = float(max(lower_percentile, upper_percentile))

    med = float(np.nanmedian(values))
    low = float(np.nanpercentile(values, lower))
    high = float(np.nanpercentile(values, upper))

    if percent:
        def format_percent_value(value: float) -> str:
            percentage = value * 100
            if percentage == 0:
                return "0%"
            if 0 < abs(percentage) < 0.01:
                return "<0.01%"
            return f"{fmt_sig(percentage)}%"

        return (
            f"{format_percent_value(med)} "
            f"[{format_percent_value(low)} - {format_percent_value(high)}]"
        )

    return fmt_interval(med, low, high)


def add_macroplastic_total(
    groups: list[tuple[str, np.ndarray]],
    include_members: bool,
) -> list[tuple[str, np.ndarray]]:
    """Add one combined macrolitter group, optionally followed by its members."""
    macro_groups = [
        (name, finite(beta))
        for name, beta in groups
        if name.startswith("Macro")
    ]
    non_macro_groups = [
        (name, beta)
        for name, beta in groups
        if not name.startswith("Macro")
    ]

    if not macro_groups:
        return groups

    combined_beta = finite(
        np.concatenate([beta for _, beta in macro_groups])
    )
    result = non_macro_groups + [("Macrolitter: total", combined_beta)]
    if include_members:
        result.extend(macro_groups)
    return result


def net_sampling_table(
    micro_ranges: list[tuple[str, float, float]],
    macro_selected: list[str],
    macro_items_selected: list[str],
    use_macro_items: bool,
    u_star: float,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    iqr_lower: float,
    iqr_upper: float,
    net_z_intervals: list[tuple[float, float]] | None = None,
    net_z_min: float | None = None,
    net_z_max: float | None = None,
    micro_df: pd.DataFrame | None = None,
    split_micro_by_direction: bool = False,
    extra_micro_groups: list[tuple[str, np.ndarray]] | None = None,
    include_micro_total: bool = True,
) -> pd.DataFrame:
    """Return captured/missed fractions for the current selected groups."""
    rows = []

    groups = selected_group_beta_values(
        micro_ranges=micro_ranges,
        macro_selected=macro_selected,
        macro_items_selected=macro_items_selected,
        use_macro_items=use_macro_items,
        u_star=u_star,
        micro_df=micro_df,
        split_micro_by_direction=False,
    )
    if include_micro_total and not split_micro_by_direction:
        groups = [
            ("Microplastics: total" if group_name == "Microplastics" else group_name, beta)
            for group_name, beta in groups
        ]
    if not include_micro_total:
        groups = [
            (group_name, beta)
            for group_name, beta in groups
            if not group_name.startswith("Microplastics")
        ]

    if split_micro_by_direction:
        expanded_groups = []
        for group_name, beta in groups:
            beta = finite(beta)
            if group_name.startswith("Microplastics"):
                expanded_groups.append((f"{group_name}: total", beta))
                buoyant = beta[beta < 0]
                settling = beta[beta >= 0]
                if len(buoyant) > 0:
                    expanded_groups.append((f"{group_name}: buoyant", buoyant))
                if len(settling) > 0:
                    expanded_groups.append((f"{group_name}: settling", settling))
            else:
                expanded_groups.append((group_name, beta))
        groups = expanded_groups

    extra_micro_groups = extra_micro_groups or []
    groups.extend(extra_micro_groups)

    groups = add_macroplastic_total(groups, include_members=True)
    micro_total_counts = {
        group_name.rsplit(": ", 1)[0]: len(finite(beta))
        for group_name, beta in groups
        if group_name.startswith("Microplastics") and group_name.endswith(": total")
    }
    total_micro_count = len(
        finite(calculate_micro_rouse_mean(u_star, micro_df=micro_df))
    ) if micro_df is not None else 0
    extra_micro_counts = {
        group_name: len(finite(beta))
        for group_name, beta in extra_micro_groups
    }

    for group_name, beta in groups:
        micro_total_count = micro_total_counts.get(group_name.rsplit(": ", 1)[0], 0)
        intervals = net_z_intervals or [(net_z_min, net_z_max)]
        captured_values, missed_values = sampling_fraction_distribution_from_beta(
            beta_values=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            net_z_intervals=intervals,
        )
        interval_text = ", ".join(
            f"{min(lower, upper):.2f}–{max(lower, upper):.2f}"
            for lower, upper in intervals
        )
        sampled_fraction = min(
            sum(abs(float(upper) - float(lower)) for lower, upper in intervals),
            1.0,
        )
        rows.append(
            {
                "Group": group_name,
                "Population (%)": (
                    f"{100 * extra_micro_counts[group_name] / total_micro_count:.1f}%"
                    if group_name in extra_micro_counts and total_micro_count > 0
                    else "100%"
                    if group_name in extra_micro_counts and len(extra_micro_groups) == 1
                    else
                    "100%"
                    if group_name.startswith("Microplastics") and group_name.endswith(": total")
                    else f"{100 * len(finite(beta)) / micro_total_count:.1f}%"
                    if group_name.startswith("Microplastics:")
                    and micro_total_count > 0
                    else ""
                ),
                "Sampled z/H interval": interval_text,
                "Water-column fraction sampled": round(sampled_fraction, 3),
                "Capture (%)": format_median_iqr(
                    captured_values,
                    iqr_lower,
                    iqr_upper,
                    percent=True,
                ),
                "Missed (%)": format_median_iqr(
                    missed_values,
                    iqr_lower,
                    iqr_upper,
                    percent=True,
                ),
            }
        )

    return pd.DataFrame(rows)



def sampling_correction_table(
    micro_ranges: list[tuple[str, float, float]],
    macro_selected: list[str],
    macro_items_selected: list[str],
    use_macro_items: bool,
    u_star: float,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    net_z_min: float,
    net_z_max: float,
    measured_concentration: float,
    concentration_units: str,
    include_discharge: bool,
    discharge: float,
    iqr_lower: float,
    iqr_upper: float,
    micro_df: pd.DataFrame | None = None,
    include_macro_members: bool = False,
    micro_concentrations: dict[str, float] | None = None,
    net_z_intervals: list[tuple[float, float]] | None = None,
    net_sample_volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Estimate depth-averaged concentration and optional river load.

    The correction uses the median particle capture fraction:

        median capture = median of the particle capture fractions
        correction factor = 1 / median capture

    Particle-level correction, concentration, and load values are also
    summarised as median [P25 - P75] to show variability.
    """
    rows = []
    c_obs = float(measured_concentration)
    q = float(discharge)
    intervals = net_z_intervals or [(net_z_min, net_z_max)]
    sampled_depth_fraction = min(sum(abs(float(upper) - float(lower)) for lower, upper in intervals), 1.0)

    groups = selected_group_beta_values(
        micro_ranges=micro_ranges,
        macro_selected=macro_selected,
        macro_items_selected=macro_items_selected,
        use_macro_items=use_macro_items,
        u_star=u_star,
        micro_df=micro_df,
        split_micro_by_direction=False,
    )
    groups = add_macroplastic_total(
        groups,
        include_members=include_macro_members,
    )

    for group_name, beta in groups:
        c_obs = float((micro_concentrations or {}).get(group_name, measured_concentration))
        if group_name.startswith("Microplastics"):
            group_name = f"{group_name}: total"
        captured_values, missed_values = sampling_fraction_distribution_from_beta(
            beta_values=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            net_z_intervals=intervals,
        )

        valid_captured = captured_values[
            np.isfinite(captured_values) & (captured_values > 0)
        ]

        observation_factors = observation_factor_distribution_from_beta(
            beta_values=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            net_z_intervals=intervals,
            net_sample_volumes=net_sample_volumes,
        )
        valid_observation_factors = observation_factors[
            np.isfinite(observation_factors) & (observation_factors > 0)
        ]

        if len(valid_captured) > 0 and len(valid_observation_factors) > 0:
            median_capture = float(np.nanmedian(valid_captured))
            reliable_capture = median_capture >= MIN_RELIABLE_CAPTURE
            if reliable_capture:
                correction_factor_values = 1.0 / valid_observation_factors
                corrected_concentration_values = c_obs * correction_factor_values
            else:
                correction_factor_values = np.array([], dtype=float)
                corrected_concentration_values = np.array([], dtype=float)
        else:
            median_capture = np.nan
            reliable_capture = False
            correction_factor_values = np.array([], dtype=float)
            corrected_concentration_values = np.array([], dtype=float)

        if (
            include_discharge
            and np.isfinite(q)
            and q >= 0
        ):
            estimated_load_values = corrected_concentration_values * q
        else:
            estimated_load_values = np.array([], dtype=float)
        
        load_units_map = {
            "particles/m3": "particles/s",
            "items/m3": "items/s",
            "mg/m3": "mg/s",
            "g/m3": "g/s",
        }

        rows.append(
            {
                "Group": group_name,
                "Sampled z/H interval": f"{min(net_z_min, net_z_max):.2f}–{max(net_z_min, net_z_max):.2f}",
                "Measured concentration": round(c_obs, 4),
                "Units": concentration_units,
                "Median capture fraction": fmt_sig(median_capture),
                "Median captured (%)": (
                    f"{fmt_sig(median_capture * 100)}%"
                    if np.isfinite(median_capture)
                    else "NA"
                ),
                "Particle capture variability": format_median_iqr(
                    captured_values,
                    iqr_lower,
                    iqr_upper,
                    percent=True,
                ),
                "Expected capture (%)": format_median_iqr(
                    captured_values,
                    iqr_lower,
                    iqr_upper,
                    percent=True,
                ),
                "Expected missed (%)": format_median_iqr(
                    missed_values,
                    iqr_lower,
                    iqr_upper,
                    percent=True,
                ),
                "Correction factor": format_median_iqr(
                    correction_factor_values,
                    iqr_lower,
                    iqr_upper,
                ) if reliable_capture else LOW_CAPTURE_WARNING,
                "Estimated depth-averaged concentration": (
                    format_median_iqr(
                        corrected_concentration_values,
                        iqr_lower,
                        iqr_upper,
                    )
                    if reliable_capture
                    else LOW_CAPTURE_WARNING
                ),
                "Discharge Q (m3/s)": round(q, 4) if include_discharge else np.nan,
                "Estimated load": (
                    format_median_iqr(
                        estimated_load_values,
                        iqr_lower,
                        iqr_upper,
                    )
                    if include_discharge and reliable_capture
                    else LOW_CAPTURE_WARNING
                    if include_discharge
                    else ""
                ),
                "Load units": (
                    load_units_map.get(
                        concentration_units,
                        f"{concentration_units} × m3/s",
                    )
                    if include_discharge
                    else ""
                ),
            }
        )

    return pd.DataFrame(rows)


def macro_item_correction_table(
    item_concentrations: dict[str, float],
    u_star: float,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    net_z_min: float,
    net_z_max: float,
    concentration_units: str,
    include_discharge: bool,
    discharge: float,
    iqr_lower: float,
    iqr_upper: float,
    component_type: str = "item",
    net_z_intervals: list[tuple[float, float]] | None = None,
    net_sample_volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Correct separately measured macrolitter item or class concentrations."""
    rows = []
    corrected_distributions = []
    capture_distributions = []
    total_measured = 0.0
    q = float(discharge)
    rng = np.random.default_rng(42)
    draw_count = 5000
    has_unreliable_positive_component = False
    intervals = net_z_intervals or [(net_z_min, net_z_max)]
    sampled_depth_fraction = min(sum(abs(float(upper) - float(lower)) for lower, upper in intervals), 1.0)

    load_units_map = {
        "particles/m3": "particles/s",
        "items/m3": "items/s",
        "mg/m3": "mg/s",
        "g/m3": "g/s",
    }
    load_units = load_units_map.get(
        concentration_units,
        f"{concentration_units} × m3/s",
    )

    for component_name, measured_concentration in item_concentrations.items():
        c_obs = max(float(measured_concentration), 0.0)
        total_measured += c_obs
        if component_type == "group":
            beta = beta_values_for_macro_group(component_name, u_star)
            row_name = macro_group_labels[component_name]
        else:
            beta = beta_values_for_macro_item(component_name, u_star)
            row_name = component_name
        captured_values, _ = sampling_fraction_distribution_from_beta(
            beta_values=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            net_z_intervals=intervals,
        )
        valid_captured = captured_values[
            np.isfinite(captured_values) & (captured_values > 0)
        ]
        observation_factors = observation_factor_distribution_from_beta(
            beta_values=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            net_z_intervals=intervals,
            net_sample_volumes=net_sample_volumes,
        )
        valid_observation_factors = observation_factors[
            np.isfinite(observation_factors) & (observation_factors > 0)
        ]
        if len(valid_captured) > 0:
            capture_distributions.append(valid_captured)

        if len(valid_captured) > 0 and len(valid_observation_factors) > 0:
            median_capture = float(np.nanmedian(valid_captured))
            raw_corrected_values = (
                c_obs / valid_observation_factors
            )
            reliable_capture = median_capture >= MIN_RELIABLE_CAPTURE
            if c_obs > 0 and not reliable_capture:
                has_unreliable_positive_component = True
            corrected_distributions.append(raw_corrected_values)
            if reliable_capture:
                corrected_values = raw_corrected_values
                median_corrected = float(np.nanmedian(corrected_values))
            else:
                corrected_values = np.array([], dtype=float)
                median_corrected = np.nan
        else:
            corrected_values = np.array([], dtype=float)
            median_capture = np.nan
            median_corrected = np.nan
            reliable_capture = False
            if c_obs > 0:
                has_unreliable_positive_component = True

        load_values = (
            corrected_values * q
            if include_discharge and np.isfinite(q) and q >= 0
            else np.array([], dtype=float)
        )
        median_load = (
            float(np.nanmedian(load_values))
            if len(load_values) > 0
            else np.nan
        )

        rows.append(
            {
                "Group": row_name,
                "Measured concentration": round(c_obs, 4),
                "Units": concentration_units,
                "Capture (%)": format_median_iqr(
                    captured_values,
                    iqr_lower,
                    iqr_upper,
                    percent=True,
                ),
                "Missed (%)": format_median_iqr(
                    1.0 - captured_values,
                    iqr_lower,
                    iqr_upper,
                    percent=True,
                ),
                "Estimated depth-averaged concentration": format_median_iqr(
                    corrected_values,
                    iqr_lower,
                    iqr_upper,
                ) if reliable_capture else LOW_CAPTURE_WARNING,
                "Discharge Q (m3/s)": round(q, 4) if include_discharge else np.nan,
                "Estimated load": (
                    format_median_iqr(load_values, iqr_lower, iqr_upper)
                    if include_discharge and reliable_capture
                    else LOW_CAPTURE_WARNING
                    if include_discharge
                    else ""
                ),
                "Load units": load_units if include_discharge else "",
                "_median_capture": median_capture,
                "_median_corrected": median_corrected,
                "_median_load": median_load,
            }
        )

    if corrected_distributions:
        if len(corrected_distributions) == 1:
            total_corrected_draws = corrected_distributions[0].copy()
        else:
            total_corrected_draws = np.zeros(draw_count, dtype=float)
            for values in corrected_distributions:
                total_corrected_draws += rng.choice(
                    values,
                    size=draw_count,
                    replace=True,
                )
        total_load_draws = (
            total_corrected_draws * q
            if include_discharge and np.isfinite(q) and q >= 0
            else np.array([], dtype=float)
        )
        if len(capture_distributions) == 1:
            effective_capture_draws = capture_distributions[0].copy()
        elif capture_distributions:
            effective_capture_draws = np.concatenate(
                [
                    rng.choice(values, size=draw_count, replace=True)
                    for values in capture_distributions
                ]
            )
        else:
            effective_capture_draws = np.array([], dtype=float)
    else:
        total_corrected_draws = np.array([], dtype=float)
        total_load_draws = np.array([], dtype=float)
        effective_capture_draws = np.array([], dtype=float)

    total_reliable = (
        len(total_corrected_draws) > 0
        and not has_unreliable_positive_component
    )
    displayed_total_corrected = (
        total_corrected_draws
        if total_reliable
        else np.array([], dtype=float)
    )
    displayed_total_load = (
        total_load_draws
        if total_reliable
        else np.array([], dtype=float)
    )

    rows.insert(
        0,
        {
            "Group": "Total",
            "Measured concentration": round(total_measured, 4),
            "Units": concentration_units,
            "Capture (%)": format_median_iqr(
                effective_capture_draws,
                iqr_lower,
                iqr_upper,
                percent=True,
            ),
            "Missed (%)": format_median_iqr(
                1.0 - effective_capture_draws,
                iqr_lower,
                iqr_upper,
                percent=True,
            ),
            "Estimated depth-averaged concentration": format_median_iqr(
                displayed_total_corrected,
                iqr_lower,
                iqr_upper,
            ) if total_reliable else MACRO_LOW_CAPTURE_WARNING,
            "Discharge Q (m3/s)": round(q, 4) if include_discharge else np.nan,
            "Estimated load": (
                format_median_iqr(displayed_total_load, iqr_lower, iqr_upper)
                if include_discharge and total_reliable
                else "Not available"
                if include_discharge
                else ""
            ),
            "Load units": load_units if include_discharge else "",
            "_median_capture": (
                float(np.nanmedian(effective_capture_draws))
                if len(effective_capture_draws) > 0
                else np.nan
            ),
            "_median_corrected": (
                float(np.nanmedian(displayed_total_corrected))
                if len(displayed_total_corrected) > 0
                else np.nan
            ),
            "_median_load": (
                float(np.nanmedian(displayed_total_load))
                if len(displayed_total_load) > 0
                else np.nan
            ),
        }
    )

    return pd.DataFrame(rows)


def selected_group_beta_values(
    micro_ranges: list[tuple[str, float, float]],
    macro_selected: list[str],
    macro_items_selected: list[str],
    use_macro_items: bool,
    u_star: float,
    micro_df: pd.DataFrame | None = None,
    split_micro_by_direction: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """Return display names and beta arrays for all selected ranges/categories/items."""
    groups = []

    for range_name, min_um, max_um in micro_ranges:
        beta = beta_values_for_micro_range(min_um=min_um, max_um=max_um, u_star=u_star, micro_df=micro_df)
        if str(range_name).lower() == "synthetic mp":
            micro_name = "Microplastics"
        else:
            micro_name = f"Microplastics: {range_name} ({min_um:g}–{max_um:g} µm)"

        if split_micro_by_direction:
            beta = finite(beta)
            buoyant = beta[beta < 0]
            sinking = beta[beta >= 0]
            if len(buoyant) > 0:
                groups.append((f"{micro_name} buoyant", buoyant))
            if len(sinking) > 0:
                groups.append((f"{micro_name} sinking", sinking))
        else:
            groups.append((micro_name, beta))

    if use_macro_items:
        for common_name in macro_items_selected:
            beta = beta_values_for_macro_item(common_name, u_star)
            groups.append((f"Macrolitter item: {common_name}", beta))
    else:
        for group_key in macro_selected:
            beta = beta_values_for_macro_group(group_key, u_star)
            groups.append((f"Macrolitter: {macro_group_labels[group_key]}", beta))

    return groups


def profile_legend_label(group_name: str) -> str:
    """Return concise labels for concentration-profile legends."""
    if group_name in {"Microplastics", "Microplastics: total"}:
        return "Total microplastics"
    if group_name == "Macrolitter: total":
        return "Total macrolitter"
    for prefix in ("Microplastics: ", "Microplastics ", "Macrolitter item: ", "Macrolitter: "):
        if group_name.startswith(prefix):
            return group_name[len(prefix):]
    return group_name


def make_profile_plot(
    micro_ranges: list[tuple[str, float, float]],
    macro_selected: list[str],
    macro_items_selected: list[str],
    use_macro_items: bool,
    u_star: float,
    H: float,
    a_bed_frac: float,
    a_surf_frac: float,
    iqr_lower: float,
    iqr_upper: float,
    show_net_interval: bool = False,
    net_z_interval: tuple[float, float] | None = None,
    net_z_intervals: list[tuple[float, float]] | None = None,
    micro_df: pd.DataFrame | None = None,
    split_micro_by_direction: bool = False,
    extra_micro_groups: list[tuple[str, np.ndarray]] | None = None,
    include_micro_total: bool = True,
    show_iqr: bool = False,
    warning_text: str | None = None,
    display_actual_depth: bool = False,
) -> plt.Figure:
    """
    Build the vertical Rouse profile figure.

    x-axis:
        Normalised relative concentration, 0 to 1.

    y-axis:
        Relative height, z/H, or depth above the riverbed in metres.
    """
    figure_size = (7.2, 4.2)
    fig, ax = plt.subplots(figsize=figure_size)

    plotted_any = False

    groups = selected_group_beta_values(
        micro_ranges=micro_ranges,
        macro_selected=macro_selected,
        macro_items_selected=macro_items_selected,
        use_macro_items=use_macro_items,
        u_star=u_star,
        micro_df=micro_df,
        split_micro_by_direction=split_micro_by_direction,
    )
    if not include_micro_total:
        groups = [
            (group_name, beta)
            for group_name, beta in groups
            if not group_name.startswith("Microplastics")
        ]
    groups.extend(extra_micro_groups or [])

    for group_name, beta in groups:
        summary = group_profile_summary(
            beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            iqr_lower=iqr_lower,
            iqr_upper=iqr_upper,
        )

        if summary.empty:
            continue
        y_values = summary["z_rel"] * H if display_actual_depth else summary["z_rel"]

        ax.plot(
            summary["median"],
            y_values,
            linewidth=2.5,
            label=profile_legend_label(group_name),
        )

        if show_iqr:
            ax.fill_betweenx(
                y_values,
                summary["q_low"],
                summary["q_high"],
                alpha=0.18,
            )

        plotted_any = True

    y_scale = H if display_actual_depth else 1.0
    ax.axhline(
        a_bed_frac * y_scale,
        linestyle="--",
        color="grey",
        linewidth=1,
        alpha=0.25,
    )

    ax.axhline(
        (1 - a_surf_frac) * y_scale,
        linestyle="--",
        color="grey",
        linewidth=1,
        alpha=0.25,
    )
    
    if show_net_interval:
        intervals = net_z_intervals or ([net_z_interval] if net_z_interval is not None else [])
        for net_number, interval in enumerate(intervals, start=1):
            net_z_min, net_z_max = interval
            requested_low = min(float(net_z_min), float(net_z_max))
            requested_high = max(float(net_z_min), float(net_z_max))
            net_low = max(0.0, requested_low)
            net_high = min(1.0, requested_high)
            display_low = net_low * y_scale
            display_high = net_high * y_scale

            if net_high > net_low:
                ax.axhspan(
                    display_low,
                    display_high,
                    alpha=0.08,
                    zorder=0,
                    label="Sampling depth" if net_number == 1 else None,
                )
                ax.axhline(
                    display_low,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=10,
                    clip_on=False,
                )
                ax.axhline(
                    display_high,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=10,
                    clip_on=False,
                )
                ax.text(
                    0.015,
                    (display_low + display_high) / 2,
                    (
                        f"Sample depth {net_number}: {requested_low * H:.2f}–{requested_high * H:.2f} m"
                        if display_actual_depth
                        else f"Sample depth {net_number}: {requested_low:.2f}–{requested_high:.2f} z/H"
                    ),
                    ha="left",
                    va="center",
                    fontsize=PLOT_FONT_STANDARD,
                    alpha=0.85,
                    transform=ax.get_yaxis_transform(),
                )

    ax.set_xlabel(
        r"Normalised concentration, $C / C_{max}$",
        fontsize=PLOT_FONT_STANDARD,
    )
    ax.set_ylabel(
        "River depth above bed (m)" if display_actual_depth else r"Relative river depth, $z/H$",
        fontsize=PLOT_FONT_STANDARD,
    )
    ax.set_ylim(0, H * 1.02 if display_actual_depth else 1.02)
    ax.set_yticks(np.linspace(0, H, 11) if display_actual_depth else np.arange(0, 1.01, 0.1))
    ax.set_xlim(0, 1)
    
    ax.tick_params(axis="both", labelsize=PLOT_FONT_STANDARD)

    ax.grid(True, alpha=0.25)

    if warning_text:
        ax.text(
            0.5, 0.985, warning_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=PLOT_FONT_STANDARD,
            fontweight="bold",
            color="#8a1c1c",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff0f0", "edgecolor": "#c84b4b", "alpha": 0.98},
            zorder=20,
        )

    if plotted_any:
        ax.legend(loc="best", fontsize=PLOT_FONT_STANDARD)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid beta values for selected categories.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    fig.tight_layout()
    return fig


# ============================================================
# SHINY UI
# ============================================================


# ============================================================
# METHODS TEXT FOR ABOUT TAB
# ============================================================
# Keep the long About & Methods text in a separate Markdown file.
# This keeps the app code cleaner and lets you edit the methods text
# without touching the Python code.
METHODS_FILE = DOCS_DIR / "methods.md"
LANDING_FILE = DOCS_DIR / "landing_page.md"

try:
    methods_text = METHODS_FILE.read_text(encoding="utf-8")
except FileNotFoundError:
    methods_text = (
        "# Methods file not found\n\n"
        "The file `docs/methods.md` was not found. "
        "Place it in the application documentation folder."
    )

try:
    landing_text = LANDING_FILE.read_text(encoding="utf-8")
except FileNotFoundError:
    landing_text = "# RIVER-PLAST"


def landing_logo(filename: str, label: str) -> ui.Tag:
    """Embed a landing-page logo so it also works in published deployments."""
    logo_path = APP_DIR / "www" / filename
    if logo_path.exists():
        mime_type = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(logo_path.suffix.lower(), "image/png")
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        return ui.tags.img(
            src=f"data:{mime_type};base64,{encoded}",
            alt=label,
            class_="landing-logo",
        )
    return ui.div(label, class_="landing-logo-placeholder")


# ============================================================
# REUSABLE UI BLOCKS
# ============================================================
def sampling_plastic_controls_ui() -> ui.Tag:
    """Plastic controls for the Sampling correction page.

    Uses sampling-specific input ids for the correction workflow.
    """
    return ui.div(
        ui.h6("Plastic type"),
        ui.layout_columns(
            ui.input_checkbox(
                "samp_select_microplastics",
                "Microplastics",
                True,
            ),
            ui.input_checkbox(
                "samp_select_macroplastics",
                "Macrolitter",
                False,
            ),
            col_widths=[6, 6],
            gap="0.5rem",
        ),
        ui.panel_conditional(
            "input.samp_select_microplastics",
            ui.div(
                ui.tags.details(
                    ui.tags.summary("Size"),
                    ui.div(
                        ui.input_slider(
                            "samp_synthetic_size_range",
                            "Particle size limits (µm)",
                            min=20,
                            max=5000,
                            value=(300, 5000),
                            step=10,
                        ),
                        ui.input_checkbox(
                            "samp_add_size_group",
                            "Add another size group",
                            False,
                        ),
                        ui.panel_conditional(
                            "input.samp_add_size_group",
                            ui.input_slider(
                                "samp_synthetic_size_range_2",
                                "Second group size limits (µm)",
                                min=20,
                                max=5000,
                                value=(20, 300),
                                step=10,
                            ),
                        ),
                        ui.input_select(
                            "samp_synthetic_size_distribution",
                            "Size distribution",
                            choices={
                                "loguniform": "Log-uniform",
                                "uniform": "Uniform",
                            },
                            selected="loguniform",
                        ),
                        class_="collapsible-control-body",
                    ),
                    open=False,
                    class_="collapsible-control nested-control",
                ),

                ui.tags.details(
                    ui.tags.summary("Shape"),
                    ui.div(
                        ui.input_slider(
                            "samp_fiber_percent",
                            "Fibres (%)",
                            min=0,
                            max=100,
                            value=50,
                            step=1,
                        ),
                        ui.input_slider(
                            "samp_fragment_percent",
                            "Fragments (%)",
                            min=0,
                            max=100,
                            value=50,
                            step=1,
                        ),
                        class_="collapsible-control-body",
                    ),
                    open=False,
                    class_="collapsible-control nested-control",
                ),

                        ui.tags.details(
                            ui.tags.summary("Polymer"),
                            ui.div(
                        ui.input_slider("samp_polymer_PP", "PP (%)", min=0, max=100, value=40, step=1),
                        ui.input_slider("samp_polymer_PE", "PE (%)", min=0, max=100, value=30, step=1),
                        ui.input_slider("samp_polymer_PS", "PS (%)", min=0, max=100, value=15, step=1),
                        ui.input_slider("samp_polymer_PET", "PET (%)", min=0, max=100, value=10, step=1),
                        ui.input_slider("samp_polymer_PA", "PA (%)", min=0, max=100, value=5, step=1),
                        ui.input_action_button(
                            "samp_reset_polymer_mix",
                            "Reset to default %",
                            class_="btn-sm btn-outline-secondary",
                        ),
                        class_="collapsible-control-body",
                    ),
                    open=False,
                            class_="collapsible-control nested-control",
                        ),
                        ui.output_ui("samp_polymer_profile_picker_ui"),
                        ui.download_button(
                    "download_sampling_synthetic_csv",
                    "Download sampled particles CSV",
                    class_="btn-sm btn-outline-primary",
                ),
                class_="collapsible-control-body",
            ),
        ),

        ui.panel_conditional(
            "input.samp_select_macroplastics",
            ui.div(
                ui.input_radio_buttons(
                    "samp_macro_mode",
                    None,
                    choices={
                        "individual": "Individual litter items",
                        "grouped": "Grouped litter items",
                    },
                    selected="individual",
                    inline=True,
                ),
                ui.panel_conditional(
                    "input.samp_macro_mode === 'grouped'",
                    ui.input_checkbox_group(
                        "samp_macro_categories",
                        "Classes",
                        choices=macro_group_labels,
                        selected=list(macro_group_labels.keys()),
                    ),
                ),
                ui.panel_conditional(
                    "input.samp_macro_mode === 'individual'",
                    ui.input_selectize(
                        "samp_macro_common_names",
                        "Individual litter items",
                        choices=macro_common_names,
                        selected=["Soft plastic pieces/films 0.5-2.5 cm"],
                        multiple=True,
                        options={
                            "placeholder": "Search or scroll through litter items",
                            "plugins": ["remove_button"],
                            "dropdownParent": "body",
                        },
                    ),
                ),
                class_="collapsible-control-body",
            ),
        ),
        class_="sampling-plastic-controls",
    )


def sample_design_plastic_controls_ui() -> ui.Tag:
    """Independent particle inputs for the Sample design page."""
    return ui.div(
        ui.input_radio_buttons(
            "design_plastic_type", ui.h6("Expected plastics"),
            choices={"microplastic": "Microplastics", "macroplastic": "Macrolitter"},
            selected="microplastic", inline=True,
        ),
        ui.panel_conditional(
            "input.design_plastic_type === 'microplastic'",
            ui.input_checkbox(
                "design_use_estimated_distribution",
                "Use a typical microplastic population",
                value=True,
            ),
            ui.tags.details(
                ui.tags.summary("Size"),
                ui.div(
                    ui.input_slider("design_size_range", "Particle size limits (µm)", 20, 5000, (20, 5000), step=10),
                    ui.input_select("design_size_distribution", "Size distribution", {"loguniform": "Log-uniform", "uniform": "Uniform"}, selected="loguniform"),
                    class_="collapsible-control-body",
                ), open=False, class_="collapsible-control nested-control",
            ),
            ui.tags.details(
                ui.tags.summary("Shape"),
                ui.div(
                    ui.input_slider("design_fiber_percent", "Fibres (%)", 0, 100, 50, step=1),
                    ui.input_slider("design_fragment_percent", "Fragments (%)", 0, 100, 50, step=1),
                    class_="collapsible-control-body",
                ), open=False, class_="collapsible-control nested-control",
            ),
            ui.tags.details(
                ui.tags.summary("Polymer"),
                ui.div(
                    *[ui.input_slider(f"design_polymer_{name}", f"{name} (%)", 0, 100, value, step=1)
                      for name, value in DEFAULT_POLYMER_PERCENTAGES.items()],
                    ui.input_action_button("design_reset_polymer_mix", "Reset to default %", class_="btn-sm btn-outline-secondary"),
                    class_="collapsible-control-body",
                ), open=False, class_="collapsible-control nested-control",
            ),
        ),
        ui.panel_conditional(
            "input.design_plastic_type === 'macroplastic'",
            ui.input_radio_buttons(
                "design_macro_mode", None,
                choices={
                    "top_five": "Top 5 most common river litter items",
                    "individual": "Individual litter items",
                    "grouped": "Grouped litter items",
                },
                selected="top_five", inline=False,
            ),
            ui.panel_conditional(
                "input.design_macro_mode === 'grouped'",
                ui.input_checkbox_group("design_macro_categories", "Classes", choices=macro_group_labels, selected=list(macro_group_labels.keys())),
            ),
            ui.panel_conditional(
                "input.design_macro_mode === 'individual' || input.design_macro_mode === 'top_five'",
                ui.input_selectize("design_macro_common_names", "Individual litter items", choices=macro_common_names,
                    selected=["Soft plastic pieces/films 0.5-2.5 cm"], multiple=True,
                    options={"placeholder": "Search or scroll through litter items", "plugins": ["remove_button"], "dropdownParent": "body"}),
            ),
        ),
        ui.panel_conditional(
            "input.design_plastic_type === 'microplastic'",
            ui.tags.details(
                ui.tags.summary("How these values are used"),
                ui.p("Particle size, shape, and polymer type are used to estimate settling or rising velocity. Together with the selected flow conditions, this velocity is used to calculate the Rouse number and predict each particle’s vertical distribution through the water column. The predicted profiles are then used to estimate the fraction captured (or missed) by the selected sampling depth interval."),
                ui.p(
                    "The typical microplastic population is based on distribution data from ",
                    ui.tags.a("Kooi et al. (2019)", href="https://doi.org/10.1021/acs.estlett.9b00379", target="_blank", rel="noopener"),
                    ".",
                ),
                ui.p(
                    "Polymer densities are based on data from the ",
                    ui.tags.a("MaTCH web tool", href="https://hannahhapich.shinyapps.io/match/", target="_blank", rel="noopener"),
                    " (",
                    ui.tags.a("Hapich et al., 2024", href="https://doi.org/10.1021/acs.est.4c02406", target="_blank", rel="noopener"),
                    ").",
                ),
                ui.p(
                    "Settling or rising velocity equations use ",
                    ui.tags.a("Dietrich (1982)", href="https://doi.org/10.1029/WR018i006p01615", target="_blank", rel="noopener"),
                    ", ",
                    ui.tags.a("Goral et al. (2023)", href="https://doi.org/10.1016/j.envres.2023.115783", target="_blank", rel="noopener"),
                    ", and ",
                    ui.tags.a("Yu et al. (2022)", href="https://doi.org/10.1016/j.marpolbul.2022.113449", target="_blank", rel="noopener"),
                    ".",
                ),
                class_="sampling-help",
            ),
        ),
        ui.panel_conditional(
            "input.design_plastic_type === 'macroplastic'",
            ui.tags.details(
                ui.tags.summary("How these values are used"),
                ui.p(
                    "The macrolitter options represent the 25 most common River-OSPAR litter items observed in rivers, accounting for approximately 80% of total litter (",
                    ui.tags.a(ui.tags.em("Lofty et al."), " 2026", href="https://doi.org/10.1021/acs.est.5c13126", target="_blank", rel="noopener"),
                    "). Settling and rising velocity data for these items are taken from ",
                    ui.tags.a(ui.tags.em("Lofty et al."), " 2026", href="https://doi.org/10.31223/X5TF4V", target="_blank", rel="noopener"),
                    ". These velocities, together with the selected flow conditions, are used to predict each item’s vertical distribution and the fraction captured (or missed) by the selected sampling depth interval.",
                ),
                class_="sampling-help",
            ),
        ),
        class_="sampling-plastic-controls",
    )


app_ui = ui.page_navbar(
    ui.nav_control(
        ui.tags.style(
            """
                    .bslib-sidebar-layout > .sidebar {
                        width: 360px !important;
                        min-width: 360px !important;
                        max-width: 360px !important;
                        max-height: calc(100vh - 72px);
                        overflow-y: auto;
                        overflow-x: hidden;
                    }
                    body {
                        font-size: 0.82rem;
                    }
                    .navbar, .nav-link {
                        font-size: 0.82rem;
                    }
                    .navbar-brand-home,
                    .navbar-brand-home:hover,
                    .navbar-brand-home:focus {
                        color: inherit;
                        text-decoration: none;
                        cursor: pointer;
                    }
                    #main_nav .nav-link[data-value="home"],
                    #main_nav .nav-link[data-value="explorer"] {
                        display: none !important;
                    }
                    .landing-page {
                        max-width: 900px;
                        margin: 3rem auto;
                        padding: 2.5rem;
                        text-align: center;
                    }
                    .landing-page h1 {
                        color: #173b53;
                        font-size: 2.6rem;
                        font-weight: 700;
                    }
                    .landing-page p {
                        font-size: 1.15rem;
                        line-height: 1.65;
                        max-width: 720px;
                        margin: 1.2rem auto 2rem;
                    }
                    .landing-start-button {
                        background: #0b7fc3;
                        border-color: #0b7fc3;
                        font-size: 1.05rem;
                        padding: 0.65rem 2rem;
                    }
                    .landing-page-actions {
                        display: flex;
                        justify-content: center;
                        gap: 1rem;
                        flex-wrap: wrap;
                        max-width: 700px;
                        margin-left: auto;
                        margin-right: auto;
                    }
                    .landing-page-option {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: space-between;
                        gap: 0.8rem;
                        height: 100%;
                        padding: 1.25rem;
                        border: 1px solid #c8dfef;
                        border-radius: 0.65rem;
                        background: #f8fbfd;
                    }
                    .landing-page-option p {
                        flex: 1 1 auto;
                        margin: 0;
                    }
                    .landing-logo-placeholder {
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        width: 160px;
                        height: 70px;
                        margin: 2rem 0.5rem 0;
                        border: 1px dashed #b8c5ce;
                        border-radius: 0.35rem;
                        color: #657580;
                        font-size: 0.8rem;
                    }
                    .explorer-page {
                        max-width: 1100px;
                        margin: 2rem auto;
                    }
                    .landing-logo {
                        width: 160px;
                        height: 70px;
                        object-fit: contain;
                        margin: 2rem 0.5rem 0;
                    }
                    h2 {
                        font-size: 1.35rem;
                        margin-bottom: 0.7rem;
                    }
                    h3 {
                        font-size: 1.08rem;
                    }
                    h4 {
                        font-size: 1.0rem;
                    }
                    .card-header {
                        font-size: 0.95rem;
                        font-weight: 650;
                    }
                    .equation {
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 0.32rem;
                        margin: 0.8rem 0 0.95rem;
                        padding: 0.55rem 0.75rem;
                        background: #f7f9fb;
                        border-left: 3px solid #3d6f8e;
                        border-radius: 0.2rem;
                        font-family: Georgia, "Times New Roman", serif;
                        font-size: 1.05rem !important;
                    }
                    .equation .fraction {
                        display: inline-flex;
                        flex-direction: column;
                        align-items: center;
                        line-height: 1.18;
                        vertical-align: middle;
                    }
                    .equation .fraction > span + span {
                        border-top: 1px solid currentColor;
                        margin-top: 0.12rem;
                        padding-top: 0.12rem;
                    }
                    .equation sub,
                    .equation sup {
                        font-size: 0.72em !important;
                    }
                    .sidebar .form-group,
                    .sidebar .shiny-input-container {
                        margin-bottom: 0.75rem;
                    }
                    .control-workflow {
                        border: 1px solid rgba(0,0,0,0.10);
                        border-radius: 0.65rem;
                        padding: 0.75rem 0.9rem;
                        background: rgba(0,0,0,0.025);
                        margin-bottom: 1rem;
                        font-size: 0.82rem;
                        line-height: 1.35;
                    }
                    .main .card {
                        margin-top: 1rem;
                    }
                    .helper-text {
                        color: #666;
                        font-size: 0.84rem;
                        margin-bottom: 0.6rem;
                    }
                    .sampling-note {
                        color: #555;
                        font-size: 0.84rem;
                        line-height: 1.32;
                        margin-top: 0.5rem;
                    }
                    .sidebar-range-block {
                        border: 1px solid rgba(0, 0, 0, 0.08);
                        border-radius: 0.55rem;
                        padding: 0.85rem 0.9rem 0.1rem 0.9rem;
                        margin-bottom: 0.8rem;
                        background: rgba(255, 255, 255, 0.65);
                    }
                    .collapsible-control {
                        border: 1px solid rgba(0, 0, 0, 0.12);
                        border-radius: 0.65rem;
                        background: rgba(0, 0, 0, 0.02);
                        margin-bottom: 1rem;
                        overflow: hidden;
                    }
                    .collapsible-control > summary {
                        cursor: pointer;
                        list-style: none;
                        font-weight: 650;
                        padding: 0.85rem 1rem;
                        user-select: none;
                    }
                    .collapsible-control > summary::-webkit-details-marker {
                        display: none;
                    }
                    .collapsible-control > summary::before {
                        content: "▶";
                        display: inline-block;
                        margin-right: 0.5rem;
                        transition: transform 0.15s ease-in-out;
                    }
                    .collapsible-control[open] > summary::before {
                        transform: rotate(90deg);
                    }
                    .collapsible-control > summary:hover {
                        background: rgba(0, 0, 0, 0.035);
                    }
                    .collapsible-control-body {
                        padding: 0 1rem 0.85rem 1rem;
                    }
                    .analysis-layout {
                        display: grid;
                        grid-template-columns: minmax(520px, 1fr) 390px;
                        gap: 1rem;
                        align-items: start;
                    }
                    .centre-analysis-panel {
                        min-width: 0;
                    }
                    .right-control-panel {
                        max-height: calc(100vh - 90px);
                        overflow-y: auto;
                        overflow-x: hidden;
                        position: sticky;
                        top: 1rem;
                        padding-right: 0.25rem;
                    }
                    .workflow-strip {
                        display: grid;
                        grid-template-columns: repeat(4, 1fr);
                        gap: 0.65rem;
                    }
                    .workflow-step {
                        border: 1px solid rgba(0,0,0,0.10);
                        border-radius: 0.65rem;
                        padding: 0.65rem 0.8rem;
                        background: rgba(0,0,0,0.025);
                        font-size: 0.84rem;
                    }
                    .warning-box {
                        border-left: 4px solid #b26a00;
                        background: rgba(255, 193, 7, 0.12);
                        padding: 0.7rem 0.9rem;
                        border-radius: 0.4rem;
                        margin: 0.75rem 0;
                    }

                    .sampling-left-tabs .nav-link {
                        padding: 0.35rem 0.45rem;
                        font-size: 0.76rem;
                    }

                    .sampling-left-tabs .card-body {
                        padding: 0.65rem 0.75rem;
                    }

                    .sampling-workflow-note {
                        border: 1px solid rgba(0,0,0,0.10);
                        border-radius: 0.65rem;
                        padding: 0.65rem 0.75rem;
                        background: rgba(0,0,0,0.025);
                        margin-bottom: 0.85rem;
                        font-size: 0.78rem;
                        line-height: 1.28;
                    }
                    .sampling-workflow-note code {
                        font-size: 0.76rem;
                    }
                    .sampling-page-intro {
                        max-width: 820px;
                        margin-top: 0;
                        margin-bottom: 0.9rem;
                        color: #4f5963;
                        font-size: 0.92rem;
                        line-height: 1.45;
                    }
                    ol.sampling-page-intro {
                        margin: 0 0 0.9rem;
                        padding-left: 0;
                        list-style-position: inside;
                    }
                    .sampling-help {
                        border: 0;
                        margin-bottom: 0.9rem;
                    }
                    .sampling-help > summary {
                        cursor: pointer;
                        color: #356a8a;
                        font-weight: 600;
                        font-size: 0.82rem;
                    }
                    .sampling-help-body {
                        margin-top: 0.55rem;
                        padding: 0.7rem 0.85rem;
                        border-radius: 0.55rem;
                        background: rgba(0,0,0,0.025);
                        font-size: 0.80rem;
                    }
                    .sampling-plastic-controls {
                        margin-top: 0;
                    }
                    .sampling-plastic-controls > h3 {
                        font-size: 0.95rem;
                        margin-bottom: 0.55rem;
                    }
                    .bslib-sidebar-layout > .sidebar.sampling-setup-sidebar:has(#design_tabs),
                    .bslib-sidebar-layout > .sidebar.sampling-setup-sidebar:has(#samp_explore_tabs),
                    .bslib-sidebar-layout > .sidebar.sampling-setup-sidebar:has(#samp_import_batch_excel) {
                        width: 500px !important;
                        min-width: 500px !important;
                        max-width: 500px !important;
                        overflow: visible !important;
                        padding: 1.5rem 1.25rem 2rem;
                    }
                    .bslib-sidebar-layout:has(#design_tabs),
                    .bslib-sidebar-layout:has(#samp_import_batch_excel) {
                        column-gap: 1.25rem;
                    }
                    .sampling-setup-sidebar:has(#design_tabs) .navset-card-tab {
                        margin-top: 0.35rem;
                        margin-bottom: 1rem;
                    }
                    .sampling-setup-sidebar:has(#design_tabs) .card {
                        margin-bottom: 1rem;
                    }
                    .sampling-setup-sidebar:has(#design_tabs) .card-body {
                        padding: 1rem;
                    }
                    .centre-analysis-panel {
                        padding: 1.75rem 1.25rem 2.5rem 0.5rem;
                    }
                    .bslib-sidebar-layout:has(#samp_import_batch_excel) .centre-analysis-panel {
                        padding-top: 0.75rem;
                    }
                    .bslib-sidebar-layout:has(#design_tabs) .centre-analysis-panel {
                        padding-top: 0.75rem;
                    }
                    .bslib-sidebar-layout > .sidebar.sampling-setup-sidebar:has(#samp_import_batch_excel) {
                        padding-top: 0.75rem;
                    }
                    .bslib-sidebar-layout > .sidebar.sampling-setup-sidebar:has(#design_tabs) {
                        padding-top: 0.75rem;
                    }
                    .centre-analysis-panel > .sampling-help {
                        margin: 0 0 1.25rem;
                    }
                    .centre-analysis-panel > .sampling-graph-controls {
                        margin: 0 0 1.15rem;
                    }
                    .centre-analysis-panel > .plot-card {
                        margin-top: 0 !important;
                    }
                    .centre-analysis-panel > .sampling-key-results {
                        margin-top: 1rem;
                    }
                    .design-export {
                        margin: 0.25rem 0 1rem;
                    }
                    body > .selectize-dropdown {
                        z-index: 100000 !important;
                    }
                    body > .selectize-dropdown .selectize-dropdown-content {
                        max-height: min(520px, 65vh) !important;
                        overflow-y: auto !important;
                    }
                    .secondary-disclosure {
                        margin-top: 0.9rem;
                        border: 1px solid rgba(0,0,0,0.10);
                        border-radius: 0.65rem;
                        overflow: hidden;
                        background: rgba(0,0,0,0.015);
                    }
                    .secondary-disclosure > summary {
                        cursor: pointer;
                        padding: 0.7rem 0.85rem;
                        font-weight: 600;
                        color: #4f5963;
                        font-size: 0.84rem;
                    }
                    .secondary-disclosure-body {
                        padding: 0 0.85rem 0.85rem 0.85rem;
                    }
                    .sampling-results-card {
                        margin-top: 1.25rem !important;
                    }
                    .sampling-graph-controls {
                        margin: 0.55rem 0;
                        padding: 0;
                    }
                    .sampling-graph-controls .shiny-input-container {
                        margin-bottom: 0.35rem;
                    }
                    .sampling-graph-controls {
                        display: flex;
                        align-items: center;
                        gap: 1rem;
                        flex-wrap: nowrap;
                    }
                    .sampling-graph-controls #samp_micro_detail .shiny-options-group {
                        display: flex;
                        flex-wrap: nowrap;
                        gap: 0.7rem;
                    }
                    .sampling-graph-controls #samp_micro_detail label {
                        white-space: nowrap;
                    }
                    .sampling-graph-controls .form-check {
                        white-space: nowrap;
                        margin-bottom: 0.25rem;
                    }
                    .batch-sample-row {
                        display: flex;
                        align-items: end;
                        gap: 1rem;
                        flex-wrap: wrap;
                    }
                    .batch-sample-row > .shiny-input-container:first-child {
                        flex: 0 1 360px;
                    }
                    .batch-sample-row .shiny-input-container {
                        margin-bottom: 0.35rem;
                    }
                    .batch-sample-row .shiny-options-group {
                        display: flex;
                        gap: 0.75rem;
                    }
                    #samp_show_iqr {
                        appearance: none;
                        width: 1.2rem;
                        height: 1.2rem;
                        border-radius: 50%;
                        border: 2px solid #8794a1;
                        vertical-align: middle;
                        cursor: pointer;
                    }
                    #samp_show_iqr:checked {
                        background: #087cc1;
                        border-color: #087cc1;
                        box-shadow: inset 0 0 0 3px #fff;
                    }
                    .sampling-results-card .nav-link {
                        font-size: 0.80rem;
                    }
                    .sampling-key-results {
                        margin: 0.35rem 0 0.85rem 0;
                    }
                    .sampling-key-results .bslib-value-box {
                        min-height: 110px;
                        height: 110px !important;
                        background: #e8f3fb !important;
                        color: #173b53 !important;
                        border: 1px solid #c8dfef;
                        box-shadow: none;
                    }
                    .sampling-key-results .bslib-value-box > .card-body {
                        padding: 12px 16px;
                    }
                    .sampling-key-results .value-box-area {
                        padding: 0;
                        gap: 0.4rem;
                        justify-content: center;
                    }
                    .sampling-key-results .value-box-title,
                    .sampling-key-results .value-box-value {
                        flex: 0 0 auto;
                        overflow-wrap: anywhere;
                    }
                    .sampling-key-results .value-box-title p,
                    .sampling-key-results .value-box-value p {
                        margin: 0;
                    }
                    .sampling-key-results .value-box-value {
                        font-size: 1.15rem;
                        line-height: 1.05;
                    }
                    .sampling-key-results .value-box-title {
                        font-size: 0.85rem;
                        line-height: 1.05;
                    }
                    .export-results-button {
                        background: #e8f3fb !important;
                        color: #173b53 !important;
                        border: 1px solid #c8dfef !important;
                        font-weight: 650;
                        box-shadow: none !important;
                    }
                    .export-results-button:hover,
                    .export-results-button:focus {
                        background: #dcecf7 !important;
                        color: #173b53 !important;
                        border-color: #b8d5e9 !important;
                    }
                    #samp_import_excel,
                    #samp_import_batch_excel,
                    #samp_import_flow_excel,
                    #samp_import_plastics_excel,
                    #samp_import_sample_excel,
                    #samp_import_excel .btn,
                    #samp_import_batch_excel .btn {
                        width: 100%;
                        background: #e8f3fb !important;
                        color: #173b53 !important;
                        border: 1px solid #c8dfef !important;
                        font-weight: 650;
                        box-shadow: none !important;
                    }
                    .btn-file:has(#samp_import_batch_excel) {
                        background: #e8f3fb !important;
                        color: #173b53 !important;
                        border: 1px solid #c8dfef !important;
                        font-weight: 650;
                        box-shadow: none !important;
                    }
                    .btn-file:has(#samp_import_batch_excel):hover,
                    .btn-file:has(#samp_import_batch_excel):focus {
                        background: #dcecf7 !important;
                        color: #173b53 !important;
                        border-color: #b8d5e9 !important;
                    }
                    #samp_import_excel:hover,
                    #samp_import_excel:focus,
                    #samp_import_batch_excel:hover,
                    #samp_import_batch_excel:focus,
                    #samp_import_excel .btn:hover,
                    #samp_import_excel .btn:focus,
                    #samp_import_batch_excel .btn:hover,
                    #samp_import_batch_excel .btn:focus {
                        background: #dcecf7 !important;
                        color: #173b53 !important;
                        border-color: #b8d5e9 !important;
                    }
                    label:has(#samp_import_excel) .btn-file {
                        background: #e8f3fb !important;
                        color: #173b53 !important;
                        border: 1px solid #c8dfef !important;
                        font-weight: 650;
                        box-shadow: none !important;
                    }
                    label:has(#samp_import_excel) .btn-file:hover,
                    label:has(#samp_import_excel) .btn-file:focus {
                        background: #dcecf7 !important;
                        color: #173b53 !important;
                        border-color: #b8d5e9 !important;
                    }
                    .shiny-input-container:has(#samp_import_excel) {
                        margin-top: 0.45rem;
                    }
                    .section-data-controls {
                        margin-top: 1rem;
                        padding-top: 0.85rem;
                        border-top: 1px solid rgba(0,0,0,0.10);
                    }
                    .section-data-controls .btn,
                    .section-data-controls .form-control {
                        width: 100%;
                    }
                    .section-data-controls .shiny-input-container {
                        margin-top: 0.45rem;
                    }
                    .section-data-controls .btn-file,
                    .section-data-controls .form-control,
                    .section-data-controls input[type=file] {
                        background: #e8f3fb !important;
                        color: #173b53 !important;
                        border: 1px solid #c8dfef !important;
                        font-weight: 650;
                        box-shadow: none !important;
                    }
                    #samp_left_tabs {
                        display: flex;
                        flex-wrap: nowrap !important;
                    }
                    #samp_left_tabs .nav-item {
                        flex: 0 1 auto;
                    }
                    #samp_left_tabs .nav-link {
                        white-space: nowrap;
                        padding: 0.42rem 0.56rem;
                        font-size: var(--app-font-standard) !important;
                    }
                    .sampling-key-group {
                        margin: 0.55rem 0 0.25rem 0;
                        color: #4f5963;
                        font-size: 0.82rem;
                        font-weight: 650;
                    }
                    .sampling-results-card shiny-data-frame
                    .shiny-data-grid > table > thead > tr > th {
                        white-space: normal !important;
                        overflow-wrap: anywhere;
                        word-break: normal;
                        line-height: 1.15;
                        vertical-align: middle;
                        height: auto !important;
                        min-height: 2.6rem;
                        padding-top: 0.45rem;
                        padding-bottom: 0.45rem;
                    }
                    .sampling-results-card shiny-data-frame
                    .shiny-data-grid > table > thead > tr > th > div {
                        white-space: normal !important;
                        overflow-wrap: anywhere;
                    }

                    .smart-table-card .datagrid,
                    .smart-table-card table,
                    .smart-table-card .rt-table {
                        font-size: 0.78rem;
                    }
                    .smart-table-card th,
                    .smart-table-card td {
                        white-space: normal !important;
                        word-break: normal;
                        overflow-wrap: anywhere;
                        line-height: 1.2;
                    }
                    .input-warning {
                        font-size: 0.76rem;
                        color: #666;
                        margin-bottom: 0.35rem;
                    }
                    .compact-note {
                        color: #555;
                        font-size: 0.76rem;
                        line-height: 1.25;
                        margin-top: 0.35rem;
                    }
                    #samp_macro_item_concentrations_ui.recalculating {
                        opacity: 1 !important;
                        pointer-events: auto !important;
                        transition: none !important;
                    }
                    .methods-content {
                        max-width: 900px;
                        margin: 0 auto;
                    }
                    .square-plot-card {
                        width: 100%;
                        max-width: 760px;
                        margin-left: auto;
                        margin-right: auto;
                    }
                    .square-plot-card .card-body {
                        display: block;
                    }
                    .square-plot-card img,
                    .square-plot-card canvas,
                    .square-plot-card svg {
                        max-width: 100%;
                        width: 100%;
                        height: auto;
                    }
                    .diagnostic-grid {
                        display: grid;
                        grid-template-columns: repeat(3, minmax(0, 1fr));
                        gap: 0.75rem;
                        max-width: 900px;
                        margin: 0.75rem auto 0 auto;
                    }
                    .smart-table-grid {
                        display: grid;
                        grid-template-columns: 1fr;
                        gap: 0.45rem;
                        margin-top: 0.55rem;
                    }
                    .smart-table-card .card-header {
                        font-size: 0.82rem;
                    }
                    .smart-table-card {
                        margin-top: 0.25rem !important;
                    }
                    .smart-table-card .card-body {
                        padding: 0.30rem 0.45rem;
                        font-size: 0.76rem;
                    }
                    .smart-table-card .datagrid {
                        max-height: 95px;
                    }
                    .mini-diagnostic-card .card-body {
                        padding: 0.45rem 0.55rem 0.55rem 0.55rem;
                    }
                    .mini-diagnostic-card .card-header {
                        font-size: 0.78rem;
                        padding: 0.45rem 0.6rem;
                    }
                    body {
                        --app-font-standard: 0.84rem;
                        --app-font-large: 1.10rem;
                        font-size: var(--app-font-standard) !important;
                    }
                    body * {
                        font-size: inherit !important;
                    }
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    h6,
                    .card-header,
                    .navbar-brand,
                    .sampling-key-results .value-box-value {
                        font-size: var(--app-font-large) !important;
                    }
                    .nav-link,
                    .btn,
                    button,
                    input,
                    select,
                    textarea,
                    label,
                    table,
                    th,
                    td,
                    code,
                    .sampling-key-results .value-box-title {
                        font-size: var(--app-font-standard) !important;
                    }
                    @media (max-width: 900px) {
                        .diagnostic-grid {
                            grid-template-columns: 1fr;
                        }
                    }
                    @media (max-width: 1300px) {
                        .analysis-layout {
                            grid-template-columns: 1fr;
                        }
                        .right-control-panel {
                            position: static;
                            max-height: none;
                        }
                        .workflow-strip {
                            grid-template-columns: 1fr;
                        }
                    }
                    """
        ),
    ),

    ui.nav_panel(
        "Home",
        ui.div(
            ui.markdown(landing_text),
            ui.layout_columns(
                ui.div(
                    ui.p(
                        "Use this page before field work to explore the vertical distribution of different plastics and to choose sampling depths suitable for the river and plastics you expect to collect."
                    ),
                    ui.input_action_button(
                        "open_sample_design",
                        "Sample design",
                        class_="btn btn-primary landing-start-button",
                    ),
                    class_="landing-page-option",
                ),
                ui.div(
                    ui.p(
                        "Use this page after field work to convert plastic concentrations measured at one or more depths to a depth-average concentration and plastic load."
                    ),
                    ui.input_action_button(
                        "open_sample_correction",
                        "Sample correction",
                        class_="btn btn-primary landing-start-button",
                    ),
                    class_="landing-page-option",
                ),
                col_widths=[6, 6],
                class_="landing-page-actions",
            ),
            ui.div(
                landing_logo("kit.svg", "Karlsruhe Institute of Technology"),
                landing_logo("humboldt.jpeg", "Humboldt"),
                landing_logo("ucr.png", "UC Riverside"),
                landing_logo("imperial.png", "Imperial College London"),
                landing_logo("moore.jpg", "Moore Institute"),
            ),
            ui.p("Citation: XXX", class_="compact-note"),
            class_="landing-page",
            id="river-plast-home",
        ),
        value="home",
    ),


    ui.nav_panel(
        "Sample design",
        ui.page_sidebar(
            ui.sidebar(
                ui.p(
                    "Use this page before field work to explore the vertical distribution of different plastics and to choose sampling depths which are suitable for the river and plastics you expect to collect.",
                    class_="sampling-page-intro",
                ),
                ui.tags.details(
                    ui.tags.summary("How to use this page"),
                    ui.div(
                        ui.tags.ol(
                            ui.tags.li("Enter the expected river flow conditions at your sampling location."),
                            ui.tags.li("Describe the microplastics or macrolitter you want to target, or use a typical plastic population."),
                            ui.tags.li("Choose potential sampling depths."),
                            ui.tags.li("Use the expected capture result to select a depth that represents the target plastics."),
                            ui.tags.li("Export the data."),
                        ),
                        class_="sampling-help-body",
                    ),
                    class_="sampling-help",
                ),
                ui.navset_card_tab(
                    ui.nav_panel(
                        "Flow",
                        ui.h6("Expected river flow conditions"),
                        ui.tags.details(
                            ui.tags.summary("How to obtain these values"),
                            ui.tags.ul(
                                ui.tags.li(ui.tags.strong("Width (m): "), "Obtain from field measurements, aerial imagery, or satellite data."),
                                ui.tags.li(ui.tags.strong("Depth (m): "), "Obtain from field measurements or bathymetric surveys."),
                                ui.tags.li(ui.tags.strong("Slope (–): "), "Obtain from field elevation measurements, digital elevation models, or satellite water-surface data."),
                                ui.tags.li(ui.tags.strong("Discharge (m³/s): "), "Obtain from field measurements, nearby gauging stations, or hydrodynamic models."),
                            ),
                            class_="sampling-help",
                        ),
                        ui.input_numeric("design_river_width", "Width (m)", value=20.0, min=0.01, step=0.1),
                        ui.input_numeric("design_river_depth", "Depth (m)", value=1.0, min=0.01, step=0.01),
                        ui.input_numeric("design_slope", "Slope (-)", value=0.00050, min=0.00001, step=0.00001),
                        ui.input_numeric("design_discharge", "Discharge (m³/s)", value=20.0, min=0.0, step=0.1),
                        ui.input_checkbox("design_direct_ustar", "Enter shear velocity directly", False),
                        ui.panel_conditional("input.design_direct_ustar", ui.input_numeric("design_u_star", "Shear velocity, u* (m/s)", value=0.15, min=0.001, step=0.001)),
                        ui.input_action_button("design_apply_flow", "Apply flow values", class_="btn-primary"),
                    ),
                    ui.nav_panel("Plastics", sample_design_plastic_controls_ui()),
                    ui.nav_panel(
                        "Sampling",
                        ui.h6("Proposed sampling depth"),
                        ui.p("Use the sliders to select your proposed sampling depth with nets, pumps, samplers or buckets.", class_="helper-text", style="margin-bottom: 1rem;"),
                        ui.input_checkbox(
                            "design_show_sampling_interval",
                            "Show sampling interval on plot",
                            True,
                        ),
                        ui.output_ui("design_net_controls"),
                    ),
                    ui.nav_panel(
                        "Advanced",
                        ui.input_slider("design_a_bed_frac", ui.tags.span("Bed reference height ", ui.tags.i("a"), ui.tags.sub("bed"), "/", ui.tags.i("H")), 0.01, 0.30, 0.05, step=0.01),
                        ui.input_slider("design_a_surf_frac", ui.tags.span("Surface reference offset ", ui.tags.i("a"), ui.tags.sub("surf"), "/", ui.tags.i("H")), 0.01, 0.30, 0.05, step=0.01),
                        ui.input_slider("design_iqr_percentiles", "Particle variability percentiles", 0, 100, (25, 75), step=1),
                    ),
                    id="design_tabs",
                ), width="500px", open="always", class_="sampling-setup-sidebar",
            ),
            ui.div(
                ui.tags.details(
                    ui.tags.summary("How to read this graph"),
                    ui.div(
                        ui.p("This graph shows how plastic concentration changes with depth in a river water column."),
                        ui.p("The y-axis is river depth."),
                        ui.p("The x-axis shows the concentration relative to the maximum. 1 is the maximum concentration. 0.5 means half of the maximum concentration. 0.1 means 10% of the maximum concentration."),
                        ui.p("Each line represents the median predicted concentration profile for the selected plastic fraction."),
                        ui.p("When shown, dashed lines mark the proposed sampling depth."),
                        ui.p("The app does not report a corrected depth-average concentration or load when the median captured fraction is below 5%, because the sampled depth is not considered sufficiently representative of that plastic fraction."),
                        ui.p("Stronger turbulent mixing, typically associated with faster flow, produces a more uniform vertical distribution of plastics."),
                        ui.p("Under weaker mixing, plastics become more vertically separated according to their buoyancy. Sinking plastics concentrate near the bottom. Floating plastics concentrate near the surface."),
                        class_="sampling-help-body",
                    ), class_="sampling-help",
                ),
                ui.panel_conditional(
                    "input.design_plastic_type === 'microplastic'",
                    ui.div(
                        ui.input_radio_buttons("design_micro_detail", None, {"total": "Total", "summary": "Buoyant and sinking", "size": "Size groups", "polymer": "Polymers"}, selected="total", inline=True),
                        class_="sampling-graph-controls",
                    ),
                ),
                ui.card(ui.output_plot("profile_plot_design", height="400px"), full_screen=True, class_="plot-card square-plot-card"),
                ui.output_ui("design_key_results"),
                ui.div(
                    ui.download_button("download_design_parameters_excel", "Export data", class_="btn-primary"),
                    class_="design-export",
                ),
                ui.card(
                    ui.card_header("Capture results by fraction"),
                    ui.output_data_frame("design_capture_results"),
                    class_="sampling-results-card",
                ),
                class_="centre-analysis-panel",
            ),
        ), value="sample_design",
    ),

    ui.nav_panel(
        "Sample correction",
        ui.page_sidebar(
            ui.sidebar(
                ui.tags.span(id="river-plast-sampling"),
                ui.p(
                    "Use this page after field work to convert plastic concentrations measured at one or more discrete depths to a depth-average concentration and plastic load.",
                    class_="sampling-page-intro",
                ),
                ui.tags.ol(
                    ui.tags.li("Download the example datasheet."),
                    ui.tags.li("Enter measured river flow, sampling-depths and plastic data from your sampling campaign."),
                    ui.tags.li("Upload the completed datasheet."),
                    ui.tags.li("Review the predicted profile and corrected results."),
                    ui.tags.li("Export the results."),
                    class_="sampling-page-intro",
                ),
                ui.div(
                        ui.h6("Download example datasheet"),
                        ui.download_button(
                            "download_batch_example_excel",
                            "Download example datasheet",
                            class_="export-results-button w-100",
                        ),
                        ui.div(style="height: 1rem;"),
                        ui.h6("Upload datasheet"),
                        ui.input_file(
                            "samp_import_batch_excel",
                            None,
                            accept=[".xlsx"],
                            button_label="Upload datasheet",
                            placeholder="Choose a RIVER-PLAST workbook",
                        ),
                        ui.h6("Export results"),
                        ui.download_button(
                            "download_batch_results_excel",
                            "Export results",
                            class_="export-results-button w-100",
                        ),
                ),
                width="500px",
                open="always",
                class_="sampling-setup-sidebar",
            ),

            ui.div(
                ui.output_ui("batch_visualisation_ui"),
                class_="centre-analysis-panel",
            ),
        ),
        value="calculator",
    ),


    ui.nav_panel(
        "Explorer",
        ui.page_sidebar(
            ui.sidebar(
                ui.tags.span(id="river-plast-explorer"),
                ui.tags.details(
                    ui.tags.summary("How to use this page"),
                    ui.div(
                        ui.markdown(
                            """
1. Set the flow conditions from the sampling campaign and click 'Apply flow values'.
2. Choose either microplastics or macrolitter and define the population from your collected sample.
3. Set the depth at which the sample was collected and the measured concentration.
4. Visualise the concentration profile and view estimated depth-average concentration and load.
5. Export the data.
                            """
                        ),
                        class_="sampling-help-body",
                    ),
                    class_="sampling-help",
                ),
                ui.navset_card_tab(
                        ui.nav_panel(
                            "Flow",
                            ui.h6("River flow conditions"),
                            ui.input_numeric(
                                "samp_river_width", "Width (m)",
                                value=20.0, min=0.01, step=0.1,
                            ),
                            ui.input_numeric(
                                "samp_river_depth", "Depth (m)",
                                value=1.0, min=0.01, step=0.01,
                            ),
                            ui.input_numeric(
                                "samp_slope", "Slope (-)",
                                value=0.00050, min=0.00001, step=0.00001,
                            ),
                            ui.input_numeric(
                                "samp_discharge", "Discharge (m³/s)",
                                value=20.0, min=0.0, step=0.1,
                            ),
                            ui.input_checkbox(
                                "samp_direct_ustar",
                                "Enter shear velocity directly",
                                False,
                            ),
                            ui.panel_conditional(
                                "input.samp_direct_ustar",
                                ui.input_numeric(
                                    "samp_u_star", "Shear velocity, u* (m/s)",
                                    value=0.15, min=0.001, step=0.001,
                                ),
                                ui.p("Overrides the width, depth and slope calculation.", class_="helper-text"),
                            ),
                            ui.input_action_button(
                                "samp_apply_flow",
                                "Apply flow values",
                                class_="btn-primary",
                            ),
                        ),
                        ui.nav_panel(
                            "Plastics",
                            sampling_plastic_controls_ui(),
                        ),
                        ui.nav_panel(
                            "Sample",
                            ui.input_slider(
                                "samp_net_z_interval",
                                "Relative sampling depth",
                                min=0.0,
                                max=1.0,
                                value=(0.85, 0.95),
                                step=0.01,
                            ),
                            ui.input_checkbox(
                                "samp_show_sampling_interval",
                                "Show sampling interval on plot",
                                True,
                            ),
                            ui.panel_conditional(
                                "!input.samp_select_macroplastics",
                                ui.input_numeric(
                                    "samp_measured_concentration",
                                    "Measured concentration in sample",
                                    value=10.0,
                                    min=0.0,
                                    step=0.1,
                                ),
                            ),
                            ui.panel_conditional(
                                "!input.samp_select_macroplastics && input.samp_add_size_group",
                                ui.input_numeric(
                                    "samp_measured_concentration_2",
                                    "Group 2 measured concentration",
                                    value=10.0,
                                    min=0.0,
                                    step=0.1,
                                ),
                            ),
                            ui.output_ui("samp_macro_item_concentrations_ui"),
                            ui.input_select(
                                "samp_concentration_units",
                                "Concentration units",
                                choices={
                                    "particles/m3": "particles/m³",
                                    "g/m3": "g/m³",
                                },
                                selected="particles/m3",
                            ),
                            ui.panel_conditional(
                                "input.samp_select_macroplastics && input.samp_macro_mode === 'grouped'",
                                ui.h6("Measured concentration by class"),
                                *[
                                    ui.input_numeric(
                                        macro_group_concentration_input_id(group_key),
                                        group_label,
                                        value=0.0,
                                        min=0.0,
                                        step=0.01,
                                    )
                                    for group_key, group_label in macro_group_labels.items()
                                ],
                            ),
                        ),
                        ui.nav_panel(
                            "Advanced",
                            ui.input_slider(
                                "samp_a_bed_frac",
                                ui.tags.span(
                                    "Bed reference height ",
                                    ui.tags.i("a"),
                                    ui.tags.sub("bed"),
                                    "/",
                                    ui.tags.i("H"),
                                ),
                                min=0.01,
                                max=0.30,
                                value=0.05,
                                step=0.01,
                            ),
                            ui.input_slider(
                                "samp_a_surf_frac",
                                ui.tags.span(
                                    "Surface reference offset ",
                                    ui.tags.i("a"),
                                    ui.tags.sub("surf"),
                                    "/",
                                    ui.tags.i("H"),
                                ),
                                min=0.01,
                                max=0.30,
                                value=0.05,
                                step=0.01,
                            ),
                            ui.input_slider(
                                "samp_iqr_percentiles",
                                "Particle variability percentiles",
                                min=0,
                                max=100,
                                value=(25, 75),
                                step=1,
                            ),
                        ),
                    id="samp_explore_tabs",
                ),
                width="500px",
                class_="sampling-setup-sidebar",
            ),
            ui.div(
                ui.output_ui("sampling_caution"),
                ui.tags.details(
                    ui.tags.summary("How to read this graph"),
                    ui.div(
                        ui.tags.ul(
                            ui.tags.li("This graph shows how plastic concentration changes in a river water column."),
                            ui.tags.li("The y-axis is river depth: y = 0 is the riverbed and y = 1 is the water surface."),
                            ui.tags.li("The x-axis shows concentration relative to the maximum. x = 1 is the maximum concentration. x = 0.6 is 60% of the maximum. x = 0.1 is 10% of the maximum and so on."),
                            ui.tags.li("When shown, dashed lines mark the sampled depth interval."),
                            ui.tags.li("The result boxes show estimated particles captured, depth-average concentration and load. See About & Methods for the calculations."),
                        ),
                        class_="sampling-help-body",
                    ),
                    class_="sampling-help",
                ),
                ui.panel_conditional(
                    "input.samp_select_microplastics",
                    ui.div(
                        ui.input_radio_buttons(
                            "samp_micro_detail",
                            None,
                            choices={
                                "total": "Total",
                                "summary": "Buoyant and sinking",
                                "size": "Size groups",
                                "polymer": "Polymers",
                            },
                            selected="total",
                            inline=True,
                        ),
                        ui.input_checkbox(
                            "samp_show_iqr",
                            "Show 25–75% particle variability",
                            False,
                        ),
                        class_="sampling-graph-controls",
                    ),
                ),
                ui.card(
                    ui.output_plot("profile_plot_sampling", height="400px"),
                    full_screen=True,
                    class_="plot-card square-plot-card",
                ),
                ui.output_ui("sampling_key_results"),
                ui.tags.details(
                    ui.tags.summary("Tables"),
                    ui.div(
                        ui.navset_card_tab(
                        ui.nav_panel(
                            "Captured fraction",
                            ui.output_data_frame("net_sampling_results"),
                            ui.download_button("download_net_sampling_results_csv", "Download CSV", class_="btn-sm btn-outline-secondary"),
                        ),
                        ui.nav_panel(
                            "Corrected concentration",
                            ui.output_data_frame("sampling_correction_results"),
                            ui.download_button("download_depth_average_results_csv", "Download CSV", class_="btn-sm btn-outline-secondary"),
                        ),
                        ui.nav_panel(
                            "Estimated load",
                            ui.output_data_frame("discharge_load_results"),
                            ui.download_button("download_load_results_csv", "Download CSV", class_="btn-sm btn-outline-secondary"),
                        ),
                        ),
                        class_="sampling-results-card",
                    ),
                    open=False,
                    class_="secondary-disclosure",
                ),
                class_="centre-analysis-panel",
            ),
        ),
        value="explorer",
    ),


    ui.nav_panel(
        "About & Methods",
        ui.div(
            ui.tags.span(id="river-plast-methods"),
            ui.card(
                ui.markdown(methods_text),
            ),
            class_="methods-content",
        ),
        value="methods",
    ),

    title=ui.tags.a(
        "RIVER-PLAST: Riverine Plastic Load Assessment and Sampling Tool",
        href="/",
        class_="navbar-brand-home",
    ),
    id="main_nav",
    selected="home",
)

# ============================================================
# SHINY SERVER
# ============================================================
def server(input: Inputs, output: Outputs, session: Session):

    applied_sampling_flow = reactive.Value(
        {
            "u_star": calculate_shear_velocity_from_slope_radius(
                hydraulic_radius=20.0 / 22.0,
                slope=0.00050,
            ),
            "discharge": 20.0,
            "mode": "hydraulic",
            "river_width_m": 20.0,
            "river_depth_m": 1.0,
            "hydraulic_radius_m": 20.0 / 22.0,
            "slope": 0.00050,
            "direct_u_star_m_s": 0.15,
        }
    )
    applied_design_flow = reactive.Value(
        {
            "u_star": calculate_shear_velocity_from_slope_radius(
                hydraulic_radius=20.0 / 22.0, slope=0.00050
            ),
            "discharge": 20.0,
            "river_depth_m": 1.0,
        }
    )
    design_net_count = reactive.Value(1)

    imported_macro_concentrations = reactive.Value({})
    imported_sample_metadata = reactive.Value(None)
    imported_particle_records = reactive.Value(None)
    imported_batch_workbook = reactive.Value({})
    imported_batch_results = reactive.Value(pd.DataFrame())
    imported_batch_particles = reactive.Value(pd.DataFrame())

    @reactive.Effect
    @reactive.event(input.open_sample_design)
    def _open_sample_design():
        ui.update_navs("main_nav", selected="sample_design", session=session)

    @reactive.Effect
    @reactive.event(input.open_sample_correction)
    def _open_sample_correction():
        ui.update_navs("main_nav", selected="calculator", session=session)

    def _number(row: pd.Series, column: str, label: str) -> float:
        """Read one required finite numeric value from an imported sheet."""
        if column not in row.index:
            raise ValueError(f"Missing '{column}' in {label}.")
        value = pd.to_numeric(row[column], errors="coerce")
        if not np.isfinite(value):
            raise ValueError(f"'{column}' in {label} must be a number.")
        return float(value)

    def _apply_flow_row(flow: pd.Series) -> None:
        """Validate and apply one long-form flow record."""
        width = _number(flow, "river_width_m", "Flow data")
        depth = _number(flow, "river_depth_m", "Flow data")
        slope = _number(flow, "slope", "Flow data")
        discharge = _number(flow, "discharge_m3_s", "Flow data")
        direct = pd.to_numeric(
            pd.Series([flow.get("direct_u_star_m_s", np.nan)]), errors="coerce"
        ).iloc[0]
        direct_u_star = float(direct) if np.isfinite(direct) and direct > 0 else 0.15
        mode = "direct" if np.isfinite(direct) and direct > 0 else "hydraulic"
        if width <= 0 or depth <= 0 or slope <= 0 or discharge < 0:
            raise ValueError("Flow inputs contain an invalid value.")

        ui.update_numeric("samp_river_width", value=width, session=session)
        ui.update_numeric("samp_river_depth", value=depth, session=session)
        ui.update_numeric("samp_slope", value=slope, session=session)
        ui.update_numeric("samp_discharge", value=discharge, session=session)
        ui.update_checkbox("samp_direct_ustar", value=mode == "direct", session=session)
        ui.update_numeric("samp_u_star", value=direct_u_star, session=session)

        hydraulic_radius = width * depth / (width + 2.0 * depth)
        u_star = direct_u_star if mode == "direct" else calculate_shear_velocity_from_slope_radius(
            hydraulic_radius=hydraulic_radius, slope=slope
        )
        applied_sampling_flow.set(
            {
                "u_star": u_star,
                "discharge": discharge,
                "mode": mode,
                "river_width_m": width,
                "river_depth_m": depth,
                "hydraulic_radius_m": hydraulic_radius,
                "slope": slope,
                "direct_u_star_m_s": direct_u_star,
            }
        )

    def _build_batch_results(
        flow_data: pd.DataFrame,
        sample_data: pd.DataFrame,
        particle_data: pd.DataFrame,
        a_bed_frac: float,
        a_surf_frac: float,
    ) -> pd.DataFrame:
        """Calculate one result set per sample and plastic type."""
        flow_lookup = flow_data.set_index("sample_id")
        result_frames = []
        for sample_id, depth_samples in sample_data.groupby("sample_id", sort=False):
            sample = depth_samples.iloc[0].copy()
            sample_id = str(sample_id)
            river_id = str(sample["river_id"])
            sample["sample_z_min"] = float(depth_samples["sample_z_min"].min())
            sample["sample_z_max"] = float(depth_samples["sample_z_max"].max())
            sample["sampled_volume_m3"] = float(depth_samples["sampled_volume_m3"].sum())
            depth_intervals = [
                (float(row["sample_z_min"]), float(row["sample_z_max"]))
                for _, row in depth_samples.iterrows()
            ]
            depth_volumes = [
                float(row["sampled_volume_m3"])
                for _, row in depth_samples.iterrows()
            ]
            if sample_id not in flow_lookup.index:
                continue
            flow = flow_lookup.loc[sample_id]
            if isinstance(flow, pd.DataFrame):
                flow = flow.iloc[0]
            width = _number(flow, "river_width_m", "Flow")
            depth = _number(flow, "river_depth_m", "Flow")
            slope = _number(flow, "slope", "Flow")
            discharge = _number(flow, "discharge_m3_s", "Flow")
            direct = pd.to_numeric(pd.Series([flow.get("direct_u_star_m_s", np.nan)]), errors="coerce").iloc[0]
            u_star = float(direct) if np.isfinite(direct) and direct > 0 else calculate_shear_velocity_from_slope_radius(
                width * depth / (width + 2 * depth), slope
            )
            z_min = _number(sample, "sample_z_min", "Sample")
            z_max = _number(sample, "sample_z_max", "Sample")
            volume = _number(sample, "sampled_volume_m3", "Sample")
            records = particle_data[particle_data["sample_id"].astype(str) == sample_id].copy()
            for plastic_type, rows in records.groupby(
                records["plastic_type"].astype(str).str.lower().str.rstrip("s")
            ):
                weights = pd.to_numeric(rows["count_weight"], errors="coerce")
                if weights.isna().any() or (weights <= 0).any():
                    continue
                concentration = float(weights.sum() / volume)
                if plastic_type == "microplastic":
                    polymers = rows["polymer"].map(standardise_polymer_name)
                    valid = polymers.notna()
                    if not valid.any():
                        continue
                    rows = rows.loc[valid]
                    weights = weights.loc[valid]
                    polymers = polymers.loc[valid]
                    input_polymer_names = rows["polymer"].astype(str).str.strip()
                    sizes = pd.to_numeric(rows["size_um"], errors="coerce")
                    shapes = rows["shape"].astype(str).str.lower().replace({"fiber": "fibre"})
                    valid = sizes.notna() & sizes.between(20, 5000) & shapes.isin({"fibre", "fragment"})
                    if not valid.any():
                        continue
                    weights = weights.loc[valid]
                    polymers = polymers.loc[valid]
                    input_polymer_names = input_polymer_names.loc[valid]
                    sizes = sizes.loc[valid]
                    shapes = shapes.loc[valid]
                    concentration = float(weights.sum() / volume)
                    total_weight = float(weights.sum())
                    polymer_mix = {
                        name: 100 * float(weights[polymers == name].sum()) / total_weight
                        for name in DEFAULT_POLYMER_PERCENTAGES
                    }
                    fibre_percent = 100 * float(weights[shapes == "fibre"].sum()) / total_weight
                    lo, hi = float(sizes.min()), float(sizes.max())
                    if lo == hi:
                        lo, hi = max(20.0, lo - 10), min(5000.0, hi + 10)
                    micro_df = generate_synthetic_microplastics(
                        n_particles=5000,
                        size_ranges_um=[(lo, hi)],
                        polymer_percentages=polymer_mix,
                        fiber_percent=fibre_percent,
                        seed=42,
                    )
                    result = sampling_correction_table(
                        micro_ranges=[("synthetic MP", lo, hi)], macro_selected=[],
                        macro_items_selected=[], use_macro_items=False, u_star=u_star,
                        micro_df=micro_df, H=depth, a_bed_frac=a_bed_frac,
                        a_surf_frac=a_surf_frac, net_z_min=z_min,
                        net_z_max=z_max, net_z_intervals=depth_intervals, net_sample_volumes=depth_volumes, measured_concentration=concentration,
                        concentration_units="particles/m3", include_discharge=True,
                        discharge=discharge, iqr_lower=25, iqr_upper=75,
                    )
                    result_sets = [("Micro total", result)]
                    beta = calculate_micro_rouse_mean(u_star, micro_df=micro_df)

                    def add_micro_export_group(
                        label: str,
                        subset: pd.DataFrame,
                        result_set: str,
                        concentration_fraction: float | None = None,
                    ) -> None:
                        if subset.empty:
                            return
                        concentration_subset = concentration * (
                            len(subset) / len(micro_df)
                            if concentration_fraction is None
                            else concentration_fraction
                        )
                        group_result = sampling_correction_table(
                            micro_ranges=[("synthetic MP", lo, hi)], macro_selected=[],
                            macro_items_selected=[], use_macro_items=False, u_star=u_star,
                            micro_df=subset, H=depth, a_bed_frac=a_bed_frac,
                            a_surf_frac=a_surf_frac, net_z_min=z_min,
                            net_z_max=z_max, net_z_intervals=depth_intervals, net_sample_volumes=depth_volumes, measured_concentration=concentration_subset,
                            concentration_units="particles/m3", include_discharge=True,
                            discharge=discharge, iqr_lower=25, iqr_upper=75,
                        )
                        group_result["Group"] = label
                        result_sets.append((result_set, group_result))

                    add_micro_export_group(
                        "Microplastics: buoyant",
                        micro_df[np.isfinite(beta) & (beta < 0)],
                        "Micro buoyant",
                    )
                    add_micro_export_group(
                        "Microplastics: sinking",
                        micro_df[np.isfinite(beta) & (beta >= 0)],
                        "Micro sinking",
                    )
                    for lower, upper, label in [
                        (20, 100, "Microplastics: 20–100 µm"),
                        (100, 300, "Microplastics: 100–300 µm"),
                        (300, 1000, "Microplastics: 300 µm–1 mm"),
                        (1000, 3000, "Microplastics: 1–3 mm"),
                        (3000, 5000, "Microplastics: 3–5 mm"),
                    ]:
                        add_micro_export_group(
                            label,
                            micro_df[micro_df["size_um"].between(lower, upper)],
                            "Micro size groups",
                        )
                    for polymer_name in input_polymer_names.drop_duplicates():
                        source_mask = input_polymer_names == polymer_name
                        model_polymer = polymers.loc[source_mask].iloc[0]
                        add_micro_export_group(
                            f"Microplastics: polymer {polymer_name}",
                            micro_df[micro_df["polymer"] == model_polymer],
                            "Micro polymers",
                            concentration_fraction=float(weights.loc[source_mask].sum() / total_weight),
                        )
                elif plastic_type == "macroplastic" and "macro_item" in rows:
                    item_counts = {
                        item: float(weights[rows["macro_item"].astype(str) == item].sum() / volume)
                        for item in rows["macro_item"].dropna().astype(str).unique()
                        if item in macro_common_names
                    }
                    if not item_counts:
                        continue
                    result = macro_item_correction_table(
                        item_concentrations=item_counts, u_star=u_star, H=depth,
                        a_bed_frac=a_bed_frac, a_surf_frac=a_surf_frac,
                        net_z_min=z_min, net_z_max=z_max, net_z_intervals=depth_intervals, net_sample_volumes=depth_volumes, concentration_units="particles/m3",
                        include_discharge=True, discharge=discharge, iqr_lower=25, iqr_upper=75,
                    )
                    result_sets = [("Macrolitter", result)]
                else:
                    continue
                for result_set, result in result_sets:
                    result.insert(0, "result_set", result_set)
                    result.insert(1, "plastic_type", plastic_type)
                    result.insert(2, "sample_id", sample_id)
                    result.insert(3, "river_id", river_id)
                    result_frames.append(result)
        return pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()

    @reactive.Effect
    @reactive.event(input.samp_import_batch_excel)
    def _import_batch_datasheet():
        uploaded = input.samp_import_batch_excel()
        if not uploaded:
            return
        try:
            sheets = pd.read_excel(
                uploaded[0]["datapath"],
                sheet_name=["Flow", "Sample", "microplastics", "macroplastic", "advanced"],
            )
            flow_data, sample_data = sheets["Flow"], sheets["Sample"]
            micro_data, macro_data = sheets["microplastics"], sheets["macroplastic"]
            advanced_data = sheets["advanced"].dropna(how="all")
            required_flow = {"river_id", "sample_id", "river_width_m", "river_depth_m", "slope", "discharge_m3_s"}
            required_sample = {"sample_id", "river_id", "sample_z_min", "sample_z_max", "sampled_volume_m3"}
            required_micro = {"river_id", "sample_id", "particle_id", "polymer", "size_um", "shape"}
            required_macro = {"river_id", "sample_id"}
            required_advanced = {"a_bed", "a_surf"}
            missing_advanced = required_advanced.difference(advanced_data.columns)
            if advanced_data.empty or missing_advanced:
                raise ValueError("advanced is missing: a_bed, a_surf.")
            advanced = advanced_data.iloc[0]
            a_bed_frac = _number(advanced, "a_bed", "advanced")
            a_surf_frac = _number(advanced, "a_surf", "advanced")
            if a_bed_frac <= 0 or a_surf_frac <= 0 or a_bed_frac + a_surf_frac >= 1:
                raise ValueError("advanced a_bed and a_surf must be positive and sum to less than 1.")
            for label, frame, columns in [
                ("Flow", flow_data, required_flow), ("Sample", sample_data, required_sample),
                ("microplastics", micro_data, required_micro), ("macroplastic", macro_data, required_macro)
            ]:
                missing = columns.difference(frame.columns)
                if missing or (label != "macroplastic" and frame.empty):
                    raise ValueError(f"{label} is missing: {', '.join(sorted(missing))}.")
            # Long-form field datasets often repeat the same flow and sampling
            # values for every particle. One unique row per sample is needed here.
            flow_data = flow_data.drop_duplicates(subset=["river_id", "sample_id"], keep="first")
            sample_data = sample_data.drop_duplicates(subset=["river_id", "sample_id", "sampling_depth_id"], keep="first") if "sampling_depth_id" in sample_data else sample_data
            sheets["Flow"] = flow_data
            sheets["Sample"] = sample_data
            if not sample_data["sample_id"].astype(str).isin(flow_data["sample_id"].astype(str)).all():
                raise ValueError("Every Sample sample_id must appear in Flow.")
            known_sample_ids = set(sample_data["sample_id"].astype(str))
            if not micro_data["sample_id"].astype(str).isin(known_sample_ids).all():
                raise ValueError("Every microplastic sample_id must appear in Sample.")
            # An empty macroplastic workflow is valid. Ignore leftover example
            # rows that are not linked to the imported sampling dataset.
            macro_data = macro_data[
                macro_data["sample_id"].astype(str).isin(known_sample_ids)
            ].copy()
            micro_data = micro_data.copy()
            micro_data["plastic_type"] = "microplastic"
            if "count_weight" not in micro_data:
                micro_data["count_weight"] = 1.0
            macro_data = macro_data.copy()
            macro_data["plastic_type"] = "macroplastic"
            if "count_weight" not in macro_data:
                if "count" in macro_data:
                    macro_data["count_weight"] = macro_data["count"]
                else:
                    macro_data["count_weight"] = 1.0
            if "particle_id" not in macro_data:
                macro_data["particle_id"] = [
                    f"{sample_id}-macro-{row_number:03d}"
                    for row_number, sample_id in enumerate(macro_data["sample_id"], start=1)
                ]
            if "macro_item" not in macro_data:
                macro_data["macro_item"] = ""
            if "OSPAR_name" in macro_data:
                names = macro_data["OSPAR_name"].fillna("").astype(str)
                macro_data.loc[names.isin(macro_common_names), "macro_item"] = names[names.isin(macro_common_names)]
            if "OSPAR_ID" in macro_data:
                ospar_ids = pd.to_numeric(macro_data["OSPAR_ID"], errors="coerce")
                mapped = ospar_ids.map(lambda value: macro_ospar_name_map.get(str(int(value)), "") if np.isfinite(value) else "")
                macro_data.loc[mapped != "", "macro_item"] = mapped[mapped != ""]
            particle_data = pd.concat([micro_data, macro_data], ignore_index=True, sort=False)
            imported_batch_workbook.set(sheets)
            imported_batch_particles.set(particle_data)
            imported_batch_results.set(
                _build_batch_results(
                    flow_data, sample_data, particle_data, a_bed_frac, a_surf_frac
                )
            )
            ui.update_slider("samp_a_bed_frac", value=a_bed_frac, session=session)
            ui.update_slider("samp_a_surf_frac", value=a_surf_frac, session=session)
            first_sample = sample_data.iloc[0]
            _apply_flow_row(flow_data[flow_data["sample_id"].astype(str) == str(first_sample["sample_id"])].iloc[0])
            imported_sample_metadata.set(first_sample)
            first_records = particle_data[particle_data["sample_id"].astype(str) == str(first_sample["sample_id"])].copy()
            imported_particle_records.set(first_records)
            first_type = first_records["plastic_type"].astype(str).str.lower().str.rstrip("s").iloc[0]
            if first_type == "microplastic":
                first_records = first_records[
                    pd.to_numeric(first_records["size_um"], errors="coerce").between(20, 5000)
                ]
            _apply_particle_records(
                first_records[first_records["plastic_type"].astype(str).str.lower().str.rstrip("s") == first_type],
                first_sample,
            )
            ui.update_slider("samp_net_z_interval", value=(float(first_sample["sample_z_min"]), float(first_sample["sample_z_max"])), session=session)
            ui.update_select("samp_concentration_units", selected="particles/m3", session=session)
            ui.notification_show(
                f"Imported {len(sample_data)} samples. Batch results are ready to export.",
                type="message", duration=7,
            )
        except Exception as error:
            ui.notification_show(f"Workbook import failed: {error}", type="error", duration=9)

    def _batch_sample_records(sample_id: str, plastic_type: str) -> pd.DataFrame:
        records = imported_batch_particles.get()
        if records is None or records.empty:
            return pd.DataFrame()
        types = records["plastic_type"].astype(str).str.lower().str.strip().str.rstrip("s")
        selected = records.loc[
            (records["sample_id"].astype(str) == str(sample_id))
            & (types == plastic_type)
        ].copy()
        if plastic_type == "macroplastic" and "macro_item" in selected:
            selected = selected[selected["macro_item"].fillna("").astype(str).str.strip() != ""]
        return selected

    @render.ui
    def batch_visualisation_ui():
        sheets = imported_batch_workbook.get()
        sample_data = sheets.get("Sample") if sheets else None
        if sample_data is None or sample_data.empty:
            return ui.p("Upload a workbook to visualise a sample.")

        sample_ids = sample_data["sample_id"].astype(str).drop_duplicates().tolist()
        return ui.div(
            ui.layout_columns(
                ui.input_selectize(
                    "batch_sample_to_visualise",
                    "Sample",
                    choices=sample_ids,
                    selected=sample_ids[0],
                    options={"dropdownParent": "body"},
                ),
                ui.input_selectize(
                    "batch_plastic_type",
                    "Plastic type",
                    choices={
                        "microplastic": "Microplastics",
                        "macroplastic": "Macrolitter",
                    },
                    selected="microplastic",
                ),
                ui.panel_conditional(
                    "input.batch_plastic_type === 'microplastic'",
                    ui.input_selectize(
                        "batch_micro_detail",
                        "Visualisation",
                        choices={
                            "total": "Total",
                            "summary": "Buoyant and sinking",
                            "size": "Size groups",
                            "polymer": "Polymers",
                        },
                        selected="total",
                    ),
                ),
                col_widths=[4, 4, 4],
            ),
            ui.tags.details(
                ui.tags.summary("How to read this graph"),
                ui.div(
                    ui.p("This graph shows how plastic concentration changes with depth in a river water column."),
                    ui.p("The y-axis is river depth."),
                    ui.p("The x-axis shows the concentration relative to the maximum. 1 is the maximum concentration. 0.5 means half of the maximum concentration. 0.1 means 10% of the maximum concentration."),
                    ui.p("Each line represents the median predicted concentration profile for the selected plastic fraction."),
                    ui.p("Dashed lines show the proposed sampling depth."),
                    ui.p("The app does not report a corrected depth-average concentration or load when the median captured fraction is below 5%, because the sampled depth is not considered sufficiently representative of that plastic fraction."),
                    ui.p("Stronger turbulent mixing, typically associated with faster flow, produces a more uniform vertical distribution of plastics."),
                    ui.p("Under weaker mixing, plastics become more vertically separated according to their buoyancy. Sinking plastics concentrate near the bottom. Floating plastics concentrate near the surface."),
                    class_="sampling-help-body",
                ),
                class_="sampling-help",
            ),
            ui.card(
                ui.output_plot("batch_profile_plot", height="400px"),
                full_screen=True,
                class_="plot-card square-plot-card",
            ),
            ui.output_ui("batch_key_results"),
            ui.card(
                ui.card_header("Results by fraction"),
                ui.output_data_frame("batch_results_table"),
                class_="sampling-results-card",
            ),
        )

    @reactive.Effect
    @reactive.event(input.batch_sample_to_visualise, input.batch_plastic_type, ignore_init=True)
    def _apply_batch_sample_for_visualisation():
        sheets = imported_batch_workbook.get()
        sample_data = sheets.get("Sample") if sheets else None
        flow_data = sheets.get("Flow") if sheets else None
        sample_id = input.batch_sample_to_visualise()
        plastic_type = input.batch_plastic_type()
        if sample_data is None or flow_data is None or not sample_id or not plastic_type:
            return
        matching = sample_data[sample_data["sample_id"].astype(str) == str(sample_id)]
        if matching.empty:
            return
        sample = matching.iloc[0].copy()
        sample["sample_z_min"] = float(matching["sample_z_min"].min())
        sample["sample_z_max"] = float(matching["sample_z_max"].max())
        sample["sampled_volume_m3"] = float(matching["sampled_volume_m3"].sum())
        flow_rows = flow_data[flow_data["sample_id"].astype(str) == str(sample_id)]
        records = _batch_sample_records(str(sample_id), str(plastic_type))
        if flow_rows.empty or records.empty:
            return
        try:
            _apply_flow_row(flow_rows.iloc[0])
            imported_sample_metadata.set(sample)
            imported_particle_records.set(records)
            _apply_particle_records(records, sample)
            ui.update_slider(
                "samp_net_z_interval",
                value=(float(sample["sample_z_min"]), float(sample["sample_z_max"])),
                session=session,
            )
            ui.update_select("samp_concentration_units", selected="particles/m3", session=session)
        except Exception as error:
            ui.notification_show(f"Could not visualise this sample: {error}", type="error", duration=8)

    def batch_selected_depth_intervals() -> list[tuple[float, float]]:
        sheets = imported_batch_workbook.get()
        sample_data = sheets.get("Sample") if sheets else None
        if sample_data is None or sample_data.empty:
            return []
        try:
            sample_id = str(input.batch_sample_to_visualise())
        except Exception:
            return []
        rows = sample_data[sample_data["sample_id"].astype(str) == sample_id]
        return [
            (float(row["sample_z_min"]), float(row["sample_z_max"]))
            for _, row in rows.iterrows()
        ]

    @reactive.Effect
    @reactive.event(input.batch_micro_detail, ignore_init=True)
    def _sync_batch_micro_detail():
        if input.batch_micro_detail() is not None:
            ui.update_radio_buttons("samp_micro_detail", selected=input.batch_micro_detail(), session=session)

    def batch_selected_results(include_all_fractions: bool = False) -> pd.DataFrame:
        """Return detailed results for the selected imported sample."""
        results = imported_batch_results.get().copy()
        if results.empty:
            return pd.DataFrame()
        try:
            sample_id = str(input.batch_sample_to_visualise())
            plastic_type = str(input.batch_plastic_type())
        except Exception:
            return pd.DataFrame()
        results = results[
            (results["sample_id"].astype(str) == sample_id)
            & (results["plastic_type"].astype(str) == plastic_type)
        ].copy()
        if results.empty:
            return results

        if plastic_type == "microplastic" and not include_all_fractions:
            try:
                detail = str(input.batch_micro_detail() or "total")
            except Exception:
                detail = "total"
            result_sets = {
                "total": ["Micro total"],
                "summary": ["Micro total", "Micro buoyant", "Micro sinking"],
                "size": ["Micro total", "Micro size groups"],
                "polymer": ["Micro total", "Micro polymers"],
            }.get(detail, ["Micro total"])
            results = results[results["result_set"].isin(result_sets)].copy()

        total_rows = results[results["Group"].astype(str).str.lower().str.contains("total")]
        if not total_rows.empty and "Measured concentration" in results:
            total_concentration = pd.to_numeric(
                total_rows.iloc[0]["Measured concentration"], errors="coerce"
            )
            measured = pd.to_numeric(results["Measured concentration"], errors="coerce")
            if np.isfinite(total_concentration) and total_concentration > 0:
                results["Population (%)"] = (100 * measured / total_concentration).map(
                    lambda value: f"{value:.1f}%" if np.isfinite(value) else ""
                )
        if "Population (%)" not in results:
            results["Population (%)"] = ""

        results["_total_first"] = ~results["Group"].astype(str).str.lower().str.contains("total")
        return results.sort_values(["_total_first", "Group"]).drop(columns="_total_first")

    @render.data_frame
    def batch_results_table():
        df = batch_selected_results(include_all_fractions=True)
        if df.empty:
            return render.DataGrid(pd.DataFrame(), width="100%", height="220px", filters=False, summary=False)

        if "Expected capture (%)" not in df:
            df["Expected capture (%)"] = df.get("Capture (%)", "")
        elif "Capture (%)" in df:
            df["Expected capture (%)"] = df["Expected capture (%)"].fillna(df["Capture (%)"])
        if "Expected missed (%)" not in df:
            df["Expected missed (%)"] = df.get("Missed (%)", "")
        elif "Missed (%)" in df:
            df["Expected missed (%)"] = df["Expected missed (%)"].fillna(df["Missed (%)"])
        df = df.drop(columns=[column for column in ["Capture (%)", "Missed (%)"] if column in df.columns])
        concentration_units = display_unit(df["Units"].dropna().iloc[0]) if "Units" in df and df["Units"].notna().any() else ""
        load_units = str(df["Load units"].dropna().iloc[0]).strip() if "Load units" in df and df["Load units"].notna().any() else ""
        measured_heading = f"Measured concentration ({concentration_units})" if concentration_units else "Measured concentration"
        depth_average_heading = f"Depth-average concentration ({concentration_units})" if concentration_units else "Depth-average concentration"
        load_heading = f"Load ({load_units})" if load_units else "Load"
        df = df.rename(
            columns={
                "Group": "Fraction",
                "Expected capture (%)": "Captured (%)",
                "Expected missed (%)": "Missed (%)",
                "Measured concentration": measured_heading,
                "Estimated depth-averaged concentration": depth_average_heading,
                "Estimated load": load_heading,
            }
        )
        columns = [
            column for column in [
                "Fraction",
                "Captured (%)",
                "Missed (%)",
                measured_heading,
                depth_average_heading,
                load_heading,
            ] if column in df.columns
        ]
        return render.DataGrid(df.loc[:, columns], width="100%", height="390px", filters=False, summary=False)

    @render.data_frame
    def batch_capture_results():
        if bool(input.samp_select_macroplastics()):
            df = current_sampling_correction_table(
                include_discharge=False,
                include_macro_members=True,
            )
            df = df.rename(columns={"Capture (%)": "Captured (%)", "Missed (%)": "Missed (%)"})
            columns = [column for column in ["Group", "Captured (%)", "Missed (%)"] if column in df.columns]
        else:
            df = net_sampling_table(
                micro_ranges=selected_samp_micro_ranges(),
                macro_selected=[],
                macro_items_selected=[],
                use_macro_items=False,
                u_star=selected_samp_u_star(),
                micro_df=selected_samp_micro_df(),
                H=selected_flow_depth(),
                a_bed_frac=float(input.samp_a_bed_frac()),
                a_surf_frac=float(input.samp_a_surf_frac()),
                net_z_min=selected_samp_net_interval()[0],
                net_z_max=selected_samp_net_interval()[1],
                iqr_lower=selected_samp_iqr_percentiles()[0],
                iqr_upper=selected_samp_iqr_percentiles()[1],
                split_micro_by_direction=show_samp_micro_direction_profiles(),
                extra_micro_groups=selected_samp_micro_detail_groups(),
                include_micro_total=True,
            )
            df = df.rename(columns={"Population (%)": "Population %", "Capture (%)": "Captured (%)", "Missed (%)": "Missed (%)"})
            columns = [column for column in ["Group", "Population %", "Captured (%)", "Missed (%)"] if column in df.columns]
        return render.DataGrid(df.loc[:, columns], width="100%", height="165px", filters=False, summary=False)

    @render.data_frame
    def batch_concentration_results():
        df = batch_selected_results()
        df = df.rename(columns={"Estimated depth-averaged concentration": "Expected depth-average concentration"})
        columns = [column for column in ["Group", "Population (%)", "Measured concentration", "Units", "Expected depth-average concentration"] if column in df.columns]
        return render.DataGrid(df.loc[:, columns], width="100%", height="220px", filters=False, summary=False)

    @render.data_frame
    def batch_load_results():
        df = batch_selected_results()
        df = df.rename(columns={"Discharge Q (m3/s)": "Discharge (m³/s)", "Estimated load": "Expected load"})
        columns = [column for column in ["Group", "Population (%)", "Discharge (m³/s)", "Expected load", "Load units"] if column in df.columns]
        return render.DataGrid(df.loc[:, columns], width="100%", height="165px", filters=False, summary=False)

    @render.plot(alt="Vertical Rouse concentration profile plot for selected imported sample")
    def batch_profile_plot():
        return make_profile_plot(
            micro_ranges=selected_samp_micro_ranges(),
            macro_selected=selected_samp_macro_categories(),
            macro_items_selected=selected_samp_macro_items(),
            use_macro_items=use_samp_macro_items(),
            u_star=selected_samp_u_star(),
            micro_df=selected_samp_micro_df(),
            H=selected_flow_depth(),
            a_bed_frac=float(input.samp_a_bed_frac()),
            a_surf_frac=float(input.samp_a_surf_frac()),
            iqr_lower=selected_samp_iqr_percentiles()[0],
            iqr_upper=selected_samp_iqr_percentiles()[1],
            show_net_interval=bool(input.samp_show_sampling_interval()),
            net_z_intervals=batch_selected_depth_intervals() or [selected_samp_net_interval()],
            split_micro_by_direction=show_samp_micro_direction_profiles(),
            extra_micro_groups=selected_samp_micro_detail_groups(),
            include_micro_total=not show_samp_micro_detail(),
            show_iqr=False,
        )

    @reactive.Effect
    @reactive.event(input.samp_import_flow_excel)
    def _import_flow_datasheet():
        uploaded = input.samp_import_flow_excel()
        if not uploaded:
            return
        try:
            flow = pd.read_excel(uploaded[0]["datapath"], sheet_name="Flow data")
            if flow.empty:
                raise ValueError("Flow data must contain one row.")
            _apply_flow_row(flow.iloc[0])
            ui.notification_show("Flow inputs imported.", type="message", duration=5)
        except Exception as error:
            ui.notification_show(f"Flow import failed: {error}", type="error", duration=8)

    def _apply_particle_records(records: pd.DataFrame, sample: pd.Series | None) -> int:
        """Use long-form particle records to configure one app population."""
        required = {"sample_id", "particle_id", "plastic_type", "count_weight"}
        missing = required.difference(records.columns)
        if missing:
            raise ValueError(f"Particle records are missing: {', '.join(sorted(missing))}.")
        if records["particle_id"].isna().any() or records["sample_id"].isna().any():
            raise ValueError("Every particle record needs sample_id and particle_id.")
        sample_ids = records["sample_id"].dropna().astype(str).unique()
        if len(sample_ids) != 1:
            raise ValueError("Upload particle records for one sample at a time.")
        if sample is not None and str(sample["sample_id"]) != sample_ids[0]:
            raise ValueError("The sample_id in the particle and sample datasheets must match.")
        weights = pd.to_numeric(records["count_weight"], errors="coerce")
        if weights.isna().any() or (weights <= 0).any():
            raise ValueError("count_weight must be greater than zero for every record.")
        types = records["plastic_type"].astype(str).str.lower().str.strip().str.rstrip("s")
        if types.nunique() != 1 or types.iloc[0] not in {"microplastic", "macroplastic"}:
            raise ValueError("Choose either microplastic or macrolitter records for one upload.")
        concentration = None
        if sample is not None:
            concentration = float(weights.sum() / float(sample["sampled_volume_m3"]))

        if types.iloc[0] == "microplastic":
            required_micro = {"polymer", "size_um", "shape"}
            missing = required_micro.difference(records.columns)
            if missing:
                raise ValueError(f"Microplastic records are missing: {', '.join(sorted(missing))}.")
            polymer_source = records.get("polymer_group", records["polymer"])
            polymers = polymer_source.map(standardise_polymer_name)
            if "polymer_group" in records.columns:
                polymers = polymers.fillna(records["polymer"].map(standardise_polymer_name))
            recognised = polymers.notna()
            omitted = int((~recognised).sum())
            if not recognised.any():
                raise ValueError(
                    "No recognised polymers. Use PE, PET, PA, PP, PS, PVA, PVC, or a supported full polymer name."
                )
            records = records.loc[recognised].copy()
            weights = weights.loc[recognised]
            polymers = polymers.loc[recognised]
            if sample is not None:
                concentration = float(weights.sum() / float(sample["sampled_volume_m3"]))
            sizes = pd.to_numeric(records["size_um"], errors="coerce")
            if sizes.isna().any() or (sizes < 20).any() or (sizes > 5000).any():
                raise ValueError("Microplastic size_um values must be between 20 and 5,000.")
            shapes = records["shape"].astype(str).str.lower().str.strip()
            shapes = shapes.replace({"fiber": "fibre", "fragment": "fragment"})
            if not shapes.isin({"fibre", "fragment"}).all():
                raise ValueError("Microplastic shape must be fibre or fragment.")
            total_weight = float(weights.sum())
            polymer_values = {
                name: 100.0 * float(weights[polymers == name].sum()) / total_weight
                for name in DEFAULT_POLYMER_PERCENTAGES
            }
            slider_polymer_values = {
                name: int(round(value)) for name, value in polymer_values.items()
            }
            largest_polymer = max(slider_polymer_values, key=slider_polymer_values.get)
            slider_polymer_values[largest_polymer] += 100 - sum(
                slider_polymer_values.values()
            )
            fibre = 100.0 * float(weights[shapes == "fibre"].sum()) / total_weight
            size_min = float(sizes.min())
            size_max = float(sizes.max())
            if size_min == size_max:
                size_min = max(20.0, size_min - 10.0)
                size_max = min(5000.0, size_max + 10.0)
            ui.update_checkbox("samp_select_microplastics", value=True, session=session)
            ui.update_checkbox("samp_select_macroplastics", value=False, session=session)
            ui.update_checkbox("samp_add_size_group", value=False, session=session)
            ui.update_slider("samp_synthetic_size_range", value=(size_min, size_max), session=session)
            ui.update_slider("samp_fiber_percent", value=fibre, session=session)
            ui.update_slider("samp_fragment_percent", value=100.0 - fibre, session=session)
            for name, value in slider_polymer_values.items():
                ui.update_slider(f"samp_polymer_{name}", value=value, session=session)
            if concentration is not None:
                ui.update_numeric("samp_measured_concentration", value=concentration, session=session)
        else:
            if "macro_item" not in records.columns:
                raise ValueError("Macrolitter records must include macro_item.")
            items = records["macro_item"].dropna().astype(str).unique().tolist()
            if not items or any(item not in macro_common_names for item in items):
                raise ValueError("Each macro_item must match an available individual litter item.")
            ui.update_checkbox("samp_select_microplastics", value=False, session=session)
            ui.update_checkbox("samp_select_macroplastics", value=True, session=session)
            ui.update_radio_buttons("samp_macro_mode", selected="individual", session=session)
            ui.update_selectize("samp_macro_common_names", selected=items, session=session)
            if concentration is not None:
                item_concentrations = {
                    item: float(weights[records["macro_item"].astype(str) == item].sum() / float(sample["sampled_volume_m3"]))
                    for item in items
                }
                imported_macro_concentrations.set(item_concentrations)
            omitted = 0
        return omitted

    @reactive.Effect
    @reactive.event(input.samp_import_sample_excel)
    def _import_sample_datasheet():
        uploaded = input.samp_import_sample_excel()
        if not uploaded:
            return
        try:
            samples = pd.read_excel(uploaded[0]["datapath"], sheet_name="Samples")
            required = {"sample_id", "sample_z_min", "sample_z_max", "sampled_volume_m3"}
            missing = required.difference(samples.columns)
            if samples.empty or missing:
                raise ValueError(f"Sample data are missing: {', '.join(sorted(missing))}.")
            if len(samples) != 1:
                raise ValueError("Upload one sample at a time.")
            row = samples.iloc[0]
            z_min = _number(row, "sample_z_min", "Samples")
            z_max = _number(row, "sample_z_max", "Samples")
            volume = _number(row, "sampled_volume_m3", "Samples")
            if not 0 <= z_min < z_max <= 1 or volume <= 0:
                raise ValueError("Use a valid depth interval and sampled_volume_m3 greater than zero.")
            imported_sample_metadata.set(row)
            ui.update_slider("samp_net_z_interval", value=(z_min, z_max), session=session)
            ui.update_select("samp_concentration_units", selected="particles/m3", session=session)
            records = imported_particle_records.get()
            if records is not None:
                omitted = _apply_particle_records(records, row)
                if omitted:
                    ui.notification_show(
                        f"{omitted} unrecognised polymer records were excluded.",
                        type="warning", duration=7,
                    )
            ui.notification_show("Sample data imported.", type="message", duration=5)
        except Exception as error:
            ui.notification_show(f"Sample import failed: {error}", type="error", duration=8)

    @reactive.Effect
    @reactive.event(input.samp_import_plastics_excel)
    def _import_plastics_datasheet():
        uploaded = input.samp_import_plastics_excel()
        if not uploaded:
            return
        try:
            records = pd.read_excel(uploaded[0]["datapath"], sheet_name="Particles").dropna(how="all")
            if records.empty:
                raise ValueError("Particles must contain at least one record.")
            imported_particle_records.set(records)
            omitted = _apply_particle_records(records, imported_sample_metadata.get())
            message = "Particle records imported."
            if omitted:
                message += f" {omitted} unrecognised polymer records were excluded."
            if imported_sample_metadata.get() is None:
                message += " Upload sample data to calculate concentration."
            ui.notification_show(message, type="message", duration=6)
        except Exception as error:
            ui.notification_show(f"Plastics import failed: {error}", type="error", duration=8)

    @reactive.Effect
    @reactive.event(input.samp_import_excel)
    def _import_sampling_inputs():
        uploaded = input.samp_import_excel()
        if not uploaded:
            return

        try:
            sheets = pd.read_excel(
                uploaded[0]["datapath"],
                sheet_name=[
                    "Flow inputs",
                    "Sample inputs",
                    "Plastic inputs",
                    "Microplastic inputs",
                ],
            )
            if any(frame.empty for frame in sheets.values()):
                raise ValueError("Each input sheet must contain at least one row.")

            flow = sheets["Flow inputs"].iloc[0]
            mode = str(flow.get("u_star_mode", "")).strip()
            width = _number(flow, "river_width_m", "Flow inputs")
            depth = _number(flow, "river_depth_m", "Flow inputs")
            slope = _number(flow, "slope", "Flow inputs")
            discharge = _number(flow, "discharge_m3_s", "Flow inputs")
            direct_u_star = _number(flow, "direct_u_star_m_s", "Flow inputs")
            if mode not in {"hydraulic", "direct"}:
                raise ValueError("Flow inputs must use hydraulic or direct u_star_mode.")
            if width <= 0 or depth <= 0 or slope <= 0 or discharge < 0 or direct_u_star <= 0:
                raise ValueError("Flow inputs contain an invalid value.")

            sample = sheets["Sample inputs"].iloc[0]
            z_min = _number(sample, "sample_z_min", "Sample inputs")
            z_max = _number(sample, "sample_z_max", "Sample inputs")
            units = str(sample.get("concentration_units", ""))
            if not 0 <= z_min < z_max <= 1:
                raise ValueError("Sample depth must be between 0 and 1, with a lower and upper limit.")
            if units not in {"particles/m3", "g/m3"}:
                raise ValueError("Sample inputs contain unsupported concentration units.")

            plastics = sheets["Plastic inputs"].dropna(how="all")
            if plastics.empty or "plastic_type" not in plastics or "selection_type" not in plastics:
                raise ValueError("Plastic inputs must include plastic_type and selection_type.")
            plastic_type = str(plastics.iloc[0]["plastic_type"])
            selection_type = str(plastics.iloc[0]["selection_type"])
            if plastic_type not in {"microplastics", "macroplastics", "macrolitter"} or not all(
                plastics["plastic_type"].astype(str) == plastic_type
            ):
                raise ValueError("Choose one plastic type in Plastic inputs.")
            if plastic_type == "macrolitter":
                plastic_type = "macroplastics"
            if not all(plastics["selection_type"].astype(str) == selection_type):
                raise ValueError("Plastic inputs must use one selection type.")
            if "measured_concentration" not in plastics:
                raise ValueError("Plastic inputs must include measured_concentration.")
            concentrations = pd.to_numeric(
                plastics["measured_concentration"], errors="coerce"
            )
            if concentrations.isna().any() or (concentrations < 0).any():
                raise ValueError("Measured concentrations must be zero or greater.")

            ui.update_numeric("samp_river_width", value=width, session=session)
            ui.update_numeric("samp_river_depth", value=depth, session=session)
            ui.update_numeric("samp_slope", value=slope, session=session)
            ui.update_numeric("samp_discharge", value=discharge, session=session)
            ui.update_checkbox("samp_direct_ustar", value=mode == "direct", session=session)
            ui.update_numeric("samp_u_star", value=direct_u_star, session=session)
            ui.update_slider("samp_net_z_interval", value=(z_min, z_max), session=session)
            ui.update_select("samp_concentration_units", selected=units, session=session)

            hydraulic_radius = width * depth / (width + 2.0 * depth)
            u_star = direct_u_star if mode == "direct" else calculate_shear_velocity_from_slope_radius(
                hydraulic_radius=hydraulic_radius, slope=slope
            )
            applied_sampling_flow.set(
                {
                    "u_star": u_star,
                    "discharge": discharge,
                    "mode": mode,
                    "river_width_m": width,
                    "river_depth_m": depth,
                    "hydraulic_radius_m": hydraulic_radius,
                    "slope": slope,
                    "direct_u_star_m_s": direct_u_star,
                }
            )

            if plastic_type == "microplastics":
                micro = sheets["Microplastic inputs"].dropna(how="all")
                if len(micro) not in {1, 2}:
                    raise ValueError("Microplastic inputs must contain one or two size groups.")
                first = micro.iloc[0]
                distribution = str(first.get("size_distribution", ""))
                if distribution not in {"loguniform", "uniform"}:
                    raise ValueError("Microplastic inputs contain an unsupported size distribution.")
                for _, row in micro.iterrows():
                    lower = _number(row, "size_min_um", "Microplastic inputs")
                    upper = _number(row, "size_max_um", "Microplastic inputs")
                    if not 20 <= lower < upper <= 5000:
                        raise ValueError("Microplastic size limits must be between 20 and 5,000 µm.")
                fibre = _number(first, "fibre_percent", "Microplastic inputs")
                fragment = _number(first, "fragment_percent", "Microplastic inputs")
                polymer_values = {
                    name: _number(first, f"{name}_percent", "Microplastic inputs")
                    for name in DEFAULT_POLYMER_PERCENTAGES
                }
                if fibre < 0 or fragment < 0 or any(value < 0 for value in polymer_values.values()):
                    raise ValueError("Microplastic percentages cannot be negative.")
                if not polymer_total_valid(sum(polymer_values.values())):
                    raise ValueError("Imported polymer percentages must total 100%.")

                ui.update_checkbox("samp_select_microplastics", value=True, session=session)
                ui.update_checkbox("samp_select_macroplastics", value=False, session=session)
                ui.update_checkbox("samp_add_size_group", value=len(micro) == 2, session=session)
                ui.update_slider(
                    "samp_synthetic_size_range",
                    value=(_number(first, "size_min_um", "Microplastic inputs"), _number(first, "size_max_um", "Microplastic inputs")),
                    session=session,
                )
                if len(micro) == 2:
                    second = micro.iloc[1]
                    ui.update_slider(
                        "samp_synthetic_size_range_2",
                        value=(_number(second, "size_min_um", "Microplastic inputs"), _number(second, "size_max_um", "Microplastic inputs")),
                        session=session,
                    )
                ui.update_select("samp_synthetic_size_distribution", selected=distribution, session=session)
                ui.update_slider("samp_fiber_percent", value=fibre, session=session)
                ui.update_slider("samp_fragment_percent", value=fragment, session=session)
                for name, value in polymer_values.items():
                    ui.update_slider(f"samp_polymer_{name}", value=value, session=session)
                ui.update_numeric("samp_measured_concentration", value=float(concentrations.iloc[0]), session=session)
                if len(micro) == 2:
                    ui.update_numeric("samp_measured_concentration_2", value=float(concentrations.iloc[1]), session=session)
            else:
                if selection_type not in {"grouped", "individual"} or "identifier" not in plastics:
                    raise ValueError("Macrolitter inputs must use grouped or individual selections.")
                identifiers = plastics["identifier"].astype(str).tolist()
                if selection_type == "grouped":
                    if any(identifier not in macro_group_labels for identifier in identifiers):
                        raise ValueError("A grouped macrolitter class was not recognised.")
                    ui.update_checkbox_group("samp_macro_categories", selected=identifiers, session=session)
                else:
                    if any(identifier not in macro_common_names for identifier in identifiers):
                        raise ValueError("An individual macrolitter item was not recognised.")
                    imported_macro_concentrations.set(
                        dict(zip(identifiers, concentrations.astype(float)))
                    )
                    ui.update_selectize("samp_macro_common_names", selected=identifiers, session=session)
                ui.update_checkbox("samp_select_microplastics", value=False, session=session)
                ui.update_checkbox("samp_select_macroplastics", value=True, session=session)
                ui.update_radio_buttons("samp_macro_mode", selected=selection_type, session=session)
                if selection_type == "grouped":
                    for identifier, concentration in zip(identifiers, concentrations):
                        ui.update_numeric(
                            macro_group_concentration_input_id(identifier),
                            value=float(concentration),
                            session=session,
                        )

            ui.notification_show("Inputs imported from Excel.", type="message", duration=5)
        except Exception as error:
            ui.notification_show(f"Excel import failed: {error}", type="error", duration=8)

    @reactive.Effect
    @reactive.event(input.samp_select_microplastics, ignore_init=True)
    def _select_only_microplastics():
        if bool(input.samp_select_microplastics()):
            ui.update_checkbox(
                "samp_select_macroplastics",
                value=False,
                session=session,
            )
        elif not bool(input.samp_select_macroplastics()):
            ui.update_checkbox(
                "samp_select_macroplastics",
                value=True,
                session=session,
            )

    @reactive.Effect
    @reactive.event(input.samp_select_macroplastics, ignore_init=True)
    def _select_only_macroplastics():
        if bool(input.samp_select_macroplastics()):
            ui.update_checkbox(
                "samp_select_microplastics",
                value=False,
                session=session,
            )
        elif not bool(input.samp_select_microplastics()):
            ui.update_checkbox(
                "samp_select_microplastics",
                value=True,
                session=session,
            )


    def polymer_total_valid(total: float) -> bool:
        """Return True only when polymer sliders sum to exactly 100%."""
        return abs(float(total) - 100.0) < 1e-9

    def empty_synthetic_microplastics_df(reason: str = "invalid polymer mix") -> pd.DataFrame:
        """Return an empty synthetic dataset with expected columns.

        Used when polymer percentages do not sum to 100%. This prevents
        silent normalisation and makes invalid custom mixtures obvious.
        """
        cols = [
            "particle_id",
            "size",
            "size_um",
            "particle_type",
            "polymer",
            "density_g_cm3",
            "velocity_dietrich",
            "velocity_goral",
            "velocity_yu",
            "validation_status",
        ]
        return pd.DataFrame(columns=cols).assign(validation_status=pd.Series(dtype="object"))

    def _integer_mix_with_fixed(
        changed_name: str,
        changed_value: float,
        current_values: dict[str, float],
    ) -> dict[str, int]:
        """Return an integer polymer percentage mix summing to 100.

        The slider the user just changed is kept fixed. The remaining
        percentage is redistributed across the other polymers in proportion
        to their previous values. This avoids hidden normalisation while
        keeping the UI total physically meaningful at 100%.
        """
        changed_value = int(round(max(0, min(100, changed_value))))
        polymer_names = list(current_values.keys())
        other_names = [name for name in polymer_names if name != changed_name]
        remaining = 100 - changed_value

        if remaining <= 0:
            return {name: (100 if name == changed_name else 0) for name in polymer_names}

        other_total = float(sum(max(0, current_values[name]) for name in other_names))

        if other_total <= 0:
            base = remaining // len(other_names)
            result = {name: base for name in other_names}
            for name in other_names[: remaining - base * len(other_names)]:
                result[name] += 1
        else:
            exact = {
                name: remaining * max(0, current_values[name]) / other_total
                for name in other_names
            }
            result = {name: int(np.floor(value)) for name, value in exact.items()}
            shortfall = remaining - sum(result.values())
            order = sorted(
                other_names,
                key=lambda name: exact[name] - result[name],
                reverse=True,
            )
            for name in order[:shortfall]:
                result[name] += 1

        result[changed_name] = changed_value
        return {name: int(result.get(name, 0)) for name in polymer_names}

    def _read_polymer_inputs(input_map: dict[str, str]) -> dict[str, int]:
        """Read a set of polymer slider inputs as integer percentages."""
        return {
            name: int(round(float(getattr(input, input_id)())))
            for name, input_id in input_map.items()
        }

    def _sync_polymer_group(
        input_map: dict[str, str],
        last_values: reactive.Value,
        update_guard: reactive.Value,
    ) -> None:
        """Synchronise one polymer-slider group so it always sums to 100%."""
        if update_guard.get():
            return

        current = _read_polymer_inputs(input_map)
        last = last_values.get()

        changed = [name for name in current if current[name] != last.get(name)]
        if not changed:
            return

        changed_name = max(
            changed,
            key=lambda name: abs(current[name] - last.get(name, 0)),
        )

        updated = _integer_mix_with_fixed(
            changed_name=changed_name,
            changed_value=current[changed_name],
            current_values=last,
        )

        update_guard.set(True)
        try:
            last_values.set(updated)
            for name, input_id in input_map.items():
                if current.get(name) != updated[name]:
                    ui.update_slider(input_id, value=updated[name])
        finally:
            update_guard.set(False)

    default_polymer_mix = {name: int(value) for name, value in DEFAULT_POLYMER_PERCENTAGES.items()}



    sampling_polymer_last_values = reactive.Value(default_polymer_mix.copy())

    sampling_polymer_update_guard = reactive.Value(False)



    @reactive.Effect
    @reactive.event(
        input.samp_polymer_PE, input.samp_polymer_PET, input.samp_polymer_PA,
        input.samp_polymer_PP, input.samp_polymer_PS, ignore_init=True,
    )
    def _sync_sampling_polymer_sliders():
        _sync_polymer_group(
            {name: f"samp_polymer_{name}" for name in DEFAULT_POLYMER_PERCENTAGES},
            sampling_polymer_last_values,
            sampling_polymer_update_guard,
        )





    @render.ui
    def sampling_caution():
        z_min, z_max = selected_samp_net_interval()
        sampled_fraction = abs(float(z_max) - float(z_min))

        messages = []
        if sampled_fraction < 0.10:
            messages.append("The sampled layer is less than 10% of the water column. Correction factors may become very sensitive.")
        if len(selected_samp_micro_ranges()) == 0 and len(selected_samp_macro_categories()) == 0 and len(selected_samp_macro_items()) == 0:
            messages.append("No plastic groups are selected on this page.")
        if (
            bool(input.samp_select_microplastics())
            and not polymer_total_valid(selected_samp_polymer_total())
        ):
            messages.append(f"Sampling-page polymer sliders total {selected_samp_polymer_total():.0f}%; linked sliders are adjusting this to 100%.")

        if not messages:
            return ui.div(
                "",
                class_="sampling-note",
            )

        return ui.div(
            ui.tags.strong("Interpret with caution"),
            ui.tags.ul(*[ui.tags.li(msg) for msg in messages]),
            class_="warning-box",
        )

    @render.ui
    def sampling_behaviour_note():
        """Explain why the active profile graph contains multiple lines."""
        if bool(input.samp_select_macroplastics()):
            selected_count = (
                len(selected_samp_macro_items())
                if use_samp_macro_items()
                else len(selected_samp_macro_categories())
            )
            if selected_count < 2:
                return ui.div()
        else:
            has_rising = False
            has_settling = False
            for _, min_um, max_um in selected_samp_micro_ranges():
                beta = beta_values_for_micro_range(
                    min_um=min_um,
                    max_um=max_um,
                    u_star=selected_samp_u_star(),
                    micro_df=selected_samp_micro_df(),
                )
                has_rising = has_rising or bool(np.any(beta < 0))
                has_settling = has_settling or bool(np.any(beta >= 0))
            if not (has_rising and has_settling):
                return ui.div()

        return ui.div(
            ui.tags.strong("Why are there multiple lines?"),
            ui.p(
                "The selected microplastics or macrolitter contain both "
                "buoyant and sinking particles. They are separated on the "
                "figure so their different vertical behaviour is easier to "
                "see. The estimated corrected concentration "
                "and loads use the total "
                "population of both sinking and buoyant items."
            ),
            class_="sampling-note",
        )

    def _key_results_ui():
        """Show compact headline outputs for every selected plastic group."""
        groups = selected_group_beta_values(
            micro_ranges=selected_samp_micro_ranges(),
            macro_selected=selected_samp_macro_categories(),
            macro_items_selected=selected_samp_macro_items(),
            use_macro_items=use_samp_macro_items(),
            u_star=selected_samp_u_star(),
            micro_df=selected_samp_micro_df(),
        )
        groups = add_macroplastic_total(groups, include_members=False)

        units_map = {
            "particles/m3": "particles/m³",
            "g/m3": "g/m³",
        }
        load_units_map = {
            "particles/m3": "particles/s",
            "g/m3": "g/s",
        }
        concentration_units = str(input.samp_concentration_units())
        discharge = selected_samp_discharge()
        group_sections = []

        if bool(input.samp_select_macroplastics()):
            item_results = current_sampling_correction_table(
                include_discharge=discharge > 0,
                include_macro_members=False,
            )
            total_rows = item_results[
                item_results["Group"] == "Total"
            ]
            if total_rows.empty:
                return ui.div()

            total = total_rows.iloc[0]
            median_capture = float(total["_median_capture"])
            median_corrected = float(total["_median_corrected"])
            median_load = float(total["_median_load"])
            measured_concentration = pd.to_numeric(total.get("Measured concentration"), errors="coerce")
            measured_text = (
                f"{fmt_sig(measured_concentration)} "
                f"{units_map.get(concentration_units, concentration_units)}"
                if np.isfinite(measured_concentration)
                else "Not available"
            )
            captured_text = (
                f"{median_capture * 100:.1f}%"
                if np.isfinite(median_capture)
                else MACRO_LOW_CAPTURE_WARNING
            )
            corrected_text = (
                f"{fmt_sig(median_corrected)} "
                f"{units_map.get(concentration_units, concentration_units)}"
                if np.isfinite(median_corrected)
                else MACRO_LOW_CAPTURE_WARNING
            )
            load_text = (
                (
                    f"{fmt_sig(median_load)} "
                    f"{load_units_map.get(concentration_units, concentration_units + ' × m³/s')}"
                    if np.isfinite(median_load)
                    else MACRO_LOW_CAPTURE_WARNING
                )
                if discharge > 0
                else "Enter discharge"
            )
            return ui.div(
                ui.layout_columns(
                    ui.value_box(
                        "Plastics captured",
                        captured_text,
                        theme="primary",
                        height="110px",
                        fill=False,
                    ),
                    ui.value_box(
                        "Measured concentration",
                        measured_text,
                        theme="primary",
                        height="110px",
                        fill=False,
                    ),
                    ui.value_box(
                        "Estimated depth-average concentration",
                        corrected_text,
                        theme="primary",
                        height="110px",
                        fill=False,
                    ),
                    ui.value_box(
                        "Estimated load",
                        load_text,
                        theme="primary",
                        height="110px",
                        fill=False,
                    ),
                    col_widths=[3, 3, 3, 3],
                    fill=False,
                ),
                class_="sampling-key-results",
            )

        if not groups:
            return ui.div()

        for group_name, beta_values in groups:
            measured_concentration = float(selected_samp_micro_concentrations().get(group_name, np.nan))
            measured_text = (
                f"{fmt_sig(measured_concentration)} "
                f"{units_map.get(concentration_units, concentration_units)}"
                if np.isfinite(measured_concentration)
                else "Not available"
            )
            captured_values, _ = sampling_fraction_distribution_from_beta(
                beta_values=beta_values,
                H=selected_flow_depth(),
                a_bed_frac=float(input.samp_a_bed_frac()),
                a_surf_frac=float(input.samp_a_surf_frac()),
                net_z_min=selected_samp_net_interval()[0],
                net_z_max=selected_samp_net_interval()[1],
            )
            captured_values = finite(captured_values)
            valid_captured = captured_values[np.isfinite(captured_values)]

            if len(valid_captured) == 0:
                captured_text = "Not available"
                corrected_text = "Not available"
                load_text = "Not available"
            else:
                median_capture = float(np.nanmedian(valid_captured))
                corrected_concentration = (
                    selected_samp_micro_concentrations()[group_name]
                    * abs(
                        selected_samp_net_interval()[1]
                        - selected_samp_net_interval()[0]
                    )
                    / median_capture
                    if median_capture > 0
                    else np.nan
                )

                captured_text = f"{median_capture * 100:.1f}%"
                corrected_text = (
                    f"{fmt_sig(corrected_concentration)} "
                    f"{units_map.get(concentration_units, concentration_units)}"
                ) if np.isfinite(corrected_concentration) else "Not available"
                load_text = (
                    f"{fmt_sig(corrected_concentration * discharge)} "
                    f"{load_units_map.get(concentration_units, concentration_units + ' × m³/s')}"
                    if discharge > 0
                    else "Enter discharge"
                )
                if median_capture < MIN_RELIABLE_CAPTURE:
                    corrected_text = LOW_CAPTURE_WARNING
                    load_text = (
                        LOW_CAPTURE_WARNING
                        if discharge > 0
                        else "Enter discharge"
                    )

            if len(groups) > 1:
                group_sections.append(ui.h6(group_name))
            group_sections.extend(
                [
                    ui.layout_columns(
                        ui.value_box(
                            "Plastics captured",
                            captured_text,
                            theme="primary",
                            height="110px",
                            fill=False,
                        ),
                        ui.value_box(
                            "Measured concentration",
                            measured_text,
                            theme="primary",
                            height="110px",
                            fill=False,
                        ),
                        ui.value_box(
                            "Estimated depth-average concentration",
                            corrected_text,
                            theme="primary",
                            height="110px",
                            fill=False,
                        ),
                        ui.value_box(
                            "Estimated load",
                            load_text,
                            theme="primary",
                            height="110px",
                            fill=False,
                        ),
                        col_widths=[3, 3, 3, 3],
                        fill=False,
                    ),
                ]
            )

        return ui.div(
            *group_sections,
            class_="sampling-key-results",
        )


    @render.ui
    def sampling_key_results():
        return _key_results_ui()

    @render.ui
    def batch_key_results():
        df = batch_selected_results()
        if df.empty:
            return ui.div()
        total = df[df["Group"].astype(str).str.lower().str.contains("total")]
        row = total.iloc[0] if not total.empty else df.iloc[0]
        median_capture = pd.to_numeric(row.get("_median_capture"), errors="coerce")
        median_corrected = pd.to_numeric(row.get("_median_corrected"), errors="coerce")
        median_load = pd.to_numeric(row.get("_median_load"), errors="coerce")
        def box_value(value: object) -> str:
            return str(value).split(" [", 1)[0]

        captured = (
            f"{fmt_sig(median_capture * 100)}%"
            if np.isfinite(median_capture)
            else box_value(row.get("Median captured (%)", "Not available"))
        )
        corrected = (
            fmt_sig(median_corrected)
            if np.isfinite(median_corrected)
            else box_value(row.get("Estimated depth-averaged concentration", "Not available"))
        )
        load = (
            fmt_sig(median_load)
            if np.isfinite(median_load)
            else box_value(row.get("Estimated load", "Not available"))
        )
        units = display_unit(row.get("Units", ""))
        load_units = str(row.get("Load units", "")).strip()
        if units and corrected not in {"Not available", LOW_CAPTURE_WARNING, MACRO_LOW_CAPTURE_WARNING}:
            corrected = f"{corrected} {units}"
        if load_units and load not in {"Not available", LOW_CAPTURE_WARNING, MACRO_LOW_CAPTURE_WARNING}:
            load = f"{load} {load_units}"
        return ui.div(
            ui.layout_columns(
                ui.value_box("Plastics captured", str(captured), theme="primary", height="110px", fill=False),
                ui.value_box("Measured concentration", f"{row.get('Measured concentration', 'Not available')} {units}".strip(), theme="primary", height="110px", fill=False),
                ui.value_box("Estimated depth-average concentration", corrected, theme="primary", height="110px", fill=False),
                ui.value_box("Estimated load", load, theme="primary", height="110px", fill=False),
                col_widths=[3, 3, 3, 3], fill=False,
            ),
            class_="sampling-key-results",
        )


    def selected_flow_depth() -> float:
        """Return the flow depth used in Rouse-profile calculations.

        Use the river depth saved with the applied flow values.
        """
        return float(applied_sampling_flow.get()["river_depth_m"])












    @reactive.Effect
    @reactive.event(input.samp_reset_polymer_mix)
    def _samp_reset_polymer_mix():
        """Reset sampling-tab polymer sliders to the default synthetic mixture."""
        for name, value in default_polymer_mix.items():
            ui.update_slider(f"samp_polymer_{name}", value=value)
        sampling_polymer_last_values.set(default_polymer_mix.copy())

    # Independent Sample design calculations.
    design_polymer_last_values = reactive.Value(default_polymer_mix.copy())
    design_polymer_update_guard = reactive.Value(False)

    @reactive.Effect
    @reactive.event(
        input.design_polymer_PE, input.design_polymer_PET, input.design_polymer_PA,
        input.design_polymer_PP, input.design_polymer_PS, ignore_init=True,
    )
    def _sync_design_polymer_sliders():
        _sync_polymer_group(
            {name: f"design_polymer_{name}" for name in DEFAULT_POLYMER_PERCENTAGES},
            design_polymer_last_values,
            design_polymer_update_guard,
        )

    @reactive.Effect
    @reactive.event(input.design_reset_polymer_mix)
    def _reset_design_polymer_mix():
        for name, value in default_polymer_mix.items():
            ui.update_slider(f"design_polymer_{name}", value=value, session=session)
        design_polymer_last_values.set(default_polymer_mix.copy())

    @reactive.Effect
    @reactive.event(input.design_use_estimated_distribution)
    def _apply_estimated_design_microplastic_distribution():
        if not bool(input.design_use_estimated_distribution()):
            return
        ui.update_slider("design_size_range", value=(20, 5000), session=session)
        ui.update_select("design_size_distribution", selected="loguniform", session=session)
        ui.update_slider("design_fiber_percent", value=50, session=session)
        ui.update_slider("design_fragment_percent", value=50, session=session)
        for name, value in default_polymer_mix.items():
            ui.update_slider(f"design_polymer_{name}", value=value, session=session)
        design_polymer_last_values.set(default_polymer_mix.copy())

    @reactive.Effect
    @reactive.event(
        input.design_size_range,
        input.design_size_distribution,
        input.design_fiber_percent,
        input.design_fragment_percent,
        input.design_polymer_PE,
        input.design_polymer_PET,
        input.design_polymer_PA,
        input.design_polymer_PP,
        input.design_polymer_PS,
        ignore_init=True,
    )
    def _clear_estimated_design_microplastic_distribution():
        is_estimated = (
            tuple(input.design_size_range()) == (20, 5000)
            and str(input.design_size_distribution()) == "loguniform"
            and float(input.design_fiber_percent()) == 50
            and float(input.design_fragment_percent()) == 50
            and all(
                float(getattr(input, f"design_polymer_{name}")()) == value
                for name, value in default_polymer_mix.items()
            )
        )
        if bool(input.design_use_estimated_distribution()) and not is_estimated:
            ui.update_checkbox(
                "design_use_estimated_distribution",
                value=False,
                session=session,
            )

    @reactive.Effect
    @reactive.event(input.design_macro_mode)
    def _apply_top_five_macroplastic_items():
        if str(input.design_macro_mode()) == "top_five":
            ui.update_selectize(
                "design_macro_common_names",
                selected=top_five_macroplastic_items,
                session=session,
            )
        elif str(input.design_macro_mode()) == "individual":
            ui.update_selectize(
                "design_macro_common_names",
                selected=["Soft plastic pieces/films 0.5-2.5 cm"],
                session=session,
            )

    @reactive.Effect
    @reactive.event(input.design_macro_common_names, ignore_init=True)
    def _clear_top_five_macroplastic_items():
        selected = list(input.design_macro_common_names() or [])
        if (
            str(input.design_macro_mode()) == "top_five"
            and set(selected) != set(top_five_macroplastic_items)
        ):
            ui.update_radio_buttons(
                "design_macro_mode",
                selected="individual",
                session=session,
            )

    @reactive.Effect
    @reactive.event(input.design_apply_flow)
    def _apply_design_flow_values():
        width = float(input.design_river_width())
        depth = float(input.design_river_depth())
        slope = float(input.design_slope())
        hydraulic_radius = width * depth / (width + 2.0 * depth)
        u_star = (
            float(input.design_u_star())
            if bool(input.design_direct_ustar())
            else calculate_shear_velocity_from_slope_radius(hydraulic_radius, slope)
        )
        applied_design_flow.set(
            {"u_star": u_star, "discharge": float(input.design_discharge()), "river_depth_m": depth}
        )

    def selected_design_u_star() -> float:
        return float(applied_design_flow.get()["u_star"])

    @render.ui
    def design_net_controls():
        depth = float(applied_design_flow.get()["river_depth_m"])
        controls = [ui.p("River depth is set in the ‘Flow’ tab.", class_="helper-text")]
        for net_number in range(1, design_net_count.get() + 1):
            sample_top = depth * (1.0 - float(input.design_a_bed_frac()))
            default_interval = (
                (max(0.0, sample_top - 0.10 * depth), sample_top)
                if net_number == 1
                else (0.0, min(0.2 * depth, depth))
            )
            controls.append(
                ui.input_slider(
                    f"design_net_interval_m_{net_number}",
                    "Sampling depth interval (m)" if net_number == 1 else f"Sampling depth interval {net_number} (m)",
                    min=0.0,
                    max=depth,
                    value=default_interval,
                    step=max(depth / 100.0, 0.01),
                )
            )
        controls.extend(
            [
                ui.input_action_button("design_add_net", "Add another sampling depth", class_="btn-sm btn-outline-secondary"),
            ]
        )
        return ui.div(*controls)

    @reactive.Effect
    @reactive.event(input.design_add_net)
    def _add_design_net():
        design_net_count.set(design_net_count.get() + 1)

    def selected_design_net_intervals() -> list[tuple[float, float]]:
        """Return non-overlapping planned net intervals as relative heights."""
        depth = float(applied_design_flow.get()["river_depth_m"])
        intervals = []
        for net_number in range(1, design_net_count.get() + 1):
            try:
                values = getattr(input, f"design_net_interval_m_{net_number}")()
            except Exception:
                values = None
            if values is None:
                if net_number == 1:
                    sample_top = 1.0 - float(input.design_a_bed_frac())
                    intervals.append((max(0.0, sample_top - 0.10), sample_top))
                continue
            lower, upper = sorted(float(value) for value in values)
            lower = max(0.0, min(lower, depth))
            upper = max(0.0, min(upper, depth))
            if upper > lower:
                intervals.append((lower / depth, upper / depth))
        intervals.sort()
        merged_intervals: list[tuple[float, float]] = []
        for lower, upper in intervals:
            if merged_intervals and lower <= merged_intervals[-1][1]:
                merged_intervals[-1] = (
                    merged_intervals[-1][0],
                    max(upper, merged_intervals[-1][1]),
                )
            else:
                merged_intervals.append((lower, upper))
        return merged_intervals

    def selected_design_polymer_percentages() -> dict[str, float]:
        values = {name: float(getattr(input, f"design_polymer_{name}")()) for name in DEFAULT_POLYMER_PERCENTAGES}
        return values if polymer_total_valid(sum(values.values())) else {name: 0.0 for name in values}

    @reactive.Effect
    @reactive.event(input.design_fiber_percent)
    def _sync_design_fragments():
        target = 100.0 - float(input.design_fiber_percent())
        if float(input.design_fragment_percent()) != target:
            ui.update_slider("design_fragment_percent", value=target, session=session)

    @reactive.Effect
    @reactive.event(input.design_fragment_percent)
    def _sync_design_fibres():
        target = 100.0 - float(input.design_fragment_percent())
        if float(input.design_fiber_percent()) != target:
            ui.update_slider("design_fiber_percent", value=target, session=session)

    def selected_design_micro_ranges() -> list[tuple[str, float, float]]:
        if str(input.design_plastic_type()) != "microplastic":
            return []
        ranges = [("synthetic MP", *input.design_size_range())]
        return [(name, float(low), float(high)) for name, low, high in ranges if high > low]

    @reactive.calc
    def selected_design_micro_df() -> pd.DataFrame:
        if str(input.design_plastic_type()) != "microplastic":
            return empty_synthetic_microplastics_df()
        polymers = selected_design_polymer_percentages()
        if not any(polymers.values()):
            return empty_synthetic_microplastics_df()
        fibre = float(input.design_fiber_percent())
        fragment = float(input.design_fragment_percent())
        fibre = 50.0 if fibre + fragment <= 0 else 100.0 * fibre / (fibre + fragment)
        return generate_synthetic_microplastics(
            n_particles=5000,
            size_ranges_um=[(low, high) for _, low, high in selected_design_micro_ranges()],
            polymer_percentages=polymers,
            fiber_percent=fibre,
            seed=84,
            size_distribution=str(input.design_size_distribution()),
        )

    def selected_design_macro_categories() -> list[str]:
        return list(input.design_macro_categories() or []) if str(input.design_plastic_type()) == "macroplastic" and str(input.design_macro_mode()) == "grouped" else []

    def selected_design_macro_items() -> list[str]:
        return list(input.design_macro_common_names() or []) if str(input.design_plastic_type()) == "macroplastic" and str(input.design_macro_mode()) in {"individual", "top_five"} else []

    @reactive.calc
    def selected_design_groups() -> list[tuple[str, np.ndarray]]:
        if str(input.design_plastic_type()) == "macroplastic":
            return selected_group_beta_values(
                micro_ranges=[], macro_selected=selected_design_macro_categories(),
                macro_items_selected=selected_design_macro_items(),
                use_macro_items=str(input.design_macro_mode()) in {"individual", "top_five"}, u_star=selected_design_u_star(),
            )
        df = selected_design_micro_df()
        return [("Microplastics", calculate_micro_rouse_mean(selected_design_u_star(), micro_df=df))] if not df.empty else []

    def selected_design_detail_groups() -> list[tuple[str, np.ndarray]]:
        if str(input.design_micro_detail()) != "size" or str(input.design_plastic_type()) != "microplastic":
            return []
        df = selected_design_micro_df()
        if df.empty:
            return []
        beta = calculate_micro_rouse_mean(selected_design_u_star(), micro_df=df)
        size_um = df["size_um"].to_numpy(dtype=float)
        groups = []
        for lower, upper, label in [(20, 100, "Size: 20–100 µm"), (100, 300, "Size: 100–300 µm"), (300, 1000, "Size: 300 µm–1 mm"), (1000, 3000, "Size: 1–3 mm"), (3000, 5000, "Size: 3–5 mm")]:
            values = finite(beta[(size_um >= lower) & (size_um <= upper)])
            if len(values):
                groups.append((f"Microplastics: {label}", values))
        return groups

    def selected_design_polymer_detail_groups() -> list[tuple[str, np.ndarray]]:
        if str(input.design_micro_detail()) != "polymer" or str(input.design_plastic_type()) != "microplastic":
            return []
        df = selected_design_micro_df()
        if df.empty:
            return []
        beta = calculate_micro_rouse_mean(selected_design_u_star(), micro_df=df)
        polymers = df["polymer"].astype(str).to_numpy()
        return [
            (f"Microplastics: polymer {polymer}", finite(beta[polymers == polymer]))
            for polymer in DEFAULT_POLYMER_PERCENTAGES
            if np.any(polymers == polymer)
        ]

    @reactive.calc
    def design_capture_values() -> np.ndarray:
        """Return capture fractions for the planned target population."""
        groups = selected_design_groups()
        beta = finite(np.concatenate([values for _, values in groups])) if groups else np.array([])
        intervals = selected_design_net_intervals()
        capture, _ = sampling_fraction_distribution_from_beta(
            beta_values=beta,
            H=float(applied_design_flow.get()["river_depth_m"]),
            a_bed_frac=float(input.design_a_bed_frac()),
            a_surf_frac=float(input.design_a_surf_frac()),
            net_z_intervals=intervals,
        )
        return finite(capture)

    @reactive.calc
    def design_capture_summary() -> tuple[float, float, float]:
        """Return median and selected percentile bounds for planned capture."""
        values = design_capture_values()
        if not len(values):
            return np.nan, np.nan, np.nan
        lower, upper = (float(value) for value in input.design_iqr_percentiles())
        return (
            float(np.nanmedian(values)),
            float(np.nanpercentile(values, lower)),
            float(np.nanpercentile(values, upper)),
        )

    @reactive.calc
    def design_median_capture() -> float:
        return design_capture_summary()[0]

    @render.plot(alt="Vertical Rouse concentration profile plot for sample design")
    def profile_plot_design():
        is_micro = str(input.design_plastic_type()) == "microplastic"
        detail = str(input.design_micro_detail()) if is_micro else "total"
        return make_profile_plot(
            micro_ranges=selected_design_micro_ranges(),
            macro_selected=selected_design_macro_categories(),
            macro_items_selected=selected_design_macro_items(),
            use_macro_items=str(input.design_macro_mode()) in {"individual", "top_five"},
            u_star=selected_design_u_star(), H=float(applied_design_flow.get()["river_depth_m"]),
            a_bed_frac=float(input.design_a_bed_frac()), a_surf_frac=float(input.design_a_surf_frac()),
            iqr_lower=float(input.design_iqr_percentiles()[0]), iqr_upper=float(input.design_iqr_percentiles()[1]),
            show_net_interval=bool(input.design_show_sampling_interval()),
            net_z_intervals=selected_design_net_intervals(),
            micro_df=selected_design_micro_df(),
            split_micro_by_direction=is_micro and detail == "summary",
            extra_micro_groups=(
                selected_design_detail_groups() if detail == "size"
                else selected_design_polymer_detail_groups() if detail == "polymer"
                else []
            ),
            include_micro_total=not (is_micro and detail in {"size", "polymer"}),
            show_iqr=False,
            display_actual_depth=True,
            warning_text=(
                "Not enough sample for representative results.\nSample a larger or different depth interval."
                if np.isfinite(design_median_capture()) and design_median_capture() < 0.05
                else None
            ),
        )

    @render.ui
    def design_key_results():
        intervals = selected_design_net_intervals()
        median_capture, _, _ = design_capture_summary()
        sampled_fraction = sum(abs(upper - lower) for lower, upper in intervals)
        capture_text = (
            f"{median_capture * 100:.1f}%"
            if np.isfinite(median_capture) else "Not available"
        )
        missed_text = (
            f"{(1.0 - median_capture) * 100:.1f}%"
            if np.isfinite(median_capture) else "Not available"
        )
        capture_note = ""
        if np.isfinite(median_capture) and median_capture < 0.05:
            capture_note = "Not enough sample for representative results. Sample a larger or different depth interval."
        return ui.div(
            ui.layout_columns(
                ui.value_box("Expected target plastics captured", capture_text, theme="primary", height="110px", fill=False),
                ui.value_box("Expected target plastics missed", missed_text, theme="primary", height="110px", fill=False),
                ui.value_box("Water-column fraction sampled", f"{sampled_fraction * 100:.1f}%", theme="primary", height="110px", fill=False),
                col_widths=[4, 4, 4], fill=False,
            ),
            ui.p(capture_note, class_="compact-note"),
            class_="sampling-key-results",
        )

    def design_capture_table() -> pd.DataFrame:
        """Return group-level capture and missed fractions for the design."""
        intervals = selected_design_net_intervals()
        is_micro = str(input.design_plastic_type()) == "microplastic"
        detail = str(input.design_micro_detail()) if is_micro else "total"
        return net_sampling_table(
            micro_ranges=selected_design_micro_ranges(),
            macro_selected=selected_design_macro_categories(),
            macro_items_selected=selected_design_macro_items(),
            use_macro_items=str(input.design_macro_mode()) in {"individual", "top_five"},
            u_star=selected_design_u_star(),
            micro_df=selected_design_micro_df(),
            H=float(applied_design_flow.get()["river_depth_m"]),
            a_bed_frac=float(input.design_a_bed_frac()),
            a_surf_frac=float(input.design_a_surf_frac()),
            net_z_intervals=intervals,
            iqr_lower=float(input.design_iqr_percentiles()[0]),
            iqr_upper=float(input.design_iqr_percentiles()[1]),
            split_micro_by_direction=is_micro and detail == "summary",
            extra_micro_groups=(
                selected_design_detail_groups() if detail == "size"
                else selected_design_polymer_detail_groups() if detail == "polymer"
                else []
            ),
            include_micro_total=True,
        )

    @render.data_frame
    def design_capture_results():
        depth = float(applied_design_flow.get()["river_depth_m"])
        lower_percentile, upper_percentile = (float(value) for value in input.design_iqr_percentiles())

        def format_result(values: np.ndarray) -> tuple[str, str]:
            captured, missed = sampling_fraction_distribution_from_beta(
                beta_values=finite(values), H=depth,
                a_bed_frac=float(input.design_a_bed_frac()),
                a_surf_frac=float(input.design_a_surf_frac()),
                net_z_intervals=selected_design_net_intervals(),
            )
            if not len(captured):
                return "Not available", "Not available"
            captured_text = f"{np.nanmedian(captured) * 100:.1f}% [{np.nanpercentile(captured, lower_percentile) * 100:.1f}–{np.nanpercentile(captured, upper_percentile) * 100:.1f}%]"
            missed_text = f"{np.nanmedian(missed) * 100:.1f}% [{np.nanpercentile(missed, lower_percentile) * 100:.1f}–{np.nanpercentile(missed, upper_percentile) * 100:.1f}%]"
            return captured_text, missed_text

        def result_row(fraction: str, plastic_type: str, values: np.ndarray) -> dict[str, str]:
            captured_text, missed_text = format_result(values)
            return {
                "Fraction": fraction,
                "Expected captured (%)": captured_text,
                "Expected missed (%)": missed_text,
            }

        if str(input.design_plastic_type()) == "microplastic":
            micro_df = selected_design_micro_df()
            beta = calculate_micro_rouse_mean(selected_design_u_star(), micro_df=micro_df)
            size_um = micro_df["size_um"].to_numpy(dtype=float)
            polymers = micro_df["polymer"].astype(str).to_numpy()
            rows = [
                result_row("Microplastics: total", "Microplastics", beta),
                result_row("Microplastics: buoyant", "Microplastics", beta[beta < 0]),
                result_row("Microplastics: sinking", "Microplastics", beta[beta >= 0]),
            ]
            for lower, upper, label in [(20, 100, "20–100 µm"), (100, 300, "100–300 µm"), (300, 1000, "300 µm–1 mm"), (1000, 3000, "1–3 mm"), (3000, 5000, "3–5 mm")]:
                values = beta[(size_um >= lower) & (size_um <= upper)]
                if len(values):
                    rows.append(result_row(f"Microplastics: {label}", "Microplastics", values))
            for polymer in DEFAULT_POLYMER_PERCENTAGES:
                values = beta[polymers == polymer]
                if len(values):
                    rows.append(result_row(f"Microplastics: polymer {polymer}", "Microplastics", values))
        else:
            groups = selected_design_groups()
            all_values = np.concatenate([values for _, values in groups]) if groups else np.array([])
            rows = [result_row("Macrolitter: total", "Macrolitter", all_values)]
            rows.extend(result_row(name, "Macrolitter", values) for name, values in groups)

        return render.DataGrid(pd.DataFrame(rows), width="100%", height="390px", filters=False, summary=False)

    @render.download(filename="river_plast_sample_design.xlsx")
    def download_design_parameters_excel():
        """Export the current Sample design settings as an Excel workbook."""
        flow = applied_design_flow.get()
        sampled_fraction = sum(abs(upper - lower) for lower, upper in selected_design_net_intervals())
        flow_sheet = pd.DataFrame(
            [
                {
                    "Parameter": "Water-column fraction sampled",
                    "Value": sampled_fraction * 100,
                },
                {"Parameter": "Width (m)", "Value": float(input.design_river_width())},
                {"Parameter": "Depth (m)", "Value": float(input.design_river_depth())},
                {"Parameter": "Slope (-)", "Value": float(input.design_slope())},
                {"Parameter": "Discharge (m³/s)", "Value": float(input.design_discharge())},
                {"Parameter": "Shear velocity mode", "Value": "Direct" if bool(input.design_direct_ustar()) else "Estimated from hydraulic radius and slope"},
                {"Parameter": "Shear velocity, u* (m/s)", "Value": float(flow["u_star"])},
            ]
        )

        plastic_type = str(input.design_plastic_type())
        if plastic_type == "microplastic":
            lower, upper = (float(value) for value in input.design_size_range())
            plastic_rows = [
                {"Parameter": "Plastic type", "Value": "Microplastics"},
                {"Parameter": "Population", "Value": "Typical microplastic population" if bool(input.design_use_estimated_distribution()) else "User-defined population"},
                {"Parameter": "Particle size range (µm)", "Value": f"{lower:g}–{upper:g}"},
                {"Parameter": "Size distribution", "Value": str(input.design_size_distribution())},
                {"Parameter": "Fibres (%)", "Value": float(input.design_fiber_percent())},
                {"Parameter": "Fragments (%)", "Value": float(input.design_fragment_percent())},
            ]
            plastic_rows.extend(
                {"Parameter": f"{name} (%)", "Value": float(getattr(input, f"design_polymer_{name}")())}
                for name in DEFAULT_POLYMER_PERCENTAGES
            )
        else:
            macro_mode = str(input.design_macro_mode())
            mode_label = {
                "top_five": "Top 5 most common river litter items",
                "individual": "Individual litter items",
                "grouped": "Grouped litter items",
            }[macro_mode]
            selected = (
                selected_design_macro_categories()
                if macro_mode == "grouped"
                else selected_design_macro_items()
            )
            plastic_rows = [
                {"Parameter": "Plastic type", "Value": "Macrolitter"},
                {"Parameter": "Selection", "Value": mode_label},
            ]
            plastic_rows.extend(
                {
                    "Parameter": "Selected item" if macro_mode != "grouped" else "Selected group",
                    "Value": macro_group_labels.get(value, value),
                }
                for value in selected
            )
        plastics_sheet = pd.DataFrame(plastic_rows)

        depth = float(flow["river_depth_m"])
        sampling_rows = []
        for number, (lower, upper) in enumerate(selected_design_net_intervals(), start=1):
            sampling_rows.append(
                {
                    "Sampling depth": number,
                    "Lower edge above bed (m)": lower * depth,
                    "Upper edge above bed (m)": upper * depth,
                    "Lower z/H": lower,
                    "Upper z/H": upper,
                }
            )
        sampling_sheet = pd.DataFrame(sampling_rows)

        advanced_sheet = pd.DataFrame(
            [
                {"Parameter": "Bed reference height, a_bed/H", "Value": float(input.design_a_bed_frac())},
                {"Parameter": "Surface reference offset, a_surf/H", "Value": float(input.design_a_surf_frac())},
                {"Parameter": "Lower particle variability percentile", "Value": float(input.design_iqr_percentiles()[0])},
                {"Parameter": "Upper particle variability percentile", "Value": float(input.design_iqr_percentiles()[1])},
            ]
        )
        lower_percentile, upper_percentile = (float(value) for value in input.design_iqr_percentiles())

        def result_row(fraction: str, plastic_type: str, beta_values: np.ndarray) -> dict[str, object]:
            captured, missed = sampling_fraction_distribution_from_beta(
                beta_values=finite(beta_values),
                H=depth,
                a_bed_frac=float(input.design_a_bed_frac()),
                a_surf_frac=float(input.design_a_surf_frac()),
                net_z_intervals=selected_design_net_intervals(),
            )
            return {
                "Fraction": fraction,
                "Plastic type": plastic_type,
                "Expected captured (%)": float(np.nanmedian(captured)) * 100 if len(captured) else np.nan,
                "Expected missed (%)": float(np.nanmedian(missed)) * 100 if len(missed) else np.nan,
                f"Captured {lower_percentile:g}th percentile (%)": float(np.nanpercentile(captured, lower_percentile)) * 100 if len(captured) else np.nan,
                f"Captured {upper_percentile:g}th percentile (%)": float(np.nanpercentile(captured, upper_percentile)) * 100 if len(captured) else np.nan,
                f"Missed {lower_percentile:g}th percentile (%)": float(np.nanpercentile(missed, lower_percentile)) * 100 if len(missed) else np.nan,
                f"Missed {upper_percentile:g}th percentile (%)": float(np.nanpercentile(missed, upper_percentile)) * 100 if len(missed) else np.nan,
            }

        if plastic_type == "microplastic":
            micro_df = selected_design_micro_df()
            beta = calculate_micro_rouse_mean(selected_design_u_star(), micro_df=micro_df)
            size_um = micro_df["size_um"].to_numpy(dtype=float)
            polymers = micro_df["polymer"].astype(str).to_numpy()
            result_rows = [result_row("Microplastics: total", "Microplastics", beta)]
            result_rows.extend(
                [
                    result_row("Microplastics: buoyant", "Microplastics", beta[beta < 0]),
                    result_row("Microplastics: sinking", "Microplastics", beta[beta >= 0]),
                ]
            )
            for lower, upper, label in [
                (20, 100, "20–100 µm"),
                (100, 300, "100–300 µm"),
                (300, 1000, "300 µm–1 mm"),
                (1000, 3000, "1–3 mm"),
                (3000, 5000, "3–5 mm"),
            ]:
                values = beta[(size_um >= lower) & (size_um <= upper)]
                if len(values):
                    result_rows.append(result_row(f"Microplastics: {label}", "Microplastics", values))
            for polymer in DEFAULT_POLYMER_PERCENTAGES:
                values = beta[polymers == polymer]
                if len(values):
                    result_rows.append(result_row(f"Microplastics: polymer {polymer}", "Microplastics", values))
        else:
            groups = selected_design_groups()
            all_values = np.concatenate([values for _, values in groups]) if groups else np.array([])
            result_rows = [result_row("Macrolitter: total", "Macrolitter", all_values)]
            result_rows.extend(
                result_row(name, "Macrolitter", values)
                for name, values in groups
            )
        results_sheet = pd.DataFrame(result_rows)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_name, frame in {
                "Results": results_sheet,
                "Flow": flow_sheet,
                "Plastics": plastics_sheet,
                "Sampling": sampling_sheet,
                "Advanced": advanced_sheet,
            }.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = "A2"
                for column_cells in worksheet.columns:
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                        max(max(len(str(cell.value or "")) for cell in column_cells) + 2, 14),
                        48,
                    )
        yield output.getvalue()

    def selected_samp_polymer_raw_percentages() -> dict[str, float]:
        return {name: float(getattr(input, f"samp_polymer_{name}")()) for name in DEFAULT_POLYMER_PERCENTAGES}

    def selected_samp_polymer_total() -> float:
        return float(sum(selected_samp_polymer_raw_percentages().values()))

    def selected_samp_polymer_percentages() -> dict[str, float]:
        raw = selected_samp_polymer_raw_percentages()
        total = float(sum(raw.values()))
        if not polymer_total_valid(total):
            return {name: 0.0 for name in raw}
        return raw

    def selected_samp_shape_percentages() -> tuple[float, float]:
        fibre = float(input.samp_fiber_percent())
        fragment = float(input.samp_fragment_percent())
        total = fibre + fragment
        if total <= 0:
            return 50.0, 50.0
        return 100.0 * fibre / total, 100.0 * fragment / total

    def selected_samp_micro_ranges() -> list[tuple[str, float, float]]:
        if not bool(input.samp_select_microplastics()):
            return []
        ranges = [("synthetic MP", *input.samp_synthetic_size_range())]
        if bool(input.samp_add_size_group()):
            ranges = [
                ("Group 1", *input.samp_synthetic_size_range()),
                ("Group 2", *input.samp_synthetic_size_range_2()),
            ]
        return [
            (name, float(lo), float(hi))
            for name, lo, hi in ranges if hi > lo
        ]

    @reactive.Effect
    def _update_micro_concentration_labels():
        if bool(input.samp_add_size_group()):
            lo, hi = input.samp_synthetic_size_range()
            label = f"Group 1 ({lo:g}–{hi:g} µm): measured concentration"
        else:
            label = "Measured concentration in sample"
        ui.update_numeric("samp_measured_concentration", label=label)
        lo, hi = input.samp_synthetic_size_range_2()
        ui.update_numeric(
            "samp_measured_concentration_2",
            label=f"Group 2 ({lo:g}–{hi:g} µm): measured concentration",
        )

    def selected_samp_micro_concentrations() -> dict[str, float]:
        concentrations = {}
        for name, lo, hi in selected_samp_micro_ranges():
            group_name = (
                "Microplastics" if name == "synthetic MP"
                else f"Microplastics: {name} ({lo:g}–{hi:g} µm)"
            )
            value = (
                input.samp_measured_concentration_2()
                if name == "Group 2"
                else input.samp_measured_concentration()
            )
            concentrations[group_name] = float(value)
        return concentrations

    @reactive.calc
    def selected_samp_micro_df() -> pd.DataFrame:
        if not bool(input.samp_select_microplastics()):
            return empty_synthetic_microplastics_df()
        if not polymer_total_valid(selected_samp_polymer_total()):
            return empty_synthetic_microplastics_df()

        ranges = selected_samp_micro_ranges()
        if not ranges:
            return empty_synthetic_microplastics_df()
        return generate_synthetic_microplastics(
            # Five thousand particles gives stable medians and quartiles for
            # the interactive app while avoiding repeated profile work over a
            # much larger Monte Carlo population.
            n_particles=5000,
            size_ranges_um=[(lo, hi) for _, lo, hi in ranges],
            polymer_percentages=selected_samp_polymer_percentages(),
            fiber_percent=selected_samp_shape_percentages()[0],
            seed=42,
            size_distribution=str(input.samp_synthetic_size_distribution()),
        )

    def selected_samp_micro_detail_groups() -> list[tuple[str, np.ndarray]]:
        """Return optional size-class and polymer beta groups for display."""
        if not bool(input.samp_select_microplastics()):
            return []

        df = selected_samp_micro_df()
        if df.empty or "size_um" not in df.columns:
            return []

        selected_mask = np.zeros(len(df), dtype=bool)
        size_um = df["size_um"].to_numpy(dtype=float)
        for _, lower, upper in selected_samp_micro_ranges():
            selected_mask |= (size_um >= lower) & (size_um <= upper)

        beta = calculate_micro_rouse_mean(selected_samp_u_star(), micro_df=df)
        groups = []
        if str(input.samp_micro_detail()) == "size":
            size_classes = [
                (20, 100, "Size: 20–100 µm"),
                (100, 300, "Size: 100–300 µm"),
                (300, 1000, "Size: 300 µm–1 mm"),
                (1000, 3000, "Size: 1–3 mm"),
                (3000, 5000, "Size: 3–5 mm"),
            ]
            for lower, upper, label in size_classes:
                mask = selected_mask & (size_um >= lower) & (size_um <= upper)
                values = finite(beta[mask])
                if len(values) > 0:
                    groups.append((f"Microplastics: {label}", values))

        if str(input.samp_micro_detail()) == "polymer" and "polymer" in df.columns:
            polymers = df["polymer"].astype(str).to_numpy()
            options = selected_samp_polymer_display_options()
            try:
                selected_names = list(input.samp_profile_polymers() or [])
            except Exception:
                selected_names = list(options)
            for display_name in selected_names:
                model_polymer = options.get(display_name)
                if model_polymer is None:
                    continue
                values = finite(beta[selected_mask & (polymers == model_polymer)])
                if len(values) > 0:
                    groups.append((f"Microplastics: polymer {display_name}", values))

        return groups

    def selected_samp_polymer_display_options() -> dict[str, str]:
        """Map the displayed polymer names to the model polymer classes."""
        records = imported_particle_records.get()
        if records is not None and not records.empty and "polymer" in records.columns:
            names = records["polymer"].dropna().astype(str).str.strip()
            options = {
                name: standardise_polymer_name(name)
                for name in names.drop_duplicates()
            }
            options = {
                name: model_name
                for name, model_name in options.items()
                if model_name is not None
            }
            if options:
                return options
        return {name: name for name in selected_samp_polymer_raw_percentages()}

    @render.ui
    def samp_polymer_profile_picker_ui():
        if str(input.samp_micro_detail()) != "polymer":
            return ui.div()
        options = selected_samp_polymer_display_options()
        try:
            current = list(input.samp_profile_polymers() or [])
        except Exception:
            current = []
        selected = [name for name in current if name in options]
        if not selected:
            selected = list(options)[:4]
        return ui.input_selectize(
            "samp_profile_polymers",
            "Polymers shown on profile",
            choices=list(options),
            selected=selected,
            multiple=True,
            options={
                "plugins": ["remove_button"],
                "placeholder": "Choose polymers to plot",
                "dropdownParent": "body",
            },
        )

    def show_samp_micro_detail() -> bool:
        return str(input.samp_micro_detail()) in {"size", "polymer"}

    def show_samp_micro_direction_profiles() -> bool:
        return str(input.samp_micro_detail()) == "summary"

    def selected_samp_macro_categories() -> list[str]:
        if not bool(input.samp_select_macroplastics()):
            return []
        return list(input.samp_macro_categories() or [])

    def use_samp_macro_items() -> bool:
        return (
            bool(input.samp_select_macroplastics())
            and str(input.samp_macro_mode()) == "individual"
        )

    def selected_samp_macro_items() -> list[str]:
        if not bool(input.samp_select_macroplastics()):
            return []
        return list(input.samp_macro_common_names() or [])

    @render.ui
    def samp_macro_item_concentrations_ui():
        if not bool(input.samp_select_macroplastics()):
            return ui.div()

        if use_samp_macro_items():
            items = selected_samp_macro_items()
            if not items:
                return ui.div()
            imported_concentrations = imported_macro_concentrations.get()
            return ui.div(
                ui.h6("Measured concentration by item"),
                *[
                    ui.input_numeric(
                        macro_item_concentration_input_id(common_name),
                        common_name,
                        value=float(imported_concentrations.get(common_name, 0.0)),
                        min=0.0,
                        step=0.01,
                    )
                    for common_name in items
                ],
            )

        return ui.div()

    def selected_samp_macro_item_concentrations() -> dict[str, float]:
        if not use_samp_macro_items():
            return {}

        concentrations = {}
        for common_name in selected_samp_macro_items():
            input_id = macro_item_concentration_input_id(common_name)
            try:
                value = getattr(input, input_id)()
            except Exception:
                value = 0.0
            concentrations[common_name] = max(float(value or 0.0), 0.0)
        return concentrations

    def selected_samp_macro_group_concentrations() -> dict[str, float]:
        if (
            not bool(input.samp_select_macroplastics())
            or use_samp_macro_items()
        ):
            return {}

        concentrations = {}
        for group_key in selected_samp_macro_categories():
            input_id = macro_group_concentration_input_id(group_key)
            try:
                value = getattr(input, input_id)()
            except Exception:
                value = 0.0
            concentrations[group_key] = max(float(value or 0.0), 0.0)
        return concentrations

    @reactive.Effect
    @reactive.event(input.samp_apply_flow)
    def _apply_sampling_flow_values():
        mode = "direct" if bool(input.samp_direct_ustar()) else "hydraulic"
        width = float(input.samp_river_width())
        depth = float(input.samp_river_depth())
        hydraulic_radius = width * depth / (width + 2.0 * depth)
        slope = float(input.samp_slope())
        direct_u_star = float(input.samp_u_star())
        if mode == "direct":
            u_star = direct_u_star
        else:
            u_star = calculate_shear_velocity_from_slope_radius(
                hydraulic_radius=hydraulic_radius,
                slope=slope,
            )

        applied_sampling_flow.set(
            {
                "u_star": u_star,
                "discharge": float(input.samp_discharge()),
                "mode": mode,
                "river_width_m": width,
                "river_depth_m": depth,
                "hydraulic_radius_m": hydraulic_radius,
                "slope": slope,
                "direct_u_star_m_s": direct_u_star,
            }
        )
        sample_top = max(0.0, min(1.0, 1.0 - float(input.samp_a_bed_frac())))
        sample_bottom = max(0.0, sample_top - 0.10)
        ui.update_slider(
            "samp_net_z_interval",
            value=(sample_bottom, sample_top),
            session=session,
        )

    def selected_samp_u_star() -> float:
        return float(applied_sampling_flow.get()["u_star"])

    def selected_samp_discharge() -> float:
        return float(applied_sampling_flow.get()["discharge"])

    @render.text
    def samp_applied_flow_text():
        flow = applied_sampling_flow.get()
        return (
            f"Applied: u* = {float(flow['u_star']):.4f} m/s; "
            f"Q = {float(flow['discharge']):.3g} m³/s"
        )

    def selected_samp_iqr_percentiles() -> tuple[float, float]:
        q_low, q_high = input.samp_iqr_percentiles()
        return float(q_low), float(q_high)

    def selected_samp_net_interval() -> tuple[float, float]:
        z_min, z_max = input.samp_net_z_interval()
        return float(z_min), float(z_max)


    def current_sampling_correction_table(
        include_discharge: bool,
        include_macro_members: bool,
    ) -> pd.DataFrame:
        """Return the correct result model for the active plastic workflow."""
        if use_samp_macro_items():
            return macro_item_correction_table(
                item_concentrations=selected_samp_macro_item_concentrations(),
                u_star=selected_samp_u_star(),
                H=selected_flow_depth(),
                a_bed_frac=float(input.samp_a_bed_frac()),
                a_surf_frac=float(input.samp_a_surf_frac()),
                net_z_min=selected_samp_net_interval()[0],
                net_z_max=selected_samp_net_interval()[1],
                concentration_units=str(input.samp_concentration_units()),
                include_discharge=include_discharge,
                discharge=selected_samp_discharge(),
                iqr_lower=selected_samp_iqr_percentiles()[0],
                iqr_upper=selected_samp_iqr_percentiles()[1],
            )

        if bool(input.samp_select_macroplastics()):
            return macro_item_correction_table(
                item_concentrations=selected_samp_macro_group_concentrations(),
                u_star=selected_samp_u_star(),
                H=selected_flow_depth(),
                a_bed_frac=float(input.samp_a_bed_frac()),
                a_surf_frac=float(input.samp_a_surf_frac()),
                net_z_min=selected_samp_net_interval()[0],
                net_z_max=selected_samp_net_interval()[1],
                concentration_units=str(input.samp_concentration_units()),
                include_discharge=include_discharge,
                discharge=selected_samp_discharge(),
                iqr_lower=selected_samp_iqr_percentiles()[0],
                iqr_upper=selected_samp_iqr_percentiles()[1],
                component_type="group",
            )

        return sampling_correction_table(
            micro_ranges=selected_samp_micro_ranges(),
            macro_selected=selected_samp_macro_categories(),
            macro_items_selected=selected_samp_macro_items(),
            use_macro_items=use_samp_macro_items(),
            u_star=selected_samp_u_star(),
            micro_df=selected_samp_micro_df(),
            H=selected_flow_depth(),
            a_bed_frac=float(input.samp_a_bed_frac()),
            a_surf_frac=float(input.samp_a_surf_frac()),
            net_z_min=selected_samp_net_interval()[0],
            net_z_max=selected_samp_net_interval()[1],
            measured_concentration=float(input.samp_measured_concentration()),
            micro_concentrations=selected_samp_micro_concentrations(),
            concentration_units=str(input.samp_concentration_units()),
            include_discharge=include_discharge,
            discharge=selected_samp_discharge(),
            iqr_lower=selected_samp_iqr_percentiles()[0],
            iqr_upper=selected_samp_iqr_percentiles()[1],
            include_macro_members=include_macro_members,
        )

    def build_net_sampling_results_df() -> pd.DataFrame:
        """Return the sampling depth estimate table as a plain DataFrame."""
        if not samp_net_sampling_enabled():
            return pd.DataFrame({"Output": ["Turn on 'Show captured/missed estimate' to calculate capture fractions."]})

        if bool(input.samp_select_macroplastics()):
            df = current_sampling_correction_table(
                include_discharge=False,
                include_macro_members=True,
            )
        else:
            df = net_sampling_table(
                micro_ranges=selected_samp_micro_ranges(),
                macro_selected=selected_samp_macro_categories(),
                macro_items_selected=selected_samp_macro_items(),
                use_macro_items=use_samp_macro_items(),
                u_star=selected_samp_u_star(),
                micro_df=selected_samp_micro_df(),
                H=selected_flow_depth(),
                a_bed_frac=float(input.samp_a_bed_frac()),
                a_surf_frac=float(input.samp_a_surf_frac()),
                net_z_min=selected_samp_net_interval()[0],
                net_z_max=selected_samp_net_interval()[1],
                iqr_lower=selected_samp_iqr_percentiles()[0],
                iqr_upper=selected_samp_iqr_percentiles()[1],
                split_micro_by_direction=show_samp_micro_direction_profiles(),
                extra_micro_groups=selected_samp_micro_detail_groups(),
                include_micro_total=True,
            )
        df = df.rename(
            columns={
                "Population (%)": "Population %",
                "Capture (%)": "Capture %",
                "Missed (%)": "Missed %",
            }
        )
        keep_cols = [
            c for c in [
                "Group",
                "Population %",
                "Capture %",
                "Missed %",
            ] if c in df.columns
        ]
        return df.loc[:, keep_cols]

    def build_depth_average_results_df() -> pd.DataFrame:
        """Return the depth-averaged concentration table as a plain DataFrame."""
        if not samp_sampling_correction_enabled():
            return pd.DataFrame({"Output": ["Turn on 'Show concentration correction' to calculate depth-averaged concentration."]})

        df = current_sampling_correction_table(
            include_discharge=selected_samp_discharge() > 0,
            include_macro_members=True,
        )
        df = df.rename(
            columns={
                "Estimated depth-averaged concentration": "Estimated depth-average concentration",
            }
        )
        keep_cols = [
            c for c in [
                "Group",
                "Measured concentration",
                "Units",
                "Estimated depth-average concentration",
            ] if c in df.columns
        ]
        return df.loc[:, keep_cols]

    def build_load_results_df() -> pd.DataFrame:
        """Return the estimated-load table as a plain DataFrame."""
        if selected_samp_discharge() <= 0:
            return pd.DataFrame({"Output": ["Enter a river discharge greater than 0 to calculate estimated load."]})
        if not samp_sampling_correction_enabled():
            return pd.DataFrame({"Output": ["Turn on 'Show concentration correction' to calculate estimated load."]})

        df = current_sampling_correction_table(
            include_discharge=True,
            include_macro_members=False,
        )
        df = df.rename(
            columns={
                "Discharge Q (m3/s)": "Discharge (m³/s)",
                "Estimated load": "Estimated load",
                "Load units": "Load units",
            }
        )
        keep_cols = [
            c for c in [
                "Group",
                "Discharge (m³/s)",
                "Estimated load",
                "Load units",
            ] if c in df.columns
        ]
        return df.loc[:, keep_cols]

    def optional_input_value(input_id: str, default):
        """Read a dynamic Shiny input without failing if it is not mounted."""
        try:
            value = getattr(input, input_id)()
        except Exception:
            return default
        return default if value is None else value

    def build_excel_input_frames() -> dict[str, pd.DataFrame]:
        """Build the editable input sheets used for Excel round trips."""
        applied_flow = applied_sampling_flow.get()
        flow_mode = str(applied_flow["mode"])
        flow = pd.DataFrame(
            [
                {
                    "u_star_mode": flow_mode,
                    "river_width_m": float(applied_flow["river_width_m"]),
                    "river_depth_m": float(applied_flow["river_depth_m"]),
                    "hydraulic_radius_m": float(
                        applied_flow["hydraulic_radius_m"]
                    ),
                    "slope": float(applied_flow["slope"]),
                    "direct_u_star_m_s": float(
                        applied_flow["direct_u_star_m_s"]
                    ),
                    "applied_u_star_m_s": selected_samp_u_star(),
                    "discharge_m3_s": selected_samp_discharge(),
                }
            ]
        )

        z_min, z_max = selected_samp_net_interval()
        sample = pd.DataFrame(
            [
                {
                    "sample_z_min": z_min,
                    "sample_z_max": z_max,
                    "concentration_units": str(
                        input.samp_concentration_units()
                    ),
                }
            ]
        )

        plastic_rows = []
        if bool(input.samp_select_microplastics()):
            for name, concentration in selected_samp_micro_concentrations().items():
                plastic_rows.append(
                    {
                        "plastic_type": "microplastics",
                        "selection_type": "population",
                        "identifier": name,
                        "name": name,
                        "measured_concentration": concentration,
                    }
                )
        elif use_samp_macro_items():
            concentrations = selected_samp_macro_item_concentrations()
            for common_name in selected_samp_macro_items():
                plastic_rows.append(
                    {
                        "plastic_type": "macrolitter",
                        "selection_type": "individual",
                        "identifier": common_name,
                        "name": common_name,
                        "measured_concentration": concentrations.get(
                            common_name,
                            0.0,
                        ),
                    }
                )
        else:
            concentrations = selected_samp_macro_group_concentrations()
            for group_key in selected_samp_macro_categories():
                plastic_rows.append(
                    {
                        "plastic_type": "macrolitter",
                        "selection_type": "grouped",
                        "identifier": group_key,
                        "name": macro_group_labels[group_key],
                        "measured_concentration": concentrations.get(
                            group_key,
                            0.0,
                        ),
                    }
                )
        plastics = pd.DataFrame(
            plastic_rows,
            columns=[
                "plastic_type",
                "selection_type",
                "identifier",
                "name",
                "measured_concentration",
            ],
        )

        size_ranges = selected_samp_micro_ranges() or [
            ("synthetic MP", *input.samp_synthetic_size_range())
        ]
        polymers = selected_samp_polymer_raw_percentages()
        fibre, fragment = selected_samp_shape_percentages()
        microplastics = pd.DataFrame(
            [
                {
                    "size_group": group_name,
                    "size_min_um": float(size_min),
                    "size_max_um": float(size_max),
                    "size_distribution": str(
                        input.samp_synthetic_size_distribution()
                    ),
                    "fibre_percent": fibre,
                    "fragment_percent": fragment,
                    **{
                        f"{name}_percent": value
                        for name, value in polymers.items()
                    },
                }
                for group_name, size_min, size_max in size_ranges
            ]
        )

        return {
            "Flow inputs": flow,
            "Sample inputs": sample,
            "Plastic inputs": plastics,
            "Microplastic inputs": microplastics,
        }

    def build_sampling_excel_bytes() -> bytes:
        """Return a complete editable-input and results workbook."""
        instructions = pd.DataFrame(
            {
                "Sheet": [
                    "Flow inputs",
                    "Sample inputs",
                    "Plastic inputs",
                    "Microplastic inputs",
                    "Captured fraction",
                    "Corrected concentration",
                    "Estimated load",
                ],
                "Purpose": [
                    "River hydraulics, shear-velocity mode, and discharge.",
                    "Sampled z/H limits and concentration units.",
                    "Selected plastic population and measured concentrations.",
                    "Synthetic microplastic size, shape, and polymer settings.",
                    "Calculated captured and missed fractions.",
                    "Measured and estimated depth-average concentrations.",
                    "Calculated discharge and load estimates.",
                ],
                "Contents": [
                    "Flow values currently applied in the app.",
                    "Current sample settings.",
                    "Current plastic selection and measured concentrations.",
                    "Current microplastic population settings.",
                    "Current calculated result.",
                    "Current calculated result.",
                    "Current calculated result.",
                ],
            }
        )
        sheets = {
            "Instructions": instructions,
            **build_excel_input_frames(),
            "Captured fraction": build_net_sampling_results_df(),
            "Corrected concentration": build_depth_average_results_df(),
            "Estimated load": build_load_results_df(),
        }
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = "A2"
                for column_cells in worksheet.columns:
                    max_length = max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in column_cells
                    )
                    worksheet.column_dimensions[
                        column_cells[0].column_letter
                    ].width = min(max(max_length + 2, 12), 48)
        return output.getvalue()

    def build_section_example_excel_bytes(sheet_names: list[str]) -> bytes:
        """Return editable long-form data sheets for one sidebar section."""
        flow = pd.DataFrame([{
            "sample_id": "RIV001",
            "river_width_m": 20.0,
            "river_depth_m": 1.2,
            "slope": 0.0005,
            "discharge_m3_s": 20.0,
            "direct_u_star_m_s": np.nan,
        }])
        samples = pd.DataFrame([{
            "sample_id": "RIV001",
            "sample_z_min": 0.80,
            "sample_z_max": 1.00,
            "sampled_volume_m3": 0.5,
            "site_name": "River Avon",
            "sample_date": "2026-09-25",
        }])
        particles = pd.DataFrame([
            {"sample_id": "RIV001", "particle_id": "RIV001_MP001", "plastic_type": "microplastic", "polymer": "PE", "size_um": 125, "shape": "fibre", "macro_item": "", "count_weight": 1},
            {"sample_id": "RIV001", "particle_id": "RIV001_MP002", "plastic_type": "microplastic", "polymer": "PET", "size_um": 300, "shape": "fragment", "macro_item": "", "count_weight": 1},
            {"sample_id": "RIV001", "particle_id": "RIV001_MP003", "plastic_type": "microplastic", "polymer": "PP", "size_um": 600, "shape": "fibre", "macro_item": "", "count_weight": 1},
        ])
        input_sheets = {"Flow data": flow, "Samples": samples, "Particles": particles}
        instructions = pd.DataFrame(
            {
                "Instructions": [
                    "Replace the example values, keep the column names unchanged, and upload one sample at a time. Upload Sample data and Particles together to calculate concentration from count_weight / sampled_volume_m3."
                ]
            }
        )
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            instructions.to_excel(writer, sheet_name="Instructions", index=False)
            for sheet_name in sheet_names:
                input_sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
            for worksheet in writer.sheets.values():
                worksheet.freeze_panes = "A2"
                for column_cells in worksheet.columns:
                    max_length = max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in column_cells
                    )
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                        max(max_length + 2, 12), 48
                    )
        return output.getvalue()

    def build_batch_example_excel_bytes() -> bytes:
        """Return the four-sheet workbook used for batch import."""
        flow = pd.DataFrame([
            {"river_id": "RIV001", "river_name": "River Exe", "sample_id": "RIV001-S1", "date": "01/01/2020", "river_width_m": 20.0, "river_depth_m": 1.2, "slope": 0.0005, "discharge_m3_s": 20.0, "direct_u_star_m_s": np.nan},
            {"river_id": "RIV001", "river_name": "River Exe", "sample_id": "RIV001-S2", "date": "01/02/2020", "river_width_m": 20.0, "river_depth_m": 1.0, "slope": 0.0005, "discharge_m3_s": 15.0, "direct_u_star_m_s": np.nan},
            {"river_id": "RIV001", "river_name": "River Exe", "sample_id": "RIV001-S3", "date": "01/03/2020", "river_width_m": 20.0, "river_depth_m": 0.8, "slope": 0.0005, "discharge_m3_s": 10.0, "direct_u_star_m_s": np.nan},
            {"river_id": "RIV002", "river_name": "River Wye", "sample_id": "RIV002-S1", "date": "24/01/2020", "river_width_m": 15.0, "river_depth_m": 1.0, "slope": 0.0003, "discharge_m3_s": 10.0, "direct_u_star_m_s": np.nan},
            {"river_id": "RIV002", "river_name": "River Wye", "sample_id": "RIV002-S2", "date": "27/02/2020", "river_width_m": 20.0, "river_depth_m": 0.5, "slope": 0.0003, "discharge_m3_s": 8.0, "direct_u_star_m_s": np.nan},
            {"river_id": "RIV003", "river_name": "River Severn", "sample_id": "RIV003-S1", "date": "17/10/2020", "river_width_m": 20.0, "river_depth_m": 0.4, "slope": 0.0003, "discharge_m3_s": 7.5, "direct_u_star_m_s": np.nan},
        ])
        sample = pd.DataFrame([
            {"river_id": "RIV001", "sample_id": "RIV001-S1", "sample_z_min": 0.80, "sample_z_max": 1.00, "sampled_volume_m3": 0.75},
            {"river_id": "RIV001", "sample_id": "RIV001-S2", "sample_z_min": 0.50, "sample_z_max": 0.70, "sampled_volume_m3": 0.60},
            {"river_id": "RIV001", "sample_id": "RIV001-S3", "sample_z_min": 0.70, "sample_z_max": 0.90, "sampled_volume_m3": 0.40},
            {"river_id": "RIV002", "sample_id": "RIV002-S1", "sample_z_min": 0.80, "sample_z_max": 1.00, "sampled_volume_m3": 0.50},
            {"river_id": "RIV002", "sample_id": "RIV002-S2", "sample_z_min": 0.40, "sample_z_max": 0.60, "sampled_volume_m3": 0.35},
            {"river_id": "RIV003", "sample_id": "RIV003-S1", "sample_z_min": 0.75, "sample_z_max": 0.95, "sampled_volume_m3": 0.30},
        ])
        sample_ids = flow["sample_id"].tolist()
        micro_polymers = ["poly(ethylene)", "pp", "pet", "poly(-methylstyrene)"]
        micro_sizes = [125, 300, 650, 1000]
        micro_shapes = ["fibre", "fragment", "fibre", "fragment"]
        microplastics = pd.DataFrame([
            {
                "sample_id": sample_id,
                "river_id": sample_id.split("-")[0],
                "particle_id": f"{sample_id}-MP{particle_number:03d}",
                "polymer": micro_polymers[(particle_number - 1) % 4],
                "size_um": micro_sizes[(particle_number - 1) % 4],
                "shape": micro_shapes[(particle_number - 1) % 4],
                "count_weight": 1,
            }
            for sample_index, sample_id in enumerate(sample_ids)
            for particle_number in range(sample_index * 12 + 1, sample_index * 12 + 13)
        ])
        macro_examples = [
            (64, "Cigarette filters"),
            (15, "Plastic bottle caps and lids"),
            (117.1, "Hard plastic pieces 0.5-2.5 cm"),
            (22.1, "Straws"),
        ]
        macroplastic = pd.DataFrame([
            {
                "sample_id": sample_id,
                "river_id": sample_id.split("-")[0],
                "particle_id": f"{sample_id}-MaP{particle_number:03d}",
                "OSPAR_ID": macro_examples[(particle_number - 1) % 4][0],
                "OSPAR_name": macro_examples[(particle_number - 1) % 4][1],
                "count_weight": 1,
            }
            for sample_index, sample_id in enumerate(sample_ids)
            for particle_number in range(sample_index * 4 + 1, sample_index * 4 + 5)
        ])
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for name, frame in {"Flow": flow, "Sample": sample, "microplastics": microplastics, "macroplastic": macroplastic}.items():
                frame.to_excel(writer, sheet_name=name, index=False)
                worksheet = writer.sheets[name]
                worksheet.freeze_panes = "A2"
                for column_cells in worksheet.columns:
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                        max(max(len(str(cell.value or "")) for cell in column_cells) + 2, 12), 48
                    )
        return output.getvalue()

    @render.download(filename="river_plast_batch_example.xlsx")
    def download_batch_example_excel():
        yield (EXAMPLES_DIR / "batch_import_example.xlsx").read_bytes()

    @render.download(filename="river_plast_batch_results.xlsx")
    def download_batch_results_excel():
        sheets = imported_batch_workbook.get()

        def format_export(frame: pd.DataFrame) -> pd.DataFrame:
            """Match the Results by fraction table shown below the graph."""
            table = frame.copy()
            captured = table.get("Expected capture (%)", pd.Series(np.nan, index=table.index, dtype=object))
            missed = table.get("Expected missed (%)", pd.Series(np.nan, index=table.index, dtype=object))
            if "Capture (%)" in table:
                captured = captured.fillna(table["Capture (%)"])
            if "Missed (%)" in table:
                missed = missed.fillna(table["Missed (%)"])
            concentration_units = display_unit(table["Units"].dropna().iloc[0]) if "Units" in table and table["Units"].notna().any() else ""
            load_units = str(table["Load units"].dropna().iloc[0]) if "Load units" in table and table["Load units"].notna().any() else ""
            return pd.DataFrame({
                "Fraction": table.get("Group", ""),
                "Captured (%)": captured,
                "Missed (%)": missed,
                f"Measured concentration ({concentration_units})" if concentration_units else "Measured concentration": table.get("Measured concentration", ""),
                f"Depth-average concentration ({concentration_units})" if concentration_units else "Depth-average concentration": table.get("Estimated depth-averaged concentration", ""),
                f"Load ({load_units})" if load_units else "Load": table.get("Estimated load", ""),
            })

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            results = imported_batch_results.get()
            if results.empty:
                pd.DataFrame({"Message": ["Import a RIVER-PLAST workbook before exporting batch results."]}).to_excel(
                    writer, sheet_name="Results", index=False
                )
            else:
                used_names = set()
                for sample_id in results["sample_id"].dropna().astype(str).unique():
                    sheet_name = re.sub(r"[\\/*?:\[\]]", "_", sample_id)[:31] or "Sample"
                    base_name = sheet_name
                    suffix = 2
                    while sheet_name in used_names:
                        sheet_name = f"{base_name[:28]}_{suffix}"
                        suffix += 1
                    used_names.add(sheet_name)

                    sample_results = results[results["sample_id"].astype(str) == sample_id]
                    flow_data = sheets.get("Flow", pd.DataFrame())
                    sample_data = sheets.get("Sample", pd.DataFrame())
                    flow_row = flow_data[flow_data["sample_id"].astype(str) == sample_id].iloc[0] if not flow_data.empty else pd.Series(dtype=object)
                    depth_rows = sample_data[sample_data["sample_id"].astype(str) == sample_id] if not sample_data.empty else pd.DataFrame()
                    sampling_depths = ", ".join(
                        f"{float(row['sample_z_min']):.2f}–{float(row['sample_z_max']):.2f} z/H"
                        for _, row in depth_rows.iterrows()
                    )
                    flow_section = pd.DataFrame({
                        "Parameter": ["River depth (m)", "Width (m)", "Slope (-)", "Discharge (m³/s)", "Sampling depths"],
                        "Value": [
                            flow_row.get("river_depth_m", ""),
                            flow_row.get("river_width_m", ""),
                            flow_row.get("slope", ""),
                            flow_row.get("discharge_m3_s", ""),
                            sampling_depths,
                        ],
                    })
                    microplastics = sample_results[
                        sample_results["result_set"].str.startswith("Micro")
                    ].rename(columns={"result_set": "Breakdown"})
                    macroplastics = sample_results[
                        sample_results["result_set"] == "Macrolitter"
                    ].drop(columns="result_set")
                    pd.DataFrame({"Flow": [""]}).to_excel(
                        writer, sheet_name=sheet_name, index=False, startrow=0
                    )
                    flow_section.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
                    start_row = len(flow_section) + 4
                    if not microplastics.empty:
                        pd.DataFrame({"Microplastics": [""]}).to_excel(
                            writer, sheet_name=sheet_name, index=False, startrow=start_row
                        )
                        start_row += 2
                        format_export(microplastics).to_excel(
                            writer, sheet_name=sheet_name, index=False, startrow=start_row
                        )
                        start_row += len(microplastics) + 3
                    if not macroplastics.empty:
                        pd.DataFrame({"Macrolitter": [""]}).to_excel(
                            writer, sheet_name=sheet_name, index=False, startrow=start_row
                        )
                        start_row += 2
                        format_export(macroplastics).to_excel(
                            writer, sheet_name=sheet_name, index=False, startrow=start_row
                        )
        yield output.getvalue()

    @render.download(filename="river_plast_flow_example.xlsx")
    def download_flow_example_excel():
        yield build_section_example_excel_bytes(["Flow data"])

    @render.download(filename="river_plast_plastics_example.xlsx")
    def download_plastics_example_excel():
        yield build_section_example_excel_bytes(["Particles"])

    @render.download(filename="river_plast_sample_example.xlsx")
    def download_sample_example_excel():
        yield build_section_example_excel_bytes(["Samples"])

    @render.download(filename="river_plastic_sampling_report.xlsx")
    def download_sampling_excel():
        yield build_sampling_excel_bytes()

    def samp_net_sampling_enabled() -> bool:
        """Captured/missed estimates are always enabled on the correction page."""
        return True

    def samp_sampling_correction_enabled() -> bool:
        """Concentration correction is always enabled on the correction page."""
        return True



    @render.text
    def samp_polymer_total_text():
        total = selected_samp_polymer_total()
        if not polymer_total_valid(total):
            return f"Polymer total is {total:.0f}%; adjusting linked sliders to 100%."
        return "Polymer total is 100%. Moving one slider rescales the others."

    @render.text
    def samp_shape_total_text():
        fibre_pct, fragment_pct = selected_samp_shape_percentages()
        return f"Total: {fibre_pct + fragment_pct:.0f}% | Fibres {fibre_pct:.0f}%, fragments {fragment_pct:.0f}%"

    @render.plot(alt="Vertical Rouse concentration profile plot")
    def profile_plot_sampling():
        return make_profile_plot(
            micro_ranges=selected_samp_micro_ranges(),
            macro_selected=selected_samp_macro_categories(),
            macro_items_selected=selected_samp_macro_items(),
            use_macro_items=use_samp_macro_items(),
            u_star=selected_samp_u_star(),
            micro_df=selected_samp_micro_df(),
            H=selected_flow_depth(),
            a_bed_frac=float(input.samp_a_bed_frac()),
            a_surf_frac=float(input.samp_a_surf_frac()),
            iqr_lower=selected_samp_iqr_percentiles()[0],
            iqr_upper=selected_samp_iqr_percentiles()[1],
            show_net_interval=bool(input.samp_show_sampling_interval()),
            net_z_interval=selected_samp_net_interval(),
            split_micro_by_direction=show_samp_micro_direction_profiles(),
            extra_micro_groups=selected_samp_micro_detail_groups(),
            include_micro_total=not show_samp_micro_detail(),
            show_iqr=bool(input.samp_show_iqr()),
        )












    @render.plot(alt="Sampling-tab synthetic microplastic size probability density plot")
    def samp_size_pdf_plot():
        df = selected_samp_micro_df()
        fig, ax = plt.subplots(figsize=(2.6, 1.7))
        size_um = pd.to_numeric(df.get("size_um"), errors="coerce").dropna().to_numpy(dtype=float)
        if len(size_um) > 0:
            ax.hist(size_um, bins=35, density=True, alpha=0.8)
            ax.set_yscale("log")
        ax.set_xlabel("Size (µm)", fontsize=PLOT_FONT_STANDARD)
        ax.set_ylabel("Density", fontsize=PLOT_FONT_STANDARD)
        ax.tick_params(axis="both", labelsize=PLOT_FONT_STANDARD)
        ax.grid(True, alpha=0.18)
        fig.tight_layout(pad=0.6)
        return fig

    @render.plot(alt="Sampling-tab synthetic microplastic shape mixture pie chart")
    def samp_shape_mix_plot():
        fibre_pct, fragment_pct = selected_samp_shape_percentages()
        fig, ax = plt.subplots(figsize=(2.4, 1.7))
        values = [fibre_pct, fragment_pct]
        labels = ["Fibres", "Fragments"]
        if sum(values) <= 0:
            ax.text(
                0.5,
                0.5,
                "No shape\nselected",
                ha="center",
                va="center",
                fontsize=PLOT_FONT_STANDARD,
            )
            ax.axis("off")
        else:
            ax.pie(
                values,
                labels=labels,
                autopct="%.0f%%",
                textprops={"fontsize": PLOT_FONT_STANDARD},
            )
        fig.tight_layout(pad=0.5)
        return fig

    @render.plot(alt="Sampling-tab synthetic microplastic polymer mixture pie chart")
    def samp_polymer_mix_plot():
        fig, ax = plt.subplots(figsize=(2.4, 1.7))
        if not polymer_total_valid(selected_samp_polymer_total()):
            ax.text(
                0.5,
                0.5,
                "Polymer total\nmust be 100%",
                ha="center",
                va="center",
                fontsize=PLOT_FONT_STANDARD,
            )
            ax.axis("off")
            fig.tight_layout(pad=0.5)
            return fig

        polymer_percentages = selected_samp_polymer_percentages()
        labels = []
        values = []
        for name, value in polymer_percentages.items():
            if value > 0:
                labels.append(name)
                values.append(value)
        if sum(values) <= 0:
            ax.text(
                0.5,
                0.5,
                "No polymer\nselected",
                ha="center",
                va="center",
                fontsize=PLOT_FONT_STANDARD,
            )
            ax.axis("off")
        else:
            ax.pie(
                values,
                labels=labels,
                autopct="%.0f%%",
                textprops={"fontsize": PLOT_FONT_STANDARD},
            )
        fig.tight_layout(pad=0.5)
        return fig

    @render.data_frame
    def net_sampling_results():
        if not samp_net_sampling_enabled():
            df = pd.DataFrame({"Output": ["Turn on 'Show captured/missed estimate' to calculate capture fractions."]})
        elif bool(input.samp_select_macroplastics()):
            df = current_sampling_correction_table(
                include_discharge=False,
                include_macro_members=True,
            )
            df = df.rename(
                columns={
                    "Population (%)": "Population %",
                    "Capture (%)": "Capture %",
                    "Missed (%)": "Missed %",
                }
            )
            df = df.loc[:, ["Group", "Capture %", "Missed %"]]
        else:
            df = net_sampling_table(
                micro_ranges=selected_samp_micro_ranges(),
                macro_selected=selected_samp_macro_categories(),
                macro_items_selected=selected_samp_macro_items(),
                use_macro_items=use_samp_macro_items(),
                u_star=selected_samp_u_star(),
                micro_df=selected_samp_micro_df(),
                H=selected_flow_depth(),
                a_bed_frac=float(input.samp_a_bed_frac()),
                a_surf_frac=float(input.samp_a_surf_frac()),
                net_z_min=selected_samp_net_interval()[0],
                net_z_max=selected_samp_net_interval()[1],
                iqr_lower=selected_samp_iqr_percentiles()[0],
                iqr_upper=selected_samp_iqr_percentiles()[1],
                split_micro_by_direction=show_samp_micro_direction_profiles(),
                extra_micro_groups=selected_samp_micro_detail_groups(),
                include_micro_total=True,
            )
            df = df.rename(
                columns={
                    "Population (%)": "Population %",
                    "Capture (%)": "Capture %",
                    "Missed (%)": "Missed %",
                }
            )
            keep_cols = [
                c for c in [
                    "Group",
                    "Population %",
                    "Capture %",
                    "Missed %",
                ] if c in df.columns
            ]
            df = df.loc[:, keep_cols]

        return render.DataGrid(
            df,
            width="100%",
            height="165px",
            filters=False,
            summary=False,
        )


    @render.data_frame
    def sampling_correction_results():
        if not samp_sampling_correction_enabled():
            df = pd.DataFrame({"Output": ["Turn on 'Show concentration correction' to calculate depth-averaged concentration."]})
        else:
            df = current_sampling_correction_table(
                include_discharge=selected_samp_discharge() > 0,
                include_macro_members=True,
            )
            df = df.rename(
                columns={
                    "Measured concentration": "Measured concentration",
                    "Estimated depth-averaged concentration": "Estimated depth-average concentration",
                    "Discharge Q (m3/s)": "Discharge (m³/s)",
                    "Estimated load": "Estimated load",
                    "Load units": "Load units",
                }
            )
            keep_cols = [
                c for c in [
                    "Group",
                    "Measured concentration",
                    "Units",
                    "Estimated depth-average concentration",
                ] if c in df.columns
            ]
            df = df.loc[:, keep_cols]

        return render.DataGrid(
            df,
            width="100%",
            height="220px",
            filters=False,
            summary=False,
        )


    @render.data_frame
    def discharge_load_results():
        if selected_samp_discharge() <= 0:
            df = pd.DataFrame({"Output": ["Enter a river discharge greater than 0 to calculate estimated load."]})
        elif not samp_sampling_correction_enabled():
            df = pd.DataFrame({"Output": ["Turn on 'Show concentration correction' to calculate estimated load."]})
        else:
            df = current_sampling_correction_table(
                include_discharge=True,
                include_macro_members=False,
            )
            df = df.rename(
                columns={
                    "Discharge Q (m3/s)": "Discharge (m³/s)",
                    "Estimated load": "Estimated load",
                    "Load units": "Load units",
                }
            )
            keep_cols = [
                c for c in [
                    "Group",
                    "Discharge (m³/s)",
                    "Estimated load",
                    "Load units",
                ] if c in df.columns
            ]
            df = df.loc[:, keep_cols]

        return render.DataGrid(
            df,
            width="100%",
            height="95px",
            filters=False,
            summary=False,
        )




    @render.download(filename="sampling_synthetic_particles.csv")
    def download_sampling_synthetic_csv():
        """Download the Sampling correction synthetic dataset."""
        yield selected_samp_micro_df().to_csv(index=False)

    @render.download(filename="sampling_depth_estimate.csv")
    def download_net_sampling_results_csv():
        """Download the sampling depth estimate table."""
        yield build_net_sampling_results_df().to_csv(index=False)

    @render.download(filename="depth_averaged_concentration.csv")
    def download_depth_average_results_csv():
        """Download the depth-averaged concentration results table."""
        yield build_depth_average_results_df().to_csv(index=False)

    @render.download(filename="estimated_load.csv")
    def download_load_results_csv():
        """Download the estimated load results table."""
        yield build_load_results_df().to_csv(index=False)



app = App(app_ui, server)
