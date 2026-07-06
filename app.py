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

Required files in the same folder:
    microplastic_particles_settling.csv
    macroplastic_particles_settling.xlsx

This version:
    - Keeps only the vertical Rouse-profile figure.
    - Uses a normal linear x-axis from 0 to 1.
    - Lets the user define up to three microplastic size ranges.
    - Places all user controls in collapsible sidebar sections.
    - Lets macroplastics be shown either by grouped category or by individual litter item.
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
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from synthetic_microplastics import (
    DEFAULT_POLYMER_PERCENTAGES,
    POLYMER_DENSITY_RANGES_G_CM3,
    generate_synthetic_microplastics,
)


# ============================================================
# LOAD DATA
# ============================================================
micro = pd.read_csv("microplastic_particles_settling.csv")
macro = pd.read_excel("macroplastic_particles_settling.xlsx")


# ============================================================
# CONSTANTS
# ============================================================
kappa = 0.41
g = 9.81

velocity_cols = [
    "velocity_dietrich",
    "velocity_goral",
    "velocity_yu",
]

# Microplastic sizes are stored in metres in the input CSV.
# The UI exposes size ranges in micrometres for readability.
micro_size_min_um = 1
micro_size_max_um = max(5000, int(np.ceil(float(micro["size"].max()) * 1e6)))

macro_mapping = {
    "PO hard": "near_neutral",
    "PO soft": "near_neutral",
    "PS": "near_neutral",
    "Multilayer": "near_neutral",
    "Textiles": "near_neutral",
    "Paper": "near_neutral",
    "PET": "near_neutral",
    "Glass": "dense",
    "Metal": "dense",
    "EPS": "buoyant",
}

macro_group_labels = {
    "buoyant": "Buoyant \nρₚ ∈ [0.02, 0.08] g cm⁻³",
    "near_neutral": "Near neutral \nρₚ ∈ [0.8, 1.5] g cm⁻³",
    "dense": "Dense\nρₚ ∈ [2.5, 4.3] g cm⁻³",
}


# ============================================================
# STATIC PREP
# ============================================================
micro["size_um"] = micro["size"].astype(float) * 1e6

macro["Material_grouped"] = macro["Material"].map(macro_mapping)

if "Common name" in macro.columns:
    macro_common_names = sorted(
        macro["Common name"].dropna().astype(str).unique().tolist()
    )
else:
    macro_common_names = []


# ============================================================
# HELPERS
# ============================================================
def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]

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


def calculate_micro_rouse_mean(u_star: float, micro_df: pd.DataFrame | None = None) -> np.ndarray:
    """
    Calculate mean microplastic Rouse number across the three velocity equations.

        beta = w / (kappa u*)

    The default source is the measured microplastic dataset loaded from CSV.
    When synthetic mode is enabled, the app passes a generated microplastic
    dataframe with the same velocity columns.
    """
    df = micro if micro_df is None else micro_df

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
    Calculate macroplastic Rouse number.

    Preserves your earlier conversion:
        macro_w = vz_mean / 100
    """
    w = (macro["vz_mean"] / 100).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    return w / (kappa * u_star)


def beta_values_for_micro_range(
    min_um: float,
    max_um: float,
    u_star: float,
    micro_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Return beta values for one user-selected microplastic size range.

    Size inputs are in micrometres. The underlying dataset stores size in metres,
    so a precomputed size_um column is used for direct filtering.
    """
    df = micro if micro_df is None else micro_df

    if max_um <= min_um or "size_um" not in df.columns:
        return np.array([])

    beta = calculate_micro_rouse_mean(u_star, micro_df=df)
    mask = (df["size_um"] >= min_um) & (df["size_um"] <= max_um)

    return finite(beta[mask.to_numpy()])


def beta_values_for_macro_group(group_key: str, u_star: float) -> np.ndarray:
    """Return beta values for one macroplastic buoyancy/density group."""
    if group_key not in macro_group_labels:
        return np.array([])

    beta = calculate_macro_rouse(u_star)
    mask = macro["Material_grouped"] == group_key

    return finite(beta[mask.to_numpy()])


def beta_values_for_macro_item(common_name: str, u_star: float) -> np.ndarray:
    """Return beta values for one macroplastic item using the Common name column."""
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

    Settling particles, beta >= 0:
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
        # Settling profile: high near bed, low toward surface.
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

    profiles = []

    for beta in beta_values:
        z_rel, c_rel = rouse_profile_from_beta(
            beta=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            n=n,
        )

        if len(z_rel) == 0 or len(c_rel) == 0:
            continue

        profiles.append(c_rel)

    if len(profiles) == 0:
        return pd.DataFrame(columns=["z_rel", "median", "q_low", "q_high"])

    profiles = np.vstack(profiles)

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
    net_z_min: float,
    net_z_max: float,
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
    captured_values = []
    missed_values = []

    for beta in beta_values:
        z_rel, c_rel = rouse_profile_from_beta(
            beta=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            n=n,
        )

        if len(z_rel) == 0 or len(c_rel) == 0:
            continue

        captured, missed = sampling_fraction_from_profile(
            z_rel=z_rel,
            c=c_rel,
            net_z_min=net_z_min,
            net_z_max=net_z_max,
        )

        if np.isfinite(captured) and np.isfinite(missed):
            captured_values.append(captured)
            missed_values.append(missed)

    return np.asarray(captured_values, dtype=float), np.asarray(missed_values, dtype=float)


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
        return (
            f"{fmt_sig(med * 100)}% "
            f"[{fmt_sig(low * 100)}% - {fmt_sig(high * 100)}%]"
        )

    return fmt_interval(med, low, high)


def net_sampling_table(
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
    iqr_lower: float,
    iqr_upper: float,
    micro_df: pd.DataFrame | None = None,
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
    )

    for group_name, beta in groups:
        captured_values, missed_values = sampling_fraction_distribution_from_beta(
            beta_values=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            net_z_min=net_z_min,
            net_z_max=net_z_max,
        )

        rows.append(
            {
                "Group": group_name,
                "Sampled z/H interval": f"{min(net_z_min, net_z_max):.2f}–{max(net_z_min, net_z_max):.2f}",
                "Water-column fraction sampled": round(abs(float(net_z_max) - float(net_z_min)), 3),
                "Capture fraction": format_median_iqr(captured_values, iqr_lower, iqr_upper),
                "Missed fraction": format_median_iqr(missed_values, iqr_lower, iqr_upper),
                "Captured (%)": format_median_iqr(captured_values, iqr_lower, iqr_upper, percent=True),
                "Missed (%)": format_median_iqr(missed_values, iqr_lower, iqr_upper, percent=True),
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
) -> pd.DataFrame:
    """Estimate depth-averaged concentration and optional river load.

    Each calculated variable is reported as:

        median [lower percentile - upper percentile]

    The percentiles are controlled by the uncertainty-band slider.
    """
    rows = []
    c_obs = float(measured_concentration)
    q = float(discharge)

    groups = selected_group_beta_values(
        micro_ranges=micro_ranges,
        macro_selected=macro_selected,
        macro_items_selected=macro_items_selected,
        use_macro_items=use_macro_items,
        u_star=u_star,
        micro_df=micro_df,
    )

    for group_name, beta in groups:
        captured_values, missed_values = sampling_fraction_distribution_from_beta(
            beta_values=beta,
            H=H,
            a_bed_frac=a_bed_frac,
            a_surf_frac=a_surf_frac,
            net_z_min=net_z_min,
            net_z_max=net_z_max,
        )

        valid_captured = captured_values[np.isfinite(captured_values) & (captured_values > 0)]

        if len(valid_captured) > 0:
            correction_factor_values = 1.0 / valid_captured
            corrected_concentration_values = c_obs * correction_factor_values
        else:
            correction_factor_values = np.array([], dtype=float)
            corrected_concentration_values = np.array([], dtype=float)

        if include_discharge and np.isfinite(q) and q >= 0:
            load_values = corrected_concentration_values * q
        else:
            load_values = np.array([], dtype=float)
        
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
                "Capture fraction": format_median_iqr(captured_values, iqr_lower, iqr_upper),
                "Missed fraction": format_median_iqr(missed_values, iqr_lower, iqr_upper),
                "Correction factor": format_median_iqr(correction_factor_values, iqr_lower, iqr_upper),
                "Estimated depth-averaged concentration": format_median_iqr(corrected_concentration_values, iqr_lower, iqr_upper),
                "Discharge Q (m3/s)": round(q, 4) if include_discharge else np.nan,
                "Estimated load": format_median_iqr(load_values, iqr_lower, iqr_upper) if include_discharge else "",
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

def selected_group_beta_values(
    micro_ranges: list[tuple[str, float, float]],
    macro_selected: list[str],
    macro_items_selected: list[str],
    use_macro_items: bool,
    u_star: float,
    micro_df: pd.DataFrame | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Return display names and beta arrays for all selected ranges/categories/items."""
    groups = []

    for range_name, min_um, max_um in micro_ranges:
        beta = beta_values_for_micro_range(min_um=min_um, max_um=max_um, u_star=u_star, micro_df=micro_df)
        if str(range_name).lower() == "synthetic mp":
            groups.append(("Microplastics", beta))
        else:
            groups.append((f"Microplastics: {range_name} ({min_um:g}–{max_um:g} µm)", beta))

    if use_macro_items:
        for common_name in macro_items_selected:
            beta = beta_values_for_macro_item(common_name, u_star)
            groups.append((f"Macro item: {common_name}", beta))
    else:
        for group_key in macro_selected:
            beta = beta_values_for_macro_group(group_key, u_star)
            groups.append((f"Macro: {macro_group_labels[group_key]}", beta))

    return groups


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
    micro_df: pd.DataFrame | None = None,
) -> plt.Figure:
    """
    Build the vertical Rouse profile figure.

    x-axis:
        Normalised relative concentration, 0 to 1.

    y-axis:
        Relative height, z/H.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    plotted_any = False

    groups = selected_group_beta_values(
        micro_ranges=micro_ranges,
        macro_selected=macro_selected,
        macro_items_selected=macro_items_selected,
        use_macro_items=use_macro_items,
        u_star=u_star,
        micro_df=micro_df,
    )

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

        ax.plot(
            summary["median"],
            summary["z_rel"],
            linewidth=2.5,
            label=group_name,
        )

        ax.fill_betweenx(
            summary["z_rel"],
            summary["q_low"],
            summary["q_high"],
            alpha=0.18,
        )

        plotted_any = True

    ax.axhline(
        a_bed_frac,
        linestyle="--",
        color="grey",
        linewidth=1,
        alpha=0.25,
    )

    ax.axhline(
        1 - a_surf_frac,
        linestyle="--",
        color="grey",
        linewidth=1,
        alpha=0.25,
    )
    
    ax.text(
        0.015,
        a_bed_frac,
        r"$a_{bed}$",
        ha="left",
        va="bottom",
        alpha=0.5,
        transform=ax.get_yaxis_transform(),
    )
    
    ax.text(
        0.015,
        1 - a_surf_frac,
        r"$a_{surf}$",
        ha="left",
        va="top",
        alpha=0.5,
        transform=ax.get_yaxis_transform(),
    )
    

    if show_net_interval and net_z_interval is not None:
        net_z_min, net_z_max = net_z_interval
        net_low = max(a_bed_frac, min(float(net_z_min), float(net_z_max)))
        net_high = min(1 - a_surf_frac, max(float(net_z_min), float(net_z_max)))

        if net_high > net_low:
            ax.axhspan(
                net_low,
                net_high,
                alpha=0.08,
                zorder=0,
                label="Sampling depth interval",
            )
            ax.axhline(
                net_low,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
            )
            ax.axhline(
                net_high,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
            )
            ax.text(
                0.015,
                (net_low + net_high) / 2,
                f"Sample: {net_low:.2f}–{net_high:.2f} z/H",
                ha="left",
                va="center",
                fontsize=9,
                alpha=0.85,
                transform=ax.get_yaxis_transform(),
            )

    ax.set_xlabel(r"Normalised concentration, $C / C_{max}$", fontsize=9)
    ax.set_ylabel(r"Relative depth, $z/H$", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks(
        np.arange(0, 1.01, 0.1)
    )
    ax.set_xlim(0, 1)
    
    ax.axhline(
    0,
    color="black",
    linewidth=1.5,
    )
    
    ax.axhline(
        1,
        color="black",
        linewidth=1.5,
    )

    ax.set_title(
        rf"Vertical profiles, $u_*$ = {u_star:.3f} m s$^{{-1}}$",
        fontsize=10,
    )
    ax.tick_params(axis="both", labelsize=8)

    ax.grid(True, alpha=0.25)

    if plotted_any:
        ax.legend(loc="best", fontsize=8)
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
APP_DIR = Path(__file__).resolve().parent
METHODS_FILE = APP_DIR / "Vertical_Plastic_Profile_App_Methods.md"

try:
    methods_text = METHODS_FILE.read_text(encoding="utf-8")
except FileNotFoundError:
    methods_text = (
        "# Methods file not found\n\n"
        "The file `Vertical_Plastic_Profile_App_Methods.md` was not found. "
        "Place it in the same folder as this app file."
    )


# ============================================================
# REUSABLE UI BLOCKS
# ============================================================
def sampling_plastic_controls_ui() -> ui.Tag:
    """Right-panel plastic controls for the Sampling correction page.

    Uses sampling-specific input ids so the sampling page can be configured
    independently from the Explorer page while keeping the same visual layout.
    """
    return ui.div(
        ui.h3("Plastic controls"),
        ui.tags.details(
            ui.tags.summary("Microplastics"),
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
                        ui.tags.details(
                            ui.tags.summary("Advanced size controls"),
                            ui.div(
                                ui.input_select(
                                    "samp_synthetic_size_distribution",
                                    "Size distribution",
                                    choices={
                                        "loguniform": "Log-uniform",
                                        "uniform": "Uniform",
                                        "lognormal": "Truncated lognormal",
                                    },
                                    selected="loguniform",
                                ),
                                class_="collapsible-control-body",
                            ),
                            open=False,
                            class_="collapsible-control nested-control",
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
                        ui.output_text("samp_shape_total_text"),
                        class_="collapsible-control-body",
                    ),
                    open=False,
                    class_="collapsible-control nested-control",
                ),

                ui.tags.details(
                    ui.tags.summary("Polymer"),
                    ui.div(
                        ui.input_action_button(
                            "samp_reset_polymer_mix",
                            "Reset to default %",
                            class_="btn-sm btn-outline-secondary",
                        ),
                        ui.div(
                            ui.output_text("samp_polymer_total_text"),
                            class_="input-warning",
                        ),
                        ui.input_slider("samp_polymer_PE", "PE (%) ρₚ = 0.89–0.98 g cm⁻³", min=0, max=100, value=25, step=1),
                        ui.input_slider("samp_polymer_PET", "PET (%) ρₚ = 0.96–1.45 g cm⁻³", min=0, max=100, value=17, step=1),
                        ui.input_slider("samp_polymer_PA", "PA (%) ρₚ = 1.02–1.16 g cm⁻³", min=0, max=100, value=12, step=1),
                        ui.input_slider("samp_polymer_PP", "PP (%) ρₚ = 0.83–0.92 g cm⁻³", min=0, max=100, value=14, step=1),
                        ui.input_slider("samp_polymer_PS", "PS (%) ρₚ = 1.04–1.10 g cm⁻³", min=0, max=100, value=9, step=1),
                        ui.input_slider("samp_polymer_PVA", "PVA (%) ρₚ = 1.19–1.31 g cm⁻³", min=0, max=100, value=6, step=1),
                        ui.input_slider("samp_polymer_PVC", "PVC (%) ρₚ = 1.10–1.58 g cm⁻³", min=0, max=100, value=17, step=1),
                        class_="collapsible-control-body",
                    ),
                    open=False,
                    class_="collapsible-control nested-control",
                ),
                class_="collapsible-control-body",
            ),
            open=True,
            class_="collapsible-control",
        ),

        ui.tags.details(
            ui.tags.summary("Macroplastics"),
            ui.div(
                ui.div(
                    "Use grouped macroplastic classes by default, or switch on item-level selection from the Common name column.",
                    class_="helper-text",
                ),
                ui.input_checkbox(
                    "samp_use_macro_items",
                    "Select individual litter items",
                    False,
                ),
                ui.panel_conditional(
                    "!input.samp_use_macro_items",
                    ui.input_checkbox_group(
                        "samp_macro_categories",
                        "Grouped macroplastic classes",
                        choices={
                            "buoyant": ui.HTML("Buoyant (Foams)<br><small>ρₚ ∈ [0.02, 0.08] g cm⁻³</small>"),
                            "near_neutral": ui.HTML("Near neutral (Plastics & others)<br><small>ρₚ ∈ [0.8, 1.5] g cm⁻³</small>"),
                            "dense": ui.HTML("Dense (Glass & metal)<br><small>ρₚ ∈ [2.5, 4.3] g cm⁻³</small>"),
                        },
                        selected=[],
                    ),
                ),
                ui.panel_conditional(
                    "input.samp_use_macro_items",
                    ui.input_selectize(
                        "samp_macro_common_names",
                        "Individual litter items",
                        choices=macro_common_names,
                        selected=[],
                        multiple=True,
                        options={"placeholder": "Search or scroll through litter items", "plugins": ["remove_button"]},
                    ),
                    ui.div("No individual items are selected by default.", class_="helper-text"),
                ),
                class_="collapsible-control-body",
            ),
            open=False,
            class_="collapsible-control",
        ),
        class_="right-control-panel",
    )


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Introduction",
        ui.tags.style(
            """
            .intro-page-wrap {
                max-width: 1180px;
                margin: 0.75rem auto;
            }
            .intro-hero-card .card-body {
                padding: 1.05rem 1.25rem;
            }
            .intro-hero-card h3 {
                margin-top: 0;
                margin-bottom: 0.45rem;
                font-size: 1.25rem;
            }
            .intro-hero-card p {
                margin-bottom: 0.45rem;
                line-height: 1.35;
            }
            .intro-card-small .card-body {
                padding: 0.85rem 1rem;
            }
            .intro-card-small h4 {
                font-size: 0.98rem;
                margin-top: 0;
                margin-bottom: 0.45rem;
            }
            .intro-card-small p,
            .intro-card-small li {
                font-size: 0.84rem;
                line-height: 1.28;
            }
            .intro-card-small ul,
            .intro-card-small ol {
                margin-top: 0.2rem;
                margin-bottom: 0.2rem;
                padding-left: 1.15rem;
            }
            .intro-citation-card .card-body {
                padding: 0.8rem 1rem;
            }
            .intro-citation-card p,
            .intro-citation-card li {
                font-size: 0.78rem;
                line-height: 1.22;
            }
            .intro-citation-card ul {
                margin-top: 0.2rem;
                margin-bottom: 0.2rem;
                padding-left: 1.05rem;
            }
            .intro-muted {
                color: #555;
            }
            """
        ),
        ui.div(
            ui.card(
                ui.card_header("River Plastic Vertical Profiler"),
                ui.markdown(
                    """
### Introduction

This app is a scientific tool for exploring how microplastics and macroplastics may be distributed vertically in a river water column.

                    """
                ),
                class_="intro-hero-card",
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("What the tool does"),
                    ui.markdown(
                        """
1. **Generates a synthetic microplastic population** from user-selected size, shape, and polymer assumptions.
2. **Uses macroplastic data** from the Lofty (2026) dataset.
3. **Calculates settling or rising velocities** using Dietrich (1982), Goral (2023), and Yu (2022) equations.
4. **Converts velocities into Rouse numbers** and estimates vertical concentration profiles.
5. **Calculates depth-average concentration corrections** from the captured or missed fraction of a defined sampling location.
6. **Exports generated particles and result tables** for checking and analysis outside the app.
                        """
                    ),
                    class_="intro-card-small",
                ),
                ui.card(
                    ui.card_header("Pages"),
                    ui.markdown(
                        """
**Explorer** — build a synthetic microplastic population, add optional macroplastic groups, and view predicted vertical concentration profiles.

**Settling and rising velocities** — generate a synthetic microplastic population and compare predicted velocity distributions from Dietrich, Goral, and Yu.

**Sampling correction** — define a sampling design, estimate captured and missed fractions, estimate depth-averaged concentration, and estimate plastic load when discharge is supplied.

**About & Methods** — read the equations, assumptions, limitations, and interpretation notes.
                        """
                    ),
                    class_="intro-card-small",
                ),
                col_widths=[6, 6],
            ),
            ui.card(
                ui.card_header("Citations"),
                ui.markdown(
                    """
**Tool**: XXXX

**Rouse profile validation**

- Valero, D., Belay, B.S., Moreno-Rodenas, A., Kramer, M. and Franca, M.J. 2022. *Water Research* 226, 119078. DOI: 10.1016/j.watres.2022.119078.
- Lofty, J., Valero, D., Moreno-Rodenas, A., Belay, B.S., Wilson, C., Ouro, P. and Franca, M.J. 2024. *Water Research* 254, 121306. DOI: 10.1016/j.watres.2024.121306.
- Born, M.P., Brüll, C., Schaefer, D., Hillebrand, G. and Schüttrumpf, H. 2023. *Environmental Science & Technology* 57(14), 5569–5579. DOI: 10.1021/acs.est.2c06885.

**Settling and rising velocity equations**

- Dietrich, W.E. 1982. *Water Resources Research* 18(6), 1615–1626. DOI: 10.1029/WR018i006p01615.
- Goral, K.D. et al. 2023. *Environmental Research* 228, 115783. DOI: 10.1016/j.envres.2023.115783.
- Yu, Z., Yang, G. and Zhang, W. 2022. *Marine Pollution Bulletin* 176, 113449. DOI: 10.1016/j.marpolbul.2022.113449.
- Lofty, J., Valero, D. and Franca, M. 2026. *Settling and Rising Dynamics of River Litter*. EarthArXiv.
                    """
                ),
                class_="intro-citation-card",
            ),
            class_="intro-page-wrap",
        ),
    ),

    ui.nav_panel(
        "Explorer",
        ui.page_sidebar(
            ui.sidebar(
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
                    .square-plot-card {
                        max-width: 520px;
                        margin-left: auto;
                        margin-right: auto;
                    }
                    .square-plot-card .card-body {
                        display: flex;
                        justify-content: center;
                    }
                    .square-plot-card img,
                    .square-plot-card canvas,
                    .square-plot-card svg {
                        max-width: 500px;
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

                ui.div(
                    ui.markdown(
                        """
                        **Workflow**

                        1. Set hydraulics  
                        2. Choose microplastic and macroplastics  
                        3. View vertical profiles  
                        """
                    ),
                    class_="control-workflow",
                ),

                ui.h3("Hydraulics"),

                ui.tags.details(
                    ui.tags.summary("Shear velocity"),
                    ui.div(
                        ui.input_radio_buttons(
                            "ustar_mode",
                            "Set shear velocity",
                            choices={
                                "direct": "Direct u*",
                                "hydraulic": "Calculate u* from R and S",
                            },
                            selected="direct",
                        ),
                        ui.output_ui("ustar_controls"),
                        class_="collapsible-control-body",
                    ),
                    open=True,
                    class_="collapsible-control",
                ),

                ui.tags.details(
                    ui.tags.summary("Reference and display"),
                    ui.div(
                        ui.input_slider(
                            "a_bed_frac",
                            "Bed reference height a_bed/H",
                            min=0.01,
                            max=0.30,
                            value=0.05,
                            step=0.01,
                        ),
                        ui.input_slider(
                            "a_surf_frac",
                            "Surface reference offset a_surf/H",
                            min=0.01,
                            max=0.30,
                            value=0.01,
                            step=0.01,
                        ),
                        ui.input_slider(
                            "iqr_percentiles",
                            "Uncertainty band percentiles",
                            min=0,
                            max=100,
                            value=(25, 75),
                            step=1,
                        ),
                        class_="collapsible-control-body",
                    ),
                    open=False,
                    class_="collapsible-control",
                ),

                width="360px",
            ),

            ui.div(
                ui.div(
                    ui.h2("Vertical concentration profiles"),
                    ui.card(
                        ui.output_plot("profile_plot_basic", height="480px"),
                        full_screen=True,
                        class_="plot-card square-plot-card",
                    ),
                    ui.div(
                        ui.card(
                            ui.card_header("Size distribution"),
                            ui.output_plot("size_pdf_plot", height="140px"),
                            class_="mini-diagnostic-card",
                        ),
                        ui.card(
                            ui.card_header("Shape mix"),
                            ui.output_plot("shape_mix_plot", height="140px"),
                            class_="mini-diagnostic-card",
                        ),
                        ui.card(
                            ui.card_header("Polymer mix"),
                            ui.output_plot("polymer_mix_plot", height="140px"),
                            class_="mini-diagnostic-card",
                        ),
                        class_="diagnostic-grid",
                    ),
                    ui.div(
                        ui.download_button(
                            "download_explorer_synthetic_csv",
                            "Download synthetic particles CSV",
                            class_="btn-sm btn-outline-primary",
                        ),
                        style="text-align:center; margin-top:0.5rem;",
                    ),
                    class_="centre-analysis-panel",
                ),

                ui.div(
                    ui.h3("Plastic controls"),
                    ui.tags.details(
                        ui.tags.summary("Microplastics"),
                        ui.div(
                            ui.tags.details(
                                ui.tags.summary("Size"),
                                ui.div(
                                    ui.input_slider(
                                        "synthetic_size_range",
                                        "Particle size limits (µm)",
                                        min=20,
                                        max=5000,
                                        value=(20, 5000),
                                        step=10,
                                    ),
                                    ui.tags.details(
                                        ui.tags.summary("Advanced size controls"),
                                        ui.div(
                                            ui.input_select(
                                                "synthetic_size_distribution",
                                                "Size distribution",
                                                choices={
                                                    "loguniform": "Log-uniform",
                                                    "uniform": "Uniform",
                                                    "lognormal": "Truncated lognormal",
                                                },
                                                selected="loguniform",
                                            ),
                                            class_="collapsible-control-body",
                                        ),
                                        open=False,
                                        class_="collapsible-control nested-control",
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
                                        "synthetic_fiber_percent",
                                        "Fibres (%)",
                                        min=0,
                                        max=100,
                                        value=50,
                                        step=1,
                                    ),
                                    ui.input_slider(
                                        "synthetic_fragment_percent",
                                        "Fragments (%)",
                                        min=0,
                                        max=100,
                                        value=50,
                                        step=1,
                                    ),
                                    ui.output_text("shape_total_text"),
                                    class_="collapsible-control-body",
                                ),
                                open=False,
                                class_="collapsible-control nested-control",
                            ),

                            ui.tags.details(
                                ui.tags.summary("Polymer"),
                                ui.div(
                                    ui.input_action_button(
                                        "reset_polymer_mix",
                                        "Reset to default %",
                                        class_="btn-sm btn-outline-secondary",
                                    ),
                                    ui.div(
                                        ui.output_text("polymer_total_text"),
                                        class_="input-warning",
                                    ),
                                    ui.input_slider("polymer_PE", "PE (%) ρₚ = 0.89–0.98 g cm⁻³", min=0, max=100, value=25, step=1),
                                    ui.input_slider("polymer_PET", "PET (%) ρₚ = 0.96–1.45 g cm⁻³", min=0, max=100, value=17, step=1),
                                    ui.input_slider("polymer_PA", "PA (%) ρₚ = 1.02–1.16 g cm⁻³", min=0, max=100, value=12, step=1),
                                    ui.input_slider("polymer_PP", "PP (%) ρₚ = 0.83–0.92 g cm⁻³", min=0, max=100, value=14, step=1),
                                    ui.input_slider("polymer_PS", "PS (%) ρₚ = 1.04–1.10 g cm⁻³", min=0, max=100, value=9, step=1),
                                    ui.input_slider("polymer_PVA", "PVA (%) ρₚ = 1.19–1.31 g cm⁻³", min=0, max=100, value=6, step=1),
                                    ui.input_slider("polymer_PVC", "PVC (%) ρₚ = 1.10–1.58 g cm⁻³", min=0, max=100, value=17, step=1),
                                    class_="collapsible-control-body",
                                ),
                                open=False,
                                class_="collapsible-control nested-control",
                            ),
                            class_="collapsible-control-body",
                        ),
                        open=True,
                        class_="collapsible-control",
                    ),

                    ui.tags.details(
                        ui.tags.summary("Macroplastics"),
                        ui.div(
                            ui.div(
                                "Use grouped macroplastic classes by default, or switch on item-level selection from the Common name column.",
                                class_="helper-text",
                            ),
                            ui.input_checkbox(
                                "use_macro_items",
                                "Select individual litter items",
                                False,
                            ),
                            ui.panel_conditional(
                                "!input.use_macro_items",
                                ui.input_checkbox_group(
                                    "macro_categories",
                                    "Grouped macroplastic classes",
                                    choices={
                                        "buoyant": ui.HTML("Buoyant (Foams)<br><small>ρₚ ∈ [0.02, 0.08] g cm⁻³</small>"),
                                        "near_neutral": ui.HTML("Near neutral (Plastics & others)<br><small>ρₚ ∈ [0.8, 1.5] g cm⁻³</small>"),
                                        "dense": ui.HTML("Dense (Glass & metal)<br><small>ρₚ ∈ [2.5, 4.3] g cm⁻³</small>"),
                                    },
                                    selected=[],
                                ),
                            ),
                            ui.panel_conditional(
                                "input.use_macro_items",
                                ui.input_selectize(
                                    "macro_common_names",
                                    "Individual litter items",
                                    choices=macro_common_names,
                                    selected=[],
                                    multiple=True,
                                    options={"placeholder": "Search or scroll through litter items", "plugins": ["remove_button"]},
                                ),
                                ui.div("No individual items are selected by default.", class_="helper-text"),
                            ),
                            class_="collapsible-control-body",
                        ),
                        open=False,
                        class_="collapsible-control",
                    ),
                    class_="right-control-panel",
                ),
                class_="analysis-layout",
            ),
        ),
    ),

    ui.nav_panel(
        "Settling and rising velocities",
        ui.page_sidebar(
            ui.sidebar(
                ui.h3("Microplastic controls"),
                ui.tags.details(
                    ui.tags.summary("Microplastics"),
                    ui.div(
                        ui.tags.details(
                            ui.tags.summary("Size"),
                            ui.div(
                                ui.input_slider(
                                    "vel_size_range",
                                    "Particle size limits (µm)",
                                    min=20,
                                    max=5000,
                                    value=(20, 5000),
                                    step=10,
                                ),
                                ui.tags.details(
                                    ui.tags.summary("Advanced size controls"),
                                    ui.div(
                                        ui.input_select(
                                            "vel_size_distribution",
                                            "Size distribution",
                                            choices={
                                                "loguniform": "Log-uniform",
                                                "uniform": "Uniform",
                                                "lognormal": "Truncated lognormal",
                                            },
                                            selected="loguniform",
                                        ),
                                        class_="collapsible-control-body",
                                    ),
                                    open=False,
                                    class_="collapsible-control nested-control",
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
                                    "vel_fiber_percent",
                                    "Fibres (%)",
                                    min=0,
                                    max=100,
                                    value=50,
                                    step=1,
                                ),
                                ui.input_slider(
                                    "vel_fragment_percent",
                                    "Fragments (%)",
                                    min=0,
                                    max=100,
                                    value=50,
                                    step=1,
                                ),
                                ui.output_text("vel_shape_total_text"),
                                class_="collapsible-control-body",
                            ),
                            open=False,
                            class_="collapsible-control nested-control",
                        ),

                        ui.tags.details(
                            ui.tags.summary("Polymer"),
                            ui.div(
                                ui.input_action_button(
                                    "vel_reset_polymer_mix",
                                    "Reset to default %",
                                    class_="btn-sm btn-outline-secondary",
                                ),
                                ui.div(
                                    ui.output_text("vel_polymer_total_text"),
                                    class_="input-warning",
                                ),
                                ui.input_slider("vel_polymer_PE", "PE (%) ρₚ = 0.89–0.98 g cm⁻³", min=0, max=100, value=25, step=1),
                                ui.input_slider("vel_polymer_PET", "PET (%) ρₚ = 0.96–1.45 g cm⁻³", min=0, max=100, value=17, step=1),
                                ui.input_slider("vel_polymer_PA", "PA (%) ρₚ = 1.02–1.16 g cm⁻³", min=0, max=100, value=12, step=1),
                                ui.input_slider("vel_polymer_PP", "PP (%) ρₚ = 0.83–0.92 g cm⁻³", min=0, max=100, value=14, step=1),
                                ui.input_slider("vel_polymer_PS", "PS (%) ρₚ = 1.04–1.10 g cm⁻³", min=0, max=100, value=9, step=1),
                                ui.input_slider("vel_polymer_PVA", "PVA (%) ρₚ = 1.19–1.31 g cm⁻³", min=0, max=100, value=6, step=1),
                                ui.input_slider("vel_polymer_PVC", "PVC (%) ρₚ = 1.10–1.58 g cm⁻³", min=0, max=100, value=17, step=1),
                                class_="collapsible-control-body",
                            ),
                            open=False,
                            class_="collapsible-control nested-control",
                        ),
                        class_="collapsible-control-body",
                    ),
                    open=True,
                    class_="collapsible-control",
                ),
                width="360px",
            ),
            ui.h2("Settling and rising velocities"),
            ui.card(
                ui.card_header("Generated settling and rising velocity distributions"),
                ui.output_plot("velocity_distribution_plot", height="420px"),
                full_screen=True,
                class_="plot-card",
            ),
            ui.div(
                ui.card(
                    ui.card_header("Size distribution"),
                    ui.output_plot("vel_size_pdf_plot", height="95px"),
                    class_="mini-diagnostic-card",
                ),
                ui.card(
                    ui.card_header("Shape mix"),
                    ui.output_plot("vel_shape_mix_plot", height="95px"),
                    class_="mini-diagnostic-card",
                ),
                ui.card(
                    ui.card_header("Polymer mix"),
                    ui.output_plot("vel_polymer_mix_plot", height="95px"),
                    class_="mini-diagnostic-card",
                ),
                class_="diagnostic-grid",
            ),
            ui.card(
                ui.card_header("Synthetic particle summary"),
                ui.output_data_frame("vel_synthetic_micro_summary"),
                ui.div(
                    ui.download_button(
                        "download_velocity_synthetic_csv",
                        "Download synthetic particles CSV",
                        class_="btn-sm btn-outline-primary",
                    ),
                    ui.download_button(
                        "download_velocity_summary_csv",
                        "Download velocity summary CSV",
                        class_="btn-sm btn-outline-secondary",
                    ),
                    style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.5rem;",
                ),
                full_screen=True,
            ),
        ),
    ),

    ui.nav_panel(
        "Sampling correction",
        ui.page_sidebar(
            ui.sidebar(
                ui.h3("Sampling setup"),

                ui.div(
                    "Set hydraulics → choose sampling depth → enter concentration/discharge → read correction.",
                    class_="sampling-workflow-note",
                ),

                ui.navset_card_tab(
                    ui.nav_panel(
                        "Hydraulics",
                        ui.input_radio_buttons(
                            "samp_ustar_mode",
                            "Set shear velocity",
                            choices={
                                "direct": "Direct u*",
                                "hydraulic": "Calculate u* from R and S",
                            },
                            selected="direct",
                        ),
                        ui.output_ui("samp_ustar_controls"),
                    ),
                    ui.nav_panel(
                        "Sampling depth",
                        ui.input_checkbox(
                            "samp_net_sampling_enabled",
                            "Show captured/missed estimate",
                            True,
                        ),
                        ui.input_slider(
                            "samp_net_z_interval",
                            "Sampling depth interval, z/H",
                            min=0.0,
                            max=1.0,
                            value=(0.80, 1.00),
                            step=0.01,
                        ),
                        ui.div(
                            "0 = bed, 1 = surface. The selected sampling depth band is drawn on the plot.",
                            class_="compact-note",
                        ),
                    ),
                    ui.nav_panel(
                        "Concentration",
                        ui.input_checkbox(
                            "samp_sampling_correction_enabled",
                            "Show concentration correction",
                            False,
                        ),
                        ui.input_numeric(
                            "samp_measured_concentration",
                            "Measured concentration in sample",
                            value=10.0,
                            min=0.0,
                            step=0.1,
                        ),
                        ui.input_select(
                            "samp_concentration_units",
                            "Concentration units",
                            choices={
                                "particles/m3": "particles/m³",
                                "items/m3": "items/m³",
                                "mg/m3": "mg/m³",
                                "g/m3": "g/m³",
                            },
                            selected="particles/m3",
                        ),
                        ui.input_checkbox(
                            "samp_include_discharge",
                            "Include river discharge Q",
                            False,
                        ),
                        ui.panel_conditional(
                            "input.samp_include_discharge",
                            ui.input_numeric(
                                "samp_discharge",
                                "Discharge Q (m³/s)",
                                value=1.0,
                                min=0.0,
                                step=0.1,
                            ),
                        ),
                        ui.div(
                            "Concentration correction and load equations are explained in About & Methods.",
                            class_="compact-note",
                        ),
                    ),
                    ui.nav_panel(
                        "Display",
                        ui.input_slider(
                            "samp_a_bed_frac",
                            "Bed reference height a_bed/H",
                            min=0.01,
                            max=0.30,
                            value=0.05,
                            step=0.01,
                        ),
                        ui.input_slider(
                            "samp_a_surf_frac",
                            "Surface reference offset a_surf/H",
                            min=0.01,
                            max=0.30,
                            value=0.01,
                            step=0.01,
                        ),
                        ui.input_slider(
                            "samp_iqr_percentiles",
                            "Displayed percentile range",
                            min=0,
                            max=100,
                            value=(25, 75),
                            step=1,
                        ),
                    ),
                    id="samp_left_tabs",
                ),
                width="360px",
            ),

            ui.div(
                ui.div(
                    ui.h2("Sampling bias and correction"),
                    ui.output_ui("sampling_caution"),
                    ui.card(
                        ui.card_header("Vertical concentration profiles"),
                        ui.output_plot("profile_plot_sampling", height="480px"),
                        full_screen=True,
                        class_="plot-card square-plot-card",
                    ),
                    ui.div(
                        ui.card(
                            ui.card_header("Size distribution"),
                            ui.output_plot("samp_size_pdf_plot", height="140px"),
                            class_="mini-diagnostic-card",
                        ),
                        ui.card(
                            ui.card_header("Shape mix"),
                            ui.output_plot("samp_shape_mix_plot", height="140px"),
                            class_="mini-diagnostic-card",
                        ),
                        ui.card(
                            ui.card_header("Polymer mix"),
                            ui.output_plot("samp_polymer_mix_plot", height="140px"),
                            class_="mini-diagnostic-card",
                        ),
                        class_="diagnostic-grid",
                    ),
                    ui.div(
                        ui.download_button(
                            "download_sampling_synthetic_csv",
                            "Download sampling particles CSV",
                            class_="btn-sm btn-outline-primary",
                        ),
                        style="text-align:center; margin-top:0.5rem;",
                    ),
                    ui.div(
                        ui.card(
                            ui.card_header("1. Sampling depth estimate — median [P25–P75]"),
                            ui.output_data_frame("net_sampling_results"),
                            ui.download_button("download_net_sampling_results_csv", "Download CSV", class_="btn-sm btn-outline-secondary"),
                            class_="smart-table-card",
                        ),
                        ui.card(
                            ui.card_header("2. Depth-averaged concentration — median [P25–P75]"),
                            ui.output_data_frame("sampling_correction_results"),
                            ui.download_button("download_depth_average_results_csv", "Download CSV", class_="btn-sm btn-outline-secondary"),
                            class_="smart-table-card",
                        ),
                        ui.card(
                            ui.card_header("3. Estimated load — median [P25–P75]"),
                            ui.output_data_frame("discharge_load_results"),
                            ui.download_button("download_load_results_csv", "Download CSV", class_="btn-sm btn-outline-secondary"),
                            class_="smart-table-card",
                        ),
                        class_="smart-table-grid",
                    ),
                    class_="centre-analysis-panel",
                ),
                sampling_plastic_controls_ui(),
                class_="analysis-layout",
            ),
        ),
    ),

    ui.nav_panel(
        "About & Methods",
        ui.layout_columns(
            ui.card(
                ui.card_header("Methods, equations, and code notes"),
                ui.markdown(methods_text),
                full_screen=True,
            ),
            col_widths=[12],
        ),
    ),

    title="River Plastic Vertical Profiler",
)

# ============================================================
# SHINY SERVER
# ============================================================
def server(input: Inputs, output: Outputs, session: Session):

    polymer_ids = [
        "polymer_PE",
        "polymer_PET",
        "polymer_PA",
        "polymer_PP",
        "polymer_PS",
        "polymer_PVA",
        "polymer_PVC",
    ]

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

    default_polymer_mix = {
        "PE": 25,
        "PET": 17,
        "PA": 12,
        "PP": 14,
        "PS": 9,
        "PVA": 6,
        "PVC": 17,
    }

    explorer_polymer_input_map = {
        "PE": "polymer_PE",
        "PET": "polymer_PET",
        "PA": "polymer_PA",
        "PP": "polymer_PP",
        "PS": "polymer_PS",
        "PVA": "polymer_PVA",
        "PVC": "polymer_PVC",
    }

    velocity_polymer_input_map = {
        "PE": "vel_polymer_PE",
        "PET": "vel_polymer_PET",
        "PA": "vel_polymer_PA",
        "PP": "vel_polymer_PP",
        "PS": "vel_polymer_PS",
        "PVA": "vel_polymer_PVA",
        "PVC": "vel_polymer_PVC",
    }

    sampling_polymer_input_map = {
        "PE": "samp_polymer_PE",
        "PET": "samp_polymer_PET",
        "PA": "samp_polymer_PA",
        "PP": "samp_polymer_PP",
        "PS": "samp_polymer_PS",
        "PVA": "samp_polymer_PVA",
        "PVC": "samp_polymer_PVC",
    }

    explorer_polymer_last_values = reactive.Value(default_polymer_mix.copy())
    velocity_polymer_last_values = reactive.Value(default_polymer_mix.copy())
    sampling_polymer_last_values = reactive.Value(default_polymer_mix.copy())

    explorer_polymer_update_guard = reactive.Value(False)
    velocity_polymer_update_guard = reactive.Value(False)
    sampling_polymer_update_guard = reactive.Value(False)

    @reactive.Effect
    @reactive.event(
        input.polymer_PE,
        input.polymer_PET,
        input.polymer_PA,
        input.polymer_PP,
        input.polymer_PS,
        input.polymer_PVA,
        input.polymer_PVC,
        ignore_init=True,
    )
    def _sync_explorer_polymer_sliders():
        _sync_polymer_group(
            explorer_polymer_input_map,
            explorer_polymer_last_values,
            explorer_polymer_update_guard,
        )

    @reactive.Effect
    @reactive.event(
        input.vel_polymer_PE,
        input.vel_polymer_PET,
        input.vel_polymer_PA,
        input.vel_polymer_PP,
        input.vel_polymer_PS,
        input.vel_polymer_PVA,
        input.vel_polymer_PVC,
        ignore_init=True,
    )
    def _sync_velocity_polymer_sliders():
        _sync_polymer_group(
            velocity_polymer_input_map,
            velocity_polymer_last_values,
            velocity_polymer_update_guard,
        )

    @reactive.Effect
    @reactive.event(
        input.samp_polymer_PE,
        input.samp_polymer_PET,
        input.samp_polymer_PA,
        input.samp_polymer_PP,
        input.samp_polymer_PS,
        input.samp_polymer_PVA,
        input.samp_polymer_PVC,
        ignore_init=True,
    )
    def _sync_sampling_polymer_sliders():
        _sync_polymer_group(
            sampling_polymer_input_map,
            sampling_polymer_last_values,
            sampling_polymer_update_guard,
        )

    @reactive.Effect
    @reactive.event(input.reset_polymer_mix)
    def _reset_polymer_mix():
        """Reset polymer sliders to the default synthetic mixture."""
        ui.update_slider("polymer_PE", value=25)
        ui.update_slider("polymer_PET", value=17)
        ui.update_slider("polymer_PA", value=12)
        ui.update_slider("polymer_PP", value=14)
        ui.update_slider("polymer_PS", value=9)
        ui.update_slider("polymer_PVA", value=6)
        ui.update_slider("polymer_PVC", value=17)
        explorer_polymer_last_values.set(default_polymer_mix.copy())


    @reactive.Effect
    @reactive.event(input.vel_reset_polymer_mix)
    def _vel_reset_polymer_mix():
        """Reset settling-tab polymer sliders to the default synthetic mixture."""
        ui.update_slider("vel_polymer_PE", value=25)
        ui.update_slider("vel_polymer_PET", value=17)
        ui.update_slider("vel_polymer_PA", value=12)
        ui.update_slider("vel_polymer_PP", value=14)
        ui.update_slider("vel_polymer_PS", value=9)
        ui.update_slider("vel_polymer_PVA", value=6)
        ui.update_slider("vel_polymer_PVC", value=17)
        velocity_polymer_last_values.set(default_polymer_mix.copy())

    def selected_polymer_raw_percentages() -> dict[str, float]:
        return {
            "PE": float(input.polymer_PE()),
            "PET": float(input.polymer_PET()),
            "PA": float(input.polymer_PA()),
            "PP": float(input.polymer_PP()),
            "PS": float(input.polymer_PS()),
            "PVA": float(input.polymer_PVA()),
            "PVC": float(input.polymer_PVC()),
        }

    def selected_polymer_percentages() -> dict[str, float]:
        """Return polymer percentages only when the slider total is 100%.

        The app no longer silently normalises polymer sliders. If the total is
        not 100%, synthetic particle generation is blocked until the user fixes
        the mixture.
        """
        raw = selected_polymer_raw_percentages()
        total = float(sum(raw.values()))

        if not polymer_total_valid(total):
            return {name: 0.0 for name in raw}

        return raw

    def selected_polymer_total() -> float:
        """Return the raw polymer slider total, not the normalised total."""
        return float(sum(selected_polymer_raw_percentages().values()))

    def selected_shape_percentages() -> tuple[float, float]:
        """Return shape percentages. UI effects keep fibres + fragments = 100."""
        fibre = float(input.synthetic_fiber_percent())
        fragment = float(input.synthetic_fragment_percent())
        total = fibre + fragment
        if total <= 0:
            return 50.0, 50.0
        if abs(total - 100.0) > 1e-6:
            return 100.0 * fibre / total, 100.0 * fragment / total
        return fibre, fragment

    @render.text
    def polymer_total_text():
        total = selected_polymer_total()
        if not polymer_total_valid(total):
            return f"Polymer total is {total:.0f}%; adjusting linked sliders to 100%."
        return "Polymer total is 100%. Moving one slider rescales the others."


    @render.text
    def shape_total_text():
        fibre_pct, fragment_pct = selected_shape_percentages()
        return f"Total: {fibre_pct + fragment_pct:.0f}% | Fibres {fibre_pct:.0f}%, fragments {fragment_pct:.0f}%"

    @render.ui
    def app_warnings():
        messages = []
        if len(selected_micro_ranges()) == 0 and len(selected_macro_categories()) == 0 and len(selected_macro_items()) == 0:
            messages.append("No plastic groups are selected, so the plot will be empty.")
        polymer_total = selected_polymer_total()
        if polymer_total <= 0:
            messages.append("Synthetic microplastics are selected, but the polymer mix is zero.")
        elif not polymer_total_valid(polymer_total):
            messages.append(f"Polymer sliders total {polymer_total:.0f}%; linked sliders are adjusting this to 100%.")

        if not messages:
            return ui.div()

        return ui.div(
            ui.tags.strong("Check inputs"),
            ui.tags.ul(*[ui.tags.li(msg) for msg in messages]),
            class_="warning-box",
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
        if not polymer_total_valid(selected_samp_polymer_total()):
            messages.append(f"Sampling-page polymer sliders total {selected_samp_polymer_total():.0f}%; linked sliders are adjusting this to 100%.")

        if not messages:
            return ui.div(
                "Sampling outputs are conditional on the selected Rouse profiles and input parameters.",
                class_="sampling-note",
            )

        return ui.div(
            ui.tags.strong("Interpret with caution"),
            ui.tags.ul(*[ui.tags.li(msg) for msg in messages]),
            class_="warning-box",
        )

    def selected_u_star() -> float:
        if input.ustar_mode() == "direct":
            return float(input.u_star())

        return calculate_shear_velocity_from_slope_radius(
            hydraulic_radius=float(input.hydraulic_radius()),
            slope=float(input.slope()),
        )

    def selected_flow_depth() -> float:
        """Return the flow depth used in Rouse-profile calculations.

        Explorer no longer exposes a separate H slider; use a fixed default
        depth for the profile unless a future hydraulic mode supplies H.
        """
        return 0.50

    def selected_micro_ranges() -> list[tuple[str, float, float]]:
        size_min_um, size_max_um = input.synthetic_size_range()
        size_min_um = float(size_min_um)
        size_max_um = float(size_max_um)
        if size_max_um <= size_min_um:
            return []
        return [("synthetic MP", size_min_um, size_max_um)]

    def selected_micro_df() -> pd.DataFrame:
        """Return the generated synthetic microplastic data for the current inputs."""
        if not polymer_total_valid(selected_polymer_total()):
            return empty_synthetic_microplastics_df()

        polymer_percentages = selected_polymer_percentages()
        size_min_um, size_max_um = input.synthetic_size_range()

        return generate_synthetic_microplastics(
            n_particles=20000,
            size_ranges_um=[(float(size_min_um), float(size_max_um))],
            polymer_percentages=polymer_percentages,
            fiber_percent=selected_shape_percentages()[0],
            seed=42,
            size_distribution=str(input.synthetic_size_distribution()),
        )



    def selected_vel_polymer_raw_percentages() -> dict[str, float]:
        return {
            "PE": float(input.vel_polymer_PE()),
            "PET": float(input.vel_polymer_PET()),
            "PA": float(input.vel_polymer_PA()),
            "PP": float(input.vel_polymer_PP()),
            "PS": float(input.vel_polymer_PS()),
            "PVA": float(input.vel_polymer_PVA()),
            "PVC": float(input.vel_polymer_PVC()),
        }

    def selected_vel_polymer_total() -> float:
        return float(sum(selected_vel_polymer_raw_percentages().values()))

    def selected_vel_polymer_percentages() -> dict[str, float]:
        raw = selected_vel_polymer_raw_percentages()
        total = float(sum(raw.values()))
        if not polymer_total_valid(total):
            return {name: 0.0 for name in raw}
        return raw

    def selected_vel_shape_percentages() -> tuple[float, float]:
        fibre = float(input.vel_fiber_percent())
        fragment = float(input.vel_fragment_percent())
        total = fibre + fragment
        if total <= 0:
            return 50.0, 50.0
        return 100.0 * fibre / total, 100.0 * fragment / total

    def selected_vel_micro_df() -> pd.DataFrame:
        if not polymer_total_valid(selected_vel_polymer_total()):
            return empty_synthetic_microplastics_df()

        size_min_um, size_max_um = input.vel_size_range()
        return generate_synthetic_microplastics(
            n_particles=20000,
            size_ranges_um=[(float(size_min_um), float(size_max_um))],
            polymer_percentages=selected_vel_polymer_percentages(),
            fiber_percent=selected_vel_shape_percentages()[0],
            seed=42,
            size_distribution=str(input.vel_size_distribution()),
        )

    def selected_macro_categories() -> list[str]:
        return list(input.macro_categories() or [])

    def use_macro_items() -> bool:
        return bool(input.use_macro_items())

    def selected_macro_items() -> list[str]:
        return list(input.macro_common_names() or [])

    def net_sampling_enabled() -> bool:
        # Explorer plot does not show the sampling interval.
        # The Sampling correction tab uses sampling-specific controls.
        return False

    def sampling_correction_enabled() -> bool:
        return False

    def selected_iqr_percentiles() -> tuple[float, float]:
        q_low, q_high = input.iqr_percentiles()
        return float(q_low), float(q_high)

    def selected_net_interval() -> tuple[float, float]:
        z_min, z_max = input.net_z_interval()
        return float(z_min), float(z_max)


    @reactive.Effect
    @reactive.event(input.samp_reset_polymer_mix)
    def _samp_reset_polymer_mix():
        """Reset sampling-tab polymer sliders to the default synthetic mixture."""
        ui.update_slider("samp_polymer_PE", value=25)
        ui.update_slider("samp_polymer_PET", value=17)
        ui.update_slider("samp_polymer_PA", value=12)
        ui.update_slider("samp_polymer_PP", value=14)
        ui.update_slider("samp_polymer_PS", value=9)
        ui.update_slider("samp_polymer_PVA", value=6)
        ui.update_slider("samp_polymer_PVC", value=17)
        sampling_polymer_last_values.set(default_polymer_mix.copy())

    def selected_samp_polymer_raw_percentages() -> dict[str, float]:
        return {
            "PE": float(input.samp_polymer_PE()),
            "PET": float(input.samp_polymer_PET()),
            "PA": float(input.samp_polymer_PA()),
            "PP": float(input.samp_polymer_PP()),
            "PS": float(input.samp_polymer_PS()),
            "PVA": float(input.samp_polymer_PVA()),
            "PVC": float(input.samp_polymer_PVC()),
        }

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
        size_min_um, size_max_um = input.samp_synthetic_size_range()
        size_min_um = float(size_min_um)
        size_max_um = float(size_max_um)
        if size_max_um <= size_min_um:
            return []
        return [("synthetic MP", size_min_um, size_max_um)]

    def selected_samp_micro_df() -> pd.DataFrame:
        if not polymer_total_valid(selected_samp_polymer_total()):
            return empty_synthetic_microplastics_df()

        size_min_um, size_max_um = input.samp_synthetic_size_range()
        return generate_synthetic_microplastics(
            n_particles=20000,
            size_ranges_um=[(float(size_min_um), float(size_max_um))],
            polymer_percentages=selected_samp_polymer_percentages(),
            fiber_percent=selected_samp_shape_percentages()[0],
            seed=42,
            size_distribution=str(input.samp_synthetic_size_distribution()),
        )

    def selected_samp_macro_categories() -> list[str]:
        return list(input.samp_macro_categories() or [])

    def use_samp_macro_items() -> bool:
        return bool(input.samp_use_macro_items())

    def selected_samp_macro_items() -> list[str]:
        return list(input.samp_macro_common_names() or [])

    def selected_samp_u_star() -> float:
        if input.samp_ustar_mode() == "direct":
            return float(input.samp_u_star())
        return calculate_shear_velocity_from_slope_radius(
            hydraulic_radius=float(input.samp_hydraulic_radius()),
            slope=float(input.samp_slope()),
        )

    def selected_samp_iqr_percentiles() -> tuple[float, float]:
        q_low, q_high = input.samp_iqr_percentiles()
        return float(q_low), float(q_high)

    def selected_samp_net_interval() -> tuple[float, float]:
        z_min, z_max = input.samp_net_z_interval()
        return float(z_min), float(z_max)

    def build_velocity_summary_df() -> pd.DataFrame:
        """Return a compact CSV-ready summary of the velocity-tab synthetic data."""
        df = selected_vel_micro_df()

        if df.empty:
            return pd.DataFrame({"Metric": ["No synthetic data"], "Value": [""]})

        rows = [
            {"Metric": "Particles generated", "Value": f"{len(df):,}"},
            {"Metric": "Size range", "Value": f"{df['size_um'].min():.3g}–{df['size_um'].max():.3g} µm"},
            {"Metric": "Density range", "Value": f"{df['density_g_cm3'].min():.3g}–{df['density_g_cm3'].max():.3g} g cm⁻³"},
        ]

        for col, label in [
            ("velocity_dietrich", "Dietrich vertical velocity"),
            ("velocity_goral", "Goral vertical velocity"),
            ("velocity_yu", "Yu vertical velocity"),
        ]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
                if len(vals) > 0:
                    rows.extend(
                        [
                            {"Metric": f"{label} median", "Value": f"{np.nanmedian(vals):.6g} m/s"},
                            {"Metric": f"{label} P25", "Value": f"{np.nanpercentile(vals, 25):.6g} m/s"},
                            {"Metric": f"{label} P75", "Value": f"{np.nanpercentile(vals, 75):.6g} m/s"},
                        ]
                    )

        return pd.DataFrame(rows)

    def build_net_sampling_results_df() -> pd.DataFrame:
        """Return the sampling depth estimate table as a plain DataFrame."""
        if not samp_net_sampling_enabled():
            return pd.DataFrame({"Output": ["Turn on 'Show captured/missed estimate' to calculate capture fractions."]})

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
        )
        df = df.rename(
            columns={
                "Sampled z/H interval": "Sampling depth interval (z/H)",
                "Water-column fraction sampled": "Fraction of water column sampled",
                "Capture fraction": "Captured fraction",
                "Missed fraction": "Missed fraction",
                "Captured (%)": "Captured percentage",
                "Missed (%)": "Missed percentage",
            }
        )
        keep_cols = [
            c for c in [
                "Group",
                "Sampling depth interval (z/H)",
                "Fraction of water column sampled",
                "Captured fraction",
                "Missed fraction",
                "Captured percentage",
                "Missed percentage",
            ] if c in df.columns
        ]
        return df.loc[:, keep_cols]

    def build_depth_average_results_df() -> pd.DataFrame:
        """Return the depth-averaged concentration table as a plain DataFrame."""
        if not samp_sampling_correction_enabled():
            return pd.DataFrame({"Output": ["Turn on 'Show concentration correction' to calculate depth-averaged concentration."]})

        df = sampling_correction_table(
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
            concentration_units=str(input.samp_concentration_units()),
            include_discharge=bool(input.samp_include_discharge()),
            discharge=float(input.samp_discharge()) if bool(input.samp_include_discharge()) else np.nan,
            iqr_lower=selected_samp_iqr_percentiles()[0],
            iqr_upper=selected_samp_iqr_percentiles()[1],
        )
        df = df.rename(
            columns={
                "Sampled z/H interval": "Sampling depth interval (z/H)",
                "Capture fraction": "Captured fraction",
                "Correction factor": "Correction factor",
                "Estimated depth-averaged concentration": "Estimated depth-averaged concentration",
            }
        )
        keep_cols = [
            c for c in [
                "Group",
                "Sampling depth interval (z/H)",
                "Measured concentration",
                "Units",
                "Captured fraction",
                "Correction factor",
                "Estimated depth-averaged concentration",
            ] if c in df.columns
        ]
        return df.loc[:, keep_cols]

    def build_load_results_df() -> pd.DataFrame:
        """Return the estimated-load table as a plain DataFrame."""
        if not bool(input.samp_include_discharge()):
            return pd.DataFrame({"Output": ["Turn on 'Include river discharge Q' to calculate estimated load."]})
        if not samp_sampling_correction_enabled():
            return pd.DataFrame({"Output": ["Turn on 'Show concentration correction' first, then include discharge to calculate estimated load."]})

        df = sampling_correction_table(
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
            concentration_units=str(input.samp_concentration_units()),
            include_discharge=True,
            discharge=float(input.samp_discharge()),
            iqr_lower=selected_samp_iqr_percentiles()[0],
            iqr_upper=selected_samp_iqr_percentiles()[1],
        )
        df = df.rename(
            columns={
                "Sampled z/H interval": "Sampling depth interval (z/H)",
                "Discharge Q (m3/s)": "Discharge (m³/s)",
                "Estimated load": "Estimated load",
                "Load units": "Load units",
            }
        )
        keep_cols = [
            c for c in [
                "Group",
                "Sampling depth interval (z/H)",
                "Discharge (m³/s)",
                "Estimated load",
                "Load units",
            ] if c in df.columns
        ]
        return df.loc[:, keep_cols]

    def samp_net_sampling_enabled() -> bool:
        return bool(input.samp_net_sampling_enabled())

    def samp_sampling_correction_enabled() -> bool:
        return bool(input.samp_sampling_correction_enabled())

    last_shape_values = reactive.Value({
        "synthetic_fiber_percent": 50,
        "synthetic_fragment_percent": 50,
    })

    @reactive.Effect
    @reactive.event(input.synthetic_fiber_percent, input.synthetic_fragment_percent, ignore_init=True)
    def _sync_shape_sliders():
        current = {
            "synthetic_fiber_percent": int(round(float(input.synthetic_fiber_percent()))),
            "synthetic_fragment_percent": int(round(float(input.synthetic_fragment_percent()))),
        }
        last = last_shape_values.get()
        changed = [name for name in current if current[name] != last.get(name)]
        if not changed:
            return

        changed_name = changed[0]
        changed_value = max(0, min(100, current[changed_name]))
        if changed_name == "synthetic_fiber_percent":
            updated = {
                "synthetic_fiber_percent": changed_value,
                "synthetic_fragment_percent": 100 - changed_value,
            }
        else:
            updated = {
                "synthetic_fragment_percent": changed_value,
                "synthetic_fiber_percent": 100 - changed_value,
            }

        last_shape_values.set(updated)
        ui.update_slider("synthetic_fiber_percent", value=updated["synthetic_fiber_percent"])
        ui.update_slider("synthetic_fragment_percent", value=updated["synthetic_fragment_percent"])


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

    @render.ui
    def samp_ustar_controls():
        if input.samp_ustar_mode() == "direct":
            return ui.TagList(
                ui.input_slider(
                    "samp_u_star",
                    "Shear velocity u* (m/s)",
                    min=0.01,
                    max=0.50,
                    value=0.15,
                    step=0.001,
                )
            )

        return ui.TagList(
            ui.input_slider(
                "samp_hydraulic_radius",
                "Hydraulic radius R, or depth H for wide channels (m)",
                min=0.01,
                max=5.00,
                value=0.50,
                step=0.01,
            ),
            ui.input_slider(
                "samp_slope",
                "Energy slope S (-)",
                min=0.00001,
                max=0.02000,
                value=0.00100,
                step=0.00001,
            ),
            ui.output_text("samp_calculated_ustar"),
        )

    @render.text
    def samp_calculated_ustar():
        u_star = calculate_shear_velocity_from_slope_radius(
            hydraulic_radius=float(input.samp_hydraulic_radius()),
            slope=float(input.samp_slope()),
        )
        return f"Calculated u*: {u_star:.4f} m/s"

    @render.ui
    def ustar_controls():
        if input.ustar_mode() == "direct":
            return ui.TagList(
                ui.input_slider(
                    "u_star",
                    "Shear velocity u* (m/s)",
                    min=0.01,
                    max=0.50,
                    value=0.15,
                    step=0.001,
                )
            )

        return ui.TagList(
            ui.input_slider(
                "hydraulic_radius",
                "Hydraulic radius R, or depth H for wide channels (m)",
                min=0.01,
                max=5.00,
                value=0.50,
                step=0.01,
            ),
            ui.input_slider(
                "slope",
                "Energy slope S (-)",
                min=0.00001,
                max=0.02000,
                value=0.00100,
                step=0.00001,
            ),
            ui.output_text("calculated_ustar"),
        )

    @render.text
    def calculated_ustar():
        u_star = calculate_shear_velocity_from_slope_radius(
            hydraulic_radius=float(input.hydraulic_radius()),
            slope=float(input.slope()),
        )
        return f"Calculated u*: {u_star:.4f} m/s"

    def make_current_profile_plot():
        return make_profile_plot(
            micro_ranges=selected_micro_ranges(),
            macro_selected=selected_macro_categories(),
            macro_items_selected=selected_macro_items(),
            use_macro_items=use_macro_items(),
            u_star=selected_u_star(),
            micro_df=selected_micro_df(),
            H=selected_flow_depth(),
            a_bed_frac=float(input.a_bed_frac()),
            a_surf_frac=float(input.a_surf_frac()),
            iqr_lower=selected_iqr_percentiles()[0],
            iqr_upper=selected_iqr_percentiles()[1],
            show_net_interval=False,
            net_z_interval=None,
        )

    @render.plot(alt="Vertical Rouse concentration profile plot")
    def profile_plot_basic():
        return make_current_profile_plot()

    @render.plot(alt="Vertical Rouse concentration profile plot")
    def profile_plot_advanced():
        return make_current_profile_plot()

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
            show_net_interval=samp_net_sampling_enabled(),
            net_z_interval=selected_samp_net_interval(),
        )

    @render.plot(alt="Synthetic microplastic size probability density plot")
    def size_pdf_plot():
        df = selected_micro_df()
        fig, ax = plt.subplots(figsize=(2.6, 1.7))
        size_um = pd.to_numeric(df.get("size_um"), errors="coerce").dropna().to_numpy(dtype=float)
        if len(size_um) > 0:
            ax.hist(size_um, bins=35, density=True, alpha=0.8)
            ax.set_yscale("log")
        ax.set_xlabel("Size (µm)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.18)
        fig.tight_layout(pad=0.6)
        return fig

    @render.plot(alt="Synthetic microplastic shape mixture pie chart")
    def shape_mix_plot():
        fibre_pct, fragment_pct = selected_shape_percentages()
        fig, ax = plt.subplots(figsize=(2.4, 1.7))
        values = [fibre_pct, fragment_pct]
        labels = ["Fibres", "Fragments"]
        if sum(values) <= 0:
            ax.text(0.5, 0.5, "No shape\nselected", ha="center", va="center", fontsize=8)
            ax.axis("off")
        else:
            ax.pie(values, labels=labels, autopct="%.0f%%", textprops={"fontsize": 7})
        fig.tight_layout(pad=0.5)
        return fig

    @render.plot(alt="Synthetic microplastic polymer mixture pie chart")
    def polymer_mix_plot():
        fig, ax = plt.subplots(figsize=(2.4, 1.7))
        if not polymer_total_valid(selected_polymer_total()):
            ax.text(0.5, 0.5, "Polymer total\nmust be 100%", ha="center", va="center", fontsize=8)
            ax.axis("off")
            fig.tight_layout(pad=0.5)
            return fig

        polymer_percentages = selected_polymer_percentages()
        labels = []
        values = []
        for name, value in polymer_percentages.items():
            if value > 0:
                labels.append(name)
                values.append(value)
        if sum(values) <= 0:
            ax.text(0.5, 0.5, "No polymer\nselected", ha="center", va="center", fontsize=8)
            ax.axis("off")
        else:
            ax.pie(values, labels=labels, autopct="%.0f%%", textprops={"fontsize": 6})
        fig.tight_layout(pad=0.5)
        return fig



    @render.text
    def vel_polymer_total_text():
        total = selected_vel_polymer_total()
        if not polymer_total_valid(total):
            return f"Polymer total is {total:.0f}%; adjusting linked sliders to 100%."
        return "Polymer total is 100%. Moving one slider rescales the others."

    @render.text
    def vel_shape_total_text():
        fibre_pct, fragment_pct = selected_vel_shape_percentages()
        return f"Total: {fibre_pct + fragment_pct:.0f}% | Fibres {fibre_pct:.0f}%, fragments {fragment_pct:.0f}%"

    @render.plot(alt="Synthetic microplastic size probability density plot")
    def vel_size_pdf_plot():
        df = selected_vel_micro_df()
        fig, ax = plt.subplots(figsize=(2.6, 1.7))
        size_um = pd.to_numeric(df.get("size_um"), errors="coerce").dropna().to_numpy(dtype=float)
        if len(size_um) > 0:
            ax.hist(size_um, bins=35, density=True, alpha=0.8)
            ax.set_yscale("log")
        ax.set_xlabel("Size (µm)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.18)
        fig.tight_layout(pad=0.6)
        return fig

    @render.plot(alt="Synthetic microplastic shape mixture pie chart")
    def vel_shape_mix_plot():
        fibre_pct, fragment_pct = selected_vel_shape_percentages()
        fig, ax = plt.subplots(figsize=(2.4, 1.7))
        values = [fibre_pct, fragment_pct]
        labels = ["Fibres", "Fragments"]
        if sum(values) <= 0:
            ax.text(0.5, 0.5, "No shape\nselected", ha="center", va="center", fontsize=8)
            ax.axis("off")
        else:
            ax.pie(values, labels=labels, autopct="%.0f%%", textprops={"fontsize": 7})
        fig.tight_layout(pad=0.5)
        return fig

    @render.plot(alt="Synthetic microplastic polymer mixture pie chart")
    def vel_polymer_mix_plot():
        fig, ax = plt.subplots(figsize=(2.4, 1.7))
        if not polymer_total_valid(selected_vel_polymer_total()):
            ax.text(0.5, 0.5, "Polymer total\nmust be 100%", ha="center", va="center", fontsize=8)
            ax.axis("off")
            fig.tight_layout(pad=0.5)
            return fig

        polymer_percentages = selected_vel_polymer_percentages()
        labels = []
        values = []
        for name, value in polymer_percentages.items():
            if value > 0:
                labels.append(name)
                values.append(value)
        if sum(values) <= 0:
            ax.text(0.5, 0.5, "No polymer\nselected", ha="center", va="center", fontsize=8)
            ax.axis("off")
        else:
            ax.pie(values, labels=labels, autopct="%.0f%%", textprops={"fontsize": 6})
        fig.tight_layout(pad=0.5)
        return fig

    @render.plot(alt="Synthetic microplastic settling and rising velocity distributions")
    def velocity_distribution_plot():
        df = selected_vel_micro_df()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        velocity_specs = [
            ("velocity_dietrich", "Dietrich"),
            ("velocity_goral", "Goral"),
            ("velocity_yu", "Yu"),
        ]

        all_values = []
        cleaned = []
        for col, label in velocity_specs:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                cleaned.append((vals, label))
                all_values.append(vals)

        if len(all_values) == 0:
            ax.text(0.5, 0.5, "No valid settling/rising velocities.", ha="center", va="center")
            ax.axis("off")
            return fig

        combined = np.concatenate(all_values)
        low, high = np.nanpercentile(combined, [0.5, 99.5])
        max_abs = max(abs(float(low)), abs(float(high)), 1e-8)
        bins = np.linspace(-max_abs, max_abs, 90)

        for vals, label in cleaned:
            ax.hist(vals, bins=bins, histtype="step", linewidth=2, density=True, label=label)

        ax.axvline(0, color="black", linewidth=1, alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Vertical velocity, w (m/s)\nnegative = rising, positive = settling", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.set_title("Generated settling and rising velocities", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    @render.data_frame
    def vel_synthetic_micro_summary():
        df = selected_vel_micro_df()
        if df.empty:
            summary = pd.DataFrame({"Metric": ["No synthetic data"], "Value": [""]})
        else:
            rows = [
                {"Metric": "Particles generated", "Value": f"{len(df):,}"},
                {"Metric": "Size range", "Value": f"{df['size_um'].min():.3g}–{df['size_um'].max():.3g} µm"},
                {"Metric": "Density range", "Value": f"{df['density_g_cm3'].min():.3g}–{df['density_g_cm3'].max():.3g} g cm⁻³"},
            ]
            for col, label in [("velocity_dietrich", "Dietrich w"), ("velocity_goral", "Goral w"), ("velocity_yu", "Yu w")]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
                    if len(vals) > 0:
                        rows.append({"Metric": label, "Value": f"median {np.nanmedian(vals):.3g} m/s; 25–75% {np.nanpercentile(vals,25):.3g} to {np.nanpercentile(vals,75):.3g}"})
            summary = pd.DataFrame(rows)

        return render.DataGrid(
            summary,
            width="100%",
            height="120px",
            filters=False,
            summary=False,
        )


    @render.plot(alt="Sampling-tab synthetic microplastic size probability density plot")
    def samp_size_pdf_plot():
        df = selected_samp_micro_df()
        fig, ax = plt.subplots(figsize=(2.6, 1.7))
        size_um = pd.to_numeric(df.get("size_um"), errors="coerce").dropna().to_numpy(dtype=float)
        if len(size_um) > 0:
            ax.hist(size_um, bins=35, density=True, alpha=0.8)
            ax.set_yscale("log")
        ax.set_xlabel("Size (µm)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
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
            ax.text(0.5, 0.5, "No shape\nselected", ha="center", va="center", fontsize=8)
            ax.axis("off")
        else:
            ax.pie(values, labels=labels, autopct="%.0f%%", textprops={"fontsize": 7})
        fig.tight_layout(pad=0.5)
        return fig

    @render.plot(alt="Sampling-tab synthetic microplastic polymer mixture pie chart")
    def samp_polymer_mix_plot():
        fig, ax = plt.subplots(figsize=(2.4, 1.7))
        if not polymer_total_valid(selected_samp_polymer_total()):
            ax.text(0.5, 0.5, "Polymer total\nmust be 100%", ha="center", va="center", fontsize=8)
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
            ax.text(0.5, 0.5, "No polymer\nselected", ha="center", va="center", fontsize=8)
            ax.axis("off")
        else:
            ax.pie(values, labels=labels, autopct="%.0f%%", textprops={"fontsize": 6})
        fig.tight_layout(pad=0.5)
        return fig

    @render.data_frame
    def net_sampling_results():
        if not samp_net_sampling_enabled():
            df = pd.DataFrame({"Output": ["Turn on 'Show captured/missed estimate' to calculate capture fractions."]})
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
            )
            df = df.rename(
                columns={
                    "Sampled z/H interval": "Sampling depth interval (z/H)",
                    "Water-column fraction sampled": "Fraction of water column sampled",
                    "Capture fraction": "Captured fraction",
                    "Missed fraction": "Missed fraction",
                    "Captured (%)": "Captured percentage",
                    "Missed (%)": "Missed percentage",
                }
            )
            keep_cols = [
                c for c in [
                    "Group",
                    "Sampling depth interval (z/H)",
                    "Fraction of water column sampled",
                    "Captured fraction",
                    "Missed fraction",
                    "Captured percentage",
                    "Missed percentage",
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


    @render.data_frame
    def sampling_correction_results():
        if not samp_sampling_correction_enabled():
            df = pd.DataFrame({"Output": ["Turn on 'Show concentration correction' to calculate depth-averaged concentration."]})
        else:
            df = sampling_correction_table(
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
                concentration_units=str(input.samp_concentration_units()),
                include_discharge=bool(input.samp_include_discharge()),
                discharge=float(input.samp_discharge()) if bool(input.samp_include_discharge()) else np.nan,
                iqr_lower=selected_samp_iqr_percentiles()[0],
                iqr_upper=selected_samp_iqr_percentiles()[1],
            )
            df = df.rename(
                columns={
                    "Sampled z/H interval": "Sampling depth interval (z/H)",
                    "Measured concentration": "Measured concentration",
                    "Capture fraction": "Captured fraction",
                    "Correction factor": "Correction factor",
                    "Estimated depth-averaged concentration": "Estimated depth-averaged concentration",
                    "Discharge Q (m3/s)": "Discharge (m³/s)",
                    "Estimated load": "Estimated load",
                    "Load units": "Load units",
                }
            )
            keep_cols = [
                c for c in [
                    "Group",
                    "Sampling depth interval (z/H)",
                    "Measured concentration",
                    "Units",
                    "Captured fraction",
                    "Correction factor",
                    "Estimated depth-averaged concentration",
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


    @render.data_frame
    def discharge_load_results():
        if not bool(input.samp_include_discharge()):
            df = pd.DataFrame({"Output": ["Turn on 'Include river discharge Q' to calculate estimated load."]})
        elif not samp_sampling_correction_enabled():
            df = pd.DataFrame({"Output": ["Turn on 'Show concentration correction' first, then include discharge to calculate estimated load."]})
        else:
            df = sampling_correction_table(
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
                concentration_units=str(input.samp_concentration_units()),
                include_discharge=True,
                discharge=float(input.samp_discharge()),
                iqr_lower=selected_samp_iqr_percentiles()[0],
                iqr_upper=selected_samp_iqr_percentiles()[1],
            )
            df = df.rename(
                columns={
                    "Sampled z/H interval": "Sampling depth interval (z/H)",
                    "Discharge Q (m3/s)": "Discharge (m³/s)",
                    "Estimated load": "Estimated load",
                    "Load units": "Load units",
                }
            )
            keep_cols = [
                c for c in [
                    "Group",
                    "Sampling depth interval (z/H)",
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


    @render.download(filename="explorer_synthetic_particles.csv")
    def download_explorer_synthetic_csv():
        """Download the Explorer-tab generated synthetic microplastic dataset."""
        yield selected_micro_df().to_csv(index=False)

    @render.download(filename="velocity_tab_synthetic_particles.csv")
    def download_velocity_synthetic_csv():
        """Download the Settling and rising velocities synthetic dataset."""
        yield selected_vel_micro_df().to_csv(index=False)

    @render.download(filename="velocity_tab_summary.csv")
    def download_velocity_summary_csv():
        """Download the velocity-tab summary table."""
        yield build_velocity_summary_df().to_csv(index=False)

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

    @render.plot(alt="Synthetic microplastic particle dataset summary")
    def synthetic_micro_plot():
        df = selected_micro_df()

        fig, ax = plt.subplots(figsize=(8.5, 3.8))

        if df.empty or "size_um" not in df.columns:
            ax.text(0.5, 0.5, "No synthetic microplastic data available.", ha="center", va="center")
            ax.axis("off")
            return fig

        # Plot absolute settling/rising velocity from the three equations.
        velocity_data = []
        labels = []

        for col, label in [
            ("velocity_dietrich", "Dietrich"),
            ("velocity_goral", "Goral"),
            ("velocity_yu", "Yu"),
        ]:
            if col in df.columns:
                vals = np.abs(df[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float))
                vals = vals[vals > 0]
                if len(vals) > 0:
                    velocity_data.append(vals)
                    labels.append(label)

        if len(velocity_data) == 0:
            ax.text(0.5, 0.5, "No valid synthetic settling velocities.", ha="center", va="center")
            ax.axis("off")
            return fig

        bins = np.geomspace(
            max(min(np.min(v) for v in velocity_data), 1e-8),
            max(np.max(v) for v in velocity_data),
            80,
        )

        for vals, label in zip(velocity_data, labels):
            ax.hist(
                vals,
                bins=bins,
                histtype="step",
                linewidth=2,
                density=True,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Absolute settling/rising velocity, |w| (m/s)")
        ax.set_ylabel("Probability density")
        ax.set_title("Synthetic microplastic settling-velocity distributions")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

        fig.tight_layout()
        return fig

    @render.data_frame
    def synthetic_micro_summary():
        df = selected_micro_df()

        if df.empty:
            summary = pd.DataFrame({"Metric": ["No synthetic data"], "Value": [""]})
        else:
            polymer_counts = df["polymer"].value_counts(normalize=True).mul(100)
            type_counts = df["particle_type"].value_counts(normalize=True).mul(100)

            rows = [
                {"Metric": "Particles generated", "Value": f"{len(df):,}"},
                {"Metric": "Size range", "Value": f"{df['size_um'].min():.3g}–{df['size_um'].max():.3g} µm"},
                {"Metric": "Density range", "Value": f"{df['density_g_cm3'].min():.3g}–{df['density_g_cm3'].max():.3g} g cm⁻³"},
                {"Metric": "Fibres", "Value": f"{type_counts.get('fiber', 0):.1f}%"},
                {"Metric": "Fragments", "Value": f"{type_counts.get('fragment', 0):.1f}%"},
            ]

            for polymer in ["PE", "PET", "PA", "PP", "PS", "PVA", "PVC"]:
                rows.append(
                    {
                        "Metric": f"{polymer} share",
                        "Value": f"{polymer_counts.get(polymer, 0):.1f}%",
                    }
                )

            summary = pd.DataFrame(rows)

        return render.DataGrid(
            summary,
            width="100%",
            height="95px",
            filters=False,
            summary=False,
        )

    @render.text
    def summary_text():
        macro_counts = macro["Material_grouped"].value_counts(dropna=False)
        macro_item_counts = (
            macro["Common name"].value_counts(dropna=False)
            if "Common name" in macro.columns
            else pd.Series(dtype=int)
        )

        current_micro = selected_micro_df()

        missing_micro_velocity = current_micro[velocity_cols].isna().sum()
        missing_macro_velocity = macro["vz_mean"].isna().sum()

        micro_range_lines = []
        for range_name, min_um, max_um in selected_micro_ranges():
            n = int(((current_micro["size_um"] >= min_um) & (current_micro["size_um"] <= max_um)).sum())
            micro_range_lines.append(f"{range_name}: {min_um:g}–{max_um:g} µm, n = {n}")

        if not micro_range_lines:
            micro_range_lines.append("No microplastic size ranges selected.")

        return (
            f"Selected/calculated u*: {selected_u_star():.4f} m/s\n"
            f"Flow depth H: {selected_flow_depth():.2f} m\n"
            f"Bed reference offset a_bed/H: {float(input.a_bed_frac()):.2f}\n"
            f"Surface reference offset a_surf/H: {float(input.a_surf_frac()):.2f}\n\n"
            f"Microplastic source: synthetic generated dataset\n\n"
            "Selected microplastic size ranges:\n"
            f"{chr(10).join(micro_range_lines)}\n\n"
            f"Macro selection mode: {'individual litter items' if use_macro_items() else 'grouped categories'}\n"
            f"Selected individual macro items: {', '.join(selected_macro_items()) if selected_macro_items() else 'None'}\n\n"
            "Macro material-group counts:\n"
            f"{macro_counts.to_string()}\n\n"
            "Available macro litter items by Common name:\n"
            f"{macro_item_counts.to_string() if not macro_item_counts.empty else 'Common name column not found.'}\n\n"
            "Missing micro velocity values by equation:\n"
            f"{missing_micro_velocity.to_string()}\n\n"
            f"Missing macro vz_mean values: {missing_macro_velocity}"
        )


app = App(app_ui, server)
