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
import re
import base64
from io import BytesIO
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
macro = pd.read_excel("macroplastic_particles_settling.xlsx")


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
    """Return a stable Shiny input id for one macroplastic item."""
    return f"samp_macro_item_concentration_{macro_common_name_to_index[common_name]}"


def macro_group_concentration_input_id(group_key: str) -> str:
    """Return a stable Shiny input id for one grouped macroplastic class."""
    return f"samp_macro_group_concentration_{group_key}"


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
    z_min = max(float(z_rel[0]), min(float(net_z_min), float(net_z_max)))
    z_max = min(float(z_rel[-1]), max(float(net_z_min), float(net_z_max)))

    if z_max <= z_min:
        captured = np.zeros(len(beta_values), dtype=float)
    else:
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
        sampled = np.trapezoid(sample_profiles, sample_z, axis=1)
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
    """Add one combined macroplastic group, optionally followed by its members."""
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
    result = non_macro_groups + [("Macroplastics: total", combined_beta)]
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
    net_z_min: float,
    net_z_max: float,
    iqr_lower: float,
    iqr_upper: float,
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
                "Population (%)": (
                    f"{100 * extra_micro_counts[group_name] / total_micro_count:.1f}%"
                    if group_name in extra_micro_counts and total_micro_count > 0
                    else
                    "100%"
                    if group_name.startswith("Microplastics") and group_name.endswith(": total")
                    else f"{100 * len(finite(beta)) / micro_total_count:.1f}%"
                    if group_name.startswith("Microplastics:")
                    and micro_total_count > 0
                    else ""
                ),
                "Sampled z/H interval": f"{min(net_z_min, net_z_max):.2f}–{max(net_z_min, net_z_max):.2f}",
                "Water-column fraction sampled": round(abs(float(net_z_max) - float(net_z_min)), 3),
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
    sampled_depth_fraction = min(
        abs(float(net_z_max) - float(net_z_min)),
        1.0,
    )

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
            net_z_min=net_z_min,
            net_z_max=net_z_max,
        )

        valid_captured = captured_values[
            np.isfinite(captured_values) & (captured_values > 0)
        ]

        if len(valid_captured) > 0:
            median_capture = float(np.nanmedian(valid_captured))
            reliable_capture = median_capture >= MIN_RELIABLE_CAPTURE
            if reliable_capture:
                correction_factor_values = (
                    sampled_depth_fraction / valid_captured
                )
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
) -> pd.DataFrame:
    """Correct separately measured macroplastic item or class concentrations."""
    rows = []
    corrected_distributions = []
    capture_distributions = []
    total_measured = 0.0
    q = float(discharge)
    rng = np.random.default_rng(42)
    draw_count = 5000
    has_unreliable_positive_component = False
    sampled_depth_fraction = min(
        abs(float(net_z_max) - float(net_z_min)),
        1.0,
    )

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
            net_z_min=net_z_min,
            net_z_max=net_z_max,
        )
        valid_captured = captured_values[
            np.isfinite(captured_values) & (captured_values > 0)
        ]
        if len(valid_captured) > 0:
            capture_distributions.append(valid_captured)

        if len(valid_captured) > 0:
            median_capture = float(np.nanmedian(valid_captured))
            raw_corrected_values = (
                c_obs * sampled_depth_fraction / valid_captured
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
            groups.append((f"Macroplastic item: {common_name}", beta))
    else:
        for group_key in macro_selected:
            beta = beta_values_for_macro_group(group_key, u_star)
            groups.append((f"Macroplastics: {macro_group_labels[group_key]}", beta))

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
    split_micro_by_direction: bool = False,
    extra_micro_groups: list[tuple[str, np.ndarray]] | None = None,
    include_micro_total: bool = True,
    show_iqr: bool = False,
    warning_text: str | None = None,
) -> plt.Figure:
    """
    Build the vertical Rouse profile figure.

    x-axis:
        Normalised relative concentration, 0 to 1.

    y-axis:
        Relative height, z/H.
    """
    figure_size = (7.2, 5.8) if show_net_interval else (8, 7)
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

        ax.plot(
            summary["median"],
            summary["z_rel"],
            linewidth=2.5,
            label=group_name,
        )

        if show_iqr:
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
    
    if show_net_interval and net_z_interval is not None:
        net_z_min, net_z_max = net_z_interval
        requested_low = min(float(net_z_min), float(net_z_max))
        requested_high = max(float(net_z_min), float(net_z_max))
        net_low = max(0.0, requested_low)
        net_high = min(1.0, requested_high)

        if net_high > net_low:
            ax.axhspan(
                net_low,
                net_high,
                alpha=0.08,
                zorder=0,
                label="Sampling interval",
            )
            ax.axhline(
                net_low,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                zorder=10,
                clip_on=False,
            )
            ax.axhline(
                net_high,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                zorder=10,
                clip_on=False,
            )
            ax.text(
                0.985,
                (net_low + net_high) / 2,
                f"Sample: {requested_low:.2f}–{requested_high:.2f} z/H",
                ha="right",
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
        r"Relative river depth, $z/H$",
        fontsize=PLOT_FONT_STANDARD,
    )
    ax.set_ylim(0, 1.02)
    ax.set_yticks(
        np.arange(0, 1.01, 0.1)
    )
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
APP_DIR = Path(__file__).resolve().parent
METHODS_FILE = APP_DIR / "Vertical_Plastic_Profile_App_Methods.md"
LANDING_FILE = APP_DIR / "RIVER_PLAST_Landing_Page.md"

try:
    methods_text = METHODS_FILE.read_text(encoding="utf-8")
except FileNotFoundError:
    methods_text = (
        "# Methods file not found\n\n"
        "The file `Vertical_Plastic_Profile_App_Methods.md` was not found. "
        "Place it in the same folder as this app file."
    )

try:
    landing_text = LANDING_FILE.read_text(encoding="utf-8")
except FileNotFoundError:
    landing_text = "# RIVER-PLAST"


def landing_logo(filename: str, label: str) -> ui.Tag:
    """Embed a landing-page logo so it also works in published deployments."""
    logo_path = APP_DIR / "www" / filename
    if logo_path.exists():
        mime_type = "image/svg+xml" if logo_path.suffix.lower() == ".svg" else "image/png"
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
                "Macroplastics",
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
                        ui.input_slider("samp_polymer_PE", "PE (%)", min=0, max=100, value=25, step=1),
                        ui.input_slider("samp_polymer_PET", "PET (%)", min=0, max=100, value=20, step=1),
                        ui.input_slider("samp_polymer_PA", "PA (%)", min=0, max=100, value=20, step=1),
                        ui.input_slider("samp_polymer_PP", "PP (%)", min=0, max=100, value=20, step=1),
                        ui.input_slider("samp_polymer_PS", "PS (%)", min=0, max=100, value=15, step=1),
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
            "design_plastic_type", "Plastic type",
            choices={"microplastic": "Microplastics", "macroplastic": "Macroplastics"},
            selected="microplastic", inline=True,
        ),
        ui.panel_conditional(
            "input.design_plastic_type === 'microplastic'",
            ui.tags.details(
                ui.tags.summary("Size"),
                ui.div(
                    ui.input_slider("design_size_range", "Particle size limits (µm)", 20, 5000, (300, 5000), step=10),
                    ui.input_checkbox("design_add_size_group", "Add another size group", False),
                    ui.panel_conditional(
                        "input.design_add_size_group",
                        ui.input_slider("design_size_range_2", "Second group size limits (µm)", 20, 5000, (20, 300), step=10),
                    ),
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
                choices={"individual": "Individual litter items", "grouped": "Grouped litter items"},
                selected="individual", inline=True,
            ),
            ui.panel_conditional(
                "input.design_macro_mode === 'grouped'",
                ui.input_checkbox_group("design_macro_categories", "Classes", choices=macro_group_labels, selected=list(macro_group_labels.keys())),
            ),
            ui.panel_conditional(
                "input.design_macro_mode === 'individual'",
                ui.input_selectize("design_macro_common_names", "Individual litter items", choices=macro_common_names,
                    selected=["Soft plastic pieces/films 0.5-2.5 cm"], multiple=True,
                    options={"placeholder": "Search or scroll through litter items", "plugins": ["remove_button"], "dropdownParent": "body"}),
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
                    .navbar-nav .nav-item:has(
                        .nav-link[data-value="Sampling correction"]
                    ) {
                        order: -1;
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
                        margin-top: 1rem;
                    }
                    .sampling-plastic-controls > h3 {
                        font-size: 0.95rem;
                        margin-bottom: 0.55rem;
                    }
                    .bslib-sidebar-layout > .sidebar.sampling-setup-sidebar {
                        width: 540px !important;
                        min-width: 540px !important;
                        max-width: 540px !important;
                        overflow: visible !important;
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
                        margin-top: 0.8rem !important;
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
                    .batch-visualisation-controls {
                        display: flex;
                        align-items: end;
                        gap: 1rem;
                        flex-wrap: wrap;
                    }
                    .batch-visualisation-controls .shiny-input-container {
                        margin-bottom: 0.35rem;
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
            ui.input_action_button(
                "start_app",
                "Start app",
                class_="btn btn-primary landing-start-button",
            ),
            ui.div(
                landing_logo("Logo_KIT.svg", "Karlsruhe Institute of Technology"),
                landing_logo("logo-horizontal-on-white_0.png", "UC Riverside"),
                landing_logo("Imperial_College_London_new_logo.png", "Imperial College London"),
            ),
            ui.p("Citation: XXX", class_="compact-note"),
            class_="landing-page",
            id="river-plast-home",
        ),
        value="home",
    ),


    ui.nav_panel(
        "Sampling correction",
        ui.page_sidebar(
            ui.sidebar(
                ui.tags.span(id="river-plast-sampling"),
                ui.p(
                    "Use this page to visualise the vertical distribution of sampled plastics and convert sample data into depth-average concentrations and loads.",
                    class_="sampling-page-intro",
                ),
                ui.tags.details(
                    ui.tags.summary("How to use this page"),
                    ui.div(
                        ui.markdown(
                            """
1. Download the example workbook.
2. Fill in flow, sampling and plastic data from your sample.
3. Upload sampling data.
4. View concentration profiles.
5. Export results.
                            """
                        ),
                        class_="sampling-help-body",
                    ),
                    class_="sampling-help",
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
                width="540px",
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
2. Choose either microplastics or macroplastics and define the population from your collected sample.
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
                                value=(0.80, 1.00),
                                step=0.01,
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
                width="540px",
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
                            ui.tags.li("Dashed lines show the depth interval sampled."),
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
        "Sample design",
        ui.page_sidebar(
            ui.sidebar(
                ui.p(
                    "Use this page before fieldwork to choose a sampling depth that is suitable for the river and plastics you expect to collect.",
                    class_="sampling-page-intro",
                ),
                ui.tags.details(
                    ui.tags.summary("How to use this page"),
                    ui.div(
                        ui.tags.ol(
                            ui.tags.li("Enter the expected river flow conditions."),
                            ui.tags.li("Describe the plastics you want to target."),
                            ui.tags.li("Choose a proposed or recommended sampling depth interval."),
                            ui.tags.li("Use the expected capture result to select a depth that represents the target plastics."),
                        ),
                        ui.p("Use Sampling correction after data collection to estimate depth-average concentration and load."),
                        class_="sampling-help-body",
                    ),
                    class_="sampling-help",
                ),
                ui.navset_card_tab(
                    ui.nav_panel(
                        "Flow",
                        ui.h6("Expected river flow conditions"),
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
                        ui.p("Choose the part of the water column you plan to sample.", class_="helper-text"),
                        ui.input_slider("design_net_z_interval", "Sampling depth interval", 0, 1, (0.80, 1.00), step=0.01),
                        ui.p("Or find a recommended sampling depth.", class_="helper-text"),
                        ui.input_action_button("design_find_sampling_depth", "Find recommended sampling depth", class_="btn-primary w-100"),
                        ui.p(
                            "This tests the selected sampling height at positions through the water column.",
                            ui.tags.br(),
                            "It recommends the interval with the highest predicted capture of the selected plastics.",
                            class_="compact-note",
                            style="margin-top: 0.65rem; margin-bottom: 0.9rem;",
                        ),
                        ui.input_numeric("design_measured_concentration", "Expected concentration in sampled water", value=10.0, min=0.0, step=0.1),
                        ui.input_select("design_concentration_units", "Concentration units", {"particles/m3": "particles/m³", "g/m3": "g/m³"}, selected="particles/m3"),
                    ),
                    ui.nav_panel(
                        "Advanced",
                        ui.input_slider("design_a_bed_frac", ui.tags.span("Bed reference height ", ui.tags.i("a"), ui.tags.sub("bed"), "/", ui.tags.i("H")), 0.01, 0.30, 0.05, step=0.01),
                        ui.input_slider("design_a_surf_frac", ui.tags.span("Surface reference offset ", ui.tags.i("a"), ui.tags.sub("surf"), "/", ui.tags.i("H")), 0.01, 0.30, 0.05, step=0.01),
                        ui.input_slider("design_iqr_percentiles", "Particle variability percentiles", 0, 100, (25, 75), step=1),
                    ),
                    id="design_tabs",
                ), width="540px", class_="sampling-setup-sidebar",
            ),
            ui.div(
                ui.tags.details(
                    ui.tags.summary("How to read this graph"),
                    ui.div(ui.tags.ul(
                        ui.tags.li("This graph shows how plastic concentration changes in a river water column."),
                        ui.tags.li("The y-axis is river depth: y = 0 is the riverbed and y = 1 is the water surface."),
                        ui.tags.li("The x-axis shows concentration relative to the maximum."),
                        ui.tags.li("Dashed lines show the depth interval sampled."),
                    ), class_="sampling-help-body"), class_="sampling-help",
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
                ui.tags.details(
                    ui.tags.summary("Tables"),
                    ui.div(
                        ui.navset_card_tab(
                            ui.nav_panel("Captured fraction", ui.output_data_frame("design_capture_results")),
                            ui.nav_panel("Expected concentration", ui.output_data_frame("design_concentration_results")),
                            ui.nav_panel("Expected load", ui.output_data_frame("design_load_results")),
                        ),
                        class_="sampling-results-card",
                    ),
                    open=False,
                    class_="secondary-disclosure",
                ),
                class_="centre-analysis-panel",
            ),
        ), value="sample_design",
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

    imported_macro_concentrations = reactive.Value({})
    imported_sample_metadata = reactive.Value(None)
    imported_particle_records = reactive.Value(None)
    imported_batch_workbook = reactive.Value({})
    imported_batch_results = reactive.Value(pd.DataFrame())
    imported_batch_particles = reactive.Value(pd.DataFrame())

    @reactive.Effect
    @reactive.event(input.start_app)
    def _open_sampling_correction():
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
        for _, sample in sample_data.iterrows():
            river_id = str(sample["river_id"])
            sample_id = str(sample["sample_id"])
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
                        net_z_max=z_max, measured_concentration=concentration,
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
                            net_z_max=z_max, measured_concentration=concentration_subset,
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
                        net_z_min=z_min, net_z_max=z_max, concentration_units="particles/m3",
                        include_discharge=True, discharge=discharge, iqr_lower=25, iqr_upper=75,
                    )
                    result_sets = [("Macroplastics", result)]
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
            sample_data = sample_data.drop_duplicates(subset=["river_id", "sample_id"], keep="first")
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

        sample_ids = sample_data["sample_id"].astype(str).tolist()
        return ui.div(
            ui.input_selectize(
                "batch_sample_to_visualise",
                "Sample",
                choices=sample_ids,
                selected=sample_ids[0],
                options={"dropdownParent": "body"},
            ),
            ui.input_radio_buttons(
                "batch_plastic_type",
                "Plastic type",
                choices={
                    "microplastic": "Microplastics",
                    "macroplastic": "Macroplastics",
                },
                selected="microplastic",
                inline=True,
            ),
            ui.panel_conditional(
                "input.batch_plastic_type === 'microplastic'",
                ui.div(
                    ui.input_radio_buttons(
                        "batch_micro_detail",
                        "Visualisation",
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
                        "batch_show_iqr",
                        "Show 25–75% particle variability",
                        False,
                    ),
                    class_="batch-visualisation-controls",
                ),
            ),
            ui.tags.details(
                ui.tags.summary("How to read this graph"),
                ui.div(
                    ui.tags.ul(
                        ui.tags.li("This graph shows how plastic concentration changes in a river water column."),
                        ui.tags.li("The y-axis is river depth: y = 0 is the riverbed and y = 1 is the water surface."),
                        ui.tags.li("The x-axis shows concentration relative to the maximum. x = 1 is the maximum concentration. x = 0.6 is 60% of the maximum. x = 0.1 is 10% of the maximum and so on."),
                        ui.tags.li("Dashed lines show the depth interval sampled."),
                        ui.tags.li("The result boxes show estimated particles captured, depth-average concentration and load. See About & Methods for the calculations."),
                    ),
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
        sample = matching.iloc[0]
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

    @reactive.Effect
    @reactive.event(input.batch_micro_detail, ignore_init=True)
    def _sync_batch_micro_detail():
        if input.batch_micro_detail() is not None:
            ui.update_radio_buttons("samp_micro_detail", selected=input.batch_micro_detail(), session=session)

    @reactive.Effect
    @reactive.event(input.batch_show_iqr, ignore_init=True)
    def _sync_batch_iqr():
        ui.update_checkbox("samp_show_iqr", value=bool(input.batch_show_iqr()), session=session)

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
            show_net_interval=samp_net_sampling_enabled(),
            net_z_interval=selected_samp_net_interval(),
            split_micro_by_direction=show_samp_micro_direction_profiles(),
            extra_micro_groups=selected_samp_micro_detail_groups(),
            include_micro_total=not show_samp_micro_detail(),
            show_iqr=bool(input.batch_show_iqr() or False),
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
            raise ValueError("Choose either microplastic or macroplastic records for one upload.")
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
                raise ValueError("Macroplastic records must include macro_item.")
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
            if plastic_type not in {"microplastics", "macroplastics"} or not all(
                plastics["plastic_type"].astype(str) == plastic_type
            ):
                raise ValueError("Choose one plastic type in Plastic inputs.")
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
                    raise ValueError("Macroplastic inputs must use grouped or individual selections.")
                identifiers = plastics["identifier"].astype(str).tolist()
                if selection_type == "grouped":
                    if any(identifier not in macro_group_labels for identifier in identifiers):
                        raise ValueError("A grouped macroplastic class was not recognised.")
                    ui.update_checkbox_group("samp_macro_categories", selected=identifiers, session=session)
                else:
                    if any(identifier not in macro_common_names for identifier in identifiers):
                        raise ValueError("An individual macroplastic item was not recognised.")
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
                "The selected microplastics or macroplastics contain both "
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
                    col_widths=[4, 4, 4],
                    fill=False,
                ),
                class_="sampling-key-results",
            )

        if not groups:
            return ui.div()

        for group_name, beta_values in groups:
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
                        col_widths=[4, 4, 4],
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
        return _key_results_ui()


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

    def selected_design_polymer_percentages() -> dict[str, float]:
        values = {name: float(getattr(input, f"design_polymer_{name}")()) for name in DEFAULT_POLYMER_PERCENTAGES}
        return values if polymer_total_valid(sum(values.values())) else {name: 0.0 for name in values}

    def selected_design_micro_ranges() -> list[tuple[str, float, float]]:
        if str(input.design_plastic_type()) != "microplastic":
            return []
        ranges = [("synthetic MP", *input.design_size_range())]
        if bool(input.design_add_size_group()):
            ranges = [("Group 1", *input.design_size_range()), ("Group 2", *input.design_size_range_2())]
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
        return list(input.design_macro_common_names() or []) if str(input.design_plastic_type()) == "macroplastic" and str(input.design_macro_mode()) == "individual" else []

    def selected_design_groups() -> list[tuple[str, np.ndarray]]:
        if str(input.design_plastic_type()) == "macroplastic":
            return selected_group_beta_values(
                micro_ranges=[], macro_selected=selected_design_macro_categories(),
                macro_items_selected=selected_design_macro_items(),
                use_macro_items=str(input.design_macro_mode()) == "individual", u_star=selected_design_u_star(),
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

    def design_median_capture() -> float:
        """Return expected median capture for the planned target population."""
        groups = selected_design_groups()
        beta = finite(np.concatenate([values for _, values in groups])) if groups else np.array([])
        interval = tuple(float(value) for value in input.design_net_z_interval())
        capture, _ = sampling_fraction_distribution_from_beta(
            beta_values=beta,
            H=float(applied_design_flow.get()["river_depth_m"]),
            a_bed_frac=float(input.design_a_bed_frac()),
            a_surf_frac=float(input.design_a_surf_frac()),
            net_z_min=interval[0], net_z_max=interval[1],
        )
        values = finite(capture)
        return float(np.nanmedian(values)) if len(values) else np.nan

    @reactive.Effect
    @reactive.event(input.design_find_sampling_depth)
    def _find_recommended_sampling_depth():
        """Move the planned interval to the position with highest expected capture."""
        groups = selected_design_groups()
        beta = finite(np.concatenate([values for _, values in groups])) if groups else np.array([])
        if not len(beta):
            ui.notification_show("Select valid target plastics before finding a sampling depth.", type="warning", duration=5)
            return

        current_low, current_high = (float(value) for value in input.design_net_z_interval())
        interval_height = abs(current_high - current_low)
        if interval_height <= 0:
            ui.notification_show("Choose a sampling interval greater than zero.", type="warning", duration=5)
            return

        starts = np.arange(0.0, 1.0 - interval_height + 0.0001, 0.01)
        captures = []
        for start in starts:
            values, _ = sampling_fraction_distribution_from_beta(
                beta_values=beta,
                H=float(applied_design_flow.get()["river_depth_m"]),
                a_bed_frac=float(input.design_a_bed_frac()),
                a_surf_frac=float(input.design_a_surf_frac()),
                net_z_min=float(start),
                net_z_max=float(start + interval_height),
            )
            values = finite(values)
            captures.append(float(np.nanmedian(values)) if len(values) else -np.inf)

        best_start = float(starts[int(np.argmax(captures))])
        best_end = min(1.0, best_start + interval_height)
        ui.update_slider("design_net_z_interval", value=(best_start, best_end), session=session)
        ui.notification_show(
            f"Recommended interval: {best_start:.2f}–{best_end:.2f} z/H.",
            type="message",
            duration=5,
        )

    @render.plot(alt="Vertical Rouse concentration profile plot for sample design")
    def profile_plot_design():
        is_micro = str(input.design_plastic_type()) == "microplastic"
        detail = str(input.design_micro_detail()) if is_micro else "total"
        return make_profile_plot(
            micro_ranges=selected_design_micro_ranges(),
            macro_selected=selected_design_macro_categories(),
            macro_items_selected=selected_design_macro_items(),
            use_macro_items=str(input.design_macro_mode()) == "individual",
            u_star=selected_design_u_star(), H=float(applied_design_flow.get()["river_depth_m"]),
            a_bed_frac=float(input.design_a_bed_frac()), a_surf_frac=float(input.design_a_surf_frac()),
            iqr_lower=float(input.design_iqr_percentiles()[0]), iqr_upper=float(input.design_iqr_percentiles()[1]),
            show_net_interval=True,
            net_z_interval=tuple(float(value) for value in input.design_net_z_interval()),
            micro_df=selected_design_micro_df(),
            split_micro_by_direction=is_micro and detail == "summary",
            extra_micro_groups=(selected_design_detail_groups() if detail == "size" else selected_design_polymer_detail_groups()),
            include_micro_total=not (is_micro and detail in {"size", "polymer"}),
            show_iqr=False,
            warning_text=(
                "Not enough sample for representative results.\nSample a larger or different depth interval."
                if np.isfinite(design_median_capture()) and design_median_capture() < 0.05
                else None
            ),
        )

    @render.ui
    def design_key_results():
        interval = tuple(float(value) for value in input.design_net_z_interval())
        median_capture = design_median_capture()
        reliable_capture = np.isfinite(median_capture) and median_capture >= 0.05
        concentration = float(input.design_measured_concentration())
        corrected = concentration * abs(interval[1] - interval[0]) / median_capture if reliable_capture else np.nan
        discharge = float(applied_design_flow.get()["discharge"])
        units = {"particles/m3": "particles/m³", "g/m3": "g/m³"}[str(input.design_concentration_units())]
        load_units = {"particles/m3": "particles/s", "g/m3": "g/s"}[str(input.design_concentration_units())]
        capture_text = f"{median_capture * 100:.1f}%" if np.isfinite(median_capture) else "Not available"
        capture_note = ""
        if np.isfinite(median_capture) and median_capture < 0.05:
            capture_note = "Not enough sample for representative results. Sample a larger or different depth interval."
        return ui.div(
            ui.layout_columns(
                ui.value_box("Expected capture of target plastics", capture_text, theme="primary", height="110px", fill=False),
                ui.value_box("Expected depth-average concentration", f"{fmt_sig(corrected)} {units}" if reliable_capture else "Not enough sample", theme="primary", height="110px", fill=False),
                ui.value_box("Expected load", f"{fmt_sig(corrected * discharge)} {load_units}" if reliable_capture and discharge > 0 else "Not enough sample" if not reliable_capture else "Enter discharge", theme="primary", height="110px", fill=False),
                col_widths=[4, 4, 4], fill=False,
            ),
            ui.p(capture_note, class_="compact-note"),
            class_="sampling-key-results",
        )

    def design_micro_concentrations() -> dict[str, float]:
        """Use the planned sampled-water concentration for each selected size range."""
        expected = float(input.design_measured_concentration())
        concentrations = {}
        for name, low, high in selected_design_micro_ranges():
            group = "Microplastics" if name == "synthetic MP" else f"Microplastics: {name} ({low:g}–{high:g} µm)"
            concentrations[group] = expected
        return concentrations

    def design_table_groups() -> list[tuple[str, np.ndarray, float]]:
        """Return the total and displayed groups with their population fractions."""
        if str(input.design_plastic_type()) == "microplastic":
            df = selected_design_micro_df()
            beta = finite(calculate_micro_rouse_mean(selected_design_u_star(), micro_df=df)) if not df.empty else np.array([])
            if not len(beta):
                return []
            detail = str(input.design_micro_detail())
            groups: list[tuple[str, np.ndarray]] = [("Microplastics: total", beta)]
            if detail == "summary":
                groups.extend([
                    ("Microplastics: buoyant", beta[beta < 0]),
                    ("Microplastics: sinking", beta[beta >= 0]),
                ])
            elif detail == "size":
                groups.extend(selected_design_detail_groups())
            elif detail == "polymer":
                groups.extend(selected_design_polymer_detail_groups())
            return [
                (name, finite(values), len(finite(values)) / len(beta))
                for name, values in groups if len(finite(values))
            ]

        components = selected_design_groups()
        if not components:
            return []
        total = finite(np.concatenate([values for _, values in components]))
        groups = [("Macroplastics: total", total, 1.0)]
        component_fraction = 1.0 / len(components)
        groups.extend((name, finite(values), component_fraction) for name, values in components if len(finite(values)))
        return groups

    def design_correction_table(include_discharge: bool, include_macro_members: bool) -> pd.DataFrame:
        """Calculate expected concentration and load for every displayed profile group."""
        del include_macro_members  # All displayed groups are included in Sample design.
        z_min, z_max = (float(value) for value in input.design_net_z_interval())
        sampled_fraction = abs(z_max - z_min)
        expected_total = float(input.design_measured_concentration())
        discharge = float(applied_design_flow.get()["discharge"])
        units = {"particles/m3": "particles/m³", "g/m3": "g/m³"}[str(input.design_concentration_units())]
        load_units = {"particles/m3": "particles/s", "g/m3": "g/s"}[str(input.design_concentration_units())]
        q_low, q_high = (float(value) for value in input.design_iqr_percentiles())
        rows = []
        for group, beta, population_fraction in design_table_groups():
            capture, _ = sampling_fraction_distribution_from_beta(
                beta_values=beta,
                H=float(applied_design_flow.get()["river_depth_m"]),
                a_bed_frac=float(input.design_a_bed_frac()),
                a_surf_frac=float(input.design_a_surf_frac()),
                net_z_min=z_min,
                net_z_max=z_max,
            )
            capture = finite(capture)
            median_capture = float(np.nanmedian(capture)) if len(capture) else np.nan
            expected_group = expected_total * population_fraction
            reliable = np.isfinite(median_capture) and median_capture >= 0.05
            corrected_draws = expected_group * sampled_fraction / capture if reliable else np.array([])
            load_draws = corrected_draws * discharge if include_discharge else np.array([])
            rows.append({
                "Group": group,
                "Population (%)": f"{population_fraction * 100:.1f}%",
                "Measured concentration": round(expected_group, 4),
                "Units": units,
                "Capture (%)": format_median_iqr(capture, q_low, q_high, percent=True) if len(capture) else "Not available",
                "Missed (%)": format_median_iqr(1.0 - capture, q_low, q_high, percent=True) if len(capture) else "Not available",
                "Estimated depth-averaged concentration": format_median_iqr(corrected_draws, q_low, q_high) if len(corrected_draws) else "Not enough sample",
                "Discharge Q (m3/s)": round(discharge, 4) if include_discharge else np.nan,
                "Estimated load": format_median_iqr(load_draws, q_low, q_high) if len(load_draws) else "Not enough sample",
                "Load units": load_units if include_discharge else "",
            })
        return pd.DataFrame(rows)

    def design_capture_table() -> pd.DataFrame:
        """Return group-level capture and missed fractions for the design."""
        interval = tuple(float(value) for value in input.design_net_z_interval())
        is_micro = str(input.design_plastic_type()) == "microplastic"
        detail = str(input.design_micro_detail()) if is_micro else "total"
        return net_sampling_table(
            micro_ranges=selected_design_micro_ranges(),
            macro_selected=selected_design_macro_categories(),
            macro_items_selected=selected_design_macro_items(),
            use_macro_items=str(input.design_macro_mode()) == "individual",
            u_star=selected_design_u_star(),
            micro_df=selected_design_micro_df(),
            H=float(applied_design_flow.get()["river_depth_m"]),
            a_bed_frac=float(input.design_a_bed_frac()),
            a_surf_frac=float(input.design_a_surf_frac()),
            net_z_min=interval[0],
            net_z_max=interval[1],
            iqr_lower=float(input.design_iqr_percentiles()[0]),
            iqr_upper=float(input.design_iqr_percentiles()[1]),
            split_micro_by_direction=is_micro and detail == "summary",
            extra_micro_groups=(selected_design_detail_groups() if detail == "size" else selected_design_polymer_detail_groups()),
            include_micro_total=True,
        )

    @render.data_frame
    def design_capture_results():
        df = design_capture_table().rename(columns={"Population (%)": "Population %", "Capture (%)": "Capture %", "Missed (%)": "Missed %"})
        columns = [column for column in ["Group", "Population %", "Capture %", "Missed %"] if column in df.columns]
        return render.DataGrid(df.loc[:, columns], width="100%", height="165px", filters=False, summary=False)

    @render.data_frame
    def design_concentration_results():
        if np.isfinite(design_median_capture()) and design_median_capture() < 0.05:
            df = pd.DataFrame({"Output": ["Not enough sample for representative results. Sample a larger or different depth interval."]})
            return render.DataGrid(df, width="100%", height="95px", filters=False, summary=False)
        df = design_correction_table(include_discharge=False, include_macro_members=True)
        df = df.rename(columns={"Estimated depth-averaged concentration": "Expected depth-average concentration"})
        columns = [column for column in ["Group", "Population (%)", "Measured concentration", "Units", "Expected depth-average concentration"] if column in df.columns]
        return render.DataGrid(df.loc[:, columns], width="100%", height="220px", filters=False, summary=False)

    @render.data_frame
    def design_load_results():
        if np.isfinite(design_median_capture()) and design_median_capture() < 0.05:
            df = pd.DataFrame({"Output": ["Not enough sample for representative results. Sample a larger or different depth interval."]})
            return render.DataGrid(df, width="100%", height="95px", filters=False, summary=False)
        df = design_correction_table(include_discharge=True, include_macro_members=True)
        df = df.rename(columns={"Discharge Q (m3/s)": "Discharge (m³/s)", "Estimated load": "Expected load"})
        columns = [column for column in ["Group", "Discharge (m³/s)", "Expected load", "Load units"] if column in df.columns]
        return render.DataGrid(df.loc[:, columns], width="100%", height="165px", filters=False, summary=False)

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
                        "plastic_type": "macroplastics",
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
                        "plastic_type": "macroplastics",
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
        yield (Path(__file__).resolve().parent / "river_plast_batch_example.xlsx").read_bytes()

    @render.download(filename="river_plast_batch_results.xlsx")
    def download_batch_results_excel():
        sheets = imported_batch_workbook.get()

        def split_median_interval(value) -> tuple[str, str]:
            text = "" if pd.isna(value) else str(value)
            if "[" in text and "]" in text:
                median, interval = text.split("[", 1)
                return median.strip(), f"[{interval}"
            return text, ""

        def format_export(frame: pd.DataFrame, include_group: bool = True) -> pd.DataFrame:
            flow = sheets.get("Flow", pd.DataFrame()).copy()
            flow_columns = [
                "sample_id", "river_width_m", "river_depth_m", "slope", "discharge_m3_s"
            ]
            if not flow.empty:
                frame = frame.merge(flow[flow_columns], on="sample_id", how="left")
            else:
                for column in flow_columns[1:]:
                    frame[column] = np.nan
            capture_source = frame.get(
                "Capture (%)", pd.Series(np.nan, index=frame.index, dtype=object)
            ).copy()
            median_capture_source = frame.get(
                "Median captured (%)", pd.Series(np.nan, index=frame.index, dtype=object)
            )
            if isinstance(capture_source, pd.Series):
                capture_missing = capture_source.isna() | capture_source.astype(str).str.strip().eq("")
                capture_source = capture_source.where(~capture_missing, median_capture_source)

            if "Missed (%)" in frame:
                missed_source = frame["Missed (%)"].copy()
            else:
                missed_source = pd.Series(np.nan, index=frame.index, dtype=object)
            if isinstance(missed_source, pd.Series) and isinstance(capture_source, pd.Series):
                def missed_from_capture(value) -> str:
                    median, _ = split_median_interval(value)
                    try:
                        return f"{100.0 - float(median.replace('%', '').strip()):.3g}%"
                    except ValueError:
                        return ""
                missed_missing = missed_source.isna() | missed_source.astype(str).str.strip().eq("")
                missed_source = missed_source.where(
                    ~missed_missing, capture_source.map(missed_from_capture)
                )
            concentration_source = frame.get("Estimated depth-averaged concentration", "")
            load_source = frame.get("Estimated load", "")
            concentration_parts = concentration_source.map(split_median_interval) if isinstance(concentration_source, pd.Series) else []
            load_parts = load_source.map(split_median_interval) if isinstance(load_source, pd.Series) else []
            sampled_interval = frame.get(
                "Sampled z/H interval", pd.Series("", index=frame.index, dtype=object)
            ).copy()
            if "Sample" in sheets:
                sample_intervals = sheets["Sample"].copy()
                sample_intervals["Sampled z/H interval"] = sample_intervals.apply(
                    lambda row: f"{row['sample_z_min']:.2f}–{row['sample_z_max']:.2f}", axis=1
                )
                imported_intervals = frame["sample_id"].map(
                    sample_intervals.set_index("sample_id")["Sampled z/H interval"]
                )
                interval_missing = sampled_interval.isna() | sampled_interval.astype(str).str.strip().eq("")
                sampled_interval = sampled_interval.where(~interval_missing, imported_intervals)
            output = pd.DataFrame({
                "Group": frame.get("Group", ""),
                "plastic_type": frame.get("plastic_type", ""),
                "sample_id": frame.get("sample_id", ""),
                "river_id": frame.get("river_id", ""),
                "Measured concentration": frame.get("Measured concentration", ""),
                "Units": frame.get("Units", "particles/m3"),
                "Capture (%)": capture_source.map(split_median_interval).map(lambda value: value[0]) if isinstance(capture_source, pd.Series) else capture_source,
                "Missed (%)": missed_source.map(split_median_interval).map(lambda value: value[0]) if isinstance(missed_source, pd.Series) else missed_source,
                "Estimated depth-averaged concentration": [value[0] for value in concentration_parts],
                "Concentration 25–75 percentiles": [value[1] for value in concentration_parts],
                "Discharge Q (m3/s)": frame.get("Discharge Q (m3/s)", ""),
                "river_width_m": frame["river_width_m"],
                "river_depth_m": frame["river_depth_m"],
                "slope": frame["slope"],
                "discharge_m3_s": frame["discharge_m3_s"],
                "Estimated load": [value[0] for value in load_parts],
                "Load 25–75 percentiles": [value[1] for value in load_parts],
                "Load units": frame.get("Load units", ""),
                "Sampled z/H interval": sampled_interval,
            })
            return output if include_group else output.drop(columns="Group")

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
                    microplastics = sample_results[
                        sample_results["result_set"].str.startswith("Micro")
                    ].rename(columns={"result_set": "Breakdown"})
                    macroplastics = sample_results[
                        sample_results["result_set"] == "Macroplastics"
                    ].drop(columns="result_set")
                    start_row = 0
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
                        pd.DataFrame({"Macroplastics": [""]}).to_excel(
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
            show_net_interval=samp_net_sampling_enabled(),
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
