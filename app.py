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
    - Includes an optional net-sampling estimator using a z/H interval for captured/missed vertical fraction.
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

from shiny import App, Inputs, Outputs, Session, render, ui


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
micro_size_min_um = 0
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


def calculate_micro_rouse_mean(u_star: float) -> np.ndarray:
    """
    Calculate mean microplastic Rouse number across the three velocity equations.

        beta = w / (kappa u*)

    Negative beta means rising/upward tendency.
    Positive beta means settling/downward tendency.
    """
    beta_arrays = []

    for col in velocity_cols:
        w = micro[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        beta_arrays.append(w / (kappa * u_star))

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
) -> np.ndarray:
    """Return beta values for one user-selected microplastic size range.

    Size inputs are in micrometres. The underlying dataset stores size in metres,
    so a precomputed size_um column is used for direct filtering.
    """
    if max_um <= min_um:
        return np.array([])

    beta = calculate_micro_rouse_mean(u_star)
    mask = (micro["size_um"] >= min_um) & (micro["size_um"] <= max_um)

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
) -> pd.DataFrame:
    """Return captured/missed fractions for the current selected groups."""
    rows = []

    groups = selected_group_beta_values(
        micro_ranges=micro_ranges,
        macro_selected=macro_selected,
        macro_items_selected=macro_items_selected,
        use_macro_items=use_macro_items,
        u_star=u_star,
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
) -> list[tuple[str, np.ndarray]]:
    """Return display names and beta arrays for all selected ranges/categories/items."""
    groups = []

    for range_name, min_um, max_um in micro_ranges:
        beta = beta_values_for_micro_range(min_um=min_um, max_um=max_um, u_star=u_star)
        groups.append((f"Micro: {range_name} ({min_um:g}–{max_um:g} µm)", beta))

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
) -> plt.Figure:
    """
    Build the vertical Rouse profile figure.

    x-axis:
        Normalised relative concentration, 0 to 1.

    y-axis:
        Relative height, z/H.
    """
    fig, ax = plt.subplots(figsize=(8.5, 7))

    plotted_any = False

    groups = selected_group_beta_values(
        micro_ranges=micro_ranges,
        macro_selected=macro_selected,
        macro_items_selected=macro_items_selected,
        use_macro_items=use_macro_items,
        u_star=u_star,
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
        0.98,
        a_bed_frac,
        r"$a_{bed}$",
        ha="right",
        va="bottom",
        alpha=0.5,
        transform=ax.get_yaxis_transform(),
    )
    
    ax.text(
        0.98,
        1 - a_surf_frac,
        r"$a_{surf}$",
        ha="right",
        va="bottom",
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
                label="Net sampled interval",
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
                0.985,
                (net_low + net_high) / 2,
                f"Net: {net_low:.2f}–{net_high:.2f} z/H",
                ha="right",
                va="center",
                fontsize=9,
                alpha=0.85,
                transform=ax.get_yaxis_transform(),
            )

    ax.set_xlabel(r"Normalised concentration, $C / C_{max}$")
    ax.set_ylabel(r"Relative depth, $z/H$")
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
        rf"Vertical profiles, $u_*$ = {u_star:.3f} m s$^{{-1}}$, "
        rf"$H$ = {H:.2f} m"
    )

    ax.grid(True, alpha=0.25)

    if plotted_any:
        ax.legend(loc="best")
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
methods_text = 'Vertical Profile Plastic Transport App: Methods and Code Notes\n==========================================================\n\n1. Purpose of the app\n---------------------\n\nThis app estimates how microplastics and macroplastics are distributed vertically in a river water column using a Rouse-profile approach.\n\nThe main aim is to estimate how much plastic is likely to be inside a sampled part of the water column, and how much may be missed by a net that only samples a limited vertical range.\n\nThe app can also correct a measured concentration from the sampled layer to estimate a depth-averaged concentration for the full water column. If river discharge is supplied, the app can estimate a plastic transport rate.\n\nThe app is intended as a first-order scientific tool. It does not remove the need for field judgement, good sampling design, or uncertainty analysis.\n\n\n2. Vertical coordinate system\n-----------------------------\n\nThe app uses a relative vertical coordinate:\n\n    z / H\n\nwhere:\n\n    z = height above the river bed\n    H = total flow depth\n\nTherefore:\n\n    z/H = 0 means the river bed\n    z/H = 1 means the water surface\n\nThis makes it easier to compare rivers or experiments with different flow depths.\n\n\n3. Rouse number\n---------------\n\nThe Rouse number controls the shape of the vertical concentration profile.\n\nIt is calculated as:\n\n    beta = w / (kappa * u_*)\n\nwhere:\n\n    beta  = Rouse number\n    w     = particle settling or rising velocity\n    kappa = von Karman constant, taken as 0.41\n    u_*   = shear velocity\n\nInterpretation:\n\n    beta > 0   particle tends to settle towards the bed\n    beta ~ 0   particle is close to well mixed\n    beta < 0   particle tends to rise towards the surface\n\nLarge positive beta values produce profiles concentrated near the bed.\nNegative beta values produce profiles concentrated near the surface.\n\n\n4. Shear velocity\n-----------------\n\nThe user can either enter shear velocity directly or calculate it from hydraulic radius and slope.\n\nThe hydraulic calculation is:\n\n    u_* = sqrt(g * R * S)\n\nwhere:\n\n    u_* = shear velocity\n    g   = gravitational acceleration, 9.81 m s^-2\n    R   = hydraulic radius\n    S   = energy slope\n\nFor wide channels, hydraulic radius can often be approximated by flow depth.\n\n\n5. Settling particle Rouse profile\n----------------------------------\n\nFor settling particles, the classical bed-referenced Rouse profile is used:\n\n    C(z) / C(a) = [ (((H - z) / z) / ((H - a) / a)) ]^beta\n\nwhere:\n\n    C(z) = concentration at elevation z\n    C(a) = concentration at reference elevation a\n    H    = flow depth\n    a    = bed reference height\n    beta = Rouse number\n\nFor settling particles, beta is positive and concentration is greatest near the bed.\n\nThe bed reference height is written in the app as:\n\n    a_bed / H\n\nThis avoids calculating the profile exactly at the bed, where the equation becomes singular.\n\n\n6. Buoyant particle Rouse profile\n---------------------------------\n\nFor buoyant particles, beta is negative. These particles tend to accumulate near the water surface.\n\nThe app uses a mirrored surface-referenced form so that the concentration increases towards the surface rather than towards the bed.\n\nThe mirrored profile is based on distance from the surface and uses the absolute value of beta.\n\nConceptually:\n\n    settling particles: high concentration near bed\n    buoyant particles:  high concentration near surface\n\nThe surface reference is written in the app as:\n\n    a_surf / H\n\nThis avoids calculating the profile exactly at the surface, where the mirrored profile would become singular.\n\n\n7. Profile normalisation\n------------------------\n\nThe app normalises each calculated concentration profile:\n\n    C_norm(z) = C(z) / C_max\n\nwhere:\n\n    C_norm(z) = normalised concentration at elevation z\n    C_max     = maximum concentration in that profile\n\nThis means:\n\n    0 <= C_norm <= 1\n\nThe plotted profiles therefore show the shape of the concentration distribution, not the absolute concentration.\n\nThis is useful because different particle types can be compared on the same graph.\n\n\n8. Net sampling interval\n------------------------\n\nThe net sampling interval is defined using a joint slider in relative water-column coordinates.\n\nFor example:\n\n    0.80 to 1.00\n\nmeans the net samples the upper 20 percent of the water column.\n\nSimilarly:\n\n    0.00 to 0.20\n\nmeans the net samples the lower 20 percent of the water column.\n\nWhen the net sampling estimate is turned on, the selected interval is shown on the plot using dashed horizontal lines and a shaded band.\n\n\n9. Capture fraction\n-------------------\n\nThe capture fraction is the fraction of the total predicted vertical concentration that lies inside the sampled interval.\n\nThe total vertically integrated concentration is:\n\n    M_total = integral from 0 to H of C(z) dz\n\nThe concentration inside the sampled layer is:\n\n    M_sample = integral from z1 to z2 of C(z) dz\n\nThe capture fraction is:\n\n    F_capture = M_sample / M_total\n\nwhere:\n\n    F_capture = fraction of the full water-column concentration inside the sampled layer\n    z1        = lower boundary of the sampled interval\n    z2        = upper boundary of the sampled interval\n\nInterpretation:\n\n    F_capture = 1.00 means the sampled layer contains all predicted particles\n    F_capture = 0.50 means the sampled layer contains half of the predicted particles\n    F_capture = 0.25 means the sampled layer contains one quarter of the predicted particles\n    F_capture = 0.10 means the sampled layer contains one tenth of the predicted particles\n\nThe missed fraction is:\n\n    F_missed = 1 - F_capture\n\n\n10. Concentration correction\n----------------------------\n\nIf a field concentration is measured in the sampled layer, the app can estimate the equivalent depth-averaged concentration for the full water column.\n\nLet:\n\n    C_obs = observed concentration in the sampled layer\n\nThe correction factor is:\n\n    CF = 1 / F_capture\n\nThe corrected full-water-column concentration is:\n\n    C_corr = C_obs * CF\n\nor equivalently:\n\n    C_corr = C_obs / F_capture\n\nExample:\n\n    Observed concentration = 10 particles m^-3\n    Capture fraction       = 0.25\n    Correction factor      = 1 / 0.25 = 4\n    Corrected concentration = 10 * 4 = 40 particles m^-3\n\nThis means that if the model predicts the net sampled only 25 percent of the vertically distributed particles, the observed concentration is multiplied by 4 to estimate the full-depth average.\n\nThis correction is the main practical value of the sampling module.\n\n\n11. Transport estimate using discharge\n--------------------------------------\n\nIf river discharge is entered, the app can estimate plastic transport rate.\n\nThe basic equation is:\n\n    Load = C_corr * Q\n\nwhere:\n\n    Load   = plastic transport rate\n    C_corr = corrected depth-averaged concentration\n    Q      = river discharge\n\nIf concentration is in particles m^-3 and discharge is in m^3 s^-1, then:\n\n    Load = particles s^-1\n\nIf concentration is in mass m^-3 and discharge is in m^3 s^-1, then:\n\n    Load = mass s^-1\n\nThis is a first-order flux estimate.\n\n\n12. Important assumption behind the correction\n----------------------------------------------\n\nThe correction assumes that the predicted Rouse profile is a reasonable representation of the true vertical distribution.\n\nThe key ratio is:\n\n    M_sample / M_total\n\nThe app assumes that this modelled ratio is representative of the real fraction of particles sampled by the net.\n\nErrors in the following will directly affect the correction:\n\n    settling or rising velocity\n    shear velocity\n    flow depth\n    chosen reference levels\n    particle grouping\n    actual particle behaviour in the river\n\nThe correction factor becomes especially sensitive when F_capture is small.\n\nFor example:\n\n    F_capture = 0.50 gives correction factor 2\n    F_capture = 0.25 gives correction factor 4\n    F_capture = 0.10 gives correction factor 10\n    F_capture = 0.05 gives correction factor 20\n\nVery small capture fractions should therefore be treated cautiously.\n\n\n13. Microplastic treatment in the code\n--------------------------------------\n\nThe microplastic dataset is read from:\n\n    microplastic_particles_settling.csv\n\nThe app uses the particle size column and three settling velocity estimates:\n\n    velocity_dietrich\n    velocity_goral\n    velocity_yu\n\nFor each particle, the app calculates beta for each velocity equation and then takes a mean beta value.\n\nThe user can define up to three custom microplastic size ranges using sliders.\n\nFor each selected range, the app filters the microplastic dataset by size and calculates the profile summary for particles inside that range.\n\nEach range can be enabled or disabled separately.\n\n\n14. Macroplastic treatment in the code\n--------------------------------------\n\nThe macroplastic dataset is read from:\n\n    macroplastic_particles_settling.xlsx\n\nThe app uses:\n\n    Material\n    Common name\n    vz_mean\n\nThe Material column is mapped into broad groups:\n\n    buoyant\n    near neutral\n    dense\n\nThe user can choose macroplastics by group.\n\nThe app also includes an optional individual litter item selection using the Common name column. This is off by default. When enabled, the user can scroll or search through litter items and choose which specific items to plot.\n\nThe macroplastic beta value is calculated from vz_mean using the conversion already present in the code:\n\n    w = vz_mean / 100\n\nThen:\n\n    beta = w / (kappa * u_*)\n\n\n15. Main code structure\n-----------------------\n\nThe app is organised into several parts.\n\nData loading:\n\n    Reads the microplastic and macroplastic datasets.\n\nConstants:\n\n    Defines kappa, gravitational acceleration, velocity column names, and macroplastic grouping rules.\n\nPre-processing:\n\n    Adds grouped macroplastic classes and prepares values used later in the app.\n\nHelper functions:\n\n    finite()\n        Removes non-finite values such as NaN and infinity.\n\n    calculate_shear_velocity_from_slope_radius()\n        Calculates shear velocity from hydraulic radius and slope.\n\n    calculate_micro_rouse_mean()\n        Calculates mean microplastic Rouse numbers across the settling velocity equations.\n\n    calculate_macro_rouse()\n        Calculates macroplastic Rouse numbers from measured vertical velocity.\n\n    rouse_profile_from_beta()\n        Creates a direction-aware concentration profile from beta.\n        Positive beta gives a bed-referenced profile.\n        Negative beta gives a surface-referenced profile.\n\n    group_profile_summary()\n        Calculates median and interquartile profiles for each selected group.\n\n    make_profile_plot()\n        Builds the main plot, including profile lines, optional interquartile bands, reference lines, and optional net sampling interval.\n\n    capture_fraction_from_summary()\n        Integrates the concentration profile inside the selected net interval and compares it with the full profile integral.\n\n    sampling/correction table functions\n        Calculate capture fraction, missed fraction, correction factor, corrected concentration, and load.\n\nUser interface:\n\n    The UI uses a sidebar with collapsible sections.\n    These include hydraulic controls, microplastic controls, macroplastic controls, reference/display settings, net sampling estimate, and sampling correction.\n\nServer:\n\n    The server reads user inputs, calculates selected beta values, updates plots, and generates tables.\n\n\n16. Why the app uses median profiles\n------------------------------------\n\nFor each group, many particles may be included.\n\nInstead of plotting every particle, the app calculates a summary profile.\n\nThe main plotted line is the median profile.\n\nIf selected, the app also shows the interquartile range:\n\n    25th percentile to 75th percentile\n\nThis gives a simple measure of variability without overcrowding the graph.\n\n\n17. Main limitations\n--------------------\n\nThe method assumes:\n\n    flow is approximately steady\n    particles are suspended or behaving like suspended material\n    shear velocity represents vertical mixing\n    settling or rising velocity is representative\n    the river is reasonably laterally mixed\n    the vertical profile is time averaged\n\nThe method is less reliable for:\n\n    very buoyant particles\n    very thin sampling layers\n    strongly unsteady flow\n    strongly stratified flow\n    large individual macroplastic items\n    particles affected strongly by shape, turbulence, vegetation, wind, or surface tension\n\nMacroplastics should be interpreted carefully. For large items, the profile may be better understood as a probability of occurrence with depth rather than a smooth concentration field.\n\n\n18. Recommended wording for results\n-----------------------------------\n\nA careful way to report results is:\n\n    Based on the predicted Rouse profile, the selected net interval is estimated to contain X percent of the vertically integrated concentration for this particle class. The measured concentration was therefore corrected using a factor of Y to estimate a depth-averaged water-column concentration.\n\nAvoid saying:\n\n    The app proves the true concentration is X.\n\nBetter wording is:\n\n    The app estimates the depth-averaged concentration as X, conditional on the assumed Rouse profile and input parameters.\n\n'

app_ui = ui.page_navbar(
    ui.nav_panel(
        "Analysis",
        ui.page_sidebar(

    ui.sidebar(
        ui.tags.style(
            """
            .bslib-sidebar-layout > .sidebar {
                width: 420px !important;
                min-width: 420px !important;
            }
            .sidebar .form-group,
            .sidebar .shiny-input-container {
                margin-bottom: 1rem;
            }
            .sidebar h3 {
                margin-top: 0.25rem;
                margin-bottom: 1rem;
            }
            .main .card {
                margin-top: 1rem;
            }
            .plot-card .card-body {
                padding: 0.75rem 1rem 1rem 1rem;
            }
            .control-card .card-header {
                font-weight: 650;
            }
            .control-card .card-body {
                padding-top: 1rem;
            }
            .range-card {
                height: 100%;
                border: 1px solid rgba(0, 0, 0, 0.08);
                box-shadow: none;
            }
            .range-card .card-body {
                padding: 0.9rem 1rem 0.75rem 1rem;
            }
            .helper-text {
                color: #666;
                font-size: 0.92rem;
                margin-bottom: 0.75rem;
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
                position: relative;
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
            .sampling-note {
                color: #555;
                font-size: 0.9rem;
                line-height: 1.35;
                margin-top: 0.5rem;
            }
            """
        ),

        ui.h3("Controls"),

        ui.tags.details(
            ui.tags.summary("Hydraulic controls"),
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
            ui.tags.summary("Microplastic size ranges"),
            ui.div(
                ui.div(
                    "Choose up to three custom microplastic size ranges. Values are in µm.",
                    class_="helper-text",
                ),

                ui.div(
                    ui.input_checkbox("micro_range_1_enabled", "Show range 1", True),
                    ui.input_slider(
                        "micro_range_1",
                        "Range 1 size limits (µm)",
                        min=micro_size_min_um,
                        max=micro_size_max_um,
                        value=(1000, 5000),
                        step=10,
                    ),
                    class_="sidebar-range-block",
                ),

                ui.div(
                    ui.input_checkbox("micro_range_2_enabled", "Show range 2", False),
                    ui.input_slider(
                        "micro_range_2",
                        "Range 2 size limits (µm)",
                        min=micro_size_min_um,
                        max=micro_size_max_um,
                        value=(400, 1000),
                        step=10,
                    ),
                    class_="sidebar-range-block",
                ),

                ui.div(
                    ui.input_checkbox("micro_range_3_enabled", "Show range 3", False),
                    ui.input_slider(
                        "micro_range_3",
                        "Range 3 size limits (µm)",
                        min=micro_size_min_um,
                        max=micro_size_max_um,
                        value=(0, 400),
                        step=10,
                    ),
                    class_="sidebar-range-block",
                ),
                class_="collapsible-control-body",
            ),
            open=True,
            class_="collapsible-control",
        ),

        ui.tags.details(
            ui.tags.summary("Macroplastic controls"),
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
                            "buoyant": ui.HTML(
                                "Buoyant (Foams)<br><small>ρₚ ∈ [0.02, 0.08] g cm⁻³</small>"
                            ),
                            "near_neutral": ui.HTML(
                                "Near neutral (Plastics & others)<br><small>ρₚ ∈ [0.8, 1.5] g cm⁻³</small>"
                            ),
                            "dense": ui.HTML(
                                "Dense (Glass & metal)<br><small>ρₚ ∈ [2.5, 4.3] g cm⁻³</small>"
                            ),
                        },
                        selected=[
                           
                        ],
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
                        options={
                            "placeholder": "Search or scroll through litter items",
                            "plugins": ["remove_button"],
                        },
                    ),
                    ui.div(
                        "No individual items are selected by default. Choose one or more items to add them to the graph.",
                        class_="helper-text",
                    ),
                ),
                class_="collapsible-control-body",
            ),
            open=True,
            class_="collapsible-control",
        ),

        ui.tags.details(
            ui.tags.summary("Net sampling estimate"),
            ui.div(
                ui.div(
                    "Estimate the fraction of each vertical profile captured within a selected z/H interval. This is off by default.",
                    class_="helper-text",
                ),

                ui.input_checkbox(
                    "net_sampling_enabled",
                    "Show captured/missed estimate",
                    False,
                ),

                ui.panel_conditional(
                    "input.net_sampling_enabled",
                    ui.input_slider(
                        "net_z_interval",
                        "Net position in water column, z/H",
                        min=0.0,
                        max=1.0,
                        value=(0.80, 1.00),
                        step=0.01,
                    ),
                    ui.div(
                        "Use this as the vertical span sampled by the net: z/H = 0 is the bed and z/H = 1 is the surface. For example, 0.80–1.00 means the net samples the upper 20% of the water column. The method integrates the relative concentration profile inside this interval and divides by the full-profile integral, so captured + missed = 1 for each selected group.",
                        class_="sampling-note",
                    ),
                ),
                class_="collapsible-control-body",
            ),
            class_="collapsible-control",
        ),

        ui.tags.details(
            ui.tags.summary("Sampling correction"),
            ui.div(
                ui.div(
                    "Convert a measured concentration from the sampled net interval into an estimated depth-averaged concentration. Optional discharge converts that concentration into an estimated flux/load.",
                    class_="helper-text",
                ),

                ui.input_checkbox(
                    "sampling_correction_enabled",
                    "Show concentration correction",
                    False,
                ),

                ui.panel_conditional(
                    "input.sampling_correction_enabled",
                    ui.input_numeric(
                        "measured_concentration",
                        "Measured concentration in net sample",
                        value=10.0,
                        min=0.0,
                        step=0.1,
                    ),
                    ui.input_select(
                        "concentration_units",
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
                        "include_discharge",
                        "Include river discharge Q",
                        False,
                    ),
                    ui.panel_conditional(
                        "input.include_discharge",
                        ui.input_numeric(
                            "discharge",
                            "Discharge Q (m³/s)",
                            value=1.0,
                            min=0.0,
                            step=0.1,
                        ),
                    ),
                    ui.div(
                        "Correction uses C_depth-avg = C_measured / capture fraction. If Q is supplied, load = C_depth-avg × Q.",
                        class_="sampling-note",
                    ),
                ),
                class_="collapsible-control-body",
            ),
            class_="collapsible-control",
        ),

        ui.tags.details(
            ui.tags.summary("Flow depth, references, and display"),
            ui.div(
                ui.input_slider(
                    "H",
                    "Flow depth H (m)",
                    min=0.05,
                    max=5.00,
                    value=0.50,
                    step=0.05,
                ),

                ui.input_slider(
                    "a_bed_frac",
                    "Bed reference offset a_bed/H",
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

                ui.hr(),

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
            class_="collapsible-control",
        ),

        ui.hr(),

        ui.markdown(
            """
            **Interpretation**

            - `β > 0`: settling profile, referenced from the bed upward using `a_bed/H`.
            - `β < 0`: buoyant profile, referenced from the surface downward using `a_surf/H`.
            - The plotted profile is `C / Cmax`, so all curves range from 0 to 1.
            - Dashed line: bed reference offset `a_bed/H`.
            - Dotted line: surface reference offset `1 - a_surf/H`.
            """
        ),
        width="420px",
    ),

    ui.h2("Vertical concentration profiles"),

    ui.card(
        ui.output_plot("profile_plot", height="760px"),
        full_screen=True,
        class_="plot-card",
    ),

    ui.panel_conditional(
        "input.net_sampling_enabled",
        ui.card(
            ui.card_header("Net sampling captured/missed estimate (median [IQR])"),
            ui.output_data_frame("net_sampling_results"),
        ),
    ),

    ui.panel_conditional(
        "input.sampling_correction_enabled",
        ui.card(
            ui.card_header("Sampling correction: estimated whole-water-column concentration (median [IQR])"),
            ui.output_data_frame("sampling_correction_results"),
        ),
    ),

    ui.card(
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

    def selected_u_star() -> float:
        if input.ustar_mode() == "direct":
            return float(input.u_star())

        return calculate_shear_velocity_from_slope_radius(
            hydraulic_radius=float(input.hydraulic_radius()),
            slope=float(input.slope()),
        )

    def selected_micro_ranges() -> list[tuple[str, float, float]]:
        ranges = []

        for idx in (1, 2, 3):
            enabled = bool(getattr(input, f"micro_range_{idx}_enabled")())
            if not enabled:
                continue

            min_um, max_um = getattr(input, f"micro_range_{idx}")()
            min_um = float(min_um)
            max_um = float(max_um)

            if max_um > min_um:
                ranges.append((f"range {idx}", min_um, max_um))

        return ranges

    def selected_macro_categories() -> list[str]:
        return list(input.macro_categories() or [])

    def use_macro_items() -> bool:
        return bool(input.use_macro_items())

    def selected_macro_items() -> list[str]:
        return list(input.macro_common_names() or [])

    def net_sampling_enabled() -> bool:
        return bool(input.net_sampling_enabled())

    def sampling_correction_enabled() -> bool:
        return bool(input.sampling_correction_enabled())

    def selected_iqr_percentiles() -> tuple[float, float]:
        q_low, q_high = input.iqr_percentiles()
        return float(q_low), float(q_high)

    def selected_net_interval() -> tuple[float, float]:
        z_min, z_max = input.net_z_interval()
        return float(z_min), float(z_max)

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

    @render.plot(alt="Vertical Rouse concentration profile plot")
    def profile_plot():
        return make_profile_plot(
            micro_ranges=selected_micro_ranges(),
            macro_selected=selected_macro_categories(),
            macro_items_selected=selected_macro_items(),
            use_macro_items=use_macro_items(),
            u_star=selected_u_star(),
            H=float(input.H()),
            a_bed_frac=float(input.a_bed_frac()),
            a_surf_frac=float(input.a_surf_frac()),
            iqr_lower=selected_iqr_percentiles()[0],
            iqr_upper=selected_iqr_percentiles()[1],
            show_net_interval=net_sampling_enabled(),
            net_z_interval=selected_net_interval(),
        )

    @render.data_frame
    def net_sampling_results():
        df = net_sampling_table(
            micro_ranges=selected_micro_ranges(),
            macro_selected=selected_macro_categories(),
            macro_items_selected=selected_macro_items(),
            use_macro_items=use_macro_items(),
            u_star=selected_u_star(),
            H=float(input.H()),
            a_bed_frac=float(input.a_bed_frac()),
            a_surf_frac=float(input.a_surf_frac()),
            net_z_min=selected_net_interval()[0],
            net_z_max=selected_net_interval()[1],
            iqr_lower=selected_iqr_percentiles()[0],
            iqr_upper=selected_iqr_percentiles()[1],
        )

        return render.DataGrid(
            df,
            width="100%",
            height="280px",
            filters=False,
            summary=False,
        )

    @render.data_frame
    def sampling_correction_results():
        df = sampling_correction_table(
            micro_ranges=selected_micro_ranges(),
            macro_selected=selected_macro_categories(),
            macro_items_selected=selected_macro_items(),
            use_macro_items=use_macro_items(),
            u_star=selected_u_star(),
            H=float(input.H()),
            a_bed_frac=float(input.a_bed_frac()),
            a_surf_frac=float(input.a_surf_frac()),
            net_z_min=selected_net_interval()[0],
            net_z_max=selected_net_interval()[1],
            measured_concentration=float(input.measured_concentration()),
            concentration_units=str(input.concentration_units()),
            include_discharge=bool(input.include_discharge()),
            discharge=float(input.discharge()) if bool(input.include_discharge()) else np.nan,
            iqr_lower=selected_iqr_percentiles()[0],
            iqr_upper=selected_iqr_percentiles()[1],
        )

        return render.DataGrid(
            df,
            width="100%",
            height="320px",
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

        missing_micro_velocity = micro[velocity_cols].isna().sum()
        missing_macro_velocity = macro["vz_mean"].isna().sum()

        micro_range_lines = []
        for range_name, min_um, max_um in selected_micro_ranges():
            n = int(((micro["size_um"] >= min_um) & (micro["size_um"] <= max_um)).sum())
            micro_range_lines.append(f"{range_name}: {min_um:g}–{max_um:g} µm, n = {n}")

        if not micro_range_lines:
            micro_range_lines.append("No microplastic size ranges selected.")

        return (
            f"Selected/calculated u*: {selected_u_star():.4f} m/s\n"
            f"Flow depth H: {float(input.H()):.2f} m\n"
            f"Bed reference offset a_bed/H: {float(input.a_bed_frac()):.2f}\n"
            f"Surface reference offset a_surf/H: {float(input.a_surf_frac()):.2f}\n\n"
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

