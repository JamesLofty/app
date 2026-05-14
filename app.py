# -*- coding: utf-8 -*-
"""
Shiny for Python app: combined microplastic + macroplastic Rouse number CDF plot

Run with:
    shiny run --reload app.py

Required files in the same folder as this app:
    microplastic_particles_settling.csv
    macroplastic_particles_settling.xlsx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shiny import App, Inputs, Outputs, Session, render, ui
from shiny.types import SafeException


# ============================================================
# LOAD DATA ONCE AT APP STARTUP
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

micro_size_bins = [0, 400e-6, 1000e-6, 5000e-6, np.inf]
micro_size_labels = ["< 400 µm", "400–1000 µm", "1–5 mm", ">5 mm"]

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

macro_order = [
    "buoyant",
    "near_neutral",
    "dense",
]

macro_legend_labels = {
    "buoyant": "Buoyant (foams)\n($\\rho < 0.8$ g cm$^{-3}$)",
    "near_neutral": "Near neutral (plastics & others)\n($0.8 \\leq \\rho \\leq 1.5$ g cm$^{-3}$)",
    "dense": "Dense (Glass & metal)\n$\\rho > 2.5$ g cm$^{-3}$",
}


# ============================================================
# PREPARE STATIC COLUMNS ONCE
# ============================================================
micro["size_class"] = pd.cut(
    micro["size"],
    bins=micro_size_bins,
    labels=micro_size_labels,
    include_lowest=True,
)

macro["Material_grouped"] = macro["Material"].map(macro_mapping)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted x and empirical cumulative probability y."""
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.array([]), np.array([])

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def calculate_micro_rouse_mean(u_star: float) -> np.ndarray:
    """
    Calculate mean microplastic Rouse number across the three velocity equations.

    Original script:
        each velocity equation produced a particle x Monte-Carlo matrix;
        then the three equations were averaged.

    Shiny version:
        one selected u_star value is used;
        result is one Rouse value per particle.
    """
    rouse_arrays = []

    for col in velocity_cols:
        w = micro[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        beta = w / (kappa * u_star)
        rouse_arrays.append(beta)

    return np.nanmean(np.vstack(rouse_arrays), axis=0)


def calculate_macro_rouse(u_star: float) -> np.ndarray:
    """
    Calculate macroplastic Rouse number.

    Original script used:
        macro_w = macro["vz_mean"] / 100

    That conversion is preserved here.
    """
    macro_w = (macro["vz_mean"] / 100).to_numpy(dtype=float)
    return macro_w / (kappa * u_star)

def calculate_shear_velocity_from_slope_radius(hydraulic_radius: float, slope: float) -> float:
    """
    Common open-channel estimate:
        u* = sqrt(g R S)

    For a wide channel, hydraulic radius R can be approximated by water depth h.
    """
    return float(np.sqrt(g * hydraulic_radius * slope))

def classify_transport_regimes(beta: np.ndarray) -> pd.Series:
    """Classify Rouse number values into simplified transport regimes."""
    beta = beta[np.isfinite(beta)]

    regimes = pd.cut(
        beta,
        bins=[-np.inf, -2.5, 2.5, np.inf],
        labels=[
            "Surface load",
            "Suspended load",
            "Bed load",
        ],
        include_lowest=True,
    )

    return pd.Series(regimes)


def regime_fraction_table(u_star: float) -> pd.DataFrame:
    """Return percentage of particle groups in each transport regime."""

    micro_rouse_mean = calculate_micro_rouse_mean(u_star)
    macro_rouse = calculate_macro_rouse(u_star)

    regime_order = [
        "Surface load",
        "Suspended load",
        "Bed load",
    ]

    rows = []

    # --------------------------------------------------
    # MICROPLASTICS BY SIZE CLASS
    # --------------------------------------------------
    for label, group in micro.groupby("size_class", observed=True):
        idx = group.index.to_numpy()
        values = micro_rouse_mean[idx]

        regimes = classify_transport_regimes(values)

        fractions = (
            regimes.value_counts(normalize=True)
            .reindex(regime_order, fill_value=0)
            * 100
        )

        row = {
            "Group": f"Micro: {label}",
        }

        for regime in regime_order:
            row[regime] = round(float(fractions.loc[regime]), 1)

        rows.append(row)

    # --------------------------------------------------
    # MACROPLASTICS BY BUOYANCY CLASS
    # --------------------------------------------------
    for label in macro_order:
        idx = macro.index[macro["Material_grouped"] == label].to_numpy()
        values = macro_rouse[idx]

        regimes = classify_transport_regimes(values)

        fractions = (
            regimes.value_counts(normalize=True)
            .reindex(regime_order, fill_value=0)
            * 100
        )

        pretty_name = {
            "buoyant": "Macro: buoyant",
            "near_neutral": "Macro: near neutral",
            "dense": "Macro: dense",
        }[label]

        row = {
            "Group": pretty_name,
        }

        for regime in regime_order:
            row[regime] = round(float(fractions.loc[regime]), 1)

        rows.append(row)

    return pd.DataFrame(rows)


def regime_fraction_wide_table(u_star: float) -> pd.DataFrame:
    """Return formatted regime-fraction table for app display."""
    return regime_fraction_table(u_star)
    


def make_rouse_cdf_plot(u_star: float) -> plt.Figure:
    """Build the combined microplastic + macroplastic CDF plot."""
    micro_rouse_mean = calculate_micro_rouse_mean(u_star)
    macro_rouse = calculate_macro_rouse(u_star)

    fig, ax = plt.subplots(figsize=(10, 6))
    tab10 = plt.get_cmap("tab10").colors

    # Transport-regime background shading
    ax.axvspan(-100, -2.5, color=tab10[0], alpha=0.12)
    ax.axvspan(-2.5, -0.08, color=tab10[2], alpha=0.12)
    ax.axvspan(-0.08, 0.08, color=tab10[3], alpha=0.15)
    ax.axvspan(0.08, 2.5, color=tab10[4], alpha=0.12)
    ax.axvspan(2.5, 100, color=tab10[1], alpha=0.12)

    # ----------------------------
    # MICRO PARTICLES
    # ----------------------------
    micro_colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(micro_size_labels)))
    micro_lines = []
    micro_labels = []

    # observed=True prevents empty category groups from being plotted
    for i, (label, group) in enumerate(micro.groupby("size_class", observed=True)):
        idx = group.index.to_numpy()
        values = micro_rouse_mean[idx]

        # Match your plotting window, otherwise extreme values dominate symlog display
        values = values[(values >= -100) & (values <= 100)]

        x, y = empirical_cdf(values)
        if len(x) == 0:
            continue

        line, = ax.plot(
            x,
            y,
            linewidth=2.5,
            linestyle="-",
            color=micro_colors[i],
        )

        micro_lines.append(line)
        micro_labels.append(str(label))

    # ----------------------------
    # MACRO PARTICLES
    # ----------------------------
    macro_colors = plt.cm.Reds(np.linspace(0.45, 0.9, len(macro_order)))
    macro_lines = []
    macro_labels = []

    for i, label in enumerate(macro_order):
        idx = macro.index[macro["Material_grouped"] == label].to_numpy()
        values = macro_rouse[idx]
        values = values[np.isfinite(values)]
        values = values[(values >= -100) & (values <= 100)]

        x, y = empirical_cdf(values)
        if len(x) == 0:
            continue

        line, = ax.plot(
            x,
            y,
            linewidth=2.5,
            linestyle="-",
            color=macro_colors[i],
        )

        macro_lines.append(line)
        macro_labels.append(macro_legend_labels[label])

    # ----------------------------
    # AXES + LABELS
    # ----------------------------
    ax.axvline(-0.08, color="black", linestyle="--", linewidth=1.5, alpha=0)
    ax.axvline(0.08, color="black", linestyle="--", linewidth=1.5, alpha=0)
    ax.axvline(-2.5, color="black", linestyle="", linewidth=1.5, alpha=0)
    ax.axvline(2.5, color="black", linestyle=":", linewidth=1.5, alpha=0)

    ax.set_xlabel(r"$β$ (-)")
    ax.set_ylabel("Cumulative probability (-)")
    ax.set_xscale("symlog")
    ax.set_xlim(-100, 100)
    ax.set_ylim(0, 1)

    ax.set_title(rf"Combined Rouse number CDF, $u_*$ = {u_star:.3f} m s$^{{-1}}$")

    # Transport-regime labels
    ax.text(-10, 0.05, "Surface load ($β < -2.5$)", ha="center")
    ax.text(-1.1, 0.05, "Rising suspended ($-2.5 < β < -0.08$)", ha="center")
    ax.text(-0.6, 0.2, "Uniform ($β < 0.08$)", ha="center")
    ax.text(1.2, 0.05, "Settling suspended ($0.08 < β < 2.5$)", ha="center")
    ax.text(10, 0.05, "Bed load ($β > 2.5$)", ha="center")

    # Separate legends
    legend_micro = ax.legend(
        micro_lines,
        micro_labels,
        title="Microplastics (< 5000 µm)",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )

    legend_macro = ax.legend(
        macro_lines,
        macro_labels,
        title="Macroplastics (> 5000 µm)",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.80),
    )

    ax.add_artist(legend_micro)

    fig.tight_layout()
    return fig



# ============================================================
# SHINY UI
# ============================================================
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Controls"),

        ui.input_radio_buttons(
            "ustar_mode",
            "How should shear velocity be set?",
            choices={
                "direct": "Use exact u* value",
                "hydraulic": "Calculate from slope + hydraulic radius/depth",
            },
            selected="direct",
        ),

        ui.output_ui("ustar_controls"),

        ui.hr(),

        ui.markdown(
            """
            **Notes**

            - Direct mode uses the selected shear velocity `u*`.
            - Hydraulic mode calculates `u* = sqrt(g R S)`.
            - `R` is hydraulic radius.
            - For a wide channel, `R ≈ h`, so water depth can be used.
            - Moving the controls recalculates all Rouse numbers.
            """
        ),
    ),

    ui.h2("Combined microplastic + macroplastic Rouse number CDF"),

    ui.output_plot("rouse_plot", height="650px"),

    ui.card(
        ui.card_header("Transport fractions by particle group"),
        ui.output_data_frame("regime_table"),
    ),

    ui.card(
        ui.card_header("Dataset summary"),
        ui.output_text_verbatim("summary_text"),
    ),
)


# ============================================================
# SHINY SERVER
# ============================================================
def server(input: Inputs, output: Outputs, session: Session):

    def selected_u_star() -> float:
        """Return either direct u* or hydraulically calculated u*."""
        if input.ustar_mode() == "direct":
            return float(input.u_star())

        return calculate_shear_velocity_from_slope_radius(
            hydraulic_radius=float(input.hydraulic_radius()),
            slope=float(input.slope()),
        )

    @render.ui
    def ustar_controls():
        """Show the correct controls depending on selected mode."""
        if input.ustar_mode() == "direct":
            return ui.TagList(
                ui.input_slider(
                    "u_star",
                    "Shear velocity u* (m/s)",
                    min=0.01,
                    max=0.30,
                    value=0.15,
                    step=0.001,
                )
            )

        return ui.TagList(
            ui.input_slider(
                "hydraulic_radius",
                "Hydraulic radius R, or water depth h for wide channels (m)",
                min=0.01,
                max=5.00,
                value=0.50,
                step=0.01,
            ),
            ui.input_slider(
                "slope",
                "Energy/water-surface slope S (-)",
                min=0.00001,
                max=0.02000,
                value=0.00100,
                step=0.00001,
            ),
            ui.output_text("calculated_ustar"),
        )

    @render.text
    def calculated_ustar():
        """Display hydraulically calculated shear velocity."""
        u_star = calculate_shear_velocity_from_slope_radius(
            hydraulic_radius=float(input.hydraulic_radius()),
            slope=float(input.slope()),
        )

        return f"Calculated u*: {u_star:.4f} m/s"

    @render.plot(alt="Combined microplastic and macroplastic Rouse number CDF plot")
    def rouse_plot():
        return make_rouse_cdf_plot(selected_u_star())

    @render.data_frame
    def regime_table():
        table = regime_fraction_wide_table(selected_u_star())

        return render.DataGrid(
            table,
            width="100%",
            height="320px",
            filters=False,
            summary=False,
        )

    @render.text
    def summary_text():
        micro_counts = micro["size_class"].value_counts().sort_index()
        macro_counts = macro["Material_grouped"].value_counts(dropna=False)

        return (
            f"Selected/calculated u*: {selected_u_star():.4f} m/s\n\n"
            "Micro particle size-class counts:\n"
            f"{micro_counts.to_string()}\n\n"
            "Macro material-group counts:\n"
            f"{macro_counts.to_string()}"
        )


app = App(app_ui, server)
