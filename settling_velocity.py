# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:19:08 2026

@author: Lofty
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('microplastic_particles.csv')

# ============================================================
# CONSTANTS
# ============================================================
rho_water = 1000.0        # kg/m3
nu = 1.004e-6      # m2/s, water at ~20 C
g = 9.81                 # m/s2

data["particle_type"] = np.where(data["CSF"] <= 0.2, "fiber", "fragment")


# %%
# # ============================================================
# # Waldschläger & Schüttrumpf (2019)
# # Fragment drag:
# #   Cd = 3 / (CSF * sqrt(Re^3))
# #
# # Fibre drag:
# #   Cd = 4.7 / sqrt(Re) + sqrt(CSF)
# # ============================================================
# def compute_velocity_waldschlaeger(d, rho_p, csf, particle_type):
#     d = np.asarray(d, dtype=float)
#     rho_p = np.asarray(rho_p, dtype=float)
#     csf = np.asarray(csf, dtype=float)
#     particle_type = np.asarray(particle_type)

#     # Avoid pathological CSF values
#     csf = np.clip(csf, 1e-6, 1.0)

#     delta_rho = np.abs(rho_p - rho_water)
#     sign = np.where(rho_p >= rho_water, 1.0, -1.0)

#     velocity_waldschlaeger = np.zeros_like(d)

#     for i in range(len(d)):

#         if d[i] <= 0 or delta_rho[i] == 0:
#             velocity_waldschlaeger[i] = 0.0
#             continue

#         # Initial Stokes velocity magnitude
#         w = delta_rho[i] * g * d[i]**2 / (18.0 * rho_water * nu)

#         for _ in range(100):

#             # Reynolds number is unsigned
#             Re = max(abs(w) * d[i] / nu, 1e-8)

#             if particle_type[i] == "fiber":
#                 Cd_empirical = 4.7 / np.sqrt(Re) + np.sqrt(csf[i])

#             else:
#                 # Waldschläger fragment correlation
#                 Cd_empirical = 3.0 / (csf[i] * Re**1.5)

#                 # Low-Re protection: prevent unphysical blow-up
#                 Cd_stokes = 24.0 / Re

#                 # Use the larger drag coefficient
#                 Cd_empirical = max(Cd_empirical, Cd_stokes)

#             w_new = np.sqrt(
#                 (4.0 * delta_rho[i] * g * d[i])
#                 / (3.0 * rho_water * Cd_empirical)
#             )

#             # Under-relaxation for numerical stability
#             w_relaxed = 0.5 * w + 0.5 * w_new

#             if abs(w_relaxed - w) < 1e-12:
#                 w = w_relaxed
#                 break

#             w = w_relaxed

#         velocity_waldschlaeger[i] = sign[i] * w

#     return velocity_waldschlaeger


# data["velocity_waldschlaeger"] = compute_velocity_waldschlaeger(
#     d=data["size"].to_numpy(),
#     rho_p=data["density"].to_numpy(),
#     csf=data["CSF"].to_numpy(),
#     particle_type=data["particle_type"].to_numpy()
# )

# %%
# ============================================================
# Goral et al. (2023)
#Fragment drag:
#   Cd = (1 + 3.2/sqrt(Re) + 32/Re)
#        * min(0.44 * psi^-2 , 1)
# Fibre drag:
#   Cd = max(19 * Re^-0.6 , 0.86)

# ============================================================
def compute_velocity_goral(d, rho_p, csf, particle_type):
    d = np.asarray(d, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    csf = np.asarray(csf, dtype=float)
    particle_type = np.asarray(particle_type)

    csf = np.clip(csf, 1e-6, 1.0)

    delta_rho = np.abs(rho_p - rho_water)
    sign = np.where(rho_p >= rho_water, 1.0, -1.0)
    velocity_goral = np.zeros_like(d)

    for i in range(len(d)):
        # psi ~ CSF
        psi = csf[i]
        # Initial Stokes guess
        w = delta_rho[i] * g * d[i]**2 / (18 * rho_water * nu)

        for _ in range(100):
            Re = max(w * d[i] / nu, 1e-12)
            
            # print(Re)
            
            if particle_type[i] == "fiber":
                Cd = max(19 * Re**(-0.6),0.86)
            else:
                Cd = (1+ 3.2 / np.sqrt(Re)+ 32 / Re) * min( 0.44 * psi**(-2),1)

            w_new = np.sqrt(
                (4.0* delta_rho[i]* g* d[i])
                /
                (3.0* rho_water* Cd)
            )

            if abs(w_new - w) < 1e-12:
                break
            w = w_new
        velocity_goral[i] = sign[i] * w
    return velocity_goral


data["velocity_goral"] = compute_velocity_goral(
    d=data["size"].to_numpy(),
    rho_p=data["density"].to_numpy(),
    csf=data["CSF"].to_numpy(),
    particle_type=data["particle_type"].to_numpy()
)
# %%
# ============================================================
# Yu et al. (2022)
#
# 𝑑∗=𝑑𝑒𝑞∙((𝜌𝑝−𝜌𝑓)∙𝑔𝜌𝑓∙𝜈2)1/3
# 𝐶𝑑,𝑠=432𝑑∗3(1+0.022𝑑∗3)0.54+0.47∙(1−exp (−0.15𝑑∗0.45))
# 𝐶𝑑=𝐶𝑑,𝑠(𝑑∗𝛽1𝜙𝑑∗𝛽2𝐶𝑆𝐹𝑑∗𝛽3)𝛽4
# 𝛽1=−0.25, 𝛽2=0.03, 𝛽3=0.33 and 𝛽4=0.25.
# 𝑤𝑠=(𝜈𝑔𝜌𝑝−𝜌𝑓𝜌𝑓)1/3∙√4𝑑∗3𝐶𝑑
# ============================================================
def compute_velocity_yu(d, rho_p, csf):

    d = np.asarray(d, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    csf = np.clip(np.asarray(csf, dtype=float), 1e-6, 1.0)

    phi = csf

    delta_rho = np.abs(rho_p - rho_water)

    sign = np.where(rho_p >= rho_water, 1.0, -1.0)

    d_star = d * ((delta_rho * g)/(rho_water * nu**2))**(1/3)

    Cd_s = (
        432 / d_star**3
        * (1 + 0.022 * d_star**3)**0.54
        + 0.47 * (1 - np.exp(-0.15 * d_star**0.45))
    )

    Cd = Cd_s / ( phi* csf* d_star**(-0.25 + 0.03 + 0.33))**0.25

    w = ((nu * g * (delta_rho / rho_water))**(1/3)* np.sqrt((4 * d_star) / (3 * Cd)))

    return sign * w


data["velocity_yu"] = compute_velocity_yu(
    d=data["size"].to_numpy(),
    rho_p=data["density"].to_numpy(),
    csf=data["CSF"].to_numpy()
)

# %%

# ============================================================
# Dietrich (1982)
#
# W* = R3 * 10^(R1 + R2)
# W* = ws^3 / (R * g * nu)
# D* = R * g * d^3 / nu^2
#
# R = abs(rho_p / rho_water - 1)
# Uses CSF as Corey shape factor.
# Uses Powers roundness index P, default 3.5.
# ============================================================

def compute_velocity_dietrich(d, rho_p, csf, powers_roundness=3.5):

    d = np.asarray(d, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    csf = np.asarray(csf, dtype=float)

    csf = np.clip(csf, 1e-6, 1.0)

    delta_rho = np.abs(rho_p - rho_water)
    sign = np.where(rho_p >= rho_water, 1.0, -1.0)

    velocity_dietrich = np.zeros_like(d)

    valid = (d > 0) & (delta_rho > 0)

    D_star = np.zeros_like(d)
    D_star[valid] = (
        delta_rho[valid] * g * d[valid]**3
        / (rho_water * nu**2)
    )

    logD = np.zeros_like(d)
    logD[valid] = np.log10(D_star[valid])

    R1 = (
        -3.76715
        + 1.92944 * logD
        - 0.09815 * logD**2
        - 0.00575 * logD**3
        + 0.00056 * logD**4
    )

    R2 = (
        np.log10(1.0 - ((1.0 - csf) / 0.85))
        - (1.0 - csf)**2.3 * np.tanh(logD - 4.6)
        + 0.3 * (0.5 - csf) * (1.0 - csf)**2 * (logD - 4.6)
    )

    P = powers_roundness

    R3 = (
        0.65
        - ((csf / 2.83) * np.tanh(logD - 4.6))
    ) ** (1.0 + ((3.5 - P) / 2.5))

    W_star = R3 * 10.0**(R1 + R2)

    velocity_dietrich[valid] = (
        W_star[valid]
        * delta_rho[valid]
        * g
        * nu
        / rho_water
    ) ** (1.0 / 3.0)

    return sign * velocity_dietrich


data["velocity_dietrich"] = compute_velocity_dietrich(
    d=data["size"].to_numpy(),
    rho_p=data["density"].to_numpy(),
    csf=data["CSF"].to_numpy(),
    powers_roundness=3.5
)


# %%
fig, ax = plt.subplots()

bins = np.geomspace(1e-6,
    np.max(np.abs([
        data["velocity_goral"],
        data["velocity_yu"]
    ])),120
)

# ax.hist(
#     data["velocity_waldschlaeger"],
#     bins=bins,
#     density=True,
#     histtype="step",
#     linewidth=2,
#     label="Waldschläger"
# )

ax.hist(
    data["velocity_dietrich"],
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Dietrich"
)

ax.hist(
    data["velocity_goral"],
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Goral"
)

ax.hist(
    data["velocity_yu"],
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Yu"
)

# ax.set_xscale("symlog")
ax.set_yscale("log")

ax.set_xlabel("Absolute velocity [m/s]")
ax.set_ylabel("Probability density")

ax.set_title("Velocity distributions")

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

data.to_csv("microplastic_particles_settling.csv", index=False)


