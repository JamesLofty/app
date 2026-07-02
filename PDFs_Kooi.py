import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norminvgauss
import pandas as pd


# ============================================================
# REFERENCES
#
# Kooi & Koelmans (2019)
#
# Size:
# power law exponent alpha = 1.6
# 20 um < d < 5000 um
#
# Shape:
# Gaussian mixture:
# lambda1 = 0.059
# lambda2 = 0.941
# mu1 = 0.076
# mu2 = 0.441
# sigma1 = 0.030
# sigma2 = 0.189
#
# Density:
# Normal Inverse Gaussian:
# alpha = 75.13
# beta  = 71.30
# delta = 0.097
# mu    = 0.839
#
# ============================================================

# ============================================================
# FLUID PROPERTIES
# ============================================================
rho_water = 1000.0      # kg/m3
mu_water = 0.001        # Pa s
nu_water = mu_water / rho_water

g = 9.81
# ============================================================
# MONTE CARLO SETTINGS
# ============================================================
N = 10000

# %%
# ============================================================
# 1. SIZE DISTRIBUTION
# p(d) ~ d^-1.6
#
# Paper:
# 20 um -> 5000 um
# ============================================================
alpha_size = 1.6

d_min = 2e-5      
d_max = 5000e-6    

u = np.random.rand(N)

sizes = (
    (
        u * (d_max**(1-alpha_size) - d_min**(1-alpha_size))
        + d_min**(1-alpha_size)
    )
    ** (1/(1-alpha_size))   
)
# ============================================================
fig, ax = plt.subplots()

ax.hist(
    sizes,
    density=True,
    alpha=0.75,
    edgecolor='black'
)
# ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Particle size [m]')
ax.set_ylabel('Probability density')
ax.set_title(r'Power-law size distribution: $p(d)\propto d^{-1.6}$')
ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 2. SHAPE DISTRIBUTION (CSF)
#
# bimodal Gaussian mixture
# ============================================================

lambda1 = 0.059
lambda2 = 0.941

mu1 = 0.076
mu2 = 0.441

sigma1 = 0.030
sigma2 = 0.189

choice = np.random.rand(N)

csf = np.zeros(N)

mask1 = choice < lambda1
mask2 = ~mask1

csf[mask1] = np.random.normal(mu1, sigma1, mask1.sum())
csf[mask2] = np.random.normal(mu2, sigma2, mask2.sum())

csf = np.clip(csf, 0.01, 1.0)
# ============================================================

fig, ax = plt.subplots()

ax.hist(
    csf,
    bins=80,
    density=True,
    alpha=0.7,
    edgecolor='black',
    label='Monte Carlo samples'
)
x = np.linspace(0.01, 1.0, 1000)
pdf1 = (
    lambda1
    * (1 / (sigma1 * np.sqrt(2*np.pi)))
    * np.exp(-0.5 * ((x - mu1)/sigma1)**2)
)
pdf2 = (
    lambda2
    * (1 / (sigma2 * np.sqrt(2*np.pi)))
    * np.exp(-0.5 * ((x - mu2)/sigma2)**2)
)
pdf_total = pdf1 + pdf2
ax.plot(
    x,
    pdf_total,
    linewidth=3,
    label='Mixture PDF'
)
# optional: show individual modes
ax.plot(x, pdf1, '--', linewidth=2, label='Mode 1')
ax.plot(x, pdf2, '--', linewidth=2, label='Mode 2')

ax.set_xlabel('Corey Shape Factor (CSF)')
ax.set_ylabel('Probability density')

ax.set_title('Bimodal Gaussian mixture for particle shape')

ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# ============================================================
# 3. DENSITY DISTRIBUTION (FITTED NIG MODEL)
# PE   : 25.0%
# PET  : 16.5%
# PA   : 12.0%
# PP   : 14.0%
# PS   :  8.5%
# PVA  :  6.0%
# PVC  :  2.0%
#
# Reported polymer density ranges [g/cm3]:
#
# PE   : 0.89 – 0.98
# PET  : 0.96 – 1.45
# PA   : 1.02 – 1.16
# PP   : 0.83 – 0.92
# PS   : 1.04 – 1.10
# PVA  : 1.19 – 1.31
# PVC  : 1.10 – 1.58
#
# BIOFULING 15% added to those less than 1g/cm3 ???
# ============================================================
# Paper NIG parameters in g/cm3
alpha = 75.1
beta = 71.3
mu = 0.84
delta = 0.097

# SciPy parameter conversion
a = alpha * delta
b = beta * delta

# Sample in g/cm3
rho_particles = norminvgauss.rvs(
    a=a,
    b=b,
    loc=mu,
    scale=delta,
    size=N,
    random_state=1
)

# Convert g/cm3 to kg/m3
rho_particles *= 1000.0

fig, ax = plt.subplots()

ax.hist(
    rho_particles,
    bins=80,
    density=True,
    alpha=0.7,
    edgecolor='black',
    label='Monte Carlo samples'
)

x = np.linspace(
    rho_particles.min(),
    rho_particles.max(),
    1000
)

# Convert x back to g/cm3 for the PDF evaluation
pdf = norminvgauss.pdf(
    x / 1000.0,
    a=a,
    b=b,
    loc=mu,
    scale=delta
)

# Unit conversion: PDF from g/cm3 to kg/m3
pdf /= 1000.0

ax.plot(
    x,
    pdf,
    linewidth=3,
    label='NIG PDF'
)

ax.set_xlabel(r'Particle density [kg m$^{-3}$]')
ax.set_ylabel('Probability density')
ax.set_title('Particle density distribution (NIG model)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()



# %%
# ============================================================
# 4. SPLIT RISING / SETTLING
# ============================================================
settling_mask = rho_particles > rho_water
rising_mask = rho_particles < rho_water

print("Settling fraction:",
      settling_mask.mean())

print("Rising fraction:",
      rising_mask.mean())


# classify particle behavior
behavior = np.where(
    rho_particles > rho_water,
    'settling',
    'rising'
)

# build dataframe
df = pd.DataFrame({
    'size': sizes,
    'CSF': csf,
    'density': rho_particles,
    'settling': behavior
})

# save to Excel
output_file = 'microplastic_particles.csv'

df.to_csv(
    output_file,
    index=False
)

print(f'Data saved to: {output_file}')

