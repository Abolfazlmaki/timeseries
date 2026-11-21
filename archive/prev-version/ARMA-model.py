
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess

# =========================
# ARIMA Parameters
# =========================
p = 1
d_model = 0
q = 0

# Number of differencing operations for visualization
d_diff = 0

n = 2000
burn_in = 0
seed = 42

ar_coefficients = [0.2]
ma_coefficients = []

noise_mean = 0.0
noise_variance = 1.0
drift = 0.15

# =========================
# Validation
# =========================
if p != len(ar_coefficients):
    raise ValueError("AR coefficient count must equal p.")

if q != len(ma_coefficients):
    raise ValueError("MA coefficient count must equal q.")

if min(p, d_model, q, d_diff) < 0:
    raise ValueError("All orders must be non-negative.")

# =========================
# Generate ARMA Process
# =========================
rng = np.random.default_rng(seed)

ar = np.r_[1.0, -np.array(ar_coefficients)]
ma = np.r_[1.0, np.array(ma_coefficients)]

arma_process = ArmaProcess(ar, ma)

if not arma_process.isstationary:
    raise ValueError("AR coefficients must define a stationary process.")

def generate_noise(size):
    return rng.normal(
        noise_mean,
        np.sqrt(noise_variance),
        size
    )

arma_values = arma_process.generate_sample(
    nsample=n + burn_in,
    distrvs=generate_noise,
    burnin=burn_in
)

arma_values += drift

# =========================
# Generate ARIMA Series
# =========================
series = arma_values.copy()

for _ in range(d_model):
    series = np.cumsum(series)

series = series[-n:]

# =========================
# Apply Independent Differencing
# =========================
diff_series = series.copy()

for _ in range(d_diff):
    diff_series = np.diff(diff_series)

# =========================
# ACF Calculation and Table
# =========================
# Number of lags (k) to analyze
max_lags = 16

# Calculate the autocorrelation values for the time series
acf_values = acf(series, nlags=max_lags, fft=False)

# Create a table of ACF values for each lag k
acf_table = pd.DataFrame({
    'Lag (k)': range(max_lags + 1),
    'ACF Value': acf_values
})

print("\n--- Autocorrelation Function (ACF) Table ---")
print(acf_table.to_string(index=False))

# =========================
# ACF Plot
# =========================
fig_acf, ax_acf = plt.subplots(figsize=(10, 5))

# Plot the autocorrelation function
plot_acf(series, lags=max_lags, ax=ax_acf, alpha=0.05)

ax_acf.set_title("Autocorrelation Function (ACF)")
ax_acf.set_xlabel("Lag (k)")
ax_acf.set_ylabel("Autocorrelation")
ax_acf.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('acf_plot.png')
# =========================
# Plot
# =========================
fig, axes = plt.subplots(2, 1, figsize=(13, 8))

axes[0].plot(series, linewidth=1)
axes[0].set_title(
    f"Generated ARIMA({p}, {d_model}, {q}) Series"
)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Value")
axes[0].grid(True, alpha=0.3)

axes[1].plot(diff_series, linewidth=1)
axes[1].set_title(
    f"Series After {d_diff} Differencing Operations"
)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Difference")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('arima_plot.png')
