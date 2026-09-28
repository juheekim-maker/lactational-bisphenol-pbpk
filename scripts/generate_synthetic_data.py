"""Generate a synthetic cohort with the same structure as the LIFE-EPIC analysis
set. NO REAL PARTICIPANT DATA ARE USED OR REPRODUCED HERE.

Marginal distributions are matched approximately to the published summary
statistics in Table 1 of the manuscript (n = 189; infant age 1-6 months, mean
2.8 +/- 1.5; body weight 5.6 +/- 1.6 kg; intake 106.6 +/- 44.1 mL/kg/day,
median 100; milk concentration medians and 95th percentiles per compound).
"""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(20250101)
N = 189

# --- infant age (months), 1-6, mean ~2.8 -------------------------------------
age = np.clip(np.round(rng.gamma(shape=4.0, scale=0.55, size=N) + 0.6), 1, 6).astype(int)

# --- body weight (kg): age-dependent, mean ~5.6, range ~3.0-9.4 --------------
bw = np.clip(3.4 + 0.72 * age + rng.normal(0, 0.55, N), 3.0, 9.4).round(2)

# --- intake (mL/kg/day): median 100, mean ~107, range 7-200 ------------------
intake = np.clip(rng.lognormal(np.log(100), 0.42, N), 7.0, 200.0).round(1)

# --- milk concentrations (ng/mL): lognormal matched to median and p95 --------
def lognorm_from_median_p95(median, p95, n):
    sigma = (np.log(p95) - np.log(median)) / 1.6449
    return rng.lognormal(np.log(median), sigma, n)

milk = {
    "c_milk_BPA": lognorm_from_median_p95(0.045, 0.932, N),
    "c_milk_BPF": lognorm_from_median_p95(0.091, 0.541, N),
    "c_milk_BPS": lognorm_from_median_p95(0.044, 0.749, N),
}

df = pd.DataFrame({
    "id": [f"SYN{i+1:03d}" for i in range(N)],
    "age_months": age,
    "sex": rng.choice(["M", "F"], N, p=[98 / 189, 91 / 189]),
    "body_weight_kg": bw,
    "intake_mL_kg_day": intake,
    **{k: v.round(4) for k, v in milk.items()},
})

out = Path(__file__).resolve().parents[1] / "data" / "synthetic_cohort.csv"
df.to_csv(out, index=False)

print(f"wrote {out}  (n = {len(df)})")
print(df[["age_months", "body_weight_kg", "intake_mL_kg_day",
          "c_milk_BPA", "c_milk_BPF", "c_milk_BPS"]]
      .describe(percentiles=[0.5, 0.95]).round(3).to_string())
