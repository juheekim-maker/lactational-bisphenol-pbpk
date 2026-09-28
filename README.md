# Lactational bisphenol PBPK model (BPA / BPF / BPS, 0–6 months)

Ontogeny-informed, reduced-form lactational physiologically based pharmacokinetic (PBPK)
model that converts **measured breast-milk concentrations and measured infant intake**
into an **internal free-fraction dose** and an **ontogeny amplification factor** for
breastfed infants aged 0–6 months.

Accompanies: Kim JH, Park S, Ji K, Im H. *Lactational PBPK modeling of bisphenol A and its
analogues: ontogeny-informed estimation of internal exposure in breastfed Korean infants.*
Food and Chemical Toxicology (submitted). Equation numbers in the code refer to
Section 2.4 of the manuscript and to Supplementary Text S1.

---

## 1. Installation

```bash
git clone https://github.com/juheekim-maker/lactational-bisphenol-pbpk.git
cd lactational-bisphenol-pbpk
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # numpy, pandas, scipy, SALib, matplotlib
```

Requires Python ≥ 3.10. Verify the installation (this reproduces Supplementary Table S2):

```bash
python lactational_bisphenol_pbpk.py
```

Expected output:

```
BPA: AF(1-6 mo) = [6.132, 5.0, 3.925, 3.125, 2.572, 2.193];  adult systemic free availability = 4.8%
BPF: AF(1-6 mo) = [6.132, 5.0, 3.925, 3.125, 2.572, 2.193];  adult systemic free availability = 7.2%
BPS: AF(1-6 mo) = [6.132, 5.0, 3.925, 3.125, 2.572, 2.193];  adult systemic free availability = 12.8%
```

---

## 2. Two-minute tutorial

### 2.1 One infant

```python
from lactational_bisphenol_pbpk import Model

m = Model(compound="BPA")

m.amplification_factor(age_months=1)          # 6.132  — Eq. 8, AF = 1/MF(t)
m.mf(age_months=1)                            # 0.163  — Eq. 3, UGT maturation fraction
m.external_dose(c_milk=0.045, intake=100)     # ng/kg/day, from measured milk data
m.internal_free_dose(c_milk=0.045, intake=100, age_months=1)   # Eq. 7
```

### 2.2 A whole cohort

Supply four columns: total milk concentration (ng/mL), body-weight-normalised
intake (mL/kg/day), infant age (months), and — if you want absolute intakes —
body weight (kg).

```python
import pandas as pd
from lactational_bisphenol_pbpk import Model

df  = pd.read_csv("data/synthetic_cohort.csv")
out = Model("BPA").apply_cohort(df.c_milk_BPA, df.intake_mL_kg_day, df.age_months)
df  = df.assign(**out)

df.groupby("age_months")[["external_EDI", "IFD", "AF"]].median()
```

### 2.3 Sensitivity and uncertainty

```python
m = Model("BPA")
m.sensitivity_oat(df.c_milk_BPA, df.intake_mL_kg_day, df.age_months)   # Table S3
mc = m.monte_carlo(df.c_milk_BPA, df.intake_mL_kg_day, df.age_months,
                   n_draws=20000, seed=20250101)                        # Table S4
mc["AF_q"]            # rows = 2.5th, 50th, 97.5th percentile; columns = age
```

### 2.4 Reproduce every figure and table in the paper

```bash
python scripts/run_all.py --data data/synthetic_cohort.csv --out results/
```

writes `Figure2.png` … `Figure6.png` and `TableS2.csv` … `TableS5.csv` to `results/`.
Substitute your own cohort file with the same column names to apply the model to
another population.

---

## 3. Re-parameterising the model

All parameters live in the frozen `Params` dataclass and can be overridden without
touching the source:

```python
from dataclasses import replace
from lactational_bisphenol_pbpk import Model, DEFAULT_PARAMS

# a slower conjugation ontogeny, and a higher assumed milk free fraction
p = replace(DEFAULT_PARAMS, mf0=0.08, tm50=12.0, f_free=0.55)
m = Model("BPA", params=p)
m.amplification_factor(1)
```

To add a new bisphenol analogue, add one entry to `COMPOUNDS` with its absorbed
fraction and adult hepatic extraction ratio:

```python
from lactational_bisphenol_pbpk import COMPOUNDS, Model
COMPOUNDS["BPAF"] = {"f_abs": 0.88, "E_h_ad": 0.90, "cl_rel": 0.55}
Model("BPAF").adult_systemic_free_availability()
```

Because `AF(t) = 1/MF(t)` (Eq. 8), the amplification factor is unchanged by
compound-specific disposition; only the internal free-dose index changes.

---

## 4. Data

Individual-level LIFE-EPIC and Ko-CHENS cohort data cannot be redistributed under the
governing ethical approvals (IRB KHSIRB-21-598(NA); KHSIRB-25-114-1(EA)).
`data/synthetic_cohort.csv` is a **fully synthetic** dataset of 189 records with the same
column names and variable definitions as the analysis set, and marginal distributions
matched to the published summary statistics (manuscript Table 1). It contains no real
participant data. The entire pipeline can therefore be executed and verified end-to-end
without the restricted data. Note that figures generated from it have the same structure
as the published ones but not identical values; Supplementary Table S2 (amplification
factors) *is* reproduced exactly, because AF depends only on postnatal age (Eq. 8).
Regenerate the synthetic file with `python scripts/generate_synthetic_data.py`.
Real data are available from the corresponding author on reasonable request and subject
to the relevant approvals.

## 5. Repository contents

```
lactational_bisphenol_pbpk.py     the model (every function annotated by equation number)
requirements.txt                  dependencies
README.md                         this tutorial
LICENSE                           MIT
CITATION.cff                      citation metadata
data/synthetic_cohort.csv         synthetic 189-record demonstration dataset
scripts/generate_synthetic_data.py  regenerates the synthetic dataset
scripts/run_all.py                regenerates all figures and supplementary tables
results/                          example output of scripts/run_all.py
```

## 6. Model structure in one screen

| Tier | Compartments | Key equations |
|---|---|---|
| 2 — lactational | measured milk → free / conjugated split | Eq. 1 |
| 3 — infant gut & liver | gut lumen, gut wall/portal, liver (UGT ontogeny) | Eqs. 2–5 |
| 1 — core systemic | plasma, richly perfused (lumped), slowly perfused (lumped), kidney→urine, feces | Eqs. 6–8 |

Distribution tissues are deliberately lumped: the reported outputs are steady-state
free-dose quantities, which are independent of tissue distribution volumes
(see manuscript Section 2.3).

## 7. Citation

```bibtex
@article{kim2026lactational,
  title  = {Lactational PBPK modeling of bisphenol A and its analogues: ontogeny-informed
            estimation of internal exposure in breastfed Korean infants},
  author = {Kim, Ju Hee and Park, Seungyoung and Ji, Kyunghee and Im, Hosub},
  journal= {Food and Chemical Toxicology},
  year   = {2026}
}
```

Archived release: [doi:10.5281/zenodo.XXXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXXX)

## 8. License

MIT (code). Documentation CC-BY-4.0.
