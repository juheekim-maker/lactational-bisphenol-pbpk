"""Reproduce every figure and supplementary table of the manuscript from a
cohort file. Run:

    python scripts/run_all.py --data data/synthetic_cohort.csv --out results/

With the synthetic dataset the figures have the same structure as the published
ones but not identical values, because the underlying cohort data cannot be
redistributed. Supplementary Table S2 (amplification factors) IS reproduced
exactly, because AF depends only on postnatal age (Eq. 8).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lactational_bisphenol_pbpk import Model, COMPOUNDS

AGES = np.arange(1, 7)


def main(data_path: Path, out_dir: Path, n_draws: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    models = {c: Model(c) for c in COMPOUNDS}

    res = {}
    for c, m in models.items():
        res[c] = pd.DataFrame(m.apply_cohort(df[f"c_milk_{c}"],
                                             df.intake_mL_kg_day,
                                             df.age_months))
        res[c]["age_months"] = df.age_months

    # ---- Table S2: AF and median EDI / IFD by age ----------------------
    rows = []
    for a in AGES:
        row = {"age_months": a, "MF": models["BPA"].mf(a),
               "AF": models["BPA"].amplification_factor(a)}
        for c in COMPOUNDS:
            sub = res[c][res[c].age_months == a]
            row[f"median_EDI_{c}"] = sub.external_EDI.median()
            row[f"median_IFD_{c}"] = sub.IFD.median()
            row[f"n_{c}"] = len(sub)
        rows.append(row)
    tS2 = pd.DataFrame(rows).round(4)
    tS2.to_csv(out_dir / "TableS2.csv", index=False)

    # ---- Figure 2: amplification factor by age -------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
    ax.plot(AGES, models["BPA"].amplification_factor(AGES), "o-", color="#B22222")
    ax.set_xlabel("Postnatal age (months)")
    ax.set_ylabel("Amplification factor, AF = 1/MF(t)")
    ax.set_title("Ontogeny amplification of internal free dose")
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "Figure2.png"); plt.close(fig)

    # ---- Figure 3: external EDI vs internal free dose ------------------
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
    sc = ax.scatter(res["BPA"].external_EDI, res["BPA"].IFD,
                    c=res["BPA"].age_months, cmap="viridis", s=22)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("External EDI (ng/kg/day)")
    ax.set_ylabel("Internal free-dose index (relative)")
    ax.set_title("External EDI does not preserve internal-dose ranking (BPA)")
    fig.colorbar(sc, ax=ax, label="Infant age (months)")
    fig.tight_layout(); fig.savefig(out_dir / "Figure3.png"); plt.close(fig)

    # ---- Figure 4: IFD distribution by age -----------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
    ax.boxplot([res["BPA"].IFD[res["BPA"].age_months == a] for a in AGES],
               tick_labels=[str(a) for a in AGES], showfliers=False)
    ax.set_xlabel("Infant age (months)")
    ax.set_ylabel("Internal free-dose index (relative)")
    ax.set_title("Distribution of infant internal free BPA dose by age")
    fig.tight_layout(); fig.savefig(out_dir / "Figure4.png"); plt.close(fig)

    # ---- Figure 5 / Table S3: one-at-a-time sensitivity ----------------
    s = models["BPA"].sensitivity_oat(df.c_milk_BPA, df.intake_mL_kg_day, df.age_months)
    tS3 = (pd.DataFrame(s["parameters"]).T
             .loc[:, ["change_low_pct", "change_high_pct"]].round(2))
    tS3.to_csv(out_dir / "TableS3.csv")
    order = tS3.abs().max(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    y = np.arange(len(order))
    ax.barh(y, tS3.loc[order, "change_low_pct"], color="#4C72B0", label="low end")
    ax.barh(y, tS3.loc[order, "change_high_pct"], color="#C44E52", label="high end")
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Change in cohort-median IFD (%)")
    ax.set_title("One-at-a-time sensitivity (BPA)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "Figure5.png"); plt.close(fig)

    # ---- Figure 6 / Table S4: Monte Carlo propagation ------------------
    mc = models["BPA"].monte_carlo(df.c_milk_BPA, df.intake_mL_kg_day,
                                   df.age_months, n_draws=n_draws)
    lo, med, hi = mc["AF_q"]
    tS4 = pd.DataFrame({"age_months": mc["ages"], "AF_p2.5": lo,
                        "AF_median": med, "AF_p97.5": hi}).round(3)
    tS4.to_csv(out_dir / "TableS4.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200)
    ax.plot(mc["ages"], med, "o-", color="#B22222", label="median")
    ax.fill_between(mc["ages"], lo, hi, alpha=0.25, color="#B22222",
                    label="95% credible interval")
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("Postnatal age (months)"); ax.set_ylabel("Amplification factor")
    ax.set_title(f"Uncertainty propagation ({n_draws:,} draws)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "Figure6.png"); plt.close(fig)

    # ---- Table S5: global sensitivity (Sobol + PRCC) -------------------
    try:
        from SALib.sample import sobol as sobol_sample
        from SALib.analyze import sobol as sobol_analyze
        from dataclasses import replace
        names = ["f_free", "k_deconj", "mf0", "tm50", "E_h_ad"]
        problem = {"num_vars": 5, "names": names,
                   "bounds": [[0.20, 0.60], [0.10, 0.60], [0.05, 0.30],
                              [4.0, 12.0], [0.85, 0.99]]}
        X = sobol_sample.sample(problem, 1024, calc_second_order=False)
        base = models["BPA"].p
        Y = np.array([
            np.median(Model("BPA", replace(base, **dict(zip(names, x))))
                      .internal_free_dose(df.c_milk_BPA, df.intake_mL_kg_day,
                                          df.age_months))
            for x in X])
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                                   print_to_console=False)
        prcc = [pd.Series(X[:, i]).rank().corr(pd.Series(Y).rank())
                for i in range(5)]
        pd.DataFrame({"parameter": names, "S1": Si["S1"].round(3),
                      "ST": Si["ST"].round(3),
                      "rank_corr": np.round(prcc, 3)}
                     ).to_csv(out_dir / "TableS5.csv", index=False)
    except ImportError:
        print("SALib not installed - Table S5 skipped (pip install SALib)")

    print(f"\nWrote figures and tables to {out_dir}/")
    print(tS2[["age_months", "MF", "AF"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_cohort.csv", type=Path)
    ap.add_argument("--out", default="results", type=Path)
    ap.add_argument("--draws", default=20000, type=int)
    a = ap.parse_args()
    main(a.data, a.out, a.draws)
