"""
lactational_bisphenol_pbpk
==========================

Ontogeny-informed, reduced-form lactational PBPK model for bisphenol A (BPA)
and its analogues bisphenol F (BPF) and bisphenol S (BPS) in breastfed infants
(0-6 months).

Reference
---------
Kim JH, Park S, Ji K, Im H. Lactational PBPK modeling of bisphenol A and its
analogues: ontogeny-informed estimation of internal exposure in breastfed
Korean infants. Food and Chemical Toxicology (submitted).

Equation numbers below refer to Section 2.4 of the manuscript and to
Supplementary Text S1.

Quick start
-----------
>>> from lactational_bisphenol_pbpk import Model, DEFAULT_PARAMS
>>> m = Model(compound="BPA")
>>> m.amplification_factor(age_months=1)      # Eq. 8
6.132...
>>> m.internal_free_dose(c_milk=0.045, intake=100, age_months=1)   # Eq. 7
...

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Sequence

import numpy as np

__all__ = ["Params", "Model", "DEFAULT_PARAMS", "COMPOUNDS",
           "maturation_factor", "hepatic_extraction"]

# --------------------------------------------------------------------------
# Compound-specific disposition (Karrer et al., 2018, Environ Health Perspect)
# --------------------------------------------------------------------------
COMPOUNDS: Dict[str, Dict[str, float]] = {
    # f_abs   : fraction of free aglycone absorbed from gut lumen
    # E_h_ad  : adult hepatic first-pass extraction ratio
    # cl_rel  : relative intrinsic clearance (BPA = 1.00), reported for reference
    "BPA": {"f_abs": 0.95, "E_h_ad": 0.95, "cl_rel": 1.00},
    "BPF": {"f_abs": 0.90, "E_h_ad": 0.92, "cl_rel": 0.70},
    "BPS": {"f_abs": 0.85, "E_h_ad": 0.85, "cl_rel": 0.30},
}


@dataclass(frozen=True)
class Params:
    """Model parameters. Base values reproduce the published analysis."""

    # --- lactational module (Tier 2) -------------------------------------
    f_free: float = 0.40        # milk free (aglycone) fraction        [Eq. 1]
    rho_milk: float = 1.03      # milk density, g/mL

    # --- infant gut and conjugation ontogeny (Tier 3) --------------------
    k_deconj: float = 0.35      # gut deconjugation / reabsorption     [Eq. 2]
    mf0: float = 0.15           # UGT activity at birth, fraction adult[Eq. 3]
    tm50: float = 8.0           # UGT maturation half-time, months     [Eq. 3]
    hill: float = 2.0           # Hill coefficient                     [Eq. 3]

    # --- core disposition (Tier 1) ---------------------------------------
    f_abs: float = 0.95         # compound-specific, see COMPOUNDS
    E_h_ad: float = 0.95        # compound-specific, see COMPOUNDS
    q_h: float = 1.0            # hepatic blood flow (normalised; cancels in AF)

    # --- reverse dosimetry (evaluation module) ---------------------------
    f_ue: float = 0.84          # urinary excretion fraction (Völkel et al., 2002)

    # --- sensitivity ranges (low, high) used by sensitivity_oat() --------
    ranges: Dict[str, Sequence[float]] = field(default_factory=lambda: {
        "f_free":   (0.20, 0.60),
        "k_deconj": (0.10, 0.60),
        "mf0":      (0.05, 0.30),
        "tm50":     (4.0, 12.0),
        "E_h_ad":   (0.85, 0.99),
    })


DEFAULT_PARAMS = Params()


# --------------------------------------------------------------------------
# Core functions (vectorised over age)
# --------------------------------------------------------------------------
def maturation_factor(age_months, mf0: float = 0.15, tm50: float = 8.0,
                      hill: float = 2.0):
    """UGT maturation fraction MF(t), Eq. 3.

    MF(t) = MF0 + (1 - MF0) * t^n / (TM50^n + t^n)
    """
    t = np.asarray(age_months, dtype=float)
    return mf0 + (1.0 - mf0) * t ** hill / (tm50 ** hill + t ** hill)


def hepatic_extraction(mf, E_h_ad: float, q_h: float = 1.0):
    """Age-scaled hepatic extraction E_H(t), Eqs. 4-5.

    CL_int,adult is back-calculated from the adult extraction ratio
    (Eq. 5) rather than fitted, so that adult behaviour is anchored to
    experimentally derived human values.
    """
    mf = np.asarray(mf, dtype=float)
    cl_int_ad = q_h * E_h_ad / (1.0 - E_h_ad)      # Eq. 5
    return mf * cl_int_ad / (q_h + mf * cl_int_ad)  # Eq. 4


class Model:
    """Ontogeny-informed lactational PBPK model for one bisphenol."""

    def __init__(self, compound: str = "BPA", params: Params | None = None):
        if compound not in COMPOUNDS:
            raise ValueError(f"compound must be one of {list(COMPOUNDS)}")
        self.compound = compound
        base = params or DEFAULT_PARAMS
        self.p = replace(base, **{k: v for k, v in COMPOUNDS[compound].items()
                                  if k in ("f_abs", "E_h_ad")})

    # -- Eq. 1-2 ---------------------------------------------------------
    def external_dose(self, c_milk, intake):
        """Daily external dose D (ng/kg/day) from measured milk data.

        c_milk : total milk concentration (ng/mL)
        intake : body-weight-normalised milk intake (mL/kg/day)
        """
        return np.asarray(c_milk, float) * np.asarray(intake, float) * self.p.rho_milk

    def portal_load(self, dose):
        """Rate of aglycone delivery to the portal vein, Eq. 2."""
        p = self.p
        return p.f_abs * (p.f_free + p.k_deconj * (1.0 - p.f_free)) * np.asarray(dose, float)

    # -- Eq. 3-6 ---------------------------------------------------------
    def mf(self, age_months):
        p = self.p
        return maturation_factor(age_months, p.mf0, p.tm50, p.hill)

    def e_h(self, age_months):
        return hepatic_extraction(self.mf(age_months), self.p.E_h_ad, self.p.q_h)

    # -- Eq. 7 -----------------------------------------------------------
    def internal_free_dose(self, c_milk, intake, age_months):
        """Relative internal free-dose index IFD(t), Eq. 7.

        Proportional to steady-state free plasma exposure. The proportionality
        constant is common across infants within a compound but differs
        between compounds; IFD is therefore for ranking within a compound and
        across age, not for absolute comparison between compounds.
        """
        e = self.e_h(age_months)
        r = self.portal_load(self.external_dose(c_milk, intake))
        return r * (1.0 - e) / (self.p.q_h * e)

    # -- Eq. 8 -----------------------------------------------------------
    def amplification_factor(self, age_months):
        """Ontogeny amplification factor AF(t), Eq. 8.

        AF(t) = [(1-E_H(t))/E_H(t)] / [(1-E_H,adult)/E_H,adult] = 1/MF(t).
        The compound-specific adult extraction cancels exactly, which is why
        the AF curves for BPA, BPF and BPS coincide under a shared maturation
        function.
        """
        return 1.0 / self.mf(age_months)

    # -- adult evaluation check (Section 3.1) ----------------------------
    def adult_systemic_free_availability(self) -> float:
        """Fraction of an oral dose reaching systemic circulation as free
        aglycone in the adult (MF = 1): f_abs * (1 - E_H,adult).

        Compare with Völkel et al. (2002): free BPA below LOQ in adult plasma,
        84-100% of dose recovered in urine as conjugate within 24 h.
        """
        return self.p.f_abs * (1.0 - self.p.E_h_ad)

    # -- cohort application ----------------------------------------------
    def apply_cohort(self, c_milk, intake, age_months):
        """Vectorised application to a cohort. Returns a dict of arrays."""
        dose = self.external_dose(c_milk, intake)
        return {
            "external_EDI": dose,
            "IFD": self.internal_free_dose(c_milk, intake, age_months),
            "AF": self.amplification_factor(age_months),
            "MF": self.mf(age_months),
            "E_H": self.e_h(age_months),
        }

    # -- sensitivity ------------------------------------------------------
    def sensitivity_oat(self, c_milk, intake, age_months):
        """One-at-a-time sensitivity of the cohort-median IFD (Table S3)."""
        base = float(np.median(self.internal_free_dose(c_milk, intake, age_months)))
        out = {}
        for name, (lo, hi) in self.p.ranges.items():
            row = {}
            for tag, val in (("low", lo), ("high", hi)):
                m = Model(self.compound, replace(self.p, **{name: val}))
                med = float(np.median(m.internal_free_dose(c_milk, intake, age_months)))
                row[tag] = med
                row[f"change_{tag}_pct"] = 100.0 * (med - base) / base
            out[name] = row
        return {"base_median_IFD": base, "parameters": out}

    def monte_carlo(self, c_milk, intake, age_months, n_draws: int = 20000,
                    seed: int = 20250101):
        """Two-dimensional Monte Carlo propagation (Section 2.10, Table S4).

        Triangular distributions with the base value as the mode.
        """
        rng = np.random.default_rng(seed)
        p = self.p
        draws = {
            "f_free":   rng.triangular(0.20, p.f_free, 0.60, n_draws),
            "k_deconj": rng.triangular(0.10, p.k_deconj, 0.60, n_draws),
            "mf0":      rng.triangular(0.05, p.mf0, 0.30, n_draws),
            "tm50":     rng.triangular(4.0, p.tm50, 12.0, n_draws),
            "E_h_ad":   rng.triangular(0.85, p.E_h_ad, 0.99, n_draws),
        }
        ages = np.asarray(sorted(set(np.asarray(age_months).tolist())), float)
        af = 1.0 / maturation_factor(ages[None, :], draws["mf0"][:, None],
                                     draws["tm50"][:, None], p.hill)
        med_ifd = np.empty(n_draws)
        for i in range(n_draws):
            m = Model(self.compound, replace(
                p, f_free=draws["f_free"][i], k_deconj=draws["k_deconj"][i],
                mf0=draws["mf0"][i], tm50=draws["tm50"][i], E_h_ad=draws["E_h_ad"][i]))
            med_ifd[i] = np.median(m.internal_free_dose(c_milk, intake, age_months))
        q = lambda a, ax=0: np.percentile(a, [2.5, 50, 97.5], axis=ax)
        return {"ages": ages, "AF_q": q(af), "median_IFD_q": q(med_ifd), "draws": draws}


# --------------------------------------------------------------------------
# Reverse dosimetry (external corroboration, Section 2.8)
# --------------------------------------------------------------------------
def reverse_dosimetry(urine_conc_ug_g_cr, creatinine_g_day, body_weight_kg,
                      f_ue: float = 0.84):
    """Creatinine-based reconstruction of daily intake (ng/kg/day).

    intake = C_urine (ug/g creatinine) * CR (g/day) / (F_ue * BW) * 1000
    """
    c = np.asarray(urine_conc_ug_g_cr, float)
    return c * creatinine_g_day / (f_ue * body_weight_kg) * 1000.0


if __name__ == "__main__":  # minimal self-check
    for cmpd in COMPOUNDS:
        m = Model(cmpd)
        af = [round(float(m.amplification_factor(t)), 3) for t in range(1, 7)]
        print(f"{cmpd}: AF(1-6 mo) = {af};  adult systemic free availability = "
              f"{m.adult_systemic_free_availability()*100:.1f}%")
