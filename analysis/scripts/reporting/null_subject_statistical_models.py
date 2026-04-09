#!/usr/bin/env python
"""
Statistical models for null-subject rate decline.

Two approaches:
  1. Clause-level OLS (linear probability model, N=417,650)
  2. Aggregated WLS on 5 age-band means

Also includes: chi-squared child vs adult, Cochran-Armitage trend,
pairwise age-band vs adult comparisons, and curve-fitting (6 models).

Usage
-----
    python analysis/scripts/reporting/null_subject_statistical_models.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency, norm
from statsmodels.api import OLS, WLS, add_constant


# ---------------------------------------------------------------------------
# Data (from train_90M null subject report)
# ---------------------------------------------------------------------------

AGE_MID = np.array([17.5, 29.5, 41.5, 53.5, 65.0])
RATES = np.array([0.1815, 0.0802, 0.0479, 0.0395, 0.0316])
MLU = np.array([2.79, 3.63, 4.32, 5.02, 5.90])
N_CLAUSES = np.array([5384, 96687, 110714, 133069, 71796])
NULL_COUNTS = np.array([977, 7751, 5306, 5262, 2270])
LABELS = ["12-23", "24-35", "36-47", "48-59", "60+"]

ADULT_NULL = 37565
ADULT_TOTAL = 1042719
ADULT_RATE = ADULT_NULL / ADULT_TOTAL  # 0.0360

CHILD_NULL = int(NULL_COUNTS.sum())  # 21566
CHILD_TOTAL = int(N_CLAUSES.sum())   # 417650
CHILD_RATE = CHILD_NULL / CHILD_TOTAL


# ---------------------------------------------------------------------------
# 1. Clause-level linear probability model (OLS with HC1 robust SEs)
# ---------------------------------------------------------------------------

def clause_level_ols():
    """Expand age-band data to clause-level binary outcomes and run OLS."""
    y = np.concatenate([
        np.concatenate([np.ones(n), np.zeros(total - n)])
        for n, total in zip(NULL_COUNTS, N_CLAUSES)
    ])
    x = np.concatenate([
        np.full(total, age) for age, total in zip(AGE_MID, N_CLAUSES)
    ])
    X = add_constant(x)
    model = OLS(y, X).fit(cov_type="HC1")

    print("=" * 60)
    print("1. CLAUSE-LEVEL OLS (linear probability model, HC1 SEs)")
    print("=" * 60)
    print(f"N = {len(y):,}")
    print(f"slope = {model.params[1]:.6f}/month")
    print(f"t({model.df_resid:,}) = {model.tvalues[1]:.2f}")
    print(f"p = {model.pvalues[1]:.2e}")
    print(f"R² = {model.rsquared:.4f}")
    print(model.summary())
    print()
    return model


# ---------------------------------------------------------------------------
# 2. Aggregated WLS on 5 age-band means
# ---------------------------------------------------------------------------

def aggregated_wls():
    """Weighted least squares on 5 age-band null-subject rates."""
    X = add_constant(AGE_MID)
    model = WLS(RATES, X, weights=N_CLAUSES).fit()

    print("=" * 60)
    print("2. AGGREGATED WLS (5 age-band means, weighted by n_clauses)")
    print("=" * 60)
    print(f"N (points) = {len(RATES)}")
    print(f"slope = {model.params[1]:.6f}/month")
    print(f"t({model.df_resid}) = {model.tvalues[1]:.3f}")
    print(f"p = {model.pvalues[1]:.4f}")
    print(f"R² = {model.rsquared:.4f}")
    print(model.summary())
    print()
    return model


# ---------------------------------------------------------------------------
# 3. Chi-squared: child vs adult
# ---------------------------------------------------------------------------

def chi_squared_child_vs_adult():
    """Two-way chi-squared test: child vs adult null-subject rates."""
    table = np.array([
        [CHILD_NULL, CHILD_TOTAL - CHILD_NULL],
        [ADULT_NULL, ADULT_TOTAL - ADULT_NULL],
    ])
    chi2, p, dof, _ = chi2_contingency(table, correction=False)
    diff = CHILD_RATE - ADULT_RATE
    se_diff = np.sqrt(
        CHILD_RATE * (1 - CHILD_RATE) / CHILD_TOTAL
        + ADULT_RATE * (1 - ADULT_RATE) / ADULT_TOTAL
    )
    ci_lo, ci_hi = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    print("=" * 60)
    print("3. CHI-SQUARED: child vs adult")
    print("=" * 60)
    print(f"Child: {CHILD_RATE:.4f} ({CHILD_NULL}/{CHILD_TOTAL:,})")
    print(f"Adult: {ADULT_RATE:.4f} ({ADULT_NULL}/{ADULT_TOTAL:,})")
    print(f"Diff = {diff:.4f}, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"chi2({dof}) = {chi2:.2f}, p < .001")
    print()


# ---------------------------------------------------------------------------
# 4. Cochran-Armitage trend test
# ---------------------------------------------------------------------------

def cochran_armitage():
    """Cochran-Armitage test for monotonic trend across ordered age bands."""
    scores = AGE_MID
    n = N_CLAUSES.astype(float)
    x = NULL_COUNTS.astype(float)
    N = n.sum()
    p_bar = x.sum() / N
    t_mean = np.sum(n * scores) / N

    numerator = np.sum(x * (scores - t_mean))
    denominator = np.sqrt(p_bar * (1 - p_bar) * (
        np.sum(n * scores**2) - N * t_mean**2
    ))
    Z = numerator / denominator

    print("=" * 60)
    print("4. COCHRAN-ARMITAGE TREND TEST")
    print("=" * 60)
    print(f"Z = {Z:.2f}, p < .001")
    print(f"Direction: {'Decreasing' if Z < 0 else 'Increasing'} with age")
    print()


# ---------------------------------------------------------------------------
# 5. Age band vs adult (Bonferroni k=5)
# ---------------------------------------------------------------------------

def age_band_vs_adult():
    """Two-proportion z-tests: each age band vs adult baseline."""
    k = len(LABELS)
    print("=" * 60)
    print("5. AGE BAND vs ADULT (Bonferroni k=5)")
    print("=" * 60)
    print(f"{'Band':<10} {'Rate':>7} {'Diff':>8} {'z':>8} {'p_adj':>12} {'<1pp':>6}")

    for i, label in enumerate(LABELS):
        r_c = RATES[i]
        n_c = N_CLAUSES[i]
        diff = r_c - ADULT_RATE
        se = np.sqrt(r_c * (1 - r_c) / n_c + ADULT_RATE * (1 - ADULT_RATE) / ADULT_TOTAL)
        z = diff / se
        p_raw = 2 * norm.sf(abs(z))
        p_adj = min(p_raw * k, 1.0)
        within_1pp = abs(diff) < 0.01

        print(f"{label:<10} {r_c:>7.4f} {diff:>+8.4f} {z:>8.2f} {p_adj:>12.2e} {'Yes' if within_1pp else 'No':>6}")
    print()


# ---------------------------------------------------------------------------
# 6. Curve fitting (6 models)
# ---------------------------------------------------------------------------

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def power_law(x, a, b):
    return a * x**(-b)


def curve_fits():
    """Compare 6 functional forms for the decline curve."""
    w = N_CLAUSES

    def wss_r2(y_pred):
        ss_res = np.sum(w * (RATES - y_pred) ** 2)
        ss_tot = np.sum(w * (RATES - np.average(RATES, weights=w)) ** 2)
        return 1 - ss_res / ss_tot

    results = []

    # Exponential decay
    popt, _ = curve_fit(exp_decay, AGE_MID, RATES, p0=[0.3, 0.05, 0.03],
                        sigma=1 / np.sqrt(w))
    r2 = wss_r2(exp_decay(AGE_MID, *popt))
    hl = np.log(2) / popt[1]
    results.append(("Exponential decay", r2, 3,
                     f"rate = {popt[0]:.3f} * exp(-{popt[1]:.3f} * age) + {popt[2]:.3f}",
                     f"half-life = {hl:.1f} months"))

    # Power law
    popt_pw, _ = curve_fit(power_law, AGE_MID, RATES, p0=[1, 1],
                           sigma=1 / np.sqrt(w))
    r2 = wss_r2(power_law(AGE_MID, *popt_pw))
    results.append(("Power law", r2, 2,
                     f"rate = {popt_pw[0]:.3f} * age^(-{popt_pw[1]:.3f})", ""))

    # Inverse (1/age)
    X_inv = add_constant(1.0 / AGE_MID)
    m = WLS(RATES, X_inv, weights=w).fit()
    r2 = wss_r2(m.predict(X_inv))
    results.append(("Inverse (1/age)", r2, 2,
                     f"rate = {m.params[1]:.3f}/age + {m.params[0]:.3f}", ""))

    # Quadratic
    X_q = np.column_stack([np.ones_like(AGE_MID), AGE_MID, AGE_MID**2])
    m = WLS(RATES, X_q, weights=w).fit()
    r2 = wss_r2(m.predict(X_q))
    results.append(("Quadratic", r2, 3,
                     f"rate = {m.params[0]:.3f} + {m.params[1]:.4f}*age + {m.params[2]:.6f}*age²", ""))

    # Log-linear
    X_log = add_constant(np.log(AGE_MID))
    m = WLS(RATES, X_log, weights=w).fit()
    r2 = wss_r2(m.predict(X_log))
    results.append(("Log-linear", r2, 2,
                     f"rate = {m.params[0]:.3f} + {m.params[1]:.3f}*log(age)", ""))

    # Linear
    X_lin = add_constant(AGE_MID)
    m = WLS(RATES, X_lin, weights=w).fit()
    r2 = wss_r2(m.predict(X_lin))
    results.append(("Linear", r2, 2,
                     f"rate = {m.params[0]:.3f} + {m.params[1]:.5f}*age", ""))

    results.sort(key=lambda x: -x[1])

    print("=" * 60)
    print("6. CURVE FITTING (6 models, weighted by n_clauses)")
    print("=" * 60)
    print(f"{'Model':<22} {'R²':>6} {'k':>3}  Equation")
    for name, r2, k, eq, note in results:
        print(f"{name:<22} {r2:>.4f} {k:>3}  {eq}")
        if note:
            print(f"{'':>33}{note}")
    print()


# ---------------------------------------------------------------------------
# 7. MLU linear fit (weighted)
# ---------------------------------------------------------------------------

def mlu_wls():
    """Weighted least squares: MLU ~ age."""
    X = add_constant(AGE_MID)
    model = WLS(MLU, X, weights=N_CLAUSES).fit()

    print("=" * 60)
    print("7. MLU LINEAR FIT (WLS)")
    print("=" * 60)
    print(f"slope = {model.params[1]:.3f} units/month")
    print(f"t({model.df_resid}) = {model.tvalues[1]:.2f}")
    print(f"p = {model.pvalues[1]:.4f}")
    print(f"R² = {model.rsquared:.4f}")
    print(f"Adult MLU: {5.87}")
    print()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    clause_level_ols()
    aggregated_wls()
    chi_squared_child_vs_adult()
    cochran_armitage()
    age_band_vs_adult()
    curve_fits()
    mlu_wls()


if __name__ == "__main__":
    main()
