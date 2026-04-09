#!/usr/bin/env python
"""
Generate publication figures for null subject analysis.

Produces three figures (PDF + PNG at 3.25x2.5in, 300 DPI):
  1. null_subject_by_age — exponential decay fit with adult baseline
  2. mlu_by_age — linear regression with adult baseline
  3. null_subject_by_age_with_mlu — dual-axis version

Usage
-----
    python analysis/scripts/reporting/generate_null_subject_figures.py \
        --output-dir data/output/train_90M/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from statsmodels.api import WLS, add_constant


# ---------------------------------------------------------------------------
# Data (from train_90M null subject report)
# ---------------------------------------------------------------------------

AGE_MID = np.array([17.5, 29.5, 41.5, 53.5, 65.0])
RATES = np.array([0.1815, 0.0802, 0.0479, 0.0395, 0.0316])
MLU = np.array([2.79, 3.63, 4.32, 5.02, 5.90])
N_CLAUSES = np.array([5384, 96687, 110714, 133069, 71796])
LABELS = ["12-23", "24-35", "36-47", "48-59", "60+"]
ADULT_RATE = 0.0360
ADULT_MLU = 5.87

FIGSIZE = (3.25, 2.5)
DPI = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def minimal_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(width=0.5)


def dot_sizes():
    return np.sqrt(N_CLAUSES / N_CLAUSES.min()) * 18


# ---------------------------------------------------------------------------
# Figure 1: Null-subject rate by age (exponential decay)
# ---------------------------------------------------------------------------

def fig_null_subject_by_age(output_dir: Path):
    popt, _ = curve_fit(exp_decay, AGE_MID, RATES, p0=[0.3, 0.05, 0.03],
                         sigma=1 / np.sqrt(N_CLAUSES))
    half_life = np.log(2) / popt[1]

    x_smooth = np.linspace(17.5, 72, 200)
    y_smooth = exp_decay(x_smooth, *popt)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.axhline(ADULT_RATE, color='#888888', linewidth=0.8, linestyle='--', zorder=1)
    ax.text(70, ADULT_RATE + 0.004, 'Adult', fontsize=6.5, color='#888888',
            ha='right', va='bottom')

    ax.plot(x_smooth, y_smooth, color='#e08ca5', linewidth=1.2, zorder=2,
            label=f'Exponential decay fit\n$t_{{1/2}}$ = {half_life:.1f} months')

    ax.scatter(AGE_MID, RATES, s=dot_sizes(), color='#c2185b',
               edgecolors='white', linewidths=0.5, zorder=3)

    ax.legend(fontsize=6, loc='upper right', frameon=False)
    ax.set_xlabel('Age (months)', fontsize=7.5)
    ax.set_ylabel('Null-subject rate', fontsize=7.5)
    ax.set_xticks(AGE_MID)
    ax.set_xticklabels(LABELS, fontsize=6.5)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.set_ylim(-0.005, 0.22)
    ax.set_xlim(10, 73)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    minimal_spines(ax)

    plt.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'null_subject_by_age.{ext}', dpi=DPI,
                    bbox_inches='tight')
    plt.close(fig)
    print(f"  null_subject_by_age (half-life={half_life:.1f}m, R²={1 - np.sum(N_CLAUSES * (RATES - exp_decay(AGE_MID, *popt))**2) / np.sum(N_CLAUSES * (RATES - np.average(RATES, weights=N_CLAUSES))**2):.4f})")


# ---------------------------------------------------------------------------
# Figure 2: MLU by age (linear fit)
# ---------------------------------------------------------------------------

def fig_mlu_by_age(output_dir: Path):
    X = add_constant(AGE_MID)
    model = WLS(MLU, X, weights=N_CLAUSES).fit()
    x_fit = np.linspace(17.5, 65, 200)
    y_fit = model.predict(add_constant(x_fit))
    slope = model.params[1]
    r2 = model.rsquared

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.axhline(ADULT_MLU, color='#888888', linewidth=0.8, linestyle='--', zorder=1)
    ax.text(55, ADULT_MLU + 0.12, 'Adult', fontsize=6.5, color='#888888',
            ha='right', va='bottom')

    ax.plot(x_fit, y_fit, color='#9bbbd4', linewidth=1.2, zorder=2,
            label=f'Linear fit ($R^2$ = {r2:.3f})\n{slope:.3f} units/month')

    ax.scatter(AGE_MID, MLU, s=dot_sizes(), color='#4e79a7',
               edgecolors='white', linewidths=0.5, zorder=3)

    ax.legend(fontsize=6, loc='upper right', frameon=False)
    ax.set_xlabel('Age (months)', fontsize=7.5)
    ax.set_ylabel('MLU (units)', fontsize=7.5)
    ax.set_xticks(AGE_MID)
    ax.set_xticklabels(LABELS, fontsize=6.5)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.set_ylim(2, 7)
    ax.set_xlim(10, 73)
    minimal_spines(ax)

    plt.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'mlu_by_age.{ext}', dpi=DPI,
                    bbox_inches='tight')
    plt.close(fig)
    print(f"  mlu_by_age (slope={slope:.3f}/month, R²={r2:.4f})")


# ---------------------------------------------------------------------------
# Figure 3: Dual-axis (null-subject rate + MLU)
# ---------------------------------------------------------------------------

def fig_dual_axis(output_dir: Path):
    popt, _ = curve_fit(exp_decay, AGE_MID, RATES, p0=[0.3, 0.05, 0.03],
                         sigma=1 / np.sqrt(N_CLAUSES))
    half_life = np.log(2) / popt[1]
    x_smooth = np.linspace(17.5, 72, 200)
    y_smooth = exp_decay(x_smooth, *popt)

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()

    ax1.axhline(ADULT_RATE, color='#888888', linewidth=0.8, linestyle='--', zorder=1)
    ax1.text(70, ADULT_RATE + 0.004, 'Adult', fontsize=6.5, color='#888888',
             ha='right', va='bottom')

    ax1.plot(x_smooth, y_smooth, color='#e08ca5', linewidth=1.2, zorder=2)
    ax1.scatter(AGE_MID, RATES, s=dot_sizes(), color='#c2185b',
                edgecolors='white', linewidths=0.5, zorder=3,
                label=f'Null-subject rate\n$t_{{1/2}}$ = {half_life:.1f} months')

    ax2.plot(AGE_MID, MLU, color='#4e79a7', linewidth=1.2, zorder=2,
             linestyle='-', marker='o', markersize=4, label='MLU (units)')

    ax1.set_xlabel('Age (months)', fontsize=7.5)
    ax1.set_ylabel('Null-subject rate', fontsize=7.5, color='#c2185b')
    ax2.set_ylabel('MLU (units)', fontsize=7.5, color='#4e79a7')
    ax1.set_xticks(AGE_MID)
    ax1.set_xticklabels(LABELS, fontsize=6.5)
    ax1.tick_params(axis='y', labelsize=6.5, colors='#c2185b')
    ax2.tick_params(axis='y', labelsize=6.5, colors='#4e79a7')
    ax1.set_ylim(-0.005, 0.22)
    ax1.set_xlim(10, 73)
    ax2.set_ylim(2, 7)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['left'].set_linewidth(0.5)
    ax1.spines['left'].set_color('#c2185b')
    ax1.spines['bottom'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    ax2.spines['right'].set_color('#4e79a7')
    ax1.tick_params(width=0.5)
    ax2.tick_params(width=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=5.5,
               loc='upper right', frameon=False)

    plt.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'null_subject_by_age_with_mlu.{ext}',
                    dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  null_subject_by_age_with_mlu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate null subject figures.")
    parser.add_argument("--output-dir", "-o", type=Path,
                        default=Path("data/output/train_90M"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {args.output_dir}/")

    fig_null_subject_by_age(args.output_dir)
    fig_mlu_by_age(args.output_dir)
    fig_dual_axis(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
