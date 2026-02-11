#!/usr/bin/env python3
"""Summarize learned parameters (mean ± std) for BKT, IRT, PFA.

Covers 2 datasets (platform_a, platform_b) × 2 modalities (error_independent, error_dependent).
Uses the global-level parameter files only.

Usage:
    python src/summarize_learned_parameters.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

DATASETS = ["platform_a", "platform_b"]
MODALITIES = ["error_independent", "error_dependent"]

DATASET_LABELS = {"platform_a": "Platform A", "platform_b": "Platform B"}
MODALITY_LABELS = {
    "error_independent": "Error-Indep.",
    "error_dependent": "Error-Dep.",
}


def _sep(n: int = 100) -> str:
    return "─" * n


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "YES ***"
    if p < 0.01:
        return "YES **"
    if p < 0.05:
        return "YES *"
    return "no"


def _run_wilcoxon(
    vals_ei: pd.Series, vals_ed: pd.Series, paired: bool,
) -> tuple[str, float, float, int]:
    """Run Wilcoxon signed-rank (paired) or Mann-Whitney U (unpaired).

    Returns (test_name, statistic, p_value, n_samples).
    """
    if paired:
        # Drop pairs where the difference is exactly 0 (Wilcoxon requirement)
        diff = vals_ei - vals_ed
        nonzero = diff[diff != 0]
        if len(nonzero) < 2:
            return "Wilcoxon signed-rank", float("nan"), float("nan"), len(nonzero)
        stat, p = wilcoxon(nonzero)
        return "Wilcoxon signed-rank", stat, p, len(vals_ei)
    else:
        if len(vals_ei) < 2 or len(vals_ed) < 2:
            return "Mann-Whitney U", float("nan"), float("nan"), 0
        stat, p = mannwhitneyu(vals_ei, vals_ed, alternative="two-sided")
        return "Mann-Whitney U", stat, p, len(vals_ei) + len(vals_ed)


def _print_test_header() -> None:
    print(f"  {_sep()}")
    print(
        f"  {'Parameter':<16} {'Test':<24} {'N':>5}"
        f" {'EI mean':>12} {'ED mean':>12}"
        f" {'Statistic':>12} {'p-value':>10} {'Significant?':>14}"
    )
    print(f"  {_sep()}")


def _print_test_row(
    param: str, test_name: str, n: int,
    mean_ei: float, mean_ed: float,
    stat: float, p: float,
) -> None:
    if pd.isna(p):
        print(
            f"  {param:<16} {test_name:<24} {n:>5}"
            f" {mean_ei:>12.6f} {mean_ed:>12.6f}"
            f" {'—':>12} {'—':>10} {'too few pairs':>14}"
        )
    else:
        print(
            f"  {param:<16} {test_name:<24} {n:>5}"
            f" {mean_ei:>12.6f} {mean_ed:>12.6f}"
            f" {stat:>12.4f} {p:>10.4f} {_sig_label(p):>14}"
        )


# ── BKT ──────────────────────────────────────────────────────────────────────

def summarize_bkt(results_path: Path) -> None:
    """Print mean ± std of BKT parameters + Wilcoxon test EI vs ED."""
    print("\n" + "=" * 100)
    print("  BKT  —  Learned Parameters  (learned_parameters.csv)")
    print("=" * 100)

    param_cols = ["prior", "learns", "guesses", "slips", "forgets"]

    for dataset in DATASETS:
        # Load both modalities
        dfs: dict[str, pd.DataFrame] = {}
        for modality in MODALITIES:
            csv_path = (
                results_path / dataset / "bkt" / "global" / modality
                / "learned_parameters.csv"
            )
            if csv_path.exists():
                dfs[modality] = pd.read_csv(csv_path, sep=None, engine="python")

        # Print per-modality summaries
        for modality in MODALITIES:
            label = f"{DATASET_LABELS[dataset]} / {MODALITY_LABELS[modality]}"
            if modality not in dfs:
                print(f"\n  {label}: file not found")
                continue
            df = dfs[modality]
            print(f"\n  {label}  (n_skills = {len(df)})")
            print(f"  {_sep()}")
            print(f"  {'Parameter':<12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print(f"  {_sep()}")
            for col in param_cols:
                if col not in df.columns:
                    continue
                vals = df[col].dropna()
                print(
                    f"  {col:<12}"
                    f" {vals.mean():>12.6f}"
                    f" {vals.std():>12.6f}"
                    f" {vals.min():>12.6f}"
                    f" {vals.max():>12.6f}"
                )
            print(f"  {_sep()}")

        # Wilcoxon test
        if len(dfs) == 2:
            df_ei = dfs["error_independent"].set_index("skill")
            df_ed = dfs["error_dependent"].set_index("skill")
            common = df_ei.index.intersection(df_ed.index)

            print(f"\n  {DATASET_LABELS[dataset]}  —  Wilcoxon test: Error-Indep. vs Error-Dep.")
            if len(common) >= 2:
                paired = True
                print(f"  Paired on {len(common)} common skills (Wilcoxon signed-rank test)")
                _print_test_header()
                for col in param_cols:
                    if col not in df_ei.columns or col not in df_ed.columns:
                        continue
                    v_ei = df_ei.loc[common, col].dropna()
                    v_ed = df_ed.loc[common, col].dropna()
                    shared = v_ei.index.intersection(v_ed.index)
                    test_name, stat, p, n = _run_wilcoxon(
                        v_ei.loc[shared], v_ed.loc[shared], paired=True,
                    )
                    _print_test_row(
                        col, test_name, len(shared),
                        v_ei.loc[shared].mean(), v_ed.loc[shared].mean(), stat, p,
                    )
            else:
                paired = False
                print(f"  No common skills — using Mann-Whitney U (unpaired)")
                _print_test_header()
                for col in param_cols:
                    if col not in df_ei.columns or col not in df_ed.columns:
                        continue
                    v_ei = dfs["error_independent"][col].dropna()
                    v_ed = dfs["error_dependent"][col].dropna()
                    test_name, stat, p, n = _run_wilcoxon(v_ei, v_ed, paired=False)
                    _print_test_row(
                        col, test_name, n, v_ei.mean(), v_ed.mean(), stat, p,
                    )
            print(f"  {_sep()}")
            print(f"  Significance levels: * p<0.05, ** p<0.01, *** p<0.001")


# ── IRT ──────────────────────────────────────────────────────────────────────

def summarize_irt(results_path: Path) -> None:
    """Print mean ± std of IRT difficulty + Wilcoxon test EI vs ED."""
    print("\n" + "=" * 100)
    print("  IRT  —  Item Parameters  (item_parameters.csv)")
    print("=" * 100)

    for dataset in DATASETS:
        dfs: dict[str, pd.DataFrame] = {}
        for modality in MODALITIES:
            csv_path = (
                results_path / dataset / "irt" / "global" / modality
                / "item_parameters.csv"
            )
            if csv_path.exists():
                dfs[modality] = pd.read_csv(csv_path, sep=None, engine="python")

        for modality in MODALITIES:
            label = f"{DATASET_LABELS[dataset]} / {MODALITY_LABELS[modality]}"
            if modality not in dfs:
                print(f"\n  {label}: file not found")
                continue
            df = dfs[modality]
            print(f"\n  {label}  (n_items = {len(df)})")
            print(f"  {_sep()}")
            print(f"  {'Parameter':<12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print(f"  {_sep()}")
            vals = df["difficulty"].dropna()
            print(
                f"  {'difficulty':<12}"
                f" {vals.mean():>12.6f}"
                f" {vals.std():>12.6f}"
                f" {vals.min():>12.6f}"
                f" {vals.max():>12.6f}"
            )
            print(f"  {_sep()}")

        if len(dfs) == 2:
            df_ei = dfs["error_independent"].set_index("item")
            df_ed = dfs["error_dependent"].set_index("item")
            common = df_ei.index.intersection(df_ed.index)

            print(f"\n  {DATASET_LABELS[dataset]}  —  Wilcoxon test: Error-Indep. vs Error-Dep.")
            if len(common) >= 2:
                print(f"  Paired on {len(common)} common items (Wilcoxon signed-rank test)")
                _print_test_header()
                v_ei = df_ei.loc[common, "difficulty"].dropna()
                v_ed = df_ed.loc[common, "difficulty"].dropna()
                shared = v_ei.index.intersection(v_ed.index)
                test_name, stat, p, n = _run_wilcoxon(
                    v_ei.loc[shared], v_ed.loc[shared], paired=True,
                )
                _print_test_row(
                    "difficulty", test_name, len(shared),
                    v_ei.loc[shared].mean(), v_ed.loc[shared].mean(), stat, p,
                )
            else:
                print(f"  No common items — using Mann-Whitney U (unpaired)")
                _print_test_header()
                v_ei = dfs["error_independent"]["difficulty"].dropna()
                v_ed = dfs["error_dependent"]["difficulty"].dropna()
                test_name, stat, p, n = _run_wilcoxon(v_ei, v_ed, paired=False)
                _print_test_row(
                    "difficulty", test_name, n, v_ei.mean(), v_ed.mean(), stat, p,
                )
            print(f"  {_sep()}")
            print(f"  Significance levels: * p<0.05, ** p<0.01, *** p<0.001")


# ── PFA ──────────────────────────────────────────────────────────────────────

def summarize_pfa(results_path: Path) -> None:
    """Print mean ± std of PFA parameters + Wilcoxon test EI vs ED on skill_beta."""
    print("\n" + "=" * 100)
    print("  PFA  —  Learned Parameters  (learned_parameters.csv)")
    print("=" * 100)

    for dataset in DATASETS:
        dfs: dict[str, pd.DataFrame] = {}
        for modality in MODALITIES:
            csv_path = (
                results_path / dataset / "pfa" / "global" / modality
                / "learned_parameters.csv"
            )
            if csv_path.exists():
                dfs[modality] = pd.read_csv(csv_path, sep=None, engine="python")

        for modality in MODALITIES:
            label = f"{DATASET_LABELS[dataset]} / {MODALITY_LABELS[modality]}"
            if modality not in dfs:
                print(f"\n  {label}: file not found")
                continue
            df = dfs[modality]
            print(f"\n  {label}")
            print(f"  {_sep()}")
            print(f"  {'Parameter':<16} {'N':>5} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
            print(f"  {_sep()}")
            for param_type in ["skill_beta", "gamma_success", "gamma_failure", "intercept"]:
                subset = df[df["parameter"] == param_type]
                if subset.empty:
                    continue
                vals = subset["value"].dropna()
                n = len(vals)
                if n == 1:
                    print(
                        f"  {param_type:<16} {n:>5}"
                        f" {vals.iloc[0]:>12.6f}"
                        f" {'—':>12}"
                        f" {vals.iloc[0]:>12.6f}"
                        f" {vals.iloc[0]:>12.6f}"
                    )
                else:
                    print(
                        f"  {param_type:<16} {n:>5}"
                        f" {vals.mean():>12.6f}"
                        f" {vals.std():>12.6f}"
                        f" {vals.min():>12.6f}"
                        f" {vals.max():>12.6f}"
                    )
            print(f"  {_sep()}")

        # Wilcoxon on skill_beta (paired by skill name) + Mann-Whitney on scalars
        if len(dfs) == 2:
            print(f"\n  {DATASET_LABELS[dataset]}  —  Wilcoxon test: Error-Indep. vs Error-Dep.")

            # skill_beta: pair by skill name
            beta_ei = (
                dfs["error_independent"]
                [dfs["error_independent"]["parameter"] == "skill_beta"]
                .set_index("name")["value"]
            )
            beta_ed = (
                dfs["error_dependent"]
                [dfs["error_dependent"]["parameter"] == "skill_beta"]
                .set_index("name")["value"]
            )
            common = beta_ei.index.intersection(beta_ed.index)

            if len(common) >= 2:
                print(f"  Paired on {len(common)} common skills (Wilcoxon signed-rank test)")
                _print_test_header()
                test_name, stat, p, n = _run_wilcoxon(
                    beta_ei.loc[common], beta_ed.loc[common], paired=True,
                )
                _print_test_row(
                    "skill_beta", test_name, len(common),
                    beta_ei.loc[common].mean(), beta_ed.loc[common].mean(), stat, p,
                )
            else:
                print(f"  No common skills — using Mann-Whitney U (unpaired)")
                _print_test_header()
                test_name, stat, p, n = _run_wilcoxon(beta_ei, beta_ed, paired=False)
                _print_test_row(
                    "skill_beta", test_name, n,
                    beta_ei.mean(), beta_ed.mean(), stat, p,
                )

            # Scalar parameters: single value each, just report the difference
            for param_type in ["gamma_success", "gamma_failure", "intercept"]:
                val_ei = dfs["error_independent"]
                val_ei = val_ei[val_ei["parameter"] == param_type]["value"]
                val_ed = dfs["error_dependent"]
                val_ed = val_ed[val_ed["parameter"] == param_type]["value"]
                if len(val_ei) == 1 and len(val_ed) == 1:
                    print(
                        f"  {param_type:<16} {'(scalar — no test)':<24}"
                        f" {'1':>5}"
                        f" {val_ei.iloc[0]:>12.6f}"
                        f" {val_ed.iloc[0]:>12.6f}"
                        f" {'—':>12} {'—':>10} {'—':>14}"
                    )

            print(f"  {_sep()}")
            print(f"  Significance levels: * p<0.05, ** p<0.01, *** p<0.001")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    base_path = Path(__file__).parent.parent
    results_path = base_path / "results"

    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return

    summarize_bkt(results_path)
    summarize_irt(results_path)
    summarize_pfa(results_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
