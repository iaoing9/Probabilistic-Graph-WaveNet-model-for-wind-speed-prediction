from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 1. Path settings
# =============================================================================

BASE_DIR = Path(r"E:\LHQ_E3\退稿返修\Case 1 Sutong Bridge\Dataset1\统计显著性分析")

INPUT_FILE = BASE_DIR / "Test_prediction_results_Dataset1_24-step_summary.xlsx"
OUTPUT_FILE = BASE_DIR / "DM_test_Dataset1_24-step_results.xlsx"


# =============================================================================
# 2. DM test settings
# =============================================================================

CRITICAL_VALUE = 1.96
SIGNIFICANCE_LEVEL = 0.05

# Loss function:
# "squared"  : squared error loss, corresponding to RMSE-type comparison
# "absolute" : absolute error loss, corresponding to MAE-type comparison
LOSS_TYPE = "squared"

# HAC_LAG = 0 means no autocorrelation correction.
# HAC_LAG = None uses floor(T^(1/3)) as the Newey-West truncation lag.
HAC_LAG = 0


# =============================================================================
# 3. Utility functions
# =============================================================================

def normal_two_sided_p_value(z_abs: float) -> float:
    """
    Compute the two-sided p-value based on the standard normal distribution.
    This function does not require scipy.
    """
    return float(math.erfc(z_abs / math.sqrt(2.0)))


def normal_one_sided_p_value_for_positive_dm(z: float) -> float:
    """
    One-sided p-value for the alternative hypothesis:
        H1: mean(d_t) > 0

    Here:
        d_t = L_baseline,t - L_GWN,t

    Therefore, positive DM indicates that GWN has a smaller loss than the baseline.
    """
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def compute_loss(error: np.ndarray, loss_type: str) -> np.ndarray:
    """
    Compute the point-wise loss sequence.
    """
    if loss_type == "squared":
        return error ** 2

    if loss_type == "absolute":
        return np.abs(error)

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def newey_west_long_run_variance(d: np.ndarray, lag: int) -> float:
    """
    Estimate the long-run variance of the loss differential sequence.
    When lag = 0, this reduces to the ordinary variance term.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    t = len(d)

    if t < 2:
        raise ValueError("The loss differential series must contain at least two valid samples.")

    d_centered = d - np.mean(d)

    gamma_0 = np.sum(d_centered * d_centered) / t
    long_run_variance = gamma_0

    if lag > 0:
        for k in range(1, lag + 1):
            gamma_k = np.sum(d_centered[k:] * d_centered[:-k]) / t
            weight = 1.0 - k / (lag + 1.0)
            long_run_variance += 2.0 * weight * gamma_k

    return float(long_run_variance)


def diebold_mariano_statistic(
    y_true: np.ndarray,
    y_gwn: np.ndarray,
    y_baseline: np.ndarray,
    loss_type: str = "squared",
    hac_lag: Optional[int] = 0,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Compute the Diebold-Mariano statistic for comparing GWN with a baseline model.

    Loss differential is defined as:
        d_t = L_baseline,t - L_GWN,t

    Therefore:
        mean(d_t) > 0 indicates that the baseline has a larger average loss than GWN.
        DM_signed > 0 indicates that GWN performs better than the baseline.
        DM_signed > 1.96 indicates that GWN significantly outperforms the baseline
        at the 5% level under the normal approximation.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_gwn = np.asarray(y_gwn, dtype=np.float64).reshape(-1)
    y_baseline = np.asarray(y_baseline, dtype=np.float64).reshape(-1)

    if not (len(y_true) == len(y_gwn) == len(y_baseline)):
        raise ValueError("Input arrays must have the same length.")

    error_gwn = y_true - y_gwn
    error_baseline = y_true - y_baseline

    loss_gwn = compute_loss(error_gwn, loss_type)
    loss_baseline = compute_loss(error_baseline, loss_type)

    loss_difference = loss_baseline - loss_gwn
    t = len(loss_difference)

    if hac_lag is None:
        lag = int(np.floor(t ** (1.0 / 3.0)))
    else:
        lag = int(hac_lag)

    if lag < 0:
        raise ValueError("hac_lag must be non-negative or None.")

    long_run_var = newey_west_long_run_variance(loss_difference, lag=lag)

    if long_run_var <= 0:
        raise ValueError(
            "The estimated long-run variance is non-positive. "
            "Please check whether the two prediction series are almost identical."
        )

    mean_d = float(np.mean(loss_difference))
    dm_signed = mean_d / math.sqrt(long_run_var / t)
    dm_abs = abs(float(dm_signed))

    details = {
        "Samples": int(t),
        "HAC_lag": int(lag),
        "Mean_loss_difference_baseline_minus_GWN": mean_d,
        "Long_run_variance": float(long_run_var),
        "GWN_mean_loss": float(np.mean(loss_gwn)),
        "Baseline_mean_loss": float(np.mean(loss_baseline)),
        "DM_signed": float(dm_signed),
        "DM_abs": dm_abs,
        "P_value_two_sided": normal_two_sided_p_value(dm_abs),
        "P_value_one_sided_GWN_better": normal_one_sided_p_value_for_positive_dm(dm_signed),
    }

    detail_df = pd.DataFrame({
        "y_true": y_true,
        "y_GWN": y_gwn,
        "y_baseline": y_baseline,
        "error_GWN": error_gwn,
        "error_baseline": error_baseline,
        "loss_GWN": loss_gwn,
        "loss_baseline": loss_baseline,
        "loss_difference_baseline_minus_GWN": loss_difference,
    })

    return dm_signed, details, detail_df


# =============================================================================
# 4. Main program
# =============================================================================

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE, header=0)

    expected_rows = 4153
    if df.shape[0] != expected_rows:
        print(f"Warning: expected {expected_rows} data rows, but got {df.shape[0]} rows.")

    if df.shape[1] < 11:
        raise ValueError(f"Expected at least 11 columns from A to K, but got {df.shape[1]} columns.")

    # Column convention:
    # A: time index
    # B: observed value
    # C: GWN prediction
    # D: Transformer prediction
    # E: LSTM prediction
    # F: GRU prediction
    # G: QRF prediction
    # H: GPR prediction
    # I: Persistence prediction
    # J: ARIMA prediction
    # K: VAR prediction

    time_col = df.iloc[:, 0]
    y_true = df.iloc[:, 1]
    y_gwn = df.iloc[:, 2]

    baseline_models = {
        "Transformer": df.iloc[:, 3],
        "LSTM": df.iloc[:, 4],
        "GRU": df.iloc[:, 5],
        "QRF": df.iloc[:, 6],
        "GPR": df.iloc[:, 7],
        "Persistence": df.iloc[:, 8],
        "ARIMA": df.iloc[:, 9],
        "VAR": df.iloc[:, 10],
    }

    summary_rows = []
    detail_sheets = {}

    for model_name, y_baseline in baseline_models.items():
        temp = pd.DataFrame({
            "Time": time_col,
            "Observed": y_true,
            "GWN": y_gwn,
            model_name: y_baseline,
        })

        before_drop = len(temp)
        temp = temp.dropna(subset=["Observed", "GWN", model_name])
        after_drop = len(temp)

        if after_drop < before_drop:
            print(f"Warning: {model_name} dropped {before_drop - after_drop} rows due to missing values.")

        dm_signed, details, detail_df = diebold_mariano_statistic(
            y_true=temp["Observed"].to_numpy(),
            y_gwn=temp["GWN"].to_numpy(),
            y_baseline=temp[model_name].to_numpy(),
            loss_type=LOSS_TYPE,
            hac_lag=HAC_LAG,
        )

        # Since d_t = L_baseline,t - L_GWN,t,
        # DM_signed > 1.96 means GWN is significantly better.
        gwn_significantly_better = dm_signed > CRITICAL_VALUE
        reject_null_two_sided = details["DM_abs"] > CRITICAL_VALUE

        if gwn_significantly_better:
            conclusion = "GWN significantly outperforms the baseline model"
        elif dm_signed < -CRITICAL_VALUE:
            conclusion = "The baseline model significantly outperforms GWN"
        else:
            conclusion = "No statistically significant difference"

        summary_rows.append({
            "Baseline_model": model_name,
            "Loss_function": "Squared error loss" if LOSS_TYPE == "squared" else "Absolute error loss",
            "Valid_samples": details["Samples"],
            "HAC_lag": details["HAC_lag"],
            "GWN_mean_loss": details["GWN_mean_loss"],
            "Baseline_mean_loss": details["Baseline_mean_loss"],
            "Mean_loss_difference_baseline_minus_GWN": details["Mean_loss_difference_baseline_minus_GWN"],
            "DM_statistic_signed": details["DM_signed"],
            "DM_statistic_absolute": details["DM_abs"],
            "P_value_two_sided": details["P_value_two_sided"],
            "P_value_one_sided_GWN_better": details["P_value_one_sided_GWN_better"],
            "Critical_value_5_percent": CRITICAL_VALUE,
            "Reject_H0_two_sided": "Yes" if reject_null_two_sided else "No",
            "GWN_significantly_better_at_5_percent": "Yes" if gwn_significantly_better else "No",
            "Conclusion": conclusion,
        })

        detail_df.insert(0, "Time", temp["Time"].to_numpy())
        detail_sheets[model_name] = detail_df

    summary_df = pd.DataFrame(summary_rows)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="DM_summary", index=False)

        for model_name, detail_df in detail_sheets.items():
            # Excel sheet names cannot exceed 31 characters.
            sheet_name = f"Detail_{model_name}"[:31]
            detail_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\nDM test results:")
    print(
        summary_df[
            [
                "Baseline_model",
                "DM_statistic_signed",
                "Critical_value_5_percent",
                "GWN_significantly_better_at_5_percent",
                "P_value_two_sided",
                "Conclusion",
            ]
        ]
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
