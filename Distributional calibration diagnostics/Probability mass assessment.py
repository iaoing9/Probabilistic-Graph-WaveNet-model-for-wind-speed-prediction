# -*- coding: utf-8 -*-
"""
Probability mass assessment for day-ahead GWN probabilistic forecasts.

Input:
E:\LHQ_E3\退稿返修\Case 2 anemometric station\Station1\GWN-station1_results\Test_prediction_results.xlsx

Column convention:
A: time index
B: observed wind speed
C: predicted wind speed mean
D: predicted wind speed variance
E: predicted wind speed standard deviation
F: lower bound of 95% prediction interval
G: upper bound of 95% prediction interval
H: residual

This script only generates the probability mass assessment scatter plot and
the corresponding plotting data.

Figure settings:
Title: Probability mass assessment
x-axis: Predicted wind gust (m/s)
y-axis: Estimated probability mass below zero
"""

import os
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm


# =========================================================
# 1. Configuration
# =========================================================

class Config:
    file_path = r"E:\LHQ_E3\退稿返修\Case 1 Sutong Bridge\Dataset1\GWN_Dataset1_results\Horizon_01_step_01h_ahead\Test_prediction_results.xlsx"

    output_prefix = "Probability_mass_assessment"

    seed = 2026
    eps = 1e-8

    # A logarithmic y-axis is recommended because the estimated probability
    # masses below zero are usually extremely small.
    use_log_scale = True

    # Only used to prevent zero values from disappearing on a logarithmic axis.
    # The original unmodified probability masses are still saved to Excel.
    plot_floor = 1e-300

    figure_dpi_png = 300
    figure_dpi_tif = 600

    fig_width = 5.2
    fig_height = 4.2


cfg = Config()


# =========================================================
# 2. Reproducibility
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


set_seed(cfg.seed)


# =========================================================
# 3. Read data
# =========================================================

def read_prediction_results(file_path, eps=1e-8):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found:\n{file_path}")

    df = pd.read_excel(file_path, engine="openpyxl")

    if df.shape[1] < 8:
        raise ValueError(
            "The input Excel file should contain at least 8 columns: "
            "A time index, B observed wind speed, C predicted mean, "
            "D predicted variance, E predicted standard deviation, "
            "F lower 95% PI, G upper 95% PI, and H residual."
        )

    data = {
        "time_index": pd.to_numeric(df.iloc[:, 0], errors="coerce").values,
        "observed": pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
        "predicted_mean": pd.to_numeric(df.iloc[:, 2], errors="coerce").values,
        "predicted_variance": pd.to_numeric(df.iloc[:, 3], errors="coerce").values,
        "predicted_std": pd.to_numeric(df.iloc[:, 4], errors="coerce").values,
        "lower_95PI": pd.to_numeric(df.iloc[:, 5], errors="coerce").values,
        "upper_95PI": pd.to_numeric(df.iloc[:, 6], errors="coerce").values,
        "residual": pd.to_numeric(df.iloc[:, 7], errors="coerce").values,
    }

    valid_mask = (
        np.isfinite(data["predicted_mean"]) &
        np.isfinite(data["predicted_std"]) &
        np.isfinite(data["observed"])
    )

    if not np.all(valid_mask):
        removed = int(np.sum(~valid_mask))
        print(f"Warning: {removed} row(s) containing non-numeric or missing values were removed.")

        for key in data:
            data[key] = np.asarray(data[key])[valid_mask]

    data["predicted_std"] = np.maximum(data["predicted_std"], eps)
    data["predicted_variance"] = np.maximum(data["predicted_variance"], eps)

    if len(data["predicted_mean"]) == 0:
        raise ValueError("No valid rows remain after data validation.")

    return data


# =========================================================
# 4. Probability mass below zero
# =========================================================

def compute_probability_mass_below_zero(data):
    predicted_mean = data["predicted_mean"]
    predicted_std = data["predicted_std"]

    # For Y ~ N(mu, sigma^2):
    # P(Y < 0) = Phi((0 - mu) / sigma)
    z_at_zero = -predicted_mean / predicted_std
    probability_mass_below_zero = norm.cdf(z_at_zero)

    return z_at_zero, probability_mass_below_zero


def prepare_plot_data(data, z_at_zero, probability_mass_below_zero, cfg):
    probability_mass_used_for_plot = np.maximum(
        probability_mass_below_zero,
        cfg.plot_floor
    )
    clipped_to_plot_floor = probability_mass_below_zero < cfg.plot_floor

    plot_data_df = pd.DataFrame({
        "time_index": data["time_index"],
        "observed_wind_gust": data["observed"],
        "predicted_mean": data["predicted_mean"],
        "predicted_variance": data["predicted_variance"],
        "predicted_std": data["predicted_std"],
        "lower_95PI": data["lower_95PI"],
        "upper_95PI": data["upper_95PI"],
        "residual": data["residual"],
        "z_at_zero": z_at_zero,
        # Direct plotting data
        "x_predicted_wind_gust_m_per_s": data["predicted_mean"],
        "y_estimated_probability_mass_below_zero": probability_mass_below_zero,
        # Auxiliary plotting column for logarithmic y-axis only
        "y_probability_mass_used_for_plot": probability_mass_used_for_plot,
        "clipped_to_plot_floor": clipped_to_plot_floor,
    })

    summary_df = pd.DataFrame([{
        "sample_size": len(probability_mass_below_zero),
        "mean_probability_mass_below_zero": np.mean(probability_mass_below_zero),
        "median_probability_mass_below_zero": np.median(probability_mass_below_zero),
        "std_probability_mass_below_zero": (
            np.std(probability_mass_below_zero, ddof=1)
            if len(probability_mass_below_zero) > 1 else 0.0
        ),
        "min_probability_mass_below_zero": np.min(probability_mass_below_zero),
        "max_probability_mass_below_zero": np.max(probability_mass_below_zero),
        "90th_percentile_probability_mass_below_zero": np.quantile(probability_mass_below_zero, 0.90),
        "95th_percentile_probability_mass_below_zero": np.quantile(probability_mass_below_zero, 0.95),
        "99th_percentile_probability_mass_below_zero": np.quantile(probability_mass_below_zero, 0.99),
        "number_clipped_only_for_plot": int(np.sum(clipped_to_plot_floor)),
        "plot_y_scale": "log" if cfg.use_log_scale else "linear",
    }])

    return plot_data_df, summary_df


# =========================================================
# 5. Plotting
# =========================================================

def plot_probability_mass_assessment(ax, plot_data_df, cfg):
    x = plot_data_df["x_predicted_wind_gust_m_per_s"].values

    if cfg.use_log_scale:
        y = plot_data_df["y_probability_mass_used_for_plot"].values
    else:
        y = plot_data_df["y_estimated_probability_mass_below_zero"].values

    ax.scatter(
        x,
        y,
        s=24,
        alpha=0.70
    )

    if cfg.use_log_scale:
        ax.set_yscale("log")

    ax.set_title("Probability mass assessment")
    ax.set_xlabel("Predicted wind gust (m/s)")
    ax.set_ylabel("Estimated probability mass below zero")
    ax.grid(True, alpha=0.25)


def save_figure(plot_data_df, output_dir, cfg):
    fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height))
    plot_probability_mass_assessment(ax, plot_data_df, cfg)

    plt.tight_layout()

    png_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}.png"
    )
    tif_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_600dpi.tif"
    )

    plt.savefig(png_path, dpi=cfg.figure_dpi_png, bbox_inches="tight")
    plt.savefig(tif_path, dpi=cfg.figure_dpi_tif, bbox_inches="tight")
    plt.close(fig)

    return png_path, tif_path


# =========================================================
# 6. Save Excel outputs
# =========================================================

def save_excel_outputs(plot_data_df, summary_df, output_dir, cfg):
    excel_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_plot_data.xlsx"
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        plot_data_df.to_excel(writer, sheet_name="mass_assessment_data", index=False)
        summary_df.to_excel(writer, sheet_name="summary_statistics", index=False)

    return excel_path


# =========================================================
# 7. Main
# =========================================================

def main(cfg):
    output_dir = os.path.dirname(cfg.file_path)

    data = read_prediction_results(cfg.file_path, eps=cfg.eps)

    z_at_zero, probability_mass_below_zero = compute_probability_mass_below_zero(data)

    plot_data_df, summary_df = prepare_plot_data(
        data=data,
        z_at_zero=z_at_zero,
        probability_mass_below_zero=probability_mass_below_zero,
        cfg=cfg
    )

    png_path, tif_path = save_figure(
        plot_data_df=plot_data_df,
        output_dir=output_dir,
        cfg=cfg
    )

    excel_path = save_excel_outputs(
        plot_data_df=plot_data_df,
        summary_df=summary_df,
        output_dir=output_dir,
        cfg=cfg
    )

    print("\nFiles saved:")
    print(png_path)
    print(tif_path)
    print(excel_path)

    print("\nDirect plotting data:")
    print("x-axis column: x_predicted_wind_gust_m_per_s")
    print("y-axis column: y_estimated_probability_mass_below_zero")

    print("\nSummary statistics:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main(cfg)
