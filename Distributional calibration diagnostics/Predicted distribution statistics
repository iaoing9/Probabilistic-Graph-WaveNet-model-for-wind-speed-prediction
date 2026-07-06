# -*- coding: utf-8 -*-
"""
PIT histogram and reliability diagram for 24 h-ahead / day-ahead GWN probabilistic forecasts.

Input:
E:\\LHQ_E3\\退稿返修\\Case 2 anemometric station\\Station1\\GWN-station1_results\\Test_prediction_results.xlsx

Column convention:
A: time index
B: observed wind speed
C: predicted wind speed mean
D: predicted wind speed variance
E: predicted wind speed standard deviation
F: lower bound of 95% prediction interval
G: upper bound of 95% prediction interval
H: residual

Note:
All 408 samples are 24 h-ahead / day-ahead forecast results.
Therefore, no per-horizon decomposition is conducted.
"""

import os
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm, kstest


# =========================================================
# 1. Configuration
# =========================================================

class Config:
    file_path = r"E:\LHQ_E3\退稿返修\Case 1 Sutong Bridge\Dataset1\GWN_Dataset1_results\Horizon_01_step_01h_ahead\Test_prediction_results.xlsx"

    output_prefix = "GWN_Test_24h"

    seed = 2026
    eps = 1e-8

    pit_bins = 19

    nominal_levels = np.array([
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75,
        0.80, 0.85, 0.90, 0.95
    ])

    figure_dpi_png = 300
    figure_dpi_tif = 600

    combined_fig_width = 10
    combined_fig_height = 4.2

    single_fig_width = 5.2
    single_fig_height = 4.2


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

def read_prediction_results(file_path):
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
        "df": df,
        "time_index": df.iloc[:, 0].values,
        "observed": df.iloc[:, 1].astype(float).values,
        "predicted_mean": df.iloc[:, 2].astype(float).values,
        "predicted_variance": df.iloc[:, 3].astype(float).values,
        "predicted_std": df.iloc[:, 4].astype(float).values,
        "lower_95_from_file": df.iloc[:, 5].astype(float).values,
        "upper_95_from_file": df.iloc[:, 6].astype(float).values,
        "residual_from_file": df.iloc[:, 7].astype(float).values,
    }

    return data


# =========================================================
# 4. PIT calculation
# =========================================================

def compute_pit(data, eps=1e-8):
    observed = data["observed"]
    predicted_mean = data["predicted_mean"]
    predicted_std = np.maximum(data["predicted_std"], eps)

    standardized_residual = (observed - predicted_mean) / predicted_std

    pit_values = norm.cdf(standardized_residual)

    negative_probability = norm.cdf((0.0 - predicted_mean) / predicted_std)

    return standardized_residual, pit_values, negative_probability


def prepare_pit_histogram_data(pit_values, pit_bins):
    bin_edges = np.linspace(0.0, 1.0, pit_bins + 1)

    counts, _ = np.histogram(pit_values, bins=bin_edges, density=False)
    density, _ = np.histogram(pit_values, bins=bin_edges, density=True)

    bin_left = bin_edges[:-1]
    bin_right = bin_edges[1:]
    bin_center = 0.5 * (bin_left + bin_right)

    expected_count = len(pit_values) / pit_bins
    expected_density = 1.0

    pit_hist_df = pd.DataFrame({
        "bin_left": bin_left,
        "bin_right": bin_right,
        "bin_center": bin_center,
        "pit_count": counts,
        "pit_density": density,
        "expected_count_under_uniform": np.full_like(bin_center, expected_count, dtype=float),
        "expected_density_under_uniform": np.full_like(bin_center, expected_density, dtype=float),
    })

    return pit_hist_df


# =========================================================
# 5. Reliability diagram calculation
# =========================================================

def compute_reliability_data(observed, predicted_mean, predicted_std, nominal_levels, eps=1e-8):
    predicted_std = np.maximum(predicted_std, eps)

    observed_range = np.max(observed) - np.min(observed)
    if observed_range < eps:
        observed_range = eps

    rows = []

    for nominal in nominal_levels:
        lower_prob = (1.0 - nominal) / 2.0
        upper_prob = (1.0 + nominal) / 2.0

        z_lower = norm.ppf(lower_prob)
        z_upper = norm.ppf(upper_prob)

        lower_bound = predicted_mean + z_lower * predicted_std
        upper_bound = predicted_mean + z_upper * predicted_std

        covered = (observed >= lower_bound) & (observed <= upper_bound)
        empirical_coverage = np.mean(covered)

        interval_width = upper_bound - lower_bound
        mean_interval_width = np.mean(interval_width)
        pinaw = mean_interval_width / observed_range

        rows.append({
            "horizon": "24 h",
            "nominal_coverage": nominal,
            "nominal_coverage_percent": nominal * 100.0,
            "lower_probability": lower_prob,
            "upper_probability": upper_prob,
            "z_lower": z_lower,
            "z_upper": z_upper,
            "empirical_coverage": empirical_coverage,
            "empirical_coverage_percent": empirical_coverage * 100.0,
            "calibration_error": empirical_coverage - nominal,
            "absolute_calibration_error": abs(empirical_coverage - nominal),
            "mean_interval_width": mean_interval_width,
            "PINAW": pinaw,
            "covered_sample_count": int(np.sum(covered)),
            "total_sample_count": int(len(observed)),
        })

    reliability_df = pd.DataFrame(rows)

    return reliability_df


# =========================================================
# 6. Plotting
# =========================================================

def plot_pit_histogram(ax, pit_hist_df):
    bar_width = pit_hist_df["bin_right"] - pit_hist_df["bin_left"]

    ax.bar(
        pit_hist_df["bin_center"],
        pit_hist_df["pit_density"],
        width=bar_width,
        alpha=0.55,
        edgecolor="black",
        label="PIT histogram"
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Uniform density"
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("(a) PIT histogram, 24 h horizon")
    ax.legend(frameon=False)


def plot_reliability_diagram(ax, reliability_df):
    ax.plot(
        reliability_df["nominal_coverage_percent"],
        reliability_df["empirical_coverage_percent"],
        marker="o",
        linewidth=2.0,
        label="Empirical coverage"
    )

    ax.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration"
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("(b) Reliability diagram, 24 h horizon")
    ax.legend(frameon=False)


def save_figures(pit_hist_df, reliability_df, output_dir, cfg):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(cfg.combined_fig_width, cfg.combined_fig_height)
    )

    plot_pit_histogram(axes[0], pit_hist_df)
    plot_reliability_diagram(axes[1], reliability_df)

    plt.tight_layout()

    combined_png_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_PIT_reliability_combined.png"
    )

    combined_tif_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_PIT_reliability_combined_600dpi.tif"
    )

    plt.savefig(combined_png_path, dpi=cfg.figure_dpi_png, bbox_inches="tight")
    plt.savefig(combined_tif_path, dpi=cfg.figure_dpi_tif, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(cfg.single_fig_width, cfg.single_fig_height))
    plot_pit_histogram(ax, pit_hist_df)
    plt.tight_layout()
    pit_png_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_PIT_histogram.png"
    )
    plt.savefig(pit_png_path, dpi=cfg.figure_dpi_png, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(cfg.single_fig_width, cfg.single_fig_height))
    plot_reliability_diagram(ax, reliability_df)
    plt.tight_layout()
    reliability_png_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_reliability_diagram.png"
    )
    plt.savefig(reliability_png_path, dpi=cfg.figure_dpi_png, bbox_inches="tight")
    plt.close(fig)

    return combined_png_path, combined_tif_path, pit_png_path, reliability_png_path


# =========================================================
# 7. Save Excel outputs
# =========================================================

def save_excel_outputs(
    data,
    standardized_residual,
    pit_values,
    negative_probability,
    pit_hist_df,
    reliability_df,
    output_dir,
    cfg
):
    observed = data["observed"]
    predicted_mean = data["predicted_mean"]
    predicted_std = np.maximum(data["predicted_std"], cfg.eps)
    predicted_variance = np.maximum(data["predicted_variance"], cfg.eps)

    lower_95_reconstructed = predicted_mean + norm.ppf(0.025) * predicted_std
    upper_95_reconstructed = predicted_mean + norm.ppf(0.975) * predicted_std

    pit_ks_statistic, pit_ks_p_value = kstest(pit_values, "uniform")

    detail_df = pd.DataFrame({
        "horizon": "24 h",
        "time_index": data["time_index"],
        "observed_wind_speed": observed,
        "predicted_mean": predicted_mean,
        "predicted_variance": predicted_variance,
        "predicted_std": predicted_std,
        "residual_observed_minus_mean": observed - predicted_mean,
        "residual_from_file": data["residual_from_file"],
        "standardized_residual": standardized_residual,
        "PIT_value": pit_values,
        "negative_probability": negative_probability,
        "lower_95PI_from_file": data["lower_95_from_file"],
        "upper_95PI_from_file": data["upper_95_from_file"],
        "lower_95PI_reconstructed_from_mu_sigma": lower_95_reconstructed,
        "upper_95PI_reconstructed_from_mu_sigma": upper_95_reconstructed,
    })

    negative_lower_ratio = np.mean(data["lower_95_from_file"] < 0.0)

    stats_df = pd.DataFrame([{
        "Case": "Case 2, Station 1, 24 h horizon",
        "Sample_size": len(pit_values),
        "Mean_PIT": np.mean(pit_values),
        "Std_PIT": np.std(pit_values, ddof=1),
        "Expected_mean_PIT_under_uniform": 0.5,
        "Expected_std_PIT_under_uniform": np.sqrt(1.0 / 12.0),
        "PIT_KS_statistic": pit_ks_statistic,
        "PIT_KS_p_value": pit_ks_p_value,
        "Mean_negative_probability": np.mean(negative_probability),
        "Max_negative_probability": np.max(negative_probability),
        "Ratio_of_negative_lower_95PI": negative_lower_ratio,
        "Mean_absolute_calibration_error": reliability_df["absolute_calibration_error"].mean(),
        "Max_absolute_calibration_error": reliability_df["absolute_calibration_error"].max(),
    }])

    stats_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_PIT_reliability_statistics.xlsx"
    )

    plot_data_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_PIT_reliability_plot_data.xlsx"
    )

    stats_df.to_excel(stats_path, index=False)

    with pd.ExcelWriter(plot_data_path, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="PIT_values", index=False)
        pit_hist_df.to_excel(writer, sheet_name="fig_a_PIT_histogram_data", index=False)
        reliability_df.to_excel(writer, sheet_name="fig_b_reliability_data", index=False)
        stats_df.to_excel(writer, sheet_name="diagnostic_statistics", index=False)

    return stats_path, plot_data_path


# =========================================================
# 8. Main
# =========================================================

def main(cfg):
    output_dir = os.path.dirname(cfg.file_path)

    data = read_prediction_results(cfg.file_path)

    standardized_residual, pit_values, negative_probability = compute_pit(
        data=data,
        eps=cfg.eps
    )

    pit_hist_df = prepare_pit_histogram_data(
        pit_values=pit_values,
        pit_bins=cfg.pit_bins
    )

    reliability_df = compute_reliability_data(
        observed=data["observed"],
        predicted_mean=data["predicted_mean"],
        predicted_std=data["predicted_std"],
        nominal_levels=cfg.nominal_levels,
        eps=cfg.eps
    )

    combined_png_path, combined_tif_path, pit_png_path, reliability_png_path = save_figures(
        pit_hist_df=pit_hist_df,
        reliability_df=reliability_df,
        output_dir=output_dir,
        cfg=cfg
    )

    stats_path, plot_data_path = save_excel_outputs(
        data=data,
        standardized_residual=standardized_residual,
        pit_values=pit_values,
        negative_probability=negative_probability,
        pit_hist_df=pit_hist_df,
        reliability_df=reliability_df,
        output_dir=output_dir,
        cfg=cfg
    )

    print("\nFiles saved:")
    print(combined_png_path)
    print(combined_tif_path)
    print(pit_png_path)
    print(reliability_png_path)
    print(stats_path)
    print(plot_data_path)

    print("\nReliability data for 24 h horizon:")
    print(reliability_df[[
        "nominal_coverage_percent",
        "empirical_coverage_percent",
        "calibration_error",
        "PINAW"
    ]].to_string(index=False))

    print("\nPIT summary for 24 h horizon:")
    print(f"Mean PIT = {np.mean(pit_values):.6f}")
    print(f"Std PIT  = {np.std(pit_values, ddof=1):.6f}")
    print("Expected mean under uniform = 0.500000")
    print(f"Expected std under uniform  = {np.sqrt(1.0 / 12.0):.6f}")
    print(f"Mean negative probability = {np.mean(negative_probability):.6f}")
    print(f"Max negative probability  = {np.max(negative_probability):.6f}")

    print("\nExcel sheets in plot data file:")
    print("PIT_values")
    print("fig_a_PIT_histogram_data")
    print("fig_b_reliability_data")
    print("diagnostic_statistics")


if __name__ == "__main__":
    main(cfg)
