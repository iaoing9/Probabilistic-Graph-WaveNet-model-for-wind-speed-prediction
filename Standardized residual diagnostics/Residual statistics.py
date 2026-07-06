# -*- coding: utf-8 -*-
"""
Residual diagnostic analysis for probabilistic GWN prediction results.

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
"""

import os
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm, gaussian_kde, probplot, skew, kurtosis


# =========================================================
# 1. Configuration
# =========================================================

class Config:
    file_path = r"E:\LHQ_E3\退稿返修\Case 1 Sutong Bridge\Dataset1\GWN_Dataset1_results\Horizon_01_step_01h_ahead\Test_prediction_results.xlsx"

    output_prefix = "GWN_Test"

    seed = 2026
    eps = 1e-8

    hist_bins = 25
    figure_dpi_png = 300
    figure_dpi_tif = 600

    combined_fig_width = 15
    combined_fig_height = 4.5

    single_fig_width = 5.2
    single_fig_height = 4.2


cfg = Config()


# =========================================================
# 2. Reproducibility
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


set_seed(cfg.seed)


# =========================================================
# 3. Data reading
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

    time_index = df.iloc[:, 0].values
    observed = df.iloc[:, 1].astype(float).values
    predicted_mean = df.iloc[:, 2].astype(float).values
    predicted_variance = df.iloc[:, 3].astype(float).values
    predicted_std = df.iloc[:, 4].astype(float).values
    lower_95 = df.iloc[:, 5].astype(float).values
    upper_95 = df.iloc[:, 6].astype(float).values
    residual_from_file = df.iloc[:, 7].astype(float).values

    return {
        "df": df,
        "time_index": time_index,
        "observed": observed,
        "predicted_mean": predicted_mean,
        "predicted_variance": predicted_variance,
        "predicted_std": predicted_std,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "residual_from_file": residual_from_file,
    }


# =========================================================
# 4. Residual diagnostics
# =========================================================

def compute_residual_diagnostics(data, eps=1e-8):
    observed = data["observed"]
    predicted_mean = data["predicted_mean"]
    predicted_variance = np.maximum(data["predicted_variance"], eps)
    predicted_std = np.maximum(data["predicted_std"], eps)

    lower_95 = data["lower_95"]
    upper_95 = data["upper_95"]
    residual_from_file = data["residual_from_file"]

    residual_calculated = observed - predicted_mean

    residual_difference = residual_calculated - residual_from_file
    max_abs_residual_difference = np.max(np.abs(residual_difference))

    standardized_residual = residual_calculated / predicted_std

    negative_probability = norm.cdf((0.0 - predicted_mean) / predicted_std)

    picp_95 = np.mean((observed >= lower_95) & (observed <= upper_95)) * 100.0

    stats = {
        "Sample_size": len(standardized_residual),
        "Mean_of_standardized_residuals": np.mean(standardized_residual),
        "Std_of_standardized_residuals": np.std(standardized_residual, ddof=1),
        "Skewness": skew(standardized_residual, bias=False),
        "Excess_kurtosis": kurtosis(standardized_residual, fisher=True, bias=False),
        "Mean_negative_probability": np.mean(negative_probability),
        "Max_negative_probability": np.max(negative_probability),
        "PICP_95_percent": picp_95,
        "Max_abs_difference_between_calculated_and_file_residual": max_abs_residual_difference,
    }

    diagnostic_data = {
        "predicted_variance": predicted_variance,
        "predicted_std": predicted_std,
        "residual_calculated": residual_calculated,
        "standardized_residual": standardized_residual,
        "negative_probability": negative_probability,
        "stats": stats,
    }

    return diagnostic_data


# =========================================================
# 5. Prepare plot data
# =========================================================

def prepare_distribution_plot_data(standardized_residual, hist_bins):
    x_min = min(np.min(standardized_residual), -4.0)
    x_max = max(np.max(standardized_residual), 4.0)

    x_grid = np.linspace(x_min, x_max, 500)

    hist_density, bin_edges = np.histogram(
        standardized_residual,
        bins=hist_bins,
        density=True
    )

    bin_left = bin_edges[:-1]
    bin_right = bin_edges[1:]
    bin_center = 0.5 * (bin_left + bin_right)

    try:
        kde = gaussian_kde(standardized_residual)
        kde_density = kde(x_grid)
    except Exception as exc:
        print(f"Warning: KDE could not be computed. Reason: {exc}")
        kde_density = np.full_like(x_grid, np.nan)

    standard_normal_density = norm.pdf(x_grid)

    hist_df = pd.DataFrame({
        "bin_left": bin_left,
        "bin_right": bin_right,
        "bin_center": bin_center,
        "histogram_density": hist_density,
    })

    curve_df = pd.DataFrame({
        "x_standardized_residual": x_grid,
        "kde_density": kde_density,
        "standard_normal_density": standard_normal_density,
    })

    return hist_df, curve_df


def prepare_qq_plot_data(standardized_residual):
    qq = probplot(standardized_residual, dist="norm")

    theoretical_quantiles = qq[0][0]
    ordered_standardized_residuals = qq[0][1]

    slope = qq[1][0]
    intercept = qq[1][1]
    r_value = qq[1][2]

    fitted_line_y = slope * theoretical_quantiles + intercept

    qq_df = pd.DataFrame({
        "x_theoretical_quantiles": theoretical_quantiles,
        "y_ordered_standardized_residuals": ordered_standardized_residuals,
        "y_fitted_reference_line": fitted_line_y,
    })

    qq_info_df = pd.DataFrame([{
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
    }])

    return qq_df, qq_info_df


def prepare_scatter_plot_data(time_index, predicted_mean, standardized_residual):
    scatter_df = pd.DataFrame({
        "time_index": time_index,
        "x_predicted_wind_speed": predicted_mean,
        "y_standardized_residual": standardized_residual,
        "reference_y_0": np.zeros_like(standardized_residual),
        "reference_y_1_96": np.full_like(standardized_residual, 1.96),
        "reference_y_minus_1_96": np.full_like(standardized_residual, -1.96),
    })

    return scatter_df


# =========================================================
# 6. Plotting functions
# =========================================================

def plot_distribution(ax, hist_df, curve_df):
    ax.bar(
        hist_df["bin_center"],
        hist_df["histogram_density"],
        width=hist_df["bin_right"] - hist_df["bin_left"],
        alpha=0.45,
        edgecolor="black",
        label="Histogram"
    )

    ax.plot(
        curve_df["x_standardized_residual"],
        curve_df["kde_density"],
        linewidth=2.0,
        label="KDE"
    )

    ax.plot(
        curve_df["x_standardized_residual"],
        curve_df["standard_normal_density"],
        linestyle="--",
        linewidth=2.0,
        label="Standard normal"
    )

    ax.axvline(
        0.0,
        linestyle=":",
        linewidth=1.5
    )

    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Density")
    ax.set_title("(a) Distribution of standardized residuals")
    ax.legend(frameon=False)


def plot_qq(ax, qq_df, qq_info_df):
    ax.scatter(
        qq_df["x_theoretical_quantiles"],
        qq_df["y_ordered_standardized_residuals"],
        s=18,
        alpha=0.75
    )

    ax.plot(
        qq_df["x_theoretical_quantiles"],
        qq_df["y_fitted_reference_line"],
        linestyle="--",
        linewidth=2.0
    )

    r_value = qq_info_df.loc[0, "r_value"]

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered standardized residuals")
    ax.set_title(f"(b) Q-Q plot, R = {r_value:.3f}")


def plot_scatter(ax, scatter_df):
    ax.scatter(
        scatter_df["x_predicted_wind_speed"],
        scatter_df["y_standardized_residual"],
        s=18,
        alpha=0.75
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5
    )

    ax.axhline(
        1.96,
        linestyle=":",
        linewidth=1.2
    )

    ax.axhline(
        -1.96,
        linestyle=":",
        linewidth=1.2
    )

    ax.set_xlabel("Predicted wind speed")
    ax.set_ylabel("Standardized residual")
    ax.set_title("(c) Residuals versus predicted wind speed")


def save_combined_figure(hist_df, curve_df, qq_df, qq_info_df, scatter_df, output_dir, cfg):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(cfg.combined_fig_width, cfg.combined_fig_height)
    )

    plot_distribution(axes[0], hist_df, curve_df)
    plot_qq(axes[1], qq_df, qq_info_df)
    plot_scatter(axes[2], scatter_df)

    plt.tight_layout()

    png_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_residual_diagnostics_combined.png"
    )

    tif_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_residual_diagnostics_combined_600dpi.tif"
    )

    plt.savefig(png_path, dpi=cfg.figure_dpi_png, bbox_inches="tight")
    plt.savefig(tif_path, dpi=cfg.figure_dpi_tif, bbox_inches="tight")
    plt.close(fig)

    return png_path, tif_path


def save_single_figures(hist_df, curve_df, qq_df, qq_info_df, scatter_df, output_dir, cfg):
    single_paths = {}

    fig, ax = plt.subplots(figsize=(cfg.single_fig_width, cfg.single_fig_height))
    plot_distribution(ax, hist_df, curve_df)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{cfg.output_prefix}_residual_distribution.png")
    plt.savefig(path, dpi=cfg.figure_dpi_png, bbox_inches="tight")
    plt.close(fig)
    single_paths["distribution"] = path

    fig, ax = plt.subplots(figsize=(cfg.single_fig_width, cfg.single_fig_height))
    plot_qq(ax, qq_df, qq_info_df)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{cfg.output_prefix}_residual_QQ_plot.png")
    plt.savefig(path, dpi=cfg.figure_dpi_png, bbox_inches="tight")
    plt.close(fig)
    single_paths["qq"] = path

    fig, ax = plt.subplots(figsize=(cfg.single_fig_width, cfg.single_fig_height))
    plot_scatter(ax, scatter_df)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{cfg.output_prefix}_residual_vs_predicted_wind_speed.png")
    plt.savefig(path, dpi=cfg.figure_dpi_png, bbox_inches="tight")
    plt.close(fig)
    single_paths["scatter"] = path

    return single_paths


# =========================================================
# 7. Save Excel outputs
# =========================================================

def save_excel_outputs(data, diagnostic_data, hist_df, curve_df, qq_df, qq_info_df, scatter_df, output_dir, cfg):
    residual_detail_df = pd.DataFrame({
        "time_index": data["time_index"],
        "observed_wind_speed": data["observed"],
        "predicted_mean": data["predicted_mean"],
        "predicted_variance": diagnostic_data["predicted_variance"],
        "predicted_std": diagnostic_data["predicted_std"],
        "residual_calculated_observed_minus_mean": diagnostic_data["residual_calculated"],
        "residual_from_file": data["residual_from_file"],
        "standardized_residual": diagnostic_data["standardized_residual"],
        "negative_probability": diagnostic_data["negative_probability"],
        "lower_95PI": data["lower_95"],
        "upper_95PI": data["upper_95"],
    })

    stats_df = pd.DataFrame([{
        "Case": "Case 2, Station 1, Test set",
        **diagnostic_data["stats"]
    }])

    stats_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_residual_statistics.xlsx"
    )

    plot_data_path = os.path.join(
        output_dir,
        f"{cfg.output_prefix}_residual_plot_data.xlsx"
    )

    stats_df.to_excel(stats_path, index=False)

    with pd.ExcelWriter(plot_data_path, engine="openpyxl") as writer:
        residual_detail_df.to_excel(writer, sheet_name="residual_details", index=False)
        stats_df.to_excel(writer, sheet_name="diagnostic_statistics", index=False)
        hist_df.to_excel(writer, sheet_name="fig_a_histogram_data", index=False)
        curve_df.to_excel(writer, sheet_name="fig_a_curve_data", index=False)
        qq_df.to_excel(writer, sheet_name="fig_b_QQ_data", index=False)
        qq_info_df.to_excel(writer, sheet_name="fig_b_QQ_info", index=False)
        scatter_df.to_excel(writer, sheet_name="fig_c_scatter_data", index=False)

    return stats_path, plot_data_path


# =========================================================
# 8. Main function
# =========================================================

def main(cfg):
    output_dir = os.path.dirname(cfg.file_path)

    data = read_prediction_results(cfg.file_path)

    diagnostic_data = compute_residual_diagnostics(
        data=data,
        eps=cfg.eps
    )

    standardized_residual = diagnostic_data["standardized_residual"]

    hist_df, curve_df = prepare_distribution_plot_data(
        standardized_residual=standardized_residual,
        hist_bins=cfg.hist_bins
    )

    qq_df, qq_info_df = prepare_qq_plot_data(
        standardized_residual=standardized_residual
    )

    scatter_df = prepare_scatter_plot_data(
        time_index=data["time_index"],
        predicted_mean=data["predicted_mean"],
        standardized_residual=standardized_residual
    )

    combined_png_path, combined_tif_path = save_combined_figure(
        hist_df=hist_df,
        curve_df=curve_df,
        qq_df=qq_df,
        qq_info_df=qq_info_df,
        scatter_df=scatter_df,
        output_dir=output_dir,
        cfg=cfg
    )

    single_paths = save_single_figures(
        hist_df=hist_df,
        curve_df=curve_df,
        qq_df=qq_df,
        qq_info_df=qq_info_df,
        scatter_df=scatter_df,
        output_dir=output_dir,
        cfg=cfg
    )

    stats_path, plot_data_path = save_excel_outputs(
        data=data,
        diagnostic_data=diagnostic_data,
        hist_df=hist_df,
        curve_df=curve_df,
        qq_df=qq_df,
        qq_info_df=qq_info_df,
        scatter_df=scatter_df,
        output_dir=output_dir,
        cfg=cfg
    )

    print("\nResidual diagnostic statistics:")
    stats_df = pd.DataFrame([{
        "Case": "Case 2, Station 1, Test set",
        **diagnostic_data["stats"]
    }])
    print(stats_df.to_string(index=False))

    print("\nFiles saved:")
    print(combined_png_path)
    print(combined_tif_path)
    print(single_paths["distribution"])
    print(single_paths["qq"])
    print(single_paths["scatter"])
    print(stats_path)
    print(plot_data_path)

    print("\nExcel sheets in plot data file:")
    print("residual_details")
    print("diagnostic_statistics")
    print("fig_a_histogram_data")
    print("fig_a_curve_data")
    print("fig_b_QQ_data")
    print("fig_b_QQ_info")
    print("fig_c_scatter_data")


if __name__ == "__main__":
    main(cfg)
