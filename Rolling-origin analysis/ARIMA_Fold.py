import os
import math
import random
import warnings
import numpy as np
import pandas as pd

from scipy.stats import norm

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for the ARIMA baseline. "
        "Please install it with: pip install statsmodels"
    ) from exc


# =========================================================
# 1. Configuration
# =========================================================

class Config:
    root_dir = r"E:\LHQ_E3\退稿返修\Case 1 Sutong Bridge\Dataset1\滚动起点分析\Fold1"

    train_file = "A_Train_Dataset1_Fold1.xlsx"
    val_file = "B_Validation_Dataset1_Fold1.xlsx"
    test_file = "C_Test_Dataset1_Fold1.xlsx"

    output_dir = os.path.join(root_dir, "ARIMA_Dataset1_Fold1_results")

    seed = 2026

    # Time resolution
    # Dataset1 uses the same time-step setting as the uploaded GWN_Dataset1.py.
    samples_per_hour = 6

    # Selected multi-step horizons, kept consistent with the uploaded GWN_Dataset1.py.
    horizon_steps = [1, 6, 12, 24]
    max_horizon_step = max(horizon_steps)

    # Input window length, kept consistent with the uploaded GWN_Dataset1.py.
    input_window_hours = 24
    input_len = input_window_hours * samples_per_hour

    # Maximum lead time used for common truncation across the selected horizons.
    max_lead_steps = max_horizon_step * samples_per_hour

    # Sliding-window moving step.
    stride = 1

    # Window alignment mode, kept consistent with the uploaded GWN_Dataset1.py.
    window_alignment = "same_input_window"

    # Grid and feature settings are retained only for data-interface and output-file consistency.
    # The ARIMA baseline uses only the bridge-site observed wind-speed series.
    num_nodes = 9
    num_grid_features = 3
    num_global_features = 1
    input_features = num_grid_features + num_global_features

    # ARIMA baseline.
    # ARIMA is fitted to the bridge-site target wind-speed series in the original scale.
    # The order can be tuned by AIC, validation NLL, or validation CRPS if needed.
    arima_order = (2, 0, 1)
    arima_trend = "c"

    # For validation and test sets, forecasts are generated in a rolling-origin manner.
    # For each sample, the ARIMA state is updated with observations up to the input-window end
    # and then used to forecast the target lead time.
    update_with_observations = True

    # Training-related settings are kept only for interface consistency with other baselines.
    batch_size = 16
    epochs = 1
    learning_rate = 1e-3
    weight_decay = 1e-5
    patience = 9999
    min_delta = 1e-5
    grad_clip = 5.0

    # Probabilistic prediction
    eps_var = 1e-6
    interval_alpha = 0.05
    z_value = 1.959963984540054

    # ARIMA must preserve chronological order.
    shuffle_train = False


cfg = Config()


# =========================================================
# 2. Reproducibility
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# =========================================================
# 3. Coordinates and adjacency matrix
# =========================================================
# These definitions are retained from the GWN code for output-file consistency.
# The ARIMA baseline does not use ECMWF grid features or graph information.

station_coord = (31.77, 120.99)

grid_coords = [
    (32.00, 120.75),
    (32.00, 121.00),
    (32.00, 121.25),
    (31.75, 120.75),
    (31.75, 121.00),
    (31.75, 121.25),
    (31.50, 120.75),
    (31.50, 121.00),
    (31.50, 121.25),
]


def haversine_distance_km(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    r = 6371.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def build_distance_adjacency(coords, sigma_scale=1.0):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            dist[i, j] = haversine_distance_km(coords[i], coords[j])

    positive_dist = dist[dist > 0]
    sigma = positive_dist.mean() * sigma_scale

    adj = np.exp(-dist ** 2 / (sigma ** 2 + 1e-12)).astype(np.float32)
    np.fill_diagonal(adj, 1.0)

    row_sum = adj.sum(axis=1, keepdims=True) + 1e-12
    adj = adj / row_sum

    return adj, dist


def build_station_readout_weights(station, coords):
    d = np.array([haversine_distance_km(station, c) for c in coords], dtype=np.float32)
    inv_d = 1.0 / (d + 1e-6)
    w = inv_d / inv_d.sum()
    return w.astype(np.float32), d


predefined_adj, grid_distance = build_distance_adjacency(grid_coords)
station_weights, station_distance = build_station_readout_weights(station_coord, grid_coords)


# =========================================================
# 4. Data loading and preprocessing
# =========================================================

def read_dataset1_excel(file_path):
    """
    Dataset1 column convention:
    A       : Time
    B:J     : ECMWF 10 m mean wind speed at 9 grid nodes
    K:S     : ECMWF 100 m mean wind speed at 9 grid nodes
    T:AB    : ECMWF 10 m gust wind speed at 9 grid nodes
    AC      : Observed bridge-site wind speed from anemometer

    The ARIMA baseline uses only AC as the univariate target series.
    x_grid and x_global are retained only for consistent sliding-window metadata.
    """
    df = pd.read_excel(file_path, engine="openpyxl")

    if df.shape[1] < 29:
        raise ValueError(
            f"{file_path} has {df.shape[1]} columns, but at least 29 columns A:AC are required."
        )

    time = df.iloc[:, 0].values

    ecmwf_raw = df.iloc[:, 1:28].astype(np.float32).values
    ane_raw = df.iloc[:, 28].astype(np.float32).values.reshape(-1, 1)

    x_grid = split_dataset1_grid_features(ecmwf_raw, num_nodes=9)
    x_global = ane_raw.astype(np.float32)
    y_raw = ane_raw.astype(np.float32)

    return time, x_grid, x_global, y_raw, df


def split_dataset1_grid_features(ecmwf_raw, num_nodes=9):
    """
    ecmwf_raw has 27 columns:
    0:9     ECMWF 10 m mean wind speed at P1-P9
    9:18    ECMWF 100 m mean wind speed at P1-P9
    18:27   ECMWF 10 m gust wind speed at P1-P9

    Return:
    x_grid: [T, N, F_grid], where F_grid = 3.
    """
    if ecmwf_raw.shape[1] != 3 * num_nodes:
        raise ValueError(
            f"Expected {3 * num_nodes} ECMWF columns, but got {ecmwf_raw.shape[1]}."
        )

    groups = []
    for g in range(3):
        groups.append(ecmwf_raw[:, g * num_nodes:(g + 1) * num_nodes])

    x_grid = np.stack(groups, axis=1).transpose(0, 2, 1).astype(np.float32)
    return x_grid


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data, axis=None):
        self.mean = np.mean(data, axis=axis, keepdims=True)
        self.std = np.std(data, axis=axis, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def build_sliding_windows(
    x_grid,
    x_global,
    y,
    time,
    input_len,
    lead_steps,
    max_lead_steps,
    stride=1,
    alignment="same_input_window"
):
    """
    Build sliding-window samples inside one chronological split only.

    For alignment = "same_input_window":
        start indices are shared by all horizons:
        start = 0, 1, ..., N - input_len - max_lead_steps.
        target index for horizon h is start + input_len + lead_steps - 1.

    For alignment = "same_target_time":
        target indices are shared by all horizons:
        target = input_len + max_lead_steps - 1, ..., N - 1.
        input window ends at target - lead_steps.
    """
    n = len(y)

    if n != len(x_grid) or n != len(x_global) or n != len(time):
        raise ValueError("x_grid, x_global, y, and time must have the same first dimension.")

    if input_len <= 0 or lead_steps <= 0 or max_lead_steps <= 0:
        raise ValueError("input_len, lead_steps, and max_lead_steps must be positive.")

    if lead_steps > max_lead_steps:
        raise ValueError("lead_steps must not be larger than max_lead_steps.")

    if n < input_len + max_lead_steps:
        raise ValueError(
            f"Split length {n} is too short for input_len={input_len} and max_lead_steps={max_lead_steps}."
        )

    xg_list = []
    xgl_list = []
    y_list = []
    target_time_list = []
    input_start_time_list = []
    input_end_time_list = []
    input_start_index_list = []
    input_end_index_list = []
    target_index_list = []

    if alignment == "same_input_window":
        max_start = n - input_len - max_lead_steps
        starts = np.arange(0, max_start + 1, stride, dtype=np.int64)

        for start in starts:
            end = start + input_len - 1
            target_idx = start + input_len + lead_steps - 1

            xg_list.append(x_grid[start:start + input_len])
            xgl_list.append(x_global[start:start + input_len])
            y_list.append(y[target_idx])

            target_time_list.append(time[target_idx])
            input_start_time_list.append(time[start])
            input_end_time_list.append(time[end])
            input_start_index_list.append(start)
            input_end_index_list.append(end)
            target_index_list.append(target_idx)

    elif alignment == "same_target_time":
        target_start = input_len + max_lead_steps - 1
        targets = np.arange(target_start, n, stride, dtype=np.int64)

        for target_idx in targets:
            end = target_idx - lead_steps
            start = end - input_len + 1

            if start < 0:
                continue

            xg_list.append(x_grid[start:start + input_len])
            xgl_list.append(x_global[start:start + input_len])
            y_list.append(y[target_idx])

            target_time_list.append(time[target_idx])
            input_start_time_list.append(time[start])
            input_end_time_list.append(time[end])
            input_start_index_list.append(start)
            input_end_index_list.append(end)
            target_index_list.append(target_idx)

    else:
        raise ValueError("alignment must be 'same_input_window' or 'same_target_time'.")

    xg = np.stack(xg_list, axis=0).astype(np.float32)
    xgl = np.stack(xgl_list, axis=0).astype(np.float32)
    yy = np.stack(y_list, axis=0).astype(np.float32)

    sample_meta = {
        "target_time": np.array(target_time_list),
        "input_start_time": np.array(input_start_time_list),
        "input_end_time": np.array(input_end_time_list),
        "input_start_index": np.array(input_start_index_list, dtype=np.int64),
        "input_end_index": np.array(input_end_index_list, dtype=np.int64),
        "target_index": np.array(target_index_list, dtype=np.int64),
    }

    return xg, xgl, yy, sample_meta


def load_raw_splits(cfg):
    train_path = os.path.join(cfg.root_dir, cfg.train_file)
    val_path = os.path.join(cfg.root_dir, cfg.val_file)
    test_path = os.path.join(cfg.root_dir, cfg.test_file)

    train_time, train_x_grid_raw, train_x_global_raw, train_y_raw, train_df = read_dataset1_excel(train_path)
    val_time, val_x_grid_raw, val_x_global_raw, val_y_raw, val_df = read_dataset1_excel(val_path)
    test_time, test_x_grid_raw, test_x_global_raw, test_y_raw, test_df = read_dataset1_excel(test_path)

    return {
        "Train": {
            "time": train_time,
            "x_grid_raw": train_x_grid_raw,
            "x_global_raw": train_x_global_raw,
            "y_raw": train_y_raw,
            "df": train_df,
        },
        "Validation": {
            "time": val_time,
            "x_grid_raw": val_x_grid_raw,
            "x_global_raw": val_x_global_raw,
            "y_raw": val_y_raw,
            "df": val_df,
        },
        "Test": {
            "time": test_time,
            "x_grid_raw": test_x_grid_raw,
            "x_global_raw": test_x_global_raw,
            "y_raw": test_y_raw,
            "df": test_df,
        },
    }


def fit_scalers_on_training(raw_data):
    scaler_grid = StandardScaler()
    scaler_global = StandardScaler()
    scaler_y = StandardScaler()

    train = raw_data["Train"]

    # Fit only on the training split to avoid normalization leakage.
    # ARIMA is evaluated in the original scale; these scalers are stored only for workflow consistency.
    scaler_grid.fit(train["x_grid_raw"], axis=(0, 1))
    scaler_global.fit(train["x_global_raw"], axis=0)
    scaler_y.fit(train["y_raw"], axis=0)

    return scaler_grid, scaler_global, scaler_y


def prepare_arima_samples_for_horizon(cfg, raw_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour

    samples = {}

    for split_name, data in raw_data.items():
        _, _, y_window, sample_meta = build_sliding_windows(
            x_grid=data["x_grid_raw"],
            x_global=data["x_global_raw"],
            y=data["y_raw"],
            time=data["time"],
            input_len=cfg.input_len,
            lead_steps=lead_steps,
            max_lead_steps=cfg.max_lead_steps,
            stride=cfg.stride,
            alignment=cfg.window_alignment,
        )

        samples[split_name] = {
            "y": y_window.reshape(-1).astype(np.float32),
            "sample_meta": sample_meta,
            "raw_size": len(data["y_raw"]),
        }

    return samples


# =========================================================
# 5. Probabilistic ARIMA baseline
# =========================================================

class ProbabilisticARIMA:
    def __init__(
        self,
        order=(2, 0, 1),
        trend="c",
        eps_var=1e-6,
        update_with_observations=True,
    ):
        self.order = order
        self.trend = trend
        self.eps_var = eps_var
        self.update_with_observations = update_with_observations
        self.result = None
        self.aic = None
        self.bic = None
        self.residual_variance = None

    def _fit_series(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(
                y,
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit()
        return result

    def fit(self, y_train):
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        self.result = self._fit_series(y_train)
        self.aic = float(self.result.aic)
        self.bic = float(self.result.bic)

        resid = np.asarray(self.result.resid, dtype=np.float64).reshape(-1)
        resid = resid[np.isfinite(resid)]
        if resid.size > 1:
            self.residual_variance = float(np.var(resid, ddof=1))
        else:
            self.residual_variance = self.eps_var
        self.residual_variance = max(self.residual_variance, self.eps_var)

        return self

    def _sanitize_variance(self, var):
        var = np.asarray(var, dtype=np.float64).reshape(-1)
        fallback = self.residual_variance if self.residual_variance is not None else self.eps_var
        var = np.where(np.isfinite(var), var, fallback)
        var = np.maximum(var, self.eps_var)
        return var

    def _prediction_variance(self, prediction_result):
        try:
            var = np.asarray(prediction_result.var_pred_mean, dtype=np.float64)
        except AttributeError:
            frame = prediction_result.summary_frame()
            var = np.asarray(frame["mean_se"], dtype=np.float64) ** 2
        return self._sanitize_variance(var)

    def predict_train_by_target_indices(self, y_train, target_indices):
        """
        In-sample probabilistic prediction for the training set.
        This is provided mainly for output consistency with the other baselines.
        """
        if self.result is None:
            raise RuntimeError("The ARIMA model must be fitted before prediction.")

        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        target_indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)

        pred_res = self.result.get_prediction(start=0, end=len(y_train) - 1)
        mu_all = np.asarray(pred_res.predicted_mean, dtype=np.float64).reshape(-1)
        var_all = self._prediction_variance(pred_res)

        y = y_train[target_indices]
        mu = mu_all[target_indices]
        var = var_all[target_indices]

        return y.astype(np.float32), mu.astype(np.float32), var.astype(np.float32)

    def rolling_forecast_for_samples(self, initial_history, eval_series, sample_meta, lead_steps):
        """
        Rolling-origin probabilistic forecasts for sliding-window samples.

        For each sample, observations up to Input_end_index are appended to the ARIMA state.
        The model then forecasts lead_steps ahead, and the last forecast step is used as
        the prediction for Target_index.
        """
        if self.result is None:
            raise RuntimeError("The ARIMA model must be fitted before prediction.")

        initial_history = np.asarray(initial_history, dtype=np.float64).reshape(-1)
        eval_series = np.asarray(eval_series, dtype=np.float64).reshape(-1)

        input_end_indices = np.asarray(sample_meta["input_end_index"], dtype=np.int64).reshape(-1)
        target_indices = np.asarray(sample_meta["target_index"], dtype=np.int64).reshape(-1)

        if input_end_indices.size == 0:
            raise ValueError("No ARIMA samples are available for rolling forecast.")

        current_result = self._fit_series(initial_history)
        appended_until = -1

        mu_list = []
        var_list = []

        for input_end_idx in input_end_indices:
            if input_end_idx >= len(eval_series):
                raise IndexError("Input_end_index is outside the evaluation series.")

            if input_end_idx > appended_until:
                new_obs = eval_series[appended_until + 1:input_end_idx + 1]
                if self.update_with_observations and new_obs.size > 0:
                    current_result = current_result.append(new_obs, refit=False)
                appended_until = input_end_idx

            forecast_res = current_result.get_forecast(steps=lead_steps)
            mu_steps = np.asarray(forecast_res.predicted_mean, dtype=np.float64).reshape(-1)
            var_steps = self._prediction_variance(forecast_res)

            mu_list.append(mu_steps[-1])
            var_list.append(var_steps[-1])

        y = eval_series[target_indices]
        mu = np.asarray(mu_list, dtype=np.float64)
        var = self._sanitize_variance(np.asarray(var_list, dtype=np.float64))

        return y.astype(np.float32), mu.astype(np.float32), var.astype(np.float32)


# =========================================================
# 6. Metrics
# =========================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def picp(y_true, lower, upper):
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside) * 100.0)


def pinaw(y_true, lower, upper):
    width = upper - lower
    data_range = np.max(y_true) - np.min(y_true)
    return float(np.mean(width) / (data_range + 1e-12))


def gaussian_crps(y_true, mu, sigma):
    sigma = np.maximum(sigma, 1e-8)
    z = (y_true - mu) / sigma
    crps_value = sigma * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_value))


def gaussian_nll_np(y_true, mu, var):
    var = np.maximum(var, 1e-8)
    nll = 0.5 * (np.log(2.0 * np.pi * var) + (y_true - mu) ** 2 / var)
    return float(np.mean(nll))


def compute_all_metrics(y_true, mu, var, z_value, alpha=0.05):
    sigma = np.sqrt(np.maximum(var, 1e-12))
    lower = mu - z_value * sigma
    upper = mu + z_value * sigma

    metrics = {
        "RMSE": rmse(y_true, mu),
        "MAE": mae(y_true, mu),
        "MAPE_percent": mape(y_true, mu),
        "R2": r2_score_np(y_true, mu),
        "PICP_percent": picp(y_true, lower, upper),
        "PINAW": pinaw(y_true, lower, upper),
        "CRPS": gaussian_crps(y_true, mu, sigma),
        "NLL": gaussian_nll_np(y_true, mu, var),
    }

    return metrics, lower, upper


# =========================================================
# 7. Output helpers
# =========================================================

def save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y):
    os.makedirs(cfg.output_dir, exist_ok=True)

    coord_df = pd.DataFrame({
        "Node": [f"P{i + 1}" for i in range(cfg.num_nodes)],
        "Latitude": [c[0] for c in grid_coords],
        "Longitude": [c[1] for c in grid_coords],
        "Distance_to_station_km": station_distance,
        "Station_readout_weight": station_weights,
    })
    coord_df.to_excel(
        os.path.join(cfg.output_dir, "grid_coordinates_and_station_weights.xlsx"),
        index=False
    )

    adj_df = pd.DataFrame(
        predefined_adj,
        index=[f"P{i + 1}" for i in range(cfg.num_nodes)],
        columns=[f"P{i + 1}" for i in range(cfg.num_nodes)]
    )
    adj_df.to_excel(
        os.path.join(cfg.output_dir, "predefined_distance_adjacency_reference.xlsx")
    )

    split_info = []
    for split_name, data in raw_data.items():
        n_raw = len(data["y_raw"])
        n_eff = n_raw - cfg.input_len - cfg.max_lead_steps + 1
        if cfg.stride > 1:
            n_eff = int(np.floor((n_eff - 1) / cfg.stride) + 1)

        split_info.append({
            "Split": split_name,
            "Raw_sample_size": n_raw,
            "Input_window_samples": cfg.input_len,
            "Input_window_hours": cfg.input_window_hours,
            "Max_lead_samples": cfg.max_lead_steps,
            "Max_lead_hours": cfg.max_horizon_step,
            "Selected_horizon_steps": str(cfg.horizon_steps),
            "Stride_samples": cfg.stride,
            "Effective_sample_size_each_horizon": n_eff,
            "Dropped_samples_due_to_common_truncation": n_raw - n_eff,
            "Window_alignment": cfg.window_alignment,
        })

    split_info_df = pd.DataFrame(split_info)
    split_info_df.to_excel(
        os.path.join(cfg.output_dir, "data_split_and_window_sample_size_summary.xlsx"),
        index=False
    )

    scaler_info = {
        "Scaler": [],
        "Shape_mean": [],
        "Shape_std": [],
        "Mean_values": [],
        "Std_values": [],
    }

    for name, scaler in [
        ("grid_features", scaler_grid),
        ("global_ANE_feature", scaler_global),
        ("target_ANE_wind_speed", scaler_y),
    ]:
        scaler_info["Scaler"].append(name)
        scaler_info["Shape_mean"].append(str(np.array(scaler.mean).shape))
        scaler_info["Shape_std"].append(str(np.array(scaler.std).shape))
        scaler_info["Mean_values"].append(np.array2string(np.squeeze(scaler.mean), precision=6, separator=", "))
        scaler_info["Std_values"].append(np.array2string(np.squeeze(scaler.std), precision=6, separator=", "))

    scaler_df = pd.DataFrame(scaler_info)
    scaler_df.to_excel(
        os.path.join(cfg.output_dir, "training_only_scaler_information_reference.xlsx"),
        index=False
    )

    feature_info = pd.DataFrame({
        "Feature_group": [
            "Target_ANE_observed_wind_speed",
            "ECMWF_grid_features",
            "ANE_historical_observed_wind_speed",
        ],
        "Excel_columns": [
            "AC",
            "B:AB",
            "AC",
        ],
        "Used_as": [
            "univariate ARIMA series",
            "not used by ARIMA",
            "not used by ARIMA",
        ],
    })
    feature_info.to_excel(
        os.path.join(cfg.output_dir, "feature_definition_arima.xlsx"),
        index=False
    )


def save_combined_outputs(cfg, all_metrics, all_predictions):
    metrics_all_df = pd.concat(all_metrics, axis=0, ignore_index=True)

    metrics_all_df.to_excel(
        os.path.join(cfg.output_dir, "metrics_summary_all_horizons_all_splits.xlsx"),
        index=False
    )

    for split_name in ["Train", "Validation", "Test"]:
        split_prediction_list = []

        for horizon_id in cfg.horizon_steps:
            pred_df = all_predictions[horizon_id][split_name].copy()
            split_prediction_list.append(pred_df)

        split_long_df = pd.concat(split_prediction_list, axis=0, ignore_index=True)
        split_long_df.to_excel(
            os.path.join(cfg.output_dir, f"{split_name}_prediction_results_all_horizons_long.xlsx"),
            index=False
        )

        # Wide-format file for quick plotting.
        first_horizon = cfg.horizon_steps[0]
        base = all_predictions[first_horizon][split_name][[
            "Sample_index",
            "Input_start_time",
            "Input_end_time",
        ]].copy()

        wide_df = base.copy()

        for horizon_id in cfg.horizon_steps:
            pred_df = all_predictions[horizon_id][split_name].copy()
            prefix = f"H{horizon_id:02d}_"

            add_cols = pred_df[[
                "Target_time",
                "Observed",
                "Predicted_mean",
                "Predicted_std",
                "Lower_95PI",
                "Upper_95PI",
                "Residual",
            ]].copy()

            add_cols = add_cols.rename(columns={c: prefix + c for c in add_cols.columns})
            wide_df = pd.concat([wide_df, add_cols], axis=1)

        wide_df.to_excel(
            os.path.join(cfg.output_dir, f"{split_name}_prediction_results_all_horizons_wide.xlsx"),
            index=False
        )

    test_metrics = metrics_all_df[metrics_all_df["Split"] == "Test"].copy()
    test_metrics.to_excel(
        os.path.join(cfg.output_dir, "Test_metrics_summary_all_horizons.xlsx"),
        index=False
    )


# =========================================================
# 8. Training and evaluation
# =========================================================

def train_one_horizon(cfg, raw_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    lead_hours = horizon_id

    horizon_dir = os.path.join(cfg.output_dir, f"Horizon_{horizon_id:02d}_step_{lead_hours:02d}h_ahead")
    os.makedirs(horizon_dir, exist_ok=True)

    samples = prepare_arima_samples_for_horizon(
        cfg=cfg,
        raw_data=raw_data,
        horizon_id=horizon_id
    )

    train_y_raw = raw_data["Train"]["y_raw"].reshape(-1)
    val_y_raw = raw_data["Validation"]["y_raw"].reshape(-1)
    test_y_raw = raw_data["Test"]["y_raw"].reshape(-1)

    model = ProbabilisticARIMA(
        order=cfg.arima_order,
        trend=cfg.arima_trend,
        eps_var=cfg.eps_var,
        update_with_observations=cfg.update_with_observations,
    )

    print("\n" + "=" * 80)
    print(f"Training ARIMA Horizon {horizon_id:02d}: {lead_hours} h ahead, lead_steps = {lead_steps} samples")
    print(f"ARIMA order = {cfg.arima_order}, trend = {cfg.arima_trend}")
    print(f"Input length = {cfg.input_len} samples ({cfg.input_window_hours} h)")
    print(f"Window alignment = {cfg.window_alignment}, stride = {cfg.stride}")
    for split_name in ["Train", "Validation", "Test"]:
        print(f"{split_name} samples: {len(samples[split_name]['y'])}")
    print("=" * 80)

    model.fit(train_y_raw)

    arima_info_df = pd.DataFrame([{
        "Model": "ARIMA",
        "Horizon_step": horizon_id,
        "Lead_hours": lead_hours,
        "Lead_samples": lead_steps,
        "order": str(cfg.arima_order),
        "trend": cfg.arima_trend,
        "AIC": model.aic,
        "BIC": model.bic,
        "Residual_variance": model.residual_variance,
        "update_with_observations": cfg.update_with_observations,
        "Window_alignment": cfg.window_alignment,
    }])
    arima_info_df.to_excel(
        os.path.join(horizon_dir, f"arima_model_info_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    history_df = pd.DataFrame([{
        "Horizon_step": horizon_id,
        "Lead_hours": lead_hours,
        "Lead_samples": lead_steps,
        "Epoch": 0,
        "Train_GNLL": np.nan,
        "Validation_GNLL": np.nan,
        "Note": "Probabilistic ARIMA baseline fitted on the training target wind-speed series."
    }])
    history_df.to_excel(
        os.path.join(horizon_dir, f"training_history_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    split_predictions = {}

    train_y, train_mu, train_var = model.predict_train_by_target_indices(
        y_train=train_y_raw,
        target_indices=samples["Train"]["sample_meta"]["target_index"]
    )
    split_predictions["Train"] = (train_y, train_mu, train_var)

    val_y, val_mu, val_var = model.rolling_forecast_for_samples(
        initial_history=train_y_raw,
        eval_series=val_y_raw,
        sample_meta=samples["Validation"]["sample_meta"],
        lead_steps=lead_steps
    )
    split_predictions["Validation"] = (val_y, val_mu, val_var)

    test_initial_history = np.concatenate([train_y_raw, val_y_raw], axis=0)
    test_y, test_mu, test_var = model.rolling_forecast_for_samples(
        initial_history=test_initial_history,
        eval_series=test_y_raw,
        sample_meta=samples["Test"]["sample_meta"],
        lead_steps=lead_steps
    )
    split_predictions["Test"] = (test_y, test_mu, test_var)

    result_summary = []
    prediction_records = {}

    for split_name in ["Train", "Validation", "Test"]:
        y_flat, mu_flat, var_flat = split_predictions[split_name]
        sample_meta = samples[split_name]["sample_meta"]

        y_flat = y_flat.reshape(-1)
        mu_flat = mu_flat.reshape(-1)
        var_flat = var_flat.reshape(-1)

        metrics, lower, upper = compute_all_metrics(
            y_flat,
            mu_flat,
            var_flat,
            cfg.z_value,
            alpha=cfg.interval_alpha
        )

        metrics["Horizon_step"] = horizon_id
        metrics["Lead_hours"] = lead_hours
        metrics["Lead_samples"] = lead_steps
        metrics["Split"] = split_name
        metrics["Raw_split_size"] = samples[split_name]["raw_size"]
        metrics["Effective_sample_size"] = len(y_flat)
        metrics["Input_window_samples"] = cfg.input_len
        metrics["Input_window_hours"] = cfg.input_window_hours
        metrics["Max_lead_samples_for_common_truncation"] = cfg.max_lead_steps
        metrics["Window_alignment"] = cfg.window_alignment
        metrics["Best_epoch"] = 0
        metrics["Best_validation_GNLL"] = np.nan

        result_summary.append(metrics)

        sigma_flat = np.sqrt(np.maximum(var_flat, 1e-12))

        pred_df = pd.DataFrame({
            "Sample_index": np.arange(1, len(y_flat) + 1),
            "Horizon_step": horizon_id,
            "Lead_hours": lead_hours,
            "Lead_samples": lead_steps,
            "Input_start_index_0based": sample_meta["input_start_index"],
            "Input_end_index_0based": sample_meta["input_end_index"],
            "Target_index_0based": sample_meta["target_index"],
            "Input_start_time": sample_meta["input_start_time"],
            "Input_end_time": sample_meta["input_end_time"],
            "Target_time": sample_meta["target_time"],
            "Observed": y_flat,
            "Predicted_mean": mu_flat,
            "Predicted_variance": var_flat,
            "Predicted_std": sigma_flat,
            "Lower_95PI": lower.reshape(-1),
            "Upper_95PI": upper.reshape(-1),
            "Residual": y_flat - mu_flat,
        })

        pred_path = os.path.join(
            horizon_dir,
            f"{split_name}_prediction_results_horizon_{horizon_id:02d}.xlsx"
        )
        pred_df.to_excel(pred_path, index=False)

        prediction_records[split_name] = pred_df

        print(f"\nHorizon {horizon_id:02d}, {split_name} metrics")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            elif k in ["Split", "Horizon_step", "Lead_hours", "Effective_sample_size", "Best_epoch"]:
                print(f"{k}: {v}")

    summary_df = pd.DataFrame(result_summary)

    cols = [
        "Horizon_step",
        "Lead_hours",
        "Lead_samples",
        "Split",
        "Raw_split_size",
        "Effective_sample_size",
        "Input_window_samples",
        "Input_window_hours",
        "Max_lead_samples_for_common_truncation",
        "Window_alignment",
        "Best_epoch",
        "Best_validation_GNLL",
        "RMSE",
        "MAE",
        "MAPE_percent",
        "R2",
        "PICP_percent",
        "PINAW",
        "CRPS",
        "NLL",
    ]
    summary_df = summary_df[cols]

    summary_df.to_excel(
        os.path.join(horizon_dir, f"metrics_summary_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    return model, summary_df, prediction_records, history_df


def train_all_horizons(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Using model: Probabilistic ARIMA")
    print(f"Selected horizon steps: {cfg.horizon_steps}")
    print("Loading Dataset1...")
    raw_data = load_raw_splits(cfg)

    print("Fitting reference scalers on training split only...")
    scaler_grid, scaler_global, scaler_y = fit_scalers_on_training(raw_data)
    save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y)

    all_metrics = []
    all_predictions = {}
    all_histories = []

    for horizon_id in cfg.horizon_steps:
        model, summary_df, prediction_records, history_df = train_one_horizon(
            cfg=cfg,
            raw_data=raw_data,
            horizon_id=horizon_id
        )

        all_metrics.append(summary_df)
        all_predictions[horizon_id] = prediction_records
        all_histories.append(history_df)

        del model

    save_combined_outputs(cfg, all_metrics, all_predictions)

    history_all_df = pd.concat(all_histories, axis=0, ignore_index=True)
    history_all_df.to_excel(
        os.path.join(cfg.output_dir, "training_history_all_horizons.xlsx"),
        index=False
    )

    metrics_all_df = pd.concat(all_metrics, axis=0, ignore_index=True)

    print("\nAll results saved to:")
    print(cfg.output_dir)

    print("\nFinal Test summary:")
    cols = [
        "Horizon_step",
        "Lead_hours",
        "Effective_sample_size",
        "RMSE",
        "MAE",
        "MAPE_percent",
        "R2",
        "PICP_percent",
        "PINAW",
        "CRPS",
        "NLL",
    ]
    print(metrics_all_df[metrics_all_df["Split"] == "Test"][cols])

    return metrics_all_df


if __name__ == "__main__":
    set_seed(cfg.seed)
    summary_df = train_all_horizons(cfg)
