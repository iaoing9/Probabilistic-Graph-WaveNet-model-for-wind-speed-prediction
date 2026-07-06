import os
import math
import random
import numpy as np
import pandas as pd

from scipy.stats import norm

import torch

try:
    from statsmodels.tsa.api import VAR
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for the VAR baseline. "
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

    output_dir = os.path.join(root_dir, "VAR_Dataset1_Fold1_results")

    seed = 2026

    # GPU setting retained only for interface consistency with the GWN script.
    # The VAR baseline is fitted by statsmodels on CPU.
    use_cuda_if_available = True
    require_cuda = False
    device = "cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu"

    # Time resolution
    # Dataset1 uses 10 min samples.
    samples_per_hour = 6

    # In the manuscript, one prediction step is defined as 1 h ahead.
    # Only the following selected horizons are trained in this version.
    # horizon_id = 1, 6, 12, 24 means 1 h, 6 h, 12 h, and 24 h ahead.
    horizon_steps = [1, 6, 12, 24]
    max_horizon_step = max(horizon_steps)

    # Input window length. The default 24 h window corresponds to 144 rows.
    input_window_hours = 24
    input_len = input_window_hours * samples_per_hour

    # Maximum lead time used for common truncation across the selected horizons.
    # This keeps the effective sample size identical for all selected horizons.
    max_lead_steps = max_horizon_step * samples_per_hour

    # Sliding-window moving step.
    # stride = 1 means the window moves by one sample each time.
    stride = 1

    # Window alignment mode:
    # "same_input_window": all horizons use the same input-window starting positions.
    # "same_target_time": all horizons are evaluated on exactly the same target timestamps.
    window_alignment = "same_input_window"

    # Grid and feature settings
    num_nodes = 9
    num_grid_features = 3
    num_global_features = 1
    input_features = num_grid_features + num_global_features

    # VAR baseline.
    # To avoid over-parameterization, the VAR model does not directly use all 27 ECMWF grid columns.
    # It uses:
    # 1 target bridge-site wind speed
    # 3 ECMWF variables averaged over the 9 grid nodes
    # Total endogenous variables = 4
    var_lag_order = 2
    var_trend = "c"
    update_with_observations = True

    # Training-related settings are retained only for output consistency.
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

    # If True, save model information for each horizon.
    save_model = True

    # DataLoader settings retained only for interface consistency.
    num_workers = 0
    pin_memory = True if device == "cuda" else False

    # Shuffling is disabled for VAR because chronological order must be preserved.
    shuffle_train = False


cfg = Config()


# =========================================================
# 2. Reproducibility
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 3. Coordinates and adjacency matrix
# =========================================================
# These definitions are retained from the GWN code for output-file consistency.
# The VAR baseline does not use graph convolution.

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

    Inputs retained from the GWN workflow:
    x_grid   = B:AB, reshaped to [T, 9 nodes, 3 grid features]
    x_global = AC, historical observed bridge-site wind speed, [T, 1]
    y        = AC, target observed bridge-site wind speed, [T, 1]
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


def build_var_endog(x_grid_raw, y_raw):
    """
    Build a low-dimensional endogenous matrix for VAR.

    Variables:
    0       target bridge-site observed wind speed
    1       spatial average of ECMWF 10 m mean wind speed over the 9 grid nodes
    2       spatial average of ECMWF 100 m mean wind speed over the 9 grid nodes
    3       spatial average of ECMWF 10 m gust wind speed over the 9 grid nodes

    The historical bridge-site wind speed is not added as a separate endogenous variable,
    because it is identical to the target observed wind-speed series in Dataset1.
    Adding it again would create a duplicate variable and may cause singularity in VAR.
    """
    y = y_raw.reshape(-1, 1).astype(np.float64)
    ecmwf_mean = np.mean(x_grid_raw.astype(np.float64), axis=1)
    endog = np.concatenate([y, ecmwf_mean], axis=1)

    names = [
        "Target_ANE_observed_wind_speed",
        "ECMWF_10m_mean_wind_speed_mean_9nodes",
        "ECMWF_100m_mean_wind_speed_mean_9nodes",
        "ECMWF_10m_gust_wind_speed_mean_9nodes",
    ]

    return endog, names


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


def build_sliding_window_metadata(
    y,
    time,
    input_len,
    lead_steps,
    max_lead_steps,
    stride=1,
    alignment="same_input_window"
):
    """
    Build only sample metadata and target values for one chronological split.
    This keeps the same sample indexing logic as the GWN sliding-window code,
    while avoiding construction of unnecessary high-dimensional input tensors.
    """
    n = len(y)

    if n != len(time):
        raise ValueError("y and time must have the same first dimension.")

    if input_len <= 0 or lead_steps <= 0 or max_lead_steps <= 0:
        raise ValueError("input_len, lead_steps, and max_lead_steps must be positive.")

    if lead_steps > max_lead_steps:
        raise ValueError("lead_steps must not be larger than max_lead_steps.")

    if n < input_len + max_lead_steps:
        raise ValueError(
            f"Split length {n} is too short for input_len={input_len} and max_lead_steps={max_lead_steps}."
        )

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

            y_list.append(y[target_idx])
            target_time_list.append(time[target_idx])
            input_start_time_list.append(time[start])
            input_end_time_list.append(time[end])
            input_start_index_list.append(start)
            input_end_index_list.append(end)
            target_index_list.append(target_idx)

    else:
        raise ValueError("alignment must be 'same_input_window' or 'same_target_time'.")

    yy = np.stack(y_list, axis=0).astype(np.float32)

    sample_meta = {
        "target_time": np.array(target_time_list),
        "input_start_time": np.array(input_start_time_list),
        "input_end_time": np.array(input_end_time_list),
        "input_start_index": np.array(input_start_index_list, dtype=np.int64),
        "input_end_index": np.array(input_end_index_list, dtype=np.int64),
        "target_index": np.array(target_index_list, dtype=np.int64),
    }

    return yy, sample_meta


def load_raw_splits(cfg):
    train_path = os.path.join(cfg.root_dir, cfg.train_file)
    val_path = os.path.join(cfg.root_dir, cfg.val_file)
    test_path = os.path.join(cfg.root_dir, cfg.test_file)

    train_time, train_x_grid_raw, train_x_global_raw, train_y_raw, train_df = read_dataset1_excel(train_path)
    val_time, val_x_grid_raw, val_x_global_raw, val_y_raw, val_df = read_dataset1_excel(val_path)
    test_time, test_x_grid_raw, test_x_global_raw, test_y_raw, test_df = read_dataset1_excel(test_path)

    train_endog, endog_names = build_var_endog(train_x_grid_raw, train_y_raw)
    val_endog, _ = build_var_endog(val_x_grid_raw, val_y_raw)
    test_endog, _ = build_var_endog(test_x_grid_raw, test_y_raw)

    return {
        "Train": {
            "time": train_time,
            "x_grid_raw": train_x_grid_raw,
            "x_global_raw": train_x_global_raw,
            "y_raw": train_y_raw,
            "endog": train_endog,
            "df": train_df,
        },
        "Validation": {
            "time": val_time,
            "x_grid_raw": val_x_grid_raw,
            "x_global_raw": val_x_global_raw,
            "y_raw": val_y_raw,
            "endog": val_endog,
            "df": val_df,
        },
        "Test": {
            "time": test_time,
            "x_grid_raw": test_x_grid_raw,
            "x_global_raw": test_x_global_raw,
            "y_raw": test_y_raw,
            "endog": test_endog,
            "df": test_df,
        },
        "Endog_names": endog_names,
    }


def fit_scalers_on_training(raw_data):
    scaler_grid = StandardScaler()
    scaler_global = StandardScaler()
    scaler_y = StandardScaler()

    train = raw_data["Train"]

    # Fit only on the training split to avoid normalization leakage.
    # The VAR baseline itself is fitted in the original physical scale.
    # These scalers are retained only for output consistency with the GWN workflow.
    scaler_grid.fit(train["x_grid_raw"], axis=(0, 1))
    scaler_global.fit(train["x_global_raw"], axis=0)
    scaler_y.fit(train["y_raw"], axis=0)

    return scaler_grid, scaler_global, scaler_y


def prepare_metadata_for_horizon(cfg, raw_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour

    target_values = {}
    sample_metas = {}

    for split_name in ["Train", "Validation", "Test"]:
        data = raw_data[split_name]
        y, sample_meta = build_sliding_window_metadata(
            y=data["y_raw"],
            time=data["time"],
            input_len=cfg.input_len,
            lead_steps=lead_steps,
            max_lead_steps=cfg.max_lead_steps,
            stride=cfg.stride,
            alignment=cfg.window_alignment,
        )

        target_values[split_name] = y.reshape(-1)
        sample_metas[split_name] = sample_meta

    return target_values, sample_metas


# =========================================================
# 5. Probabilistic VAR baseline
# =========================================================

class ProbabilisticVAR:
    def __init__(
        self,
        lag_order=2,
        trend="c",
        eps_var=1e-6,
        update_with_observations=True,
        target_index=0,
    ):
        self.lag_order = lag_order
        self.trend = trend
        self.eps_var = eps_var
        self.update_with_observations = update_with_observations
        self.target_index = target_index

        self.result = None
        self.aic = None
        self.bic = None
        self.hqic = None
        self.fpe = None
        self.neqs = None
        self.k_ar = None
        self._forecast_var_cache = {}

    def fit(self, endog_train):
        endog_train = np.asarray(endog_train, dtype=np.float64)

        model = VAR(endog_train)
        self.result = model.fit(
            maxlags=self.lag_order,
            ic=None,
            trend=self.trend
        )

        self.aic = float(self.result.aic)
        self.bic = float(self.result.bic)
        self.hqic = float(self.result.hqic)
        self.fpe = float(self.result.fpe)
        self.neqs = int(self.result.neqs)
        self.k_ar = int(self.result.k_ar)

        return self

    def _target_forecast_variance(self, steps):
        if steps not in self._forecast_var_cache:
            cov_all = self.result.forecast_cov(steps=steps)
            var = float(cov_all[steps - 1, self.target_index, self.target_index])
            self._forecast_var_cache[steps] = max(var, self.eps_var)
        return self._forecast_var_cache[steps]

    def _forecast_from_history(self, history, steps):
        if self.result is None:
            raise RuntimeError("The VAR model must be fitted before prediction.")

        history = np.asarray(history, dtype=np.float64)

        if history.shape[0] < self.k_ar:
            raise ValueError(
                f"At least {self.k_ar} observations are required for VAR forecasting, "
                f"but only {history.shape[0]} were provided."
            )

        mean_all = self.result.forecast(
            y=history[-self.k_ar:],
            steps=steps
        )

        mu = float(mean_all[steps - 1, self.target_index])
        var = self._target_forecast_variance(steps)

        return mu, var

    def predict_samples(self, split_name, raw_data, sample_meta, lead_steps):
        """
        Forecast one value for each sliding-window sample.

        For each sample, the VAR history ends at Input_end_index_0based.
        Thus the target observation is never used as input for that sample.
        Validation and test histories are initialized chronologically using
        preceding splits, consistent with Train -> Validation -> Test ordering.
        """
        if split_name == "Train":
            prefix_history = None
            split_endog = raw_data["Train"]["endog"]
        elif split_name == "Validation":
            prefix_history = raw_data["Train"]["endog"]
            split_endog = raw_data["Validation"]["endog"]
        elif split_name == "Test":
            prefix_history = np.vstack([
                raw_data["Train"]["endog"],
                raw_data["Validation"]["endog"],
            ])
            split_endog = raw_data["Test"]["endog"]
        else:
            raise ValueError("split_name must be 'Train', 'Validation', or 'Test'.")

        target_indices = sample_meta["target_index"]
        input_end_indices = sample_meta["input_end_index"]

        y = split_endog[target_indices, self.target_index].astype(np.float32)
        mu = np.zeros(len(target_indices), dtype=np.float32)
        var = np.zeros(len(target_indices), dtype=np.float32)

        for i, input_end_idx in enumerate(input_end_indices):
            split_history = split_endog[:input_end_idx + 1]
            if prefix_history is None:
                history = split_history
            else:
                history = np.vstack([prefix_history, split_history])

            mu_i, var_i = self._forecast_from_history(history, steps=lead_steps)
            mu[i] = mu_i
            var[i] = var_i

        return y, mu, var


# =========================================================
# 6. Loss and metrics
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
# 7. Training and evaluation
# =========================================================

def train_one_horizon(cfg, raw_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    lead_hours = horizon_id

    horizon_dir = os.path.join(cfg.output_dir, f"Horizon_{horizon_id:02d}_step_{lead_hours:02d}h_ahead")
    os.makedirs(horizon_dir, exist_ok=True)

    set_seed(cfg.seed + horizon_id)

    target_values, sample_metas = prepare_metadata_for_horizon(
        cfg=cfg,
        raw_data=raw_data,
        horizon_id=horizon_id
    )

    model = ProbabilisticVAR(
        lag_order=cfg.var_lag_order,
        trend=cfg.var_trend,
        eps_var=cfg.eps_var,
        update_with_observations=cfg.update_with_observations,
        target_index=0,
    )
    model.fit(raw_data["Train"]["endog"])

    best_epoch = 0
    best_val_loss = np.nan

    var_info_df = pd.DataFrame([{
        "Model": "VAR",
        "Horizon_step": horizon_id,
        "Lead_hours": lead_hours,
        "Lead_samples": lead_steps,
        "lag_order_requested": cfg.var_lag_order,
        "lag_order_used": model.k_ar,
        "trend": cfg.var_trend,
        "num_endogenous_variables": model.neqs,
        "AIC": model.aic,
        "BIC": model.bic,
        "HQIC": model.hqic,
        "FPE": model.fpe,
        "update_with_observations": cfg.update_with_observations,
        "note": "VAR coefficients are fitted on the training endogenous series. Sliding-window forecasts use observations up to the input-window end only."
    }])
    var_info_df.to_excel(
        os.path.join(horizon_dir, f"var_model_info_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    endog_info_df = pd.DataFrame({
        "Variable_index": np.arange(len(raw_data["Endog_names"])),
        "Variable_name": raw_data["Endog_names"],
    })
    endog_info_df.to_excel(
        os.path.join(horizon_dir, f"var_endogenous_variables_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    history_df = pd.DataFrame([{
        "Horizon_step": horizon_id,
        "Lead_hours": lead_hours,
        "Lead_samples": lead_steps,
        "Epoch": 0,
        "Train_GNLL": np.nan,
        "Validation_GNLL": np.nan,
        "note": "Probabilistic VAR baseline fitted on the training multivariate wind-speed-related series."
    }])
    history_df.to_excel(
        os.path.join(horizon_dir, f"training_history_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    print("\n" + "=" * 80)
    print(f"Evaluating VAR Horizon {horizon_id:02d}: {lead_hours} h ahead, lead_steps = {lead_steps} samples")
    print(f"Input length = {cfg.input_len} samples ({cfg.input_window_hours} h)")
    print(f"Window alignment = {cfg.window_alignment}, stride = {cfg.stride}")
    print(f"VAR lag order used = {model.k_ar}, endogenous variables = {model.neqs}")
    for split_name in ["Train", "Validation", "Test"]:
        print(f"{split_name} samples: {len(target_values[split_name])}")
    print("=" * 80)

    result_summary = []
    prediction_records = {}

    for split_name in ["Train", "Validation", "Test"]:
        sample_meta = sample_metas[split_name]

        y_flat, mu_flat, var_flat = model.predict_samples(
            split_name=split_name,
            raw_data=raw_data,
            sample_meta=sample_meta,
            lead_steps=lead_steps,
        )

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
        metrics["Raw_split_size"] = len(raw_data[split_name]["y_raw"])
        metrics["Effective_sample_size"] = len(y_flat)
        metrics["Input_window_samples"] = cfg.input_len
        metrics["Input_window_hours"] = cfg.input_window_hours
        metrics["Max_lead_samples_for_common_truncation"] = cfg.max_lead_steps
        metrics["Window_alignment"] = cfg.window_alignment
        metrics["Best_epoch"] = best_epoch
        metrics["Best_validation_GNLL"] = best_val_loss

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
            if isinstance(v, float) and not np.isnan(v):
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
    for split_name in ["Train", "Validation", "Test"]:
        data = raw_data[split_name]
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
            "VAR_endogenous_variables": str(raw_data["Endog_names"]),
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
            "ECMWF_10m_mean_wind_speed_mean_9nodes",
            "ECMWF_100m_mean_wind_speed_mean_9nodes",
            "ECMWF_10m_gust_wind_speed_mean_9nodes",
        ],
        "Excel_columns": [
            "AC",
            "B:J averaged over P1-P9",
            "K:S averaged over P1-P9",
            "T:AB averaged over P1-P9",
        ],
        "Used_as": [
            "VAR endogenous variable and prediction target",
            "VAR endogenous variable",
            "VAR endogenous variable",
            "VAR endogenous variable",
        ],
    })
    feature_info.to_excel(
        os.path.join(cfg.output_dir, "var_feature_definition.xlsx"),
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


def train_all_horizons(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Using device: CPU for statsmodels VAR baseline")
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
