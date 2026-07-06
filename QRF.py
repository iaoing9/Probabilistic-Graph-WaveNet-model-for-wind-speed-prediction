import os
import math
import random
import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# 1. Configuration
# =========================================================

class Config:
    root_dir = r"E:\LHQ_E3\退稿返修\Case 1 Sutong Bridge\Dataset1"

    train_file = "A_Train_Dataset1.xlsx"
    val_file = "B_Validation_Dataset1.xlsx"
    test_file = "C_Test_Dataset1.xlsx"

    output_dir = os.path.join(root_dir, "QRF_Dataset1_results")

    seed = 2026

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
    # This keeps the effective sample size identical for 1, 6, 12, and 24 steps.
    max_lead_steps = max_horizon_step * samples_per_hour

    # Sliding-window moving step.
    # stride = 1 means the window moves by one 10 min sample each time.
    stride = 1

    # Window alignment mode:
    # "same_input_window": all horizons use the same input-window starting positions.
    #   This matches the example:
    #   1-step: input rows 1-144, target row 150;
    #   24-step: input rows 1-144, target row 288.
    # "same_target_time": all horizons are evaluated on exactly the same target timestamps.
    #   This is stricter for horizon-wise comparison, but it does not match the row example above.
    window_alignment = "same_input_window"

    # Grid and feature settings
    num_nodes = 9
    num_grid_features = 3
    num_global_features = 1
    input_features = num_grid_features + num_global_features

    # QRF settings.
    # The QRF baseline treats each sliding-window sample as one tabular sample.
    # For each window, all temporal grid-node features and historical ANE features
    # are flattened into a single feature vector.
    n_estimators = 500
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 5
    max_features = "sqrt"
    bootstrap = True
    n_jobs = -1

    # Probabilistic prediction
    interval_alpha = 0.05
    lower_quantile = interval_alpha / 2.0
    upper_quantile = 1.0 - interval_alpha / 2.0
    z_value = 1.959963984540054

    # Small positive value for variance-based compatibility metrics.
    eps_var = 1e-8


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
    A       : Time, 10 min resolution
    B:J     : ECMWF 10 m mean wind speed at 9 grid nodes
    K:S     : ECMWF 100 m mean wind speed at 9 grid nodes
    T:AB    : ECMWF 10 m gust wind speed at 9 grid nodes
    AC      : Observed bridge-site wind speed from anemometer

    Inputs:
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

    x_grid   : [N, num_nodes, num_grid_features]
    x_global : [N, num_global_features]
    y        : [N, 1]
    time     : [N]

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


def flatten_windows_for_qrf(x_grid, x_global):
    """
    Convert sliding-window samples into tabular samples for QRF.

    x_grid:   [S, T, N, F_grid]
    x_global: [S, T, F_global]

    Return:
    x: [S, T * (N * F_grid + F_global)]
    """
    s, t, n, f_grid = x_grid.shape

    x_grid_flat = x_grid.reshape(s, t, n * f_grid)
    x = np.concatenate([x_grid_flat, x_global], axis=-1)
    x = x.reshape(s, t * (n * f_grid + x_global.shape[-1])).astype(np.float32)

    return x


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
    scaler_grid.fit(train["x_grid_raw"], axis=(0, 1))
    scaler_global.fit(train["x_global_raw"], axis=0)
    scaler_y.fit(train["y_raw"], axis=0)

    return scaler_grid, scaler_global, scaler_y


def transform_raw_splits(raw_data, scaler_grid, scaler_global, scaler_y):
    transformed = {}

    for split_name, data in raw_data.items():
        transformed[split_name] = {
            "time": data["time"],
            "x_grid": scaler_grid.transform(data["x_grid_raw"]),
            "x_global": scaler_global.transform(data["x_global_raw"]),
            "y": scaler_y.transform(data["y_raw"]),
            "y_raw": data["y_raw"],
            "raw_size": len(data["y_raw"]),
        }

    return transformed


def prepare_data_for_horizon(cfg, transformed_data, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour

    qrf_data = {}
    sample_metas = {}

    for split_name, data in transformed_data.items():
        x_grid, x_global, y, sample_meta = build_sliding_windows(
            x_grid=data["x_grid"],
            x_global=data["x_global"],
            y=data["y"],
            time=data["time"],
            input_len=cfg.input_len,
            lead_steps=lead_steps,
            max_lead_steps=cfg.max_lead_steps,
            stride=cfg.stride,
            alignment=cfg.window_alignment,
        )

        x_tabular = flatten_windows_for_qrf(x_grid, x_global)
        y_tabular = y.reshape(-1).astype(np.float32)

        qrf_data[split_name] = {
            "x": x_tabular,
            "y": y_tabular,
        }
        sample_metas[split_name] = sample_meta

    return qrf_data, sample_metas


# =========================================================
# 5. Quantile Random Forest
# =========================================================

class QuantileRandomForest:
    def __init__(
        self,
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=2026,
    ):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        self.leaf_y_by_tree = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        self.rf.fit(x_train, self.y_train)

        train_leaf = self.rf.apply(x_train)
        self.leaf_y_by_tree = []

        for tree_idx in range(train_leaf.shape[1]):
            leaf_map = {}
            leaves = train_leaf[:, tree_idx]

            for leaf_id, y_value in zip(leaves, self.y_train):
                if leaf_id not in leaf_map:
                    leaf_map[leaf_id] = []
                leaf_map[leaf_id].append(float(y_value))

            for leaf_id in leaf_map:
                leaf_map[leaf_id] = np.asarray(leaf_map[leaf_id], dtype=np.float32)

            self.leaf_y_by_tree.append(leaf_map)

        return self

    def _distribution_for_one_sample(self, sample_leaf_ids):
        values = []

        for tree_idx, leaf_id in enumerate(sample_leaf_ids):
            leaf_map = self.leaf_y_by_tree[tree_idx]
            leaf_values = leaf_map.get(leaf_id)

            if leaf_values is not None and leaf_values.size > 0:
                values.append(leaf_values)

        if len(values) == 0:
            # This should almost never happen because each tree must assign
            # the sample to one of its terminal leaves. The fallback avoids
            # runtime failure in pathological cases.
            return self.y_train

        return np.concatenate(values, axis=0)

    def predict_distribution(self, x):
        pred_leaf = self.rf.apply(x)

        distributions = []
        for i in range(pred_leaf.shape[0]):
            distributions.append(self._distribution_for_one_sample(pred_leaf[i]))

        return distributions

    def predict(self, x, lower_q=0.025, upper_q=0.975, eps_var=1e-8):
        distributions = self.predict_distribution(x)

        mean = np.zeros(len(distributions), dtype=np.float32)
        median = np.zeros(len(distributions), dtype=np.float32)
        lower = np.zeros(len(distributions), dtype=np.float32)
        upper = np.zeros(len(distributions), dtype=np.float32)
        var = np.zeros(len(distributions), dtype=np.float32)

        for i, values in enumerate(distributions):
            mean[i] = np.mean(values)
            median[i] = np.quantile(values, 0.50)
            lower[i] = np.quantile(values, lower_q)
            upper[i] = np.quantile(values, upper_q)
            var[i] = np.var(values) + eps_var

        return mean, median, lower, upper, var, distributions


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


def empirical_crps_one(y_true, samples):
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)

    if samples.size == 0:
        return np.nan

    term_1 = np.mean(np.abs(samples - y_true))

    samples_sorted = np.sort(samples)
    n = samples_sorted.size
    idx = np.arange(1, n + 1, dtype=np.float64)

    # mean_{i,j}|x_i - x_j| computed without constructing the n x n matrix.
    pairwise_sum = np.sum((2.0 * idx - n - 1.0) * samples_sorted)
    mean_pairwise_abs = 2.0 * pairwise_sum / (n ** 2)

    return float(term_1 - 0.5 * mean_pairwise_abs)


def empirical_crps(y_true, distributions):
    values = [
        empirical_crps_one(y_i, dist_i)
        for y_i, dist_i in zip(y_true.reshape(-1), distributions)
    ]
    return float(np.nanmean(values))


def gaussian_nll_np(y_true, mu, var):
    var = np.maximum(var, 1e-8)
    nll = 0.5 * (np.log(2.0 * np.pi * var) + (y_true - mu) ** 2 / var)
    return float(np.mean(nll))


def compute_all_metrics(y_true, mean, median, var, lower, upper, distributions):
    sigma = np.sqrt(np.maximum(var, 1e-12))

    metrics = {
        "RMSE": rmse(y_true, mean),
        "MAE": mae(y_true, mean),
        "MAPE_percent": mape(y_true, mean),
        "R2": r2_score_np(y_true, mean),
        "PICP_percent": picp(y_true, lower, upper),
        "PINAW": pinaw(y_true, lower, upper),
        "CRPS": empirical_crps(y_true, distributions),
        "NLL": gaussian_nll_np(y_true, mean, var),
        "GaussianApprox_CRPS": gaussian_crps(y_true, mean, sigma),
        "Median_RMSE": rmse(y_true, median),
        "Median_MAE": mae(y_true, median),
    }

    return metrics


# =========================================================
# 7. Inverse transformation
# =========================================================

def inverse_y(y_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    y_mean = float(np.squeeze(scaler_y.mean))
    return y_norm * y_std + y_mean


def inverse_var(var_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    return var_norm * (y_std ** 2)


def inverse_distribution_list(distributions_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    y_mean = float(np.squeeze(scaler_y.mean))

    distributions = []
    for values in distributions_norm:
        distributions.append(values * y_std + y_mean)

    return distributions


# =========================================================
# 8. Training and evaluation
# =========================================================

def train_one_horizon(cfg, transformed_data, scaler_y, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    lead_hours = horizon_id

    horizon_dir = os.path.join(cfg.output_dir, f"Horizon_{horizon_id:02d}_step_{lead_hours:02d}h_ahead")
    os.makedirs(horizon_dir, exist_ok=True)

    set_seed(cfg.seed + horizon_id)

    qrf_data, sample_metas = prepare_data_for_horizon(
        cfg=cfg,
        transformed_data=transformed_data,
        horizon_id=horizon_id
    )

    train_x = qrf_data["Train"]["x"]
    train_y = qrf_data["Train"]["y"]

    model = QuantileRandomForest(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        bootstrap=cfg.bootstrap,
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed + horizon_id,
    )

    print("\n" + "=" * 80)
    print(f"Training Horizon {horizon_id:02d}: {lead_hours} h ahead, lead_steps = {lead_steps} samples")
    print(f"Input length = {cfg.input_len} samples ({cfg.input_window_hours} h)")
    print(f"Window alignment = {cfg.window_alignment}, stride = {cfg.stride}")
    print(f"QRF input dimension = {train_x.shape[1]}")
    for split_name in ["Train", "Validation", "Test"]:
        print(f"{split_name} samples: {qrf_data[split_name]['x'].shape[0]}")
    print("=" * 80)

    print("Training Quantile Random Forest...")
    model.fit(train_x, train_y)
    print("QRF training completed.")

    training_info_df = pd.DataFrame({
        "Item": [
            "Model",
            "Horizon_step",
            "Lead_hours",
            "Lead_samples",
            "Input_window_samples",
            "Input_window_hours",
            "Window_alignment",
            "QRF_input_dimension",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "random_state",
        ],
        "Value": [
            "Quantile Random Forest",
            horizon_id,
            lead_hours,
            lead_steps,
            cfg.input_len,
            cfg.input_window_hours,
            cfg.window_alignment,
            train_x.shape[1],
            cfg.n_estimators,
            str(cfg.max_depth),
            cfg.min_samples_split,
            cfg.min_samples_leaf,
            str(cfg.max_features),
            cfg.bootstrap,
            cfg.seed + horizon_id,
        ],
    })
    training_info_df.to_excel(
        os.path.join(horizon_dir, f"qrf_training_information_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    feature_importance_df = pd.DataFrame({
        "Feature_index": np.arange(train_x.shape[1]) + 1,
        "Importance": model.rf.feature_importances_,
    })
    feature_importance_df.to_excel(
        os.path.join(horizon_dir, f"feature_importance_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    result_summary = []
    prediction_records = {}

    for split_name in ["Train", "Validation", "Test"]:
        x = qrf_data[split_name]["x"]
        y_norm = qrf_data[split_name]["y"]

        mean_norm, median_norm, lower_norm, upper_norm, var_norm, distributions_norm = model.predict(
            x,
            lower_q=cfg.lower_quantile,
            upper_q=cfg.upper_quantile,
            eps_var=cfg.eps_var,
        )

        y = inverse_y(y_norm, scaler_y)
        mean = inverse_y(mean_norm, scaler_y)
        median = inverse_y(median_norm, scaler_y)
        lower = inverse_y(lower_norm, scaler_y)
        upper = inverse_y(upper_norm, scaler_y)
        var = inverse_var(var_norm, scaler_y)
        distributions = inverse_distribution_list(distributions_norm, scaler_y)

        y_flat = y.reshape(-1)
        mean_flat = mean.reshape(-1)
        median_flat = median.reshape(-1)
        var_flat = var.reshape(-1)
        lower_flat = lower.reshape(-1)
        upper_flat = upper.reshape(-1)

        metrics = compute_all_metrics(
            y_true=y_flat,
            mean=mean_flat,
            median=median_flat,
            var=var_flat,
            lower=lower_flat,
            upper=upper_flat,
            distributions=distributions,
        )

        metrics["Horizon_step"] = horizon_id
        metrics["Lead_hours"] = lead_hours
        metrics["Lead_samples"] = lead_steps
        metrics["Split"] = split_name
        metrics["Raw_split_size"] = transformed_data[split_name]["raw_size"]
        metrics["Effective_sample_size"] = len(y_flat)
        metrics["Input_window_samples"] = cfg.input_len
        metrics["Input_window_hours"] = cfg.input_window_hours
        metrics["Max_lead_samples_for_common_truncation"] = cfg.max_lead_steps
        metrics["Window_alignment"] = cfg.window_alignment
        metrics["Best_epoch"] = 0
        metrics["Best_validation_GNLL"] = np.nan

        result_summary.append(metrics)

        sigma_flat = np.sqrt(np.maximum(var_flat, 1e-12))
        sample_meta = sample_metas[split_name]

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
            "Predicted_mean": mean_flat,
            "Predicted_median": median_flat,
            "Predicted_variance": var_flat,
            "Predicted_std": sigma_flat,
            "Lower_95PI": lower_flat,
            "Upper_95PI": upper_flat,
            "Residual": y_flat - mean_flat,
            "Residual_median": y_flat - median_flat,
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
        "GaussianApprox_CRPS",
        "Median_RMSE",
        "Median_MAE",
    ]
    summary_df = summary_df[cols]

    summary_df.to_excel(
        os.path.join(horizon_dir, f"metrics_summary_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    return model, summary_df, prediction_records, training_info_df


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
        os.path.join(cfg.output_dir, "training_only_scaler_information.xlsx"),
        index=False
    )

    feature_info = pd.DataFrame({
        "Feature_group": [
            "ECMWF_10m_mean_wind_speed",
            "ECMWF_100m_mean_wind_speed",
            "ECMWF_10m_gust_wind_speed",
            "ANE_historical_observed_wind_speed",
            "Target_ANE_observed_wind_speed",
        ],
        "Excel_columns": [
            "B:J",
            "K:S",
            "T:AB",
            "AC",
            "AC shifted according to horizon",
        ],
        "Used_as": [
            "grid input",
            "grid input",
            "grid input",
            "global historical input",
            "prediction target",
        ],
    })
    feature_info.to_excel(
        os.path.join(cfg.output_dir, "feature_definition.xlsx"),
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
        # Since common truncation is used, all horizons have the same number of samples.
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
                "Predicted_median",
                "Predicted_std",
                "Lower_95PI",
                "Upper_95PI",
                "Residual",
                "Residual_median",
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

    print(f"Selected horizon steps: {cfg.horizon_steps}")
    print("Loading Dataset1...")
    raw_data = load_raw_splits(cfg)

    print("Fitting scalers on training split only...")
    scaler_grid, scaler_global, scaler_y = fit_scalers_on_training(raw_data)

    print("Transforming train, validation, and test splits using training-only scalers...")
    transformed_data = transform_raw_splits(raw_data, scaler_grid, scaler_global, scaler_y)

    save_static_information(cfg, raw_data, scaler_grid, scaler_global, scaler_y)

    all_metrics = []
    all_predictions = {}
    all_training_info = []

    for horizon_id in cfg.horizon_steps:
        model, summary_df, prediction_records, training_info_df = train_one_horizon(
            cfg=cfg,
            transformed_data=transformed_data,
            scaler_y=scaler_y,
            horizon_id=horizon_id
        )

        all_metrics.append(summary_df)
        all_predictions[horizon_id] = prediction_records
        all_training_info.append(training_info_df.assign(Horizon_step=horizon_id))

        del model

    save_combined_outputs(cfg, all_metrics, all_predictions)

    training_info_all_df = pd.concat(all_training_info, axis=0, ignore_index=True)
    training_info_all_df.to_excel(
        os.path.join(cfg.output_dir, "qrf_training_information_all_horizons.xlsx"),
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
