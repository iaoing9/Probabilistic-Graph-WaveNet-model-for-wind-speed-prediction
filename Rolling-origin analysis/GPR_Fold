import os
import math
import random
import pickle
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist

from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Configuration
# =========================================================

class Config:
    root_dir = r"E:\LHQ_E3\退稿返修\Case 1 Sutong Bridge\Dataset1\滚动起点分析\Fold1"

    train_file = "A_Train_Dataset1_Fold1.xlsx"
    val_file = "B_Validation_Dataset1_Fold1.xlsx"
    test_file = "C_Test_Dataset1_Fold1.xlsx"

    output_dir = os.path.join(root_dir, "GPR_Dataset1_Fold1_results")

    seed = 2026

    # GPU setting. The code will use CUDA automatically when a CUDA-enabled
    # NVIDIA GPU and a CUDA-version PyTorch installation are available.
    # Set require_cuda = True if you want the script to stop when CUDA is unavailable.
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

    # SGPR settings
    # Exact GPR has O(n^3) training complexity and O(n^2) memory complexity.
    # SGPR reduces the main matrix operations to the inducing-point scale M.
    # Larger M improves accuracy but increases cost approximately as O(n M^2 + M^3).
    num_inducing = 512
    inducing_selection = "random"  # "random" or "kmeans"

    # If True, a small exact GPR is fitted only on the inducing subset to obtain
    # reasonable kernel hyperparameters. This avoids exact GPR on all training samples.
    optimize_kernel_on_inducing_subset = True
    n_restarts_optimizer = 0

    # If optimize_kernel_on_inducing_subset is False, these values are used.
    signal_variance = 1.0
    length_scale = None  # None uses a median-distance heuristic on inducing points.
    noise_variance = 1e-2

    # Kernel hyperparameter bounds used only for inducing-subset optimization.
    constant_value = 1.0
    constant_bounds = (1e-3, 1e3)
    length_scale_initial = 1.0
    length_scale_bounds = (1e-2, 1e2)
    noise_level = 1e-2
    noise_level_bounds = (1e-6, 1e1)
    alpha = 1e-8

    # Numerical jitter for Cholesky factorization.
    jitter = 1e-6

    # Training
    batch_size = 16
    epochs = 150
    learning_rate = 1e-3
    weight_decay = 1e-5
    patience = 9999
    min_delta = 1e-5
    grad_clip = 5.0

    # Probabilistic prediction
    eps_var = 1e-6
    interval_alpha = 0.05
    z_value = 1.959963984540054

    # If True, save the best model file for each horizon.
    save_model = True

    # DataLoader settings. For Windows, num_workers = 0 is the safest choice.
    # If your GPU utilization is low, you can try num_workers = 2 or 4.
    num_workers = 0
    pin_memory = True if device == "cuda" else False

    # Training-window shuffling is acceptable because samples are generated
    # after chronological split and no window crosses split boundaries.
    shuffle_train = True


cfg = Config()


# =========================================================
# 2. Reproducibility
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        # benchmark=True can accelerate fixed-size convolution on GPU.
        # deterministic=False is faster; set it to True only when strict bitwise reproducibility is required.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


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


class WindWindowDataset(Dataset):
    def __init__(self, x_grid, x_global, y):
        self.x_grid = torch.tensor(x_grid, dtype=torch.float32)
        self.x_global = torch.tensor(x_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x_grid[idx], self.x_global[idx], self.y[idx]


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

    datasets = {}
    loaders = {}
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

        ds = WindWindowDataset(x_grid, x_global, y)

        shuffle = cfg.shuffle_train if split_name == "Train" else False
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )

        datasets[split_name] = ds
        loaders[split_name] = loader
        sample_metas[split_name] = sample_meta

    return datasets, loaders, sample_metas


# =========================================================
# 5. Probabilistic Sparse Gaussian Process Regression
# =========================================================

def flatten_windows_for_sgpr(x_grid, x_global):
    """
    Convert sliding-window samples into tabular samples for SGPR.

    x_grid:   [S, T, N, F_grid]
    x_global: [S, T, F_global]

    Return:
    x: [S, T * (N * F_grid + F_global)]

    The SSGPR baseline treats the 9 ECMWF grid-point variables and the historical
    bridge-site wind speed as ordinary tabular predictors. It does not explicitly
    model graph-based spatial dependence.
    """
    if hasattr(x_grid, "detach"):
        x_grid = x_grid.detach().cpu().numpy()
    if hasattr(x_global, "detach"):
        x_global = x_global.detach().cpu().numpy()

    s, t, n, f_grid = x_grid.shape
    x_grid_flat = x_grid.reshape(s, t, n * f_grid)
    x = np.concatenate([x_grid_flat, x_global], axis=-1)
    x = x.reshape(s, -1).astype(np.float64)

    return x


def flatten_y_for_sgpr(y):
    if hasattr(y, "detach"):
        y = y.detach().cpu().numpy()
    return y.reshape(-1).astype(np.float64)


def squared_exponential_kernel(x1, x2, signal_variance, length_scale):
    """
    Isotropic RBF kernel:
    k(x, x') = signal_variance * exp(-0.5 * ||x-x'||^2 / length_scale^2)
    """
    d2 = cdist(x1 / length_scale, x2 / length_scale, metric="sqeuclidean")
    return signal_variance * np.exp(-0.5 * d2)


def estimate_median_length_scale(x, seed=2026, max_points=1000):
    n = x.shape[0]
    rng = np.random.default_rng(seed)

    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        z = x[idx]
    else:
        z = x

    d = cdist(z, z, metric="euclidean")
    d = d[np.triu_indices_from(d, k=1)]
    d = d[np.isfinite(d)]
    d = d[d > 0]

    if len(d) == 0:
        return 1.0

    return float(np.median(d))


def select_inducing_points(x_train, cfg):
    n = x_train.shape[0]
    m = min(cfg.num_inducing, n)

    if m == n:
        return x_train.copy(), np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(cfg.seed)

    if cfg.inducing_selection.lower() == "random":
        idx = rng.choice(n, size=m, replace=False)
        idx = np.sort(idx).astype(np.int64)
        return x_train[idx].copy(), idx

    if cfg.inducing_selection.lower() == "kmeans":
        kmeans = MiniBatchKMeans(
            n_clusters=m,
            random_state=cfg.seed,
            batch_size=min(4096, max(256, m * 4)),
            n_init=3,
            max_iter=100,
            reassignment_ratio=0.01
        )
        kmeans.fit(x_train)
        xu = kmeans.cluster_centers_.astype(np.float64)
        return xu, np.arange(m, dtype=np.int64)

    raise ValueError("cfg.inducing_selection must be either 'random' or 'kmeans'.")


def optimize_kernel_on_subset(xu, y_subset, cfg):
    """
    Fit a small exact GPR only on the inducing subset to obtain reasonable
    kernel hyperparameters for SGPR. This avoids exact GPR on all samples.
    """
    kernel = (
        ConstantKernel(
            constant_value=cfg.constant_value,
            constant_value_bounds=cfg.constant_bounds
        )
        * RBF(
            length_scale=cfg.length_scale_initial,
            length_scale_bounds=cfg.length_scale_bounds
        )
        + WhiteKernel(
            noise_level=cfg.noise_level,
            noise_level_bounds=cfg.noise_level_bounds
        )
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=cfg.alpha,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=cfg.n_restarts_optimizer,
        normalize_y=False,
        random_state=cfg.seed
    )
    gpr.fit(xu, y_subset)

    learned_kernel = gpr.kernel_

    # Kernel structure: ConstantKernel * RBF + WhiteKernel
    signal_variance = float(learned_kernel.k1.k1.constant_value)
    length_scale = float(np.squeeze(learned_kernel.k1.k2.length_scale))
    noise_variance = float(learned_kernel.k2.noise_level)

    return signal_variance, length_scale, noise_variance, learned_kernel


class SparseGPRFITC:
    """
    Sparse Gaussian Process Regression using a FITC-style inducing-point approximation.

    The model is fitted in standardized input/output space. The predictive variance
    returned by predict() is the observation-level variance, i.e. latent uncertainty
    plus estimated noise variance.
    """

    def __init__(
        self,
        num_inducing=512,
        inducing_selection="random",
        signal_variance=1.0,
        length_scale=None,
        noise_variance=1e-2,
        optimize_kernel_on_inducing_subset=True,
        eps_var=1e-6,
        jitter=1e-6,
        seed=2026,
    ):
        self.num_inducing = num_inducing
        self.inducing_selection = inducing_selection
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.optimize_kernel_on_inducing_subset = optimize_kernel_on_inducing_subset
        self.eps_var = eps_var
        self.jitter = jitter
        self.seed = seed

        self.xu = None
        self.inducing_indices = None
        self.learned_kernel = None

        self.l_kmm = None
        self.l_a = None
        self.alpha = None
        self.lambda_diag = None

    def _kernel(self, x1, x2):
        return squared_exponential_kernel(
            x1,
            x2,
            signal_variance=self.signal_variance,
            length_scale=self.length_scale
        )

    def fit(self, x_train, y_train, cfg):
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)

        self.xu, self.inducing_indices = select_inducing_points(x_train, cfg)

        if cfg.optimize_kernel_on_inducing_subset:
            if cfg.inducing_selection.lower() == "random":
                y_inducing_subset = y_train[self.inducing_indices]
            else:
                # For k-means centers, no exact y value exists at the center.
                # Use nearest-neighbor target values only for kernel hyperparameter optimization.
                nearest_idx = np.argmin(cdist(self.xu, x_train, metric="sqeuclidean"), axis=1)
                y_inducing_subset = y_train[nearest_idx]

            (
                self.signal_variance,
                self.length_scale,
                self.noise_variance,
                self.learned_kernel,
            ) = optimize_kernel_on_subset(self.xu, y_inducing_subset, cfg)
        else:
            if self.length_scale is None:
                self.length_scale = estimate_median_length_scale(
                    self.xu,
                    seed=self.seed
                )

        self.noise_variance = max(float(self.noise_variance), self.eps_var)
        self.signal_variance = max(float(self.signal_variance), self.eps_var)
        self.length_scale = max(float(self.length_scale), 1e-8)

        m = self.xu.shape[0]

        kmm = self._kernel(self.xu, self.xu)
        kmm = kmm + self.jitter * np.eye(m)

        knm = self._kernel(x_train, self.xu)

        self.l_kmm = np.linalg.cholesky(kmm)

        # V = Lmm^{-1} Kmn, shape [M, N]
        v = solve_triangular(self.l_kmm, knm.T, lower=True, check_finite=False)
        qnn_diag = np.sum(v ** 2, axis=0)

        # FITC diagonal correction plus observation noise.
        lambda_diag = self.signal_variance - qnn_diag + self.noise_variance
        lambda_diag = np.maximum(lambda_diag, self.eps_var)
        self.lambda_diag = lambda_diag

        beta = 1.0 / lambda_diag

        # A = Kmm + Kmn Lambda^{-1} Knm
        weighted_knm = knm * beta[:, None]
        a = kmm + knm.T @ weighted_knm
        a = a + self.jitter * np.eye(m)

        self.l_a = np.linalg.cholesky(a)

        b = knm.T @ (beta * y_train)

        # alpha = A^{-1} b
        tmp = solve_triangular(self.l_a, b, lower=True, check_finite=False)
        self.alpha = solve_triangular(self.l_a.T, tmp, lower=False, check_finite=False)

        return self

    def predict(self, x_test, batch_size=4096):
        if self.xu is None or self.alpha is None:
            raise RuntimeError("SparseGPRFITC must be fitted before prediction.")

        x_test = np.asarray(x_test, dtype=np.float64)
        n_test = x_test.shape[0]

        mu_all = []
        var_all = []

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            xb = x_test[start:end]

            ksm = self._kernel(xb, self.xu)

            mu = ksm @ self.alpha

            # q_star = ksm Kmm^{-1} kms
            v_kmm = solve_triangular(self.l_kmm, ksm.T, lower=True, check_finite=False)
            q_star = np.sum(v_kmm ** 2, axis=0)

            # a_star = ksm A^{-1} kms
            v_a = solve_triangular(self.l_a, ksm.T, lower=True, check_finite=False)
            a_star = np.sum(v_a ** 2, axis=0)

            # FITC latent variance plus observation noise.
            latent_var = self.signal_variance - q_star + a_star
            obs_var = latent_var + self.noise_variance
            obs_var = np.maximum(obs_var, self.eps_var)

            mu_all.append(mu)
            var_all.append(obs_var)

        mu_all = np.concatenate(mu_all, axis=0)
        var_all = np.concatenate(var_all, axis=0)

        return mu_all, var_all


def predict_sgpr(model, dataset, cfg):
    x = flatten_windows_for_sgpr(dataset.x_grid, dataset.x_global)
    y = flatten_y_for_sgpr(dataset.y)

    mu, var = model.predict(x)
    var = np.maximum(var, cfg.eps_var)

    y = y.reshape(-1, 1).astype(np.float32)
    mu = mu.reshape(-1, 1).astype(np.float32)
    var = var.reshape(-1, 1).astype(np.float32)

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

def inverse_y_and_var(y_norm, mu_norm, var_norm, scaler_y):
    y_std = float(np.squeeze(scaler_y.std))
    y_mean = float(np.squeeze(scaler_y.mean))

    y = y_norm * y_std + y_mean
    mu = mu_norm * y_std + y_mean

    # Variance scales with std squared.
    var = var_norm * (y_std ** 2)

    return y, mu, var


def train_one_horizon(cfg, transformed_data, scaler_y, horizon_id):
    lead_steps = horizon_id * cfg.samples_per_hour
    lead_hours = horizon_id

    horizon_dir = os.path.join(cfg.output_dir, f"Horizon_{horizon_id:02d}_step_{lead_hours:02d}h_ahead")
    os.makedirs(horizon_dir, exist_ok=True)

    set_seed(cfg.seed + horizon_id)

    datasets, loaders, sample_metas = prepare_data_for_horizon(
        cfg=cfg,
        transformed_data=transformed_data,
        horizon_id=horizon_id
    )

    x_train_full = flatten_windows_for_sgpr(
        datasets["Train"].x_grid,
        datasets["Train"].x_global
    )
    y_train_full = flatten_y_for_sgpr(datasets["Train"].y)

    model = SparseGPRFITC(
        num_inducing=cfg.num_inducing,
        inducing_selection=cfg.inducing_selection,
        signal_variance=cfg.signal_variance,
        length_scale=cfg.length_scale,
        noise_variance=cfg.noise_variance,
        optimize_kernel_on_inducing_subset=cfg.optimize_kernel_on_inducing_subset,
        eps_var=cfg.eps_var,
        jitter=cfg.jitter,
        seed=cfg.seed + horizon_id,
    )

    print("\n" + "=" * 80)
    print(f"Training SGPR Horizon {horizon_id:02d}: {lead_hours} h ahead, lead_steps = {lead_steps} samples")
    print(f"Input length = {cfg.input_len} samples ({cfg.input_window_hours} h)")
    print(f"Window alignment = {cfg.window_alignment}, stride = {cfg.stride}")
    for split_name in ["Train", "Validation", "Test"]:
        print(f"{split_name} samples: {len(datasets[split_name])}")
    print(f"SGPR full training samples: {x_train_full.shape[0]}")
    print(f"SGPR input dimension: {x_train_full.shape[1]}")
    print(f"Number of inducing points: {min(cfg.num_inducing, x_train_full.shape[0])}")
    print(f"Inducing selection: {cfg.inducing_selection}")
    print(f"Optimize kernel on inducing subset: {cfg.optimize_kernel_on_inducing_subset}")
    print("=" * 80)

    model.fit(x_train_full, y_train_full, cfg)

    best_model_path = os.path.join(horizon_dir, f"best_sgpr_fitc_model_horizon_{horizon_id:02d}.pkl")
    if cfg.save_model:
        with open(best_model_path, "wb") as f:
            pickle.dump(model, f)

    history_df = pd.DataFrame([{
        "Horizon_step": horizon_id,
        "Lead_hours": lead_hours,
        "Lead_samples": lead_steps,
        "Epoch": 0,
        "Train_GNLL": np.nan,
        "Validation_GNLL": np.nan,
        "Full_retained_training_samples": x_train_full.shape[0],
        "Num_inducing": model.xu.shape[0],
        "Input_dimension": x_train_full.shape[1],
        "Inducing_selection": cfg.inducing_selection,
        "Optimize_kernel_on_inducing_subset": cfg.optimize_kernel_on_inducing_subset,
        "Signal_variance": model.signal_variance,
        "Length_scale": model.length_scale,
        "Noise_variance": model.noise_variance,
        "Learned_kernel_on_inducing_subset": str(model.learned_kernel),
        "Note": "Probabilistic SSGPR baseline with FITC-style inducing-point approximation fitted on standardized sliding-window samples."
    }])
    history_df.to_excel(
        os.path.join(horizon_dir, f"training_history_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    pd.DataFrame({
        "Inducing_point_index_or_cluster_id": model.inducing_indices
    }).to_excel(
        os.path.join(horizon_dir, f"sgpr_inducing_indices_or_cluster_ids_horizon_{horizon_id:02d}.xlsx"),
        index=False
    )

    result_summary = []
    prediction_records = {}

    for split_name in ["Train", "Validation", "Test"]:
        y_norm, mu_norm, var_norm = predict_sgpr(model, datasets[split_name], cfg)

        y, mu, var = inverse_y_and_var(
            y_norm=y_norm,
            mu_norm=mu_norm,
            var_norm=var_norm,
            scaler_y=scaler_y
        )

        y_flat = y.reshape(-1)
        mu_flat = mu.reshape(-1)
        var_flat = var.reshape(-1)

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
        os.path.join(cfg.output_dir, "predefined_distance_adjacency.xlsx")
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

    if cfg.require_cuda and cfg.device != "cuda":
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and use an NVIDIA GPU, "
            "or set cfg.require_cuda = False."
        )

    print("SSGPR baseline uses scikit-learn and runs on CPU.")

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
    all_histories = []

    for horizon_id in cfg.horizon_steps:
        model, summary_df, prediction_records, history_df = train_one_horizon(
            cfg=cfg,
            transformed_data=transformed_data,
            scaler_y=scaler_y,
            horizon_id=horizon_id
        )

        all_metrics.append(summary_df)
        all_predictions[horizon_id] = prediction_records
        all_histories.append(history_df)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
