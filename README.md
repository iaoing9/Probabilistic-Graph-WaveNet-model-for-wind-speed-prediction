# Probabilistic Graph WaveNet for Wind Speed Prediction

This repository provides the source code and reproducibility materials for the study:

**A probabilistic Graph WaveNet for wind speed prediction combining meteorological data and on-site measurements**

The proposed probabilistic Graph WaveNet (GWN) jointly estimates the conditional mean and sample-dependent variance of wind speed using the Gaussian negative log-likelihood (GNLL). The framework is evaluated in two cases:

- **Case 1:** Retrospective hindcasting of 10-min mean wind speed using ECMWF reanalysis data and bridge-site measurements.
- **Case 2:** Day-ahead wind gust forecasting using operational ECMWF forecasts and on-site measurements from two anemometric stations.

## Repository contents

The repository includes:

- Data preprocessing and sample construction
- Probabilistic GWN implementation
- Benchmark models
- Rolling-origin evaluation
- Model component ablation experiments
- Multi-seed robustness analysis
- Diebold–Mariano statistical significance tests
- Probabilistic evaluation metrics
- Numerical results reported in the response to the reviewers

The evaluated models include:

- Probabilistic GWN
- Probabilistic Transformer
- Probabilistic LSTM
- Probabilistic GRU
- Quantile regression forest
- Gaussian process regression
- Persistence model
- ARIMA
- VAR
