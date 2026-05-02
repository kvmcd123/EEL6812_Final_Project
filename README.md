# Weather Station Graph Forecasting

This repository contains an EEL6812 final project for next-hour temperature forecasting from weather-station time series. It builds station networks from Meteostat data, trains graph-aware neural models, and compares those models against an LSTM baseline.

The main experiment uses a `full_us` station graph with 2,728 nodes. Each node represents a weather station, edges connect nearby stations, and the dynamic features include rain, wind speed, temperature, humidity, hour-of-day encodings, and day-of-year encodings.

## Project Overview

The project is organized around three workflows:

1. Build station datasets and graph tables with `build_weather_station_dataset.ipynb`.
2. Train forecasting models with `train_model.ipynb`.
3. Compare trained checkpoints with `compare_models.py` and visualize station networks with `plot_networks.py`.

The implemented model classes are in `models.py`:

- `GraphNN`: a graph convolution model that uses static station features plus a flattened weather lookback window.
- `GraphRNN`: a graph convolution plus GRU model that combines graph context with sequential weather features.
- `LSTMRegressor`: a non-graph LSTM baseline.

## Repository Layout

```text
.
+-- GraphData/
|   +-- *_nodeList.csv          # station nodes with location/static metadata
|   +-- *_edgeList.csv          # weighted station-neighbor edges
|   +-- *_stations.csv          # raw station metadata
|   +-- network_plots/          # generated station-network maps
+-- WeatherData/
|   +-- full_us/                # daily hourly weather matrices
+-- model_error_plots/          # model comparison CSVs and plots
+-- build_weather_station_dataset.ipynb
+-- train_model.ipynb
+-- WorkingBaselines.ipynb
+-- models.py
+-- compare_models.py
+-- plot_networks.py
```

Each daily weather folder contains one CSV per feature:

- `temp_data.csv`
- `rain_data.csv`
- `wind_speed_data.csv`
- `humidity_data.csv`

Rows are station node IDs and columns are `hour_00` through `hour_23`.

## Setup

Create and activate a Python environment, then install the core dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn meteostat jupyter
pip install torch
pip install torch-geometric
```

For CUDA training, install the PyTorch and PyTorch Geometric builds that match your CUDA version. CPU works, but the full-US graph checkpoints are large and training is much faster on a GPU.

## Data Preparation

Open and run:

```bash
jupyter notebook build_weather_station_dataset.ipynb
```

The dataset notebook:

- discovers Meteostat stations inside the configured bounds,
- builds `nodeList`, `edgeList`, and station metadata CSVs,
- fetches hourly 2018 weather data,
- fills missing values using nearby stations and interpolation,
- exports daily weather feature matrices under `WeatherData/<dataset_name>/`.

Adjust `DATASET_NAME`, `BOUNDS`, `START`, and `END` in the notebook to generate a new dataset.

## Training

Open and run:

```bash
jupyter notebook train_model.ipynb
```

Important settings near the top of the notebook:

- `MODEL_NAME`: `GraphNN`, `GraphRNN`, or `baseline_lstm`
- `DATASET_NAME`: `full_us` or another dataset defined in the notebook
- `LOOKBACK`: number of previous hourly steps used for prediction, currently `24`
- `TRAIN_RATIO`: chronological train/test split, currently `0.80`
- `VAL_RATIO`: validation split from the training portion, currently `0.20`

Training writes local checkpoint files to the paths configured by `CHECKPOINT_DIR` and `CHECKPOINT_PATH` in the notebook.

## Model Comparison

Run the checkpoint comparison script:

```bash
python compare_models.py
```

By default, `compare_models.py` loads checkpoint files from its configured `CHECKPOINT_DIR`, evaluates the selected dataset list, and writes:

- `model_error_plots/model_error_summary.csv`
- `model_error_plots/all_model_average_node_errors.csv`
- `model_error_plots/<dataset>_average_node_errors.csv`
- `model_error_plots/<dataset>_average_node_errors.png`

The script has simple configuration constants near the top, including `DATASET_NAMES`, `MODEL_NAMES`, `BATCH_SIZE`, `DEVICE_CHOICE`, and `LOG_Y`.

### Current Full-US Results

From `model_error_plots/model_error_summary.csv`:

| Dataset | Model | MAE | MSE | RMSE | Predictions | Nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full_us | GraphNN | 1.0160 | 2.2986 | 1.5161 | 4,779,456 | 2,728 |
| full_us | GraphRNN | 1.0270 | 2.2941 | 1.5146 | 4,779,456 | 2,728 |
| full_us | Baseline LSTM | 1.0541 | 2.5097 | 1.5842 | 4,779,456 | 2,728 |

The generated full-US node-error plot is available at:

```text
model_error_plots/full_us_average_node_errors.png
```

## Network Plots

Generate station network maps with:

```bash
python plot_networks.py
```

The script discovers matching `*nodeList.csv` and `*edgeList.csv` files in `GraphData/` and writes maps to:

```text
GraphData/network_plots/
```

If OpenStreetMap tiles are missing from the local cache, the script will try to download them.

## Notes

- The comparison script uses `torch.load(..., weights_only=False)` because the checkpoints store tensors and metadata needed for reconstruction and evaluation.
