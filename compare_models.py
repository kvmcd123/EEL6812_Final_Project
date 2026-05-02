import gc
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import GraphNN, GraphRNN, LSTMRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "saved_models"
OUTPUT_DIR = SCRIPT_DIR / "model_error_plots"

CHECKPOINTS = None
BATCH_SIZE = 8
DEVICE_CHOICE = "auto"
DATASET_NAMES = ["full_us"]
MODEL_NAMES = None
LOG_Y = False


def pick_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def checkpoint_paths():
    if CHECKPOINTS:
        paths = [path if path.is_absolute() else Path.cwd() / path for path in CHECKPOINTS]
    else:
        paths = sorted(CHECKPOINT_DIR.glob("*_checkpoint.pt"))

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(str(path) for path in missing))
    if not paths:
        raise FileNotFoundError(f"No *_checkpoint.pt files found in {CHECKPOINT_DIR}")
    return paths


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def model_display_name(model_name):
    if model_name == "baseline_lstm":
        return "Baseline LSTM"
    if model_name == "GraphRNN":
        return "GraphRNN"
    return "GraphNN"


def infer_model_name(checkpoint, path):
    model_name = checkpoint.get("model_name")
    if model_name:
        return str(model_name)

    stem = path.stem.lower()
    if "graphsage" in stem or "graph_sage" in stem:
        return "GraphRNN"
    if "graphnn" in stem:
        return "GraphNN"
    if "baseline" in stem or "lstm" in stem:
        return "baseline_lstm"
    raise ValueError(f"Could not infer model type for {path}")


def infer_dataset_name(checkpoint, path):
    if checkpoint.get("dataset_name"):
        return str(checkpoint["dataset_name"])

    stem = path.stem
    for suffix in ("_GraphRNN_checkpoint", "_GraphNN_checkpoint", "_baseline_lstm_checkpoint", "_checkpoint"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def make_model(checkpoint, model_name, device):
    state = checkpoint["model_state_dict"]

    for item in state.items():
        print(item[0], item[1].shape)

    node_features = checkpoint.get("node_features")
    x_test = checkpoint["X_test"]
    config = checkpoint.get("config", {})
    dropout = float(config.get("dropout", 0.0))

    if model_name == "GraphNN":
        hidden_size = int(state["input_proj.0.weight"].shape[0])
        head_hidden_size = int(state["head.1.weight"].shape[0])
        model = GraphNN(
            num_nodes=len(checkpoint["node_ids"]),
            node_static_dim=int(node_features.shape[1]),
            node_window_dim=int(x_test.shape[-1]),
            gnn_hidden_size=hidden_size,
            head_hidden_size=head_hidden_size,
            dropout=dropout,
        )
    elif model_name == "GraphRNN":
        sage_hidden_size = int(state["node_proj.weight"].shape[0])
        rnn_hidden_size = int(state["gru.weight_hh_l0"].shape[1])
        head_hidden_size = int(state["head.1.weight"].shape[0])
        num_targets_per_node = int(state["head.4.weight"].shape[0])
        feature_names = config.get("feature_names")
        if feature_names:
            num_weather_features = len(feature_names)
        else:
            num_weather_features = int(x_test.shape[1] // len(checkpoint["node_ids"]))
        model = GraphRNN(
            num_nodes=len(checkpoint["node_ids"]),
            num_node_features=int(node_features.shape[1]),
            num_edge_features=1,
            num_weather_features=num_weather_features,
            sage_hidden_size=sage_hidden_size,
            rnn_hidden_size=rnn_hidden_size,
            head_hidden_size=head_hidden_size,
            num_targets_per_node=num_targets_per_node,
            dropout=dropout,
        )
    elif model_name == "baseline_lstm":
        num_layers = 1 + max(
            int(key.removeprefix("lstm.weight_ih_l"))
            for key in state
            if key.startswith("lstm.weight_ih_l")
        )
        hidden_size = int(state["fc.weight"].shape[1])
        model = LSTMRegressor(
            input_size=int(x_test.shape[-1]),
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def forward_model(model, model_name, batch, checkpoint, device):
    if model_name == "GraphNN":
        return model(
            checkpoint["node_features"].to(device),
            checkpoint["edge_index"].to(device),
            checkpoint["edge_weight"].to(device),
            batch,
        )

    if model_name == "GraphRNN":
        edge_features = checkpoint.get("edge_features")
        if edge_features is None:
            edge_features = checkpoint["edge_weight"].reshape(-1, 1)
        return model(
            checkpoint["node_features"].to(device),
            edge_features.to(device),
            batch,
            checkpoint["edge_index"].to(device),
        )

    if model_name == "baseline_lstm":
        return model(batch)

    raise ValueError(model_name)


def predict_checkpoint(checkpoint, model_name, device, batch_size):
    model = make_model(checkpoint, model_name, device)
    x_test = checkpoint["X_test"]
    chunks = []

    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            batch = x_test[start : start + batch_size].to(device)
            chunks.append(forward_model(model, model_name, batch, checkpoint, device).cpu())

    pred_delta = torch.cat(chunks, dim=0) if chunks else torch.empty_like(checkpoint["last_temp_test"])
    pred_scaled = pred_delta + checkpoint["last_temp_test"]
    truth_scaled = checkpoint.get("y_test_level", checkpoint.get("y_test"))
    if truth_scaled is None:
        raise KeyError("Checkpoint must contain y_test_level or y_test.")

    pred = pred_scaled * checkpoint["target_std"] + checkpoint["target_mean"]
    truth = truth_scaled * checkpoint["target_std"] + checkpoint["target_mean"]
    return pred.detach().cpu(), truth.detach().cpu()


def node_error_frame(checkpoint, pred, truth, dataset_name, model_name, checkpoint_path):
    error = pred - truth
    abs_error = torch.abs(error).numpy()
    squared_error = torch.square(error).numpy()

    if "test_station_ids" in checkpoint:
        station_ids = np.asarray(checkpoint["test_station_ids"], dtype=int)
        frame = pd.DataFrame(
            {
                "node_id": station_ids,
                "absolute_error": abs_error.reshape(-1),
                "squared_error": squared_error.reshape(-1),
            }
        )
        grouped = (
            frame.groupby("node_id", as_index=False)
            .agg(mae=("absolute_error", "mean"), mse=("squared_error", "mean"), n_samples=("absolute_error", "size"))
            .sort_values("node_id")
        )
        grouped["node_order"] = np.arange(len(grouped))
    else:
        node_ids = np.asarray(checkpoint["node_ids"], dtype=int)
        if abs_error.ndim != 2 or abs_error.shape[1] != len(node_ids):
            raise ValueError(
                f"{checkpoint_path.name}: expected graph predictions shaped "
                f"[num_samples, {len(node_ids)}], got {tuple(abs_error.shape)}"
            )
        grouped = pd.DataFrame(
            {
                "node_order": np.arange(len(node_ids)),
                "node_id": node_ids,
                "mae": abs_error.mean(axis=0),
                "mse": squared_error.mean(axis=0),
                "n_samples": abs_error.shape[0],
            }
        )

    grouped.insert(0, "checkpoint", checkpoint_path.name)
    grouped.insert(0, "model_label", model_display_name(model_name))
    grouped.insert(0, "model_name", model_name)
    grouped.insert(0, "dataset_name", dataset_name)
    return grouped


def overall_error_row(checkpoint, pred, truth, dataset_name, model_name, checkpoint_path):
    error = pred - truth
    abs_error = torch.abs(error)
    squared_error = torch.square(error)

    if "test_station_ids" in checkpoint:
        node_count = int(pd.Series(checkpoint["test_station_ids"]).nunique())
    else:
        node_count = int(len(checkpoint["node_ids"]))

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "model_label": model_display_name(model_name),
        "checkpoint": checkpoint_path.name,
        "overall_mae": float(abs_error.mean().item()),
        "overall_mse": float(squared_error.mean().item()),
        "overall_rmse": float(torch.sqrt(squared_error.mean()).item()),
        "n_predictions": int(abs_error.numel()),
        "nodes": node_count,
    }


def evaluate_checkpoint(path, device):
    print(f"Loading {path.name}...")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_name = infer_model_name(checkpoint, path)
    dataset_name = infer_dataset_name(checkpoint, path)

    if DATASET_NAMES and dataset_name not in set(DATASET_NAMES):
        print(f"Skipping {path.name}: dataset {dataset_name!r} is not selected.")
        return None
    if MODEL_NAMES and model_name not in set(MODEL_NAMES):
        print(f"Skipping {path.name}: model {model_name!r} is not selected.")
        return None

    print(f"Evaluating {dataset_name}/{model_name} on {device}...")
    pred, truth = predict_checkpoint(checkpoint, model_name, device, max(BATCH_SIZE, 1))
    frame = node_error_frame(checkpoint, pred, truth, dataset_name, model_name, path)
    overall = overall_error_row(checkpoint, pred, truth, dataset_name, model_name, path)
    print(
        f"Overall {dataset_name}/{overall['model_label']}: "
        f"MAE={overall['overall_mae']:.5f}, "
        f"MSE={overall['overall_mse']:.5f}, "
        f"RMSE={overall['overall_rmse']:.5f} "
        f"over {overall['n_predictions']} predictions across {overall['nodes']} nodes"
    )

    del checkpoint, pred, truth
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return frame, overall


def sparse_node_ticks(ax, ordered_nodes):
    if ordered_nodes.empty:
        return
    max_ticks = 14
    if len(ordered_nodes) <= max_ticks:
        tick_rows = ordered_nodes
    else:
        tick_positions = np.linspace(0, len(ordered_nodes) - 1, max_ticks).round().astype(int)
        tick_rows = ordered_nodes.iloc[np.unique(tick_positions)]
    ax.set_xticks(tick_rows["plot_x"])
    ax.set_xticklabels(tick_rows["node_id"].astype(str), rotation=45, ha="right")


def plot_dataset_errors(dataset_name, frame):
    first_model = frame.sort_values(["model_name", "node_order"]).iloc[0]["model_name"]
    node_order = (
        frame[frame["model_name"] == first_model][["node_id", "node_order"]]
        .drop_duplicates()
        .sort_values("node_order")
        .reset_index(drop=True)
    )
    node_order["plot_x"] = np.arange(len(node_order))
    plot_frame = frame.merge(node_order[["node_id", "plot_x"]], on="node_id", how="left")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    metrics = [("mae", "Average MAE"), ("mse", "Average MSE")]
    for ax, (metric, title) in zip(axes, metrics):
        for model_name, model_frame in plot_frame.groupby("model_name", sort=True):
            ordered = model_frame.sort_values("plot_x")
            ax.plot(
                ordered["plot_x"],
                ordered[metric],
                marker="o" if len(ordered) <= 80 else None,
                linewidth=1.5,
                markersize=3,
                label=model_display_name(model_name),
            )
        ax.set_title(f"{title} by Node: {dataset_name}")
        ax.set_ylabel("Temperature error")
        ax.grid(True, alpha=0.3)
        if LOG_Y:
            ax.set_yscale("log")
        ax.legend()

    axes[-1].set_xlabel("Node ID")
    sparse_node_ticks(axes[-1], node_order)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{safe_name(dataset_name)}_average_node_errors.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_outputs(all_errors, overall_errors):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    all_csv = OUTPUT_DIR / "all_model_average_node_errors.csv"
    all_errors.to_csv(all_csv, index=False)
    written.append(all_csv)

    summary_csv = OUTPUT_DIR / "model_error_summary.csv"
    overall_errors.sort_values(["dataset_name", "overall_mae"]).to_csv(summary_csv, index=False)
    written.append(summary_csv)

    for dataset_name, dataset_frame in all_errors.groupby("dataset_name", sort=True):
        dataset_csv = OUTPUT_DIR / f"{safe_name(dataset_name)}_average_node_errors.csv"
        dataset_frame.sort_values(["model_name", "node_order"]).to_csv(dataset_csv, index=False)
        written.append(dataset_csv)
        written.append(plot_dataset_errors(dataset_name, dataset_frame))

    return written


def run_comparison():
    device = pick_device(DEVICE_CHOICE)
    paths = checkpoint_paths()
    frames = []
    overall_rows = []

    for path in paths:
        result = evaluate_checkpoint(path, device)
        if result is not None:
            frame, overall = result
            frames.append(frame)
            overall_rows.append(overall)

    if not frames:
        raise ValueError("No checkpoints matched the selected filters.")

    all_errors = pd.concat(frames, ignore_index=True)
    overall_errors = pd.DataFrame(overall_rows)
    written = write_outputs(all_errors, overall_errors)

    print("\nOverall errors across all nodes and all test data:")
    printable = overall_errors.sort_values(["dataset_name", "overall_mae"])[
        ["dataset_name", "model_label", "overall_mae", "overall_mse", "overall_rmse", "n_predictions", "nodes"]
    ]
    print(printable.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    print("\nWrote:")
    for path in written:
        print(f"  {path}")

    return all_errors, overall_errors


all_node_errors, overall_errors = run_comparison()
