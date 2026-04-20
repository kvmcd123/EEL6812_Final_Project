import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path


class GraphSAGEGRUForecaster(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_node_features,
        num_edge_features,
        num_weather_features,
        sage_hidden_size=16,
        rnn_hidden_size=64,
        head_hidden_size=64,
        num_targets_per_node=1,
        dropout=0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_weather_features = num_weather_features
        self.num_targets_per_node = num_targets_per_node

        self.sage = SAGEConv(
            in_channels=num_node_features,
            out_channels=sage_hidden_size,
        )
        self.sage_norm = nn.LayerNorm(sage_hidden_size)
        self.sage_activation = nn.GELU()
        self.sage_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=num_weather_features + sage_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, num_targets_per_node),
        )

    def forward(self, node_features, edge_features, node_dynamic_features, edge_index):
        if node_dynamic_features.dim() == 2:
            node_dynamic_features = node_dynamic_features.unsqueeze(0)

        batch_size = node_dynamic_features.size(0)
        seq_len = node_dynamic_features.size(-1)

        graph_context = self.sage(node_features, edge_index)
        graph_context = self.sage_norm(graph_context)
        graph_context = self.sage_activation(graph_context)
        graph_context = self.sage_dropout(graph_context)

        dynamic_features = node_dynamic_features.view(
            batch_size,
            self.num_nodes,
            self.num_weather_features,
            seq_len,
        )
        dynamic_features = dynamic_features.permute(0, 1, 3, 2)

        graph_context = graph_context.unsqueeze(0).unsqueeze(2).expand(batch_size, self.num_nodes, seq_len, -1)
        gru_input = torch.cat((dynamic_features, graph_context), dim=-1)
        gru_input = gru_input.reshape(batch_size * self.num_nodes, seq_len, -1)

        _, hidden = self.gru(gru_input)
        hidden = hidden[-1]
        output = self.head(hidden)
        output = output.view(batch_size, self.num_nodes * self.num_targets_per_node)
        return output


def main():
    checkpoint_path = Path("saved_models/temp_only_graphsage_checkpoint.pt")
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    if not checkpoint_path.exists():
        print("Checkpoint not found.")
        print("Run the training notebook first so it saves:")
        print(checkpoint_path)
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = GraphSAGEGRUForecaster(
        num_nodes=config["num_nodes"],
        num_node_features=config["num_node_features"],
        num_edge_features=config["num_edge_features"],
        num_weather_features=config["num_weather_features"],
        sage_hidden_size=config["sage_hidden_size"],
        rnn_hidden_size=config["rnn_hidden_size"],
        head_hidden_size=config["head_hidden_size"],
        num_targets_per_node=config["num_targets_per_node"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    node_features = checkpoint["node_features"]
    edge_features = checkpoint["edge_features"]
    edge_index = checkpoint["edge_index"]
    x_test = checkpoint["X_test"]
    y_test = checkpoint["y_test"]
    target_mean = checkpoint["target_mean"].squeeze(1)
    target_std = checkpoint["target_std"].squeeze(1)
    node_ids = checkpoint["node_ids"]
    timestamps = pd.to_datetime(checkpoint["test_timestamps"])

    if len(x_test) == 0:
        print("The saved checkpoint has an empty test set, so there is nothing to show.")
        return

    with torch.no_grad():
        pred_scaled = model(node_features, edge_features, x_test, edge_index)

    pred = pred_scaled * target_std.unsqueeze(0) + target_mean.unsqueeze(0)
    truth = y_test * target_std.unsqueeze(0) + target_mean.unsqueeze(0)

    sample_idx = 0
    node_idx = 0
    lookback = config["lookback"]
    num_weather_features = config["num_weather_features"]

    feature_names = ["Rain", "Wind", "Temperature", "Humidity"]
    feature_rows = [
        node_idx * num_weather_features + 0,
        node_idx * num_weather_features + 1,
        node_idx * num_weather_features + 2,
        node_idx * num_weather_features + 3,
    ]

    input_end_time = timestamps[sample_idx] - pd.Timedelta(hours=1)
    input_times = pd.date_range(end=input_end_time, periods=lookback, freq="h")

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, name, row_idx in zip(axes, feature_names, feature_rows):
        ax.plot(input_times, x_test[sample_idx, row_idx, :].numpy(), linewidth=2)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f"Test Input Sequence for Weather Station near St. Augustine\n(values are standardized inputs)")
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(output_dir / "input_sequence_example.png", dpi=150)
    plt.show()
    plt.close()

    plot_len = min(72, len(pred))
    plot_times = timestamps[:plot_len]

    plt.figure(figsize=(12, 4))
    plt.plot(plot_times, truth[:plot_len, node_idx].numpy(), label="Actual temperature", linewidth=2)
    plt.plot(plot_times, pred[:plot_len, node_idx].numpy(), label="Predicted temperature", linewidth=2)
    plt.ylabel("Temperature (°F)")
    plt.xlabel("Time")
    plt.title(f"Temperature Output Time Series for Weather Station near St. Augustine")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "output_timeseries_example.png", dpi=150)
    plt.show()
    plt.close()

    node_data = pd.read_csv("Florida_meteo_nodeList.csv")
    edge_data = pd.read_csv("Florida_meteo_edgeList.csv")

    positions = {
        int(row.id): (float(row.longitude), float(row.latitude))
        for row in node_data[["id", "longitude", "latitude"]].itertuples(index=False)
    }

    graph = nx.Graph()
    graph.add_nodes_from(int(node_id) for node_id in node_data["id"])
    graph.add_edges_from(
        (int(row.source), int(row.target))
        for row in edge_data[["source", "target"]].itertuples(index=False)
    )

    node_order = [int(node_id) for node_id in node_data["id"]]
    graph_plot_len = min(4, len(pred))
    graph_indices = np.linspace(0, len(pred) - 1, graph_plot_len, dtype=int)

    graph_actual = truth[graph_indices].numpy()
    graph_pred = pred[graph_indices].numpy()
    all_graph_values = np.concatenate([graph_actual.reshape(-1), graph_pred.reshape(-1)])
    vmin = float(np.min(all_graph_values))
    vmax = float(np.max(all_graph_values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(graph_plot_len, 2, constrained_layout=True)
    if graph_plot_len == 1:
        axes = np.array([axes])

    for row_idx, time_idx in enumerate(graph_indices):
        current_time = timestamps[time_idx]
        actual_vals = truth[time_idx].numpy()
        pred_vals = pred[time_idx].numpy()

        for ax, values, title in [
            (axes[row_idx, 0], actual_vals, "Actual"),
            (axes[row_idx, 1], pred_vals, "Predicted"),
        ]:
            nx.draw_networkx_edges(
                graph,
                pos=positions,
                ax=ax,
                width=0.7,
                edge_color="gray",
                alpha=0.2,
            )
            nx.draw_networkx_nodes(
                graph,
                pos=positions,
                ax=ax,
                nodelist=node_order,
                node_color=values,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                node_size=50,
                linewidths=0.2,
                edgecolors="black",
            )
            ax.set_title(f"{title}\n{current_time}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    fig.suptitle("Graph Time-Series Comparison Across Several Test Times", fontsize=14)
    # plt.tight_layout()
    plt.savefig(output_dir / "graph_timeseries_comparison.png", dpi=150)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
