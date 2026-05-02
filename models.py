import torch
import torch.nn as nn
import torch_geometric.nn as tg

class GraphNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_static_dim,
        node_window_dim,
        gnn_hidden_size=96,
        head_hidden_size=64,
        dropout=0.10,
    ):
        super().__init__()

        self.num_nodes = num_nodes

        input_dim = node_static_dim + node_window_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, gnn_hidden_size),
            nn.LayerNorm(gnn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gcn1 = tg.GraphConv(gnn_hidden_size, gnn_hidden_size)
        self.gcn2 = tg.GraphConv(gnn_hidden_size, gnn_hidden_size)

        self.norm1 = nn.LayerNorm(gnn_hidden_size)
        self.norm2 = nn.LayerNorm(gnn_hidden_size)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(gnn_hidden_size),
            nn.Linear(gnn_hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )

    def forward(self, node_static_features, edge_index, edge_weight, node_window_features):
        if node_window_features.dim() == 2:
            node_window_features = node_window_features.unsqueeze(0)

        batch_size = node_window_features.size(0)

        static_features = node_static_features.unsqueeze(0).expand(batch_size, -1, -1)

        node_inputs = torch.cat(
            (static_features, node_window_features),
            dim=-1,
        )

        node_inputs = node_inputs.reshape(batch_size * self.num_nodes, -1)

        x = self.input_proj(node_inputs)

        batch_offsets = torch.arange(
            batch_size,
            device=edge_index.device,
        ) * self.num_nodes

        batched_edge_index = edge_index.unsqueeze(0) + batch_offsets.view(-1, 1, 1)
        batched_edge_index = batched_edge_index.permute(1, 0, 2).reshape(2, -1)

        batched_edge_weight = edge_weight.repeat(batch_size)

        h1 = self.gcn1(
            x,
            batched_edge_index,
            edge_weight=batched_edge_weight,
        )

        h1 = self.norm1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        h2 = self.gcn2(
            h1,
            batched_edge_index,
            edge_weight=batched_edge_weight,
        )

        h2 = self.norm2(h2 + h1)
        h2 = self.act(h2)
        h2 = self.dropout(h2)

        out = self.head(h2)
        out = out.view(batch_size, self.num_nodes)

        return out

    
class GraphRNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_node_features,
        num_edge_features,
        num_weather_features,
        sage_hidden_size=16,
        rnn_hidden_size=32,
        head_hidden_size=32,
        num_targets_per_node=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_weather_features = num_weather_features
        self.num_targets_per_node = num_targets_per_node

        self.node_proj = nn.Linear(num_node_features, sage_hidden_size)
        self.graph_in = tg.GraphConv(
            in_channels=num_node_features,
            out_channels=sage_hidden_size,
        )
        self.graph_out = tg.GraphConv(
            in_channels=sage_hidden_size,
            out_channels=sage_hidden_size,
        )
        self.graph_norm1 = nn.LayerNorm(sage_hidden_size)
        self.graph_norm2 = nn.LayerNorm(sage_hidden_size)
        #self.graph_activation = nn.GELU()
        self.graph_activation = nn.Tanh()
        self.graph_dropout = nn.Dropout(dropout)

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

        graph_context = self.graph_in(node_features, edge_index)
        graph_context = self.graph_norm1(graph_context)
        graph_context = self.graph_activation(graph_context)
        graph_context = self.graph_dropout(graph_context)
        graph_context = graph_context + self.node_proj(node_features)

        graph_context = self.graph_out(graph_context, edge_index)
        graph_context = self.graph_norm2(graph_context)
        graph_context = self.graph_activation(graph_context)
        graph_context = self.graph_dropout(graph_context)

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


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])