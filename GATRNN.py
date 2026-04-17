import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, BatchNorm

torch.manual_seed(0)
 
class GATRNN(nn.Module):
    def __init__(self, num_static_node_features, num_static_edge_features, num_weather_features, hidden_size, lstm_hidden_size=64, num_heads=8):
        super().__init__()
       
        self.gat_conv1 = GATConv(
            num_static_node_features, hidden_size, heads=num_heads, concat=False, edge_dim=num_static_edge_features
        )
        self.batch_norm1 = BatchNorm(hidden_size)
        self.lstm = nn.LSTM(input_size=num_weather_features, hidden_size=lstm_hidden_size, batch_first=True)
       
        combined_feature_size = hidden_size + lstm_hidden_size
        self.hidden_linear = nn.Linear(combined_feature_size, hidden_size // 2)
        self.output_linear = nn.Linear(hidden_size // 2, 1)
       
        self.dropout = nn.Dropout(0.60)
 
    def forward(self, node_static_features, edge_static_features, node_dynamic_features, edge_index):
        x1 = self.gat_conv1(node_static_features, edge_index.t(), edge_static_features)
        # x1 = self.batch_norm1(x1)
        x1 = torch.relu(self.batch_norm1(x1))
        x1 = self.dropout(x1)
       
        lstm_out, _ = self.lstm(node_dynamic_features)
        lstm_out = lstm_out[:, -1, :]
       
        combined_features = torch.cat((x1, lstm_out), dim=1)
       
        hidden_output = torch.relu(self.hidden_linear(combined_features))
        output = self.output_linear(hidden_output)
       
        return output