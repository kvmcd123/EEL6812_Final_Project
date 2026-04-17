import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, BatchNorm

torch.manual_seed(0)
 
# class GATRNN(nn.Module):
#     def __init__(self, num_static_node_features, num_static_edge_features, num_weather_features, hidden_size, lstm_hidden_size=64, num_heads=8):
#         super().__init__()
       
#         self.gat_conv1 = GATConv(
#             num_static_node_features, hidden_size, heads=num_heads, concat=False, edge_dim=num_static_edge_features
#         )
#         self.batch_norm1 = BatchNorm(hidden_size)
#         self.lstm = nn.LSTM(input_size=num_weather_features, hidden_size=lstm_hidden_size, batch_first=True)
       
#         combined_feature_size = hidden_size + lstm_hidden_size
#         self.hidden_linear = nn.Linear(combined_feature_size, hidden_size // 2)
#         self.output_linear = nn.Linear(hidden_size // 2, 1)
       
#         self.dropout = nn.Dropout(0.60)
 
#     def forward(self, node_static_features, edge_static_features, node_dynamic_features, edge_index):
#         x1 = self.gat_conv1(node_static_features, edge_index.t(), edge_static_features)
#         # x1 = self.batch_norm1(x1)
#         x1 = torch.relu(self.batch_norm1(x1))
#         x1 = self.dropout(x1)
       
#         lstm_out, _ = self.lstm(node_dynamic_features)
#         lstm_out = lstm_out[:, -1, :]
       
#         combined_features = torch.cat((x1, lstm_out), dim=1)
       
#         hidden_output = torch.relu(self.hidden_linear(combined_features))
#         output = self.output_linear(hidden_output)
       
#         return output



class GATRNN(nn.Module):
    def __init__(self, num_nodes, num_node_features, num_edge_features, num_weather_features, hidden_size, lstm_hidden_size=64, num_heads=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat_conv = GATConv(num_node_features, hidden_size, heads=num_heads, concat=False, edge_dim=num_edge_features)
        self.gat_linear = nn.Linear(hidden_size, 1)
        self.gat_activation = torch.tanh()

        # self.batch_norm1 = BatchNorm(hidden_size)
        self.lstm = nn.LSTM(input_size=num_weather_features*num_nodes, hidden_size=lstm_hidden_size, batch_first=True)
        self.lstm_linear = nn.Linear(lstm_hidden_size, num_nodes)
        self.lstm_activation = torch.tanh()
        
 
    def forward(self, node_features, edge_features, node_dynamic_features, edge_index):
        
        x1 = self.gat_conv(node_features, edge_index.t(), edge_features)
        x1 = self.gat_linear(x1)
        x1 = self.gat_activation(x1)

        x2 = torch.zeros((node_dynamic_features.size(0), node_dynamic_features.size(1) * node_dynamic_features.size(2)), device=node_dynamic_features.device)
        
        for i in range(self.num_nodes):

            x2[3*i,:] = 0 * x2[3*i,:] + x1[i]
            x2[3*i+1,:] = node_dynamic_features[2*i,:]
            x2[3*i+2,:] = node_dynamic_features[2*i+1,:]

       
        x3, _ = self.lstm(x2)
        x4 = self.lstm_linear(x3)
        output = self.lstm_activation(x4)
       
        return output