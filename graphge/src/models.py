import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, force_dropout=None):
        use_dropout = self.training if force_dropout is None else force_dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=use_dropout)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)