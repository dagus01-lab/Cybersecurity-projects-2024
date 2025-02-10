import torch
from torch_geometric.nn import GCNConv, TransformerConv, TopKPooling
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv, TransformerConv, TopKPooling
import torch.nn.functional as F

class STGAETransformerPoolingMultiGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, pool_ratio=0.3):
        super().__init__()
        # Encoder
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.trans_conv1 = TransformerConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.pool1 = TopKPooling(hidden_channels, ratio=pool_ratio)
        self.conv4 = GCNConv(hidden_channels, latent_dim)
        
        # Decoder
        self.conv5 = GCNConv(latent_dim, hidden_channels)
        self.trans_conv2 = TransformerConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.conv7 = GCNConv(hidden_channels, hidden_channels)
        self.conv8 = GCNConv(hidden_channels, in_channels)

    def encode(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.trans_conv1(x, edge_index))
        x, edge_index, _, batch, perm, _ = self.pool1(x, edge_index, None, batch)
        z = self.conv4(x, edge_index)
        return z, edge_index, batch, perm

    def decode(self, z, edge_index):
        x = F.relu(self.conv5(z, edge_index))
        x = F.relu(self.trans_conv2(x, edge_index))
        x = F.relu(self.conv6(x, edge_index))
        x = F.relu(self.conv7(x, edge_index))
        return self.conv8(x, edge_index)

    def forward(self, data):
        batch = getattr(data, 'batch', torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device))
        z, pooled_edge, pooled_batch, perm = self.encode(data.x, data.edge_index, batch)
        return self.decode(z, pooled_edge), z, perm