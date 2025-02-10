import numpy as np
import torch
import random
from torch_geometric.data import Data, Dataset 
from torch_geometric.nn import radius_graph
from torch_sparse import SparseTensor


def chunk_to_graph(data_array, temp_window, neigh_radius=1.0):
    """
    Convert a data chunk (xarray.DataArray with shape [T, H, W])
    into a PyTorch Geometric Data object.
    
    Each grid cell becomes a node with a feature vector that is the
    concatenation of:
      - the time series (length T) of temperature values (with NaNs imputed to 0)
      - a corresponding mask (length T) where 1.0 indicates a NaN and 0.0 indicates a valid value.
    
    Edges are created between 4-connected neighbors.
    """
    try:
        data_np = data_array.values  # shape: (T, H, W)
    except Exception as e:
        raise ValueError("Failed to retrieve values from data_array: " + str(e))
    
    T, H, W = data_np.shape
    if T != temp_window:
        raise ValueError(f"Expected temporal dimension of {temp_window}, got {T}")
    
    num_nodes = H * W
    
    mask_np = np.isnan(data_np).astype(np.float16)
    data_filled = np.nan_to_num(data_np, nan=0.0)
    
    node_temps = data_filled.reshape(T, -1).transpose(1, 0)  
    node_mask = mask_np.reshape(T, -1).transpose(1, 0)  
    
    features = np.concatenate((node_temps, node_mask), axis=1)
    features_tensor = torch.from_numpy(features.copy()).to(torch.float32)
    
    coords = []
    for i in range(H):
        for j in range(W):
            coords.append([i, j])
    pos = torch.tensor(coords, dtype=torch.float32)  
    edge_index = radius_graph(pos, r=neigh_radius, loop=False)
    
    return Data(x=features_tensor, edge_index=edge_index)


class TemperatureGraphDataset(Dataset):
    """
    Custom dataset that loads chunks from the NETCDF temperature data,
    applies spatial subsetting, and converts each chunk into a spatio-temporal graph.
    Each graph's node features are the concatenation of temperature values and the NaN mask.
    """
    def __init__(self, ds, var_name, temp_window, rnd_offset=True):
        super(TemperatureGraphDataset, self).__init__()
        self.ds = ds
        self.var_name = var_name
        self.temp_window = temp_window
        self.chunk_size = ds[self.var_name].chunks[0][0]
        self.rnd_offset = rnd_offset
        self.num_chunks = int(np.floor(ds[self.var_name].sizes['time'] / self.chunk_size))
    
    def len(self):
        return self.num_chunks
    
    def get(self, idx):
        if self.rnd_offset:
            rnd_offset = random.randint(0, self.temp_window) if idx < self.temp_window - 1 else 0

            start = idx * self.temp_window + rnd_offset
        else:
            start = idx
        end = start + self.temp_window
        try:
            chunk = self.ds[self.var_name].isel(time=slice(start, end))
        except Exception as e:
            raise ValueError(f"Error extracting chunk at index {idx}: {e}")
        #with ProgressBar():
        chunk = chunk.compute()
        graph = chunk_to_graph(chunk, self.temp_window)
        return graph
 