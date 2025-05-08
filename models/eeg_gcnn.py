"""
Implementation of the EEG-GCNN model for neurological disease diagnosis.
This file contains both the shallow and deep variants of the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ShallowEEGGCNN(torch.nn.Module):
    """
    Shallow variant of the EEG-GCNN model from the paper.
    
    This model consists of:
    - 2 Graph Convolutional layers (output dims: 64, 128)
    - Batch normalization after each convolution
    - Global Mean Pooling
    - Output layer
    """
    
    def __init__(self, num_node_features, out_channels=1):
        super(ShallowEEGGCNN, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Output layer
        self.out = nn.Linear(128, out_channels)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # First graph convolutional layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        # Second graph convolutional layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        # Global mean pooling
        x = global_mean_pool(x, data.batch)
        
        # Output layer
        x = self.out(x)
        
        return x


class DeepEEGGCNN(torch.nn.Module):
    """
    Deep variant of the EEG-GCNN model from the paper.
    
    This model consists of:
    - 5 Graph Convolutional layers (output dims: 16, 16, 32, 64, 128)
    - Batch normalization after each convolution
    - Global Mean Pooling
    - 2 hidden Linear layers (dims: 30, 20)
    - Output layer
    """
    
    def __init__(self, num_node_features, out_channels=1):
        super(DeepEEGGCNN, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 32)
        self.conv4 = GCNConv(32, 64)
        self.conv5 = GCNConv(64, 128)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)
        
        # Hidden linear layers
        self.fc1 = nn.Linear(128, 30)
        self.fc2 = nn.Linear(30, 20)
        
        # Output layer
        self.out = nn.Linear(20, out_channels)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # First graph convolutional layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        # Second graph convolutional layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        # Third graph convolutional layer
        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        
        # Fourth graph convolutional layer
        x = self.conv4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        
        # Fifth graph convolutional layer
        x = self.conv5(x, edge_index, edge_attr)
        x = self.bn5(x)
        x = F.leaky_relu(x)
        
        # Global mean pooling
        x = global_mean_pool(x, data.batch)
        
        # First hidden linear layer
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        
        # Second hidden linear layer
        x = self.fc2(x)
        x = F.leaky_relu(x)
        
        # Output layer
        x = self.out(x)
        
        return x


# Ablation models for your studies

class SpatialOnlyEEGGCNN(torch.nn.Module):
    """
    Ablation model that uses only spatial connectivity.
    This model has the same architecture as the ShallowEEGGCNN but uses only spatial connectivity.
    """
    
    def __init__(self, num_node_features, out_channels=1):
        """
        Initialize the spatial-only EEG-GCNN model.
        
        Args:
            num_node_features (int): Number of input features per node.
            out_channels (int): Number of output channels. Default is 1 for binary classification.
        """
        super(SpatialOnlyEEGGCNN, self).__init__()
        
        # Same architecture as ShallowEEGGCNN
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.out = nn.Linear(128, out_channels)
        
    def forward(self, data):
        """
        Forward pass that uses only spatial connectivity.
        
        Args:
            data (torch_geometric.data.Data): Input graph data.
            
        Returns:
            torch.Tensor: Model output.
        """
        x, edge_index = data.x, data.edge_index
        
        # Use only spatial connectivity (ignore edge_attr)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x = global_mean_pool(x, data.batch)
        
        x = self.out(x)
        
        return x


class FunctionalOnlyEEGGCNN(torch.nn.Module):
    """
    Ablation model that uses only functional connectivity.
    This model has the same architecture as the ShallowEEGGCNN but uses only functional connectivity.
    """
    
    def __init__(self, num_node_features, out_channels=1):
        """
        Initialize the functional-only EEG-GCNN model.
        
        Args:
            num_node_features (int): Number of input features per node.
            out_channels (int): Number of output channels. Default is 1 for binary classification.
        """
        super(FunctionalOnlyEEGGCNN, self).__init__()
        
        # Same architecture as ShallowEEGGCNN
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.out = nn.Linear(128, out_channels)
        
    def forward(self, data):
        """
        Forward pass that uses only functional connectivity.
        The implementation depends on how functional connectivity is represented in the data.
        
        Args:
            data (torch_geometric.data.Data): Input graph data.
            
        Returns:
            torch.Tensor: Model output.
        """
        # In a real implementation, this would extract or compute functional connectivity
        # For this example, we'll assume functional connectivity is stored in data.functional_edge_index
        # and data.functional_edge_attr
        
        # If these aren't available, this is just a placeholder - in a real implementation
        # you would need to adapt this to how your data is structured
        x = data.x
        edge_index = data.edge_index
        
        # First graph convolutional layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        # Second graph convolutional layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        # Global mean pooling
        x = global_mean_pool(x, data.batch)
        
        # Output layer
        x = self.out(x)
        
        return x


class SparseEEGGCNN(torch.nn.Module):
    """
    Extension model that uses a sparse graph instead of a fully connected graph.
    This model has the same architecture as the ShallowEEGGCNN but uses a sparse graph.
    """
    
    def __init__(self, num_node_features, sparsity_threshold=0.5, out_channels=1):
        """
        Initialize the sparse EEG-GCNN model.
        
        Args:
            num_node_features (int): Number of input features per node.
            sparsity_threshold (float): Threshold for edge pruning. Edges with weights below this threshold
                                        will be removed from the graph. Default is 0.5.
            out_channels (int): Number of output channels. Default is 1 for binary classification.
        """
        super(SparseEEGGCNN, self).__init__()
        
        self.sparsity_threshold = sparsity_threshold
        
        # Same architecture as ShallowEEGGCNN
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.out = nn.Linear(128, out_channels)
        
    def forward(self, data):
        """
        Forward pass with sparse graph.
        
        Args:
            data (torch_geometric.data.Data): Input graph data.
            
        Returns:
            torch.Tensor: Model output.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply sparsity threshold to create a sparse graph
        # Only keep edges with weights above the threshold
        mask = edge_attr.squeeze() > self.sparsity_threshold
        sparse_edge_index = edge_index[:, mask]
        sparse_edge_attr = edge_attr[mask]
        
        # First graph convolutional layer
        x = self.conv1(x, sparse_edge_index, sparse_edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        
        # Second graph convolutional layer
        x = self.conv2(x, sparse_edge_index, sparse_edge_attr)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        
        # Global mean pooling
        x = global_mean_pool(x, data.batch)
        
        # Output layer
        x = self.out(x)
        
        return x