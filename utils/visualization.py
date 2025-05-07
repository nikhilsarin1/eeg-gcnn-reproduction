"""
Visualization tools for EEG-GCNN model interpretation and results analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from torch_geometric.data import Data


def visualize_connectivity_matrix(connectivity_matrix, channel_names, title="Connectivity Matrix"):
    """
    Visualize a connectivity matrix between EEG channels.
    
    Args:
        connectivity_matrix (numpy.ndarray): Connectivity matrix of shape (n_channels, n_channels).
        channel_names (list): List of channel names.
        title (str): Title for the plot.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(connectivity_matrix, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Connectivity Strength", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(channel_names)))
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_xticklabels(channel_names)
    ax.set_yticklabels(channel_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(channel_names)):
        for j in range(len(channel_names)):
            text = ax.text(j, i, f"{connectivity_matrix[i, j]:.2f}",
                          ha="center", va="center", color="w" if connectivity_matrix[i, j] > 0.5 else "black")
    
    ax.set_title(title)
    fig.tight_layout()
    
    return fig


def visualize_spectral_features(psd_features, channel_names, freq_bands, class_labels=None):
    """
    Visualize spectral features across channels and frequency bands.
    
    Args:
        psd_features (numpy.ndarray): PSD features of shape (n_samples, n_channels, n_bands).
        channel_names (list): List of channel names.
        freq_bands (list): List of frequency band names.
        class_labels (numpy.ndarray, optional): Class labels for each sample. Default is None.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    n_channels = len(channel_names)
    n_bands = len(freq_bands)
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3*n_channels), sharex=True)
    
    if class_labels is not None:
        # Get unique class labels
        unique_labels = np.unique(class_labels)
        n_classes = len(unique_labels)
        class_names = ["Healthy" if label == 0 else "Patient" for label in unique_labels]
        
        # Calculate mean PSD values for each class
        class_means = []
        for label in unique_labels:
            mask = (class_labels == label)
            class_mean = np.mean(psd_features[mask], axis=0)
            class_means.append(class_mean)
        
        # Plot mean PSD values for each channel
        for i, channel in enumerate(channel_names):
            ax = axes[i] if n_channels > 1 else axes
            
            x = np.arange(n_bands)
            width = 0.35
            
            for j, (label, class_mean) in enumerate(zip(unique_labels, class_means)):
                ax.bar(x + j*width - width/2, class_mean[i], width, 
                       label=class_names[j], 
                       alpha=0.7)
            
            ax.set_title(f"Channel: {channel}")
            ax.set_ylabel("Power")
            ax.set_xticks(x)
            ax.set_xticklabels(freq_bands)
            
            if i == 0:
                ax.legend()
    
    else:
        # Plot mean PSD values across all samples
        mean_psd = np.mean(psd_features, axis=0)
        
        for i, channel in enumerate(channel_names):
            ax = axes[i] if n_channels > 1 else axes
            
            ax.bar(freq_bands, mean_psd[i])
            ax.set_title(f"Channel: {channel}")
            ax.set_ylabel("Power")
    
    plt.xlabel("Frequency Bands")
    fig.tight_layout()
    
    return fig


def visualize_feature_importance(model, loader, device="cpu"):
    """
    Visualize feature importance using gradients with respect to input features.
    
    Args:
        model (torch.nn.Module): Trained EEG-GCNN model.
        loader (torch_geometric.data.DataLoader): DataLoader with input data.
        device (str): Device to run the model on.
        
    Returns:
        numpy.ndarray: Feature importance scores of shape (n_channels, n_bands).
    """
    model.eval()
    model = model.to(device)
    
    # Get a single batch
    batch = next(iter(loader))
    batch = batch.to(device)
    
    # Enable gradient computation for input features
    batch.x.requires_grad_(True)
    
    # Forward pass
    output = model(batch)
    
    # Target is the predicted class
    target = output.argmax(dim=1) if output.shape[1] > 1 else torch.ones_like(output)
    
    # Backward pass
    output.backward(gradient=target)
    
    # Get gradients
    gradients = batch.x.grad.abs()
    
    # Average across the batch dimension
    feature_importance = gradients.mean(dim=0).detach().cpu().numpy()
    
    # Reshape to (n_channels, n_bands)
    n_channels = 8  # Assuming 8 channels
    n_bands = 6     # Assuming 6 frequency bands
    feature_importance = feature_importance.reshape(n_channels, n_bands)
    
    # Visualize
    channel_names = [
        'F7-F3', 'F8-F4', 'T7-C3', 'T8-C4',
        'P7-P3', 'P8-P4', 'O1-P3', 'O2-P4'
    ]
    
    freq_bands = [
        'delta', 'theta', 'alpha', 'lower_beta', 'higher_beta', 'gamma'
    ]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(feature_importance, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Feature Importance", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(freq_bands)))
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_xticklabels(freq_bands)
    ax.set_yticklabels(channel_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(channel_names)):
        for j in range(len(freq_bands)):
            text = ax.text(j, i, f"{feature_importance[i, j]:.2f}",
                          ha="center", va="center", color="w" if feature_importance[i, j] > 0.5 else "black")
    
    ax.set_title("Feature Importance")
    fig.tight_layout()
    
    return feature_importance, fig


def visualize_embeddings(model, loader, device="cpu"):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        model (torch.nn.Module): Trained EEG-GCNN model.
        loader (torch_geometric.data.DataLoader): DataLoader with input data.
        device (str): Device to run the model on.
        
    Returns:
        matplotlib.figure.Figure: Figure object.
    """
    model.eval()
    model = model.to(device)
    
    # Get all embeddings and labels
    embeddings = []
    labels = []
    subject_ids = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass to get embeddings (modified to return embeddings)
            # This requires modifying the model to return embeddings
            embedding = model.get_embedding(batch)
            
            # Store embeddings and labels
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch.y.detach().cpu().numpy())
            subject_ids.extend([s.item() for s in batch.subject_id])
    
    # Concatenate all embeddings and labels
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    
    # Plot t-SNE result
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                        c=labels, cmap='viridis', 
                        alpha=0.7, s=10)
    
    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)
    
    ax.set_title("t-SNE Visualization of Embeddings")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    
    fig.tight_layout()
    
    return fig


def add_embedding_method_to_models(model):
    """
    Add a method to get embeddings from the model.
    This is a monkey patch to add functionality without modifying the original code.
    
    Args:
        model (torch.nn.Module): EEG-GCNN model.
        
    Returns:
        torch.nn.Module: Modified model with get_embedding method.
    """
    # Define a function to get embeddings
    def get_embedding(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Process through convolution layers
        if hasattr(self, 'conv1'):
            x = self.conv1(x, edge_index, edge_attr)
            if hasattr(self, 'bn1'):
                x = self.bn1(x)
            x = torch.nn.functional.leaky_relu(x)
        
        if hasattr(self, 'conv2'):
            x = self.conv2(x, edge_index, edge_attr)
            if hasattr(self, 'bn2'):
                x = self.bn2(x)
            x = torch.nn.functional.leaky_relu(x)
        
        if hasattr(self, 'conv3'):
            x = self.conv3(x, edge_index, edge_attr)
            if hasattr(self, 'bn3'):
                x = self.bn3(x)
            x = torch.nn.functional.leaky_relu(x)
        
        if hasattr(self, 'conv4'):
            x = self.conv4(x, edge_index, edge_attr)
            if hasattr(self, 'bn4'):
                x = self.bn4(x)
            x = torch.nn.functional.leaky_relu(x)
        
        if hasattr(self, 'conv5'):
            x = self.conv5(x, edge_index, edge_attr)
            if hasattr(self, 'bn5'):
                x = self.bn5(x)
            x = torch.nn.functional.leaky_relu(x)
        
        # Global pooling to get graph-level embedding
        if hasattr(data, 'batch'):
            x = torch.nn.functional.global_mean_pool(x, data.batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x
    
    # Add the method to the model
    import types
    model.get_embedding = types.MethodType(get_embedding, model)
    
    return model


def compute_channel_contribution(model, loader, device="cpu"):
    """
    Compute the contribution of each channel to the classification.
    
    Args:
        model (torch.nn.Module): Trained EEG-GCNN model.
        loader (torch_geometric.data.DataLoader): DataLoader with input data.
        device (str): Device to run the model on.
        
    Returns:
        numpy.ndarray: Channel contribution scores.
    """
    model.eval()
    model = model.to(device)
    
    # Get a batch
    batch = next(iter(loader))
    batch = batch.to(device)
    
    # Initial prediction
    with torch.no_grad():
        original_output = model(batch)
    
    # Channel names
    channel_names = [
        'F7-F3', 'F8-F4', 'T7-C3', 'T8-C4',
        'P7-P3', 'P8-P4', 'O1-P3', 'O2-P4'
    ]
    
    # Initialize contribution scores
    n_channels = 8
    contribution_scores = np.zeros(n_channels)
    
    # For each channel, zero out its features and compute the change in output
    for i in range(n_channels):
        modified_x = batch.x.clone()
        
        # Zero out features for this channel
        modified_x[:, i, :] = 0
        
        # Create new batch with modified features
        modified_batch = Data(
            x=modified_x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            y=batch.y,
            batch=batch.batch if hasattr(batch, 'batch') else None
        )
        
        # Compute output with modified features
        with torch.no_grad():
            modified_output = model(modified_batch)
        
        # Compute change in output
        if modified_output.shape[1] > 1:
            # Multi-class
            original_probs = torch.nn.functional.softmax(original_output, dim=1)
            modified_probs = torch.nn.functional.softmax(modified_output, dim=1)
            
            # Compute KL divergence
            contribution = torch.nn.functional.kl_div(
                torch.log(modified_probs), original_probs, reduction='batchmean'
            ).item()
        else:
            # Binary classification
            contribution = torch.abs(original_output - modified_output).mean().item()
        
        contribution_scores[i] = contribution
    
    # Normalize scores
    if np.sum(contribution_scores) > 0:
        contribution_scores = contribution_scores / np.sum(contribution_scores)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(channel_names, contribution_scores)
    ax.set_ylabel("Contribution")
    ax.set_title("Channel Contribution to Classification")
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    return contribution_scores, fig