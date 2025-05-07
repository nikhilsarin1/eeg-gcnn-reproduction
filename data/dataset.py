"""
Dataset implementation for EEG-GCNN reproduction.
This file handles loading the EEG data, extracting features, and constructing graphs.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import mne
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal

class EEGGraphDataset(Dataset):
    """
    Dataset class for EEG graph data.
    
    Attributes:
        data_dir (str): Directory containing the EEG data.
        subject_ids (list): List of subject IDs.
        windows (list): List of EEG windows.
        labels (list): List of labels (0: healthy, 1: patient).
        transforms (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, data_dir, subject_ids=None, window_size=10, 
                 sampling_rate=250, transform=None, load_precomputed=False,
                 precomputed_features_path=None, use_github_lemon=True, github_lemon_url="https://github.com/OpenNeuroDatasets/ds000221"):
        """
        Initialize the EEG graph dataset.
        
        Args:
            data_dir (str): Directory containing the EEG data.
            subject_ids (list, optional): List of subject IDs to include. Defaults to None (all subjects).
            window_size (int, optional): Size of each window in seconds. Defaults to 10.
            sampling_rate (int, optional): Sampling rate in Hz. Defaults to 250.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_precomputed (bool, optional): Whether to load precomputed features. Defaults to False.
            precomputed_features_path (str, optional): Path to precomputed features. Required if load_precomputed is True.
            use_github_lemon (bool, optional): Whether to use the LEMON dataset from GitHub. Defaults to False.
            github_lemon_url (str, optional): URL to the LEMON dataset on GitHub. Defaults to "https://github.com/OpenNeuroDatasets/ds000221"
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.load_precomputed = load_precomputed
        self.use_github_lemon = use_github_lemon
        self.github_lemon_url = github_lemon_url
        
        # Frequency bands (Hz)
        self.freq_bands = {
            'delta': (1, 4),
            'theta': (4, 7.5),
            'alpha': (7.5, 13),
            'lower_beta': (13, 16),
            'higher_beta': (16, 30),
            'gamma': (30, 40)
        }
        
        # Selected bipolar channels
        self.bipolar_channels = [
            ('F7', 'F3'),  # F7-F3
            ('F8', 'F4'),  # F8-F4
            ('T7', 'C3'),  # T7-C3
            ('T8', 'C4'),  # T8-C4
            ('P7', 'P3'),  # P7-P3
            ('P8', 'P4'),  # P8-P4
            ('O1', 'P3'),  # O1-P3
            ('O2', 'P4')   # O2-P4
        ]
        
        if load_precomputed:
            if precomputed_features_path is None:
                raise ValueError("precomputed_features_path must be provided when load_precomputed is True")
            self._load_precomputed_features(precomputed_features_path)
        else:
            self._load_and_preprocess_data(subject_ids)
            
    def _load_precomputed_features(self, precomputed_features_path):
        """
        Load precomputed features from a file.
        
        Args:
            precomputed_features_path (str): Path to precomputed features.
        """
        # Load precomputed PSD features
        data = np.load(precomputed_features_path, allow_pickle=True)
        self.psd_features = data['psd_features']
        self.subject_ids = data['subject_ids']
        self.labels = data['labels']
        self.windows = data['windows']
        
        # Load precomputed connectivity matrices if available
        connectivity_path = os.path.join(os.path.dirname(precomputed_features_path), 'connectivity.npz')
        if os.path.exists(connectivity_path):
            connectivity_data = np.load(connectivity_path, allow_pickle=True)
            self.spatial_connectivity = connectivity_data['spatial_connectivity']
            self.functional_connectivity = connectivity_data['functional_connectivity']
        else:
            # Compute connectivity matrices
            self._compute_connectivity_matrices()
            
    def _load_and_preprocess_data(self, subject_ids=None):
        """
        Load and preprocess EEG data.
        
        Args:
            subject_ids (list, optional): List of subject IDs to include. Defaults to None (all subjects).
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, this would load and preprocess the TUH EEG and MPI LEMON datasets
        
        # For now, we'll just create some dummy data for demonstration
        print("Note: Using dummy data for demonstration. Replace with actual data loading.")
        
        n_subjects = 100 if subject_ids is None else len(subject_ids)
        n_windows_per_subject = 50
        n_channels = len(self.bipolar_channels)
        n_features = len(self.freq_bands)
        
        self.psd_features = np.random.rand(n_subjects * n_windows_per_subject, n_channels, n_features)
        self.subject_ids = np.repeat(np.arange(n_subjects), n_windows_per_subject)
        self.labels = np.random.randint(0, 2, n_subjects * n_windows_per_subject)
        self.windows = np.arange(n_subjects * n_windows_per_subject)
        
        # Compute connectivity matrices
        self._compute_connectivity_matrices()
        
    def _compute_connectivity_matrices(self):
        """
        Compute spatial and functional connectivity matrices.
        """
        n_channels = len(self.bipolar_channels)
        
        # Compute spatial connectivity (geodesic distances)
        # This would typically use the positions of the electrodes on a unit sphere
        # For now, we'll use a simplified approach with random distances
        self.spatial_connectivity = np.random.rand(n_channels, n_channels)
        self.spatial_connectivity = (self.spatial_connectivity + self.spatial_connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(self.spatial_connectivity, 0)  # Zero diagonal
        
        # Normalize spatial connectivity to [0, 1]
        self.spatial_connectivity = self.spatial_connectivity / np.max(self.spatial_connectivity)
        
        # Compute functional connectivity (spectral coherence)
        # In a real implementation, this would use the EEG time series data
        # For now, we'll use random values
        self.functional_connectivity = np.random.rand(n_channels, n_channels)
        self.functional_connectivity = (self.functional_connectivity + self.functional_connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(self.functional_connectivity, 1)  # One on diagonal
        
    def get_spatial_connectivity(self):
        """
        Calculate spatial connectivity matrix based on geodesic distances.
        
        Returns:
            np.ndarray: Spatial connectivity matrix.
        """
        # In the actual implementation, this would calculate geodesic distances
        # using the positions of the electrodes on a unit sphere
        return self.spatial_connectivity
    
    def get_functional_connectivity(self, window_idx):
        """
        Calculate functional connectivity matrix based on spectral coherence.
        
        Args:
            window_idx (int): Index of the window.
            
        Returns:
            np.ndarray: Functional connectivity matrix.
        """
        # In the actual implementation, this would calculate spectral coherence
        # using the EEG time series data for the given window
        return self.functional_connectivity
    
    def __len__(self):
        """
        Return the number of windows in the dataset.
        
        Returns:
            int: Number of windows.
        """
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            torch_geometric.data.Data: Graph data sample.
        """
        # Get PSD features for the window
        x = torch.FloatTensor(self.psd_features[idx])
        
        # Get label for the window
        y = torch.FloatTensor([self.labels[idx]])
        
        # Get subject ID for the window
        subject_id = self.subject_ids[idx]
        
        # Get connectivity matrices
        spatial_connectivity = self.get_spatial_connectivity()
        functional_connectivity = self.get_functional_connectivity(idx)
        
        # Combine spatial and functional connectivity
        # using the formula from the paper: A_ij = (A^s_ij + A^f_ij) / 2
        edge_weights = (spatial_connectivity + functional_connectivity) / 2
        
        # Create edge index and edge attributes
        # Edge index is a tensor of shape [2, num_edges] where each column represents an edge (source, target)
        # For a fully connected graph, we have n_channels * (n_channels - 1) edges
        n_channels = len(self.bipolar_channels)
        edge_index = []
        edge_attr = []
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:  # Exclude self-loops
                    edge_index.append([i, j])
                    edge_attr.append(edge_weights[i, j])
        
        edge_index = torch.LongTensor(edge_index).t()
        edge_attr = torch.FloatTensor(edge_attr).unsqueeze(1)
        
        # Create graph data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        # Add metadata
        data.subject_id = subject_id
        data.window_idx = idx
        
        if self.transform:
            data = self.transform(data)
            
        return data
    
    def get_subject_windows(self, subject_id):
        """
        Get all windows for a given subject.
        
        Args:
            subject_id (int): Subject ID.
            
        Returns:
            list: List of window indices for the subject.
        """
        return [i for i, sid in enumerate(self.subject_ids) if sid == subject_id]
    
    def get_unique_subjects(self):
        """
        Get unique subject IDs.
        
        Returns:
            list: List of unique subject IDs.
        """
        return np.unique(self.subject_ids)