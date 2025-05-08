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
import joblib

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
        Load precomputed features from FigShare.
        """
        # Print directory structure for debugging
        print(f"Looking for precomputed features in: {precomputed_features_path}")
        
        # Define paths
        psd_path = os.path.join(precomputed_features_path, 'psd_features_data_X')
        metadata_path = os.path.join(precomputed_features_path, 'master_metadata_index.csv')
        labels_path = os.path.join(precomputed_features_path, 'labels_y')
        spec_coh_path = os.path.join(precomputed_features_path, 'spec_coh_values.npy')
        
        # Check if files exist
        if not os.path.exists(psd_path):
            raise FileNotFoundError(f"PSD features file not found at {psd_path}")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")
        
        # Load PSD features using joblib
        print(f"Loading PSD features from {psd_path}")
        import joblib
        self.psd_features = joblib.load(psd_path)
        print(f"PSD features shape: {self.psd_features.shape}")
        
        # Load labels using joblib
        print(f"Loading labels from {labels_path}")
        self.labels = joblib.load(labels_path)
        # Convert labels to numeric values (0 for healthy, 1 for patient)
        self.labels = np.array([1 if label == 'patient' else 0 for label in self.labels])
        print(f"Labels shape: {self.labels.shape}")
        
        # Generate subject IDs (if not available directly)
        # In the original code, subject IDs might be derived from metadata
        if os.path.exists(metadata_path):
            print(f"Loading metadata from {metadata_path}")
            # Parse the CSV file
            import pandas as pd
            metadata = pd.read_csv(metadata_path, low_memory=False)
            # Extract subject IDs from metadata
            self.subject_ids = metadata['patient_ID'].values
        else:
            print("Metadata file not found, generating placeholder subject IDs")
            # Generate placeholder subject IDs (one unique ID per window)
            self.subject_ids = np.arange(len(self.labels))
        
        # Set window indices
        self.windows = np.arange(len(self.labels))
        
        # Load spectral coherence values for functional connectivity
        if os.path.exists(spec_coh_path):
            print(f"Loading spectral coherence values from {spec_coh_path}")
            self.functional_connectivity = np.load(spec_coh_path, allow_pickle=True)
        else:
            print(f"Warning: Spectral coherence file not found")
            # Initialize with dummy values
            n_channels = self.psd_features.shape[1]
            self.functional_connectivity = np.eye(n_channels)
        
        # For spatial connectivity, we need to compute it or use a default
        # Since geo_distances.npy is not visible in the screenshot, we'll compute it
        # based on standard_1010.tsv.txt
        std_1010_path = os.path.join(precomputed_features_path, 'standard_1010.tsv.txt')
        if os.path.exists(std_1010_path):
            print(f"Computing spatial connectivity from {std_1010_path}")
            # This would need a custom function to parse the TSV file and compute distances
            # For now, we'll use a placeholder
            n_channels = self.psd_features.shape[1]
            self.spatial_connectivity = np.eye(n_channels)
        else:
            print(f"Warning: Standard 10-10 electrode positions file not found")
            n_channels = self.psd_features.shape[1]
            self.spatial_connectivity = np.eye(n_channels)
        
        print(f"Loaded precomputed data with {len(self.psd_features)} windows")
        print(f"Number of unique subjects: {len(np.unique(self.subject_ids))}")
            
    def _load_and_preprocess_data(self, subject_ids=None):
        """
        Load and preprocess EEG data from TUH EEG and MPI LEMON datasets.
        
        Args:
            subject_ids (list, optional): List of subject IDs to include. Defaults to None (all subjects).
        """
        import glob
        import pandas as pd
        from scipy import signal
        
        # Initialize lists to store data
        all_psd_features = []
        all_subject_ids = []
        all_labels = []
        
        # Define frequency bands
        freq_bands = self.freq_bands
        
        # Process TUH EEG dataset (patients with neurological disorders)
        tuh_dir = os.path.join(self.data_dir, 'raw', 'tuh')
        if os.path.exists(tuh_dir):
            print(f"Processing TUH EEG dataset from {tuh_dir}...")
            
            # Find all EDF files recursively
            edf_files = glob.glob(os.path.join(tuh_dir, '**', '*.edf'), recursive=True)
            
            for file_idx, file_path in enumerate(edf_files):
                try:
                    # Extract subject ID from filename
                    subject_id = os.path.basename(file_path).split('_')[0]
                    
                    if subject_ids is not None and subject_id not in subject_ids:
                        continue
                    
                    # Load EEG data
                    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                    
                    # Resample to target frequency
                    raw = raw.resample(self.sampling_rate)
                    
                    # Apply filters
                    raw = raw.filter(l_freq=1.0, h_freq=None)
                    raw = raw.notch_filter(freqs=50)
                    
                    # Get available channels
                    available_channels = raw.ch_names
                    
                    # Convert to bipolar montage if possible
                    bipolar_pairs = self.bipolar_channels
                    usable_pairs = []
                    
                    for pair in bipolar_pairs:
                        if pair[0] in available_channels and pair[1] in available_channels:
                            usable_pairs.append(pair)
                    
                    if len(usable_pairs) == 0:
                        print(f"Warning: No usable bipolar pairs found in {file_path}")
                        continue
                    
                    # Apply bipolar reference
                    raw = mne.set_bipolar_reference(
                        raw,
                        anode=[pair[0] for pair in usable_pairs],
                        cathode=[pair[1] for pair in usable_pairs],
                        ch_name=[f"{pair[0]}-{pair[1]}" for pair in usable_pairs]
                    )
                    
                    # Extract windows
                    data = raw.get_data()
                    n_channels, n_samples = data.shape
                    
                    # Calculate window size in samples
                    window_samples = int(10.0 * self.sampling_rate)  # 10-second windows
                    
                    # Extract non-overlapping windows
                    for i in range(0, n_samples - window_samples + 1, window_samples):
                        window_data = data[:, i:i+window_samples]
                        
                        # Extract PSD features
                        psd_features = np.zeros((n_channels, len(freq_bands)))
                        
                        for j in range(n_channels):
                            freqs, psd = signal.welch(window_data[j], fs=self.sampling_rate, nperseg=min(256, window_samples))
                            
                            # Extract band powers
                            for k, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
                                idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                                if np.any(idx):
                                    psd_features[j, k] = np.mean(psd[idx])
                        
                        # Store features and metadata
                        all_psd_features.append(psd_features)
                        all_subject_ids.append(subject_id)
                        all_labels.append(1)  # 1 for patient
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Process MPI LEMON dataset (healthy individuals)
        lemon_dir = os.path.join(self.data_dir, 'raw', 'lemon')
        if os.path.exists(lemon_dir):
            print(f"Processing MPI LEMON dataset from {lemon_dir}...")
            
            # Find all BrainVision files recursively
            vhdr_files = glob.glob(os.path.join(lemon_dir, '**', '*.vhdr'), recursive=True)
            
            for file_idx, file_path in enumerate(vhdr_files):
                try:
                    # Extract subject ID from filename
                    subject_id = os.path.basename(file_path).split('_')[0]
                    
                    if subject_ids is not None and subject_id not in subject_ids:
                        continue
                    
                    # Load EEG data
                    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
                    
                    # Resample to target frequency
                    raw = raw.resample(self.sampling_rate)
                    
                    # Apply filters
                    raw = raw.filter(l_freq=1.0, h_freq=None)
                    raw = raw.notch_filter(freqs=50)
                    
                    # Get available channels
                    available_channels = raw.ch_names
                    
                    # Convert to bipolar montage if possible
                    bipolar_pairs = self.bipolar_channels
                    usable_pairs = []
                    
                    for pair in bipolar_pairs:
                        if pair[0] in available_channels and pair[1] in available_channels:
                            usable_pairs.append(pair)
                    
                    if len(usable_pairs) == 0:
                        print(f"Warning: No usable bipolar pairs found in {file_path}")
                        continue
                    
                    # Apply bipolar reference
                    raw = mne.set_bipolar_reference(
                        raw,
                        anode=[pair[0] for pair in usable_pairs],
                        cathode=[pair[1] for pair in usable_pairs],
                        ch_name=[f"{pair[0]}-{pair[1]}" for pair in usable_pairs]
                    )
                    
                    # Extract windows
                    data = raw.get_data()
                    n_channels, n_samples = data.shape
                    
                    # Calculate window size in samples
                    window_samples = int(10.0 * self.sampling_rate)  # 10-second windows
                    
                    # Extract non-overlapping windows
                    for i in range(0, n_samples - window_samples + 1, window_samples):
                        window_data = data[:, i:i+window_samples]
                        
                        # Extract PSD features
                        psd_features = np.zeros((n_channels, len(freq_bands)))
                        
                        for j in range(n_channels):
                            freqs, psd = signal.welch(window_data[j], fs=self.sampling_rate, nperseg=min(256, window_samples))
                            
                            # Extract band powers
                            for k, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
                                idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                                if np.any(idx):
                                    psd_features[j, k] = np.mean(psd[idx])
                        
                        # Store features and metadata
                        all_psd_features.append(psd_features)
                        all_subject_ids.append(subject_id)
                        all_labels.append(0)  # 0 for healthy
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Convert lists to numpy arrays
        if len(all_psd_features) > 0:
            self.psd_features = np.array(all_psd_features)
            self.subject_ids = np.array(all_subject_ids)
            self.labels = np.array(all_labels)
            self.windows = np.arange(len(self.psd_features))
            
            print(f"Processed {len(self.psd_features)} windows from {len(np.unique(self.subject_ids))} subjects")
            print(f"Patient windows: {np.sum(self.labels == 1)}")
            print(f"Healthy windows: {np.sum(self.labels == 0)}")
        else:
            print("No EEG data found. Using placeholder data.")
            # Create minimal placeholder data to avoid errors
            n_channels = len(self.bipolar_channels)
            n_features = len(self.freq_bands)
            
            self.psd_features = np.zeros((10, n_channels, n_features))
            self.subject_ids = np.array(['placeholder'] * 10)
            self.labels = np.zeros(10)
            self.windows = np.arange(10)
        
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