"""
Functions for computing spatial and functional connectivity between EEG electrodes.
"""

import numpy as np
from scipy import signal
import math


def compute_geodesic_distance(pos_1, pos_2):
    """
    Compute the geodesic distance between two points on a unit sphere.
    
    Args:
        pos_1 (tuple): 3D coordinates of the first point (x, y, z).
        pos_2 (tuple): 3D coordinates of the second point (x, y, z).
        
    Returns:
        float: Geodesic distance between the two points.
    """
    # Normalize positions to unit vectors
    norm_1 = np.sqrt(np.sum(np.array(pos_1) ** 2))
    norm_2 = np.sqrt(np.sum(np.array(pos_2) ** 2))
    
    if norm_1 == 0 or norm_2 == 0:
        return float('inf')
    
    unit_1 = np.array(pos_1) / norm_1
    unit_2 = np.array(pos_2) / norm_2
    
    # Calculate the angle between the two unit vectors
    dot_product = np.clip(np.dot(unit_1, unit_2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    return angle


def compute_spatial_connectivity(channel_positions):
    """
    Compute the spatial connectivity matrix based on geodesic distances.
    
    Args:
        channel_positions (dict): Dictionary mapping channel names to 3D positions.
        
    Returns:
        numpy.ndarray: Spatial connectivity matrix.
    """
    n_channels = len(channel_positions)
    channels = list(channel_positions.keys())
    
    # Initialize connectivity matrix
    connectivity = np.zeros((n_channels, n_channels))
    
    # Compute pairwise geodesic distances
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                connectivity[i, j] = compute_geodesic_distance(
                    channel_positions[channels[i]],
                    channel_positions[channels[j]]
                )
    
    # Normalize connectivity matrix
    max_dist = np.max(connectivity)
    if max_dist > 0:
        connectivity = connectivity / max_dist
    
    # Invert distances to get connectivity (closer = stronger connection)
    connectivity = 1.0 - connectivity
    
    return connectivity


def compute_coherence(signal1, signal2, fs=250, nperseg=512):
    """
    Compute the coherence between two time series.
    
    Args:
        signal1 (numpy.ndarray): First time series.
        signal2 (numpy.ndarray): Second time series.
        fs (int): Sampling frequency in Hz.
        nperseg (int): Length of each segment for computing coherence.
        
    Returns:
        float: Mean coherence value in the frequency range of interest.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length")
    
    # Compute coherence
    f, Cxy = signal.coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    
    # Define frequency bands of interest
    freq_bands = {
        'delta': (1, 4),
        'theta': (4, 7.5),
        'alpha': (7.5, 13),
        'beta_low': (13, 16),
        'beta_high': (16, 30),
        'gamma': (30, 40)
    }
    
    # Calculate average coherence across all frequency bands
    avg_coherence = 0
    total_bandwidth = 0
    
    for band_name, (low_freq, high_freq) in freq_bands.items():
        # Find indices of frequencies within this band
        idx = np.logical_and(f >= low_freq, f <= high_freq)
        if np.any(idx):
            # Calculate mean coherence in this band
            band_coherence = np.mean(Cxy[idx])
            band_width = high_freq - low_freq
            avg_coherence += band_coherence * band_width
            total_bandwidth += band_width
    
    if total_bandwidth > 0:
        avg_coherence /= total_bandwidth
    
    return avg_coherence


def compute_functional_connectivity(eeg_data, channels, fs=250):
    """
    Compute the functional connectivity matrix based on spectral coherence.
    
    Args:
        eeg_data (numpy.ndarray): EEG data with shape (n_channels, n_samples).
        channels (list): List of channel names.
        fs (int): Sampling frequency in Hz.
        
    Returns:
        numpy.ndarray: Functional connectivity matrix.
    """
    n_channels = len(channels)
    
    # Initialize connectivity matrix
    connectivity = np.zeros((n_channels, n_channels))
    
    # Compute pairwise coherence
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            coh = compute_coherence(eeg_data[i], eeg_data[j], fs=fs)
            connectivity[i, j] = coh
            connectivity[j, i] = coh  # Symmetric matrix
    
    # Set diagonal to 1 (self-coherence)
    np.fill_diagonal(connectivity, 1.0)
    
    return connectivity


def combine_connectivity_matrices(spatial_conn, functional_conn, alpha=0.5):
    """
    Combine spatial and functional connectivity matrices.
    
    Args:
        spatial_conn (numpy.ndarray): Spatial connectivity matrix.
        functional_conn (numpy.ndarray): Functional connectivity matrix.
        alpha (float): Weight for spatial connectivity (1-alpha for functional).
        
    Returns:
        numpy.ndarray: Combined connectivity matrix.
    """
    if spatial_conn.shape != functional_conn.shape:
        raise ValueError("Connectivity matrices must have the same shape")
    
    # Linear combination of the two connectivity matrices
    combined = alpha * spatial_conn + (1 - alpha) * functional_conn
    
    return combined


def get_electrode_positions():
    """
    Return the standard 10-20 system electrode positions.
    
    Returns:
        dict: Mapping from electrode names to 3D coordinates.
    """
    # Define the positions of electrodes in the 10-20 system
    # Coordinates are on a unit sphere
    # Based on standard 10-20 system
    positions = {
        'Fp1': (-0.309, 0.851, 0.425),
        'Fp2': (0.309, 0.851, 0.425),
        'F7': (-0.628, 0.628, 0.459),
        'F3': (-0.410, 0.683, 0.605),
        'Fz': (0.000, 0.724, 0.690),
        'F4': (0.410, 0.683, 0.605),
        'F8': (0.628, 0.628, 0.459),
        'T3': (-0.887, 0.000, 0.461),  # Also known as T7
        'C3': (-0.545, 0.000, 0.839),
        'Cz': (0.000, 0.000, 1.000),
        'C4': (0.545, 0.000, 0.839),
        'T4': (0.887, 0.000, 0.461),  # Also known as T8
        'T5': (-0.628, -0.628, 0.459),  # Also known as P7
        'P3': (-0.410, -0.683, 0.605),
        'Pz': (0.000, -0.724, 0.690),
        'P4': (0.410, -0.683, 0.605),
        'T6': (0.628, -0.628, 0.459),  # Also known as P8
        'O1': (-0.309, -0.851, 0.425),
        'O2': (0.309, -0.851, 0.425),
        # Additional 10-20 system electrodes
        'Fpz': (0.000, 0.951, 0.309),
        'Oz': (0.000, -0.951, 0.309),
        # Alternative names for electrodes (newer nomenclature)
        'T7': (-0.887, 0.000, 0.461),  # Same as T3
        'T8': (0.887, 0.000, 0.461),   # Same as T4
        'P7': (-0.628, -0.628, 0.459), # Same as T5
        'P8': (0.628, -0.628, 0.459),  # Same as T6
    }
    
    return positions


def get_bipolar_positions():
    """
    Get the positions for bipolar electrode montage.
    
    Returns:
        dict: Mapping from bipolar channel names to 3D coordinates.
    """
    positions = get_electrode_positions()
    
    # Define bipolar montage as pairs of electrodes
    bipolar_pairs = {
        'F7-F3': ('F7', 'F3'),
        'F8-F4': ('F8', 'F4'),
        'T7-C3': ('T7', 'C3'),  # or T3-C3 in older nomenclature
        'T8-C4': ('T8', 'C4'),  # or T4-C4 in older nomenclature
        'P7-P3': ('P7', 'P3'),  # or T5-P3 in older nomenclature
        'P8-P4': ('P8', 'P4'),  # or T6-P4 in older nomenclature
        'O1-P3': ('O1', 'P3'),
        'O2-P4': ('O2', 'P4')
    }
    
    # Calculate the midpoint between each electrode pair
    bipolar_positions = {}
    for bipolar_name, (e1, e2) in bipolar_pairs.items():
        if e1 in positions and e2 in positions:
            pos1 = np.array(positions[e1])
            pos2 = np.array(positions[e2])
            midpoint = (pos1 + pos2) / 2
            # Normalize to unit sphere
            norm = np.linalg.norm(midpoint)
            if norm > 0:
                midpoint = midpoint / norm
            bipolar_positions[bipolar_name] = tuple(midpoint)
    
    return bipolar_positions