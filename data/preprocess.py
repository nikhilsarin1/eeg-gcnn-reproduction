"""
Script to preprocess EEG data from TUH and MPI LEMON datasets.
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy import signal
import h5py
import argparse
import glob
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.connectivity import get_electrode_positions, compute_spatial_connectivity, compute_functional_connectivity
from utils.github_downloader import download_eeg_gcnn_precomputed


def load_tuh_eeg(file_path):
    """
    Load EEG data from the TUH EEG Abnormal Corpus.
    
    Args:
        file_path (str): Path to the EEG file.
        
    Returns:
        mne.io.Raw: MNE Raw object containing the EEG data.
    """
    # Load EDF file using MNE
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    return raw


def load_mpi_lemon_eeg(file_path):
    """
    Load EEG data from the MPI LEMON dataset.
    
    Args:
        file_path (str): Path to the EEG file.
        
    Returns:
        mne.io.Raw: MNE Raw object containing the EEG data.
    """
    # Load BrainVision file using MNE
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
    
    return raw


def preprocess_eeg(raw, fs=250.0, bipolar_montage=True):
    """
    Preprocess an EEG recording.
    
    Args:
        raw (mne.io.Raw): MNE Raw object containing the EEG data.
        fs (float): Target sampling frequency in Hz.
        bipolar_montage (bool): Whether to convert to bipolar montage.
        
    Returns:
        mne.io.Raw: Preprocessed MNE Raw object.
    """
    # Resample to target frequency
    raw = raw.resample(fs)
    
    # Apply highpass filter at 1 Hz
    raw = raw.filter(l_freq=1.0, h_freq=None)
    
    # Apply notch filter at 50 Hz (power line noise)
    raw = raw.notch_filter(freqs=50)
    
    if bipolar_montage:
        # Define bipolar montage
        bipolar_pairs = [
            ('F7', 'F3'),
            ('F8', 'F4'),
            ('T7', 'C3'),
            ('T8', 'C4'),
            ('P7', 'P3'),
            ('P8', 'P4'),
            ('O1', 'P3'),
            ('O2', 'P4')
        ]
        
        # Create bipolar montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # Apply bipolar reference
        raw = mne.set_bipolar_reference(raw, anode=np.array([pair[0] for pair in bipolar_pairs]),
                                       cathode=np.array([pair[1] for pair in bipolar_pairs]),
                                       ch_name=[f"{pair[0]}-{pair[1]}" for pair in bipolar_pairs])
    
    return raw


def extract_windows(raw, window_size=10.0, overlap=0.0):
    """
    Extract windows from an EEG recording.
    
    Args:
        raw (mne.io.Raw): MNE Raw object containing the EEG data.
        window_size (float): Window size in seconds.
        overlap (float): Window overlap in seconds.
        
    Returns:
        list: List of MNE Epochs objects, one for each window.
    """
    fs = raw.info['sfreq']
    data = raw.get_data()
    n_channels, n_samples = data.shape
    
    # Calculate window and step sizes in samples
    window_samples = int(window_size * fs)
    step_samples = int((window_size - overlap) * fs)
    
    # Calculate number of windows
    n_windows = (n_samples - window_samples) // step_samples + 1
    
    # Extract windows
    windows = []
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        
        if end <= n_samples:
            window_data = data[:, start:end]
            windows.append(window_data)
    
    return windows


def compute_psd_features(window, fs=250.0, freq_bands=None):
    """
    Compute power spectral density features for a window.
    
    Args:
        window (numpy.ndarray): EEG data for one window with shape (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        freq_bands (dict): Dictionary mapping frequency band names to (low, high) frequency tuples.
        
    Returns:
        numpy.ndarray: PSD features with shape (n_channels, n_bands).
    """
    if freq_bands is None:
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 7.5),
            'alpha': (7.5, 13),
            'lower_beta': (13, 16),
            'higher_beta': (16, 30),
            'gamma': (30, 40)
        }
    
    n_channels, n_samples = window.shape
    n_bands = len(freq_bands)
    
    # Initialize PSD features array
    psd_features = np.zeros((n_channels, n_bands))
    
    # Compute PSD for each channel
    for i in range(n_channels):
        freqs, psd = signal.welch(window[i], fs=fs, nperseg=min(256, n_samples))
        
        # Extract band powers
        for j, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
            # Find indices of frequencies within this band
            idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            
            # Calculate mean power in this band
            if np.any(idx):
                psd_features[i, j] = np.mean(psd[idx])
    
    return psd_features


def preprocess_dataset(data_dir, output_dir, dataset_type='tuh', fs=250.0, window_size=10.0):
    """
    Preprocess an entire dataset.
    
    Args:
        data_dir (str): Directory containing the EEG data.
        output_dir (str): Directory to save the processed data.
        dataset_type (str): Type of dataset ('tuh' or 'lemon').
        fs (float): Sampling frequency in Hz.
        window_size (float): Window size in seconds.
        
    Returns:
        dict: Dictionary containing the preprocessed data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define frequency bands
    freq_bands = {
        'delta': (1, 4),
        'theta': (4, 7.5),
        'alpha': (7.5, 13),
        'lower_beta': (13, 16),
        'higher_beta': (16, 30),
        'gamma': (30, 40)
    }
    
    # Initialize arrays for storing features and metadata
    all_psd_features = []
    all_subject_ids = []
    all_window_indices = []
    all_labels = []  # 0 for healthy, 1 for patient
    
    # Get list of EEG files
    if dataset_type == 'tuh':
        # For TUH dataset, look for EDF files
        file_paths = glob.glob(os.path.join(data_dir, '**', '*.edf'), recursive=True)
        label = 1  # Patient
    elif dataset_type == 'lemon':
        # For LEMON dataset, look for BrainVision files
        file_paths = glob.glob(os.path.join(data_dir, '**', '*.vhdr'), recursive=True)
        label = 0  # Healthy
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Process each file
    for file_idx, file_path in enumerate(tqdm(file_paths, desc=f"Processing {dataset_type} files")):
        try:
            # Extract subject ID from the file path
            subject_id = f"{dataset_type}_{file_idx}"
            
            # Load and preprocess the EEG data
            if dataset_type == 'tuh':
                raw = load_tuh_eeg(file_path)
            else:  # dataset_type == 'lemon'
                raw = load_mpi_lemon_eeg(file_path)
            
            raw = preprocess_eeg(raw, fs=fs)
            
            # Extract windows
            windows = extract_windows(raw, window_size=window_size)
            
            # Compute PSD features for each window
            for window_idx, window in enumerate(windows):
                psd_features = compute_psd_features(window, fs=fs, freq_bands=freq_bands)
                
                # Store features and metadata
                all_psd_features.append(psd_features)
                all_subject_ids.append(subject_id)
                all_window_indices.append(window_idx)
                all_labels.append(label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert lists to arrays
    all_psd_features = np.array(all_psd_features)
    all_subject_ids = np.array(all_subject_ids)
    all_window_indices = np.array(all_window_indices)
    all_labels = np.array(all_labels)
    
    # Save processed data
    output_file = os.path.join(output_dir, f"{dataset_type}_processed.npz")
    np.savez(
        output_file,
        psd_features=all_psd_features,
        subject_ids=all_subject_ids,
        window_indices=all_window_indices,
        labels=all_labels
    )
    
    print(f"Processed {len(all_psd_features)} windows from {len(np.unique(all_subject_ids))} subjects")
    print(f"Saved processed data to {output_file}")
    
    return {
        'psd_features': all_psd_features,
        'subject_ids': all_subject_ids,
        'window_indices': all_window_indices,
        'labels': all_labels
    }


def compute_connectivity_matrices(data_dir, output_dir):
    """
    Compute and save connectivity matrices.
    
    Args:
        data_dir (str): Directory containing the processed data.
        output_dir (str): Directory to save the connectivity matrices.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get electrode positions
    electrode_positions = get_electrode_positions()
    
    # Compute spatial connectivity
    spatial_connectivity = compute_spatial_connectivity(electrode_positions)
    
    # Save spatial connectivity
    np.save(os.path.join(output_dir, 'spatial_connectivity.npy'), spatial_connectivity)
    
    # For functional connectivity, we would need to compute it for each window
    # This would be done during model training, so we'll just save the spatial connectivity for now
    
    print(f"Saved connectivity matrices to {output_dir}")


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Preprocess EEG data')
    
    parser.add_argument('--tuh_dir', type=str, default=None, help='Directory containing TUH EEG data')
    parser.add_argument('--lemon_dir', type=str, default=None, help='Directory containing MPI LEMON EEG data')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Directory to save processed data')
    parser.add_argument('--fs', type=float, default=250.0, help='Sampling frequency in Hz')
    parser.add_argument('--window_size', type=float, default=10.0, help='Window size in seconds')
    parser.add_argument('--use_github_lemon', action='store_true', help='Whether to use the LEMON dataset from GitHub')
    parser.add_argument('--github_lemon_url', type=str, default='https://github.com/OpenNeuroDatasets/ds000221', 
                        help='URL to the LEMON dataset on GitHub')
    parser.add_argument('--use_precomputed_eeg_gcnn', action='store_true', 
                        help='Download and use precomputed EEG-GCNN features')
    
    return parser.parse_args()


def main(args=None):
    """
    Main function.
    
    Args:
        args (argparse.Namespace, optional): Command-line arguments. If None, parse arguments.
    """
    if args is None:
        args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_precomputed_eeg_gcnn:
        print("Downloading precomputed EEG-GCNN features...")
        download_eeg_gcnn_precomputed(args.output_dir)
        print("Precomputed features downloaded successfully!")
        return
    
    # If using GitHub LEMON dataset, download it
    if args.use_github_lemon:
        from utils.github_downloader import download_lemon_dataset
        lemon_dir = os.path.join(os.path.dirname(args.output_dir), 'raw', 'lemon')
        download_lemon_dataset(os.path.dirname(lemon_dir), args.github_lemon_url)
        args.lemon_dir = lemon_dir
    
    # Process TUH dataset if specified
    if args.tuh_dir is not None:
        print(f"Processing TUH dataset from {args.tuh_dir}...")
        preprocess_dataset(args.tuh_dir, args.output_dir, dataset_type='tuh', fs=args.fs, window_size=args.window_size)
    
    # Process LEMON dataset if specified
    if args.lemon_dir is not None:
        print(f"Processing LEMON dataset from {args.lemon_dir}...")
        preprocess_dataset(args.lemon_dir, args.output_dir, dataset_type='lemon', fs=args.fs, window_size=args.window_size)
    
    # Compute connectivity matrices
    print("Computing connectivity matrices...")
    compute_connectivity_matrices(args.output_dir, args.output_dir)
    
    print("Preprocessing complete.")


if __name__ == '__main__':
    main()