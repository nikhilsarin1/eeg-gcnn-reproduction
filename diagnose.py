#!/usr/bin/env python3
# diagnose.py - Standalone script to diagnose data loading and shape issues

import numpy as np
import os
import sys
from pathlib import Path

def main():
    print("=== EEG-GCNN Diagnostic Tool ===")
    print("Checking for precomputed features...")
    
    # Define paths
    base_path = Path("data/processed/eeg_gcnn")
    psd_path = base_path / "psd_features_data_X"
    labels_path = base_path / "labels_y"
    metadata_path = base_path / "master_metadata_index.csv"
    spec_coh_path = base_path / "spec_coh_values.npy"
    electrode_path = base_path / "standard_1010.tsv.txt"
    
    # Check if files exist
    for path in [psd_path, labels_path, metadata_path, spec_coh_path, electrode_path]:
        print(f"Checking {path}... {'Exists' if path.exists() else 'Not found'}")
    
    # Load data with allow_pickle=True
    print("\nLoading PSD features...")
    try:
        psd_features = np.load(psd_path, allow_pickle=True)
        print(f"PSD features shape: {psd_features.shape}")
        print(f"PSD features dtype: {psd_features.dtype}")
        print(f"First few values: {psd_features[:3] if len(psd_features) > 0 else 'No data'}")
    except Exception as e:
        print(f"Error loading PSD features: {e}")
    
    print("\nLoading labels...")
    try:
        labels = np.load(labels_path, allow_pickle=True)
        print(f"Labels shape: {labels.shape}")
        print(f"Labels dtype: {labels.dtype}")
        print(f"First few values: {labels[:5] if len(labels) > 0 else 'No data'}")
    except Exception as e:
        print(f"Error loading labels: {e}")
    
    print("\nLoading metadata...")
    try:
        # If it's a CSV file, use pandas
        import pandas as pd
        metadata = pd.read_csv(metadata_path)
        print(f"Metadata shape: {metadata.shape}")
        print(f"Metadata columns: {metadata.columns.tolist()}")
        print(f"First few rows: {metadata.head(2)}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
    
    print("\nLoading spectral coherence values...")
    try:
        spec_coh_values = np.load(spec_coh_path, allow_pickle=True)
        print(f"Spectral coherence values shape: {spec_coh_values.shape}")
        print(f"Spectral coherence values dtype: {spec_coh_values.dtype}")
        
        # Check first few values
        if isinstance(spec_coh_values, np.ndarray) and spec_coh_values.size > 0:
            if len(spec_coh_values.shape) == 1:
                print(f"First few values: {spec_coh_values[:5]}")
            else:
                print(f"First value shape: {spec_coh_values[0].shape if hasattr(spec_coh_values[0], 'shape') else 'Not an array'}")
        
        # Check for NaNs or infinities
        if np.issubdtype(spec_coh_values.dtype, np.number):
            nan_count = np.isnan(spec_coh_values).sum()
            inf_count = np.isinf(spec_coh_values).sum()
            print(f"NaN count in spectral coherence: {nan_count}")
            print(f"Infinity count in spectral coherence: {inf_count}")
    except Exception as e:
        print(f"Error loading spectral coherence values: {e}")
    
    print("\nComputing spatial connectivity...")
    try:
        if electrode_path.exists():
            with open(electrode_path, 'r') as f:
                electrode_data = [line.strip().split() for line in f if line.strip()]
                
            # Display first few electrode positions
            print(f"First few electrode positions: {electrode_data[:2]}")
            
            # Check if the data has the expected format
            if len(electrode_data) > 1 and len(electrode_data[1]) >= 4:
                try:
                    # Assume electrode_data has the format: [name, x, y, z]
                    electrode_positions = np.array([[float(row[1]), float(row[2]), float(row[3])] for row in electrode_data[1:]])
                    print(f"Electrode positions shape: {electrode_positions.shape}")
                    
                    # Compute geodesic distances between electrodes
                    from scipy.spatial.distance import pdist, squareform
                    dist_matrix = squareform(pdist(electrode_positions))
                    print(f"Distance matrix shape: {dist_matrix.shape}")
                    
                    # Normalize distances to [0, 1]
                    max_dist = np.max(dist_matrix)
                    spatial_connectivity = 1 - (dist_matrix / max_dist)
                    print(f"Spatial connectivity shape: {spatial_connectivity.shape}")
                    
                    # Check for NaNs or infinities
                    nan_count = np.isnan(spatial_connectivity).sum()
                    inf_count = np.isinf(spatial_connectivity).sum()
                    print(f"NaN count in spatial connectivity: {nan_count}")
                    print(f"Infinity count in spatial connectivity: {inf_count}")
                except Exception as e:
                    print(f"Error processing electrode data: {e}")
            else:
                print(f"Unexpected electrode data format")
                print(f"First line: {electrode_data[0] if electrode_data else 'No data'}")
        else:
            print(f"Electrode file not found: {electrode_path}")
    except Exception as e:
        print(f"Error computing spatial connectivity: {e}")
    
    print("\nSimulating a getitem call...")
    try:
        # Only proceed if we have loaded both features and coherence values
        if 'psd_features' in locals() and 'spec_coh_values' in locals():
            idx = 0  # Get first item
            
            # Get PSD feature for this window
            if idx < len(psd_features):
                psd_feature = psd_features[idx]
                print(f"Feature shape for window {idx}: {psd_feature.shape if hasattr(psd_feature, 'shape') else 'Not an array'}")
            else:
                print(f"Index {idx} out of bounds for psd_features with length {len(psd_features)}")
            
            # Try getting label for this window
            if 'labels' in locals() and idx < len(labels):
                label = labels[idx]
                print(f"Label for window {idx}: {label}")
            
            # Try getting functional connectivity for this window
            if len(spec_coh_values.shape) > 2:
                if idx < spec_coh_values.shape[0]:
                    functional_connectivity = spec_coh_values[idx]
                    print(f"Functional connectivity shape for window {idx}: {functional_connectivity.shape if hasattr(functional_connectivity, 'shape') else 'Not an array'}")
                else:
                    print(f"Index {idx} out of bounds for spec_coh_values with shape {spec_coh_values.shape}")
            else:
                print(f"Spectral coherence values has shape {spec_coh_values.shape}, not per-window")
                functional_connectivity = spec_coh_values
                print(f"Using global functional connectivity with shape: {functional_connectivity.shape if hasattr(functional_connectivity, 'shape') else 'Not an array'}")
            
            # Check if shapes match for combining
            if 'spatial_connectivity' in locals() and 'functional_connectivity' in locals():
                if hasattr(spatial_connectivity, 'shape') and hasattr(functional_connectivity, 'shape'):
                    print(f"Spatial connectivity shape: {spatial_connectivity.shape}")
                    print(f"Functional connectivity shape: {functional_connectivity.shape}")
                    
                    if spatial_connectivity.shape == functional_connectivity.shape:
                        print("Shapes match! Can combine matrices.")
                        edge_weights = (spatial_connectivity + functional_connectivity) / 2
                        print(f"Edge weights shape: {edge_weights.shape}")
                    else:
                        print("Shape mismatch! Cannot combine matrices directly.")
                        print("This is likely the cause of the error in your code.")
                        
                        # Try to reshape or resize functional_connectivity to match spatial_connectivity
                        if len(functional_connectivity.shape) == 1:
                            print("Trying to reshape 1D functional_connectivity...")
                            
                            # Check if it's a flattened square matrix
                            size = int(np.sqrt(len(functional_connectivity)))
                            if size * size == len(functional_connectivity):
                                reshaped = functional_connectivity.reshape(size, size)
                                print(f"Reshaped to {reshaped.shape}")
                                
                                # If the shapes still don't match, we might need to resize
                                if reshaped.shape != spatial_connectivity.shape:
                                    from scipy.ndimage import zoom
                                    target_shape = spatial_connectivity.shape
                                    zoom_factor = target_shape[0] / reshaped.shape[0]
                                    print(f"Resizing with zoom factor {zoom_factor}...")
                                    try:
                                        resized = zoom(reshaped, zoom_factor)
                                        print(f"Resized to {resized.shape}")
                                        if resized.shape == spatial_connectivity.shape:
                                            print("Shapes now match! Can combine matrices.")
                                            edge_weights = (spatial_connectivity + resized) / 2
                                            print(f"Edge weights shape: {edge_weights.shape}")
                                        else:
                                            print(f"Shapes still don't match after resizing: {resized.shape} vs {spatial_connectivity.shape}")
                                    except Exception as e:
                                        print(f"Error during resizing: {e}")
                            else:
                                print(f"Cannot reshape array of length {len(functional_connectivity)} to a square matrix")
                else:
                    print("One or both connectivity matrices do not have a shape attribute")
        else:
            print("Could not simulate getitem call because features or coherence values were not loaded")
    except Exception as e:
        print(f"Error simulating getitem call: {e}")
    
    print("\nAnalyzing project structure...")
    try:
        # Find dataset.py
        dataset_file_paths = list(Path('.').rglob('dataset.py'))
        if dataset_file_paths:
            dataset_file = dataset_file_paths[0]
            print(f"Found dataset.py at: {dataset_file}")
            
            # Look for __getitem__ method
            with open(dataset_file, 'r') as f:
                content = f.read()
                if '__getitem__' in content:
                    print("Found __getitem__ method in dataset.py")
                    # Find the relevant section with edge_weights calculation
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'edge_weights' in line and 'spatial_connectivity' in line and 'functional_connectivity' in line:
                            context = lines[max(0, i-5):min(len(lines), i+6)]
                            print("Found potential problematic code:")
                            print('\n'.join(context))
                            break
                else:
                    print("Could not find __getitem__ method in dataset.py")
        else:
            print("Could not find dataset.py")
            
        # Find other important files
        train_file_paths = list(Path('.').rglob('train.py'))
        if train_file_paths:
            print(f"Found train.py at: {train_file_paths[0]}")
        else:
            print("Could not find train.py")
            
        test_file_paths = list(Path('.').rglob('test.py'))
        if test_file_paths:
            print(f"Found test.py at: {test_file_paths[0]}")
        else:
            print("Could not find test.py")
            
        # Find main.py
        main_file_paths = list(Path('.').rglob('main.py'))
        if main_file_paths:
            print(f"Found main.py at: {main_file_paths[0]}")
        else:
            print("Could not find main.py")
    except Exception as e:
        print(f"Error analyzing project structure: {e}")
    
    print("\n=== Diagnostic Complete ===")
    print("\nRecommendation based on common issues:")
    print("1. Need to fix the shape mismatch between spatial and functional connectivity matrices")
    print("2. Check if spectral coherence values are in the expected format")
    print("3. Ensure that the dataset.__getitem__ method properly handles different shapes")

if __name__ == "__main__":
    main()