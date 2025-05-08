#!/bin/bash
# run_debug.sh - Run EEG-GCNN with debugging support

# Create directory structure
mkdir -p data/raw/tuh
mkdir -p data/raw/lemon
mkdir -p data/processed/eeg_gcnn
mkdir -p results

# First fix the PSD data format
python fix_psd_data.py

# Run main script with proper arguments
# From the error message, we can see the correct format of arguments

# Run preprocessing, training, and testing with precomputed features
python main.py --mode all \
               --model shallow \
               --load_precomputed \
               --use_precomputed_eeg_gcnn

echo "Run completed! Check results directory for output."