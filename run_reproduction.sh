#!/bin/bash
# Script to run the complete EEG-GCNN reproduction

# Set environment variables
PYTHON=python3  # Change to your Python executable if needed

# Create directory structure
$PYTHON setup.py

# Parse command line arguments
PREPROCESS=0
TRAIN=0
TEST=0
RUN_ABLATIONS=0
MODEL="shallow"  # Default model
DOWNLOAD_DATA=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --preprocess)
      PREPROCESS=1
      shift
      ;;
    --train)
      TRAIN=1
      shift
      ;;
    --test)
      TEST=1
      shift
      ;;
    --all)
      PREPROCESS=1
      TRAIN=1
      TEST=1
      shift
      ;;
    --model)
      MODEL="$2"
      shift
      shift
      ;;
    --ablations)
      RUN_ABLATIONS=1
      shift
      ;;
    --download-data)
      DOWNLOAD_DATA=1
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --preprocess       Run preprocessing"
      echo "  --train            Run training"
      echo "  --test             Run testing"
      echo "  --all              Run preprocessing, training, and testing"
      echo "  --model MODEL      Specify model (shallow, deep, spatial_only, functional_only, sparse)"
      echo "  --ablations        Run ablation studies"
      echo "  --download-data    Download datasets automatically"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# If no options specified, run all
if [ $PREPROCESS -eq 0 ] && [ $TRAIN -eq 0 ] && [ $TEST -eq 0 ]; then
  PREPROCESS=1
  TRAIN=1
  TEST=1
fi

# Set up data directories
mkdir -p data/raw/tuh
mkdir -p data/raw/lemon
mkdir -p data/processed
mkdir -p saved_models
mkdir -p results

# Print configuration
echo "=== EEG-GCNN Reproduction ==="
echo "Model: $MODEL"
echo "Run preprocess: $PREPROCESS"
echo "Run training: $TRAIN"
echo "Run testing: $TEST"
echo "Run ablations: $RUN_ABLATIONS"
echo "==========================="

# Download data if requested
if [ $DOWNLOAD_DATA -eq 1 ]; then
  echo "Downloading datasets..."
  
  # Download LEMON dataset from GitHub
  echo "Downloading MPI LEMON dataset from GitHub..."
  mkdir -p temp_download
  cd temp_download
  
  # Clone only the latest commit with minimal depth to save time and space
  git clone --depth 1 https://github.com/OpenNeuroDatasets/ds000221.git
  
  # Move files to the appropriate location
  mv ds000221/* ../data/raw/lemon/
  
  # Clean up
  cd ..
  rm -rf temp_download
  
  echo "LEMON dataset downloaded."
  
  # For TUH dataset, provide instructions as it requires registration
  echo "Note: The TUH EEG dataset requires registration and cannot be downloaded automatically."
  echo "Please visit: https://www.isip.piconepress.com/projects/tuh_eeg/"
  echo "After downloading, place the files in the data/raw/tuh directory."
fi

# Step 1: Preprocess data
if [ $PREPROCESS -eq 1 ]; then
  echo "Step 1: Preprocessing data..."
  
  # Check if data is available
  if [ -d "data/raw/tuh" ] && [ "$(ls -A data/raw/tuh)" ]; then
    echo "TUH EEG data found."
  else
    echo "Warning: TUH EEG data not found in data/raw/tuh."
    echo "You can download it from: https://www.isip.piconepress.com/projects/tuh_eeg/"
    echo "Or you can use precomputed features with the --load_precomputed flag."
  fi
  
  if [ -d "data/raw/lemon" ] && [ "$(ls -A data/raw/lemon)" ]; then
    echo "MPI LEMON data found."
  else
    echo "Warning: MPI LEMON data not found in data/raw/lemon."
    echo "You can download it with the --download-data flag."
    echo "Or you can use precomputed features with the --load_precomputed flag."
  fi
  
  # Run preprocessing
  echo "Running preprocessing..."
  if [ $DOWNLOAD_DATA -eq 1 ]; then
    $PYTHON main.py --mode preprocess --use_github_lemon
  else
    $PYTHON main.py --mode preprocess
  fi
  
  echo "Preprocessing complete."
fi

# Step 2: Train model
if [ $TRAIN -eq 1 ]; then
  echo "Step 2: Training model(s)..."
  
  # Run training
  if [ $RUN_ABLATIONS -eq 1 ]; then
    echo "Running training with ablations..."
    $PYTHON main.py --mode train --model $MODEL --run_ablations
  else
    echo "Running training for model: $MODEL"
    $PYTHON main.py --mode train --model $MODEL
  fi
  
  echo "Training complete."
fi

# Step 3: Test model
if [ $TEST -eq 1 ]; then
  echo "Step 3: Testing model(s)..."
  
  # Run testing
  if [ $RUN_ABLATIONS -eq 1 ]; then
    echo "Running testing with ablations..."
    $PYTHON main.py --mode test --model $MODEL --run_ablations --visualize --feature_importance
  else
    echo "Running testing for model: $MODEL"
    $PYTHON main.py --mode test --model $MODEL --visualize --feature_importance
  fi
  
  echo "Testing complete."
fi

# Generate final report if all steps were run
if [ $PREPROCESS -eq 1 ] && [ $TRAIN -eq 1 ] && [ $TEST -eq 1 ]; then
  echo "Generating final report..."
  
  # Create summary of results
  echo "=== EEG-GCNN Reproduction Results ===" > results/summary.txt
  echo "Model: $MODEL" >> results/summary.txt
  echo "Date: $(date)" >> results/summary.txt
  echo >> results/summary.txt
  
  # Add results
  if [ -f "results/${MODEL}_test_metrics.npz" ]; then
    echo "Results available in: results/${MODEL}_test_metrics.npz" >> results/summary.txt
  fi
  
  # Add ablation results
  if [ $RUN_ABLATIONS -eq 1 ]; then
    echo >> results/summary.txt
    echo "Ablation results:" >> results/summary.txt
    for ablation in "spatial_only" "functional_only" "sparse"; do
      if [ -f "results/${ablation}_test_metrics.npz" ]; then
        echo "- $ablation: results/${ablation}_test_metrics.npz" >> results/summary.txt
      fi
    done
  fi
  
  echo >> results/summary.txt
  echo "Visualizations available in the results directory." >> results/summary.txt
  
  echo "Final report generated: results/summary.txt"
fi

echo "All done!"