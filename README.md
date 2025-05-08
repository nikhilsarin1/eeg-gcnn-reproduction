# EEG-GCNN: Graph Convolutional Neural Networks for EEG-Based Neurological Disease Diagnosis

This repository contains a reproduction attempt of the paper ["EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network"](https://arxiv.org/abs/2011.12107) by Wagh and Varatharajah (2020).

## Overview

Neurological disorders affect approximately 1 billion people worldwide, with electroencephalography (EEG) serving as a primary diagnostic tool. However, visual identification of abnormalities in EEG suffers from a major limitation: approximately 50% of EEGs from patients with seizures are deemed "normal" based on expert visual review. This low sensitivity creates delays in diagnosis and treatment.

The EEG-GCNN approach demonstrates significant improvement (AUC 0.90) in distinguishing "normal" EEGs of patients with neurological diseases from healthy individuals' EEGs by utilizing a graph-based representation that captures both spatial and functional brain connectivity.

## Reproducibility Challenges

**Note:** Despite multiple attempts, I encountered significant challenges in reproducing the results from the original paper:

1. **Initial Raw Data Approach**: I first attempted to use raw data from both TUH and LEMON datasets to reproduce the results following the paper's methodology. However, this approach failed due to incompatibilities between the datasets and the preprocessing pipeline. I encountered a critical error in the data processing phase where the electrode montage conversion between the different EEG datasets could not be properly aligned.

2. **Precomputed Features Approach**: Following the original paper's GitHub repository instructions, I then tried using the precomputed features approach. This led to a critical pickle error when loading the data:

```
Traceback (most recent call last):
File "/Users/nikhilsarin/eeg-gcnn-project/main.py", line 256, in <module> main()
File "/Users/nikhilsarin/eeg-gcnn-project/main.py", line 246, in main run_train(args)
File "/Users/nikhilsarin/eeg-gcnn-project/main.py", line 181, in run_train train_main(train_args)
File "/Users/nikhilsarin/eeg-gcnn-project/experiments/train.py", line 314, in main dataset = EEGGraphDataset(
File "/Users/nikhilsarin/eeg-gcnn-project/data/dataset.py", line 77, in __init__ self._load_precomputed_features(precomputed_features_path)
File "/Users/nikhilsarin/eeg-gcnn-project/data/dataset.py", line 107, in _load_precomputed_features self.psd_features = np.load(psd_path, allow_pickle=True)
pickle.UnpicklingError: Failed to interpret file 'data/processed/eeg_gcnn/psd_features_data_X' as a pickle
```

3. **Data Format Mismatch**: Even after attempting to fix the pickle error, subsequent data loading issues emerged with shape mismatches between the spatial connectivity matrix (71,71) and the functional connectivity matrix (64,), preventing the successful training of the model.

These challenges highlight the difficulties in reproducing complex deep learning research, particularly in the neurological domain where data preprocessing and handling are especially intricate.

## Key Contributions of Original Paper

1. Novel graph representation for EEG data capturing both spatial and functional connectivity
2. First large-scale evaluation (1,593 subjects) of distinguishing "normal" patient EEGs from healthy EEGs
3. State-of-the-art performance (AUC 0.90) outperforming both human experts and classical ML baselines

## Repository Structure

```
eeg-gcnn-reproduction/
├── data/                # Data preprocessing and dataset classes
│   ├── dataset.py       # Dataset implementation
│   ├── preprocess.py    # Preprocessing pipeline
│   ├── raw/             # Raw data (not included in repo)
│   └── processed/       # Processed data (not included in repo)
├── models/              # Model implementations
│   └── eeg_gcnn.py      # EEG-GCNN model and variants
├── utils/               # Utility functions
│   ├── connectivity.py  # Connectivity calculations
│   ├── evaluation.py    # Evaluation metrics
│   └── visualization.py # Visualization tools
├── experiments/         # Training and testing scripts
│   ├── train.py         # Training script
│   ├── test.py          # Testing script
│   └── config.py        # Configuration settings
├── notebooks/           # Jupyter notebooks for analysis
├── results/             # Results directory (not included in repo)
├── saved_models/        # Saved model checkpoints (not included in repo)
├── run_reproduction.sh  # Main script to run the reproduction
├── setup.py             # Setup script
├── main.py              # Main entry point
└── README.md            # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/nikhilsarin1/eeg-gcnn-reproduction.git
cd eeg-gcnn-reproduction
```

2. Create a conda environment:
```bash
conda create -n eeg-gcnn python=3.9
conda activate eeg-gcnn
```

3. Install required packages:
```bash
# Core packages
conda install numpy pandas matplotlib seaborn scikit-learn
conda install mne -c conda-forge

# PyTorch packages
conda install pytorch -c pytorch
conda install pyg -c pyg
```

## Data

This project uses two public EEG datasets:

1. **Temple University Hospital (TUH) EEG Corpus** - Contains "normal" EEGs from patients with neurological disorders.
   - Note: You must request access to the TUH EEG Corpus before you can download the data. Approval from the dataset administrators is required.
   - Download: [https://www.isip.piconepress.com/projects/tuh_eeg/](https://www.isip.piconepress.com/projects/tuh_eeg/)
   - Place in `data/raw/tuh/`

2. **Max Planck Institute Leipzig Mind-Brain-Body (MPI LEMON) Dataset** - Contains EEGs from healthy individuals.
   - Download: [https://openneuro.org/datasets/ds000221/versions/00002](https://openneuro.org/datasets/ds000221/versions/00002)
   - Alternatively, can be downloaded automatically with the `--download-data` flag
   - Place in `data/raw/lemon/`

## Usage

### Complete Reproduction

To run the complete reproduction pipeline (preprocessing, training, and testing):

```bash
bash run_reproduction.sh --all
```

**Note:** As mentioned in the Reproducibility Challenges section, running this script will likely result in errors due to the issues described above.

### Step-by-Step Execution

1. **Preprocessing**:
```bash
bash run_reproduction.sh --preprocess
```

2. **Training**:
```bash
bash run_reproduction.sh --train --model shallow
```

3. **Testing**:
```bash
bash run_reproduction.sh --test --model shallow --visualize
```

### Precomputed Features

To use the precomputed features (though this also encounters errors as described above):

```bash
bash run_reproduction.sh --all --use-precomputed-eeg-gcnn
```

## Implementation Details

### Model Architecture

- **Shallow EEG-GCNN**: 2 Graph Convolution layers (output dimensions: 64, 128) + Global Mean Pooling
- **Deep EEG-GCNN**: 5 Graph Convolution layers (output dimensions: 16, 16, 32, 64, 128) + Global Mean Pooling + 2 hidden Linear layers (dimensions: 30, 20)

### Graph Representation

- **Nodes**: 8 EEG channels (bipolar montage)
- **Node Features**: 6 frequency band powers (delta, theta, alpha, lower beta, higher beta, gamma)
- **Edge Weights**: Combination of spatial connectivity (geodesic distance) and functional connectivity (spectral coherence)

### Training

- **Cross-validation**: 10-fold with disjoint subjects
- **Optimization**: Adam optimizer with learning rate 0.001
- **Early Stopping**: Patience of 10 epochs based on validation AUC
- **Class Imbalance**: Weighted loss function based on class distribution

## Results from Original Paper

Results comparing the EEG-GCNN approach with baselines:

| Model | AUC | Balanced Accuracy | Precision | Recall | F1 |
|-------|-----|-------------------|-----------|--------|----| 
| Shallow EEG-GCNN | 0.90 | 0.85 | 0.99 | 0.72 | 0.83 |
| Deep EEG-GCNN | 0.90 | 0.85 | 0.99 | 0.74 | 0.84 |
| FCNN (baseline) | 0.71 | 0.66 | 0.94 | 0.66 | 0.77 |
| Random Forest (baseline) | 0.80 | 0.74 | 0.95 | 0.79 | 0.86 |

## Computational Requirements

- **Training Time**: ~2-3 hours per model on a single GPU
- **GPU Memory**: ~2-4GB
- **Disk Space**: ~5GB for processed data and results

## Acknowledgements

This project is based on the work of Neeraj Wagh and Yogatheesan Varatharajah. We thank the authors for making their paper and code publicly available.

## Citation

If you use this code in your research, please cite the original paper:

```
@article{wagh2020eeg,
  title={EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network},
  author={Wagh, Neeraj and Varatharajah, Yogatheesan},
  journal={Proceedings of Machine Learning Research},
  pages={1--12},
  year={2020}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.