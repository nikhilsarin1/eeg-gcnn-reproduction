"""
Configuration settings for EEG-GCNN experiments.
"""

import os

# Path configurations
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = 'results'
SAVED_MODELS_DIR = 'saved_models'

# Dataset configurations
TUH_DATASET = {
    'name': 'tuh',
    'raw_dir': os.path.join(RAW_DATA_DIR, 'tuh'),
    'processed_file': os.path.join(PROCESSED_DATA_DIR, 'tuh_processed.npz'),
    'label': 1  # Patient
}

LEMON_DATASET = {
    'name': 'lemon',
    'raw_dir': os.path.join(RAW_DATA_DIR, 'lemon'),
    'processed_file': os.path.join(PROCESSED_DATA_DIR, 'lemon_processed.npz'),
    'label': 0  # Healthy
}

# EEG preprocessing configurations
EEG_CONFIG = {
    'fs': 250.0,  # Sampling frequency in Hz
    'window_size': 10.0,  # Window size in seconds
    'freq_bands': {
        'delta': (1, 4),
        'theta': (4, 7.5),
        'alpha': (7.5, 13),
        'lower_beta': (13, 16),
        'higher_beta': (16, 30),
        'gamma': (30, 40)
    },
    'bipolar_channels': [
        ('F7', 'F3'),
        ('F8', 'F4'),
        ('T7', 'C3'),
        ('T8', 'C4'),
        ('P7', 'P3'),
        ('P8', 'P4'),
        ('O1', 'P3'),
        ('O2', 'P4')
    ]
}

# Model configurations
MODEL_CONFIG = {
    'shallow': {
        'hidden_dims': [64, 128],
        'batch_norm': True,
        'dropout': 0.3
    },
    'deep': {
        'hidden_dims': [16, 16, 32, 64, 128],
        'batch_norm': True,
        'dropout': 0.3,
        'fc_dims': [30, 20]
    },
    'spatial_only': {
        'hidden_dims': [64, 128],
        'batch_norm': True,
        'dropout': 0.3
    },
    'functional_only': {
        'hidden_dims': [64, 128],
        'batch_norm': True,
        'dropout': 0.3
    },
    'sparse': {
        'hidden_dims': [64, 128],
        'batch_norm': True,
        'dropout': 0.3,
        'sparsity_threshold': 0.5
    }
}

# Training configurations
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 100,
    'lr': 0.001,
    'weight_decay': 0.0001,
    'patience': 10,
    'n_folds': 10,
    'test_split': 0.3,
    'seed': 42
}

# Evaluation configurations
EVAL_CONFIG = {
    'metrics': ['auc', 'balanced_accuracy', 'precision', 'recall', 'f1'],
    'visualize': True,
    'feature_importance': True
}