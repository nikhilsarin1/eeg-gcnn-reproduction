"""
Main script for running the EEG-GCNN reproduction experiments.
"""

import os
import argparse
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from experiments.train import main as train_main
from experiments.test import main as test_main
from data.preprocess import main as preprocess_main
from experiments.config import MODEL_CONFIG, TRAIN_CONFIG, EVAL_CONFIG
from data.preprocess import compute_connectivity_matrices


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run EEG-GCNN reproduction experiments')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='all', choices=['preprocess', 'train', 'test', 'all'],
                        help='Mode to run: preprocess, train, test, or all')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the EEG data')
    parser.add_argument('--tuh_dir', type=str, default=None, help='Directory containing TUH EEG data')
    parser.add_argument('--lemon_dir', type=str, default=None, help='Directory containing MPI LEMON EEG data')
    parser.add_argument('--precomputed_features', type=str, default=None, 
                        help='Path to precomputed features, ignored if mode is preprocess')
    parser.add_argument('--load_precomputed', action='store_true', 
                        help='Whether to load precomputed features, ignored if mode is preprocess')
    parser.add_argument('--use_github_lemon', action='store_true',
                        help='Whether to use the LEMON dataset from GitHub')
    parser.add_argument('--github_lemon_url', type=str, 
                        default='https://github.com/OpenNeuroDatasets/ds000221',
                        help='URL to the LEMON dataset on GitHub')
    parser.add_argument('--use_precomputed_eeg_gcnn', action='store_true',
                        help='Use precomputed EEG-GCNN features')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='shallow', 
                        choices=['shallow', 'deep', 'spatial_only', 'functional_only', 'sparse'],
                        help='Model architecture')
    
    # Ablation arguments
    parser.add_argument('--run_ablations', action='store_true', 
                        help='Whether to run ablation studies')
    parser.add_argument('--sparsity_threshold', type=float, default=0.5, 
                        help='Threshold for edge pruning in sparse model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['batch_size'], 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'], 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['lr'], 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=TRAIN_CONFIG['weight_decay'], 
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=TRAIN_CONFIG['patience'], 
                        help='Patience for early stopping')
    parser.add_argument('--n_folds', type=int, default=TRAIN_CONFIG['n_folds'], 
                        help='Number of folds for cross-validation')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='saved_models', 
                        help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', 
                        help='Whether to visualize results')
    parser.add_argument('--feature_importance', action='store_true', 
                        help='Whether to visualize feature importance')
    parser.add_argument('--seed', type=int, default=TRAIN_CONFIG['seed'], 
                        help='Random seed')
    
    return parser.parse_args()


def run_preprocess(args):
    """
    Run preprocessing.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If using precomputed EEG-GCNN features
    if args.use_precomputed_eeg_gcnn:
        print("\n=== Downloading precomputed EEG-GCNN features ===")
        from utils.github_downloader import download_eeg_gcnn_precomputed
        
        precomputed_dir = os.path.join(args.data_dir, 'processed', 'eeg_gcnn')
        os.makedirs(precomputed_dir, exist_ok=True)

        if os.path.isdir(precomputed_dir) and os.listdir(precomputed_dir):
            print(f"Precomputed features already in {precomputed_dir}, skipping download.")
            return
        
        # Download precomputed features
        success = download_eeg_gcnn_precomputed(precomputed_dir)
        
        if success:
            print(f"Successfully downloaded precomputed EEG-GCNN features to {precomputed_dir}")
            return
        else:
            print("Failed to download precomputed EEG-GCNN features")
            print("Please check your internet connection or try again later")
            sys.exit(1)
            
        # Create a metadata file to indicate we're using precomputed features
        with open(os.path.join(precomputed_dir, 'using_precomputed.txt'), 'w') as f:
            f.write("This directory contains precomputed EEG-GCNN features from FigShare.")
        
        print("Precomputed features setup complete.")
        return
    
    # Original preprocessing for TUH dataset
    if args.tuh_dir is not None:
        print(f"Processing TUH dataset from {args.tuh_dir}...")
        tuh_args = argparse.Namespace(
            tuh_dir=args.tuh_dir,
            lemon_dir=None,
            output_dir=args.output_dir,
            fs=250.0,
            window_size=10.0,
            use_github_lemon=False,
            github_lemon_url=None
        )
        preprocess_main(tuh_args)
    
    # Original preprocessing for LEMON dataset
    if args.lemon_dir is not None or args.use_github_lemon:
        print(f"Processing LEMON dataset...")
        lemon_args = argparse.Namespace(
            tuh_dir=None,
            lemon_dir=args.lemon_dir,
            output_dir=args.output_dir,
            fs=250.0,
            window_size=10.0,
            use_github_lemon=args.use_github_lemon,
            github_lemon_url=args.github_lemon_url
        )
        preprocess_main(lemon_args)
    
    # Compute connectivity matrices if not using precomputed features
    print("Computing connectivity matrices...")
    compute_connectivity_matrices(args.output_dir, args.output_dir)
    
    print("Preprocessing complete.")


def run_train(args):
    """
    Run training.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Train model
    train_args = argparse.Namespace(
        data_dir=args.data_dir,
        precomputed_features=args.precomputed_features,
        load_precomputed=args.load_precomputed,
        model=args.model,
        sparsity_threshold=args.sparsity_threshold if hasattr(args, 'sparsity_threshold') else 0.5,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        n_folds=args.n_folds,
        save_dir=args.save_dir,
        seed=args.seed,
        use_precomputed_eeg_gcnn=args.use_precomputed_eeg_gcnn
    )
    train_main(train_args)
    
    # Train ablation models if requested
    if args.run_ablations:
        models = ['spatial_only', 'functional_only', 'sparse']
        for model in models:
            print(f"\nTraining ablation model: {model}")
            train_args.model = model
            train_main(train_args)


def run_test(args):
    """
    Run testing.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Test model
    test_args = argparse.Namespace(
        data_dir=args.data_dir,
        precomputed_features=args.precomputed_features,
        load_precomputed=args.load_precomputed,
        test_split=TRAIN_CONFIG['test_split'],
        model=args.model,
        sparsity_threshold=args.sparsity_threshold,
        model_dir=args.save_dir,
        output_dir=args.output_dir,
        visualize=args.visualize,
        feature_importance=args.feature_importance,
        seed=args.seed,
        use_precomputed_eeg_gcnn=args.use_precomputed_eeg_gcnn if hasattr(args, 'use_precomputed_eeg_gcnn') else False
    )
    test_main(test_args)
    
    # Test ablation models if requested
    if args.run_ablations:
        models = ['spatial_only', 'functional_only', 'sparse']
        for model in models:
            print(f"\nTesting ablation model: {model}")
            test_args.model = model
            test_main(test_args)


def main():
    """
    Main function.
    """
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run mode
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n=== Running preprocessing ===")
        run_preprocess(args)
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n=== Running training ===")
        run_train(args)
    
    if args.mode == 'test' or args.mode == 'all':
        print("\n=== Running testing ===")
        run_test(args)
    
    print("\n=== All done! ===")


if __name__ == '__main__':
    main()