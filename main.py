"""
Main script for running the EEG-GCNN reproduction experiments.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from experiments.train import main as train_main
from experiments.test import main as test_main
from data.preprocess import main as preprocess_main
from experiments.config import MODEL_CONFIG, TRAIN_CONFIG, EVAL_CONFIG


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
    parser.add_argument('--precomputed_features', type=str, default=None, 
                        help='Path to precomputed features, ignored if mode is preprocess')
    parser.add_argument('--load_precomputed', action='store_true', 
                        help='Whether to load precomputed features, ignored if mode is preprocess')
    parser.add_argument('--use_github_lemon', action='store_true',
                        help='Whether to use the LEMON dataset from GitHub')
    parser.add_argument('--github_lemon_url', type=str, 
                        default='https://github.com/OpenNeuroDatasets/ds000221',
                        help='URL to the LEMON dataset on GitHub')
    
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
    # Preprocess TUH dataset
    tuh_args = argparse.Namespace(
        tuh_dir=os.path.join(args.data_dir, 'raw', 'tuh'),
        lemon_dir=None,
        output_dir=os.path.join(args.data_dir, 'processed'),
        fs=250.0,
        window_size=10.0,
        use_github_lemon=args.use_github_lemon,
        github_lemon_url=args.github_lemon_url
    )
    preprocess_main(tuh_args)
    
    # Preprocess LEMON dataset
    lemon_args = argparse.Namespace(
        tuh_dir=None,
        lemon_dir=os.path.join(args.data_dir, 'raw', 'lemon'),
        output_dir=os.path.join(args.data_dir, 'processed'),
        fs=250.0,
        window_size=10.0,
        use_github_lemon=args.use_github_lemon,
        github_lemon_url=args.github_lemon_url
    )
    preprocess_main(lemon_args)


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
        sparsity_threshold=args.sparsity_threshold,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        n_folds=args.n_folds,
        save_dir=args.save_dir,
        seed=args.seed
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
        seed=args.seed
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