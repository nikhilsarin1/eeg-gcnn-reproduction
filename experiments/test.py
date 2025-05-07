"""
Testing script for the EEG-GCNN model.
This file handles the evaluation of trained models on the test set.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import EEGGraphDataset
from models.eeg_gcnn import ShallowEEGGCNN, DeepEEGGCNN, SpatialOnlyEEGGCNN, FunctionalOnlyEEGGCNN, SparseEEGGCNN
from utils.evaluation import (
    aggregate_windows_to_subjects, 
    plot_roc_curve, 
    plot_precision_recall_curve, 
    plot_confusion_matrix, 
    compute_metrics,
    visualize_feature_importance
)


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Test EEG-GCNN model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the EEG data')
    parser.add_argument('--precomputed_features', type=str, default=None, help='Path to precomputed features')
    parser.add_argument('--load_precomputed', action='store_true', help='Whether to load precomputed features')
    parser.add_argument('--test_split', type=float, default=0.3, help='Fraction of data to use for testing')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='shallow', choices=['shallow', 'deep', 'spatial_only', 'functional_only', 'sparse'], help='Model architecture')
    parser.add_argument('--sparsity_threshold', type=float, default=0.5, help='Threshold for edge pruning in sparse model')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='Directory containing saved models')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--visualize', action='store_true', help='Whether to visualize results')
    parser.add_argument('--feature_importance', action='store_true', help='Whether to visualize feature importance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def load_dataset_and_models(args):
    """
    Load dataset and trained models.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
        
    Returns:
        tuple: Dataset, model states, and fold metrics.
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print(f'Loading dataset...')
    if args.load_precomputed:
        dataset = EEGGraphDataset(
            data_dir=args.data_dir,
            load_precomputed=True,
            precomputed_features_path=args.precomputed_features
        )
    else:
        dataset = EEGGraphDataset(
            data_dir=args.data_dir
        )
    
    print(f'Dataset loaded with {len(dataset)} windows')
    
    # Load model states and fold metrics
    model_states_path = os.path.join(args.model_dir, f'{args.model}_model_states.pt')
    fold_metrics_path = os.path.join(args.model_dir, f'{args.model}_fold_metrics.pt')
    
    model_states = torch.load(model_states_path)
    fold_metrics = torch.load(fold_metrics_path)
    
    print(f'Loaded {len(model_states)} trained models')
    
    return dataset, model_states, fold_metrics


def evaluate_on_test_set(dataset, model_states, args):
    """
    Evaluate trained models on the test set.
    
    Args:
        dataset (EEGGraphDataset): Dataset.
        model_states (dict): Dictionary of model state dicts for each fold.
        args (argparse.Namespace): Command-line arguments.
        
    Returns:
        tuple: Dictionary of test metrics for each fold and average metrics.
    """
    # Get unique subject IDs
    unique_subjects = dataset.get_unique_subjects()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Shuffle subjects
    np.random.shuffle(unique_subjects)
    
    # Split subjects into train and test
    test_size = int(len(unique_subjects) * args.test_split)
    test_subjects = unique_subjects[:test_size]
    
    print(f'Using {len(test_subjects)} subjects for testing')
    
    # Get window indices for test subjects
    test_windows = []
    for subject_id in test_subjects:
        test_windows.extend(dataset.get_subject_windows(subject_id))
    
    print(f'Using {len(test_windows)} windows for testing')
    
    # Create test data loader
    test_loader = DataLoader([dataset[i] for i in test_windows], batch_size=64, shuffle=False)
    
    # Initialize dictionary to store test metrics for each fold
    test_metrics = {}
    
    # Initialize arrays to store subject-level predictions and targets for each fold
    all_subject_preds = []
    all_subject_targets = []
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate each fold model on the test set
    for fold, model_state in model_states.items():
        print(f'\nEvaluating fold {fold+1} model...')
        
        # Create model
        if args.model == 'shallow':
            model = ShallowEEGGCNN(num_node_features=dataset.psd_features.shape[2])
        elif args.model == 'deep':
            model = DeepEEGGCNN(num_node_features=dataset.psd_features.shape[2])
        elif args.model == 'spatial_only':
            model = SpatialOnlyEEGGCNN(num_node_features=dataset.psd_features.shape[2])
        elif args.model == 'functional_only':
            model = FunctionalOnlyEEGGCNN(num_node_features=dataset.psd_features.shape[2])
        elif args.model == 'sparse':
            model = SparseEEGGCNN(num_node_features=dataset.psd_features.shape[2], sparsity_threshold=args.sparsity_threshold)
        else:
            raise ValueError(f'Unknown model: {args.model}')
            
        # Load model state
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        
        # Initialize arrays for test predictions and targets
        test_preds = []
        test_targets = []
        test_subject_ids = []
        
        # Evaluate model on test set
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                # Forward pass
                outputs = model(batch).squeeze()
                
                # Store predictions and targets
                test_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
                test_targets.append(batch.y.detach().cpu().numpy())
                test_subject_ids.extend([s.item() for s in batch.subject_id])
        
        # Concatenate predictions and targets
        test_preds = np.concatenate([p for p in test_preds])
        test_targets = np.concatenate([t for t in test_targets])
        
        # Aggregate window-level predictions to subject-level
        subject_preds, subject_targets = aggregate_windows_to_subjects(test_preds, test_targets, test_subject_ids)
        
        # Store subject-level predictions and targets for this fold
        all_subject_preds.append(subject_preds)
        all_subject_targets.append(subject_targets)
        
        # Compute metrics
        metrics = compute_metrics(subject_targets, subject_preds)
        
        # Store metrics for this fold
        test_metrics[fold] = metrics
        
        # Print fold metrics
        print(f'Fold {fold+1} test metrics:')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')
    
    # Compute average metrics across folds
    avg_metrics = {}
    std_metrics = {}
    
    for metric in test_metrics[0].keys():
        avg_metrics[metric] = np.mean([test_metrics[fold][metric] for fold in range(len(model_states))])
        std_metrics[metric] = np.std([test_metrics[fold][metric] for fold in range(len(model_states))])
        
        print(f'Average {metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}')
    
    # Store average predictions and targets
    avg_subject_preds = np.mean(all_subject_preds, axis=0)
    avg_subject_targets = all_subject_targets[0]  # All folds have the same targets
    
    return test_metrics, avg_metrics, std_metrics, avg_subject_preds, avg_subject_targets


def visualize_results(avg_subject_preds, avg_subject_targets, args):
    """
    Visualize evaluation results.
    
    Args:
        avg_subject_preds (numpy.ndarray): Average subject-level predictions.
        avg_subject_targets (numpy.ndarray): Subject-level targets.
        args (argparse.Namespace): Command-line arguments.
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(avg_subject_targets, avg_subject_preds, title=f'{args.model} ROC Curve')
    plt.savefig(os.path.join(args.output_dir, f'{args.model}_roc_curve.png'))
    
    # Plot precision-recall curve
    avg_precision, best_f1, best_threshold = plot_precision_recall_curve(
        avg_subject_targets, avg_subject_preds, title=f'{args.model} Precision-Recall Curve'
    )
    plt.savefig(os.path.join(args.output_dir, f'{args.model}_pr_curve.png'))
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(
        avg_subject_targets, avg_subject_preds, threshold=best_threshold, title=f'{args.model} Confusion Matrix'
    )
    plt.savefig(os.path.join(args.output_dir, f'{args.model}_confusion_matrix.png'))
    
    print(f'Visualizations saved to {args.output_dir}')


def visualize_feature_importance_wrapper(dataset, model_states, args):
    """
    Wrapper function to visualize feature importance.
    
    Args:
        dataset (EEGGraphDataset): Dataset.
        model_states (dict): Dictionary of model state dicts for each fold.
        args (argparse.Namespace): Command-line arguments.
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    if args.model == 'shallow':
        model = ShallowEEGGCNN(num_node_features=dataset.psd_features.shape[2])
    elif args.model == 'deep':
        model = DeepEEGGCNN(num_node_features=dataset.psd_features.shape[2])
    elif args.model == 'spatial_only':
        model = SpatialOnlyEEGGCNN(num_node_features=dataset.psd_features.shape[2])
    elif args.model == 'functional_only':
        model = FunctionalOnlyEEGGCNN(num_node_features=dataset.psd_features.shape[2])
    elif args.model == 'sparse':
        model = SparseEEGGCNN(num_node_features=dataset.psd_features.shape[2], sparsity_threshold=args.sparsity_threshold)
    else:
        raise ValueError(f'Unknown model: {args.model}')
        
    # Load model state from the first fold
    model.load_state_dict(model_states[0])
    model = model.to(device)
    
    # Visualize feature importance
    feature_importance = visualize_feature_importance(model, dataset, device=device)
    plt.savefig(os.path.join(args.output_dir, f'{args.model}_feature_importance.png'))
    
    print(f'Feature importance visualization saved to {args.output_dir}')


def main(args):
    """
    Main function.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Load dataset and models
    dataset, model_states, fold_metrics = load_dataset_and_models(args)
    
    # Evaluate on test set
    test_metrics, avg_metrics, std_metrics, avg_subject_preds, avg_subject_targets = evaluate_on_test_set(
        dataset, model_states, args
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    np.savez(
        os.path.join(args.output_dir, f'{args.model}_test_metrics.npz'),
        test_metrics=test_metrics,
        avg_metrics=avg_metrics,
        std_metrics=std_metrics
    )
    
    # Visualize results if requested
    if args.visualize:
        visualize_results(avg_subject_preds, avg_subject_targets, args)
    
    # Visualize feature importance if requested
    if args.feature_importance:
        visualize_feature_importance_wrapper(dataset, model_states, args)
    
    print(f'Testing complete. Results saved to {args.output_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)