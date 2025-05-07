"""
Training script for the EEG-GCNN model.
This file handles the training process, including cross-validation and hyperparameter tuning.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import EEGGraphDataset
from models.eeg_gcnn import ShallowEEGGCNN, DeepEEGGCNN, SpatialOnlyEEGGCNN, FunctionalOnlyEEGGCNN, SparseEEGGCNN
from utils.evaluation import aggregate_windows_to_subjects

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train EEG-GCNN model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the EEG data')
    parser.add_argument('--precomputed_features', type=str, default=None, help='Path to precomputed features')
    parser.add_argument('--load_precomputed', action='store_true', help='Whether to load precomputed features')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='shallow', choices=['shallow', 'deep', 'spatial_only', 'functional_only', 'sparse'], help='Model architecture')
    parser.add_argument('--sparsity_threshold', type=float, default=0.5, help='Threshold for edge pruning in sparse model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds for cross-validation')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, criterion, optimizer, args):
    """
    Train a model for a specified number of epochs.
    
    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch_geometric.loader.DataLoader): Training data loader.
        val_loader (torch_geometric.loader.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        args (argparse.Namespace): Command-line arguments.
        
    Returns:
        tuple: Best model state dict and validation metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch).squeeze()
            
            # Calculate loss with class weights to handle imbalance
            loss = criterion(outputs, batch.y.squeeze())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_targets.append(batch.y.detach().cpu().numpy())
        
        # Concatenate predictions and targets
        train_preds = np.concatenate([p for p in train_preds])
        train_targets = np.concatenate([t for t in train_targets])
        
        # Calculate training metrics
        train_auc = roc_auc_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Forward pass
                outputs = model(batch).squeeze()
                
                # Calculate loss
                loss = criterion(outputs, batch.y.squeeze())
                
                val_loss += loss.item()
                val_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
                val_targets.append(batch.y.detach().cpu().numpy())
        
        # Concatenate predictions and targets
        val_preds = np.concatenate([p for p in val_preds])
        val_targets = np.concatenate([t for t in val_targets])
        
        # Calculate validation metrics
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Print progress
        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss/len(train_loader):.4f} | '
              f'Train AUC: {train_auc:.4f} | '
              f'Val Loss: {val_loss/len(val_loader):.4f} | '
              f'Val AUC: {val_auc:.4f}')
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    # Calculate final metrics using the best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    val_preds = []
    val_targets = []
    val_subject_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch).squeeze()
            
            val_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            val_targets.append(batch.y.detach().cpu().numpy())
            val_subject_ids.extend([s.item() for s in batch.subject_id])
    
    # Concatenate predictions and targets
    val_preds = np.concatenate([p for p in val_preds])
    val_targets = np.concatenate([t for t in val_targets])
    
    # Aggregate window-level predictions to subject-level
    subject_preds, subject_targets = aggregate_windows_to_subjects(val_preds, val_targets, val_subject_ids)
    
    # Calculate subject-level metrics
    subject_auc = roc_auc_score(subject_targets, subject_preds)
    
    # Apply threshold of 0.5 for binary classification
    subject_pred_labels = (subject_preds > 0.5).astype(int)
    
    subject_balanced_acc = balanced_accuracy_score(subject_targets, subject_pred_labels)
    subject_precision = precision_score(subject_targets, subject_pred_labels)
    subject_recall = recall_score(subject_targets, subject_pred_labels)
    subject_f1 = f1_score(subject_targets, subject_pred_labels)
    
    # Create dictionary of metrics
    metrics = {
        'auc': subject_auc,
        'balanced_accuracy': subject_balanced_acc,
        'precision': subject_precision,
        'recall': subject_recall,
        'f1': subject_f1
    }
    
    return best_model_state, metrics


def cross_validation(dataset, args):
    """
    Perform k-fold cross-validation.
    
    Args:
        dataset (EEGGraphDataset): Dataset to use for cross-validation.
        args (argparse.Namespace): Command-line arguments.
        
    Returns:
        tuple: Dictionary of model state dicts and dictionary of metrics for each fold.
    """
    # Get unique subject IDs
    unique_subjects = dataset.get_unique_subjects()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Create k-fold cross-validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Dictionaries to store model state dicts and metrics for each fold
    model_states = {}
    fold_metrics = {}
    
    # Loop over folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_subjects)):
        print(f'\nFold {fold+1}/{args.n_folds}')
        
        # Get train and validation subject IDs
        train_subjects = unique_subjects[train_idx]
        val_subjects = unique_subjects[val_idx]
        
        # Get window indices for train and validation subjects
        train_windows = []
        for subject_id in train_subjects:
            train_windows.extend(dataset.get_subject_windows(subject_id))
            
        val_windows = []
        for subject_id in val_subjects:
            val_windows.extend(dataset.get_subject_windows(subject_id))
            
        # Create data loaders
        train_loader = DataLoader([dataset[i] for i in train_windows], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader([dataset[i] for i in val_windows], batch_size=args.batch_size, shuffle=False)
        
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
        
        # Define loss function with class weights to handle imbalance
        pos_weight = torch.tensor([sum(dataset.labels == 0) / sum(dataset.labels == 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train model
        model_state, metrics = train_model(model, train_loader, val_loader, criterion, optimizer, args)
        
        # Store model state dict and metrics
        model_states[fold] = model_state
        fold_metrics[fold] = metrics
        
        # Print fold metrics
        print(f'Fold {fold+1} metrics:')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        avg_metrics[metric] = np.mean([fold_metrics[fold][metric] for fold in range(args.n_folds)])
        std_metrics = np.std([fold_metrics[fold][metric] for fold in range(args.n_folds)])
        print(f'Average {metric}: {avg_metrics[metric]:.4f} ± {std_metrics:.4f}')
    
    return model_states, fold_metrics


def main(args):
    """
    Main function.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
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
    
    # Perform cross-validation
    print(f'Performing {args.n_folds}-fold cross-validation...')
    model_states, fold_metrics = cross_validation(dataset, args)
    
    # Save model states and metrics
    torch.save(model_states, os.path.join(args.save_dir, f'{args.model}_model_states.pt'))
    torch.save(fold_metrics, os.path.join(args.save_dir, f'{args.model}_fold_metrics.pt'))
    
    print(f'Training complete. Models saved to {args.save_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)