"""
Utility functions for evaluating EEG-GCNN models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns


def aggregate_windows_to_subjects(window_preds, window_targets, subject_ids):
    """
    Aggregate window-level predictions to subject-level using maximum likelihood estimation.
    
    Args:
        window_preds (numpy.ndarray): Window-level predictions.
        window_targets (numpy.ndarray): Window-level targets.
        subject_ids (list): List of subject IDs for each window.
        
    Returns:
        tuple: Subject-level predictions and targets.
    """
    # Convert to numpy arrays
    window_preds = np.array(window_preds)
    window_targets = np.array(window_targets)
    subject_ids = np.array(subject_ids)
    
    # Get unique subject IDs
    unique_subjects = np.unique(subject_ids)
    
    # Initialize arrays for subject-level predictions and targets
    subject_preds = np.zeros(len(unique_subjects))
    subject_targets = np.zeros(len(unique_subjects))
    
    # Aggregate window-level predictions to subject-level
    for i, subject_id in enumerate(unique_subjects):
        # Get indices of windows for this subject
        subject_mask = (subject_ids == subject_id)
        
        # Get predictions and targets for this subject
        subject_window_preds = window_preds[subject_mask]
        subject_window_targets = window_targets[subject_mask]
        
        # Aggregate predictions using maximum likelihood estimation
        # This is equivalent to taking the mean of the predictions
        subject_preds[i] = np.mean(subject_window_preds)
        
        # The subject-level target is the same for all windows of the subject
        subject_targets[i] = subject_window_targets[0]
    
    return subject_preds, subject_targets


def plot_roc_curve(targets, predictions, title='ROC Curve'):
    """
    Plot ROC curve.
    
    Args:
        targets (numpy.ndarray): Target values.
        predictions (numpy.ndarray): Predicted values.
        title (str, optional): Plot title. Defaults to 'ROC Curve'.
        
    Returns:
        float: Area under the ROC curve.
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    
    return roc_auc


def plot_precision_recall_curve(targets, predictions, title='Precision-Recall Curve'):
    """
    Plot precision-recall curve.
    
    Args:
        targets (numpy.ndarray): Target values.
        predictions (numpy.ndarray): Predicted values.
        title (str, optional): Plot title. Defaults to 'Precision-Recall Curve'.
        
    Returns:
        tuple: Average precision and F1 score.
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(targets, predictions)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    # Calculate average precision
    avg_precision = np.mean(precision)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AP = {avg_precision:.3f}, Best F1 = {best_f1:.3f})')
    plt.axhline(y=sum(targets)/len(targets), color='navy', linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    
    return avg_precision, best_f1, best_threshold


def plot_confusion_matrix(targets, predictions, threshold=0.5, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        targets (numpy.ndarray): Target values.
        predictions (numpy.ndarray): Predicted values.
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        title (str, optional): Plot title. Defaults to 'Confusion Matrix'.
        
    Returns:
        numpy.ndarray: Confusion matrix.
    """
    # Convert predictions to binary using threshold
    binary_preds = (predictions > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, binary_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'Patient'],
                yticklabels=['Healthy', 'Patient'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    return cm


def compute_metrics(targets, predictions, threshold=0.5):
    """
    Compute various evaluation metrics.
    
    Args:
        targets (numpy.ndarray): Target values.
        predictions (numpy.ndarray): Predicted values.
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        
    Returns:
        dict: Dictionary of metrics.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, 
        recall_score, f1_score, roc_auc_score
    )
    
    # Convert predictions to binary using threshold
    binary_preds = (predictions > threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(targets, binary_preds),
        'balanced_accuracy': balanced_accuracy_score(targets, binary_preds),
        'precision': precision_score(targets, binary_preds),
        'recall': recall_score(targets, binary_preds),
        'f1': f1_score(targets, binary_preds),
        'auc': roc_auc_score(targets, predictions)
    }
    
    return metrics


def visualize_feature_importance(model, dataset, device='cpu'):
    """
    Visualize feature importance using a saliency map for the EEG-GCNN model.
    
    Args:
        model (torch.nn.Module): Trained model.
        dataset (EEGGraphDataset): Dataset.
        device (str, optional): Device to use. Defaults to 'cpu'.
        
    Returns:
        numpy.ndarray: Feature importance map.
    """
    import torch
    
    # Switch model to evaluation mode
    model.eval()
    
    # Initialize feature importance map
    n_channels = len(dataset.bipolar_channels)
    n_freq_bands = len(dataset.freq_bands)
    feature_importance = np.zeros((n_channels, n_freq_bands))
    
    # Loop over dataset
    for idx in range(len(dataset)):
        # Get sample
        data = dataset[idx]
        
        # Move data to device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.edge_attr = data.edge_attr.to(device)
        data.y = data.y.to(device)
        
        # Enable gradient computation
        data.x.requires_grad_(True)
        
        # Forward pass
        output = model(data)
        
        # Backward pass
        output.backward()
        
        # Get gradients
        gradients = data.x.grad.abs().detach().cpu().numpy()
        
        # Add to feature importance map
        feature_importance += gradients
    
    # Normalize feature importance map
    feature_importance /= len(dataset)
    
    # Plot feature importance map
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        feature_importance, 
        xticklabels=list(dataset.freq_bands.keys()),
        yticklabels=[f"{ch1}-{ch2}" for ch1, ch2 in dataset.bipolar_channels],
        cmap='viridis',
        annot=True,
        fmt='.2f'
    )
    plt.xlabel('Frequency Bands')
    plt.ylabel('EEG Channels')
    plt.title('Feature Importance Map')
    
    return feature_importance