import os
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_training_curves(train_losses, val_losses, save_path='./analysis/training_curves.png'):
    """Plot training and validation loss curves"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_predictions(predictions, labels, save_path='./analysis/predictions_vs_actual.png'):
    """Plot predicted vs actual values"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, predictions, alpha=0.5, s=20)
    
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title('Predicted vs Actual Irradiance')
    plt.xlabel('Actual Irradiance (W/m²)')
    plt.ylabel('Predicted Irradiance (W/m²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_error_distribution(predictions, labels, save_path='./analysis/error_distribution.png'):
    """Plot error distribution"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    errors = predictions - labels
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Error Distribution')
    axes[0].set_xlabel('Prediction Error (W/m²)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    axes[1].scatter(labels, errors, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Residual Plot')
    axes[1].set_xlabel('Actual Irradiance (W/m²)')
    axes[1].set_ylabel('Prediction Error (W/m²)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")
