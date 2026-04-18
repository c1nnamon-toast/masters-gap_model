"""
Evaluation and testing.
"""
import numpy as np
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def test_model(model, test_loader, criterion, device, model_path=None):
    """
    Test model on test set.
    
    Returns:
        Dictionary with test metrics and predictions
    """
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Loaded model from {model_path}")
    
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    predictions = np.array(all_predictions).flatten()
    labels = np.array(all_labels).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(test_loss)
    mae = np.mean(np.abs(predictions - labels))
    
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    logger.info("Test Results:")
    logger.info(f"  MSE Loss: {test_loss:.4f}")
    logger.info(f"  RMSE: {rmse:.4f} W/m²")
    logger.info(f"  MAE: {mae:.4f} W/m²")
    logger.info(f"  R² Score: {r2_score:.4f}")
    
    return {
        'loss': test_loss,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2_score,
        'predictions': predictions,
        'labels': labels
    }
