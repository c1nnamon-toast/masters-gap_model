import os
import datetime
import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_timestamped_analysis_dir(base_analysis_dir):
    """Create and return a timestamped subdirectory for storing analysis outputs"""

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    timestamped_dir = os.path.join(base_analysis_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)

    return timestamped_dir

def create_training_report(config_dict, training_history, test_results, 
                           save_dir='./analysis', model_save_path=None):
    """Creates a markdown training report"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = os.path.join(save_dir, f"training_report_{timestamp}.md")
    
    with open(report_path, 'w') as f:
        f.write("# Training Report\n\n")
        f.write(f"**Date**: "\
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for key, value in config_dict.items():
            f.write(f"| {key} | {value} |\n")
        f.write("\n")
        
        f.write("## Training Results\n\n")
        f.write(f"- **Training Time**: "\
                f"{training_history['training_time']/60:.1f} minutes\n")
        f.write(f"- **Best Epoch**: "\
                f"{training_history['best_epoch']}/{config_dict.get('Epochs', '?')}\n")
        f.write(f"- **Best Validation Loss**: "\
                f"{training_history['best_val_loss']:.4f}\n")
        f.write(f"- **Best Validation RMSE**: "\
                f"{np.sqrt(training_history['best_val_loss']):.4f} W/m²\n")
        if 'peak_vram_mb' in training_history and training_history['peak_vram_mb'] > 0:
            f.write(f"- **Peak GPU Memory (VRAM)**: "\
                    f"{training_history['peak_vram_mb']:.2f} MB\n")
        f.write("\n")
        
        if test_results:
            f.write("## Test Results\n\n")
            f.write(f"- **MSE Loss**: {test_results['loss']:.4f}\n")
            f.write(f"- **RMSE**: {test_results['rmse']:.4f} W/m²\n")
            f.write(f"- **MAE**: {test_results['mae']:.4f} W/m²\n")
            f.write(f"- **R² Score**: {test_results['r2_score']:.4f}\n\n")
        
        f.write("## Generated Files\n\n")
        model_filename = os.path.basename(model_save_path) \
            if model_save_path else 'model.pth'
        f.write(f"- Model checkpoint: `{model_filename}`\n")
        f.write("- Training curves: `training_curves.png`\n")
        f.write("- Predictions plot: `predictions_vs_actual.png`\n")
        f.write("- Error distribution: `error_distribution.png`\n")
    
    logger.info(f"Saved: {report_path}")

    return report_path
