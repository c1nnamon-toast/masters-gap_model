# Training Report

**Date**: 2026-04-18 00:51:10

## Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0001 |
| Batch Size | 16 |
| Epochs | 27 |
| Dataset | /home/xtsimbota_135764/masters/dataset/dataset_rubsheet_square |
| Device | cuda |

## Training Results

- **Training Time**: 71.6 minutes
- **Best Epoch**: 27/27
- **Best Validation Loss**: 262.3138
- **Best Validation RMSE**: 16.1961 W/m²
- **Peak GPU Memory (VRAM)**: 5119.57 MB

## Test Results

- **MSE Loss**: 227.9706
- **RMSE**: 15.0987 W/m²
- **MAE**: 9.0492 W/m²
- **R² Score**: 0.9969

## Generated Files

- Model checkpoint: `rubsheet_square_model_gap_2026-04-17_23-39.pth`
- Training curves: `training_curves.png`
- Predictions plot: `predictions_vs_actual.png`
- Error distribution: `error_distribution.png`
