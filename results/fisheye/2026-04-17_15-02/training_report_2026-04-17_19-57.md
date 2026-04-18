# Training Report

**Date**: 2026-04-17 19:57:36

## Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0001 |
| Batch Size | 16 |
| Epochs | 27 |
| Dataset | /home/xtsimbota_135764/masters/dataset/original |
| Device | cuda |

## Training Results

- **Training Time**: 294.2 minutes
- **Best Epoch**: 21/27
- **Best Validation Loss**: 346.4688
- **Best Validation RMSE**: 18.6137 W/m²
- **Peak GPU Memory (VRAM)**: 20420.05 MB

## Test Results

- **MSE Loss**: 325.8132
- **RMSE**: 18.0503 W/m²
- **MAE**: 12.4328 W/m²
- **R² Score**: 0.9956

## Generated Files

- Model checkpoint: `fisheye_model_gap_2026-04-17_15-02.pth`
- Training curves: `training_curves.png`
- Predictions plot: `predictions_vs_actual.png`
- Error distribution: `error_distribution.png`
