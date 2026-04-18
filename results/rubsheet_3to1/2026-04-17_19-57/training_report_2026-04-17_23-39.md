# Training Report

**Date**: 2026-04-17 23:39:19

## Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0001 |
| Batch Size | 16 |
| Epochs | 27 |
| Dataset | /home/xtsimbota_135764/masters/dataset/dataset_rubsheet_3to1 |
| Device | cuda |

## Training Results

- **Training Time**: 221.1 minutes
- **Best Epoch**: 23/27
- **Best Validation Loss**: 379.2517
- **Best Validation RMSE**: 19.4744 W/m²
- **Peak GPU Memory (VRAM)**: 15308.87 MB

## Test Results

- **MSE Loss**: 347.5263
- **RMSE**: 18.6421 W/m²
- **MAE**: 11.3993 W/m²
- **R² Score**: 0.9953

## Generated Files

- Model checkpoint: `rubsheet_3to1_model_gap_2026-04-17_19-57.pth`
- Training curves: `training_curves.png`
- Predictions plot: `predictions_vs_actual.png`
- Error distribution: `error_distribution.png`
