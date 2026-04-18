"""
EDA plotting functions.
"""
import os
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def plot_irradiance_distribution(df, save_path='./analysis/irradiance_distribution.png'):
    """Plot histogram of irradiance values."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use IrradianceToPredict if it exists, otherwise use Irradiance
    irradiance_col = 'IrradianceToPredict' if 'IrradianceToPredict' in df.columns else 'Irradiance'
    
    plt.figure(figsize=(10, 4))
    df[irradiance_col].hist(bins=100)
    plt.title(f'Distribution of {irradiance_col}')
    plt.xlabel('Irradiance (W/m²)')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")

