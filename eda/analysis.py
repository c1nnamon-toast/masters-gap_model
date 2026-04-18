"""
EDA analysis functions.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def analyze_dataframe(df):
    """Print basic statistics about the dataframe."""
    logger.info("DataFrame Info:")
    logger.info(f"Total samples: {len(df)}")
    
    # Use IrradianceToPredict if it exists, otherwise use Irradiance
    irradiance_col = 'IrradianceToPredict' if 'IrradianceToPredict' in df.columns else 'Irradiance'
    
    logger.info(f"Irradiance statistics ({irradiance_col}):")
    logger.info(f"\n{df[irradiance_col].describe()}")
    logger.info(f"Missing values: {df[irradiance_col].isna().sum()}")
    logger.info(f"Zero values: {(df[irradiance_col] == 0).sum()}")
    
    if 'Hour' in df.columns:
        logger.info("Hour distribution:")
        logger.info(f"\n{df['Hour'].value_counts().sort_index()}")

