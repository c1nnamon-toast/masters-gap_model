import sys
import os
import logging

# Ensure access to root directory
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EXPERIMENT_DIR)

import config_fisheye as config
from nn.loader import get_dataloaders
from eda.analysis import analyze_dataframe
from eda.plots import plot_irradiance_distribution
from nn.model import SkyCNN
from nn.trainer import Trainer
from evaluation.evaluator import test_model
from utils.plots import plot_training_curves, plot_predictions, plot_error_distribution
from utils.reporter import create_training_report, get_timestamped_analysis_dir
from logger_helper import logger_setup

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def main():
    logger_setup(experiment_logfile=config.LOG_FILE)

    if not torch.cuda.is_available():
        logger.error("no GPU")
        return

    # Create necessary directories
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    analysis_dir = get_timestamped_analysis_dir(config.ANALYSIS_DIR)
    run_timestamp = os.path.basename(analysis_dir)
    model_stem, model_ext = os.path.splitext(config.MODEL_SAVE_PATH)
    model_save_path = f"{model_stem}_{run_timestamp}{model_ext}"

    logger.info("-" * 80)
    logger.info("Fisheye Experiment [Custom CNN]")
    logger.info("-" * 80)

    logger.info("Loading dataset")
    train_loader, val_loader, test_loader, full_df = get_dataloaders(
        train_csv=config.TRAIN_CSV,
        val_csv=config.VAL_CSV,
        test_csv=config.TEST_CSV,
        train_image_dir=config.TRAIN_IMAGE_DIR,
        val_image_dir=config.VAL_IMAGE_DIR,
        test_image_dir=config.TEST_IMAGE_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        normalize_mean=config.NORMALIZE_MEAN,
        normalize_std=config.NORMALIZE_STD
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # EDA
    logger.info("-" * 80)
    logger.info("Exploratory Data Analysis")
    logger.info("-" * 80)
    analyze_dataframe(full_df)
    plot_irradiance_distribution(
        full_df, save_path=f'{analysis_dir}/irradiance_distribution.png')

    # Create model
    logger.info("-" * 80)
    logger.info("Creating model")
    logger.info("-" * 80)
    model = SkyCNN().to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Training
    logger.info("-" * 80)
    logger.info("Training")
    logger.info("-" * 80)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS,
        save_path=model_save_path
    )

    training_history = trainer.train()

    # Testing
    logger.info("-" * 80)
    logger.info("Testing")
    logger.info("-" * 80)

    test_results = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=config.DEVICE,
        model_path=model_save_path
    )

    # Plots
    logger.info("-" * 80)
    logger.info("Generating plots")
    logger.info("-" * 80)

    plot_training_curves(
        training_history['train_losses'],
        training_history['val_losses'],
        save_path=f'{analysis_dir}/training_curves.png'
    )

    plot_predictions(
        test_results['predictions'],
        test_results['labels'],
        save_path=f'{analysis_dir}/predictions_vs_actual.png'
    )

    plot_error_distribution(
        test_results['predictions'],
        test_results['labels'],
        save_path=f'{analysis_dir}/error_distribution.png'
    )

    # Report
    logger.info("-" * 80)
    logger.info("Creating report")
    logger.info("-" * 80)

    config_dict = {
        'Learning Rate': config.LEARNING_RATE,
        'Batch Size': config.BATCH_SIZE,
        'Epochs': config.NUM_EPOCHS,
        'Dataset': config.DATASET_ROOT,
        'Device': str(config.DEVICE)
    }

    create_training_report(config_dict, training_history, test_results, 
                           save_dir=analysis_dir, model_save_path=model_save_path)

    logger.info("-" * 80)
    logger.info("Training Complete")
    logger.info("-" * 80)
    logger.info(f"Model saved: {model_save_path}")

if __name__ == "__main__":
    main()
