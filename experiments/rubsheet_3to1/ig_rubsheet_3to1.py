"""Integrated Gradients attribution analysis for the rubsheet_3to1 experiment.

Uses Captum's IntegratedGradients to explain SkyCNN's irradiance predictions
on the rubsheet_3to1 test set.

For each of six representative groups (low / medium / high irradiance, best /
worst prediction errors, random) a grid of 7 images is saved showing:
  - the original sky strip image
  - the IG negative attribution heatmap (GnBu)
  - the IG positive attribution heatmap (YlOrRd)
  - the mixed signed attribution heatmap (RdBu_r)
  - the mixed attribution blended onto the original

A summary CSV of all test-set predictions is also saved, along with:
  - attribution_values.csv  : per-image attribution statistics
  - attribution_extremes.csv: global top-40 / bottom-40 pixel values

Note: rubsheet_3to1 images are 1602x534 px (wide panoramic strips).

Run
---
    python experiments/rubsheet_3to1/ig_rubsheet_3to1.py
"""

import sys
import os
import glob
import re

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EXPERIMENT_DIR)

import torch
import numpy as np
import logging
from captum.attr import IntegratedGradients

import config_rubsheet_3to1 as config
from nn.model import SkyCNN
from nn.loader import get_dataloaders
from utils.attribution_map import (
    collect_all_results,
    select_groups,
    collect_group_attributions,
    plot_ig_signed_grid,
    save_summary_csv,
    save_attribution_stats,
    log_top_attribution_values,
)
from logger_helper import logger_setup

logger = logging.getLogger(__name__)


N_STEPS = 20
N_PER_GROUP     = 7
TOP_N           = 40
EXPERIMENT_NAME = "Rubsheet 3-to-1"
METHOD_NAME     = "Integrated Gradients"


def find_latest_model(base_path):
    """Return the most recent model checkpoint (by timestamp suffix)."""
    model_dir   = os.path.dirname(base_path)
    base_name   = os.path.basename(base_path).replace(".pth", "")
    pattern     = os.path.join(model_dir, f"{base_name}_*.pth")
    model_files = glob.glob(pattern)
    model_files = [f for f in model_files if "gap" in os.path.basename(f)]
    if not model_files:
        if os.path.exists(base_path):
            return base_path
        raise FileNotFoundError(f"No gap model files found matching {pattern}")
    def extract_timestamp(path):
        match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.pth$", path)
        return match.group(1) if match else ""
    model_files.sort(key=extract_timestamp, reverse=True)
    return model_files[0]


def make_ig_attr_fn(ig, device, n_steps):
    """Return a callable that computes a raw signed IG attribution map."""
    def attr_fn(img_tensor):
        inp      = img_tensor.to(device)
        baseline = torch.zeros_like(inp)
        attributions = ig.attribute(inp, baseline, n_steps=n_steps, target=0)
        attr_np  = attributions.detach().cpu().numpy()[0]
        attr_map = np.sum(attr_np, axis=0)
        del attributions, inp, baseline
        return attr_map
    return attr_fn


def main():
    logger_setup(experiment_logfile=config.LOG_FILE)
    device = config.DEVICE

    logger.info("=== Integrated Gradients — Rubsheet 3-to-1 ===")
    logger.info(f"IG params: n_steps={N_STEPS}")

    logger.info("Loading test data …")
    _, _, test_loader, _ = get_dataloaders(
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
        normalize_std=config.NORMALIZE_STD,
    )

    model_path = find_latest_model(config.MODEL_SAVE_PATH)
    logger.info(f"Loading model from {model_path}")
    model = SkyCNN().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    results = collect_all_results(model, test_loader, device)

    output_dir = os.path.join(config.ANALYSIS_DIR, "ig")
    os.makedirs(output_dir, exist_ok=True)

    save_summary_csv(results, os.path.join(output_dir, "summary.csv"))
    groups = select_groups(results, n=N_PER_GROUP)

    ig      = IntegratedGradients(model)
    attr_fn = make_ig_attr_fn(ig, device, n_steps=N_STEPS)

    logger.info("Pre-computing IG attribution maps …")
    group_attr_data = collect_group_attributions(groups, attr_fn)

    save_attribution_stats(group_attr_data, output_dir, top_n=TOP_N)
    log_top_attribution_values(group_attr_data, top_n=TOP_N)

    group_titles = {
        "low":    "Low Irradiance",
        "medium": "Medium Irradiance",
        "high":   "High Irradiance",
        "best":   "Best Predictions (lowest error)",
        "worst":  "Worst Predictions (highest error)",
        "random": "Random Sample",
    }

    for group_name, cases_maps in group_attr_data.items():
        cases     = [c for c, _ in cases_maps]
        attr_maps = [m for _, m in cases_maps]
        save_path = os.path.join(output_dir, f"{group_name}_ig.png")
        plot_ig_signed_grid(
            cases=cases,
            attr_fn=None,
            precomputed_maps=attr_maps,
            title_prefix=group_titles[group_name],
            method_name=METHOD_NAME,
            experiment_name=EXPERIMENT_NAME,
            save_path=save_path,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
        )

    logger.info("Integrated Gradients analysis complete.")


if __name__ == "__main__":
    main()
