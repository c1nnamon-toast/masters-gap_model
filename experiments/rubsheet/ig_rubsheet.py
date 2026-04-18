"""Integrated Gradients attribution analysis for the rubsheet experiment.

Uses Captum's IntegratedGradients to explain SkyCNN's irradiance predictions
on the rubsheet test set (standard polar-to-Cartesian panoramic unwrap).

For each of six representative groups (low / medium / high irradiance, best /
worst prediction errors, random) a grid of 7 images is saved showing:
  - the original sky strip image
  - the IG attribution map (L2-aggregated over channels, normalized to [0, 1])
  - the attribution blended onto the original (red-tint overlay)

Two attribution CSVs are written to the output directory:
  attribution_values.csv   — per-image distribution stats for all 42 images
  attribution_extremes.csv — global top-40 and bottom-40 pixel values with
                             group, image index, and (row, col) coordinates

A summary CSV of all test-set predictions is also saved.

Baseline choice
---------------
We use an all-zeros tensor as the IG baseline.  For images normalised to
[-1, 1] with mean=0.5 / std=0.5, the zero tensor corresponds to a
mid-grey image — a natural "uninformative" reference point.

Run
---
    python experiments/rubsheet/ig_rubsheet.py
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

import config_rubsheet as config
from nn.model import SkyCNN
from nn.loader import get_dataloaders
from utils.attribution_map import (
    collect_all_results,
    select_groups,
    collect_group_attributions,
    save_attribution_stats,
    log_top_attribution_values,
    plot_ig_signed_grid,
    save_summary_csv,
)
from logger_helper import logger_setup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Integrated Gradients parameters
# ---------------------------------------------------------------------------

N_STEPS = 50

N_PER_GROUP     = 7
TOP_N           = 40
EXPERIMENT_NAME = "Rubsheet"
METHOD_NAME     = "Integrated Gradients"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_model(base_path):
    """Return the most recent model checkpoint (by timestamp suffix)."""
    model_dir   = os.path.dirname(base_path)
    base_name   = os.path.basename(base_path).replace(".pth", "")
    pattern     = os.path.join(model_dir, f"{base_name}_*.pth")
    model_files = glob.glob(pattern)

    if not model_files:
        if os.path.exists(base_path):
            return base_path
        raise FileNotFoundError(f"No model files found matching {pattern}")

    def extract_timestamp(path):
        match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.pth$", path)
        return match.group(1) if match else ""

    model_files.sort(key=extract_timestamp, reverse=True)
    return model_files[0]


def make_ig_attr_fn(ig, device, n_steps):
    """Return a callable that computes a signed IG attribution map.

    The map is NOT normalised here so that raw values can be inspected for
    outliers via :func:`save_attribution_stats` and
    :func:`log_top_attribution_values` before display normalisation.
    """
    def attr_fn(img_tensor):
        inp      = img_tensor.to(device)
        baseline = torch.zeros_like(inp)

        attributions = ig.attribute(inp, baseline, n_steps=n_steps, target=0)

        attr_np  = attributions.detach().cpu().numpy()[0]   # (C, H, W)
        attr_map = np.sum(attr_np, axis=0)                   # signed (H, W)

        del attributions, inp, baseline

        return attr_map  # raw, unnormalized, signed

    return attr_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger_setup(experiment_logfile=config.LOG_FILE)
    device = config.DEVICE

    logger.info("=== Integrated Gradients — Rubsheet ===")
    logger.info(f"IG params: n_steps={N_STEPS}, top_n={TOP_N}")

    # ── Load test data ──────────────────────────────────────────────────────
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

    # ── Load model ──────────────────────────────────────────────────────────
    model_path = find_latest_model(config.MODEL_SAVE_PATH)
    logger.info(f"Loading model from {model_path}")
    model = SkyCNN().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # ── Collect all test-set predictions ────────────────────────────────────
    results = collect_all_results(model, test_loader, device)

    # ── Output directory ────────────────────────────────────────────────────
    output_dir = os.path.join(config.ANALYSIS_DIR, "ig")
    os.makedirs(output_dir, exist_ok=True)

    # ── Save summary CSV ────────────────────────────────────────────────────
    save_summary_csv(results, os.path.join(output_dir, "summary.csv"))

    # ── Select representative groups ────────────────────────────────────────
    groups = select_groups(results, n=N_PER_GROUP)

    # ── Set up Integrated Gradients ─────────────────────────────────────────
    ig      = IntegratedGradients(model)
    attr_fn = make_ig_attr_fn(ig, device, n_steps=N_STEPS)

    # ── Pre-compute all attribution maps (single pass) ───────────────────────
    group_attr_data = collect_group_attributions(groups, attr_fn)

    # ── Attribution statistics: top/bottom values + per-image stats ──────────
    save_attribution_stats(group_attr_data, output_dir, top_n=TOP_N)
    log_top_attribution_values(group_attr_data, top_n=TOP_N)

    # ── Generate one grid per group (reuse precomputed maps) ─────────────────
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
