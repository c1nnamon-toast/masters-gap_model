"""Occlusion attribution analysis for the rubsheet_square experiment.

Uses Captum's Occlusion (sliding-window perturbation) to explain SkyCNN's
irradiance predictions on the rubsheet_square test set.

For each of six representative groups (low / medium / high irradiance, best /
worst prediction errors, random) a grid of 7 images is saved showing:
  - the original sky image
  - the Occlusion attribution map (L2-aggregated over channels, [0,1])
  - the attribution blended onto the original (red-tint overlay)

A summary CSV of all test-set predictions is also saved, along with:
  - attribution_values.csv  : per-image attribution statistics
  - attribution_extremes.csv: global top-40 / bottom-40 pixel values

Run
---
    python experiments/rubsheet_square/occlusion_rubsheet_square.py
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
from captum.attr import Occlusion

import config_rubsheet_square as config
from nn.model import SkyCNN
from nn.loader import get_dataloaders
from utils.attribution_map import (
    collect_all_results,
    select_groups,
    collect_group_attributions,
    plot_attribution_grid,
    save_summary_csv,
    save_attribution_stats,
    log_top_attribution_values,
)
from logger_helper import logger_setup

logger = logging.getLogger(__name__)


PATCH_SIZE     = 50
STRIDE         = 25
BASELINE_VALUE = 0.0

N_PER_GROUP     = 7
TOP_N           = 40
EXPERIMENT_NAME = "Rubsheet Square"
METHOD_NAME     = "Occlusion"


def find_latest_model(base_path):
    """Return the most recent gap model checkpoint (by timestamp suffix)."""
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


def make_occlusion_attr_fn(occ, device, patch_size, stride, baseline_value):
    """Return a callable that computes a raw (unnormalized) Occlusion attribution map."""
    if isinstance(patch_size, int):
        sliding_window = (3, patch_size, patch_size)
    else:
        sliding_window = (3, patch_size[0], patch_size[1])

    if isinstance(stride, int):
        stride_tuple = (1, stride, stride)
    else:
        stride_tuple = (1, stride[0], stride[1])

    def attr_fn(img_tensor):
        inp      = img_tensor.to(device)
        baseline = torch.full_like(inp, fill_value=baseline_value)
        attributions = occ.attribute(
            inp,
            sliding_window_shapes=sliding_window,
            strides=stride_tuple,
            baselines=baseline,
            target=0,
        )
        attr_np  = attributions.detach().cpu().numpy()[0]
        attr_map = np.linalg.norm(attr_np, axis=0)
        del attributions, inp, baseline
        return attr_map

    return attr_fn


def main():
    logger_setup(experiment_logfile=config.LOG_FILE)
    device = config.DEVICE

    logger.info("=== Occlusion Attribution — Rubsheet Square ===")
    logger.info(
        f"Occlusion params: patch_size={PATCH_SIZE}, stride={STRIDE}, "
        f"baseline_value={BASELINE_VALUE}"
    )

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

    output_dir = os.path.join(config.ANALYSIS_DIR, "occlusion")
    os.makedirs(output_dir, exist_ok=True)

    save_summary_csv(results, os.path.join(output_dir, "summary.csv"))
    groups = select_groups(results, n=N_PER_GROUP)

    occ     = Occlusion(model)
    attr_fn = make_occlusion_attr_fn(
        occ, device,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        baseline_value=BASELINE_VALUE,
    )

    logger.info("Pre-computing Occlusion attribution maps …")
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
        norm_maps = [m / m.max() if m.max() > 0 else m for m in attr_maps]
        save_path = os.path.join(output_dir, f"{group_name}_occlusion.png")
        plot_attribution_grid(
            cases=cases,
            attr_fn=None,
            precomputed_maps=norm_maps,
            title_prefix=group_titles[group_name],
            method_name=METHOD_NAME,
            experiment_name=EXPERIMENT_NAME,
            save_path=save_path,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
        )

    logger.info("Occlusion analysis complete.")


if __name__ == "__main__":
    main()
