"""Occlusion attribution analysis for the fisheye experiment.

Uses Captum's Occlusion (sliding-window perturbation) to explain SkyCNN's
irradiance predictions on the fisheye test set.

Occlusion works by systematically replacing patches of the input with a
baseline value and measuring how much the model's prediction changes.
Regions whose occlusion causes a large drop in predicted irradiance are
deemed highly important.

For each of six representative groups (low / medium / high irradiance, best /
worst prediction errors, random) a grid of 7 images is saved showing:
  - the original sky image
  - the Occlusion attribution map (L2-aggregated over channels, [0,1])
  - the attribution blended onto the original (red-tint overlay)

A summary CSV of all test-set predictions is also saved.

Runtime note
------------
Occlusion is slower than Integrated Gradients because it requires one
forward pass per sliding-window position.  For 1068×1068 fisheye images the
default parameters below give roughly 32×32 ≈ 1 024 passes per image, which
is manageable on a GPU but will be slow on CPU.  Increase STRIDE or
PATCH_SIZE to trade resolution for speed.

Tunable parameters
------------------
PATCH_SIZE      : int — side length of the occluded square (pixels).
                  Larger → faster, coarser attribution map.
STRIDE          : int — step between patch centres (pixels).
                  Larger → faster, coarser attribution map.
BASELINE_VALUE  : float — fill value in normalised pixel space.
                  0.0 = mid-grey for images normalised to [-1, 1].

Run
---
    python experiments/fisheye/occlusion_fisheye.py
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
from tqdm import tqdm
from captum.attr import Occlusion

import config_fisheye as config
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


# ---------------------------------------------------------------------------
# Occlusion parameters  ← adjust these if runtime is too slow / coarse
# ---------------------------------------------------------------------------

# Fisheye images are 1068×1068 px.
# patch=67 (≈ 1/16 of image width) and stride=34 (≈ half-patch) yield
# roughly 32×32 = 1 024 forward passes per image.
PATCH_SIZE     = 50     # occluded square patch side length (pixels)
STRIDE         = 20     # sliding-window step (pixels)
BASELINE_VALUE = 0.0    # fill value in normalised space (0 = mid-grey)

N_PER_GROUP     = 7
TOP_N           = 40
EXPERIMENT_NAME = "Fisheye"
METHOD_NAME     = "Occlusion"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_model(base_path):
    """Return the most recent model checkpoint (by timestamp suffix).

    Model files are named like: fisheye_model_2026-03-03_19-56.pth
    Falls back to ``base_path`` if no timestamped files exist.
    """
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
    """Return a callable that computes a raw (unnormalized) Occlusion attribution map.

    Returns
    -------
    callable : (img_tensor: Tensor[1,C,H,W]) -> np.ndarray[H,W]  raw values
    """
    # Expand scalar args to 3-channel tuples as Captum expects (C, H, W)
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

        # target=0 selects the single scalar output (shape: batch×1)
        attributions = occ.attribute(
            inp,
            sliding_window_shapes=sliding_window,
            strides=stride_tuple,
            baselines=baseline,
            target=0,
        )

        # Aggregate channels: L2 norm across RGB -> (H, W), raw (unnormalized)
        attr_np  = attributions.detach().cpu().numpy()[0]   # (C, H, W)
        attr_map = np.linalg.norm(attr_np, axis=0)          # (H, W)
        del attributions, inp, baseline
        return attr_map  # raw — caller normalizes for display

    return attr_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger_setup(experiment_logfile=config.LOG_FILE)
    device = config.DEVICE

    logger.info("=== Occlusion Attribution — Fisheye ===")
    logger.info(
        f"Occlusion params: patch_size={PATCH_SIZE}, stride={STRIDE}, "
        f"baseline_value={BASELINE_VALUE}"
    )

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
    output_dir = os.path.join(config.ANALYSIS_DIR, "occlusion")
    os.makedirs(output_dir, exist_ok=True)

    # ── Save summary CSV ────────────────────────────────────────────────────
    save_summary_csv(results, os.path.join(output_dir, "summary.csv"))

    # ── Select representative groups ────────────────────────────────────────
    groups = select_groups(results, n=N_PER_GROUP)

    # ── Set up Occlusion ────────────────────────────────────────────────────
    occ     = Occlusion(model)
    attr_fn = make_occlusion_attr_fn(
        occ, device,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        baseline_value=BASELINE_VALUE,
    )

    # ── Pre-compute all attribution maps (single pass per image) ─────────────
    logger.info("Pre-computing Occlusion attribution maps …")
    group_attr_data = collect_group_attributions(groups, attr_fn)

    # ── Attribution statistics: CSV + console ────────────────────────────────
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
