"""
Grad-CAM visualization for the rubsheet_3to1 experiment.
Loads the trained rubsheet_3to1 model (latest by timestamp), runs inference
on the test set, selects the 10 best and 10 worst predictions, and generates
Grad-CAM heatmap grids for each group.

Note: rubsheet_3to1 images are 1119x534 (3:1 aspect ratio, compressed).
"""
import sys
import os
import glob
import re

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EXPERIMENT_DIR)

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import logging
from tqdm import tqdm

import config_rubsheet_3to1 as config
from nn.model import SkyCNN
from nn.loader import get_dataloaders
from utils.cgradcam import RegressionGradCAM, overlay_heatmap
from logger_helper import logger_setup

logger = logging.getLogger(__name__)


def find_latest_model(base_path):
    """
    Find the latest model file based on timestamp suffix.
    Model files are named like: rubsheet_3to1_model_2026-03-05_01-36.pth
    """
    model_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path).replace('.pth', '')
    
    pattern = os.path.join(model_dir, f"{base_name}_*.pth")
    model_files = glob.glob(pattern)
    model_files = [f for f in model_files if "gap" in os.path.basename(f)]

    if not model_files:
        if os.path.exists(base_path):
            return base_path
        raise FileNotFoundError(f"No gap model files found matching {pattern}")
    
    def extract_timestamp(path):
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.pth$', path)
        return match.group(1) if match else ''
    
    model_files.sort(key=extract_timestamp, reverse=True)
    return model_files[0]


def select_test_cases(model, dataloader, device, n=10):
    """
    Run inference on the full test set and return the n best and n worst
    predictions ranked by absolute error.
    """
    model.eval()
    results = []

    logger.info("Running inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Scanning test set"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            for j in range(images.size(0)):
                results.append({
                    "image": images[j].unsqueeze(0).cpu(),
                    "label": labels[j].item(),
                    "prediction": outputs[j].item(),
                    "error": abs(outputs[j].item() - labels[j].item()),
                })

    results.sort(key=lambda x: x["error"])

    best = results[:n]
    worst = results[-n:]
    worst.reverse()

    logger.info(f"Best error range : {best[0]['error']:.2f} – {best[-1]['error']:.2f}")
    logger.info(f"Worst error range: {worst[0]['error']:.2f} – {worst[-1]['error']:.2f}")

    return best, worst


def plot_gradcam_grid(cases, cam_obj, device, title_prefix, save_path):
    """
    For each case, plot 3 columns:
      Col 1 – Original Input Image
      Col 2 – Pixel Contribution Map (Grad-CAM)
      Col 3 – Contribution Applied to Original
    
    Note: Images are 1119x534 (wider aspect ratio), so we use a wider figure.
    """
    n = len(cases)
    # Wider figure for 3:1 aspect ratio images
    fig, axes = plt.subplots(n, 3, figsize=(18, 3.5 * n + 1.2))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    col_headers = [
        "Original Input Image",
        "Pixel Contribution Map (Grad-CAM)",
        "Contribution Applied to Original",
    ]

    logger.info(f"Generating Grad-CAM grid: {title_prefix} ({n} images)...")

    for i, case in enumerate(cases):
        img_tensor = case["image"].to(device)

        with torch.enable_grad():
            heatmap, pred = cam_obj(img_tensor)

        orig_img, overlay = overlay_heatmap(
            img_tensor, heatmap,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
        )

        # Col 1 – Original Input Image
        axes[i, 0].imshow(orig_img)
        label_text = f"True: {case['label']:.0f} W/m²  |  Pred: {case['prediction']:.0f} W/m²"
        if i == 0:
            axes[i, 0].set_title(f"{col_headers[0]}\n{label_text}", fontsize=10)
        else:
            axes[i, 0].set_title(label_text, fontsize=10)
        axes[i, 0].axis("off")

        # Col 2 – Pixel Contribution Map
        axes[i, 1].imshow(heatmap, cmap="Reds", vmin=0, vmax=1)
        if i == 0:
            axes[i, 1].set_title(col_headers[1], fontsize=10)
        axes[i, 1].axis("off")

        # Col 3 – Contribution Applied to Original
        axes[i, 2].imshow(overlay)
        if i == 0:
            axes[i, 2].set_title(f"{col_headers[2]}\nAbs Error: {case['error']:.1f}", fontsize=10)
        else:
            axes[i, 2].set_title(f"Abs Error: {case['error']:.1f}", fontsize=10)
        axes[i, 2].axis("off")

    # Colorbar legend
    cbar_ax = fig.add_axes([0.15, 0.025, 0.7, 0.012])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["No contribution", "Medium", "High contribution"])
    cbar.ax.tick_params(labelsize=9)

    caption = (
        "Col 1: The original sky image fed to the model.  "
        "Col 2: Pixel contribution map — the more intense the red, "
        "the more that region contributed to the predicted irradiance.  "
        "Col 3: Contribution overlaid on the original."
    )
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8, style="italic",
             wrap=True, color="0.3")

    plt.suptitle(f"{title_prefix}  (Regression Grad-CAM — Rubsheet 3-to-1)", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {save_path}")


def main():
    logger_setup(experiment_logfile=config.LOG_FILE)
    device = config.DEVICE

    logger.info("Loading test data...")
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

    grad_cam = RegressionGradCAM(model, model.conv_block4)

    best_cases, worst_cases = select_test_cases(model, test_loader, device, n=10)

    output_dir = os.path.join(config.ANALYSIS_DIR, "gradcam")
    os.makedirs(output_dir, exist_ok=True)

    plot_gradcam_grid(
        best_cases, grad_cam, device,
        "Top 10 Best Predictions",
        os.path.join(output_dir, "best_10_gradcam.png"),
    )
    plot_gradcam_grid(
        worst_cases, grad_cam, device,
        "Top 10 Worst Predictions",
        os.path.join(output_dir, "worst_10_gradcam.png"),
    )

    logger.info("Grad-CAM visualisation complete.")


if __name__ == "__main__":
    main()
