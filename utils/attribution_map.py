"""Shared helpers for attribution-map analysis (Integrated Gradients, Occlusion, …).

Provides
--------
collect_all_results       : run inference on a dataloader and record results
select_groups             : pick representative examples for visualization
collect_group_attributions: pre-compute attribution maps for all groups
save_attribution_stats    : save per-image stats and global extremes to CSV
log_top_attribution_values: log top/bottom N attribution values
denormalize_image         : undo torchvision Normalize for display
plot_attribution_grid     : save a 3-column figure for occlusion groups
plot_ig_signed_grid       : save a 5-column signed-attribution figure for IG groups
save_summary_csv          : persist per-sample results as a CSV
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def collect_all_results(model, dataloader, device):
    """Run inference on every sample in *dataloader* and collect metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode (or will be set to eval).
    dataloader : torch.utils.data.DataLoader
        Yields (images, labels) batches.
    device : torch.device

    Returns
    -------
    list[dict]
        Each dict has keys:
          - ``image``      : CPU tensor, shape (1, C, H, W)
          - ``label``      : float — ground-truth irradiance (W/m²)
          - ``prediction`` : float — model output (W/m²)
          - ``error``      : float — absolute prediction error (W/m²)
    """
    model.eval()
    results = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting results"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            for j in range(images.size(0)):
                results.append({
                    "image":      images[j].unsqueeze(0).cpu(),
                    "label":      labels[j].item(),
                    "prediction": outputs[j].item(),
                    "error":      abs(outputs[j].item() - labels[j].item()),
                })

    logger.info(f"Collected {len(results)} test samples.")
    return results


def select_groups(results, n=7, seed=42):
    """Select *n* representative examples for each visualization group.

    Groups
    ------
    low    : samples with the lowest irradiance labels
    medium : samples near the median irradiance
    high   : samples with the highest irradiance labels
    best   : smallest absolute prediction error
    worst  : largest absolute prediction error
    random : reproducible random sample (seeded via *seed*)

    Parameters
    ----------
    results : list[dict]
        Output of :func:`collect_all_results`.
    n : int
        Number of samples per group.
    seed : int
        Random seed for the "random" group.

    Returns
    -------
    dict[str, list[dict]]
        Keys: ``"low"``, ``"medium"``, ``"high"``, ``"best"``, ``"worst"``,
        ``"random"``.
    """
    if len(results) < n:
        raise ValueError(f"Need at least {n} samples, got {len(results)}")

    # --- By irradiance label ---
    by_label = sorted(results, key=lambda x: x["label"])
    low  = by_label[:n]
    high = by_label[-n:]

    # Medium: n samples centred around the median index
    mid_idx = len(by_label) // 2
    half    = n // 2
    start   = max(0, mid_idx - half)
    end     = min(len(by_label), start + n)
    medium  = by_label[start:end]

    # --- By error ---
    by_error = sorted(results, key=lambda x: x["error"])
    best  = by_error[:n]
    worst = list(reversed(by_error[-n:]))  # worst first

    # --- Random ---
    rng          = np.random.default_rng(seed)
    indices      = rng.choice(len(results), size=n, replace=False)
    random_group = [results[int(i)] for i in indices]

    groups = {
        "low":    low,
        "medium": medium,
        "high":   high,
        "best":   best,
        "worst":  worst,
        "random": random_group,
    }

    for name, grp in groups.items():
        labels = [c["label"] for c in grp]
        errors = [c["error"] for c in grp]
        logger.info(
            f"Group '{name}': label [{min(labels):.0f}–{max(labels):.0f}] W/m², "
            f"error [{min(errors):.1f}–{max(errors):.1f}]"
        )

    return groups


def collect_group_attributions(groups, attr_fn):
    """Pre-compute attribution maps for all groups.

    Calls ``attr_fn`` exactly once per image and caches the result so that
    the maps can be reused for statistics, logging, and visualization without
    redundant forward passes.

    Parameters
    ----------
    groups  : dict[str, list[dict]]
        Output of :func:`select_groups`.
    attr_fn : callable
        ``attr_fn(img_tensor) -> np.ndarray (H, W)`` — raw (unnormalized)
        attribution map (signed for IG, magnitude for Occlusion).

    Returns
    -------
    dict[str, list[tuple[dict, np.ndarray]]]
        Same keys as *groups*; each value is a list of ``(case, attr_map)``
        pairs where *attr_map* is the raw attribution map.
    """
    group_attr = {}
    for group_name, cases in groups.items():
        maps = []
        for case in tqdm(cases, desc=f"  {group_name}", leave=False):
            try:
                maps.append(attr_fn(case["image"]))
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    logger.warning(
                        f"CUDA OOM on group '{group_name}' — skipping this image."
                    )
                    maps.append(None)
                else:
                    raise
        group_attr[group_name] = list(zip(cases, maps))
    return group_attr


def save_attribution_stats(group_attr_data, save_dir, top_n=40):
    """Save per-image attribution statistics and global extremes to CSV.

    Writes two files into *save_dir*:

    ``attribution_values.csv``
        One row per image with summary statistics of the raw attribution map:
        max, min, mean, std, 90th/95th/99th percentiles.

    ``attribution_extremes.csv``
        The global top-*top_n* and bottom-*top_n* raw attribution pixel values
        across all images, with their (group, image index, row, col)
        coordinates.

    Parameters
    ----------
    group_attr_data : dict[str, list[tuple[dict, np.ndarray]]]
        Output of :func:`collect_group_attributions`.
    save_dir : str
        Directory where both CSV files are written.
    top_n : int
        Number of extreme pixels to record (default 40).
    """
    os.makedirs(save_dir, exist_ok=True)

    per_image_rows = []
    top_pool = []   # global top-N candidates
    bot_pool = []   # global bottom-N candidates

    for group_name, cases_maps in group_attr_data.items():
        for img_idx, (case, attr_map) in enumerate(cases_maps):
            if attr_map is None:
                continue
            flat = attr_map.flatten()

            per_image_rows.append({
                "group":        group_name,
                "img_idx":      img_idx,
                "label_wm2":    round(case["label"], 2),
                "pred_wm2":     round(case["prediction"], 2),
                "abs_err_wm2":  round(case["error"], 2),
                "max_attr":     round(float(flat.max()), 6),
                "min_attr":     round(float(flat.min()), 6),
                "mean_attr":    round(float(flat.mean()), 6),
                "std_attr":     round(float(flat.std()), 6),
                "p90_attr":     round(float(np.percentile(flat, 90)), 6),
                "p95_attr":     round(float(np.percentile(flat, 95)), 6),
                "p99_attr":     round(float(np.percentile(flat, 99)), 6),
            })

            # Collect top-N and bottom-N pixel values per image
            sorted_idx = np.argsort(flat)
            # Bottom N
            for j in range(min(top_n, len(sorted_idx))):
                ridx = sorted_idx[j]
                row, col = np.unravel_index(ridx, attr_map.shape)
                bot_pool.append({
                    "value":     float(flat[ridx]),
                    "group":     group_name,
                    "img_idx":   img_idx,
                    "row":       int(row),
                    "col":       int(col),
                    "rank_type": "bottom",
                })
            # Top N (descending)
            for j in range(1, min(top_n, len(sorted_idx)) + 1):
                ridx = sorted_idx[-j]
                row, col = np.unravel_index(ridx, attr_map.shape)
                top_pool.append({
                    "value":     float(flat[ridx]),
                    "group":     group_name,
                    "img_idx":   img_idx,
                    "row":       int(row),
                    "col":       int(col),
                    "rank_type": "top",
                })

    # --- attribution_values.csv ---
    values_path = os.path.join(save_dir, "attribution_values.csv")
    pd.DataFrame(per_image_rows).to_csv(values_path, index=False)
    logger.info(f"Attribution values CSV → {values_path} ({len(per_image_rows)} rows)")

    # --- attribution_extremes.csv ---
    # Keep global top-N and bottom-N
    top_pool.sort(key=lambda x: x["value"], reverse=True)
    bot_pool.sort(key=lambda x: x["value"])
    extremes = top_pool[:top_n] + bot_pool[:top_n]
    extremes_path = os.path.join(save_dir, "attribution_extremes.csv")
    pd.DataFrame(extremes).to_csv(extremes_path, index=False)
    logger.info(
        f"Attribution extremes CSV → {extremes_path} "
        f"(top {top_n} + bottom {top_n} pixels)"
    )


def log_top_attribution_values(group_attr_data, top_n=40):
    """Log the global top-N and bottom-N raw attribution values.

    Collects the maximum and minimum pixel values from every image across all
    groups, sorts them, and logs the *top_n* highest and *top_n* lowest with
    their group and image context.

    Parameters
    ----------
    group_attr_data : dict[str, list[tuple[dict, np.ndarray]]]
        Output of :func:`collect_group_attributions`.
    top_n : int
        How many extreme values to log (default 40).
    """
    all_top = []
    all_bot = []

    for group_name, cases_maps in group_attr_data.items():
        for img_idx, (case, attr_map) in enumerate(cases_maps):
            if attr_map is None:
                continue
            flat = attr_map.flatten()
            sorted_idx = np.argsort(flat)
            tag = f"[{group_name}#{img_idx} lbl={case['label']:.0f}]"
            # Per-image top and bottom N
            for j in range(1, min(top_n, len(sorted_idx)) + 1):
                all_top.append((float(flat[sorted_idx[-j]]), tag))
            for j in range(min(top_n, len(sorted_idx))):
                all_bot.append((float(flat[sorted_idx[j]]), tag))

    # Sort globally
    all_top.sort(key=lambda x: x[0], reverse=True)
    all_bot.sort(key=lambda x: x[0])

    logger.info(f"=== Top {top_n} attribution values (descending) ===")
    for rank, (val, tag) in enumerate(all_top[:top_n], 1):
        logger.info(f"  #{rank:>3}  {val:.6f}  {tag}")

    logger.info(f"=== Bottom {top_n} attribution values (ascending) ===")
    for rank, (val, tag) in enumerate(all_bot[:top_n], 1):
        logger.info(f"  #{rank:>3}  {val:.6f}  {tag}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def denormalize_image(img_tensor, normalize_mean=None, normalize_std=None):
    """Convert a normalized image tensor to a uint8 numpy RGB array.

    Reverses the torchvision ``Normalize`` transform:
    ``original = img * std + mean``

    Parameters
    ----------
    img_tensor : torch.Tensor, shape (1, C, H, W)
    normalize_mean : list[float] or None
    normalize_std  : list[float] or None

    Returns
    -------
    np.ndarray, dtype uint8, shape (H, W, 3)
    """
    img = img_tensor.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    if normalize_mean is not None and normalize_std is not None:
        std  = np.asarray(normalize_std,  dtype=np.float32)
        mean = np.asarray(normalize_mean, dtype=np.float32)
        img  = img * std + mean
    img = np.clip(img, 0.0, 1.0)
    return np.uint8(255 * img)


def _make_overlay(orig_uint8, attr_map, alpha=0.55):
    """Blend a [0,1] heatmap over an original image using a red tint.

    Parameters
    ----------
    orig_uint8 : np.ndarray, uint8, shape (H, W, 3)
    attr_map   : np.ndarray, float, shape (H, W), values in [0, 1]
    alpha      : float — maximum blend weight for the red channel

    Returns
    -------
    np.ndarray, uint8, shape (H, W, 3)
    """
    img_f     = orig_uint8.astype(np.float32) / 255.0
    red       = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    alpha_map = (attr_map * alpha)[:, :, np.newaxis]           # (H, W, 1)
    blended   = (1.0 - alpha_map) * img_f + alpha_map * red
    return np.uint8(255 * np.clip(blended, 0, 1))


def _make_diverging_overlay(orig_uint8, attr_map_norm, alpha=0.75):
    """Blend a signed [-1,1] attribution map onto original.

    Red tint for positive (energy boosters), blue tint for negative
    (energy blockers).

    Parameters
    ----------
    orig_uint8    : np.ndarray, uint8, (H, W, 3)
    attr_map_norm : np.ndarray, float, (H, W), values in [-1, 1]
    alpha         : float — maximum blend weight

    Returns
    -------
    np.ndarray, uint8, (H, W, 3)
    """
    img_f = orig_uint8.astype(np.float32) / 255.0

    pos = np.clip(attr_map_norm, 0, 1)
    neg = np.clip(-attr_map_norm, 0, 1)

    red  = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    a_pos = (pos * alpha)[:, :, np.newaxis]
    a_neg = (neg * alpha)[:, :, np.newaxis]

    blended = (1.0 - a_pos - a_neg) * img_f + a_pos * red + a_neg * blue
    return np.uint8(255 * np.clip(blended, 0, 1))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_attribution_grid(
    cases,
    attr_fn,
    title_prefix,
    method_name,
    experiment_name,
    save_path,
    normalize_mean=None,
    normalize_std=None,
    alpha=0.55,
    precomputed_maps=None,
):
    """Generate and save a visualization grid for one group of examples.

    For each case the figure shows three columns:
      Col 1 – Original image (denormalized to uint8 RGB)
      Col 2 – Attribution map (absolute values, normalized 0→1, "Reds" colormap)
      Col 3 – Attribution overlaid on the original (red-tint blend)

    Parameters
    ----------
    cases : list[dict]
        Dicts with keys: ``image``, ``label``, ``prediction``, ``error``.
    attr_fn : callable or None
        ``attr_fn(img_tensor) -> np.ndarray (H, W)`` — must return values in
        [0, 1] (attribution magnitude, already normalized).  May be ``None``
        when ``precomputed_maps`` is provided.
    title_prefix : str
        Human-readable group label (e.g. ``"Low Irradiance"``).
    method_name : str
        Attribution method name shown in the super-title (e.g.
        ``"Integrated Gradients"``).
    experiment_name : str
        Experiment tag shown in the super-title (e.g. ``"Fisheye"``).
    save_path : str
        Full path (including filename) where the PNG is saved.
    normalize_mean, normalize_std : list[float] or None
        Passed to :func:`denormalize_image`.
    alpha : float
        Blending strength for the overlay (0 = no tint, 1 = fully red).
    precomputed_maps : list[np.ndarray] or None
        If provided, these normalized [0,1] maps are used directly instead of
        calling ``attr_fn``.  ``attr_fn`` may be ``None`` when this is given.
    """
    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.5 * n + 1.2))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    col_headers = [
        "Original Image",
        f"Attribution Map ({method_name})",
        "Attribution Overlay",
    ]

    logger.info(f"Generating {method_name} grid: {title_prefix} ({n} images) …")

    # Use precomputed maps if provided, otherwise call attr_fn
    if precomputed_maps is not None:
        all_attr_maps = list(precomputed_maps)
    else:
        all_attr_maps = [attr_fn(case["image"]) for case in cases]

    # Determine if signed attribution (IG) or magnitude-only (Occlusion)
    is_signed = any(np.any(a < 0) for a in all_attr_maps)

    # Compute global vmin/vmax for color normalization
    if is_signed:
        absmax = max(abs(np.min(a)) for a in all_attr_maps)
        absmax = max(absmax, max(abs(np.max(a)) for a in all_attr_maps))
        vmin, vmax = -absmax, absmax
        cmap = "RdBu_r"
    else:
        vmin, vmax = 0, 1
        cmap = "Reds"

    for i, (case, attr_map) in enumerate(zip(cases, all_attr_maps)):
        img_tensor = case["image"]

        # Normalize signed maps to [-1, 1] for visualization
        if is_signed and absmax > 0:
            attr_map_vis = np.clip(attr_map / absmax, -1, 1)
        elif not is_signed and np.max(attr_map) > 0:
            attr_map_vis = attr_map / np.max(attr_map)
        else:
            attr_map_vis = attr_map

        # Denormalize image and build overlay (red tint for positive only)
        orig_uint8 = denormalize_image(img_tensor, normalize_mean, normalize_std)
        overlay = _make_overlay(orig_uint8, np.clip(attr_map_vis, 0, 1), alpha=alpha)

        label_text = (
            f"True: {case['label']:.0f} W/m²  |  "
            f"Pred: {case['prediction']:.0f} W/m²  |  "
            f"Err: {case['error']:.1f}"
        )

        # Col 1 – Original image
        axes[i, 0].imshow(orig_uint8)
        if i == 0:
            axes[i, 0].set_title(f"{col_headers[0]}\n{label_text}", fontsize=9)
        else:
            axes[i, 0].set_title(label_text, fontsize=9)
        axes[i, 0].axis("off")

        # Col 2 – Attribution map
        axes[i, 1].imshow(attr_map_vis, cmap=cmap, vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 1].set_title(col_headers[1], fontsize=9)
        axes[i, 1].axis("off")

        # Col 3 – Attribution overlay
        axes[i, 2].imshow(overlay)
        if i == 0:
            axes[i, 2].set_title(col_headers[2], fontsize=9)
        axes[i, 2].axis("off")

    # Colorbar legend
    cbar_ax = fig.add_axes([0.15, 0.025, 0.7, 0.012])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    if is_signed:
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_ticklabels(["Negative (cloud blocks)", "Zero", "Positive (sun boosts)"])
    else:
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["No attribution", "Medium", "High attribution"])
    cbar.ax.tick_params(labelsize=9)

    caption = (
        f"Col 1: Original sky image.  "
        f"Col 2: {method_name} attribution — red = positive (increases output), blue = negative (decreases output).  "
        f"Col 3: Attribution blended onto the original image (red tint = positive only)."
    )
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8, style="italic",
             wrap=True, color="0.3")

    plt.suptitle(
        f"{title_prefix}  ({method_name} — {experiment_name})",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {save_path}")


def plot_ig_signed_grid(
    cases,
    attr_fn,
    title_prefix,
    method_name,
    experiment_name,
    save_path,
    normalize_mean=None,
    normalize_std=None,
    alpha=0.75,
    precomputed_maps=None,
):
    """5-column IG grid: original | negative | positive | mixed heatmap | mixed overlay.

    ``attr_fn`` must return a **signed** attribution map (np.sum across
    channels, not L2 norm).  Positive = increases prediction, negative =
    decreases prediction.  May be ``None`` when ``precomputed_maps`` is given.

    Group-level normalization: all rows share the same absmax scale so
    inter-image comparisons are valid.  No per-image clipping or gamma.

    Parameters
    ----------
    cases : list[dict]
        Dicts with keys: ``image``, ``label``, ``prediction``, ``error``.
    attr_fn : callable or None
        ``attr_fn(img_tensor) -> np.ndarray (H, W)`` — signed attribution.
        May be ``None`` when ``precomputed_maps`` is provided.
    title_prefix, method_name, experiment_name, save_path : str
    normalize_mean, normalize_std : list[float] or None
    alpha : float
        Overlay blend strength.
    precomputed_maps : list[np.ndarray or None] or None
        Raw signed attribution maps (one per case, or None for OOM).
        If provided, ``attr_fn`` is not called.
    """
    n = len(cases)
    fig, axes = plt.subplots(n, 5, figsize=(27, 5.0 * n + 1.6))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    col_headers = [
        "Original Image",
        "Negative Attribution\n(blocks energy)",
        "Positive Attribution\n(boosts energy)",
        "Mixed Attribution\n(RdBu_r heatmap)",
        "Mixed Overlay\non Original",
    ]

    logger.info(
        f"Generating signed {method_name} grid: {title_prefix} ({n} images) …"
    )

    # Use precomputed maps or compute via attr_fn
    if precomputed_maps is not None:
        all_attr_maps = list(precomputed_maps)
    else:
        all_attr_maps = []
        for case in tqdm(cases, desc=f"{title_prefix} (attr)", leave=False):
            all_attr_maps.append(attr_fn(case["image"]))

    # Group-level absmax (skip None/OOM maps)
    valid_maps = [m for m in all_attr_maps if m is not None]
    if valid_maps:
        absmax = max(np.abs(m).max() for m in valid_maps)
    else:
        absmax = 1.0
    if absmax == 0:
        absmax = 1.0

    for i, (case, attr_map) in enumerate(zip(cases, all_attr_maps)):
        if attr_map is None:
            for col in range(5):
                axes[i, col].axis("off")
                if col == 0:
                    axes[i, col].set_title(
                        f"OOM — skipped\n"
                        f"True: {case['label']:.0f} W/m²  |  "
                        f"Pred: {case['prediction']:.0f} W/m²",
                        fontsize=9,
                    )
            continue

        # Normalize to [-1, 1] using group absmax
        attr_norm = np.clip(attr_map / absmax, -1.0, 1.0)

        neg_display = np.clip(-attr_norm, 0, 1)
        pos_display = np.clip(attr_norm, 0, 1)

        orig_uint8 = denormalize_image(
            case["image"], normalize_mean, normalize_std
        )
        overlay = _make_diverging_overlay(orig_uint8, attr_norm, alpha=alpha)

        label_text = (
            f"True: {case['label']:.0f} W/m²  |  "
            f"Pred: {case['prediction']:.0f} W/m²  |  "
            f"Err: {case['error']:.1f}"
        )

        # Col 1 — Original
        axes[i, 0].imshow(orig_uint8)
        if i == 0:
            axes[i, 0].set_title(
                f"{col_headers[0]}\n{label_text}", fontsize=9,
            )
        else:
            axes[i, 0].set_title(label_text, fontsize=9)
        axes[i, 0].axis("off")

        # Col 2 — Negative (white = nothing, dark blue = strong blocker)
        axes[i, 1].imshow(neg_display, cmap="GnBu", vmin=0, vmax=1)
        if i == 0:
            axes[i, 1].set_title(col_headers[1], fontsize=9)
        axes[i, 1].axis("off")

        # Col 3 — Positive (white = nothing, dark red = strong booster)
        axes[i, 2].imshow(pos_display, cmap="YlOrRd", vmin=0, vmax=1)
        if i == 0:
            axes[i, 2].set_title(col_headers[2], fontsize=9)
        axes[i, 2].axis("off")

        # Col 4 — Mixed/signed diverging heatmap (RdBu_r standalone)
        axes[i, 3].imshow(attr_norm, cmap="RdBu_r", vmin=-1, vmax=1)
        if i == 0:
            axes[i, 3].set_title(col_headers[3], fontsize=9)
        axes[i, 3].axis("off")

        # Col 5 — Diverging overlay on original
        axes[i, 4].imshow(overlay)
        if i == 0:
            axes[i, 4].set_title(col_headers[4], fontsize=9)
        axes[i, 4].axis("off")

    # Colorbars
    neg_ax = fig.add_axes([0.04, 0.025, 0.28, 0.012])
    neg_sm = ScalarMappable(
        cmap="GnBu", norm=mcolors.Normalize(vmin=0, vmax=1),
    )
    neg_sm.set_array([])
    neg_cb = fig.colorbar(neg_sm, cax=neg_ax, orientation="horizontal")
    neg_cb.set_ticks([0, 0.5, 1])
    neg_cb.set_ticklabels(["No effect", "", "Strong blocker"])
    neg_cb.ax.tick_params(labelsize=8)

    pos_ax = fig.add_axes([0.36, 0.025, 0.28, 0.012])
    pos_sm = ScalarMappable(
        cmap="YlOrRd", norm=mcolors.Normalize(vmin=0, vmax=1),
    )
    pos_sm.set_array([])
    pos_cb = fig.colorbar(pos_sm, cax=pos_ax, orientation="horizontal")
    pos_cb.set_ticks([0, 0.5, 1])
    pos_cb.set_ticklabels(["No effect", "", "Strong booster"])
    pos_cb.ax.tick_params(labelsize=8)

    mix_ax = fig.add_axes([0.68, 0.025, 0.28, 0.012])
    mix_sm = ScalarMappable(
        cmap="RdBu_r", norm=mcolors.Normalize(vmin=-1, vmax=1),
    )
    mix_sm.set_array([])
    mix_cb = fig.colorbar(mix_sm, cax=mix_ax, orientation="horizontal")
    mix_cb.set_ticks([-1, 0, 1])
    mix_cb.set_ticklabels(["Negative (blocks)", "Neutral", "Positive (boosts)"])
    mix_cb.ax.tick_params(labelsize=8)

    caption = (
        f"Col 1: Original sky image.  "
        f"Col 2: Negative IG — blue regions decrease predicted irradiance "
        f"(clouds, shadows).  "
        f"Col 3: Positive IG — red/orange regions increase predicted "
        f"irradiance (sun, clear sky).  "
        f"Col 4: Mixed attribution heatmap (RdBu_r: red=positive, blue=negative).  "
        f"Col 5: Mixed overlay on original (red=positive, blue=negative tint).  "
        f"Group-level absmax normalization (all rows share the same scale)."
    )
    fig.text(
        0.5, 0.003, caption, ha="center", fontsize=7,
        style="italic", wrap=True, color="0.3",
    )

    plt.suptitle(
        f"{title_prefix}  ({method_name} — {experiment_name})",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def save_summary_csv(results, save_path):
    """Save per-sample inference metadata to a CSV file.

    Parameters
    ----------
    results : list[dict]
        From :func:`collect_all_results` (keys: ``label``, ``prediction``,
        ``error``).
    save_path : str
        Full path for the CSV file (parent directories are created if needed).
    """
    rows = [
        {
            "sample_index":   i,
            "label_wm2":      r["label"],
            "prediction_wm2": r["prediction"],
            "abs_error_wm2":  r["error"],
        }
        for i, r in enumerate(results)
    ]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Summary CSV saved → {save_path} ({len(df)} rows)")
