"""Grid search over IG, Occlusion, and Grad-CAM visualization hyperparameters.

Runs Integrated Gradients, Occlusion, and Grad-CAM attribution analysis
across a small grid of hyperparameters for each configured experiment.

Data and model are loaded once per experiment; only the attribution step is
repeated for each hyperparameter combination.  GPU memory is freed after
every trial.  Any trial that raises an exception is logged and skipped.

Output structure
----------------
    visualization/gap/
        fisheye/
            summary.csv
            ig_nsteps15/   ... ig_nsteps80/
            occ_p15_s12/   ... occ_p25_s15/
            gradcam_conv_block4/
        rubsheet_3to1/
            summary.csv
            ig_nsteps15/   ... ig_nsteps80/
            occ_p15_s12/   ... occ_p25_s15/
            gradcam_conv_block4/
        rubsheet_square/
            ...
        gridsearch_results.md

Run
---
    python main_viz.py
"""

import sys
import os
import gc
import glob
import re
import time
import logging
import traceback
import importlib.util
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from captum.attr import IntegratedGradients, Occlusion, LayerGradCam

from nn.model import SkyCNN
from nn.loader import get_dataloaders
from utils.attribution_map import (
    collect_all_results,
    select_groups,
    plot_attribution_grid,
    plot_ig_signed_grid,
    save_summary_csv,
)
from logger_helper import logger_setup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid search configuration
# ---------------------------------------------------------------------------

N_PER_GROUP = 7

EXPERIMENTS = {
    "fisheye": {
        "config_path": os.path.join(
            PROJECT_ROOT, "experiments", "fisheye", "config_fisheye.py"
        ),
        "experiment_name": "Fisheye",
        "ig_grid": [
            {"n_steps": 15},
            {"n_steps": 20},
            {"n_steps": 40},
            {"n_steps": 80},
        ],
        "occlusion_patch_sizes": [15, 25],
        "occlusion_strides":     [12, 15],
        "gradcam_grid": [
            {"layer_name": "conv_block4"},
        ],
    },
    "rubsheet_3to1": {
        "config_path": os.path.join(
            PROJECT_ROOT, "experiments", "rubsheet_3to1", "config_rubsheet_3to1.py"
        ),
        "experiment_name": "Rubsheet 3-to-1",
        "ig_grid": [
            {"n_steps": 15},
            {"n_steps": 20},
            {"n_steps": 40},
            {"n_steps": 80},
        ],
        "occlusion_patch_sizes": [15, 25],
        "occlusion_strides":     [12, 15],
        "gradcam_grid": [
            {"layer_name": "conv_block4"},
        ],
    },
    "rubsheet_square": {
        "config_path": os.path.join(
            PROJECT_ROOT, "experiments", "rubsheet_square", "config_rubsheet_square.py"
        ),
        "experiment_name": "Rubsheet Square",
        "ig_grid": [
            {"n_steps": 15},
            {"n_steps": 20},
            {"n_steps": 40},
            {"n_steps": 80},
        ],
        "occlusion_patch_sizes": [15, 25],
        "occlusion_strides":     [12, 15],
        "gradcam_grid": [
            {"layer_name": "conv_block4"},
        ],
    },
}

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "visualization", "gap")

GROUP_TITLES = {
    "low":    "Low Irradiance",
    "medium": "Medium Irradiance",
    "high":   "High Irradiance",
    "best":   "Best Predictions (lowest error)",
    "worst":  "Worst Predictions (highest error)",
    "random": "Random Sample",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path):
    """Dynamically import a config module from an absolute path."""
    spec = importlib.util.spec_from_file_location("_cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def find_latest_model(base_path):
    """Return the most recent timestamped .pth checkpoint (gap models only)."""
    model_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path).replace(".pth", "")
    pattern = os.path.join(model_dir, f"{base_name}_*.pth")
    model_files = glob.glob(pattern)
    model_files = [f for f in model_files if "gap" in os.path.basename(f)]

    if not model_files:
        if os.path.exists(base_path):
            return base_path
        raise FileNotFoundError(f"No gap model files found matching {pattern}")

    def _ts(path):
        m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.pth$", path)
        return m.group(1) if m else ""

    model_files.sort(key=_ts, reverse=True)
    return model_files[0]


def cleanup_gpu():
    """Release GPU memory between trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def resolve_layer(model, layer_name):
    """Resolve a dot-separated layer name to the corresponding nn.Module.

    Examples
    --------
    >>> resolve_layer(model, "conv_block4")        # -> model.conv_block4
    >>> resolve_layer(model, "conv_block4.3")       # -> model.conv_block4[3]
    """
    module = model
    for part in layer_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


# ---------------------------------------------------------------------------
# Attribution function factories
# ---------------------------------------------------------------------------


def make_ig_attr_fn(ig, device, n_steps):
    """Return a callable: img_tensor (1,C,H,W) -> attr_map (H,W) with sign (sum across channels)."""

    def attr_fn(img_tensor):
        inp = img_tensor.to(device)
        baseline = torch.zeros_like(inp)
        attributions = ig.attribute(
            inp, baseline, n_steps=n_steps, target=0, internal_batch_size=4
        )

        attr_np = attributions.detach().cpu().numpy()[0]  # (3, H, W)
        # Sum across channels to preserve sign
        attr_map = np.sum(attr_np, axis=0)
        del attributions, inp, baseline
        torch.cuda.empty_cache()
        return attr_map  # signed map

    return attr_fn


def make_occlusion_attr_fn(occ, device, patch_size, stride, baseline_value=0.0):
    """Return a callable: img_tensor (1,C,H,W) -> attr_map (H,W) in [0,1]."""
    sliding_window = (3, patch_size, patch_size)
    stride_tuple = (1, stride, stride)

    def attr_fn(img_tensor):
        inp = img_tensor.to(device)
        baseline = torch.full_like(inp, fill_value=baseline_value)
        attributions = occ.attribute(
            inp,
            sliding_window_shapes=sliding_window,
            strides=stride_tuple,
            baselines=baseline,
            target=0,
        )

        attr_np = attributions.detach().cpu().numpy()[0]
        attr_map = np.linalg.norm(attr_np, axis=0)
        del attributions, inp, baseline
        torch.cuda.empty_cache()

        amax = attr_map.max()
        if amax > 0:
            attr_map /= amax
        return attr_map

    return attr_fn


def make_gradcam_attr_fn(gradcam, device):
    """Return a callable: img_tensor (1,C,H,W) -> attr_map (H,W) in [0,1].

    Grad-CAM produces a coarse heatmap at the spatial resolution of the
    target convolutional layer.  This function upsamples it back to the
    input resolution via bilinear interpolation and normalises to [0, 1].
    ``relu_attributions=True`` is used so that only positively-contributing
    regions are shown — standard practice for Grad-CAM.
    """

    def attr_fn(img_tensor):
        inp = img_tensor.to(device)
        input_h, input_w = inp.shape[2], inp.shape[3]

        attributions = gradcam.attribute(
            inp, target=0, relu_attributions=True
        )
        # attributions shape: (1, C_layer, H_layer, W_layer)
        # Sum across channel dimension to get a single spatial map
        attr_map = attributions.sum(dim=1, keepdim=True)  # (1, 1, H_l, W_l)

        # Upsample to input resolution
        attr_map = F.interpolate(
            attr_map,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )

        attr_np = attr_map.detach().cpu().numpy()[0, 0]  # (H, W)

        del attributions, inp, attr_map
        torch.cuda.empty_cache()

        # Normalise to [0, 1]
        amax = attr_np.max()
        if amax > 0:
            attr_np /= amax
        return attr_np

    return attr_fn


# ---------------------------------------------------------------------------
# Trial runners
# ---------------------------------------------------------------------------


def run_ig_trial(model, groups, config, output_dir, experiment_name, n_steps):
    """Run one IG hyperparameter trial and save all group grids."""
    ig = IntegratedGradients(model)
    attr_fn = make_ig_attr_fn(ig, config.DEVICE, n_steps=n_steps)

    completeness_rows = []
    baseline_value = 0.0  # IG baseline is zeros_like (black)

    for group_name, cases in groups.items():
        save_path = os.path.join(output_dir, f"{group_name}_ig.png")

        # For completeness: collect per-image completeness info
        for idx, case in enumerate(cases):
            img_tensor = case["image"]
            inp = img_tensor.to(config.DEVICE)
            baseline = torch.zeros_like(inp)
            # Model prediction for image
            with torch.no_grad():
                pred = float(model(inp).cpu().numpy().squeeze())
                pred_baseline = float(model(baseline).cpu().numpy().squeeze())
            # Attribution map (signed)
            attributions = ig.attribute(
                inp, baseline, n_steps=n_steps, target=0, internal_batch_size=4
            )
            attr_np = attributions.detach().cpu().numpy()[0]  # (3, H, W)
            attr_map = np.sum(attr_np, axis=0)
            attr_sum = float(np.sum(attr_map))
            diff = pred - pred_baseline
            error = attr_sum - diff
            completeness_rows.append({
                "group": group_name,
                "index": idx,
                "label": case["label"],
                "prediction": pred,
                "baseline_prediction": pred_baseline,
                "attr_sum": attr_sum,
                "diff": diff,
                "completeness_error": error,
            })

        plot_ig_signed_grid(
            cases=cases,
            attr_fn=attr_fn,
            title_prefix=GROUP_TITLES[group_name],
            method_name=f"IG (n_steps={n_steps})",
            experiment_name=experiment_name,
            save_path=save_path,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
        )

    # Save completeness report
    import pandas as pd
    completeness_path = os.path.join(output_dir, "completeness.csv")
    pd.DataFrame(completeness_rows).to_csv(completeness_path, index=False)
    logger.info(f"IG completeness report saved → {completeness_path}")

    del ig, attr_fn


def run_occlusion_trial(
    model, groups, config, output_dir, experiment_name, patch_size, stride
):
    """Run one Occlusion hyperparameter trial and save all group grids."""
    occ = Occlusion(model)
    attr_fn = make_occlusion_attr_fn(
        occ, config.DEVICE, patch_size=patch_size, stride=stride
    )

    for group_name, cases in groups.items():
        save_path = os.path.join(output_dir, f"{group_name}_occlusion.png")
        plot_attribution_grid(
            cases=cases,
            attr_fn=attr_fn,
            title_prefix=GROUP_TITLES[group_name],
            method_name=f"Occlusion (patch={patch_size}, stride={stride})",
            experiment_name=experiment_name,
            save_path=save_path,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
        )

    del occ, attr_fn


def run_gradcam_trial(
    model, groups, config, output_dir, experiment_name, layer_name
):
    """Run one Grad-CAM trial and save all group grids.

    Grad-CAM targets a specific convolutional layer and produces a coarse
    heatmap that is upsampled to input resolution.  In the regression
    setting the resulting maps tend to be diffuse — this is expected and
    documented in the thesis (Section 2.4 / Section 3.6).
    """
    layer = resolve_layer(model, layer_name)
    gradcam = LayerGradCam(model, layer)
    attr_fn = make_gradcam_attr_fn(gradcam, config.DEVICE)

    for group_name, cases in groups.items():
        save_path = os.path.join(output_dir, f"{group_name}_gradcam.png")
        plot_attribution_grid(
            cases=cases,
            attr_fn=attr_fn,
            title_prefix=GROUP_TITLES[group_name],
            method_name=f"Grad-CAM (layer={layer_name})",
            experiment_name=experiment_name,
            save_path=save_path,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
        )

    del gradcam, attr_fn


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_summary_md(run_log, total_time, output_path):
    """Write a markdown report summarising every grid-search trial."""
    lines = [
        "# Visualization Grid Search Results",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total runtime**: {total_time:.1f} s ({total_time / 60:.1f} min)",
        "",
        "---",
        "",
        "## Results",
        "",
        "| # | Experiment | Method | Hyperparameters | Status | Time (s) | Output |",
        "|--:|-----------|--------|-----------------|:------:|----------|--------|",
    ]

    n_ok = n_fail = 0
    for i, e in enumerate(run_log, 1):
        ok = e["status"] == "SUCCESS"
        n_ok += ok
        n_fail += not ok
        lines.append(
            f"| {i} | {e['experiment']} | {e['method']} | "
            f"{e['params']} | {'OK' if ok else 'FAIL'} | "
            f"{e['time']:.1f} | `{e['output_dir']}` |"
        )

    lines += [
        "",
        f"**Completed**: {n_ok}/{len(run_log)} succeeded, {n_fail} failed",
        "",
    ]

    failures = [e for e in run_log if e["status"] != "SUCCESS"]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for e in failures:
            lines.append(
                f"### {e['experiment']} / {e['method']} / {e['params']}"
            )
            lines.append("")
            lines.append(f"```\n{e['error']}\n```")
            lines.append("")

    lines += [
        "---",
        "",
        "## Grid Search Configuration",
        "",
        "### Integrated Gradients",
        "",
        "| Experiment | n_steps values |",
        "|-----------|----------------|",
    ]
    for spec in EXPERIMENTS.values():
        if "ig_grid" in spec and spec["ig_grid"]:
            steps = ", ".join(str(p["n_steps"]) for p in spec["ig_grid"])
        else:
            steps = "-"
        lines.append(f"| {spec['experiment_name']} | {steps} |")
    lines += [
        "",
        "### Occlusion (full cross-product: patch_size x stride)",
        "",
        "| Experiment | patch_sizes | strides | combos (stride <= patch) |",
        "|-----------|-------------|---------|--------------------------|",
    ]
    for spec in EXPERIMENTS.values():
        if (
            "occlusion_patch_sizes" in spec
            and "occlusion_strides" in spec
            and spec["occlusion_patch_sizes"]
            and spec["occlusion_strides"]
        ):
            patches = ", ".join(str(p) for p in spec["occlusion_patch_sizes"])
            strides = ", ".join(str(s) for s in spec["occlusion_strides"])
            n_combos = sum(
                1
                for p in spec["occlusion_patch_sizes"]
                for s in spec["occlusion_strides"]
                if s <= p
            )
            lines.append(
                f"| {spec['experiment_name']} | {patches} | {strides} | {n_combos} |"
            )
        else:
            lines.append(f"| {spec['experiment_name']} | - | - | - |")

    lines += [
        "",
        "### Grad-CAM",
        "",
        "| Experiment | target layers |",
        "|-----------|---------------|",
    ]
    for spec in EXPERIMENTS.values():
        if "gradcam_grid" in spec and spec["gradcam_grid"]:
            layers = ", ".join(p["layer_name"] for p in spec["gradcam_grid"])
        else:
            layers = "-"
        lines.append(f"| {spec['experiment_name']} | {layers} |")
    lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info(f"Summary written -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger_setup()

    if not torch.cuda.is_available():
        logger.error("No GPU found — aborting.")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("  Visualization Grid Search — IG & Occlusion & Grad-CAM")
    logger.info("=" * 80)

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    run_log: list[dict] = []
    total_start = time.time()

    for exp_key, exp_spec in EXPERIMENTS.items():
        exp_name = exp_spec["experiment_name"]
        logger.info("")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Experiment: {exp_name}")
        logger.info(f"{'=' * 60}")

        # -- Load config dynamically --
        config = load_config(exp_spec["config_path"])
        device = config.DEVICE

        # -- Load data (once per experiment) --
        logger.info("Loading test data ...")
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

        # -- Load model (once per experiment) --
        model_path = find_latest_model(config.MODEL_SAVE_PATH)
        logger.info(f"Model: {model_path}")
        try:
            model = SkyCNN().to(device)
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
        except RuntimeError as _oom:
            if "out of memory" in str(_oom).lower():
                logger.warning("CUDA OOM loading model — falling back to CPU.")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                _, _, test_loader, _ = get_dataloaders(
                    train_csv=config.TRAIN_CSV,
                    val_csv=config.VAL_CSV,
                    test_csv=config.TEST_CSV,
                    train_image_dir=config.TRAIN_IMAGE_DIR,
                    val_image_dir=config.VAL_IMAGE_DIR,
                    test_image_dir=config.TEST_IMAGE_DIR,
                    batch_size=config.BATCH_SIZE,
                    num_workers=config.NUM_WORKERS,
                    pin_memory=False,
                    normalize_mean=config.NORMALIZE_MEAN,
                    normalize_std=config.NORMALIZE_STD,
                )
                model = SkyCNN().to(device)
                model.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=True)
                )
            else:
                raise
        model.eval()

        # -- Collect predictions & pick groups (once per experiment) --
        results = collect_all_results(model, test_loader, device)
        groups = select_groups(results, n=N_PER_GROUP)

        exp_output = os.path.join(OUTPUT_BASE, exp_key)
        os.makedirs(exp_output, exist_ok=True)
        save_summary_csv(results, os.path.join(exp_output, "summary.csv"))

        # ── IG trials ──────────────────────────────────────────────────────
        if "ig_grid" in exp_spec and exp_spec["ig_grid"]:
            for ig_params in exp_spec["ig_grid"]:
                n_steps = ig_params["n_steps"]
                tag = f"ig_nsteps{n_steps}"
                trial_dir = os.path.join(exp_output, tag)
                os.makedirs(trial_dir, exist_ok=True)
                params_str = f"n_steps={n_steps}"

                logger.info(f"  [IG] {params_str}")
                t0 = time.time()
                try:
                    run_ig_trial(
                        model, groups, config, trial_dir, exp_name, n_steps
                    )
                    elapsed = time.time() - t0
                    run_log.append(dict(
                        experiment=exp_name, method="IG",
                        params=params_str, status="SUCCESS",
                        time=elapsed,
                        output_dir=os.path.relpath(trial_dir, PROJECT_ROOT),
                        error="",
                    ))
                    logger.info(f"       OK  ({elapsed:.1f} s)")
                except Exception as exc:
                    elapsed = time.time() - t0
                    tb = traceback.format_exc()
                    run_log.append(dict(
                        experiment=exp_name, method="IG",
                        params=params_str, status="FAILED",
                        time=elapsed,
                        output_dir=os.path.relpath(trial_dir, PROJECT_ROOT),
                        error=tb,
                    ))
                    logger.error(f"       FAILED  ({elapsed:.1f} s): {exc}")
                finally:
                    cleanup_gpu()

        # ── Occlusion trials ───────────────────────────────────────────────
        if (
            "occlusion_patch_sizes" in exp_spec
            and exp_spec["occlusion_patch_sizes"]
            and "occlusion_strides" in exp_spec
            and exp_spec["occlusion_strides"]
        ):
            for ps in exp_spec["occlusion_patch_sizes"]:
                for st in exp_spec["occlusion_strides"]:
                    if st > ps:
                        logger.info(
                            f"  [Occ] skip stride={st} > patch={ps}"
                        )
                        continue
                    tag = f"occ_p{ps}_s{st}"
                    trial_dir = os.path.join(exp_output, tag)
                    os.makedirs(trial_dir, exist_ok=True)
                    params_str = f"patch={ps}, stride={st}"

                    logger.info(f"  [Occ] {params_str}")
                    t0 = time.time()
                    try:
                        run_occlusion_trial(
                            model, groups, config, trial_dir, exp_name, ps, st
                        )
                        elapsed = time.time() - t0
                        run_log.append(dict(
                            experiment=exp_name, method="Occlusion",
                            params=params_str, status="SUCCESS",
                            time=elapsed,
                            output_dir=os.path.relpath(trial_dir, PROJECT_ROOT),
                            error="",
                        ))
                        logger.info(f"       OK  ({elapsed:.1f} s)")
                    except Exception as exc:
                        elapsed = time.time() - t0
                        tb = traceback.format_exc()
                        run_log.append(dict(
                            experiment=exp_name, method="Occlusion",
                            params=params_str, status="FAILED",
                            time=elapsed,
                            output_dir=os.path.relpath(trial_dir, PROJECT_ROOT),
                            error=tb,
                        ))
                        logger.error(f"       FAILED  ({elapsed:.1f} s): {exc}")
                    finally:
                        cleanup_gpu()

        # ── Grad-CAM trials ────────────────────────────────────────────────
        if "gradcam_grid" in exp_spec and exp_spec["gradcam_grid"]:
            for gc_params in exp_spec["gradcam_grid"]:
                layer_name = gc_params["layer_name"]
                tag = f"gradcam_{layer_name.replace('.', '_')}"
                trial_dir = os.path.join(exp_output, tag)
                os.makedirs(trial_dir, exist_ok=True)
                params_str = f"layer={layer_name}"

                logger.info(f"  [GradCAM] {params_str}")
                t0 = time.time()
                try:
                    run_gradcam_trial(
                        model, groups, config, trial_dir, exp_name,
                        layer_name,
                    )
                    elapsed = time.time() - t0
                    run_log.append(dict(
                        experiment=exp_name, method="Grad-CAM",
                        params=params_str, status="SUCCESS",
                        time=elapsed,
                        output_dir=os.path.relpath(trial_dir, PROJECT_ROOT),
                        error="",
                    ))
                    logger.info(f"       OK  ({elapsed:.1f} s)")
                except Exception as exc:
                    elapsed = time.time() - t0
                    tb = traceback.format_exc()
                    run_log.append(dict(
                        experiment=exp_name, method="Grad-CAM",
                        params=params_str, status="FAILED",
                        time=elapsed,
                        output_dir=os.path.relpath(trial_dir, PROJECT_ROOT),
                        error=tb,
                    ))
                    logger.error(f"       FAILED  ({elapsed:.1f} s): {exc}")
                finally:
                    cleanup_gpu()

        # -- Release experiment-level resources --
        del model, results, groups, test_loader
        cleanup_gpu()
        logger.info(f"  GPU memory released after {exp_name}.")

    total_time = time.time() - total_start

    # -- Write markdown report --
    summary_path = os.path.join(OUTPUT_BASE, "gridsearch_results.md")
    generate_summary_md(run_log, total_time, summary_path)

    # -- Final log --
    logger.info("")
    logger.info("=" * 80)
    logger.info(
        f"  Grid search complete.  {total_time:.1f} s  ({total_time / 60:.1f} min)"
    )
    logger.info("=" * 80)

    n_ok = sum(1 for r in run_log if r["status"] == "SUCCESS")
    n_fail = len(run_log) - n_ok
    logger.info(f"  {n_ok}/{len(run_log)} succeeded, {n_fail} failed")
    for e in run_log:
        flag = "OK" if e["status"] == "SUCCESS" else "FAIL"
        logger.info(
            f"    [{flag}]  {e['experiment']} / {e['method']} / "
            f"{e['params']}  ({e['time']:.1f} s)"
        )


if __name__ == "__main__":
    main()
