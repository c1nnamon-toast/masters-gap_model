# Visualization Grid Search Results

**Generated**: 2026-04-04 17:43:51  
**Total runtime**: 4051.3 s (67.5 min)

---

## Results

| # | Experiment | Method | Hyperparameters | Status | Time (s) | Output |
|--:|-----------|--------|-----------------|:------:|----------|--------|
| 1 | Fisheye | IG | n_steps=20 | OK | 76.3 | `analysis/viz_gridsearch/fisheye/ig_nsteps20` |
| 2 | Fisheye | IG | n_steps=40 | OK | 98.4 | `analysis/viz_gridsearch/fisheye/ig_nsteps40` |
| 3 | Fisheye | IG | n_steps=80 | OK | 144.5 | `analysis/viz_gridsearch/fisheye/ig_nsteps80` |
| 4 | Fisheye | IG | n_steps=120 | OK | 190.0 | `analysis/viz_gridsearch/fisheye/ig_nsteps120` |
| 5 | Fisheye | Occlusion | patch=30, stride=13 | OK | 1968.7 | `analysis/viz_gridsearch/fisheye/occ_p30_s13` |
| 6 | Rubsheet 3-to-1 | IG | n_steps=20 | OK | 56.5 | `analysis/viz_gridsearch/rubsheet_3to1/ig_nsteps20` |
| 7 | Rubsheet 3-to-1 | IG | n_steps=40 | OK | 72.9 | `analysis/viz_gridsearch/rubsheet_3to1/ig_nsteps40` |
| 8 | Rubsheet 3-to-1 | IG | n_steps=80 | OK | 107.5 | `analysis/viz_gridsearch/rubsheet_3to1/ig_nsteps80` |
| 9 | Rubsheet 3-to-1 | IG | n_steps=120 | OK | 141.6 | `analysis/viz_gridsearch/rubsheet_3to1/ig_nsteps120` |
| 10 | Rubsheet 3-to-1 | Occlusion | patch=30, stride=13 | OK | 1065.9 | `analysis/viz_gridsearch/rubsheet_3to1/occ_p30_s13` |

**Completed**: 10/10 succeeded, 0 failed

---

## Grid Search Configuration

### Integrated Gradients

| Experiment | n_steps values |
|-----------|----------------|
| Fisheye | 20, 40, 80, 120 |
| Rubsheet 3-to-1 | 20, 40, 80, 120 |

### Occlusion (full cross-product: patch_size x stride)

| Experiment | patch_sizes | strides | combos (stride <= patch) |
|-----------|-------------|---------|--------------------------|
| Fisheye | 30 | 13 | 1 |
| Rubsheet 3-to-1 | 30 | 13 | 1 |
