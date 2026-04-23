# Visualization Grid Search Results

**Generated**: 2026-04-20 15:59:47  
**Total runtime**: 13387.1 s (223.1 min)

---

## Results

| # | Experiment | Method | Hyperparameters | Status | Time (s) | Output |
|--:|-----------|--------|-----------------|:------:|----------|--------|
| 1 | Fisheye | IG | n_steps=15 | OK | 68.3 | `visualization/gap/fisheye/ig_nsteps15` |
| 2 | Fisheye | IG | n_steps=20 | OK | 72.9 | `visualization/gap/fisheye/ig_nsteps20` |
| 3 | Fisheye | IG | n_steps=40 | OK | 95.8 | `visualization/gap/fisheye/ig_nsteps40` |
| 4 | Fisheye | IG | n_steps=80 | OK | 141.9 | `visualization/gap/fisheye/ig_nsteps80` |
| 5 | Fisheye | Occlusion | patch=15, stride=12 | OK | 2368.7 | `visualization/gap/fisheye/occ_p15_s12` |
| 6 | Fisheye | Occlusion | patch=15, stride=15 | OK | 1556.8 | `visualization/gap/fisheye/occ_p15_s15` |
| 7 | Fisheye | Occlusion | patch=25, stride=12 | OK | 2316.3 | `visualization/gap/fisheye/occ_p25_s12` |
| 8 | Fisheye | Occlusion | patch=25, stride=15 | OK | 1516.5 | `visualization/gap/fisheye/occ_p25_s15` |
| 9 | Rubsheet 3-to-1 | IG | n_steps=15 | OK | 48.6 | `visualization/gap/rubsheet_3to1/ig_nsteps15` |
| 10 | Rubsheet 3-to-1 | IG | n_steps=20 | OK | 51.3 | `visualization/gap/rubsheet_3to1/ig_nsteps20` |
| 11 | Rubsheet 3-to-1 | IG | n_steps=40 | OK | 68.2 | `visualization/gap/rubsheet_3to1/ig_nsteps40` |
| 12 | Rubsheet 3-to-1 | IG | n_steps=80 | OK | 102.4 | `visualization/gap/rubsheet_3to1/ig_nsteps80` |
| 13 | Rubsheet 3-to-1 | Occlusion | patch=15, stride=12 | OK | 1310.3 | `visualization/gap/rubsheet_3to1/occ_p15_s12` |
| 14 | Rubsheet 3-to-1 | Occlusion | patch=15, stride=15 | OK | 841.9 | `visualization/gap/rubsheet_3to1/occ_p15_s15` |
| 15 | Rubsheet 3-to-1 | Occlusion | patch=25, stride=12 | OK | 1271.2 | `visualization/gap/rubsheet_3to1/occ_p25_s12` |
| 16 | Rubsheet 3-to-1 | Occlusion | patch=25, stride=15 | OK | 819.5 | `visualization/gap/rubsheet_3to1/occ_p25_s15` |
| 17 | Rubsheet Square | IG | n_steps=15 | OK | 37.9 | `visualization/gap/rubsheet_square/ig_nsteps15` |
| 18 | Rubsheet Square | IG | n_steps=20 | OK | 39.0 | `visualization/gap/rubsheet_square/ig_nsteps20` |
| 19 | Rubsheet Square | IG | n_steps=40 | OK | 44.1 | `visualization/gap/rubsheet_square/ig_nsteps40` |
| 20 | Rubsheet Square | IG | n_steps=80 | OK | 54.4 | `visualization/gap/rubsheet_square/ig_nsteps80` |
| 21 | Rubsheet Square | Occlusion | patch=15, stride=12 | OK | 124.7 | `visualization/gap/rubsheet_square/occ_p15_s12` |
| 22 | Rubsheet Square | Occlusion | patch=15, stride=15 | OK | 84.1 | `visualization/gap/rubsheet_square/occ_p15_s15` |
| 23 | Rubsheet Square | Occlusion | patch=25, stride=12 | OK | 119.6 | `visualization/gap/rubsheet_square/occ_p25_s12` |
| 24 | Rubsheet Square | Occlusion | patch=25, stride=15 | OK | 80.6 | `visualization/gap/rubsheet_square/occ_p25_s15` |

**Completed**: 24/24 succeeded, 0 failed

---

## Grid Search Configuration

### Integrated Gradients

| Experiment | n_steps values |
|-----------|----------------|
| Fisheye | 15, 20, 40, 80 |
| Rubsheet 3-to-1 | 15, 20, 40, 80 |
| Rubsheet Square | 15, 20, 40, 80 |

### Occlusion (full cross-product: patch_size x stride)

| Experiment | patch_sizes | strides | combos (stride <= patch) |
|-----------|-------------|---------|--------------------------|
| Fisheye | 15, 25 | 12, 15 | 4 |
| Rubsheet 3-to-1 | 15, 25 | 12, 15 | 4 |
| Rubsheet Square | 15, 25 | 12, 15 | 4 |
