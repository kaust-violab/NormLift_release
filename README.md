# NormLift

Open-vocabulary 3D semantic segmentation on ScanNet: lift SAM+CLIP features onto
3D Gaussians, then refine them with a norm-based confidence measure and
confidence-guided semantic mode voting over spatial KNN neighborhoods.

## Setup

Requires two external repos:

- [Splat Feature Solver / splat-distiller](https://github.com/saliteta/splat-distiller) —
  `pip install .` inside it (provides `gsplat_ext` and `gaussian_splatting`);
  set `SPLAT_DISTILLER_DIR` to its source root.
- [OpenGaussian](https://github.com/yanmin-wu/OpenGaussian) — used for 3DGS training;
  set `OPENGAUSSIAN_DIR` to its root.

Plus `faiss-gpu` and `plyfile`.

## Data

10 ScanNet v2 scenes (`scene0000_00 0062 0070 0097 0140 0200 0347 0400 0590 0645`),
in OpenGaussian layout under `$NORMLIFT_DATA/<scene>/`:
`points3d.ply`, `transforms_{train,test}.json`, `<scene>_vh_clean_2.labels.ply`,
and LangSplat-style features in `language_features/language_features/`
(SAM ViT-H + OpenCLIP, `*_f.npy` / `*_s.npy`).

```sh
# COLMAP-format copy, needed by feature lifting and the visibility steps:
python preprocess/convert_to_colmap.py --src $NORMLIFT_DATA/<scene> --dst $NORMLIFT_DATA/<scene>_colmap
```

## Run

```sh
export NORMLIFT_DATA=/path/to/data  NORMLIFT_OUT=/path/to/outputs
export OPENGAUSSIAN_DIR=/path/to/OpenGaussian  SPLAT_DISTILLER_DIR=/path/to/splat-distiller

# Per scene: train RGB 3DGS (frozen points), convert ckpt, lift features
python train/train_rgb.py --scene scene0200_00
python train/ply_to_pt.py --scene scene0200_00
NORMLIFT_SCENE=scene0200_00 python lift/distill_features.py

# All scenes: post-processing + evaluation
python pipeline/run_pipeline.py --classes 19
python pipeline/run_pipeline.py --classes 15 --steps eval
python pipeline/run_pipeline.py --classes 10 --steps eval
```

The pipeline runs 5 steps per scene — visibility count → effective views →
confidence → KNN vote → eval — and prints per-scene and mean mIoU/mAcc.
Voting is class-agnostic (it operates purely in feature space) and runs once;
`--classes 15/10` re-evaluates the same voting products with a smaller label
set, so those runs only need `--steps eval`.

## Hyperparameters

| Stage | Parameter | Value |
|---|---|---|
| Training | iterations / resolution | 30 000 / `-r 2` |
| Lifting | Tikhonov / SAM level | 1.0 / 3 |
| Confidence | β (view gate) | 1.0 |
| KNN vote | K / τ_r / σ_d / τ / γ / Δ | 36 / 0.75 / 0.3 m / 0.55 / 0.1 / 0.05 |
| Eval | opacity threshold | 0.1 |
