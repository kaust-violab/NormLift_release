"""
Sequentially runs the 5-step post-training pipeline for a list of scenes.

Pipeline (per scene):
    1. get_see_count.py      -- per-view visibility count
    2. get_eff_view.py       -- effective view count N_eff
    3. get_confidence.py     -- confidence c_i = ||f_i|| * N_eff/(N_eff+beta)
    4. knn_vote.py           -- confidence-guided semantic mode voting
    5. eval_scannet.py       -- ScanNet semantic segmentation eval (mIoU / mAcc)

Implementation strategy: import each script as a module, patch its path
constants per scene, call main(), then GC. This avoids edits to the
underlying scripts.

评测协议 (--classes): KNN 投票只跑一次, 固定在完整的 19 类标签空间;
--classes 只决定 eval 用哪套 TARGET_ID (论文口径). 15/10 类协议复用
19 类的投票产物, 只需 --steps eval.

Directory conventions:
    DATA_ROOT/<scene>/           ScanNet-style scene (points3d.ply, transforms_*.json,
                                 <scene>_vh_clean_2.labels.ply, language_features/)
    DATA_ROOT/<scene>_colmap/    COLMAP-format copy (preprocess/convert_to_colmap.py)
    OUT_ROOT/<scene>/            training / lifting / pipeline outputs

Usage:
    python run_pipeline.py --data-root <dir> --out-root <dir> --classes 19
    python run_pipeline.py --scenes scene0000_00 scene0062_00 --steps knn_vote eval
"""

import argparse
import gc
import importlib
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = PIPELINE_DIR.parent

# pipeline dir for sibling step modules; splat-distiller root (optional) for
# gsplat_ext / gaussian_splatting when they are not pip-installed.
for p in (str(PIPELINE_DIR), os.environ.get("SPLAT_DISTILLER_DIR")):
    if p and p not in sys.path:
        sys.path.insert(0, p)

ALL_SCENES = [
    "scene0000_00", "scene0062_00", "scene0070_00", "scene0097_00",
    "scene0140_00", "scene0200_00", "scene0347_00", "scene0400_00",
    "scene0590_00", "scene0645_00",
]
ALL_STEPS = ["see_count", "eff_view", "confidence", "knn_vote", "eval"]


def parse_args():
    ap = argparse.ArgumentParser(description="NormLift 5-step post-training pipeline")
    ap.add_argument("--data-root", type=Path,
                    default=Path(os.environ.get("NORMLIFT_DATA", RELEASE_ROOT / "data")),
                    help="dir containing <scene>/ and <scene>_colmap/")
    ap.add_argument("--out-root", type=Path,
                    default=Path(os.environ.get("NORMLIFT_OUT", RELEASE_ROOT / "outputs")),
                    help="dir containing per-scene training/lifting outputs")
    ap.add_argument("--text-feat", type=Path,
                    default=Path(os.environ.get("NORMLIFT_TEXT_FEAT",
                                                RELEASE_ROOT / "assets" / "text_features.json")),
                    help="CLIP text features json (NYU40 class names)")
    ap.add_argument("--classes", type=int, choices=[19, 15, 10], default=19,
                    help="eval protocol (TARGET_ID set); voting always runs in the full 19-class space")
    ap.add_argument("--scenes", nargs="+", default=ALL_SCENES)
    ap.add_argument("--steps", nargs="+", default=ALL_STEPS, choices=ALL_STEPS)
    return ap.parse_args()


def cuda_cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def step_see_count(paths, args):
    mod = importlib.import_module("get_see_count")
    mod.SCENE_DIR  = str(paths["scene_colmap"])
    mod.CKPT_PATH  = str(paths["ckpt"])
    mod.OUTPUT_DIR = paths["see_count_dir"]
    mod.main()


def step_eff_view(paths, args):
    mod = importlib.import_module("get_eff_view")
    mod.SCENE_DIR      = str(paths["scene_colmap"])
    mod.CKPT_PATH      = str(paths["ckpt"])
    mod.SEE_COUNT_PATH = str(paths["see_count_pt"])
    mod.OUTPUT_DIR     = paths["eff_view_dir"]
    mod.main()


def step_confidence(paths, args):
    mod = importlib.import_module("get_confidence")
    mod.EFF_PATH   = str(paths["eff_view_pt"])
    mod.FEAT_PATH  = str(paths["feat"])
    mod.OUTPUT_DIR = paths["confidence_dir"]
    mod.main()


def step_knn_vote(paths, args):
    mod = importlib.import_module("knn_vote")
    mod.SCENE_DIR = str(paths["scene"])
    mod.CKPT_PATH = str(paths["ckpt"])
    mod.FEAT_PATH = str(paths["feat"])
    mod.CONF_PATH = str(paths["confidence_pt"])
    mod.TEXT_FEAT = str(args.text_feat)
    mod.OUT_DIR   = paths["knn_vote_dir"]
    mod.main()


def step_eval(paths, args):
    mod = importlib.import_module("eval_scannet")
    mod.SCENE_DIR = str(paths["scene"])
    mod.GT_PLY    = str(paths["gt_ply"])
    mod.CKPT_PATH = str(paths["ckpt"])
    mod.FEAT_PATH = str(paths["voted_feat"])
    mod.TEXT_FEAT = str(args.text_feat)
    mod.TARGET_ID = mod.TARGET_ID_SETS[args.classes]
    return mod.main()


STEP_RUNNERS = {
    "see_count":  (step_see_count,  ["ckpt", "scene_colmap"]),
    "eff_view":   (step_eff_view,   ["ckpt", "scene_colmap", "see_count_pt"]),
    "confidence": (step_confidence, ["eff_view_pt", "feat"]),
    "knn_vote":   (step_knn_vote,   ["ckpt", "feat", "confidence_pt", "scene"]),
    "eval":       (step_eval,       ["ckpt", "voted_feat", "gt_ply", "scene"]),
}


def build_paths(scene_name: str, args) -> dict:
    out_root = args.out_root / scene_name
    vote_dir = out_root / "knn_vote"
    return {
        "scene":          args.data_root / scene_name,
        "scene_colmap":   args.data_root / f"{scene_name}_colmap",
        "gt_ply":         args.data_root / scene_name / f"{scene_name}_vh_clean_2.labels.ply",
        "out_root":       out_root,
        "ckpt":           out_root / "point_cloud" / "iteration_30000" / "point_cloud.pt",
        "feat":           out_root / "ckpts"       / "point_cloud_features.pt",
        "see_count_dir":  out_root / "see_count",
        "see_count_pt":   out_root / "see_count"   / "see_count.pt",
        "eff_view_dir":   out_root / "eff_view",
        "eff_view_pt":    out_root / "eff_view"    / "eff_view.pt",
        "confidence_dir": out_root / "confidence",
        "confidence_pt":  out_root / "confidence"  / "confidence.pt",
        "knn_vote_dir":   vote_dir,
        "voted_feat":     vote_dir / "voted_features.pt",
    }


def run_step(step: str, scene_name: str, paths: dict, args):
    runner, required_keys = STEP_RUNNERS[step]
    missing = [k for k in required_keys if not Path(paths[k]).exists()]
    if missing:
        print(f"  [SKIP {scene_name}/{step}] missing inputs: "
              + ", ".join(f"{k}={paths[k]}" for k in missing))
        return None

    print(f"\n  ── [{scene_name}] step: {step} ─────────────────────────")
    try:
        return runner(paths, args)
    except Exception:
        print(f"  [FAIL {scene_name}/{step}] exception below — continuing.")
        traceback.print_exc()
        return None
    finally:
        cuda_cleanup()


def main():
    args = parse_args()
    print(f"== NormLift pipeline started at {datetime.now():%Y-%m-%d %H:%M:%S} ==")
    print(f"  data root : {args.data_root}")
    print(f"  out root  : {args.out_root}")
    print(f"  protocol  : {args.classes}-class")
    print(f"  scenes    : {args.scenes}")
    print(f"  steps     : {args.steps}")

    results = []
    for scene_name in args.scenes:
        print(f"\n========================================================")
        print(f"  Scene: {scene_name}")
        print(f"========================================================")

        paths = build_paths(scene_name, args)
        if not paths["scene"].is_dir():
            print(f"  [SKIP] scene dir missing: {paths['scene']}")
            continue

        for step in args.steps:
            ret = run_step(step, scene_name, paths, args)
            if step == "eval" and ret is not None:
                results.append(ret)

    if results:
        print(f"\n========== Summary ({args.classes}-class) ==========")
        print(f"  {'scene':<16s} {'mIoU':>8s} {'mAcc':>8s} {'Acc':>8s}")
        for r in results:
            print(f"  {r['scene']:<16s} {r['mIoU']:>8.4f} {r['mAcc']:>8.4f} {r['Acc']:>8.4f}")
        n = len(results)
        print(f"\n  mean over {n} scenes: "
              f"mIoU={sum(r['mIoU'] for r in results)/n:.4f}  "
              f"mAcc={sum(r['mAcc'] for r in results)/n:.4f}  "
              f"Acc={sum(r['Acc'] for r in results)/n:.4f}")
    print(f"== Done at {datetime.now():%Y-%m-%d %H:%M:%S} ==")


if __name__ == "__main__":
    main()
