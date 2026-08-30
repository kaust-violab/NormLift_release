"""
Train ONLY the geometry and color of 3DGS on a ScanNet scene, reusing
OpenGaussian training machinery. No instance-feature learning, no codebook
quantization.

Strategy: delegate to OpenGaussian's `training()` and push every stage threshold
beyond `--iterations`, so execution never leaves Stage 0 (plain 3DGS RGB loss).
Combined with `--frozen_init_pts`, the Gaussian count stays equal to the input
point cloud (no densification, no pruning) — this 1:1 correspondence with the
GT labels.ply is what eval_scannet.py relies on.

Usage:
    OPENGAUSSIAN_DIR=/path/to/OpenGaussian \\
    python train_rgb.py --scene scene0200_00 --data-root <dir> --out-root <dir>

Scene layout (ScanNet-style): <data-root>/<scene>/points3d.ply + transforms_{train,test}.json
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

_RELEASE_ROOT = Path(__file__).resolve().parents[1]

# ─── Pre-parse our own args (before OpenGaussian's parser sees argv) ─
_pre = ArgumentParser(add_help=False)
_pre.add_argument("--scene", default=os.environ.get("NORMLIFT_SCENE", "scene0200_00"))
_pre.add_argument("--data-root", default=os.environ.get("NORMLIFT_DATA", str(_RELEASE_ROOT / "data")))
_pre.add_argument("--out-root",  default=os.environ.get("NORMLIFT_OUT",  str(_RELEASE_ROOT / "outputs")))
_pre_args, _rest_argv = _pre.parse_known_args()

# Resolve all paths BEFORE os.chdir below.
SCENE_DIR  = str(Path(_pre_args.data_root).resolve() / _pre_args.scene)
OUTPUT_DIR = str(Path(_pre_args.out_root).resolve() / _pre_args.scene)
ITERATIONS = 30_000
RESOLUTION = 2          # -r 2, ScanNet-style (half res)

# ─── Make OpenGaussian importable ────────────────────────────────────
OPENGS_DIR = str(Path(os.environ.get("OPENGAUSSIAN_DIR", "OpenGaussian")).resolve())
sys.path.insert(0, OPENGS_DIR)
# Relative imports inside OpenGaussian (utils, gaussian_renderer, scene, …)
# need cwd == OPENGS_DIR to resolve.
os.chdir(OPENGS_DIR)

from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training
from utils.general_utils import safe_state
from gaussian_renderer import network_gui


def parse_args():
    parser = ArgumentParser(description="Geometry+color only training (no quantization)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip",              type=str, default="127.0.0.1")
    parser.add_argument("--port",            type=int, default=6009)
    parser.add_argument("--debug_from",      type=int, default=-1)
    parser.add_argument("--detect_anomaly",  action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ITERATIONS])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ITERATIONS])
    parser.add_argument("--quiet",           action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # Construct synthetic argv forcing our Stage-0-only config.
    forced_argv = [
        "-s", SCENE_DIR,
        "-m", OUTPUT_DIR,
        "-r", str(RESOLUTION),
        "--iterations",          str(ITERATIONS),
        "--frozen_init_pts",
        # Push every OpenGaussian stage gate past --iterations so Stage 0 runs to the end.
        "--start_ins_feat_iter", str(ITERATIONS + 1),
        "--start_root_cb_iter",  str(ITERATIONS + 2),
        "--start_leaf_cb_iter",  str(ITERATIONS + 3),
        "--eval",
    ] + _rest_argv

    args = parser.parse_args(forced_argv)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    return args, lp, op, pp


def main():
    args, lp, op, pp = parse_args()

    print(f"Scene:   {args.source_path}")
    print(f"Output:  {args.model_path}")
    print(f"Iterations: {args.iterations} (Stage 0 only; frozen init points)")
    sys.stdout.flush()

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)

    import torch
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
