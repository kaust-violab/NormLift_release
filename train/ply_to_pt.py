"""
将 INRIA-style 3DGS ply (OpenGaussian 保存的 point_cloud.ply) 转换为
splat-distiller/gsplat 风格的 .pt 文件，以便被 GaussianPrimitive.from_file
直接加载。

对照参考:
  - INRIA ply 字段定义: OpenGaussian/scene/gaussian_model.py::save_ply
  - splat-distiller .pt schema: gaussian_splatting/simple_trainer.py (line 277)
      self.splats.state_dict() = {"means","scales","quats","opacities","sh0","shN"}
  - .pt 加载逻辑: gsplat_ext.utils.primitives._from_ckpt
      means  : 原值 (not transformed)
      scales : 原值 (log scale), 加载时 exp()
      quats  : 原值, 加载时 F.normalize
      opacities: 原值 (logit), 加载时 sigmoid
      sh0    : (N, 1,   3)
      shN    : (N, K-1, 3)      其中 K = (sh_degree + 1)^2

用法:
    python ply_to_pt.py --scene scene0200_00 --out-root <dir>
    python ply_to_pt.py --ply <path/to/point_cloud.ply>          # 显式指定
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData


_RELEASE_ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description="Convert INRIA-style 3DGS ply to gsplat .pt")
_ap.add_argument("--scene", default=os.environ.get("NORMLIFT_SCENE", "scene0200_00"))
_ap.add_argument("--out-root", type=Path,
                 default=Path(os.environ.get("NORMLIFT_OUT", _RELEASE_ROOT / "outputs")))
_ap.add_argument("--ply", type=Path, default=None, help="explicit ply path (overrides --scene)")
_ap.add_argument("--out", type=Path, default=None, help="explicit output .pt path")
_args = _ap.parse_args()

PLY_PATH = _args.ply or (_args.out_root / _args.scene
                         / "point_cloud" / "iteration_30000" / "point_cloud.ply")
OUT_PATH = _args.out or PLY_PATH.with_suffix(".pt")


def main():
    assert PLY_PATH.exists(), f"ply not found: {PLY_PATH}"

    ply = PlyData.read(str(PLY_PATH))
    verts = ply.elements[0]
    names = {p.name for p in verts.properties}
    N = verts.count
    print(f"Loaded {PLY_PATH}  ({N} points)")

    # ── xyz ──────────────────────────────────────────────────────────
    xyz = np.stack([np.asarray(verts["x"]),
                    np.asarray(verts["y"]),
                    np.asarray(verts["z"])], axis=1).astype(np.float32)

    # ── opacity (raw logit) ──────────────────────────────────────────
    opacities = np.asarray(verts["opacity"]).astype(np.float32)          # (N,)

    # ── scales (raw log-scale) ───────────────────────────────────────
    scale_names = sorted([n for n in names if n.startswith("scale_")],
                         key=lambda s: int(s.split("_")[-1]))
    scales = np.stack([np.asarray(verts[n]) for n in scale_names],
                      axis=1).astype(np.float32)                         # (N, 3)

    # ── rotations (raw quats, normalized on load) ────────────────────
    rot_names = sorted([n for n in names if n.startswith("rot_") or n == "rot"],
                       key=lambda s: int(s.split("_")[-1]) if "_" in s else 0)
    # Fall back to strict "rot_0..3" if the above grabbed something odd
    if not all(n.startswith("rot_") for n in rot_names):
        rot_names = sorted([n for n in names if n.startswith("rot_")],
                           key=lambda s: int(s.split("_")[-1]))
    rots = np.stack([np.asarray(verts[n]) for n in rot_names],
                    axis=1).astype(np.float32)                           # (N, 4)

    # ── SH: DC + rest ────────────────────────────────────────────────
    f_dc = np.stack([np.asarray(verts["f_dc_0"]),
                     np.asarray(verts["f_dc_1"]),
                     np.asarray(verts["f_dc_2"])], axis=1).astype(np.float32)  # (N, 3)

    rest_names = sorted([n for n in names if n.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))
    n_rest = len(rest_names)
    if n_rest == 0:
        sh_degree = 0
    else:
        # INRIA layout: 3 channels × (K-1) coeffs flattened as [ch0_c0..ch0_c{K-2}, ch1..., ch2...]
        # total = 3*(K-1), so K = n_rest/3 + 1, sh_degree = sqrt(K)-1
        assert n_rest % 3 == 0, f"f_rest_* count {n_rest} not divisible by 3"
        K_minus_1 = n_rest // 3
        K = K_minus_1 + 1
        sh_degree = int(np.sqrt(K)) - 1
        assert (sh_degree + 1) ** 2 == K, f"bad sh_degree inference: K={K}"

    # sh0 : (N, 1, 3)
    sh0 = f_dc[:, None, :]

    # shN : (N, K-1, 3)
    if n_rest > 0:
        rest = np.stack([np.asarray(verts[n]) for n in rest_names],
                        axis=1).astype(np.float32)                       # (N, 3*(K-1))
        # INRIA order: [ch, coef] flattened → reshape (N, 3, K-1) then transpose to (N, K-1, 3)
        rest = rest.reshape(N, 3, K - 1).transpose(0, 2, 1)              # (N, K-1, 3)
        shN = np.ascontiguousarray(rest)
    else:
        shN = np.zeros((N, 0, 3), dtype=np.float32)

    # ── build the ParameterDict-style state dict ─────────────────────
    splats = {
        "means":     torch.from_numpy(xyz),
        "scales":    torch.from_numpy(scales),
        "quats":     torch.from_numpy(rots),
        "opacities": torch.from_numpy(opacities),
        "sh0":       torch.from_numpy(sh0),
        "shN":       torch.from_numpy(shN),
    }

    print(f"sh_degree (inferred): {sh_degree}")
    for k, v in splats.items():
        print(f"  {k:10s}: {tuple(v.shape)}  {v.dtype}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"splats": splats, "step": 30000}, OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
