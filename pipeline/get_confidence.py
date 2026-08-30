"""
按 readme.md 的定义算 per-Gaussian confidence:

    c_i = ||f_i||  *  N_eff(i) / (N_eff(i) + β),     β = 1

直观:
    N_eff -> 0   : gate -> 0     (没看到 / 单视角主导被压到 0)
    N_eff = 1    : gate = 0.5    (单视角)
    N_eff = 2    : gate ≈ 0.667
    N_eff = 5    : gate ≈ 0.833
    N_eff -> ∞   : gate -> 1     (真多视角时 confidence -> ||f_i||)

输入:
    <NORMLIFT_OUT>/<scene>/eff_view/eff_view.pt          -> N_eff   (N,) float
    <NORMLIFT_OUT>/<scene>/ckpts/point_cloud_features.pt -> features (N, 512), 取 norm
输出:
    <NORMLIFT_OUT>/<scene>/confidence/confidence.pt      -> (N,) float32
"""

import os
from pathlib import Path

import torch


# ─── Paths (run_pipeline.py 会按场景覆写; 单独运行时用下面的默认值) ──
_OUT   = Path(os.environ.get("NORMLIFT_OUT", Path(__file__).resolve().parents[1] / "outputs"))
_SCENE = os.environ.get("NORMLIFT_SCENE", "scene0140_00")

EFF_PATH   = str(_OUT / _SCENE / "eff_view" / "eff_view.pt")
FEAT_PATH  = str(_OUT / _SCENE / "ckpts" / "point_cloud_features.pt")
OUTPUT_DIR = _OUT / _SCENE / "confidence"

BETA = 1.0


def main():
    print(f"\n=== Computing confidence  c_i = ||f_i|| * N_eff / (N_eff + β),  β = {BETA} ===")

    # (1) 加载 N_eff
    n_eff = torch.load(EFF_PATH, map_location="cpu").float()           # (N,)
    print(f"  Loaded N_eff   : shape={tuple(n_eff.shape)}  "
          f"min={n_eff.min().item():.3f}  max={n_eff.max().item():.3f}  "
          f"mean={n_eff.mean().item():.3f}")

    # (2) 加载特征 -> 取模长
    feats = torch.load(FEAT_PATH, map_location="cpu").float()          # (N, 512)
    feat_norms = feats.norm(dim=1)                                     # (N,)
    print(f"  Loaded ||f||   : shape={tuple(feat_norms.shape)}  "
          f"min={feat_norms.min().item():.4f}  max={feat_norms.max().item():.4f}  "
          f"mean={feat_norms.mean().item():.4f}")

    if n_eff.shape[0] != feat_norms.shape[0]:
        raise RuntimeError(
            f"shape mismatch: N_eff N ({n_eff.shape[0]})  "
            f"vs  feat N ({feat_norms.shape[0]})"
        )

    # (3) confidence
    gate       = n_eff / (n_eff + BETA)                                # ∈ [0, 1)
    confidence = feat_norms * gate                                     # (N,)

    # ─── 统计 ────────────────────────────────────────────────────
    print(f"\n  ── Confidence stats ──────────────────────────")
    print(f"    mean={confidence.mean().item():.4f}  "
          f"median={confidence.median().item():.4f}  "
          f"min={confidence.min().item():.4f}  "
          f"max={confidence.max().item():.4f}")

    qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    qv = torch.quantile(confidence, qs)
    print(f"  Confidence percentiles:")
    for q, v in zip(qs.tolist(), qv.tolist()):
        print(f"    p{int(q*100):>2d}  =  {v:.4f}")

    bands = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7),
             (0.7, 0.9), (0.9, float("inf"))]
    M = confidence.numel()
    print(f"\n  Confidence buckets:")
    for lo, hi in bands:
        m = (confidence >= lo) & (confidence < hi)
        n = int(m.sum())
        print(f"    [{lo:>4.1f}, {hi:>4.1f}) :  {n:>8d}  ({n / M * 100:5.2f}%)")

    # gate 与 norm 各自的统计 (帮助看是哪一项压低了 confidence)
    print(f"\n  ── Component breakdown ────────────────────────")
    print(f"    gate = N_eff/(N_eff+β):  mean={gate.mean().item():.4f}  "
          f"median={gate.median().item():.4f}")
    n_low_eff  = int((n_eff < 1.5).sum())
    n_zero_eff = int((n_eff == 0).sum())
    n_zero_f   = int((feat_norms == 0).sum())
    print(f"    N_eff == 0     (uncovered)        : {n_zero_eff}  "
          f"({n_zero_eff / M * 100:.2f}%)")
    print(f"    N_eff <  1.5   (single-view-ish)  : {n_low_eff}  "
          f"({n_low_eff / M * 100:.2f}%)")
    print(f"    ||f|| == 0                        : {n_zero_f}  "
          f"({n_zero_f / M * 100:.2f}%)")

    # ─── Save ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "confidence.pt"
    torch.save(confidence.cpu(), out_path)
    print(f"\nSaved confidence to: {out_path}   "
          f"shape={tuple(confidence.shape)}  dtype={confidence.dtype}")


if __name__ == "__main__":
    main()
