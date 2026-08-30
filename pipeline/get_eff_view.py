"""
把每个高斯的"被看到视角数"替换成"有效视角数" (effective sample size).

定义:
    N_eff(i) = (Σ_s W_i^(s))^2 / Σ_s (W_i^(s))^2

性质:
    - N_eff ∈ [1, see_count]   (任意视角都看不到 -> 设 0)
    - 所有 W^(s) 相等 -> N_eff = see_count
    - 一个视角占绝对主导 -> N_eff -> 1
所以 N_eff 把"名义视角数"修正为"实际有跨视角平均效应"的视角数,
单视角主导的高斯 N_eff ≈ 1, 真正多视角观测的高斯 N_eff ≈ see_count.

参考:
    get_see_count.py    -- per-view "看得到" 计数 (整数)
    whether_domain.py   -- 同时算了 N_eff 但只给子集统计 + 保存 dict

输出:
    <NORMLIFT_OUT>/<scene>/eff_view/eff_view.pt   # (N,) float32
"""

import os
import sys
from pathlib import Path

# gsplat_ext / gaussian_splatting 随 splat-distiller `pip install .` 安装;
# 未安装时可用 SPLAT_DISTILLER_DIR 指向其源码根目录.
_SD = os.environ.get("SPLAT_DISTILLER_DIR")
if _SD and _SD not in sys.path:
    sys.path.insert(0, _SD)

import torch
from tqdm import tqdm

from gsplat_ext import Dataset, Parser, GaussianPrimitive, GaussianRenderer
from gaussian_splatting.utils import set_random_seed


# ─── Paths (run_pipeline.py 会按场景覆写; 单独运行时用下面的默认值) ──
_DATA  = Path(os.environ.get("NORMLIFT_DATA", Path(__file__).resolve().parents[1] / "data"))
_OUT   = Path(os.environ.get("NORMLIFT_OUT",  Path(__file__).resolve().parents[1] / "outputs"))
_SCENE = os.environ.get("NORMLIFT_SCENE", "scene0140_00")

SCENE_DIR      = str(_DATA / f"{_SCENE}_colmap")
CKPT_PATH      = str(_OUT / _SCENE / "point_cloud" / "iteration_30000" / "point_cloud.pt")
SEE_COUNT_PATH = str(_OUT / _SCENE / "see_count" / "see_count.pt")
OUTPUT_DIR     = _OUT / _SCENE / "eff_view"

TIKHONOV  = 1.0
DUMMY_DIM = 1
DEVICE    = "cuda"
EPS       = 1e-12


@torch.no_grad()
def main():
    set_random_seed(42)

    parser = Parser(data_dir=SCENE_DIR, factor=1, test_every=1000)
    trainset = Dataset(parser, split="train", load_features=False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    splats = GaussianPrimitive()
    splats.from_file(CKPT_PATH, tikhonov=TIKHONOV)
    splats.to(DEVICE)
    renderer = GaussianRenderer(splats)

    N = splats.geometry["means"].shape[0]
    n_views = len(train_loader)
    print(f"N gaussians   : {N}")
    print(f"N train views : {n_views}")

    # float64 累加: W^(s) 可能很小, 平方后更小, 防 underflow
    accum_W    = torch.zeros(N, dtype=torch.float64, device=DEVICE)
    accum_W_sq = torch.zeros(N, dtype=torch.float64, device=DEVICE)

    for data in tqdm(train_loader, desc="Per-view accumulation", total=n_views):
        camtoworlds = data["camtoworld"].to(DEVICE)
        Ks          = data["K"].to(DEVICE)
        pixels      = data["image"].to(DEVICE) / 255.0
        height, width = pixels.shape[1:3]

        features = torch.zeros((1, height, width, DUMMY_DIM), device=DEVICE)
        _splat_feats_view, splat_weights_view, ids = renderer.inverse_render(
            K=Ks, extrinsic=camtoworlds, width=width, height=height, features=features,
        )

        ids_t = ids if torch.is_tensor(ids) else torch.as_tensor(ids, device=DEVICE)
        seen  = splat_weights_view > 0
        if not seen.any():
            del _splat_feats_view, splat_weights_view, ids, ids_t, seen, features
            continue

        ids_seen = ids_t[seen]
        w64 = splat_weights_view[seen].to(torch.float64)
        accum_W   [ids_seen] += w64
        accum_W_sq[ids_seen] += w64 * w64

        del _splat_feats_view, splat_weights_view, ids, ids_t, seen, ids_seen, w64, features
        torch.cuda.empty_cache()

    # ─── N_eff ───────────────────────────────────────────────────────
    covered = accum_W > 0
    n_eff = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    safe_sq = accum_W_sq.clamp(min=EPS)
    n_eff_full = (accum_W * accum_W) / safe_sq
    n_eff[covered] = n_eff_full[covered].float()
    # uncovered (accum_W == 0): n_eff 保持为 0

    # ─── 与 see_count.pt 对比 ────────────────────────────────────────
    if Path(SEE_COUNT_PATH).exists():
        see_count = torch.load(SEE_COUNT_PATH, map_location=DEVICE)
        if see_count.shape[0] == N:
            sc = see_count.float()
            ne = n_eff
            both = covered & (see_count > 0)
            ratio = torch.zeros_like(ne)
            ratio[both] = ne[both] / sc[both]

            print(f"\n  ── Comparison vs see_count.pt ──────────────")
            print(f"    covered (accum_W > 0) : {int(covered.sum())}")
            print(f"    see_count > 0         : {int((see_count > 0).sum())}  "
                  f"(应与 covered 一致)")
            print(f"    N_eff       : mean={ne[both].mean().item():.3f}  "
                  f"median={ne[both].median().item():.3f}  "
                  f"max={ne[both].max().item():.3f}")
            print(f"    see_count   : mean={sc[both].mean().item():.3f}  "
                  f"median={sc[both].median().item():.1f}  "
                  f"max={int(sc[both].max())}")
            print(f"    N_eff/see   : mean={ratio[both].mean().item():.3f}  "
                  f"median={ratio[both].median().item():.3f}     "
                  f"(1.0 = 视角间均匀, 越小越偏单视角)")
        else:
            print(f"\n  WARNING: see_count.pt N ({see_count.shape[0]}) != gaussian N ({N}), "
                  f"跳过对比.")
    else:
        print(f"\n  (see_count.pt 不存在, 跳过对比)")

    # ─── 整体 N_eff 分布 ─────────────────────────────────────────────
    ne_covered = n_eff[covered]
    print(f"\n  ── Overall N_eff distribution (covered = {int(covered.sum())}) ──")
    print(f"    mean = {ne_covered.mean().item():.3f}    "
          f"median = {ne_covered.median().item():.3f}    "
          f"max = {ne_covered.max().item():.3f}")
    bands = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0),
             (5.0, 10.0), (10.0, float("inf"))]
    print(f"  N_eff buckets:")
    n_total = int(covered.sum())
    for lo, hi in bands:
        m = (ne_covered >= lo) & (ne_covered < hi)
        n_b = int(m.sum())
        pct = n_b / n_total * 100 if n_total else 0
        print(f"    [{lo:>4.1f}, {hi:>4.1f}) :  {n_b:>8d}  ({pct:5.2f}%)")

    # ─── Save ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "eff_view.pt"
    torch.save(n_eff.cpu(), out_path)
    print(f"\nSaved eff_view to: {out_path}   shape={tuple(n_eff.shape)}  "
          f"dtype={n_eff.dtype}")


if __name__ == "__main__":
    main()
