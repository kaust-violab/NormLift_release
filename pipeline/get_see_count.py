"""
统计每个高斯被多少个训练视角"看到".

沿用特征 lifting (distill_features.py) 的反投影流程, 但只保留 visibility
计数 —— 不需要语言特征 / SAM mask.

判定规则 (与特征聚合口径完全一致):
    lifting 时没有任何 per-view 阈值, 只要 inverse_render 把高斯返回到 `ids` 中
    且 W_i^(s) > 0, 它就被纳入累加. 所以这里同样只看 W_i^(s) > 0 -> 看到一次.
    这样 see_count == 0 当且仅当 lifting 中 covered = (accum_W > 0) 为 False.

输出:
    <NORMLIFT_OUT>/<scene>/see_count/see_count.pt   # (N,) int32

    另存一份累计权重供后续按 cumulative 阈值过滤:
    <NORMLIFT_OUT>/<scene>/see_count/accum_W.pt     # (N,) float32
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

SCENE_DIR  = str(_DATA / f"{_SCENE}_colmap")
CKPT_PATH  = str(_OUT / _SCENE / "point_cloud" / "iteration_30000" / "point_cloud.pt")
OUTPUT_DIR = _OUT / _SCENE / "see_count"

TIKHONOV  = 1.0     # 与 lifting 对齐
DUMMY_DIM = 1       # 我们不关心特征值, 只用占位 0 张量喂给 renderer
DEVICE    = "cuda"


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
    print(f"Number of Gaussians : {N}")
    print(f"Number of train views: {n_views}")
    print(f"Visibility criterion : W_i^(s) > 0  (与 lifting 累加口径一致)")

    seen_count = torch.zeros(N, dtype=torch.int32, device=DEVICE)
    accum_W    = torch.zeros(N, dtype=torch.float32, device=DEVICE)

    for data in tqdm(train_loader, desc="Per-view counting", total=n_views):
        camtoworlds = data["camtoworld"].to(DEVICE)
        Ks          = data["K"].to(DEVICE)
        pixels      = data["image"].to(DEVICE) / 255.0
        height, width = pixels.shape[1:3]

        # 占位 features: 我们只关心 splat_weights_view, 特征值无所谓
        features = torch.zeros((1, height, width, DUMMY_DIM), device=DEVICE)

        _splat_feats_view, splat_weights_view, ids = renderer.inverse_render(
            K=Ks, extrinsic=camtoworlds, width=width, height=height, features=features,
        )
        # splat_weights_view[i] = Σ_u w_i^(s)(u) for the i-th gaussian among `ids`

        seen_mask = splat_weights_view > 0                 # bool (n_in_ids,)
        ids_t = ids if torch.is_tensor(ids) else torch.as_tensor(ids, device=DEVICE)
        seen_ids = ids_t[seen_mask]
        seen_count[seen_ids] += 1
        accum_W[ids_t] += splat_weights_view               # 直接累加, 与 lifting 一致

        del _splat_feats_view, splat_weights_view, ids, ids_t, seen_mask, seen_ids, features
        torch.cuda.empty_cache()

    # ─── Stats ───────────────────────────────────────────────────
    sc = seen_count.float()
    n_unseen = int((seen_count == 0).sum())
    n_once   = int((seen_count == 1).sum())
    print(f"\nProcessed {n_views} train views.")
    print(f"  see_count   mean={sc.mean().item():.2f}  "
          f"median={int(sc.median())}  max={int(seen_count.max())}")
    print(f"  unseen (count == 0) : {n_unseen:>8d}  ({n_unseen / N * 100:5.2f}%)")
    print(f"  seen once           : {n_once:>8d}  ({n_once   / N * 100:5.2f}%)")

    print("\n  bucketed:")
    for lo, hi in [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50),
                   (50, max(51, n_views + 1))]:
        n = int(((seen_count >= lo) & (seen_count < hi)).sum())
        print(f"    [{lo:>3d}, {hi:>4d}):  {n:>8d}  ({n / N * 100:5.2f}%)")

    # ─── 与 lifting 覆盖口径一致性比对 ──────────────────────────
    covered = accum_W > 0
    n_uncovered_seen = int(((seen_count > 0) & ~covered).sum())
    n_uncovered      = int((~covered).sum())
    print(f"\n  (sanity) covered = (accum_W > 0)        : {int(covered.sum())}  "
          f"-- 与 lifting 的 covered 同口径")
    print(f"  (sanity) uncovered (accum_W == 0)        : {n_uncovered}")
    print(f"  (sanity) seen_count > 0 但 uncovered     : {n_uncovered_seen}  "
          f"(应该恒为 0)")

    # ─── Save ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_count = OUTPUT_DIR / "see_count.pt"
    out_accum = OUTPUT_DIR / "accum_W.pt"
    torch.save(seen_count.cpu(), out_count)
    torch.save(accum_W.cpu(),    out_accum)
    print(f"\nSaved see_count to: {out_count}   shape={tuple(seen_count.shape)}  "
          f"dtype={seen_count.dtype}")
    print(f"Saved accum_W   to: {out_accum}   shape={tuple(accum_W.shape)}  "
          f"dtype={accum_W.dtype}")


if __name__ == "__main__":
    main()
