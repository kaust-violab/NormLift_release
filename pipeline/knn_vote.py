"""
Reliability-guided KNN mode voting (Sec. 3.4 of the paper).

Refines lifted features in feature space without averaging: a low-reliability
Gaussian copies the feature of its best-supported neighbour instead of blending
neighbours or being replaced by a text embedding. The step is class-agnostic;
text features are only used for the before/after diagnostics printed at the end.

Support score (Eq. 9), for target i and candidate j in N_K(i) ∪ {i}:
    S_ij = R(j) * Σ_{k ∈ N_K(i) ∪ {i}} R(k) * d_ik * g_jk
    R(j)  reliability of candidate j                  (confidence/confidence.pt)
    d_ik  spatial decay   exp(-||μ_i - μ_k||² / (2 σ_d²))
    g_jk  semantic gate   sigmoid((cos(f_j, f_k) - τ) / γ)

Conservative replacement (Eq. 10), with j* = argmax_{j ∈ N_K(i)} S_ij:
    f_i ← f_{j*}   iff   S_ij* > S_ii + Δ,   otherwise f_i is kept.

Every Gaussian is a target; reliability enters only through the weights R(j), R(k).
Gaussians with f_i = 0 (R(i) = 0, never observed) are no exception: S_ii = 0, so
they take the best-supported neighbour whenever it clears the margin (Appendix D).

Outputs:
    <NORMLIFT_OUT>/<scene>/knn_vote/voted_features.pt  # (N, D), FEAT_PATH for eval_scannet.py
    <NORMLIFT_OUT>/<scene>/knn_vote/voted_labels.pt    # (N,) int64, diagnostic text-argmax labels
The 15/10-class protocols re-evaluate this same output with another TARGET_ID.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import faiss


# ─── Paths (run_pipeline.py 会按场景覆写; 单独运行时用下面的默认值) ──
_DATA  = Path(os.environ.get("NORMLIFT_DATA", Path(__file__).resolve().parents[1] / "data"))
_OUT   = Path(os.environ.get("NORMLIFT_OUT",  Path(__file__).resolve().parents[1] / "outputs"))
_SCENE = os.environ.get("NORMLIFT_SCENE", "scene0140_00")

SCENE_DIR = str(_DATA / _SCENE)

CKPT_PATH = str(_OUT / _SCENE / "point_cloud" / "iteration_30000" / "point_cloud.pt")
FEAT_PATH = str(_OUT / _SCENE / "ckpts" / "point_cloud_features.pt")
CONF_PATH = str(_OUT / _SCENE / "confidence" / "confidence.pt")
TEXT_FEAT = str(Path(os.environ.get(
    "NORMLIFT_TEXT_FEAT",
    Path(__file__).resolve().parents[1] / "assets" / "text_features.json")))
OUT_DIR   = _OUT / _SCENE / "knn_vote"

# ─── Hyperparameters ─────────────────────────────────────────────────
K            = 36     # spatial KNN size
SIGMA_D      = 0.3    # spatial bandwidth (m); d_ik = exp(-||Δμ||² / (2σ_d²))
TAU_COS      = 0.55   # semantic-agreement threshold τ
GAMMA        = 0.1    # gate sharpness γ
DELTA        = 0.05   # margin Δ: replace only if S_ij* > S_ii + Δ
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE   = 4096   # GPU 每批处理的目标高斯数 (控制显存)


# ─── Labels (compact 1..19) — 仅诊断 ────────────────────────────────
nyu40_dict = {
    0: "unlabeled",      1: "wall",           2: "floor",         3: "cabinet",
    4: "bed",            5: "chair",          6: "sofa",          7: "table",
    8: "door",           9: "window",        10: "bookshelf",    11: "picture",
   12: "counter",       13: "blinds",        14: "desk",         15: "shelves",
   16: "curtain",       17: "dresser",       18: "pillow",       19: "mirror",
   20: "floormat",      21: "clothes",       22: "ceiling",      23: "books",
   24: "refrigerator",  25: "television",    26: "paper",        27: "towel",
   28: "shower curtain",29: "box",           30: "whiteboard",   31: "person",
   32: "nightstand",    33: "toilet",        34: "sink",         35: "lamp",
   36: "bathtub",       37: "bag",           38: "otherstructure",
   39: "otherfurniture",40: "otherprop",
}
TARGET_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
NUM_CLASSES = len(TARGET_ID)


@torch.no_grad()
def main():
    print(f"\n=== Confidence-Guided Semantic Mode Voting ===")
    print(f"  K={K}  σ_d={SIGMA_D}m  τ={TAU_COS}  γ={GAMMA}  Δ={DELTA}")

    # ─── Load ────────────────────────────────────────────────────────
    print("\nLoading data ...")
    ckpt     = torch.load(CKPT_PATH, map_location="cpu")
    means_np = ckpt["splats"]["means"].float().numpy().astype(np.float32)   # (N, 3)
    feats    = torch.load(FEAT_PATH, map_location="cpu").float()            # (N, D)
    conf     = torch.load(CONF_PATH, map_location="cpu").float()            # (N,)
    N = means_np.shape[0]
    D = feats.shape[1]
    if feats.shape[0] != N or conf.shape[0] != N:
        raise RuntimeError(f"shape mismatch: means={N}, feats={feats.shape[0]}, conf={conf.shape[0]}")
    print(f"  N={N}  D={D}")

    # ─── Step 1: KNN graph ───────────────────────────────────────────
    print(f"\nStep 1: KNN graph (K={K}) ...")
    index = faiss.IndexFlatL2(3)
    index.add(means_np)
    D_sq, knn_idx_np = index.search(means_np, K + 1)        # (N, K+1) 第一列是自己
    knn_idx_np = knn_idx_np[:, 1:]                           # (N, K)  排除自己
    knn_dsq_np = D_sq[:, 1:]                                 # (N, K)  squared L2
    knn_dist   = np.sqrt(np.clip(knn_dsq_np, 0, None))
    print(f"  KNN distance: mean={knn_dist.mean():.4f}m  median={float(np.median(knn_dist)):.4f}m  max={knn_dist.max():.4f}m")
    avg_dw = float(np.mean(np.exp(-knn_dsq_np / (2 * SIGMA_D ** 2))))
    print(f"  avg dist-decay weight at σ_d={SIGMA_D}: {avg_dw:.4f}")

    # ─── Step 2: 上 GPU ──────────────────────────────────────────────
    device = DEVICE
    feats_n_dev = F.normalize(feats, dim=1, p=2).to(device)                     # (N, D)
    conf_dev    = conf.to(device)                                               # (N,)
    knn_idx_dev = torch.from_numpy(knn_idx_np).long().to(device)                # (N, K)
    knn_dw_np   = np.exp(-knn_dsq_np / (2 * SIGMA_D ** 2)).astype(np.float32)
    knn_dw_dev  = torch.from_numpy(knn_dw_np).to(device)                        # (N, K)

    # ─── Step 3: 分块跑 mode voting (every Gaussian is a target) ─────
    print(f"\nStep 3: chunked mode voting (chunk={CHUNK_SIZE}) ...")
    n_to_vote        = N
    target_idx_all   = torch.arange(N, device=device)                           # (N,)
    new_feats        = feats.clone()                                            # (N, D) on cpu
    n_replaced       = 0

    for start in range(0, n_to_vote, CHUNK_SIZE):
        end           = min(start + CHUNK_SIZE, n_to_vote)
        chunk_targets = target_idx_all[start:end]                                # (M,)
        M             = chunk_targets.shape[0]

        # 扩展邻域索引: [self, knn_1, ..., knn_K]   (M, K+1)
        ext_idx     = torch.cat([
            chunk_targets.unsqueeze(1),                                          # (M, 1)
            knn_idx_dev[chunk_targets],                                          # (M, K)
        ], dim=1)                                                                # (M, K+1)

        ext_feats_n = feats_n_dev[ext_idx]                                       # (M, K+1, D) normalized
        ext_conf    = conf_dev[ext_idx]                                          # (M, K+1)
        # d_ik: k=0 是自己 (dist=0 → weight=1), k=1..K 是 KNN
        ext_dw      = torch.cat([
            torch.ones(M, 1, device=device, dtype=knn_dw_dev.dtype),
            knn_dw_dev[chunk_targets],                                           # (M, K)
        ], dim=1)                                                                # (M, K+1)

        # 邻域内两两 cosine 相似度
        cos_mat = torch.bmm(ext_feats_n, ext_feats_n.transpose(1, 2))            # (M, K+1, K+1)
        # 软语义一致性 g_jk
        g_jk    = torch.sigmoid((cos_mat - TAU_COS) / GAMMA)                     # (M, K+1, K+1)

        # 支持者权重: r_k * d_ik
        weight_k    = ext_conf * ext_dw                                          # (M, K+1)
        # weighted_g[m, j] = Σ_k weight_k[m, k] * g_jk[m, j, k]
        weighted_g  = (g_jk * weight_k.unsqueeze(1)).sum(dim=-1)                 # (M, K+1)
        # S_ij = r_j * weighted_g[m, j]
        S           = ext_conf * weighted_g                                      # (M, K+1)

        # 拆出 self-candidate 与邻居候选
        S_ii                  = S[:, 0]                                          # (M,) self
        S_neighbors           = S[:, 1:]                                         # (M, K)
        max_S, argmax_in_nbrs = S_neighbors.max(dim=1)                           # (M,) (M,)
        # 对应到全局索引: ext_idx[m, argmax_in_nbrs[m] + 1]
        chosen_global_idx = ext_idx.gather(
            1, (argmax_in_nbrs + 1).unsqueeze(1)
        ).squeeze(1)                                                             # (M,)

        # 保守替换: max_S > S_ii + Δ
        score_pass   = max_S > (S_ii + DELTA)                                    # (M,)

        replace_mask = score_pass                                                # (M,)

        if int(replace_mask.sum()) > 0:
            replace_targets = chunk_targets[replace_mask].cpu()                  # (M_repl,)
            replace_sources = chosen_global_idx[replace_mask].cpu()              # (M_repl,)
            # 用未归一化的原 feature 直接复制 (cos sim 看方向, 模长无所谓)
            new_feats[replace_targets] = feats[replace_sources]
            n_replaced += int(replace_mask.sum())

    print(f"\n  replaced (mode voted):                {n_replaced:>7d}  ({n_replaced/N*100:5.2f}%)")
    n_kept = N - n_replaced
    print(f"  kept (no consensus margin):           {n_kept:>7d}  ({n_kept/N*100:5.2f}%)")

    # ─── Step 4: 诊断 — 类分布前后对比 (cos sim with text features) ──
    print(f"\nStep 4: class distribution before/after (cos-argmax with text) ...")
    target_names = [nyu40_dict[i] for i in TARGET_ID]
    with open(TEXT_FEAT) as fp:
        text_data = json.load(fp)
    qf = torch.zeros(NUM_CLASSES, D)
    missing = []
    for i, name in enumerate(target_names):
        if name in text_data:
            qf[i] = torch.tensor(text_data[name], dtype=torch.float32)
        else:
            missing.append(name)
    if missing:
        print(f"  WARNING: no text feature for: {missing}")

    qf_n = F.normalize(qf, dim=1, p=2)
    def predict(f):
        fn   = F.normalize(f, dim=1, p=2)
        sim  = fn @ qf_n.T
        pred = sim.argmax(dim=1) + 1
        pred[f.norm(dim=1) == 0] = 0
        return pred.numpy()

    pred_old = predict(feats)
    pred_new = predict(new_feats)
    n_changed_pred = int((pred_old != pred_new).sum())
    print(f"  predictions changed: {n_changed_pred}/{N}  ({n_changed_pred/N*100:.2f}%)")

    print(f"\n  {'class':<24s} {'baseline':>10s} {'mode_voted':>12s} {'Δ':>10s}")
    for c in range(NUM_CLASSES + 1):
        n_old = int((pred_old == c).sum())
        n_new = int((pred_new == c).sum())
        if n_old == 0 and n_new == 0:
            continue
        cname = "unlabeled" if c == 0 else nyu40_dict.get(TARGET_ID[c-1], "?")
        delta = n_new - n_old
        print(f"  [{c:>2d}] {cname:<20s} {n_old:>10d} {n_new:>12d} "
              f"{('+' if delta >= 0 else '') + str(delta):>10s}")

    # ─── Save ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_out  = OUT_DIR / "voted_features.pt"
    label_out = OUT_DIR / "voted_labels.pt"
    torch.save(new_feats, feat_out)
    torch.save(torch.from_numpy(pred_new).long(), label_out)
    print(f"\nSaved features: {feat_out}   shape={tuple(new_feats.shape)}  dtype={new_feats.dtype}")
    print(f"Saved labels  : {label_out}   shape={tuple(pred_new.shape)}")
    print(f"\n下一步: 把 eval_scannet.py 的 FEAT_PATH 指向 {feat_out} 跑 eval "
          f"(run_pipeline.py 的 eval step 会自动做这件事).")


if __name__ == "__main__":
    main()
