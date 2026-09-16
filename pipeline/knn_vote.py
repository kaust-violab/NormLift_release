"""
Confidence-Guided Semantic Mode Voting (改 feature, 不改 label).

跟"标签级投票"的本质区别:
    label vote:  硬 argmax 1-of-C 分类后按类投票, 再用 text feature 重新生成 feature.
                 封闭词汇假设, 离散化损失大.
    mode vote:   改 feature 方向, 但不取均值, 而是从 "邻域 ∪ {自己}" 候选集里
                 挑出"支持度最高"的某个邻居 feature 直接复制过来.
                 → 避免 CLIP feature 被线性混合 (semantic 不平均);
                 → 避免被 text feature 替换 (开放词汇友好);
                 → 自带保守阈值: 只有当 mode 比"自己"明显强时才替换.

支持度公式 (对目标 Gaussian i, 候选 j ∈ N(i) ∪ {i}):
    S_ij = r_j *  Σ_{k ∈ N(i) ∪ {i}}  r_k * d_ik * g_jk
其中:
    r_j        : 候选 j 的可靠性 (= conf[j])
    r_k        : 支持者 k 的可靠性
    d_ik       : 支持者 k 到目标 i 的距离权重 = exp(-||μ_i - μ_k||² / (2σ_d²))
    g_jk       : 支持者 k 与候选 j 的语义一致性 = sigmoid((cos(f_j, f_k) - τ) / γ)
                 (软阈值: 相似度 < τ 几乎贡献 0; > τ 贡献接近 1)

更新规则:
    1. conf[i] >= TAU_R       → 高置信, 不动
    2. 否则 j* = argmax_{j∈N(i)} S_ij;
       若 S_ij* > S_ii + Δ    → f_i ← f_{j*}
       否则                    → 保持原 f_i  (margin 防止过度修改)

投票是类无关的 (只在 feature 空间进行); text features 仅用于打印前后类分布诊断.
15/10 类协议直接用同一份投票产物、换 eval_scannet.py 的 TARGET_ID 重新评测.

输出文件:
    <NORMLIFT_OUT>/<scene>/knn_vote/voted_features.pt  # (N, D), 喂给 eval_scannet.py 的 FEAT_PATH
    <NORMLIFT_OUT>/<scene>/knn_vote/voted_labels.pt    # (N,) int64, cos-argmax(text) 诊断标签
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
K            = 36     # KNN size (空间 K 近邻)
TAU_R        = 0.75   # 高置信阈值: conf >= 此值则跳过更新
SIGMA_D      = 0.3    # 距离带宽 (米); d_ik = exp(-||Δμ||² / (2σ_d²))
TAU_COS      = 0.55   # 语义一致性阈值
GAMMA        = 0.1    # sigmoid 软化程度
DELTA        = 0.05   # margin: 仅当 S_ij* > S_ii + Δ 才替换
TAU_REPLACE  = 0.0    # replacement cosine guard: 仅当 cos(f_i, f_{j*}) > 此值才替换
                      #   防止小类点的残留方向被 mode (大类) 完全吞并
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
    print(f"  K={K}  τ_r={TAU_R}  σ_d={SIGMA_D}m  τ={TAU_COS}  γ={GAMMA}  Δ={DELTA}  τ_replace={TAU_REPLACE}")

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

    # ─── Step 3: 选出待更新高斯 ──────────────────────────────────────
    print(f"\nStep 3: 划分高/低置信高斯 ...")
    needs_vote   = conf_dev < TAU_R                                             # (N,) bool
    n_to_vote    = int(needs_vote.sum())
    n_high_conf  = N - n_to_vote
    print(f"  high-conf (conf >= {TAU_R}, skip):  {n_high_conf:>7d} ({n_high_conf/N*100:5.2f}%)")
    print(f"  to vote   (conf <  {TAU_R}):        {n_to_vote:>7d} ({n_to_vote/N*100:5.2f}%)")

    # ─── Step 4: 分块跑 mode voting ──────────────────────────────────
    print(f"\nStep 4: chunked mode voting (chunk={CHUNK_SIZE}) ...")
    target_idx_all   = torch.nonzero(needs_vote, as_tuple=False).squeeze(1)     # (n_to_vote,)
    new_feats        = feats.clone()                                            # (N, D) on cpu
    n_replaced       = 0
    n_blocked_guard  = 0    # 被 replacement cosine guard 否决的次数

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

        # ─── replacement cosine guard ────────────────────────────────
        # 即便 S_ij* 占优, 也要求 f_{j*} 与原 f_i 的方向不能差太远.
        # 这样能在低置信小类高斯的"语义残留方向"上设一个安全网,
        # 避免被语义不近但支持度高的大类 mode 一把吞并.
        # cos(f_i, f_{j*}) 取归一化后的 inner product, 自己=ext_feats_n[:, 0, :]
        f_i_n             = ext_feats_n[:, 0, :]                                 # (M, D)
        chosen_local_idx  = (argmax_in_nbrs + 1).view(M, 1, 1).expand(-1, 1, D)  # (M, 1, D)
        f_jstar_n         = ext_feats_n.gather(1, chosen_local_idx).squeeze(1)   # (M, D)
        cos_i_jstar       = (f_i_n * f_jstar_n).sum(dim=-1)                      # (M,)
        guard_pass        = cos_i_jstar > TAU_REPLACE                            # (M,)

        # 仅做诊断: 多少次 score 通过但被 guard 拦下
        n_blocked_guard += int((score_pass & ~guard_pass).sum())

        replace_mask = score_pass & guard_pass                                   # (M,)

        if int(replace_mask.sum()) > 0:
            replace_targets = chunk_targets[replace_mask].cpu()                  # (M_repl,)
            replace_sources = chosen_global_idx[replace_mask].cpu()              # (M_repl,)
            # 用未归一化的原 feature 直接复制 (cos sim 看方向, 模长无所谓)
            new_feats[replace_targets] = feats[replace_sources]
            n_replaced += int(replace_mask.sum())

    print(f"\n  replaced (mode voted):                {n_replaced:>7d}  ({n_replaced/N*100:5.2f}%)")
    print(f"  blocked by cos guard (τ_replace={TAU_REPLACE}):  {n_blocked_guard:>7d}  ({n_blocked_guard/N*100:5.2f}%)")
    n_kept = n_to_vote - n_replaced
    print(f"  kept (low conf, no consensus margin): {n_kept:>7d}  ({n_kept/N*100:5.2f}%)")

    # ─── Step 5: 诊断 — 类分布前后对比 (cos sim with text features) ──
    print(f"\nStep 5: class distribution before/after (cos-argmax with text) ...")
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
