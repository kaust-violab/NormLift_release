"""
KNN 标签级投票 (不改特征, 只改预测).

跟 knn_filter*.py 的本质区别:
    knn_filter:  改 feature  -> 强行把一个邻居的 feature 套给 weak -> 单点错就翻车
    knn_vote:    改 label    -> K 个邻居按 confidence 加权投票决定 weak 的预测类
                              单个错邻居被多数票稀释, 边界场景多数类自然胜出.

策略:
    SELF_TRUST = 0.75:   conf[i] > 此值 -> 高斯 i 直接用自己的预测 (不参加投票)
    VOTER_THRESH = 0.58: 只有 conf > 此值的邻居才能投票 (低 conf 邻居被剥夺投票权)
    K = 36:              每个高斯的空间 K 近邻

投票公式 (对待投票的高斯 i):
    w_ij = conf_j * exp(-||p_i - p_j||^2 / SIGMA^2)
    votes[c] = Σ_{j ∈ N(i), conf_j > VOTER_THRESH, pred_j > 0} w_ij * 1[pred_j == c]
    +  (可选) conf_i * 1[pred_i == c]   (INCLUDE_SELF=True 时把自己的票也算上)
    new_pred[i] = argmax_c votes[c]
    若 votes 全为 0 (无合法选民): 保留原 pred

距离衰减:
    SIGMA: 高斯核宽度. dist² = SIGMA² -> 权重 = e^(-1) ≈ 0.37
                    dist  = 2σ      -> 权重 ≈ 0.018
    用来保护小目标 / 边界场景 -- 远邻居的"多数票"被衰减, 近邻居的局部结构主导.

迭代投票:
    N_ITER 轮投票. 每轮用上一轮 voted_pred 当邻居标签源.
    self-trust 高斯始终锚定 baseline pred (不漂移).
    收敛 (changes==0) 提前停止.
    通过迭代让"局部一致区"的纠正能传播得更远, 但要警惕过度平滑.

输出做成"喂给 eval_scannet.py 即可"的形式:
    self-trust 的高斯 -> 保持原 feature (cos sim argmax 给出原 pred)
    voted 的高斯      -> feature 改为 text_features[new_pred - 1] (cos sim argmax 给 new_pred)
    new_pred == 0 的  -> feature 置 0 (eval 里 ||f||==0 -> unlabeled)

注意: 投票只跑一次, 固定在完整的 19 类标签空间. 15/10 类协议直接用同一份
投票产物、换 eval_scannet.py 的 TARGET_ID 重新评测 (论文口径).

输出文件:
    <NORMLIFT_OUT>/<scene>/knn_vote/voted_features.pt  # (N, 512), 喂给 eval_scannet.py 的 FEAT_PATH
    <NORMLIFT_OUT>/<scene>/knn_vote/voted_labels.pt    # (N,) int64, per-Gaussian 投票后 compact label
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
K            = 36      # K nearest neighbors (空间)
SELF_TRUST   = 0.75    # conf > 此值: 不参加投票, 保留原 pred
VOTER_THRESH = 0.58    # conf > 此值: 才有投票权
INCLUDE_SELF = False   # 投票时是否把自己的预测算进去 (按自己的 conf 加权)
SIGMA        = 0.05    # m, 距离衰减带宽; weight *= exp(-dist^2 / SIGMA^2)
N_ITER       = 20       # 迭代投票轮数 (changes==0 自动 early stop)


# ─── Labels (compact 1..19) ──────────────────────────────────────────
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
# 投票固定在完整的 19 类标签空间 (OpenGaussian ScanNet eval 的全类别集).
# 15/10 类协议不重新投票, 只在 eval_scannet.py 里换 TARGET_ID 重评.
TARGET_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]


def main():
    NUM_CLASSES = len(TARGET_ID)
    print(f"\n=== KNN label voting (K={K}, self_trust={SELF_TRUST}, voter_thresh={VOTER_THRESH}, "
          f"{NUM_CLASSES}-class protocol) ===")

    # ─── Load ────────────────────────────────────────────────────────
    print("Loading data ...")
    ckpt  = torch.load(CKPT_PATH, map_location="cpu")
    means = ckpt["splats"]["means"].float().numpy().astype(np.float32)   # (N, 3)
    feats = torch.load(FEAT_PATH, map_location="cpu").float()            # (N, D)
    conf  = torch.load(CONF_PATH, map_location="cpu").float()            # (N,)
    N = means.shape[0]
    D = feats.shape[1]
    if feats.shape[0] != N or conf.shape[0] != N:
        raise RuntimeError(f"shape mismatch: means={N}, feats={feats.shape[0]}, conf={conf.shape[0]}")
    print(f"  N={N}  D={D}")

    # ─── Step 1: 文本特征 + 当前 per-Gaussian 预测 ───────────────────
    print("Computing baseline per-Gaussian predictions ...")
    target_names = [nyu40_dict[i] for i in TARGET_ID]
    with open(TEXT_FEAT) as f:
        text_data = json.load(f)
    qf = torch.zeros(NUM_CLASSES, D)
    missing = []
    for i, name in enumerate(target_names):
        if name in text_data:
            qf[i] = torch.tensor(text_data[name], dtype=torch.float32)
        else:
            missing.append(name)
    if missing:
        print(f"  WARNING: no text feature for: {missing}")

    pf_n = F.normalize(feats, dim=1, p=2)
    qf_n = F.normalize(qf,    dim=1, p=2)
    sim          = pf_n @ qf_n.T                       # (N, C)
    pred_compact = sim.argmax(dim=1) + 1                # 1..C
    feat_norms   = feats.norm(dim=1)
    pred_compact[feat_norms == 0] = 0                   # zero feature -> unlabeled
    pred_np = pred_compact.numpy().astype(np.int64)
    conf_np = conf.numpy().astype(np.float32)
    print(f"  baseline pred class distribution:")
    for c in range(NUM_CLASSES + 1):
        n = int((pred_np == c).sum())
        if n > 0:
            cname = "unlabeled" if c == 0 else nyu40_dict.get(TARGET_ID[c-1], "?")
            print(f"    [{c:>2d}] {cname:<20s}: {n:>7d}")

    # ─── Step 2: KNN by 空间位置 (同时保留距离用于衰减) ──────────────
    print(f"\nStep 2: KNN graph (K={K}, sigma={SIGMA}m) ...")
    index = faiss.IndexFlatL2(3)
    index.add(means)
    D_sq, knn_idx = index.search(means, K + 1)         # (N, K+1) 第一列是自己 (dist=0)
    knn_idx = knn_idx[:, 1:]                            # (N, K)  排除自己
    knn_dsq = D_sq[:, 1:]                               # (N, K)  squared L2 距离

    # 距离统计 (用于判断 SIGMA 是否合理)
    knn_dist = np.sqrt(np.clip(knn_dsq, 0, None))
    print(f"  KNN distance: mean={knn_dist.mean():.4f}m  "
          f"median={float(np.median(knn_dist)):.4f}m  "
          f"max={knn_dist.max():.4f}m")
    typical_w = float(np.mean(np.exp(-knn_dsq / (SIGMA ** 2))))
    print(f"  avg dist-decay weight at SIGMA={SIGMA}: {typical_w:.4f} "
          f"(贴近 1 = 衰减太弱; 贴近 0 = 衰减太强)")

    # ─── Step 3: 决定哪些高斯需要投票 ────────────────────────────────
    self_trust_mask = conf_np > SELF_TRUST              # 不投票
    needs_vote      = ~self_trust_mask                  # 待投票
    n_self_trust    = int(self_trust_mask.sum())
    n_to_vote       = int(needs_vote.sum())
    print(f"  self_trust  (conf > {SELF_TRUST}): {n_self_trust:>7d}  ({n_self_trust/N*100:5.2f}%)")
    print(f"  to vote     (conf <= {SELF_TRUST}): {n_to_vote:>7d}  ({n_to_vote/N*100:5.2f}%)")

    # ─── Step 4: 迭代投票 ────────────────────────────────────────────
    print(f"\nStep 4: iterative voting (N_ITER={N_ITER}) ...")
    # 静态量
    dist_w           = np.exp(-knn_dsq / (SIGMA ** 2)).astype(np.float32)   # (N, K)
    neighbor_confs   = conf_np[knn_idx]                                     # (N, K)
    baseline_pred    = pred_np.copy()                                       # 永不改变的锚
    cur_pred         = baseline_pred.copy()

    for it in range(N_ITER):
        # 用 cur_pred 取邻居标签
        neighbor_preds = cur_pred[knn_idx]                                  # (N, K)
        voter_mask     = (neighbor_confs > VOTER_THRESH) & (neighbor_preds > 0)
        weighted_confs = (neighbor_confs * dist_w * voter_mask).astype(np.float32)

        # 投票表
        votes = torch.zeros(N, NUM_CLASSES + 1, dtype=torch.float64)
        votes.scatter_add_(
            1,
            torch.from_numpy(neighbor_preds).long(),
            torch.from_numpy(weighted_confs).double(),
        )

        if INCLUDE_SELF:
            self_valid = (cur_pred > 0)
            for c in range(1, NUM_CLASSES + 1):
                mask_c = self_valid & (cur_pred == c)
                if mask_c.any():
                    votes[mask_c, c] += torch.from_numpy(conf_np[mask_c]).double()

        votes[:, 0] = -1.0     # 不让 argmax 选到 unlabeled
        voted_pred  = votes.argmax(dim=1).numpy().astype(np.int64)
        no_voter    = (votes[:, 1:].sum(dim=1).numpy() == 0)

        # 合成 next_pred:
        #   self-trust   -> baseline (永远锚定)
        #   no_voter     -> cur_pred (保留当前)
        #   else         -> voted_pred
        next_pred = baseline_pred.copy()
        for_voting = needs_vote & ~no_voter
        keep_no_voter = needs_vote & no_voter
        next_pred[for_voting]    = voted_pred[for_voting]
        next_pred[keep_no_voter] = cur_pred[keep_no_voter]

        n_diff_step  = int((next_pred != cur_pred).sum())
        n_diff_base  = int((next_pred != baseline_pred).sum())
        n_voted      = int(for_voting.sum())
        n_no_voter_i = int(keep_no_voter.sum())
        print(f"  iter {it+1}/{N_ITER}: "
              f"voted={n_voted:>6d}  no_voter={n_no_voter_i:>5d}  "
              f"changes_vs_prev={n_diff_step:>5d}  "
              f"changes_vs_baseline={n_diff_base:>6d}")

        cur_pred = next_pred
        if n_diff_step == 0:
            print(f"  -> converged at iter {it+1}, early stop.")
            break

    new_pred = cur_pred
    n_changed = int((new_pred != baseline_pred).sum())
    print(f"\nFinal: {n_changed} gaussians changed vs baseline pred.")

    # ─── Stats: 投票后的类分布 + 类间转移矩阵 ────────────────────────
    print(f"\nNew pred class distribution:")
    for c in range(NUM_CLASSES + 1):
        n_old = int((pred_np  == c).sum())
        n_new = int((new_pred == c).sum())
        if n_old > 0 or n_new > 0:
            cname = "unlabeled" if c == 0 else nyu40_dict.get(TARGET_ID[c-1], "?")
            delta = n_new - n_old
            print(f"    [{c:>2d}] {cname:<20s}: {n_old:>7d} -> {n_new:>7d}  "
                  f"({'+' if delta >= 0 else ''}{delta})")

    # ─── Step 6: 把 new_pred 写回 feature 文件 ───────────────────────
    print("\nStep 6: building voted_features.pt ...")
    new_feats = feats.clone()
    # changed 高斯: feature 改成 text_feature[new_pred-1]
    changed_mask = torch.from_numpy(new_pred != pred_np)
    if changed_mask.any():
        # 直接用未归一的文本特征 -- cosine sim 看方向不看模长
        new_feats[changed_mask] = qf[torch.from_numpy(new_pred[changed_mask.numpy()] - 1).long()]
    # new_pred == 0 的高斯: 置 0 (其实不应出现, 因为我们 votes[:, 0] = -1)
    zero_mask = torch.from_numpy(new_pred == 0)
    new_feats[zero_mask] = 0.0

    # Sanity: 重新跑一次 cos-argmax, 确认它给出 new_pred
    pf_n2 = F.normalize(new_feats, dim=1, p=2)
    sim2  = pf_n2 @ qf_n.T
    check_pred = sim2.argmax(dim=1).numpy() + 1
    check_pred[new_feats.norm(dim=1).numpy() == 0] = 0
    n_consistent = int((check_pred == new_pred).sum())
    print(f"  feature -> argmax sanity: {n_consistent}/{N} 一致 "
          f"({n_consistent/N*100:.2f}%)")

    # ─── Save ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_out  = OUT_DIR / "voted_features.pt"
    label_out = OUT_DIR / "voted_labels.pt"
    torch.save(new_feats, feat_out)
    torch.save(torch.from_numpy(new_pred).long(), label_out)
    print(f"\nSaved features: {feat_out}   shape={tuple(new_feats.shape)}")
    print(f"Saved labels  : {label_out}   shape={tuple(new_pred.shape)}")
    print(f"\n下一步: 把 eval_scannet.py 的 FEAT_PATH 指向 {feat_out} 跑 eval "
          f"(run_pipeline.py 的 eval step 会自动做这件事).")


if __name__ == "__main__":
    main()
