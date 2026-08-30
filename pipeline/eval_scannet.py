"""
ScanNet 3D semantic segmentation eval — 1:1 版本.

因为训练用 --frozen_init_pts, 高斯数量 == 初始点云 == GT labels.ply 的顶点数,
顺序保留. 所以 GT point 第 i 个 <-> Gaussian 第 i 个, 直接 index 取特征,
不需要 faiss NN 匹配.

对照:
    OpenGaussian/scripts/eval_scannet.py  —— 同款 1:1 index 策略
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData


# ─── Paths (run_pipeline.py 会按场景覆写; 单独运行时用下面的默认值) ──
_DATA  = Path(os.environ.get("NORMLIFT_DATA", Path(__file__).resolve().parents[1] / "data"))
_OUT   = Path(os.environ.get("NORMLIFT_OUT",  Path(__file__).resolve().parents[1] / "outputs"))
_SCENE = os.environ.get("NORMLIFT_SCENE", "scene0140_00")
_NCLS  = int(os.environ.get("NORMLIFT_CLASSES", "19"))

SCENE_DIR = str(_DATA / _SCENE)
GT_PLY    = os.path.join(SCENE_DIR, f"{_SCENE}_vh_clean_2.labels.ply")

CKPT_PATH = str(_OUT / _SCENE / "point_cloud" / "iteration_30000" / "point_cloud.pt")
FEAT_PATH = str(_OUT / _SCENE / "knn_vote" / "voted_features.pt")
TEXT_FEAT = str(Path(os.environ.get(
    "NORMLIFT_TEXT_FEAT",
    Path(__file__).resolve().parents[1] / "assets" / "text_features.json")))

OPACITY_THRESH = 0.1   # Gaussian (after sigmoid) 小于此值的点, 把对应 GT label 置为 unlabeled


# ─── NYU40 label names ──────────────────────────────────────────────
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

# 三套评测协议的类别集合 (OpenGaussian ScanNet eval).
# run_pipeline.py 按 --classes 覆写 TARGET_ID; 单独运行时由 NORMLIFT_CLASSES 选择.
TARGET_ID_SETS = {
    19: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    15: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34],
    10: [1, 2, 4, 5, 6, 7, 8, 9, 10, 33],
}
TARGET_ID = TARGET_ID_SETS[_NCLS]


# ─── Helpers ─────────────────────────────────────────────────────────
def read_labels_from_ply(file_path):
    plydata = PlyData.read(file_path)
    vd = plydata["vertex"].data
    pts    = np.vstack([vd["x"], vd["y"], vd["z"]]).T.astype(np.float32)
    labels = np.array(vd["label"], dtype=np.int64)
    return pts, labels


def calculate_metrics(gt, pred, total_classes):
    """Per-class IoU, mIoU, Acc, mAcc. Class 0 is ignored."""
    gt   = gt.cpu()
    pred = pred.cpu()
    pred[gt == 0] = 0

    intersection = torch.zeros(total_classes)
    union        = torch.zeros(total_classes)
    correct      = torch.zeros(total_classes)
    total        = torch.zeros(total_classes)

    for cls in range(1, total_classes):
        gt_mask   = gt   == cls
        pred_mask = pred == cls
        intersection[cls] = (gt_mask & pred_mask).sum().float()
        union[cls]        = (gt_mask | pred_mask).sum().float()
        correct[cls]      = (gt_mask & pred_mask).sum().float()
        total[cls]        = gt_mask.sum().float()

    ious = torch.zeros(total_classes)
    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    gt_classes       = torch.unique(gt)
    valid_gt_classes = gt_classes[(gt_classes != 0) & (gt_classes < total_classes)]

    mean_iou       = ious[valid_gt_classes].mean().item() if len(valid_gt_classes) > 0 else float("nan")
    valid_mask     = gt != 0
    accuracy       = (((gt == pred) & valid_mask).sum() / valid_mask.sum().clamp(min=1)).item()
    class_acc      = correct / total.clamp(min=1)
    mean_class_acc = class_acc[valid_gt_classes].mean().item() if len(valid_gt_classes) > 0 else float("nan")

    return ious, mean_iou, accuracy, mean_class_acc, valid_gt_classes


# ─── Main ────────────────────────────────────────────────────────────
def main():
    scene_name = Path(SCENE_DIR).name
    print(f"\n=== Evaluating {scene_name}  ({len(TARGET_ID)}-class protocol, 1:1 indexing, no faiss) ===")

    # (1) GT
    print("Loading GT point cloud + labels ...")
    gt_points, gt_labels_raw = read_labels_from_ply(GT_PLY)
    N_gt = len(gt_points)
    print(f"  {N_gt} GT points")

    # (2) Gaussian geometry (for opacity filtering + 1:1 sanity check)
    print("Loading Gaussian ckpt ...")
    ckpt   = torch.load(CKPT_PATH, map_location="cpu")
    splats = ckpt["splats"]
    gauss_means     = splats["means"].float()          # (M, 3) raw
    gauss_opacities = splats["opacities"].float()      # (M,) raw logits
    M = gauss_means.shape[0]
    print(f"  {M} Gaussians")

    if M != N_gt:
        raise RuntimeError(
            f"Gaussian count ({M}) != GT point count ({N_gt}).  "
            f"1:1 index 不成立. 要么 frozen_init_pts 不是用 labels.ply 点云作为初始化, "
            f"要么训练时点数被改过. 如需 NN 匹配, 请改用 my_eval_scannet.py."
        )

    # 顺便做个弱 sanity check: 比较坐标是否一致 (允许一定漂移, 因为 means 允许梯度更新)
    diff = (gauss_means.numpy() - gt_points).astype(np.float64)
    mean_disp = float(np.linalg.norm(diff, axis=1).mean())
    max_disp  = float(np.linalg.norm(diff, axis=1).max())
    print(f"  1:1 position drift: mean={mean_disp:.4e}  max={max_disp:.4e} "
          f"(允许梯度更新过的小漂移, 不影响 label 对应关系)")

    # (3) Features (per-Gaussian language feature)
    print("Loading per-Gaussian language features ...")
    gauss_feats = torch.load(FEAT_PATH, map_location="cpu").float()   # (M, 512)
    print(f"  feature shape: {tuple(gauss_feats.shape)}")
    if gauss_feats.shape[0] != M:
        raise RuntimeError(
            f"Feature N ({gauss_feats.shape[0]}) != Gaussian N ({M}).  "
            f"ckpt 与 feature 文件不匹配."
        )

    # (4) 按 1:1 对应: 第 i 个 GT point 的 Gaussian 就是 splats[i] / gauss_feats[i]
    #    不做额外 NN 查询.
    per_point_feat = gauss_feats                                      # (N_gt, 512)

    # (5) 低 opacity 的高斯对应的 GT label 置为 unlabeled
    low_opacity = torch.sigmoid(gauss_opacities) < OPACITY_THRESH     # (N_gt,)

    # (6) Remap NYU40 -> compact 1..19
    target_names  = [nyu40_dict[i] for i in TARGET_ID]
    id_to_compact = {nyu40_id: idx + 1 for idx, nyu40_id in enumerate(TARGET_ID)}
    updated = np.zeros_like(gt_labels_raw)
    for orig, new in id_to_compact.items():
        updated[gt_labels_raw == orig] = new
    total_classes = len(TARGET_ID) + 1

    gt_tensor = torch.from_numpy(updated).long()
    gt_tensor[low_opacity] = 0
    print(f"  Ignored {int(low_opacity.sum())} low-opacity points (< {OPACITY_THRESH}).")

    compact_present = torch.unique(gt_tensor).tolist()
    print(f"  compact labels present: {compact_present}")

    # (7) Text features for target classes
    print("Loading text features ...")
    with open(TEXT_FEAT) as f:
        text_data = json.load(f)

    query_text_feats = torch.zeros(len(target_names), 512)
    missing = []
    for i, name in enumerate(target_names):
        if name in text_data:
            query_text_feats[i] = torch.tensor(text_data[name], dtype=torch.float32)
        else:
            missing.append(name)
    if missing:
        print(f"  WARNING: no text feature for: {missing}")

    # (8) Cosine sim -> predicted compact label per point
    print("Computing cosine similarities ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pf = F.normalize(per_point_feat.to(device),    dim=1, p=2)        # (N, 512)
    qf = F.normalize(query_text_feats.to(device),  dim=1, p=2)        # (C, 512)
    sim          = pf @ qf.t()                                         # (N, C)
    pred_compact = sim.argmax(dim=1).cpu() + 1                         # 1..C

    # 零特征的高斯给 unlabeled (避免把它们硬分类)
    zero_feat = per_point_feat.norm(dim=1) == 0
    pred_compact[zero_feat] = 0

    # (9) Metrics
    print("Computing metrics ...")
    ious, mean_iou, accuracy, mean_acc, valid_classes = calculate_metrics(
        gt_tensor, pred_compact, total_classes
    )

    print(f"\n  Per-class IoU (classes present in GT):")
    for cidx in valid_classes.tolist():
        orig_nyu40 = TARGET_ID[cidx - 1]
        cname      = nyu40_dict.get(orig_nyu40, "?")
        print(f"    [{orig_nyu40:2d}] {cname:<20s}  IoU = {ious[cidx]:.4f}")

    print(f"\n  mIoU  = {mean_iou:.4f}")
    print(f"  mAcc  = {mean_acc:.4f}")
    print(f"  Acc   = {accuracy:.4f}\n")

    return {"scene": scene_name, "mIoU": mean_iou, "mAcc": mean_acc, "Acc": accuracy}


if __name__ == "__main__":
    main()
