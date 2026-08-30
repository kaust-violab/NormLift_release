import os
import sys

# 本脚本依赖 splat-distiller 源码树里的顶层模块 (argparser, application_beta 等),
# 用 SPLAT_DISTILLER_DIR 指向其根目录 (或直接在该目录下运行).
_SD = os.environ.get("SPLAT_DISTILLER_DIR")
if _SD and _SD not in sys.path:
    sys.path.insert(0, _SD)

from nerfview import apply_float_colormap
import torch
import numpy as np
from gsplat_ext import Dataset, Parser
from gaussian_splatting.utils import set_random_seed
from gsplat.distributed import cli
import os
from tqdm import tqdm
import torch.nn.functional as F
from gaussian_splatting.analysis import compute_prototypes, fast_hdbscan_with_pos_gpu
from typing import Tuple, Union, Dict
from gsplat_ext import GaussianPrimitive, GaussianPrimitive2D, BetaSplatPrimitive
from gsplat_ext import GaussianRenderer, GaussianRenderer2D, BetaSplatRenderer
from pathlib import Path
from PIL import Image
from application_beta.semantic_query.mask_label_registration import mask_register
from argparser import DataArgs, DistillArgs, build_parser

# -----------------------------------------------------------------------
# Configuration — per-scene language feature distillation (NormLift lifting)
# -----------------------------------------------------------------------
_RELEASE_ROOT = Path(__file__).resolve().parents[1]
_DATA  = Path(os.environ.get("NORMLIFT_DATA", _RELEASE_ROOT / "data"))
_OUT   = Path(os.environ.get("NORMLIFT_OUT",  _RELEASE_ROOT / "outputs"))
_SCENE = os.environ.get("NORMLIFT_SCENE", "scene0200_00")

SCENE_DIR      = str(_DATA / f"{_SCENE}_colmap")
LANG_FEAT_DIR  = _DATA / _SCENE / "language_features" / "language_features"
CKPT_PATH      = str(_OUT / _SCENE / "point_cloud" / "iteration_30000" / "point_cloud.pt")
OUTPUT_DIR     = _OUT / _SCENE / "ckpts"
TIKHONOV       = 1.0
QUANTIZE       = False    # enable HDBSCAN post-processing + mask-level refinement
SAM_LEVEL      = 3      # which SAM level to use from _s.npy  (0-3)
# -----------------------------------------------------------------------

beta_filter_args = {
    "beta_filter_larde": 3.0,
}










class Runner:
    """Engine for training and testing."""

    def __init__(self, data_args: DataArgs, distill_args: DistillArgs) -> None:
        set_random_seed(42)

        self.data_args = data_args
        self.distill_args = distill_args
        self.device = f"cuda"

        self.set_dataset(data_args)
        self.set_splat(distill_args)  # set the splat primitive and renderer

    def set_dataset(self, data_args: DataArgs):
        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=data_args.dir,
            factor=data_args.factor,
            test_every=data_args.test_every,
        )
        # load_features=False: we load language features manually from npy files
        trainset = Dataset(
            self.parser,
            split="train",
            load_features=False,
        )
        valset = Dataset(self.parser, split="val")

        self.trainLoader = torch.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=False
        )
        self.valLoader = torch.utils.data.DataLoader(
            valset, batch_size=1, shuffle=False
        )

        self.scene_scale = self.parser.scene_scale * 1.1

    def set_splat(self, distill_args: DistillArgs):
        if distill_args.method == "2DGS":
            self.splats = GaussianPrimitive2D()
            self.splats.from_file(distill_args.ckpt, tikhonov=distill_args.tikhonov)
            self.renderer = GaussianRenderer2D(self.splats)
        elif distill_args.method == "3DGS":
            self.splats = GaussianPrimitive()
            self.splats.from_file(distill_args.ckpt, tikhonov=distill_args.tikhonov)
            self.renderer = GaussianRenderer(self.splats)
        elif distill_args.method == "DBS":
            self.splats = BetaSplatPrimitive()
            self.splats.from_file(distill_args.ckpt)
            self.renderer = BetaSplatRenderer(self.splats)
        else:
            raise ValueError(f"Invalid splat method: {distill_args.method}")

        self.splats.to(self.device)



    def quantize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize the features to the prototypes
        We don't think this is a good idea when dealing with general tasks
        for example, lifting resnet, vit, or general features, especially visual features
        such as SHIFT features are not good for this, because they are not supposed to be quantized

        For 3D segmentation, it is also not a good idea, we recommend to use the original features
        plus our contrastive comparison to segment the splats. Since one can observe the background

        I implement this function is simply it works for metrics evaluation. During real application
        we find that using contrastive comparison is good enough. HDBSCAN for large scale clustering
        is too slow, and the level is hard to control.

        For now, we only use this function for metrics evaluation, not for real application.
        """
        labels, probs, pca_feat, pca_pos = fast_hdbscan_with_pos_gpu(
            embeddings=self.splat_features,
            positions=None,
            num_freqs=4,
            n_components=50,
            min_samples=10,
            use_incremental_pca=False,
        )
        labels = torch.tensor(labels)
        prototypes, assigned_features = compute_prototypes(
            torch.tensor(labels), self.splat_features
        )
        return assigned_features, prototypes, labels

    @torch.no_grad()
    def distill(self, distill_args: DistillArgs, data_args: DataArgs):
        """Entry for distillation using language features from npy files."""
        print("Running distillation with language features from:", LANG_FEAT_DIR)
        means = self.splats.geometry["means"]
        feature_dim = 512  # language features are always 512-dim (CLIP)

        self.splat_features = torch.zeros(
            (means.shape[0], feature_dim), dtype=torch.float32, device=self.device
        )
        self.splat_weights = torch.zeros(
            (means.shape[0]), dtype=torch.float32, device=self.device
        )

        skipped = 0
        for i, data in tqdm(
            enumerate(self.trainLoader),
            desc="Distilling language features",
            total=len(self.trainLoader),
        ):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            height, width = pixels.shape[1:3]

            # Resolve image stem (e.g. "0", "20", ...) to find the npy files
            image_stem = Path(data["image_name"][0]).stem
            feat_path = LANG_FEAT_DIR / f"{image_stem}_f.npy"
            mask_path = LANG_FEAT_DIR / f"{image_stem}_s.npy"

            if not feat_path.exists() or not mask_path.exists():
                skipped += 1
                continue

            # Load language features: mask_feats [num_masks, 512], mask_segs [4, H, W]
            mask_feats_np = np.load(feat_path).astype(np.float32)  # [num_masks, 512]
            mask_segs_np  = np.load(mask_path).astype(np.float32)  # [4, H_orig, W_orig]

            # Build dense feature map from SAM level SAM_LEVEL
            mask_ids = mask_segs_np[SAM_LEVEL].astype(np.int32)    # [H_orig, W_orig]
            H_orig, W_orig = mask_ids.shape
            dense_feat = np.zeros((H_orig, W_orig, 512), dtype=np.float32)
            valid = mask_ids >= 0
            if valid.any():
                # Clamp indices to valid range in case of edge IDs
                safe_ids = np.clip(mask_ids[valid], 0, len(mask_feats_np) - 1)
                dense_feat[valid] = mask_feats_np[safe_ids]

            # Convert to tensor [1, H_orig, W_orig, 512]
            features = torch.from_numpy(dense_feat).to(self.device).unsqueeze(0)

            # Resize to match rendered resolution (in case factor != 1)
            if H_orig != height or W_orig != width:
                features = features.permute(0, 3, 1, 2)   # [1, 512, H, W]
                features = torch.nn.functional.interpolate(
                    features, size=(height, width), mode="bilinear", align_corners=False
                )
                features = features.permute(0, 2, 3, 1)   # [1, H, W, 512]

            features = F.normalize(features, p=2, dim=-1)

            (
                splat_features_per_image,
                splat_weights_per_image,
                ids,
            ) = self.renderer.inverse_render(
                K=Ks,
                extrinsic=camtoworlds,
                width=width,
                height=height,
                features=features,
            )

            self.splat_features[ids] += splat_features_per_image[:, :feature_dim]
            self.splat_weights[ids] += splat_weights_per_image
            del splat_features_per_image, splat_weights_per_image, ids, features
            torch.cuda.empty_cache()

        if skipped > 0:
            print(f"WARNING: skipped {skipped} frames (no matching npy files)")

        self.splat_features /= self.splat_weights[..., None]
        self.splat_features = torch.nan_to_num(self.splat_features, nan=0.0)

        del self.splat_weights
        torch.cuda.empty_cache()

        # Save results to OUTPUT_DIR
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_stem = Path(distill_args.ckpt).stem          # e.g. "ckpt_29999_rank0"
        self.basename = str(OUTPUT_DIR / ckpt_stem)
        
        self.splat_features = self.splat_features.cpu()

        if distill_args.quantize == "True":
            print("Quantize is set to True, running HDBSCAN ...")
            _, _, self.labels = self.quantize()
            torch.save(self.labels, self.basename + "_labels.pt")
            print("Applying Post-Aggregation Filtering with language features ...")
            image_labels: Dict[str, torch.Tensor] = self.label_projection(self.labels)
            self.splats.to("cpu")
            torch.cuda.empty_cache()
            self.splat_features = self.mask_level_refinement(data_args, image_labels).to('cpu')

        if distill_args.filter != -1:
            self.splats.to("cuda")
            filter_set = Dataset(self.parser, split="train")
            filter_loader = torch.utils.data.DataLoader(
                filter_set, batch_size=1, shuffle=False
            )
            masks = self.splats.filtering(
                filter_loader,
                self.splats.geometry["means"],
                torch.device(self.device),
                threshold=distill_args.filter,
                args=None,
            )
            ckpt_path = Path(distill_args.ckpt)
            self.splats.mask(masks)
            self.splats.save(ckpt_path.parent / (ckpt_path.stem + "_filtered.pt"))
            self.splat_features = self.splat_features.cuda()[masks]


        suffix = f"_{distill_args.out_name}" if distill_args.out_name else ""
        if distill_args.filter != -1 and distill_args.quantize == "True":
            torch.save(self.splat_features, self.basename + f"_filtered_refined_features{suffix}.pt")
        elif distill_args.filter != -1:
            torch.save(self.splat_features, self.basename + f"_filtered_features{suffix}.pt")
        elif distill_args.quantize == "True":
            torch.save(self.splat_features, self.basename + f"_refined_features{suffix}.pt")
        else:
            torch.save(self.splat_features, self.basename + f"_features{suffix}.pt")
    
    def mask_level_refinement(self, data_args: DataArgs, image_labels: Dict[str, torch.Tensor]):
        # Use LANG_FEAT_DIR directly: contains *_s.npy (SAM masks) and *_f.npy (CLIP features)
        # in OpenGaussian format, which mask_register already supports natively.
        self.mask_registrater = mask_register(image_labels, LANG_FEAT_DIR)
        self.mask_registrater.reg()
        refined = self.mask_registrater.GaussianFeatureRefinement(self.splat_features.cuda(), self.labels)
        return refined

    @torch.no_grad()
    def label_projection(
        self, labels: torch.Tensor, debugg_path: Union[Path, None] = None
    ):
        """
        The stratgy is the following, we can assign the label as a one hot vector, or other orthogonal vector,
        and then we project those orthogonal labels from high dimension to camera position, and then determine
        pixel level correspondance.
        That is at position (image_number i, height x ,width y) -> cluster c. If we have so called masked level
        level correspondance, we then have a mask to Gaussian Cluster Correspondance, therefore, we can somehow
        keep the multiview difference for each Gaussian

        We try not to save it on disks
        """
        print("beta version will try to establish mask to Gaussian Cluster relation")
        image_labels = {}

        # labels: (n,) input, where n is the number of elements
        labels = labels + 1  # Shift -1 to 0, 0 to 1, etc.
        num_classes = (
            int(labels.max().item()) + 1
        )  # Since -1 is now 0, this covers all classes including background
        one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()
        #### Then we start to do the projection
        #### We over write the feature in this way, and trigger feature map rendering
        self.splats._feature = one_hot.cuda()
        store_folder = Path(self.basename).parent / "labels"
        store_folder.mkdir(exist_ok=True)

        for i, data in tqdm(
            enumerate(self.trainLoader),
            desc="Project the Cluster Labels",
            total=len(self.trainLoader),
        ):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device)
            height, width = pixels.shape[1:3]
            image_name = Path(data["image_name"][0])

            projected_labels = self.renderer.render(
                K=Ks, extrinsic=camtoworlds, width=width, height=height, mode="Feature"
            )
            assert (
                projected_labels != None
            ), "render has some issue, expect labels as shape of H,W,m shape images"
            # Labels should be a (H,W,N) tensor, and we do argmax to get the cluster it belongs to
            label_dense = projected_labels.argmax(dim=-1)

            image_labels[image_name.stem] = label_dense.cpu()


            # Save the color-mapped image using PIL
            if debugg_path != None:
                assert isinstance(debugg_path, Path)
                map = label_dense / num_classes
                images_for_debugging = apply_float_colormap(map.reshape(-1, 1))
                images_for_debugging = images_for_debugging.reshape(height, width, 3)
                save_path = debugg_path / (image_name.stem + "jpg")
                # images_for_debugging is assumed to be a numpy array or torch tensor in HWC, float [0,1] or [0,255]
                img_np = images_for_debugging.cpu().numpy()
                img_pil = Image.fromarray(img_np)
                img_pil.save(save_path)
        
        return image_labels

def main(local_rank: int, world_rank, world_size: int, args):
    data_args, distill_args = args

    runner = Runner(data_args, distill_args)
    runner.distill(distill_args, data_args)


if __name__ == "__main__":
    data_args = DataArgs(
        dir=SCENE_DIR,
        factor=1,
        test_every=1000,
        feature_folder="language_features",   # not used for loading, informational only
        mask_folder=None,
    )
    distill_args = DistillArgs(
        method="3DGS",
        ckpt=CKPT_PATH,
        tikhonov=TIKHONOV,
        quantize="True" if QUANTIZE else "False",
        filter=-1,
        out_name="",
    )
    cli(main, (data_args, distill_args))
