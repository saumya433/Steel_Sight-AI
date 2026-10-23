"""Stage 1: unsupervised anomaly screen (PatchCore-lite).

Trained ONLY on defect-free surface texture. Anything that does not look like
normal surface gets a high score - including defect types the detector has
never been trained on. This is what turns a closed-set detector into an
open-set inspector.

Method (simplified from Roth et al., "Towards Total Recall in Industrial
Anomaly Detection", CVPR 2022):

  1. A frozen ImageNet ResNet-18 turns each image into a grid of patch
     descriptors (layer2 + layer3 features, locally averaged).
  2. The memory bank is the set of descriptors from NORMAL patches, reduced
     with greedy coreset subsampling so scoring stays fast.
  3. A test patch is scored by its distance to the nearest bank entry.
     The image score is the max patch score; the patch grid is the heatmap.
  4. The threshold is calibrated on held-out normal patches (e.g. the 99.5th
     percentile) so the false-alarm rate is an explicit, chosen number.

On NEU-DET there are no defect-free images, so ``scripts/train_anomaly.py``
builds the bank from the regions OUTSIDE the annotated boxes. On a real line
you simply record a few minutes of good surface - no labelling needed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class AnomalyConfig:
    image_size: int = 224          # images are resized to this before feature extraction
    proj_dim: int = 128            # random projection of the patch descriptor (speed/memory)
    coreset_fraction: float = 0.10 # keep this fraction of normal patches in the bank
    max_bank: int = 20000          # hard cap on bank size
    neighbourhood: int = 3         # local average pooling kernel over the feature grid
    backbone: str = "resnet18"     # "resnet18" (ImageNet, the PatchCore default) or "yolo:<weights.pt>"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class _ResNetExtractor(torch.nn.Module):
    """Frozen ImageNet ResNet-18 up to layer3, returning the layer2 (+ upsampled layer3) grid."""

    out_dim = 384

    def __init__(self):
        super().__init__()
        import torchvision

        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        f2 = self.layer2(self.stem(x))            # stride 8,  128 ch
        f3 = self.layer3(f2)                      # stride 16, 256 ch
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)         # (B, 384, H/8, W/8)


class _YoloExtractor(torch.nn.Module):
    """Frozen YOLOv8 backbone (layers 0-6): stride-8 and stride-16 C2f features.

    Lets Stage 1 share the detector's own backbone ("one backbone, two heads"),
    and works wherever the YOLO weights are available even if ImageNet weights
    cannot be downloaded.
    """

    def __init__(self, weights: str = "yolov8n.pt"):
        super().__init__()
        from ultralytics import YOLO

        det = YOLO(weights).model
        self.layers = torch.nn.ModuleList(list(det.model)[:7])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        with torch.no_grad():
            self.out_dim = int(self.forward(torch.zeros(1, 3, 64, 64)).shape[1])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f8 = f16 = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 4:
                f8 = x
            elif i == 6:
                f16 = x
        f16 = F.interpolate(f16, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f8, f16], dim=1)


def build_extractor(backbone: str) -> tuple[torch.nn.Module, str]:
    """Return (extractor, backbone_name_actually_used)."""
    if backbone.startswith("yolo:"):
        return _YoloExtractor(backbone.split(":", 1)[1]), backbone
    try:
        return _ResNetExtractor(), "resnet18"
    except Exception as e:  # weights download blocked, no torchvision, ...
        import warnings

        warnings.warn(f"ImageNet ResNet-18 unavailable ({type(e).__name__}: {str(e)[:80]}); "
                      "falling back to the YOLOv8n backbone for Stage 1 features.")
        return _YoloExtractor("yolov8n.pt"), "yolo:yolov8n.pt"


class AnomalyScreen:
    """Fit on normal texture, then score images and produce heatmaps."""

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.cfg = config or AnomalyConfig()
        self.device = torch.device(self.cfg.device)
        self.extractor, self.cfg.backbone = build_extractor(self.cfg.backbone)
        self.extractor = self.extractor.to(self.device)
        in_dim = int(self.extractor.out_dim)
        proj_dim = min(self.cfg.proj_dim, in_dim)
        gen = torch.Generator().manual_seed(self.cfg.seed)
        self.projection = (torch.randn(in_dim, proj_dim, generator=gen) / np.sqrt(proj_dim)).to(self.device)
        self.bank: Optional[torch.Tensor] = None     # (N, proj_dim)
        self.threshold: Optional[float] = None
        self.calibration: dict = {}

    # ------------------------------------------------------------------ features
    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        import cv2

        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)

    @torch.no_grad()
    def patch_features(self, images: Sequence[np.ndarray], batch: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
        """Return (n_images, H, W, proj_dim) patch descriptors and the grid size."""
        feats = []
        k = self.cfg.neighbourhood
        for i in range(0, len(images), batch):
            x = torch.cat([self._to_tensor(im) for im in images[i:i + batch]]).to(self.device)
            f = self.extractor(x)
            f = F.avg_pool2d(f, kernel_size=k, stride=1, padding=k // 2)       # local aggregation
            f = f.permute(0, 2, 3, 1) @ self.projection                          # (B, H, W, D)
            feats.append(f.cpu())
        feats_t = torch.cat(feats)
        return feats_t, (feats_t.shape[1], feats_t.shape[2])

    # ---------------------------------------------------------------------- fit
    @staticmethod
    def _coreset(x: torch.Tensor, n_keep: int, seed: int = 0) -> torch.Tensor:
        """Greedy k-center coreset: keeps the bank diverse instead of random."""
        if n_keep >= len(x):
            return x
        g = torch.Generator().manual_seed(seed)
        # Start from a random subset to bound the O(N * n_keep) cost on large banks.
        if len(x) > 4 * n_keep:
            x = x[torch.randperm(len(x), generator=g)[: 4 * n_keep].to(x.device)]
        idx = [int(torch.randint(len(x), (1,), generator=g))]
        min_d = torch.cdist(x, x[idx[-1]].unsqueeze(0)).squeeze(1)
        for _ in range(n_keep - 1):
            nxt = int(torch.argmax(min_d))
            idx.append(nxt)
            d = torch.cdist(x, x[nxt].unsqueeze(0)).squeeze(1)
            min_d = torch.minimum(min_d, d)
        return x[idx]

    def fit(self, images: Sequence[np.ndarray], masks: Optional[Sequence[Optional[np.ndarray]]] = None) -> "AnomalyScreen":
        """Build the memory bank from normal texture.

        masks: optional per-image uint8 arrays where 1 marks DEFECT pixels.
               Patches whose cell overlaps a defect are excluded from the bank.
               On a real line pass masks=None and feed defect-free footage.
        """
        feats, (gh, gw) = self.patch_features(images)
        if masks is None:
            normal = feats.reshape(-1, feats.shape[-1])
        else:
            keep = []
            for i, m in enumerate(masks):
                if m is None:
                    keep.append(feats[i].reshape(-1, feats.shape[-1]))
                    continue
                cell = _downsample_mask(m, gh, gw)       # 1 where the cell touches a defect
                keep.append(feats[i][torch.from_numpy(cell == 0)])
            normal = torch.cat(keep)
        n_keep = int(min(self.cfg.max_bank, max(1, self.cfg.coreset_fraction * len(normal))))
        # Greedy coreset is O(N * n_keep): run it on the GPU when there is one.
        self.bank = self._coreset(normal.to(self.device), n_keep, self.cfg.seed)
        self.calibration = {"backbone": self.cfg.backbone, "n_normal_patches": int(len(normal)), "bank_size": int(len(self.bank))}
        return self

    # -------------------------------------------------------------------- score
    @torch.no_grad()
    def _nn_distance(self, feats: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        assert self.bank is not None, "call fit() or load() first"
        flat = feats.reshape(-1, feats.shape[-1]).to(self.device)
        out = torch.empty(len(flat))
        for i in range(0, len(flat), chunk):
            d = torch.cdist(flat[i:i + chunk], self.bank)
            out[i:i + chunk] = d.min(dim=1).values.cpu()
        return out.reshape(feats.shape[:-1])

    def score(self, image_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (image_score, heatmap) where heatmap is HxW float32 at input size."""
        import cv2

        feats, _ = self.patch_features([image_bgr])
        grid = self._nn_distance(feats)[0].numpy()
        heat = cv2.resize(grid, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=4)
        return float(grid.max()), heat.astype(np.float32)

    def patch_scores(self, images: Sequence[np.ndarray]) -> np.ndarray:
        feats, _ = self.patch_features(images)
        return self._nn_distance(feats).numpy()

    def calibrate(self, normal_patch_scores: np.ndarray, percentile: float = 99.5) -> float:
        """Set the threshold as a percentile of scores on held-out NORMAL patches.

        percentile 99.5 means: about 0.5% of normal patches would be flagged,
        i.e. a chosen, explicit false-alarm budget.
        """
        self.threshold = float(np.percentile(normal_patch_scores, percentile))
        self.calibration.update({"percentile": percentile, "threshold": self.threshold})
        return self.threshold

    def is_anomalous(self, score: float) -> bool:
        return self.threshold is not None and score >= self.threshold

    # ------------------------------------------------------------------ persist
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bank": self.bank.cpu() if self.bank is not None else None,
                "projection": self.projection.cpu(),
                "threshold": self.threshold,
                "calibration": self.calibration,
                "config": dataclasses.asdict(self.cfg),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "AnomalyScreen":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = AnomalyConfig(**blob["config"])
        if device:
            cfg.device = device
        obj = cls(cfg)
        obj.projection = blob["projection"].to(obj.device)
        obj.bank = blob["bank"].to(obj.device) if blob["bank"] is not None else None
        obj.threshold = blob["threshold"]
        obj.calibration = blob.get("calibration", {})
        return obj


def _downsample_mask(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """1 where a feature-grid cell overlaps any defect pixel (with a 1-cell margin)."""
    import cv2

    m = (mask > 0).astype(np.uint8)
    cell = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)      # fraction of defect pixels
    cell = (cell > 0).astype(np.uint8)
    cell = cv2.dilate(cell, np.ones((3, 3), np.uint8))                # margin so edges are not "normal"
    return cell


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Area under ROC for scores of positives vs negatives (rank-based, no sklearn)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv)
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def boxes_to_mask(boxes_xyxy: Iterable[Sequence[float]], h: int, w: int) -> np.ndarray:
    """YOLO-style boxes (pixel xyxy) -> uint8 mask with 1 inside boxes."""
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        m[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))] = 1
    return m


def summarize(screen: AnomalyScreen) -> str:
    return json.dumps(screen.calibration, indent=2)
