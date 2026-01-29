# core.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Sequence
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class PathsCfg:
    data_root: Path = Path("data/ds_augmantation")
    ckpt_dir: Path = Path("checkpoints")


@dataclass
class TrainCfg:
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    val_ratio: float = 0.2
    seed: int = 42
    num_workers: int = 2


def get_device() -> str:
    return "cuda"


def build_mobilenet(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        model = mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def make_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class FolderImageDataset(Dataset):
    """
    Структура:
      data_root/
        cl0_augmented/
        cl1_augmented/
        ...
    """
    def __init__(self, data_root: Path, img_size: int = 224):
        self.data_root = Path(data_root)
        self.tf = make_transform(img_size)

        self.class_dirs = sorted([p for p in self.data_root.iterdir() if p.is_dir()])
        if not self.class_dirs:
            raise RuntimeError(f"Не знайдено папок класів у: {self.data_root}")

        self.classes = [p.name for p in self.class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        for cdir in self.class_dirs:
            y = self.class_to_idx[cdir.name]
            for p in cdir.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, y))

        if not self.samples:
            raise RuntimeError(f"Не знайдено зображень у: {self.data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return x, y


def stratified_split_indices(
    samples: Sequence[Tuple[Path, int]],
    val_ratio: float,
    seed: int
) -> Tuple[List[int], List[int]]:
    rnd = random.Random(seed)
    by_class: Dict[int, List[int]] = {}

    for i, (_, y) in enumerate(samples):
        by_class.setdefault(y, []).append(i)

    train_idx, val_idx = [], []
    for y, idxs in by_class.items():
        rnd.shuffle(idxs)
        k = max(1, int(len(idxs) * val_ratio))  # хоча б 1 у val
        val_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])

    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    return train_idx, val_idx


def make_weighted_sampler(train_indices: List[int], samples: Sequence[Tuple[Path, int]]) -> torch.utils.data.WeightedRandomSampler:
    counts: Dict[int, int] = {}
    for i in train_indices:
        _, y = samples[i]
        counts[y] = counts.get(y, 0) + 1

    weights = []
    for i in train_indices:
        _, y = samples[i]
        weights.append(1.0 / max(1, counts[y]))

    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device: str) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()

    return total_loss / max(1, total), correct / max(1, total)


def save_ckpt(path: Path, model: nn.Module, classes: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": classes}, str(path))


def load_ckpt(path: Path, device: str) -> Tuple[nn.Module, List[str]]:
    ckpt = torch.load(str(path), map_location=device)
    classes: List[str] = ckpt["classes"]
    model = build_mobilenet(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, classes


def pretty_class_name(name: str) -> str:
    return name.replace("_augmented", "")


# ---------------- Grad-CAM ----------------

class GradCAM:
    """
    Grad-CAM для класифікації (працює з MobileNetV3).
    target_layer: наприклад model.features[-1]
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        x: (1,3,H,W)
        returns: cam01 (H,W) in [0,1], logits (1,num_classes)
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        A = self.activations  # (1,K,h,w)
        dA = self.gradients   # (1,K,h,w)
        w = dA.mean(dim=(2, 3), keepdim=True)
        cam = (w * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam, logits.detach()
