# video_core.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class MultiCfg:
    images_root: Path = Path("data/ds_augmantation")
    img_size: int = 224
    frames_per_sample: int = 4   # вимога complexity 3
    seed: int = 42

    # скільки "псевдо-кліпів" генерити на клас (на епоху)
    samples_per_class: int = 1200

    # якщо True — кадри беруться з повторенням (коли клас малий)
    allow_replacement: bool = True


def get_device() -> str:
    if torch.cuda.is_available() and torch.version.cuda is not None:
        return "cuda"
    return "cpu"


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def pretty_class_name(name: str) -> str:
    return name.replace("_augmented", "")


def _list_class_dirs(images_root: Path) -> List[Path]:
    return sorted([p for p in images_root.iterdir() if p.is_dir()])


def _list_images(class_dir: Path) -> List[Path]:
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


class MultiImage4Dataset(torch.utils.data.Dataset):
    """
    Один семпл = 4 фото з ОДНОГО класу.
    Повертає:
      x: (T=4, 3, H, W)
      y: int
    """
    def __init__(self, cfg: MultiCfg):
        self.cfg = cfg
        self.tf = make_transform(cfg.img_size)
        self.rnd = random.Random(cfg.seed)

        class_dirs = _list_class_dirs(cfg.images_root)
        if not class_dirs:
            raise RuntimeError(f"Нема папок класів у: {cfg.images_root}")

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # збираємо список зображень по кожному класу
        self.images_by_class: Dict[int, List[Path]] = {}
        for d in class_dirs:
            y = self.class_to_idx[d.name]
            imgs = _list_images(d)
            if len(imgs) == 0:
                continue
            self.images_by_class[y] = imgs

        if not self.images_by_class:
            raise RuntimeError("Не знайдено жодного зображення в жодному класі.")

        # попередньо генеруємо "items" як (label, [img1,img2,img3,img4])
        self.items: List[Tuple[int, List[Path]]] = []
        for y, imgs in self.images_by_class.items():
            n = len(imgs)
            for _ in range(cfg.samples_per_class):
                if n >= cfg.frames_per_sample and not cfg.allow_replacement:
                    chosen = self.rnd.sample(imgs, cfg.frames_per_sample)
                else:
                    chosen = [self.rnd.choice(imgs) for _ in range(cfg.frames_per_sample)]
                self.items.append((y, chosen))

        self.rnd.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        y, paths = self.items[idx]
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.tf(img))   # (3,H,W)
        x = torch.stack(frames, dim=0)   # (4,3,H,W)
        return x, y


class VideoMobileNetMean(nn.Module):
    """
    Lightweight "hand-crafted" відео-модель:
    - MobileNetV3 encoder для кожного кадру
    - усереднення по часу (mean)
    - linear classifier
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_dim = backbone.classifier[-1].in_features
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.avgpool(z).flatten(1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,3,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        z = self.encode_frame(x)            # (B*T, D)
        z = z.view(B, T, -1).mean(dim=1)    # (B, D)
        return self.classifier(z)           # (B, num_classes)


def model_size_mb(model: nn.Module) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 4 / (1024 ** 2)


@torch.no_grad()
def estimate_fps(model: nn.Module, device: str, img_size: int = 224, T: int = 4, iters: int = 100) -> float:
    model.eval()
    x = torch.randn(1, T, 3, img_size, img_size, device=device)

    for _ in range(10):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    import time
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    sec = max(1e-9, (t1 - t0))
    return iters / sec
