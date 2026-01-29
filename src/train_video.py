# train_video.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .video_core import (
    MultiCfg, get_device,
    MultiImage4Dataset, VideoMobileNetMean,
    model_size_mb, estimate_fps,
)


def parse_args():
    p = argparse.ArgumentParser("Train Complexity-3 (T=4) using ONLY images (pseudo-video samples)")
    p.add_argument("--images", type=str, default=str(MultiCfg().images_root))
    p.add_argument("--out", type=str, default="checkpoints/video_best.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--spc", type=int, default=1200, help="samples per class per epoch (dataset size knob)")
    return p.parse_args()


@torch.no_grad()
def eval_epoch(model, loader, device: str):
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


def main():
    args = parse_args()
    device = get_device()
    print("Device:", device)

    cfg = MultiCfg(
        images_root=Path(args.images),
        img_size=args.img,
        samples_per_class=args.spc
    )

    ds = MultiImage4Dataset(cfg)

    # простий random split (на марафон норм)
    n = len(ds)
    n_val = max(1, int(n * args.val))
    idx = torch.randperm(n).tolist()
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=(device == "cuda"))

    model = VideoMobileNetMean(num_classes=len(ds.classes), pretrained=(not args.no_pretrained)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Model size (MB):", round(model_size_mb(model), 2))
    print("FPS (rough):", round(estimate_fps(model, device, img_size=args.img, T=4, iters=50), 2))
    print("Classes order:", ds.classes)

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"train ep {ep}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            pbar.set_postfix(loss=total_loss / max(1, total))

        vloss, vacc = eval_epoch(model, val_loader, device)
        print(f"EP {ep}: val_loss={vloss:.4f} val_acc={vacc:.4f}")

        if vacc > best_acc:
            best_acc = vacc
            torch.save({"model": model.state_dict(), "classes": ds.classes}, str(out_path))
            print("saved:", out_path)

    print("Done. best_val_acc =", best_acc)


if __name__ == "__main__":
    main()
