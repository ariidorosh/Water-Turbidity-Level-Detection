# train.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .core import (
    PathsCfg, TrainCfg,
    get_device, build_mobilenet,
    FolderImageDataset,
    stratified_split_indices,
    make_weighted_sampler,
    eval_epoch,
    save_ckpt,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Complexity-2 model (single frame classifier)")
    p.add_argument("--data", type=str, default=str(PathsCfg().data_root), help="Path to ds_augmantation root")
    p.add_argument("--out", type=str, default=str(PathsCfg().ckpt_dir / "single_best.pt"), help="Checkpoint output path")
    p.add_argument("--epochs", type=int, default=TrainCfg().epochs)
    p.add_argument("--batch", type=int, default=TrainCfg().batch_size)
    p.add_argument("--lr", type=float, default=TrainCfg().lr)
    p.add_argument("--img", type=int, default=TrainCfg().img_size)
    p.add_argument("--val", type=float, default=TrainCfg().val_ratio)
    p.add_argument("--seed", type=int, default=TrainCfg().seed)
    p.add_argument("--workers", type=int, default=TrainCfg().num_workers)
    p.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet pretrained weights")
    return p.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data)
    out_path = Path(args.out)

    device = get_device()
    print("Device:", device)

    ds = FolderImageDataset(data_root, img_size=args.img)
    train_idx, val_idx = stratified_split_indices(ds.samples, val_ratio=args.val, seed=args.seed)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    sampler = make_weighted_sampler(train_idx, ds.samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )

    model = build_mobilenet(num_classes=len(ds.classes), pretrained=(not args.no_pretrained)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_val = -1.0

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

        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f"EP {ep}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            save_ckpt(out_path, model, ds.classes)
            print("saved:", out_path)

    print("Done. best_val_acc =", best_val)
    print("Classes order:", ds.classes)


if __name__ == "__main__":
    main()
