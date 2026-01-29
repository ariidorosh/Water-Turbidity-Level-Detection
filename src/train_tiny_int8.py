# train_tiny_int8.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision.models import mobilenet_v3_small

from .core import (
    PathsCfg, TrainCfg,
    get_device,
    FolderImageDataset,
    stratified_split_indices,
    make_weighted_sampler,
    eval_epoch,
)

# =========================
# Models
# =========================
def build_teacher(num_classes: int) -> nn.Module:
    m = mobilenet_v3_small(weights=None)  # ваги прийдуть з ckpt
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def build_student(num_classes: int, width_mult: float) -> nn.Module:
    m = mobilenet_v3_small(weights=None, width_mult=width_mult)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


# =========================
# Distillation
# =========================
def distill_loss(student_logits, teacher_logits, y, alpha: float, T: float) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, y)
    kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    return (1 - alpha) * ce + alpha * kd


# =========================
# Checkpoint utils
# =========================
def save_ckpt_minimal(path: Path, model: nn.Module, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "classes": classes},
        str(path),
        _use_new_zipfile_serialization=True,
    )


def save_ckpt_fp32_student(path: Path, model: nn.Module, classes: list[str], width_mult: float) -> None:
    """
    FP32 (float) tiny checkpoint для fallback/eval, коли INT8 не запускається на системі.
    ВАЖЛИВО: model має бути НЕ quantized (звичайний float).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "classes": classes,
            "width_mult": float(width_mult),
        },
        str(path),
        _use_new_zipfile_serialization=True,
    )


def sizeof_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def load_teacher_ckpt(ckpt_path: Path, device: str) -> tuple[nn.Module, list[str]]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    classes = ckpt["classes"]
    model = build_teacher(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, classes


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Train tiny INT8 student from teacher (distill + QAT)")
    p.add_argument("--data", type=str, default=str(PathsCfg().data_root), help="Path to ds_augmantation root")

    # teacher (велика модель, яку НЕ тренимо заново)
    p.add_argument("--teacher", type=str, default=str(PathsCfg().ckpt_dir / "single_best.pt"), help="Teacher checkpoint")

    # output (INT8)
    p.add_argument("--out", type=str, default=str(PathsCfg().ckpt_dir / "single_tiny_int8.pt"), help="Output INT8 checkpoint")

    # training
    p.add_argument("--epochs", type=int, default=40, help="Student distillation epochs")
    p.add_argument("--qat", type=int, default=2, help="QAT fine-tune epochs (CPU)")
    p.add_argument("--batch", type=int, default=TrainCfg().batch_size)
    p.add_argument("--lr", type=float, default=3e-4, help="Student lr")
    p.add_argument("--lr_qat", type=float, default=1e-4, help="QAT lr")

    # data
    p.add_argument("--img", type=int, default=TrainCfg().img_size)
    p.add_argument("--val", type=float, default=TrainCfg().val_ratio)
    p.add_argument("--seed", type=int, default=TrainCfg().seed)
    p.add_argument("--workers", type=int, default=TrainCfg().num_workers)

    # distill knobs
    p.add_argument("--alpha", type=float, default=0.6, help="Distillation weight (0..1)")
    p.add_argument("--T", type=float, default=2.0, help="Temperature")

    # size knob
    p.add_argument("--width", type=float, default=0.35, help="Student width_mult (try 0.35, 0.30, 0.25)")
    p.add_argument("--max_mb", type=float, default=1.2, help="Desired max checkpoint size in MB")
    return p.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()

    data_root = Path(args.data)
    teacher_path = Path(args.teacher)
    out_path = Path(args.out)

    device = get_device()
    print("Device:", device)
    print("Teacher:", teacher_path)
    print("Out (INT8):", out_path)

    # dataset
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

    # load teacher
    teacher, classes = load_teacher_ckpt(teacher_path, device=device)
    num_classes = len(classes)
    print("Classes:", classes)

    # build student
    student = build_student(num_classes=num_classes, width_mult=float(args.width)).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    best_val = -1.0
    best_state = None

    # 1) Distillation training (GPU/CPU залежно від device)
    for ep in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"student ep {ep}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                t_logits = teacher(x)

            opt.zero_grad(set_to_none=True)
            s_logits = student(x)

            loss = distill_loss(s_logits, t_logits, y, alpha=float(args.alpha), T=float(args.T))
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            pbar.set_postfix(loss=total_loss / max(1, total))

        val_loss, val_acc = eval_epoch(student, val_loader, device)
        print(f"EP {ep}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            print("best updated")

    print("Done. best_val_acc =", best_val)

    if best_state is not None:
        student.load_state_dict(best_state, strict=True)

    # =========================
    # ✅ SAVE FP32 tiny fallback (перед quantization!)
    # =========================
    fp32_path = out_path.with_name("single_tiny_fp32.pt")
    student_fp32_cpu = student.to("cpu").eval()
    save_ckpt_fp32_student(fp32_path, student_fp32_cpu, classes, width_mult=float(args.width))
    print("saved fp32 tiny:", fp32_path)

    # =========================
    # 2) QAT + INT8 convert (тільки CPU)
    # =========================
    student_cpu = student_fp32_cpu  # вже на cpu
    teacher_cpu = teacher.to("cpu").eval()

    torch.backends.quantized.engine = "fbgemm"
    student_cpu.train()
    student_cpu.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    torch.ao.quantization.prepare_qat(student_cpu, inplace=True)

    opt_q = torch.optim.AdamW(student_cpu.parameters(), lr=args.lr_qat)

    for ep in range(1, args.qat + 1):
        total_loss = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"qat ep {ep}", leave=False)
        for x, y in pbar:
            x = x.to("cpu")
            y = y.to("cpu")

            with torch.no_grad():
                t_logits = teacher_cpu(x)

            opt_q.zero_grad(set_to_none=True)
            s_logits = student_cpu(x)
            loss = distill_loss(s_logits, t_logits, y, alpha=float(args.alpha), T=float(args.T))
            loss.backward()
            opt_q.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            pbar.set_postfix(loss=total_loss / max(1, total))

    student_cpu.eval()
    torch.ao.quantization.convert(student_cpu, inplace=True)

    # save INT8
    save_ckpt_minimal(out_path, student_cpu, classes)
    mb = sizeof_mb(out_path)

    print("saved int8:", out_path)
    print("checkpoint size:", f"{mb:.3f} MB")

    if mb > float(args.max_mb):
        print(
            f"\n⚠️ Не влізло в {args.max_mb}MB.\n"
            "Спробуй:\n"
            f"  1) --width {args.width} -> 0.30 або 0.25\n"
            "  2) якщо просяде точність — підніми --alpha (0.6->0.7) або --T (2->3)\n"
            "  3) або +2..4 епохи distill (epochs), QAT зазвичай лишається 2\n"
        )


if __name__ == "__main__":
    main()
