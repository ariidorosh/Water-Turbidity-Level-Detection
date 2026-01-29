# src/infer_video.py
from __future__ import annotations

import argparse
from pathlib import Path
from collections import deque
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .core import (
    load_ckpt as load_single_ckpt,
    make_transform as make_single_transform,
    pretty_class_name as pretty_single_name,
)
from .video_core import get_device


CONF_TH = 0.45

FEATURE_NAMES_10 = [
    "EntropyShannon",
    "BackProjection",
    "GammaCorrection",
    "HarrisCorner",
    "SharpnessDOM",
    "FFT_Blur",
    "Mean_L",
    "Std_L",
    "Mean_B",
    "Std_B",
]

FEATURES_ORDER = [
    "EntropyShannon",
    "BackProjection",
    "GammaCorrection",
    "HarrisCorner",
    "SharpnessDOM",
    "FFT_Blur",
    "Mean_L",
    "Std_L",
    "Mean_B",
    "Std_B",
]


def parse_args():
    p = argparse.ArgumentParser("Video inference (3 models)")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out", type=str, default="out_video.mp4")
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument(
        "--model",
        choices=["single_best", "turbidity_net_optimized", "best_turbidity_model_pkl"],
        default="single_best",
    )
    return p.parse_args()


def _pick_existing(ckpt_dir: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Не знайшов жодного з файлів: {candidates} у папці {ckpt_dir}")


def _default_classes_5() -> list[str]:
    return ["cl0_augmented", "cl1_augmented", "cl2_augmented", "cl3_augmented", "cl4_augmented"]


# =========================
# Model #2: CIFAR-ResNet-like
# =========================
class _BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class TurbidityNetOptimized(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 2, 1)
        self.layer2 = self._make_layer(32, 2, 2)
        self.layer3 = self._make_layer(64, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(_BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out).flatten(1)
        return self.fc(out)


def _load_turbidity_net_optimized(pth_path: Path, device: str) -> tuple[nn.Module, list[str]]:
    sd = torch.load(str(pth_path), map_location="cpu")
    classes = _default_classes_5()
    model = TurbidityNetOptimized(num_classes=len(classes))
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, classes


# =========================
# Model #3: joblib / imblearn Pipeline
# =========================
def _joblib_load(path: Path) -> Any:
    try:
        import joblib  # type: ignore
    except Exception as e:
        raise RuntimeError("Потрібен joblib (і зазвичай sklearn + imblearn) для best_turbidity_model.pkl") from e

    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(
            f"Не зміг завантажити {path.name} через joblib. "
            "Ймовірно, різні версії sklearn/imblearn або не вистачає залежностей."
        ) from e


# --- Robust features for PKL SVC pipeline ---
def _extract_10_features(pil: Image.Image) -> np.ndarray:
    """
    Витяг 10 фіч для Team model B (best_turbidity_model.pkl).

    КРИТИЧНО:
    - Старий HarrisCorner через cornerHarris+threshold "вибухає" (ти бачив 19817) і дає OOD,
      тоді SVC повертає константні ймовірності.
    - Тут HarrisCorner = кількість кутів через goodFeaturesToTrack з лімітом.
    - SharpnessDOM логарифмуємо (log1p), щоб не було диких стрибків.
    - Повертаємо (1,10) у чіткому порядку FEATURES_ORDER.
    """
    img = pil.convert("RGB")

    # легкий даунскейл для стабільності/швидкості
    w, h = img.size
    target_w = 320
    if w > target_w:
        new_h = int(h * (target_w / w))
        img = img.resize((target_w, new_h))

    rgb = np.asarray(img, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1) EntropyShannon
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1).astype(np.float64)
    s = float(hist.sum())
    if s <= 0:
        ent = 0.0
    else:
        p = hist / s
        p = p[p > 1e-12]
        ent = float(-(p * np.log2(p)).sum())

    # 2) BackProjection (mean 0..255)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hst = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hst, hst, 0, 255, cv2.NORM_MINMAX)
    back = cv2.calcBackProject([hsv], [0, 1], hst, [0, 180, 0, 256], 1)
    bp = float(np.mean(back))

    # 3) GammaCorrection (mean after correction, 0..255)
    gamma = 1.5
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    corrected = cv2.LUT(gray, lut)
    gam = float(np.mean(corrected))

    # 4) HarrisCorner (ROBUST): count of good corners (0..maxCorners)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3,
        useHarrisDetector=True,
        k=0.04,
    )
    har = float(0 if corners is None else len(corners))

    # 5) SharpnessDOM (log1p(laplacian var))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sha = float(np.log1p(lap.var()))

    # 6) FFT_Blur (high/total ratio)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    H, W = gray.shape
    cy, cx = H // 2, W // 2
    r = int(min(H, W) * 0.10)

    mask = np.ones((H, W), dtype=np.float64)
    y0, y1 = max(0, cy - r), min(H, cy + r)
    x0, x1 = max(0, cx - r), min(W, cx + r)
    mask[y0:y1, x0:x1] = 0.0

    high = mag * mask
    fft = float(high.mean() / (mag.mean() + 1e-9))

    # 7..10) LAB stats (OpenCV LAB: 0..255)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[:, :, 0]
    B = lab[:, :, 2]
    mean_L = float(np.mean(L))
    std_L = float(np.std(L))
    mean_B = float(np.mean(B))
    std_B = float(np.std(B))

    x = np.array([ent, bp, gam, har, sha, fft, mean_L, std_L, mean_B, std_B], dtype=np.float64).reshape(1, -1)
    return x


@dataclass
class _PKLWrapper:
    model: Any
    classes: list[str]

    def predict_proba(self, pil: Image.Image) -> np.ndarray:
        x = _extract_10_features(pil)

        # ✅ важливо: передати feature names і порядок, як у тренуванні
        X = x
        try:
            import pandas as pd  # type: ignore
            X = pd.DataFrame(x, columns=FEATURES_ORDER)
        except Exception:
            pass

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            proba = np.asarray(proba, dtype=np.float32)
            return proba[0] if proba.ndim == 2 else proba

        if hasattr(self.model, "decision_function"):
            df = np.asarray(self.model.decision_function(X), dtype=np.float32).reshape(-1)
            e = np.exp(df - np.max(df))
            return e / (np.sum(e) + 1e-9)

        raise RuntimeError("PKL-модель не має ні predict_proba(), ні decision_function().")


def _load_pkl_model(pkl_path: Path) -> tuple[_PKLWrapper, list[str]]:
    obj = _joblib_load(pkl_path)
    classes = _default_classes_5()

    if isinstance(obj, dict):
        if "classes" in obj:
            classes = list(obj["classes"])
        if "model" in obj:
            obj = obj["model"]

    return _PKLWrapper(model=obj, classes=classes), classes


# =========================
# FFMPEG
# =========================
def _get_ffmpeg_exe() -> str | None:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _encode_h264_for_browser(raw_mp4: Path, final_mp4: Path) -> Path:
    ffmpeg = _get_ffmpeg_exe()
    if not ffmpeg:
        return raw_mp4

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_mp4),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_mp4),
    ]
    try:
        subprocess.run(cmd, check=True)
        raw_mp4.unlink(missing_ok=True)
        return final_mp4
    except Exception:
        return raw_mp4


# =========================
# DRAW (PIL Cyrillic)
# =========================
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for p in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "DejaVuSans.ttf",
    ]:
        try:
            f = ImageFont.truetype(p, size=size)
            _FONT_CACHE[size] = f
            return f
        except Exception:
            continue
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def _bgr(hex_rgb: str):
    h = hex_rgb.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def draw_text_pil(
    frame,
    text: str,
    xy: tuple[int, int],
    size: int = 20,
    fill=(255, 255, 255),
    stroke_fill=(0, 0, 0),
    stroke_width: int = 2,
):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(
        xy,
        text,
        font=_get_font(size),
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_probs_panel(
    frame,
    probs: np.ndarray,
    classes: list[str],
    pretty_fn,
    chosen_idx: int,
    confident: bool,
    panel_xy=(20, 60),
    panel_w=460,
    bar_h=22,
    gap=10,
):
    x0, y0 = panel_xy
    H, W = frame.shape[:2]
    panel_w = min(panel_w, W - x0 - 20)

    c_bg = _bgr("#000000")
    c_border = _bgr("#FFFFFF")
    c_bar_bg = _bgr("#2B2B2B")
    c_fill = _bgr("#3B82F6")
    c_fill_hi = _bgr("#22C55E")

    rows = len(classes)
    panel_h = rows * bar_h + (rows - 1) * gap + 38

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 12, y0 - 28), (x0 + panel_w + 12, y0 - 28 + panel_h), c_bg, -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    cv2.rectangle(frame, (x0 - 12, y0 - 28), (x0 + panel_w + 12, y0 - 28 + panel_h), c_border, 1)

    draw_text_pil(frame, "Ймовірності класів", (x0, y0 - 30), size=20)

    for i, cls in enumerate(classes):
        p = float(probs[i])
        y = y0 + i * (bar_h + gap)
        highlight = confident and (i == chosen_idx)

        cv2.rectangle(frame, (x0, y), (x0 + panel_w, y + bar_h), c_bar_bg, -1)
        fill_w = int(panel_w * max(0.0, min(1.0, p)))
        if fill_w > 0:
            cv2.rectangle(frame, (x0, y), (x0 + fill_w, y + bar_h), c_fill_hi if highlight else c_fill, -1)
        cv2.rectangle(frame, (x0, y), (x0 + panel_w, y + bar_h), c_border, 1)

        draw_text_pil(frame, f"{pretty_fn(cls)}: {p:.2f}", (x0 + 10, y + 2), size=18)

    footer = (
        f"Поточний рівень: {pretty_fn(classes[chosen_idx])} (>= {CONF_TH:.2f})"
        if confident
        else f"Поточний рівень: не впевнено (top < {CONF_TH:.2f})"
    )
    draw_text_pil(frame, footer, (x0, y0 + rows * (bar_h + gap) - gap + 12), size=18)


# =========================
# INFERENCE
# =========================
def run_infer_video(
    video_path: str | Path,
    out_path: str | Path,
    img: int = 224,
    every: int = 1,
    ckpt_dir: str | Path = "checkpoints",
    model_choice: str = "single_best",
) -> tuple[str, str]:
    video_path = Path(video_path)
    out_path = Path(out_path)
    ckpt_dir = Path(ckpt_dir)

    device = get_device()
    used_mode = ""

    if model_choice == "single_best":
        ckpt = _pick_existing(ckpt_dir, ["single_best.pt"])
        model, classes = load_single_ckpt(ckpt, device=device)
        tf = make_single_transform(img)
        pretty = pretty_single_name
        used_mode = "SINGLE_BEST(single_best.pt)"
        smooth_len = 4

        def infer_probs(pil_img: Image.Image) -> np.ndarray:
            x = tf(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                return F.softmax(model(x), dim=1)[0].detach().cpu().numpy()

    elif model_choice == "turbidity_net_optimized":
        ckpt = _pick_existing(ckpt_dir, ["turbidity_net_optimized (1).pth", "turbidity_net_optimized (1).pth"])
        model, classes = _load_turbidity_net_optimized(ckpt, device=device)
        tf = make_single_transform(img)
        pretty = pretty_single_name
        used_mode = f"TURBIDITY_NET_OPTIMIZED({ckpt.name})"
        smooth_len = 4

        def infer_probs(pil_img: Image.Image) -> np.ndarray:
            x = tf(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                return F.softmax(model(x), dim=1)[0].detach().cpu().numpy()

    elif model_choice == "best_turbidity_model_pkl":
        ckpt = _pick_existing(ckpt_dir, ["best_turbidity_model.pkl"])
        wrapper, classes = _load_pkl_model(ckpt)
        pretty = pretty_single_name
        used_mode = f"PKL_PIPELINE({ckpt.name})"
        smooth_len = 1  # без згладжування, щоб бачити реальну динаміку

        def infer_probs(pil_img: Image.Image) -> np.ndarray:
            probs = wrapper.predict_proba(pil_img)
            probs = np.asarray(probs, dtype=np.float32).reshape(-1)
            if probs.shape[0] != len(classes):
                probs = probs[: len(classes)] if probs.shape[0] > len(classes) else np.pad(
                    probs, (0, len(classes) - probs.shape[0])
                )
                s = float(probs.sum())
                if s > 0:
                    probs = probs / s
            return probs

    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не можу відкрити відео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out = out_path.with_suffix(".raw.mp4")

    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Не можу створити вихідний mp4 (raw).")

    prob_buf = deque(maxlen=smooth_len)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        i += 1
        if every > 1 and (i % every != 0):
            writer.write(frame)
            continue

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        probs = infer_probs(pil)

        prob_buf.append(probs)
        probs_smooth = probs if smooth_len == 1 else np.mean(prob_buf, axis=0)

        idx = int(np.argmax(probs_smooth))
        confident = float(probs_smooth[idx]) >= CONF_TH

        draw_probs_panel(frame, probs_smooth, classes, pretty, idx, confident, panel_xy=(20, 60))
        draw_text_pil(frame, f"{used_mode} | frame={i}", (20, 25), size=20)

        writer.write(frame)

    cap.release()
    writer.release()

    final_out = _encode_h264_for_browser(raw_out, out_path)
    return str(final_out), used_mode


def main():
    args = parse_args()
    out_path, used = run_infer_video(
        video_path=args.video,
        out_path=args.out,
        img=args.img,
        every=args.every,
        ckpt_dir=args.ckpt_dir,
        model_choice=args.model,
    )
    print("Saved:", out_path)
    print("Used:", used)


if __name__ == "__main__":
    main()
