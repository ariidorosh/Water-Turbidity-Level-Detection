# infer_video.py
from __future__ import annotations

import argparse
from pathlib import Path
from collections import deque
import shutil
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .core import (
    load_ckpt as load_single_ckpt,
    make_transform as make_single_transform,
    pretty_class_name as pretty_single_name,
)
from .video_core import (
    get_device,
    VideoMobileNetMean,
    make_transform as make_video_transform,
    pretty_class_name as pretty_video_name,
)

CONF_TH = 0.45  # поріг впевненості


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        "Video inference (prefers video_best.pt, falls back to single_best.pt with win4 averaging)"
    )
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out", type=str, default="out_video.mp4")
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--every", type=int, default=1, help="process every Nth frame (speed)")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument(
        "--force",
        choices=["auto", "t4", "single"],
        default="auto",
        help="auto = use video_best if exists else single_best; t4 = force video_best; single = force single_best(win4)",
    )
    return p.parse_args()


# =========================
# MODELS
# =========================
def _load_video_model(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    classes = ckpt["classes"]
    model = VideoMobileNetMean(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, classes


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
        ffmpeg, "-y",
        "-i", str(raw_mp4),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(final_mp4),
    ]

    try:
        subprocess.run(cmd, check=True)
        raw_mp4.unlink(missing_ok=True)
        return final_mp4
    except Exception:
        return raw_mp4


# =========================
# DRAW (PIL text for Cyrillic)
# =========================
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _get_font(size: int):
    """TrueType шрифт з кирилицею. Працює на Windows."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        # DejaVu (часто є, якщо встановлювався разом з Python/пакетами)
        "DejaVuSans.ttf",
    ]

    font = None
    for p in candidates:
        try:
            font = ImageFont.truetype(p, size=size)
            break
        except Exception:
            continue

    if font is None:
        # fallback (може не мати кирилиці, але хоч не впаде)
        font = ImageFont.load_default()

    _FONT_CACHE[size] = font
    return font


def _bgr(hex_rgb: str):
    h = hex_rgb.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def draw_text_pil(frame, text: str, xy: tuple[int, int], size: int = 20, fill=(255, 255, 255), stroke_fill=(0, 0, 0), stroke_width: int = 2):
    """Малює текст (в т.ч. кирилицю) на кадрі через PIL."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _get_font(size)
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
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

    # colors
    c_bg = _bgr("#000000")
    c_border = _bgr("#FFFFFF")
    c_bar_bg = _bgr("#2B2B2B")

    # щоб текст НЕ зникав — fill не білий
    c_fill = _bgr("#3B82F6")      # синій
    c_fill_hi = _bgr("#22C55E")   # зелений для "підсвічених"

    rows = len(classes)
    panel_h = rows * bar_h + (rows - 1) * gap + 38

    # translucent panel bg
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 12, y0 - 28), (x0 + panel_w + 12, y0 - 28 + panel_h), c_bg, -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # border
    cv2.rectangle(frame, (x0 - 12, y0 - 28), (x0 + panel_w + 12, y0 - 28 + panel_h), c_border, 1)

    # title (PIL text)
    draw_text_pil(frame, "Ймовірності класів", (x0, y0 - 30), size=20)

    # fixed order: 0..4
    for i, cls in enumerate(classes):
        p = float(probs[i])
        y = y0 + i * (bar_h + gap)

        highlight = confident and (i <= chosen_idx)

        # bar background
        cv2.rectangle(frame, (x0, y), (x0 + panel_w, y + bar_h), c_bar_bg, -1)

        # bar fill
        fill_w = int(panel_w * max(0.0, min(1.0, p)))
        if fill_w > 0:
            cv2.rectangle(frame, (x0, y), (x0 + fill_w, y + bar_h), c_fill_hi if highlight else c_fill, -1)

        # bar border
        cv2.rectangle(frame, (x0, y), (x0 + panel_w, y + bar_h), c_border, 1)

        # label (PIL text) — білий з чорним stroke
        label = f"{pretty_fn(cls)}: {p:.2f}"
        draw_text_pil(frame, label, (x0 + 10, y + 2), size=18)

    # footer
    if confident:
        footer = f"Поточний рівень: {pretty_fn(classes[chosen_idx])}  (>= {CONF_TH:.2f})"
    else:
        footer = f"Поточний рівень: не впевнено  (top < {CONF_TH:.2f})"
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
    force: str = "auto",
) -> tuple[str, str]:
    video_path = Path(video_path)
    out_path = Path(out_path)
    ckpt_dir = Path(ckpt_dir)

    device = get_device()

    video_ckpt = ckpt_dir / "video_best.pt"
    single_ckpt = ckpt_dir / "single_best.pt"

    if force == "t4":
        if not video_ckpt.exists():
            raise RuntimeError(f"Ти форсиш t4, але нема {video_ckpt}")
        use_video_model = True
    elif force == "single":
        if not single_ckpt.exists():
            raise RuntimeError(f"Ти форсиш single, але нема {single_ckpt}")
        use_video_model = False
    else:
        use_video_model = video_ckpt.exists()

    if use_video_model:
        model, classes = _load_video_model(video_ckpt, device)
        tf = make_video_transform(img)
        mode_name = "T4_MODEL(video_best.pt)"
        pretty = pretty_video_name
    else:
        if not single_ckpt.exists():
            raise RuntimeError(f"Нема ні {video_ckpt}, ні {single_ckpt}. Нема чим робити інференс.")
        model, classes = load_single_ckpt(single_ckpt, device=device)
        tf = make_single_transform(img)
        mode_name = "WIN4_AVG(single_best.pt)"
        pretty = pretty_single_name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не можу відкрити відео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_out = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Не можу створити вихідний mp4 (raw). Спробуй інший шлях/назву.")

    buf = deque(maxlen=4)
    prob_buf = deque(maxlen=4)

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
        x = tf(pil)
        buf.append(x)

        if len(buf) < 4:
            # щоб не було "????" — також PIL
            draw_text_pil(frame, f"{mode_name}: прогрів (потрібно 4 кадри)...", (20, 25), size=20)
            writer.write(frame)
            continue

        if use_video_model:
            clip = torch.stack(list(buf), dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(clip), dim=1)[0].detach().cpu().numpy()
        else:
            with torch.no_grad():
                p = F.softmax(model(buf[-1].unsqueeze(0).to(device)), dim=1)[0].detach().cpu().numpy()
            prob_buf.append(p)
            probs = np.mean(prob_buf, axis=0)

        idx = int(np.argmax(probs))
        confident = float(probs[idx]) >= CONF_TH

        # панель 5 класів
        draw_probs_panel(
            frame=frame,
            probs=probs,
            classes=classes,
            pretty_fn=pretty,
            chosen_idx=idx,
            confident=confident,
            panel_xy=(20, 60),
        )

        # режим зверху
        draw_text_pil(frame, mode_name, (20, 25), size=20)

        writer.write(frame)

    cap.release()
    writer.release()

    final_out = _encode_h264_for_browser(raw_out, out_path)
    return str(final_out), mode_name


def main():
    args = parse_args()
    print("Device:", get_device())

    out_path, used = run_infer_video(
        video_path=args.video,
        out_path=args.out,
        img=args.img,
        every=args.every,
        ckpt_dir=args.ckpt_dir,
        force=args.force,
    )

    print("Saved:", out_path)
    print("Used:", used)


if __name__ == "__main__":
    main()
