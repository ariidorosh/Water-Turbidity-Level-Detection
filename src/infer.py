# infer.py
from __future__ import annotations

import argparse
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .core import (
    get_device,
    load_ckpt,
    make_transform,
    pretty_class_name,
    GradCAM,
)


def parse_args():
    p = argparse.ArgumentParser(description="Inference on video with single-frame model (+win4 smoothing + optional Grad-CAM)")
    p.add_argument("--ckpt", type=str, default="checkpoints/single_best.pt", help="Path to checkpoint")
    p.add_argument("--video", type=str, required=True, help="Input video path")
    p.add_argument("--out", type=str, default="out.mp4", help="Output video path")
    p.add_argument("--mode", choices=["single", "win4"], default="win4", help="single=1 frame; win4=avg probs over last 4 frames")
    p.add_argument("--gradcam", action="store_true", help="Overlay Grad-CAM heatmap (uses current frame)")
    p.add_argument("--every", type=int, default=1, help="Process every N-th frame (CPU спасалка). 1=all frames")
    p.add_argument("--img", type=int, default=224, help="Input size for model")
    return p.parse_args()


def overlay_heatmap(frame_bgr, cam01: np.ndarray):
    h, w = frame_bgr.shape[:2]
    cam = cv2.resize((cam01 * 255).astype(np.uint8), (w, h))
    heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 0.65, heat, 0.35, 0)


def preprocess_frame(frame_bgr, tf, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = tf(rgb)  # tf expects PIL/np? In our tf: Resize expects PIL, so wrap carefully
    return x


def main():
    args = parse_args()
    device = get_device()
    print("Device:", device)

    model, classes = load_ckpt(Path(args.ckpt), device=device)

    # transform for numpy -> PIL safe
    tf = make_transform(args.img)

    # Grad-CAM init (якщо треба)
    cam_tool = None
    if args.gradcam:
        cam_tool = GradCAM(model, target_layer=model.features[-1])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не можу відкрити відео: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, H))

    prob_buf = deque(maxlen=4)
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_i += 1
        if args.every > 1 and (frame_i % args.every != 0):
            # просто записуємо як є, щоб вихід мав ту ж довжину
            writer.write(frame)
            continue

        # preprocess: потрібно PIL, тому робимо через Image.fromarray
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = tf(pil).unsqueeze(0).to(device)  # (1,3,H,W)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        if args.mode == "single":
            used_probs = probs
        else:
            prob_buf.append(probs)
            used_probs = np.mean(prob_buf, axis=0)

        cls_idx = int(np.argmax(used_probs))
        cls_name = pretty_class_name(classes[cls_idx])

        vis = frame
        if args.gradcam and cam_tool is not None:
            cam01, _ = cam_tool(x, class_idx=cls_idx)  # cam for chosen class
            vis = overlay_heatmap(frame, cam01)

        text = f"{args.mode.upper()} {cls_name} | " + " ".join(
            [f"{pretty_class_name(classes[i])}:{used_probs[i]:.2f}" for i in range(len(classes))]
        )
        cv2.putText(vis, text[:150], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        writer.write(vis)

    cap.release()
    writer.release()

    if cam_tool is not None:
        cam_tool.remove()

    print("Saved:", args.out)


if __name__ == "__main__":
    main()
