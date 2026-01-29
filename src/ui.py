# ui.py
from __future__ import annotations

from pathlib import Path
import time

import gradio as gr

from .infer_video import run_infer_video
from .video_core import get_device


PROJECT_ROOT = Path(".")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = PROJECT_ROOT / "checkpoints"


def _safe_name(stem: str) -> str:
    keep = []
    for ch in stem:
        if ch.isalnum() or ch in ("_", "-", " "):
            keep.append(ch)
    return "".join(keep).strip().replace(" ", "_") or "video"


def process_video(video_in, mode: str, every: int, img: int):
    if video_in is None:
        raise gr.Error("Завантаж відео.")

    in_path = Path(video_in)
    if not in_path.exists():
        raise gr.Error("Не бачу відео-файл. Спробуй завантажити ще раз.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"{_safe_name(in_path.stem)}_{mode}_{stamp}.mp4"
    out_path = OUTPUT_DIR / out_name

    out_path_str, used = run_infer_video(
        video_path=in_path,
        out_path=out_path,
        img=img,
        every=every,
        ckpt_dir=CKPT_DIR,
        force=mode,
    )

    status = f"Готово ✅ (модель: {used})"
    # Повертаємо і для плеєра, і для скачування
    return out_path_str, out_path_str, used, status


def build_ui():
    device = get_device()

    with gr.Blocks(title="MLWeek26 — Відео класифікатор мутності води") as demo:
        gr.Markdown(
            f"""
# MLWeek26 — Класифікатор мутності води (відео)

1) Завантаж відео  
2) Натисни **Запустити**  
3) Дивись результат справа або скачай файл  

**Пристрій:** `{device}`
"""
        )

        with gr.Row():
            video_in = gr.Video(label="Вхідне відео", format="mp4")
            video_out = gr.Video(label="Результат", format="mp4")

        with gr.Accordion("Налаштування (можна не чіпати)", open=False):
            mode = gr.Radio(
                choices=[
                    ("Авто (рекомендовано)", "auto"),
                    ("T=4 модель (video_best.pt)", "t4"),
                    ("Single модель (single_best.pt + усереднення)", "single"),
                ],
                value="auto",
                label="Режим",
            )
            every = gr.Slider(1, 10, value=1, step=1, label="Прискорення: обробляти кожен N-й кадр")
            img = gr.Dropdown([160, 192, 224, 256], value=224, label="Розмір кадру для моделі")

        run_btn = gr.Button("Запустити", variant="primary")

        download = gr.File(label="Скачати результат")
        used_model = gr.Textbox(label="Яку модель використано", interactive=False)
        status = gr.Textbox(label="Статус", interactive=False)

        run_btn.click(
            fn=process_video,
            inputs=[video_in, mode, every, img],
            outputs=[video_out, download, used_model, status],
        )

        gr.Markdown("Результати зберігаються в папці **outputs/**")

    return demo


def main():
    demo = build_ui()
    # allowed_paths інколи допомагає, якщо Gradio капризнічає з локальними файлами
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, allowed_paths=["outputs"])


if __name__ == "__main__":
    main()
