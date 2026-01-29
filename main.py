# main.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import gradio as gr

from src.infer_video import run_infer_video
from src.video_core import get_device


APP_TITLE = "MLWeek26 — Класифікатор мутності води (відео)"


def _safe_stem(p: str) -> str:
    stem = Path(p).stem
    stem = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in stem)
    return stem[:80] if len(stem) > 80 else stem


def process_video(
    video_path: str | None,
    model_choice: str,
    every_n: int,
    img_size: int,
    ckpt_dir: str,
):
    if not video_path:
        return None, gr.update(value=None, visible=False), "", "Завантаж відео зліва."

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(video_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = outputs_dir / f"{stem}_{model_choice}_viz_{ts}.mp4"

    try:
        out_video, used_mode = run_infer_video(
            video_path=video_path,
            out_path=str(out_path),
            img=int(img_size),
            every=max(1, int(every_n)),
            ckpt_dir=ckpt_dir,
            model_choice=model_choice,
        )
        status = f"Готово (модель: {used_mode})"
        return out_video, gr.update(value=out_video, visible=True), used_mode, status
    except Exception as e:
        return None, gr.update(value=None, visible=False), "", f"Помилка: {e}"


def build_ui():
    device = get_device()

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            f"1) Завантаж відео зліва  \n"
            f"2) Прогін стартує автоматично  \n"
            f"3) Дивись результат справа і (якщо треба) завантаж нижче\n\n"
            f"Пристрій: `{device}`"
        )

        with gr.Row():
            in_video = gr.Video(
                label="Вхідне відео",
                elem_id="in_video",
                format="mp4",
            )
            out_video = gr.Video(
                label="Результат (панель ймовірностей + вибраний клас)",
                elem_id="out_video",
                format="mp4",
                interactive=False,
            )

        with gr.Accordion("Налаштування (можна не чіпати)", open=False):
            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=[
                        ("Single Best (single_best.pt)", "single_best"),
                        ("Team model A (turbidity_net_optimized (1).pth)", "turbidity_net_optimized"),
                        ("Team model B (best_turbidity_model.pkl)", "best_turbidity_model_pkl"),
                    ],
                    value="single_best",
                    label="Модель",
                )
                every_n = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Обробляти кожен N-й кадр (швидше)",
                )
                img_size = gr.Dropdown(
                    choices=[224, 192, 160],
                    value=224,
                    label="Розмір інпуту (img)",
                )
                ckpt_dir = gr.Textbox(
                    value="checkpoints",
                    label="Папка з вагами (checkpoints)",
                )

        run_btn = gr.Button("Перезапустити (якщо змінив налаштування)", variant="primary")

        with gr.Group():
            download_btn = gr.DownloadButton(
                label="Завантажити результат",
                value=None,
                visible=False,
                elem_id="download_btn",
            )
            used_model = gr.Textbox(label="Яку модель використано", interactive=False)
            status = gr.Textbox(label="Статус", interactive=False)

        gr.Markdown("Результати зберігаються в папці `outputs/`.")

        in_video.change(
            fn=process_video,
            inputs=[in_video, model_choice, every_n, img_size, ckpt_dir],
            outputs=[out_video, download_btn, used_model, status],
            queue=True,
        )

        run_btn.click(
            fn=process_video,
            inputs=[in_video, model_choice, every_n, img_size, ckpt_dir],
            outputs=[out_video, download_btn, used_model, status],
            queue=True,
        )

    return demo


CSS = """
#in_video, #out_video {
    height: 420px !important;
    max-height: 420px !important;
    min-height: 420px !important;
}
#in_video video, #out_video video {
    height: 420px !important;
    max-height: 420px !important;
    object-fit: contain;
    background: #111;
}
#download_btn { margin-top: 10px; }
"""


if __name__ == "__main__":
    app = build_ui()
    app.launch(css=CSS)
