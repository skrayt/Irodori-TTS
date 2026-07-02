#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from gradio_common import (
    CUSTOM_CSS,
    DEFAULT_MAX_SECONDS,
    MAX_GRADIO_CANDIDATES,
    attach_emoji_palette,
    build_candidate_audio_grid,
    build_history_panel,
    build_model_control_row,
    build_runtime_key,
    build_runtime_settings_row,
    candidate_audio_updates,
    default_checkpoint,
    format_timings,
    make_stage_logger,
    parse_optional_float,
    parse_optional_int,
    parse_seed,
    resolve_decode_mode,
    save_generation_outputs,
)
from irodori_tts.inference_runtime import SamplingRequest, get_cached_runtime

LOG_PREFIX = "gradio-caption"
OUTPUT_DIR = Path("gradio_outputs_voicedesign")
DEFAULT_CHECKPOINT_FALLBACK = "Aratako/Irodori-TTS-500M-v2-VoiceDesign"

TEXT_CAPTION_EXAMPLES = [
    [
        "いらっしゃいませ！本日のおすすめは、こちらの新作スイーツです！😊",
        "元気で明るい若い女性の声。ハキハキとした接客口調。",
    ],
    [
        "……もう夜も遅いから、そろそろ寝ようか。😮‍💨",
        "落ち着いた低めの男性の声。囁くように優しく話す。",
    ],
    [
        "え、ほんとに？やったー！🎵",
        "無邪気で高めの少女の声。喜びに満ちた話し方。",
    ],
]


def _build_key(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
):
    return build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
        log_prefix=LOG_PREFIX,
    )


def _describe_runtime(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> str:
    runtime_key = _build_key(
        checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark
    )
    runtime, reloaded = get_cached_runtime(runtime_key)
    status = (
        "loaded model into memory" if reloaded else "model already loaded; reused existing runtime"
    )
    notes: list[str] = []
    if not runtime.model_cfg.use_caption_condition:
        notes.append(
            "warning: this checkpoint does not enable caption conditioning. Use gradio_app.py for reference-audio inference."
        )
    if runtime.model_cfg.use_speaker_condition:
        notes.append(
            "info: this checkpoint still supports speaker conditioning, but this UI always runs without reference audio."
        )
    return "\n".join(
        [
            status,
            f"checkpoint: {runtime_key.checkpoint}",
            f"model_device: {runtime_key.model_device}",
            f"model_precision: {runtime_key.model_precision}",
            f"codec_device: {runtime_key.codec_device}",
            f"codec_precision: {runtime_key.codec_precision}",
            f"use_caption_condition: {runtime.model_cfg.use_caption_condition}",
            f"use_speaker_condition: {runtime.model_cfg.use_speaker_condition}",
            *notes,
        ]
    )


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
    text: str,
    caption: str,
    seconds: float,
    num_steps: int,
    num_candidates: int,
    seed_value: float | None,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    decode_mode: str,
    max_text_len_raw: str,
    max_caption_len_raw: str,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[object, ...]:
    progress(0.0, desc="モデルを準備中...")
    stdout_log = make_stage_logger(progress)

    text_value = str(text).strip()
    caption_value = str(caption).strip()
    if text_value == "":
        raise gr.Error("Text を入力してください。")
    requested_candidates = int(num_candidates)
    if not (1 <= requested_candidates <= MAX_GRADIO_CANDIDATES):
        raise gr.Error(f"Num Candidates は 1〜{MAX_GRADIO_CANDIDATES} で指定してください。")

    cfg_scale = parse_optional_float(cfg_scale_raw, "CFG Scale Override")
    max_text_len = parse_optional_int(max_text_len_raw, "Max Text Len")
    max_caption_len = parse_optional_int(max_caption_len_raw, "Max Caption Len")
    truncation_factor = parse_optional_float(truncation_factor_raw, "Truncation Factor")
    rescale_k = parse_optional_float(rescale_k_raw, "Rescale k")
    rescale_sigma = parse_optional_float(rescale_sigma_raw, "Rescale sigma")
    seed = parse_seed(seed_value)

    runtime_key = _build_key(
        checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark
    )
    runtime, reloaded = get_cached_runtime(runtime_key)
    if not runtime.model_cfg.use_caption_condition:
        raise gr.Error(
            "このチェックポイントは caption conditioning に対応していません。リファレンス音声モデルには gradio_app.py を使用してください。"
        )

    decode_mode_value = resolve_decode_mode(
        decode_mode, codec_device=codec_device, num_candidates=requested_candidates
    )
    stdout_log(f"[{LOG_PREFIX}] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        (
            "[{}] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "watermark={} mode={} seconds={} steps={} seed={} candidates={} decode_mode={}"
        ).format(
            LOG_PREFIX,
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            enable_watermark,
            cfg_guidance_mode,
            seconds,
            num_steps,
            "random" if seed is None else seed,
            requested_candidates,
            decode_mode_value,
        )
    )
    stdout_log(
        "[{}] conditioning: text={} caption={}".format(
            LOG_PREFIX,
            "on" if text_value else "off",
            "on" if caption_value else "off (text-only)",
        )
    )
    progress(0.05, desc="テキストをトークナイズ中...")

    result = runtime.synthesize(
        SamplingRequest(
            text=text_value,
            caption=caption_value or None,
            ref_wav=None,
            ref_latent=None,
            no_ref=True,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=requested_candidates,
            decode_mode=decode_mode_value,
            seconds=float(seconds),
            max_ref_seconds=30.0,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            num_steps=int(num_steps),
            seed=seed,
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_caption=float(cfg_scale_caption),
            cfg_scale_speaker=0.0,
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            trim_tail=True,
        ),
        log_fn=stdout_log,
    )

    out_paths = save_generation_outputs(
        out_dir=OUTPUT_DIR,
        result=result,
        text=text_value,
        caption=caption_value,
        extra_metadata={
            "seconds": float(seconds),
            "num_steps": int(num_steps),
            "cfg_guidance_mode": str(cfg_guidance_mode),
            "cfg_scale_text": float(cfg_scale_text),
            "cfg_scale_caption": float(cfg_scale_caption),
        },
    )

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {result.used_seed}",
        f"candidates: {len(result.audios)}",
        f"decode_mode: {decode_mode_value}",
        *[f"saved[{i}]: {path}" for i, path in enumerate(out_paths, start=1)],
        *result.messages,
    ]
    if runtime.model_cfg.use_speaker_condition:
        detail_lines.append(
            "info: speaker conditioning exists in this checkpoint, but this UI forced no-reference mode."
        )
    detail_text = "\n".join(detail_lines)
    timing_text = format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[{LOG_PREFIX}] saved {len(out_paths)} candidates")

    return (*candidate_audio_updates(out_paths), detail_text, timing_text)


def build_ui() -> gr.Blocks:
    default_ckpt = default_checkpoint(
        fallback=DEFAULT_CHECKPOINT_FALLBACK,
        prefer_keywords=("caption", "voice_design"),
    )

    with gr.Blocks(title="Irodori-TTS VoiceDesign Gradio", css=CUSTOM_CSS) as demo:
        gr.Markdown("# Irodori-TTS VoiceDesign Inference")
        gr.Markdown(
            "VoiceDesign版モデル向けのUIです。caption を入れると caption / style conditioning、空欄なら text-only conditioning で推論します。"
        )

        (
            checkpoint,
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            enable_watermark,
        ) = build_runtime_settings_row(default_ckpt)
        runtime_inputs = [
            checkpoint,
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            enable_watermark,
        ]
        build_model_control_row(load_fn=_describe_runtime, runtime_inputs=runtime_inputs)

        text = gr.Textbox(label="Text", lines=4)
        attach_emoji_palette(text)

        caption = gr.Textbox(
            label="Caption / Style Prompt (optional)",
            lines=4,
        )
        gr.Examples(examples=TEXT_CAPTION_EXAMPLES, inputs=[text, caption], label="入力例")

        with gr.Accordion("Sampling", open=True):
            with gr.Row():
                seconds = gr.Slider(
                    label="Max Seconds (生成する最大の長さ。短いほど高速)",
                    minimum=1.0,
                    maximum=DEFAULT_MAX_SECONDS,
                    value=DEFAULT_MAX_SECONDS,
                    step=1.0,
                )
                num_steps = gr.Slider(label="Num Steps", minimum=1, maximum=120, value=20, step=1)
                num_candidates = gr.Slider(
                    label="Num Candidates",
                    minimum=1,
                    maximum=MAX_GRADIO_CANDIDATES,
                    value=1,
                    step=1,
                )
                seed_value = gr.Number(
                    label="Seed (-1 または空欄でランダム)", value=-1, precision=0
                )

            with gr.Row():
                cfg_guidance_mode = gr.Dropdown(
                    label="CFG Guidance Mode",
                    choices=["independent", "joint", "alternating"],
                    value="independent",
                )
                cfg_scale_text = gr.Slider(
                    label="CFG Scale Text",
                    minimum=0.0,
                    maximum=10.0,
                    value=2.0,
                    step=0.1,
                )
                cfg_scale_caption = gr.Slider(
                    label="CFG Scale Caption",
                    minimum=0.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.1,
                )

        with gr.Accordion("Advanced (Optional)", open=False):
            cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
            with gr.Row():
                cfg_min_t = gr.Number(label="CFG Min t", value=0.5)
                cfg_max_t = gr.Number(label="CFG Max t", value=1.0)
                context_kv_cache = gr.Checkbox(label="Context KV Cache", value=True)
                decode_mode = gr.Dropdown(
                    label="Decode Mode",
                    choices=["auto", "sequential", "batch"],
                    value="auto",
                )
            with gr.Row():
                max_text_len_raw = gr.Textbox(label="Max Text Len (optional)", value="")
                max_caption_len_raw = gr.Textbox(label="Max Caption Len (optional)", value="")
            with gr.Row():
                truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")

        generate_btn = gr.Button("Generate", variant="primary")

        out_audios = build_candidate_audio_grid()
        out_log = gr.Textbox(label="Run Log", lines=8)
        out_timing = gr.Textbox(label="Timing", lines=8)

        refresh_history, history_outputs = build_history_panel(
            out_dir=OUTPUT_DIR, text_box=text, seed_box=seed_value, caption_box=caption
        )

        generate_btn.click(
            lambda: gr.update(interactive=False),
            outputs=[generate_btn],
        ).then(
            _run_generation,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                enable_watermark,
                text,
                caption,
                seconds,
                num_steps,
                num_candidates,
                seed_value,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_caption,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                decode_mode,
                max_text_len_raw,
                max_caption_len_raw,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
            ],
            outputs=[*out_audios, out_log, out_timing],
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[generate_btn],
        ).then(
            refresh_history,
            outputs=history_outputs,
        )

        demo.load(refresh_history, outputs=history_outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradio app for caption-conditioned Irodori-TTS checkpoints."
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
        show_error=True,
    )


if __name__ == "__main__":
    main()
