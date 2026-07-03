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

LOG_PREFIX = "gradio"
OUTPUT_DIR = Path("gradio_outputs")
DEFAULT_CHECKPOINT_FALLBACK = "Aratako/Irodori-TTS-500M-v2"

TEXT_EXAMPLES = [
    ["こんにちは、今日はとてもいい天気ですね。😊"],
    ["ねえ、ちょっと待って……⏸️ ほんとに行っちゃうの？🥺"],
    ["はぁ……😮‍💨 今日も一日、よく頑張ったなぁ。🥱"],
]


def _load_model(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> str:
    runtime_key = build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
        log_prefix=LOG_PREFIX,
    )
    _, reloaded = get_cached_runtime(runtime_key)
    if reloaded:
        status = "loaded model into memory"
    else:
        status = "model already loaded; reused existing runtime"
    return (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device}\n"
        f"model_precision: {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device}\n"
        f"codec_precision: {runtime_key.codec_precision}"
    )


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
    text: str,
    uploaded_audio: str | None,
    seconds: float,
    num_steps: int,
    num_candidates: int,
    seed_value: float | None,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    decode_mode: str,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[object, ...]:
    progress(0.0, desc="モデルを準備中...")
    stdout_log = make_stage_logger(progress)

    text_value = str(text).strip()
    if text_value == "":
        raise gr.Error("Text を入力してください。")
    requested_candidates = int(num_candidates)
    if not (1 <= requested_candidates <= MAX_GRADIO_CANDIDATES):
        raise gr.Error(f"Num Candidates は 1〜{MAX_GRADIO_CANDIDATES} で指定してください。")

    cfg_scale = parse_optional_float(cfg_scale_raw, "CFG Scale Override")
    truncation_factor = parse_optional_float(truncation_factor_raw, "Truncation Factor")
    rescale_k = parse_optional_float(rescale_k_raw, "Rescale k")
    rescale_sigma = parse_optional_float(rescale_sigma_raw, "Rescale sigma")
    speaker_kv_scale = parse_optional_float(speaker_kv_scale_raw, "Speaker KV Scale")
    speaker_kv_min_t = parse_optional_float(speaker_kv_min_t_raw, "Speaker KV Min t")
    speaker_kv_max_layers = parse_optional_int(speaker_kv_max_layers_raw, "Speaker KV Max Layers")
    seed = parse_seed(seed_value)

    ref_wav = str(uploaded_audio) if uploaded_audio and str(uploaded_audio).strip() else None
    no_ref = ref_wav is None

    runtime_key = build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
        log_prefix=LOG_PREFIX,
    )
    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[{LOG_PREFIX}] runtime: {'reloaded' if reloaded else 'reused'}")

    decode_mode_value = resolve_decode_mode(
        decode_mode, codec_device=codec_device, num_candidates=requested_candidates
    )
    stdout_log(
        (
            "[{}] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "watermark={} mode={} seconds={} steps={} seed={} no_ref={} candidates={} decode_mode={}"
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
            no_ref,
            requested_candidates,
            decode_mode_value,
        )
    )
    progress(0.05, desc="テキストをトークナイズ中...")

    result = runtime.synthesize(
        SamplingRequest(
            text=text_value,
            ref_wav=ref_wav,
            ref_latent=None,
            no_ref=bool(no_ref),
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=requested_candidates,
            decode_mode=decode_mode_value,
            seconds=float(seconds),
            max_ref_seconds=30.0,
            max_text_len=None,
            num_steps=int(num_steps),
            seed=seed,
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            trim_tail=True,
        ),
        log_fn=stdout_log,
    )

    out_paths = save_generation_outputs(
        out_dir=OUTPUT_DIR,
        result=result,
        text=text_value,
        extra_metadata={
            "seconds": float(seconds),
            "num_steps": int(num_steps),
            "cfg_guidance_mode": str(cfg_guidance_mode),
            "cfg_scale_text": float(cfg_scale_text),
            "cfg_scale_speaker": float(cfg_scale_speaker),
            "ref_audio": ref_wav,
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
    detail_text = "\n".join(detail_lines)
    timing_text = format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[{LOG_PREFIX}] saved {len(out_paths)} candidates")

    return (*candidate_audio_updates(out_paths), detail_text, timing_text)


def build_ui() -> gr.Blocks:
    default_ckpt = default_checkpoint(fallback=DEFAULT_CHECKPOINT_FALLBACK)

    with gr.Blocks(title="Irodori-TTS Gradio", css=CUSTOM_CSS) as demo:
        gr.Markdown("# Irodori-TTS Inference (Cached Runtime)")
        gr.Markdown(
            "When settings are unchanged, runtime is reused and only sampling/decoding runs."
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
        build_model_control_row(load_fn=_load_model, runtime_inputs=runtime_inputs)

        text = gr.Textbox(label="Text", lines=4)
        attach_emoji_palette(text)
        gr.Examples(examples=TEXT_EXAMPLES, inputs=[text], label="入力例")

        uploaded_audio = gr.Audio(
            label="Reference Audio Upload (optional, blank = no-reference mode)",
            type="filepath",
        )

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
                    value=3.0,
                    step=0.1,
                )
                cfg_scale_speaker = gr.Slider(
                    label="CFG Scale Speaker",
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
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
                truncation_factor_raw = gr.Textbox(label="Truncation Factor (optional)", value="")
                rescale_k_raw = gr.Textbox(label="Rescale k (optional)", value="")
                rescale_sigma_raw = gr.Textbox(label="Rescale sigma (optional)", value="")
            with gr.Row():
                speaker_kv_scale_raw = gr.Textbox(label="Speaker KV Scale (optional)", value="")
                speaker_kv_min_t_raw = gr.Textbox(label="Speaker KV Min t (optional)", value="0.9")
                speaker_kv_max_layers_raw = gr.Textbox(
                    label="Speaker KV Max Layers (optional)", value=""
                )

        generate_btn = gr.Button("Generate", variant="primary")

        out_audios = build_candidate_audio_grid()
        out_log = gr.Textbox(label="Run Log", lines=8)
        out_timing = gr.Textbox(label="Timing", lines=8)

        refresh_history, history_outputs = build_history_panel(
            out_dir=OUTPUT_DIR, text_box=text, seed_box=seed_value
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
                uploaded_audio,
                seconds,
                num_steps,
                num_candidates,
                seed_value,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                decode_mode,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
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
    parser = argparse.ArgumentParser(description="Gradio app for Irodori-TTS with cached runtime.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
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
