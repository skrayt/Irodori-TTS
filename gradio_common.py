#!/usr/bin/env python3
"""Shared helpers and UI builders for the Irodori-TTS Gradio apps."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingResult,
    clear_cached_runtime,
    default_runtime_device,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)

MAX_GRADIO_CANDIDATES = 32
GRADIO_AUDIO_COLS_PER_ROW = 4
HISTORY_SIZE = 5
CLEANUP_KEEP_FILES = 50
DEFAULT_MAX_SECONDS = 30.0
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"

CUSTOM_CSS = """
.emoji-scroll {
    max-height: 350px;
    overflow-y: auto;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 8px;
    margin-top: 5px;
}
"""

EMOJI_PALETTE: list[tuple[str, str]] = [
    ("👂", "囁き、耳元の音"), ("😮‍💨", "吐息、溜息、寝息"), ("⏸️", "間、沈黙"),
    ("🤭", "笑い（くすくす等）"), ("🥵", "喘ぎ、うめき声、唸り声"), ("📢", "エコー、リバーブ"),
    ("😏", "からかう、甘える"), ("🥺", "声を震わせる"), ("🌬️", "息切れ、呼吸音"),
    ("😮", "息をのむ"), ("👅", "舐める音、水音"), ("💋", "リップノイズ"),
    ("🫶", "優しく"), ("😭", "嗚咽、悲しみ"), ("😱", "悲鳴、叫び、絶叫"),
    ("😪", "眠そうに、気だるげ"), ("⏩", "早口、急いで"), ("📞", "電話越し風の音"),
    ("🐢", "ゆっくりと"), ("🥤", "唾を飲み込む音"), ("🤧", "咳き込み、鼻すすり"),
    ("😒", "舌打ち"), ("😰", "動揺、緊張、どもり"), ("😆", "喜びながら"),
    ("😠", "怒り、拗ねながら"), ("😲", "驚き、感嘆"), ("🥱", "あくび"),
    ("😖", "苦しげに"), ("😟", "心配そうに"), ("🫣", "照れながら"),
    ("🙄", "呆れたように"), ("😊", "楽しげに"), ("👌", "相槌、頷く音"),
    ("🙏", "懇願するように"), ("🥴", "酔っ払って"), ("🎵", "鼻歌"),
    ("🤐", "口を塞がれて"), ("😌", "安堵、満足げに"), ("🤔", "疑問の声"),
]

_CHECKPOINT_PATTERNS = ("checkpoint_*.pt", "checkpoint_*.safetensors")
# Scanning the whole working tree at startup gets slow once gradio_outputs/
# accumulates files, so only look in likely checkpoint locations.
_CHECKPOINT_SEARCH_DIRS = ("checkpoints", "outputs", "runs")


def default_checkpoint(*, fallback: str, prefer_keywords: Sequence[str] = ()) -> str:
    candidates: list[Path] = []
    for pattern in _CHECKPOINT_PATTERNS:
        candidates.extend(Path(".").glob(pattern))
        for dir_name in _CHECKPOINT_SEARCH_DIRS:
            root = Path(dir_name)
            if root.is_dir():
                candidates.extend(root.glob(f"**/{pattern}"))
    candidates = sorted(set(candidates))
    preferred = [
        path
        for path in candidates
        if any(keyword in str(path).lower() for keyword in prefer_keywords)
    ]
    if preferred:
        return str(preferred[-1])
    if candidates:
        return str(candidates[-1])
    return fallback


def parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        raise gr.Error(f"{label} には数値または空欄を指定してください。") from None


def parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError:
        raise gr.Error(f"{label} には整数または空欄を指定してください。") from None


def parse_seed(raw: float | int | None) -> int | None:
    """Seed from a gr.Number: blank or negative means random."""
    if raw is None:
        return None
    seed = int(raw)
    return None if seed < 0 else seed


def format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)


_HF_RESOLVE_LOCK = threading.Lock()
_HF_RESOLVE_CACHE: dict[str, str] = {}


def resolve_checkpoint_path(raw_checkpoint: str, *, log_prefix: str = "gradio") -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise gr.Error("checkpoint を指定してください。")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    # Cache resolved HF paths so each generation does not hit the network
    # again (and keeps working offline once downloaded).
    with _HF_RESOLVE_LOCK:
        cached = _HF_RESOLVE_CACHE.get(checkpoint)
    if cached is not None and Path(cached).exists():
        return cached

    resolved = str(hf_hub_download(repo_id=checkpoint, filename="model.safetensors"))
    with _HF_RESOLVE_LOCK:
        _HF_RESOLVE_CACHE[checkpoint] = resolved
    print(f"[{log_prefix}] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return resolved


def build_runtime_key(
    *,
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
    log_prefix: str = "gradio",
) -> RuntimeKey:
    checkpoint_path = resolve_checkpoint_path(checkpoint, log_prefix=log_prefix)
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo=CODEC_REPO,
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        enable_watermark=bool(enable_watermark),
        compile_model=False,
        compile_dynamic=False,
    )


def resolve_decode_mode(decode_mode: str, *, codec_device: str, num_candidates: int) -> str:
    mode = str(decode_mode).strip().lower()
    if mode != "auto":
        return mode
    # Batch decode is faster on GPU, but very large batches of long audio can OOM.
    if str(codec_device) == "cuda" and 1 < int(num_candidates) <= 8:
        return "batch"
    return "sequential"


# Runtime stage logs are emitted when a stage *finishes*, so each entry maps to
# the progress fraction reached and the description of the next stage.
_PROGRESS_STAGES: list[tuple[str, float, str]] = [
    ("tokenize_text", 0.2, "リファレンスを準備中..."),
    ("prepare_reference", 0.25, "音声をサンプリング中..."),
    ("sample_rf", 0.85, "音声をデコード中..."),
    ("decode_latent", 0.95, "ファイルを保存中..."),
]


def make_stage_logger(progress: gr.Progress | None) -> Callable[[str], None]:
    def log(msg: str) -> None:
        print(msg, flush=True)
        if progress is None:
            return
        for stage, fraction, desc in _PROGRESS_STAGES:
            if msg.startswith(f"[runtime] {stage}:"):
                progress(fraction, desc=desc)
                break

    return log


def save_generation_outputs(
    *,
    out_dir: Path,
    result: SamplingResult,
    text: str,
    caption: str | None = None,
    extra_metadata: dict | None = None,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metadata: dict[str, object] = {"text": text, "seed": result.used_seed}
    if caption is not None:
        metadata["caption"] = caption
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)

    out_paths: list[str] = []
    for i, audio in enumerate(result.audios, start=1):
        base = out_dir / f"sample_{stamp}_{i:03d}"
        out_path = save_wav(base.with_suffix(".wav"), audio.float(), result.sample_rate)
        out_paths.append(str(out_path))
        base.with_suffix(".json").write_text(metadata_json, encoding="utf-8")
    return out_paths


def cleanup_outputs(out_dir: Path, *, keep: int = CLEANUP_KEEP_FILES) -> str:
    if not out_dir.exists():
        return "削除対象のファイルはありません。"
    wav_files = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for wav_path in wav_files[keep:]:
        for path in (wav_path, wav_path.with_suffix(".json"), wav_path.with_suffix(".txt")):
            if path.exists():
                path.unlink()
        removed += 1
    if removed == 0:
        return f"削除対象はありません（保存件数が {keep} 件以下です）。"
    return f"古い生成結果 {removed} 件を削除しました（最新 {keep} 件を保持）。"


def clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


def build_runtime_settings_row(
    default_checkpoint_value: str,
) -> tuple[gr.Textbox, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.State]:
    device_choices = list_available_runtime_devices()
    default_device = default_runtime_device()
    precision_choices = list_available_runtime_precisions(default_device)

    with gr.Row():
        checkpoint = gr.Textbox(
            label="Checkpoint (.pt/.safetensors or HF repo id)",
            value=default_checkpoint_value,
            scale=4,
        )
        model_device = gr.Dropdown(
            label="Model Device", choices=device_choices, value=default_device, scale=1
        )
        model_precision = gr.Dropdown(
            label="Model Precision",
            choices=precision_choices,
            value=precision_choices[0],
            scale=1,
        )
        codec_device = gr.Dropdown(
            label="Codec Device", choices=device_choices, value=default_device, scale=1
        )
        codec_precision = gr.Dropdown(
            label="Codec Precision",
            choices=precision_choices,
            value=precision_choices[0],
            scale=1,
        )
        enable_watermark = gr.State(False)

    def _on_device_change(device: str) -> gr.Dropdown:
        choices = list_available_runtime_precisions(device)
        return gr.Dropdown(choices=choices, value=choices[0])

    model_device.change(_on_device_change, inputs=[model_device], outputs=[model_precision])
    codec_device.change(_on_device_change, inputs=[codec_device], outputs=[codec_precision])
    return checkpoint, model_device, model_precision, codec_device, codec_precision, enable_watermark


def build_model_control_row(*, load_fn: Callable, runtime_inputs: list) -> gr.Textbox:
    with gr.Row():
        load_model_btn = gr.Button("Load Model")
        clear_cache_btn = gr.Button("Unload Model")
        status = gr.Textbox(label="Model Status", interactive=False)

    load_model_btn.click(
        lambda: ("モデルをロード中... (初回はダウンロードのため数分かかることがあります)", gr.update(interactive=False)),
        outputs=[status, load_model_btn],
    ).then(
        load_fn,
        inputs=runtime_inputs,
        outputs=[status],
    ).then(
        lambda: gr.update(interactive=True),
        outputs=[load_model_btn],
    )
    clear_cache_btn.click(clear_runtime_cache, outputs=[status])
    return status


def attach_emoji_palette(text_box: gr.Textbox) -> None:
    with gr.Accordion("🎨 絵文字パレット (クリックでテキストに追加)", open=False):
        gr.Markdown("タイルをクリックすると、**絵文字のみ**がテキストの末尾に追加されます。")
        with gr.Column(elem_classes="emoji-scroll"):
            for i in range(0, len(EMOJI_PALETTE), 3):
                with gr.Row():
                    for emoji, desc in EMOJI_PALETTE[i : i + 3]:
                        btn = gr.Button(value=f"{emoji} {desc}", size="sm")
                        btn.click(
                            lambda t, e=emoji: (t or "") + e,
                            inputs=[text_box],
                            outputs=[text_box],
                        )


def build_candidate_audio_grid() -> list[gr.Audio]:
    out_audios: list[gr.Audio] = []
    num_rows = (
        MAX_GRADIO_CANDIDATES + GRADIO_AUDIO_COLS_PER_ROW - 1
    ) // GRADIO_AUDIO_COLS_PER_ROW
    with gr.Column():
        for row_idx in range(num_rows):
            with gr.Row():
                for col_idx in range(GRADIO_AUDIO_COLS_PER_ROW):
                    i = row_idx * GRADIO_AUDIO_COLS_PER_ROW + col_idx
                    if i >= MAX_GRADIO_CANDIDATES:
                        break
                    out_audios.append(
                        gr.Audio(
                            label=f"Generated Audio {i + 1}",
                            type="filepath",
                            interactive=False,
                            visible=(i == 0),
                            min_width=160,
                        )
                    )
    return out_audios


def candidate_audio_updates(out_paths: Sequence[str]) -> list[object]:
    updates: list[object] = []
    for i in range(MAX_GRADIO_CANDIDATES):
        if i < len(out_paths):
            updates.append(gr.update(value=out_paths[i], visible=True))
        else:
            updates.append(gr.update(value=None, visible=False))
    return updates


def _list_history_entries(out_dir: Path, limit: int) -> list[dict]:
    if not out_dir.exists():
        return []
    wav_files = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    entries: list[dict] = []
    for wav_path in wav_files[:limit]:
        meta: dict = {}
        json_path = wav_path.with_suffix(".json")
        txt_path = wav_path.with_suffix(".txt")
        if json_path.exists():
            try:
                loaded = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except (json.JSONDecodeError, OSError):
                meta = {}
        elif txt_path.exists():
            meta = {"text": txt_path.read_text(encoding="utf-8")}
        meta["wav"] = str(wav_path)
        entries.append(meta)
    return entries


def build_history_panel(
    *,
    out_dir: Path,
    text_box: gr.Textbox,
    seed_box: gr.Number,
    caption_box: gr.Textbox | None = None,
) -> tuple[Callable[[], tuple], list]:
    """Build the generation-history accordion.

    Returns (refresh_fn, refresh_outputs) so callers can also refresh the
    history right after a generation finishes and on app load.
    """
    include_caption = caption_box is not None

    with gr.Accordion(f"📂 最近の生成履歴 (最新{HISTORY_SIZE}件)", open=False):
        with gr.Row():
            refresh_btn = gr.Button("履歴を更新", size="sm")
            cleanup_btn = gr.Button(
                f"古い生成ファイルを削除 (最新{CLEANUP_KEEP_FILES}件を残す)", size="sm"
            )
        rows: list[dict] = []
        for i in range(HISTORY_SIZE):
            with gr.Row():
                audio = gr.Audio(
                    label=f"音声 {i + 1}", interactive=False, visible=False, scale=3
                )
                text_view = gr.Textbox(
                    label="テキスト", interactive=False, visible=False, scale=3
                )
                caption_view = None
                if include_caption:
                    caption_view = gr.Textbox(
                        label="Caption", interactive=False, visible=False, scale=3
                    )
                restore_btn = gr.Button("設定を復元", size="sm", visible=False, scale=1)
                meta_state = gr.State({})
            rows.append(
                {
                    "audio": audio,
                    "text": text_view,
                    "caption": caption_view,
                    "restore": restore_btn,
                    "meta": meta_state,
                }
            )

    refresh_outputs: list = []
    for row in rows:
        refresh_outputs.extend([row["audio"], row["text"]])
        if include_caption:
            refresh_outputs.append(row["caption"])
        refresh_outputs.extend([row["restore"], row["meta"]])

    def refresh_history() -> tuple:
        entries = _list_history_entries(out_dir, HISTORY_SIZE)
        updates: list[object] = []
        for i in range(HISTORY_SIZE):
            if i < len(entries):
                entry = entries[i]
                updates.append(gr.update(value=entry.get("wav"), visible=True))
                updates.append(
                    gr.update(value=entry.get("text") or "（テキスト情報なし）", visible=True)
                )
                if include_caption:
                    updates.append(gr.update(value=entry.get("caption") or "", visible=True))
                updates.append(gr.update(visible=True))
                updates.append(entry)
            else:
                updates.append(gr.update(value=None, visible=False))
                updates.append(gr.update(value=None, visible=False))
                if include_caption:
                    updates.append(gr.update(value=None, visible=False))
                updates.append(gr.update(visible=False))
                updates.append({})
        return tuple(updates)

    restore_outputs = [text_box] + ([caption_box] if include_caption else []) + [seed_box]

    def restore_settings(meta: dict) -> tuple:
        meta = meta or {}
        updates: list[object] = [gr.update(value=str(meta.get("text") or ""))]
        if include_caption:
            updates.append(gr.update(value=str(meta.get("caption") or "")))
        seed = meta.get("seed")
        updates.append(gr.update(value=int(seed) if isinstance(seed, int) else -1))
        return tuple(updates)

    for row in rows:
        row["restore"].click(restore_settings, inputs=[row["meta"]], outputs=restore_outputs)

    def cleanup_and_refresh() -> tuple:
        gr.Info(cleanup_outputs(out_dir))
        return refresh_history()

    refresh_btn.click(refresh_history, outputs=refresh_outputs)
    cleanup_btn.click(cleanup_and_refresh, outputs=refresh_outputs)
    return refresh_history, refresh_outputs
