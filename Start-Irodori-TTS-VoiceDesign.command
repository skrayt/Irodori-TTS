#!/bin/bash
echo "=========================================="
echo "Irodori-TTS (VoiceDesign版) を起動しています..."
echo "=========================================="
# このスクリプトが置かれているディレクトリへ移動（配置場所に依存しない）
cd "$(dirname "$0")"
# 既定ではローカルのみ公開 (http://127.0.0.1:7861)。
# 同一LANの他端末からアクセスしたい場合は --server-name 0.0.0.0 に変更してください。
uv run python gradio_app_voicedesign.py --server-name 127.0.0.1 --server-port 7861
