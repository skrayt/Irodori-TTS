#!/bin/bash
echo "=========================================="
echo "Irodori-TTS (VoiceDesign版) を起動しています..."
echo "=========================================="
cd "/Users/bingoshouhei/Documents/pgm/MyProject/Irodori-TTS"
uv run python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
