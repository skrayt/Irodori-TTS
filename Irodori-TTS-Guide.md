# Irodori-TTS 使い方ガイド

このドキュメントは、[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) のローカル環境での使い方をまとめたものです。

## Irodori-TTS の特徴

Irodori-TTS はFlow Matchingをベースとした日本語特化の音声合成(TTS)モデルです。
主な特徴は以下の通りです：

- **絵文字による感情・スタイル制御**: テキスト内に絵文字を含めることで、直感的に音声のスタイルをコントロールできます。
- **Zero-shot Voice Cloning**: 数秒の参照音声(Reference Audio)を渡すだけで、その声質を模倣した音声を生成できます。
- **VoiceDesign機能**: プロンプト（テキスト）の指示に従って、特定の声質（例: 「落ち着いた女性の声で」）を生成する専用モデルも用意されています。

---

## 使い方の基本

Irodori-TTSは、初心者でも使いやすい **Webブラウザ上のUI (Gradio)** と、バッチ処理等に適した **コマンドライン (CLI)** の2つの方法で実行できます。

モデルのデータは、コマンド実行時に Hugging Face から自動的にダウンロードされます。
（※ 初回実行時は数GBのモデルダウンロードが行われるため、時間がかかります）

### 1. Webブラウザ (Gradio UI) で使う

もっとも手軽な方法です。ターミナルで以下のコマンドを実行すると、ローカルサーバーが立ち上がります。

#### 通常のIrodori-TTS (Voice Cloning用)
参照音声を用いて自分の好きな声でテキストを読み上げさせたい場合はこちらを使用します。

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```
起動後、ブラウザで `http://localhost:7860` にアクセスしてください。

#### VoiceDesign版 (プロンプトでの声質指定用)
参照音声ではなく、「明るい女性の声」などのテキスト指定で声質を作りたい場合はこちらを使用します。

```bash
uv run python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```
起動後、ブラウザで `http://localhost:7861` にアクセスしてください。

---

### 2. コマンドライン (CLI) で使う

スクリプト等から自動化したい場合は `infer.py` を使用します。

#### 基本的なテキスト読み上げ（参照音声あり）
```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

#### 参照音声なしでの読み上げ
```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

#### VoiceDesign版での読み上げ（プロンプト指定）
```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "今日はいい天気ですね。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

---

## 感情表現（絵文字）の使い方

テキスト入力欄に絵文字や特定の記号を混ぜることで、読み上げのニュアンスを変えることができます。

**例:**
- 「今日はいい天気ですね😊」 -> 嬉しそうに読む
- 「ああっ！大変だ💦」 -> 焦ったように読む
- 「ふふっ(笑) それでね、」 -> 笑いを含んで読む

※ Irodori-TTSのトークナイザーは日本語の一般的な顔文字や絵文字のニュアンスを学習しています。Gradio UI上で色々な絵文字を試して、狙った感情が出るか調整するのがおすすめです。

> [!TIP]
> 途中でエラーが出たり、メモリ不足 (OOM) になる場合は、一度に読み上げるテキストの文字数（`--max-text-len`）を減らすか、文章を区切って生成すると安定します。
