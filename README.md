# ShortsClipper v2.0

YouTubeライブ配信から高精度な日本語音声認識とAI感情分析を行い、効率的なショート動画編集を支援する純粋関数ベースツール

## 🎯 概要

ShortsClipperは、長時間のYouTubeライブ配信動画を分析し、技術的内容に特化した高精度転写と感情変化検出により、ショート動画作成に最適な編集ポイントを自動特定するPythonツールです。

### ✨ 主な機能

- **🎤 適応的音声転写**: 音声長に応じた最適処理方式の自動選択
- **🔄 オーバーラップ分割**: 音声境界での単語切断を防ぐ高精度処理
- **🧠 AI感情分析**: ML-Ask による日本語感情分析とハイライト検出
- **⚙️ 純粋関数設計**: 段階的処理による効率的パラメーター調整
- **🖥️ CLI統合**: 個別実行とパイプライン実行の柔軟なワークフロー

## 🚀 クイックスタート

### 個別ステップ実行

```bash
# Stage 1: 音声抽出
python main.py audio your_video.mp4

# Stage 2: 適応的文字起こし（推奨）
python main.py adaptive audio_file.wav --model base

# Stage 3: 感情分析
python main.py emotions transcript.json

# Stage 4: ハイライト検出
python main.py highlights emotions.json --preset aggressive
```

### パイプライン一括実行

```bash
# 基本パイプライン（base モデル）
python main.py pipeline your_video.mp4 --model base --preset standard

# 高精度パイプライン（large-v3 モデル）
python main.py pipeline your_video.mp4 --model large-v3 --preset aggressive
```

### 長時間音声対応

```bash
# オーバーラップ分割による高精度処理
python main.py adaptive long_audio.wav --overlap-seconds 5 --force-split

# 品質優先モード
python main.py adaptive audio.wav --quality --profile high_quality
```

## 🛠 技術スタック

### コア技術
- **音声認識**: Faster Whisper (base/large-v3)
- **感情分析**: ML-Ask（日本語特化）
- **音声処理**: FFmpeg（高速・軽量）
- **CLI Framework**: Click（コマンドライン統合）

### 先進的機能
- **適応的転写**: 音声長ベース処理方式自動選択
- **オーバーラップ分割**: 境界音声の重複処理
- **純粋関数設計**: 中間ファイル保存による効率化
- **並列処理**: マルチセグメント同時転写

## 📋 出力ファイル構造

### データディレクトリ
```
data/
├── stage1_audio/          # 音声抽出結果
│   ├── video_audio.wav
│   └── video_audio_clean.wav
├── stage2_transcript/     # 文字起こし結果  
│   ├── video_raw.json
│   ├── video_cleaned.json
│   └── video_adaptive_*.json
├── stage3_emotions/       # 感情分析結果
│   └── video_text_emotions.json
└── stage4_highlights/     # ハイライト検出結果
    └── video_highlights_*.json
```

### JSON出力例
```json
{
  "segments": [
    {
      "start": 126.86,
      "end": 138.46,
      "text": "そして、ここにやりたいことを入れますと、YouTubeで技術調査。",
      "confidence": -0.6507778662555622,
      "emotions": {"joy": 0.75, "surprise": 0.45},
      "highlight_level": "high"
    }
  ],
  "metadata": {
    "model_size": "base",
    "processing_method": "adaptive",
    "segments_count": 52,
    "total_processing_time": 140.5
  }
}
```

## 🎬 処理ワークフロー

### 純粋関数ベース設計
```
動画ファイル → Stage 1 → 音声ファイル → Stage 2 → 転写JSON → Stage 3 → 感情JSON → Stage 4 → ハイライトJSON
```

### ステージ詳細

1. **🎵 Stage 1 (音声抽出)**
   - FFmpegによる高速音声抽出
   - 音楽部分自動スキップ（冒頭3分）
   - 16kHz mono WAV出力

2. **📝 Stage 2 (適応的転写)**
   - 音声長による処理方式自動選択
   - オーバーラップ分割による高精度化
   - 日本語技術用語最適化プロンプト

3. **🧠 Stage 3 (感情分析)**
   - ML-Ask による日本語感情分析
   - 時系列感情変化検出
   - 軽量処理（0.2秒/10分動画）

4. **⭐ Stage 4 (ハイライト検出)**
   - 感情変化パターン検出
   - キーワードベース重要度計算
   - 3段階ハイライトレベル分類

## 📦 インストール & セットアップ

### 必要要件
- Python 3.8+
- FFmpeg
- CUDA対応GPU（オプション、転写高速化）

### インストール手順
```bash
# リポジトリクローン
git clone https://github.com/sumihagoromo/ShortsClipper.git
cd ShortsClipper

# 依存関係インストール
pip install -r requirements.txt

# FFmpeg確認
ffmpeg -version

# 動作確認
python main.py version
```

## ⚡ パフォーマンス

### 処理時間実測値
- **音声抽出**: 1時間動画 → 30秒
- **base モデル**: 10分音声 → 3-5分転写
- **large-v3 モデル**: 10分音声 → 8-12分転写
- **感情分析**: 10分動画 → 0.2秒
- **ハイライト検出**: 10分動画 → 0.1秒

### 最適化機能
- **適応的処理**: 音声長に応じた自動最適化
- **並列処理**: マルチセグメント同時転写
- **キャッシュ**: 中間ファイル保存によるパラメーター調整効率化

## 🔧 設定ファイル

プロジェクト固有の詳細設定は [`CLAUDE.md`](CLAUDE.md) を参照  
技術仕様の詳細は [`DESIGN.md`](DESIGN.md) を参照

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照
