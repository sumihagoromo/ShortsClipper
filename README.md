# ShortsClipper

YouTubeライブ配信から感情分析とタイムスタンプ付き文字起こしを行い、手動動画編集を支援するツール

## 🎯 概要

ShortsClipperは、YouTubeライブ配信動画を分析し、感情の変化や重要な発言を検出して、ショート動画作成に適した編集ポイントを特定するPythonツールです。

### 主な機能

- **時刻付き文字起こし**: OpenAI Whisperによる高精度な日本語音声認識
- **感情分析**: テキストと音声から感情変化を検出
- **ハイライト検出**: 感情の急変や重要なポイントを自動特定
- **編集支援データ出力**: CSV/JSON形式でタイムスタンプ付きデータを提供

## 🚀 使用方法

### 1. 動画準備
YouTubeライブ配信動画を手動でダウンロード

### 2. 分析実行
```bash
python main.py your_video.mp4
```

### 3. 結果確認
- `your_video_analysis.csv`: 動画編集ソフト用のタイムスタンプデータ
- `your_video_highlights.json`: 詳細な分析結果とハイライト候補

### 4. 動画編集
出力されたタイムスタンプを参考に、Adobe Premiere Pro等でショート動画を作成

## 🛠 技術スタック

- **音声認識**: Faster Whisper (OpenAI Whisperの高速版)
- **感情分析**: ML-Ask + Empath API
- **音声処理**: librosa, pydub
- **データ出力**: pandas, json

## 📋 出力形式

### CSV (動画編集ソフト対応)
```csv
timestamp_start,timestamp_end,text,emotion_score,emotion_type,highlight_level
00:01:23.45,00:01:27.12,"すごく嬉しいです！",0.85,joy,high
```

### JSON (詳細データ)
```json
{
  "segments": [
    {
      "start": 83.45,
      "end": 87.12,
      "text": "すごく嬉しいです！",
      "emotion": {"joy": 0.85, "sadness": 0.12},
      "highlight_level": "high"
    }
  ]
}
```

## 🎬 ワークフロー

1. **入力**: YouTubeライブ配信動画 (手動ダウンロード)
2. **音声抽出**: FFmpegで音声ファイル生成
3. **文字起こし**: Faster Whisperでタイムスタンプ付きテキスト
4. **感情分析**: ML-Ask/Empathで感情スコア計算
5. **ハイライト検出**: 感情変化点と重要発言の特定
6. **データ出力**: CSV/JSONファイル生成
7. **手動編集**: 動画編集ソフトでショート動画作成

## 📦 インストール

```bash
pip install -r requirements.txt
```

## 🔧 設定

詳細な設定については `DESIGN.md` を参照してください。

## 🗺 開発ロードマップ

開発計画については `ROADMAP.md` を参照してください。

## 📄 ライセンス

MIT License
