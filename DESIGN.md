# ShortsClipper 技術設計書

## 🏗 システム概要

ShortsClipperは、YouTubeライブ配信動画を分析して手動動画編集を支援するツールです。音声認識、感情分析、ハイライト検出の3つの主要コンポーネントで構成されます。

## 📁 プロジェクト構造

```
ShortsClipper/
├── main.py                    # メインエントリーポイント
├── src/
│   ├── transcriber.py         # 音声認識・文字起こし
│   ├── emotion_analyzer.py    # 感情分析
│   ├── highlight_detector.py  # ハイライト検出
│   ├── output_formatter.py    # データ出力処理
│   └── utils/
│       ├── audio_processor.py # 音声ファイル処理
│       └── config.py          # 設定管理
├── config/
│   └── settings.yaml          # 設定ファイル
├── tests/
│   ├── test_transcriber.py
│   ├── test_emotion_analyzer.py
│   └── test_highlight_detector.py
├── docs/
│   ├── README.md
│   ├── DESIGN.md
│   └── ROADMAP.md
├── requirements.txt
└── .gitignore
```

## 🔧 技術アーキテクチャ

### 1. 音声認識モジュール (transcriber.py)

**ライブラリ**: Faster Whisper

**機能**:
- 動画ファイルから音声抽出
- 単語レベルのタイムスタンプ付き文字起こし
- 日本語特化の設定

**処理フロー**:
```
動画ファイル → FFmpeg音声抽出 → Faster Whisper → タイムスタンプ付きテキスト
```

**出力形式**:
```python
{
    "segments": [
        {
            "start": 83.45,
            "end": 87.12,
            "text": "すごく嬉しいです！",
            "words": [
                {"word": "すごく", "start": 83.45, "end": 84.12},
                {"word": "嬉しい", "start": 84.15, "end": 85.80},
                {"word": "です", "start": 85.85, "end": 87.12}
            ]
        }
    ]
}
```

### 2. 感情分析モジュール (emotion_analyzer.py)

**ライブラリ**: ML-Ask + Empath API (オプション)

**機能**:
- テキストベース感情分析
- 音声ベース感情分析 (Phase 2)
- 感情スコアの時系列データ生成

**感情カテゴリ**:
- joy (喜び)
- sadness (悲しみ)
- anger (怒り)
- fear (恐れ)
- surprise (驚き)
- neutral (中性)

**出力形式**:
```python
{
    "timestamp": 83.45,
    "emotions": {
        "joy": 0.85,
        "sadness": 0.12,
        "anger": 0.03,
        "fear": 0.02,
        "surprise": 0.15,
        "neutral": 0.08
    },
    "dominant_emotion": "joy",
    "confidence": 0.85
}
```

### 3. ハイライト検出モジュール (highlight_detector.py)

**検出アルゴリズム**:

#### 感情変化検出
- **感情スコア急変**: 連続する時間窓での感情スコア変化が閾値(0.3)以上
- **高感情値**: 任意の感情が閾値(0.7)以上
- **感情持続**: 高感情状態が一定時間(3秒)以上継続

#### キーワードベース検出
- **反応語**: 「すごい」「やばい」「最高」「ひどい」等
- **感嘆詞**: 「おー」「うわー」「えー」等
- **強調表現**: 「めちゃくちゃ」「超」「本当に」等

#### 音響特徴検出 (Phase 2)
- **音量変化**: 急激な音量上昇・下降
- **発話速度**: 通常より早い・遅い発話
- **無音区間**: 長い沈黙の前後

**ハイライトレベル**:
- **high**: 複数の検出条件を満たす
- **medium**: 1-2個の検出条件を満たす
- **low**: 軽微な感情変化

### 4. データ出力モジュール (output_formatter.py)

**CSV出力** (動画編集ソフト用):
```csv
timestamp_start,timestamp_end,text,emotion_score,emotion_type,highlight_level,keywords
00:01:23.45,00:01:27.12,"すごく嬉しいです！",0.85,joy,high,"すごく"
00:03:45.67,00:03:48.90,"ちょっと困ったな...",0.72,sadness,medium,""
```

**JSON出力** (詳細データ):
```json
{
    "metadata": {
        "video_file": "sample.mp4",
        "duration": 3600.0,
        "processed_at": "2024-01-01T12:00:00Z",
        "model_versions": {
            "whisper": "faster-whisper-large-v3",
            "emotion": "ml-ask-1.0"
        }
    },
    "segments": [...],
    "highlights": [
        {
            "timestamp": 83.45,
            "duration": 4.67,
            "level": "high",
            "reasons": ["emotion_spike", "keyword_match"],
            "context": {
                "before": "...",
                "highlight": "すごく嬉しいです！",
                "after": "..."
            }
        }
    ],
    "statistics": {
        "total_highlights": 15,
        "emotion_distribution": {...},
        "average_emotion_intensity": 0.42
    }
}
```

## 🔧 設定管理

**settings.yaml**:
```yaml
transcription:
  model_size: "large-v3"
  language: "ja"
  initial_prompt: "以下は日本語の音声です。"
  temperature: 0.0

emotion_analysis:
  provider: "ml-ask"  # or "empath"
  batch_size: 10
  confidence_threshold: 0.5

highlight_detection:
  emotion_change_threshold: 0.3
  high_emotion_threshold: 0.7
  keyword_weight: 1.5
  duration_weight: 1.2
  
output:
  csv_format: true
  json_format: true
  include_statistics: true
  
audio:
  sample_rate: 16000
  channels: 1
```

## 🚀 パフォーマンス考慮事項

### CPU・GPU使用量
- **Faster Whisper**: GPU利用で4倍高速化
- **感情分析**: CPU処理、バッチ処理で最適化

### メモリ使用量
- **大容量動画**: チャンク分割処理
- **中間ファイル**: 一時ディレクトリでの管理

### 処理時間目安
- 1時間の動画: 約10-15分 (GPU使用時)
- 1時間の動画: 約30-45分 (CPU使用時)

## 🔒 セキュリティ・プライバシー

- **ローカル処理**: 動画データはローカルでのみ処理
- **API使用**: Empath使用時のみ外部API通信
- **一時ファイル**: 処理完了後の自動削除

## 🧪 テスト戦略

### 単体テスト
- 各モジュールの個別機能テスト
- エラーハンドリングテスト

### 統合テスト
- エンドツーエンドの処理フローテスト
- 様々な動画形式での動作確認

### パフォーマンステスト
- 大容量ファイルでの処理時間測定
- メモリ使用量の監視

## 📊 ログ・モニタリング

```python
# ログレベル
DEBUG: 詳細な処理情報
INFO: 処理進捗状況
WARNING: 警告事項
ERROR: エラー情報
```

**ログ出力例**:
```
2024-01-01 12:00:00 INFO [main] 動画ファイル読み込み開始: sample.mp4
2024-01-01 12:01:30 INFO [transcriber] 音声抽出完了: 3600秒
2024-01-01 12:05:45 INFO [transcriber] 文字起こし完了: 245セグメント
2024-01-01 12:07:20 INFO [emotion_analyzer] 感情分析完了: 245セグメント
2024-01-01 12:08:10 INFO [highlight_detector] ハイライト検出完了: 15箇所
2024-01-01 12:08:30 INFO [output_formatter] 結果出力完了
```

## 🔄 エラーハンドリング

### 想定エラーと対処
- **動画読み込みエラー**: サポート形式チェック・変換提案
- **音声抽出エラー**: FFmpegインストール確認
- **メモリ不足**: チャンク分割処理への自動切り替え
- **API制限**: ローカル処理への切り替え提案

## 🔧 拡張性

### Phase 2追加予定機能
- 音声感情分析 (Empath API)
- リアルタイム処理
- 複数話者対応

### Phase 3追加予定機能
- 機械学習モデルの精度向上
- WebUI提供
- 複数プラットフォーム対応