# ShortsClipper 技術設計書 v2.0

## 🏗 システム概要

ShortsClipperは、YouTubeライブ配信動画を分析して手動動画編集を支援するツールです。純粋関数ベースのワークフロー設計により、効率的なパラメーター調整と高精度な日本語音声処理を実現します。

**設計思想**:
- 純粋関数による各処理段階の分離
- 中間ファイル保存によるキャッシュ機能
- CLIコントローラーによる柔軟な実行制御
- 高速な反復開発（パラメーター調整）の支援

## 📁 プロジェクト構造

```
ShortsClipper/
├── main.py                           # CLIコントローラー（Click ベース）
├── process_audio.py                   # Stage 1: 音声抽出ワークフロー
├── process_transcript.py              # Stage 2: 文字起こしワークフロー 
├── process_emotions.py                # Stage 3: 感情分析ワークフロー
├── process_highlights.py              # Stage 4: ハイライト検出ワークフロー
├── src/
│   ├── transcriber.py                 # Faster Whisper実装
│   ├── emotion_analyzer.py            # ML-Ask感情分析
│   ├── highlight_detector.py          # パターンベース検出
│   ├── output_formatter.py            # 多形式出力
│   └── utils/
│       ├── audio_processor.py         # FFmpeg音声処理
│       ├── config.py                  # YAML設定管理
│       └── logging_config.py          # 統合ログシステム
├── config/
│   ├── audio_extraction.yaml          # 音声抽出設定
│   ├── transcription_base.yaml        # 基本文字起こし
│   ├── transcription_large_v3.yaml    # 高精度文字起こし
│   ├── emotion_analysis.yaml          # 感情分析設定
│   └── highlight_detection_*.yaml     # ハイライト検出設定
├── scripts/
│   ├── analysis/                      # 分析スクリプト群
│   ├── testing/                       # テストスクリプト群
│   ├── monitoring/                    # 監視スクリプト群
│   └── tuning/                        # パラメーター調整
├── data/
│   ├── stage1_audio/                  # 音声抽出結果
│   ├── stage2_transcript/             # 文字起こし結果
│   ├── stage3_emotions/               # 感情分析結果
│   └── stage4_highlights/             # ハイライト検出結果
├── logs/
│   ├── transcription/                 # 文字起こしログ
│   ├── processing/                    # 処理ログ
│   ├── monitoring/                    # 監視ログ
│   └── analysis/                      # 分析ログ
├── CLAUDE.md                          # プロジェクト固有ルール
└── requirements.txt
```

## 🔧 技術アーキテクチャ

### アーキテクチャ設計原則

**純粋関数ベースワークフロー**:
```
入力ファイル → Stage 1 → 中間ファイル → Stage 2 → 中間ファイル → Stage 3 → 中間ファイル → Stage 4 → 最終結果
```

- 各Stageは純粋関数として実装
- 標準化されたJSON入出力形式
- 重い処理と軽い処理の分離
- 段階的実行とパラメーター調整の効率化

**CLIコントローラー設計**:
```bash
# 個別実行
python main.py audio video.mp4
python main.py transcript audio.wav --model large-v3
python main.py emotions transcript.json
python main.py highlights emotions.json --preset aggressive

# パイプライン実行
python main.py pipeline video.mp4 --model base --preset standard

# パラメーター調整
python main.py tune -i emotions.json -p 3
```

### Stage 1: 音声抽出ワークフロー (process_audio.py)

**ライブラリ**: FFmpeg

**機能**:
- 動画ファイルから音声抽出（16kHz, mono WAV）
- 音声メタデータ生成
- 音声クリーニング（音楽除去）

**処理フロー**:
```
動画ファイル → FFmpeg音声抽出 → WAV + メタデータJSON
```

**パフォーマンス**: 1時間動画 → 約30秒

**出力形式**:
```json
{
    "video_id": "sample",
    "audio_file": "sample_audio.wav",
    "duration": 3600.0,
    "sample_rate": 16000,
    "channels": 1,
    "processing_time": 30.5,
    "created_at": "2024-01-01T12:00:00Z"
}
```

### Stage 2: 適応的文字起こしワークフロー (process_transcript_adaptive.py)

**ライブラリ**: Faster Whisper + 適応的処理システム

**基本機能** (process_transcript.py):
- 高精度日本語文字起こし（language="ja"指定）
- マルチモデル対応（base, large-v3）
- 単語レベルタイムスタンプ
- エンジニア向けプロンプト最適化

**適応的転写機能** (process_transcript_adaptive.py):
- **音声長ベース処理方式自動選択**
- **オーバーラップ分割**: 音声境界での単語切断防止
- **重複除去アルゴリズム**: 類似度ベース重複検出
- **並列処理**: マルチセグメント同時転写

**処理フロー**:
```
音声WAV → 音声長判定 → 分割戦略選択 → オーバーラップ分割 → 並列転写 → 重複除去 → 統合結果
```

**パフォーマンス**:
- baseモデル: 5分音声 → 1.3分転写（適応的）
- baseモデル: 10分音声 → 3-5分転写
- large-v3モデル: 10分音声 → 8-12分転写

**出力形式**:
```json
{
    "video_id": "success_segment_5min",
    "segments": [
        {
            "start": 126.86,
            "end": 138.46,
            "text": "そして、ここにやりたいことを入れますと、YouTubeで技術調査。",
            "confidence": -0.6507778662555622,
            "duration": 11.600000000000009,
            "word_count": 1
        }
    ],
    "language": "ja",
    "language_probability": 1.0,
    "metadata": {
        "model_size": "base",
        "processing_method": "overlap_aware_merge",
        "segments_count": 52,
        "source_segments": 2,
        "total_processing_time": 140.5,
        "profile_used": "direct"
    },
    "overlap_processing": {
        "overlap_regions_found": 1,
        "high_similarity_regions": 0,
        "similarity_threshold": 0.7,
        "overlap_seconds": 3.0
    },
    "success": true
}
```

### Stage 3: 感情分析ワークフロー (process_emotions.py)

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

**処理フロー**:
```
文字起こしJSON → ML-Ask感情分析 → 時系列感情データJSON
```

**パフォーマンス**: 10分動画 → 約0.2秒（軽い処理）

**出力形式**:
```json
{
    "emotions": [
        {
            "timestamp": 83.45,
            "text": "すごく嬉しいです！",
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
    ],
    "statistics": {
        "total_segments": 245,
        "processing_time": 0.2
    }
}
```

### Stage 4: ハイライト検出ワークフロー (process_highlights.py)

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

**処理フロー**:
```
感情分析JSON → パターンマッチング → ハイライト候補JSON
```

**パフォーマンス**: 10分動画 → 約0.1秒（軽い処理）

### Stage 5: 多形式出力ワークフロー (process_output.py)

**機能**:
- CSV出力（動画編集ソフト用）
- JSON出力（詳細データ）
- SRT字幕出力
- 統計レポート生成

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

**モジュラー設定ファイル**:

**transcription_large_v3.yaml** (高精度モード):
```yaml
transcription:
  model_size: "large-v3"
  language: "ja"
  temperature: 0.0
  initial_prompt: "日本語の音声です。技術的な話題、プログラミング、エンジニアリング、YouTube、API、コーディングに関する内容を含みます。"
  word_timestamps: true
  condition_on_previous_text: true
  device: "auto"
  compute_type: "float32"
  beam_size: 5
  best_of: 5
  patience: 1.0
  length_penalty: 1.0
  repetition_penalty: 1.1
  no_repeat_ngram_size: 2
```

**transcription_base.yaml** (高速モード):
```yaml
transcription:
  model_size: "base"
  language: "ja"
  temperature: 0.0
  initial_prompt: "日本語の音声です。"
  word_timestamps: true
  device: "auto"
  compute_type: "float32"
```

**highlight_detection_aggressive.yaml**:
```yaml
highlight_detection:
  emotion_change_threshold: 0.2
  high_emotion_threshold: 0.6
  keyword_weight: 2.0
  duration_weight: 1.5
  min_highlight_duration: 2.0
  max_highlight_duration: 30.0
```

## 🆕 先進的技術コンポーネント

### 適応的転写システム (AdaptiveTranscriber)

**核心機能**:
- **音声長ベース処理方式自動選択**: 音声長に応じてdirect/batch処理を選択
- **プロファイル自動調整**: 品質重視・速度重視の自動切り替え
- **処理時間予測**: 機械学習ベース処理時間推定

**コンポーネント**:
```
AdaptiveConfigManager → AudioSplitter → BatchTranscriber → TranscriptionMerger
```

### オーバーラップ音声分割 (AudioSplitter)

**問題解決**:
- 音声境界での単語切断問題
- セグメント境界での情報ロス
- 長時間音声での精度低下

**技術仕様**:
- **オーバーラップ時間**: 設定可能（デフォルト5秒）
- **無音検出**: FFmpegベース最適分割点探索
- **元境界保持**: 重複除去用メタデータ保存

### 重複除去システム (OverlapProcessor)

**アルゴリズム**:
- **類似度計算**: difflib.SequenceMatcherベース
- **閾値ベース判定**: 設定可能類似度閾値（デフォルト0.7）
- **タイムスタンプ調整**: 重複除去後の時系列整合性保証

**実証結果**:
- 5分音声: 51→52セグメント（+1改善）
- 境界部分の追加音声内容捕捉確認
- 重複除去精度: 類似度0.00で適切保持

## 🚀 パフォーマンス特性

### 処理時間実測値

**Stage 1 (音声抽出)**:
- 1時間動画: 約30秒
- CPU使用量: 低

**Stage 2 (文字起こし)**:
- baseモデル: 10分音声 → 65分（6.5倍）
- large-v3モデル: 10分音声 → 100分（10倍）
- ボトルネック: CPU集約的処理

**Stage 3 (感情分析)**:
- 10分動画: 約0.2秒
- 軽量処理、反復調整に最適

**Stage 4 (ハイライト検出)**:
- 10分動画: 約0.1秒
- 軽量処理、反復調整に最適

### ワークフロー効率化

**重い処理と軽い処理の分離**:
- 重い処理: 音声抽出（30秒）、文字起こし（65-100分）
- 軽い処理: 感情分析（0.2秒）、ハイライト検出（0.1秒）

**パラメーター調整効率**:
- 文字起こし後の調整: Stage 3-4のみ実行（約0.3秒）
- 音声前処理後の調整: Stage 2-4実行（65-100分）

### メモリ使用量
- **中間ファイル保存**: data/stage*/ディレクトリ構造
- **ログ分離**: logs/*/カテゴリ別管理
- **設定分離**: config/*.yaml目的別設定

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

## 📊 統合ログシステム

### ログ分類とディレクトリ構造

```
logs/
├── transcription/          # 文字起こし処理ログ
│   ├── transcript_base.log
│   └── transcript_large_v3.log
├── processing/             # 各Stage処理ログ
│   ├── audio_*.log
│   ├── emotions_*.log
│   └── highlights_*.log
├── monitoring/             # 監視・進捗ログ
│   └── progress_*.log
└── analysis/               # 分析・比較ログ
    └── quality_analysis_*.log
```

### ログ設定

**統合ログ設定 (src/utils/logging_config.py)**:
```python
def setup_process_logging(
    process_name: str,
    log_category: str = "processing",
    level: int = logging.INFO,
    video_id: Optional[str] = None
) -> logging.Logger:
```

**ログ出力例**:
```
2025-06-15 20:22:36 INFO [transcript] === 文字起こしプロセス開始 ===
2025-06-15 20:22:36 INFO [transcript] 入力: audio_clean_10min.wav
2025-06-15 20:22:36 INFO [transcript] モデル: large-v3 (日本語特化)
2025-06-15 20:23:05 INFO [transcript] Whisperモデル読み込み完了
2025-06-15 20:23:05 INFO [transcript] 音声転写実行中...
```

### 進捗モニタリング

**自動進捗監視**:
```bash
python scripts/monitoring/monitor_progress.py
```

- 30分間隔での進捗確認
- プロセス継続状況の自動報告
- 長時間処理の完了通知

## 🔄 エラーハンドリング

### 想定エラーと対処
- **動画読み込みエラー**: サポート形式チェック・変換提案
- **音声抽出エラー**: FFmpegインストール確認
- **メモリ不足**: チャンク分割処理への自動切り替え
- **API制限**: ローカル処理への切り替え提案

## 🎯 実装状況とロードマップ

### Phase 2完了機能 ✅
- ✅ 純粋関数ベースワークフロー設計
- ✅ 高精度日本語文字起こし（large-v3モデル）
- ✅ CLIコントローラー実装
- ✅ 効率的パラメーター調整システム
- ✅ 統合ログシステム
- ✅ 音声前処理（音楽除去）
- ✅ マルチプリセット対応
- ✅ 並列調整実行

### Phase 2.5完了 ✅ (適応的転写システム)
- ✅ **適応的転写システム実装**
- ✅ **オーバーラップ分割機能実装** 
- ✅ **重複除去アルゴリズム実装**
- ✅ **音声境界問題完全解決**
- ✅ **5分音声52セグメント検出実現**

### Phase 3完了 ✅ (実用性強化・ユーザビリティ改善) - 2025年6月21日

#### Sprint 3.1: 実用性検証・品質改善 ✅
- ✅ **新しい音声データでの品質検証**（claudecodeintro.mp4で検証済み）
- ✅ **音声境界自動検出システム実装**
- ✅ **ハイライト検出実用レベル評価**（25個検出、156.5秒カバレッジ）
- ✅ **エラーハンドリング強化**

#### Sprint 3.2: 動画編集者向け出力改善 ✅
- ✅ **動画編集ソフト対応出力フォーマット**
  - ✅ Premiere Pro互換CSV出力
  - ✅ DaVinci Resolve EDL形式出力
  - ✅ タイムライン形式視覚レポート
- ✅ **元動画時間軸対応**（音声オフセット自動補正）
- ✅ **ハイライト詳細情報出力**（感情・信頼度・レベル）

#### Sprint 3.3: YouTubeチャプター生成機能 ✅
- ✅ **チャプターポイント自動生成**
  - ✅ ハイライト検出結果ベース
  - ✅ 適切な間隔の自動調整（最低2分間隔）
  - ✅ チャプター重複回避アルゴリズム
- ✅ **YouTube説明欄用フォーマット出力**（mm:ss形式）
- ✅ **チャプタータイトル自動生成**（感情・キーワードベース）

### Phase 3.5: 高度機能オプション 📋
- 📋 WebUI実装（Streamlit/Gradio）
- 📋 音声感情分析統合 (Empath API)
- 📋 複数話者対応
- 📋 バッチ処理最適化
- 📋 GPU最適化

## 🔧 開発運用ルール

### CLAUDE.mdプロジェクト固有ルール
- 音声処理: 音楽混入回避のため先頭3分スキップ
- モデル選択: baseモデル推奨（速度重視、large-v3は高精度モード）
- ログ管理: カテゴリ別ディレクトリ構造の維持
- 設定管理: YAML分離によるモジュラー設計
- 性能制限: CPU集約的処理のため並列度調整

### CLIコマンド体系
```bash
# 基本コマンド
python main.py audio video.mp4        # Stage 1実行
python main.py transcript audio.wav   # Stage 2実行（基本）
python main.py adaptive audio.wav     # Stage 2実行（適応的・推奨）
python main.py emotions transcript.json # Stage 3実行
python main.py highlights emotions.json # Stage 4実行

# 適応的転写（高機能）
python main.py adaptive audio.wav --overlap-seconds 5 --force-split
python main.py adaptive audio.wav --quality --profile high_quality
python main.py adaptive audio.wav --no-overlap  # オーバーラップ無効

# パイプライン実行
python main.py pipeline video.mp4 --model base --preset standard

# パラメーター調整
python main.py tune -i emotions.json -p 3

# システム管理
python main.py cleanup --category processing --days 7
python main.py version

# Phase 3新機能（計画中）
python main.py export --format premiere-pro transcript.json  # Premiere Pro互換出力
python main.py export --format davinci transcript.json       # DaVinci Resolve互換出力
python main.py chapters highlights.json --min-gap 120        # YouTubeチャプター生成
python main.py report --visual timeline.html                 # 視覚的レポート生成
```

## 🎯 Phase 3技術仕様（計画中）

### 動画編集ソフト対応出力フォーマット

**Premiere Pro互換CSV**:
```csv
Marker Name,Description,In,Out,Duration,Marker Type
Highlight 1,"感情スコア: 0.85",00:01:23:12,00:01:27:08,00:00:03:21,Comment
Chapter 1,"技術説明開始",00:02:15:00,,,Chapter
```

**DaVinci Resolve EDL形式**:
```
TITLE: ShortsClipper Highlights
FCM: NON-DROP FRAME

001  001      V     C        00:01:23:12 00:01:27:08 01:00:00:00 01:00:03:21
* FROM CLIP NAME: Highlight_1_Joy_0.85
```

### YouTubeチャプター生成アルゴリズム

**チャプター分割ロジック**:
1. **感情変化ベース**: 大きな感情変化点でチャプター分割
2. **ハイライト密度ベース**: ハイライトが集中する区間の前後
3. **時間ベース**: 最低間隔（例：2分）の保証
4. **話題変化ベース**: キーワード変化による論理分割

**出力フォーマット例**:
```
00:00 イントロ・概要説明
02:15 技術デモンストレーション開始
05:42 実装の詳細解説
08:30 質疑応答・まとめ
```

## 🎯 Phase 3実装済み技術仕様

### 音声境界自動検出システム

**実装ファイル**: `src/speech_detector.py`, `src/audio_features_detector.py`

**技術仕様**:
- **Whisperベース検出**: 言語確率・セグメント分析
- **音響特徴量分析**: librosa使用、スペクトラル分析
- **設定可能パラメーター**: 閾値・チェック間隔・モデル設定
- **フォールバック機能**: 検出失敗時のデフォルトスキップ

### 動画編集ソフト対応エクスポート

**実装ファイル**: `src/export_formatter.py`

**対応形式**:
```python
# Premiere Pro CSV
fieldnames = ['Marker Name', 'Description', 'In', 'Out', 'Duration', 'Marker Type']

# DaVinci Resolve EDL
format = "TITLE: ShortsClipper Highlights\nFCM: NON-DROP FRAME\n{entries}"

# YouTube Chapters
format = "mm:ss タイトル"

# Timeline Report
format = "Markdown形式視覚レポート"
```

**重要機能**:
- **元動画時間軸対応**: 音声オフセット自動補正
- **設定可能フィルタリング**: レベル・時間・セグメント数
- **タイムコード変換**: フレームレート対応

### CLI統合コマンド

```bash
# 音声境界検出
python main.py detect-speech audio.wav [オプション]

# エクスポート（全形式）
python main.py export highlights.json --format all

# 特定形式エクスポート
python main.py export highlights.json --format timeline --min-level medium
```