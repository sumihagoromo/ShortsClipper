# ShortsClipper プロジェクト固有ルール

## プロジェクト概要
YouTube live stream analysis tool for creating highlight clips

## 音声処理ルール

### 音声抽出
- 元動画は手動ダウンロードしてパス指定する方針
- YouTube APIは使用しない
- 音楽イントロ部分（冒頭3分）は自動スキップ
- クリーン音声ファイルを作成して処理

### 文字起こし設定
- **Whisper large-v3は重すぎて非効率** - 使用禁止
- **baseモデルを使用** - 実用的な処理速度と十分な精度
- 日本語指定必須: `language: "ja"`
- 音楽部分をスキップしたクリーン音声で処理
- 処理時間目安: 10分音声 → 3-5分処理

### ファイル命名規則
- 音声ファイル: `{video_id}_audio.wav`
- クリーン音声: `{video_id}_audio_clean.wav` 
- 文字起こし: `{video_id}_raw.json`, `{video_id}_cleaned.json`
- 感情分析: `{video_id}_text_emotions.json`
- ハイライト: `{video_id}_highlights_{config_name}.json`

## アーキテクチャルール

### 純粋関数アーキテクチャ
- 各プロセスは純粋関数として実装
- 標準化されたJSON入出力
- 中間ファイル保存による効率的なパラメーター調整
- 重い処理（音声抽出・文字起こし）と軽い処理（感情分析・ハイライト検出）の分離

### プロセス分離
1. **Stage 1**: 音声抽出 (`process_audio.py`)
2. **Stage 2**: 文字起こし (`process_transcript.py`) 
3. **Stage 3**: 感情分析 (`process_emotions.py`)
4. **Stage 4**: ハイライト検出 (`process_highlights.py`)

### 設定管理
- YAMLファイルによる設定外部化
- プロセス別設定ファイル
- パラメーター調整用の複数設定プリセット

## パフォーマンスルール

### 効率的なワークフロー
- **重要**: 日本語文字起こし精度を最優先
- 後続処理は意味のある文字起こし結果が前提
- パラメーター調整のため高速な反復実行を重視
- バックグラウンド処理とモニタリングツール活用

### 処理時間制限
- 単一プロセステスト: 10分以内
- インタラクティブ調整: 1分以内の応答
- フルパイプライン: 1日以内完了

## 技術スタック
- Python 3.x
- faster-whisper (baseモデル)
- ML-Ask (感情分析、ollama fallback)
- FFmpeg (音声処理)
- YAML設定管理

## 禁止事項
- large-v3モデルの常用（テスト以外）
- 音楽混入音声での直接処理
- 設定ハードコーディング
- 重い処理の同期実行