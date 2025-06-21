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

## 🚀 現在の開発状況（Phase 2完了）

### 技術的成果 ✅
- **適応的転写システム完全実装**: オーバーラップ分割・重複除去機能
- **高精度処理実現**: 5分音声で52セグメント検出（境界問題解決）
- **処理速度最適化**: baseモデルで2.3分/5分音声の実用的処理時間
- **技術用語認識**: 「YouTube」「ショート動画」「自動化」等の正確認識
- **純粋関数ワークフロー**: 効率的パラメーター調整システム構築

### 実装完了機能
1. **Stage 1**: 音声抽出（音楽部分自動スキップ）
2. **Stage 2**: 適応的転写（オーバーラップ処理対応）
3. **Stage 3**: 感情分析（ML-Ask統合）
4. **Stage 4**: ハイライト検出（3段階レベル分類）
5. **CLI統合**: 個別実行・パイプライン実行対応

## 📋 Phase 3開発計画（実用性強化・ユーザビリティ改善）

### 次回セッション開始時の優先アクション

#### Sprint 3.1: 実用性検証（最優先）
1. **新しい音声データでの品質検証**
   - 最新適応的転写システムでのフルパイプライン実行
   - 転写精度・感情分析・ハイライト検出の実用レベル評価
   - 長時間音声（30分以上）での安定性テスト

2. **現在の出力形式の実用性評価**
   - 既存JSON出力の動画編集での使いやすさ確認
   - CSV出力機能の動画編集ソフトでの互換性テスト

#### Sprint 3.2: 出力改善（実装優先）
1. **動画編集者向けフォーマット設計**
   - Premiere Pro互換CSV出力形式
   - DaVinci Resolve対応フォーマット
   - タイムライン形式の視覚的レポート

2. **YouTubeチャプター生成機能**
   - ハイライト検出結果からのチャプターポイント自動生成
   - YouTube説明欄用フォーマット出力
   - 適切なチャプター間隔の自動調整

### セッション継続のための技術ノート

#### 重要ファイル位置
- **適応的転写**: `process_transcript_adaptive.py`
- **CLI統合**: `main.py` の adaptive コマンド
- **出力フォーマッター**: `src/output_formatter.py`
- **設定管理**: `config/` ディレクトリ各種YAML

#### 検証用コマンド例
```bash
# 新しい音声での適応的転写テスト
python main.py adaptive new_audio.wav --overlap-seconds 5

# フルパイプライン実行
python main.py pipeline new_video.mp4 --model base --preset standard

# 現在の出力確認
ls data/stage4_highlights/
```

## 🚀 Phase 3 完了状況（2025年6月21日）

### 新規実装完了機能 ✅
1. **音声境界自動検出システム**
   - Whisper言語確率ベースの自動検出
   - 音響特徴量分析による補完機能（librosa）
   - 設定可能なパラメーター・フォールバック機能
   - CLI統合（`detect-speech`コマンド）

2. **動画編集ソフト対応エクスポート機能** ✅
   - Premiere Pro互換CSV出力
   - DaVinci Resolve EDL形式出力
   - YouTubeチャプター自動生成
   - タイムライン視覚レポート生成
   - 元動画時間軸対応（音声オフセット自動補正）

3. **実用性強化機能** ✅
   - 設定可能フィルタリング・レベル分類
   - CLI統合（`export`コマンド）
   - エラーハンドリング強化
   - 純粋関数アーキテクチャ維持

### 使用例
```bash
# 音声境界自動検出テスト
python main.py detect-speech audio.wav

# フルパイプライン実行
python main.py audio video.mp4
python process_transcript.py -i data/stage1_audio/video_audio_clean.wav -c config/transcription_base.yaml
python process_emotions.py -i data/stage2_transcript/video_clean_cleaned.json
python process_highlights.py -i data/stage3_emotions/video_clean_text_emotions.json -c config/highlight_detection_aggressive.yaml

# エクスポート（全形式）
python main.py export data/stage4_highlights/video_highlights_aggressive.json --format all

# 特定形式エクスポート
python main.py export highlights.json --format timeline --min-level medium
python main.py export highlights.json --format premiere --max-segments 10
```

### 品質基準（達成済み）
- 転写精度: 技術用語90%以上認識 ✅
- 処理速度: baseモデルで実用的処理時間 ✅
- ハイライト検出: 実用レベルの精度 ✅
- 出力形式: 動画編集ソフトで直接利用可能 ✅
- 元動画時間軸対応: 音声オフセット自動補正 ✅