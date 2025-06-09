# GitHub Issues 作成ガイド

このファイルには、ShortsClipperプロジェクトのPhase 1 (MVP) に必要なGitHub Issuesのテンプレートが含まれています。

## ⚠️ GitHub CLI セットアップ

まず以下のコマンドでGitHub CLIにログインしてください：

```bash
gh auth login
```

その後、以下のIssue作成コマンドを実行してください。

---

## 📋 Phase 1: MVP Issues

### 1. プロジェクト基盤構築

```bash
gh issue create \
  --title "🏗 プロジェクト基盤構築" \
  --label "enhancement,phase-1" \
  --milestone "MVP" \
  --body "$(cat <<'EOF'
## 概要
ShortsClipperプロジェクトの基本的なプロジェクト構造とセットアップを行う

## タスク
- [ ] プロジェクトディレクトリ構造作成
- [ ] requirements.txt作成
- [ ] .gitignore設定
- [ ] 基本設定ファイル (config/settings.yaml) 作成
- [ ] ログ設定実装
- [ ] main.py エントリーポイント作成

## 受け入れ基準
- [ ] プロジェクト構造がDESIGN.mdの仕様と一致している
- [ ] 依存関係が正しく管理されている
- [ ] 基本的なログ出力が動作する

## 見積もり
**ストーリーポイント**: 3  
**予想工数**: 1日

## 関連ドキュメント
- DESIGN.md#プロジェクト構造
- ROADMAP.md#Sprint-1.1
EOF
)"
```

### 2. 音声抽出機能実装

```bash
gh issue create \
  --title "🎵 FFmpegによる音声抽出機能実装" \
  --label "feature,phase-1" \
  --milestone "MVP" \
  --body "$(cat <<'EOF'
## 概要
動画ファイルからFFmpegを使用して音声を抽出する機能を実装

## タスク
- [ ] FFmpeg依存関係チェック機能
- [ ] 動画ファイル形式検証
- [ ] 音声抽出処理 (16kHz, モノラル)
- [ ] 一時ファイル管理
- [ ] エラーハンドリング実装

## 技術仕様
- **入力**: mp4, mov, avi, mkv等の動画ファイル
- **出力**: 16kHz, モノラルのwavファイル
- **使用ライブラリ**: ffmpeg-python

## 受け入れ基準
- [ ] 主要な動画形式から音声抽出可能
- [ ] 一時ファイルが適切にクリーンアップされる
- [ ] FFmpeg未インストール時に適切なエラーメッセージ表示

## 見積もり
**ストーリーポイント**: 5  
**予想工数**: 2日

## 関連ドキュメント
- DESIGN.md#技術アーキテクチャ
EOF
)"
```

### 3. Faster Whisper統合

```bash
gh issue create \
  --title "🗣 Faster Whisper統合とタイムスタンプ付き文字起こし" \
  --label "feature,phase-1" \
  --milestone "MVP" \
  --body "$(cat <<'EOF'
## 概要
Faster Whisperを統合してタイムスタンプ付きの文字起こし機能を実装

## タスク
- [ ] Faster Whisperライブラリ統合
- [ ] 日本語最適化設定 (initial_prompt等)
- [ ] セグメント・単語レベルタイムスタンプ取得
- [ ] 進捗表示機能
- [ ] メモリ効率的な処理実装

## 技術仕様
- **モデル**: large-v3 (設定可能)
- **言語**: 日本語
- **出力**: セグメント情報 + 単語レベルタイムスタンプ
- **パフォーマンス**: GPU対応

## 受け入れ基準
- [ ] 10分程度の動画で正確な文字起こしが可能
- [ ] タイムスタンプが秒単位で正確
- [ ] 日本語の認識精度が実用レベル
- [ ] プログレスバーで進捗確認可能

## 見積もり
**ストーリーポイント**: 8  
**予想工数**: 3日

## 関連Issue
- 音声抽出機能実装 (#2)
EOF
)"
```

### 4. 感情分析機能実装

```bash
gh issue create \
  --title "😊 ML-Askによる感情分析機能実装" \
  --label "feature,phase-1" \
  --milestone "MVP" \
  --body "$(cat <<'EOF'
## 概要
ML-Askライブラリを使用してテキストベースの感情分析機能を実装

## タスク
- [ ] ML-Askライブラリ統合
- [ ] 感情スコア計算機能
- [ ] タイムスタンプとの紐付け
- [ ] 感情データの正規化
- [ ] バッチ処理最適化

## 技術仕様
- **入力**: Whisperの文字起こし結果
- **出力**: 6種類の感情スコア (joy, sadness, anger, fear, surprise, neutral)
- **処理単位**: セグメント単位

## 感情カテゴリ
- joy (喜び)
- sadness (悲しみ)  
- anger (怒り)
- fear (恐れ)
- surprise (驚き)
- neutral (中性)

## 受け入れ基準
- [ ] 各セグメントに対して感情スコアを計算可能
- [ ] タイムスタンプが正確に保持される
- [ ] 処理速度が実用的レベル
- [ ] 感情スコアが0-1の範囲で正規化される

## 見積もり
**ストーリーポイント**: 5  
**予想工数**: 2日

## 関連Issue
- Faster Whisper統合 (#3)
EOF
)"
```

### 5. 基本ハイライト検出機能

```bash
gh issue create \
  --title "⭐ 基本ハイライト検出機能実装" \
  --label "feature,phase-1" \
  --milestone "MVP" \
  --body "$(cat <<'EOF'
## 概要
感情分析結果を基にした基本的なハイライト検出機能を実装

## タスク
- [ ] 感情変化検出アルゴリズム
- [ ] 高感情値検出
- [ ] キーワードベース検出 (基本版)
- [ ] ハイライトレベル分類 (high/medium/low)
- [ ] 検出閾値の設定機能

## 検出ロジック
### 感情変化検出
- 連続する時間窓での感情スコア変化が閾値(0.3)以上

### 高感情値検出  
- 任意の感情が閾値(0.7)以上

### 基本キーワード
- 反応語: 「すごい」「やばい」「最高」「ひどい」
- 感嘆詞: 「おー」「うわー」「えー」

## 受け入れ基準
- [ ] 明確な感情変化を検出可能
- [ ] ハイライトレベルが適切に分類される
- [ ] 設定ファイルで閾値変更可能
- [ ] 基本的なキーワード検出が動作

## 見積もり
**ストーリーポイント**: 8  
**予想工数**: 3日

## 関連Issue
- 感情分析機能実装 (#4)
EOF
)"
```

### 6. CSV/JSON出力機能

```bash
gh issue create \
  --title "📄 CSV/JSON出力機能実装" \
  --label "feature,phase-1" \
  --milestone "MVP" \
  --body "$(cat <<'EOF'
## 概要
分析結果をCSVとJSON形式で出力する機能を実装

## タスク
- [ ] CSV出力フォーマッター (動画編集ソフト用)
- [ ] JSON出力フォーマッター (詳細データ)
- [ ] メタデータ情報の含有
- [ ] 統計情報の生成
- [ ] ファイル名の自動生成

## 出力仕様

### CSV形式 (動画編集ソフト対応)
```csv
timestamp_start,timestamp_end,text,emotion_score,emotion_type,highlight_level,keywords
00:01:23.45,00:01:27.12,"すごく嬉しいです！",0.85,joy,high,"すごく"
```

### JSON形式 (詳細データ)
- メタデータ (動画ファイル情報、処理時刻、モデルバージョン)
- セグメント詳細データ
- ハイライト一覧
- 統計情報

## 受け入れ基準
- [ ] CSV形式でAdobe Premiere Pro等にインポート可能
- [ ] JSON形式で全詳細データが含まれる
- [ ] ファイル名が重複しない自動命名
- [ ] エラー時に適切なメッセージ表示

## 見積もり
**ストーリーポイント**: 5  
**予想工数**: 2日

## 関連Issue
- ハイライト検出機能 (#5)
EOF
)"
```

### 7. テストスイート作成

```bash
gh issue create \
  --title "🧪 テストスイート作成" \
  --label "testing,phase-1" \
  --milestone "MVP" \
  --body "$(cat <<'EOF'
## 概要
各機能の単体テストと統合テストを作成

## タスク
- [ ] pytest環境セットアップ
- [ ] 音声抽出機能のテスト
- [ ] Whisper統合のテスト (モックを使用)
- [ ] 感情分析機能のテスト
- [ ] ハイライト検出のテスト
- [ ] 出力機能のテスト
- [ ] 統合テストシナリオ作成

## テスト方針
- **単体テスト**: 各モジュールの個別機能
- **統合テスト**: エンドツーエンドの処理フロー
- **モックテスト**: 外部依存関係のテスト
- **サンプルデータ**: テスト用の小サイズ動画

## 受け入れ基準
- [ ] 各モジュールで80%以上のコードカバレッジ
- [ ] CI/CDでテストが自動実行される
- [ ] テスト用サンプルデータが準備される
- [ ] テスト実行時間が合理的

## 見積もり
**ストーリーポイント**: 8  
**予想工数**: 3日

## 関連Issue
- 全ての機能実装Issue
EOF
)"
```

### 8. エラーハンドリング強化

```bash
gh issue create \
  --title "⚠️ エラーハンドリング強化" \
  --label "bug,phase-1" \
  --milestone "MVP" \
  --body "$(cat <<'EOF'
## 概要
想定されるエラーケースに対する適切なハンドリングを実装

## タスク
- [ ] ファイル関連エラー処理
- [ ] FFmpeg関連エラー処理  
- [ ] Whisper関連エラー処理
- [ ] メモリ不足エラー処理
- [ ] ネットワーク関連エラー処理
- [ ] ユーザーフレンドリーなエラーメッセージ
- [ ] ログレベル別の出力制御

## 想定エラーケース

### ファイル関連
- 存在しないファイル
- 権限不足
- 破損ファイル
- サポート外形式

### システム関連
- FFmpeg未インストール
- メモリ不足  
- ディスク容量不足
- GPU利用不可

### 処理関連
- 音声抽出失敗
- 文字起こし失敗
- 感情分析失敗

## 受け入れ基準
- [ ] 全ての想定エラーで適切なメッセージ表示
- [ ] ログファイルに詳細エラー記録
- [ ] ユーザーが対処法を理解できる説明
- [ ] プログラムが予期しない終了をしない

## 見積もり
**ストーリーポイント**: 5  
**予想工数**: 2日

## 関連Issue
- 全ての機能実装Issue
EOF
)"
```

---

## 🏷 ラベル設定

以下のラベルを事前に作成してください：

```bash
# 機能カテゴリ
gh label create "feature" --color "0075ca" --description "新機能"
gh label create "enhancement" --color "a2eeef" --description "機能改善"
gh label create "bug" --color "d73a4a" --description "バグ修正"
gh label create "testing" --color "1d76db" --description "テスト関連"

# 優先度
gh label create "priority-high" --color "d93f0b" --description "高優先度"
gh label create "priority-medium" --color "fbca04" --description "中優先度"  
gh label create "priority-low" --color "0e8a16" --description "低優先度"

# フェーズ
gh label create "phase-1" --color "5319e7" --description "Phase 1: MVP"
gh label create "phase-2" --color "7057ff" --description "Phase 2: 機能強化"
gh label create "phase-3" --color "8b5cf6" --description "Phase 3: 高度化"
```

## 📊 マイルストーン設定

```bash
gh milestone create "MVP" --description "Phase 1: 最小機能製品" --due-date "2024-07-07"
```

---

## 📝 Issue作成後の作業

1. **Project作成**: GitHub ProjectsでKanbanボード作成
2. **ブランチ戦略**: 各Issue用のfeatureブランチ作成
3. **CI/CD設定**: GitHub Actionsでテスト自動化
4. **レビュープロセス**: プルリクエスト必須とするブランチ保護設定

## 🚀 開発開始コマンド

すべてのIssue作成後、以下で開発を開始できます：

```bash
# 最初のIssueで開発開始
git checkout -b feature/project-foundation
# 開発作業...
git commit -m "feat: implement project foundation"
git push origin feature/project-foundation
gh pr create --title "🏗 プロジェクト基盤構築" --body "Closes #1"
```