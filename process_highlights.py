#!/usr/bin/env python3
"""
純粋関数ベースハイライト検出プロセス

Input: 
  - data/stage3_emotions/{video_id}_text_emotions.json
Output:
  - data/stage4_highlights/{video_id}_highlights_{config_name}.json

Usage:
  python process_highlights.py --input data/stage3_emotions/video_text_emotions.json --output data/stage4_highlights/ --config config/highlight_detection.yaml

純粋関数設計:
- 同じ入力に対して同じ出力
- 副作用なし（ファイル作成以外）
- 高速パラメーター調整対応（最重要プロセス）
- 設定名でファイル名を区別し、複数設定の並列実行可能
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import click
import yaml

# プロジェクトルートをPythonパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


@dataclass
class HighlightDetectionConfig:
    """ハイライト検出設定（イミュータブル）"""
    emotion_change_threshold: float = 0.1
    high_emotion_threshold: float = 0.2
    keyword_weight: float = 3.0
    duration_weight: float = 1.0
    min_highlight_duration: float = 0.5
    min_highlight_score: float = 0.1
    max_highlights: int = 15
    min_confidence: float = 0.3
    merge_distance: float = 1.0
    config_name: str = "default"
    
    # エンゲージメントパターン（技術系に特化）
    engagement_patterns: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.engagement_patterns is None:
            self.engagement_patterns = {
                "questions": ["どう", "なぜ", "何", "どこ", "いつ", "誰", "どれ", "どの", "？", "?"],
                "exclamations": ["！", "すごい", "やばい", "ヤバい", "マジ", "本当", "ほんと", "えー", "うわー", "おー"],
                "emphasis": ["これ", "この", "その", "あの", "特に", "実際", "本当に", "マジで", "絶対", "確実"],
                "reactions": ["あー", "おー", "うー", "へー", "ふーん", "なるほど", "そうか", "そっか", "わかる"],
                "technical_interest": ["面白い", "興味深い", "便利", "使える", "いい", "良い", "素晴らしい", "完璧"],
                "emotional_markers": ["感動", "驚", "ショック", "びっくり", "嬉しい", "楽しい", "困った", "大変"],
                "programming_terms": ["エラー", "バグ", "デバッグ", "コード", "プログラム", "アルゴリズム", "データ"],
                "completion_markers": ["完成", "成功", "達成", "解決", "できた", "動いた", "うまくいった"]
            }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HighlightDetectionConfig':
        """辞書から設定を作成"""
        valid_keys = {k for k in cls.__annotations__ if k != 'engagement_patterns'}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        config = cls(**filtered_dict)
        
        # engagement_patternsの個別設定
        if 'engagement_patterns' in config_dict:
            config.engagement_patterns.update(config_dict['engagement_patterns'])
        
        return config


@dataclass
class HighlightSegment:
    """ハイライトセグメント（イミュータブル）"""
    start: float
    end: float
    text: str
    highlight_score: float
    dominant_emotion: str
    emotions: Dict[str, float]
    confidence: float
    rank: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def duration(self) -> float:
        """セグメントの継続時間"""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'highlight_score': self.highlight_score,
            'dominant_emotion': self.dominant_emotion,
            'emotions': self.emotions,
            'confidence': self.confidence,
            'duration': self.duration(),
            'rank': self.rank,
            'metadata': self.metadata or {}
        }


@dataclass
class HighlightDetectionResult:
    """ハイライト検出結果（イミュータブル）"""
    video_id: str
    config_name: str
    input_emotions_path: str
    config: HighlightDetectionConfig
    highlights: List[HighlightSegment]
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['config'] = asdict(self.config)
        result['highlights'] = [h.to_dict() for h in self.highlights]
        return result


def calculate_highlight_scores_pure(
    emotion_segments: List[Dict[str, Any]], 
    config: HighlightDetectionConfig
) -> List[Dict[str, Any]]:
    """
    各セグメントのハイライトスコアを計算する純粋関数
    
    Args:
        emotion_segments: 感情分析済みセグメントのリスト
        config: ハイライト検出設定
        
    Returns:
        List[Dict[str, Any]]: スコア付きセグメントのリスト
    """
    scored_segments = []
    
    for segment in emotion_segments:
        emotions = segment.get("emotions", {})
        text = segment.get("text", "")
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        duration = end - start
        
        # 基本感情スコア
        non_neutral_emotions = {k: v for k, v in emotions.items() if k != "neutral"}
        if non_neutral_emotions:
            max_emotion_score = max(non_neutral_emotions.values())
        else:
            # neutral優勢の場合、低いベースラインスコア
            neutral_score = emotions.get("neutral", 0.0)
            max_emotion_score = max(0.05, neutral_score * 0.2)
        
        # 感情強度ボーナス
        emotion_bonus = 0.0
        if max_emotion_score >= config.high_emotion_threshold:
            emotion_bonus = (max_emotion_score - config.high_emotion_threshold) * 2.0
        
        # エンゲージメントパターンボーナス
        engagement_bonus = 0.0
        text_lower = text.lower()
        
        for pattern_type, patterns in config.engagement_patterns.items():
            pattern_count = 0
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    pattern_count += 1
                # 部分マッチも評価
                elif len(pattern) >= 3 and any(pattern.lower()[:3] in word for word in text_lower.split()):
                    pattern_count += 0.5
            
            if pattern_count > 0:
                # パターンタイプごとに異なる重み付け
                type_weights = {
                    "questions": 0.4,
                    "exclamations": 0.35,
                    "emotional_markers": 0.3,
                    "programming_terms": 0.25,
                    "completion_markers": 0.3,
                    "technical_interest": 0.2,
                    "emphasis": 0.15,
                    "reactions": 0.1
                }
                
                weight = type_weights.get(pattern_type, 0.1)
                engagement_bonus += pattern_count * weight * config.keyword_weight
        
        # 継続時間ボーナス
        duration_bonus = 0.0
        if duration > 1.0:  # 1秒以上の場合
            duration_bonus = min(0.15, (duration - 1.0) * 0.03) * config.duration_weight
        
        # 総合スコア計算
        base_score = max_emotion_score
        total_bonus = emotion_bonus + engagement_bonus + duration_bonus
        highlight_score = base_score + total_bonus
        
        # メタデータ作成
        metadata = {
            "base_emotion_score": max_emotion_score,
            "emotion_bonus": emotion_bonus,
            "engagement_bonus": engagement_bonus,
            "duration_bonus": duration_bonus,
            "total_bonus": total_bonus,
            "duration": duration
        }
        
        # スコア付きセグメントを作成
        scored_segment = {
            **segment,
            "highlight_score": highlight_score,
            "metadata": metadata
        }
        
        scored_segments.append(scored_segment)
    
    logger.info(f"{len(scored_segments)}セグメントのハイライトスコアを計算しました")
    return scored_segments


def apply_threshold_filters_pure(
    candidates: List[Dict[str, Any]], 
    config: HighlightDetectionConfig
) -> List[Dict[str, Any]]:
    """
    閾値ベースでハイライト候補をフィルタリングする純粋関数
    
    Args:
        candidates: ハイライト候補のリスト
        config: ハイライト検出設定
        
    Returns:
        List[Dict[str, Any]]: フィルタリング済みハイライトのリスト
    """
    filtered = []
    
    for candidate in candidates:
        score = candidate.get("highlight_score", 0.0)
        duration = candidate.get("end", 0.0) - candidate.get("start", 0.0)
        confidence = candidate.get("confidence", 1.0)
        
        # スコア閾値チェック
        if score < config.min_highlight_score:
            continue
            
        # 継続時間閾値チェック
        if duration < config.min_highlight_duration:
            continue
        
        # 信頼度閾値チェック
        if confidence < config.min_confidence:
            continue
        
        filtered.append(candidate)
    
    logger.info(f"閾値フィルタリング: {len(candidates)} → {len(filtered)}個")
    return filtered


def merge_nearby_highlights_pure(
    highlights: List[Dict[str, Any]], 
    config: HighlightDetectionConfig
) -> List[Dict[str, Any]]:
    """
    近接するハイライトを統合する純粋関数
    
    Args:
        highlights: ハイライトのリスト
        config: ハイライト検出設定
        
    Returns:
        List[Dict[str, Any]]: 統合済みハイライトのリスト
    """
    if len(highlights) <= 1:
        return highlights[:]
    
    # 開始時間でソート
    sorted_highlights = sorted(highlights, key=lambda x: x.get("start", 0.0))
    merged = []
    current_group = [sorted_highlights[0]]
    
    for highlight in sorted_highlights[1:]:
        current_start = highlight.get("start", 0.0)
        last_end = current_group[-1].get("end", 0.0)
        
        # 近接または重複チェック
        if current_start <= last_end + config.merge_distance:
            current_group.append(highlight)
        else:
            # 現在のグループを統合して追加
            merged_highlight = merge_highlight_group_pure(current_group)
            merged.append(merged_highlight)
            current_group = [highlight]
    
    # 最後のグループを処理
    if current_group:
        merged_highlight = merge_highlight_group_pure(current_group)
        merged.append(merged_highlight)
    
    logger.info(f"ハイライト統合: {len(highlights)} → {len(merged)}個")
    return merged


def merge_highlight_group_pure(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    ハイライトグループを1つに統合するヘルパー関数
    
    Args:
        group: 統合対象のハイライトグループ
        
    Returns:
        Dict[str, Any]: 統合されたハイライト
    """
    if len(group) == 1:
        return group[0]
    
    # 最初と最後の時間を取得
    start_time = min(h.get("start", 0.0) for h in group)
    end_time = max(h.get("end", 0.0) for h in group)
    
    # 最高スコアのハイライトを基準にする
    best_highlight = max(group, key=lambda x: x.get("highlight_score", 0.0))
    
    # テキストを結合
    texts = [h.get("text", "") for h in group if h.get("text", "").strip()]
    combined_text = " ".join(texts)
    
    # 統合されたハイライトを作成
    merged = {
        **best_highlight,
        "start": start_time,
        "end": end_time,
        "text": combined_text,
        "duration": end_time - start_time,
        "merged_count": len(group),
        "metadata": {
            **best_highlight.get("metadata", {}),
            "merged_from": len(group)
        }
    }
    
    return merged


def rank_highlights_by_score_pure(
    highlights: List[Dict[str, Any]], 
    config: HighlightDetectionConfig
) -> List[HighlightSegment]:
    """
    ハイライトをスコア順にランキングする純粋関数
    
    Args:
        highlights: ハイライトのリスト
        config: ハイライト検出設定
        
    Returns:
        List[HighlightSegment]: ランキング済みハイライトのリスト
    """
    # スコア降順でソート、同点時は継続時間の長い順
    sorted_highlights = sorted(
        highlights,
        key=lambda x: (x.get("highlight_score", 0.0), x.get("end", 0.0) - x.get("start", 0.0)),
        reverse=True
    )
    
    # 最大数で切り取り
    top_highlights = sorted_highlights[:config.max_highlights]
    
    # HighlightSegmentオブジェクトに変換してランキング情報を付与
    ranked_highlights = []
    for i, highlight in enumerate(top_highlights):
        highlight_segment = HighlightSegment(
            start=highlight.get("start", 0.0),
            end=highlight.get("end", 0.0),
            text=highlight.get("text", ""),
            highlight_score=highlight.get("highlight_score", 0.0),
            dominant_emotion=highlight.get("dominant_emotion", "neutral"),
            emotions=highlight.get("emotions", {}),
            confidence=highlight.get("confidence", 0.0),
            rank=i + 1,
            metadata=highlight.get("metadata", {})
        )
        ranked_highlights.append(highlight_segment)
    
    logger.info(f"ハイライトランキング: 上位{len(ranked_highlights)}個を選択")
    return ranked_highlights


def calculate_summary_statistics(
    highlights: List[HighlightSegment],
    total_segments: int
) -> Dict[str, Any]:
    """
    ハイライト検出結果のサマリー統計を計算する純粋関数
    
    Args:
        highlights: ハイライトのリスト
        total_segments: 総セグメント数
        
    Returns:
        Dict[str, Any]: サマリー統計
    """
    if not highlights:
        return {
            "total_highlights": 0,
            "average_score": 0.0,
            "total_duration": 0.0,
            "top_emotions": [],
            "emotion_distribution": {},
            "score_distribution": {}
        }
    
    # 基本統計
    total_duration = sum(h.duration() for h in highlights)
    average_score = sum(h.highlight_score for h in highlights) / len(highlights)
    
    # 感情分布
    emotion_counts = {}
    for highlight in highlights:
        emotion = highlight.dominant_emotion
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    top_emotions = sorted(emotion_counts.keys(), key=emotion_counts.get, reverse=True)
    
    # スコア分布
    scores = [h.highlight_score for h in highlights]
    score_distribution = {
        "min": min(scores),
        "max": max(scores),
        "mean": average_score,
        "median": sorted(scores)[len(scores) // 2] if scores else 0.0
    }
    
    return {
        "total_highlights": len(highlights),
        "average_score": round(average_score, 3),
        "total_duration": round(total_duration, 1),
        "coverage_ratio": round(total_duration / (total_segments * 2.0) * 100, 1) if total_segments > 0 else 0.0,
        "top_emotions": top_emotions,
        "emotion_distribution": emotion_counts,
        "score_distribution": score_distribution
    }


def highlight_detection_workflow(
    input_emotions_path: str,
    output_dir: str,
    config: HighlightDetectionConfig
) -> HighlightDetectionResult:
    """
    ハイライト検出ワークフローを実行する純粋関数
    
    Args:
        input_emotions_path: 入力感情分析ファイルのパス
        output_dir: 出力ディレクトリ
        config: ハイライト検出設定
        
    Returns:
        HighlightDetectionResult: ハイライト検出結果
    """
    start_time = time.time()
    
    try:
        # パス正規化
        emotions_path = Path(input_emotions_path).resolve()
        output_path = Path(output_dir).resolve()
        
        # 検証
        if not emotions_path.exists():
            raise ValueError(f"感情分析ファイルが見つかりません: {emotions_path}")
        
        # 出力ディレクトリ作成
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 動画IDを取得
        video_id = emotions_path.stem.replace('_text_emotions', '').replace('_audio_emotions', '')
        
        # 感情分析結果読み込み
        with open(emotions_path, 'r', encoding='utf-8') as f:
            emotions_data = json.load(f)
        
        segments = emotions_data.get('segments', [])
        if not segments:
            raise ValueError("感情分析セグメントが見つかりません")
        
        logger.info(f"感情分析結果を読み込み: {len(segments)}セグメント")
        
        # 1. ハイライトスコア計算
        logger.info("ハイライトスコア計算中...")
        scored_segments = calculate_highlight_scores_pure(segments, config)
        
        # 2. 閾値フィルタリング
        logger.info("閾値フィルタリング中...")
        filtered_candidates = apply_threshold_filters_pure(scored_segments, config)
        
        # 3. 近接ハイライト統合
        logger.info("ハイライト統合中...")
        merged_highlights = merge_nearby_highlights_pure(filtered_candidates, config)
        
        # 4. ランキング
        logger.info("ランキング中...")
        ranked_highlights = rank_highlights_by_score_pure(merged_highlights, config)
        
        # 5. サマリー統計計算
        summary = calculate_summary_statistics(ranked_highlights, len(segments))
        
        # メタデータ作成
        processing_time = time.time() - start_time
        metadata = {
            "total_segments": len(segments),
            "candidates_found": len(filtered_candidates),
            "merged_highlights": len(merged_highlights),
            "final_highlights": len(ranked_highlights),
            "processing_time": processing_time,
            "config_hash": hashlib.md5(str(asdict(config)).encode()).hexdigest()[:8],
            "timestamp": time.time()
        }
        
        # 結果作成
        result = HighlightDetectionResult(
            video_id=video_id,
            config_name=config.config_name,
            input_emotions_path=str(emotions_path),
            config=config,
            highlights=ranked_highlights,
            metadata=metadata,
            summary=summary,
            processing_time=processing_time,
            success=True
        )
        
        logger.info(f"ハイライト検出ワークフロー完了: {processing_time:.2f}秒")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"ハイライト検出ワークフローエラー: {error_msg}")
        
        return HighlightDetectionResult(
            video_id=Path(input_emotions_path).stem.replace('_text_emotions', '').replace('_audio_emotions', ''),
            config_name=config.config_name,
            input_emotions_path=str(input_emotions_path),
            config=config,
            highlights=[],
            metadata={},
            summary={},
            processing_time=processing_time,
            success=False,
            error_message=error_msg
        )


def save_highlight_result(result: HighlightDetectionResult, output_dir: Path) -> None:
    """
    ハイライト検出結果をJSONファイルに保存する純粋関数
    
    Args:
        result: ハイライト検出結果
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file_path = output_dir / f"{result.video_id}_highlights_{result.config_name}.json"
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"ハイライト検出結果を保存: {output_file_path}")


def load_config(config_path: str) -> HighlightDetectionConfig:
    """
    設定ファイルを読み込む純粋関数
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        HighlightDetectionConfig: ハイライト検出設定
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        return HighlightDetectionConfig()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() == '.yaml':
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # highlight_detection設定セクションを取得
        highlight_config = config_dict.get('highlight_detection', {})
        
        # config_nameを設定ファイル名から自動設定
        if 'config_name' not in highlight_config:
            highlight_config['config_name'] = config_file.stem
        
        return HighlightDetectionConfig.from_dict(highlight_config)
        
    except Exception as e:
        logger.warning(f"設定ファイル読み込みエラー: {e}")
        return HighlightDetectionConfig()


@click.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='入力感情分析ファイルのパス')
@click.option('--output', '-o', 'output_dir', default='data/stage4_highlights',
              help='出力ディレクトリ (デフォルト: data/stage4_highlights)')
@click.option('--config', '-c', 'config_path', default='config/highlight_detection.yaml',
              help='設定ファイルのパス')
@click.option('--config-name', default=None,
              help='設定名（出力ファイル名に使用、指定しない場合は設定ファイル名を使用）')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(input_path: str, output_dir: str, config_path: str, config_name: str, verbose: bool):
    """
    純粋関数ベースハイライト検出プロセス
    
    INPUT_PATH: 処理する感情分析ファイルのパス
    """
    # ログ設定
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=== ハイライト検出プロセス開始 ===")
    logger.info(f"入力: {input_path}")
    logger.info(f"出力: {output_dir}")
    logger.info(f"設定: {config_path}")
    
    try:
        # 設定読み込み
        config = load_config(config_path)
        
        # コマンドライン引数で設定名を上書き
        if config_name:
            config.config_name = config_name
        
        logger.info(f"ハイライト検出設定: {config.config_name}")
        logger.info(f"  スコア閾値: {config.min_highlight_score}")
        logger.info(f"  感情閾値: {config.high_emotion_threshold}")
        logger.info(f"  キーワード重み: {config.keyword_weight}")
        logger.info(f"  最大ハイライト数: {config.max_highlights}")
        
        # ハイライト検出実行
        result = highlight_detection_workflow(input_path, output_dir, config)
        
        if result.success:
            # 結果保存
            save_highlight_result(result, Path(output_dir))
            
            logger.info("=== ハイライト検出プロセス完了 ===")
            logger.info(f"処理時間: {result.processing_time:.2f}秒")
            logger.info(f"検出ハイライト数: {len(result.highlights)}")
            logger.info(f"平均スコア: {result.summary.get('average_score', 0.0):.3f}")
            logger.info(f"総時間: {result.summary.get('total_duration', 0.0)}秒")
            logger.info(f"カバレッジ: {result.summary.get('coverage_ratio', 0.0)}%")
            
            # トップ感情分布を表示
            emotion_dist = result.summary.get('emotion_distribution', {})
            if emotion_dist:
                logger.info("感情分布:")
                for emotion, count in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {emotion}: {count}個")
            
            click.echo(f"✅ ハイライト検出成功: {len(result.highlights)}個")
            
        else:
            logger.error(f"ハイライト検出失敗: {result.error_message}")
            click.echo(f"❌ ハイライト検出失敗: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        click.echo(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()