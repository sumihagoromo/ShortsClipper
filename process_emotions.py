#!/usr/bin/env python3
"""
純粋関数ベース感情分析プロセス

Input: 
  - data/stage2_transcript/{video_id}_cleaned.json
Output:
  - data/stage3_emotions/{video_id}_text_emotions.json

Usage:
  python process_emotions.py --input data/stage2_transcript/video_cleaned.json --output data/stage3_emotions/ --config config/emotion_analysis.yaml

純粋関数設計:
- 同じ入力に対して同じ出力
- 副作用なし（ファイル作成以外）
- 高速パラメーター調整対応（テキスト感情分析は軽い処理）
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import click
import yaml

# プロジェクトルートをPythonパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_config import setup_process_logging

logger = logging.getLogger(__name__)


@dataclass
class EmotionAnalysisConfig:
    """感情分析設定（イミュータブル）"""
    provider: str = "ml-ask"
    batch_size: int = 10
    confidence_threshold: float = 0.5
    normalize_scores: bool = True
    emotion_categories: List[str] = None
    
    def __post_init__(self):
        if self.emotion_categories is None:
            self.emotion_categories = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmotionAnalysisConfig':
        """辞書から設定を作成"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class EmotionSegment:
    """感情分析済みセグメント（イミュータブル）"""
    start: float
    end: float
    text: str
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    original_segment: Dict[str, Any] = None
    
    def duration(self) -> float:
        """セグメントの継続時間"""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'emotions': self.emotions,
            'dominant_emotion': self.dominant_emotion,
            'confidence': self.confidence,
            'duration': self.duration()
        }


@dataclass
class EmotionAnalysisResult:
    """感情分析結果（イミュータブル）"""
    video_id: str
    input_transcript_path: str
    config: EmotionAnalysisConfig
    segments: List[EmotionSegment]
    statistics: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['config'] = asdict(self.config)
        result['segments'] = [seg.to_dict() for seg in self.segments]
        return result


def initialize_emotion_analyzer(config: EmotionAnalysisConfig):
    """
    感情分析器を初期化する関数
    
    Args:
        config: 感情分析設定
        
    Returns:
        感情分析器インスタンス
    """
    try:
        if config.provider == "ml-ask":
            try:
                import oseti
                analyzer = oseti.Analyzer()
                logger.info("ML-Ask (oseti) 感情分析器を初期化しました")
                return analyzer
            except ImportError:
                logger.warning("osetiが見つかりません。代替実装を使用します")
                return create_mock_analyzer()
        else:
            logger.warning(f"未サポートのプロバイダー: {config.provider}。代替実装を使用します")
            return create_mock_analyzer()
            
    except Exception as e:
        logger.warning(f"感情分析器初期化失敗: {e}。代替実装を使用します")
        return create_mock_analyzer()


def create_mock_analyzer():
    """
    簡易感情分析器を作成する関数（oseti未インストール時のフォールバック）
    """
    class MockAnalyzer:
        def __init__(self):
            # 簡単な感情キーワード辞書
            self.positive_words = [
                "嬉しい", "楽しい", "素晴らしい", "最高", "感動", "良い", "いい", "すごい",
                "ありがと", "感謝", "幸せ", "満足", "成功", "達成", "完璧", "素敵"
            ]
            self.negative_words = [
                "悲しい", "残念", "つらい", "困った", "大変", "悪い", "だめ", "失敗",
                "問題", "エラー", "心配", "不安", "怒り", "イライラ", "疲れ", "しんどい"
            ]
            self.surprise_words = [
                "驚", "びっくり", "まさか", "信じられない", "すごい", "やばい", "えー", "おー"
            ]
        
        def analyze(self, text: str) -> float:
            """
            簡易感情分析
            """
            if not text or not text.strip():
                return 0.0
            
            text_lower = text.lower()
            
            pos_count = sum(1 for word in self.positive_words if word in text)
            neg_count = sum(1 for word in self.negative_words if word)
            surprise_count = sum(1 for word in self.surprise_words if word in text)
            
            # 単純な極性スコア
            if pos_count > neg_count:
                return min(0.8, pos_count * 0.3)
            elif neg_count > pos_count:
                return max(-0.8, -neg_count * 0.3)
            elif surprise_count > 0:
                return 0.5  # 中程度のポジティブ
            else:
                return 0.0
    
    return MockAnalyzer()


def analyze_text_emotion_pure(analyzer, text: str) -> Dict[str, float]:
    """
    単一テキストの感情分析を行う純粋関数
    
    Args:
        analyzer: 感情分析器
        text: 分析対象テキスト
        
    Returns:
        Dict[str, float]: 感情スコア辞書
    """
    if not text or not text.strip():
        return {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 1.0
        }
    
    try:
        scores = analyzer.analyze(text)
        
        if isinstance(scores, dict):
            # 辞書形式の場合はそのまま使用
            emotion_scores = scores
        else:
            # 単一値の場合は極性に基づいて変換
            if scores > 0.3:
                emotion_scores = {
                    "joy": float(scores),
                    "sadness": 0.0,
                    "anger": 0.0,
                    "fear": 0.0,
                    "surprise": 0.1,
                    "neutral": max(0.0, 1.0 - float(scores))
                }
            elif scores < -0.3:
                emotion_scores = {
                    "joy": 0.0,
                    "sadness": float(abs(scores)),
                    "anger": float(abs(scores)) * 0.5,
                    "fear": float(abs(scores)) * 0.3,
                    "surprise": 0.0,
                    "neutral": max(0.0, 1.0 - float(abs(scores)))
                }
            else:
                emotion_scores = {
                    "joy": max(0.0, float(scores)) * 0.5,
                    "sadness": 0.0,
                    "anger": 0.0,
                    "fear": 0.0,
                    "surprise": abs(float(scores)) * 0.3,
                    "neutral": 0.8
                }
        
        # 欠損している感情カテゴリを補完
        standard_emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        for emotion in standard_emotions:
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0.0
        
        return emotion_scores
        
    except Exception as e:
        logger.warning(f"感情分析に失敗しました（テキスト: {text[:20]}...）: {e}")
        return {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 1.0
        }


def normalize_emotion_scores(
    raw_scores: Dict[str, float], 
    confidence_threshold: float = 0.5
) -> Dict[str, float]:
    """
    感情スコアを正規化する純粋関数
    
    Args:
        raw_scores: 生の感情スコア
        confidence_threshold: 信頼度閾値
        
    Returns:
        Dict[str, float]: 正規化された感情スコア
    """
    total_score = sum(abs(score) for score in raw_scores.values())
    
    if total_score == 0:
        num_emotions = len(raw_scores)
        return {emotion: 1.0 / num_emotions for emotion in raw_scores.keys()}
    
    # 正規化
    normalized = {}
    for emotion, score in raw_scores.items():
        normalized[emotion] = max(0.0, score) / total_score
    
    # 信頼度チェック
    max_score = max(normalized.values())
    if max_score < confidence_threshold:
        neutral_boost = confidence_threshold - max_score
        normalized["neutral"] = min(1.0, normalized.get("neutral", 0.0) + neutral_boost)
        
        # 再正規化
        total = sum(normalized.values())
        if total > 0:
            normalized = {emotion: score / total for emotion, score in normalized.items()}
    
    return normalized


def batch_analyze_emotions_pure(
    analyzer,
    segments: List[Dict[str, Any]],
    config: EmotionAnalysisConfig,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[EmotionSegment]:
    """
    複数のセグメントを一括で感情分析する純粋関数
    
    Args:
        analyzer: 感情分析器
        segments: 分析対象セグメントのリスト
        config: 感情分析設定
        progress_callback: 進捗コールバック
        
    Returns:
        List[EmotionSegment]: 感情分析結果のリスト
    """
    batch_size = config.batch_size
    confidence_threshold = config.confidence_threshold
    normalize_scores = config.normalize_scores
    
    emotion_segments = []
    total_segments = len(segments)
    
    logger.info(f"感情分析開始: {total_segments}セグメント")
    
    for i in range(0, total_segments, batch_size):
        batch = segments[i:i + batch_size]
        
        if progress_callback:
            progress = min(i + batch_size, total_segments)
            progress_callback(f"感情分析中 ({progress}/{total_segments})")
        
        for segment in batch:
            text = segment.get("text", "")
            
            # 感情分析実行
            raw_emotion_scores = analyze_text_emotion_pure(analyzer, text)
            
            # 正規化
            if normalize_scores:
                emotion_scores = normalize_emotion_scores(raw_emotion_scores, confidence_threshold)
            else:
                emotion_scores = raw_emotion_scores
            
            # 支配的感情を計算
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            # EmotionSegmentを作成
            emotion_segment = EmotionSegment(
                start=segment.get("start", 0.0),
                end=segment.get("end", 0.0),
                text=text,
                emotions=emotion_scores,
                dominant_emotion=dominant_emotion,
                confidence=confidence,
                original_segment=segment
            )
            
            emotion_segments.append(emotion_segment)
    
    logger.info(f"感情分析完了: {len(emotion_segments)}セグメント")
    return emotion_segments


def calculate_emotion_statistics(emotion_segments: List[EmotionSegment]) -> Dict[str, Any]:
    """
    感情分析結果の統計を計算する純粋関数
    
    Args:
        emotion_segments: 感情分析結果のリスト
        
    Returns:
        Dict[str, Any]: 感情統計
    """
    if not emotion_segments:
        return {
            "average_emotions": {},
            "dominant_emotion_distribution": {},
            "emotion_changes": 0,
            "total_duration": 0.0,
            "segments_count": 0,
            "emotion_intensity": 0.0
        }
    
    # 平均感情スコアを計算
    emotion_sums = {}
    dominant_counts = {}
    total_duration = 0.0
    emotion_changes = 0
    previous_dominant = None
    
    for segment in emotion_segments:
        emotions = segment.emotions
        dominant = segment.dominant_emotion
        duration = segment.duration()
        
        # 感情スコアの累積
        for emotion, score in emotions.items():
            emotion_sums[emotion] = emotion_sums.get(emotion, 0.0) + score
        
        # 支配的感情の分布
        dominant_counts[dominant] = dominant_counts.get(dominant, 0) + 1
        
        # 感情変化回数
        if previous_dominant is not None and previous_dominant != dominant:
            emotion_changes += 1
        previous_dominant = dominant
        
        total_duration += duration
    
    # 平均を計算
    segments_count = len(emotion_segments)
    average_emotions = {
        emotion: total / segments_count 
        for emotion, total in emotion_sums.items()
    }
    
    # 支配的感情の分布を割合に変換
    dominant_distribution = {
        emotion: count / segments_count 
        for emotion, count in dominant_counts.items()
    }
    
    # 感情強度（neutralを除いた平均）
    non_neutral_emotions = {k: v for k, v in average_emotions.items() if k != "neutral"}
    emotion_intensity = sum(non_neutral_emotions.values()) / len(non_neutral_emotions) if non_neutral_emotions else 0.0
    
    return {
        "average_emotions": average_emotions,
        "dominant_emotion_distribution": dominant_distribution,
        "emotion_changes": emotion_changes,
        "total_duration": total_duration,
        "segments_count": segments_count,
        "emotion_intensity": emotion_intensity
    }


def emotion_analysis_workflow(
    input_transcript_path: str,
    output_dir: str,
    config: EmotionAnalysisConfig
) -> EmotionAnalysisResult:
    """
    感情分析ワークフローを実行する純粋関数
    
    Args:
        input_transcript_path: 入力文字起こしファイルのパス
        output_dir: 出力ディレクトリ
        config: 感情分析設定
        
    Returns:
        EmotionAnalysisResult: 感情分析結果
    """
    start_time = time.time()
    
    try:
        # パス正規化
        transcript_path = Path(input_transcript_path).resolve()
        output_path = Path(output_dir).resolve()
        
        # 検証
        if not transcript_path.exists():
            raise ValueError(f"文字起こしファイルが見つかりません: {transcript_path}")
        
        # 出力ディレクトリ作成
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 動画IDを取得
        video_id = transcript_path.stem.replace('_cleaned', '').replace('_raw', '')
        
        # 文字起こし結果読み込み
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        segments = transcript_data.get('segments', [])
        if not segments:
            raise ValueError("文字起こしセグメントが見つかりません")
        
        logger.info(f"文字起こし結果を読み込み: {len(segments)}セグメント")
        
        # 感情分析器初期化
        logger.info("感情分析器を初期化中...")
        analyzer = initialize_emotion_analyzer(config)
        
        # 感情分析実行
        logger.info("感情分析を実行中...")
        
        def progress_callback(message):
            logger.info(message)
        
        emotion_segments = batch_analyze_emotions_pure(
            analyzer, segments, config, progress_callback
        )
        
        # 統計計算
        logger.info("統計を計算中...")
        statistics = calculate_emotion_statistics(emotion_segments)
        
        # メタデータ作成
        processing_time = time.time() - start_time
        metadata = {
            "provider": config.provider,
            "confidence_threshold": config.confidence_threshold,
            "segments_analyzed": len(emotion_segments),
            "total_duration": statistics["total_duration"],
            "emotion_changes": statistics["emotion_changes"],
            "timestamp": time.time()
        }
        
        # 結果作成
        result = EmotionAnalysisResult(
            video_id=video_id,
            input_transcript_path=str(transcript_path),
            config=config,
            segments=emotion_segments,
            statistics=statistics,
            metadata=metadata,
            processing_time=processing_time,
            success=True
        )
        
        logger.info(f"感情分析ワークフロー完了: {processing_time:.2f}秒")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"感情分析ワークフローエラー: {error_msg}")
        
        return EmotionAnalysisResult(
            video_id=Path(input_transcript_path).stem.replace('_cleaned', '').replace('_raw', ''),
            input_transcript_path=str(input_transcript_path),
            config=config,
            segments=[],
            statistics={},
            metadata={},
            processing_time=processing_time,
            success=False,
            error_message=error_msg
        )


def save_emotion_result(result: EmotionAnalysisResult, output_dir: Path) -> None:
    """
    感情分析結果をJSONファイルに保存する純粋関数
    
    Args:
        result: 感情分析結果
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file_path = output_dir / f"{result.video_id}_text_emotions.json"
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"感情分析結果を保存: {output_file_path}")


def load_config(config_path: str) -> EmotionAnalysisConfig:
    """
    設定ファイルを読み込む純粋関数
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        EmotionAnalysisConfig: 感情分析設定
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        return EmotionAnalysisConfig()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() == '.yaml':
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # emotion_analysis設定セクションを取得
        emotion_config = config_dict.get('emotion_analysis', {})
        return EmotionAnalysisConfig.from_dict(emotion_config)
        
    except Exception as e:
        logger.warning(f"設定ファイル読み込みエラー: {e}")
        return EmotionAnalysisConfig()


@click.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='入力文字起こしファイルのパス')
@click.option('--output', '-o', 'output_dir', default='data/stage3_emotions',
              help='出力ディレクトリ (デフォルト: data/stage3_emotions)')
@click.option('--config', '-c', 'config_path', default='config/emotion_analysis.yaml',
              help='設定ファイルのパス')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(input_path: str, output_dir: str, config_path: str, verbose: bool):
    """
    純粋関数ベース感情分析プロセス
    
    INPUT_PATH: 処理する文字起こしファイルのパス
    """
    # ログ設定
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=== 感情分析プロセス開始 ===")
    logger.info(f"入力: {input_path}")
    logger.info(f"出力: {output_dir}")
    logger.info(f"設定: {config_path}")
    
    try:
        # 設定読み込み
        config = load_config(config_path)
        logger.info(f"感情分析設定: {asdict(config)}")
        
        # 感情分析実行
        result = emotion_analysis_workflow(input_path, output_dir, config)
        
        if result.success:
            # 結果保存
            save_emotion_result(result, Path(output_dir))
            
            logger.info("=== 感情分析プロセス完了 ===")
            logger.info(f"処理時間: {result.processing_time:.2f}秒")
            logger.info(f"分析セグメント数: {len(result.segments)}")
            logger.info(f"感情変化回数: {result.statistics.get('emotion_changes', 0)}")
            logger.info(f"感情強度: {result.statistics.get('emotion_intensity', 0.0):.3f}")
            
            # 支配的感情分布を表示
            distribution = result.statistics.get('dominant_emotion_distribution', {})
            for emotion, ratio in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {emotion}: {ratio:.1%}")
            
            click.echo(f"✅ 感情分析成功: {len(result.segments)}セグメント")
            
        else:
            logger.error(f"感情分析失敗: {result.error_message}")
            click.echo(f"❌ 感情分析失敗: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        click.echo(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()