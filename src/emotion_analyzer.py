"""
感情分析モジュール

関数型プログラミングの原則に従って実装:
- 純粋関数（副作用なし）
- イミュータブルなデータ
- 関数の合成
- 明示的なエラーハンドリング
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import math


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmotionScore:
    """イミュータブルな感情スコア"""
    joy: float
    sadness: float
    anger: float
    fear: float
    surprise: float
    neutral: float
    
    def to_dict(self) -> Dict[str, float]:
        """辞書形式に変換"""
        return {
            "joy": self.joy,
            "sadness": self.sadness,
            "anger": self.anger,
            "fear": self.fear,
            "surprise": self.surprise,
            "neutral": self.neutral
        }
    
    def dominant_emotion(self) -> str:
        """最も高いスコアの感情を返す"""
        scores = self.to_dict()
        return max(scores, key=scores.get)


@dataclass(frozen=True)
class EmotionResult:
    """イミュータブルな感情分析結果"""
    text: str
    emotions: EmotionScore
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None


def create_emotion_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    感情分析設定を作成する純粋関数
    
    Args:
        base_config: 基本設定辞書
        
    Returns:
        Dict[str, Any]: 完全な感情分析設定
    """
    # デフォルト設定
    default_config = {
        "provider": "ml-ask",
        "batch_size": 10,
        "confidence_threshold": 0.5,
        "normalize_scores": True,
        "emotion_categories": ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
    }
    
    # ベース設定をマージ
    merged_config = {**default_config, **base_config}
    
    # プロバイダーに基づくモデル選択
    provider_models = {
        "ml-ask": "oseti",
        "empath": "empath_api",
        "local": "transformers_model"
    }
    
    merged_config["model_type"] = provider_models.get(
        merged_config["provider"], 
        "oseti"
    )
    
    return merged_config


def initialize_emotion_analyzer(config: Dict[str, Any]):
    """
    感情分析器を初期化する関数
    
    Args:
        config: 感情分析設定
        
    Returns:
        感情分析器インスタンス
        
    Raises:
        ValueError: サポートされていないプロバイダーの場合
        Exception: 初期化に失敗した場合
    """
    provider = config.get("provider", "ml-ask")
    
    try:
        if provider == "ml-ask":
            # osetiを使用
            import oseti
            analyzer = oseti.Analyzer()
            logger.info("ML-Ask (oseti) 感情分析器を初期化しました")
            return analyzer
            
        elif provider == "empath":
            # Empath APIの場合（将来の拡張用）
            logger.warning("Empath APIは未実装です。ML-Askを使用します。")
            import oseti
            return oseti.Analyzer()
            
        else:
            raise ValueError(f"サポートされていないプロバイダー: {provider}")
            
    except ImportError as e:
        error_msg = f"感情分析ライブラリのインポートに失敗しました: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
    except Exception as e:
        error_msg = f"感情分析器の初期化に失敗しました: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def analyze_text_emotion(analyzer, text: str) -> Dict[str, float]:
    """
    単一テキストの感情分析を行う関数
    
    Args:
        analyzer: 感情分析器
        text: 分析対象テキスト
        
    Returns:
        Dict[str, float]: 感情スコア辞書
    """
    if not text or not text.strip():
        # 空文字列の場合は中性的な結果を返す
        return {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 1.0
        }
    
    try:
        # osetiによる感情分析
        scores = analyzer.analyze(text)
        
        # スコアの形式を統一（osetiの出力形式に依存）
        if isinstance(scores, dict):
            # 辞書形式の場合はそのまま使用
            emotion_scores = scores
        else:
            # 単一値の場合は極性に基づいて変換
            if scores > 0:
                emotion_scores = {
                    "joy": float(scores),
                    "sadness": 0.0,
                    "anger": 0.0,
                    "fear": 0.0,
                    "surprise": 0.0,
                    "neutral": 1.0 - float(abs(scores))
                }
            elif scores < 0:
                emotion_scores = {
                    "joy": 0.0,
                    "sadness": float(abs(scores)),
                    "anger": 0.0,
                    "fear": 0.0,
                    "surprise": 0.0,
                    "neutral": 1.0 - float(abs(scores))
                }
            else:
                emotion_scores = {
                    "joy": 0.0,
                    "sadness": 0.0,
                    "anger": 0.0,
                    "fear": 0.0,
                    "surprise": 0.0,
                    "neutral": 1.0
                }
        
        # 欠損している感情カテゴリを補完
        standard_emotions = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
        for emotion in standard_emotions:
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0.0
        
        return emotion_scores
        
    except Exception as e:
        logger.warning(f"感情分析に失敗しました（テキスト: {text[:20]}...）: {e}")
        # エラー時は中性的な結果を返す
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
    # 合計スコアを計算
    total_score = sum(abs(score) for score in raw_scores.values())
    
    if total_score == 0:
        # 全て0の場合は均等に分散
        num_emotions = len(raw_scores)
        return {emotion: 1.0 / num_emotions for emotion in raw_scores.keys()}
    
    # 正規化
    normalized = {}
    for emotion, score in raw_scores.items():
        normalized[emotion] = max(0.0, score) / total_score
    
    # 信頼度チェック
    max_score = max(normalized.values())
    if max_score < confidence_threshold:
        # 信頼度が低い場合はneutralを優勢にする
        neutral_boost = confidence_threshold - max_score
        normalized["neutral"] = min(1.0, normalized.get("neutral", 0.0) + neutral_boost)
        
        # 再正規化
        total = sum(normalized.values())
        if total > 0:
            normalized = {emotion: score / total for emotion, score in normalized.items()}
    
    return normalized


def batch_analyze_emotions(
    analyzer,
    segments: List[Dict[str, Any]],
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> List[Dict[str, Any]]:
    """
    複数のセグメントを一括で感情分析する関数
    
    Args:
        analyzer: 感情分析器
        segments: 分析対象セグメントのリスト
        config: 感情分析設定
        progress_callback: 進捗コールバック関数
        
    Returns:
        List[Dict[str, Any]]: 感情分析結果のリスト
    """
    batch_size = config.get("batch_size", 10)
    confidence_threshold = config.get("confidence_threshold", 0.5)
    normalize_scores = config.get("normalize_scores", True)
    
    results = []
    total_segments = len(segments)
    
    logger.info(f"感情分析開始: {total_segments}セグメント")
    
    for i in range(0, total_segments, batch_size):
        batch = segments[i:i + batch_size]
        
        # 進捗更新
        if progress_callback:
            progress_callback(
                min(i + batch_size, total_segments),
                total_segments,
                f"感情分析中 ({min(i + batch_size, total_segments)}/{total_segments})"
            )
        
        # バッチ内の各セグメントを処理
        for segment in batch:
            text = segment.get("text", "")
            
            # 感情分析実行
            raw_emotion_scores = analyze_text_emotion(analyzer, text)
            
            # 正規化
            if normalize_scores:
                emotion_scores = normalize_emotion_scores(raw_emotion_scores, confidence_threshold)
            else:
                emotion_scores = raw_emotion_scores
            
            # 支配的感情を計算
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            # 結果を作成
            result = {
                **segment,  # 元のセグメント情報を保持
                "emotions": emotion_scores,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence
            }
            
            results.append(result)
    
    logger.info(f"感情分析完了: {len(results)}セグメント")
    return results


def merge_emotion_with_timestamps(
    transcription_segments: List[Dict[str, Any]],
    emotion_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    転写結果と感情分析結果を統合する純粋関数
    
    Args:
        transcription_segments: 転写セグメント
        emotion_results: 感情分析結果
        
    Returns:
        List[Dict[str, Any]]: 統合された結果
    """
    # 短い方の長さに合わせる
    min_length = min(len(transcription_segments), len(emotion_results))
    
    merged_results = []
    for i in range(min_length):
        transcription = transcription_segments[i]
        emotion = emotion_results[i]
        
        # 統合結果を作成
        merged = {
            "start": transcription.get("start", 0.0),
            "end": transcription.get("end", 0.0),
            "text": transcription.get("text", ""),
            "emotions": emotion.get("emotions", {}),
            "dominant_emotion": emotion.get("dominant_emotion", "neutral"),
            "confidence": emotion.get("confidence", 0.0)
        }
        
        merged_results.append(merged)
    
    if len(transcription_segments) != len(emotion_results):
        logger.warning(
            f"セグメント数不一致: 転写={len(transcription_segments)}, "
            f"感情分析={len(emotion_results)}, 使用={min_length}"
        )
    
    return merged_results


def calculate_emotion_statistics(emotion_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    感情分析結果の統計を計算する純粋関数
    
    Args:
        emotion_data: 感情分析結果のリスト
        
    Returns:
        Dict[str, Any]: 感情統計
    """
    if not emotion_data:
        return {
            "average_emotions": {},
            "dominant_emotion_distribution": {},
            "emotion_changes": 0,
            "total_duration": 0.0,
            "segments_count": 0
        }
    
    # 平均感情スコアを計算
    emotion_sums = {}
    dominant_counts = {}
    total_duration = 0.0
    emotion_changes = 0
    previous_dominant = None
    
    for segment in emotion_data:
        emotions = segment.get("emotions", {})
        dominant = segment.get("dominant_emotion", "neutral")
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        duration = end - start
        
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
    segments_count = len(emotion_data)
    average_emotions = {
        emotion: total / segments_count 
        for emotion, total in emotion_sums.items()
    }
    
    # 支配的感情の分布を割合に変換
    dominant_distribution = {
        emotion: count / segments_count 
        for emotion, count in dominant_counts.items()
    }
    
    return {
        "average_emotions": average_emotions,
        "dominant_emotion_distribution": dominant_distribution,
        "emotion_changes": emotion_changes,
        "total_duration": total_duration,
        "segments_count": segments_count,
        "emotion_intensity": sum(average_emotions.values()) / len(average_emotions) if average_emotions else 0.0
    }


def full_emotion_analysis_workflow(
    transcription_result: Dict[str, Any],
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    感情分析の全体ワークフローを実行する関数
    
    Args:
        transcription_result: 転写結果
        config: 感情分析設定
        progress_callback: 進捗コールバック関数
        
    Returns:
        Dict[str, Any]: 完全な感情分析結果
        
    Raises:
        ValueError: 入力データエラー
        Exception: 処理エラー
    """
    segments = transcription_result.get("segments", [])
    if not segments:
        raise ValueError("転写セグメントが見つかりません")
    
    # 1. 感情分析設定作成
    emotion_config = create_emotion_config(config)
    logger.info(f"感情分析設定: {emotion_config['provider']} (バッチサイズ: {emotion_config['batch_size']})")
    
    # 2. 感情分析器初期化
    if progress_callback:
        progress_callback(1, 4, "感情分析器初期化中...")
    analyzer = initialize_emotion_analyzer(emotion_config)
    
    # 3. バッチ感情分析実行
    if progress_callback:
        progress_callback(2, 4, "感情分析実行中...")
    emotion_results = batch_analyze_emotions(analyzer, segments, emotion_config, progress_callback)
    
    # 4. タイムスタンプとの統合
    if progress_callback:
        progress_callback(3, 4, "結果統合中...")
    merged_segments = merge_emotion_with_timestamps(segments, emotion_results)
    
    # 5. 統計計算
    if progress_callback:
        progress_callback(4, 4, "統計計算中...")
    statistics = calculate_emotion_statistics(merged_segments)
    
    logger.info("感情分析ワークフロー完了")
    
    return {
        "segments": merged_segments,
        "statistics": statistics,
        "metadata": {
            "provider": emotion_config["provider"],
            "confidence_threshold": emotion_config["confidence_threshold"],
            "segments_analyzed": len(merged_segments),
            "total_duration": statistics["total_duration"]
        }
    }


# 関数合成のヘルパー
def compose_emotion_pipeline(*functions):
    """
    感情分析関数を合成してパイプラインを作成する高階関数
    """
    def pipeline(initial_value):
        result = initial_value
        for func in functions:
            result = func(result)
        return result
    return pipeline