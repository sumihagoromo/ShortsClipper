"""
ハイライト検出モジュール

関数型プログラミングの原則に従って実装:
- 純粋関数（副作用なし）
- イミュータブルなデータ
- 関数の合成
- 明示的なエラーハンドリング
"""

import logging
import re
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HighlightSegment:
    """イミュータブルなハイライトセグメント"""
    start: float
    end: float
    text: str
    highlight_score: float
    dominant_emotion: str
    emotions: Dict[str, float]
    rank: Optional[int] = None
    
    def duration(self) -> float:
        """セグメントの継続時間を取得"""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "highlight_score": self.highlight_score,
            "dominant_emotion": self.dominant_emotion,
            "emotions": self.emotions,
            "rank": self.rank,
            "duration": self.duration()
        }


@dataclass(frozen=True)
class EmotionChange:
    """イミュータブルな感情変化点"""
    change_point: float
    from_emotion: str
    to_emotion: str
    intensity: float
    confidence: float


def create_highlight_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    ハイライト検出設定を作成する純粋関数
    
    Args:
        base_config: 基本設定辞書
        
    Returns:
        Dict[str, Any]: 完全なハイライト検出設定
    """
    # デフォルト設定
    default_config = {
        "emotion_change_threshold": 0.3,
        "high_emotion_threshold": 0.7,
        "keyword_weight": 1.5,
        "duration_weight": 1.2,
        "min_highlight_duration": 2.0,
        "min_highlight_score": 0.5,
        "max_highlights": 10,
        "merge_distance": 1.0,
        "min_confidence": 0.5,
        "engagement_patterns": {
            "questions": ["どう", "なぜ", "何", "どこ", "いつ", "誰", "どれ", "どの", "？"],
            "exclamations": ["！", "すごい", "やばい", "ヤバい", "マジ", "本当", "ほんと", "えー", "うわー"],
            "emphasis": ["これ", "この", "その", "あの", "特に", "実際", "本当に", "マジで", "絶対"],
            "reactions": ["あー", "おー", "うー", "へー", "ふーん", "なるほど", "そうか", "そっか"],
            "technical_interest": ["面白い", "興味深い", "すごい", "便利", "使える", "いい", "良い", "悪い"],
            "emotional_markers": ["感動", "驚", "ショック", "びっくり", "嬉しい", "楽しい", "困った", "大変"]
        }
    }
    
    # ベース設定をマージ
    merged_config = {**default_config, **base_config}
    
    # 閾値の妥当性チェック
    if merged_config["emotion_change_threshold"] < 0 or merged_config["emotion_change_threshold"] > 1:
        merged_config["emotion_change_threshold"] = 0.3
        logger.warning("無効な感情変化閾値を修正しました")
    
    if merged_config["high_emotion_threshold"] < 0 or merged_config["high_emotion_threshold"] > 1:
        merged_config["high_emotion_threshold"] = 0.7
        logger.warning("無効な高感情閾値を修正しました")
    
    return merged_config


def detect_emotion_changes(
    emotion_data: List[Dict[str, Any]], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    感情変化点を検出する純粋関数
    
    Args:
        emotion_data: 感情分析結果のリスト
        config: ハイライト検出設定
        
    Returns:
        List[Dict[str, Any]]: 感情変化点のリスト
    """
    if len(emotion_data) < 2:
        return []
    
    change_threshold = config.get("emotion_change_threshold", 0.3)
    min_confidence = config.get("min_confidence", 0.5)
    
    changes = []
    previous_segment = None
    
    for segment in emotion_data:
        current_emotion = segment.get("dominant_emotion", "neutral")
        current_confidence = segment.get("confidence", 0.0)
        
        # 信頼度チェック
        if current_confidence < min_confidence:
            continue
            
        if previous_segment is not None:
            prev_emotion = previous_segment.get("dominant_emotion", "neutral")
            prev_confidence = previous_segment.get("confidence", 0.0)
            
            # 前のセグメントの信頼度もチェック
            if prev_confidence < min_confidence:
                previous_segment = segment
                continue
            
            # 感情変化を検出
            if current_emotion != prev_emotion:
                # 変化強度を計算
                current_emotions = segment.get("emotions", {})
                prev_emotions = previous_segment.get("emotions", {})
                
                # 現在の支配的感情のスコアと前の支配的感情のスコア
                current_score = current_emotions.get(current_emotion, 0.0)
                prev_score = prev_emotions.get(prev_emotion, 0.0)
                
                # 変化の強度を計算（より敏感に）
                intensity = max(current_score, prev_score)
                
                # 閾値を超える変化のみ記録（デフォルト0.3なので大部分の変化が検出される）
                if intensity >= change_threshold:
                    change = {
                        "change_point": segment.get("start", 0.0),
                        "from_emotion": prev_emotion,
                        "to_emotion": current_emotion,
                        "intensity": intensity,
                        "confidence": min(current_confidence, prev_confidence)
                    }
                    changes.append(change)
        
        previous_segment = segment
    
    logger.info(f"感情変化点を{len(changes)}個検出しました")
    return changes


def calculate_highlight_scores(
    segments: List[Dict[str, Any]], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    各セグメントのハイライトスコアを計算する純粋関数
    
    Args:
        segments: 感情分析済みセグメントのリスト
        config: ハイライト検出設定
        
    Returns:
        List[Dict[str, Any]]: スコア付きセグメントのリスト
    """
    high_emotion_threshold = config.get("high_emotion_threshold", 0.7)
    keyword_weight = config.get("keyword_weight", 1.5)
    duration_weight = config.get("duration_weight", 1.2)
    emotion_keywords = config.get("emotion_keywords", {})
    
    scored_segments = []
    
    for segment in segments:
        emotions = segment.get("emotions", {})
        text = segment.get("text", "")
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        duration = end - start
        
        # 基本感情スコア（neutralのみの場合でもベースラインスコアを提供）
        if emotions:
            # neutralを除く感情から最高値を取得
            non_neutral_emotions = {k: v for k, v in emotions.items() if k != "neutral"}
            if non_neutral_emotions:
                max_emotion_score = max(non_neutral_emotions.values())
            else:
                # neutral優勢の場合、低いベースラインスコアを設定（感情分析の制限を補完）
                neutral_score = emotions.get("neutral", 0.0)
                max_emotion_score = max(0.1, neutral_score * 0.3)  # neutralでも最低0.1のスコア
        else:
            max_emotion_score = 0.1  # データが無い場合のベースライン
        
        # 感情強度ボーナス
        emotion_bonus = 0.0
        if max_emotion_score >= high_emotion_threshold:
            emotion_bonus = (max_emotion_score - high_emotion_threshold) * 2.0
        
        # エンゲージメントパターンボーナス（従来のキーワードボーナスを置き換え）
        engagement_bonus = 0.0
        engagement_patterns = config.get("engagement_patterns", {})
        
        # 様々なエンゲージメントパターンを評価（日本語転写エラー対応付き）
        for pattern_type, patterns in engagement_patterns.items():
            pattern_count = 0
            for pattern in patterns:
                # 基本マッチング（大文字小文字を区別しない）
                if pattern.lower() in text.lower():
                    pattern_count += 1
                # 日本語転写エラー対応の近似マッチング
                elif _fuzzy_japanese_match(pattern, text):
                    pattern_count += 0.7  # 近似マッチは重みを下げる
            
            if pattern_count > 0:
                # パターンタイプごとに異なる重み付け
                if pattern_type == "questions":
                    engagement_bonus += pattern_count * 0.3 * keyword_weight  # 質問は高評価
                elif pattern_type == "exclamations":
                    engagement_bonus += pattern_count * 0.25 * keyword_weight  # 感嘆も高評価
                elif pattern_type == "emotional_markers":
                    engagement_bonus += pattern_count * 0.2 * keyword_weight  # 感情マーカー
                else:
                    engagement_bonus += pattern_count * 0.15 * keyword_weight  # その他のパターン
        
        # 従来のキーワードボーナスとの互換性
        keyword_bonus = engagement_bonus
        
        # 継続時間ボーナス
        duration_bonus = 0.0
        if duration > 2.0:  # 2秒以上の場合
            duration_bonus = min(0.2, (duration - 2.0) * 0.05) * duration_weight
        
        # 総合スコア計算
        base_score = max_emotion_score
        total_bonus = emotion_bonus + keyword_bonus + duration_bonus
        highlight_score = base_score + total_bonus  # 上限を撤廃してより区別しやすく
        
        # セグメント情報を保持
        scored_segment = {
            **segment,  # 元の情報を保持
            "highlight_score": highlight_score,
            "duration": duration,
            "emotion_bonus": emotion_bonus,
            "keyword_bonus": keyword_bonus,
            "duration_bonus": duration_bonus,
            "keyword_boost": keyword_bonus  # テスト用にエイリアスも追加
        }
        
        scored_segments.append(scored_segment)
    
    logger.info(f"{len(scored_segments)}セグメントのハイライトスコアを計算しました")
    return scored_segments


def apply_threshold_filters(
    candidates: List[Dict[str, Any]], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    閾値ベースでハイライト候補をフィルタリングする純粋関数
    
    Args:
        candidates: ハイライト候補のリスト
        config: ハイライト検出設定
        
    Returns:
        List[Dict[str, Any]]: フィルタリング済みハイライトのリスト
    """
    min_score = config.get("min_highlight_score", 0.5)
    min_duration = config.get("min_highlight_duration", 2.0)
    
    filtered = []
    
    for candidate in candidates:
        score = candidate.get("highlight_score", 0.0)
        duration = candidate.get("duration", 0.0)
        
        # スコア閾値チェック
        if score < min_score:
            continue
            
        # 継続時間閾値チェック
        if duration < min_duration:
            continue
        
        filtered.append(candidate)
    
    logger.info(f"閾値フィルタリング: {len(candidates)} → {len(filtered)}個")
    return filtered


def merge_nearby_highlights(
    highlights: List[Dict[str, Any]], 
    config: Dict[str, Any]
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
    
    merge_distance = config.get("merge_distance", 1.0)
    
    # 開始時間でソート
    sorted_highlights = sorted(highlights, key=lambda x: x.get("start", 0.0))
    merged = []
    current_group = [sorted_highlights[0]]
    
    for highlight in sorted_highlights[1:]:
        current_start = highlight.get("start", 0.0)
        last_end = current_group[-1].get("end", 0.0)
        
        # 近接または重複チェック
        if current_start <= last_end + merge_distance:
            # 統合対象
            current_group.append(highlight)
        else:
            # 現在のグループを統合して追加
            merged_highlight = _merge_highlight_group(current_group)
            merged.append(merged_highlight)
            
            # 新しいグループを開始
            current_group = [highlight]
    
    # 最後のグループを処理
    if current_group:
        merged_highlight = _merge_highlight_group(current_group)
        merged.append(merged_highlight)
    
    logger.info(f"ハイライト統合: {len(highlights)} → {len(merged)}個")
    return merged


def _merge_highlight_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    texts = [h.get("text", "") for h in group]
    combined_text = " ".join(filter(None, texts))
    
    # 統合されたハイライトを作成
    merged = {
        **best_highlight,  # 基準ハイライトの属性を継承
        "start": start_time,
        "end": end_time,
        "text": combined_text,
        "duration": end_time - start_time,
        "merged_count": len(group)
    }
    
    return merged


def rank_highlights_by_score(
    highlights: List[Dict[str, Any]], 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    ハイライトをスコア順にランキングする純粋関数
    
    Args:
        highlights: ハイライトのリスト
        config: ハイライト検出設定
        
    Returns:
        List[Dict[str, Any]]: ランキング済みハイライトのリスト
    """
    max_highlights = config.get("max_highlights", 10)
    
    # スコア降順でソート、同点時は継続時間の長い順
    sorted_highlights = sorted(
        highlights,
        key=lambda x: (x.get("highlight_score", 0.0), x.get("duration", 0.0)),
        reverse=True
    )
    
    # 最大数で切り取り
    top_highlights = sorted_highlights[:max_highlights]
    
    # ランキング情報を付与
    ranked_highlights = []
    for i, highlight in enumerate(top_highlights):
        ranked_highlight = {
            **highlight,
            "rank": i + 1
        }
        ranked_highlights.append(ranked_highlight)
    
    logger.info(f"ハイライトランキング: 上位{len(ranked_highlights)}個を選択")
    return ranked_highlights


def format_highlight_results(
    highlights: List[Dict[str, Any]], 
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    ハイライト検出結果をフォーマットする純粋関数
    
    Args:
        highlights: ハイライトのリスト
        metadata: メタデータ
        
    Returns:
        Dict[str, Any]: フォーマット済み結果
    """
    if not highlights:
        return {
            "highlights": [],
            "metadata": {
                "total_highlights": 0,
                "total_segments": metadata.get("total_segments", 0),
                "processing_time": metadata.get("processing_time", 0.0)
            },
            "summary": {
                "top_emotions": [],
                "average_score": 0.0,
                "total_duration": 0.0
            }
        }
    
    # サマリー統計を計算
    total_duration = sum(h.get("duration", 0.0) for h in highlights)
    average_score = sum(h.get("highlight_score", 0.0) for h in highlights) / len(highlights)
    
    # 上位感情を計算
    emotion_counts = {}
    for highlight in highlights:
        emotion = highlight.get("dominant_emotion", "neutral")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    top_emotions = sorted(emotion_counts.keys(), key=emotion_counts.get, reverse=True)
    
    return {
        "highlights": highlights,
        "metadata": {
            "total_highlights": len(highlights),
            "total_segments": metadata.get("total_segments", 0),
            "candidates_found": metadata.get("candidates_found", 0),
            "processing_time": metadata.get("processing_time", 0.0)
        },
        "summary": {
            "top_emotions": top_emotions,
            "average_score": round(average_score, 3),
            "total_duration": round(total_duration, 1),
            "emotion_distribution": emotion_counts
        }
    }


def full_highlight_detection_workflow(
    emotion_result: Dict[str, Any],
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    ハイライト検出の全体ワークフローを実行する関数
    
    Args:
        emotion_result: 感情分析結果
        config: ハイライト検出設定
        progress_callback: 進捗コールバック関数
        
    Returns:
        Dict[str, Any]: 完全なハイライト検出結果
        
    Raises:
        ValueError: 入力データエラー
        Exception: 処理エラー
    """
    segments = emotion_result.get("segments", [])
    if not segments:
        raise ValueError("感情分析セグメントが見つかりません")
    
    import time
    start_time = time.time()
    
    # 1. ハイライト検出設定作成
    highlight_config = create_highlight_config(config)
    logger.info(f"ハイライト検出設定: 閾値={highlight_config['high_emotion_threshold']}")
    
    # 2. 感情変化検出
    if progress_callback:
        progress_callback(1, 6, "感情変化点検出中...")
    emotion_changes = detect_emotion_changes(segments, highlight_config)
    
    # 3. ハイライトスコア計算
    if progress_callback:
        progress_callback(2, 6, "ハイライトスコア計算中...")
    scored_segments = calculate_highlight_scores(segments, highlight_config)
    
    # 4. 閾値フィルタリング
    if progress_callback:
        progress_callback(3, 6, "閾値フィルタリング中...")
    filtered_candidates = apply_threshold_filters(scored_segments, highlight_config)
    
    # 5. 近接ハイライト統合
    if progress_callback:
        progress_callback(4, 6, "ハイライト統合中...")
    merged_highlights = merge_nearby_highlights(filtered_candidates, highlight_config)
    
    # 6. ランキング
    if progress_callback:
        progress_callback(5, 6, "ランキング中...")
    ranked_highlights = rank_highlights_by_score(merged_highlights, highlight_config)
    
    # 7. 結果フォーマット
    if progress_callback:
        progress_callback(6, 6, "結果フォーマット中...")
    
    processing_time = time.time() - start_time
    metadata = {
        "total_segments": len(segments),
        "candidates_found": len(filtered_candidates),
        "emotion_changes": len(emotion_changes),
        "processing_time": processing_time
    }
    
    formatted_result = format_highlight_results(ranked_highlights, metadata)
    
    logger.info("ハイライト検出ワークフロー完了")
    return formatted_result


# 関数合成のヘルパー
def compose_highlight_pipeline(*functions):
    """
    ハイライト検出関数を合成してパイプラインを作成する高階関数
    """
    def pipeline(initial_value):
        result = initial_value
        for func in functions:
            result = func(result)
        return result
    return pipeline


# キーワード検出用のヘルパー関数
def extract_engagement_patterns(text: str, pattern_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    テキストからエンゲージメントパターンを抽出する純粋関数
    
    Args:
        text: 分析対象テキスト
        pattern_dict: エンゲージメントパターン辞書
        
    Returns:
        Dict[str, List[str]]: 検出されたパターン
    """
    found_patterns = {}
    text_lower = text.lower()
    
    for pattern_type, patterns in pattern_dict.items():
        found = []
        for pattern in patterns:
            # 大文字小文字を区別しない検索
            if pattern.lower() in text_lower:
                found.append(pattern)
        
        if found:
            found_patterns[pattern_type] = found
    
    return found_patterns


def _fuzzy_japanese_match(pattern: str, text: str) -> bool:
    """
    日本語転写エラーと表記揺れに対応した近似マッチング関数
    
    Args:
        pattern: 検索パターン
        text: 検索対象テキスト
        
    Returns:
        bool: 近似マッチした場合True
    """
    import re
    
    # 共通の転写エラーパターン
    transcription_variants = {
        # カタカナ・ひらがな相互変換の一般的なケース
        'ク': ['く', 'ッ'],
        'ド': ['ど', 'ト'],
        'コ': ['こ', 'ゴ'],
        'エ': ['え', 'ェ'],
        'ン': ['ん'],
        # 長音記号の有無
        'ー': [''],
        # 促音・拗音
        'ッ': ['っ', ''],
        'ャ': ['や', 'ゃ'],
        'ュ': ['ゆ', 'ゅ'],
        'ョ': ['よ', 'ょ'],
        # 語尾変化
        'です': ['だ', 'である'],
        'ます': ['る', '']
    }
    
    pattern_lower = pattern.lower()
    text_lower = text.lower()
    
    # 1. 部分一致チェック（短縮形）
    if len(pattern) >= 3:
        # パターンの最初の3文字が含まれているかチェック
        if pattern_lower[:3] in text_lower:
            return True
        # パターンの最後の3文字が含まれているかチェック
        if len(pattern) > 3 and pattern_lower[-3:] in text_lower:
            return True
    
    # 2. 文字レベルでの近似マッチング（編集距離ベース）
    if len(pattern) >= 2 and len(text) >= 2:
        # 簡単なレーベンシュタイン距離近似
        max_distance = max(1, len(pattern) // 3)  # パターン長の1/3まで差異を許容
        if _simple_edit_distance(pattern_lower, text_lower) <= max_distance:
            return True
    
    # 3. 共通の転写エラーに対する置換チェック
    pattern_variants = [pattern_lower]
    for original, replacements in transcription_variants.items():
        if original in pattern_lower:
            for replacement in replacements:
                pattern_variants.append(pattern_lower.replace(original, replacement))
    
    for variant in pattern_variants:
        if variant and variant in text_lower:
            return True
    
    return False


def _simple_edit_distance(s1: str, s2: str) -> int:
    """
    簡単な編集距離計算（パフォーマンス重視）
    
    Args:
        s1, s2: 比較する文字列
        
    Returns:
        int: 編集距離
    """
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    # 長い文字列は切り捨てて高速化
    max_len = 10
    s1 = s1[:max_len]
    s2 = s2[:max_len]
    
    # 2行のDPテーブルで空間効率化
    prev_row = list(range(len(s2) + 1))
    curr_row = [0] * (len(s2) + 1)
    
    for i in range(1, len(s1) + 1):
        curr_row[0] = i
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                curr_row[j] = prev_row[j-1]
            else:
                curr_row[j] = 1 + min(prev_row[j], curr_row[j-1], prev_row[j-1])
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[len(s2)]


def calculate_emotion_intensity(emotions: Dict[str, float]) -> float:
    """
    感情の強度を計算する純粋関数
    
    Args:
        emotions: 感情スコア辞書
        
    Returns:
        float: 感情強度（0-1）
    """
    if not emotions:
        return 0.0
    
    # 最大値と分散を考慮した強度計算
    max_score = max(emotions.values())
    variance = sum((score - max_score) ** 2 for score in emotions.values()) / len(emotions)
    
    # 高い感情と低い分散ほど強い強度
    intensity = max_score * (1 - min(0.5, variance))
    return intensity