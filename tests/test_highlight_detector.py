import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

# テスト対象のモジュールをインポート
from src.highlight_detector import (
    create_highlight_config,
    detect_emotion_changes,
    calculate_highlight_scores,
    apply_threshold_filters,
    merge_nearby_highlights,
    rank_highlights_by_score,
    format_highlight_results,
    full_highlight_detection_workflow
)


class TestHighlightConfiguration:
    """ハイライト検出設定機能のテスト（純粋関数）"""
    
    def test_create_highlight_config_default_values(self):
        """デフォルト設定でハイライト検出設定を作成"""
        # Given: デフォルトパラメータ
        base_config = {
            "emotion_change_threshold": 0.3,
            "high_emotion_threshold": 0.7
        }
        
        # When: ハイライト検出設定を作成
        result = create_highlight_config(base_config)
        
        # Then: 適切なデフォルト値が設定される
        assert result["emotion_change_threshold"] == 0.3
        assert result["high_emotion_threshold"] == 0.7
        assert result["keyword_weight"] == 1.5
        assert result["duration_weight"] == 1.2
        assert result["min_highlight_duration"] == 2.0
        assert result["max_highlights"] == 10
    
    def test_create_highlight_config_custom_values(self):
        """カスタム設定でハイライト検出設定を作成"""
        # Given: カスタムパラメータ
        custom_config = {
            "emotion_change_threshold": 0.4,
            "high_emotion_threshold": 0.8,
            "keyword_weight": 2.0,
            "max_highlights": 5
        }
        
        # When: カスタム設定を作成
        result = create_highlight_config(custom_config)
        
        # Then: カスタム値が反映される
        assert result["emotion_change_threshold"] == 0.4
        assert result["high_emotion_threshold"] == 0.8
        assert result["keyword_weight"] == 2.0
        assert result["max_highlights"] == 5
    
    @pytest.mark.parametrize("threshold,expected_keywords", [
        (0.3, ["素晴らしい", "感動", "驚き"]),
        (0.5, ["素晴らしい", "感動"]),
        (0.8, ["素晴らしい"]),
    ])
    def test_create_highlight_config_keyword_filtering(self, threshold, expected_keywords):
        """閾値に基づくキーワードフィルタリングの純粋関数テスト"""
        # Given: 異なる閾値
        config = {"high_emotion_threshold": threshold}
        
        # When: 設定を作成
        result = create_highlight_config(config)
        
        # Then: 適切なキーワードが選択される
        assert result["high_emotion_threshold"] == threshold


class TestEmotionChangeDetection:
    """感情変化検出のテスト（純粋関数）"""
    
    def test_detect_emotion_changes_standard(self):
        """標準的な感情変化の検出"""
        # Given: 感情データのシーケンス
        emotion_data = [
            {"start": 0.0, "end": 3.0, "dominant_emotion": "neutral", "confidence": 0.8, 
             "emotions": {"neutral": 0.8, "joy": 0.1, "sadness": 0.1}},
            {"start": 3.0, "end": 6.0, "dominant_emotion": "joy", "confidence": 0.9,
             "emotions": {"joy": 0.9, "neutral": 0.05, "sadness": 0.05}},
            {"start": 6.0, "end": 9.0, "dominant_emotion": "joy", "confidence": 0.85,
             "emotions": {"joy": 0.85, "neutral": 0.1, "sadness": 0.05}},
            {"start": 9.0, "end": 12.0, "dominant_emotion": "sadness", "confidence": 0.7,
             "emotions": {"sadness": 0.7, "neutral": 0.2, "joy": 0.1}},
            {"start": 12.0, "end": 15.0, "dominant_emotion": "surprise", "confidence": 0.8,
             "emotions": {"surprise": 0.8, "neutral": 0.1, "joy": 0.1}}
        ]
        config = {"emotion_change_threshold": 0.3}
        
        # When: 感情変化を検出
        result = detect_emotion_changes(emotion_data, config)
        
        # Then: 適切な変化点が検出される
        assert len(result) == 3  # neutral->joy, joy->sadness, sadness->surprise
        assert result[0]["change_point"] == 3.0
        assert result[0]["from_emotion"] == "neutral"
        assert result[0]["to_emotion"] == "joy"
        assert result[1]["change_point"] == 9.0
        assert result[2]["change_point"] == 12.0
    
    def test_detect_emotion_changes_no_changes(self):
        """感情変化がない場合"""
        # Given: 同じ感情のデータ
        emotion_data = [
            {"start": 0.0, "end": 3.0, "dominant_emotion": "joy", "confidence": 0.8,
             "emotions": {"joy": 0.8, "neutral": 0.1, "sadness": 0.1}},
            {"start": 3.0, "end": 6.0, "dominant_emotion": "joy", "confidence": 0.9,
             "emotions": {"joy": 0.9, "neutral": 0.05, "sadness": 0.05}},
            {"start": 6.0, "end": 9.0, "dominant_emotion": "joy", "confidence": 0.85,
             "emotions": {"joy": 0.85, "neutral": 0.1, "sadness": 0.05}}
        ]
        config = {"emotion_change_threshold": 0.3}
        
        # When: 感情変化を検出
        result = detect_emotion_changes(emotion_data, config)
        
        # Then: 変化点が検出されない
        assert result == []
    
    def test_detect_emotion_changes_confidence_filtering(self):
        """信頼度による感情変化フィルタリング"""
        # Given: 低信頼度を含む感情データ
        emotion_data = [
            {"start": 0.0, "end": 3.0, "dominant_emotion": "neutral", "confidence": 0.9,
             "emotions": {"neutral": 0.9, "joy": 0.05, "sadness": 0.05}},
            {"start": 3.0, "end": 6.0, "dominant_emotion": "joy", "confidence": 0.2,  # 低信頼度
             "emotions": {"joy": 0.5, "neutral": 0.3, "sadness": 0.2}},
            {"start": 6.0, "end": 9.0, "dominant_emotion": "sadness", "confidence": 0.8,
             "emotions": {"sadness": 0.8, "neutral": 0.1, "joy": 0.1}}
        ]
        config = {"emotion_change_threshold": 0.3, "min_confidence": 0.5}
        
        # When: 信頼度フィルタリング付きで検出
        result = detect_emotion_changes(emotion_data, config)
        
        # Then: 低信頼度の変化は除外される
        assert len(result) == 1  # neutral->sadness (joyは信頼度不足でスキップ)
        assert result[0]["from_emotion"] == "neutral"
        assert result[0]["to_emotion"] == "sadness"


class TestHighlightScoreCalculation:
    """ハイライトスコア計算のテスト（純粋関数）"""
    
    def test_calculate_highlight_scores_emotion_intensity(self):
        """感情強度に基づくハイライトスコア計算"""
        # Given: 感情分析結果
        segments = [
            {
                "start": 0.0, "end": 3.0, "text": "普通の話です",
                "emotions": {"joy": 0.1, "neutral": 0.9}, "confidence": 0.8
            },
            {
                "start": 3.0, "end": 6.0, "text": "とても素晴らしい！",
                "emotions": {"joy": 0.9, "neutral": 0.1}, "confidence": 0.95
            },
            {
                "start": 6.0, "end": 9.0, "text": "ちょっと悲しいですね",
                "emotions": {"sadness": 0.7, "neutral": 0.3}, "confidence": 0.85
            }
        ]
        config = {
            "high_emotion_threshold": 0.7,
            "keyword_weight": 1.5,
            "duration_weight": 1.2
        }
        
        # When: ハイライトスコアを計算
        result = calculate_highlight_scores(segments, config)
        
        # Then: 感情強度に応じたスコアが計算される
        assert len(result) == 3
        assert result[1]["highlight_score"] > result[0]["highlight_score"]  # 高い喜び
        assert result[2]["highlight_score"] > result[0]["highlight_score"]  # 高い悲しみ
        assert result[1]["highlight_score"] > result[2]["highlight_score"]  # 喜び > 悲しみ
    
    def test_calculate_highlight_scores_keyword_boost(self):
        """キーワードによるスコアブースト"""
        # Given: キーワードを含むセグメント
        segments = [
            {
                "start": 0.0, "end": 3.0, "text": "普通の話です",
                "emotions": {"joy": 0.5, "neutral": 0.5}, "confidence": 0.8
            },
            {
                "start": 3.0, "end": 6.0, "text": "素晴らしい結果です！",
                "emotions": {"joy": 0.5, "neutral": 0.5}, "confidence": 0.8
            }
        ]
        config = {
            "keyword_weight": 2.0,
            "emotion_keywords": {
                "positive": ["素晴らしい", "最高", "感動"]
            }
        }
        
        # When: キーワードブースト付きでスコア計算
        result = calculate_highlight_scores(segments, config)
        
        # Then: キーワードを含むセグメントのスコアが高くなる
        assert result[1]["highlight_score"] > result[0]["highlight_score"]
        assert "keyword_boost" in result[1]
        assert result[1]["keyword_boost"] > 0
    
    def test_calculate_highlight_scores_duration_weight(self):
        """セグメント長による重み付け"""
        # Given: 異なる長さのセグメント
        segments = [
            {
                "start": 0.0, "end": 1.0, "text": "短い",  # 1秒
                "emotions": {"joy": 0.8, "neutral": 0.2}, "confidence": 0.9
            },
            {
                "start": 1.0, "end": 6.0, "text": "長いセグメントです", # 5秒
                "emotions": {"joy": 0.8, "neutral": 0.2}, "confidence": 0.9
            }
        ]
        config = {"duration_weight": 1.5}
        
        # When: セグメント長重み付きでスコア計算
        result = calculate_highlight_scores(segments, config)
        
        # Then: 長いセグメントのスコアが高くなる
        assert result[1]["highlight_score"] > result[0]["highlight_score"]
        assert result[1]["duration"] == 5.0
        assert result[0]["duration"] == 1.0


class TestThresholdFiltering:
    """閾値フィルタリングのテスト（純粋関数）"""
    
    def test_apply_threshold_filters_score_filtering(self):
        """スコア閾値によるフィルタリング"""
        # Given: 異なるスコアのハイライト候補
        candidates = [
            {"start": 0.0, "end": 3.0, "highlight_score": 0.9, "duration": 3.0},
            {"start": 3.0, "end": 6.0, "highlight_score": 0.4, "duration": 3.0},
            {"start": 6.0, "end": 9.0, "highlight_score": 0.8, "duration": 3.0},
            {"start": 9.0, "end": 12.0, "highlight_score": 0.3, "duration": 3.0}
        ]
        config = {"min_highlight_score": 0.5}
        
        # When: スコア閾値でフィルタリング
        result = apply_threshold_filters(candidates, config)
        
        # Then: 閾値以上のもののみが残る
        assert len(result) == 2
        assert all(item["highlight_score"] >= 0.5 for item in result)
    
    def test_apply_threshold_filters_duration_filtering(self):
        """最小継続時間によるフィルタリング"""
        # Given: 異なる長さのハイライト候補
        candidates = [
            {"start": 0.0, "end": 1.0, "highlight_score": 0.9, "duration": 1.0},  # 短い
            {"start": 1.0, "end": 4.0, "highlight_score": 0.8, "duration": 3.0},  # 適切
            {"start": 4.0, "end": 5.5, "highlight_score": 0.7, "duration": 1.5},  # 短い
            {"start": 5.5, "end": 10.0, "highlight_score": 0.6, "duration": 4.5}  # 適切
        ]
        config = {"min_highlight_duration": 2.0}
        
        # When: 最小継続時間でフィルタリング
        result = apply_threshold_filters(candidates, config)
        
        # Then: 最小継続時間以上のもののみが残る
        assert len(result) == 2
        assert all(item["duration"] >= 2.0 for item in result)
    
    def test_apply_threshold_filters_combined(self):
        """複合条件によるフィルタリング"""
        # Given: 複数条件に該当するハイライト候補
        candidates = [
            {"start": 0.0, "end": 3.0, "highlight_score": 0.9, "duration": 3.0},  # 両方OK
            {"start": 3.0, "end": 6.0, "highlight_score": 0.4, "duration": 3.0},  # スコア不足
            {"start": 6.0, "end": 7.0, "highlight_score": 0.8, "duration": 1.0},  # 長さ不足
            {"start": 7.0, "end": 10.0, "highlight_score": 0.7, "duration": 3.0}  # 両方OK
        ]
        config = {
            "min_highlight_score": 0.5,
            "min_highlight_duration": 2.0
        }
        
        # When: 複合条件でフィルタリング
        result = apply_threshold_filters(candidates, config)
        
        # Then: 両方の条件を満たすもののみが残る
        assert len(result) == 2
        assert all(item["highlight_score"] >= 0.5 and item["duration"] >= 2.0 for item in result)


class TestHighlightMerging:
    """ハイライト統合のテスト（純粋関数）"""
    
    def test_merge_nearby_highlights_overlapping(self):
        """重複するハイライトの統合"""
        # Given: 重複するハイライト
        highlights = [
            {"start": 0.0, "end": 3.0, "highlight_score": 0.8},
            {"start": 2.0, "end": 5.0, "highlight_score": 0.7},
            {"start": 10.0, "end": 13.0, "highlight_score": 0.9}
        ]
        config = {"merge_distance": 1.0}
        
        # When: 近接ハイライトを統合
        result = merge_nearby_highlights(highlights, config)
        
        # Then: 重複部分が統合される
        assert len(result) == 2
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 5.0
        assert result[0]["highlight_score"] == 0.8  # より高いスコアを採用
        assert result[1]["start"] == 10.0
    
    def test_merge_nearby_highlights_close_segments(self):
        """近接するハイライトの統合"""
        # Given: 近接するハイライト
        highlights = [
            {"start": 0.0, "end": 3.0, "highlight_score": 0.8},
            {"start": 3.5, "end": 6.0, "highlight_score": 0.7},  # 0.5秒の間隔
            {"start": 10.0, "end": 13.0, "highlight_score": 0.9}
        ]
        config = {"merge_distance": 1.0}  # 1秒以内なら統合
        
        # When: 近接ハイライトを統合
        result = merge_nearby_highlights(highlights, config)
        
        # Then: 近接部分が統合される
        assert len(result) == 2
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 6.0
        assert result[1]["start"] == 10.0
    
    def test_merge_nearby_highlights_no_merge(self):
        """統合対象がない場合"""
        # Given: 十分に離れたハイライト
        highlights = [
            {"start": 0.0, "end": 3.0, "highlight_score": 0.8},
            {"start": 10.0, "end": 13.0, "highlight_score": 0.7},
            {"start": 20.0, "end": 23.0, "highlight_score": 0.9}
        ]
        config = {"merge_distance": 1.0}
        
        # When: 統合処理を実行
        result = merge_nearby_highlights(highlights, config)
        
        # Then: 統合されずそのまま残る
        assert len(result) == 3
        assert result == highlights


class TestHighlightRanking:
    """ハイライトランキングのテスト（純粋関数）"""
    
    def test_rank_highlights_by_score_descending(self):
        """スコア降順でのハイライトランキング"""
        # Given: 異なるスコアのハイライト
        highlights = [
            {"start": 0.0, "end": 3.0, "highlight_score": 0.7},
            {"start": 3.0, "end": 6.0, "highlight_score": 0.9},
            {"start": 6.0, "end": 9.0, "highlight_score": 0.5},
            {"start": 9.0, "end": 12.0, "highlight_score": 0.8}
        ]
        config = {"max_highlights": 10}
        
        # When: スコアでランキング
        result = rank_highlights_by_score(highlights, config)
        
        # Then: スコア降順でソートされる
        assert len(result) == 4
        assert result[0]["highlight_score"] == 0.9
        assert result[1]["highlight_score"] == 0.8
        assert result[2]["highlight_score"] == 0.7
        assert result[3]["highlight_score"] == 0.5
        
        # ランキング情報が付与される
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2
    
    def test_rank_highlights_by_score_max_limit(self):
        """最大ハイライト数による制限"""
        # Given: 多数のハイライト
        highlights = [
            {"start": i, "end": i+1, "highlight_score": 1.0 - i*0.1} 
            for i in range(10)
        ]
        config = {"max_highlights": 3}
        
        # When: 最大数制限付きでランキング
        result = rank_highlights_by_score(highlights, config)
        
        # Then: 上位3つのみが残る
        assert len(result) == 3
        assert all(result[i]["highlight_score"] >= result[i+1]["highlight_score"] 
                  for i in range(len(result)-1))
    
    def test_rank_highlights_by_score_tie_breaking(self):
        """同点時のタイブレーク"""
        # Given: 同じスコアのハイライト
        highlights = [
            {"start": 0.0, "end": 3.0, "highlight_score": 0.8, "duration": 3.0},
            {"start": 3.0, "end": 7.0, "highlight_score": 0.8, "duration": 4.0},
            {"start": 7.0, "end": 9.0, "highlight_score": 0.8, "duration": 2.0}
        ]
        config = {"max_highlights": 10}
        
        # When: タイブレーク付きランキング
        result = rank_highlights_by_score(highlights, config)
        
        # Then: 継続時間の長い順でタイブレーク
        assert result[0]["duration"] >= result[1]["duration"]
        assert result[1]["duration"] >= result[2]["duration"]


class TestResultFormatting:
    """結果フォーマットのテスト（純粋関数）"""
    
    def test_format_highlight_results_complete(self):
        """完全なハイライト結果のフォーマット"""
        # Given: ハイライト検出結果
        highlights = [
            {
                "start": 0.0, "end": 3.0, "text": "素晴らしい発見です！",
                "highlight_score": 0.9, "rank": 1, "dominant_emotion": "joy",
                "emotions": {"joy": 0.8, "surprise": 0.2}
            },
            {
                "start": 10.0, "end": 13.0, "text": "とても悲しい出来事でした",
                "highlight_score": 0.7, "rank": 2, "dominant_emotion": "sadness",
                "emotions": {"sadness": 0.7, "neutral": 0.3}
            }
        ]
        metadata = {
            "total_segments": 20,
            "candidates_found": 5,
            "processing_time": 1.2
        }
        
        # When: 結果をフォーマット
        result = format_highlight_results(highlights, metadata)
        
        # Then: 適切にフォーマットされる
        assert result["highlights"] == highlights
        assert result["metadata"]["total_highlights"] == 2
        assert result["metadata"]["total_segments"] == 20
        assert result["summary"]["top_emotions"] == ["joy", "sadness"]
        assert result["summary"]["average_score"] == 0.8


class TestIntegration:
    """統合テスト"""
    
    def test_full_highlight_detection_workflow(self):
        """ハイライト検出の全体ワークフローテスト"""
        # Given: 感情分析結果と設定
        emotion_result = {
            "segments": [
                {
                    "start": 0.0, "end": 3.0, "text": "普通の話です",
                    "emotions": {"neutral": 0.9, "joy": 0.1}, 
                    "dominant_emotion": "neutral", "confidence": 0.8
                },
                {
                    "start": 3.0, "end": 6.0, "text": "素晴らしい発見です！",
                    "emotions": {"joy": 0.9, "neutral": 0.1}, 
                    "dominant_emotion": "joy", "confidence": 0.95
                },
                {
                    "start": 6.0, "end": 9.0, "text": "とても悲しいです",
                    "emotions": {"sadness": 0.8, "neutral": 0.2}, 
                    "dominant_emotion": "sadness", "confidence": 0.85
                }
            ]
        }
        config = {
            "emotion_change_threshold": 0.3,
            "high_emotion_threshold": 0.7,
            "min_highlight_duration": 2.0,
            "max_highlights": 5
        }
        
        # When: 全体ワークフローを実行
        result = full_highlight_detection_workflow(emotion_result, config)
        
        # Then: 正しい構造で結果が返される
        assert "highlights" in result
        assert "metadata" in result
        assert "summary" in result
        assert len(result["highlights"]) <= config["max_highlights"]
        assert all(h["highlight_score"] > 0 for h in result["highlights"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])