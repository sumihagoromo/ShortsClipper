import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

# テスト対象のモジュールをインポート
from src.emotion_analyzer import (
    create_emotion_config,
    initialize_emotion_analyzer,
    analyze_text_emotion,
    normalize_emotion_scores,
    batch_analyze_emotions,
    merge_emotion_with_timestamps,
    calculate_emotion_statistics,
    full_emotion_analysis_workflow
)


class TestEmotionConfiguration:
    """感情分析設定機能のテスト（純粋関数）"""
    
    def test_create_emotion_config_default_values(self):
        """デフォルト設定で感情分析設定を作成"""
        # Given: デフォルトパラメータ
        base_config = {
            "provider": "ml-ask",
            "batch_size": 10
        }
        
        # When: 感情分析設定を作成
        result = create_emotion_config(base_config)
        
        # Then: 適切なデフォルト値が設定される
        assert result["provider"] == "ml-ask"
        assert result["batch_size"] == 10
        assert result["confidence_threshold"] == 0.5
        assert result["normalize_scores"] is True
        assert "emotion_categories" in result
        assert result["model_type"] == "oseti"
    
    def test_create_emotion_config_custom_values(self):
        """カスタム設定で感情分析設定を作成"""
        # Given: カスタムパラメータ
        custom_config = {
            "provider": "empath",
            "batch_size": 20,
            "confidence_threshold": 0.7,
            "emotion_categories": ["joy", "sadness", "anger"]
        }
        
        # When: カスタム設定を作成
        result = create_emotion_config(custom_config)
        
        # Then: カスタム値が反映される
        assert result["provider"] == "empath"
        assert result["batch_size"] == 20
        assert result["confidence_threshold"] == 0.7
        assert len(result["emotion_categories"]) == 3
    
    @pytest.mark.parametrize("provider,expected_model", [
        ("ml-ask", "oseti"),
        ("empath", "empath_api"),
        ("local", "transformers_model"),
    ])
    def test_create_emotion_config_provider_selection(self, provider, expected_model):
        """プロバイダーに基づくモデル選択の純粋関数テスト"""
        # Given: 異なるプロバイダー
        config = {"provider": provider}
        
        # When: 設定を作成
        result = create_emotion_config(config)
        
        # Then: 適切なモデルが選択される
        assert result["model_type"] == expected_model


class TestEmotionAnalyzerInitialization:
    """感情分析器初期化のテスト"""
    
    def test_initialize_emotion_analyzer_ml_ask(self):
        """ML-Ask感情分析器の初期化"""
        # Given: ML-Ask設定
        config = {
            "provider": "ml-ask",
            "model_type": "oseti"
        }
        
        with patch('oseti.Analyzer') as mock_analyzer:
            mock_instance = MagicMock()
            mock_analyzer.return_value = mock_instance
            
            # When: 感情分析器を初期化
            # result = initialize_emotion_analyzer(config)
            
            # Then: 正しい分析器が作成される
            # mock_analyzer.assert_called_once()
            # assert result == mock_instance
            pass  # 実装後に有効化
    
    def test_initialize_emotion_analyzer_error_handling(self):
        """感情分析器初期化エラーのハンドリング"""
        # Given: 無効な設定
        config = {"provider": "invalid-provider"}
        
        # When & Then: 適切な例外が発生
        # with pytest.raises(ValueError):
        #     initialize_emotion_analyzer(config)
        pass  # 実装後に有効化


class TestTextEmotionAnalysis:
    """テキスト感情分析のテスト（純粋関数）"""
    
    def test_analyze_text_emotion_single_text(self):
        """単一テキストの感情分析"""
        # Given: テキストとモック分析器
        text = "今日はとても嬉しい日です"
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {
            "joy": 0.85,
            "sadness": 0.02,
            "anger": 0.01,
            "fear": 0.01,
            "surprise": 0.11,
            "neutral": 0.00
        }
        
        # When: テキストを感情分析
        # result = analyze_text_emotion(mock_analyzer, text)
        
        # Then: 正しい感情スコアが返される
        # assert result["joy"] == 0.85
        # assert result["sadness"] == 0.02
        # assert len(result) == 6  # 6種類の感情
        pass  # 実装後に有効化
    
    def test_analyze_text_emotion_empty_text(self):
        """空文字列の感情分析"""
        # Given: 空文字列
        text = ""
        mock_analyzer = MagicMock()
        
        # When: 空文字列を分析
        # result = analyze_text_emotion(mock_analyzer, text)
        
        # Then: 中性的な結果が返される
        # assert result["neutral"] == 1.0
        # assert sum(result.values()) == 1.0
        pass  # 実装後に有効化
    
    def test_analyze_text_emotion_japanese_text(self):
        """日本語テキストの感情分析"""
        # Given: 日本語の感情表現
        japanese_texts = [
            ("素晴らしい！", "joy"),
            ("悲しいです", "sadness"),
            ("怒っています", "anger"),
            ("怖いです", "fear"),
            ("びっくりした", "surprise")
        ]
        
        mock_analyzer = MagicMock()
        
        for text, expected_emotion in japanese_texts:
            # モックの戻り値を設定
            emotion_scores = {emotion: 0.1 for emotion in ["joy", "sadness", "anger", "fear", "surprise", "neutral"]}
            emotion_scores[expected_emotion] = 0.8
            mock_analyzer.analyze.return_value = emotion_scores
            
            # When: 日本語テキストを分析
            # result = analyze_text_emotion(mock_analyzer, text)
            
            # Then: 期待される感情が最高スコアになる
            # dominant_emotion = max(result, key=result.get)
            # assert dominant_emotion == expected_emotion
            pass  # 実装後に有効化


class TestEmotionNormalization:
    """感情スコア正規化のテスト（純粋関数）"""
    
    def test_normalize_emotion_scores_standard(self):
        """標準的な感情スコアの正規化"""
        # Given: 生の感情スコア
        raw_scores = {
            "joy": 0.7,
            "sadness": 0.3,
            "anger": 0.1,
            "fear": 0.05,
            "surprise": 0.2,
            "neutral": 0.0
        }
        
        # When: スコアを正規化
        result = normalize_emotion_scores(raw_scores)
        
        # Then: 合計が1.0になる
        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert all(0 <= score <= 1 for score in result.values())
    
    def test_normalize_emotion_scores_zero_sum(self):
        """合計が0のスコアの正規化"""
        # Given: 全て0のスコア
        raw_scores = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "neutral": 0.0
        }
        
        # When: スコアを正規化
        result = normalize_emotion_scores(raw_scores)
        
        # Then: 均等に分散される
        expected_value = 1.0 / len(raw_scores)
        assert all(abs(score - expected_value) < 1e-6 for score in result.values())
    
    def test_normalize_emotion_scores_confidence_threshold(self):
        """信頼度閾値を考慮した正規化"""
        # Given: 低信頼度スコア
        raw_scores = {
            "joy": 0.3,
            "sadness": 0.2,
            "anger": 0.1,
            "fear": 0.1,
            "surprise": 0.1,
            "neutral": 0.2
        }
        confidence_threshold = 0.5
        
        # When: 閾値を考慮して正規化
        result = normalize_emotion_scores(raw_scores, confidence_threshold)
        
        # Then: 低信頼度の場合はneutralが優勢になる
        assert result["neutral"] > max(result[emotion] for emotion in result if emotion != "neutral")


class TestBatchProcessing:
    """バッチ処理のテスト"""
    
    def test_batch_analyze_emotions_standard(self):
        """標準的なバッチ感情分析"""
        # Given: 複数のテキストセグメント
        segments = [
            {"text": "嬉しいです", "start": 0.0, "end": 2.0},
            {"text": "悲しいです", "start": 2.0, "end": 4.0},
            {"text": "普通です", "start": 4.0, "end": 6.0}
        ]
        config = {"batch_size": 2}
        
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = [
            {"joy": 0.8, "sadness": 0.1, "anger": 0.1, "fear": 0.0, "surprise": 0.0, "neutral": 0.0},
            {"joy": 0.1, "sadness": 0.8, "anger": 0.1, "fear": 0.0, "surprise": 0.0, "neutral": 0.0},
            {"joy": 0.1, "sadness": 0.1, "anger": 0.1, "fear": 0.1, "surprise": 0.1, "neutral": 0.5}
        ]
        
        # When: バッチで感情分析
        # result = batch_analyze_emotions(mock_analyzer, segments, config)
        
        # Then: 各セグメントに感情スコアが付与される
        # assert len(result) == 3
        # assert result[0]["emotions"]["joy"] == 0.8
        # assert result[1]["emotions"]["sadness"] == 0.8
        # assert result[2]["emotions"]["neutral"] == 0.5
        pass  # 実装後に有効化
    
    def test_batch_analyze_emotions_progress_callback(self):
        """進捗コールバック付きバッチ処理"""
        # Given: 大量のセグメント
        segments = [{"text": f"テキスト{i}", "start": i, "end": i+1} for i in range(10)]
        config = {"batch_size": 3}
        progress_data = []
        
        def progress_callback(current: int, total: int, message: str):
            progress_data.append({"current": current, "total": total, "message": message})
        
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {"neutral": 1.0, "joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0}
        
        # When: 進捗コールバック付きで処理
        # result = batch_analyze_emotions(mock_analyzer, segments, config, progress_callback)
        
        # Then: 進捗が記録される
        # assert len(progress_data) > 0
        # assert progress_data[-1]["current"] == len(segments)
        pass  # 実装後に有効化


class TestTimestampMerging:
    """タイムスタンプ統合のテスト（純粋関数）"""
    
    def test_merge_emotion_with_timestamps_success(self):
        """感情結果とタイムスタンプの正常統合"""
        # Given: 転写結果と感情分析結果
        transcription_segments = [
            {"start": 0.0, "end": 3.0, "text": "こんにちは"},
            {"start": 3.0, "end": 6.0, "text": "元気です"}
        ]
        
        emotion_results = [
            {"emotions": {"joy": 0.7, "neutral": 0.3}, "dominant_emotion": "joy"},
            {"emotions": {"joy": 0.8, "neutral": 0.2}, "dominant_emotion": "joy"}
        ]
        
        # When: 感情とタイムスタンプを統合
        result = merge_emotion_with_timestamps(transcription_segments, emotion_results)
        
        # Then: 正しく統合される
        assert len(result) == 2
        assert result[0]["start"] == 0.0
        assert result[0]["emotions"]["joy"] == 0.7
        assert result[1]["dominant_emotion"] == "joy"
    
    def test_merge_emotion_with_timestamps_length_mismatch(self):
        """セグメント数の不一致ハンドリング"""
        # Given: 異なる長さのリスト
        transcription_segments = [{"start": 0.0, "end": 3.0, "text": "テスト"}]
        emotion_results = [
            {"emotions": {"joy": 0.7}},
            {"emotions": {"sadness": 0.8}}
        ]
        
        # When & Then: 適切にハンドリングされる
        result = merge_emotion_with_timestamps(transcription_segments, emotion_results)
        assert len(result) == 1  # 短い方に合わせる


class TestEmotionStatistics:
    """感情統計計算のテスト（純粋関数）"""
    
    def test_calculate_emotion_statistics_complete(self):
        """完全な感情統計の計算"""
        # Given: 感情分析結果
        emotion_data = [
            {"emotions": {"joy": 0.8, "sadness": 0.1, "neutral": 0.1}, "start": 0.0, "end": 3.0},
            {"emotions": {"joy": 0.2, "sadness": 0.7, "neutral": 0.1}, "start": 3.0, "end": 6.0},
            {"emotions": {"joy": 0.1, "sadness": 0.1, "neutral": 0.8}, "start": 6.0, "end": 9.0}
        ]
        
        # When: 統計を計算
        result = calculate_emotion_statistics(emotion_data)
        
        # Then: 適切な統計が計算される
        assert "average_emotions" in result
        assert "dominant_emotion_distribution" in result
        assert "emotion_changes" in result
        assert "total_duration" in result
        assert result["total_duration"] == 9.0


class TestIntegration:
    """統合テスト"""
    
    def test_full_emotion_analysis_workflow(self):
        """感情分析の全体ワークフローテスト"""
        # Given: 転写結果と設定
        transcription_result = {
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "とても嬉しいです"},
                {"start": 3.0, "end": 6.0, "text": "少し悲しいです"}
            ]
        }
        config = {
            "provider": "ml-ask",
            "batch_size": 10,
            "confidence_threshold": 0.5
        }
        
        with patch('src.emotion_analyzer.initialize_emotion_analyzer') as mock_init, \
             patch('src.emotion_analyzer.batch_analyze_emotions') as mock_batch, \
             patch('src.emotion_analyzer.calculate_emotion_statistics') as mock_stats:
            
            mock_analyzer = MagicMock()
            mock_init.return_value = mock_analyzer
            
            mock_emotion_results = [
                {"emotions": {"joy": 0.8, "neutral": 0.2}, "dominant_emotion": "joy"},
                {"emotions": {"sadness": 0.7, "neutral": 0.3}, "dominant_emotion": "sadness"}
            ]
            mock_batch.return_value = mock_emotion_results
            
            mock_statistics = {
                "average_emotions": {"joy": 0.4, "sadness": 0.35},
                "total_duration": 6.0
            }
            mock_stats.return_value = mock_statistics
            
            # When: 全体ワークフローを実行
            # result = full_emotion_analysis_workflow(transcription_result, config)
            
            # Then: 正しい順序で各関数が呼ばれる
            # mock_init.assert_called_once()
            # mock_batch.assert_called_once()
            # mock_stats.assert_called_once()
            # assert "segments" in result
            # assert "statistics" in result
            pass  # 実装後に有効化


if __name__ == "__main__":
    pytest.main([__file__, "-v"])