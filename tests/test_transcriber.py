import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

# テスト対象のモジュールをインポート
from src.transcriber import (
    create_whisper_config,
    load_whisper_model,
    transcribe_audio_segments,
    extract_word_timestamps,
    format_transcription_result,
    full_transcription_workflow,
    update_progress,
    calculate_audio_chunks
)


class TestWhisperConfiguration:
    """Whisper設定機能のテスト（純粋関数）"""
    
    def test_create_whisper_config_default_values(self):
        """デフォルト設定でWhisper設定を作成"""
        # Given: デフォルトパラメータ
        base_config = {
            "model_size": "large-v3",
            "language": "ja",
            "temperature": 0.0
        }
        
        # When: Whisper設定を作成
        result = create_whisper_config(base_config)
        
        # Then: 適切なデフォルト値が設定される
        assert result["model_size"] == "large-v3"
        assert result["language"] == "ja"
        assert result["temperature"] == 0.0
        assert result["initial_prompt"] == "以下は日本語の音声です。"
        assert result["word_timestamps"] is True
        assert result["condition_on_previous_text"] is False
    
    def test_create_whisper_config_custom_values(self):
        """カスタム設定でWhisper設定を作成"""
        # Given: カスタムパラメータ
        custom_config = {
            "model_size": "medium",
            "language": "en",
            "temperature": 0.2,
            "initial_prompt": "Technical discussion about AI."
        }
        
        # When: カスタム設定を作成
        result = create_whisper_config(custom_config)
        
        # Then: カスタム値が反映される
        assert result["model_size"] == "medium"
        assert result["language"] == "en"
        assert result["initial_prompt"] == "Technical discussion about AI."
        assert result["temperature"] == 0.2
    
    @pytest.mark.parametrize("model_size,expected_device", [
        ("tiny", "cpu"),
        ("base", "cpu"), 
        ("small", "auto"),
        ("medium", "auto"),
        ("large-v3", "auto"),
    ])
    def test_create_whisper_config_device_selection(self, model_size, expected_device):
        """モデルサイズに基づくデバイス選択の純粋関数テスト"""
        # Given: 異なるモデルサイズ
        config = {"model_size": model_size}
        
        # When: 設定を作成
        result = create_whisper_config(config)
        
        # Then: 適切なデバイスが選択される
        assert result["device"] == expected_device


class TestWhisperModelLoading:
    """Whisperモデル読み込みのテスト"""
    
    def test_load_whisper_model_success(self):
        """Whisperモデルの正常読み込み"""
        # Given: 有効な設定
        config = {
            "model_size": "base",
            "device": "cpu",
            "compute_type": "float32"
        }
        
        with patch('faster_whisper.WhisperModel') as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            
            # When: モデルを読み込み
            result = load_whisper_model(config)
            
            # Then: 正しいパラメータでモデルが作成される
            mock_model.assert_called_once_with(
                "base",
                device="cpu",
                compute_type="float32"
            )
            assert result == mock_instance
    
    def test_load_whisper_model_error_handling(self):
        """モデル読み込みエラーのハンドリング"""
        # Given: 無効な設定
        config = {"model_size": "invalid-model"}
        
        with patch('faster_whisper.WhisperModel') as mock_model:
            mock_model.side_effect = Exception("Model not found")
            
            # When & Then: 適切な例外が発生
            with pytest.raises(Exception):
                load_whisper_model(config)


class TestAudioTranscription:
    """音声文字起こしのテスト"""
    
    def test_transcribe_audio_segments_success(self):
        """音声セグメントの正常な文字起こし"""
        # Given: モックモデルと音声ファイル
        mock_model = MagicMock()
        audio_path = Path("/tmp/test_audio.wav")
        config = {
            "language": "ja",
            "temperature": 0.0,
            "initial_prompt": "以下は日本語の音声です。",
            "word_timestamps": True
        }
        
        # モックの転写結果
        mock_segments = [
            MagicMock(
                start=0.0,
                end=3.5,
                text="こんにちは",
                words=[
                    MagicMock(start=0.0, end=1.5, word="こんにちは", probability=0.95)
                ]
            ),
            MagicMock(
                start=3.5,
                end=6.0,
                text="今日はいい天気ですね",
                words=[
                    MagicMock(start=3.5, end=4.0, word="今日", probability=0.92),
                    MagicMock(start=4.0, end=4.2, word="は", probability=0.98),
                    MagicMock(start=4.2, end=4.8, word="いい", probability=0.89),
                    MagicMock(start=4.8, end=5.5, word="天気", probability=0.94),
                    MagicMock(start=5.5, end=6.0, word="ですね", probability=0.91)
                ]
            )
        ]
        
        mock_model.transcribe.return_value = (mock_segments, {"language": "ja"})
        
        # When: 音声を文字起こし
        # result = transcribe_audio_segments(mock_model, audio_path, config)
        
        # Then: 正しい形式で結果が返される
        # expected_structure = {
        #     "segments": [
        #         {
        #             "start": 0.0,
        #             "end": 3.5,
        #             "text": "こんにちは",
        #             "words": [{"start": 0.0, "end": 1.5, "word": "こんにちは", "probability": 0.95}]
        #         }
        #     ],
        #     "language": "ja"
        # }
        # assert len(result["segments"]) == 2
        # assert result["language"] == "ja"
        pass  # 実装後に有効化
    
    def test_transcribe_audio_segments_empty_audio(self):
        """空の音声ファイルの処理"""
        # Given: 空の結果を返すモックモデル
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], {"language": "ja"})
        
        audio_path = Path("/tmp/empty_audio.wav")
        config = {"language": "ja"}
        
        # When: 空の音声を文字起こし
        # result = transcribe_audio_segments(mock_model, audio_path, config)
        
        # Then: 空の結果が返される
        # assert result["segments"] == []
        # assert result["language"] == "ja"
        pass  # 実装後に有効化


class TestWordTimestampExtraction:
    """単語レベルタイムスタンプ抽出のテスト（純粋関数）"""
    
    def test_extract_word_timestamps_success(self):
        """単語タイムスタンプの正常抽出"""
        # Given: セグメントデータ
        segments_data = [
            {
                "start": 0.0,
                "end": 3.5,
                "text": "こんにちは",
                "words": [
                    {"start": 0.0, "end": 1.5, "word": "こんにちは", "probability": 0.95}
                ]
            },
            {
                "start": 3.5,
                "end": 6.0,
                "text": "今日はいい天気ですね",
                "words": [
                    {"start": 3.5, "end": 4.0, "word": "今日", "probability": 0.92},
                    {"start": 4.0, "end": 4.2, "word": "は", "probability": 0.98}
                ]
            }
        ]
        
        # When: 単語タイムスタンプを抽出
        # result = extract_word_timestamps(segments_data)
        
        # Then: 正しい形式で単語が抽出される
        # expected = [
        #     {"start": 0.0, "end": 1.5, "word": "こんにちは", "probability": 0.95, "segment_id": 0},
        #     {"start": 3.5, "end": 4.0, "word": "今日", "probability": 0.92, "segment_id": 1},
        #     {"start": 4.0, "end": 4.2, "word": "は", "probability": 0.98, "segment_id": 1}
        # ]
        # assert result == expected
        pass  # 実装後に有効化
    
    def test_extract_word_timestamps_empty_segments(self):
        """空のセグメントデータの処理"""
        # Given: 空のセグメントデータ
        segments_data = []
        
        # When: 単語タイムスタンプを抽出
        result = extract_word_timestamps(segments_data)
        
        # Then: 空のリストが返される
        assert result == []
    
    def test_extract_word_timestamps_no_words(self):
        """単語データがないセグメントの処理"""
        # Given: 単語データがないセグメント
        segments_data = [
            {"start": 0.0, "end": 1.0, "text": "...", "words": []}
        ]
        
        # When: 単語タイムスタンプを抽出
        result = extract_word_timestamps(segments_data)
        
        # Then: 空のリストが返される
        assert result == []


class TestTranscriptionFormatting:
    """転写結果フォーマットのテスト（純粋関数）"""
    
    def test_format_transcription_result_complete(self):
        """完全な転写結果のフォーマット"""
        # Given: 生の転写データ
        raw_data = {
            "segments": [
                {"start": 0.0, "end": 3.5, "text": "こんにちは", "words": []},
                {"start": 3.5, "end": 6.0, "text": "今日は", "words": []}
            ],
            "language": "ja"
        }
        metadata = {
            "model_size": "large-v3",
            "audio_duration": 6.0,
            "processing_time": 2.1
        }
        
        # When: 結果をフォーマット
        result = format_transcription_result(raw_data, metadata)
        
        # Then: 適切にフォーマットされる
        assert result["metadata"]["model_size"] == "large-v3"
        assert result["metadata"]["language"] == "ja"
        assert result["metadata"]["segments_count"] == 2
        assert result["full_text"] == "こんにちは 今日は"
        assert "segments" in result
        assert "word_timestamps" in result


class TestProgressTracking:
    """進捗追跡のテスト"""
    
    def test_progress_callback_function(self):
        """進捗コールバック関数のテスト"""
        # Given: 進捗追跡用のコールバック
        progress_data = []
        
        def progress_callback(current: int, total: int, message: str):
            progress_data.append({"current": current, "total": total, "message": message})
        
        # When: 進捗を更新
        update_progress(progress_callback, 1, 5, "セグメント 1/5 処理中")
        update_progress(progress_callback, 3, 5, "セグメント 3/5 処理中")
        
        # Then: 進捗が記録される
        assert len(progress_data) == 2
        assert progress_data[0]["current"] == 1
        assert progress_data[1]["current"] == 3


class TestMemoryEfficiency:
    """メモリ効率的な処理のテスト"""
    
    def test_chunk_audio_processing(self):
        """音声チャンク処理のテスト"""
        # Given: 大きな音声ファイルの情報
        audio_info = {
            "duration": 3600.0,  # 1時間
            "sample_rate": 16000,
            "channels": 1
        }
        chunk_size = 30.0  # 30秒チャンク
        
        # When: チャンク情報を計算
        chunks = calculate_audio_chunks(audio_info, chunk_size)
        
        # Then: 適切なチャンク分割される
        expected_chunks = 120  # 3600 / 30
        assert len(chunks) == expected_chunks
        assert chunks[0]["start"] == 0.0
        assert chunks[0]["end"] == 30.0
        assert chunks[-1]["end"] == 3600.0


class TestIntegration:
    """統合テスト"""
    
    def test_full_transcription_workflow(self):
        """文字起こしの全体ワークフローテスト"""
        # Given: 音声ファイルと設定
        audio_path = Path("test_audio.wav")
        config = {
            "model_size": "base",
            "language": "ja",
            "temperature": 0.0
        }
        
        with patch('src.transcriber.load_whisper_model') as mock_load, \
             patch('src.transcriber.transcribe_audio_segments') as mock_transcribe, \
             patch('src.transcriber.format_transcription_result') as mock_format:
            
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            mock_raw_result = {
                "segments": [{"start": 0.0, "end": 1.0, "text": "テスト"}],
                "language": "ja"
            }
            mock_transcribe.return_value = mock_raw_result
            
            mock_formatted = {
                "metadata": {"language": "ja"},
                "segments": mock_raw_result["segments"],
                "full_text": "テスト"
            }
            mock_format.return_value = mock_formatted
            
            # When: 全体ワークフローを実行
            result = full_transcription_workflow(audio_path, config)
            
            # Then: 正しい順序で各関数が呼ばれる
            mock_load.assert_called_once()
            mock_transcribe.assert_called_once()
            mock_format.assert_called_once()
            assert result == mock_formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])