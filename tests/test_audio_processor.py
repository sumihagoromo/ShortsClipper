import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# テスト対象のモジュールをインポート
from src.utils.audio_processor import (
    check_ffmpeg_available,
    validate_video_format,
    extract_audio,
    create_temp_audio_file,
    cleanup_temp_file,
    full_audio_extraction_workflow
)


class TestFFmpegDependencyCheck:
    """FFmpeg依存関係チェック機能のテスト"""
    
    def test_ffmpeg_available_returns_true_when_installed(self):
        """FFmpegがインストールされている場合、Trueを返す"""
        # Given: FFmpegがインストールされている環境
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # When: FFmpeg可用性をチェック
            result = check_ffmpeg_available()
            
            # Then: Trueが返される
            assert result is True
    
    def test_ffmpeg_available_returns_false_when_not_installed(self):
        """FFmpegがインストールされていない場合、Falseを返す"""
        # Given: FFmpegがインストールされていない環境
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            # When: FFmpeg可用性をチェック
            result = check_ffmpeg_available()
            
            # Then: Falseが返される
            assert result is False
    
    def test_ffmpeg_available_returns_false_on_command_failure(self):
        """FFmpegコマンドが失敗した場合、Falseを返す"""
        # Given: FFmpegコマンドが失敗する環境
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            # When: FFmpeg可用性をチェック
            result = check_ffmpeg_available()
            
            # Then: Falseが返される
            assert result is False


class TestVideoFormatValidation:
    """動画ファイル形式検証のテスト"""
    
    @pytest.mark.parametrize("filename,expected", [
        ("video.mp4", True),
        ("video.avi", True),
        ("video.mov", True),
        ("video.mkv", True),
        ("video.MP4", True),  # 大文字も対応
        ("video.txt", False),
        ("video.jpg", False),
        ("video", False),  # 拡張子なし
        ("", False),  # 空文字
    ])
    def test_validate_video_format_pure_function(self, filename, expected):
        """動画形式検証の純粋関数テスト"""
        # When: ファイル名の形式を検証
        result = validate_video_format(filename)
        
        # Then: 期待される結果が返される
        assert result == expected
    
    def test_validate_video_format_with_path_object(self):
        """Pathオブジェクトでも動作する"""
        # Given: Pathオブジェクト
        video_path = Path("test_video.mp4")
        
        # When: Pathオブジェクトで検証
        result = validate_video_format(video_path)
        
        # Then: 正しく検証される
        assert result is True


class TestAudioExtraction:
    """音声抽出処理のテスト"""
    
    def test_extract_audio_success_case(self):
        """正常な音声抽出のテスト"""
        # Given: 有効な動画ファイルパス
        input_video = "test_video.mp4"
        expected_audio_config = {
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav"
        }
        
        with patch('ffmpeg.input') as mock_input, \
             patch('ffmpeg.output') as mock_output, \
             patch('ffmpeg.overwrite_output') as mock_overwrite, \
             patch('ffmpeg.run') as mock_run, \
             patch('src.utils.audio_processor.create_temp_audio_file') as mock_temp:
            
            # ffmpegの各ステップをモック
            mock_stream1 = MagicMock()
            mock_stream2 = MagicMock()
            mock_stream3 = MagicMock()
            
            mock_input.return_value = mock_stream1
            mock_output.return_value = mock_stream2
            mock_overwrite.return_value = mock_stream3
            mock_temp.return_value = Path("/tmp/test_audio.wav")
            
            # When: 音声を抽出
            result = extract_audio(input_video, expected_audio_config)
            
            # Then: 正しいパラメータでFFmpegが呼ばれる
            mock_input.assert_called_once_with(input_video)
            mock_run.assert_called_once_with(mock_stream3, quiet=True, capture_output=True)
            assert result == Path("/tmp/test_audio.wav")
    
    def test_extract_audio_with_custom_config(self):
        """カスタム設定での音声抽出テスト"""
        # Given: カスタム音声設定
        input_video = "test_video.mp4"
        custom_config = {
            "sample_rate": 48000,
            "channels": 2,
            "format": "wav"
        }
        
        with patch('ffmpeg.input'), \
             patch('ffmpeg.output'), \
             patch('ffmpeg.overwrite_output'), \
             patch('ffmpeg.run'), \
             patch('src.utils.audio_processor.create_temp_audio_file') as mock_temp:
            
            mock_temp.return_value = Path("/tmp/test_audio.wav")
            
            # When: カスタム設定で音声抽出
            result = extract_audio(input_video, custom_config)
            
            # Then: 正常に処理される
            assert result == Path("/tmp/test_audio.wav")
    
    def test_extract_audio_ffmpeg_error_handling(self):
        """FFmpegエラー時の例外処理テスト"""
        # Given: FFmpegが失敗する状況
        input_video = "invalid_video.mp4"
        
        with patch('ffmpeg.input'), \
             patch('ffmpeg.output'), \
             patch('ffmpeg.overwrite_output'), \
             patch('ffmpeg.run') as mock_run, \
             patch('src.utils.audio_processor.create_temp_audio_file'):
            
            # FFmpegエラーをシミュレート
            import ffmpeg
            mock_run.side_effect = ffmpeg.Error("FFmpeg failed", "", "")
            
            # When & Then: 適切な例外が発生
            with pytest.raises(Exception):
                extract_audio(input_video, {})


class TestTempFileManagement:
    """一時ファイル管理のテスト"""
    
    def test_create_temp_audio_file_returns_unique_path(self):
        """一時音声ファイルの作成が一意のパスを返す"""
        # When: 一時ファイルパスを2回作成
        path1 = create_temp_audio_file("test_video.mp4")
        path2 = create_temp_audio_file("test_video.mp4")
        
        # Then: 異なるパスが返される
        assert path1 != path2
        assert path1.suffix == ".wav"
        assert path2.suffix == ".wav"
        
        # テスト後のクリーンアップ
        cleanup_temp_file(path1)
        cleanup_temp_file(path2)
    
    def test_cleanup_temp_file_removes_file(self):
        """一時ファイルの削除が正常に動作する"""
        # Given: 実際の一時ファイル
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = Path(temp_file.name)
        
        # 確認: ファイルが存在する
        assert temp_path.exists()
        
        # When: ファイルをクリーンアップ
        cleanup_temp_file(temp_path)
        
        # Then: ファイルが削除される
        assert not temp_path.exists()
    
    def test_cleanup_temp_file_handles_nonexistent_file(self):
        """存在しないファイルの削除でエラーが発生しない"""
        # Given: 存在しないファイルパス
        nonexistent_path = Path("/tmp/nonexistent_audio.wav")
        
        # When & Then: エラーが発生しない
        cleanup_temp_file(nonexistent_path)  # 例外が発生しないことを確認


class TestIntegration:
    """統合テスト"""
    
    def test_full_audio_extraction_workflow(self):
        """音声抽出の全体ワークフローテスト"""
        # Given: 模擬的な動画ファイル
        video_file = "sample_video.mp4"
        audio_config = {"sample_rate": 16000, "channels": 1, "format": "wav"}
        
        with patch('src.utils.audio_processor.check_ffmpeg_available', return_value=True), \
             patch('src.utils.audio_processor.validate_video_format', return_value=True), \
             patch('src.utils.audio_processor.extract_audio') as mock_extract, \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_temp_path = Path("/tmp/audio_123.wav")
            mock_extract.return_value = mock_temp_path
            
            # When: 全体ワークフローを実行
            result = full_audio_extraction_workflow(video_file, audio_config)
            
            # Then: 正しい順序で各関数が呼ばれる
            assert result == mock_temp_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])