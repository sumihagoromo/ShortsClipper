"""
音声処理モジュール

関数型プログラミングの原則に従って実装:
- 純粋関数（副作用なし）
- イミュータブルなデータ
- 関数の合成
"""

import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import ffmpeg


logger = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """
    FFmpegが利用可能かチェックする純粋関数
    
    Returns:
        bool: FFmpegが利用可能な場合True
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def validate_video_format(file_path: Union[str, Path]) -> bool:
    """
    動画ファイル形式を検証する純粋関数
    
    Args:
        file_path: ファイルパス（文字列またはPathオブジェクト）
        
    Returns:
        bool: サポートされている形式の場合True
    """
    if not file_path:
        return False
        
    path = Path(file_path)
    supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    
    return path.suffix.lower() in supported_formats


def create_temp_audio_file(video_path: Union[str, Path]) -> Path:
    """
    一時音声ファイルのパスを生成する純粋関数
    
    Args:
        video_path: 元動画ファイルのパス
        
    Returns:
        Path: 一時音声ファイルのパス
    """
    video_name = Path(video_path).stem
    temp_dir = Path(tempfile.gettempdir())
    
    # 一意のファイル名を生成
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.wav',
        prefix=f'shorts_clipper_{video_name}_',
        dir=temp_dir
    )
    
    # ファイルディスクリプタを閉じる（ファイルは残す）
    import os
    os.close(temp_fd)
    
    return Path(temp_path)


def cleanup_temp_file(file_path: Path) -> None:
    """
    一時ファイルを削除する関数
    
    Args:
        file_path: 削除するファイルのパス
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"一時ファイルを削除しました: {file_path}")
    except OSError as e:
        logger.warning(f"一時ファイルの削除に失敗しました: {file_path}, エラー: {e}")


def extract_audio(
    input_video: Union[str, Path], 
    audio_config: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Path:
    """
    動画から音声を抽出する関数
    
    Args:
        input_video: 入力動画ファイルのパス
        audio_config: 音声設定辞書
        output_path: 出力パス（Noneの場合は一時ファイル）
        
    Returns:
        Path: 抽出された音声ファイルのパス
        
    Raises:
        Exception: FFmpegでの処理が失敗した場合
    """
    input_path = Path(input_video)
    
    if output_path is None:
        output_path = create_temp_audio_file(input_path)
    
    try:
        # FFmpegストリームを構築
        stream = ffmpeg.input(str(input_path))
        
        # 音声設定を適用
        audio_args = {
            'acodec': 'pcm_s16le',  # WAV形式
            'ac': audio_config.get('channels', 1),  # チャンネル数
            'ar': audio_config.get('sample_rate', 16000),  # サンプリングレート
        }
        
        stream = ffmpeg.output(stream, str(output_path), **audio_args)
        
        # 既存ファイルを上書き
        stream = ffmpeg.overwrite_output(stream)
        
        # 実行
        ffmpeg.run(stream, quiet=True)
        
        logger.info(f"音声抽出完了: {input_path} -> {output_path}")
        return output_path
        
    except ffmpeg.Error as e:
        error_msg = f"FFmpegでの音声抽出に失敗しました: {e}"
        logger.error(error_msg)
        # 失敗時は一時ファイルをクリーンアップ
        if output_path and output_path.exists():
            cleanup_temp_file(output_path)
        raise Exception(error_msg) from e


def full_audio_extraction_workflow(
    video_file: Union[str, Path], 
    audio_config: Dict[str, Any]
) -> Path:
    """
    音声抽出の全体ワークフローを実行する関数
    
    Args:
        video_file: 動画ファイルのパス
        audio_config: 音声設定
        
    Returns:
        Path: 抽出された音声ファイルのパス
        
    Raises:
        ValueError: 検証エラー
        Exception: 処理エラー
    """
    # 1. FFmpeg依存関係チェック
    if not check_ffmpeg_available():
        raise ValueError("FFmpegがインストールされていません。FFmpegをインストールしてください。")
    
    # 2. 動画形式検証
    if not validate_video_format(video_file):
        raise ValueError(f"サポートされていない動画形式です: {video_file}")
    
    # 3. ファイル存在確認
    video_path = Path(video_file)
    if not video_path.exists():
        raise ValueError(f"動画ファイルが見つかりません: {video_file}")
    
    # 4. 音声抽出実行
    logger.info(f"音声抽出を開始します: {video_file}")
    audio_path = extract_audio(video_path, audio_config)
    
    return audio_path


# 関数の合成例
def compose_audio_pipeline(*functions):
    """
    関数を合成してパイプラインを作成する高階関数
    """
    def pipeline(initial_value):
        result = initial_value
        for func in functions:
            result = func(result)
        return result
    return pipeline