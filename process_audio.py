#!/usr/bin/env python3
"""
純粋関数ベース音声抽出プロセス

Input: 動画ファイル (MP4, AVI, MOV等)
Output: 
  - data/stage1_audio/{video_id}_audio.wav
  - data/stage1_audio/{video_id}_audio_meta.json

Usage:
  python process_audio.py --input data/input/video.mp4 --output data/stage1_audio/ --config config/audio_extraction.yaml

純粋関数設計:
- 同じ入力に対して同じ出力
- 副作用なし（ファイル作成以外）
- 設定外部化によるパラメーター調整
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import dataclass, asdict
import click
import yaml
import ffmpeg
import subprocess

# プロジェクトルートをPythonパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.audio_processor import (
    check_ffmpeg_available,
    validate_video_format,
    cleanup_temp_file
)
from src.utils.logging_config import setup_process_logging

logger = logging.getLogger(__name__)


@dataclass
class AudioExtractionConfig:
    """音声抽出設定（イミュータブル）"""
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    codec: str = "pcm_s16le"
    quality_level: str = "standard"  # standard, high
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AudioExtractionConfig':
        """辞書から設定を作成"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class AudioExtractionResult:
    """音声抽出結果（イミュータブル）"""
    input_video_path: str
    output_audio_path: str
    video_id: str
    video_hash: str
    config: AudioExtractionConfig
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['config'] = asdict(self.config)
        return result


def calculate_video_hash(video_path: Path) -> str:
    """
    動画ファイルのハッシュ値を計算する純粋関数
    
    Args:
        video_path: 動画ファイルのパス
        
    Returns:
        str: SHA256ハッシュ値（最初の8文字）
    """
    hash_sha256 = hashlib.sha256()
    try:
        with open(video_path, 'rb') as f:
            # ファイルサイズが大きい場合は先頭部分のみ読み取り
            chunk_size = 64 * 1024  # 64KB
            while chunk := f.read(chunk_size):
                hash_sha256.update(chunk)
                # 大きなファイルの場合は最初の1MBのみでハッシュ計算
                if f.tell() > 1024 * 1024:
                    break
    except Exception as e:
        logger.warning(f"ハッシュ計算に失敗、ファイル名を使用: {e}")
        # ハッシュ計算に失敗した場合はファイル名とサイズを使用
        hash_sha256.update(video_path.name.encode())
        hash_sha256.update(str(video_path.stat().st_size).encode())
    
    return hash_sha256.hexdigest()[:8]


def generate_video_id(video_path: Path) -> str:
    """
    動画IDを生成する純粋関数
    
    Args:
        video_path: 動画ファイルのパス
        
    Returns:
        str: 動画ID (ファイル名から拡張子を除いたもの)
    """
    return video_path.stem


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    """
    動画のメタデータを取得する純粋関数
    
    Args:
        video_path: 動画ファイルのパス
        
    Returns:
        Dict[str, Any]: メタデータ辞書
    """
    try:
        # ffprobeを使ってメタデータを取得
        probe = ffmpeg.probe(str(video_path))
        
        # 動画ストリーム情報を取得
        video_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
            None
        )
        
        # 音声ストリーム情報を取得
        audio_stream = next(
            (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
            None
        )
        
        metadata = {
            'file_size': video_path.stat().st_size,
            'duration': float(probe['format'].get('duration', 0)),
            'bitrate': int(probe['format'].get('bit_rate', 0)),
            'format_name': probe['format'].get('format_name', 'unknown'),
        }
        
        if video_stream:
            metadata.update({
                'video_codec': video_stream.get('codec_name', 'unknown'),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
            })
        
        if audio_stream:
            metadata.update({
                'audio_codec': audio_stream.get('codec_name', 'unknown'),
                'audio_sample_rate': int(audio_stream.get('sample_rate', 0)),
                'audio_channels': int(audio_stream.get('channels', 0)),
            })
        
        return metadata
        
    except Exception as e:
        logger.warning(f"メタデータ取得に失敗: {e}")
        return {
            'file_size': video_path.stat().st_size,
            'duration': 0.0,
            'error': str(e)
        }


def extract_audio_pure(
    input_video_path: Path,
    output_audio_path: Path,
    config: AudioExtractionConfig
) -> Dict[str, Any]:
    """
    音声抽出を行う純粋関数
    
    Args:
        input_video_path: 入力動画ファイルのパス
        output_audio_path: 出力音声ファイルのパス
        config: 音声抽出設定
        
    Returns:
        Dict[str, Any]: 抽出結果のメタデータ
        
    Raises:
        Exception: 音声抽出に失敗した場合
    """
    try:
        # FFmpegストリームを構築
        stream = ffmpeg.input(str(input_video_path))
        
        # 音声設定を適用
        audio_args = {
            'acodec': config.codec,
            'ac': config.channels,
            'ar': config.sample_rate,
        }
        
        # 品質レベルに応じた追加設定
        if config.quality_level == "high":
            audio_args['q:a'] = 0  # 最高品質
        
        stream = ffmpeg.output(stream, str(output_audio_path), **audio_args)
        stream = ffmpeg.overwrite_output(stream)
        
        # 実行時間計測
        start_time = time.time()
        ffmpeg.run(stream, quiet=True, capture_stdout=True, capture_stderr=True)
        processing_time = time.time() - start_time
        
        # 出力ファイルの検証
        if not output_audio_path.exists():
            raise Exception("音声ファイルが作成されませんでした")
        
        output_size = output_audio_path.stat().st_size
        if output_size == 0:
            raise Exception("空の音声ファイルが作成されました")
        
        logger.info(f"音声抽出完了: {input_video_path} -> {output_audio_path}")
        
        return {
            'output_file_size': output_size,
            'processing_time': processing_time,
            'extraction_success': True
        }
        
    except ffmpeg.Error as e:
        error_msg = f"FFmpegでの音声抽出に失敗: {e}"
        logger.error(error_msg)
        # 失敗時は不完全ファイルを削除
        if output_audio_path.exists():
            cleanup_temp_file(output_audio_path)
        raise Exception(error_msg) from e


def audio_extraction_workflow(
    input_video_path: str,
    output_dir: str,
    config: AudioExtractionConfig
) -> AudioExtractionResult:
    """
    音声抽出ワークフローを実行する純粋関数
    
    Args:
        input_video_path: 入力動画ファイルのパス
        output_dir: 出力ディレクトリ
        config: 音声抽出設定
        
    Returns:
        AudioExtractionResult: 抽出結果
    """
    start_time = time.time()
    
    try:
        # パス正規化
        video_path = Path(input_video_path).resolve()
        output_path = Path(output_dir).resolve()
        
        # 検証
        if not check_ffmpeg_available():
            raise ValueError("FFmpegがインストールされていません")
        
        if not validate_video_format(video_path):
            raise ValueError(f"サポートされていない動画形式: {video_path.suffix}")
        
        if not video_path.exists():
            raise ValueError(f"動画ファイルが見つかりません: {video_path}")
        
        # 出力ディレクトリ作成
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 動画ID・ハッシュ生成
        video_id = generate_video_id(video_path)
        video_hash = calculate_video_hash(video_path)
        
        # 出力ファイルパス生成
        audio_file_path = output_path / f"{video_id}_audio.wav"
        
        # 動画メタデータ取得
        video_metadata = get_video_metadata(video_path)
        
        # 音声抽出実行
        extraction_metadata = extract_audio_pure(video_path, audio_file_path, config)
        
        # 結果作成
        processing_time = time.time() - start_time
        
        result = AudioExtractionResult(
            input_video_path=str(video_path),
            output_audio_path=str(audio_file_path),
            video_id=video_id,
            video_hash=video_hash,
            config=config,
            metadata={
                'video_metadata': video_metadata,
                'extraction_metadata': extraction_metadata,
                'timestamp': time.time()
            },
            processing_time=processing_time,
            success=True
        )
        
        logger.info(f"音声抽出ワークフロー完了: {processing_time:.2f}秒")
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"音声抽出ワークフローエラー: {error_msg}")
        
        return AudioExtractionResult(
            input_video_path=str(input_video_path),
            output_audio_path="",
            video_id=generate_video_id(Path(input_video_path)),
            video_hash="",
            config=config,
            metadata={},
            processing_time=processing_time,
            success=False,
            error_message=error_msg
        )


def save_audio_result(result: AudioExtractionResult, output_dir: Path) -> None:
    """
    音声抽出結果をJSONファイルに保存する純粋関数
    
    Args:
        result: 音声抽出結果
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    meta_file_path = output_dir / f"{result.video_id}_audio_meta.json"
    
    with open(meta_file_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"音声抽出結果を保存: {meta_file_path}")


def load_config(config_path: str) -> AudioExtractionConfig:
    """
    設定ファイルを読み込む純粋関数
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        AudioExtractionConfig: 音声抽出設定
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        return AudioExtractionConfig()  # デフォルト設定
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() == '.yaml':
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # audio設定セクションを取得
        audio_config = config_dict.get('audio', {})
        return AudioExtractionConfig.from_dict(audio_config)
        
    except Exception as e:
        logger.warning(f"設定ファイル読み込みエラー: {e}")
        return AudioExtractionConfig()  # デフォルト設定


@click.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='入力動画ファイルのパス')
@click.option('--output', '-o', 'output_dir', default='data/stage1_audio',
              help='出力ディレクトリ (デフォルト: data/stage1_audio)')
@click.option('--config', '-c', 'config_path', default='config/audio_extraction.yaml',
              help='設定ファイルのパス')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(input_path: str, output_dir: str, config_path: str, verbose: bool):
    """
    純粋関数ベース音声抽出プロセス
    
    INPUT_PATH: 処理する動画ファイルのパス
    """
    # 動画IDを抽出してログ設定
    input_file = Path(input_path)
    video_id = input_file.stem
    
    # 統一ログ設定
    log_level = logging.DEBUG if verbose else logging.INFO
    global logger
    logger = setup_process_logging('audio', 'processing', log_level, video_id)
    
    logger.info("=== 音声抽出プロセス開始 ===")
    logger.info(f"入力: {input_path}")
    logger.info(f"出力: {output_dir}")
    logger.info(f"設定: {config_path}")
    
    try:
        # 設定読み込み
        config = load_config(config_path)
        logger.info(f"音声抽出設定: {asdict(config)}")
        
        # 音声抽出実行
        result = audio_extraction_workflow(input_path, output_dir, config)
        
        if result.success:
            # 結果保存
            save_audio_result(result, Path(output_dir))
            
            logger.info("=== 音声抽出プロセス完了 ===")
            logger.info(f"処理時間: {result.processing_time:.2f}秒")
            logger.info(f"出力ファイル: {result.output_audio_path}")
            
            click.echo(f"✅ 音声抽出成功: {result.output_audio_path}")
            
        else:
            logger.error(f"音声抽出失敗: {result.error_message}")
            click.echo(f"❌ 音声抽出失敗: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        click.echo(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()