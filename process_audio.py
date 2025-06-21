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
from typing import Dict, Any, Union, Optional, Tuple
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
from src.speech_detector import (
    detect_speech_boundary,
    SpeechDetectionConfig,
    save_detection_result
)

logger = logging.getLogger(__name__)


@dataclass
class AudioExtractionConfig:
    """音声抽出設定（イミュータブル）"""
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"
    codec: str = "pcm_s16le"
    quality_level: str = "standard"  # standard, high
    
    # 音声境界検出関連
    speech_detection_enabled: bool = True
    skip_seconds: int = 0  # 自動検出時は0、手動指定時は具体的な秒数
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AudioExtractionConfig':
        """辞書から設定を作成"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class AudioExtractionResult:
    """音声抽出結果（イミュータブル）"""
    input_video_path: str
    output_audio_path: str
    output_clean_audio_path: str = ""  # クリーン音声ファイルパス
    video_id: str = ""
    video_hash: str = ""
    config: AudioExtractionConfig = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    success: bool = False
    error_message: str = ""
    
    # 音声境界検出結果
    speech_detection_used: bool = False
    detected_speech_start: Optional[int] = None
    speech_detection_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        if self.config:
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


def create_clean_audio_with_speech_detection(
    input_audio_path: Path,
    output_dir: Path,
    video_id: str,
    speech_detection_config: SpeechDetectionConfig,
    audio_config: AudioExtractionConfig
) -> Tuple[Optional[Path], Optional[int], float]:
    """
    音声境界自動検出を使用してクリーン音声ファイルを作成する純粋関数
    
    Args:
        input_audio_path: 入力音声ファイルのパス
        output_dir: 出力ディレクトリ
        video_id: 動画ID
        speech_detection_config: 音声検出設定
        audio_config: 音声抽出設定
        
    Returns:
        Tuple[Optional[Path], Optional[int], float]: 
            (クリーン音声パス, 検出された開始時刻, 信頼度スコア)
    """
    try:
        # 手動スキップが指定されている場合
        if not audio_config.speech_detection_enabled or audio_config.skip_seconds > 0:
            skip_seconds = audio_config.skip_seconds or speech_detection_config.default_skip_seconds
            logger.info(f"手動スキップ設定を使用: {skip_seconds}秒")
            
            clean_audio_path = output_dir / f"{video_id}_audio_clean.wav"
            
            # FFmpegで指定秒数をスキップ
            subprocess.run([
                "ffmpeg", "-i", str(input_audio_path),
                "-ss", str(skip_seconds),
                "-y", str(clean_audio_path)
            ], check=True, capture_output=True)
            
            return clean_audio_path, skip_seconds, 1.0
        
        # 自動検出実行
        logger.info("音声境界自動検出を実行中...")
        detection_result = detect_speech_boundary(input_audio_path, speech_detection_config)
        
        if detection_result.success and detection_result.detected_speech_start is not None:
            skip_seconds = detection_result.detected_speech_start
            confidence = detection_result.confidence_score
            
            logger.info(
                f"音声境界検出成功: {skip_seconds}秒 (信頼度: {confidence:.3f})"
            )
            
            # 検出結果を保存
            save_detection_result(detection_result, output_dir)
            
            # クリーン音声作成
            clean_audio_path = output_dir / f"{video_id}_audio_clean.wav"
            
            subprocess.run([
                "ffmpeg", "-i", str(input_audio_path),
                "-ss", str(skip_seconds),
                "-y", str(clean_audio_path)
            ], check=True, capture_output=True)
            
            return clean_audio_path, skip_seconds, confidence
        
        else:
            # 自動検出失敗時のフォールバック
            skip_seconds = speech_detection_config.default_skip_seconds
            logger.warning(f"自動検出失敗、フォールバック使用: {skip_seconds}秒")
            
            clean_audio_path = output_dir / f"{video_id}_audio_clean.wav"
            
            subprocess.run([
                "ffmpeg", "-i", str(input_audio_path),
                "-ss", str(skip_seconds),
                "-y", str(clean_audio_path)
            ], check=True, capture_output=True)
            
            return clean_audio_path, skip_seconds, 0.0
            
    except Exception as e:
        logger.error(f"クリーン音声作成エラー: {e}")
        return None, None, 0.0


def audio_extraction_workflow(
    input_video_path: str,
    output_dir: str,
    config: AudioExtractionConfig,
    speech_detection_config: Optional[SpeechDetectionConfig] = None
) -> AudioExtractionResult:
    """
    音声抽出ワークフローを実行する純粋関数
    
    Args:
        input_video_path: 入力動画ファイルのパス
        output_dir: 出力ディレクトリ
        config: 音声抽出設定
        speech_detection_config: 音声境界検出設定（オプション）
        
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
        
        # 音声境界検出とクリーン音声作成
        clean_audio_path = None
        detected_speech_start = None
        speech_detection_confidence = 0.0
        speech_detection_used = False
        
        if speech_detection_config:
            logger.info("音声境界検出を実行...")
            clean_audio_path, detected_speech_start, speech_detection_confidence = \
                create_clean_audio_with_speech_detection(
                    audio_file_path, output_path, video_id, 
                    speech_detection_config, config
                )
            speech_detection_used = True
        
        # 結果作成
        processing_time = time.time() - start_time
        
        result = AudioExtractionResult(
            input_video_path=str(video_path),
            output_audio_path=str(audio_file_path),
            output_clean_audio_path=str(clean_audio_path) if clean_audio_path else "",
            video_id=video_id,
            video_hash=video_hash,
            config=config,
            metadata={
                'video_metadata': video_metadata,
                'extraction_metadata': extraction_metadata,
                'timestamp': time.time()
            },
            processing_time=processing_time,
            success=True,
            speech_detection_used=speech_detection_used,
            detected_speech_start=detected_speech_start,
            speech_detection_confidence=speech_detection_confidence
        )
        
        logger.info(f"音声抽出ワークフロー完了: {processing_time:.2f}秒")
        if speech_detection_used and detected_speech_start is not None:
            logger.info(f"音声開始位置: {detected_speech_start}秒 (信頼度: {speech_detection_confidence:.3f})")
        
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


def load_config(config_path: str) -> Tuple[AudioExtractionConfig, Optional[SpeechDetectionConfig]]:
    """
    設定ファイルを読み込む純粋関数
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        Tuple[AudioExtractionConfig, Optional[SpeechDetectionConfig]]: 
            音声抽出設定と音声境界検出設定
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        return AudioExtractionConfig(), None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() == '.yaml':
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # audio設定セクションを取得
        audio_config_dict = config_dict.get('audio', {})
        audio_config = AudioExtractionConfig.from_dict(audio_config_dict)
        
        # 音声境界検出設定を取得
        speech_detection_dict = config_dict.get('speech_detection', {})
        speech_detection_config = None
        
        if speech_detection_dict.get('enabled', False):
            speech_detection_config = SpeechDetectionConfig.from_dict(speech_detection_dict)
            # audio_configの設定を反映
            if not audio_config.speech_detection_enabled:
                speech_detection_config = None
        
        return audio_config, speech_detection_config
        
    except Exception as e:
        logger.warning(f"設定ファイル読み込みエラー: {e}")
        return AudioExtractionConfig(), None


@click.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='入力動画ファイルのパス')
@click.option('--output', '-o', 'output_dir', default='data/stage1_audio',
              help='出力ディレクトリ (デフォルト: data/stage1_audio)')
@click.option('--config', '-c', 'config_path', default='config/audio_extraction.yaml',
              help='設定ファイルのパス')
@click.option('--skip-seconds', type=int, help='手動スキップ秒数（自動検出無効化）')
@click.option('--no-speech-detection', is_flag=True, help='音声境界自動検出を無効化')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(input_path: str, output_dir: str, config_path: str, skip_seconds: Optional[int], 
         no_speech_detection: bool, verbose: bool):
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
        config, speech_detection_config = load_config(config_path)
        
        # CLI オプションで設定をオーバーライド
        if skip_seconds is not None:
            config.skip_seconds = skip_seconds
            config.speech_detection_enabled = False
            speech_detection_config = None
            
        if no_speech_detection:
            config.speech_detection_enabled = False
            speech_detection_config = None
        
        logger.info(f"音声抽出設定: {asdict(config)}")
        if speech_detection_config:
            logger.info("音声境界自動検出: 有効")
        else:
            logger.info("音声境界自動検出: 無効")
        
        # 音声抽出実行
        result = audio_extraction_workflow(input_path, output_dir, config, speech_detection_config)
        
        if result.success:
            # 結果保存
            save_audio_result(result, Path(output_dir))
            
            logger.info("=== 音声抽出プロセス完了 ===")
            logger.info(f"処理時間: {result.processing_time:.2f}秒")
            logger.info(f"出力ファイル: {result.output_audio_path}")
            
            if result.output_clean_audio_path:
                logger.info(f"クリーン音声ファイル: {result.output_clean_audio_path}")
                
            if result.speech_detection_used and result.detected_speech_start is not None:
                logger.info(f"検出された音声開始位置: {result.detected_speech_start}秒")
                logger.info(f"検出信頼度: {result.speech_detection_confidence:.3f}")
            
            click.echo(f"✅ 音声抽出成功: {result.output_audio_path}")
            if result.output_clean_audio_path:
                click.echo(f"✅ クリーン音声作成: {result.output_clean_audio_path}")
            
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