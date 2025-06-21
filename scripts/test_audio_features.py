#!/usr/bin/env python3
"""
音響特徴ベース音声境界検出のテストスクリプト

Usage:
  python scripts/test_audio_features.py --audio data/stage1_audio/sample_audio.wav
"""

import sys
from pathlib import Path
import logging
import click

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_features_detector import (
    detect_speech_boundary_with_audio_features,
    AudioFeaturesConfig,
    save_audio_features_result
)
from src.utils.logging_config import setup_process_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option('--audio', '-a', required=True, type=click.Path(exists=True),
              help='音声ファイルのパス')
@click.option('--output', '-o', default='data/test_output',
              help='出力ディレクトリ (デフォルト: data/test_output)')
@click.option('--segment-duration', type=int, default=10,
              help='分析セグメント長（秒）')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(audio: str, output: str, segment_duration: int, verbose: bool):
    """
    音響特徴ベース音声境界検出のテストツール
    
    AUDIO: テストする音声ファイルのパス
    """
    # ログ設定
    log_level = logging.DEBUG if verbose else logging.INFO
    global logger
    logger = setup_process_logging('audio_features_test', 'analysis', log_level)
    
    logger.info("=== 音響特徴ベース境界検出テスト開始 ===")
    logger.info(f"音声ファイル: {audio}")
    logger.info(f"出力ディレクトリ: {output}")
    
    audio_path = Path(audio)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 検出設定作成
        config = AudioFeaturesConfig(
            segment_duration=segment_duration
        )
        
        logger.info(f"検出設定: {config}")
        
        # 音響特徴検出実行
        result = detect_speech_boundary_with_audio_features(audio_path, config)
        
        # 結果保存
        save_audio_features_result(result, output_path)
        
        # 結果表示
        logger.info("=== 検出結果 ===")
        logger.info(f"成功: {result.success}")
        logger.info(f"検出された音声開始位置: {result.detected_speech_start}秒")
        logger.info(f"検出方法: {result.detection_method}")
        logger.info(f"信頼度: {result.confidence_score:.3f}")
        logger.info(f"処理時間: {result.processing_time:.2f}秒")
        logger.info(f"分析セグメント数: {len(result.segments_analyzed)}")
        logger.info(f"音楽区間数: {len(result.music_segments)}")
        logger.info(f"音声区間数: {len(result.speech_segments)}")
        
        if result.segments_analyzed:
            logger.info("\n--- セグメント分析詳細 ---")
            for i, segment in enumerate(result.segments_analyzed[:10]):  # 最初の10セグメント
                logger.info(
                    f"セグメント {i+1} ({segment.start_time:.1f}s-{segment.end_time:.1f}s): "
                    f"音楽={segment.is_music}, 音声={segment.is_speech}, "
                    f"信頼度={segment.confidence_score:.3f}"
                )
                logger.info(
                    f"  音量: {segment.avg_volume_db:.1f}dB, "
                    f"スペクトラル対比: {segment.spectral_contrast:.1f}, "
                    f"調和波エネルギー: {segment.harmonic_energy:.3f}, "
                    f"テンポ: {segment.tempo_estimate:.1f}BPM"
                )
        
        # 音楽区間表示
        if result.music_segments:
            logger.info("\n--- 検出された音楽区間 ---")
            for i, (start, end) in enumerate(result.music_segments):
                logger.info(f"音楽区間 {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}秒間)")
        
        # 音声区間表示
        if result.speech_segments:
            logger.info("\n--- 検出された音声区間 ---")
            for i, (start, end) in enumerate(result.speech_segments):
                logger.info(f"音声区間 {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}秒間)")
        
        # 結果ファイルパス表示
        result_file = output_path / f"{audio_path.stem}_audio_features_detection.json"
        
        if result.success:
            click.echo(f"✅ 検出成功: {result.detected_speech_start}秒")
            click.echo(f"🎯 信頼度: {result.confidence_score:.3f}")
            click.echo(f"🎵 音楽区間: {len(result.music_segments)}個")
            click.echo(f"🗣️  音声区間: {len(result.speech_segments)}個")
        else:
            click.echo(f"❌ 検出失敗: {result.error_message}")
        
        click.echo(f"⏱️ 処理時間: {result.processing_time:.2f}秒")
        click.echo(f"📄 詳細結果: {result_file}")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        click.echo(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()