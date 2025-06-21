#!/usr/bin/env python3
"""
音声境界自動検出システムのテストスクリプト

Usage:
  python scripts/test_speech_detection.py --audio data/stage1_audio/sample_audio.wav
  python scripts/test_speech_detection.py --audio data/stage1_audio/sample_audio.wav --config custom_config.yaml
"""

import sys
from pathlib import Path
import logging
import time
import click

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.speech_detector import (
    detect_speech_boundary,
    SpeechDetectionConfig,
    save_detection_result,
    quick_speech_detection
)
from src.utils.logging_config import setup_process_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option('--audio', '-a', required=True, type=click.Path(exists=True),
              help='音声ファイルのパス')
@click.option('--output', '-o', default='data/test_output',
              help='出力ディレクトリ (デフォルト: data/test_output)')
@click.option('--quick', '-q', is_flag=True,
              help='クイック検出モード（簡易テスト用）')
@click.option('--max-check-minutes', type=int, default=10,
              help='最大チェック時間（分）')
@click.option('--skip-default', type=int, default=180,
              help='デフォルトスキップ時間（秒）')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(audio: str, output: str, quick: bool, max_check_minutes: int, 
         skip_default: int, verbose: bool):
    """
    音声境界自動検出システムのテストツール
    
    AUDIO: テストする音声ファイルのパス
    """
    # ログ設定
    log_level = logging.DEBUG if verbose else logging.INFO
    global logger
    logger = setup_process_logging('speech_detection_test', 'analysis', log_level)
    
    logger.info("=== 音声境界自動検出テスト開始 ===")
    logger.info(f"音声ファイル: {audio}")
    logger.info(f"出力ディレクトリ: {output}")
    
    audio_path = Path(audio)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if quick:
            # クイック検出モード
            logger.info("クイック検出モードで実行中...")
            start_time = time.time()
            
            detected_start = quick_speech_detection(
                audio_path, 
                skip_seconds=skip_default,
                max_check_minutes=max_check_minutes
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"検出結果: {detected_start}秒")
            logger.info(f"処理時間: {processing_time:.2f}秒")
            
            click.echo(f"✅ クイック検出完了: {detected_start}秒")
            click.echo(f"⏱️ 処理時間: {processing_time:.2f}秒")
            
        else:
            # 詳細検出モード
            logger.info("詳細検出モードで実行中...")
            
            # 検出設定作成
            config = SpeechDetectionConfig(
                max_check_duration=max_check_minutes * 60,
                default_skip_seconds=skip_default
            )
            
            logger.info(f"検出設定: {config}")
            
            # 検出実行
            result = detect_speech_boundary(audio_path, config)
            
            # 結果保存
            save_detection_result(result, output_path)
            
            # 結果表示
            logger.info("=== 検出結果 ===")
            logger.info(f"成功: {result.success}")
            logger.info(f"検出された音声開始位置: {result.detected_speech_start}秒")
            logger.info(f"検出方法: {result.detection_method}")
            logger.info(f"信頼度: {result.confidence_score:.3f}")
            logger.info(f"フォールバック使用: {result.fallback_used}")
            logger.info(f"処理時間: {result.processing_time:.2f}秒")
            logger.info(f"分析セグメント数: {len(result.segments_analyzed)}")
            
            if result.segments_analyzed:
                logger.info("\n--- セグメント分析詳細 ---")
                for i, segment in enumerate(result.segments_analyzed):
                    logger.info(
                        f"セグメント {i+1} ({segment.start_time}s-{segment.end_time}s): "
                        f"セグメント数={segment.segment_count}, "
                        f"有意義セグメント={segment.meaningful_segments}, "
                        f"言語確率={segment.language_probability:.3f}, "
                        f"信頼度={segment.confidence_score:.3f}"
                    )
                    if segment.sample_text:
                        logger.info(f"  サンプルテキスト: {segment.sample_text[:2]}")
            
            # 結果ファイルパス表示
            result_file = output_path / f"{audio_path.stem}_speech_detection.json"
            
            if result.success:
                click.echo(f"✅ 検出成功: {result.detected_speech_start}秒")
                click.echo(f"🎯 信頼度: {result.confidence_score:.3f}")
                if result.fallback_used:
                    click.echo("⚠️ フォールバック使用（自動検出失敗）")
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