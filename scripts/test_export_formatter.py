#!/usr/bin/env python3
"""
動画編集ソフト対応エクスポート機能のテストスクリプト

Usage:
  python scripts/test_export_formatter.py --highlights data/stage4_highlights/sample_highlights.json
"""

import sys
from pathlib import Path
import logging
import click

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export_formatter import (
    export_highlights_all_formats,
    ExportConfig,
    load_highlights_data
)
from src.utils.logging_config import setup_process_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option('--highlights', '-h', required=True, type=click.Path(exists=True),
              help='ハイライト検出結果ファイルのパス')
@click.option('--output', '-o', default='data/stage5_output',
              help='出力ディレクトリ (デフォルト: data/stage5_output)')
@click.option('--min-level', type=click.Choice(['low', 'medium', 'high']), default='medium',
              help='最小ハイライトレベル')
@click.option('--max-segments', type=int, default=15,
              help='最大セグメント数')
@click.option('--frame-rate', type=float, default=30.0,
              help='フレームレート')
@click.option('--context-seconds', type=float, default=1.0,
              help='前後の文脈時間（秒）')
@click.option('--video-duration', type=float,
              help='動画の総時間（秒）- YouTubeチャプター用')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(highlights: str, output: str, min_level: str, max_segments: int, 
         frame_rate: float, context_seconds: float, video_duration: float, verbose: bool):
    """
    動画編集ソフト対応エクスポート機能のテストツール
    
    HIGHLIGHTS: ハイライト検出結果ファイルのパス
    """
    # ログ設定
    log_level = logging.DEBUG if verbose else logging.INFO
    global logger
    logger = setup_process_logging('export_test', 'processing', log_level)
    
    logger.info("=== 動画編集ソフト対応エクスポートテスト開始 ===")
    logger.info(f"ハイライトファイル: {highlights}")
    logger.info(f"出力ディレクトリ: {output}")
    
    highlights_path = Path(highlights)
    output_path = Path(output)
    
    try:
        # エクスポート設定作成
        config = ExportConfig(
            min_highlight_level=min_level,
            max_segments=max_segments,
            frame_rate=frame_rate,
            context_seconds=context_seconds
        )
        
        logger.info(f"エクスポート設定: {config}")
        
        # ハイライトデータ読み込み
        highlights_data = load_highlights_data(highlights_path)
        logger.info(f"読み込み済みハイライト数: {len(highlights_data)}個")
        
        if not highlights_data:
            click.echo("❌ ハイライトデータが見つかりません")
            return
        
        # ハイライト情報表示
        logger.info("\n--- ハイライト詳細 ---")
        for i, highlight in enumerate(highlights_data[:5], 1):  # 最初の5個
            logger.info(
                f"ハイライト {i}: {highlight.start_time:.1f}s-{highlight.end_time:.1f}s "
                f"({highlight.highlight_level}, {highlight.emotion_type}, "
                f"信頼度: {highlight.confidence:.2f})"
            )
            if highlight.text:
                logger.info(f"  テキスト: {highlight.text[:50]}...")
        
        # 全形式エクスポート実行
        output_files = export_highlights_all_formats(
            highlights_path, 
            output_path, 
            config, 
            video_duration
        )
        
        # 結果表示
        logger.info("=== エクスポート結果 ===")
        for format_name, file_path in output_files.items():
            logger.info(f"{format_name}: {file_path}")
        
        # ファイルサイズ確認
        for format_name, file_path in output_files.items():
            file_size = Path(file_path).stat().st_size
            logger.info(f"{format_name} ファイルサイズ: {file_size}バイト")
        
        click.echo(f"✅ エクスポート成功: {len(output_files)}個のファイル生成")
        
        # 生成されたファイルの場所を表示
        click.echo("\n📁 生成されたファイル:")
        for format_name, file_path in output_files.items():
            click.echo(f"  {format_name}: {file_path}")
        
        # 使用方法の説明
        click.echo("\n🎬 使用方法:")
        if 'premiere_pro' in output_files:
            click.echo("  • Premiere Pro: ファイル > 読み込み でCSVファイルをマーカーとして読み込み")
        if 'davinci_resolve' in output_files:
            click.echo("  • DaVinci Resolve: ファイル > 読み込み > タイムライン でEDLファイルを読み込み")
        if 'youtube_chapters' in output_files:
            click.echo("  • YouTube: チャプターファイルの内容を動画説明欄にコピー")
        if 'timeline_report' in output_files:
            click.echo("  • 解析レポート: MarkdownビューアーまたはGitHubで表示")
        
    except Exception as e:
        logger.error(f"エクスポートテスト実行エラー: {e}")
        click.echo(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()