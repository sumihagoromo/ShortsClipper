#!/usr/bin/env python3
"""
ShortsClipper - メインコントローラー
各処理を個別実行またはパイプライン実行できるコマンドラインインターフェース
"""

import sys
import logging
from pathlib import Path
import click

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_config import setup_process_logging


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
@click.pass_context
def cli(ctx, verbose):
    """ShortsClipper - YouTube動画ハイライト検出ツール"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/stage1_audio', help='出力ディレクトリ')
@click.option('--config', '-c', default='config/audio_extraction.yaml', help='設定ファイル')
@click.option('--skip-seconds', type=int, help='手動スキップ秒数（自動検出無効化）')
@click.option('--no-speech-detection', is_flag=True, help='音声境界自動検出を無効化')
@click.pass_context
def audio(ctx, input_path, output, config, skip_seconds, no_speech_detection):
    """動画から音声を抽出（音声境界自動検出対応）"""
    from process_audio import main as audio_main
    
    logger = setup_process_logging('main_audio', 'processing', 
                                 logging.DEBUG if ctx.obj.get('verbose', False) else logging.INFO)
    logger.info(f"音声抽出: {input_path} -> {output}")
    
    # process_audio.pyのmain関数を呼び出し
    audio_main.callback(input_path, output, config, skip_seconds, 
                       no_speech_detection, ctx.obj.get('verbose', False))


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/stage2_transcript', help='出力ディレクトリ')
@click.option('--config', '-c', default='config/transcription_base.yaml', help='設定ファイル')
@click.option('--model', type=click.Choice(['base', 'small', 'medium', 'large-v3']), help='Whisperモデル')
@click.pass_context
def transcript(ctx, input_path, output, config, model):
    """音声を文字起こし"""
    from process_transcript import main as transcript_main
    
    logger = setup_process_logging('main_transcript', 'processing', ctx.obj.get('verbose', False))
    logger.info(f"文字起こし: {input_path} -> {output}")
    
    # モデル指定時は設定を動的変更
    if model:
        config = f"config/transcription_{model}.yaml"
        logger.info(f"モデル指定: {model}")
    
    transcript_main(input_path, output, config, ctx.obj.get('verbose', False))


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/stage2_transcript', help='出力ディレクトリ')
@click.option('--profile', '-p', type=click.Choice(['direct', 'batch_standard', 'batch_fast', 'high_quality']), 
              help='処理プロファイル指定')
@click.option('--quality', is_flag=True, help='品質優先モード')
@click.option('--force-split', is_flag=True, help='強制分割処理')
@click.option('--overlap-seconds', default=5.0, help='オーバーラップ時間（秒）')
@click.option('--no-overlap', is_flag=True, help='オーバーラップ処理を無効化')
@click.pass_context
def adaptive(ctx, input_path, output, profile, quality, force_split, overlap_seconds, no_overlap):
    """適応的音声転写（長時間音声対応）"""
    from process_transcript_adaptive import AdaptiveTranscriber
    
    logger = setup_process_logging('main_adaptive', 'transcription', 
                                 logging.DEBUG if ctx.obj.get('verbose', False) else logging.INFO)
    logger.info(f"適応的転写: {input_path} -> {output}")
    
    try:
        transcriber = AdaptiveTranscriber(logger)
        result = transcriber.process_audio_adaptive(
            input_path, output, profile, quality, force_split,
            overlap_seconds, not no_overlap
        )
        
        if result['success']:
            logger.info(f"✅ 適応的転写完了: {len(result['segments'])}セグメント")
        else:
            logger.error(f"❌ 適応的転写失敗: {result['error_message']}")
            
    except Exception as e:
        logger.error(f"適応的転写エラー: {e}")
        raise


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/stage3_emotions', help='出力ディレクトリ')
@click.option('--config', '-c', default='config/emotion_analysis.yaml', help='設定ファイル')
@click.pass_context
def emotions(ctx, input_path, output, config):
    """文字起こしから感情分析"""
    from process_emotions import main as emotions_main
    
    logger = setup_process_logging('main_emotions', 'processing', ctx.obj.get('verbose', False))
    logger.info(f"感情分析: {input_path} -> {output}")
    
    emotions_main(input_path, output, config, ctx.obj.get('verbose', False))


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/stage4_highlights', help='出力ディレクトリ')
@click.option('--config', '-c', default='config/highlight_detection.yaml', help='設定ファイル')
@click.option('--preset', type=click.Choice(['conservative', 'standard', 'aggressive']), help='プリセット設定')
@click.pass_context
def highlights(ctx, input_path, output, config, preset):
    """感情分析からハイライト検出"""
    from process_highlights import main as highlights_main
    
    logger = setup_process_logging('main_highlights', 'processing', ctx.obj.get('verbose', False))
    logger.info(f"ハイライト検出: {input_path} -> {output}")
    
    # プリセット指定時は設定を動的変更
    if preset:
        config = f"config/highlight_detection_{preset}.yaml"
        logger.info(f"プリセット指定: {preset}")
    
    highlights_main(input_path, output, config, ctx.obj.get('verbose', False))


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--model', default='base', type=click.Choice(['base', 'small', 'medium', 'large-v3']), help='Whisperモデル')
@click.option('--preset', default='standard', type=click.Choice(['conservative', 'standard', 'aggressive']), help='ハイライト検出プリセット')
@click.option('--skip-audio', is_flag=True, help='音声抽出をスキップ（音声ファイル指定時）')
@click.option('--skip-clean', is_flag=True, help='音声クリーニングをスキップ')
@click.pass_context
def pipeline(ctx, input_path, model, preset, skip_audio, skip_clean):
    """完全パイプライン実行（動画 -> ハイライト）"""
    import subprocess
    from pathlib import Path
    
    logger = setup_process_logging('main_pipeline', 'processing', ctx.obj.get('verbose', False))
    logger.info(f"パイプライン開始: {input_path}")
    
    input_file = Path(input_path)
    video_id = input_file.stem
    verbose_flag = ['-v'] if ctx.obj.get('verbose', False) else []
    
    try:
        current_file = input_path
        
        # Stage 1: 音声抽出
        if not skip_audio:
            logger.info("Stage 1: 音声抽出")
            subprocess.run([
                'python', 'main.py', 'audio', current_file,
                '-o', 'data/stage1_audio',
                '-c', 'config/audio_extraction.yaml'
            ] + verbose_flag, check=True)
            current_file = f"data/stage1_audio/{video_id}_audio.wav"
        
        # 音声クリーニング
        if not skip_clean:
            logger.info("音声クリーニング")
            subprocess.run([
                'python', 'scripts/analysis/create_clean_audio.py'
            ], check=True)
            current_file = f"data/stage1_audio/{video_id}_audio_clean.wav"
        
        # Stage 2: 文字起こし
        logger.info(f"Stage 2: 文字起こし ({model})")
        subprocess.run([
            'python', 'main.py', 'transcript', current_file,
            '-o', 'data/stage2_transcript',
            '--model', model
        ] + verbose_flag, check=True)
        current_file = f"data/stage2_transcript/{Path(current_file).stem}_cleaned.json"
        
        # Stage 3: 感情分析
        logger.info("Stage 3: 感情分析")
        subprocess.run([
            'python', 'main.py', 'emotions', current_file,
            '-o', 'data/stage3_emotions'
        ] + verbose_flag, check=True)
        current_file = f"data/stage3_emotions/{Path(current_file).stem.replace('_cleaned', '')}_text_emotions.json"
        
        # Stage 4: ハイライト検出
        logger.info(f"Stage 4: ハイライト検出 ({preset})")
        subprocess.run([
            'python', 'main.py', 'highlights', current_file,
            '-o', 'data/stage4_highlights',
            '--preset', preset
        ] + verbose_flag, check=True)
        
        logger.info("✅ パイプライン完了")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ パイプライン失敗: {e}")
        sys.exit(1)


@cli.command()
@click.option('--configs', '-c', default='config/', help='設定ディレクトリ')
@click.option('--input', '-i', required=True, help='入力ファイル')
@click.option('--parallel', '-p', default=3, help='並列実行数')
@click.pass_context
def tune(ctx, configs, input, parallel):
    """パラメーター調整（複数設定で並列実行）"""
    from scripts.tuning.batch_tune import main as tune_main
    
    logger = setup_process_logging('main_tune', 'processing', ctx.obj.get('verbose', False))
    logger.info(f"パラメーター調整: {input}")
    
    tune_main(input, configs, parallel, ctx.obj.get('verbose', False))


@cli.command()
@click.option('--category', type=click.Choice(['transcription', 'processing', 'monitoring']), 
              default='processing', help='ログカテゴリ')
@click.option('--days', default=7, help='保持日数')
@click.option('--dry-run', is_flag=True, help='実際には削除しない')
def cleanup(category, days, dry_run):
    """古いログファイルをクリーンアップ"""
    from src.utils.logging_config import cleanup_old_logs
    
    removed_files = cleanup_old_logs(days, dry_run)
    if dry_run:
        click.echo(f"削除対象: {len(removed_files)}ファイル")
        for file in removed_files:
            click.echo(f"  {file}")
    else:
        click.echo(f"削除完了: {len(removed_files)}ファイル")


@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/test_output', help='出力ディレクトリ')
@click.option('--quick', '-q', is_flag=True, help='クイック検出モード')
@click.option('--max-check-minutes', default=10, help='最大チェック時間（分）')
@click.option('--default-skip', default=180, help='デフォルトスキップ時間（秒）')
@click.pass_context
def detect_speech(ctx, audio_path, output, quick, max_check_minutes, default_skip):
    """音声境界自動検出（テスト用）"""
    from scripts.test_speech_detection import main as test_main
    
    logger = setup_process_logging('main_detect_speech', 'analysis', 
                                 logging.DEBUG if ctx.obj.get('verbose', False) else logging.INFO)
    logger.info(f"音声境界検出テスト: {audio_path}")
    
    # テストスクリプトを呼び出し
    test_main.callback(audio_path, output, quick, max_check_minutes, 
                      default_skip, ctx.obj.get('verbose', False))


@cli.command()
@click.argument('highlights_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/stage5_output', help='出力ディレクトリ')
@click.option('--format', '-f', 
              type=click.Choice(['all', 'premiere', 'davinci', 'youtube', 'timeline']),
              default='all', help='出力形式')
@click.option('--min-level', type=click.Choice(['low', 'medium', 'high']), 
              default='medium', help='最小ハイライトレベル')
@click.option('--max-segments', type=int, default=15, help='最大セグメント数')
@click.option('--video-duration', type=float, help='動画の総時間（秒）')
@click.pass_context
def export(ctx, highlights_path, output, format, min_level, max_segments, video_duration):
    """ハイライトを動画編集ソフト対応形式でエクスポート"""
    from src.export_formatter import export_highlights_all_formats, ExportConfig
    
    logger = setup_process_logging('main_export', 'processing', 
                                 logging.DEBUG if ctx.obj.get('verbose', False) else logging.INFO)
    logger.info(f"ハイライトエクスポート: {highlights_path} -> {output}")
    
    try:
        # エクスポート設定
        config = ExportConfig(
            min_highlight_level=min_level,
            max_segments=max_segments
        )
        
        # エクスポート実行
        output_files = export_highlights_all_formats(
            highlights_path, output, config, video_duration
        )
        
        if format != 'all':
            # 特定形式のみ表示
            format_map = {
                'premiere': 'premiere_pro',
                'davinci': 'davinci_resolve',
                'youtube': 'youtube_chapters',
                'timeline': 'timeline_report'
            }
            target_format = format_map.get(format, format)
            if target_format in output_files:
                click.echo(f"✅ {format}形式エクスポート完了: {output_files[target_format]}")
            else:
                click.echo(f"❌ {format}形式のエクスポートに失敗")
        else:
            # 全形式の結果表示
            click.echo(f"✅ 全形式エクスポート完了: {len(output_files)}個のファイル生成")
            for format_name, file_path in output_files.items():
                click.echo(f"  📄 {format_name}: {file_path}")
        
    except Exception as e:
        logger.error(f"エクスポートエラー: {e}")
        click.echo(f"❌ エクスポート失敗: {e}")
        sys.exit(1)


@cli.command()
def version():
    """バージョン情報"""
    click.echo("ShortsClipper v2.0.0")
    click.echo("純粋関数ベース YouTube ハイライト検出ツール")
    click.echo("音声境界自動検出機能搭載")


if __name__ == '__main__':
    cli()