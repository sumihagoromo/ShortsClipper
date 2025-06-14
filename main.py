#!/usr/bin/env python3
"""
ShortsClipper - YouTubeライブ配信動画分析ツール

音声認識、感情分析、ハイライト検出を行い、動画編集を支援します。
"""

import sys
import logging
from pathlib import Path
import click

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import ConfigManager


@click.command()
@click.argument('video_file', type=click.Path(exists=True))
@click.option('--config', '-c', default='config/settings.yaml', 
              help='設定ファイルのパス')
@click.option('--output-dir', '-o', default=None,
              help='出力ディレクトリ (設定ファイルの値を上書き)')
@click.option('--verbose', '-v', is_flag=True,
              help='詳細ログ出力')
@click.option('--model-size', default=None,
              help='Whisperモデルサイズ (tiny, base, small, medium, large, large-v3)')
@click.option('--language', default=None,
              help='音声言語 (ja, en, etc.)')
def main(video_file, config, output_dir, verbose, model_size, language):
    """
    動画ファイルを分析してハイライトを検出します。
    
    VIDEO_FILE: 分析する動画ファイルのパス
    """
    try:
        # 設定管理の初期化
        config_manager = ConfigManager(config)
        
        # ログレベルの調整
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger = logging.getLogger(__name__)
        logger.info("ShortsClipper を開始します")
        logger.info(f"動画ファイル: {video_file}")
        
        # 設定の表示
        logger.info("=== 設定情報 ===")
        transcription_config = config_manager.get_transcription_config()
        logger.info(f"Whisperモデル: {transcription_config.get('model_size', 'unknown')}")
        logger.info(f"言語: {transcription_config.get('language', 'unknown')}")
        
        # コマンドライン引数での設定上書き
        if model_size:
            transcription_config['model_size'] = model_size
            logger.info(f"モデルサイズを {model_size} に変更")
            
        if language:
            transcription_config['language'] = language
            logger.info(f"言語を {language} に変更")
            
        if output_dir:
            output_config = config_manager.get_output_config()
            output_config['output_dir'] = output_dir
            logger.info(f"出力ディレクトリを {output_dir} に変更")
        
        # TODO: 各モジュールの実装後に有効化
        logger.info("=== 処理開始 ===")
        
        # 1. 音声抽出
        logger.info("1. 音声抽出を開始...")
        from src.utils.audio_processor import full_audio_extraction_workflow
        
        audio_config = config_manager.get_audio_config()
        try:
            audio_file = full_audio_extraction_workflow(video_file, audio_config)
            logger.info(f"音声抽出完了: {audio_file}")
        except Exception as e:
            logger.error(f"音声抽出に失敗しました: {e}")
            raise
        
        # 2. 文字起こし
        logger.info("2. 文字起こしを開始...")
        from src.transcriber import full_transcription_workflow
        
        transcription_config = config_manager.get_transcription_config()
        try:
            transcription_result = full_transcription_workflow(audio_file, transcription_config)
            logger.info(f"文字起こし完了: {transcription_result['metadata']['segments_count']}セグメント")
        except Exception as e:
            logger.error(f"文字起こしに失敗しました: {e}")
            raise
        
        # 3. 感情分析
        logger.info("3. 感情分析を開始...")
        from src.emotion_analyzer import full_emotion_analysis_workflow
        
        emotion_config = config_manager.get_emotion_config()
        try:
            emotion_result = full_emotion_analysis_workflow(transcription_result, emotion_config)
            logger.info(f"感情分析完了: {emotion_result['metadata']['segments_analyzed']}セグメント")
        except Exception as e:
            logger.error(f"感情分析に失敗しました: {e}")
            raise
        
        # 4. ハイライト検出
        logger.info("4. ハイライト検出を開始...")
        from src.highlight_detector import full_highlight_detection_workflow
        
        highlight_config = config_manager.get_highlight_config()
        try:
            highlight_result = full_highlight_detection_workflow(emotion_result, highlight_config)
            logger.info(f"ハイライト検出完了: {highlight_result['metadata']['total_highlights']}個")
        except Exception as e:
            logger.error(f"ハイライト検出に失敗しました: {e}")
            raise
        
        # 5. 結果出力
        logger.info("5. 結果出力を開始...")
        from src.output_formatter import format_and_save_results
        
        output_config = config_manager.get_output_config()
        try:
            saved_files = format_and_save_results(
                transcription_result,
                emotion_result,
                highlight_result,
                output_config,
                video_file
            )
            logger.info(f"結果出力完了: {sum(len(files) for files in saved_files.values())}ファイル")
        except Exception as e:
            logger.error(f"結果出力に失敗しました: {e}")
            raise
        
        logger.info("=== 処理完了 ===")
        logger.info("すべての処理が正常に完了しました")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        if verbose:
            logger.exception("詳細なエラー情報:")
        sys.exit(1)


@click.group()
def cli():
    """ShortsClipper コマンドラインツール"""
    pass


@cli.command()
@click.option('--config', '-c', default='config/settings.yaml')
def validate_config(config):
    """設定ファイルの検証"""
    try:
        config_manager = ConfigManager(config)
        click.echo("✅ 設定ファイルは正常です")
        
        # 設定の詳細表示
        click.echo("\n=== 設定内容 ===")
        click.echo(f"Whisperモデル: {config_manager.get('transcription.model_size')}")
        click.echo(f"言語: {config_manager.get('transcription.language')}")
        click.echo(f"出力ディレクトリ: {config_manager.get('output.output_dir')}")
        click.echo(f"ログレベル: {config_manager.get('logging.level')}")
        
    except Exception as e:
        click.echo(f"❌ 設定ファイルにエラーがあります: {e}")
        sys.exit(1)


@cli.command()
def version():
    """バージョン情報を表示"""
    click.echo("ShortsClipper v1.0.0-alpha")
    click.echo("YouTube動画ハイライト検出ツール")


if __name__ == '__main__':
    # CLIグループコマンドとして実行する場合
    if len(sys.argv) > 1 and sys.argv[1] in ['validate-config', 'version']:
        cli()
    else:
        # メインコマンドとして実行
        main()