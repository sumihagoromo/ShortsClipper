#!/usr/bin/env python3
"""
適応的転写処理モジュール
音声長に応じて最適な処理方法を自動選択し実行
"""

import sys
import json
import time
import yaml
from pathlib import Path
from typing import Optional, Dict
import click
import logging

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_config import setup_process_logging
from src.utils.audio_splitter import AudioSplitter
from src.utils.batch_transcription import BatchTranscriber
from src.utils.result_merger import TranscriptionMerger
from src.utils.adaptive_config import AdaptiveConfigManager
# Process transcript modules will be called as subprocess


class AdaptiveTranscriber:
    """適応的転写処理クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = AdaptiveConfigManager(self.logger)
        self.audio_splitter = AudioSplitter(self.logger)
        self.merger = TranscriptionMerger(self.logger)
    
    def process_audio_adaptive(self, audio_path: str, output_dir: str,
                             force_profile: Optional[str] = None,
                             quality_priority: bool = False,
                             force_split: bool = False,
                             overlap_seconds: float = 5.0,
                             use_overlap_processing: bool = True) -> Dict:
        """適応的音声転写処理のメイン関数"""
        start_time = time.time()
        
        self.logger.info(f"=== 適応的転写処理開始 ===")
        self.logger.info(f"入力音声: {audio_path}")
        self.logger.info(f"出力ディレクトリ: {output_dir}")
        
        try:
            # 1. 音声情報取得
            audio_duration = self.audio_splitter.get_audio_duration(audio_path)
            video_id = Path(audio_path).stem
            
            # 2. 適応的設定生成
            adaptive_config = self.config_manager.create_adaptive_config(
                audio_duration, force_profile, quality_priority
            )
            
            self.logger.info(f"選択プロファイル: {adaptive_config['profile']['name']}")
            self.logger.info(f"推定処理時間: {adaptive_config['processing']['estimated_time_seconds']/60:.1f}分")
            
            # 3. 処理方法決定
            should_split = adaptive_config['splitting']['should_split'] or force_split
            
            if should_split:
                # 分割処理
                result = self._process_with_splitting(
                    audio_path, output_dir, video_id, adaptive_config,
                    overlap_seconds, use_overlap_processing
                )
            else:
                # 直接処理
                result = self._process_directly(
                    audio_path, output_dir, video_id, adaptive_config
                )
            
            # 4. 処理時間記録
            total_time = time.time() - start_time
            result['metadata']['total_processing_time'] = total_time
            result['metadata']['profile_used'] = adaptive_config['profile']['name']
            
            # 5. 結果保存
            self._save_results(result, output_dir, video_id)
            
            self.logger.info(f"=== 適応的転写処理完了 ===")
            self.logger.info(f"総処理時間: {total_time/60:.1f}分")
            self.logger.info(f"検出セグメント数: {len(result['segments'])}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"適応的転写処理エラー: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _process_directly(self, audio_path: str, output_dir: str, 
                         video_id: str, adaptive_config: Dict) -> Dict:
        """直接処理（分割なし）"""
        self.logger.info("=== 直接転写処理実行 ===")
        
        # 設定ファイルを一時作成
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(adaptive_config['transcription'], f, default_flow_style=False, allow_unicode=True)
            temp_config_path = f.name
        
        try:
            # process_transcript.pyを直接呼び出し
            import subprocess
            cmd = [
                sys.executable, "process_transcript.py",
                "-i", audio_path,
                "-o", output_dir,
                "-c", temp_config_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=Path(__file__).parent,
                timeout=3600  # 1時間タイムアウト
            )
            
            if result.returncode == 0:
                # 結果ファイルを読み込み
                output_files = list(Path(output_dir).glob(f"{video_id}_cleaned.json"))
                if output_files:
                    with open(output_files[0], 'r', encoding='utf-8') as f:
                        transcription_result = json.load(f)
                    
                    # 統一フォーマットに変換
                    unified_result = {
                        "video_id": video_id,
                        "segments": transcription_result.get('segments', []),
                        "language": transcription_result.get('language', 'ja'),
                        "language_probability": transcription_result.get('language_probability', 1.0),
                        "metadata": {
                            "model_size": adaptive_config['transcription']['transcription'].get('model_size', 'base'),
                            "processing_method": "direct",
                            "audio_duration": adaptive_config['audio_info']['duration'],
                            "segments_count": len(transcription_result.get('segments', [])),
                            "processing_time": transcription_result.get('processing_time', 0),
                            "timestamp": time.time()
                        },
                        "processing_time": transcription_result.get('processing_time', 0),
                        "success": True,
                        "error_message": ""
                    }
                    
                    self.logger.info(f"直接転写完了: {len(unified_result['segments'])}セグメント")
                    return unified_result
                else:
                    raise Exception("出力ファイルが見つかりません")
            else:
                error_message = result.stderr or "Unknown subprocess error"
                self.logger.error(f"直接転写失敗: {error_message}")
                return self._create_error_result(error_message, 0)
                
        finally:
            # 一時設定ファイル削除
            Path(temp_config_path).unlink(missing_ok=True)
    
    def _process_with_splitting(self, audio_path: str, output_dir: str,
                              video_id: str, adaptive_config: Dict,
                              overlap_seconds: float = 5.0,
                              use_overlap_processing: bool = True) -> Dict:
        """分割処理"""
        self.logger.info("=== 分割転写処理実行 ===")
        
        # 一時ディレクトリ作成
        temp_dir = Path(output_dir) / "temp_segments"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 音声分割（オーバーラップ付き）
            segments = self.audio_splitter.split_audio_adaptive(
                audio_path, 
                str(temp_dir),
                force_split=True,
                overlap_seconds=overlap_seconds
            )
            
            self.logger.info(f"音声分割完了: {len(segments)}セグメント")
            
            # 2. バッチ転写用の一時設定ファイル作成
            import tempfile
            transcription_config = adaptive_config['transcription']
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(transcription_config, f, default_flow_style=False, allow_unicode=True)
                temp_config_path = f.name
            
            try:
                batch_transcriber = BatchTranscriber(
                    config_path=temp_config_path,
                    logger=self.logger
                )
            
                parallel_workers = adaptive_config['processing']['parallel_workers']
                transcription_results = batch_transcriber.transcribe_batch_parallel(
                    segments, max_workers=parallel_workers
                )
                
                # 3. 結果統合（オーバーラップ処理設定に応じて）
                merged_result = self.merger.merge_transcription_results(
                    transcription_results, video_id, use_overlap_processing=use_overlap_processing
                )
                
                # 4. 一時ファイル削除
                self._cleanup_temp_files(temp_dir)
                
            finally:
                # 一時設定ファイル削除
                Path(temp_config_path).unlink(missing_ok=True)
            
            self.logger.info(f"分割転写完了: {len(merged_result['segments'])}セグメント")
            return merged_result
            
        except Exception as e:
            self.logger.error(f"分割転写処理エラー: {e}")
            # 一時ファイル削除
            self._cleanup_temp_files(temp_dir)
            raise
    
    def _cleanup_temp_files(self, temp_dir: Path):
        """一時ファイルをクリーンアップ"""
        try:
            if temp_dir.exists():
                for file in temp_dir.glob("*.wav"):
                    file.unlink()
                temp_dir.rmdir()
                self.logger.info("一時ファイル削除完了")
        except Exception as e:
            self.logger.warning(f"一時ファイル削除エラー: {e}")
    
    def _save_results(self, result: Dict, output_dir: str, video_id: str):
        """結果をファイルに保存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Raw結果保存
        raw_file = output_path / f"{video_id}_adaptive_raw.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Cleaned結果保存（後処理適用）
        cleaned_result = self._apply_post_processing(result)
        cleaned_file = output_path / f"{video_id}_adaptive_cleaned.json"
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"結果保存: {raw_file}, {cleaned_file}")
    
    def _apply_post_processing(self, result: Dict) -> Dict:
        """後処理を適用（簡易版）"""
        if not result['success'] or not result['segments']:
            return result
        
        cleaned_result = result.copy()
        cleaned_segments = []
        
        for segment in result['segments']:
            cleaned_segment = segment.copy()
            
            # 基本的なテキストクリーニング
            text = cleaned_segment.get('text', '')
            if text:
                # 不要な繰り返し句読点を除去
                text = text.replace('。。。', '。')
                text = text.replace('、、、', '、')
                cleaned_segment['text'] = text
            
            cleaned_segments.append(cleaned_segment)
        
        cleaned_result['segments'] = cleaned_segments
        return cleaned_result
    
    def _create_error_result(self, error_message: str, processing_time: float) -> Dict:
        """エラー結果を作成"""
        return {
            "video_id": "unknown",
            "segments": [],
            "language": "ja",
            "language_probability": 0.0,
            "metadata": {
                "processing_method": "adaptive",
                "timestamp": time.time()
            },
            "processing_time": processing_time,
            "success": False,
            "error_message": error_message
        }


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='data/stage2_transcript', help='出力ディレクトリ')
@click.option('--profile', '-p', help='強制プロファイル指定 (direct/batch_standard/batch_fast/high_quality)')
@click.option('--quality', is_flag=True, help='品質優先モード')
@click.option('--force-split', is_flag=True, help='強制分割処理')
@click.option('--overlap-seconds', default=5.0, help='オーバーラップ時間（秒）')
@click.option('--no-overlap', is_flag=True, help='オーバーラップ処理を無効化')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(input_path, output, profile, quality, force_split, overlap_seconds, no_overlap, verbose):
    """適応的音声転写処理
    
    音声の長さに応じて最適な処理方法を自動選択し実行します。
    """
    
    # ロガー設定
    log_level = logging.DEBUG if verbose else logging.INFO
    video_id = Path(input_path).stem
    logger = setup_process_logging(
        f"adaptive_transcript_{video_id}", 
        "transcription", 
        log_level
    )
    
    try:
        # 適応的転写処理実行
        transcriber = AdaptiveTranscriber(logger)
        result = transcriber.process_audio_adaptive(
            input_path, output, profile, quality, force_split, 
            overlap_seconds, not no_overlap
        )
        
        if result['success']:
            print(f"✅ 転写完了: {len(result['segments'])}セグメント")
            print(f"   処理時間: {result['processing_time']/60:.1f}分")
            print(f"   使用プロファイル: {result['metadata'].get('profile_used', 'unknown')}")
        else:
            print(f"❌ 転写失敗: {result['error_message']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"メイン処理エラー: {e}")
        print(f"❌ 処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()