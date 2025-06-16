#!/usr/bin/env python3
"""
バッチ転写処理ユーティリティ
分割された音声セグメントの並行転写処理を提供
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import yaml

from .audio_splitter import AudioSegment
import subprocess
import sys


@dataclass
class TranscriptionResult:
    """転写結果データクラス"""
    segment_id: str
    success: bool
    segments: List[Dict] = None
    error_message: str = ""
    processing_time: float = 0.0
    language: str = "ja"
    language_probability: float = 0.0
    segments_count: int = 0
    audio_duration: float = 0.0
    start_offset: float = 0.0  # 元音声での開始時刻オフセット
    audio_segment: AudioSegment = None  # オーバーラップ処理用


class BatchTranscriber:
    """バッチ転写処理クラス"""
    
    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None):
        self.config_path = config_path
        self.logger = logger or logging.getLogger(__name__)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """設定ファイルを読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"設定読み込み完了: {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")
            raise
    
    def transcribe_single_segment(self, audio_segment: AudioSegment) -> TranscriptionResult:
        """単一セグメントの転写処理"""
        start_time = time.time()
        
        try:
            self.logger.info(f"セグメント転写開始: {audio_segment.segment_id}")
            
            # 一時設定ファイルを作成
            temp_config_path = f"/tmp/{audio_segment.segment_id}_config.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            # process_transcript.pyを直接呼び出し
            temp_output_dir = f"/tmp/{audio_segment.segment_id}_output"
            Path(temp_output_dir).mkdir(exist_ok=True)
            
            cmd = [
                sys.executable, "process_transcript.py",
                "-i", audio_segment.file_path,
                "-o", temp_output_dir,
                "-c", temp_config_path
            ]
            
            # サブプロセス実行
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=Path(__file__).parent.parent.parent,  # プロジェクトルート
                timeout=1800  # 30分タイムアウト
            )
            
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                # 結果ファイルを読み込み
                output_files = list(Path(temp_output_dir).glob("*_cleaned.json"))
                if output_files:
                    with open(output_files[0], 'r', encoding='utf-8') as f:
                        segment_result = json.load(f)
                    
                    # タイムスタンプをグローバル時刻に調整
                    adjusted_segments = self._adjust_timestamps(
                        segment_result.get('segments', []), 
                        audio_segment.start_time
                    )
                    
                    transcription_result = TranscriptionResult(
                        segment_id=audio_segment.segment_id,
                        success=True,
                        segments=adjusted_segments,
                        processing_time=processing_time,
                        language=segment_result.get('language', 'ja'),
                        language_probability=segment_result.get('language_probability', 0.0),
                        segments_count=len(adjusted_segments),
                        audio_duration=audio_segment.duration,
                        start_offset=audio_segment.start_time,
                        audio_segment=audio_segment
                    )
                    
                    self.logger.info(
                        f"セグメント転写完了: {audio_segment.segment_id} "
                        f"({len(adjusted_segments)}セグメント, {processing_time:.1f}秒)"
                    )
                else:
                    raise Exception("出力ファイルが見つかりません")
            else:
                error_message = result.stderr or "Unknown subprocess error"
                transcription_result = TranscriptionResult(
                    segment_id=audio_segment.segment_id,
                    success=False,
                    error_message=error_message,
                    processing_time=processing_time,
                    audio_duration=audio_segment.duration,
                    start_offset=audio_segment.start_time,
                    audio_segment=audio_segment
                )
                
                self.logger.error(
                    f"セグメント転写エラー: {audio_segment.segment_id} - {error_message}"
                )
            
            # 一時ファイル削除
            self._cleanup_temp_files(temp_config_path, temp_output_dir)
            
            return transcription_result
            
        except Exception as e:
            self.logger.error(f"セグメント転写例外: {audio_segment.segment_id} - {e}")
            return TranscriptionResult(
                segment_id=audio_segment.segment_id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                audio_duration=audio_segment.duration,
                start_offset=audio_segment.start_time,
                audio_segment=audio_segment
            )
    
    def _adjust_timestamps(self, segments: List[Dict], start_offset: float) -> List[Dict]:
        """セグメント内のタイムスタンプをグローバル時刻に調整"""
        adjusted_segments = []
        
        for segment in segments:
            adjusted_segment = segment.copy()
            adjusted_segment['start'] += start_offset
            adjusted_segment['end'] += start_offset
            
            # 単語レベルタイムスタンプも調整
            if 'words' in adjusted_segment:
                adjusted_words = []
                for word in adjusted_segment['words']:
                    adjusted_word = word.copy()
                    adjusted_word['start'] += start_offset
                    adjusted_word['end'] += start_offset
                    adjusted_words.append(adjusted_word)
                adjusted_segment['words'] = adjusted_words
            
            adjusted_segments.append(adjusted_segment)
        
        return adjusted_segments
    
    def _cleanup_temp_files(self, temp_config_path: str, temp_output_dir: str):
        """一時ファイルをクリーンアップ"""
        try:
            # 設定ファイル削除
            config_path = Path(temp_config_path)
            if config_path.exists():
                config_path.unlink()
            
            # 出力ディレクトリ削除
            output_path = Path(temp_output_dir)
            if output_path.exists():
                for file in output_path.glob("*"):
                    file.unlink()
                output_path.rmdir()
        except Exception as e:
            self.logger.warning(f"一時ファイル削除エラー: {e}")
    
    def transcribe_batch_parallel(self, audio_segments: List[AudioSegment], 
                                max_workers: int = 3) -> List[TranscriptionResult]:
        """並行バッチ転写処理"""
        self.logger.info(f"=== 並行バッチ転写開始 ===")
        self.logger.info(f"セグメント数: {len(audio_segments)}")
        self.logger.info(f"並行度: {max_workers}")
        
        results = []
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 全セグメントを並行実行
            future_to_segment = {
                executor.submit(self.transcribe_single_segment, segment): segment
                for segment in audio_segments
            }
            
            # 完了順に結果を収集
            for future in as_completed(future_to_segment):
                segment = future_to_segment[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if not result.success:
                        failed_count += 1
                    
                    # 進捗報告
                    completed = len(results)
                    self.logger.info(
                        f"進捗: {completed}/{len(audio_segments)} "
                        f"({completed/len(audio_segments)*100:.1f}%) "
                        f"完了"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Future処理エラー {segment.segment_id}: {e}")
                    failed_count += 1
                    results.append(TranscriptionResult(
                        segment_id=segment.segment_id,
                        success=False,
                        error_message=str(e),
                        audio_duration=segment.duration,
                        start_offset=segment.start_time
                    ))
        
        # 結果をstart_offsetでソート（時系列順）
        results.sort(key=lambda x: x.start_offset)
        
        self.logger.info(f"=== 並行バッチ転写完了 ===")
        self.logger.info(f"成功: {len(results) - failed_count}/{len(results)}")
        self.logger.info(f"失敗: {failed_count}/{len(results)}")
        
        return results
    
    def transcribe_batch_sequential(self, audio_segments: List[AudioSegment]) -> List[TranscriptionResult]:
        """逐次バッチ転写処理（デバッグ用）"""
        self.logger.info(f"=== 逐次バッチ転写開始 ===")
        self.logger.info(f"セグメント数: {len(audio_segments)}")
        
        results = []
        failed_count = 0
        
        for i, segment in enumerate(audio_segments):
            self.logger.info(f"処理中: {i+1}/{len(audio_segments)} - {segment.segment_id}")
            
            result = self.transcribe_single_segment(segment)
            results.append(result)
            
            if not result.success:
                failed_count += 1
            
            # 進捗報告
            progress = (i + 1) / len(audio_segments) * 100
            self.logger.info(f"進捗: {progress:.1f}%")
        
        self.logger.info(f"=== 逐次バッチ転写完了 ===")
        self.logger.info(f"成功: {len(results) - failed_count}/{len(results)}")
        self.logger.info(f"失敗: {failed_count}/{len(results)}")
        
        return results
    
    def save_batch_results(self, results: List[TranscriptionResult], 
                          output_path: str) -> str:
        """バッチ転写結果を保存"""
        batch_result = {
            "batch_metadata": {
                "total_segments": len(results),
                "successful_segments": sum(1 for r in results if r.success),
                "failed_segments": sum(1 for r in results if not r.success),
                "total_processing_time": sum(r.processing_time for r in results),
                "total_audio_duration": sum(r.audio_duration for r in results),
                "config_path": self.config_path,
                "timestamp": time.time()
            },
            "segment_results": [asdict(result) for result in results]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"バッチ結果保存: {output_file}")
        return str(output_file)


def main():
    """テスト用メイン関数"""
    import sys
    from src.utils.logging_config import setup_process_logging
    from src.utils.audio_splitter import AudioSplitter
    
    if len(sys.argv) < 4:
        print("Usage: python batch_transcription.py <audio_file> <config_file> <output_dir>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    config_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    logger = setup_process_logging("batch_transcription", "processing")
    
    # 1. 音声分割
    splitter = AudioSplitter(logger)
    segments = splitter.split_audio_adaptive(audio_file, f"{output_dir}/segments")
    
    # 2. バッチ転写
    transcriber = BatchTranscriber(config_file, logger)
    results = transcriber.transcribe_batch_parallel(segments, max_workers=2)
    
    # 3. 結果保存
    transcriber.save_batch_results(results, f"{output_dir}/batch_results.json")
    
    print(f"\nバッチ処理完了:")
    print(f"  総セグメント: {len(results)}")
    print(f"  成功: {sum(1 for r in results if r.success)}")
    print(f"  失敗: {sum(1 for r in results if not r.success)}")


if __name__ == "__main__":
    main()