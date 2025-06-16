#!/usr/bin/env python3
"""
音声ファイル分割ユーティリティ
音声長に応じた適応的分割処理を提供
"""

import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class AudioSegment:
    """音声セグメント情報"""
    segment_id: str
    file_path: str
    start_time: float
    end_time: float
    duration: float
    overlap_start: Optional[float] = None  # 前セグメントとのオーバーラップ開始時間
    overlap_end: Optional[float] = None    # 次セグメントとのオーバーラップ終了時間
    original_start: Optional[float] = None  # 元のセグメント開始時刻（重複除去用）
    original_end: Optional[float] = None    # 元のセグメント終了時刻（重複除去用）


class AudioSplitter:
    """音声分割処理クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def get_audio_duration(self, audio_path: str) -> float:
        """音声ファイルの長さを取得"""
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", audio_path
            ], capture_output=True, text=True, check=True)
            
            info = json.loads(result.stdout)
            duration = float(info['format']['duration'])
            self.logger.info(f"音声長取得: {duration:.1f}秒 ({duration/60:.1f}分)")
            return duration
            
        except Exception as e:
            self.logger.error(f"音声長取得エラー: {e}")
            raise
    
    def determine_split_strategy(self, duration: float) -> Tuple[float, bool]:
        """音声長に応じた分割戦略を決定"""
        if duration <= 10 * 60:  # 10分以下
            return duration, False  # 分割なし
        elif duration <= 60 * 60:  # 1時間以下
            return 10 * 60, True    # 10分分割
        else:  # 1時間超
            return 15 * 60, True    # 15分分割
    
    def find_silence_boundaries(self, audio_path: str, segment_duration: float, 
                              silence_threshold: float = -30, 
                              silence_duration: float = 0.5) -> List[float]:
        """無音区間を検出して最適な分割点を見つける"""
        try:
            # ffmpegで無音区間を検出
            result = subprocess.run([
                "ffmpeg", "-i", audio_path,
                "-af", f"silencedetect=noise={silence_threshold}dB:d={silence_duration}",
                "-f", "null", "-"
            ], capture_output=True, text=True, stderr=subprocess.STDOUT)
            
            # 無音区間の開始時刻を抽出
            silence_starts = []
            for line in result.stdout.split('\n'):
                if 'silence_start:' in line:
                    try:
                        time_str = line.split('silence_start:')[1].strip()
                        silence_starts.append(float(time_str))
                    except (IndexError, ValueError):
                        continue
            
            self.logger.info(f"無音区間検出: {len(silence_starts)}箇所")
            return silence_starts
            
        except Exception as e:
            self.logger.warning(f"無音区間検出エラー: {e}")
            return []
    
    def calculate_split_points(self, duration: float, target_duration: float, 
                             silence_boundaries: List[float]) -> List[float]:
        """最適な分割点を計算"""
        if not silence_boundaries:
            # 無音区間がない場合は等間隔分割
            return [i * target_duration for i in range(1, int(duration / target_duration) + 1)]
        
        split_points = []
        current_target = target_duration
        
        while current_target < duration:
            # 目標時刻の前後1分以内で最も近い無音区間を探す
            candidates = [s for s in silence_boundaries 
                         if abs(s - current_target) <= 60 and s > current_target - target_duration]
            
            if candidates:
                # 最も目標に近い無音区間を選択
                split_point = min(candidates, key=lambda x: abs(x - current_target))
                split_points.append(split_point)
                current_target = split_point + target_duration
            else:
                # 無音区間がない場合は目標時刻で分割
                split_points.append(current_target)
                current_target += target_duration
        
        self.logger.info(f"分割点計算: {len(split_points)}個の分割点")
        return split_points
    
    def create_segments(self, audio_path: str, split_points: List[float], 
                       duration: float, overlap_seconds: float = 5.0) -> List[AudioSegment]:
        """オーバーラップ付き音声セグメント情報を生成"""
        segments = []
        audio_name = Path(audio_path).stem
        
        # 分割点リストの準備（開始点0と終了点を追加）
        all_points = [0.0] + split_points + [duration]
        
        for i in range(len(all_points) - 1):
            original_start = all_points[i]
            original_end = all_points[i + 1]
            
            # オーバーラップを含む実際の抽出範囲を計算
            extract_start = max(0, original_start - overlap_seconds) if i > 0 else original_start
            extract_end = min(duration, original_end + overlap_seconds) if i < len(all_points) - 2 else original_end
            extract_duration = extract_end - extract_start
            
            # オーバーラップ情報を記録
            overlap_start = extract_start if i > 0 else None
            overlap_end = extract_end if i < len(all_points) - 2 else None
            
            segment = AudioSegment(
                segment_id=f"{audio_name}_seg_{i+1:03d}",
                file_path="",  # 後で設定
                start_time=extract_start,  # 実際の抽出開始時刻
                end_time=extract_end,      # 実際の抽出終了時刻
                duration=extract_duration,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                original_start=original_start,  # 元のセグメント境界
                original_end=original_end
            )
            
            segments.append(segment)
        
        self.logger.info(f"オーバーラップセグメント生成: {len(segments)}個")
        for i, seg in enumerate(segments):
            overlap_info = ""
            if seg.overlap_start is not None:
                overlap_info += f"前{overlap_seconds}s "
            if seg.overlap_end is not None:
                overlap_info += f"後{overlap_seconds}s"
            self.logger.debug(f"セグメント{i+1}: {seg.duration:.1f}s {overlap_info}")
        
        return segments
    
    def extract_audio_segments(self, audio_path: str, segments: List[AudioSegment], 
                             output_dir: str) -> List[AudioSegment]:
        """音声セグメントファイルを実際に抽出"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        updated_segments = []
        
        for segment in segments:
            segment_file = output_path / f"{segment.segment_id}.wav"
            
            try:
                # ffmpegで音声セグメントを抽出
                subprocess.run([
                    "ffmpeg", "-i", audio_path,
                    "-ss", str(segment.start_time),
                    "-t", str(segment.duration),
                    "-y", str(segment_file)
                ], check=True, capture_output=True)
                
                # ファイルパスを更新
                updated_segment = AudioSegment(
                    segment_id=segment.segment_id,
                    file_path=str(segment_file),
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    duration=segment.duration,
                    overlap_start=segment.overlap_start,
                    overlap_end=segment.overlap_end,
                    original_start=segment.original_start,
                    original_end=segment.original_end
                )
                updated_segments.append(updated_segment)
                
                self.logger.debug(f"セグメント抽出完了: {segment_file.name}")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"セグメント抽出エラー {segment.segment_id}: {e}")
                continue
        
        self.logger.info(f"音声分割完了: {len(updated_segments)}/{len(segments)}個成功")
        return updated_segments
    
    def split_audio_adaptive(self, audio_path: str, output_dir: str, 
                           force_split: bool = False, overlap_seconds: float = 5.0) -> List[AudioSegment]:
        """適応的音声分割のメイン処理"""
        self.logger.info(f"=== 適応的音声分割開始 ===")
        self.logger.info(f"入力: {audio_path}")
        self.logger.info(f"出力: {output_dir}")
        
        # 1. 音声長を取得
        duration = self.get_audio_duration(audio_path)
        
        # 2. 分割戦略を決定
        target_duration, should_split = self.determine_split_strategy(duration)
        
        if not should_split and not force_split:
            self.logger.info("分割不要: 元ファイルをそのまま使用")
            audio_name = Path(audio_path).stem
            return [AudioSegment(
                segment_id=f"{audio_name}_full",
                file_path=audio_path,
                start_time=0.0,
                end_time=duration,
                duration=duration,
                original_start=0.0,
                original_end=duration
            )]
        
        self.logger.info(f"分割戦略: {target_duration/60:.1f}分単位で分割")
        
        # 3. 無音区間を検出
        silence_boundaries = self.find_silence_boundaries(audio_path, target_duration)
        
        # 4. 最適な分割点を計算
        split_points = self.calculate_split_points(duration, target_duration, silence_boundaries)
        
        # 5. セグメント情報を生成（オーバーラップ付き）
        segments = self.create_segments(audio_path, split_points, duration, overlap_seconds)
        
        # 6. 音声セグメントファイルを抽出
        final_segments = self.extract_audio_segments(audio_path, segments, output_dir)
        
        self.logger.info(f"=== 適応的音声分割完了 ===")
        self.logger.info(f"総セグメント数: {len(final_segments)}")
        
        return final_segments


def main():
    """テスト用メイン関数"""
    import sys
    from src.utils.logging_config import setup_process_logging
    
    if len(sys.argv) < 3:
        print("Usage: python audio_splitter.py <audio_file> <output_dir>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    logger = setup_process_logging("audio_splitter", "processing")
    splitter = AudioSplitter(logger)
    
    segments = splitter.split_audio_adaptive(audio_file, output_dir)
    
    print(f"\n分割結果:")
    for segment in segments:
        print(f"  {segment.segment_id}: {segment.duration:.1f}秒 ({segment.file_path})")


if __name__ == "__main__":
    main()