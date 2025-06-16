#!/usr/bin/env python3
"""
転写結果統合ユーティリティ
分割セグメントの転写結果を統合して完全な転写を生成
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import logging

from .batch_transcription import TranscriptionResult
from .overlap_processor import OverlapProcessor


@dataclass
class MergerStats:
    """統合処理統計"""
    total_segments: int
    successful_segments: int
    failed_segments: int
    total_audio_duration: float
    total_processing_time: float
    merged_segments_count: int
    duplicate_removals: int
    timestamp_adjustments: int


class TranscriptionMerger:
    """転写結果統合クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.overlap_processor = OverlapProcessor(logger)
    
    def merge_transcription_results(self, batch_results: List[TranscriptionResult],
                                  video_id: str, use_overlap_processing: bool = True) -> Dict:
        """バッチ転写結果を統合（オーバーラップ処理対応）"""
        self.logger.info(f"=== 転写結果統合開始 ===")
        self.logger.info(f"入力セグメント数: {len(batch_results)}")
        self.logger.info(f"オーバーラップ処理: {'有効' if use_overlap_processing else '無効'}")
        
        # 成功したセグメントのみを処理
        successful_results = [r for r in batch_results if r.success]
        failed_results = [r for r in batch_results if not r.success]
        
        if failed_results:
            self.logger.warning(f"失敗セグメント: {len(failed_results)}個")
            for failed in failed_results:
                self.logger.warning(f"  {failed.segment_id}: {failed.error_message}")
        
        if not successful_results:
            self.logger.error("統合可能なセグメントがありません")
            return self._create_empty_result(video_id, batch_results)
        
        # セグメントを時系列順にソート
        successful_results.sort(key=lambda x: x.start_offset)
        
        if use_overlap_processing and len(successful_results) > 1:
            # オーバーラップ処理を使用
            transcription_dicts = self._convert_to_transcription_dicts(successful_results)
            merged_result = self.overlap_processor.merge_overlapped_results(transcription_dicts)
            
            # メタデータを追加
            merged_result['video_id'] = video_id
            merged_result['metadata'].update({
                'total_segments': len(batch_results),
                'successful_segments': len(successful_results),
                'failed_segments': len(failed_results),
                'processing_time': sum(r.processing_time for r in batch_results),
            })
            merged_result['processing_time'] = sum(r.processing_time for r in batch_results)
            merged_result['success'] = True
            merged_result['error_message'] = ""
            
        else:
            # 従来の統合処理
            merged_segments = self._merge_segments(successful_results)
            cleaned_segments = self._remove_duplicates(merged_segments)
            stats = self._calculate_stats(batch_results, successful_results, cleaned_segments)
            merged_result = self._build_final_result(
                video_id, cleaned_segments, successful_results, stats
            )
        
        self.logger.info(f"=== 転写結果統合完了 ===")
        self.logger.info(f"統合セグメント数: {len(merged_result['segments'])}")
        
        return merged_result
    
    def _convert_to_transcription_dicts(self, results: List[TranscriptionResult]) -> List[Dict]:
        """TranscriptionResultをdict形式に変換"""
        transcription_dicts = []
        
        for result in results:
            # 各セグメントにオーディオセグメント情報を付加
            transcription_dict = {
                'video_id': result.segment_id,
                'segments': result.segments,
                'language': result.language,
                'language_probability': result.language_probability,
                'audio_segment': result.audio_segment  # AudioSegmentオブジェクト
            }
            transcription_dicts.append(transcription_dict)
        
        return transcription_dicts
    
    def _merge_segments(self, results: List[TranscriptionResult]) -> List[Dict]:
        """セグメントデータを時系列順に統合"""
        merged_segments = []
        
        for result in results:
            if result.segments:
                merged_segments.extend(result.segments)
        
        # タイムスタンプでソート
        merged_segments.sort(key=lambda x: x['start'])
        
        self.logger.info(f"セグメント統合: {len(merged_segments)}個")
        return merged_segments
    
    def _remove_duplicates(self, segments: List[Dict], 
                          time_threshold: float = 1.0) -> List[Dict]:
        """重複セグメントを除去"""
        if not segments:
            return segments
        
        cleaned_segments = []
        removed_count = 0
        
        # 最初のセグメントは常に追加
        cleaned_segments.append(segments[0])
        
        for current_segment in segments[1:]:
            last_segment = cleaned_segments[-1]
            
            # 時間的重複をチェック
            time_overlap = self._calculate_time_overlap(last_segment, current_segment)
            text_similarity = self._calculate_text_similarity(
                last_segment.get('text', ''), 
                current_segment.get('text', '')
            )
            
            # 重複判定条件
            is_duplicate = (
                time_overlap > time_threshold and text_similarity > 0.8
            ) or (
                time_overlap > 0.5 and text_similarity > 0.95
            )
            
            if is_duplicate:
                removed_count += 1
                self.logger.debug(
                    f"重複除去: {current_segment['start']:.1f}s "
                    f"(overlap={time_overlap:.1f}s, similarity={text_similarity:.2f})"
                )
                # より信頼度の高いセグメントを保持
                if current_segment.get('confidence', 0) > last_segment.get('confidence', 0):
                    cleaned_segments[-1] = current_segment
            else:
                cleaned_segments.append(current_segment)
        
        self.logger.info(f"重複除去完了: {removed_count}個除去")
        return cleaned_segments
    
    def _calculate_time_overlap(self, seg1: Dict, seg2: Dict) -> float:
        """2つのセグメント間の時間的重複を計算"""
        start1, end1 = seg1['start'], seg1['end']
        start2, end2 = seg2['start'], seg2['end']
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        return max(0, overlap_end - overlap_start)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """テキストの類似度を計算（簡単なアルゴリズム）"""
        if not text1 or not text2:
            return 0.0
        
        # 単語レベルでの類似度計算
        words1 = set(text1.replace('。', '').replace('、', '').split())
        words2 = set(text2.replace('。', '').replace('、', '').split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _validate_timestamp_sequence(self, segments: List[Dict]) -> List[Dict]:
        """タイムスタンプの整合性を検証・修正"""
        if not segments:
            return segments
        
        validated_segments = []
        adjustment_count = 0
        
        for i, segment in enumerate(segments):
            current_segment = segment.copy()
            
            # 前のセグメントとの整合性チェック
            if validated_segments:
                prev_segment = validated_segments[-1]
                if current_segment['start'] < prev_segment['end']:
                    # 重複がある場合は調整
                    current_segment['start'] = prev_segment['end']
                    adjustment_count += 1
            
            # start <= end の保証
            if current_segment['start'] >= current_segment['end']:
                current_segment['end'] = current_segment['start'] + 0.1
                adjustment_count += 1
            
            validated_segments.append(current_segment)
        
        if adjustment_count > 0:
            self.logger.info(f"タイムスタンプ調整: {adjustment_count}箇所")
        
        return validated_segments
    
    def _calculate_stats(self, all_results: List[TranscriptionResult],
                        successful_results: List[TranscriptionResult],
                        final_segments: List[Dict]) -> MergerStats:
        """統合処理の統計を計算"""
        return MergerStats(
            total_segments=len(all_results),
            successful_segments=len(successful_results),
            failed_segments=len(all_results) - len(successful_results),
            total_audio_duration=sum(r.audio_duration for r in all_results),
            total_processing_time=sum(r.processing_time for r in all_results),
            merged_segments_count=len(final_segments),
            duplicate_removals=sum(r.segments_count for r in successful_results) - len(final_segments),
            timestamp_adjustments=0  # _validate_timestamp_sequenceで設定
        )
    
    def _build_final_result(self, video_id: str, segments: List[Dict],
                           successful_results: List[TranscriptionResult],
                           stats: MergerStats) -> Dict:
        """最終的な転写結果を構築"""
        # タイムスタンプ検証
        validated_segments = self._validate_timestamp_sequence(segments)
        
        return {
            "video_id": video_id,
            "segments": validated_segments,
            "language": "ja",
            "language_probability": self._calculate_average_language_probability(successful_results),
            "metadata": {
                "model_size": "base",  # 設定から取得すべき
                "processing_method": "adaptive_batch",
                "total_segments": stats.total_segments,
                "successful_segments": stats.successful_segments,
                "failed_segments": stats.failed_segments,
                "audio_duration": stats.total_audio_duration,
                "segments_count": len(validated_segments),
                "processing_time": stats.total_processing_time,
                "duplicate_removals": stats.duplicate_removals,
                "timestamp_adjustments": stats.timestamp_adjustments,
                "timestamp": time.time()
            },
            "processing_time": stats.total_processing_time,
            "success": True,
            "error_message": ""
        }
    
    def _calculate_average_language_probability(self, results: List[TranscriptionResult]) -> float:
        """平均言語確率を計算"""
        if not results:
            return 0.0
        
        probabilities = [r.language_probability for r in results if r.language_probability > 0]
        return sum(probabilities) / len(probabilities) if probabilities else 0.0
    
    def _create_empty_result(self, video_id: str, all_results: List[TranscriptionResult]) -> Dict:
        """失敗時の空結果を作成"""
        return {
            "video_id": video_id,
            "segments": [],
            "language": "ja",
            "language_probability": 0.0,
            "metadata": {
                "model_size": "base",
                "processing_method": "adaptive_batch",
                "total_segments": len(all_results),
                "successful_segments": 0,
                "failed_segments": len(all_results),
                "audio_duration": sum(r.audio_duration for r in all_results),
                "segments_count": 0,
                "processing_time": sum(r.processing_time for r in all_results),
                "timestamp": time.time()
            },
            "processing_time": sum(r.processing_time for r in all_results),
            "success": False,
            "error_message": "No successful segments to merge"
        }
    
    def save_merged_result(self, merged_result: Dict, output_path: str) -> str:
        """統合結果を保存"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"統合結果保存: {output_file}")
        return str(output_file)


def main():
    """テスト用メイン関数"""
    import sys
    from src.utils.logging_config import setup_process_logging
    
    if len(sys.argv) < 3:
        print("Usage: python result_merger.py <batch_results.json> <output_file>")
        sys.exit(1)
    
    batch_file = sys.argv[1]
    output_file = sys.argv[2]
    
    logger = setup_process_logging("result_merger", "processing")
    
    # バッチ結果を読み込み
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    # TranscriptionResultオブジェクトに変換
    results = []
    for segment_data in batch_data['segment_results']:
        result = TranscriptionResult(**segment_data)
        results.append(result)
    
    # 統合処理
    merger = TranscriptionMerger(logger)
    merged_result = merger.merge_transcription_results(results, "test_video")
    
    # 結果保存
    merger.save_merged_result(merged_result, output_file)
    
    print(f"\n統合処理完了:")
    print(f"  入力セグメント: {len(results)}")
    print(f"  出力セグメント: {len(merged_result['segments'])}")
    print(f"  成功率: {merged_result['metadata']['successful_segments']}/{merged_result['metadata']['total_segments']}")


if __name__ == "__main__":
    main()