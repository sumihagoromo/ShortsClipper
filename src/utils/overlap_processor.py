#!/usr/bin/env python3
"""
オーバーラップ処理と重複除去ユーティリティ
音声分割時のオーバーラップ部分から重複テキストを特定・除去
"""

import re
import difflib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class OverlapRegion:
    """オーバーラップ領域情報"""
    segment1_id: str
    segment2_id: str
    overlap_start: float  # セグメント1内での開始時刻
    overlap_end: float    # セグメント1内での終了時刻
    overlap_duration: float
    text1: str = ""       # セグメント1のオーバーラップ部分テキスト
    text2: str = ""       # セグメント2のオーバーラップ部分テキスト
    similarity: float = 0.0


class OverlapProcessor:
    """オーバーラップ処理クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def identify_overlap_regions(self, transcription_results: List[Dict], 
                               overlap_seconds: float = 5.0) -> List[OverlapRegion]:
        """転写結果からオーバーラップ領域を特定"""
        overlap_regions = []
        
        for i in range(len(transcription_results) - 1):
            current = transcription_results[i]
            next_result = transcription_results[i + 1]
            
            # セグメント情報取得
            current_end = getattr(current.get('audio_segment'), 'original_end', None)
            next_start = getattr(next_result.get('audio_segment'), 'original_start', None)
            
            if current_end is None or next_start is None:
                continue
                
            # オーバーラップ領域計算
            overlap_start = current_end - overlap_seconds
            overlap_end = current_end
            
            region = OverlapRegion(
                segment1_id=current['video_id'],
                segment2_id=next_result['video_id'],
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                overlap_duration=overlap_seconds
            )
            
            # オーバーラップ部分のテキストを抽出
            region.text1 = self._extract_overlap_text(
                current['segments'], overlap_start, overlap_end
            )
            region.text2 = self._extract_overlap_text(
                next_result['segments'], 0, overlap_seconds  # 次セグメントの開始から
            )
            
            # 類似度計算
            region.similarity = self._calculate_text_similarity(
                region.text1, region.text2
            )
            
            overlap_regions.append(region)
            
            self.logger.debug(f"オーバーラップ検出: {region.segment1_id} ↔ {region.segment2_id}, "
                            f"類似度: {region.similarity:.2f}")
        
        self.logger.info(f"オーバーラップ領域特定: {len(overlap_regions)}箇所")
        return overlap_regions
    
    def _extract_overlap_text(self, segments: List[Dict], start_time: float, 
                            end_time: float) -> str:
        """指定時間範囲のテキストを抽出"""
        overlap_texts = []
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # セグメントがオーバーラップ範囲と重複するかチェック
            if seg_end <= start_time or seg_start >= end_time:
                continue
                
            # 完全に範囲内にあるセグメント
            if seg_start >= start_time and seg_end <= end_time:
                overlap_texts.append(segment.get('text', ''))
            # 部分的に重複する場合（簡易的に全体を含める）
            elif (seg_start < end_time and seg_end > start_time):
                overlap_texts.append(segment.get('text', ''))
        
        return ''.join(overlap_texts)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """2つのテキストの類似度を計算"""
        if not text1 or not text2:
            return 0.0
            
        # 正規化（空白・句読点を除去）
        normalized1 = re.sub(r'[^\w]', '', text1)
        normalized2 = re.sub(r'[^\w]', '', text2)
        
        if not normalized1 or not normalized2:
            return 0.0
        
        # シーケンスマッチングで類似度計算
        matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
        similarity = matcher.ratio()
        
        return similarity
    
    def remove_duplicates(self, transcription_results: List[Dict], 
                         overlap_regions: List[OverlapRegion],
                         similarity_threshold: float = 0.7) -> List[Dict]:
        """重複テキストを除去"""
        self.logger.info(f"重複除去開始: 閾値={similarity_threshold}")
        
        # 結果をコピー
        cleaned_results = [result.copy() for result in transcription_results]
        
        duplicates_removed = 0
        
        for region in overlap_regions:
            if region.similarity < similarity_threshold:
                continue
                
            # 高い類似度の場合、後のセグメントから重複部分を除去
            next_idx = None
            for i, result in enumerate(cleaned_results):
                if result['video_id'] == region.segment2_id:
                    next_idx = i
                    break
            
            if next_idx is None:
                continue
                
            # 次セグメントの冒頭部分（オーバーラップ時間内）のセグメントを削除
            original_count = len(cleaned_results[next_idx]['segments'])
            cleaned_segments = []
            
            for segment in cleaned_results[next_idx]['segments']:
                seg_start = segment.get('start', 0)
                
                # オーバーラップ時間を超えているセグメントのみ保持
                if seg_start >= region.overlap_duration:
                    # タイムスタンプを調整（オーバーラップ分を引く）
                    adjusted_segment = segment.copy()
                    adjusted_segment['start'] = seg_start - region.overlap_duration
                    adjusted_segment['end'] = adjusted_segment.get('end', 0) - region.overlap_duration
                    cleaned_segments.append(adjusted_segment)
            
            cleaned_results[next_idx]['segments'] = cleaned_segments
            removed_count = original_count - len(cleaned_segments)
            duplicates_removed += removed_count
            
            self.logger.debug(f"重複除去: {region.segment2_id} から {removed_count}セグメント削除")
        
        self.logger.info(f"重複除去完了: 総計{duplicates_removed}セグメント削除")
        return cleaned_results
    
    def merge_overlapped_results(self, transcription_results: List[Dict],
                               overlap_seconds: float = 5.0,
                               similarity_threshold: float = 0.7) -> Dict:
        """オーバーラップ処理付きで転写結果をマージ"""
        self.logger.info("=== オーバーラップ処理開始 ===")
        
        if not transcription_results:
            return self._create_empty_result()
        
        # 1. オーバーラップ領域を特定
        overlap_regions = self.identify_overlap_regions(
            transcription_results, overlap_seconds
        )
        
        # 2. 重複を除去
        cleaned_results = self.remove_duplicates(
            transcription_results, overlap_regions, similarity_threshold
        )
        
        # 3. 結果をマージ
        merged_result = self._merge_cleaned_results(cleaned_results)
        
        # 4. 統計情報を追加
        merged_result['overlap_processing'] = {
            'overlap_regions_found': len(overlap_regions),
            'high_similarity_regions': len([r for r in overlap_regions 
                                          if r.similarity >= similarity_threshold]),
            'similarity_threshold': similarity_threshold,
            'overlap_seconds': overlap_seconds
        }
        
        self.logger.info("=== オーバーラップ処理完了 ===")
        self.logger.info(f"処理前セグメント数: {sum(len(r['segments']) for r in transcription_results)}")
        self.logger.info(f"処理後セグメント数: {len(merged_result['segments'])}")
        
        return merged_result
    
    def _merge_cleaned_results(self, cleaned_results: List[Dict]) -> Dict:
        """クリーニング済み結果をマージ"""
        if not cleaned_results:
            return self._create_empty_result()
        
        base_result = cleaned_results[0].copy()
        all_segments = []
        current_time_offset = 0
        
        for i, result in enumerate(cleaned_results):
            segments = result.get('segments', [])
            
            for segment in segments:
                adjusted_segment = segment.copy()
                # グローバル時刻に調整
                adjusted_segment['start'] += current_time_offset
                adjusted_segment['end'] += current_time_offset
                all_segments.append(adjusted_segment)
            
            # 次のセグメントのための時刻オフセット更新
            if segments:
                last_segment = segments[-1]
                current_time_offset = last_segment.get('end', 0) + current_time_offset
        
        # 結果を構築
        merged_result = {
            'video_id': base_result.get('video_id', 'merged'),
            'segments': all_segments,
            'language': base_result.get('language', 'ja'),
            'language_probability': base_result.get('language_probability', 1.0),
            'metadata': {
                'processing_method': 'overlap_aware_merge',
                'segments_count': len(all_segments),
                'source_segments': len(cleaned_results)
            }
        }
        
        return merged_result
    
    def _create_empty_result(self) -> Dict:
        """空の結果を作成"""
        return {
            'video_id': 'empty',
            'segments': [],
            'language': 'ja',
            'language_probability': 0.0,
            'metadata': {
                'processing_method': 'overlap_aware_merge',
                'segments_count': 0,
                'source_segments': 0
            }
        }


def main():
    """テスト用メイン関数"""
    import sys
    import json
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python overlap_processor.py <transcription_results_dir>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    # テスト用ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    processor = OverlapProcessor(logger)
    
    # 転写結果ファイルを読み込み
    transcription_files = list(results_dir.glob("*_cleaned.json"))
    transcription_results = []
    
    for file_path in sorted(transcription_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            transcription_results.append(result)
    
    # オーバーラップ処理実行
    merged_result = processor.merge_overlapped_results(transcription_results)
    
    # 結果出力
    output_file = results_dir / "merged_overlap_processed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=2)
    
    print(f"オーバーラップ処理完了: {output_file}")
    print(f"最終セグメント数: {len(merged_result['segments'])}")


if __name__ == "__main__":
    main()