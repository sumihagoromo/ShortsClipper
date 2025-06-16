#!/usr/bin/env python3
"""
既存の出力結果を新しい純粋関数ベース形式にコンバートするユーティリティ

Usage:
  python convert_legacy_data.py --input output/sumi-claude-code-04_transcription.json --output data/stage2_transcript/
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import click


def convert_transcription_to_new_format(
    legacy_data: Dict[str, Any],
    video_id: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    既存の文字起こしデータを新しい形式にコンバートする
    
    Args:
        legacy_data: 既存のデータ
        video_id: 動画ID
        
    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: (raw結果, cleaned結果)
    """
    segments = legacy_data.get('segments', [])
    metadata = legacy_data.get('metadata', {})
    
    # 生の結果形式に変換
    raw_segments = []
    for segment in segments:
        raw_segment = {
            'start': segment.get('start', 0.0),
            'end': segment.get('end', 0.0),
            'text': segment.get('text', ''),
            'words': segment.get('words', []),
            'confidence': None  # 既存データにはconfidenceがない
        }
        raw_segments.append(raw_segment)
    
    raw_result = {
        'video_id': video_id,
        'input_audio_path': f'data/stage1_audio/{video_id}_audio.wav',
        'config': {
            'model_size': metadata.get('model_size', 'base'),
            'language': metadata.get('language', 'ja'),
            'temperature': 0.0,
            'initial_prompt': '以下は日本語の音声です。',
            'word_timestamps': True,
            'condition_on_previous_text': False,
            'device': 'cpu',
            'compute_type': 'float32',
            'beam_size': 5,
            'best_of': 5,
            'patience': 1.0,
            'length_penalty': 1.0,
            'repetition_penalty': 1.1,
            'no_repeat_ngram_size': 0
        },
        'segments': raw_segments,
        'language': metadata.get('language', 'ja'),
        'language_probability': metadata.get('language_probability', 1.0),
        'metadata': {
            'model_size': metadata.get('model_size', 'base'),
            'audio_file': f'data/stage1_audio/{video_id}_audio.wav',
            'audio_duration': metadata.get('total_duration', 0.0),
            'language': metadata.get('language', 'ja'),
            'language_probability': metadata.get('language_probability', 1.0),
            'processing_time': metadata.get('processing_time', 0.0),
            'segments_count': metadata.get('segments_count', len(segments)),
            'total_duration': metadata.get('total_duration', 0.0),
            'word_count': metadata.get('word_count', len(segments)),
            'timestamp': time.time(),
            'converted_from_legacy': True
        },
        'processing_time': metadata.get('processing_time', 0.0),
        'success': True,
        'error_message': ''
    }
    
    # 清浄化された結果を作成
    cleaned_segments = []
    total_corrections = 0
    
    for raw_segment in raw_segments:
        text = raw_segment['text']
        
        # 既存データの問題点を修正
        corrections = []
        
        # 大量の句点を除去
        if text and len(text) > 10 and text.count('。') / len(text) > 0.8:
            text = ""
            corrections.append("大量の句点を除去")
        
        # 空白の正規化
        text = text.strip()
        
        cleaned_segment = {
            'start': raw_segment['start'],
            'end': raw_segment['end'],
            'text': text,
            'original_text': raw_segment['text'],
            'corrections': corrections,
            'confidence': raw_segment['confidence'],
            'duration': raw_segment['end'] - raw_segment['start'],
            'word_count': len(text.split()) if text else 0
        }
        
        cleaned_segments.append(cleaned_segment)
        total_corrections += len(corrections)
    
    cleaned_result = {
        'video_id': video_id,
        'segments': cleaned_segments,
        'correction_stats': {
            'total_corrections': total_corrections,
            'correction_types': {'句点除去': total_corrections},
            'segments_processed': len(cleaned_segments),
            'processing_time': 0.1
        },
        'metadata': {
            'based_on_raw_result': True,
            'raw_segments_count': len(raw_segments),
            'cleaned_segments_count': len(cleaned_segments),
            'timestamp': time.time(),
            'converted_from_legacy': True
        },
        'processing_time': 0.1
    }
    
    return raw_result, cleaned_result


@click.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='入力する既存の転写結果JSONファイル')
@click.option('--output', '-o', 'output_dir', default='data/stage2_transcript',
              help='出力ディレクトリ')
@click.option('--video-id', default=None,
              help='動画ID（指定しない場合はファイル名から自動推測）')
def main(input_path: str, output_dir: str, video_id: str):
    """
    既存の転写結果を新しい形式にコンバート
    """
    input_file = Path(input_path)
    output_path = Path(output_dir)
    
    # 動画IDの推測
    if not video_id:
        video_id = input_file.stem.replace('_transcription', '')
    
    print(f"入力ファイル: {input_file}")
    print(f"出力ディレクトリ: {output_path}")
    print(f"動画ID: {video_id}")
    
    # 既存データ読み込み
    with open(input_file, 'r', encoding='utf-8') as f:
        legacy_data = json.load(f)
    
    print(f"既存データを読み込み: {len(legacy_data.get('segments', []))}セグメント")
    
    # 新しい形式にコンバート
    raw_result, cleaned_result = convert_transcription_to_new_format(legacy_data, video_id)
    
    # 出力ディレクトリ作成
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ファイル保存
    raw_file = output_path / f"{video_id}_raw.json"
    with open(raw_file, 'w', encoding='utf-8') as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)
    
    cleaned_file = output_path / f"{video_id}_cleaned.json"
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ コンバート完了:")
    print(f"  Raw: {raw_file}")
    print(f"  Cleaned: {cleaned_file}")
    print(f"  セグメント数: {len(raw_result['segments'])}")
    print(f"  修正適用数: {cleaned_result['correction_stats']['total_corrections']}")


if __name__ == '__main__':
    main()