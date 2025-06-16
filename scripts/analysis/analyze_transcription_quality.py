#!/usr/bin/env python3
"""
高精度文字起こし結果の分析と表示
"""

import json
from collections import Counter


def analyze_transcription():
    """文字起こし結果を分析"""
    with open('data/stage2_transcript/sumi-claude-code-04_cleaned.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data['segments']
    
    print('=== 高精度文字起こし結果分析 ===')
    print(f'総セグメント数: {len(segments)}')
    print()
    
    # テキスト統計
    all_texts = [seg['text'].strip() for seg in segments if seg['text'].strip()]
    unique_texts = list(set(all_texts))
    
    print(f'ユニークテキスト数: {len(unique_texts)}')
    print(f'総文字数: {sum(len(text) for text in all_texts)}')
    print()
    
    # 頻出フレーズ分析
    text_counts = Counter(all_texts)
    print('=== 頻出フレーズ TOP 10 ===')
    for text, count in text_counts.most_common(10):
        print(f'{count:3d}回 | {text}')
    print()
    
    # 多様な発話サンプル（頻出フレーズ以外）
    common_phrases = {text for text, count in text_counts.most_common(5)}
    
    diverse_samples = []
    seen_texts = set()
    
    for segment in segments:
        text = segment['text'].strip()
        if (text and 
            text not in common_phrases and 
            text not in seen_texts and
            len(text) > 3):  # 3文字以上
            seen_texts.add(text)
            diverse_samples.append({
                'time': f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}",
                'text': text,
                'confidence': segment['confidence'],
                'duration': segment['duration']
            })
            if len(diverse_samples) >= 20:
                break
    
    print('=== 多様な発話内容 (頻出フレーズ除外) ===')
    for sample in diverse_samples:
        print(f"{sample['time']} | {sample['text'][:50]}{'...' if len(sample['text']) > 50 else ''}")
    print()
    
    # 長い発話セグメント
    long_segments = [seg for seg in segments if len(seg['text'].strip()) > 20]
    long_segments.sort(key=lambda x: len(x['text']), reverse=True)
    
    print('=== 長い発話セグメント TOP 10 ===')
    for i, segment in enumerate(long_segments[:10]):
        time_str = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
        print(f"{i+1:2d}. {time_str} | {segment['text'][:80]}{'...' if len(segment['text']) > 80 else ''}")
    print()
    
    # 信頼度分析
    confidences = [seg['confidence'] for seg in segments]
    avg_confidence = sum(confidences) / len(confidences)
    print(f'平均信頼度: {avg_confidence:.3f}')
    print(f'最高信頼度: {max(confidences):.3f}')
    print(f'最低信頼度: {min(confidences):.3f}')
    
    # 高信頼度セグメント
    high_conf_segments = [seg for seg in segments if seg['confidence'] > 0.5]
    print(f'高信頼度セグメント (>0.5): {len(high_conf_segments)}個')
    
    if high_conf_segments:
        print('\n=== 高信頼度発話サンプル ===')
        for i, segment in enumerate(high_conf_segments[:5]):
            time_str = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
            print(f"{time_str} | {segment['text']} | 信頼度:{segment['confidence']:.3f}")


if __name__ == "__main__":
    analyze_transcription()