#!/usr/bin/env python3
"""
40-45分セグメントでのWhisperテスト
"""

from faster_whisper import WhisperModel
import json


def test_short_segment():
    """5分間のセグメントをテスト"""
    audio_path = "/tmp/test_40-45min.wav"
    
    print("=== 40-45分セグメント Whisperテスト ===")
    
    # Faster-Whisper medium model
    print("Whisper medium model で文字起こし中...")
    
    model = WhisperModel("medium", device="auto", compute_type="float32")
    
    # シンプルな設定でテスト
    segments, info = model.transcribe(
        audio_path,
        language="ja",
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt="",
        word_timestamps=True,
        vad_filter=True,  # 音声活動検出
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    print(f"検出言語: {info.language} (確率: {info.language_probability:.3f})")
    
    segments_list = list(segments)
    print(f"検出セグメント数: {len(segments_list)}")
    
    if len(segments_list) == 0:
        print("セグメントが検出されませんでした")
        return
    
    print("\n=== 検出された全セグメント ===")
    unique_texts = set()
    
    for i, segment in enumerate(segments_list):
        time_str = f"{int(segment.start)//60:02d}:{int(segment.start)%60:02d}"
        print(f"{i+1:2d}. {time_str} | {segment.text} | 信頼度:{segment.avg_logprob:.3f}")
        unique_texts.add(segment.text.strip())
    
    print(f"\nユニークテキスト数: {len(unique_texts)}")
    print("ユニークテキスト一覧:")
    for text in unique_texts:
        print(f"  - {text}")
    
    # より詳細な分析
    if len(segments_list) > 0:
        avg_confidence = sum(seg.avg_logprob for seg in segments_list) / len(segments_list)
        print(f"\n平均信頼度: {avg_confidence:.3f}")
        
        total_duration = sum(seg.end - seg.start for seg in segments_list)
        print(f"音声のある部分の総時間: {total_duration:.1f}秒 / 300秒")


if __name__ == "__main__":
    test_short_segment()