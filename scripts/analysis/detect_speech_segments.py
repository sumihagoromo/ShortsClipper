#!/usr/bin/env python3
"""
音声ファイルから音楽と音声部分を自動検出
"""

from faster_whisper import WhisperModel
import json


def find_speech_start():
    """音声開始位置を検出"""
    audio_path = "data/stage1_audio/sumi-claude-code-04_audio.wav"
    
    print("=== 音声開始位置検出 ===")
    
    # 最初の10分間をチェック
    test_segments = [
        (0, 120),      # 0-2分
        (120, 240),    # 2-4分  
        (240, 360),    # 4-6分
        (360, 480),    # 6-8分
        (480, 600),    # 8-10分
    ]
    
    model = WhisperModel("base", device="auto", compute_type="float32")
    
    for start_time, end_time in test_segments:
        print(f"\n--- {start_time//60}:{start_time%60:02d} - {end_time//60}:{end_time%60:02d} ---")
        
        # セグメントを抽出
        import subprocess
        temp_file = f"/tmp/test_{start_time}-{end_time}.wav"
        
        try:
            subprocess.run([
                "ffmpeg", "-i", audio_path,
                "-ss", str(start_time), "-t", str(end_time - start_time),
                "-y", temp_file
            ], capture_output=True, check=True)
            
            # Whisperで処理
            segments, info = model.transcribe(
                temp_file,
                language="ja",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False
            )
            
            segments_list = list(segments)
            
            print(f"検出セグメント数: {len(segments_list)}")
            print(f"言語確率: {info.language_probability:.3f}")
            
            if len(segments_list) > 0:
                print("検出内容:")
                for i, seg in enumerate(segments_list[:3]):
                    print(f"  {i+1}. {seg.text}")
                    
                # 実際の音声があるかチェック
                speech_segments = [seg for seg in segments_list if len(seg.text.strip()) > 2]
                if len(speech_segments) > 0:
                    print(f"✅ 音声検出！ 実質的な発話: {len(speech_segments)}個")
                    print(f"推奨開始位置: {start_time}秒 ({start_time//60}:{start_time%60:02d})")
                    return start_time
                else:
                    print("❌ 音楽/ノイズのみ")
            else:
                print("❌ 無音またはノイズ")
                
        except Exception as e:
            print(f"エラー: {e}")
    
    return None


if __name__ == "__main__":
    speech_start = find_speech_start()
    
    if speech_start is not None:
        print(f"\n🎯 推奨: {speech_start}秒 ({speech_start//60}:{speech_start%60:02d}) から音声処理を開始")
    else:
        print("\n⚠️ 明確な音声開始位置が見つかりませんでした")