#!/usr/bin/env python3
"""
Whisperの基本テスト - 設定を変えながら音声認識テスト
"""

import whisper
from faster_whisper import WhisperModel
import json
from pathlib import Path


def test_whisper_basic():
    """基本的なWhisperテスト"""
    audio_path = "data/stage1_audio/sumi-claude-code-04_audio.wav"
    
    print("=== 音声ファイルの最初30秒をテスト ===")
    
    # テスト1: OpenAI Whisperライブラリ
    print("\n--- テスト1: OpenAI Whisper (medium) ---")
    try:
        model = whisper.load_model("medium")
        result = model.transcribe(
            audio_path,
            language="ja",
            verbose=True,
            initial_prompt="",
            temperature=0.0,
            word_timestamps=True
        )
        
        print(f"検出言語: {result['language']}")
        print("最初の5セグメント:")
        for i, segment in enumerate(result['segments'][:5]):
            print(f"{i+1}. {segment['start']:.1f}s-{segment['end']:.1f}s: {segment['text']}")
            
    except Exception as e:
        print(f"エラー: {e}")
    
    # テスト2: Faster-Whisper (設定変更)
    print("\n--- テスト2: Faster-Whisper (設定調整) ---")
    try:
        model = WhisperModel("medium", device="auto", compute_type="float32")
        
        # より緩い設定でテスト
        segments, info = model.transcribe(
            audio_path,
            language="ja",
            beam_size=1,  # より単純
            best_of=1,    # より単純
            temperature=0.2,  # 少し高め
            condition_on_previous_text=False,  # 前文脈無視
            initial_prompt="",  # プロンプト無し
            word_timestamps=True,
            vad_filter=True,  # 音声活動検出
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"検出言語: {info.language} (確率: {info.language_probability:.3f})")
        
        segments_list = list(segments)
        print(f"セグメント数: {len(segments_list)}")
        
        print("最初の10セグメント:")
        for i, segment in enumerate(segments_list[:10]):
            print(f"{i+1}. {segment.start:.1f}s-{segment.end:.1f}s: {segment.text}")
            
        # ユニークなテキストを確認
        unique_texts = set(seg.text.strip() for seg in segments_list)
        print(f"\nユニークテキスト数: {len(unique_texts)}")
        for text in list(unique_texts)[:5]:
            print(f"  {text}")
            
    except Exception as e:
        print(f"エラー: {e}")
    
    # テスト3: 音声の一部だけテスト（最初の30秒）
    print("\n--- テスト3: 最初30秒のみ ---")
    try:
        import subprocess
        
        # 最初の30秒を抽出
        temp_audio = "/tmp/test_30s.wav"
        subprocess.run([
            "ffmpeg", "-i", audio_path, 
            "-t", "30", "-y", temp_audio
        ], capture_output=True)
        
        model = WhisperModel("medium", device="auto")
        segments, info = model.transcribe(
            temp_audio,
            language="ja",
            beam_size=1,
            temperature=0.0
        )
        
        segments_list = list(segments)
        print(f"30秒テスト - セグメント数: {len(segments_list)}")
        print(f"検出言語: {info.language} (確率: {info.language_probability:.3f})")
        
        for i, segment in enumerate(segments_list):
            print(f"{i+1}. {segment.start:.1f}s-{segment.end:.1f}s: {segment.text}")
            
    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    test_whisper_basic()