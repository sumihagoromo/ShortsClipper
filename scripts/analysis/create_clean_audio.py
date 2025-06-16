#!/usr/bin/env python3
"""
音楽部分をスキップして音声のみの部分を抽出
"""

import subprocess
from pathlib import Path


def create_clean_audio():
    """音楽部分をスキップした音声ファイルを作成"""
    input_audio = "data/stage1_audio/sumi-claude-code-04_audio.wav"
    output_audio = "data/stage1_audio/sumi-claude-code-04_audio_clean.wav"
    
    print("=== 音楽部分をスキップしたクリーン音声作成 ===")
    
    # 仮説: 最初の3分は音楽、その後から音声開始
    skip_seconds = 180  # 3分スキップ
    
    print(f"最初の{skip_seconds}秒をスキップして音声抽出中...")
    
    try:
        subprocess.run([
            "ffmpeg", "-i", input_audio,
            "-ss", str(skip_seconds),  # 3分スキップ
            "-y", output_audio
        ], check=True)
        
        output_path = Path(output_audio)
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ クリーン音声作成完了: {output_audio}")
            print(f"ファイルサイズ: {size_mb:.1f}MB")
            
            # 長さを確認
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", output_audio
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                duration = float(info['format']['duration'])
                print(f"音声長: {duration/60:.1f}分 (元の65分から{skip_seconds/60:.1f}分短縮)")
                
            return output_audio
        else:
            print("❌ ファイル作成に失敗")
            return None
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None


def test_clean_audio(clean_audio_path):
    """クリーン音声の最初の部分をテスト"""
    print(f"\n=== クリーン音声テスト ===")
    
    # 最初の2分をテスト
    test_file = "/tmp/clean_test.wav"
    
    try:
        subprocess.run([
            "ffmpeg", "-i", clean_audio_path,
            "-t", "120",  # 最初の2分
            "-y", test_file
        ], check=True, capture_output=True)
        
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="auto", compute_type="float32")
        
        segments, info = model.transcribe(
            test_file,
            language="ja",
            beam_size=1,
            temperature=0.0
        )
        
        segments_list = list(segments)
        print(f"検出セグメント数: {len(segments_list)}")
        print(f"言語確率: {info.language_probability:.3f}")
        
        if len(segments_list) > 0:
            print("最初の5セグメント:")
            for i, seg in enumerate(segments_list[:5]):
                print(f"  {i+1}. {seg.start:5.1f}s | {seg.text}")
                
            # 有意義な音声があるかチェック
            meaningful_segments = [
                seg for seg in segments_list 
                if len(seg.text.strip()) > 3 and 
                not any(repeat in seg.text for repeat in ['このステージ', 'このように', '【音楽】'])
            ]
            
            if len(meaningful_segments) > 0:
                print(f"✅ 有意義な音声検出: {len(meaningful_segments)}個")
                return True
            else:
                print("❌ まだ音楽/ノイズが多い")
                return False
        else:
            print("❌ セグメントが検出されない")
            return False
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False


if __name__ == "__main__":
    clean_audio = create_clean_audio()
    
    if clean_audio:
        if test_clean_audio(clean_audio):
            print(f"\n🎯 成功！{clean_audio} を使用して文字起こしを再実行することを推奨")
        else:
            print(f"\n⚠️ {clean_audio} でもまだ音楽が混入している可能性があります")
            print("より多くの時間をスキップするか、手動で音声開始位置を特定してください")