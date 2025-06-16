#!/usr/bin/env python3
"""
音楽終了の正確な位置（2分27秒）から音声を抽出
"""

import subprocess
from pathlib import Path
import json


def create_precise_clean_audio():
    """実際の音楽終了位置から音声ファイルを作成"""
    input_audio = "data/stage1_audio/sumi-claude-code-04_audio.wav"
    output_audio = "data/stage1_audio/sumi-claude-code-04_audio_clean_precise.wav"
    
    print("=== 正確な音楽終了位置からクリーン音声作成 ===")
    
    # 実際の音楽終了: 2分27秒 = 147秒
    skip_seconds = 147
    
    print(f"最初の{skip_seconds}秒（2分27秒）をスキップして音声抽出中...")
    
    try:
        subprocess.run([
            "ffmpeg", "-i", input_audio,
            "-ss", str(skip_seconds),  # 2分27秒スキップ
            "-y", output_audio
        ], check=True, capture_output=True)
        
        output_path = Path(output_audio)
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✅ 正確なクリーン音声作成完了: {output_audio}")
            print(f"ファイルサイズ: {size_mb:.1f}MB")
            
            # 長さを確認
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", output_audio
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                duration = float(info['format']['duration'])
                print(f"音声長: {duration/60:.1f}分 (元の65分から{skip_seconds/60:.1f}分短縮)")
                print(f"実際の音声時間: {duration:.1f}秒")
                
            return output_audio
        else:
            print("❌ ファイル作成に失敗")
            return None
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None


def extract_problem_segment():
    """問題区間（147-381秒相当）を抽出"""
    input_audio = "data/stage1_audio/sumi-claude-code-04_audio_clean_precise.wav"
    output_audio = "data/stage1_audio/problem_segment_147-381.wav"
    
    print("\n=== 問題区間（最初の234秒間）抽出 ===")
    
    # 147秒スキップ後の最初の234秒（147+234=381秒まで）
    duration_seconds = 234  # 381 - 147 = 234秒
    
    try:
        subprocess.run([
            "ffmpeg", "-i", input_audio,
            "-t", str(duration_seconds),  # 最初の234秒のみ
            "-y", output_audio
        ], check=True, capture_output=True)
        
        output_path = Path(output_audio)
        if output_path.exists():
            print(f"✅ 問題区間抽出完了: {output_audio}")
            print(f"区間: 元動画の147秒-381秒（{duration_seconds/60:.1f}分間）")
            return output_audio
        else:
            print("❌ 問題区間抽出に失敗")
            return None
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None


def extract_success_segment():
    """成功区間（381秒以降）を抽出"""
    input_audio = "data/stage1_audio/sumi-claude-code-04_audio.wav"
    output_audio = "data/stage1_audio/success_segment_381plus.wav"
    
    print("\n=== 成功区間（381秒以降）抽出 ===")
    
    # 381秒以降
    skip_seconds = 381
    
    try:
        subprocess.run([
            "ffmpeg", "-i", input_audio,
            "-ss", str(skip_seconds),  # 381秒スキップ
            "-y", output_audio
        ], check=True, capture_output=True)
        
        output_path = Path(output_audio)
        if output_path.exists():
            print(f"✅ 成功区間抽出完了: {output_audio}")
            print(f"区間: 元動画の381秒以降")
            return output_audio
        else:
            print("❌ 成功区間抽出に失敗")
            return None
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None


if __name__ == "__main__":
    # 1. 正確な位置からのクリーン音声作成
    clean_audio = create_precise_clean_audio()
    
    if clean_audio:
        # 2. 問題区間の抽出
        problem_segment = extract_problem_segment()
        
        # 3. 成功区間の抽出
        success_segment = extract_success_segment()
        
        print(f"\n🎯 次のステップ:")
        print(f"1. 正確なクリーン音声での転写: {clean_audio}")
        print(f"2. 問題区間の個別分析: {problem_segment}")
        print(f"3. 成功区間の個別分析: {success_segment}")
        print(f"\n実行コマンド例:")
        print(f"python main.py transcript {clean_audio} --model base")