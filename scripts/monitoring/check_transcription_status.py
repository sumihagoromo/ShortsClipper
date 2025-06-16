#!/usr/bin/env python3
"""
定期的な文字起こし進捗確認スクリプト
30分間隔でプロセス状況を確認し報告
"""

import time
import subprocess
import os
from datetime import datetime
from pathlib import Path


def check_process_status(pid):
    """プロセスの実行状況をチェック"""
    try:
        result = subprocess.run(['ps', '-p', str(pid)], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def get_latest_segment(log_file):
    """最新の処理セグメントを取得"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in reversed(lines):
            if "Processing segment at" in line:
                # タイムスタンプとセグメント位置を抽出
                parts = line.strip().split()
                if len(parts) >= 8:
                    timestamp = parts[0] + " " + parts[1]
                    segment_time = parts[-1]
                    return timestamp, segment_time
        return None, None
    except Exception as e:
        return f"Error: {e}", None


def calculate_progress(segment_time_str):
    """進捗率を計算"""
    try:
        if segment_time_str:
            # "03:30.000" -> 3.5分
            time_parts = segment_time_str.split(':')
            minutes = int(time_parts[0])
            seconds = float(time_parts[1])
            current_minutes = minutes + seconds / 60
            total_minutes = 65  # 65分の動画
            progress = (current_minutes / total_minutes) * 100
            return progress, current_minutes
        return 0, 0
    except:
        return 0, 0


def estimate_completion_time(progress, start_time_str):
    """完了予定時刻を推定"""
    try:
        start_time = datetime.strptime("2025-06-15 " + start_time_str.split()[1], "%Y-%m-%d %H:%M:%S,%f")
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds() / 3600  # 時間
        
        if progress > 1:  # 1%以上進んでいる場合
            estimated_total_hours = elapsed / (progress / 100)
            estimated_completion = start_time + timedelta(hours=estimated_total_hours)
            return estimated_completion.strftime("%H:%M")
        return "計算中"
    except:
        return "不明"


def main():
    """30分間隔での監視"""
    pid = 90497
    log_file = "transcript_progress.log"
    
    print("=== 文字起こし進捗定期確認開始 ===")
    print(f"監視PID: {pid}")
    print("30分間隔で確認します (Ctrl+C で停止)")
    print()
    
    check_count = 1
    
    try:
        while True:
            current_time = datetime.now()
            print(f"\n📅 確認 #{check_count} - {current_time.strftime('%H:%M:%S')}")
            print("-" * 40)
            
            # プロセス状況確認
            if check_process_status(pid):
                print("🟢 プロセス実行中")
                
                # 最新進捗取得
                timestamp, segment_time = get_latest_segment(log_file)
                
                if segment_time:
                    progress, current_minutes = calculate_progress(segment_time)
                    print(f"📊 進捗: {progress:.1f}% ({current_minutes:.1f}/65.0分)")
                    print(f"🔄 最新セグメント: {segment_time}")
                    print(f"⏰ 最新更新: {timestamp}")
                    
                    # 推定残り時間
                    if progress > 1:
                        remaining_minutes = (65 - current_minutes) * (current_minutes / progress * 100)
                        remaining_hours = remaining_minutes / 60
                        print(f"⏳ 推定残り: {remaining_hours:.1f}時間")
                else:
                    print("📊 進捗情報取得できず")
                    
            else:
                print("🔴 プロセス停止 - 完了または異常終了")
                
                # 出力ファイル確認
                output_dir = Path("data/stage2_transcript")
                if output_dir.exists():
                    output_files = list(output_dir.glob("sumi-claude-code-04*"))
                    if output_files:
                        print("📁 出力ファイル:")
                        for file in output_files:
                            size_mb = file.stat().st_size / (1024 * 1024)
                            modified = datetime.fromtimestamp(file.stat().st_mtime)
                            print(f"  {file.name}: {size_mb:.1f}MB ({modified.strftime('%H:%M:%S')})")
                
                print("\n✅ 監視終了")
                break
            
            check_count += 1
            
            # 30分待機 (1800秒)
            print(f"\n⏱️  次回確認: {(current_time + timedelta(minutes=30)).strftime('%H:%M:%S')}")
            time.sleep(1800)  # 30分
            
    except KeyboardInterrupt:
        print("\n\n⏹️  監視を手動停止しました")
        if check_process_status(pid):
            print(f"ℹ️  プロセス {pid} はまだ実行中です")


if __name__ == "__main__":
    from datetime import timedelta
    main()