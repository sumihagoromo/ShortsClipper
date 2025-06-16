#!/usr/bin/env python3
"""
文字起こし進捗監視ツール
バックグラウンド実行中の文字起こし処理の進捗を監視する
"""

import time
import subprocess
import os
from pathlib import Path
from datetime import datetime


def check_process_status(pid):
    """プロセスの実行状況をチェック"""
    try:
        result = subprocess.run(['ps', '-p', str(pid)], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def get_latest_log_entries(log_file, n=10):
    """最新のログエントリを取得"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n:] if lines else []
    except Exception as e:
        return [f"ログ読み込みエラー: {e}"]


def get_output_files_status():
    """出力ファイルの状況を確認"""
    output_dir = Path("data/stage2_transcript")
    
    files = []
    if output_dir.exists():
        for file in output_dir.glob("sumi-claude-code-04*"):
            stat = file.stat()
            files.append({
                'name': file.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
    
    return files


def estimate_progress(log_entries):
    """ログからおおよその進捗を推定"""
    progress_indicators = {
        "モデルを読み込み中": 10,
        "音声転写を実行中": 20,
        "Processing audio with duration": 25,
        "Processing segment at": 30,  # これが増えていく
        "転写完了": 80,
        "後処理中": 85,
        "結果保存": 95,
        "プロセス完了": 100
    }
    
    current_progress = 0
    segment_count = 0
    
    for line in log_entries:
        for indicator, progress in progress_indicators.items():
            if indicator in line:
                current_progress = max(current_progress, progress)
        
        # セグメント処理数をカウント
        if "Processing segment at" in line:
            segment_count += 1
    
    # セグメント数に基づく進捗補正（65分≈130セグメント程度と仮定）
    if segment_count > 0:
        segment_progress = min(50, segment_count * 0.4)  # 30-80%の範囲
        current_progress = max(current_progress, 30 + segment_progress)
    
    return min(100, current_progress), segment_count


def main():
    """メイン監視ループ"""
    pid = 90497  # バックグラウンドプロセスのPID
    log_file = "transcript_progress.log"
    
    print("=== 文字起こし進捗監視 ===")
    print(f"監視対象PID: {pid}")
    print(f"ログファイル: {log_file}")
    print("Ctrl+C で監視を停止")
    print()
    
    start_time = datetime.now()
    
    try:
        while True:
            current_time = datetime.now()
            elapsed = current_time - start_time
            
            print(f"\n⏰ {current_time.strftime('%H:%M:%S')} (経過: {elapsed})")
            print("-" * 50)
            
            # プロセス状況
            if check_process_status(pid):
                print("🟢 プロセス実行中")
            else:
                print("🔴 プロセス停止")
                print("処理が完了したか、エラーで停止した可能性があります")
                break
            
            # ログ解析
            log_entries = get_latest_log_entries(log_file, 20)
            progress, segment_count = estimate_progress(log_entries)
            
            print(f"📊 推定進捗: {progress}%")
            if segment_count > 0:
                print(f"🔄 処理済みセグメント: {segment_count}")
            
            # 最新ログ（重要なもののみ）
            print("📋 最新ログ:")
            important_logs = []
            for line in log_entries[-5:]:
                line = line.strip()
                if any(keyword in line for keyword in [
                    "INFO", "ERROR", "WARNING", "モデル", "転写", "Processing", 
                    "完了", "エラー", "失敗", "成功"
                ]):
                    # タイムスタンプを除去して短縮
                    if " - " in line:
                        parts = line.split(" - ", 2)
                        if len(parts) >= 3:
                            important_logs.append(f"  {parts[2]}")
                        else:
                            important_logs.append(f"  {line}")
                    else:
                        important_logs.append(f"  {line}")
            
            for log in important_logs[-3:]:  # 最新3件
                print(log)
            
            # 出力ファイル状況
            output_files = get_output_files_status()
            if output_files:
                print("📁 出力ファイル:")
                for file in output_files:
                    size_mb = file['size'] / (1024 * 1024)
                    print(f"  {file['name']}: {size_mb:.1f}MB ({file['modified'].strftime('%H:%M:%S')})")
            else:
                print("📁 出力ファイル: まだ作成されていません")
            
            # 推定残り時間（rough estimate）
            if progress > 25 and progress < 95:
                # 実際の処理が始まってからの推定
                processing_elapsed = elapsed.total_seconds()
                estimated_total = processing_elapsed / (progress / 100)
                remaining = estimated_total - processing_elapsed
                remaining_min = remaining / 60
                print(f"⏳ 推定残り時間: {remaining_min:.0f}分")
            
            # 30秒待機
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n監視を停止しました")
        
        # 最終状況を確認
        if check_process_status(pid):
            print(f"⚠️  プロセス {pid} はまだ実行中です")
            print("停止する場合: kill {pid}")
        else:
            print("✅ プロセスは正常に完了しています")
            
            # 結果ファイル確認
            output_files = get_output_files_status()
            if output_files:
                print("📊 生成された結果ファイル:")
                for file in output_files:
                    size_mb = file['size'] / (1024 * 1024)
                    print(f"  {file['name']}: {size_mb:.1f}MB")
    
    except Exception as e:
        print(f"\n❌ 監視中にエラー: {e}")


if __name__ == "__main__":
    main()