#!/usr/bin/env python3
"""
プロセス監視とログ確認スクリプト
"""

import time
import subprocess
import os
from datetime import datetime

def check_process_status():
    """Pythonプロセスの状態をチェック"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'python main.py sumi-claude-code-04.mp4' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu_time = parts[9]
                    memory = parts[5]
                    return {
                        'pid': pid,
                        'cpu_time': cpu_time,
                        'memory': memory,
                        'running': True
                    }
    except Exception as e:
        print(f"プロセスチェックエラー: {e}")
    
    return {'running': False}

def get_latest_log_entries(n=5):
    """最新のログエントリを取得"""
    log_file = "/Users/rysh/sumi/ShortsClipper/logs/shorts_clipper.log"
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n:]
    except Exception as e:
        return [f"ログ読み込みエラー: {e}"]

def monitor_loop():
    """監視ループ"""
    start_time = datetime.now()
    check_count = 0
    
    while True:
        check_count += 1
        current_time = datetime.now()
        elapsed = current_time - start_time
        
        print(f"\n=== 監視チェック #{check_count} ({current_time.strftime('%H:%M:%S')}) ===")
        print(f"監視開始からの経過時間: {elapsed}")
        
        # プロセス状態チェック
        status = check_process_status()
        if status['running']:
            print(f"✅ プロセス実行中")
            print(f"   PID: {status['pid']}")
            print(f"   CPU時間: {status['cpu_time']}")
            print(f"   メモリ: {status['memory']}")
        else:
            print("❌ プロセスが見つかりません")
            break
        
        # 最新ログエントリ
        print("📋 最新ログ:")
        log_entries = get_latest_log_entries(3)
        for entry in log_entries:
            print(f"   {entry.strip()}")
        
        # 出力ファイル確認
        if os.path.exists("output"):
            output_files = [f for f in os.listdir("output") if "sumi-claude-code-04" in f]
            if output_files:
                print(f"📁 出力ファイル: {len(output_files)}個作成済み")
            else:
                print("📁 出力ファイルはまだ作成されていません")
        
        print("---")
        
        # 5分待機
        time.sleep(300)  # 5分

if __name__ == "__main__":
    print("ShortsClipper プロセス監視を開始します")
    print("Ctrl+C で監視を停止")
    
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\n監視を停止しました")
    except Exception as e:
        print(f"監視中にエラーが発生しました: {e}")