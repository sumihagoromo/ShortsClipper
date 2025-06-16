#!/usr/bin/env python3
"""
統一ログ設定モジュール
プロジェクト全体でログファイルの配置を標準化
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional, List


def setup_process_logging(
    process_name: str,
    log_category: str = "processing",
    level: int = logging.INFO,
    video_id: Optional[str] = None
) -> logging.Logger:
    """
    プロセス用ログ設定
    
    Args:
        process_name: プロセス名 (audio, transcript, emotions, highlights)
        log_category: ログカテゴリ (transcription, monitoring, processing)
        level: ログレベル
        video_id: 動画ID（指定時はファイル名に含める）
    
    Returns:
        設定済みロガー
    """
    # ログディレクトリ構造を作成
    log_dir = Path("logs") / log_category
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ログファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if video_id:
        log_filename = f"{process_name}_{video_id}_{timestamp}.log"
    else:
        log_filename = f"{process_name}_{timestamp}.log"
    
    log_path = log_dir / log_filename
    
    # ロガー設定
    logger = logging.getLogger(process_name)
    logger.setLevel(level)
    
    # 既存のハンドラーをクリア
    logger.handlers.clear()
    
    # ファイルハンドラー
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # プロパゲーションを無効化（重複ログ防止）
    logger.propagate = False
    
    return logger


def setup_transcription_logging(
    model_name: str,
    video_id: str,
    level: int = logging.INFO
) -> logging.Logger:
    """文字起こし専用ログ設定"""
    process_name = f"transcript_{model_name}"
    return setup_process_logging(
        process_name=process_name,
        log_category="transcription",
        level=level,
        video_id=video_id
    )


def setup_monitoring_logging(
    monitor_type: str,
    level: int = logging.INFO
) -> logging.Logger:
    """監視ツール専用ログ設定"""
    return setup_process_logging(
        process_name=monitor_type,
        log_category="monitoring",
        level=level
    )


def setup_analysis_logging(
    analysis_type: str,
    level: int = logging.INFO
) -> logging.Logger:
    """分析ツール専用ログ設定"""
    return setup_process_logging(
        process_name=analysis_type,
        log_category="analysis",
        level=level
    )


def get_latest_log_path(process_name: str, log_category: str = "processing") -> Optional[Path]:
    """最新のログファイルパスを取得"""
    log_dir = Path("logs") / log_category
    if not log_dir.exists():
        return None
    
    # プロセス名で始まるログファイルを検索
    log_files = list(log_dir.glob(f"{process_name}_*.log"))
    if not log_files:
        return None
    
    # 最新のファイルを返す
    return max(log_files, key=lambda f: f.stat().st_mtime)


def cleanup_old_logs(days_to_keep: int = 7, dry_run: bool = False) -> List[Path]:
    """古いログファイルをクリーンアップ"""
    from datetime import timedelta
    
    cutoff_time = datetime.now() - timedelta(days=days_to_keep)
    removed_files = []
    
    logs_root = Path("logs")
    if not logs_root.exists():
        return removed_files
    
    for log_file in logs_root.rglob("*.log"):
        if log_file.stat().st_mtime < cutoff_time.timestamp():
            if not dry_run:
                log_file.unlink()
            removed_files.append(log_file)
    
    return removed_files