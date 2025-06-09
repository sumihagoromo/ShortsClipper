import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
        self._ensure_directories()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"設定ファイルが見つかりません: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"設定ファイルの読み込みに失敗しました: {e}")
            raise

    def _setup_logging(self):
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', './logs/shorts_clipper.log')
        
        # ログディレクトリを作成
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # ローテーティングファイルハンドラーの設定
        from logging.handlers import RotatingFileHandler
        
        # ルートロガーの設定
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # 既存のハンドラーをクリア
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # ファイルハンドラー
        max_bytes = self._parse_size(log_config.get('max_file_size', '10MB'))
        backup_count = log_config.get('backup_count', 3)
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # ハンドラーを追加
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info("ログ設定が完了しました")

    def _parse_size(self, size_str: str) -> int:
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        size_str = str(size_str).upper()
        
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                size_num_str = size_str[:-len(unit)].strip()
                try:
                    size_num = float(size_num_str)
                    return int(size_num * multiplier)
                except ValueError:
                    logging.warning(f"サイズの解析に失敗しました: {size_str}")
                    return 10 * 1024 * 1024  # デフォルト10MB
        
        try:
            return int(size_str)
        except ValueError:
            logging.warning(f"サイズの解析に失敗しました: {size_str}")
            return 10 * 1024 * 1024  # デフォルト10MB

    def _ensure_directories(self):
        output_dir = Path(self.config.get('output', {}).get('output_dir', './output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logs_dir = Path('./logs')
        logs_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key_path: str, default=None):
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_transcription_config(self) -> Dict[str, Any]:
        return self.config.get('transcription', {})

    def get_emotion_config(self) -> Dict[str, Any]:
        return self.config.get('emotion_analysis', {})

    def get_highlight_config(self) -> Dict[str, Any]:
        return self.config.get('highlight_detection', {})

    def get_output_config(self) -> Dict[str, Any]:
        return self.config.get('output', {})

    def get_audio_config(self) -> Dict[str, Any]:
        return self.config.get('audio', {})

    def get_performance_config(self) -> Dict[str, Any]:
        return self.config.get('performance', {})