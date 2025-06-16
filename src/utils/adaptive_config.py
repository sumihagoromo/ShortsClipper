#!/usr/bin/env python3
"""
適応的設定管理ユーティリティ
音声の長さや特性に応じた最適設定の自動選択
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class ProcessingProfile:
    """処理プロファイル"""
    name: str
    description: str
    config_path: str
    max_duration: float  # 最大適用時間（秒）
    parallel_workers: int
    memory_efficient: bool
    quality_priority: bool


class AdaptiveConfigManager:
    """適応的設定管理クラス"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.profiles = self._initialize_profiles()
    
    def _initialize_profiles(self) -> Dict[str, ProcessingProfile]:
        """処理プロファイルを初期化"""
        profiles = {
            "direct": ProcessingProfile(
                name="direct",
                description="直接処理（短時間音声用）",
                config_path="config/transcription_base_no_repeat.yaml",
                max_duration=10 * 60,  # 10分
                parallel_workers=1,
                memory_efficient=True,
                quality_priority=True
            ),
            "batch_standard": ProcessingProfile(
                name="batch_standard",
                description="標準バッチ処理（中時間音声用）",
                config_path="config/transcription_base_no_repeat.yaml",
                max_duration=60 * 60,  # 1時間
                parallel_workers=3,
                memory_efficient=True,
                quality_priority=True
            ),
            "batch_fast": ProcessingProfile(
                name="batch_fast",
                description="高速バッチ処理（長時間音声用）",
                config_path="config/transcription_base_no_repeat.yaml",
                max_duration=float('inf'),
                parallel_workers=4,
                memory_efficient=True,
                quality_priority=False
            ),
            "high_quality": ProcessingProfile(
                name="high_quality",
                description="高品質処理（重要音声用）",
                config_path="config/transcription_large_v3.yaml",
                max_duration=30 * 60,  # 30分
                parallel_workers=2,
                memory_efficient=False,
                quality_priority=True
            )
        }
        
        self.logger.info(f"処理プロファイル初期化: {len(profiles)}種類")
        return profiles
    
    def select_optimal_profile(self, audio_duration: float, 
                             force_profile: Optional[str] = None,
                             quality_priority: bool = False) -> ProcessingProfile:
        """音声長と要件に基づく最適プロファイル選択"""
        if force_profile:
            if force_profile in self.profiles:
                profile = self.profiles[force_profile]
                self.logger.info(f"強制プロファイル選択: {profile.name}")
                return profile
            else:
                self.logger.warning(f"不明なプロファイル: {force_profile}, 自動選択に切り替え")
        
        # 品質優先モード
        if quality_priority and audio_duration <= 30 * 60:
            profile = self.profiles["high_quality"]
            self.logger.info(f"品質優先プロファイル選択: {profile.name}")
            return profile
        
        # 時間ベースの自動選択
        for profile in self.profiles.values():
            if audio_duration <= profile.max_duration:
                self.logger.info(
                    f"自動プロファイル選択: {profile.name} "
                    f"(音声{audio_duration/60:.1f}分 <= 上限{profile.max_duration/60:.1f}分)"
                )
                return profile
        
        # デフォルト（最も制限の緩いプロファイル）
        profile = self.profiles["batch_fast"]
        self.logger.info(f"デフォルトプロファイル選択: {profile.name}")
        return profile
    
    def load_transcription_config(self, profile: ProcessingProfile) -> Dict:
        """プロファイルに基づく転写設定を読み込み"""
        config_path = Path(profile.config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # プロファイル固有の調整
            config = self._adjust_config_for_profile(config, profile)
            
            self.logger.info(f"設定読み込み完了: {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"設定読み込みエラー {config_path}: {e}")
            raise
    
    def _adjust_config_for_profile(self, config: Dict, profile: ProcessingProfile) -> Dict:
        """プロファイルに基づく設定調整"""
        adjusted_config = config.copy()
        
        # パフォーマンス設定の調整
        if 'performance' not in adjusted_config:
            adjusted_config['performance'] = {}
        
        adjusted_config['performance']['num_workers'] = profile.parallel_workers
        adjusted_config['performance']['memory_efficient'] = profile.memory_efficient
        
        # 品質優先の場合の調整
        if profile.quality_priority and 'transcription' in adjusted_config:
            transcription = adjusted_config['transcription']
            
            # より慎重な設定に調整
            if transcription.get('beam_size', 1) == 1:
                transcription['beam_size'] = 3
            if transcription.get('best_of', 1) == 1:
                transcription['best_of'] = 3
        
        # 高速処理の場合の調整
        if not profile.quality_priority and 'transcription' in adjusted_config:
            transcription = adjusted_config['transcription']
            
            # より高速な設定に調整
            transcription['beam_size'] = 1
            transcription['best_of'] = 1
            transcription['condition_on_previous_text'] = False
        
        self.logger.debug(f"プロファイル調整完了: {profile.name}")
        return adjusted_config
    
    def estimate_processing_time(self, audio_duration: float, 
                               profile: ProcessingProfile) -> Tuple[float, float]:
        """処理時間を推定"""
        # 基本的な処理時間係数（実測値ベース）
        base_coefficients = {
            "direct": 6.5,      # 6.5倍（baseモデル直接処理）
            "batch_standard": 5.0,  # 5.0倍（並行処理効果）
            "batch_fast": 4.0,      # 4.0倍（高速設定+並行処理）
            "high_quality": 10.0    # 10.0倍（large-v3モデル）
        }
        
        coefficient = base_coefficients.get(profile.name, 6.5)
        
        # 並行処理による効率改善
        if profile.parallel_workers > 1:
            parallel_efficiency = min(0.8, 0.3 + 0.2 * profile.parallel_workers)
            coefficient *= parallel_efficiency
        
        estimated_time = audio_duration * coefficient
        
        # 信頼区間（±30%）
        lower_bound = estimated_time * 0.7
        upper_bound = estimated_time * 1.3
        
        self.logger.info(
            f"処理時間推定: {estimated_time/60:.1f}分 "
            f"({lower_bound/60:.1f}-{upper_bound/60:.1f}分)"
        )
        
        return estimated_time, upper_bound
    
    def get_split_strategy(self, audio_duration: float, 
                          profile: ProcessingProfile) -> Dict:
        """分割戦略を取得"""
        if profile.name == "direct":
            return {
                "should_split": False,
                "segment_duration": audio_duration,
                "overlap_seconds": 0.0,
                "strategy": "no_split"
            }
        
        # 分割サイズの決定
        if audio_duration <= 30 * 60:  # 30分以下
            segment_duration = 10 * 60  # 10分分割
        elif audio_duration <= 120 * 60:  # 2時間以下
            segment_duration = 15 * 60  # 15分分割
        else:  # 2時間超
            segment_duration = 20 * 60  # 20分分割
        
        # オーバーラップ時間（分割境界での文脈保持）
        overlap_seconds = 5.0 if profile.quality_priority else 3.0
        
        strategy = {
            "should_split": True,
            "segment_duration": segment_duration,
            "overlap_seconds": overlap_seconds,
            "strategy": f"{segment_duration/60:.0f}min_split"
        }
        
        self.logger.info(
            f"分割戦略: {segment_duration/60:.0f}分単位分割 "
            f"(オーバーラップ{overlap_seconds}秒)"
        )
        
        return strategy
    
    def create_adaptive_config(self, audio_duration: float,
                             force_profile: Optional[str] = None,
                             quality_priority: bool = False) -> Dict:
        """音声に最適化された完全な設定を作成"""
        # プロファイル選択
        profile = self.select_optimal_profile(
            audio_duration, force_profile, quality_priority
        )
        
        # 基本設定読み込み
        transcription_config = self.load_transcription_config(profile)
        
        # 分割戦略取得
        split_strategy = self.get_split_strategy(audio_duration, profile)
        
        # 処理時間推定
        estimated_time, max_time = self.estimate_processing_time(audio_duration, profile)
        
        # 統合設定作成
        adaptive_config = {
            "profile": {
                "name": profile.name,
                "description": profile.description,
                "selected_reason": f"audio_duration_{audio_duration/60:.1f}min"
            },
            "transcription": transcription_config,
            "splitting": split_strategy,
            "processing": {
                "parallel_workers": profile.parallel_workers,
                "estimated_time_seconds": estimated_time,
                "max_time_seconds": max_time,
                "memory_efficient": profile.memory_efficient,
                "quality_priority": profile.quality_priority
            },
            "audio_info": {
                "duration": audio_duration,
                "duration_minutes": audio_duration / 60
            }
        }
        
        self.logger.info(f"適応的設定作成完了: {profile.name}プロファイル")
        return adaptive_config


def main():
    """テスト用メイン関数"""
    import sys
    from src.utils.logging_config import setup_process_logging
    
    logger = setup_process_logging("adaptive_config", "processing")
    manager = AdaptiveConfigManager(logger)
    
    # テストケース
    test_durations = [
        5 * 60,    # 5分
        15 * 60,   # 15分
        45 * 60,   # 45分
        90 * 60,   # 90分
        180 * 60   # 3時間
    ]
    
    print("適応的設定テスト:")
    for duration in test_durations:
        print(f"\n=== {duration/60:.0f}分音声 ===")
        
        config = manager.create_adaptive_config(duration)
        profile = config['profile']
        processing = config['processing']
        
        print(f"選択プロファイル: {profile['name']}")
        print(f"推定処理時間: {processing['estimated_time_seconds']/60:.1f}分")
        print(f"並行度: {processing['parallel_workers']}")
        print(f"分割戦略: {config['splitting']['strategy']}")


if __name__ == "__main__":
    main()