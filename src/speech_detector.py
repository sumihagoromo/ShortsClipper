#!/usr/bin/env python3
"""
音声境界自動検出モジュール

音楽部分と音声部分の境界を自動検出する機能を提供する。
複数の検出手法を組み合わせて信頼性の高い境界検出を実現。

検出手法:
1. Whisper音声認識による言語確率ベース検出
2. 音声活動検出（VAD）による音声セグメント分析
3. 音響特徴分析による音楽/音声判別

関数型プログラミング原則:
- 純粋関数として実装
- イミュータブルなデータ構造
- 副作用の最小化
"""

import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
import time

# プロジェクトルートをPythonパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class SpeechDetectionConfig:
    """音声検出設定"""
    # 基本設定
    check_interval: int = 120  # チェック間隔（秒）
    max_check_duration: int = 600  # 最大チェック時間（秒、10分）
    segment_duration: int = 120  # セグメント長（秒）
    
    # Whisper設定
    model_size: str = "base"
    language: str = "ja"
    temperature: float = 0.0
    beam_size: int = 1
    
    # 判定閾値
    language_probability_threshold: float = 0.8  # 言語確率閾値（厳格化）
    minimum_segments_count: int = 5  # 最小セグメント数（厳格化）
    meaningful_text_min_length: int = 10  # 有意義なテキスト最小長（厳格化）
    
    # フォールバック設定
    default_skip_seconds: int = 180  # デフォルトスキップ時間（3分）
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SpeechDetectionConfig':
        """辞書から設定を作成"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class SpeechSegmentInfo:
    """音声セグメント情報"""
    start_time: int
    end_time: int
    segment_count: int
    language_probability: float
    meaningful_segments: int
    sample_text: List[str]
    confidence_score: float


@dataclass
class SpeechDetectionResult:
    """音声検出結果"""
    audio_file_path: str
    detected_speech_start: Optional[int]
    detection_method: str
    confidence_score: float
    segments_analyzed: List[SpeechSegmentInfo]
    processing_time: float
    success: bool
    fallback_used: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


def create_temp_audio_segment(
    audio_path: Union[str, Path], 
    start_time: int, 
    duration: int
) -> Path:
    """
    音声ファイルから指定区間の一時ファイルを作成する純粋関数
    
    Args:
        audio_path: 元音声ファイルのパス
        start_time: 開始時間（秒）
        duration: 継続時間（秒）
        
    Returns:
        Path: 一時音声ファイルのパス
        
    Raises:
        Exception: FFmpegでの処理が失敗した場合
    """
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.wav',
        prefix=f'speech_detect_{start_time}_{duration}_'
    )
    
    # ファイルディスクリプタを閉じる
    import os
    os.close(temp_fd)
    temp_file_path = Path(temp_path)
    
    try:
        subprocess.run([
            "ffmpeg", "-i", str(audio_path),
            "-ss", str(start_time), 
            "-t", str(duration),
            "-y", str(temp_file_path)
        ], capture_output=True, check=True, timeout=60)
        
        return temp_file_path
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        # 失敗時は一時ファイルを削除
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise Exception(f"音声セグメント作成に失敗: {e}")


def analyze_audio_segment_with_whisper(
    segment_path: Path,
    config: SpeechDetectionConfig
) -> SpeechSegmentInfo:
    """
    Whisperを使用して音声セグメントを分析する純粋関数
    
    Args:
        segment_path: 音声セグメントファイルのパス
        config: 音声検出設定
        
    Returns:
        SpeechSegmentInfo: セグメント分析結果
        
    Raises:
        Exception: Whisper処理が失敗した場合
    """
    try:
        # Whisperモデル初期化
        model = WhisperModel(
            config.model_size, 
            device="auto", 
            compute_type="float32"
        )
        
        # 音声認識実行
        segments, info = model.transcribe(
            str(segment_path),
            language=config.language,
            beam_size=config.beam_size,
            temperature=config.temperature,
            condition_on_previous_text=False
        )
        
        segments_list = list(segments)
        
        # 有意義なセグメントをフィルタリング
        music_patterns = [
            '【音楽】', 'music', '♪', 
            'このステージ', 'このように', 'この', 'ステージ',
            'stage', 'の', 'を', 'は', 'が', 'て', 'に'
        ]
        
        # 繰り返しパターンの検出
        repetitive_patterns = ['このステージ', 'このように', 'ステージ']
        
        meaningful_segments = []
        for seg in segments_list:
            text = seg.text.strip().lower()
            
            # 最小長チェック
            if len(text) < config.meaningful_text_min_length:
                continue
                
            # 音楽パターン除外
            if any(pattern in text for pattern in music_patterns):
                continue
                
            # 繰り返しパターンの検出（同じ単語が多数繰り返される場合は除外）
            words = text.split()
            if len(words) > 5:
                # 最も頻出する単語の出現率をチェック
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                if word_counts:
                    max_count = max(word_counts.values())
                    repetition_ratio = max_count / len(words)
                    
                    # 50%以上が同じ単語の繰り返しの場合は除外
                    if repetition_ratio > 0.5:
                        continue
            
            # 短い単語のみで構成される場合は除外
            if all(len(word) <= 2 for word in words):
                continue
                
            meaningful_segments.append(seg)
        
        # サンプルテキスト抽出
        sample_text = [seg.text.strip() for seg in segments_list[:3]]
        
        # 信頼度スコア計算
        confidence_score = calculate_segment_confidence(
            len(segments_list),
            len(meaningful_segments),
            info.language_probability,
            config
        )
        
        return SpeechSegmentInfo(
            start_time=0,  # セグメント内での相対時間
            end_time=config.segment_duration,
            segment_count=len(segments_list),
            language_probability=info.language_probability,
            meaningful_segments=len(meaningful_segments),
            sample_text=sample_text,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.warning(f"Whisper分析エラー: {e}")
        return SpeechSegmentInfo(
            start_time=0,
            end_time=config.segment_duration,
            segment_count=0,
            language_probability=0.0,
            meaningful_segments=0,
            sample_text=[],
            confidence_score=0.0
        )


def calculate_segment_confidence(
    total_segments: int,
    meaningful_segments: int,
    language_probability: float,
    config: SpeechDetectionConfig
) -> float:
    """
    セグメントの信頼度スコアを計算する純粋関数
    
    Args:
        total_segments: 総セグメント数
        meaningful_segments: 有意義なセグメント数  
        language_probability: 言語確率
        config: 検出設定
        
    Returns:
        float: 信頼度スコア（0.0-1.0）
    """
    # 基本スコア（言語確率）
    base_score = language_probability
    
    # セグメント数によるボーナス
    segment_bonus = min(meaningful_segments / config.minimum_segments_count, 1.0) * 0.2
    
    # 総セグメントに対する有意義セグメントの割合
    if total_segments > 0:
        ratio_bonus = (meaningful_segments / total_segments) * 0.1
    else:
        ratio_bonus = 0.0
    
    # 最終スコア計算
    confidence_score = min(base_score + segment_bonus + ratio_bonus, 1.0)
    
    return confidence_score


def is_speech_segment_valid(
    segment_info: SpeechSegmentInfo,
    config: SpeechDetectionConfig
) -> bool:
    """
    音声セグメントが有効かどうかを判定する純粋関数
    
    Args:
        segment_info: セグメント情報
        config: 検出設定
        
    Returns:
        bool: 有効な音声セグメントの場合True
    """
    return (
        segment_info.language_probability >= config.language_probability_threshold and
        segment_info.meaningful_segments >= config.minimum_segments_count and
        segment_info.confidence_score > 0.5
    )


def detect_speech_boundary(
    audio_file_path: Union[str, Path],
    config: SpeechDetectionConfig
) -> SpeechDetectionResult:
    """
    音声ファイルから音声境界を自動検出する主要関数
    
    Args:
        audio_file_path: 音声ファイルのパス
        config: 音声検出設定
        
    Returns:
        SpeechDetectionResult: 検出結果
    """
    start_time = time.time()
    audio_path = Path(audio_file_path)
    segments_analyzed = []
    
    logger.info(f"音声境界検出開始: {audio_path}")
    
    try:
        # 音声ファイル存在確認
        if not audio_path.exists():
            raise ValueError(f"音声ファイルが見つかりません: {audio_path}")
        
        # 検出処理実行
        for check_time in range(0, config.max_check_duration, config.check_interval):
            logger.debug(f"音声チェック: {check_time}秒位置")
            
            # 一時セグメント作成
            temp_segment = None
            try:
                temp_segment = create_temp_audio_segment(
                    audio_path, 
                    check_time, 
                    config.segment_duration
                )
                
                # Whisper分析実行
                segment_info = analyze_audio_segment_with_whisper(temp_segment, config)
                segment_info.start_time = check_time
                segment_info.end_time = check_time + config.segment_duration
                
                segments_analyzed.append(segment_info)
                
                logger.debug(
                    f"セグメント分析結果 ({check_time}s): "
                    f"セグメント数={segment_info.segment_count}, "
                    f"言語確率={segment_info.language_probability:.3f}, "
                    f"有意義セグメント={segment_info.meaningful_segments}, "
                    f"信頼度={segment_info.confidence_score:.3f}"
                )
                
                # 有効な音声セグメントが見つかった場合
                if is_speech_segment_valid(segment_info, config):
                    processing_time = time.time() - start_time
                    
                    logger.info(
                        f"音声境界検出成功: {check_time}秒位置 "
                        f"(信頼度: {segment_info.confidence_score:.3f})"
                    )
                    
                    return SpeechDetectionResult(
                        audio_file_path=str(audio_path),
                        detected_speech_start=check_time,
                        detection_method="whisper_language_probability",
                        confidence_score=segment_info.confidence_score,
                        segments_analyzed=segments_analyzed,
                        processing_time=processing_time,
                        success=True
                    )
                
            except Exception as e:
                logger.warning(f"セグメント分析エラー ({check_time}s): {e}")
                
            finally:
                # 一時ファイルクリーンアップ
                if temp_segment and temp_segment.exists():
                    temp_segment.unlink()
        
        # 音声境界が見つからなかった場合はフォールバック
        processing_time = time.time() - start_time
        
        logger.warning(
            f"音声境界検出失敗、フォールバック使用: {config.default_skip_seconds}秒"
        )
        
        return SpeechDetectionResult(
            audio_file_path=str(audio_path),
            detected_speech_start=config.default_skip_seconds,
            detection_method="fallback_default",
            confidence_score=0.0,
            segments_analyzed=segments_analyzed,
            processing_time=processing_time,
            success=True,
            fallback_used=True
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"音声境界検出エラー: {error_msg}")
        
        return SpeechDetectionResult(
            audio_file_path=str(audio_path),
            detected_speech_start=None,
            detection_method="error",
            confidence_score=0.0,
            segments_analyzed=segments_analyzed,
            processing_time=processing_time,
            success=False,
            error_message=error_msg
        )


def save_detection_result(
    result: SpeechDetectionResult, 
    output_dir: Union[str, Path]
) -> None:
    """
    検出結果をJSONファイルに保存する純粋関数
    
    Args:
        result: 検出結果
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ファイル名生成
    audio_name = Path(result.audio_file_path).stem
    result_file = output_path / f"{audio_name}_speech_detection.json"
    
    # JSON保存
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"音声検出結果を保存: {result_file}")


# 関数型プログラミング用のヘルパー関数
def create_speech_detector(config: SpeechDetectionConfig):
    """
    設定を部分適用した音声検出関数を作成する高階関数
    
    Args:
        config: 音声検出設定
        
    Returns:
        function: 音声ファイルパスを受け取る検出関数
    """
    def detector(audio_file_path: Union[str, Path]) -> SpeechDetectionResult:
        return detect_speech_boundary(audio_file_path, config)
    
    return detector


# テスト用の便利関数
def quick_speech_detection(
    audio_file_path: Union[str, Path],
    skip_seconds: int = 180,
    max_check_minutes: int = 10
) -> int:
    """
    簡単な音声検出（テスト用）
    
    Args:
        audio_file_path: 音声ファイルのパス
        skip_seconds: デフォルトスキップ時間
        max_check_minutes: 最大チェック時間（分）
        
    Returns:
        int: 検出された音声開始時刻（秒）
    """
    config = SpeechDetectionConfig(
        default_skip_seconds=skip_seconds,
        max_check_duration=max_check_minutes * 60
    )
    
    result = detect_speech_boundary(audio_file_path, config)
    
    if result.success and result.detected_speech_start is not None:
        return result.detected_speech_start
    else:
        return skip_seconds