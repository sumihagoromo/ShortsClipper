#!/usr/bin/env python3
"""
音響特徴ベース音声境界検出モジュール

音楽とマイクテスト・会話を音響特徴で判別:
1. 音量変化パターン分析
2. 周波数スペクトラム分析  
3. 音響エネルギー分布
4. 無音区間検出
5. 音楽特有の周期性検出

関数型プログラミング原則に従って実装
"""

import numpy as np
import librosa
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class AudioFeaturesConfig:
    """音響特徴検出設定"""
    # 基本設定
    sample_rate: int = 16000
    segment_duration: int = 10  # 分析セグメント長（秒）
    overlap_duration: int = 2   # オーバーラップ時間（秒）
    
    # 音量分析
    volume_threshold_db: float = -30.0  # 音量閾値（dB）
    silence_threshold_db: float = -40.0  # 無音閾値（dB）
    
    # 音楽検出閾値
    music_energy_threshold: float = 0.6    # 音楽エネルギー閾値
    harmonic_ratio_threshold: float = 0.7  # 調和波成分比閾値
    tempo_stability_threshold: float = 0.8 # テンポ安定性閾値
    
    # 音声検出閾値
    speech_energy_threshold: float = 0.3   # 音声エネルギー閾値
    spectral_contrast_threshold: float = 15.0  # スペクトラル対比閾値
    
    # フィルタリング
    min_speech_duration: int = 5    # 最小音声継続時間（秒）
    max_music_gap: int = 3         # 音楽間の最大ギャップ（秒）
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AudioFeaturesConfig':
        """辞書から設定を作成"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class AudioSegmentFeatures:
    """音響セグメントの特徴量"""
    start_time: float
    end_time: float
    
    # 音量特徴
    rms_energy: float
    peak_volume_db: float
    avg_volume_db: float
    volume_variance: float
    
    # 周波数特徴
    spectral_centroid: float
    spectral_contrast: float
    spectral_rolloff: float
    mfcc_features: List[float]
    
    # 音楽特有特徴
    harmonic_energy: float
    percussive_energy: float
    tempo_estimate: float
    beat_strength: float
    
    # 判定結果
    is_music: bool
    is_speech: bool
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AudioBoundaryResult:
    """音響分析による境界検出結果"""
    audio_file_path: str
    detected_speech_start: Optional[float]
    detection_method: str
    confidence_score: float
    segments_analyzed: List[AudioSegmentFeatures]
    music_segments: List[Tuple[float, float]]  # 音楽区間
    speech_segments: List[Tuple[float, float]]  # 音声区間
    processing_time: float
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_audio_segment(
    audio_path: Union[str, Path],
    start_time: float,
    duration: float,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    音声ファイルから指定区間を読み込む純粋関数
    
    Args:
        audio_path: 音声ファイルのパス
        start_time: 開始時間（秒）
        duration: 継続時間（秒）
        sample_rate: サンプリングレート
        
    Returns:
        np.ndarray: 音声データ
    """
    try:
        y, sr = librosa.load(
            str(audio_path),
            sr=sample_rate,
            offset=start_time,
            duration=duration
        )
        return y
    except Exception as e:
        logger.warning(f"音声読み込みエラー ({start_time}s): {e}")
        return np.array([])


def extract_volume_features(audio_data: np.ndarray) -> Dict[str, float]:
    """
    音量特徴を抽出する純粋関数
    
    Args:
        audio_data: 音声データ
        
    Returns:
        Dict[str, float]: 音量特徴辞書
    """
    if len(audio_data) == 0:
        return {
            'rms_energy': 0.0,
            'peak_volume_db': -80.0,
            'avg_volume_db': -80.0,
            'volume_variance': 0.0
        }
    
    # RMSエネルギー
    rms_energy = float(np.sqrt(np.mean(audio_data**2)))
    
    # ピーク音量（dB）
    peak_volume = np.max(np.abs(audio_data))
    peak_volume_db = 20 * np.log10(peak_volume + 1e-10)
    
    # 平均音量（dB）  
    avg_volume = np.mean(np.abs(audio_data))
    avg_volume_db = 20 * np.log10(avg_volume + 1e-10)
    
    # 音量分散
    volume_variance = float(np.var(np.abs(audio_data)))
    
    return {
        'rms_energy': rms_energy,
        'peak_volume_db': float(peak_volume_db),
        'avg_volume_db': float(avg_volume_db),
        'volume_variance': volume_variance
    }


def extract_spectral_features(
    audio_data: np.ndarray, 
    sample_rate: int = 16000
) -> Dict[str, Any]:
    """
    スペクトラル特徴を抽出する純粋関数
    
    Args:
        audio_data: 音声データ
        sample_rate: サンプリングレート
        
    Returns:
        Dict[str, Any]: スペクトラル特徴辞書
    """
    if len(audio_data) == 0:
        return {
            'spectral_centroid': 0.0,
            'spectral_contrast': 0.0,
            'spectral_rolloff': 0.0,
            'mfcc_features': [0.0] * 13
        }
    
    try:
        # スペクトラル重心
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate
        )
        spectral_centroid_mean = float(np.mean(spectral_centroid))
        
        # スペクトラル対比
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio_data, sr=sample_rate
        )
        spectral_contrast_mean = float(np.mean(spectral_contrast))
        
        # スペクトラルロールオフ
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate
        )
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))
        
        # MFCC特徴
        mfcc = librosa.feature.mfcc(
            y=audio_data, sr=sample_rate, n_mfcc=13
        )
        mfcc_features = [float(np.mean(mfcc[i])) for i in range(13)]
        
        return {
            'spectral_centroid': spectral_centroid_mean,
            'spectral_contrast': spectral_contrast_mean,
            'spectral_rolloff': spectral_rolloff_mean,
            'mfcc_features': mfcc_features
        }
        
    except Exception as e:
        logger.warning(f"スペクトラル特徴抽出エラー: {e}")
        return {
            'spectral_centroid': 0.0,
            'spectral_contrast': 0.0,
            'spectral_rolloff': 0.0,
            'mfcc_features': [0.0] * 13
        }


def extract_music_features(
    audio_data: np.ndarray,
    sample_rate: int = 16000
) -> Dict[str, float]:
    """
    音楽特有の特徴を抽出する純粋関数
    
    Args:
        audio_data: 音声データ
        sample_rate: サンプリングレート
        
    Returns:
        Dict[str, float]: 音楽特徴辞書
    """
    if len(audio_data) == 0:
        return {
            'harmonic_energy': 0.0,
            'percussive_energy': 0.0,
            'tempo_estimate': 0.0,
            'beat_strength': 0.0
        }
    
    try:
        # 調和波成分と打楽器成分の分離
        y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
        
        # 調和波エネルギー
        harmonic_energy = float(np.sqrt(np.mean(y_harmonic**2)))
        
        # 打楽器エネルギー
        percussive_energy = float(np.sqrt(np.mean(y_percussive**2)))
        
        # テンポ推定
        tempo, beats = librosa.beat.beat_track(
            y=audio_data, sr=sample_rate
        )
        tempo_estimate = float(tempo)
        
        # ビート強度
        beat_strength = float(np.mean(librosa.onset.onset_strength(
            y=audio_data, sr=sample_rate
        )))
        
        return {
            'harmonic_energy': harmonic_energy,
            'percussive_energy': percussive_energy,
            'tempo_estimate': tempo_estimate,
            'beat_strength': beat_strength
        }
        
    except Exception as e:
        logger.warning(f"音楽特徴抽出エラー: {e}")
        return {
            'harmonic_energy': 0.0,
            'percussive_energy': 0.0,
            'tempo_estimate': 0.0,
            'beat_strength': 0.0
        }


def classify_audio_segment(
    features: AudioSegmentFeatures,
    config: AudioFeaturesConfig
) -> Tuple[bool, bool, float]:
    """
    音響特徴から音楽/音声を分類する純粋関数
    
    Args:
        features: 音響特徴
        config: 検出設定
        
    Returns:
        Tuple[bool, bool, float]: (is_music, is_speech, confidence)
    """
    # 音楽判定スコア
    music_score = 0.0
    
    # 調和波成分が多い（音楽特徴）
    if features.harmonic_energy > config.music_energy_threshold:
        music_score += 0.3
    
    # テンポが検出される（音楽特徴）
    if features.tempo_estimate > 60 and features.tempo_estimate < 180:
        music_score += 0.2
    
    # ビート強度が高い（音楽特徴）
    if features.beat_strength > 0.5:
        music_score += 0.2
    
    # 音量が安定している（音楽特徴）
    if features.volume_variance < 0.01:
        music_score += 0.1
    
    # スペクトラル対比が低い（音楽特徴）
    if features.spectral_contrast < config.spectral_contrast_threshold:
        music_score += 0.2
    
    # 音声判定スコア
    speech_score = 0.0
    
    # スペクトラル対比が高い（音声特徴）
    if features.spectral_contrast > config.spectral_contrast_threshold:
        speech_score += 0.5
    
    # 音量変動が大きい（音声特徴）
    if features.volume_variance > 0.0005:  # 閾値を下げる
        speech_score += 0.2
    
    # 中域周波数に集中（音声特徴）
    if 1000 < features.spectral_centroid < 4000:
        speech_score += 0.3
    
    # 調和波エネルギーが適度（音声特徴）
    if 0.01 < features.harmonic_energy < 0.1:
        speech_score += 0.2
    
    # 音量レベルが適切（音声特徴）
    if features.avg_volume_db > -35:
        speech_score += 0.1
    
    # 判定
    is_music = music_score > 0.6
    is_speech = speech_score > 0.5  # 閾値を下げて検出しやすくする
    confidence = max(music_score, speech_score)
    
    # 両方該当する場合（マイクテスト中の音楽など）
    if is_music and is_speech:
        # より高いスコアを採用
        if music_score > speech_score:
            is_speech = False
        else:
            is_music = False
    
    return is_music, is_speech, confidence


def analyze_audio_segment(
    audio_path: Union[str, Path],
    start_time: float,
    duration: float,
    config: AudioFeaturesConfig
) -> AudioSegmentFeatures:
    """
    音響セグメントを分析する純粋関数
    
    Args:
        audio_path: 音声ファイルのパス
        start_time: 開始時間
        duration: 継続時間
        config: 検出設定
        
    Returns:
        AudioSegmentFeatures: セグメント特徴
    """
    # 音声データ読み込み
    audio_data = load_audio_segment(
        audio_path, start_time, duration, config.sample_rate
    )
    
    # 各特徴抽出
    volume_features = extract_volume_features(audio_data)
    spectral_features = extract_spectral_features(audio_data, config.sample_rate)
    music_features = extract_music_features(audio_data, config.sample_rate)
    
    # セグメント特徴作成
    features = AudioSegmentFeatures(
        start_time=start_time,
        end_time=start_time + duration,
        
        # 音量特徴
        rms_energy=volume_features['rms_energy'],
        peak_volume_db=volume_features['peak_volume_db'],
        avg_volume_db=volume_features['avg_volume_db'],
        volume_variance=volume_features['volume_variance'],
        
        # 周波数特徴
        spectral_centroid=spectral_features['spectral_centroid'],
        spectral_contrast=spectral_features['spectral_contrast'],
        spectral_rolloff=spectral_features['spectral_rolloff'],
        mfcc_features=spectral_features['mfcc_features'],
        
        # 音楽特徴
        harmonic_energy=music_features['harmonic_energy'],
        percussive_energy=music_features['percussive_energy'],
        tempo_estimate=music_features['tempo_estimate'],
        beat_strength=music_features['beat_strength'],
        
        # 初期値
        is_music=False,
        is_speech=False,
        confidence_score=0.0
    )
    
    # 分類実行
    is_music, is_speech, confidence = classify_audio_segment(features, config)
    
    features.is_music = is_music
    features.is_speech = is_speech
    features.confidence_score = confidence
    
    return features


def detect_speech_boundary_with_audio_features(
    audio_path: Union[str, Path],
    config: AudioFeaturesConfig
) -> AudioBoundaryResult:
    """
    音響特徴を使用して音声境界を検出する主要関数
    
    Args:
        audio_path: 音声ファイルのパス
        config: 検出設定
        
    Returns:
        AudioBoundaryResult: 検出結果
    """
    start_time = time.time()
    audio_file_path = str(Path(audio_path).resolve())
    
    logger.info(f"音響特徴ベース境界検出開始: {audio_file_path}")
    
    try:
        # 音声ファイル情報取得
        duration = librosa.get_duration(path=audio_file_path)
        logger.info(f"音声長: {duration:.1f}秒")
        
        # セグメント分析
        segments_analyzed = []
        current_time = 0.0
        
        while current_time < duration:
            segment_duration = min(config.segment_duration, duration - current_time)
            
            logger.debug(f"セグメント分析: {current_time:.1f}s-{current_time + segment_duration:.1f}s")
            
            segment_features = analyze_audio_segment(
                audio_path, current_time, segment_duration, config
            )
            
            segments_analyzed.append(segment_features)
            
            # 次のセグメントへ（オーバーラップ考慮）
            current_time += config.segment_duration - config.overlap_duration
        
        # 音楽区間と音声区間の特定
        music_segments = []
        speech_segments = []
        
        for segment in segments_analyzed:
            if segment.is_music:
                music_segments.append((segment.start_time, segment.end_time))
            elif segment.is_speech:
                speech_segments.append((segment.start_time, segment.end_time))
        
        # 音声開始位置の決定
        detected_speech_start = None
        confidence_score = 0.0
        
        # 最初の有意義な音声セグメントを検索
        for segment in segments_analyzed:
            if segment.is_speech and segment.confidence_score > 0.7:
                detected_speech_start = segment.start_time
                confidence_score = segment.confidence_score
                break
        
        # 連続する音声セグメントの結合
        if detected_speech_start is not None:
            # 音声開始前に短い音楽ギャップがある場合は調整
            for i, segment in enumerate(segments_analyzed):
                if segment.start_time <= detected_speech_start <= segment.end_time:
                    # 前のセグメントをチェック
                    if i > 0 and not segments_analyzed[i-1].is_music:
                        detected_speech_start = segments_analyzed[i-1].start_time
                    break
        
        processing_time = time.time() - start_time
        
        success = detected_speech_start is not None
        detection_method = "audio_features_analysis" if success else "no_speech_detected"
        
        logger.info(
            f"音響特徴検出結果: "
            f"音声開始={detected_speech_start}s, "
            f"信頼度={confidence_score:.3f}, "
            f"音楽区間数={len(music_segments)}, "
            f"音声区間数={len(speech_segments)}"
        )
        
        return AudioBoundaryResult(
            audio_file_path=audio_file_path,
            detected_speech_start=detected_speech_start,
            detection_method=detection_method,
            confidence_score=confidence_score,
            segments_analyzed=segments_analyzed,
            music_segments=music_segments,
            speech_segments=speech_segments,
            processing_time=processing_time,
            success=success
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"音響特徴検出エラー: {error_msg}")
        
        return AudioBoundaryResult(
            audio_file_path=audio_file_path,
            detected_speech_start=None,
            detection_method="error",
            confidence_score=0.0,
            segments_analyzed=[],
            music_segments=[],
            speech_segments=[],
            processing_time=processing_time,
            success=False,
            error_message=error_msg
        )


def save_audio_features_result(
    result: AudioBoundaryResult,
    output_dir: Union[str, Path]
) -> None:
    """
    音響分析結果をJSONファイルに保存する純粋関数
    
    Args:
        result: 分析結果
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ファイル名生成
    audio_name = Path(result.audio_file_path).stem
    result_file = output_path / f"{audio_name}_audio_features_detection.json"
    
    # JSON保存
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"音響分析結果を保存: {result_file}")