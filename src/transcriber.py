"""
文字起こしモジュール

関数型プログラミングの原則に従って実装:
- 純粋関数（副作用なし）
- イミュータブルなデータ
- 関数の合成
- 明示的なエラーハンドリング
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionSegment:
    """イミュータブルな転写セグメント"""
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]]
    confidence: Optional[float] = None


@dataclass(frozen=True)
class TranscriptionResult:
    """イミュータブルな転写結果"""
    segments: List[TranscriptionSegment]
    language: str
    metadata: Dict[str, Any]
    full_text: str


def create_whisper_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Whisper設定を作成する純粋関数
    
    Args:
        base_config: 基本設定辞書
        
    Returns:
        Dict[str, Any]: 完全なWhisper設定
    """
    # デフォルト設定
    default_config = {
        "model_size": "large-v3",
        "language": "ja",
        "temperature": 0.0,
        "initial_prompt": "以下は日本語の音声です。",
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "device": "auto",
        "compute_type": "float16"
    }
    
    # ベース設定をマージ
    merged_config = {**default_config, **base_config}
    
    # モデルサイズに基づくデバイス選択ロジック
    model_size = merged_config["model_size"]
    if merged_config["device"] == "auto":
        if model_size in ["tiny", "base"]:
            merged_config["device"] = "cpu"
        else:
            # GPU利用可能性をチェック（簡易版）
            merged_config["device"] = "auto"
    
    return merged_config


def load_whisper_model(config: Dict[str, Any]):
    """
    Whisperモデルを読み込む関数
    
    Args:
        config: Whisper設定
        
    Returns:
        WhisperModel: 読み込まれたモデル
        
    Raises:
        Exception: モデル読み込みに失敗した場合
    """
    try:
        # faster-whisperの実際のインポートはここで行う
        from faster_whisper import WhisperModel
        
        model_size = config["model_size"]
        device = config["device"]
        compute_type = config.get("compute_type", "float32")  # デフォルトをfloat32に変更
        
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        
        logger.info(f"Whisperモデル読み込み完了: {model_size} on {device}")
        return model
        
    except ValueError as e:
        if "float16" in str(e):
            # float16がサポートされていない場合はfloat32にフォールバック
            logger.warning(f"float16がサポートされていません。float32にフォールバックします: {e}")
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel(
                    config["model_size"],
                    device=config["device"],
                    compute_type="float32"
                )
                logger.info(f"Whisperモデル読み込み完了（float32）: {config['model_size']} on {config['device']}")
                return model
            except Exception as fallback_error:
                error_msg = f"float32フォールバックも失敗しました: {fallback_error}"
                logger.error(error_msg)
                raise Exception(error_msg) from fallback_error
        else:
            error_msg = f"Whisperモデルの読み込みに失敗しました: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        
    except Exception as e:
        error_msg = f"Whisperモデルの読み込みに失敗しました: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def transcribe_audio_segments(
    model, 
    audio_path: Path, 
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    音声セグメントを文字起こしする関数
    
    Args:
        model: Whisperモデル
        audio_path: 音声ファイルのパス
        config: 転写設定
        progress_callback: 進捗コールバック関数
        
    Returns:
        Dict[str, Any]: 生の転写結果
    """
    try:
        logger.info(f"文字起こし開始: {audio_path}")
        start_time = time.time()
        
        # Whisperで転写実行
        segments, info = model.transcribe(
            str(audio_path),
            language=config.get("language"),
            temperature=config.get("temperature", 0.0),
            initial_prompt=config.get("initial_prompt"),
            word_timestamps=config.get("word_timestamps", True),
            condition_on_previous_text=config.get("condition_on_previous_text", False)
        )
        
        # セグメントを処理
        processed_segments = []
        segments_list = list(segments)  # ジェネレータをリストに変換
        total_segments = len(segments_list)
        
        for i, segment in enumerate(segments_list):
            # 進捗更新
            if progress_callback:
                progress_callback(i + 1, total_segments, f"セグメント {i + 1}/{total_segments} 処理中")
            
            # 単語データの処理
            words_data = []
            if hasattr(segment, 'words') and segment.words:
                words_data = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": getattr(word, 'probability', 0.0)
                    }
                    for word in segment.words
                ]
            
            processed_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words_data
            })
        
        processing_time = time.time() - start_time
        logger.info(f"文字起こし完了: {total_segments}セグメント, {processing_time:.2f}秒")
        
        return {
            "segments": processed_segments,
            "language": info.language,
            "language_probability": getattr(info, 'language_probability', 0.0),
            "processing_time": processing_time
        }
        
    except Exception as e:
        error_msg = f"文字起こし処理に失敗しました: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def extract_word_timestamps(segments_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    セグメントから単語レベルタイムスタンプを抽出する純粋関数
    
    Args:
        segments_data: セグメントデータのリスト
        
    Returns:
        List[Dict[str, Any]]: 単語タイムスタンプのリスト
    """
    word_timestamps = []
    
    for segment_id, segment in enumerate(segments_data):
        words = segment.get("words", [])
        for word_data in words:
            word_timestamps.append({
                **word_data,
                "segment_id": segment_id
            })
    
    return word_timestamps


def format_transcription_result(
    raw_data: Dict[str, Any], 
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    転写結果をフォーマットする純粋関数
    
    Args:
        raw_data: 生の転写データ
        metadata: メタデータ
        
    Returns:
        Dict[str, Any]: フォーマットされた結果
    """
    segments = raw_data.get("segments", [])
    
    # 全文テキストを結合
    full_text = " ".join(segment["text"] for segment in segments if segment["text"].strip())
    
    # メタデータを拡張
    enhanced_metadata = {
        **metadata,
        "language": raw_data.get("language", "unknown"),
        "language_probability": raw_data.get("language_probability", 0.0),
        "processing_time": raw_data.get("processing_time", 0.0),
        "segments_count": len(segments),
        "total_duration": segments[-1]["end"] if segments else 0.0,
        "word_count": sum(len(segment.get("words", [])) for segment in segments)
    }
    
    return {
        "metadata": enhanced_metadata,
        "segments": segments,
        "word_timestamps": extract_word_timestamps(segments),
        "full_text": full_text
    }


def update_progress(
    callback: Optional[Callable[[int, int, str], None]], 
    current: int, 
    total: int, 
    message: str
) -> None:
    """
    進捗を更新する関数
    
    Args:
        callback: 進捗コールバック関数
        current: 現在の進捗
        total: 全体の数
        message: 進捗メッセージ
    """
    if callback:
        callback(current, total, message)


def calculate_audio_chunks(
    audio_info: Dict[str, Any], 
    chunk_size: float
) -> List[Dict[str, Any]]:
    """
    音声チャンク分割を計算する純粋関数
    
    Args:
        audio_info: 音声ファイル情報
        chunk_size: チャンクサイズ（秒）
        
    Returns:
        List[Dict[str, Any]]: チャンク情報のリスト
    """
    duration = audio_info["duration"]
    chunks = []
    
    current_start = 0.0
    while current_start < duration:
        current_end = min(current_start + chunk_size, duration)
        chunks.append({
            "start": current_start,
            "end": current_end,
            "duration": current_end - current_start
        })
        current_start = current_end
    
    return chunks


def full_transcription_workflow(
    audio_path: Union[str, Path], 
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    文字起こしの全体ワークフローを実行する関数
    
    Args:
        audio_path: 音声ファイルのパス
        config: 転写設定
        progress_callback: 進捗コールバック関数
        
    Returns:
        Dict[str, Any]: 完全な転写結果
        
    Raises:
        ValueError: 検証エラー
        Exception: 処理エラー
    """
    audio_path = Path(audio_path)
    
    # 1. 音声ファイル存在確認
    if not audio_path.exists():
        raise ValueError(f"音声ファイルが見つかりません: {audio_path}")
    
    # 2. Whisper設定作成
    whisper_config = create_whisper_config(config)
    logger.info(f"Whisper設定: {whisper_config['model_size']} ({whisper_config['language']})")
    
    # 3. モデル読み込み
    update_progress(progress_callback, 1, 4, "Whisperモデル読み込み中...")
    model = load_whisper_model(whisper_config)
    
    # 4. 文字起こし実行
    update_progress(progress_callback, 2, 4, "音声文字起こし実行中...")
    raw_result = transcribe_audio_segments(model, audio_path, whisper_config, progress_callback)
    
    # 5. 結果フォーマット
    update_progress(progress_callback, 3, 4, "結果フォーマット中...")
    metadata = {
        "model_size": whisper_config["model_size"],
        "audio_file": str(audio_path),
        "audio_duration": 0.0  # 実際の実装では音声ファイルから取得
    }
    
    formatted_result = format_transcription_result(raw_result, metadata)
    
    # 6. 完了
    update_progress(progress_callback, 4, 4, "文字起こし完了")
    logger.info("文字起こしワークフロー完了")
    
    return formatted_result


# 関数合成のヘルパー
def compose_transcription_pipeline(*functions):
    """
    文字起こし関数を合成してパイプラインを作成する高階関数
    """
    def pipeline(initial_value):
        result = initial_value
        for func in functions:
            result = func(result)
        return result
    return pipeline