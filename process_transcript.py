#!/usr/bin/env python3
"""
純粋関数ベース文字起こしプロセス

Input: 
  - data/stage1_audio/{video_id}_audio.wav
  - data/stage1_audio/{video_id}_audio_meta.json
Output:
  - data/stage2_transcript/{video_id}_raw.json
  - data/stage2_transcript/{video_id}_cleaned.json

Usage:
  python process_transcript.py --input data/stage1_audio/video_audio.wav --output data/stage2_transcript/ --config config/transcription.yaml

純粋関数設計:
- 同じ入力に対して同じ出力
- 副作用なし（ファイル作成以外）
- キャッシュ機能による高速パラメーター調整対応
"""

import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
import click
import yaml

# プロジェクトルートをPythonパスに追加
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logging_config import setup_transcription_logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionConfig:
    """文字起こし設定（イミュータブル）"""
    model_size: str = "large-v3"
    language: str = "ja"
    temperature: float = 0.0
    initial_prompt: str = "以下は日本語の音声です。技術的な話題、プログラミング、エンジニアリングに関する内容を含みます。"
    word_timestamps: bool = True
    condition_on_previous_text: bool = True
    device: str = "auto"
    compute_type: str = "float32"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TranscriptionConfig':
        """辞書から設定を作成"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class TranscriptionSegment:
    """文字起こしセグメント（イミュータブル）"""
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]]
    confidence: Optional[float] = None
    
    def duration(self) -> float:
        """セグメントの継続時間"""
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """文字起こし結果（イミュータブル）"""
    video_id: str
    input_audio_path: str
    config: TranscriptionConfig
    segments: List[TranscriptionSegment]
    language: str
    language_probability: float
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['config'] = asdict(self.config)
        return result


@dataclass
class CleanedTranscriptionResult:
    """後処理済み文字起こし結果（イミュータブル）"""
    video_id: str
    segments: List[Dict[str, Any]]
    correction_stats: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


def load_whisper_model(config: TranscriptionConfig):
    """
    Whisperモデルを読み込む純粋関数
    
    Args:
        config: 文字起こし設定
        
    Returns:
        WhisperModel: 読み込まれたモデル
    """
    try:
        from faster_whisper import WhisperModel
        
        model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type
        )
        
        logger.info(f"Whisperモデル読み込み完了: {config.model_size} on {config.device}")
        return model
        
    except ValueError as e:
        if "float16" in str(e):
            logger.warning("float16がサポートされていません。float32にフォールバックします")
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel(
                    config.model_size,
                    device=config.device,
                    compute_type="float32"
                )
                logger.info(f"Whisperモデル読み込み完了（float32）: {config.model_size}")
                return model
            except Exception as fallback_error:
                raise Exception(f"float32フォールバックも失敗: {fallback_error}") from fallback_error
        else:
            raise Exception(f"Whisperモデル読み込み失敗: {e}") from e
    except Exception as e:
        raise Exception(f"Whisperモデル読み込み失敗: {e}") from e


def transcribe_audio_pure(
    model,
    audio_path: Path,
    config: TranscriptionConfig,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[List[TranscriptionSegment], str, float]:
    """
    音声を文字起こしする純粋関数
    
    Args:
        model: Whisperモデル
        audio_path: 音声ファイルのパス
        config: 文字起こし設定
        progress_callback: 進捗コールバック
        
    Returns:
        Tuple[List[TranscriptionSegment], str, float]: (セグメント, 言語, 言語確率)
    """
    try:
        if progress_callback:
            progress_callback("音声転写中...")
        
        # Whisperで転写実行
        segments, info = model.transcribe(
            str(audio_path),
            language=config.language,
            temperature=config.temperature,
            initial_prompt=config.initial_prompt,
            word_timestamps=config.word_timestamps,
            condition_on_previous_text=config.condition_on_previous_text,
            beam_size=config.beam_size,
            best_of=config.best_of,
            patience=config.patience,
            length_penalty=config.length_penalty,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size
        )
        
        # セグメントをリストに変換
        transcription_segments = []
        total_segments = 0
        
        for segment in segments:
            words = []
            if hasattr(segment, 'words') and segment.words:
                words = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": getattr(word, 'probability', 0.0)
                    }
                    for word in segment.words
                ]
            
            transcription_segment = TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words,
                confidence=getattr(segment, 'avg_logprob', None)
            )
            
            transcription_segments.append(transcription_segment)
            total_segments += 1
            
            if progress_callback and total_segments % 100 == 0:
                progress_callback(f"転写中... {total_segments}セグメント処理済み")
        
        logger.info(f"転写完了: {len(transcription_segments)}セグメント")
        
        return transcription_segments, info.language, info.language_probability
        
    except Exception as e:
        raise Exception(f"音声転写失敗: {e}") from e


def clean_transcription_text(text: str) -> Tuple[str, List[str]]:
    """
    転写テキストを後処理する純粋関数
    
    Args:
        text: 生の転写テキスト
        
    Returns:
        tuple[str, List[str]]: (清浄化されたテキスト, 適用された修正のリスト)
    """
    corrections = []
    cleaned_text = text
    
    # 技術用語辞書
    tech_corrections = {
        # プログラミング関連
        'プログラマー': 'プログラマー',
        'エンジニア': 'エンジニア',
        'コーディング': 'コーディング',
        'デバッグ': 'デバッグ',
        'リファクタリング': 'リファクタリング',
        'デプロイ': 'デプロイ',
        'レポジトリ': 'リポジトリ',
        'コミット': 'コミット',
        'プルリクエスト': 'プルリクエスト',
        'マージ': 'マージ',
        
        # AI・機械学習関連
        'アルゴリズム': 'アルゴリズム',
        'ニューラルネットワーク': 'ニューラルネットワーク',
        'ディープラーニング': 'ディープラーニング',
        '機械学習': '機械学習',
        'データサイエンス': 'データサイエンス',
        'ビッグデータ': 'ビッグデータ',
        
        # 一般的な転写エラー修正
        'ッ': '',  # 不適切な促音削除
        '。。。': '...',  # 連続句点の修正
        '、、': '、',  # 連続読点の修正
        '  ': ' ',  # 連続スペースの修正
    }
    
    # 技術用語の修正
    for incorrect, correct in tech_corrections.items():
        if incorrect in cleaned_text and incorrect != correct:
            old_text = cleaned_text
            cleaned_text = cleaned_text.replace(incorrect, correct)
            if old_text != cleaned_text:
                corrections.append(f"{incorrect} → {correct}")
    
    # 空白の正規化
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    # 明らかに不正な文字列の除去
    if re.match(r'^[。、]+$', cleaned_text):
        cleaned_text = ""
        corrections.append("不正な文字列を除去")
    
    return cleaned_text, corrections


def clean_transcription_result(
    raw_result: TranscriptionResult
) -> CleanedTranscriptionResult:
    """
    転写結果を後処理する純粋関数
    
    Args:
        raw_result: 生の転写結果
        
    Returns:
        CleanedTranscriptionResult: 後処理済み転写結果
    """
    start_time = time.time()
    
    cleaned_segments = []
    total_corrections = 0
    correction_types = {}
    
    for segment in raw_result.segments:
        original_text = segment.text
        cleaned_text, segment_corrections = clean_transcription_text(original_text)
        
        # 修正統計の更新
        total_corrections += len(segment_corrections)
        for correction in segment_corrections:
            correction_type = correction.split(' → ')[0] if ' → ' in correction else 'other'
            correction_types[correction_type] = correction_types.get(correction_type, 0) + 1
        
        # 清浄化されたセグメントを作成
        cleaned_segment = {
            "start": segment.start,
            "end": segment.end,
            "text": cleaned_text,
            "original_text": original_text,
            "corrections": segment_corrections,
            "confidence": segment.confidence,
            "duration": segment.duration(),
            "word_count": len(cleaned_text.split()) if cleaned_text else 0
        }
        
        cleaned_segments.append(cleaned_segment)
    
    processing_time = time.time() - start_time
    
    correction_stats = {
        "total_corrections": total_corrections,
        "correction_types": correction_types,
        "segments_processed": len(cleaned_segments),
        "processing_time": processing_time
    }
    
    return CleanedTranscriptionResult(
        video_id=raw_result.video_id,
        segments=cleaned_segments,
        correction_stats=correction_stats,
        metadata={
            "based_on_raw_result": True,
            "raw_segments_count": len(raw_result.segments),
            "cleaned_segments_count": len(cleaned_segments),
            "timestamp": time.time()
        },
        processing_time=processing_time
    )


def transcription_workflow(
    input_audio_path: str,
    output_dir: str,
    config: TranscriptionConfig
) -> Tuple[TranscriptionResult, CleanedTranscriptionResult]:
    """
    文字起こしワークフローを実行する純粋関数
    
    Args:
        input_audio_path: 入力音声ファイルのパス
        output_dir: 出力ディレクトリ
        config: 文字起こし設定
        
    Returns:
        tuple[TranscriptionResult, CleanedTranscriptionResult]: (生結果, 後処理済み結果)
    """
    start_time = time.time()
    
    try:
        # パス正規化
        audio_path = Path(input_audio_path).resolve()
        output_path = Path(output_dir).resolve()
        
        # 検証
        if not audio_path.exists():
            raise ValueError(f"音声ファイルが見つかりません: {audio_path}")
        
        # 出力ディレクトリ作成
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 動画IDを取得
        video_id = audio_path.stem.replace('_audio', '')
        
        # Whisperモデル読み込み
        logger.info("Whisperモデルを読み込み中...")
        model = load_whisper_model(config)
        
        # 音声転写実行
        logger.info("音声転写を実行中...")
        
        def progress_callback(message):
            logger.info(message)
        
        segments, language, language_prob = transcribe_audio_pure(
            model, audio_path, config, progress_callback
        )
        
        # メタデータ作成
        processing_time = time.time() - start_time
        metadata = {
            "model_size": config.model_size,
            "audio_file": str(audio_path),
            "audio_duration": segments[-1].end if segments else 0.0,
            "language": language,
            "language_probability": language_prob,
            "processing_time": processing_time,
            "segments_count": len(segments),
            "total_duration": segments[-1].end if segments else 0.0,
            "word_count": sum(len(s.words) for s in segments),
            "timestamp": time.time()
        }
        
        # 生の転写結果を作成
        raw_result = TranscriptionResult(
            video_id=video_id,
            input_audio_path=str(audio_path),
            config=config,
            segments=segments,
            language=language,
            language_probability=language_prob,
            metadata=metadata,
            processing_time=processing_time,
            success=True
        )
        
        # 後処理実行
        logger.info("転写結果を後処理中...")
        cleaned_result = clean_transcription_result(raw_result)
        
        logger.info(f"文字起こしワークフロー完了: {processing_time:.2f}秒")
        
        return raw_result, cleaned_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"文字起こしワークフローエラー: {error_msg}")
        
        # エラー結果を作成
        error_result = TranscriptionResult(
            video_id=Path(input_audio_path).stem.replace('_audio', ''),
            input_audio_path=str(input_audio_path),
            config=config,
            segments=[],
            language="",
            language_probability=0.0,
            metadata={},
            processing_time=processing_time,
            success=False,
            error_message=error_msg
        )
        
        # 空の後処理結果
        empty_cleaned = CleanedTranscriptionResult(
            video_id=error_result.video_id,
            segments=[],
            correction_stats={},
            metadata={},
            processing_time=0.0
        )
        
        return error_result, empty_cleaned


def save_transcription_results(
    raw_result: TranscriptionResult,
    cleaned_result: CleanedTranscriptionResult,
    output_dir: Path
) -> None:
    """
    文字起こし結果をJSONファイルに保存する純粋関数
    
    Args:
        raw_result: 生の転写結果
        cleaned_result: 後処理済み転写結果
        output_dir: 出力ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生結果の保存
    raw_file_path = output_dir / f"{raw_result.video_id}_raw.json"
    with open(raw_file_path, 'w', encoding='utf-8') as f:
        json.dump(raw_result.to_dict(), f, ensure_ascii=False, indent=2)
    
    # 後処理済み結果の保存
    cleaned_file_path = output_dir / f"{cleaned_result.video_id}_cleaned.json"
    with open(cleaned_file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_result.to_dict(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"文字起こし結果を保存: {raw_file_path}, {cleaned_file_path}")


def load_config(config_path: str) -> TranscriptionConfig:
    """
    設定ファイルを読み込む純粋関数
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        TranscriptionConfig: 文字起こし設定
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        return TranscriptionConfig()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() == '.yaml':
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # transcription設定セクションを取得
        transcription_config = config_dict.get('transcription', {})
        return TranscriptionConfig.from_dict(transcription_config)
        
    except Exception as e:
        logger.warning(f"設定ファイル読み込みエラー: {e}")
        return TranscriptionConfig()


@click.command()
@click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True),
              help='入力音声ファイルのパス')
@click.option('--output', '-o', 'output_dir', default='data/stage2_transcript',
              help='出力ディレクトリ (デフォルト: data/stage2_transcript)')
@click.option('--config', '-c', 'config_path', default='config/transcription.yaml',
              help='設定ファイルのパス')
@click.option('--verbose', '-v', is_flag=True, help='詳細ログ出力')
def main(input_path: str, output_dir: str, config_path: str, verbose: bool):
    """
    純粋関数ベース文字起こしプロセス
    
    INPUT_PATH: 処理する音声ファイルのパス
    """
    # 動画IDとモデル名を抽出してログ設定
    input_file = Path(input_path)
    video_id = input_file.stem.replace('_audio', '').replace('_clean', '').replace('_10min', '')
    
    # 設定からモデル名を推定
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        model_name = config_data.get('transcription', {}).get('model_size', 'unknown')
    except:
        model_name = 'unknown'
    
    # 統一ログ設定
    log_level = logging.DEBUG if verbose else logging.INFO
    global logger
    logger = setup_transcription_logging(model_name, video_id, log_level)
    
    logger.info("=== 文字起こしプロセス開始 ===")
    logger.info(f"入力: {input_path}")
    logger.info(f"出力: {output_dir}")
    logger.info(f"設定: {config_path}")
    
    try:
        # 設定読み込み
        config = load_config(config_path)
        logger.info(f"文字起こし設定: {asdict(config)}")
        
        # 文字起こし実行
        raw_result, cleaned_result = transcription_workflow(input_path, output_dir, config)
        
        if raw_result.success:
            # 結果保存
            save_transcription_results(raw_result, cleaned_result, Path(output_dir))
            
            logger.info("=== 文字起こしプロセス完了 ===")
            logger.info(f"処理時間: {raw_result.processing_time:.2f}秒")
            logger.info(f"検出セグメント数: {len(raw_result.segments)}")
            logger.info(f"言語: {raw_result.language} (確率: {raw_result.language_probability:.3f})")
            logger.info(f"修正適用数: {cleaned_result.correction_stats.get('total_corrections', 0)}")
            
            click.echo(f"✅ 文字起こし成功: {len(raw_result.segments)}セグメント")
            
        else:
            logger.error(f"文字起こし失敗: {raw_result.error_message}")
            click.echo(f"❌ 文字起こし失敗: {raw_result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        click.echo(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()