"""
結果出力フォーマッター

文字起こし、感情分析、ハイライト検出の結果をファイルに出力する
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


def save_transcription_results(
    transcription_result: Dict[str, Any],
    output_dir: Path,
    base_filename: str
) -> List[Path]:
    """
    文字起こし結果をファイルに保存
    
    Args:
        transcription_result: 文字起こし結果
        output_dir: 出力ディレクトリ
        base_filename: ベースファイル名
        
    Returns:
        List[Path]: 作成されたファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    
    # JSON形式で保存
    json_path = output_dir / f"{base_filename}_transcription.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcription_result, f, ensure_ascii=False, indent=2)
    created_files.append(json_path)
    logger.info(f"文字起こし結果(JSON)を保存: {json_path}")
    
    # CSV形式で保存
    csv_path = output_dir / f"{base_filename}_transcription.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'duration', 'text'])
        
        for segment in transcription_result.get('segments', []):
            writer.writerow([
                segment.get('start', 0),
                segment.get('end', 0),
                segment.get('end', 0) - segment.get('start', 0),
                segment.get('text', '')
            ])
    created_files.append(csv_path)
    logger.info(f"文字起こし結果(CSV)を保存: {csv_path}")
    
    # テキスト形式で保存
    txt_path = output_dir / f"{base_filename}_transcript.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 文字起こし結果 ===\n")
        f.write(f"ファイル: {base_filename}\n")
        f.write(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"セグメント数: {len(transcription_result.get('segments', []))}\n\n")
        
        for i, segment in enumerate(transcription_result.get('segments', []), 1):
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '')
            f.write(f"[{i:02d}] {start:06.1f}s - {end:06.1f}s: {text}\n")
    created_files.append(txt_path)
    logger.info(f"文字起こし結果(TXT)を保存: {txt_path}")
    
    return created_files


def save_emotion_results(
    emotion_result: Dict[str, Any],
    output_dir: Path,
    base_filename: str
) -> List[Path]:
    """
    感情分析結果をファイルに保存
    
    Args:
        emotion_result: 感情分析結果
        output_dir: 出力ディレクトリ
        base_filename: ベースファイル名
        
    Returns:
        List[Path]: 作成されたファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    
    # JSON形式で保存
    json_path = output_dir / f"{base_filename}_emotions.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(emotion_result, f, ensure_ascii=False, indent=2)
    created_files.append(json_path)
    logger.info(f"感情分析結果(JSON)を保存: {json_path}")
    
    # CSV形式で保存
    csv_path = output_dir / f"{base_filename}_emotions.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['start', 'end', 'text', 'dominant_emotion', 'confidence', 'joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral'])
        
        for segment in emotion_result.get('segments', []):
            emotions = segment.get('emotions', {})
            writer.writerow([
                segment.get('start', 0),
                segment.get('end', 0),
                segment.get('text', ''),
                segment.get('dominant_emotion', ''),
                segment.get('confidence', 0),
                emotions.get('joy', 0),
                emotions.get('sadness', 0),
                emotions.get('anger', 0),
                emotions.get('fear', 0),
                emotions.get('surprise', 0),
                emotions.get('neutral', 0)
            ])
    created_files.append(csv_path)
    logger.info(f"感情分析結果(CSV)を保存: {csv_path}")
    
    return created_files


def save_highlight_results(
    highlight_result: Dict[str, Any],
    output_dir: Path,
    base_filename: str
) -> List[Path]:
    """
    ハイライト検出結果をファイルに保存
    
    Args:
        highlight_result: ハイライト検出結果
        output_dir: 出力ディレクトリ
        base_filename: ベースファイル名
        
    Returns:
        List[Path]: 作成されたファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    
    # JSON形式で保存
    json_path = output_dir / f"{base_filename}_highlights.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(highlight_result, f, ensure_ascii=False, indent=2)
    created_files.append(json_path)
    logger.info(f"ハイライト結果(JSON)を保存: {json_path}")
    
    # CSV形式で保存
    csv_path = output_dir / f"{base_filename}_highlights.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'start', 'end', 'duration', 'text', 'highlight_score', 'dominant_emotion', 'confidence'])
        
        for highlight in highlight_result.get('highlights', []):
            writer.writerow([
                highlight.get('rank', 0),
                highlight.get('start', 0),
                highlight.get('end', 0),
                highlight.get('duration', 0),
                highlight.get('text', ''),
                highlight.get('highlight_score', 0),
                highlight.get('dominant_emotion', ''),
                highlight.get('confidence', 0)
            ])
    created_files.append(csv_path)
    logger.info(f"ハイライト結果(CSV)を保存: {csv_path}")
    
    return created_files


def save_summary_report(
    transcription_result: Dict[str, Any],
    emotion_result: Dict[str, Any],
    highlight_result: Dict[str, Any],
    output_dir: Path,
    base_filename: str,
    video_file: str
) -> Path:
    """
    サマリーレポートを作成
    
    Args:
        transcription_result: 文字起こし結果
        emotion_result: 感情分析結果
        highlight_result: ハイライト検出結果
        output_dir: 出力ディレクトリ
        base_filename: ベースファイル名
        video_file: 元動画ファイル名
        
    Returns:
        Path: 作成されたレポートファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{base_filename}_summary.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ShortsClipper 処理結果サマリー\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"元動画ファイル: {video_file}\n")
        f.write(f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 文字起こし統計
        f.write("--- 文字起こし統計 ---\n")
        segments = transcription_result.get('segments', [])
        f.write(f"検出セグメント数: {len(segments)}セグメント\n")
        if segments:
            total_duration = segments[-1].get('end', 0) - segments[0].get('start', 0)
            f.write(f"音声総時間: {total_duration:.1f}秒\n")
            
            full_text = ' '.join([s.get('text', '') for s in segments])
            f.write(f"文字数: {len(full_text)}文字\n")
        f.write("\n")
        
        # 感情分析統計
        f.write("--- 感情分析統計 ---\n")
        emotion_stats = emotion_result.get('statistics', {})
        f.write(f"分析セグメント数: {emotion_stats.get('segments_count', 0)}セグメント\n")
        
        avg_emotions = emotion_stats.get('average_emotions', {})
        if avg_emotions:
            f.write("平均感情スコア:\n")
            for emotion, score in sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {emotion}: {score:.3f}\n")
        
        emotion_dist = emotion_stats.get('dominant_emotion_distribution', {})
        if emotion_dist:
            f.write("支配的感情分布:\n")
            for emotion, ratio in sorted(emotion_dist.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {emotion}: {ratio:.1%}\n")
        f.write("\n")
        
        # ハイライト統計
        f.write("--- ハイライト検出統計 ---\n")
        highlights = highlight_result.get('highlights', [])
        f.write(f"検出ハイライト数: {len(highlights)}個\n")
        
        if highlights:
            total_highlight_duration = sum(h.get('duration', 0) for h in highlights)
            f.write(f"ハイライト総時間: {total_highlight_duration:.1f}秒\n")
            
            avg_score = sum(h.get('highlight_score', 0) for h in highlights) / len(highlights)
            f.write(f"平均ハイライトスコア: {avg_score:.3f}\n")
            
            f.write("\nトップ3ハイライト:\n")
            for i, highlight in enumerate(highlights[:3], 1):
                start = highlight.get('start', 0)
                end = highlight.get('end', 0)
                score = highlight.get('highlight_score', 0)
                text = highlight.get('text', '')[:50]
                f.write(f"  {i}. {start:.1f}s-{end:.1f}s (スコア:{score:.3f}) {text}...\n")
        else:
            f.write("ハイライトが検出されませんでした。\n")
            f.write("（閾値を下げるとより多くのハイライトが検出される可能性があります）\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    logger.info(f"サマリーレポートを保存: {report_path}")
    return report_path


def format_and_save_results(
    transcription_result: Dict[str, Any],
    emotion_result: Dict[str, Any],
    highlight_result: Dict[str, Any],
    config: Dict[str, Any],
    video_file: str
) -> Dict[str, List[Path]]:
    """
    すべての結果をフォーマットして保存
    
    Args:
        transcription_result: 文字起こし結果
        emotion_result: 感情分析結果
        highlight_result: ハイライト検出結果
        config: 出力設定
        video_file: 元動画ファイル名
        
    Returns:
        Dict[str, List[Path]]: 保存されたファイルのパス（種類別）
    """
    output_dir = Path(config.get('output_dir', './output'))
    base_filename = Path(video_file).stem
    
    saved_files = {
        'transcription': [],
        'emotion': [],
        'highlight': [],
        'summary': []
    }
    
    try:
        # 文字起こし結果保存
        if config.get('save_transcription', True):
            saved_files['transcription'] = save_transcription_results(
                transcription_result, output_dir, base_filename
            )
        
        # 感情分析結果保存
        if config.get('save_emotion', True):
            saved_files['emotion'] = save_emotion_results(
                emotion_result, output_dir, base_filename
            )
        
        # ハイライト結果保存
        if config.get('save_highlight', True):
            saved_files['highlight'] = save_highlight_results(
                highlight_result, output_dir, base_filename
            )
        
        # サマリーレポート作成
        if config.get('save_summary', True):
            summary_path = save_summary_report(
                transcription_result, emotion_result, highlight_result,
                output_dir, base_filename, video_file
            )
            saved_files['summary'] = [summary_path]
        
        logger.info(f"すべての結果を {output_dir} に保存しました")
        
    except Exception as e:
        logger.error(f"結果保存中にエラーが発生しました: {e}")
        raise
    
    return saved_files