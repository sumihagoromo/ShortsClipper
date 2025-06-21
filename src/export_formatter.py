#!/usr/bin/env python3
"""
動画編集ソフト対応エクスポートフォーマッター

ハイライト検出結果を動画編集ソフトで直接利用可能な形式に変換:
1. Premiere Pro互換CSV形式
2. DaVinci Resolve EDL形式
3. Final Cut Pro XML形式
4. タイムライン視覚レポート
5. YouTubeチャプター形式

関数型プログラミング原則に従って実装
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class HighlightSegment:
    """ハイライトセグメント情報"""
    start_time: float
    end_time: float
    text: str
    emotion_type: str
    emotion_score: float
    highlight_level: str  # high, medium, low
    keywords: List[str]
    confidence: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HighlightSegment':
        """辞書からハイライトセグメントを作成"""
        return cls(
            start_time=data.get('start_time', 0.0),
            end_time=data.get('end_time', 0.0),
            text=data.get('text', ''),
            emotion_type=data.get('emotion_type', 'neutral'),
            emotion_score=data.get('emotion_score', 0.0),
            highlight_level=data.get('highlight_level', 'low'),
            keywords=data.get('keywords', []),
            confidence=data.get('confidence', 0.0)
        )


@dataclass
class ExportConfig:
    """エクスポート設定"""
    # フィルタリング設定
    min_highlight_level: str = "medium"  # low, medium, high
    min_duration: float = 0.5  # 最小ハイライト長（秒）
    max_segments: int = 20  # 最大セグメント数
    
    # フォーマット設定
    time_format: str = "hh:mm:ss.ff"  # タイムコード形式
    frame_rate: float = 30.0  # フレームレート
    include_context: bool = True  # 前後の文脈を含める
    context_seconds: float = 1.0  # 前後の文脈時間
    
    # Premiere Pro設定
    premiere_marker_type: str = "Comment"  # マーカータイプ
    premiere_include_description: bool = True
    
    # DaVinci設定
    davinci_track_name: str = "Highlights"
    davinci_include_edl_header: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExportConfig':
        """辞書から設定を作成"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


def seconds_to_timecode(seconds: float, frame_rate: float = 30.0, format: str = "hh:mm:ss.ff") -> str:
    """
    秒をタイムコードに変換する純粋関数
    
    Args:
        seconds: 秒数
        frame_rate: フレームレート
        format: タイムコード形式
        
    Returns:
        str: タイムコード文字列
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    frames = int((seconds % 1) * frame_rate)
    
    if format == "hh:mm:ss.ff":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{frames:02d}"
    elif format == "hh:mm:ss:ff":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
    elif format == "seconds":
        return f"{seconds:.2f}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{frames:02d}"


def load_highlights_data(file_path: Union[str, Path]) -> List[HighlightSegment]:
    """
    ハイライト検出結果ファイルを読み込む純粋関数
    
    Args:
        file_path: ハイライト結果ファイルのパス
        
    Returns:
        List[HighlightSegment]: ハイライトセグメントのリスト
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ハイライトファイルが見つかりません: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ハイライトセグメントを抽出
        highlights = []
        
        if 'highlights' in data:
            # Stage 4の出力形式
            for highlight in data['highlights']:
                # 感情データから最も高いスコアの感情を取得
                emotions = highlight.get('emotions', {})
                if emotions:
                    dominant_emotion = max(emotions.keys(), key=lambda k: emotions[k])
                    emotion_score = emotions[dominant_emotion]
                else:
                    dominant_emotion = highlight.get('dominant_emotion', 'neutral')
                    emotion_score = 0.0
                
                # ハイライトレベルをスコアから推定
                highlight_score = highlight.get('highlight_score', 0.0)
                if highlight_score > 1.0:
                    level = 'high'
                elif highlight_score > 0.5:
                    level = 'medium'
                else:
                    level = 'low'
                
                segment = HighlightSegment(
                    start_time=highlight.get('start', 0.0),
                    end_time=highlight.get('end', 0.0),
                    text=highlight.get('text', ''),
                    emotion_type=dominant_emotion,
                    emotion_score=emotion_score,
                    highlight_level=level,
                    keywords=[],  # TODO: キーワード抽出の実装
                    confidence=highlight.get('confidence', 0.0)
                )
                highlights.append(segment)
        
        elif 'segments' in data:
            # Stage 2/3の出力形式から推定
            for segment in data['segments']:
                if segment.get('highlight_score', 0.0) > 0.5:
                    highlight_segment = HighlightSegment(
                        start_time=segment.get('start', 0.0),
                        end_time=segment.get('end', 0.0),
                        text=segment.get('text', ''),
                        emotion_type='unknown',
                        emotion_score=segment.get('highlight_score', 0.0),
                        highlight_level='medium',
                        keywords=[],
                        confidence=segment.get('confidence', 0.0)
                    )
                    highlights.append(highlight_segment)
        
        return highlights
        
    except Exception as e:
        logger.error(f"ハイライトデータ読み込みエラー: {e}")
        return []


def filter_highlights(
    highlights: List[HighlightSegment], 
    config: ExportConfig
) -> List[HighlightSegment]:
    """
    ハイライトをフィルタリングする純粋関数
    
    Args:
        highlights: ハイライトセグメントのリスト
        config: エクスポート設定
        
    Returns:
        List[HighlightSegment]: フィルタリング済みハイライト
    """
    # レベルフィルタリング
    level_priority = {'low': 0, 'medium': 1, 'high': 2}
    min_level = level_priority.get(config.min_highlight_level, 1)
    
    filtered = [
        h for h in highlights 
        if level_priority.get(h.highlight_level, 0) >= min_level
    ]
    
    # 継続時間フィルタリング
    filtered = [h for h in filtered if h.duration >= config.min_duration]
    
    # 信頼度順にソート
    filtered.sort(key=lambda x: x.confidence, reverse=True)
    
    # 最大セグメント数制限
    if len(filtered) > config.max_segments:
        filtered = filtered[:config.max_segments]
    
    # 時系列順にソート
    filtered.sort(key=lambda x: x.start_time)
    
    return filtered


def export_premiere_pro_csv(
    highlights: List[HighlightSegment],
    output_path: Union[str, Path],
    config: ExportConfig
) -> None:
    """
    Premiere Pro互換CSV形式でエクスポートする純粋関数
    
    Args:
        highlights: ハイライトセグメントのリスト
        output_path: 出力ファイルパス
        config: エクスポート設定
    """
    output_path = Path(output_path)
    
    # フィルタリング
    filtered_highlights = filter_highlights(highlights, config)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Marker Name', 'Description', 'In', 'Out', 'Duration', 'Marker Type'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, highlight in enumerate(filtered_highlights, 1):
            # コンテキストを考慮した時間調整
            start_time = max(0, highlight.start_time - config.context_seconds)
            end_time = highlight.end_time + config.context_seconds
            
            # タイムコード変換
            in_timecode = seconds_to_timecode(start_time, config.frame_rate, "hh:mm:ss:ff")
            out_timecode = seconds_to_timecode(end_time, config.frame_rate, "hh:mm:ss:ff")
            duration_timecode = seconds_to_timecode(end_time - start_time, config.frame_rate, "hh:mm:ss:ff")
            
            # マーカー名生成
            marker_name = f"Highlight_{i:02d}_{highlight.highlight_level.title()}"
            
            # 説明文生成
            description_parts = []
            if highlight.text:
                description_parts.append(f"テキスト: {highlight.text[:50]}...")
            description_parts.append(f"感情: {highlight.emotion_type} ({highlight.emotion_score:.2f})")
            description_parts.append(f"信頼度: {highlight.confidence:.2f}")
            if highlight.keywords:
                description_parts.append(f"キーワード: {', '.join(highlight.keywords[:3])}")
            
            description = " | ".join(description_parts) if config.premiere_include_description else marker_name
            
            writer.writerow({
                'Marker Name': marker_name,
                'Description': description,
                'In': in_timecode,
                'Out': out_timecode,
                'Duration': duration_timecode,
                'Marker Type': config.premiere_marker_type
            })
    
    logger.info(f"Premiere Pro CSV出力完了: {output_path} ({len(filtered_highlights)}個のハイライト)")


def export_davinci_edl(
    highlights: List[HighlightSegment],
    output_path: Union[str, Path],
    config: ExportConfig
) -> None:
    """
    DaVinci Resolve EDL形式でエクスポートする純粋関数
    
    Args:
        highlights: ハイライトセグメントのリスト
        output_path: 出力ファイルパス
        config: エクスポート設定
    """
    output_path = Path(output_path)
    
    # フィルタリング
    filtered_highlights = filter_highlights(highlights, config)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # EDLヘッダー
        if config.davinci_include_edl_header:
            f.write("TITLE: ShortsClipper Highlights\n")
            f.write("FCM: NON-DROP FRAME\n\n")
        
        for i, highlight in enumerate(filtered_highlights, 1):
            # コンテキストを考慮した時間調整
            start_time = max(0, highlight.start_time - config.context_seconds)
            end_time = highlight.end_time + config.context_seconds
            
            # タイムコード変換
            source_in = seconds_to_timecode(start_time, config.frame_rate, "hh:mm:ss:ff")
            source_out = seconds_to_timecode(end_time, config.frame_rate, "hh:mm:ss:ff")
            record_in = seconds_to_timecode((i - 1) * 10, config.frame_rate, "hh:mm:ss:ff")  # 10秒間隔
            record_out = seconds_to_timecode((i - 1) * 10 + (end_time - start_time), config.frame_rate, "hh:mm:ss:ff")
            
            # EDLエントリ
            f.write(f"{i:03d}  001      V     C        {source_in} {source_out} {record_in} {record_out}\n")
            
            # クリップ名
            clip_name = f"Highlight_{i:02d}_{highlight.highlight_level.title()}_{highlight.emotion_type}"
            f.write(f"* FROM CLIP NAME: {clip_name}\n")
            
            # コメント
            if highlight.text:
                f.write(f"* COMMENT: {highlight.text[:100]}\n")
            
            f.write("\n")
    
    logger.info(f"DaVinci Resolve EDL出力完了: {output_path} ({len(filtered_highlights)}個のハイライト)")


def export_youtube_chapters(
    highlights: List[HighlightSegment],
    output_path: Union[str, Path],
    config: ExportConfig,
    video_duration: Optional[float] = None
) -> None:
    """
    YouTubeチャプター形式でエクスポートする純粋関数
    
    Args:
        highlights: ハイライトセグメントのリスト
        output_path: 出力ファイルパス
        config: エクスポート設定
        video_duration: 動画の総時間（秒）
    """
    output_path = Path(output_path)
    
    # フィルタリング
    filtered_highlights = filter_highlights(highlights, config)
    
    # チャプターポイント生成
    chapters = []
    
    # 最初のチャプター（イントロ）
    if filtered_highlights and filtered_highlights[0].start_time > 60:
        chapters.append({
            'time': 0,
            'title': 'イントロ・概要'
        })
    
    # ハイライトベースのチャプター
    prev_end = 0
    for i, highlight in enumerate(filtered_highlights):
        # 前のハイライトから離れている場合は中間チャプターを追加
        if highlight.start_time - prev_end > 120:  # 2分以上の間隔
            mid_time = prev_end + (highlight.start_time - prev_end) / 2
            chapters.append({
                'time': mid_time,
                'title': f'解説・詳細 {len(chapters)}'
            })
        
        # ハイライトチャプター
        chapter_title = generate_chapter_title(highlight, i + 1)
        chapters.append({
            'time': highlight.start_time,
            'title': chapter_title
        })
        
        prev_end = highlight.end_time
    
    # 最後のチャプター（まとめ）
    if video_duration and prev_end < video_duration - 60:
        chapters.append({
            'time': prev_end,
            'title': 'まとめ・結論'
        })
    
    # チャプター間隔の調整（最低2分間隔）
    adjusted_chapters = adjust_chapter_intervals(chapters, min_interval=120)
    
    # YouTube形式で出力
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# YouTubeチャプター\n")
        f.write("# 以下の内容を動画の説明欄にコピーしてください\n\n")
        
        for chapter in adjusted_chapters:
            # YouTube形式では mm:ss
            minutes = int(chapter['time'] // 60)
            seconds = int(chapter['time'] % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"
            f.write(f"{time_str} {chapter['title']}\n")
    
    logger.info(f"YouTubeチャプター出力完了: {output_path} ({len(adjusted_chapters)}個のチャプター)")


def generate_chapter_title(highlight: HighlightSegment, index: int) -> str:
    """
    ハイライトからチャプタータイトルを生成する純粋関数
    
    Args:
        highlight: ハイライトセグメント
        index: チャプター番号
        
    Returns:
        str: チャプタータイトル
    """
    # 感情タイプベースのタイトル
    emotion_titles = {
        'joy': '興奮・盛り上がりポイント',
        'surprise': '驚き・発見シーン',
        'anger': '強調・重要ポイント',
        'fear': '注意・警告部分',
        'sadness': '考察・深掘り',
        'neutral': '説明・解説'
    }
    
    base_title = emotion_titles.get(highlight.emotion_type, '重要ポイント')
    
    # キーワードがある場合は具体化
    if highlight.keywords:
        main_keyword = highlight.keywords[0]
        if main_keyword:
            base_title = f"{main_keyword}について"
    
    # テキストから具体的な内容を抽出
    if highlight.text:
        text_words = highlight.text.strip().split()
        if len(text_words) > 0:
            # 最初の数語でタイトル改善
            first_words = ' '.join(text_words[:4])
            if len(first_words) > 5:
                base_title = f"{first_words}..."
    
    return f"{base_title}"


def adjust_chapter_intervals(
    chapters: List[Dict[str, Any]], 
    min_interval: float = 120
) -> List[Dict[str, Any]]:
    """
    チャプター間隔を調整する純粋関数
    
    Args:
        chapters: チャプターのリスト
        min_interval: 最小間隔（秒）
        
    Returns:
        List[Dict[str, Any]]: 調整済みチャプター
    """
    if not chapters:
        return chapters
    
    adjusted = [chapters[0]]  # 最初のチャプターは保持
    
    for chapter in chapters[1:]:
        last_time = adjusted[-1]['time']
        
        # 最小間隔をチェック
        if chapter['time'] - last_time >= min_interval:
            adjusted.append(chapter)
        # 間隔が短すぎる場合は前のチャプターとマージ
        else:
            # より重要なタイトルを選択
            if 'ハイライト' in chapter['title'] or '重要' in chapter['title']:
                adjusted[-1] = chapter
    
    return adjusted


def export_timeline_report(
    highlights: List[HighlightSegment],
    output_path: Union[str, Path],
    config: ExportConfig
) -> None:
    """
    タイムライン形式の視覚レポートを生成する純粋関数
    
    Args:
        highlights: ハイライトセグメントのリスト
        output_path: 出力ファイルパス
        config: エクスポート設定
    """
    output_path = Path(output_path)
    
    # フィルタリング
    filtered_highlights = filter_highlights(highlights, config)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# ShortsClipper タイムライン解析レポート\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"ハイライト数: {len(filtered_highlights)}個\n\n")
        
        f.write("## ハイライト一覧\n\n")
        
        # ハイライトレベル別の色分け
        level_symbols = {
            'high': '🔥',
            'medium': '⭐',
            'low': '📌'
        }
        
        for i, highlight in enumerate(filtered_highlights, 1):
            symbol = level_symbols.get(highlight.highlight_level, '📌')
            start_time = seconds_to_timecode(highlight.start_time, format="mm:ss")
            end_time = seconds_to_timecode(highlight.end_time, format="mm:ss")
            
            f.write(f"### {symbol} ハイライト {i}: {start_time} - {end_time}\n\n")
            f.write(f"**レベル**: {highlight.highlight_level.upper()}\n")
            f.write(f"**感情**: {highlight.emotion_type} (スコア: {highlight.emotion_score:.2f})\n")
            f.write(f"**信頼度**: {highlight.confidence:.2f}\n")
            
            if highlight.keywords:
                f.write(f"**キーワード**: {', '.join(highlight.keywords)}\n")
            
            if highlight.text:
                f.write(f"**内容**: {highlight.text}\n")
            
            f.write("\n")
        
        # 統計情報
        f.write("## 統計情報\n\n")
        
        if filtered_highlights:
            total_duration = sum(h.duration for h in filtered_highlights)
            avg_confidence = sum(h.confidence for h in filtered_highlights) / len(filtered_highlights)
            
            level_counts = {}
            emotion_counts = {}
            
            for h in filtered_highlights:
                level_counts[h.highlight_level] = level_counts.get(h.highlight_level, 0) + 1
                emotion_counts[h.emotion_type] = emotion_counts.get(h.emotion_type, 0) + 1
            
            f.write(f"- **総ハイライト時間**: {total_duration:.1f}秒\n")
            f.write(f"- **平均信頼度**: {avg_confidence:.2f}\n")
            f.write(f"- **レベル分布**: {level_counts}\n")
            f.write(f"- **感情分布**: {emotion_counts}\n")
    
    logger.info(f"タイムライン解析レポート出力完了: {output_path}")


# 統合エクスポート関数
def export_highlights_all_formats(
    highlights_file: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[ExportConfig] = None,
    video_duration: Optional[float] = None
) -> Dict[str, str]:
    """
    全ての形式でハイライトをエクスポートする統合関数
    
    Args:
        highlights_file: ハイライト検出結果ファイル
        output_dir: 出力ディレクトリ
        config: エクスポート設定
        video_duration: 動画の総時間（秒）
        
    Returns:
        Dict[str, str]: 出力ファイルパスの辞書
    """
    if config is None:
        config = ExportConfig()
    
    highlights_path = Path(highlights_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ハイライトデータ読み込み
    highlights = load_highlights_data(highlights_path)
    
    if not highlights:
        logger.warning("ハイライトデータが見つかりませんでした")
        return {}
    
    base_name = highlights_path.stem
    output_files = {}
    
    try:
        # Premiere Pro CSV
        premiere_file = output_path / f"{base_name}_premiere_pro.csv"
        export_premiere_pro_csv(highlights, premiere_file, config)
        output_files['premiere_pro'] = str(premiere_file)
        
        # DaVinci Resolve EDL
        davinci_file = output_path / f"{base_name}_davinci_resolve.edl"
        export_davinci_edl(highlights, davinci_file, config)
        output_files['davinci_resolve'] = str(davinci_file)
        
        # YouTubeチャプター
        youtube_file = output_path / f"{base_name}_youtube_chapters.txt"
        export_youtube_chapters(highlights, youtube_file, config, video_duration)
        output_files['youtube_chapters'] = str(youtube_file)
        
        # タイムライン解析レポート
        timeline_file = output_path / f"{base_name}_timeline_report.md"
        export_timeline_report(highlights, timeline_file, config)
        output_files['timeline_report'] = str(timeline_file)
        
        logger.info(f"全形式エクスポート完了: {len(output_files)}個のファイル生成")
        
    except Exception as e:
        logger.error(f"エクスポートエラー: {e}")
        raise
    
    return output_files