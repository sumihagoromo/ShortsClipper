#!/usr/bin/env python3
"""
ハイライト検出結果比較ツール

複数の設定で生成されたハイライト検出結果を比較し、
パラメーター調整の効果を可視化する。

Usage:
  python compare_highlights.py --results data/stage4_highlights/video_highlights_*.json
  python compare_highlights.py --video sumi-claude-code-04 --configs standard,aggressive,conservative
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import click


def load_highlight_result(file_path: Path) -> Dict[str, Any]:
    """
    ハイライト検出結果ファイルを読み込む
    
    Args:
        file_path: 結果ファイルのパス
        
    Returns:
        Dict[str, Any]: ハイライト検出結果
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_highlight_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    複数のハイライト検出結果を比較する
    
    Args:
        results: ハイライト検出結果のリスト
        
    Returns:
        Dict[str, Any]: 比較結果
    """
    comparison = {
        "summary": {},
        "details": [],
        "recommendations": []
    }
    
    # 各結果の詳細分析
    for result in results:
        config_name = result.get("config_name", "unknown")
        highlights = result.get("highlights", [])
        summary = result.get("summary", {})
        processing_time = result.get("processing_time", 0.0)
        
        detail = {
            "config_name": config_name,
            "highlight_count": len(highlights),
            "total_duration": summary.get("total_duration", 0.0),
            "average_score": summary.get("average_score", 0.0),
            "coverage_ratio": summary.get("coverage_ratio", 0.0),
            "processing_time": processing_time,
            "emotion_distribution": summary.get("emotion_distribution", {}),
            "score_distribution": summary.get("score_distribution", {}),
            "top_highlights": [
                {
                    "start": h.get("start", 0.0),
                    "end": h.get("end", 0.0),
                    "text": h.get("text", "")[:50] + "..." if len(h.get("text", "")) > 50 else h.get("text", ""),
                    "score": h.get("highlight_score", 0.0),
                    "emotion": h.get("dominant_emotion", "neutral")
                }
                for h in highlights[:3]  # 上位3個
            ]
        }
        
        comparison["details"].append(detail)
    
    # サマリー統計
    if comparison["details"]:
        highlight_counts = [d["highlight_count"] for d in comparison["details"]]
        avg_scores = [d["average_score"] for d in comparison["details"]]
        processing_times = [d["processing_time"] for d in comparison["details"]]
        
        comparison["summary"] = {
            "total_configs": len(comparison["details"]),
            "highlight_count_range": {"min": min(highlight_counts), "max": max(highlight_counts)},
            "avg_score_range": {"min": min(avg_scores), "max": max(avg_scores)},
            "processing_time_range": {"min": min(processing_times), "max": max(processing_times)},
            "fastest_config": min(comparison["details"], key=lambda x: x["processing_time"])["config_name"],
            "most_highlights_config": max(comparison["details"], key=lambda x: x["highlight_count"])["config_name"],
            "highest_score_config": max(comparison["details"], key=lambda x: x["average_score"])["config_name"]
        }
    
    # 推奨事項生成
    comparison["recommendations"] = generate_recommendations(comparison["details"])
    
    return comparison


def generate_recommendations(details: List[Dict[str, Any]]) -> List[str]:
    """
    比較結果に基づいて推奨事項を生成する
    
    Args:
        details: 比較詳細のリスト
        
    Returns:
        List[str]: 推奨事項のリスト
    """
    recommendations = []
    
    if not details:
        return ["比較対象の結果がありません"]
    
    # ハイライト数の分析
    highlight_counts = [d["highlight_count"] for d in details]
    max_count = max(highlight_counts)
    min_count = min(highlight_counts)
    
    if max_count == 0:
        recommendations.append("⚠️  すべての設定でハイライトが検出されませんでした。閾値を下げることを検討してください。")
    elif min_count == 0:
        recommendations.append("⚠️  一部の設定でハイライトが検出されませんでした。保守的すぎる可能性があります。")
    elif max_count > 20:
        recommendations.append("⚠️  ハイライト数が多すぎる設定があります。閾値を上げることを検討してください。")
    
    # スコアの分析
    avg_scores = [d["average_score"] for d in details]
    if max(avg_scores) < 0.5:
        recommendations.append("💡 平均スコアが低めです。キーワード重みを上げることを検討してください。")
    
    # 処理時間の分析
    processing_times = [d["processing_time"] for d in details]
    if max(processing_times) > 1.0:
        recommendations.append("⏱️  処理時間が長めです。パラメーター最適化を検討してください。")
    
    # 設定別の推奨
    best_config = max(details, key=lambda x: x["highlight_count"] if x["highlight_count"] > 0 else 0)
    if best_config["highlight_count"] > 0:
        recommendations.append(f"🎯 '{best_config['config_name']}' 設定が最も効果的です（{best_config['highlight_count']}個検出）。")
    
    # バランスの良い設定の推奨
    balanced_configs = [d for d in details if 3 <= d["highlight_count"] <= 15 and d["average_score"] >= 0.3]
    if balanced_configs:
        best_balanced = max(balanced_configs, key=lambda x: x["average_score"])
        recommendations.append(f"⚖️  バランスの良い設定: '{best_balanced['config_name']}'")
    
    return recommendations


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """
    比較結果をレポート形式にフォーマットする
    
    Args:
        comparison: 比較結果
        
    Returns:
        str: フォーマット済みレポート
    """
    report = []
    report.append("=" * 80)
    report.append("ハイライト検出結果比較レポート")
    report.append("=" * 80)
    report.append("")
    
    # サマリー
    summary = comparison.get("summary", {})
    if summary:
        report.append("📊 サマリー統計")
        report.append("-" * 40)
        report.append(f"比較設定数: {summary.get('total_configs', 0)}")
        
        highlight_range = summary.get("highlight_count_range", {})
        report.append(f"ハイライト数範囲: {highlight_range.get('min', 0)} - {highlight_range.get('max', 0)}個")
        
        score_range = summary.get("avg_score_range", {})
        report.append(f"平均スコア範囲: {score_range.get('min', 0):.3f} - {score_range.get('max', 0):.3f}")
        
        time_range = summary.get("processing_time_range", {})
        report.append(f"処理時間範囲: {time_range.get('min', 0):.3f} - {time_range.get('max', 0):.3f}秒")
        
        report.append("")
        report.append(f"🏆 最高性能:")
        report.append(f"  最速設定: {summary.get('fastest_config', 'N/A')}")
        report.append(f"  最多検出設定: {summary.get('most_highlights_config', 'N/A')}")
        report.append(f"  最高スコア設定: {summary.get('highest_score_config', 'N/A')}")
        report.append("")
    
    # 詳細比較
    details = comparison.get("details", [])
    if details:
        report.append("📋 詳細比較")
        report.append("-" * 40)
        
        for detail in details:
            config_name = detail["config_name"]
            report.append(f"🔧 {config_name.upper()} 設定")
            report.append(f"  ハイライト数: {detail['highlight_count']}個")
            report.append(f"  総時間: {detail['total_duration']:.1f}秒")
            report.append(f"  平均スコア: {detail['average_score']:.3f}")
            report.append(f"  処理時間: {detail['processing_time']:.3f}秒")
            
            # 感情分布
            emotion_dist = detail.get("emotion_distribution", {})
            if emotion_dist:
                report.append(f"  感情分布: {', '.join([f'{e}: {c}個' for e, c in emotion_dist.items()])}")
            
            # トップハイライト
            top_highlights = detail.get("top_highlights", [])
            if top_highlights:
                report.append(f"  トップハイライト:")
                for i, h in enumerate(top_highlights, 1):
                    report.append(f"    {i}. {h['start']:.1f}s-{h['end']:.1f}s (スコア:{h['score']:.3f}) {h['text']}")
            
            report.append("")
    
    # 推奨事項
    recommendations = comparison.get("recommendations", [])
    if recommendations:
        report.append("💡 推奨事項")
        report.append("-" * 40)
        for rec in recommendations:
            report.append(f"  {rec}")
        report.append("")
    
    report.append("=" * 80)
    report.append(f"レポート生成時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    return "\n".join(report)


@click.command()
@click.option('--results', '-r', multiple=True, type=click.Path(exists=True),
              help='比較する結果ファイルのパス（複数指定可能）')
@click.option('--video', '-v', default=None,
              help='動画ID（設定名から自動検索）')
@click.option('--configs', '-c', default=None,
              help='設定名をカンマ区切りで指定（例: standard,aggressive,conservative）')
@click.option('--output', '-o', default=None,
              help='比較レポート出力ファイル（指定しない場合は標準出力）')
@click.option('--format', 'output_format', default='text', type=click.Choice(['text', 'json']),
              help='出力形式')
def main(results, video, configs, output, output_format):
    """
    ハイライト検出結果比較ツール
    """
    result_files = []
    
    # 結果ファイルの収集
    if results:
        result_files = [Path(r) for r in results]
    elif video and configs:
        # 動画IDと設定名から自動検索
        config_names = [c.strip() for c in configs.split(',')]
        highlights_dir = Path('data/stage4_highlights')
        
        for config_name in config_names:
            result_file = highlights_dir / f"{video}_highlights_{config_name}.json"
            if result_file.exists():
                result_files.append(result_file)
            else:
                click.echo(f"⚠️  結果ファイルが見つかりません: {result_file}")
    else:
        click.echo("❌ --results または --video と --configs を指定してください")
        return
    
    if not result_files:
        click.echo("❌ 比較する結果ファイルが見つかりません")
        return
    
    click.echo(f"🔍 {len(result_files)}個の結果ファイルを比較中...")
    
    # 結果読み込み
    highlight_results = []
    for file_path in result_files:
        try:
            result = load_highlight_result(file_path)
            highlight_results.append(result)
            click.echo(f"  ✅ 読み込み: {file_path.name}")
        except Exception as e:
            click.echo(f"  ❌ 読み込み失敗: {file_path.name} - {e}")
    
    if not highlight_results:
        click.echo("❌ 有効な結果ファイルがありません")
        return
    
    # 比較実行
    comparison = compare_highlight_results(highlight_results)
    
    # 出力
    if output_format == 'json':
        output_data = json.dumps(comparison, ensure_ascii=False, indent=2)
    else:
        output_data = format_comparison_report(comparison)
    
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_data)
        click.echo(f"📝 比較レポートを保存: {output_path}")
    else:
        click.echo(output_data)


if __name__ == '__main__':
    main()