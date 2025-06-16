#!/usr/bin/env python3
"""
バッチパラメーター調整ツール

複数の設定を一括で実行し、結果を比較する。
パラメーター調整の効率化を目的とする。

Usage:
  python batch_tune.py --video sumi-claude-code-04 --configs config/highlight_detection*.yaml
  python batch_tune.py --video sumi-claude-code-04 --preset all
"""

import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import click
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_highlight_detection(
    emotions_file: str,
    config_file: str,
    output_dir: str = "data/stage4_highlights"
) -> Dict[str, Any]:
    """
    ハイライト検出プロセスを実行する
    
    Args:
        emotions_file: 感情分析結果ファイル
        config_file: 設定ファイル
        output_dir: 出力ディレクトリ
        
    Returns:
        Dict[str, Any]: 実行結果
    """
    config_path = Path(config_file)
    config_name = config_path.stem
    
    start_time = time.time()
    
    try:
        cmd = [
            "python", "process_highlights.py",
            "--input", emotions_file,
            "--output", output_dir,
            "--config", config_file
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30秒タイムアウト
        )
        
        processing_time = time.time() - start_time
        
        if result.returncode == 0:
            # 結果ファイルを読み込み
            video_id = Path(emotions_file).stem.replace('_text_emotions', '').replace('_audio_emotions', '')
            result_file = Path(output_dir) / f"{video_id}_highlights_{config_name}.json"
            
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    highlight_result = json.load(f)
                
                return {
                    "config_name": config_name,
                    "config_file": str(config_file),
                    "success": True,
                    "processing_time": processing_time,
                    "highlight_count": len(highlight_result.get("highlights", [])),
                    "average_score": highlight_result.get("summary", {}).get("average_score", 0.0),
                    "total_duration": highlight_result.get("summary", {}).get("total_duration", 0.0),
                    "result_file": str(result_file),
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "config_name": config_name,
                    "config_file": str(config_file),
                    "success": False,
                    "processing_time": processing_time,
                    "error": "結果ファイルが見つかりません",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            return {
                "config_name": config_name,
                "config_file": str(config_file),
                "success": False,
                "processing_time": processing_time,
                "error": f"プロセス実行失敗 (code: {result.returncode})",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            "config_name": config_name,
            "config_file": str(config_file),
            "success": False,
            "processing_time": time.time() - start_time,
            "error": "タイムアウト"
        }
    except Exception as e:
        return {
            "config_name": config_name,
            "config_file": str(config_file),
            "success": False,
            "processing_time": time.time() - start_time,
            "error": str(e)
        }


def get_preset_configs(preset: str) -> List[str]:
    """
    プリセットに基づいて設定ファイルのリストを取得する
    
    Args:
        preset: プリセット名
        
    Returns:
        List[str]: 設定ファイルのパスのリスト
    """
    config_dir = Path("config")
    
    if preset == "all":
        pattern = "highlight_detection*.yaml"
    elif preset == "basic":
        pattern = "highlight_detection.yaml"
    elif preset == "comparison":
        return [
            "config/highlight_detection.yaml",
            "config/highlight_detection_aggressive.yaml",
            "config/highlight_detection_conservative.yaml"
        ]
    else:
        raise ValueError(f"不明なプリセット: {preset}")
    
    return [str(f) for f in config_dir.glob(pattern)]


def generate_batch_report(results: List[Dict[str, Any]], video_id: str) -> str:
    """
    バッチ実行結果のレポートを生成する
    
    Args:
        results: 実行結果のリスト
        video_id: 動画ID
        
    Returns:
        str: レポート文字列
    """
    report = []
    report.append("=" * 80)
    report.append(f"バッチパラメーター調整レポート - {video_id}")
    report.append("=" * 80)
    report.append("")
    
    # 成功・失敗統計
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    report.append(f"📊 実行統計")
    report.append(f"  総実行数: {len(results)}")
    report.append(f"  成功: {len(successful)}")
    report.append(f"  失敗: {len(failed)}")
    report.append("")
    
    # 成功した結果の詳細
    if successful:
        report.append("✅ 成功した設定")
        report.append("-" * 60)
        
        # パフォーマンス順でソート
        successful.sort(key=lambda x: (-x.get("highlight_count", 0), x.get("processing_time", 999)))
        
        for result in successful:
            config_name = result["config_name"]
            highlight_count = result.get("highlight_count", 0)
            avg_score = result.get("average_score", 0.0)
            total_duration = result.get("total_duration", 0.0)
            processing_time = result.get("processing_time", 0.0)
            
            report.append(f"🔧 {config_name.upper()}")
            report.append(f"  ハイライト数: {highlight_count}個")
            report.append(f"  平均スコア: {avg_score:.3f}")
            report.append(f"  総時間: {total_duration:.1f}秒")
            report.append(f"  処理時間: {processing_time:.3f}秒")
            report.append(f"  結果ファイル: {result.get('result_file', 'N/A')}")
            report.append("")
    
    # 失敗した結果
    if failed:
        report.append("❌ 失敗した設定")
        report.append("-" * 60)
        
        for result in failed:
            config_name = result["config_name"]
            error = result.get("error", "不明なエラー")
            processing_time = result.get("processing_time", 0.0)
            
            report.append(f"⚠️ {config_name.upper()}")
            report.append(f"  エラー: {error}")
            report.append(f"  処理時間: {processing_time:.3f}秒")
            if result.get("stderr"):
                report.append(f"  詳細: {result['stderr'][:100]}...")
            report.append("")
    
    # 推奨事項
    if successful:
        report.append("💡 推奨事項")
        report.append("-" * 60)
        
        best_count = max(successful, key=lambda x: x.get("highlight_count", 0))
        best_score = max(successful, key=lambda x: x.get("average_score", 0.0))
        fastest = min(successful, key=lambda x: x.get("processing_time", 999))
        
        report.append(f"🎯 最多検出: {best_count['config_name']} ({best_count.get('highlight_count', 0)}個)")
        report.append(f"🏆 最高スコア: {best_score['config_name']} ({best_score.get('average_score', 0.0):.3f})")
        report.append(f"⚡ 最高速: {fastest['config_name']} ({fastest.get('processing_time', 0.0):.3f}秒)")
        report.append("")
        
        # バランスの良い設定を推奨
        balanced = [r for r in successful if 2 <= r.get("highlight_count", 0) <= 15 and r.get("average_score", 0.0) >= 0.5]
        if balanced:
            best_balanced = max(balanced, key=lambda x: x.get("average_score", 0.0))
            report.append(f"⚖️  推奨設定: {best_balanced['config_name']} (バランス重視)")
        elif successful:
            report.append(f"⚖️  推奨設定: {best_count['config_name']} (検出数重視)")
        
        report.append("")
    
    # 次のステップ
    report.append("🚀 次のステップ")
    report.append("-" * 60)
    if successful:
        best_configs = [r['config_name'] for r in successful[:3]]
        report.append(f"1. 上位設定での詳細確認:")
        for config in best_configs:
            report.append(f"   python compare_highlights.py --video {video_id} --configs {config}")
        report.append("")
        report.append("2. 最適設定でのファイナル実行:")
        report.append(f"   python process_highlights.py --input data/stage3_emotions/{video_id}_text_emotions.json --config config/highlight_detection_{best_configs[0]}.yaml")
    else:
        report.append("1. 設定ファイルの見直し")
        report.append("2. 閾値の調整")
        report.append("3. 感情分析結果の確認")
    
    report.append("")
    report.append("=" * 80)
    report.append(f"レポート生成時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    return "\n".join(report)


@click.command()
@click.option('--video', '-v', required=True,
              help='動画ID')
@click.option('--configs', '-c', multiple=True, type=click.Path(),
              help='設定ファイルのパス（複数指定可能）')
@click.option('--preset', '-p', type=click.Choice(['all', 'basic', 'comparison']),
              help='プリセット設定 (all: 全設定, basic: 基本設定のみ, comparison: 比較用3設定)')
@click.option('--output-dir', default='data/stage4_highlights',
              help='出力ディレクトリ')
@click.option('--parallel', '-j', default=3, type=int,
              help='並列実行数（デフォルト: 3）')
@click.option('--report', '-r', default=None,
              help='レポート出力ファイル（指定しない場合は標準出力）')
@click.option('--verbose', is_flag=True,
              help='詳細ログ出力')
def main(video, configs, preset, output_dir, parallel, report, verbose):
    """
    バッチパラメーター調整ツール
    """
    click.echo(f"🎬 動画: {video}")
    
    # 設定ファイルの取得
    config_files = []
    
    if configs:
        config_files = list(configs)
    elif preset:
        config_files = get_preset_configs(preset)
    else:
        click.echo("❌ --configs または --preset を指定してください")
        return
    
    if not config_files:
        click.echo("❌ 実行する設定ファイルが見つかりません")
        return
    
    click.echo(f"🔧 実行設定数: {len(config_files)}")
    for config_file in config_files:
        click.echo(f"  - {config_file}")
    
    # 感情分析ファイルの確認
    emotions_file = f"data/stage3_emotions/{video}_text_emotions.json"
    if not Path(emotions_file).exists():
        click.echo(f"❌ 感情分析ファイルが見つかりません: {emotions_file}")
        return
    
    click.echo(f"📊 入力ファイル: {emotions_file}")
    click.echo(f"📁 出力ディレクトリ: {output_dir}")
    click.echo(f"⚡ 並列実行数: {parallel}")
    click.echo("")
    
    # バッチ実行
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        # 全タスクを投入
        future_to_config = {
            executor.submit(run_highlight_detection, emotions_file, config_file, output_dir): config_file
            for config_file in config_files
        }
        
        # 完了したものから結果を取得
        for future in as_completed(future_to_config):
            config_file = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                
                config_name = result["config_name"]
                if result["success"]:
                    highlight_count = result.get("highlight_count", 0)
                    processing_time = result.get("processing_time", 0.0)
                    click.echo(f"✅ {config_name}: {highlight_count}個 ({processing_time:.3f}秒)")
                else:
                    error = result.get("error", "不明なエラー")
                    click.echo(f"❌ {config_name}: {error}")
                    
                if verbose and result.get("stderr"):
                    click.echo(f"   詳細: {result['stderr']}")
                    
            except Exception as e:
                click.echo(f"❌ {config_file}: 例外発生 - {e}")
                results.append({
                    "config_name": Path(config_file).stem,
                    "config_file": config_file,
                    "success": False,
                    "error": str(e),
                    "processing_time": 0.0
                })
    
    total_time = time.time() - start_time
    
    click.echo("")
    click.echo(f"🏁 バッチ実行完了 (総時間: {total_time:.2f}秒)")
    
    # レポート生成
    report_content = generate_batch_report(results, video)
    
    if report:
        report_path = Path(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        click.echo(f"📝 レポートを保存: {report_path}")
    else:
        click.echo("")
        click.echo(report_content)


if __name__ == '__main__':
    main()