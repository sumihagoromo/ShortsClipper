#!/usr/bin/env python3
"""
改善されたハイライト検出パラメーターのテストスクリプト
"""

import json
import yaml
from src.highlight_detector import full_highlight_detection_workflow
from src.utils.config import ConfigManager

def test_improved_highlights():
    """既存の感情分析結果を使って改善されたハイライト検出をテスト"""
    
    # 設定読み込み
    config_manager = ConfigManager()
    config = config_manager.config
    
    # 既存の感情分析結果を読み込み
    with open('output/sumi-claude-code-01_60-62min_emotions.json', 'r', encoding='utf-8') as f:
        emotion_result = json.load(f)
    
    print(f"読み込んだ感情分析結果: {len(emotion_result.get('segments', []))}セグメント")
    
    # ハイライト検出設定を表示
    highlight_config = config.get('highlight_detection', {})
    print("\\n=== ハイライト検出設定 ===")
    for key, value in highlight_config.items():
        print(f"{key}: {value}")
    
    # ハイライト検出実行
    print("\\n=== ハイライト検出実行 ===")
    try:
        result = full_highlight_detection_workflow(emotion_result, highlight_config)
        
        print(f"ハイライト検出完了: {result['metadata']['total_highlights']}個")
        print(f"処理時間: {result['metadata']['processing_time']:.3f}秒")
        
        # 結果の詳細表示
        if result['highlights']:
            print("\\n=== 検出されたハイライト ===")
            for i, highlight in enumerate(result['highlights'][:5]):  # 上位5個まで表示
                print(f"[{i+1}] {highlight['start']:.1f}s-{highlight['end']:.1f}s: ")
                print(f"    スコア: {highlight['highlight_score']:.3f}")
                print(f"    テキスト: {highlight['text']}")
                print(f"    感情: {highlight['dominant_emotion']}")
                print()
        else:
            print("ハイライトが検出されませんでした")
            
        # 改善された結果をファイルに保存
        with open('output/improved_highlights_test.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print("\\n改善された結果を output/improved_highlights_test.json に保存しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_highlights()