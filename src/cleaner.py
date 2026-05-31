"""
cleaner.py
【ゴミ清掃工程】Whisperが文字起こししたデータから、不要なフィラー（口癖）を除去する専用モジュール。

責務:
  - タイムスタンプを一切壊さずに、指定された単語だけを空文字に置き換える
  - 元のデータを書き換えないよう、新しいデータを作成して返す
"""

import copy
from tqdm import tqdm

def remove_fillers(segments: list, filler_list: list[str]) -> list:
    """
    タイムスタンプを維持したまま、フィラー（口癖）だけを空文字に置換します。
    
    Args:
        segments: Whisperが出力した生データのリスト
        filler_list: 除去したい単語のリスト
        
    Returns:
        フィラー除去済みの新しいセグメントリスト
    """
    if not filler_list:
        tqdm.write("[*] フィラーリストが空のため、清掃処理をスキップします。")
        return copy.deepcopy(segments)
    
    tqdm.write("[*] タイムスタンプ維持型フィラー除去を実行中...")
    
    # 💡 重要: 元のデータを直接書き換えないよう、ディープコピー（完全な複製）を作成します
    cleaned_segments = copy.deepcopy(segments)

    for seg in cleaned_segments:
        # 1. セグメント全体のテキストから置換
        seg_text = seg.get("text", "")
        for filler in filler_list:
            seg_text = seg_text.replace(filler, "")

        # 2. 1文字/1単語単位（words）のデータを精査し、フィラー単語のみ空文字化
        for w in seg.get("words", []):
            if w.get("word", "").strip() in filler_list:
                w["word"] = ""  # 時間枠（start/end）はそのまま残し、文字だけ消す

        seg["text"] = seg_text

    return cleaned_segments