"""
forced_aligner.py
【タイムスタンプ同期工程】AIによって文字数や表現が変わったテキストに対し、
元のWhisper音声認識が持っていた時間情報を強制的に再割り当て（Forced Alignment）するモジュール。

責務:
  - AI校正後のテキストと、元の音声時間データを比較・照合する
  - 動的計画法（DP）などを用いて、各文字に適切な start / end の秒数を割り当てる
"""

import copy
from tqdm import tqdm
from src.aligner import align_text_and_timestamps # 既存のDPアルゴリズムを呼び出します

def run_forced_alignment(original_segments: list, ai_refined_segments: list) -> list:
    """
    Whisperの元データが持つ正確な時間を、AI校正後のテキストに強制同期します。
    
    Args:
        original_segments: Whisperが出力したタイムスタンプ付きの元データ
        ai_refined_segments: AIが校正した後のテキストデータ
        
    Returns:
        時間が正確に割り当てられた、最終成果物用の新しいセグメントリスト
    """
    tqdm.write("[*] Forced Alignment（強制タイムスタンプ同期）を実行中...")
    
    # 💡 重要: 元データを壊さないよう完全な複製を作成
    aligned_segments = copy.deepcopy(ai_refined_segments)

    for i in range(min(len(original_segments), len(aligned_segments))):
        orig_seg = original_segments[i]
        ref_seg = aligned_segments[i]

        source_words = orig_seg.get("words", [])
        refined_text = ref_seg.get("text", "")

        # 既存の src/aligner.py にあるアルゴリズムを呼び出して文字と時間を同期
        try:
            # ※ align_text_and_timestampsは 1セグメント分の同期ができるように実装されている前提
            aligned_words = align_text_and_timestamps(source_words, refined_text)
            ref_seg["words"] = aligned_words
            
        except Exception as e:
            # エラーが起きた場合は、元のセグメント枠の時間を借りて安全に代替する
            seg_start = float(orig_seg.get("start", 0.0))
            seg_end = float(orig_seg.get("end", 0.0))
            
            # 【重要】単語(word)ではなく、テキスト(text)キーに間違えてアクセスしないよう明記
            tqdm.write(f"[警告] ID:{orig_seg.get('id', '?')} の同期に失敗。セグメント時間({seg_start}〜{seg_end})で代替します: {e}")
            
            # 代替データは "word" キーで保存する（BudouXのformatterが後で読み込めるように）
            ref_seg["words"] = [{"word": refined_text, "start": seg_start, "end": seg_end}]

    return aligned_segments