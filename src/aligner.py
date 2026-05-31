"""
aligner.py
5. 時間同期：動的計画法（DP）で、修正後テキストとタイムスタンプを照合
校正前後のテキストを動的計画法で比較し、消失したタイムスタンプ情報を文字単位で復元するモジュール。

責務:
  - 編集距離に基づく2次元DP行列の計算
  - 最短編集経路を特定するためのバックトラック処理
  - 校正後の各文字に対する適切な開始・終了時間の割り当ておよび補間
"""

from src.exceptions import WhisperSrtBaseError

class AlignmentError(WhisperSrtBaseError):
    """アライメント処理およびタイムスタンプ復元中に発生した致命的なエラー。"""
    pass

def calculate_dp_matrix(source: str, target: str) -> list[list[int]]:
    """校正前テキスト(source)と校正後テキスト(target)の最小編集距離を計算し、DP行列を構築する関数。

    Args:
        source: 校正前の文字列（Whisperが出力した元の文字の並び）。
        target: 校正後の文字列（DeBERTa等が修正した綺麗な文字の並び）。

    Returns:
        コスト計算結果が格納された2次元配列の行列。
    """
    m, n = len(source), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 行列の境界条件（初期値）を設定します。
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # 動的計画法による最小コストの計算ループ。
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 文字が一致する場合は追加コストなし
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    # 削除コスト
                    dp[i][j - 1] + 1,    # 挿入コスト
                    dp[i - 1][j - 1] + 1 # 置換コスト
                )
    return dp

def backtrack_alignment(dp: list[list[int]], source: str, target: str) -> list[tuple[int, int]]:
    """DP行列を終点から逆算し、ソース文字列とターゲット文字列の最適な文字対応ルートを導き出す関数。

    Args:
        dp: コスト計算済みの2次元DP行列。
        source: 校正前の文字列。
        target: 校正後の文字列。

    Returns:
        (sourceのインデックス, targetのインデックス) の対応ペアを格納したリスト。
    """
    i, j = len(source), len(target)
    alignment_path = []
    
    # 行列の右下（終点）から左上（始点）に向かって経路を探索します。
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source[i - 1] == target[j - 1]:
            alignment_path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment_path.append((i - 1, j - 1))  # 置換が発生した箇所
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
            alignment_path.append((i - 1, -1))     # ターゲット側で文字が削除された箇所
            i -= 1
        else:
            alignment_path.append((-1, j - 1))     # ターゲット側で新しく文字が挿入された箇所
            j -= 1
            
    # 逆順で取得した経路を正順に反転させます。
    alignment_path.reverse()
    return alignment_path

def restore_char_timestamps(alignment_path: list[tuple[int, int]], source_words: list) -> list[dict]:
    """文字の対応ルートとWhisperの元の時間データを基に、校正後の文字に対するタイムスタンプを復元する関数。

    Args:
        alignment_path: バックトラックによって得られた文字対応インデックスのペアリスト。
        source_words: Whisperが計測した1文字ごとの時間情報を含む辞書リスト。

    Returns:
        校正後の各文字位置にマッピングされる時間情報（開始秒・終了秒）のリスト。
    """
    refined_timestamps = []
    
    # 時間情報が欠損した場合の補間用として、直前の有効なタイムスタンプ情報を保持します。
    last_valid_start = 0.0
    last_valid_end = 0.0
    if source_words:
        last_valid_start = source_words[0].get("start", 0.0)
        last_valid_end = source_words[0].get("end", 0.0)

    for s_idx, t_idx in alignment_path:
        if t_idx == -1:
            # 校正によって文字が完全に削除された場合は、時間情報の復元が不要なためスキップします。
            continue
            
        if s_idx != -1 and s_idx < len(source_words):
            # 元の文字と綺麗に対応が取れた場合、Whisperが記録していた高精度タイムスタンプをそのまま引き継ぎます。
            start_time = source_words[s_idx].get("start", last_valid_start)
            end_time = source_words[s_idx].get("end", last_valid_end)
            last_valid_start = start_time
            last_valid_end = end_time
        else:
            # AIの校正により新しく挿入された文字のケース。
            # 直前の確定終了時間を起点とし、極小の表示時間（0.05秒）を疑似的に割り振って時間を進めます。
            start_time = last_valid_end
            end_time = start_time + 0.05
            last_valid_end = end_time

        refined_timestamps.append({
            "start": start_time,
            "end": end_time
        })
        
    return refined_timestamps

def align_text_and_timestamps(source_words: list, refined_text: str) -> list[dict]:
    """外部モジュールから呼び出されるアライメント制御のメイン関数。

    Args:
        source_words: Whisperのセグメントから集約された1文字ごとのオリジナル時間データリスト。
        refined_text: 校正AIによって生成された綺麗な文字列全体。

    Returns:
        各文字に正確なタイムスタンプ（start, end）が結合された辞書データのリスト。

    Raises:
        AlignmentError: アライメント計算中に予期せぬシステム例外が発生した場合。
    """
    try:
        # 集約されているwordsリストから、校正前のオリジナル文字列を再構築します。
        source_text = "".join([item.get("word", item.get("text", "")) for item in source_words])
        
        # 1. 最小編集距離のDPマトリクスを計算
        dp = calculate_dp_matrix(source_text, refined_text)
        
        # 2. マトリクスを逆書き出しして文字の対応ルートを特定
        path = backtrack_alignment(dp, source_text, refined_text)
        
        # 3. ルートに従って各文字のタイムスタンプを復元
        char_timestamps = restore_char_timestamps(path, source_words)
        
        # 完成した時間配列に、校正後の文字自体をマッピングして構造化データを構築します。
        final_result = []
        for idx, char_str in enumerate(refined_text):
            final_result.append({
                "text": char_str,
                "start": char_timestamps[idx]["start"],
                "end": char_timestamps[idx]["end"]
            })
            
        return final_result
        
    except Exception as e:
        raise AlignmentError(f"タイムスタンプのアライメント処理に失敗しました。原因: {str(e)}")