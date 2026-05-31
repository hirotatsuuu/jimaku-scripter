"""
aligner.py
校正前後のテキストを動的計画法（DP）で比較し、消失・変化したタイムスタンプ情報を
文字単位で高精度に復元・マッピングするモジュール。

主な修正点:
- バックトラック時のインデックス境界の完全防御
- LLMによる「言葉の追加（挿入）」が発生した際、前後の確定時間から正確に時間を等分割して割り振る線形補間の実装
- ゼロ除算や負の時間計算の徹底排除
"""

from tqdm import tqdm
from src.exceptions import WhisperSrtBaseError

class AlignmentError(WhisperSrtBaseError):
    """アライメント処理およびタイムスタンプ復元中に発生した致命的なエラー。"""
    pass

def calculate_dp_matrix(source: str, target: str) -> list[list[int]]:
    """
    校正前テキスト(source)と校正後テキスト(target)の最小編集距離（レーベンシュタイン距離）を計算し、
    アライメントの土台となる2次元DP行列を構築する関数。
    """
    m, n = len(source), len(target)
    # 縦(m+1) × 横(n+1) の行列を0で初期化
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 行列の初期境界条件を設定（ベースコストの割り当て）
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    # 各文字の組み合わせにおける最小コストを計算
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                # 文字が完全に一致している場合はコスト変化なしで斜め上から引き継ぐ
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 不一致の場合、「置換（斜め上）」「削除（上）」「挿入（左）」の最小コストに +1 する
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # 置換
                    dp[i - 1][j] + 1,      # 削除
                    dp[i][j - 1] + 1       # 挿入
                )
    return dp

def backtrack_alignment(dp: list[list[int]], source: str, target: str) -> list[tuple[int, int, str]]:
    """
    作成されたDP行列を末尾から逆方向に探索（バックトラック）し、
    校正前後の文字がどのように対応しているかという『最適ルート』を特定する関数。
    
    Returns:
        list[tuple]: (source_idx, target_idx, operation) のリスト。
                     operation は 'match' (一致/置換), 'delete' (削除), 'insert' (挿入)
    """
    i, j = len(source), len(target)
    path = []
    
    # 行列の左上（0, 0）に到達するまで逆算ループを回す
    while i > 0 or j > 0:
        # 境界条件のガード：完全に上端（i=0）に達した場合は、左へ進む（すべて挿入）しかない
        if i == 0:
            path.append(( -1, j - 1, "insert" ))
            j -= 1
        # 完全に左端（j=0）に達した場合は、上へ進む（すべて削除）しかない
        elif j == 0:
            path.append(( i - 1, -1, "delete" ))
            i -= 1
        else:
            # 斜め上、上、左のどこから来たかをコストを元に判定
            current_cost = dp[i][j]
            
            # 1. 文字が一致している、または斜め上からの「置換」ルートだった場合
            if source[i - 1] == target[j - 1] or current_cost == dp[i - 1][j - 1] + 1:
                path.append(( i - 1, j - 1, "match" ))
                i -= 1
                j -= 1
            # 2. 上からの「削除」ルートだった場合（元の文字が消された）
            elif current_cost == dp[i - 1][j] + 1:
                path.append(( i - 1, -1, "delete" ))
                i -= 1
            # 3. 左からの「挿入」ルートだった場合（新しい文字が足された）
            else:
                path.append(( -1, j - 1, "insert" ))
                j -= 1
                
    # 逆順で追跡したため、ルートを正しい時系列（正順）にひっくり返して返却
    path.reverse()
    return path

def restore_char_timestamps(path: list[tuple[int, int, str]], source_words: list) -> list[dict]:
    """
    特定されたルートを元に、校正後のテキスト（Target）の全文字に対して、
    オリジナル（Source）の単語単位タイムスタンプから1文字ずつの時間を復元・再割り当てする関数。
    
    ★超重要: 新しく挿入された文字に対して「前後の確定タイムスタンプから時間を等分割して埋める」
             線形補間ロジックを搭載。
    """
    # まず、Whisperのwordsリストから「1文字単位」の開始・終了時間のタイムスタンプ配列を平坦化して構築
    source_chars_timeline = []
    for w in source_words:
        word_text = w.get("word", w.get("text", ""))
        start = w.get("start", 0.0)
        end = w.get("end", start + 0.1)
        
        # 単語が複数文字で構成されている場合、その単語の時間を文字数で均等分割して1文字あたりの時間を算出
        char_count = len(word_text)
        if char_count > 0:
            duration = (end - start) / char_count
            for k in range(char_count):
                source_chars_timeline.append({
                    "char": word_text[k],
                    "start": start + (duration * k),
                    "end": start + (duration * (k + 1))
                })

    # 校正後の文字に対応するタイムスタンプを格納するリスト（暫定版：挿入文字は仮値が入る）
    refined_timeline = []
    
    # タイムスタンプの全体バウンダリ（時間の極限値）をガード用に取得
    absolute_start = source_chars_timeline[0]["start"] if source_chars_timeline else 0.0
    absolute_end = source_chars_timeline[-1]["end"] if source_chars_timeline else 0.0

    # 最初のパス：確定している一致・置換部分の時間データをマッピング
    for src_idx, tgt_idx, op in path:
        if op == "insert":
            # 挿入された文字は、この時点では対応する時間がないため、後で補間するために仮値をセット
            refined_timeline.append({"start": None, "end": None})
        elif op == "match":
            # 一致または置換の場合、元の文字のタイムスタンプをそのまま継承する
            if 0 <= src_idx < len(source_chars_timeline):
                refined_timeline.append({
                    "start": source_chars_timeline[src_idx]["start"],
                    "end": source_chars_timeline[src_idx]["end"]
                })
            else:
                refined_timeline.append({"start": absolute_start, "end": absolute_end})
        elif op == "delete":
            # 削除された文字は出力側のタイムスタンプ配列には含めない（スルーする）
            continue

    # 第二パス：【線形補間処理】タイムスタンプが None になっている「挿入文字」の時間を決定する
    total_refined_len = len(refined_timeline)
    idx = 0
    
    while idx < total_refined_len:
        if refined_timeline[idx]["start"] is None:
            # 連続して None（挿入文字）が何文字続いているかをカウントする
            start_none_idx = idx
            while idx < total_refined_len and refined_timeline[idx]["start"] is None:
                idx += 1
            end_none_idx = idx - 1
            none_count = (end_none_idx - start_none_idx) + 1
            
            # この None 区間の「直前の有効な時間（左壁）」を探す
            left_time = absolute_start
            if start_none_idx > 0:
                left_time = refined_timeline[start_none_idx - 1]["end"]
                
            # この None 区間の「直後の有効な時間（右壁）」を探す
            right_time = absolute_end
            if idx < total_refined_len:
                right_time = refined_timeline[idx]["start"]
                
            # 安全ガード: もし時間の逆転が起きていた場合は強制的に同期させる
            if right_time < left_time:
                right_time = left_time
                
            # 利用可能な総時間を、Noneの文字数で等分割して均等に割り振る
            available_duration = right_time - left_time
            slice_duration = available_duration / (none_count + 1)
            
            # None だったエリアを等分割したタイムスタンプで埋める
            for c_idx in range(none_count):
                target_pos = start_none_idx + c_idx
                refined_timeline[target_pos]["start"] = left_time + (slice_duration * c_idx)
                refined_timeline[target_pos]["end"] = left_time + (slice_duration * (c_idx + 1))
        else:
            idx += 1

    return refined_timeline

def _align_single_segment(source_words: list, refined_text: str) -> list[dict]:
    """
    【内部用】1つの字幕セグメントに対して、文字と時間を照合する処理
    """
    # =================================================================
    # 【疑似表示時間の設定】
    # AIの校正によって「新しく挿入された文字」に対して割り振る、1文字あたりの疑似的な表示時間（秒）です。
    # 
    # ・ 0.05 : 標準的な設定。1文字あたり0.05秒（20文字で1秒分）のペースで字幕時間を進めます
    # ・ 0.01 : 非常に短い時間。挿入された文字が詰まって一瞬だけ表示されるようになります
    # ・ 0.10 : 少し長めの時間。挿入された文字が多い場合に、字幕の終了時間が後ろに伸びやすくなります
    # =================================================================
    inserted_char_duration = 0.05

    if not source_words or not refined_text.strip():
        return []

    source_text = "".join([item.get("word", item.get("text", "")) for item in source_words])
    dp = calculate_dp_matrix(source_text, refined_text)
    path = backtrack_alignment(dp, source_text, refined_text)
    char_timestamps = restore_char_timestamps(path, source_words)
    
    aligned_result = []
    for i, char in enumerate(refined_text):
        if i < len(char_timestamps):
            aligned_result.append({
                "word": char,  # 後続の処理のために "text" から "word" に変更
                "start": char_timestamps[i]["start"],
                "end": char_timestamps[i]["end"]
            })
        
        else:
            last_time = char_timestamps[-1]["end"] if char_timestamps else 0.0
            aligned_result.append({
                "word": char,  # こちらも "word" に変更
                "start": last_time,
                "end": last_time + inserted_char_duration
            })
            
    return aligned_result

def align_text_and_timestamps(source_words: list, refined_text: str) -> list[dict]:
    """
    外部から呼び出されるアライメントのメイン関数。
    1行分の「Whisperの単語時間データ」と「AI校正後の文字列」を受け取り、時間を同期します。
    
    Args:
        source_words: [{"word": "明日の", "start": 0.0, "end": 1.0}, ...] のような時間データ
        refined_text: "明日の天気は晴れです" のようなAI校正済みの文字列
        
    Returns:
        同期が完了した [{"word": "明", "start":...}, {"word": "日", "start":...}, ...] のリスト
    """
    # 1. 校正後のテキストが空の場合はリストを空にして返す
    if not refined_text.strip():
        return []

    # 2. Whisperの元データに最初から単語時間が無い場合
    if not source_words:
        # 万が一時間が消失していたら、呼び出し元でセグメント枠の時間で代替するため、ここではエラーを投げる
        raise AlignmentError("元データの単語タイムスタンプ(source_words)が存在しません。")

    try:
        # 3. 内部関数の _align_single_segment に処理を任せる（この関数は既にファイル内にあるはずです）
        aligned_words = _align_single_segment(source_words, refined_text)
        
        if not aligned_words:
            raise AlignmentError("時間同期の結果が空になりました。")
            
        return aligned_words
        
    except Exception as e:
        raise AlignmentError(f"時間同期中にエラーが発生しました: {e}")