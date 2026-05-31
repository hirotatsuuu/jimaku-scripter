"""
pipeline.py
システムの全工程を統括する司令塔モジュール。
各処理を独立したブロックとして扱い、引数のフラグに応じて実行を制御します。

責務:
  - 実行時引数ごとの処理分岐（default, filler, llm, deberta, dp, fa）
  - 各モジュールの呼び出しとデータの中継ぎ
  - 全体進捗バー（tqdm）の管理とコンソールへのログ出力
  - Whisperファイルの保存（text, raw, srt）と完成したファイルの保存（text, raw, srt）
"""

import os
import sys
import copy
import time
from tqdm import tqdm

from src.config import REMOVE_TEMP_AUDIO, VIDEO_EXTENSIONS
from src.exceptions import AudioExtractionError, FfmpegNotFoundError, FileWriteError
from src.transcriber import load_filler_list, load_word_dictionary, run_whisper_transcribe, extract_audio_from_video
from src.formatter import write_srt_file
from src.utils import build_output_paths, save_segments_as_json, save_segments_as_plaintext
from src.proofreader import proofread_text
from src.forcedaligner import run_forced_alignment
from src.cleaner import remove_fillers
from src.aligner import align_text_and_timestamps
from src.refiner import refine_context_with_llm


def run(args) -> None:
    start_time = time.perf_counter()
    tqdm.write("="*50)
    tqdm.write("[*] whisper-llm-srt 字幕生成スクリプトを起動")
    tqdm.write("="*50)

    # ユーザーが入力したパスの「\」を「/」に統一する
    # clean_input_file = args.input_file.replace("\\", "/")

    # 1. パスの構築とディレクトリ準備
    try:
        paths = build_output_paths(args.input_file)
    except FileWriteError as e:
        tqdm.write(f"[エラー] 出力ディレクトリの作成に失敗しました: {e}")
        sys.exit(1)

    # 2. 動画の場合は音声を抽出
    target_audio_file = args.input_file
    is_video_input = os.path.splitext(args.input_file)[1].lower() in VIDEO_EXTENSIONS
    if is_video_input:
        tqdm.write("\n--- [1] 音声前処理 ---")
        extracted_audio_path = paths["extracted_audio"]
        try:
            extract_audio_from_video(args.input_file, extracted_audio_path)
            target_audio_file = extracted_audio_path
        except Exception as e:
            tqdm.write(f"[エラー] 音声抽出に失敗しました: {e}")
            sys.exit(1)

    # -----------------------------------------------------------------
    #  進行ステップ数の計算（ONになっているフラグの数だけプログレスバーが増えます）
    # -----------------------------------------------------------------
    total_steps = 1 # Whisper音声認識 (必須)
    if getattr(args, "run_filler", False): total_steps += 1
    if getattr(args, "run_llm", False): total_steps += 1
    if getattr(args, "run_deberta", False): total_steps += 1
    if getattr(args, "run_dp", False) or getattr(args, "run_fa", False): total_steps += 1
    total_steps += 1 # 最終工程 (必須)

    try:
        with tqdm(total=total_steps, desc="全体進捗", unit="step") as pbar:
            
            # =================================================================
            # 【工程 2】 Whisper 音声認識 (常に実行)
            # =================================================================
            tqdm.write("\n--- [2] Whisper 音声認識 ---")
            word_dict = load_word_dictionary(args.dict)

            tqdm.write("[*] Whisper 音声認識処理を実行中...")

            # 生の音声からテキストと時間を抽出
            whisper_segments, whisper_time = run_whisper_transcribe(
                target_audio_file, word_dict, args.model 
            )
            tqdm.write(f"[+] Whisper 音声認識処理 完了 (所要時間: {whisper_time:.2f} 秒)")

            # 以降の全工程で、常にこの変数に「最新の磨かれたデータ」を格納してリレーします
            current_segments = whisper_segments
            pbar.update(1)

            # 【中間保存】引数無しの時はこれが最終結果となり、引数ありの時は比較用の生データ(_wsp)として機能します
            write_srt_file(current_segments, paths["whisper_srt"], args.min_char_len, args.max_char_len)
            save_segments_as_json(current_segments, paths["whisper_json"])
            save_segments_as_plaintext(current_segments, paths["whisper_txt"])

            # =================================================================
            # 【工程 3】 フィラー除去
            # =================================================================
            if getattr(args, "run_filler", False):
                tqdm.write("\n--- [3] フィラー除去 ---")
                filler_list = load_filler_list(args.filler)

                # ディープコピーを渡すことで、もしこの処理でエラーが起きても元のデータは無傷で保たれます
                current_segments = remove_fillers(copy.deepcopy(current_segments), filler_list)
                pbar.update(1)

            # 💡 【重要】タイムスタンプ復元のための「時間の正解データ」を保護
            # AIがテキストを改変する直前の、正確な時間枠を持ったデータをここで保管します
            reference_segments = copy.deepcopy(current_segments)

            # =================================================================
            # 【工程 4】 LLM 文脈校正
            # =================================================================
            if getattr(args, "run_llm", False):
                tqdm.write("\n--- [4] LLM 文脈校正 ---")
                tqdm.write("[*] LLMによる校正処理を実行中...")

                current_segments, llm_time = refine_context_with_llm(
                    copy.deepcopy(current_segments), args.prompt, args.batch_size_llm
                )

                tqdm.write(f"[+] LLM 校正処理 完了 (所要時間: {llm_time:.2f} 秒)")
                pbar.update(1)

            # =================================================================
            # 【工程 5】 DeBERTa 精度補強校正
            # =================================================================
            if getattr(args, "run_deberta", False):
                tqdm.write("\n--- [5] DeBERTa 精度補強校正 ---")
                tqdm.write("[*] DeBERTaによる校正処理を実行中...")

                current_segments, deberta_time = proofread_text(
                    copy.deepcopy(current_segments)
                )
                tqdm.write(f"[+] DeBERTa 校正処理 完了 (所要時間: {deberta_time:.2f} 秒)")
                pbar.update(1)

            # =================================================================
            # 【工程 6】 タイムスタンプ同期 (DP または FA)
            # =================================================================
            # ※ AI校正をすると文字数や表現が変わって時間がズレるため、元の時間(reference)と照合します
            if getattr(args, "run_dp", False) or getattr(args, "run_fa", False):
                
                if getattr(args, "run_fa", False):
                    # FA（Forced Alignment）フラグが立っている場合
                    tqdm.write("\n--- [6] Forced Alignment タイムスタンプ同期 ---")

                    current_segments = run_forced_alignment(
                        reference_segments, copy.deepcopy(current_segments)
                    )
                else:
                    # DPフラグのみが立っている場合（既存のDPアルゴリズムでリスト全体を回す）
                    tqdm.write("\n--- [7] DP(動的計画法) タイムスタンプ同期  ---")

                    aligned_segments = copy.deepcopy(current_segments)

                    for i in range(min(len(reference_segments), len(aligned_segments))):
                        try:
                            # 1行ずつ既存のDP関数に渡して時間を再計算
                            aligned_words = align_text_and_timestamps(
                                reference_segments[i].get("words", []), 
                                aligned_segments[i].get("text", "")
                            )
                            aligned_segments[i]["words"] = aligned_words
                        except Exception:
                            # エラー時は安全のためセグメント枠の時間を借りる
                            aligned_segments[i]["words"] = [{"word": aligned_segments[i].get("text", ""), "start": reference_segments[i].get("start", 0.0), "end": reference_segments[i].get("end", 0.0)}]
                    current_segments = aligned_segments
                    
                pbar.update(1)
            
            # =================================================================
            # 【最終工程】 BudouX 整形と SRT 出力
            # =================================================================
            # 引数に関係なく、ここまでリレーされてきた「current_segments」が最終成果物となります。
            # フラグ無しならWhisperのまま、フラグ有りなら全ての校正を乗り越えたデータが出力されます。
            tqdm.write("\n--- [8] BudouX 文節整形 SRT 書き出し ---")
            tqdm.write(f"[*] ファイル保存先: {paths['final_srt']}")
            
            write_srt_file(current_segments, paths["final_srt"], args.min_char_len, args.max_char_len)
            save_segments_as_json(current_segments, paths["final_json"])
            save_segments_as_plaintext(current_segments, paths["final_txt"])
            
            pbar.update(1)

    finally:
        # ---------------------------------------------------------
        # [後始末クリーンアップ処理]
        # ---------------------------------------------------------
        tqdm.write("\n--- [9] 音声後処理 ---")

        # エラーで異常終了した場合でも、必ずここを通過してファイルを消すか判断する
        if is_video_input and REMOVE_TEMP_AUDIO:
            try:
                if os.path.exists(target_audio_file):
                    os.remove(target_audio_file)
                    tqdm.write(f"[*] 生成した音声ファイルを削除: {target_audio_file}")
            except OSError as e:
                tqdm.write(f"[警告] 一時音声ファイルの削除中にエラーが発生しました: {e}")

        elif is_video_input and not REMOVE_TEMP_AUDIO:
            tqdm.write(f"[*] 生成した音声ファイルを保持: {target_audio_file}")

        # 進捗バーがまだ100%に達していない場合、強制的に最大値まで進めて終了する
        if 'pbar' in locals() and pbar is not None:
            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)  # 残りのステップ数を一気に進める
            pbar.refresh()                        # 画面表示を最新（100%）に更新
            pbar.close()                          # 進捗バーの制御を安全に終了

    tqdm.write("\n" + "="*50)
    tqdm.write(f"[*] 字幕生成処理 完了 （総所要時間: {time.perf_counter() - start_time:.2f} 秒）")
    tqdm.write("[*] 字幕生成スクリプトを終了しました")
    tqdm.write("="*50)