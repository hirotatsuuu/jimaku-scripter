"""
pipeline.py
全体のデータフローをコントロールする司令塔。

責務:
  - モードごとの処理分岐（default, wsp, llm, all）
  - 各モジュールの呼び出しとデータの中継ぎ
  - 全体進捗バー（tqdm）の管理とコンソールへのログ出力
  - 各工程完了ごとの中間ファイル（JSON, TXT）の保存
"""

import os
import sys
import copy
import time
from tqdm import tqdm

from src.config import REMOVE_TEMP_AUDIO
from src.utils import build_output_paths, save_segments_as_json, save_segments_as_plaintext
from src.processor import process_audio
from src.transcriber import load_word_dictionary, load_filler_list, run_whisper_transcribe, clean_fillers_keep_timing
from src.refiner import refine_context_with_llm
from src.proofreader import proofread_text
from src.aligner import align_text_and_timestamps
from src.formatter import format_segments_to_lines
from src.srtwriter import write_srt_file

def run(args) -> None:
    start_time = time.perf_counter()
    tqdm.write("="*50)
    tqdm.write("[*] 字幕生成パイプラインを起動しました")
    tqdm.write(f"[*] 選択モード: {args.mode.upper()}")
    tqdm.write("="*50)

    # 1. パスの構築とディレクトリ準備
    # ユーザーが入力したパスの「\」を「/」に統一する
    clean_input_file = args.input_file.replace("\\", "/")
    try:
        paths = build_output_paths(clean_input_file)
    except Exception as e:
        tqdm.write(f"[エラー] 出力先の準備に失敗しました: {e}")
        sys.exit(1)

    # 全工程のステップ数をモードに応じて決定する
    total_steps = 4  # 基本工程: パス準備, 前処理, Whisper, フィラー除去

    if args.mode in ["llm", "all"]:  # LLMを通るのは all と llm モードだけ
        total_steps += 1 # LLM工程
    if args.mode in ["default", "all"]:
        total_steps += 2 # DeBERTa校正, DP同期
    total_steps += 1 # 最終SRT出力

    # 音声クリーンアップ用に変数を準備
    target_audio_file = ""
    is_temp_audio = False

    try:
        with tqdm(total=total_steps, desc="全体進捗", unit="step") as pbar:
            
            # ---------------------------------------------------------
            # [工程 A] 音声前処理
            # ---------------------------------------------------------
            tqdm.write("\n--- [1] 音声前処理 ---")
            extracted_audio_path = paths["extracted_audio"]
        
            try:
                # 戻り値を2つ受け取るように修正（パス, 一時ファイルフラグ）
                target_audio_file, is_temp_audio = process_audio(clean_input_file, extracted_audio_path)
            except Exception as e:
                tqdm.write(f"[エラー] {e}")
                sys.exit(1)
            pbar.update(1)

            # ---------------------------------------------------------
            # [工程 B] Whisper 音声認識
            # ---------------------------------------------------------
            tqdm.write("\n--- [2] 音声認識 (Whisper) ---")
            word_dict = load_word_dictionary(args.dict)
            tqdm.write("[*] Whisper 音声認識処理を実行中...")

            try:
                raw_segments, whisper_time = run_whisper_transcribe(target_audio_file, word_dict, args.model)
                tqdm.write(f"[+] Whisper 音声認識処理 完了 (所要時間: {whisper_time:.2f} 秒)")
            except Exception as e:
                tqdm.write(f"[エラー] {e}")
                sys.exit(1)
            
            # 生データの保存
            save_segments_as_json(raw_segments, paths["whisper_json"])
            save_segments_as_plaintext(raw_segments, paths["whisper_txt"])
            pbar.update(1)

            # ---------------------------------------------------------
            # [工程 C] フィラー除去
            # ---------------------------------------------------------
            tqdm.write("\n--- [3] フィラー除去 ---")
            filler_list = load_filler_list(args.filler)
            cleaned_segments = clean_fillers_keep_timing(raw_segments, filler_list)
            pbar.update(1)

            current_segments = copy.deepcopy(cleaned_segments)

            # モード「wsp」の場合はここでSRT出力へスキップ
            if args.mode == "wsp":
                tqdm.write("[*] WSPモードのため、AI校正処理をスキップして出力へ進みます。")
                final_segments = current_segments

            else:
                # ---------------------------------------------------------
                # [工程 D] LLM 校正
                # ---------------------------------------------------------
                # defaultモードではスキップされる
                if args.mode in ["all", "llm"]:  
                    tqdm.write("\n--- [4] LLM 文脈校正 ---")
                    tqdm.write("[*] LLM 校正処理を実行中...")
                    try:
                        llm_segments, llm_time = refine_context_with_llm(
                            current_segments, args.prompt, args.batch_size_llm
                        )
                        current_segments = llm_segments
                        tqdm.write(f"[+] LLM 校正処理 完了 (所要時間: {llm_time:.2f} 秒)")
                        
                        # LLM出力データの保存
                        save_segments_as_json(current_segments, paths["refined_json"])
                        save_segments_as_plaintext(current_segments, paths["refined_txt"])

                    except Exception as e:
                        tqdm.write(f"[警告] LLM処理で問題が発生しました。元のデータを維持します: {e}")
                    pbar.update(1)
                else:
                    tqdm.write("\n[*] DEFAULTモードのため、LLM文脈校正処理をスキップします。")

                if args.mode == "llm":
                    tqdm.write("[*] LLMモードのため、DeBERTa校正をスキップして出力へ進みます。")
                    final_segments = current_segments
                else:
                    # ---------------------------------------------------------
                    # [工程 E] DeBERTa 穴埋め校正 (default, all 向け)
                    # ---------------------------------------------------------
                    tqdm.write("\n--- [5] DeBERTa 精度補強校正 ---")
                    tqdm.write("[*] DeBERTa 校正処理を実行中...")

                    proofread_segments, deberta_time = proofread_text(current_segments)
                    tqdm.write(f"[+] DeBERTa 校正処理 完了 (所要時間: {deberta_time:.2f} 秒)")
                    pbar.update(1)
                    
                    # ---------------------------------------------------------
                    # [工程 F] DP タイムスタンプ同期照合
                    # ---------------------------------------------------------
                    tqdm.write("\n--- [6] DP タイムスタンプ同期 ---")
                    
                    # LLMをスキップした場合は cleaned_segments
                    # LLMを通った場合は LLMの出力 をベースに照合する
                    base_segments = cleaned_segments if args.mode == "default" else current_segments
                    aligned_segments = align_text_and_timestamps(base_segments, proofread_segments)

                    final_segments = aligned_segments
                    pbar.update(1)

            # ---------------------------------------------------------
            # [工程 G] BudouX 整形と SRT 出力
            # ---------------------------------------------------------
            tqdm.write("\n--- [最終工程] 文節整形とSRT書き出し ---")
            
            # 【比較用】Whisper生データ（AI校正前）のSRT出力
            tqdm.write("[*] Whisper版 SRTファイルを生成中...")
            # (cleaned_segments には フィラー除去直後 のデータが安全に保管されています)
            # 読点や文字数に応じて美しく改行する処理
            formatted_whisper = format_segments_to_lines(
                cleaned_segments, args.min_char_len, args.max_char_len
            )
            write_srt_file(formatted_whisper, paths["whisper_srt"])

            # ② 【本番用】AI校正版のSRT出力 (wspモード以外の場合のみ)
            if args.mode in ["default", "all", "llm"]:
                tqdm.write("[*] AI校正版 SRTファイルを生成中...")
                # 読点や文字数に応じて美しく改行する処理
                formatted_refined = format_segments_to_lines(
                    final_segments, args.min_char_len, args.max_char_len
                )
                write_srt_file(formatted_refined, paths["refined_srt"])

            pbar.update(1)

            # 進捗バーがまだ100%に達していない場合、強制的に最大値まで進めて終了する
            if 'pbar' in locals() and pbar is not None:
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)  # 残りのステップ数を一気に進める
                pbar.refresh()                        # 画面表示を最新（100%）に更新
                pbar.close()                          # 進捗バーの制御を安全に終了

    finally:
        # ---------------------------------------------------------
        # [後始末クリーンアップ処理]
        # ---------------------------------------------------------
        # エラーで異常終了した場合でも、必ずここを通過してファイルを消すか判断する
        if is_temp_audio and REMOVE_TEMP_AUDIO:
            try:
                if os.path.exists(target_audio_file):
                    os.remove(target_audio_file)
                    tqdm.write(f"\n[*] 設定(REMOVE_TEMP_AUDIO=True)に基づき、一時音声ファイルを削除しました: {target_audio_file}")
            except OSError as e:
                tqdm.write(f"\n[警告] 一時音声ファイルの削除中にエラーが発生しました: {e}")
        elif is_temp_audio and not REMOVE_TEMP_AUDIO:
            tqdm.write(f"\n[*] 設定(REMOVE_TEMP_AUDIO=False)に基づき、一時音声ファイルは保持されました: {target_audio_file}")

    tqdm.write("\n" + "="*50)
    tqdm.write(f"[*] すべての処理が完了しました！ 総所要時間: {time.perf_counter() - start_time:.2f} 秒")
    tqdm.write("="*50)