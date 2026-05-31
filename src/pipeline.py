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
    try:
        paths = build_output_paths(args.input_file)
    except Exception as e:
        tqdm.write(f"[エラー] 出力先の準備に失敗しました: {e}")
        sys.exit(1)

    # 全工程のステップ数をモードに応じて決定する
    total_steps = 4 # 基本工程: パス準備, 前処理, Whisper, フィラー除去
    if args.mode in ["llm", "default", "all"]:
        total_steps += 1 # LLM工程
    if args.mode in ["default", "all"]:
        total_steps += 2 # DeBERTa校正, DP同期
    total_steps += 1 # 最終SRT出力

    with tqdm(total=total_steps, desc="全体進捗", unit="step") as pbar:
        
        # ---------------------------------------------------------
        # [工程 A] 音声前処理
        # ---------------------------------------------------------
        tqdm.write("\n--- [1] 音声前処理 ---")
        target_audio_file = paths["extracted_audio"]
        try:
            target_audio_file = process_audio(args.input_file, target_audio_file)
        except Exception as e:
            tqdm.write(f"[エラー] {e}")
            sys.exit(1)
        pbar.update(1)

        # ---------------------------------------------------------
        # [工程 B] Whisper 音声認識
        # ---------------------------------------------------------
        tqdm.write("\n--- [2] 音声認識 (Whisper) ---")
        word_dict = load_word_dictionary(args.dict)
        try:
            raw_segments, whisper_time = run_whisper_transcribe(target_audio_file, word_dict, args.model)
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
            tqdm.write("\n--- [4] LLM 文脈校正 ---")
            try:
                llm_segments, llm_time = refine_context_with_llm(
                    current_segments, args.prompt, args.batch_size_llm
                )
                current_segments = llm_segments
                
                # LLM出力データの保存
                save_segments_as_json(current_segments, paths["refined_json"])
                save_segments_as_plaintext(current_segments, paths["refined_txt"])
            except Exception as e:
                tqdm.write(f"[警告] LLM処理で問題が発生しました。元のデータを維持します: {e}")
            pbar.update(1)

            if args.mode == "llm":
                tqdm.write("[*] LLMモードのため、DeBERTa校正をスキップして出力へ進みます。")
                final_segments = current_segments
            else:
                # ---------------------------------------------------------
                # [工程 E] DeBERTa 穴埋め校正 (default, all 向け)
                # ---------------------------------------------------------
                tqdm.write("\n--- [5] DeBERTa 精度補強校正 ---")
                proofread_segments = proofread_text(current_segments)
                pbar.update(1)
                
                # ---------------------------------------------------------
                # [工程 F] DP タイムスタンプ同期照合
                # ---------------------------------------------------------
                tqdm.write("\n--- [6] DP タイムスタンプ同期 ---")
                aligned_segments = align_text_and_timestamps(cleaned_segments, proofread_segments)
                final_segments = aligned_segments
                pbar.update(1)

        # ---------------------------------------------------------
        # [工程 G] BudouX 整形と SRT 出力
        # ---------------------------------------------------------
        tqdm.write("\n--- [最終工程] 文節整形とSRT書き出し ---")
        
        # 読点や文字数に応じて美しく改行する処理
        formatted_lines = format_segments_to_lines(
            final_segments, args.min_char_len, args.max_char_len
        )
        
        # 指定されたモードに応じたパスにSRTを出力する
        output_srt_path = paths["whisper_srt"] if args.mode == "wsp" else paths["refined_srt"]
        
        write_srt_file(formatted_lines, output_srt_path)
        pbar.update(1)

    # 処理完了
    tqdm.write("\n" + "="*50)
    tqdm.write(f"[*] すべての処理が完了しました！ 総所要時間: {time.perf_counter() - start_time:.2f} 秒")
    tqdm.write("="*50)