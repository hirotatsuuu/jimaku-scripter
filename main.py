"""
main.py
プログラムのエントリーポイント。

引数の解析のみを担当し、処理の実体は src/pipeline.py の run() に委譲します。
設定値のデフォルトは src/config.py に一元管理されています。
"""

import argparse  # コマンドライン引数を受け取って解析するための標準ライブラリ
import os        # 入力ファイルの存在確認に使うライブラリ
import sys       # ファイルが見つからない場合の強制終了に使うライブラリ

from src.config import (
    DEFAULT_AUDIO_FILE,
    DEFAULT_DICT_FILE,
    DEFAULT_FILLER_FILE,
    DEFAULT_PROMPT_FILE,
    DEFAULT_MODEL_SIZE,
    BATCH_SIZE_LLM,
    MIN_CHAR_LEN,
    MAX_CHAR_LEN
)
from src.pipeline import run


def main() -> None:
    """引数を解析してパイプラインを起動する関数"""
    parser = argparse.ArgumentParser(
        description="音声ファイルを生成AIで最適化してSRT字幕を自動出力するスクリプト"
    )
    parser.add_argument(
        "input_file", 
        nargs="?", 
        default=DEFAULT_AUDIO_FILE,
        help="入力ファイル（動画または音声）のパス"
    )
    parser.add_argument(
        "--dict",
        default=DEFAULT_DICT_FILE,
        help="優先単語リスト（dictionary.txt）のパス"
    )
    parser.add_argument(
        "--filler",
        default=DEFAULT_FILLER_FILE,
        help="フィラーリスト（filler.txt）のパス"
    )
    parser.add_argument(
        "--prompt", 
        default=DEFAULT_PROMPT_FILE, 
        help="LLM プロンプト（prompt.txt）のパス"
    )
    parser.add_argument(
        "--model",  
        default=DEFAULT_MODEL_SIZE,  
        help="Whisper のモデルサイズ指定（tiny/base/small/medium/large）"
    )
    parser.add_argument(
        "-min", "--min-chars",
        type=int, default=MIN_CHAR_LEN,
        dest="min_char_len",
        help=f"字幕 1 行あたりの最低文字数（デフォルト: {MIN_CHAR_LEN}）"
    )
    parser.add_argument(
        "-max", "--max-chars",
        type=int, default=MAX_CHAR_LEN,
        dest="max_char_len",
        help=f"字幕 1 行あたりの最大文字数（デフォルト: {MAX_CHAR_LEN}）"
    ) 
    parser.add_argument(
        "--batch-size",
        type=int, default=BATCH_SIZE_LLM,
        dest="batch_size_llm",
        help=f"LLM に一度に送るセグメント数（デフォルト: {BATCH_SIZE_LLM}）"
    )
    parser.add_argument(
        "-f",
        action="store_true",
        dest="run_filler",  
        help="フィラー除去処理をする"
    )
    parser.add_argument(
        "--llm",  
        action="store_true",
        dest="run_llm",
        help="LLMによる校正をする"
    )
    parser.add_argument(
        "--deberta",  
        action="store_true",
        dest="run_deberta",
        help="DeBERTaによる校正をする"
    )
    parser.add_argument(
        "--dp",  
        action="store_true",
        dest="run_dp",
        help="AIによる校正後にDP処理をする（必ず比較対象のAI校正後のテキストがある）"
    )
    parser.add_argument(
        "--fa",  
        action="store_true",
        dest="run_fa",
        help="AIによる校正後にForced Alignment処理をする"
    )
    # 何も指定が無い場合は、音声ファイル➡Whisper音声認識➡SRTファイルの生成のみ

    args = parser.parse_args()

    # 指定されたファイルが実在するか確認する（パイプライン開始前に早期終了させる）
    if not os.path.exists(args.input_file):
        print(f"[エラー] 入力ファイルが見つかりません: {args.input_file}")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()