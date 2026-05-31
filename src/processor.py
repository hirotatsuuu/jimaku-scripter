"""
processor.py
1. 音声前処理：FFmpegで動画/音声を 16kHz WAV化
入力ファイル（動画や音声）を受け取り、Whisperが最も得意とする標準フォーマットに変換するモジュール。
"""

import os
import subprocess
from src.config import VIDEO_EXTENSIONS, OUTPUT_AUDIO_DIR
from src.exceptions import AudioExtractionError, FfmpegNotFoundError

def process_audio(input_file_path: str) -> str:
    """入力ファイルをチェックし、動画であれば 16kHz / モノラル / WAV に変換してそのパスを返す関数。
    音声ファイルであれば何もせずそのままパスを返します。
    """
    # 1. ファイルの拡張子を取得して小文字にする
    input_ext = os.path.splitext(input_file_path)[1].lower()
    
    # 2. 拡張子が動画用のもの（.mp4, .mkv等）に含まれているかチェック
    if input_ext not in VIDEO_EXTENSIONS:
        # 動画ではない（既に音声ファイルなど）の場合は、何もせずそのままのパスを返す
        return input_file_path
        
    # 3. 動画だった場合の一時WAV出力先パスを組み立てる
    stem = os.path.splitext(os.path.basename(input_file_path))[0]
    output_wav_path = os.path.join(OUTPUT_AUDIO_DIR, f"{stem}_temp.wav")
    
    # すでに一時ファイルが存在する場合は、新しくFFmpegを回さずにそのパスを再利用する（高速化）
    if os.path.exists(output_wav_path):
        return output_wav_path

    # 4. FFmpegの外部コマンドを組み立ててバックグラウンドで実行
    # （※ ここの詳細なコマンド実装は、次の processor.py の編集フェーズで作り込みます）
    print(f"[*] 動画ファイルを検出したため、音声を抽出します: {input_file_path}")
    
    # 仮のパス返却（のちほど実装を肉付けします）
    return output_wav_path