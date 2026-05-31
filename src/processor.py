"""
processor.py
【前処理工程】動画や音声ファイルから、AIが処理しやすい形式（16kHz m4a）に変換するモジュール。

責務:
  - 入力ファイルが動画かどうかを判定する
  - 動画の場合はFFmpegを使って音声ストリームのみを抽出する
  - 音声のサンプリングレートをWhisper推奨の16kHzに変換する
"""

import os
import subprocess
from tqdm import tqdm
from src.exceptions import AudioExtractionError, FfmpegNotFoundError
from src.config import VIDEO_EXTENSIONS

def is_video_file(file_path: str) -> bool:
    """
    指定されたファイルが動画かどうかを拡張子から判定する関数です。
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower() in VIDEO_EXTENSIONS

def process_audio(input_file: str, output_audio_path: str) -> str:
    """
    入力ファイルをチェックし、必要に応じて音声抽出・変換を行います。
    
    Args:
        input_file: ユーザーが指定した入力ファイルのパス
        output_audio_path: 変換後の音声を保存するパス
        
    Returns:
        Whisperに渡すべき最終的な音声ファイルのパス
    """
    tqdm.write("[*] ffmpegによる音声変換処理を開始します...")

    # ログ出力が綺麗になるよう、Windows特有の「\」を「/」に統一する
    input_file_clean = input_file.replace("\\", "/")
    output_audio_clean = output_audio_path.replace("\\", "/")

    # 入力ファイルが既に m4a の場合、再変換をスキップして元のファイルをそのまま返す
    # メモ：サンプリングレートを16kHzに統一するため、音声ファイルであってもFFmpegを通したほうがよい
    if input_file_clean.lower().endswith(".m4a"):
        tqdm.write(f"[*] 入力ファイルは既に m4a 形式なので前処理をスキップします: {input_file_clean}")
        return input_file_clean, False  # False = 一時ファイルではない（元のファイル）

    tqdm.write(f"[*] 音声前処理対象ファイル: {input_file}")
    
    command = [
        "ffmpeg",
        "-y",                   # 既にファイルが存在する場合は強制的に上書きする
        "-i", input_file,       # 入力ファイルを指定
        "-vn",                  # 映像（Video）ストリームを除外する（No Video）
        "-acodec", "aac",       # 音声コーデックをAACに指定して再エンコードする
        "-ar", "16000",         # 音声の周波数を16kHz（Whisperが最も得意とする形式）に統一する
        "-ac", "1",             # モノラル（1チャンネル）に変換する
        "-b:a", "128k",         # 音質を128kbpsに設定する
        output_audio_path,      # 出力先のファイルパス
    ]

    try:
        # コマンドをバックグラウンドで実行します
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        tqdm.write(f"[*] 音声の最適化・抽出が完了しました: {output_audio_clean}")
        return output_audio_clean, True # True = この処理で一時的に生成したファイルである

    except subprocess.CalledProcessError as e:
        # FFmpegの実行が失敗した場合（ファイルが壊れているなど）
        stderr_message = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else "詳細不明"
        raise AudioExtractionError(
            f"音声の処理に失敗しました。ファイルが破損している可能性があります。\n"
            f"エラー詳細: {stderr_message}"
        ) from e

    except FileNotFoundError as e:
        # PCにFFmpegがインストールされていない場合
        raise FfmpegNotFoundError(
            "PCに 'ffmpeg' がインストールされていません。"
            "コマンドプロンプト等でインストールを行ってください。"
        ) from e