"""
srtwriter.py
【最終工程】確定したテキストとタイムスタンプを、動画編集ソフト等で読み込めるSRT形式のファイルとして出力するモジュール。

責務:
  - タイムスタンプ（秒数）を SRT規格（HH:MM:SS,mmm）に変換する
  - BudouX等で整形された行データを連番とともにテキストファイルへ書き出す
"""

import os
from tqdm import tqdm
from src.exceptions import FileWriteError

def format_timestamp(seconds: float) -> str:
    """
    ただの秒数（例: 65.5秒）を、SRT形式の文字列（例: 00:01:05,500）に変換します。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds % 1) * 1000))

    if milliseconds >= 1000:
        milliseconds -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def write_srt_file(formatted_lines: list, output_srt_path: str) -> None:
    """
    整形済みの行データリストを読み込み、SRTファイルとして保存します。
    
    Args:
        formatted_lines: [{"text": "字幕テキスト", "start": 0.0, "end": 2.5}, ...] の形式のリスト
        output_srt_path: 保存先のSRTファイルパス
    """
    tqdm.write(f"[*] SRTファイルの生成を開始します。書き出し先: {output_srt_path}")
    
    try:
        with open(output_srt_path, "w", encoding="utf-8") as f:
            for index, line_data in enumerate(formatted_lines, start=1):
                text = line_data.get("text", "").strip()
                if not text:
                    continue
                
                # 1. 字幕の連番を出力
                f.write(f"{index}\n")
                
                # 2. 開始時間と終了時間を出力
                start_time = format_timestamp(line_data["start"])
                end_time = format_timestamp(line_data["end"])
                f.write(f"{start_time} --> {end_time}\n")
                
                # 3. 字幕テキスト本体と、区切りのための空行を出力
                f.write(f"{text}\n\n")

        tqdm.write(f"[+] SRTファイルの出力が完了しました。")

    except OSError as e:
        raise FileWriteError(
            f"SRTファイルの書き込みに失敗しました: {output_srt_path} / 原因: {e}"
        ) from e