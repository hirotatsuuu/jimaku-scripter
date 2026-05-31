"""
proofreader.py
【校正工程】DeBERTaを用いてテキストの不自然な箇所（確信度不足）を検知し、
文脈から正しい単語を推測・穴埋め補完するモジュール。
"""

import os
import warnings

# 環境変数での警告抑制
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Pythonの警告システムを使って、Hugging Face絡みのWarningを根こそぎ無視する
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

import logging
# さらにロガーの出力レベルもエラーのみに絞る
try:
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

import copy
import time  # 処理時間計測用に追加
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from src.config import DEBERTA_MODEL_NAME

def _proofread_single_text(text: str, tokenizer: AutoTokenizer, model: AutoModelForMaskedLM, threshold: float) -> tuple[str, int]:
    """
    1つのテキスト（文字列）に対してDeBERTa校正を行う内部用の専用関数。
    外部からは直接呼び出さず、メイン関数からループで呼び出されます。
    """
    # =================================================================
    # 【安全ガード：最大許容文字数差】
    # AIの校正前後で、1文あたりの文字数がこの値（文字数）以上変わった場合は、
    # AIが暴走（過剰な削除や過剰な挿入）したとみなして元のテキストを維持します。
    # 
    # ・ 15 : 標準的なバランス（おすすめ）
    # ・ 5  : 非常に厳格。少しでも長さが変わったら元に戻す（安全重視）
    # ・ 50 : 非常に緩い。大幅な言い換えや文章の要約を許容する
    # =================================================================
    max_char_diff = 15

    if not text.strip():
        return text, 0

    # 長い文章によるエラーを防ぐため、句点で分割
    sentences = text.split("。")
    corrected_sentences = []
    total_low_confidence_count = 0

    for sentence in sentences:
        if not sentence.strip():
            continue
            
        current_sentence = sentence + "。"
        
        # テキストをAIが理解できるトークンに変換
        inputs = tokenizer(current_sentence, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        
        # 予測の実行
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits[0]
            
        probabilities = torch.softmax(predictions, dim=-1)

        # 修正候補の位置（インデックス）と、その元の信頼度をペアで保存する
        mask_candidates = []
        
        # 確信度不足のトークンを検出
        for idx in range(1, len(input_ids) - 1):
            token_id = input_ids[idx].item()
            original_word_prob = probabilities[idx][token_id].item()
            
            if original_word_prob < threshold:
                # カッコを2重にして (インデックス, 信頼度) の「1組のペア」として追加します
                mask_candidates.append((idx, original_word_prob))

        total_low_confidence_count += len(mask_candidates)

        # マスク候補がない場合はそのまま採用
        if len(mask_candidates) == 0:
            corrected_sentences.append(current_sentence)
            continue

        # 不自然な箇所を [MASK] に置き換えて再予測
        masked_input_ids = input_ids.clone()
        # ペア（タプル）から「位置」と「信頼度」をバラバラに分解して受け取ります
        # （後ろの _ には信頼度が入りますが、ここでは使わないので _ にしています）
        for pos, _ in mask_candidates:
            masked_input_ids[pos] = tokenizer.mask_token_id
            
        with torch.no_grad():
            masked_inputs = {"input_ids": masked_input_ids.unsqueeze(0)}
            if "attention_mask" in inputs:
                masked_inputs["attention_mask"] = inputs["attention_mask"]
            
            masked_outputs = model(**masked_inputs)
            masked_predictions = masked_outputs.logits[0]

        # 最適な単語を当てはめる
        final_input_ids = input_ids.clone()
    
        # 修正の適用とビフォーアフター（＋信頼度）の表示
        for pos, original_prob in mask_candidates:

            # 修正「前」の単語を翻訳（デコード）して取得
            original_id = input_ids[pos].item()
            original_word = tokenizer.decode([original_id]).strip()
        
            # 修正「後」の単語を翻訳（デコード）して取得
            top_candidate_id = torch.argmax(masked_predictions[pos]).item()
            final_input_ids[pos] = top_candidate_id
            corrected_word = tokenizer.decode([top_candidate_id]).strip()

            # 信頼度をパーセント（%）で分かりやすく表示
            confidence_percent = original_prob * 100

            # もし単語が書き換わっていたら、コンソールにビフォーアフターを表示
            if original_word != corrected_word:
                tqdm.write(f"\n[DeBERTa 修正検出]")
                tqdm.write(f"  BEFORE: {original_word:<5} (文脈信頼度: {confidence_percent:15.10f}%)")
                tqdm.write(f"  AFTER : {corrected_word}")

        corrected_sentence = tokenizer.decode(final_input_ids, skip_special_tokens=True)
        
        # 安全ガード: AIが暴走して文字数が大きく変わった場合は元のテキストを守る
        if abs(len(current_sentence) - len(corrected_sentence)) > max_char_diff:
            corrected_sentences.append(current_sentence)
        else:
            corrected_sentences.append(corrected_sentence)

    return "".join(corrected_sentences), total_low_confidence_count


def proofread_text(segments: list) -> tuple[list, float]:
    """
    pipeline.py から呼び出されるメイン関数。
    セグメントのリストを受け取り、モデルのロードから各テキストの校正までを一元管理します。
    
    Args:
        segments: WhisperやLLMから渡されたセグメントのリスト
        threshold: この数値以下の確信度の単語を修正対象とする（0.0〜1.0）
           
    Returns:
        校正後のテキストを含むセグメントリスト
    """
    # =================================================================
    # 【AIの校正感度（閾値）設定】
    # 確信度がこの数値（0.0 〜 1.0）以下の単語が、修正候補（MASK対象）になります。
    # DeBERTaは非常に自信満々に予測を出す（0.999 または 0.001 のような両極端な値になりやすい）ため、
    # 閾値はかなり小さく設定しないと過剰に修正されてしまいます。
    #
    # ・ 0.5   : 50%未満なら修正。少しでも不自然なら直す（アグレッシブ・過剰修正の危険あり）
    # ・ 0.05  : 5%未満なら修正。標準的なバランス（おすすめ）
    # ・ 0.001 : 0.1%未満なら修正。文法的に絶対におかしいレベルの単語だけ直す（保守的）
    # =================================================================
    threshold = 0.05

    start_time = time.perf_counter()  # 時間計測スタート

    tqdm.write(f"[*] DeBERTaモデル ({DEBERTA_MODEL_NAME}) を読み込んでいます（初回はダウンロードに時間がかかります）...")
    
    try:
        # この処理の中でのみモデルを呼び出すことで、pipeline.py を綺麗に保ちます
        tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(DEBERTA_MODEL_NAME)
    except Exception as e:
        tqdm.write(f"[エラー] DeBERTaモデルの読み込みに失敗しました。校正をスキップして元のデータを維持します。\n詳細: {e}")
        return segments, 0.0

    tqdm.write(f"[*] DeBERTaによるテキスト校正処理を開始します... (感度閾値: {threshold*100}%)")
    
    corrected_segments = copy.deepcopy(segments)
    total_masks_in_all_segments = 0
    
    # 渡されたリストから1つずつデータを取り出し、テキストだけを校正用関数に投げる
    for seg in corrected_segments:
        original_text = seg.get("text", "")
        if original_text:
            # 内部関数 _proofread_single_text を呼び出して文字列を綺麗にする
            corrected_txt, masks_count = _proofread_single_text(
                original_text, tokenizer, model, threshold
            )
            seg["text"] = corrected_txt  # 綺麗になったテキストで上書き
            total_masks_in_all_segments += masks_count

    tqdm.write(f"[+] DeBERTa校正完了。確信度不足により修正候補となった総単語数: {total_masks_in_all_segments} 個")
    
    elapsed_time = time.perf_counter() - start_time  # 処理にかかった時間を計算

    return corrected_segments, elapsed_time  # セグメントデータと一緒にかかった時間も返す