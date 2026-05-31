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

# ★追加: Pythonの警告システムを使って、Hugging Face絡みのWarningを根こそぎ無視する
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
        mask_positions = []
        
        # 確信度不足のトークンを検出
        for idx in range(1, len(input_ids) - 1):
            token_id = input_ids[idx].item()
            original_word_prob = probabilities[idx][token_id].item()
            
            if original_word_prob < threshold:
                mask_positions.append(idx)

        total_low_confidence_count += len(mask_positions)

        # マスク候補がない場合はそのまま採用
        if len(mask_positions) == 0:
            corrected_sentences.append(current_sentence)
            continue

        # 不自然な箇所を [MASK] に置き換えて再予測
        masked_input_ids = input_ids.clone()
        for pos in mask_positions:
            masked_input_ids[pos] = tokenizer.mask_token_id
            
        with torch.no_grad():
            masked_inputs = {"input_ids": masked_input_ids.unsqueeze(0)}
            if "attention_mask" in inputs:
                masked_inputs["attention_mask"] = inputs["attention_mask"]
            
            masked_outputs = model(**masked_inputs)
            masked_predictions = masked_outputs.logits[0]

        # 最適な単語を当てはめる
        final_input_ids = input_ids.clone()
        for pos in mask_positions:
            top_candidate_id = torch.argmax(masked_predictions[pos]).item()
            final_input_ids[pos] = top_candidate_id

        corrected_sentence = tokenizer.decode(final_input_ids, skip_special_tokens=True)
        
        # 安全ガード: AIが暴走して文字数が大きく変わった場合は元のテキストを守る
        if abs(len(current_sentence) - len(corrected_sentence)) > 15:
            corrected_sentences.append(current_sentence)
        else:
            corrected_sentences.append(corrected_sentence)

    return "".join(corrected_sentences), total_low_confidence_count


def proofread_text(segments: list, threshold: float = 0.7) -> list:
    """
    pipeline.py から呼び出されるメイン関数。
    セグメントのリストを受け取り、モデルのロードから各テキストの校正までを一元管理します。
    
    Args:
        segments: WhisperやLLMから渡されたセグメントのリスト
        threshold: この数値以下の確信度の単語を修正対象とする（0.0〜1.0）
        
    Returns:
        校正後のテキストを含むセグメントリスト
    """

    start_time = time.perf_counter()  # 時間計測スタート

    tqdm.write(f"[*] DeBERTaモデル ({DEBERTA_MODEL_NAME}) を読み込んでいます（初回はダウンロードに時間がかかります）...")
    
    try:
        # この処理の中でのみモデルを呼び出すことで、pipeline.py を綺麗に保ちます
        tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(DEBERTA_MODEL_NAME)
    except Exception as e:
        tqdm.write(f"[エラー] DeBERTaモデルの読み込みに失敗しました。校正をスキップして元のデータを維持します。\n詳細: {e}")
        return segments

    tqdm.write("[*] DeBERTaによるテキスト校正処理を開始します...")
    
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