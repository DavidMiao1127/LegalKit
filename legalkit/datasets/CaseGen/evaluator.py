from typing import List, Dict
from legalkit.datasets.base import BaseEvaluator
from legalkit.datasets.utils import clean_prediction
import numpy as np
import jieba
import re
import string
from collections import OrderedDict
from rouge import Rouge

import sys
sys.setrecursionlimit(10000)

# ---------------------util functions---------------------
def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""
    
    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    if s is None:
        return ""
    return white_space_fix(remove_punc(lower(s)))

def find_valid_substrings(s):
    if s is None:
        return ''
    s = s.split('解析')[0].split('分析')[0]
    s = s.replace("、", "") \
         .replace(".", "") \
         .replace(",", "") \
         .replace(";", "") \
         .replace("，", "") \
         .replace("和", "") \
         .replace(", ", "")
    pattern = r'[ABCDE]{1,5}'
    substrings = re.findall(pattern, s)
    valid_substrings = [substring for substring in substrings if len(substring) == len(set(substring))]
    valid_substrings = "".join(valid_substrings)
    valid_substrings = ''.join(OrderedDict.fromkeys(valid_substrings))
    return valid_substrings

def extract_choices(text):
    text = normalize_zh_answer(text)
    choices = "".join([char for char in text if char.isalpha()])  # 只保留字母
    return choices

# ---------------------compute functions---------------------
def compute_accuracy(data_dict):
    return {
        'score': sum([find_valid_substrings(p) == find_valid_substrings(r) for p, r in zip([d['prediction'] for d in data_dict], [d['refr'] for d in data_dict])]) / len(data_dict)
    }

def compute_rouge_l(data_dict):
    pred = [d['prediction'] for d in data_dict]
    ref = [d['refr'] for d in data_dict]
    
    return {
        'score': lambda pred, ref: (
            lambda rouge: np.mean([
                rouge.get_scores(pred_text, ref_text, avg=True)["rouge-l"]["f"]
                if pred_text and ref_text else 0.0
                for pred_text, ref_text in zip(
                    [
                        " ".join(list(jieba.cut(normalize_zh_answer(p), cut_all=False)))
                        for p in pred if p and isinstance(p, str) and p.strip()
                    ],
                    [
                        " ".join(list(jieba.cut(normalize_zh_answer(r), cut_all=False)))
                        for r in ref if r and isinstance(r, str) and r.strip()
                    ]
                )
            ])
        )(Rouge())
    }

# map task_id to compute function
funct_dict = {
    '1_1': compute_accuracy,
    '1_2': compute_accuracy,
    '1_3': compute_accuracy,
    '2_1': compute_accuracy,
    '2_2': compute_accuracy,
    '2_3': compute_accuracy,
    '2_4': compute_accuracy,
    '2_5': compute_accuracy,
    '3_1': compute_accuracy,
    '3_2': compute_accuracy,
    '3_3': compute_accuracy,
    '3_4': compute_accuracy,
    '3_5': compute_accuracy,
    '3_6': compute_accuracy,
    '4_1': compute_accuracy,
    '4_2': compute_accuracy,
    '5_1': compute_rouge_l,
    '5_2': compute_rouge_l,
    '5_3': compute_rouge_l,
    '5_4': compute_rouge_l,
    '6_1': compute_accuracy,
    '6_2': compute_accuracy,
    '6_3': compute_accuracy
}



class Evaluator(BaseEvaluator):
    def evaluate(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str],
        origin_prompts: List[str] = None
    ) -> Dict[str, float]:
        scorer = funct_dict.get(task_id)
        if not scorer:
            return {'error': f"Unsupported task '{task_id}'"}

        data_dict = []
        for i, rec in enumerate(records):
            orig = origin_prompts[i] if origin_prompts else f"{rec['instruction']}\n{rec['question']}"
            data_dict.append({
                'origin_prompt': orig,
                'prediction': clean_prediction(predictions.get(rec['id'], '')),
                'refr': rec['answer']
            })

        score_result = scorer(data_dict)
        print(score_result)

        # normalize float scores
        return {k: (v * 100 if 0.0 <= v <= 1.0 else v) for k, v in score_result.items()}

