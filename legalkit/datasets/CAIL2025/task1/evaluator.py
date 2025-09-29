from typing import List, Dict
from legalkit.datasets.base import BaseEvaluator
from legalkit.datasets.utils import clean_prediction
import re
import string
from collections import OrderedDict

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    all_punctuation = set(string.punctuation + cn_punctuation)
    s = s.lower().strip()
    return "".join(ch for ch in s if ch not in all_punctuation and not ch.isspace())

def _extract_choice_seq(s: str) -> str:
    if not s:
        return ""
    for kw in ("解析", "分析"):
        if kw in s:
            s = s.split(kw)[0]
    s = _normalize_text(s)
    s = s.replace("、", "")
    letters = [c.upper() for c in s if c.upper() in "ABCDE"]
    return "".join(OrderedDict.fromkeys(letters))

class Evaluator(BaseEvaluator):
    def evaluate(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str],
        origin_prompts: List[str] = None
    ) -> Dict[str, float]:
        if not records:
            return {'error': 'No records', 'score': 0.0}

        total = 0
        correct = 0
        for i, rec in enumerate(records):
            gold_raw = rec.get('answer', '')
            pred_raw = clean_prediction(predictions.get(rec['id'], ''))

            gold_seq = _extract_choice_seq(gold_raw)
            pred_seq = _extract_choice_seq(pred_raw)

            if gold_seq:
                total += 1
                if pred_seq == gold_seq:
                    correct += 1

        acc = (correct / total) if total else 0.0
        return {
            'score': acc * 100,
            'accuracy': acc * 100,
            'total': total,
            'correct': correct
        }
