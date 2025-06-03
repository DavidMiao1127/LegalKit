from typing import List, Dict
from legalkit.datasets.base import BaseEvaluator
from legalkit.datasets.utils import clean_prediction
import re
import string
from collections import OrderedDict

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

def compute_accuracy(data_dict):
    return {
        'score': sum([find_valid_substrings(p) == find_valid_substrings(r) for p, r in zip([d['prediction'] for d in data_dict], [d['refr'] for d in data_dict])]) / len(data_dict)
    }


# map task_id to compute function
funct_dict = {
    '0_test': compute_accuracy,
    '1_test': compute_accuracy
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
            orig = origin_prompts[i] if origin_prompts else f"{rec['instruction']}\n{rec['statement']}\n{rec['option_list']}"
            data_dict.append({
                'origin_prompt': orig,
                'prediction': clean_prediction(predictions.get(rec['id'], '')),
                'refr': rec['answer']
            })

        score_result = scorer(data_dict)

        # normalize float scores
        return {k: (v * 100 if 0.0 <= v <= 1.0 else v) for k, v in score_result.items()}

