from typing import List, Dict, Tuple
from legalkit.datasets.base import BaseEvaluator
from legalkit.datasets.utils import clean_prediction
import numpy as np
import jieba
import re
import string
from collections import OrderedDict
from rouge import Rouge
# Optional external metrics libs
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except Exception:  # noqa
    sentence_bleu = None
    SmoothingFunction = None
try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod
except Exception:  # noqa
    _rouge_scorer_mod = None
try:
    from bert_score import score as bert_score_func
except Exception:  # noqa
    bert_score_func = None
import json
import os

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

# --------------------- task id mapping (refactored) ---------------------
# New canonical CaseGen subtasks: defense / fact / reasoning / judgement
# We use rouge-l as the default lexical similarity metric for all four long-form tasks.
# If later a task needs a different base metric (e.g. classification accuracy), extend here.

NEW_CASEGEN_FUNCT = {
    'defense': compute_rouge_l,
    'fact': compute_rouge_l,
    'reasoning': compute_rouge_l,
    'judgement': compute_rouge_l,
}

# Legacy numeric task ids (multiple small granular tasks) mapped heuristically to new categories.
# This preserves backward compatibility if upstream dataset still emits old ids during transition.
# Adjust mapping as needed when the dataset generation layer is migrated.
LEGACY_TO_NEW = {
    # group 1.x,2.x -> defense (e.g., various preliminary defenses)
    '1_1': 'defense','1_2': 'defense','1_3': 'defense',
    '2_1': 'defense','2_2': 'defense','2_3': 'defense','2_4': 'defense','2_5': 'defense',
    # group 3.x reasoning analysis style
    '3_1': 'reasoning','3_2': 'reasoning','3_3': 'reasoning','3_4': 'reasoning','3_5': 'reasoning','3_6': 'reasoning',
    # group 4.x factual extraction style
    '4_1': 'fact','4_2': 'fact',
    # group 5.x long-form judgement elements
    '5_1': 'judgement','5_2': 'judgement','5_3': 'judgement','5_4': 'judgement',
    # group 6.x (if existed) mapping: treat as reasoning or fact — choose reasoning by default; adjust if needed
    '6_1': 'reasoning','6_2': 'reasoning','6_3': 'reasoning'
}

def resolve_task_category(task_id: str) -> str:
    """Return the canonical four-category task label for CaseGen.
    1. If already a new category, return directly.
    2. Else map via legacy dict.
    3. If still unknown, default to 'reasoning' (neutral choice) to avoid hard failure.
    """
    if task_id in NEW_CASEGEN_FUNCT:
        return task_id
    if task_id in LEGACY_TO_NEW:
        return LEGACY_TO_NEW[task_id]
    return 'reasoning'



class Evaluator(BaseEvaluator):
    def supports_llm_judge(self) -> bool:
        return True

    _TEMPLATE_CACHE = {}
    _TEMPLATE_FILENAMES = {
        'defense': 'defense_judge_template.txt',
        'fact': 'fact_judge_template.txt',
        'reasoning': 'reasoning_judge_template.txt',
        'judgement': 'judgement_judge_template.txt'
    }
    _TEMPLATE_DEFAULTS = {
        'defense': (
            "你是法律文书质量评审专家。\n"
            "请根据起诉书与参考答辩书评估 AI 生成的答辩书质量，并仅输出 JSON 格式："
            "{\"score\": <0-1 小数>, \"explanation\": \"简要理由\"}.\n"
            "起诉书:\n{起诉书}\n参考答辩书:\n{参考答辩书}\nAI助手撰写的答辩书:\n{AI助手撰写的答辩书}\n"
        ),
        'fact': (
            "你是法律文书事实部分评审专家。\n"
            "请比较 AI 生成的审理事实与参考答案，输出 JSON：{\"score\": <0-1>, \"explanation\": \"简要理由\"}.\n"
            "参考答案:\n{参考答案}\n审理事实:\n{审理事实}\n"
        ),
        'reasoning': (
            "你是法律说理部分评审专家。\n"
            "请比较 AI 生成的说理部分与参考答案，输出 JSON：{\"score\": <0-1>, \"explanation\": \"简要理由\"}.\n"
            "参考答案:\n{参考答案}\n判决说理部分:\n{判决说理部分}\n"
        ),
        'judgement': (
            "你是法律判决结果部分评审专家。\n"
            "请比较 AI 生成的判决结果与参考答案，输出 JSON：{\"score\": <0-1>, \"explanation\": \"简要理由\"}.\n"
            "参考答案:\n{参考答案}\nAI助手撰写的判决结果部分:\n{AI助手撰写的判决结果部分}\n"
        )
    }

    def _template_dir(self) -> str:
        # data directory: legalkit/data/casegen_templates
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'CaseGen','templates'))

    def _load_template(self, category: str) -> str:
        if category in self._TEMPLATE_CACHE:
            return self._TEMPLATE_CACHE[category]
        fdir = self._template_dir()
        fname = self._TEMPLATE_FILENAMES.get(category)
        path = os.path.join(fdir, fname) if fname else None
        content = None
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                content = None
        if not content:
            content = self._TEMPLATE_DEFAULTS[category]
        self._TEMPLATE_CACHE[category] = content
        return content

    def _sanitize_generation(self, text: str) -> str:
        if not text:
            return ''
        # remove tags like <antThinking>...</antThinking> and any other <> blocks
        while '<' in text and '>' in text:
            new_text = re.sub(r"<antThinking>.*?</antThinking>", "", text, flags=re.S)
            new_text = re.sub(r"<.*?>", "", new_text, flags=re.S)
            if new_text == text:
                break
            text = new_text
        return text.strip()

    def _build_casegen_prompt(self, category: str, record: Dict, prediction: str) -> Tuple[str, bool]:
        tpl = self._load_template(category)
        # mapping placeholders depending on category
        try:
            if category == 'defense':
                prompt = tpl.replace('{起诉书}', record.get('prosecution', '') or '') \
                             .replace('{参考答辩书}', record.get('defense', '') or '') \
                             .replace('{AI助手撰写的答辩书}', self._sanitize_generation(prediction))
            elif category == 'fact':
                prompt = tpl.replace('{参考答案}', record.get('fact', '') or '') \
                             .replace('{审理事实}', self._sanitize_generation(prediction))
            elif category == 'reasoning':
                prompt = tpl.replace('{参考答案}', record.get('reasoning', '') or '') \
                             .replace('{判决说理部分}', self._sanitize_generation(prediction))
            elif category == 'judgement':
                prompt = tpl.replace('{参考答案}', record.get('judgement', '') or '') \
                             .replace('{AI助手撰写的判决结果部分}', self._sanitize_generation(prediction))
            else:
                return '', False
        except Exception:
            return '', False
        return prompt, True

    def _collect_casegen_judge_batches(self, records: List[Dict], predictions: Dict[int, str], categories: List[str]):
        cat_prompts = {c: [] for c in categories}
        cat_ids = {c: [] for c in categories}
        for rec in records:
            pred = clean_prediction(predictions.get(rec['id'], ''))
            for cat in categories:
                prompt, ok = self._build_casegen_prompt(cat, rec, pred)
                if ok and prompt.strip():
                    cat_prompts[cat].append(prompt)
                    cat_ids[cat].append(rec['id'])
        return cat_prompts, cat_ids

    def evaluate(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str],
        origin_prompts: List[str] = None
    ) -> Dict[str, float]:
        cat = resolve_task_category(task_id)
        scorer = NEW_CASEGEN_FUNCT.get(cat)
        if not scorer:
            return {'error': f"Unsupported task '{task_id}' (resolved category {cat})"}

        data_dict = []
        for i, rec in enumerate(records):
            orig = origin_prompts[i] if origin_prompts else f"{rec['instruction']}\n{rec['question']}"
            data_dict.append({
                'origin_prompt': orig,
                'prediction': clean_prediction(predictions.get(rec['id'], '')),
                'refr': rec['answer']
            })

        score_result = scorer(data_dict)
        # For rouge-l compute function we stored under 'score' a lambda returning mean rouge f.
        # Keep same pattern but ensure numeric value extraction.
        metrics = {}
        for k, v in score_result.items():
            if callable(v):
                try:
                    val = v(None, None)  # lambda ignores params due to closure
                except Exception:
                    val = 0.0
            else:
                val = v
            if 0.0 <= val <= 1.0:
                val = val * 100
            metrics[k] = val

        # Integrate classic text generation metrics for CaseGen long-form sections.
        classic = self._maybe_compute_classic_metrics(task_id, records, predictions)
        if classic:
            metrics.update(classic)

        if self.has_judge():
            judge_metrics = self.evaluate_with_judge(cat, records, predictions)
            if judge_metrics:
                metrics.update(judge_metrics)

        return metrics

    def evaluate_with_judge(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str]
    ) -> Dict[str, float]:
        if not self.judge_runner:
            return {}

        # Category definitions and desired metric keys (Chinese kept to align with external script semantics)
        category_metric_keys = {
            'defense': ['事实准确性', '法律关系准确性', '逻辑性', '完备性', '综合得分'],
            'fact': ['事实准确性', '相关性', '逻辑性', '综合得分'],
            'reasoning': ['争议焦点准确性', '法律关系准确性', '逻辑性', '伦理性', '综合得分'],
            'judgement': ['判决结果准确性', '引用法条完整性和准确性', '综合得分']
        }

        # Synonym / variant mapping -> canonical key
        synonym_map = {
            '事实正确性': '事实准确性',
            '法律关系正确性': '法律关系准确性',
            '争议焦点正确性': '争议焦点准确性',
            '判决问题准确性': '判决结果准确性',
            '判决结果正确性': '判决结果准确性',
            '引用法条完整性和正确性': '引用法条完整性和准确性',
            '引用法条完整正确确性': '引用法条完整性和准确性',
            '引用法条完整正确性': '引用法条完整性和准确性',
            '引用法条完整性': '引用法条完整性和准确性',
            '引用法条完整准确性': '引用法条完整性和准确性',
            '论理性': '伦理性'
        }

        categories = list(category_metric_keys.keys())
        cat_prompts, cat_ids = self._collect_casegen_judge_batches(records, predictions, categories)

        all_metrics: Dict[str, float] = {}
        overall_scores = []  # collect per-category 综合得分 (normalized 0-1)
        total_invalid = 0
        total_votes = 0
        sample_rationales = []

        def _clean_response(txt: str) -> str:
            if not txt:
                return ''
            # remove think blocks
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
            return txt

        def _extract_json_fragment(txt: str) -> str:
            # find last '{' and matching '}' naive slice similar to external script
            if not txt:
                return ''
            left = txt.rfind('{')
            if left == -1:
                return ''
            frag = txt[left:]
            # trim after first '}' that would make a valid JSON (simple heuristic)
            right_rel = frag.find('}')
            if right_rel != -1:
                frag = frag[:right_rel+1]
            # basic sanitization
            frag = frag.replace("'", '"').replace('\\', '')
            # remove Chinese '分' suffix right after a number 0-10
            frag = re.sub(r'(?<=\D)([0-9]|10)分', r'\1', frag)
            # collapse duplicated opening braces
            while frag.startswith('{{') and not frag.startswith('{{"'):
                frag = frag[1:]
            return frag

        def _to_float(v):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            # normalize if looks like 0-10 scale
            if fv > 1.0 and fv <= 10.0:
                fv = fv / 10.0
            # clip
            return max(0.0, min(1.0, fv))

        # metric accumulators per category
        cat_metric_acc: Dict[str, Dict[str, List[float]]] = {c: {k: [] for k in ks} for c, ks in category_metric_keys.items()}

        for cat in categories:
            prompts = cat_prompts[cat]
            ids = cat_ids[cat]
            if not prompts:
                continue
            responses = self.judge_runner.generate(prompts)
            invalid = 0
            rationales = []
            for rid, resp in zip(ids, responses):
                raw = _clean_response(resp)
                json_frag = _extract_json_fragment(raw)
                parsed_obj = None
                if json_frag:
                    try:
                        parsed_obj = json.loads(json_frag)
                    except Exception:
                        parsed_obj = None
                # build canonical metrics dict
                metrics_map = {k: -1 for k in category_metric_keys[cat]}
                rationale_text = ''
                if parsed_obj and isinstance(parsed_obj, dict):
                    # rename synonyms
                    canon_obj = {}
                    for k, v in parsed_obj.items():
                        canon_k = synonym_map.get(k, k)
                        canon_obj[canon_k] = v
                    # prefer explicit 综合得分 else fallback to 'score'
                    if '综合得分' not in canon_obj and 'score' in canon_obj:
                        canon_obj['综合得分'] = canon_obj['score']
                    # explanation fields
                    rationale_text = canon_obj.get('explanation') or canon_obj.get('reason') or ''
                    for mk in metrics_map.keys():
                        if mk in canon_obj:
                            val = _to_float(canon_obj[mk])
                            if val is not None:
                                metrics_map[mk] = val
                else:
                    invalid += 1
                # accumulate
                for mk, val in metrics_map.items():
                    if val != -1:
                        cat_metric_acc[cat][mk].append(val)
                if rationale_text and len(sample_rationales) < 5:
                    sample_rationales.append({'id': rid, 'type': cat, 'rationale': rationale_text})
            # aggregate category metrics
            for mk, vals in cat_metric_acc[cat].items():
                if vals:
                    # export as percentage *100
                    all_metrics[f'judge_{cat}_{mk}'] = (sum(vals)/len(vals))*100
                else:
                    all_metrics[f'judge_{cat}_{mk}'] = 0.0
            # Backward compatible primary score per category
            main_vals = cat_metric_acc[cat].get('综合得分') or []
            if main_vals:
                cat_main_avg = sum(main_vals)/len(main_vals)
                overall_scores.append(cat_main_avg)
                total_votes += len(main_vals)
                all_metrics[f'judge_{cat}_score'] = cat_main_avg * 100
            else:
                all_metrics[f'judge_{cat}_score'] = 0.0
            # votes/invalid
            all_metrics[f'judge_{cat}_votes'] = len(cat_metric_acc[cat].get('综合得分') or [])
            # invalid includes parse failures + generation shortfall
            all_metrics[f'judge_{cat}_invalid'] = invalid + max(0, len(prompts) - len(responses))
            total_invalid += all_metrics[f'judge_{cat}_invalid']

        # overall aggregated judge_score (average of category 综合得分)
        if overall_scores:
            all_metrics['judge_score'] = (sum(overall_scores)/len(overall_scores))*100
        else:
            all_metrics['judge_score'] = 0.0
        all_metrics['judge_votes'] = total_votes
        all_metrics['judge_invalid'] = total_invalid
        if sample_rationales:
            all_metrics['judge_samples'] = json.dumps(sample_rationales, ensure_ascii=False)
        return all_metrics

    # ---------------- Classic Metrics (BLEU / Rouge-L / BERTScore) ------------------
    def _maybe_compute_classic_metrics(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str]
    ) -> Dict[str, float]:
        """Compute BLEU(1-4), Rouge-L (P/R/F), BERTScore (P/R/F1) for applicable CaseGen categories.

        We treat task_id groups (5_x for narrative, maybe others) but since CaseGen evaluator
        receives mixed task_ids, we fallback to record-level category inference via keys.
        For simplicity, we apply classic metrics only when records contain long-form
        fields among {'defense','fact','reasoning','judgement'}.
        """
        # Quick exit if libs missing
        if not any([sentence_bleu, _rouge_scorer_mod, bert_score_func]):
            return {}

        # Determine which field is relevant per record by heuristic: prefer keys present
        candidate_fields = ['defense', 'fact', 'reasoning', 'judgement']
        # Collect pairs per category
        cat_pairs: Dict[str, List[Tuple[str, str]]] = {c: [] for c in candidate_fields}
        for rec in records:
            rid = rec['id']
            pred_text = clean_prediction(predictions.get(rid, ''))
            # Choose category if ref exists and prediction non-empty
            for c in candidate_fields:
                if c in rec and isinstance(rec[c], str) and rec[c].strip():
                    # treat that as reference candidate, add even if pred empty (will handle)
                    cat_pairs[c].append((rec[c], pred_text))
        # Remove empty categories
        cat_pairs = {k: v for k, v in cat_pairs.items() if v}
        if not cat_pairs:
            return {}

        smoothing = SmoothingFunction().method4 if SmoothingFunction else None
        rouge_scorer = _rouge_scorer_mod.RougeScorer(['rougeL'], use_stemmer=False) if _rouge_scorer_mod else None
        results: Dict[str, float] = {}

        for cat, pairs in cat_pairs.items():
            refs = [r for r, _ in pairs]
            hyps = [h for _, h in pairs]
            # Tokenize using jieba for BLEU
            if sentence_bleu and smoothing:
                bleu_ng = {1: [], 2: [], 3: [], 4: []}
                for ref, hyp in pairs:
                    if not ref or not hyp:
                        continue
                    ref_tokens = list(jieba.cut(ref))
                    hyp_tokens = list(jieba.cut(hyp))
                    weights_list = {
                        1: (1, 0, 0, 0),
                        2: (0.5, 0.5, 0, 0),  # standard cumulative variant alternative
                        3: (1/3, 1/3, 1/3, 0),
                        4: (0.25, 0.25, 0.25, 0.25)
                    }
                    for n in range(1,5):
                        try:
                            score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights_list[n], smoothing_function=smoothing)
                            bleu_ng[n].append(score)
                        except Exception:
                            continue
                for n in range(1,5):
                    vals = bleu_ng[n]
                    if vals:
                        results[f'classic_{cat}_bleu{n}'] = (sum(vals)/len(vals))*100
                    else:
                        results[f'classic_{cat}_bleu{n}'] = 0.0

            # Rouge-L (only F previously; we export P/R/F)
            if rouge_scorer:
                rouge_p = []
                rouge_r = []
                rouge_f = []
                for ref, hyp in pairs:
                    if not ref or not hyp:
                        continue
                    try:
                        sc = rouge_scorer.score(ref, hyp)['rougeL']
                        rouge_p.append(sc.precision)
                        rouge_r.append(sc.recall)
                        rouge_f.append(sc.fmeasure)
                    except Exception:
                        continue
                if rouge_p:
                    results[f'classic_{cat}_rougeL_p'] = (sum(rouge_p)/len(rouge_p))*100
                    results[f'classic_{cat}_rougeL_r'] = (sum(rouge_r)/len(rouge_r))*100
                    results[f'classic_{cat}_rougeL_f'] = (sum(rouge_f)/len(rouge_f))*100
                else:
                    results[f'classic_{cat}_rougeL_p'] = 0.0
                    results[f'classic_{cat}_rougeL_r'] = 0.0
                    results[f'classic_{cat}_rougeL_f'] = 0.0

            # BERTScore (expensive) - run once per category if library available
            if bert_score_func and refs and hyps:
                try:
                    P, R, F1 = bert_score_func(hyps, refs, lang='zh', verbose=False, model_type='bert-base-chinese')
                    results[f'classic_{cat}_bertscore_p'] = float(P.mean().item())*100
                    results[f'classic_{cat}_bertscore_r'] = float(R.mean().item())*100
                    results[f'classic_{cat}_bertscore_f1'] = float(F1.mean().item())*100
                except Exception:
                    results[f'classic_{cat}_bertscore_p'] = 0.0
                    results[f'classic_{cat}_bertscore_r'] = 0.0
                    results[f'classic_{cat}_bertscore_f1'] = 0.0

        return results

    def _parse_judge_response(self, response: str):
        if not response:
            return None
        try:
            start = response.index('{')
            end = response.rindex('}') + 1
            payload = json.loads(response[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

        score = payload.get('score')
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            return None

        score_val = max(0.0, min(1.0, score_val))
        rationale = payload.get('explanation') or payload.get('reason') or ''
        return score_val, str(rationale)

