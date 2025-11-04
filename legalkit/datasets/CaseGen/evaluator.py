from typing import List, Dict, Tuple
from legalkit.datasets.base import BaseEvaluator
from legalkit.datasets.utils import clean_prediction
from legalkit.storage import StorageManager
import numpy as np
import jieba
import re
import string
from collections import OrderedDict
from rouge import Rouge
try:
    from tqdm.auto import tqdm
except Exception:  # noqa
    class _TqdmFallback:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n=1):
            pass

        def close(self):
            pass

    def tqdm(*args, **kwargs):  # type: ignore
        return _TqdmFallback()
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

# --------------------- task id mapping (simplified) ---------------------
def resolve_task_category(task_id: str) -> str:
    """Return the canonical CaseGen category when applicable.
    This evaluator now relies solely on LLM-as-judge. We keep a minimal helper
    to pass through known categories and default to 'reasoning' otherwise.
    """
    categories = {'defense', 'fact', 'reasoning', 'judgement'}
    return task_id if task_id in categories else 'reasoning'



class Evaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self._run_root = None
        self._model_id = None
        self._debug = False

    def configure_judge(
        self,
        judge_runner,
        dataset: str = None,
        run_root: str = None,
        model_id: str = None,
        debug: bool = False,
        **kwargs
    ):
        super().configure_judge(judge_runner, **kwargs)
        self.dataset = dataset
        self._run_root = run_root
        self._model_id = model_id
        self._debug = bool(debug)

    def supports_llm_judge(self) -> bool:
        return True

    def _record_judge_entry(self, subtask: str, entry: Dict):
        if not self._run_root or not self._model_id or not entry:
            return
        try:
            StorageManager.append_judge_result(
                self._run_root,
                self._model_id,
                subtask,
                entry
            )
        except Exception as exc:
            if self._debug:
                print(f"[Evaluator] failed to append judge entry for {subtask}: {exc}")

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

    _BLEU_WEIGHTS = [
        ('1_gram', (1, 0, 0, 0)),
        ('2_gram', (0, 1, 0, 0)),
        ('3_gram', (0, 0, 1, 0)),
        ('4_gram', (0, 0, 0, 1))
    ]
    _BERT_MODEL = "bert-base-chinese"

    def _template_dir(self) -> str:
        # data directory: legalkit/data/casegen_templates
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'CaseGen','templates'))

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

    def _reference_text_for_category(self, category: str, record: Dict) -> str:
        if category == 'defense':
            val = record.get('defense') or record.get('defence')
        elif category == 'fact':
            val = record.get('fact')
        elif category == 'reasoning':
            val = record.get('reasoning')
        elif category == 'judgement':
            val = record.get('judgement') or record.get('judgment')
        else:
            val = None
        if val:
            return val
        return ''

    def _compute_text_metrics(self, reference: str, generated: str) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {
            'bleu': None,
            'rouge': None,
            'bertscore': None
        }
        if not reference or not generated:
            return metrics

        reference = str(reference).strip()
        generated = str(generated).strip()
        if not reference or not generated:
            return metrics

        try:
            ref_tokens = list(jieba.cut(reference))
            gen_tokens = list(jieba.cut(generated))
        except Exception:
            ref_tokens = reference.split()
            gen_tokens = generated.split()

        if sentence_bleu is not None:
            smooth_fn = None
            if SmoothingFunction is not None:
                try:
                    smooth_fn = SmoothingFunction().method4
                except Exception:
                    smooth_fn = None
            bleu_scores: Dict[str, float] = {}
            for label, weights in self._BLEU_WEIGHTS:
                try:
                    kwargs = {'weights': weights}
                    if smooth_fn is not None:
                        score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth_fn, **kwargs)
                    else:
                        score = sentence_bleu([ref_tokens], gen_tokens, **kwargs)
                except Exception:
                    score = None
                bleu_scores[label] = score
            metrics['bleu'] = bleu_scores

        if _rouge_scorer_mod is not None:
            try:
                if not hasattr(self, '_rouge_scorer_instance'):
                    self._rouge_scorer_instance = _rouge_scorer_mod.RougeScorer(['rougeL'], use_stemmer=False)
                rouge_scores = self._rouge_scorer_instance.score(reference, generated)['rougeL']
                metrics['rouge'] = {
                    'precision': rouge_scores.precision,
                    'recall': rouge_scores.recall,
                    'f1': rouge_scores.fmeasure
                }
            except Exception:
                metrics['rouge'] = None

        if bert_score_func is not None:
            try:
                P, R, F1 = bert_score_func([generated], [reference], lang="zh", verbose=False, model_type=self._BERT_MODEL)
                metrics['bertscore'] = {
                    'precision': float(P.mean().item()),
                    'recall': float(R.mean().item()),
                    'f1': float(F1.mean().item())
                }
            except Exception:
                metrics['bertscore'] = None

        return metrics

    def _append_quality_metrics(self, aggregator: Dict[str, Dict[str, List[float]]], metrics: Dict[str, Dict[str, float]]) -> None:
        if not metrics:
            return
        bleu_bucket = aggregator.get('bleu')
        if bleu_bucket is not None and metrics.get('bleu'):
            for label, val in metrics['bleu'].items():
                if val is not None and label in bleu_bucket:
                    try:
                        bleu_bucket[label].append(float(val))
                    except Exception:
                        continue
        rouge_bucket = aggregator.get('rouge')
        if rouge_bucket is not None and metrics.get('rouge'):
            for label in ('precision', 'recall', 'f1'):
                val = metrics['rouge'].get(label) if metrics['rouge'] else None
                if val is not None and label in rouge_bucket:
                    try:
                        rouge_bucket[label].append(float(val))
                    except Exception:
                        continue
        bert_bucket = aggregator.get('bertscore')
        if bert_bucket is not None and metrics.get('bertscore'):
            for label in ('precision', 'recall', 'f1'):
                val = metrics['bertscore'].get(label) if metrics['bertscore'] else None
                if val is not None and label in bert_bucket:
                    try:
                        bert_bucket[label].append(float(val))
                    except Exception:
                        continue

    def _collect_casegen_judge_batches(self, records: List[Dict], predictions: Dict[int, str], categories: List[str]):
        cat_prompts = {c: [] for c in categories}
        cat_ids = {c: [] for c in categories}
        cat_refs = {c: [] for c in categories}
        cat_pred_texts = {c: [] for c in categories}
        for rec in records:
            pred = clean_prediction(predictions.get(rec['id'], ''))
            for cat in categories:
                prompt, ok = self._build_casegen_prompt(cat, rec, pred)
                if ok and prompt.strip():
                    cat_prompts[cat].append(prompt)
                    cat_ids[cat].append(rec['id'])
                    reference_text = self._reference_text_for_category(cat, rec)
                    cat_refs[cat].append(reference_text)
                    cat_pred_texts[cat].append(self._sanitize_generation(pred))
        return cat_prompts, cat_ids, cat_refs, cat_pred_texts

    def evaluate(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str],
        origin_prompts: List[str] = None
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        cat = resolve_task_category(task_id)
        # If an LLM judge is available, run the full judge+quality pipeline.
        if self.has_judge():
            judge_metrics = self.evaluate_with_judge(cat, records, predictions, subtask_id=task_id)
            if judge_metrics:
                metrics.update(judge_metrics)
                # Provide a primary 'score' equal to judge_score for downstream consumers
                if 'judge_score' in judge_metrics:
                    metrics['score'] = judge_metrics['judge_score']
            return metrics

        # No LLM judge configured: compute traditional text-quality metrics only
        quality_only = self.compute_quality_metrics_only(cat, records, predictions, subtask_id=task_id)
        if quality_only:
            metrics.update(quality_only)

        return metrics

    def evaluate_with_judge(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str],
        subtask_id: str = None
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

        categories = [task_id] if task_id in category_metric_keys else []
        if not categories:
            return {}
        cat_prompts, cat_ids, cat_refs, cat_pred_texts = self._collect_casegen_judge_batches(records, predictions, categories)

        quality_acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        precomputed_quality: Dict[str, List[Dict[str, Dict[str, float]]]] = {}
        for cat in categories:
            quality_acc[cat] = {
                'bleu': {label: [] for label, _ in self._BLEU_WEIGHTS} if sentence_bleu is not None else None,
                'rouge': {'precision': [], 'recall': [], 'f1': []} if _rouge_scorer_mod is not None else None,
                'bertscore': {'precision': [], 'recall': [], 'f1': []} if bert_score_func is not None else None
            }
            precomputed_quality[cat] = []
            refs = cat_refs.get(cat, [])
            preds = cat_pred_texts.get(cat, [])
            for ref, pred_text in zip(refs, preds):
                qm = self._compute_text_metrics(ref, pred_text)
                precomputed_quality[cat].append(qm)
                self._append_quality_metrics(quality_acc[cat], qm)

        all_metrics: Dict[str, float] = {}
        overall_scores = [] 
        total_invalid = 0
        total_votes = 0
        sample_rationales = []
        subtask = subtask_id or task_id

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
            processed_indices = set()
            invalid = 0
            category_entries: List[Dict] = []

            def _handle_entry(idx_in_cat: int, resp: str) -> None:
                nonlocal invalid
                rid = ids[idx_in_cat]
                prompt_text = prompts[idx_in_cat]
                raw = _clean_response(resp)
                json_frag = _extract_json_fragment(raw)
                parsed_obj = None
                parse_error = None
                if json_frag:
                    try:
                        parsed_obj = json.loads(json_frag)
                    except Exception as exc:
                        parsed_obj = None
                        parse_error = str(exc)
                else:
                    parse_error = 'missing_json'

                metrics_map = {k: -1 for k in category_metric_keys[cat]}
                rationale_text = ''
                if parsed_obj and isinstance(parsed_obj, dict):
                    canon_obj = {}
                    for k, v in parsed_obj.items():
                        canon_k = synonym_map.get(k, k)
                        canon_obj[canon_k] = v
                    if '综合得分' not in canon_obj and 'score' in canon_obj:
                        canon_obj['综合得分'] = canon_obj['score']
                    rationale_text = canon_obj.get('explanation') or canon_obj.get('reason') or ''
                    for mk in list(metrics_map.keys()):
                        if mk in canon_obj:
                            val = _to_float(canon_obj[mk])
                            if val is not None:
                                metrics_map[mk] = val
                else:
                    invalid += 1
                    if parse_error is None:
                        parse_error = 'parse_failed'

                for mk, val in metrics_map.items():
                    if val != -1:
                        cat_metric_acc[cat][mk].append(val)
                if rationale_text and len(sample_rationales) < 5:
                    sample_rationales.append({'id': rid, 'type': cat, 'rationale': rationale_text})

                entry_metrics = {mk: (val if val != -1 else None) for mk, val in metrics_map.items()}
                entry = {
                    'id': rid,
                    'category': cat,
                    'prompt': prompt_text,
                    'raw_response': raw,
                    'json_fragment': json_frag,
                    'parsed': parsed_obj if isinstance(parsed_obj, dict) else None,
                    'metrics': entry_metrics,
                    'rationale': rationale_text or None,
                    'parse_error': parse_error,
                }
                q_cache = precomputed_quality.get(cat) or []
                if idx_in_cat < len(q_cache):
                    entry['quality_metrics'] = q_cache[idx_in_cat]
                else:
                    entry['quality_metrics'] = {
                        'bleu': None,
                        'rouge': None,
                        'bertscore': None
                    }
                category_entries.append(entry)
                processed_indices.add(idx_in_cat)
                try:
                    if self._run_root and self._model_id:
                        StorageManager.append_judge_result(self._run_root, self._model_id, subtask, entry)
                except Exception as exc:
                    if self._debug:
                        print(f"[Evaluator] failed to append judge result for id={rid}: {exc}")

            def _batch_callback(start_idx: int, batch_outputs: List[str]) -> None:
                for offset, resp in enumerate(batch_outputs):
                    idx_in_cat = start_idx + offset
                    if idx_in_cat < len(ids):
                        _handle_entry(idx_in_cat, resp)

            # run judge in batches and update progress bar, streaming results via callback
            with tqdm(total=len(prompts), desc=f"Judge {cat}", unit="sample", leave=False) as pbar:
                self.judge_runner.generate(
                    prompts,
                    progress=pbar.update,
                    batch_callback=_batch_callback
                )

            # handle missing responses (judge returned fewer outputs than prompts)
            missing_count = max(0, len(ids) - len(processed_indices))
            if missing_count:
                for idx_in_cat in range(len(ids)):
                    if idx_in_cat in processed_indices:
                        continue
                    rid = ids[idx_in_cat]
                    prompt_text = prompts[idx_in_cat]
                    entry = {
                        'id': rid,
                        'category': cat,
                        'prompt': prompt_text,
                        'raw_response': '',
                        'json_fragment': None,
                        'parsed': None,
                        'metrics': {mk: None for mk in category_metric_keys[cat]},
                        'rationale': None,
                        'parse_error': 'missing_response',
                    }
                    q_cache = precomputed_quality.get(cat) or []
                    if idx_in_cat < len(q_cache):
                        entry['quality_metrics'] = q_cache[idx_in_cat]
                    else:
                        entry['quality_metrics'] = {
                            'bleu': None,
                            'rouge': None,
                            'bertscore': None
                        }
                    category_entries.append(entry)
                    try:
                        if self._run_root and self._model_id:
                            StorageManager.append_judge_result(self._run_root, self._model_id, subtask, entry)
                    except Exception as exc:
                        if self._debug:
                            print(f"[Evaluator] failed to append missing_response entry for id={rid}: {exc}")
                invalid += missing_count

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
            all_metrics[f'judge_{cat}_invalid'] = invalid
            total_invalid += all_metrics[f'judge_{cat}_invalid']

            qa = quality_acc.get(cat, {})
            bleu_bucket = qa.get('bleu') if qa else None
            if isinstance(bleu_bucket, dict):
                for label, vals in bleu_bucket.items():
                    key = f'quality_{cat}_bleu_{label}'
                    all_metrics[key] = (sum(vals) / len(vals)) if vals else 0.0
            rouge_bucket = qa.get('rouge') if qa else None
            if isinstance(rouge_bucket, dict):
                for label, vals in rouge_bucket.items():
                    key = f'quality_{cat}_rouge_{label}'
                    all_metrics[key] = (sum(vals) / len(vals)) if vals else 0.0
            bert_bucket = qa.get('bertscore') if qa else None
            if isinstance(bert_bucket, dict):
                for label, vals in bert_bucket.items():
                    key = f'quality_{cat}_bertscore_{label}'
                    all_metrics[key] = (sum(vals) / len(vals)) if vals else 0.0

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

    def compute_quality_metrics_only(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str],
        subtask_id: str = None
    ) -> Dict[str, float]:
        """Compute BLEU / ROUGE-L / BERTScore aggregates for the given task/category.

        This runs the same text-quality computations used when the LLM judge is present,
        but does not call or require the LLM judge. Returns a flat metrics dict.
        """
        category_metric_keys = {
            'defense': ['事实准确性', '法律关系准确性', '逻辑性', '完备性', '综合得分'],
            'fact': ['事实准确性', '相关性', '逻辑性', '综合得分'],
            'reasoning': ['争议焦点准确性', '法律关系准确性', '逻辑性', '伦理性', '综合得分'],
            'judgement': ['判决结果准确性', '引用法条完整性和准确性', '综合得分']
        }
        categories = [task_id] if task_id in category_metric_keys else []
        if not categories:
            return {}

        cat_prompts, cat_ids, cat_refs, cat_pred_texts = self._collect_casegen_judge_batches(records, predictions, categories)

        quality_acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for cat in categories:
            quality_acc[cat] = {
                'bleu': {label: [] for label, _ in self._BLEU_WEIGHTS} if sentence_bleu is not None else None,
                'rouge': {'precision': [], 'recall': [], 'f1': []} if _rouge_scorer_mod is not None else None,
                'bertscore': {'precision': [], 'recall': [], 'f1': []} if bert_score_func is not None else None
            }
            refs = cat_refs.get(cat, [])
            preds = cat_pred_texts.get(cat, [])
            for ref, pred_text in zip(refs, preds):
                qm = self._compute_text_metrics(ref, pred_text)
                self._append_quality_metrics(quality_acc[cat], qm)

        all_metrics: Dict[str, float] = {}
        for cat in categories:
            qa = quality_acc.get(cat, {})
            bleu_bucket = qa.get('bleu') if qa else None
            if isinstance(bleu_bucket, dict):
                for label, vals in bleu_bucket.items():
                    key = f'quality_{cat}_bleu_{label}'
                    all_metrics[key] = (sum(vals) / len(vals)) if vals else 0.0
            rouge_bucket = qa.get('rouge') if qa else None
            if isinstance(rouge_bucket, dict):
                for label, vals in rouge_bucket.items():
                    key = f'quality_{cat}_rouge_{label}'
                    all_metrics[key] = (sum(vals) / len(vals)) if vals else 0.0
            bert_bucket = qa.get('bertscore') if qa else None
            if isinstance(bert_bucket, dict):
                for label, vals in bert_bucket.items():
                    key = f'quality_{cat}_bertscore_{label}'
                    all_metrics[key] = (sum(vals) / len(vals)) if vals else 0.0

        return all_metrics

    # Classic metrics removed in judge-only mode.

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

