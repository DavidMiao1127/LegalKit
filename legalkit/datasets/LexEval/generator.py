from typing import List, Dict, Tuple
import os
import json

class Generator:
    def __init__(self, model):
        self.model = model
        self.is_few_shot = str(os.getenv("FEW_SHOT", "0")).lower() in ("1", "true", "yes")
        pkg_dir = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(pkg_dir, "..", "..", ".."))
        self.few_shot_root = os.path.join(repo_root, "data", "lexeval", "few_shot")
        self.tokenizer = getattr(model, "tokenizer", None)
        self.max_model_len = getattr(model, "max_model_len", 8192)

    def _label_for_task(self, task_name: str) -> str:
        if task_name in ("5_1", "5_1_few_shot"):
            return "摘要:"
        if task_name in ("5_2", "5_2_few_shot"):
            return "裁判分析过程:"
        if task_name in ("5_3", "5_3_few_shot"):
            return "翻译结果:"
        return "答案:"

    def _get_fewshot_examples(self, task_name: str, max_examples: int = 3, include_prompt_header: bool = True) -> str:
        """Build few-shot examples string from jsonl under few_shot_root/<task_name>.jsonl.
        Fallback to base name without suffix _few_shot. Optionally limit to `max_examples` and
        control whether to append the trailing prompt header ("请你回答:\n问题:").
        """
        fname = f"{task_name}.jsonl"
        path = os.path.join(self.few_shot_root, fname)
        if not os.path.exists(path) and task_name.endswith("_few_shot"):
            base = task_name.replace("_few_shot", "")
            path = os.path.join(self.few_shot_root, f"{base}.jsonl")

        if not os.path.exists(path) or max_examples <= 0:
            return "请你回答:\n问题:" if include_prompt_header else ""

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            return "请你回答:\n问题:" if include_prompt_header else ""

        # Only keep up to max_examples lines
        lines = lines[:max_examples]

        header = f"以下是{len(lines)}个例子:\n" if lines else ""
        examples = header
        for line in lines:
            try:
                ex = json.loads(line)
            except Exception:
                continue
            # ex should have input and answer fields
            inp = ex.get("input", "")
            ans = ex.get("answer", "")
            if task_name in ("5_1", "5_1_few_shot"):
                examples += f"问题: {inp}\n摘要: {ans}\n\n"
            elif task_name in ("5_2", "5_2_few_shot"):
                examples += f"问题: {inp}\n裁判分析过程: {ans}\n\n"
            elif task_name in ("5_3", "5_3_few_shot"):
                examples += f"问题: {inp}\n翻译结果: {ans}\n\n"
            else:
                examples += f"问题: {inp}\n答案: {ans}\n\n"
        if include_prompt_header:
            examples += "请你回答:\n问题:"
        return examples

    @staticmethod
    def _truncate_long(prompt: str, context_length: int, tokenizer, q_type: str) -> str:
        if tokenizer is None:
            return prompt
        ori_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        if q_type == 'generation':
            if len(ori_prompt) > context_length - 512:
                half = int((context_length - 512) / 2)
                left = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True)
                right = tokenizer.decode(ori_prompt[-half:], skip_special_tokens=True)
                prompt = left + right
        elif q_type == 'multiple_choice':
            if len(ori_prompt) > context_length - 20:
                half = int((context_length - 20) / 2)
                left = tokenizer.decode(ori_prompt[:half], skip_special_tokens=True)
                right = tokenizer.decode(ori_prompt[-half:], skip_special_tokens=True)
                prompt = left + right
        else:
            raise ValueError(f"Wrong question type: {q_type}")
        return prompt

    def generate(
        self,
        task_id: str,
        records: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        prompts: List[str] = []
        # Determine question type heuristic: text generation for 5_1/5_2/5_3 else multiple-choice/short
        q_type = 'generation' if task_id.split(':')[-1] in ("5_1", "5_2", "5_3", "5_1_few_shot", "5_2_few_shot", "5_3_few_shot") else 'multiple_choice'

        label = self._label_for_task(task_id)

        # Token length budget based on question type
        context_length = int(self.max_model_len)
        reserve = 512 if q_type == 'generation' else 20
        budget = max(0, context_length - reserve)

        def token_len(text: str) -> int:
            if self.tokenizer is None:
                return len(text)
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
            if self.tokenizer is None:
                return text[:max_tokens]
            toks = self.tokenizer.encode(text, add_special_tokens=False)
            if len(toks) <= max_tokens:
                return text
            # Keep the tail of instruction to preserve most recent guidance
            kept = toks[-max_tokens:]
            return self.tokenizer.decode(kept, skip_special_tokens=True)

        for rec in records:
            instruction = rec.get('instruction', '')
            # LexEval dataset stores the question under 'question' (sourced from original 'input')
            q_text = rec.get('question')
            if q_text is None:
                q_text = rec.get('input', '')

            # Build the immutable question part; we prioritize keeping it intact
            question_part = f"请你回答:\n问题: {q_text}\n{label}"

            if not self.is_few_shot:
                base_prompt = f"{instruction}\n{question_part}"
                # If still too long, only truncate instruction; keep question intact
                if token_len(base_prompt) > budget:
                    allow_instr = max(0, budget - token_len(question_part) - 1)
                    instruction_cut = truncate_text_to_tokens(instruction, allow_instr)
                    base_prompt = f"{instruction_cut}\n{question_part}"
                prompts.append(base_prompt)
                continue

            # Try with decreasing number of few-shot examples: 3 -> 1 -> 0
            chosen_prompt = None
            for ex_num in (3, 1, 0):
                examples = self._get_fewshot_examples(task_id, max_examples=ex_num, include_prompt_header=False)
                if examples:
                    prompt_candidate = f"{instruction}\n{examples}请你回答:\n问题: {q_text}\n{label}"
                else:
                    prompt_candidate = f"{instruction}\n{question_part}"

                if token_len(prompt_candidate) <= budget:
                    chosen_prompt = prompt_candidate
                    break

            if chosen_prompt is None:
                # Even without examples it's too long; drop examples and truncate only instruction
                allow_instr = max(0, budget - token_len(question_part) - 1)
                instruction_cut = truncate_text_to_tokens(instruction, allow_instr)
                chosen_prompt = f"{instruction_cut}\n{question_part}"

            prompts.append(chosen_prompt)
        generated_list = self.model.generate(prompts)
        return prompts, generated_list

