from typing import List, Dict, Tuple

class Generator:
    def __init__(self, model):
        self.model = model

    def generate(
        self,
        task_id: str,
        records: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        def fmt_options(option_list: Dict[str, str]) -> str:
            if not isinstance(option_list, dict):
                return ""
            # Sort by key A, B, C, D...
            keys = sorted(option_list.keys())
            return "\n".join([f"{k}. {option_list[k]}" for k in keys])

        prompts = []
        for rec in records:
            statement = rec.get('statement', '')
            options_str = fmt_options(rec.get('option_list', {}))
            prompt = (
                "请依据题干，从给定选项中选择所有正确的选项，直接输出选项字母（如 A 或 AB），无需解释。\n"
                f"题干：{statement}\n"
                f"选项：\n{options_str}\n"
            )
            prompts.append(prompt)
        generated_list = self.model.generate(prompts)
        return prompts, generated_list

