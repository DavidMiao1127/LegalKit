def clean_prediction(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()
