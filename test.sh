MODELS="/data/glm-train-prod/SFT/Qwen3-4B-Base/iter_0010000/HF"

python legalkit/main.py --config example/config_jecqa.yaml --models $MODELS
# python legalkit/main.py --config example/config_lexeval.yaml --models $MODELS