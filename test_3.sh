MODELS="/data/ms-swift-new/megatron_output/Qwen3-4B-Mix/v1-20250728-052810/iter_0016000/HF"

python legalkit/main.py --config example/config_jecqa.yaml --models $MODELS
# python legalkit/main.py --config example/config_lexeval.yaml --models $MODELS