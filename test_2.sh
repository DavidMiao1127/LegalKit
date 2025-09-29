MODELS="/data/ms/ms-swift/megatron_output/Qwen3-4B-SFT/v86-20250808-011350/HF"

python legalkit/main.py --config example/config_jecqa.yaml --models $MODELS
# python legalkit/main.py --config example/config_lexeval.yaml --models $MODELS