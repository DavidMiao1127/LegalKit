# LegalKit

**LegalKit** is a practical and extensible evaluation toolkit for legal-domain Large Language Models (LLMs). It unifies **dataset adapters, model generation, offline JSON evaluation, and LLM-as-Judge scoring** into a single workflow, and provides an optional lightweight Web UI for non-terminal users.

---

## ✨ Features

* 📚 **Multi-dataset support**: legal QA, case reasoning, judgment generation, etc.
* 🔌 **Modular design**: pluggable `load_tasks`, `Generator`, `Evaluator`
* ⚙️ **Unified model specs**: local paths, HuggingFace (`hf:`), API endpoints (`api:`)
* ⚡ **Acceleration**: support for `vllm` and `lmdeploy` (tensor parallel + data parallel)
* 📄 **Offline evaluation**: evaluate directly from JSON predictions
* 🧑‍⚖️ **LLM-as-Judge**: independent configuration for judge models, batch evaluation supported
* 📊 **Multi-dimensional evaluation**: BLEU, Rouge-L, BERTScore + law-specific criteria
* 🔄 **Resumable runs**: checkpointing via sharded prediction storage
* 🌐 **Optional Web UI**: submit tasks and browse evaluation results

---

## 📦 Installation

### From source

```bash
conda create -n legalkit python=3.10 -y
conda activate legalkit

git clone https://github.com/DavidMiao1127/LegalKit.git
cd LegalKit
pip install -e .
```

### From PyPI

```bash
pip install legalkit
```

---

## 🗂 Project Structure

```
LegalKit/
  legalkit/
    main.py            # CLI entry: argument parsing & multiprocess orchestration
    judge.py           # Judge config + LLM-as-Judge runner
    storage.py         # Prediction sharding & aggregation
    datasets/          # Dataset adapters
  web/                 # Flask-based Web UI
  data/                # Built-in dataset artifacts & templates
  example/             # Example YAML configs
  run_output/          # Output directory
  README.md / README_zh.md
```

---

## ⚙️ Core Concepts

| Concept            | Description                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| **Model Spec**     | Local path, `hf:` (HuggingFace repo), or `api:` (remote endpoint)       |
| **Dataset**        | Defined under `legalkit/datasets/<Name>`, must implement `load_tasks()` |
| **Task**           | Unit of grouped records identified by ID                                |
| **Generator**      | Executes batched model inference                                        |
| **Evaluator**      | Computes metrics or rule-based evaluation                               |
| **Judge Runner**   | Independent model used for evaluation only                              |
| **StorageManager** | Handles sharded prediction files and merging                            |

---

## 🧪 CLI Parameters

| Flag                         | Type / Values       | Default                                        | Description                                                                              |
| ---------------------------- | ------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `--models`                   | list[str]           | None (optional in JSON mode)                   | Model specs: local dir, `hf:Org/Repo`, `api:Name`, or directory with multiple sub-models |
| `--datasets`                 | list[str]           | None (required)                                | Dataset names (under `legalkit/datasets/`)                                               |
| `--accelerator` / `-a`       | `vllm` / `lmdeploy` | None                                           | Accelerator backend for generation                                                       |
| `--num_workers`              | int                 | 1                                              | Number of data-parallel worker processes                                                 |
| `--tensor_parallel`          | int                 | 1                                              | Tensor/model parallel degree (scales GPU demand)                                         |
| `--task`                     | infer / eval / all  | all                                            | Phase: generation only / evaluation only / both. Forced to `eval` in JSON mode           |
| `--resume` / `-r`            | path                | –                                              | Resume from existing run directory (skip completed records)                              |
| `--output_dir`               | path                | ./run_output                                   | Root output directory (with timestamp subfolders)                                        |
| `--temperature`              | float               | 1.0                                            | Sampling temperature (primary model)                                                     |
| `--top_p`                    | float               | 1.0                                            | Nucleus sampling parameter                                                               |
| `--max_tokens`               | int                 | 8192                                           | Maximum generation tokens                                                                |
| `--sub_tasks`                | list[str]           | None                                           | Restrict to specific subtasks (dataset-defined)                                          |
| `--batch_size`               | int                 | 1                                              | Batch size per worker                                                                    |
| `--repetition_penalty`       | float               | 1.0                                            | Repetition penalty                                                                       |
| `--api_url`                  | str                 | None                                           | Base URL for `api:` model                                                                |
| `--api_key`                  | str                 | None                                           | API authentication key                                                                   |
| `--json_eval`                | flag                | False                                          | Enable offline JSON evaluation mode                                                      |
| `--json_paths`               | list[str]           | – (required in JSON mode)                      | Prediction files: single path or `Dataset=/abs/path.json` pairs                          |
| `--json_model_label`         | str                 | json_eval                                      | Synthetic model name in JSON mode (if `--models` not given)                              |
| `--judge`                    | str                 | None                                           | Judge model spec (same as `--models`)                                                    |
| `--judge_accelerator`        | `vllm` / `lmdeploy` | None                                           | Accelerator backend for judge model                                                      |
| `--judge_tensor_parallel`    | int                 | 1                                              | Tensor parallelism for judge model                                                       |
| `--judge_batch_size`         | int                 | 4 in JSON mode; else same as `batch_size` or 1 | Judge evaluation batch size                                                              |
| `--judge_temperature`        | float               | temperature or 0.0                             | Sampling temperature for judge                                                           |
| `--judge_top_p`              | float               | top_p or 1.0                                   | Nucleus sampling for judge                                                               |
| `--judge_max_tokens`         | int                 | max_tokens or 512                              | Maximum tokens for judge outputs                                                         |
| `--judge_repetition_penalty` | float               | repetition_penalty or 1.0                      | Repetition penalty for judge                                                             |
| `--judge_api_url`            | str                 | api_url                                        | Override API URL for judge                                                               |
| `--judge_api_key`            | str                 | api_key                                        | Override API key for judge                                                               |

---

## 🚀 Quick Start

Minimal run (generation + evaluation):

```bash
python legalkit/main.py \
  --models /path/to/local/model \
  --datasets CaseGen \
  --task all \
  --num_workers 1 --tensor_parallel 1 \
  --max_tokens 4096 --temperature 0.0
```

With vLLM acceleration:

```bash
python legalkit/main.py \
  --models /path/to/model \
  --datasets LawBench \
  --accelerator vllm \
  --tensor_parallel 2 \
  --num_workers 1
```

Resume from checkpoint:

```bash
python legalkit/main.py --resume run_output/20250710-093723 --task all
```

---

## 📄 Config-Driven Execution

Example `config.yaml`:

```yaml
args:
  models: ["/models/Qwen2.5-32B-Instruct"]
  datasets: ["JECQA"]
  sub_tasks: ["1-1", "1-2"]
  accelerator: lmdeploy
  num_workers: 2
  tensor_parallel: 4
  temperature: 0.0
  max_tokens: 4096
  task: all
```

Run with:

```bash
python legalkit/main.py --config example/config_jecqa.yaml
```

---

## 📥 Offline Evaluation (JSON Predictions)

Evaluate using existing predictions only (no model inference):

```bash
python legalkit/main.py \
  --datasets JECQA \
  --json_eval \
  --json_paths /data/jecqa_preds.json
```

Multiple datasets:

```bash
python legalkit/main.py \
  --datasets LawBench JECQA \
  --json_eval \
  --json_paths LawBench=/data/lawbench.json JECQA=/data/jecqa.json \
  --json_model_label merged_external
```

---

## 🧑‍⚖️ LLM-as-Judge

Introduce a secondary model to produce qualitative or structured evaluation scores.

Example:

```bash
python legalkit/main.py \
  --datasets CaseGen \
  --json_eval --json_paths /data/casegen_preds.json \
  --judge hf:Qwen/Qwen2.5-7B-Instruct \
  --judge_batch_size 2 --judge_max_tokens 256
```

---

## 🌐 Web UI (Optional)

Install dependencies:

```bash
pip install flask flask-cors
```

Run:

```bash
./start.sh
```

Access via browser:

```
http://localhost:5000
```

---

## 🔍 Output Structure

```
run_output/<TIMESTAMP>/
  config.yaml
  <model_id>/
    result.json        # Task -> score dictionary
    predict/           # Sharded predictions
```

---

## 📚 Supported Datasets

* **Comprehensive benchmarks**: LawBench, LegalBench, LexEval
* **Case generation / reasoning**: CaseGen, CaseHold
* **QA / Knowledge**: JECQA, LAiW, LexGLUE
* **Retrieval / RAG**: LexRAG
* **Judicial benchmarks (CAIL series 2019–2025)**
