# LegalKit

**LegalKit** 是一个实用且可扩展的法律领域大模型（LLM）评测工具包。它统一了 **数据集适配、模型生成、离线 JSON 评测、LLM-as-Judge 评审** 等流程，并提供一个可选的轻量级 Web UI，方便非命令行用户操作。

---

## ✨ 特性

* 📚 **多数据集支持**：法律问答、案例推理、裁判文书生成等
* 🔌 **模块化设计**：`load_tasks`、`Generator`、`Evaluator` 可插拔
* ⚙️ **统一模型规格**：支持本地路径、HuggingFace（`hf:`）、API（`api:`）
* ⚡ **推理加速**：支持 `vllm`、`lmdeploy`（张量并行 + 数据并行）
* 📄 **离线评测**：仅基于 JSON 预测结果进行评估
* 🧑‍⚖️ **LLM-as-Judge**：独立配置评审模型，支持批量评审
* 📊 **多维度评估**：BLEU、Rouge-L、BERTScore + 法律维度化评审
* 🔄 **可恢复运行**：断点续跑（按任务分片存储预测结果）
* 🌐 **可选 Web 界面**：提交任务、浏览评测结果

---

## 📦 安装

### 源码安装

```bash
conda create -n legalkit python=3.10 -y
conda activate legalkit

git clone https://github.com/DavidMiao1127/LegalKit.git
cd LegalKit
pip install -e .
```

### 从 PyPI 安装

```bash
pip install legalkit
```

---

## 🗂 项目结构

```
LegalKit/
  legalkit/
    main.py            # CLI 主入口：参数解析 & 多进程调度
    judge.py           # 评审模型配置 + LLM-as-Judge Runner
    storage.py         # 预测结果分片与聚合
    datasets/          # 数据集适配模块
  web/                 # Flask Web 界面
  data/                # 内置数据集 & 模板
  example/             # 示例配置（YAML）
  run_output/          # 运行输出目录
  README.md / README_zh.md
```

---

## ⚙️ 核心概念

| 概念                 | 说明                                            |
| ------------------ | --------------------------------------------- |
| **Model Spec**     | 支持本地路径、`hf:`（HuggingFace Repo）、`api:`（远程接口）   |
| **Dataset**        | `legalkit/datasets/<Name>`，需实现 `load_tasks()` |
| **Task**           | 按 ID 分组的任务单元                                  |
| **Generator**      | 执行模型推理（批量生成）                                  |
| **Evaluator**      | 指标计算或基于规则的评估                                  |
| **Judge Runner**   | 独立的评审模型，仅用于打分                                 |
| **StorageManager** | 管理预测文件分片与合并                                   |

---

## 🧪 参数列表

| 参数 | 类型 / 取值 | 默认值 | 功能说明 |
|------|-------------|----------------|----------|
| `--models` | 列表[str] | None（JSON 模式可省） | 模型规格：本地目录 / `hf:Org/Repo` / `api:名称` / 含多个子模型的目录。 |
| `--datasets` | 列表[str] | None（必填） | 评测数据集名称（对应 `legalkit/datasets/` 子目录）。 |
| `--accelerator` / `-a` | `vllm` / `lmdeploy` | None | 主模型生成加速后端。 |
| `--num_workers` | int | 1 | 数据并行进程数。 |
| `--tensor_parallel` | int | 1 | 张量/模型并行度（放大 GPU 需求）。 |
| `--task` | infer / eval / all | all | 阶段选择：仅生成 / 仅评估 / 二者。JSON 模式强制为 eval。 |
| `--resume` / `-r` | 路径 | – | 恢复既有运行目录（跳过已完成记录）。 |
| `--output_dir` | 路径 | ./run_output | 新运行根目录（内含时间戳子目录）。 |
| `--temperature` | float | 1.0 | 主模型采样温度。 |
| `--top_p` | float | 1.0 | 主模型 nucleus top-p。 |
| `--max_tokens` | int | 8192 | 主模型最大生成 token 数。 |
| `--sub_tasks` | 列表[str] | None | 只运行特定子任务（视数据集定义）。 |
| `--batch_size` | int | 1 | 主模型生成批大小（单 worker）。 |
| `--repetition_penalty` | float | 1.0 | 主模型重复惩罚。 |
| `--api_url` | str | None | `api:` 模型基础 URL。 |
| `--api_key` | str | None | `api:` 模型认证 Key。 |
| `--json_eval` | flag | False | 启用离线 JSON 评测模式。 |
| `--json_paths` | 列表[str] | –（需配合 json_eval） | 预测文件：单一路径或多条 `Dataset=/abs/path.json`。 |
| `--json_model_label` | str | json_eval | JSON 模式虚拟模型名（未提供 --models 时）。 |
| `--judge` | str | None | Judge 模型规格（与 --models 同格式）。 |
| `--judge_accelerator` | `vllm` / `lmdeploy` | None | Judge 模型加速后端。 |
| `--judge_tensor_parallel` | int | 1 | Judge 张量并行度。 |
| `--judge_batch_size` | int | JSON 模式=4，否则 batch_size 或 1 | Judge 生成批大小。 |
| `--judge_temperature` | float | temperature 或 0.0 | Judge 温度。 |
| `--judge_top_p` | float | top_p 或 1.0 | Judge top-p。 |
| `--judge_max_tokens` | int | max_tokens 或 512 | Judge 最大生成 tokens。 |
| `--judge_repetition_penalty` | float | repetition_penalty 或 1.0 | Judge 重复惩罚。 |
| `--judge_api_url` | str | api_url | 覆盖 Judge API URL。 |
| `--judge_api_key` | str | api_key | 覆盖 Judge API Key。 |

## 🚀 快速开始

最小化运行（生成 + 评估）：

```bash
python legalkit/main.py \
  --models /path/to/local/model \
  --datasets CaseGen \
  --task all \
  --num_workers 1 --tensor_parallel 1 \
  --max_tokens 4096 --temperature 0.0
```

使用 vLLM 加速：

```bash
python legalkit/main.py \
  --models /path/to/model \
  --datasets LawBench \
  --accelerator vllm \
  --tensor_parallel 2 \
  --num_workers 1
```

恢复运行：

```bash
python legalkit/main.py --resume run_output/20250710-093723 --task all
```

---

## 📄 配置驱动运行

示例 `config.yaml`：

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

运行：

```bash
python legalkit/main.py --config example/config_jecqa.yaml
```

---

## 📥 离线评测 (JSON Predictions)

只对已有预测结果进行评估（无需模型推理）。

```bash
python legalkit/main.py \
  --datasets JECQA \
  --json_eval \
  --json_paths /data/jecqa_preds.json
```

多数据集：

```bash
python legalkit/main.py \
  --datasets LawBench JECQA \
  --json_eval \
  --json_paths LawBench=/data/lawbench.json JECQA=/data/jecqa.json \
  --json_model_label merged_external
```

---

## 🧑‍⚖️ LLM-as-Judge

为评测任务引入二级模型，生成定性或结构化打分。

示例：

```bash
python legalkit/main.py \
  --datasets CaseGen \
  --json_eval --json_paths /data/casegen_preds.json \
  --judge hf:Qwen/Qwen2.5-7B-Instruct \
  --judge_batch_size 2 --judge_max_tokens 256
```

---

## 🌐 Web 界面（可选）

安装依赖：

```bash
pip install flask flask-cors
```

运行：

```bash
./start.sh
```

浏览器访问：

```
http://localhost:5000
```

---

## 🔍 输出结构

```
run_output/<TIMESTAMP>/
  config.yaml
  <model_id>/
    result.json        # 任务 -> 指标分数字典
    predict/           # 分片预测结果
```

---


## 📚 当前支持的数据集

* **综合基准**：LawBench, LegalBench, LexEval
* **案例生成/推理**：CaseGen, CaseHold
* **问答/知识**：JECQA, LAiW, LexGLUE
* **检索/RAG**：LexRAG
* **司法赛题（CAIL 系列 2019–2025）**