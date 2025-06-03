# LegalKit: a toolbox for legal models evaluation

## Install
```bash
conda create -n legalkit python=3.10
conda activate legalkit
cd LegalKit
pip install -e .
```

## Run
- run by config.yaml
    - write a config file
    
    ```yaml
    args:
      models: ["/haitao/workspace/model/Qwen2.5-32B-Instruct"]		# list of models, also can be local path or local folder containing multiple checkpoints
      datasets: ["JECQA"]	# list of datasets
      sub_tasks: ["1-1","1-2"]  	# list of subtasks(if enabled, datasets have to be single)
      accelerator: "lmdeploy"		# accelerator, currently support lmdeploy
      num_workers: 2				# number of workers that split the task
      tensor_parallel: 4			# number of tp for each worker
      temperature: 0.0
      top_p: 0.9
      max_tokens: 4096
      task: "all"					# choice: all, infer, eval
      resume: null					# direct of path for resuming (folder name should be a timestamp)
    ```
    
    - load the config file and run
    
    ```bash
    python legalkit/main.py --config example/config.yaml
    ```
- run by CLI

```bash
 python legalkit/main.py
  
  --models MODELS [MODELS ...]
                        Model specs: local paths, hf shortcuts, or api labels
  --datasets DATASETS [DATASETS ...]
                        Dataset names
  -a {vllm,lmdeploy}, --accelerator {vllm,lmdeploy}
                        Acceleration backend
  --num_workers NUM_WORKERS
                        Number of parallel workers (data parallelism)
  --tensor_parallel TENSOR_PARALLEL
                        Tensor parallelism degree (model parallelism)
  --task {infer,eval,all}
                        Phase to run
  -r RESUME, --resume RESUME
                        Path to existing run directory to resume
  --output_dir OUTPUT_DIR
                        Directory for new outputs
  --temperature TEMPERATURE
                        Sampling temperature
  --top_p TOP_P         Top-p sampling
  --max_tokens MAX_TOKENS
                        Maximum number of tokens to generate
  --sub_tasks SUB_TASKS [SUB_TASKS ...]
                        Sub-tasks to run
```
After running, the evaluation result will be stored in output_dir, with generation output and scores.

## API Use
```yaml
args:
  models: ["api:Qwen/Qwen3-8B"]     # format:"api:{model_name}"
  api_url: https://api.siliconflow.cn/v1/chat/completions       # base url of api(take siliconflow as an example)
  api_key: sk-ylmpihzxhcjcmralmhgzoxhdzvfbmxkdcelnlcrqqhjisvky  # api key
  datasets: ["JECQA"]
  sub_tasks: ["0_test"]
  temperature: 0.6
  top_p: 0.9
  max_tokens: 8192
  batch_size: 8
  resume: null
```