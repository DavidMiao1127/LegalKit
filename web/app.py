#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import uuid
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import yaml
from typing import Dict, Any, List
import queue
import multiprocessing as mp
from pathlib import Path

# Add parent directory to Python path to import legalkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from legalkit.main import run_worker, discover_models, parse_json_path_specs, JSON_MODEL_PREFIX
from legalkit.storage import StorageManager
from multiprocessing import Barrier, get_context
import torch

app = Flask(__name__)
CORS(app)

# Global state for task management
running_tasks = {}  # task_id -> task_info
task_lock = threading.Lock()

# Available datasets and models
AVAILABLE_DATASETS = ["CaseGen", "JECQA", "LAiW", "LawBench", "LegalBench", "LexEval", "LexRAG", "LegalAgentBench", "CAIL2025"]
ACCELERATORS = ["vllm", "lmdeploy"]
TASK_PHASES = ["infer", "eval", "all"]

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
        
    def create_task(self, config: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        with self.lock:
            self.tasks[task_id] = {
                'id': task_id,
                'config': config,
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'started_at': None,
                'completed_at': None,
                'progress': 0,
                'results': None,
                'error': None,
                'log': []
            }
        return task_id
    
    def get_task(self, task_id: str) -> Dict[str, Any]:
        with self.lock:
            return self.tasks.get(task_id, {})
    
    def update_task(self, task_id: str, **kwargs):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.tasks.values())

task_manager = TaskManager()

def run_evaluation_task(task_id: str, config: Dict[str, Any]):
    """Run evaluation task in a separate thread"""
    try:
        task_manager.update_task(task_id, 
                                status='running', 
                                started_at=datetime.now().isoformat())
        
        # Prepare merged_args similar to main.py
        merged_args = config.copy()
        # Normalize datasets/models types
        if isinstance(merged_args.get('datasets'), str):
            merged_args['datasets'] = [merged_args['datasets']]

        # JSON evaluation mode handling
        json_eval_mode = bool(merged_args.get('json_eval'))
        if json_eval_mode:
            json_paths = merged_args.get('json_paths')
            if isinstance(json_paths, str):
                json_paths = [json_paths]
            if not json_paths:
                raise ValueError("json_eval 已开启但未提供 json_paths")
            if not merged_args.get('datasets'):
                raise ValueError("json_eval 模式需要 datasets")
            merged_args['json_inputs'] = parse_json_path_specs(json_paths, merged_args['datasets'])
            # force task phase to eval
            merged_args['task'] = 'eval'
            # Ensure a JSON model spec if user didn't give real models
            if not merged_args.get('models'):
                label = merged_args.get('json_model_label', 'json_eval')
                merged_args['models'] = [f"{JSON_MODEL_PREFIX}{label}"]
        else:
            merged_args['json_inputs'] = {}
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_root = os.path.join("./web_run_output", f"{task_id}_{timestamp}")
        os.makedirs(run_root, exist_ok=True)
        
        # Save config
        with open(os.path.join(run_root, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump({'args': merged_args}, f, allow_unicode=True)
        
        # Set up multiprocessing parameters
        num_workers = int(merged_args.get('num_workers', 1))
        tensor_parallel = int(merged_args.get('tensor_parallel', 1))
        
        # Judge config normalization (optional)
        judge_spec = merged_args.get('judge')
        if judge_spec:
            judge_gen_cfg = {
                'temperature': merged_args.get('judge_temperature', merged_args.get('temperature', 0.0)),
                'top_p': merged_args.get('judge_top_p', merged_args.get('top_p', 1.0)),
                'max_tokens': merged_args.get('judge_max_tokens', merged_args.get('max_tokens', 512)),
                'rep_penalty': merged_args.get('judge_repetition_penalty', merged_args.get('repetition_penalty', 1.0)),
            }
            merged_args['judge_config'] = {
                'model_spec': judge_spec,
                'accelerator': merged_args.get('judge_accelerator'),
                'tensor_parallel': int(merged_args.get('judge_tensor_parallel', 1)),
                'batch_size': int(merged_args.get('judge_batch_size', 4 if json_eval_mode else merged_args.get('batch_size', 1) or 1)),
                'gen_cfg': judge_gen_cfg,
                'api_url': merged_args.get('judge_api_url', merged_args.get('api_url')),
                'api_key': merged_args.get('judge_api_key', merged_args.get('api_key')),
            }
        else:
            merged_args['judge_config'] = None

        # Validate GPU requirements only if generation required
        requires_inference = (merged_args.get('task') in ('infer', 'all')) and not json_eval_mode
        has_real_models = any(not str(spec).startswith(JSON_MODEL_PREFIX) for spec in merged_args.get('models', []))
        total_gpus_required = num_workers * tensor_parallel if requires_inference and has_real_models else 0
        available_gpus = torch.cuda.device_count()
        if total_gpus_required > available_gpus:
            raise ValueError(f"Not enough GPUs: requested {total_gpus_required}, available {available_gpus}")
        
        task_manager.update_task(task_id, progress=10)
        
        # Run the evaluation
        mp.set_start_method("spawn", force=True)
        ctx = get_context('spawn')
        barrier = ctx.Barrier(num_workers)
        
        if num_workers > 1:
            mp.spawn(
                run_worker,
                args=(num_workers, merged_args, {}, run_root, barrier),
                nprocs=num_workers,
                join=True
            )
        else:
            run_worker(0, 1, merged_args, {}, run_root, barrier)
        
        # Load results
        results = {}
        for model_spec in merged_args['models']:
            if os.path.isdir(model_spec):
                model_configs = discover_models(model_spec)
                if not model_configs:
                    model_configs = [{"model_type": "local", "model_path": model_spec}]
            else:
                if model_spec.startswith("hf:"):
                    model_configs = [{"model_type": "hf", "model_name": model_spec.split("hf:")[1]}]
                elif model_spec.startswith("api:"):
                    model_configs = [{"model_type": "api", "model_name": model_spec.split("api:")[1]}]
                else:
                    model_configs = [{"model_type": "local", "model_path": model_spec}]
            
            for model_cfg in model_configs:
                model_id = model_cfg.get('model_path', model_cfg.get('model_name', 'unknown'))
                result_file = os.path.join(run_root, model_id.replace("/", "_"), 'result.json')
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        model_results = json.load(f)
                        results[model_id] = model_results
        
        task_manager.update_task(task_id, 
                                status='completed',
                                completed_at=datetime.now().isoformat(),
                                progress=100,
                                results=results)
        
    except Exception as e:
        task_manager.update_task(task_id, 
                                status='failed',
                                completed_at=datetime.now().isoformat(),
                                error=str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/datasets')
def get_datasets():
    """Get available datasets"""
    return jsonify(AVAILABLE_DATASETS)

@app.route('/api/discover_models', methods=['POST'])
def discover_models_api():
    """Discover models from a directory path"""
    data = request.get_json()
    path = data.get('path', '')
    
    if not path or not os.path.exists(path):
        return jsonify({'error': 'Invalid path'}), 400
    
    models = discover_models(path)
    return jsonify(models)

@app.route('/api/submit_task', methods=['POST'])
def submit_task():
    """Submit a new evaluation task"""
    try:
        config = request.get_json()
        
        # Validate required fields
        if not config.get('models') or not config.get('datasets'):
            return jsonify({'error': 'Models and datasets are required'}), 400
        
        # Create task
        task_id = task_manager.create_task(config)
        
        # Start evaluation in background thread
        thread = threading.Thread(target=run_evaluation_task, args=(task_id, config))
        thread.daemon = True
        thread.start()
        
        return jsonify({'task_id': task_id, 'status': 'submitted'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks')
def list_tasks():
    """List all tasks"""
    tasks = task_manager.list_tasks()
    return jsonify(tasks)

@app.route('/api/tasks/<task_id>')
def get_task(task_id):
    """Get specific task details"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)

@app.route('/api/tasks/<task_id>/results')
def get_task_results(task_id):
    """Get task results"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    if task['status'] != 'completed':
        return jsonify({'error': 'Task not completed yet'}), 400
    
    return jsonify(task.get('results', {}))

@app.route('/api/system_info')
def get_system_info():
    """Get system information"""
    try:
        gpu_count = torch.cuda.device_count()
        gpu_info = []
        for i in range(gpu_count):
            gpu_info.append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory // 1024**3,  # GB
            })
        
        return jsonify({
            'gpu_count': gpu_count,
            'gpu_info': gpu_info,
            'datasets': AVAILABLE_DATASETS,
            'accelerators': ACCELERATORS,
            'task_phases': TASK_PHASES
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create output directory
    os.makedirs('./web_run_output', exist_ok=True)
    
    print("Starting LegalKit Web Interface...")
    print("Available datasets:", AVAILABLE_DATASETS)
    print("Available accelerators:", ACCELERATORS)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)