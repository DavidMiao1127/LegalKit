#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LegalKit Web Interface Test Script
测试Web界面的基本功能
"""

import requests
import json
import time
import sys

def test_web_interface():
    base_url = "http://localhost:5000"
    
    print("🧪 LegalKit Web Interface 测试")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("1. 检查服务器状态...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ 服务器运行正常")
        else:
            print(f"   ❌ 服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ 无法连接到服务器: {e}")
        print("   请先启动Web服务: python app.py")
        return False
    
    # Test 2: Get system info
    print("2. 获取系统信息...")
    try:
        response = requests.get(f"{base_url}/api/system_info")
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ GPU数量: {info.get('gpu_count', 0)}")
            print(f"   ✅ 支持数据集: {len(info.get('datasets', []))}")
            print(f"   ✅ 支持加速器: {len(info.get('accelerators', []))}")
        else:
            print(f"   ❌ 获取系统信息失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 系统信息请求失败: {e}")
    
    # Test 3: Get datasets
    print("3. 获取数据集列表...")
    try:
        response = requests.get(f"{base_url}/api/datasets")
        if response.status_code == 200:
            datasets = response.json()
            print(f"   ✅ 可用数据集: {datasets}")
        else:
            print(f"   ❌ 获取数据集失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 数据集请求失败: {e}")
    
    # Test 4: Test model discovery (with a dummy path)
    print("4. 测试模型发现功能...")
    try:
        test_path = "/nonexistent/path"
        response = requests.post(f"{base_url}/api/discover_models", 
                               json={"path": test_path})
        if response.status_code == 400:  # Expected for non-existent path
            print("   ✅ 模型发现API正常响应")
        else:
            print(f"   ⚠️  模型发现返回异常状态码: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 模型发现请求失败: {e}")
    
    # Test 5: Get tasks list
    print("5. 获取任务列表...")
    try:
        response = requests.get(f"{base_url}/api/tasks")
        if response.status_code == 200:
            tasks = response.json()
            print(f"   ✅ 当前任务数量: {len(tasks)}")
        else:
            print(f"   ❌ 获取任务列表失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 任务列表请求失败: {e}")
    
    print("\n🎉 基本功能测试完成!")
    print("\n📋 使用指南:")
    print("1. 打开浏览器访问: http://localhost:5000")
    print("2. 在评测配置页面设置模型和数据集")
    print("3. 点击'开始评测'提交任务")
    print("4. 在结果管理页面查看任务状态")
    
    return True

def test_submit_dummy_task():
    """提交一个测试任务（不会真正执行）"""
    base_url = "http://localhost:5000"
    
    print("\n🚀 提交测试任务...")
    
    # 这是一个模拟任务，不会真正执行评测
    test_config = {
        "models": ["dummy_model_for_test"],
        "datasets": ["LawBench"],
        "task": "infer",
        "num_workers": 1,
        "tensor_parallel": 1,
        "batch_size": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 100,
        "repetition_penalty": 1.0
    }
    
    try:
        response = requests.post(f"{base_url}/api/submit_task",
                               json=test_config)
        if response.status_code == 200:
            result = response.json()
            task_id = result.get('task_id')
            print(f"   ✅ 任务提交成功! Task ID: {task_id}")
            
            # 等待几秒然后检查任务状态
            print("   ⏳ 等待3秒后检查任务状态...")
            time.sleep(3)
            
            status_response = requests.get(f"{base_url}/api/tasks/{task_id}")
            if status_response.status_code == 200:
                task_info = status_response.json()
                print(f"   📊 任务状态: {task_info.get('status')}")
                print(f"   📈 任务进度: {task_info.get('progress', 0)}%")
            
        else:
            error_info = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"   ❌ 任务提交失败: {error_info}")
    except Exception as e:
        print(f"   ❌ 任务提交请求失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--submit-test":
        # 只运行基本测试，然后提交测试任务
        if test_web_interface():
            test_submit_dummy_task()
    else:
        # 只运行基本功能测试
        test_web_interface()