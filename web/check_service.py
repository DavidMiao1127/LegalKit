#!/usr/bin/env python3

import requests
import json

def check_web_service():
    """检查Web服务状态"""
    
    print("🔍 检查LegalKit Web服务状态...")
    
    # 检查本地访问
    try:
        response = requests.get("http://localhost:5000/api/system_info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print("✅ Web服务运行正常！")
            print(f"   - GPU数量: {info.get('gpu_count', 0)}")
            print(f"   - 支持数据集: {len(info.get('datasets', []))}")
            print(f"   - 服务地址: http://localhost:5000")
            return True
        else:
            print(f"❌ Web服务响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到Web服务: {e}")
        return False

def show_access_info():
    """显示访问信息"""
    print("\n📱 Web界面功能:")
    print("   - 📊 评测配置: 模型选择、数据集配置、参数设置")
    print("   - 📈 结果管理: 任务监控、进度跟踪、结果查看") 
    print("   - 🔧 系统信息: GPU状态、资源监控")
    
    print("\n🌟 主要特性:")
    print("   - 支持本地模型、HuggingFace模型、API模型")
    print("   - 支持6个数据集: LawBench、LexEval、JECQA等")
    print("   - 实时任务监控和进度更新")
    print("   - 美观的响应式界面")

if __name__ == "__main__":
    if check_web_service():
        show_access_info()
        print("\n🎉 LegalKit Web界面已就绪，开始您的法律大模型评测之旅！")
    else:
        print("\n⚠️  Web服务似乎未正常运行，请检查启动状态")