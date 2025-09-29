# LegalKit Web Interface

LegalKit Web Interface 是一个基于Web的交互式评测平台，为LegalKit法律大模型评测工具包提供了友好的图形化界面。

## 功能特性

### 🎯 核心功能
- **模型配置**: 支持本地模型、HuggingFace模型和API模型
- **数据集选择**: 支持CaseGen、JECQA、LAiW、LawBench、LegalBench、LexEval等数据集
- **加速器支持**: 支持VLLM和LMDeploy加速
- **任务管理**: 实时监控评测任务状态和进度
- **结果可视化**: 直观展示评测结果和性能指标

### 🖥️ 界面特性
- **响应式设计**: 支持桌面和移动设备
- **实时更新**: 自动刷新任务状态
- **美观简洁**: 现代化的Material Design风格
- **多语言支持**: 中文界面

## 快速开始

### 1. 安装依赖

```bash
cd /data/LegalKit/web
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 使用启动脚本（推荐）
./start.sh

# 或直接运行
python app.py
```

### 3. 访问界面

打开浏览器访问：`http://localhost:5000`

## 使用指南

### 📊 评测配置

1. **模型配置**
   - **本地模型**: 输入模型路径，支持模型发现功能
   - **HuggingFace模型**: 使用 `hf:model_name` 格式
   - **API模型**: 使用 `api:model_name` 格式，需配置API URL和Key

2. **数据集选择**
   - 从支持的数据集中选择一个或多个
   - 可选择特定子任务

3. **运行配置**
   - **任务类型**: 完整流程/仅推理/仅评估
   - **加速器**: 选择VLLM或LMDeploy加速
   - **并行配置**: 设置工作进程数和张量并行度

4. **生成参数**
   - 温度、Top-p、最大Token数等参数调整

### 📈 结果管理

- **任务列表**: 查看所有评测任务的状态
- **实时监控**: 任务进度实时更新
- **结果查看**: 详细的评测结果和性能指标
- **任务详情**: 查看配置参数和错误信息

### 🔧 系统监控

- **GPU状态**: 查看可用GPU和内存信息
- **系统信息**: 查看支持的数据集和加速器
- **资源监控**: 实时显示系统状态

## API 接口

### 主要端点

- `GET /api/datasets` - 获取可用数据集
- `POST /api/discover_models` - 发现模型
- `POST /api/submit_task` - 提交评测任务
- `GET /api/tasks` - 获取任务列表
- `GET /api/tasks/{task_id}` - 获取任务详情
- `GET /api/tasks/{task_id}/results` - 获取任务结果
- `GET /api/system_info` - 获取系统信息

### 请求示例

```bash
# 提交评测任务
curl -X POST http://localhost:5000/api/submit_task \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["path/to/model"],
    "datasets": ["LawBench"],
    "task": "all",
    "num_workers": 1,
    "temperature": 0.7
  }'
```

## 配置说明

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA支持（可选，用于GPU加速）

### 目录结构

```
web/
├── app.py              # Flask应用主文件
├── requirements.txt    # Python依赖
├── start.sh           # 启动脚本
├── static/
│   ├── css/
│   │   └── style.css  # 样式文件
│   └── js/
│       └── app.js     # 前端JavaScript
├── templates/
│   └── index.html     # 主页模板
└── web_run_output/    # 评测结果输出目录
```

### 配置选项

可以通过修改 `app.py` 中的变量来调整配置：

- `AVAILABLE_DATASETS`: 支持的数据集列表
- `ACCELERATORS`: 支持的加速器列表
- `TASK_PHASES`: 支持的任务类型

## 故障排除

### 常见问题

1. **模型导入失败**
   - 确保模型路径正确
   - 检查模型文件是否完整

2. **GPU不可用**
   - 检查CUDA安装
   - 确认GPU驱动正常

3. **任务执行失败**
   - 查看任务详情中的错误信息
   - 检查系统资源是否充足

4. **界面无法访问**
   - 确认Flask服务正常启动
   - 检查端口5000是否被占用

### 日志查看

Flask应用的日志会显示在终端中，包含：
- 任务提交信息
- 错误详情
- 系统状态

## 扩展开发

### 添加新数据集

1. 在LegalKit中实现数据集类
2. 在 `AVAILABLE_DATASETS` 中添加数据集名称
3. 重启Web服务

### 自定义界面

- 修改 `templates/index.html` 调整页面结构
- 修改 `static/css/style.css` 调整样式
- 修改 `static/js/app.js` 添加新功能

## 许可证

此项目遵循LegalKit的许可证条款。

## 贡献

欢迎提交问题和改进建议！