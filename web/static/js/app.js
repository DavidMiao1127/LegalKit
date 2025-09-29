// LegalKit Web Interface JavaScript

class LegalKitApp {
    constructor() {
        this.baseUrl = '/api';
        this.tasks = new Map();
        this.refreshInterval = null;
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadInitialData();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('evaluationForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitEvaluationTask();
        });

        // Model type change
        document.getElementById('modelType').addEventListener('change', (e) => {
            this.toggleApiConfig(e.target.value === 'api');
        });

        // Model discovery
        document.getElementById('discoverModels').addEventListener('click', () => {
            this.discoverModels();
        });

        // JSON eval toggle
        const jsonSwitch = document.getElementById('jsonEvalSwitch');
        jsonSwitch.addEventListener('change', (e) => {
            document.getElementById('jsonEvalConfig').style.display = e.target.checked ? 'block' : 'none';
            // If enabled, force task phase UI to eval (visual hint only)
            if (e.target.checked) {
                document.getElementById('taskPhase').value = 'eval';
            }
        });

        // Judge toggle
        const judgeSwitch = document.getElementById('judgeSwitch');
        judgeSwitch.addEventListener('change', (e) => {
            document.getElementById('judgeConfig').style.display = e.target.checked ? 'block' : 'none';
        });

        // Refresh tasks
        document.getElementById('refreshTasks').addEventListener('click', () => {
            this.loadTasks();
        });

        // Tab switching
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const target = e.target.getAttribute('href').substring(1);
                if (target === 'results') {
                    this.loadTasks();
                } else if (target === 'system') {
                    this.loadSystemInfo();
                }
            });
        });
    }

    async loadInitialData() {
        try {
            await Promise.all([
                this.loadDatasets(),
                this.loadSystemInfo(),
                this.loadRecentTasks()
            ]);
        } catch (error) {
            this.showError('加载初始数据失败: ' + error.message);
        }
    }

    async loadDatasets() {
        try {
            const response = await fetch(`${this.baseUrl}/datasets`);
            const datasets = await response.json();
            
            const select = document.getElementById('datasetSelect');
            select.innerHTML = '';
            
            datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset;
                option.textContent = dataset;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading datasets:', error);
            this.showError('加载数据集失败');
        }
    }

    async loadSystemInfo() {
        try {
            const response = await fetch(`${this.baseUrl}/system_info`);
            const info = await response.json();
            
            this.displaySystemInfo(info);
            this.displayGpuInfo(info.gpu_info);
            this.displaySupportedDatasets(info.datasets);
        } catch (error) {
            console.error('Error loading system info:', error);
            document.getElementById('systemInfo').innerHTML = 
                '<div class="alert alert-danger">加载系统信息失败</div>';
        }
    }

    displaySystemInfo(info) {
        const systemInfoDiv = document.getElementById('systemInfo');
        systemInfoDiv.innerHTML = `
            <div class="system-metric">
                <span class="metric-value">${info.gpu_count}</span>
                <span class="metric-label">可用GPU</span>
            </div>
            <div class="row">
                <div class="col-4">
                    <div class="text-center">
                        <strong>12</strong><br>
                        <small class="text-muted">数据集</small>
                    </div>
                </div>
                <div class="col-4">
                    <div class="text-center">
                        <strong>${info.accelerators.length}</strong><br>
                        <small class="text-muted">加速器</small>
                    </div>
                </div>
                <div class="col-4">
                    <div class="text-center">
                        <strong>6</strong><br>
                        <small class="text-muted">任务类型</small>
                    </div>
                </div>
            </div>
        `;
    }

    displayGpuInfo(gpuInfo) {
        const gpuInfoDiv = document.getElementById('gpuInfo');
        if (!gpuInfo || gpuInfo.length === 0) {
            gpuInfoDiv.innerHTML = '<div class="alert alert-warning">未检测到GPU</div>';
            return;
        }

        const gpuCards = gpuInfo.map(gpu => `
            <div class="gpu-card">
                <div class="gpu-name">GPU ${gpu.id}: ${gpu.name}</div>
                <div class="gpu-memory">
                    <i class="bi bi-memory"></i> ${gpu.memory_total} GB
                </div>
            </div>
        `).join('');

        gpuInfoDiv.innerHTML = gpuCards;
    }

    displaySupportedDatasets(datasets) {
        const datasetsDiv = document.getElementById('supportedDatasets');
        const datasetItems = datasets.map(dataset => `
            <div class="dataset-item">
                <i class="bi bi-database"></i>
                ${dataset}
            </div>
        `).join('');

        datasetsDiv.innerHTML = datasetItems;
    }

    async loadRecentTasks() {
        try {
            const response = await fetch(`${this.baseUrl}/tasks`);
            const tasks = await response.json();
            
            // Sort by creation time and take the 5 most recent
            const recentTasks = tasks
                .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
                .slice(0, 5);

            this.displayRecentTasks(recentTasks);
        } catch (error) {
            console.error('Error loading recent tasks:', error);
        }
    }

    displayRecentTasks(tasks) {
        const recentTasksDiv = document.getElementById('recentTasks');
        
        if (tasks.length === 0) {
            recentTasksDiv.innerHTML = '<p class="text-muted">暂无任务</p>';
            return;
        }

        const taskItems = tasks.map(task => `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <div>
                    <div class="fw-bold">${task.id.substring(0, 8)}...</div>
                    <small class="text-muted">${this.formatDate(task.created_at)}</small>
                </div>
                <span class="status-badge status-${task.status}">${this.getStatusText(task.status)}</span>
            </div>
        `).join('');

        recentTasksDiv.innerHTML = taskItems;
    }

    async loadTasks() {
        try {
            const response = await fetch(`${this.baseUrl}/tasks`);
            const tasks = await response.json();
            
            this.displayTasksList(tasks);
        } catch (error) {
            console.error('Error loading tasks:', error);
            this.showError('加载任务列表失败');
        }
    }

    displayTasksList(tasks) {
        const tasksListDiv = document.getElementById('tasksList');
        
        if (tasks.length === 0) {
            tasksListDiv.innerHTML = '<div class="alert alert-info">暂无评测任务</div>';
            return;
        }

        const tasksTable = `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>任务ID</th>
                            <th>状态</th>
                            <th>数据集</th>
                            <th>模型</th>
                            <th>创建时间</th>
                            <th>进度</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tasks.map(task => this.renderTaskRow(task)).join('')}
                    </tbody>
                </table>
            </div>
        `;

        tasksListDiv.innerHTML = tasksTable;
    }

    renderTaskRow(task) {
        const datasets = task.config.datasets ? task.config.datasets.join(', ') : 'N/A';
        const models = task.config.models ? task.config.models.map(m => 
            m.length > 30 ? m.substring(0, 30) + '...' : m
        ).join(', ') : 'N/A';

        return `
            <tr class="task-row" onclick="app.showTaskDetail('${task.id}')">
                <td>
                    <code>${task.id.substring(0, 8)}...</code>
                </td>
                <td>
                    <span class="status-badge status-${task.status}">
                        ${this.getStatusText(task.status)}
                    </span>
                </td>
                <td>${datasets}</td>
                <td title="${task.config.models ? task.config.models.join(', ') : ''}">${models}</td>
                <td>${this.formatDate(task.created_at)}</td>
                <td>
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar" style="width: ${task.progress || 0}%"></div>
                    </div>
                    <small>${task.progress || 0}%</small>
                </td>
                <td>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary" onclick="event.stopPropagation(); app.showTaskDetail('${task.id}')">
                            <i class="bi bi-eye"></i>
                        </button>
                        ${task.status === 'completed' ? `
                            <button class="btn btn-outline-success" onclick="event.stopPropagation(); app.showTaskResults('${task.id}')">
                                <i class="bi bi-graph-up"></i>
                            </button>
                        ` : ''}
                    </div>
                </td>
            </tr>
        `;
    }

    toggleApiConfig(show) {
        const apiConfig = document.getElementById('apiConfig');
        apiConfig.style.display = show ? 'block' : 'none';
    }

    async discoverModels() {
        const modelPath = document.getElementById('modelPath').value;
        if (!modelPath) {
            this.showError('请输入模型路径');
            return;
        }

        try {
            const response = await fetch(`${this.baseUrl}/discover_models`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: modelPath })
            });

            const result = await response.json();
            if (response.ok) {
                this.displayDiscoveredModels(result);
            } else {
                this.showError(result.error || '模型发现失败');
            }
        } catch (error) {
            this.showError('模型发现失败: ' + error.message);
        }
    }

    displayDiscoveredModels(models) {
        const modelListDiv = document.getElementById('modelList');
        
        if (models.length === 0) {
            modelListDiv.innerHTML = '<div class="alert alert-warning">未找到有效模型</div>';
            return;
        }

        const modelItems = models.map(model => `
            <div class="model-item">
                <div class="model-path">${model.model_path}</div>
                <span class="model-type">${model.model_type}</span>
            </div>
        `).join('');

        modelListDiv.innerHTML = `
            <div class="mt-2">
                <strong>发现的模型:</strong>
                ${modelItems}
            </div>
        `;
    }

    async submitEvaluationTask() {
        try {
            const config = this.getFormConfig();
            this.validateConfig(config);

            const response = await fetch(`${this.baseUrl}/submit_task`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const result = await response.json();
            if (response.ok) {
                this.showSuccess(`任务提交成功! 任务ID: ${result.task_id}`);
                this.loadRecentTasks();
                // Switch to results tab
                document.querySelector('[href="#results"]').click();
            } else {
                this.showError(result.error || '任务提交失败');
            }
        } catch (error) {
            this.showError('任务提交失败: ' + error.message);
        }
    }

    getFormConfig() {
        const modelType = document.getElementById('modelType').value;
        const modelPath = document.getElementById('modelPath').value;
        
        let models = [];
        if (modelType === 'api') {
            models = [`api:${modelPath}`];
        } else if (modelType === 'hf') {
            models = [`hf:${modelPath}`];
        } else {
            models = [modelPath];
        }

        const selectedDatasets = Array.from(document.getElementById('datasetSelect').selectedOptions)
            .map(option => option.value);

        const subTasksValue = document.getElementById('subTasks').value.trim();
        const subTasks = subTasksValue ? subTasksValue.split(',').map(s => s.trim()) : null;

        const config = {
            models: models,
            datasets: selectedDatasets,
            task: document.getElementById('taskPhase').value,
            num_workers: parseInt(document.getElementById('numWorkers').value),
            tensor_parallel: parseInt(document.getElementById('tensorParallel').value),
            batch_size: parseInt(document.getElementById('batchSize').value),
            temperature: parseFloat(document.getElementById('temperature').value),
            top_p: parseFloat(document.getElementById('topP').value),
            max_tokens: parseInt(document.getElementById('maxTokens').value),
            repetition_penalty: parseFloat(document.getElementById('repetitionPenalty').value)
        };

        const accelerator = document.getElementById('accelerator').value;
        if (accelerator) {
            config.accelerator = accelerator;
        }

        if (subTasks) {
            config.sub_tasks = subTasks;
        }

        // JSON eval section
        const jsonEnabled = document.getElementById('jsonEvalSwitch').checked;
        if (jsonEnabled) {
            config.json_eval = true;
            const rawPaths = document.getElementById('jsonPaths').value.trim();
            if (rawPaths) {
                // split by newline, filter empty
                const lines = rawPaths.split(/\n+/).map(l => l.trim()).filter(Boolean);
                config.json_paths = lines;
            }
            const jl = document.getElementById('jsonModelLabel').value.trim();
            if (jl) config.json_model_label = jl;
            // Enforce eval phase for backend consistency
            config.task = 'eval';
        }

        // Judge section
        const judgeEnabled = document.getElementById('judgeSwitch').checked;
        if (judgeEnabled) {
            const spec = document.getElementById('judgeModelSpec').value.trim();
            if (spec) config.judge = spec;
            const jb = parseInt(document.getElementById('judgeBatchSize').value);
            if (!isNaN(jb)) config.judge_batch_size = jb;
            const jtp = parseInt(document.getElementById('judgeTensorParallel').value);
            if (!isNaN(jtp)) config.judge_tensor_parallel = jtp;
            const jt = parseFloat(document.getElementById('judgeTemperature').value);
            if (!isNaN(jt)) config.judge_temperature = jt;
            const jtop = parseFloat(document.getElementById('judgeTopP').value);
            if (!isNaN(jtop)) config.judge_top_p = jtop;
            const jmax = parseInt(document.getElementById('judgeMaxTokens').value);
            if (!isNaN(jmax)) config.judge_max_tokens = jmax;
            const jrep = parseFloat(document.getElementById('judgeRepPenalty').value);
            if (!isNaN(jrep)) config.judge_repetition_penalty = jrep;
            const jacc = document.getElementById('judgeAccelerator').value;
            if (jacc) config.judge_accelerator = jacc;
            const japi = document.getElementById('judgeApiUrl').value.trim();
            if (japi) config.judge_api_url = japi;
            const japikey = document.getElementById('judgeApiKey').value.trim();
            if (japikey) config.judge_api_key = japikey;
        }

        if (modelType === 'api') {
            config.api_url = document.getElementById('apiUrl').value;
            config.api_key = document.getElementById('apiKey').value;
        }

        return config;
    }

    validateConfig(config) {
        if (!config.models || config.models.length === 0) {
            throw new Error('请指定至少一个模型');
        }

        if (!config.datasets || config.datasets.length === 0) {
            throw new Error('请选择至少一个数据集');
        }

        if (config.models.some(m => m.startsWith('api:')) && (!config.api_url || !config.api_key)) {
            throw new Error('API模型需要提供API URL和API Key');
        }

        if (config.json_eval) {
            if (!config.json_paths || config.json_paths.length === 0) {
                throw new Error('JSON 评测模式需要提供 json_paths');
            }
            if (config.task !== 'eval') {
                throw new Error('JSON 评测模式下任务类型必须为 eval');
            }
        }
        if (config.judge && !config.judge_batch_size) {
            // Provide default for safety
            config.judge_batch_size = 4;
        }
    }

    async showTaskDetail(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/tasks/${taskId}`);
            const task = await response.json();

            if (!response.ok) {
                throw new Error(task.error || '获取任务详情失败');
            }

            this.displayTaskDetail(task);
            new bootstrap.Modal(document.getElementById('taskDetailModal')).show();
        } catch (error) {
            this.showError('获取任务详情失败: ' + error.message);
        }
    }

    displayTaskDetail(task) {
        const content = `
            <div class="row">
                <div class="col-md-6">
                    <h6>基本信息</h6>
                    <table class="table table-sm">
                        <tr><td>任务ID</td><td><code>${task.id}</code></td></tr>
                        <tr><td>状态</td><td><span class="status-badge status-${task.status}">${this.getStatusText(task.status)}</span></td></tr>
                        <tr><td>创建时间</td><td>${this.formatDate(task.created_at)}</td></tr>
                        <tr><td>开始时间</td><td>${task.started_at ? this.formatDate(task.started_at) : '-'}</td></tr>
                        <tr><td>完成时间</td><td>${task.completed_at ? this.formatDate(task.completed_at) : '-'}</td></tr>
                        <tr><td>进度</td><td>${task.progress || 0}%</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>配置参数</h6>
                    <pre class="bg-light p-2 rounded"><code>${JSON.stringify(task.config, null, 2)}</code></pre>
                </div>
            </div>
            ${task.error ? `
                <div class="mt-3">
                    <h6>错误信息</h6>
                    <div class="alert alert-danger">${task.error}</div>
                </div>
            ` : ''}
        `;

        document.getElementById('taskDetailContent').innerHTML = content;
    }

    async showTaskResults(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/tasks/${taskId}/results`);
            const results = await response.json();

            if (!response.ok) {
                throw new Error(results.error || '获取结果失败');
            }

            this.displayTaskResults(results);
            new bootstrap.Modal(document.getElementById('resultsModal')).show();
        } catch (error) {
            this.showError('获取结果失败: ' + error.message);
        }
    }

    displayTaskResults(results) {
        let content = '<div class="row">';
        const groupMetricKeys = (resultObj) => {
            const groups = { primary: {}, judge: {}, classic: {}, other: {} };
            for (const [k, v] of Object.entries(resultObj)) {
                if (k === 'score') { groups.primary[k] = v; continue; }
                if (k.startsWith('judge_')) { groups.judge[k] = v; continue; }
                if (k.startsWith('classic_')) { groups.classic[k] = v; continue; }
                groups.other[k] = v;
            }
            return groups;
        };

        const renderGroup = (title, data) => {
            if (!data || Object.keys(data).length === 0) return '';
            return `
                <div class="mb-2">
                    <strong>${title}</strong><br>
                    <small class="text-muted">${Object.entries(data).map(([k,v]) => `${k}: ${typeof v === 'number' ? v.toFixed(3) : v}`).join(', ')}</small>
                </div>
            `;
        };

        Object.entries(results).forEach(([modelId, modelResults]) => {
            content += `<div class="col-12 mb-4"><h5>${modelId}</h5>`;
            Object.entries(modelResults).forEach(([taskId, result]) => {
                const groups = groupMetricKeys(result);
                const primaryScore = result.score;
                content += `
                    <div class="card mb-3">
                      <div class="card-header d-flex justify-content-between align-items-center">
                        <span><code>${taskId}</code></span>
                        <span class="score-badge ${this.getScoreClass(primaryScore)}">${primaryScore !== undefined ? primaryScore.toFixed(3) : 'N/A'}</span>
                      </div>
                      <div class="card-body">
                        ${renderGroup('主指标', groups.primary)}
                        ${renderGroup('LLM Judge 指标', groups.judge)}
                        ${renderGroup('经典指标 (BLEU/Rouge/BERTScore)', groups.classic)}
                        ${renderGroup('其它', groups.other)}
                      </div>
                    </div>`;
            });
            content += '</div>';
        });
        content += '</div>';
        document.getElementById('resultsContent').innerHTML = content;
    }

    getScoreClass(score) {
        if (score >= 0.8) return 'score-high';
        if (score >= 0.6) return 'score-medium';
        return 'score-low';
    }

    startAutoRefresh() {
        this.refreshInterval = setInterval(() => {
            this.loadRecentTasks();
            // Refresh tasks list if currently viewing it
            if (document.querySelector('[href="#results"]').classList.contains('active')) {
                this.loadTasks();
            }
        }, 5000); // Refresh every 5 seconds
    }

    getStatusText(status) {
        const statusMap = {
            'pending': '等待中',
            'running': '运行中',
            'completed': '已完成',
            'failed': '失败'
        };
        return statusMap[status] || status;
    }

    formatDate(dateString) {
        return new Date(dateString).toLocaleString('zh-CN');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'danger');
    }

    showNotification(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LegalKitApp();
});