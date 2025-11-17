// LegalKit Web Interface JavaScript

class LegalKitApp {
    constructor() {
        this.baseUrl = '/api';
        this.tasks = new Map();
        this.refreshInterval = null;
        this.lang = (localStorage.getItem('ui_lang') || 'en');
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.applyStaticTranslations();
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

        // Retrieval toggle and method-dependent UI
        const retrievalSwitch = document.getElementById('retrievalSwitch');
        const retrievalConfig = document.getElementById('retrievalConfig');
        const retrievalMethod = document.getElementById('retrievalMethod');
        const embedApiGroup = document.getElementById('embedApiGroup');
        const embedBatchGroup = document.getElementById('embedBatchGroup');

        const updateRetrievalMethodUI = () => {
            const method = retrievalMethod.value;
            const isDense = method && method.startsWith('dense');
            const isApi = method === 'dense-api';
            embedApiGroup.style.display = isApi ? 'flex' : 'none';
            embedBatchGroup.style.display = isDense ? 'flex' : 'none';
        };

        retrievalSwitch.addEventListener('change', (e) => {
            retrievalConfig.style.display = e.target.checked ? 'block' : 'none';
        });
        retrievalMethod.addEventListener('change', updateRetrievalMethodUI);
        // Initialize retrieval UI state
        updateRetrievalMethodUI();

        // Judge toggle
        const judgeSwitch = document.getElementById('judgeSwitch');
        judgeSwitch.addEventListener('change', (e) => {
            document.getElementById('judgeConfig').style.display = e.target.checked ? 'block' : 'none';
        });

        // Refresh tasks
        document.getElementById('refreshTasks').addEventListener('click', () => {
            this.loadTasks();
        });

        // Language toggle button
        const langBtn = document.getElementById('langToggle');
        if (langBtn) {
            langBtn.addEventListener('click', () => {
                const next = this.lang === 'en' ? 'zh' : 'en';
                this.setLanguage(next);
            });
        }

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

    setLanguage(lang) {
        this.lang = lang;
        localStorage.setItem('ui_lang', lang);
        this.applyStaticTranslations();
        // Reload dynamic panels to reflect i18n texts
        this.loadSystemInfo();
        this.loadRecentTasks();
        if (document.querySelector('[href="#results"]').classList.contains('active')) {
            this.loadTasks();
        }
    }

    t(key) {
        return (I18N[this.lang] && I18N[this.lang][key]) || key;
    }

    applyStaticTranslations() {
        // Update document title and html lang
        document.documentElement.setAttribute('lang', this.lang === 'zh' ? 'zh' : 'en');
        document.title = this.t('title');
        // Update any element with data-i18n
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (key) {
                el.textContent = this.t(key);
            }
        });
        // Update placeholders
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            if (key) {
                el.setAttribute('placeholder', this.t(key));
            }
        });
        // Update values (for disabled/display-only inputs)
        document.querySelectorAll('[data-i18n-value]').forEach(el => {
            const key = el.getAttribute('data-i18n-value');
            if (key) {
                el.value = this.t(key);
            }
        });
        // Update language button label: show the target language label for clarity
        const langBtn = document.getElementById('langToggle');
        if (langBtn) {
            langBtn.innerHTML = this.lang === 'en'
                ? '<i class="bi bi-translate"></i> 中文'
                : '<i class="bi bi-translate"></i> EN';
        }
    }

    async loadInitialData() {
        try {
            await Promise.all([
                this.loadDatasets(),
                this.loadSystemInfo(),
                this.loadRecentTasks()
            ]);
        } catch (error) {
            this.showError(this.t('err_init') + error.message);
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
            this.showError(this.t('err_load_datasets'));
        }
    }

    async loadSystemInfo() {
        try {
            const response = await fetch(`${this.baseUrl}/system_info`);
            const info = await response.json();
            
            this.displaySystemInfo(info);
            this.displayGpuInfo(info.gpu_info);
            this.displaySupportedDatasets(info.datasets);
            this.displaySupportedBackends(info.accelerators || []);
        } catch (error) {
            console.error('Error loading system info:', error);
            document.getElementById('systemInfo').innerHTML = 
                `<div class="alert alert-danger">${this.t('err_load_system')}</div>`;
        }
    }

    displaySystemInfo(info) {
        const systemInfoDiv = document.getElementById('systemInfo');
        const datasetsCount = (info.datasets && info.datasets.length) ? info.datasets.length : 18;
        const subtasksCount = info.subtasks_total || info.subtasks_count || 312;
        const acceleratorsCount = (info.accelerators && info.accelerators.length) ? info.accelerators.length : 3;

        systemInfoDiv.innerHTML = `
            <div class="system-metric">
                <span class="metric-value">${info.gpu_count}</span>
                <span class="metric-label">${this.t('metric_gpu_available')}</span>
            </div>
            <div class="row">
                <div class="col-4">
                    <div class="text-center">
                        <strong>${datasetsCount}</strong><br>
                        <small class="text-muted">${this.t('metric_datasets')}</small>
                    </div>
                </div>
                <div class="col-4">
                    <div class="text-center">
                        <strong>${subtasksCount}</strong><br>
                        <small class="text-muted">${this.t('metric_subtasks')}</small>
                    </div>
                </div>
                <div class="col-4">
                    <div class="text-center">
                        <strong>${acceleratorsCount}</strong><br>
                        <small class="text-muted">${this.t('metric_accelerators')}</small>
                    </div>
                </div>
            </div>
        `;
    }

    displayGpuInfo(gpuInfo) {
        const gpuInfoDiv = document.getElementById('gpuInfo');
        if (!gpuInfo || gpuInfo.length === 0) {
            gpuInfoDiv.innerHTML = `<div class="alert alert-warning">${this.t('no_gpu')}</div>`;
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

    displaySupportedBackends(backends) {
        const div = document.getElementById('supportedBackends');
        if (!div) return;
        if (!backends || backends.length === 0) {
            div.innerHTML = `<div class="alert alert-warning">${this.t('no_backends') || 'No accelerators configured'}</div>`;
            return;
        }
        const items = backends.map(b => `
            <div class="dataset-item">
                <i class="bi bi-cpu"></i>
                ${b}
            </div>
        `).join('');
        div.innerHTML = items;
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
            recentTasksDiv.innerHTML = `<p class="text-muted">${this.t('no_tasks')}</p>`;
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
            this.showError(this.t('err_load_tasks'));
        }
    }

    displayTasksList(tasks) {
        const tasksListDiv = document.getElementById('tasksList');
        
        if (tasks.length === 0) {
            tasksListDiv.innerHTML = `<div class="alert alert-info">${this.t('no_eval_tasks')}</div>`;
            return;
        }

        const tasksTable = `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>${this.t('detail_task_id')}</th>
                            <th>${this.t('detail_status')}</th>
                            <th>${this.t('metric_datasets')}</th>
                            <th>${this.t('model')}</th>
                            <th>${this.t('detail_created_at')}</th>
                            <th>${this.t('detail_progress')}</th>
                            <th>${this.t('action')}</th>
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
            this.showError(this.t('err_input_model_path'));
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
                this.showError(result.error || this.t('err_model_discovery_failed'));
            }
        } catch (error) {
            this.showError(this.t('err_model_discovery_failed') + ': ' + error.message);
        }
    }

    displayDiscoveredModels(models) {
        const modelListDiv = document.getElementById('modelList');
        
        if (models.length === 0) {
            modelListDiv.innerHTML = `<div class="alert alert-warning">${this.t('no_valid_models')}</div>`;
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
                <strong>${this.t('discovered_models')}</strong>
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
                this.showSuccess(`${this.t('submit_success_prefix')}${result.task_id}`);
                this.loadRecentTasks();
                // Switch to results tab
                document.querySelector('[href="#results"]').click();
            } else {
                this.showError(result.error || this.t('err_task_submit_failed'));
            }
        } catch (error) {
            this.showError(this.t('err_task_submit_failed') + ': ' + error.message);
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

        // Retrieval section
        const retrievalEnabled = document.getElementById('retrievalSwitch').checked;
        if (retrievalEnabled) {
            const rMethod = document.getElementById('retrievalMethod').value;
            const rK = parseInt(document.getElementById('retrievalK').value);
            const rFaiss = document.getElementById('retrievalFaissType').value;
            if (rMethod) config.retrieval_method = rMethod;
            if (!isNaN(rK)) config.retrieval_k = rK;
            if (rMethod && rMethod.startsWith('dense')) {
                if (rFaiss) config.retrieval_faiss_type = rFaiss;
                const ebs = parseInt(document.getElementById('embedBatchSize').value);
                if (!isNaN(ebs)) config.embed_batch_size = ebs;
                if (rMethod === 'dense-api') {
                    const emn = document.getElementById('embedModelName').value.trim();
                    const eurl = document.getElementById('embedApiUrl').value.trim();
                    const ekey = document.getElementById('embedApiKey').value.trim();
                    if (emn) config.embed_model_name = emn;
                    if (eurl) config.embed_api_url = eurl;
                    if (ekey) config.embed_api_key = ekey;
                }
            }
        } else {
            // Explicitly disable retrieval if toggle off
            config.retrieval_method = 'none';
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
            throw new Error(this.t('validate_need_model'));
        }

        if (!config.datasets || config.datasets.length === 0) {
            throw new Error(this.t('validate_need_dataset'));
        }

        if (config.models.some(m => m.startsWith('api:')) && (!config.api_url || !config.api_key)) {
            throw new Error(this.t('validate_need_api'));
        }

        if (config.json_eval) {
            if (!config.json_paths || config.json_paths.length === 0) {
                throw new Error(this.t('validate_need_json_paths'));
            }
            if (config.task !== 'eval') {
                throw new Error(this.t('validate_json_task_eval'));
            }
        }
        if (config.judge && !config.judge_batch_size) {
            // Provide default for safety
            config.judge_batch_size = 4;
        }

        // Retrieval validation
        if (config.retrieval_method && config.retrieval_method !== 'none') {
            if (config.retrieval_method === 'dense-api') {
                if (!config.embed_api_url || !config.embed_api_key || !config.embed_model_name) {
                    throw new Error(this.t('validate_need_embed_api'));
                }
            }
        }
    }

    async showTaskDetail(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/tasks/${taskId}`);
            const task = await response.json();

            if (!response.ok) {
                throw new Error(task.error || this.t('err_get_task_detail'));
            }

            let results = task.results;
            let resultsError = null;

            if ((!results || Object.keys(results).length === 0) && task.status === 'completed') {
                try {
                    const resultsResponse = await fetch(`${this.baseUrl}/tasks/${taskId}/results`);
                    const payload = await resultsResponse.json().catch(() => ({}));
                    if (!resultsResponse.ok) {
                        throw new Error(payload.error || resultsResponse.statusText);
                    }
                    results = payload;
                } catch (error) {
                    resultsError = error.message;
                }
            }

            this.displayTaskDetail(task, results, resultsError);
            new bootstrap.Modal(document.getElementById('taskDetailModal')).show();
        } catch (error) {
            this.showError(this.t('err_get_task_detail') + ': ' + error.message);
        }
    }

    displayTaskDetail(task, results = null, resultsError = null) {
        const resultsSection = (() => {
            if (task.status === 'completed') {
                if (resultsError) {
                    return `
                        <div class="mt-3">
                            <h6>${this.t('modal_results')}</h6>
                            <div class="alert alert-warning">${this.t('err_get_results')}: ${resultsError}</div>
                        </div>
                    `;
                }
                if (results && Object.keys(results).length > 0) {
                    return `
                        <div class="mt-3">
                            <h6>${this.t('modal_results')}</h6>
                            ${this.renderResultsContent(results)}
                        </div>
                    `;
                }
                return `
                    <div class="mt-3">
                        <h6>${this.t('modal_results')}</h6>
                        <div class="alert alert-info">${this.t('detail_no_results')}</div>
                    </div>
                `;
            }
            return '';
        })();

        const content = `
            <div class="row">
                <div class="col-md-6">
                    <h6>${this.t('detail_basic')}</h6>
                    <table class="table table-sm">
                        <tr><td>${this.t('detail_task_id')}</td><td><code>${task.id}</code></td></tr>
                        <tr><td>${this.t('detail_status')}</td><td><span class="status-badge status-${task.status}">${this.getStatusText(task.status)}</span></td></tr>
                        <tr><td>${this.t('detail_created_at')}</td><td>${this.formatDate(task.created_at)}</td></tr>
                        <tr><td>${this.t('detail_started_at')}</td><td>${task.started_at ? this.formatDate(task.started_at) : '-'}</td></tr>
                        <tr><td>${this.t('detail_completed_at')}</td><td>${task.completed_at ? this.formatDate(task.completed_at) : '-'}</td></tr>
                        <tr><td>${this.t('detail_progress')}</td><td>${task.progress || 0}%</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>${this.t('detail_config')}</h6>
                    <pre class="bg-light p-2 rounded"><code>${JSON.stringify(task.config, null, 2)}</code></pre>
                </div>
            </div>
            ${task.error ? `
                <div class="mt-3">
                    <h6>${this.t('detail_error')}</h6>
                    <div class="alert alert-danger">${task.error}</div>
                </div>
            ` : ''}
            ${resultsSection}
        `;

        document.getElementById('taskDetailContent').innerHTML = content;
    }

    renderResultsContent(results) {
        if (!results || Object.keys(results).length === 0) {
            return `<div class="alert alert-info">${this.t('detail_no_results')}</div>`;
        }

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

        let content = '<div class="row">';
        Object.entries(results).forEach(([modelId, modelResults]) => {
            content += `<div class="col-12 mb-4"><h5>${modelId}</h5>`;
            Object.entries(modelResults).forEach(([taskId, result]) => {
                const groups = groupMetricKeys(result);
                const primaryScore = result.score;
                content += `
                    <div class="card mb-3">
                      <div class="card-header d-flex justify-content-between align-items-center">
                        <span><code>${taskId}</code></span>
                        <span class="score-badge ${this.getScoreClass(primaryScore)}">${primaryScore !== undefined ? primaryScore.toFixed(3) : this.t('na')}</span>
                      </div>
                      <div class="card-body">
                        ${renderGroup(this.t('results_primary'), groups.primary)}
                        ${renderGroup(this.t('results_judge'), groups.judge)}
                        ${renderGroup(this.t('results_classic'), groups.classic)}
                        ${renderGroup(this.t('results_other'), groups.other)}
                      </div>
                    </div>`;
            });
            content += '</div>';
        });
        content += '</div>';
        return content;
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
            'pending': this.t('status_pending'),
            'running': this.t('status_running'),
            'completed': this.t('status_completed'),
            'failed': this.t('status_failed')
        };
        return statusMap[status] || status;
    }

    formatDate(dateString) {
        const locale = this.lang === 'zh' ? 'zh-CN' : 'en-US';
        return new Date(dateString).toLocaleString(locale);
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

// i18n resources
const I18N = {
    en: {
        title: 'LegalKit: Legal Model Evaluation Toolkit',
        brand: 'LegalKit: Legal Model Evaluation Toolkit',
        nav_dashboard: 'Evaluation',
        nav_results: 'Results',
        nav_system: 'System',
        tab_eval_config: 'Evaluation Config',
        tab_results_mgmt: 'Results Management',
        tab_system_status: 'System Status',
        settings_title: 'Task Settings',
        system_title: 'System Status',
        recent_title: 'Recent Tasks',
        tasks_title: 'Tasks',
        refresh: 'Refresh',
        gpu_title: 'GPU Info',
        datasets_title: 'Supported Datasets',
    backends_title: 'Supported Backends',
        start_eval: 'Start Evaluation',
    model: 'Model',
    action: 'Action',
    modal_task_detail: 'Task Detail',
    modal_results: 'Evaluation Results',
        // Retrieval
        section_retrieval_title: 'Retrieval Configuration',
        switch_retrieval_enable: 'Enable retrieval stage before generation',
        label_retrieval_method: 'Retrieval method',
        label_retrieval_k: 'Top-k',
        label_faiss_type: 'FAISS index',
        label_embed_model_name: 'Embedding model',
        label_embed_api_url: 'Embedding API URL',
        label_embed_api_key: 'Embedding API Key',
        label_embed_batch_size: 'Embedding batch size',
        hint_retrieval_scope: 'Applies to datasets that implement a retrieval stage (e.g., LexRAG). Artifacts are saved under run_output.',
        // Form sections and labels
        section_model_config_title: 'Model Configuration',
        label_model_type: 'Model Type',
        opt_model_local: 'Local model',
        opt_model_hf: 'HuggingFace',
        opt_model_api: 'API model',
        label_model_path: 'Model path/name',
        ph_model_path: 'Enter model path or name',
        label_dataset_select: 'Dataset Selection',
    label_dataset_list: 'Datasets',
        hint_multi_select: 'Hold Ctrl to multi-select',
        label_subtasks: 'Subtasks (optional)',
        ph_subtasks: 'e.g.: 2-1,2-2',
        hint_subtasks: 'Separate multiple subtasks with commas',
        section_run_config_title: 'Run Configuration',
        label_task_type: 'Task Type',
        opt_task_all: 'Full pipeline',
        opt_task_infer: 'Inference only',
        opt_task_eval: 'Evaluation only',
        label_accelerator: 'Accelerator',
        opt_acc_none: 'None',
        label_num_workers: 'Worker processes',
        label_tensor_parallel: 'Tensor parallelism',
        label_batch_size: 'Batch size',
        section_generation_params_title: 'Generation Parameters',
        label_temperature: 'Temperature',
        label_top_p: 'Top-p',
        label_max_tokens: 'Max tokens',
        label_repetition_penalty: 'Repetition penalty',
        section_json_eval_title: 'JSON offline evaluation (optional)',
        switch_json_eval: 'Enable evaluation of JSON prediction files (skip inference)',
        label_json_paths: 'JSON file paths (use dataset=path per line)',
        ph_json_paths: 'CaseGen=/path/to/casegen_preds.json',
        label_json_model_label: 'Evaluation model label',
        ph_json_model_label: 'json_eval',
        label_override_task_type: 'Override task type',
        override_eval_value: 'Auto-set to eval',
        hint_json_models: 'If models are not specified, a placeholder model json::label will be injected',
        section_judge_title: 'LLM Judge configuration (optional)',
        switch_judge_enable: 'Use LLM as judge',
        label_judge_model_spec: 'Judge model spec',
        ph_judge_model_spec: 'hf:Qwen/Qwen2.5-7B',
        label_judge_batch: 'Batch',
        label_judge_tp: 'Tensor parallelism',
        label_judge_temperature: 'Temperature',
        label_judge_top_p: 'Top-p',
        label_judge_max_tokens: 'MaxTokens',
        label_judge_rep_penalty: 'Repetition penalty',
        label_judge_accelerator: 'Judge accelerator',
        opt_judge_none: 'None',
        label_judge_api_mode: 'API mode (optional)',
        ph_judge_api_url: 'https://api.example.com/v1/chat',
        ph_judge_api_key: 'judge-api-key',
        hint_judge_decoupled: 'The judge model is decoupled from the main model and used only during evaluation.',
        loading: 'Loading...',
        no_tasks: 'No tasks',
        no_eval_tasks: 'No evaluation tasks',
        no_gpu: 'No GPU detected',
        discovered_models: 'Discovered models:',
        no_valid_models: 'No valid models found',
        status_pending: 'Pending',
        status_running: 'Running',
        status_completed: 'Completed',
        status_failed: 'Failed',
        metric_gpu_available: 'GPUs',
        metric_datasets: 'Datasets',
        metric_accelerators: 'Accelerators',
        metric_task_types: 'Task types',
        metric_subtasks: 'Subtasks',
    no_backends: 'No accelerators configured',
        err_init: 'Failed to load initial data: ',
        err_load_datasets: 'Failed to load datasets',
        err_load_system: 'Failed to load system info',
        err_load_tasks: 'Failed to load tasks list',
        err_input_model_path: 'Please input model path',
        err_model_discovery_failed: 'Model discovery failed',
        err_task_submit_failed: 'Task submission failed',
        err_get_task_detail: 'Failed to get task detail',
        err_get_results: 'Failed to get results',
        submit_success_prefix: 'Task submitted! ID: ',
        validate_need_model: 'Please specify at least one model',
        validate_need_dataset: 'Please select at least one dataset',
        validate_need_api: 'API model requires API URL and API Key',
        validate_need_json_paths: 'JSON evaluation needs json_paths',
        validate_json_task_eval: 'Task must be eval in JSON mode',
    validate_need_embed_api: 'Embedding API mode requires API URL, API Key and model name',
        detail_basic: 'Basic Info',
        detail_task_id: 'Task ID',
        detail_status: 'Status',
        detail_created_at: 'Created',
        detail_started_at: 'Started',
        detail_completed_at: 'Completed',
        detail_progress: 'Progress',
        detail_config: 'Config',
        detail_error: 'Error',
    detail_no_results: 'No evaluation results available yet',
        results_primary: 'Primary',
        results_judge: 'LLM Judge Metrics',
        results_classic: 'Classic Metrics (BLEU/Rouge/BERTScore)',
        results_other: 'Other',
        na: 'N/A',
    },
    zh: {
        title: 'LegalKit：法律模型快速评测工具包',
        brand: 'LegalKit：法律模型快速评测工具包',
        nav_dashboard: '评测任务',
        nav_results: '结果查看',
        nav_system: '系统信息',
        tab_eval_config: '评测配置',
        tab_results_mgmt: '结果管理',
        tab_system_status: '系统状态',
        settings_title: '评测任务配置',
        system_title: '系统状态',
        recent_title: '最近任务',
        tasks_title: '任务列表',
        refresh: '刷新',
        gpu_title: 'GPU信息',
        datasets_title: '支持的数据集',
    backends_title: '支持的加速后端',
        start_eval: '开始评测',
    model: '模型',
    action: '操作',
        modal_task_detail: '任务详情',
        modal_results: '评测结果',
        // Retrieval
        section_retrieval_title: '检索配置',
        switch_retrieval_enable: '在生成前启用检索阶段',
        label_retrieval_method: '检索方法',
        label_retrieval_k: 'Top-k',
        label_faiss_type: 'FAISS 索引',
        label_embed_model_name: 'Embedding 模型',
        label_embed_api_url: 'Embedding API 地址',
        label_embed_api_key: 'Embedding API Key',
        label_embed_batch_size: 'Embedding 批大小',
        hint_retrieval_scope: '仅对实现了检索阶段的数据集生效（如 LexRAG），产物保存在 run_output。',
        // Form sections and labels
        section_model_config_title: '模型配置',
        label_model_type: '模型类型',
        opt_model_local: '本地模型',
        opt_model_hf: 'HuggingFace模型',
        opt_model_api: 'API模型',
        label_model_path: '模型路径/名称',
        ph_model_path: '输入模型路径或名称',
        label_dataset_select: '数据集选择',
    label_dataset_list: '数据集',
        hint_multi_select: '按住Ctrl键可多选',
        label_subtasks: '子任务（可选）',
        ph_subtasks: '例如: 2-1,2-2',
        hint_subtasks: '用逗号分隔多个子任务',
        section_run_config_title: '运行配置',
        label_task_type: '任务类型',
        opt_task_all: '完整流程',
        opt_task_infer: '仅推理',
        opt_task_eval: '仅评估',
        label_accelerator: '加速器',
        opt_acc_none: '无',
        label_num_workers: '并行工作进程',
        label_tensor_parallel: '张量并行度',
        label_batch_size: '批次大小',
        section_generation_params_title: '生成参数',
        label_temperature: '温度',
        label_top_p: 'Top-p',
        label_max_tokens: '最大Token数',
        label_repetition_penalty: '重复惩罚',
        section_json_eval_title: 'JSON 离线评测 (可选)',
        switch_json_eval: '启用 JSON 预测文件直接评测（跳过推理）',
        label_json_paths: 'JSON 文件路径（多数据集用 dataset=path，每行一个）',
        ph_json_paths: 'CaseGen=/path/to/casegen_preds.json',
        label_json_model_label: '评测模型标签',
        ph_json_model_label: 'json_eval',
        label_override_task_type: '覆盖任务类型',
        override_eval_value: '自动设为 eval',
        hint_json_models: '若未指定 models，将自动注入占位模型 json::label',
        section_judge_title: 'LLM 评审 (Judge) 配置 (可选)',
        switch_judge_enable: '启用 LLM 作为评审',
        label_judge_model_spec: 'Judge 模型 Spec',
        ph_judge_model_spec: 'hf:Qwen/Qwen2.5-7B',
        label_judge_batch: 'Batch',
        label_judge_tp: '并行度',
        label_judge_temperature: '温度',
        label_judge_top_p: 'Top-p',
        label_judge_max_tokens: 'MaxTokens',
        label_judge_rep_penalty: '重复惩罚',
        label_judge_accelerator: 'Judge 加速器',
        opt_judge_none: '无',
        label_judge_api_mode: 'API 模式 (可选)',
        ph_judge_api_url: 'https://api.example.com/v1/chat',
        ph_judge_api_key: 'judge-api-key',
        hint_judge_decoupled: 'Judge 模型与主模型解耦，仅在评测阶段调用。',
    modal_task_detail: '任务详情',
    modal_results: '评测结果',
        loading: '加载中...',
        no_tasks: '暂无任务',
        no_eval_tasks: '暂无评测任务',
        no_gpu: '未检测到GPU',
        discovered_models: '发现的模型:',
        no_valid_models: '未找到有效模型',
        status_pending: '等待中',
        status_running: '运行中',
        status_completed: '已完成',
        status_failed: '失败',
        metric_gpu_available: '可用GPU',
        metric_datasets: '数据集',
        metric_accelerators: '加速器',
        metric_task_types: '任务类型',
        metric_subtasks: '子任务',
    no_backends: '未配置加速后端',
        err_init: '加载初始数据失败: ',
        err_load_datasets: '加载数据集失败',
        err_load_system: '加载系统信息失败',
        err_load_tasks: '加载任务列表失败',
        err_input_model_path: '请输入模型路径',
        err_model_discovery_failed: '模型发现失败',
        err_task_submit_failed: '任务提交失败',
        err_get_task_detail: '获取任务详情失败',
        err_get_results: '获取结果失败',
        submit_success_prefix: '任务提交成功! 任务ID: ',
        validate_need_model: '请指定至少一个模型',
        validate_need_dataset: '请选择至少一个数据集',
        validate_need_api: 'API模型需要提供API URL和API Key',
        validate_need_json_paths: 'JSON 评测模式需要提供 json_paths',
        validate_json_task_eval: 'JSON 评测模式下任务类型必须为 eval',
    validate_need_embed_api: 'Embedding API 模式需要提供 API URL、API Key 和模型名',
        detail_basic: '基本信息',
        detail_task_id: '任务ID',
        detail_status: '状态',
        detail_created_at: '创建时间',
        detail_started_at: '开始时间',
        detail_completed_at: '完成时间',
        detail_progress: '进度',
        detail_config: '配置参数',
        detail_error: '错误信息',
    detail_no_results: '暂无评测结果',
        results_primary: '主指标',
        results_judge: 'LLM Judge 指标',
        results_classic: '经典指标 (BLEU/Rouge/BERTScore)',
        results_other: '其它',
        na: 'N/A',
    }
};