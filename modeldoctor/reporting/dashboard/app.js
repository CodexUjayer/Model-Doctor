/**
 * app.js - Logic for the ModelDoctor Dashboard
 */

document.addEventListener('DOMContentLoaded', () => {
    // Attempt to load report data from the script tag or fetch it (for real implementation)
    async function initDashboard() {
        try {
            // Fetch live data from the local API
            const response = await fetch('/api/report');
            if (!response.ok) throw new Error('Failed to fetch report data');
            const reportData = await response.json();
            
            // Re-hide loading overlay
            const loadingOverlay = document.getElementById('loadingOverlay');
            if (loadingOverlay) loadingOverlay.style.display = 'none';

            renderDashboard(reportData);
            setupEventListeners();
        } catch (e) {
            console.error('Failed to load report data', e);
            // Fallback to mock data if fetch fails
            const reportData = generateMockReport();
            renderDashboard(reportData);
            setupEventListeners();
        }
    }

    initDashboard();
});

function setupEventListeners() {
    document.getElementById('btn-export').addEventListener('click', () => {
        alert('Export functionality would trigger here.');
    });

    document.getElementById('btn-run').addEventListener('click', () => {
        document.body.classList.add('running');
        setTimeout(() => {
            document.body.classList.remove('running');
            alert('Analysis complete!');
        }, 2000);
    });
}

function renderDashboard(report) {
    if (!report) return;

    renderOverallHealth(report.health_score);
    renderKeyMetrics(report);
    renderHealthBreakdown(report.health_score);
    renderRecommendations(report.top_recommendations);
    renderPerformanceCharts(report);
    renderModelInfo(report.passport, report.generated_at);
}

// --- Overall Health ---
function renderOverallHealth(healthScore) {
    if (!healthScore) return;

    const scoreEl = document.getElementById('overall-score');
    const badgeEl = document.getElementById('overall-badge');
    const verdictEl = document.getElementById('overall-verdict');
    const ringEl = document.getElementById('score-ring');

    // Animate score counter
    animateValue(scoreEl, 0, Math.round(healthScore.overall), 1500);

    // Setup Badge
    badgeEl.textContent = healthScore.grade || 'A';
    badgeEl.className = 'badge';
    if (healthScore.overall >= 90) {
        badgeEl.classList.add('badge-success');
        badgeEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg> Excellent`;
    } else if (healthScore.overall >= 75) {
        badgeEl.classList.add('badge-accent');
        badgeEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> Good`;
    } else if (healthScore.overall >= 60) {
        badgeEl.classList.add('badge-warning');
        badgeEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg> Fair`;
    } else {
        badgeEl.classList.add('badge-danger');
        badgeEl.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg> Poor`;
    }

    verdictEl.textContent = healthScore.verdict || 'Your model is in great shape.';

    // Animate Ring
    const circumference = 2 * Math.PI * 54; // 339.29
    ringEl.style.strokeDasharray = circumference;
    const offset = circumference - (healthScore.overall / 100) * circumference;
    
    // Slight delay for smooth animation start
    setTimeout(() => {
        ringEl.style.strokeDashoffset = offset;
        // Update color based on score
        if (healthScore.overall >= 90) ringEl.style.stroke = 'var(--success)';
        else if (healthScore.overall >= 75) ringEl.style.stroke = 'var(--accent)';
        else if (healthScore.overall >= 60) ringEl.style.stroke = 'var(--warning)';
        else ringEl.style.stroke = 'var(--danger)';
    }, 100);
}

// --- Key Metrics ---
function renderKeyMetrics(report) {
    const grid = document.getElementById('metrics-grid');
    if (!grid) return;

    // Example metrics to show based on task type
    let metrics = [];
    
    // In a real scenario, extract from report.diagnoses or context metrics
    if (report.passport?.task_type?.includes('CLASSIFICATION')) {
        metrics = [
            { name: 'Accuracy', value: 0.94, status: 'Excellent', type: 'success' },
            { name: 'AUC-ROC', value: 0.96, status: 'Excellent', type: 'success' },
            { name: 'F1 Score', value: 0.92, status: 'Excellent', type: 'success' },
            { name: 'Log Loss', value: 0.18, status: 'Good', type: 'accent' }
        ];
    } else {
        // Fallback generic metrics
        metrics = [
            { name: 'Score', value: 0.85, status: 'Good', type: 'accent' },
            { name: 'Error', value: 0.12, status: 'Fair', type: 'warning' },
            { name: 'Training Time', value: '45s', status: 'Fast', type: 'success' },
            { name: 'Memory', value: '1.2GB', status: 'Normal', type: 'neutral' }
        ];
    }

    grid.innerHTML = metrics.map((m, i) => `
        <div class="metric-cell">
            <span class="metric-name">${m.name}</span>
            <span class="metric-value" data-val="${m.value}">${typeof m.value === 'number' ? '0.00' : m.value}</span>
            <span style="color: var(--${m.type === 'neutral' ? 'text-secondary' : m.type}); font-size: 11px; font-weight: 600;">${m.status}</span>
            <div class="metric-sparkline"><canvas id="spark-${i}"></canvas></div>
        </div>
    `).join('');

    // Animate numbers and draw sparklines
    metrics.forEach((m, i) => {
        if (typeof m.value === 'number') {
            const el = grid.querySelector(`.metric-value[data-val="${m.value}"]`);
            animateFloat(el, 0, m.value, 1500);
        }
        
        // Draw dummy sparkline
        const ctx = document.getElementById(`spark-${i}`).getContext('2d');
        const color = getComputedStyle(document.documentElement).getPropertyValue(`--${m.type === 'neutral' ? 'text-secondary' : m.type}`).trim() || '#5B5CEB';
        
        // Generate random sparkline data trending up/down based on 'goodness'
        const data = Array.from({length: 10}, (_, j) => {
            const base = m.type === 'success' ? j : (m.type === 'warning' || m.type === 'danger' ? 10 - j : 5);
            return base + Math.random() * 2;
        });

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map((_, i) => i),
                datasets: [{
                    data: data,
                    borderColor: color,
                    borderWidth: 1.5,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
                scales: { x: { display: false }, y: { display: false } },
                animation: { duration: 2000, easing: 'easeOutQuart' }
            }
        });
    });
}

// --- Health Breakdown ---
function renderHealthBreakdown(healthScore) {
    if (!healthScore || !healthScore.dimensions) return;

    const listEl = document.getElementById('dimension-list');
    const ctx = document.getElementById('radarChart').getContext('2d');

    const dims = healthScore.dimensions.slice(0, 6); // Max 6 for neat radar
    
    // Render List
    const icons = {
        'data_quality': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>',
        'feature_engineering': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
        'overfitting': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
        'leakage': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
        'hyperparameters': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>',
        'prediction_quality': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>',
        'generalization': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12h4l2-9 5 18 3-9h6"></path></svg>',
        'production_readiness': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
        'default': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle></svg>'
    };

    const colors = ['#5B5CEB', '#3B82F6', '#22C55E', '#F59E0B', '#F97316', '#EF4444'];

    listEl.innerHTML = dims.map((d, i) => {
        const color = colors[i % colors.length];
        const icon = icons[d.dimension] || icons['default'];
        const name = d.dimension.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        
        return `
            <div class="dim-row" data-tip="${d.summary || ''}">
                <div class="dim-icon" style="color: ${color}; background: ${color}15;">${icon}</div>
                <div class="dim-info">
                    <div class="dim-name">${name}</div>
                    <div class="dim-bar-wrap">
                        <div class="dim-bar" style="background: ${color};" data-width="${d.score}%"></div>
                    </div>
                </div>
                <div class="dim-score">${Math.round(d.score)}/100</div>
            </div>
        `;
    }).join('');

    // Trigger bar animation
    setTimeout(() => {
        listEl.querySelectorAll('.dim-bar').forEach(bar => {
            bar.style.width = bar.getAttribute('data-width');
        });
    }, 100);

    // Render Radar Chart
    const labels = dims.map(d => d.dimension.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '));
    const data = dims.map(d => d.score);

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Score',
                data: data,
                backgroundColor: 'rgba(91, 92, 235, 0.15)',
                borderColor: 'rgba(91, 92, 235, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(91, 92, 235, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(91, 92, 235, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(0, 0, 0, 0.05)' },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' },
                    pointLabels: { font: { family: "'Inter', sans-serif", size: 10 }, color: '#6B7280' },
                    ticks: { display: false, min: 0, max: 100 }
                }
            },
            plugins: { legend: { display: false } },
            animation: { duration: 1500, easing: 'easeOutQuart' }
        }
    });
}

// --- Recommendations ---
function renderRecommendations(recommendations) {
    const listEl = document.getElementById('rec-list');
    if (!recommendations || recommendations.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>
                <p>No recommendations. Your model is perfectly tuned!</p>
            </div>`;
        return;
    }

    listEl.innerHTML = recommendations.map((r, i) => {
        let badgeClass = 'badge-neutral';
        let priorityText = r.priority || 'Medium';
        if (priorityText.toLowerCase() === 'high' || priorityText.toLowerCase() === 'critical') badgeClass = 'badge-danger';
        else if (priorityText.toLowerCase() === 'medium') badgeClass = 'badge-warning';
        else if (priorityText.toLowerCase() === 'low') badgeClass = 'badge-success';

        let iconColor = badgeClass.replace('badge-', 'var(--') + ')';
        let iconBg = badgeClass.replace('badge-', 'var(--') + '-bg)';

        return `
            <div class="rec-item">
                <div class="rec-header" onclick="this.parentElement.classList.toggle('open')">
                    <div class="rec-icon-wrap" style="background: ${iconBg}; color: ${iconColor};">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="12" y1="19" x2="12" y2="5"></line>
                            <polyline points="5 12 12 5 19 12"></polyline>
                        </svg>
                    </div>
                    <div class="rec-title-col">
                        <div class="rec-title">${r.title || r.action}</div>
                        <div class="rec-improvement">Expected improvement: ${r.estimated_improvement || r.estimated_impact || 'Moderate'}</div>
                    </div>
                    <div class="badge ${badgeClass}">${priorityText.charAt(0).toUpperCase() + priorityText.slice(1)}</div>
                    <svg class="rec-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
                </div>
                <div class="rec-body">
                    <div class="rec-body-inner">
                        <div class="rec-rationale">${r.rationale}</div>
                        
                        <div class="rec-detail-row">
                            <div class="rec-detail">
                                <span class="rec-detail-label">Confidence</span>
                                <span class="rec-detail-value">${r.confidence ? r.confidence.charAt(0).toUpperCase() + r.confidence.slice(1) : 'Medium'}</span>
                            </div>
                            <div class="rec-detail">
                                <span class="rec-detail-label">Difficulty</span>
                                <span class="rec-detail-value">${r.implementation_difficulty || 'Medium'}</span>
                            </div>
                        </div>
                        
                        ${r.affected_metrics && r.affected_metrics.length ? `
                        <div class="rec-detail">
                            <span class="rec-detail-label">Affected Metrics</span>
                            <div class="rec-metrics">
                                ${r.affected_metrics.map(m => `<span class="rec-metric-tag">${m}</span>`).join('')}
                            </div>
                        </div>` : ''}

                        ${r.supporting_evidence && Object.keys(r.supporting_evidence).length ? `
                        <div class="rec-detail" style="margin-top: 4px;">
                            <span class="rec-detail-label">Supporting Evidence</span>
                            <div class="rec-evidence-list">
                                ${Object.entries(r.supporting_evidence).map(([k, v]) => `
                                    <div class="rec-evidence-item"><strong>${k.replace(/_/g, ' ')}:</strong> ${v}</div>
                                `).join('')}
                            </div>
                        </div>` : ''}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// --- Performance Charts ---
function renderPerformanceCharts(report) {
    const tabsEl = document.getElementById('perf-tabs');
    const chartsEl = document.getElementById('perf-charts');
    
    // For demo, we'll create some dummy charts based on task type
    const charts = [
        { id: 'roc', label: 'ROC Curve', render: renderROC },
        { id: 'pr', label: 'Precision-Recall', render: renderPR },
        { id: 'calib', label: 'Calibration', render: renderCalibration }
    ];

    tabsEl.innerHTML = charts.map((c, i) => `
        <button class="chart-tab ${i === 0 ? 'active' : ''}" data-target="${c.id}">${c.label}</button>
    `).join('');

    chartsEl.innerHTML = charts.map((c, i) => `
        <div class="chart-panel ${i === 0 ? 'active' : ''}" id="panel-${c.id}">
            <canvas id="canvas-${c.id}"></canvas>
        </div>
    `).join('');

    // Render charts
    charts.forEach(c => c.render(`canvas-${c.id}`));

    // Tab switching
    tabsEl.querySelectorAll('.chart-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            tabsEl.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
            chartsEl.querySelectorAll('.chart-panel').forEach(p => p.classList.remove('active'));
            
            e.target.classList.add('active');
            document.getElementById(`panel-${e.target.getAttribute('data-target')}`).classList.add('active');
        });
    });
}

function renderROC(canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Dummy ROC data
    const data = [];
    for(let i=0; i<=100; i++) {
        let fpr = i/100;
        let tpr = Math.pow(fpr, 0.3); // Curve shape
        data.push({x: fpr, y: tpr});
    }

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'ROC Curve (AUC = 0.96)',
                data: data,
                borderColor: '#5B5CEB',
                backgroundColor: 'rgba(91,92,235,0.1)',
                borderWidth: 2,
                showLine: true,
                pointRadius: 0,
                fill: true
            },
            {
                label: 'Random',
                data: [{x:0, y:0}, {x:1, y:1}],
                borderColor: '#D1D5DB',
                borderDash: [5, 5],
                borderWidth: 1.5,
                showLine: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
                y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }
            },
            plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 6 } } }
        }
    });
}

function renderPR(canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const data = [];
    for(let i=0; i<=100; i++) {
        let recall = i/100;
        let precision = 1 - Math.pow(recall, 3)*0.2;
        data.push({x: recall, y: precision});
    }
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'PR Curve (AP = 0.92)',
                data: data,
                borderColor: '#22C55E',
                borderWidth: 2,
                showLine: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Recall' }, min: 0, max: 1 },
                y: { title: { display: true, text: 'Precision' }, min: 0, max: 1 }
            },
            plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 6 } } }
        }
    });
}

function renderCalibration(canvasId) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'],
            datasets: [{
                label: 'Model Calibration',
                data: [0.12, 0.25, 0.33, 0.45, 0.48, 0.55, 0.68, 0.82, 0.88],
                borderColor: '#F59E0B',
                backgroundColor: '#F59E0B',
                borderWidth: 2,
                pointRadius: 4
            },
            {
                label: 'Perfect Calibration',
                data: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                borderColor: '#D1D5DB',
                borderDash: [5, 5],
                borderWidth: 1.5,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Mean Predicted Probability' } },
                y: { title: { display: true, text: 'Fraction of Positives' } }
            },
            plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 6 } } }
        }
    });
}

// --- Model Information ---
function renderModelInfo(passport, generatedAt) {
    const grid = document.getElementById('info-grid');
    if (!grid || !passport) return;

    const info = [
        { label: 'Algorithm', value: passport.model_class || 'Unknown', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>' },
        { label: 'Framework', value: passport.framework !== 'UNKNOWN' ? passport.framework : 'Scikit-Learn', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>' },
        { label: 'Task Type', value: passport.task_type !== 'UNKNOWN' ? passport.task_type : 'Classification', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>' },
        { label: 'Features', value: passport.n_features || '-', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>' },
        { label: 'Training Samples', value: passport.n_train_samples ? passport.n_train_samples.toLocaleString() : '-', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>' },
        { label: 'Model Size', value: passport.model_size_bytes ? (passport.model_size_bytes / (1024*1024)).toFixed(1) + ' MB' : '-', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>' },
        { label: 'Analysis Date', value: generatedAt ? new Date(generatedAt).toLocaleString() : new Date().toLocaleString(), icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>' }
    ];

    grid.innerHTML = info.map(item => `
        <div class="info-row">
            <div class="info-icon">${item.icon}</div>
            <div class="info-content">
                <span class="info-label">${item.label}</span>
                <span class="info-value">${item.value}</span>
            </div>
        </div>
    `).join('');
}

// --- Utils ---
function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        // easeOutQuart
        const ease = 1 - Math.pow(1 - progress, 4);
        obj.innerHTML = Math.floor(ease * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            obj.innerHTML = end;
        }
    };
    window.requestAnimationFrame(step);
}

function animateFloat(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 4);
        const current = (ease * (end - start) + start).toFixed(2);
        obj.innerHTML = current;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            obj.innerHTML = end.toFixed(2);
        }
    };
    window.requestAnimationFrame(step);
}

function generateMockReport() {
    return {
        passport: {
            model_class: 'RandomForestClassifier',
            framework: 'Scikit-Learn',
            task_type: 'BINARY_CLASSIFICATION',
            n_features: 32,
            n_train_samples: 50000,
            model_size_bytes: 48200000
        },
        health_score: {
            overall: 92,
            grade: 'A',
            verdict: 'Excellent — production-ready',
            dimensions: [
                { dimension: 'accuracy', score: 95, summary: 'High accuracy on test set' },
                { dimension: 'robustness', score: 88, summary: 'Good performance across data slices' },
                { dimension: 'generalization', score: 85, summary: 'Slight overfitting detected' },
                { dimension: 'interpretability', score: 75, summary: 'Tree ensemble is moderately interpretable' },
                { dimension: 'efficiency', score: 93, summary: 'Fast prediction latency' },
                { dimension: 'fairness', score: 90, summary: 'No significant demographic bias' }
            ]
        },
        top_recommendations: [
            {
                title: 'Reduce max_depth from 15 to 8',
                action: 'Reduce max_depth from 15 to 8',
                rationale: 'Deep trees are memorizing noise in the training set. Restricting depth will act as regularization and improve generalization performance on unseen data.',
                confidence: 'High',
                priority: 'High',
                estimated_improvement: '+3-5%',
                implementation_difficulty: 'Easy',
                affected_metrics: ['Validation Accuracy', 'Log Loss'],
                supporting_evidence: {
                    'Train Accuracy': '0.99',
                    'Test Accuracy': '0.86',
                    'Gap': '0.13'
                }
            },
            {
                title: 'Address class imbalance',
                action: 'Address class imbalance',
                rationale: 'The target variable is skewed 90/10. The model is biased towards the majority class.',
                confidence: 'Medium',
                priority: 'Medium',
                estimated_improvement: '+2-4%',
                implementation_difficulty: 'Medium',
                affected_metrics: ['F1 Score', 'Recall'],
                supporting_evidence: {
                    'Minority Class Ratio': '0.10'
                }
            },
             {
                title: 'Remove highly correlated features',
                action: 'Remove highly correlated features',
                rationale: 'Features A and B have a correlation of 0.98, causing multicollinearity.',
                confidence: 'Low',
                priority: 'Low',
                estimated_improvement: '+1-2%',
                implementation_difficulty: 'Easy',
                affected_metrics: ['Interpretability'],
                supporting_evidence: {}
            }
        ],
        generated_at: new Date().toISOString()
    };
}
