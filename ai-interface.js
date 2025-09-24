class AIInterface {
    constructor(aiManager) {
        this.aiManager = aiManager;
        this.elements = {};
        this.isVisible = false;
        this.updateInterval = null;
        
        this.initializeInterface();
        this.setupEventListeners();
    }

    initializeInterface() {
        // Create AI section HTML elements
        this.createAISection();
        this.initializeElements();
    }

    createAISection() {
        const mainContent = document.querySelector('.main-content');
        
        const aiSectionHTML = `
            <section class="ai-section" id="aiSection">
                <div class="ai-header">
                    <h3>
                        <span class="ai-icon">🤖</span>
                        AI Analysis & Insights
                    </h3>
                    <div class="ai-status" id="aiStatus">
                        <span class="ai-indicator" id="aiIndicator"></span>
                        <span class="ai-status-text" id="aiStatusText">Loading AI...</span>
                    </div>
                </div>

                <div class="ai-content" id="aiContent">
                    <div class="ai-loading" id="aiLoading">
                        <div class="loading-spinner"></div>
                        <p>Initializing AI model...</p>
                    </div>

                    <div class="ai-dashboard" id="aiDashboard" style="display: none;">
                        <!-- Current Analysis -->
                        <div class="analysis-card">
                            <h4>Current Analysis</h4>
                            <div class="prediction-display">
                                <div class="prediction-result" id="predictionResult">
                                    <span class="prediction-class" id="predictionClass">--</span>
                                    <span class="prediction-confidence" id="predictionConfidence">--</span>
                                </div>
                                <div class="prediction-probabilities" id="predictionProbs">
                                    <div class="prob-bar normal-prob">
                                        <label>Normal</label>
                                        <div class="prob-value" id="normalProb">--%</div>
                                        <div class="prob-bar-fill" id="normalBar"></div>
                                    </div>
                                    <div class="prob-bar mild-prob">
                                        <label>Mild Hypoxia</label>
                                        <div class="prob-value" id="mildProb">--%</div>
                                        <div class="prob-bar-fill" id="mildBar"></div>
                                    </div>
                                    <div class="prob-bar severe-prob">
                                        <label>Severe Hypoxia</label>
                                        <div class="prob-value" id="severeProb">--%</div>
                                        <div class="prob-bar-fill" id="severeBar"></div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Risk Assessment -->
                        <div class="risk-card">
                            <h4>Risk Assessment</h4>
                            <div class="risk-display">
                                <div class="risk-level" id="riskLevel">Low</div>
                                <div class="risk-meter">
                                    <div class="risk-bar" id="riskBar"></div>
                                </div>
                                <div class="risk-score" id="riskScore">0/7</div>
                            </div>
                        </div>

                        <!-- Insights & Recommendations -->
                        <div class="insights-card">
                            <h4>AI Insights & Recommendations</h4>
                            <div class="insights-list" id="insightsList">
                                <p class="no-insights">No insights available yet.</p>
                            </div>
                        </div>

                        <!-- Trend Analysis -->
                        <div class="trend-card">
                            <h4>Trend Analysis</h4>
                            <div class="trend-display">
                                <canvas id="trendChart" width="400" height="200"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="ai-controls">
                    <button id="toggleAIBtn" class="ai-toggle-btn">
                        <span class="btn-icon">👁️</span>
                        Hide AI Analysis
                    </button>
                    <button id="exportAIBtn" class="ai-export-btn">
                        <span class="btn-icon">📊</span>
                        Export AI Data
                    </button>
                </div>
            </section>
        `;

        // Insert AI section after chart section
        const chartSection = document.querySelector('.chart-section');
        chartSection.insertAdjacentHTML('afterend', aiSectionHTML);
    }

    initializeElements() {
        this.elements = {
            aiSection: document.getElementById('aiSection'),
            aiStatus: document.getElementById('aiStatus'),
            aiIndicator: document.getElementById('aiIndicator'),
            aiStatusText: document.getElementById('aiStatusText'),
            aiLoading: document.getElementById('aiLoading'),
            aiDashboard: document.getElementById('aiDashboard'),
            predictionClass: document.getElementById('predictionClass'),
            predictionConfidence: document.getElementById('predictionConfidence'),
            normalProb: document.getElementById('normalProb'),
            mildProb: document.getElementById('mildProb'),
            severeProb: document.getElementById('severeProb'),
            normalBar: document.getElementById('normalBar'),
            mildBar: document.getElementById('mildBar'),
            severeBar: document.getElementById('severeBar'),
            riskLevel: document.getElementById('riskLevel'),
            riskBar: document.getElementById('riskBar'),
            riskScore: document.getElementById('riskScore'),
            insightsList: document.getElementById('insightsList'),
            trendChart: document.getElementById('trendChart'),
            toggleAIBtn: document.getElementById('toggleAIBtn'),
            exportAIBtn: document.getElementById('exportAIBtn')
        };
    }

    setupEventListeners() {
        // AI model ready event
        window.addEventListener('aiModelReady', (event) => {
            this.handleModelReady(event.detail.isReady);
        });

        // AI error event
        window.addEventListener('aiError', (event) => {
            this.handleAIError(event.detail.message);
        });

        // Toggle visibility
        this.elements.toggleAIBtn.addEventListener('click', () => {
            this.toggleVisibility();
        });

        // Export AI data
        this.elements.exportAIBtn.addEventListener('click', () => {
            this.exportAIData();
        });
    }

    handleModelReady(isReady) {
        if (isReady) {
            this.elements.aiLoading.style.display = 'none';
            this.elements.aiDashboard.style.display = 'block';
            this.elements.aiIndicator.classList.add('connected');
            this.elements.aiStatusText.textContent = 'AI Ready';
            this.initializeTrendChart();
        }
    }

    handleAIError(message) {
        this.elements.aiLoading.style.display = 'none';
        this.elements.aiStatusText.textContent = 'AI Error';
        this.elements.aiIndicator.classList.add('error');
        
        this.elements.aiDashboard.innerHTML = `
            <div class="ai-error">
                <div class="error-icon">⚠️</div>
                <h4>AI Analysis Unavailable</h4>
                <p>${message}</p>
                <p>Manual monitoring is still available.</p>
            </div>
        `;
        this.elements.aiDashboard.style.display = 'block';
    }

    async updateAnalysis(spo2, heartRate) {
        if (!this.aiManager.isModelLoaded) return;

        try {
            // Get AI prediction
            const prediction = await this.aiManager.predictHypoxiaStatus(spo2, heartRate);
            
            // Generate insights
            const insights = this.aiManager.generateInsights(spo2, heartRate, prediction);
            
            // Get risk assessment
            const riskAssessment = this.aiManager.getRiskAssessment(spo2, heartRate, prediction);

            // Update UI
            this.updatePredictionDisplay(prediction);
            this.updateRiskDisplay(riskAssessment);
            this.updateInsightsDisplay(insights);
            this.updateTrendChart();

        } catch (error) {
            console.error('AI analysis update failed:', error);
        }
    }

    updatePredictionDisplay(prediction) {
        // Update prediction class and confidence
        this.elements.predictionClass.textContent = prediction.predictedClass;
        this.elements.predictionConfidence.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

        // Update probability bars
        this.updateProbabilityBar('normal', prediction.normal);
        this.updateProbabilityBar('mild', prediction.mildHypoxia);
        this.updateProbabilityBar('severe', prediction.severeHypoxia);

        // Style based on prediction
        this.elements.predictionClass.className = `prediction-class ${prediction.predictedClass.toLowerCase().replace(' ', '-')}`;
    }

    updateProbabilityBar(type, probability) {
        const percentage = (probability * 100).toFixed(1);
        
        this.elements[`${type}Prob`].textContent = `${percentage}%`;
        this.elements[`${type}Bar`].style.width = `${percentage}%`;
    }

    updateRiskDisplay(riskAssessment) {
        this.elements.riskLevel.textContent = riskAssessment.level;
        this.elements.riskScore.textContent = `${riskAssessment.score}/${riskAssessment.maxScore}`;
        
        const riskPercentage = (riskAssessment.score / riskAssessment.maxScore) * 100;
        this.elements.riskBar.style.width = `${riskPercentage}%`;
        
        // Style based on risk level
        this.elements.riskLevel.className = `risk-level ${riskAssessment.level.toLowerCase()}`;
        this.elements.riskBar.className = `risk-bar ${riskAssessment.level.toLowerCase()}`;
    }

    updateInsightsDisplay(insights) {
        if (insights.length === 0) {
            this.elements.insightsList.innerHTML = '<p class="no-insights">No insights available.</p>';
            return;
        }

        const insightsHTML = insights.map(insight => `
            <div class="insight-item ${insight.type}">
                <div class="insight-icon">${this.getInsightIcon(insight.type)}</div>
                <div class="insight-content">
                    <p class="insight-message">${insight.message}</p>
                    <p class="insight-recommendation">${insight.recommendation}</p>
                </div>
            </div>
        `).join('');

        this.elements.insightsList.innerHTML = insightsHTML;
    }

    getInsightIcon(type) {
        const icons = {
            'positive': '✅',
            'warning': '⚠️',
            'critical': '🚨',
            'info': 'ℹ️'
        };
        return icons[type] || 'ℹ️';
    }

    initializeTrendChart() {
        const ctx = this.elements.trendChart.getContext('2d');
        
        this.trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Severe Hypoxia Probability',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Mild Hypoxia Probability',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Normal Probability',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    },
                    x: {
                        display: false
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }

    updateTrendChart() {
        if (!this.trendChart) return;

        const history = this.aiManager.getAnalysisHistory();
        const recentHistory = history.slice(-20); // Last 20 predictions

        const labels = recentHistory.map((_, index) => index + 1);
        const severeData = recentHistory.map(entry => entry.severeHypoxia);
        const mildData = recentHistory.map(entry => entry.mildHypoxia);
        const normalData = recentHistory.map(entry => entry.normal);

        this.trendChart.data.labels = labels;
        this.trendChart.data.datasets[0].data = severeData;
        this.trendChart.data.datasets[1].data = mildData;
        this.trendChart.data.datasets[2].data = normalData;

        this.trendChart.update('none'); // Update without animation for real-time feel
    }

    toggleVisibility() {
        this.isVisible = !this.isVisible;
        
        if (this.isVisible) {
            this.elements.aiSection.classList.add('expanded');
            this.elements.toggleAIBtn.innerHTML = '<span class="btn-icon">👁️</span> Hide AI Analysis';
        } else {
            this.elements.aiSection.classList.remove('expanded');
            this.elements.toggleAIBtn.innerHTML = '<span class="btn-icon">👁️</span> Show AI Analysis';
        }
    }

    exportAIData() {
        try {
            const aiData = this.aiManager.exportPredictionData();
            const blob = new Blob([aiData], { type: 'application/json' });
            
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', `ai-analysis-data-${new Date().toISOString().split('T')[0]}.json`);
            link.style.visibility = 'hidden';
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error('Failed to export AI data:', error);
            alert('Failed to export AI data');
        }
    }

    // Method to be called from main app when new data arrives
    onNewVitalSigns(spo2, heartRate) {
        this.updateAnalysis(spo2, heartRate);
    }

    // Clear AI analysis data
    clearAnalysisData() {
        this.aiManager.clearAnalysisHistory();
        if (this.trendChart) {
            this.trendChart.data.labels = [];
            this.trendChart.data.datasets.forEach(dataset => {
                dataset.data = [];
            });
            this.trendChart.update();
        }
        
        // Reset displays
        this.elements.predictionClass.textContent = '--';
        this.elements.predictionConfidence.textContent = '--';
        this.elements.normalProb.textContent = '--%';
        this.elements.mildProb.textContent = '--%';
        this.elements.severeProb.textContent = '--%';
        this.elements.normalBar.style.width = '0%';
        this.elements.mildBar.style.width = '0%';
        this.elements.severeBar.style.width = '0%';
        this.elements.riskLevel.textContent = 'Low';
        this.elements.riskScore.textContent = '0/7';
        this.elements.riskBar.style.width = '0%';
        this.elements.insightsList.innerHTML = '<p class="no-insights">No insights available yet.</p>';
    }
}
// At the end of the file
window.AIInterface = AIInterface;