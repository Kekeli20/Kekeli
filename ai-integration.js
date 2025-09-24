class HypoxiaMonitorAppWithAI extends HypoxiaMonitorApp {
    constructor() {
        super();

        // Initialize AI components
        this.aiManager = new HypoxiaClassifier(); // Use your production classifier
        this.aiInterface = null;

        // Register event listener BEFORE loading the model
        this.initializeAI();

        // Now load the model
        this.aiManager.loadModel();
    }

    initializeAI() {
        window.addEventListener('aiModelReady', () => {
            console.log('aiModelReady event received');
            this.aiInterface = new AIInterface(this.aiManager);
            console.log('AIInterface created:', this.aiInterface);
        });

        window.addEventListener('aiError', (event) => {
            console.error('AI Error:', event.detail.message);
            this.showError(`AI Analysis Error: ${event.detail.message}`);
        });
    }

    // Override the updateSpo2 method to include AI analysis
    updateSpo2(value, timestamp) {
        super.updateSpo2(value, timestamp);

        if (this.aiInterface && this.currentSpo2 !== null && this.currentHR !== null) {
            this.aiInterface.onNewVitalSigns(this.currentSpo2, this.currentHR);
        }
    }

    // Override the updateHeartRate method to include AI analysis
    updateHeartRate(value, timestamp) {
        super.updateHeartRate(value, timestamp);

        if (this.aiInterface && this.currentSpo2 !== null && this.currentHR !== null) {
            this.aiInterface.onNewVitalSigns(this.currentSpo2, this.currentHR);
        }
    }

    // Override the clearAllData method to also clear AI data
    clearAllData() {
        if (confirm('Are you sure you want to clear all recorded data including AI analysis? This action cannot be undone.')) {
            super.clearAllData();
            if (this.aiInterface) {
                this.aiInterface.clearAnalysisData();
            }
        }
    }

    getSessionDuration() {
        if (!this.sessionStartTime) return 0;
        return Math.floor((new Date() - this.sessionStartTime) / 1000);
    }

    getAIInsights() {
        if (!this.aiManager) return null;
        return this.aiManager.getAnalysisHistory();
    }

    generateCSV() {
        const headers = ['Timestamp', 'Date', 'Time', 'Type', 'Value', 'Unit', 'Status', 'AI_Prediction', 'AI_Confidence', 'Risk_Level'];
        const rows = [headers];

        const aiHistory = this.aiManager ? this.aiManager.getAnalysisHistory() : [];

        this.dataStorage.forEach((entry, index) => {
            const date = entry.timestamp.toLocaleDateString();
            const time = entry.timestamp.toLocaleTimeString();
            const type = entry.type === 'spo2' ? 'SpO2' : 'Heart Rate';
            const unit = entry.type === 'spo2' ? '%' : 'BPM';

            let status = 'Normal';
            if (entry.type === 'spo2') {
                if (entry.value < 90) status = 'Severe Hypoxia';
                else if (entry.value < 95) status = 'Mild Hypoxia';
            } else {
                if (entry.value < 60 || entry.value > 100) status = 'Abnormal';
            }

            let aiPrediction = '';
            let aiConfidence = '';
            let riskLevel = '';

            const correspondingAI = aiHistory.find(ai =>
                Math.abs(ai.timestamp - entry.timestamp) < 5000 // Within 5 seconds
            );

            if (correspondingAI) {
                aiPrediction = correspondingAI.predictedClass;
                aiConfidence = (correspondingAI.confidence * 100).toFixed(1) + '%';

                const riskAssessment = this.aiManager.getRiskAssessment(
                    entry.type === 'spo2' ? entry.value : this.currentSpo2,
                    entry.type === 'heartRate' ? entry.value : this.currentHR,
                    correspondingAI
                );
                riskLevel = riskAssessment.level;
            }

            rows.push([
                entry.timestamp.toISOString(),
                date,
                time,
                type,
                entry.value,
                unit,
                status,
                aiPrediction,
                aiConfidence,
                riskLevel
            ]);
        });

        return rows.map(row => row.join(',')).join('\n');
    }
}

// Configuration object for easy customization (updated for production model)
const AIConfig = {
    modelPath: '/web_model/model.json', // Path to your trained model
    updateInterval: 2000,
    featureRanges: {
        spo2: { min: 70, max: 100 },
        heartRate: { min: 40, max: 180 }
    },
    alertThresholds: {
        severeHypoxiaProb: 0.7,
        criticalRiskScore: 4
    }
};

const AIUtils = {
    async loadTrainedModel(modelPath) {
        try {
            const model = await tf.loadLayersModel(modelPath);
            console.log('Trained model loaded successfully');
            return model;
        } catch (error) {
            console.error('Failed to load trained model:', error);
            throw error;
        }
    },

    validateVitalSigns(spo2, heartRate) {
        return {
            isValid: spo2 >= 0 && spo2 <= 100 && heartRate >= 0 && heartRate <= 300,
            spo2InRange: spo2 >= 70 && spo2 <= 100,
            heartRateInRange: heartRate >= 30 && heartRate <= 200
        };
    },

    formatPrediction(prediction) {
        return {
            class: prediction.predictedClass,
            confidence: `${(prediction.confidence * 100).toFixed(1)}%`,
            probabilities: {
                normal: `${(prediction.normal * 100).toFixed(1)}%`,
                mild: `${(prediction.mildHypoxia * 100).toFixed(1)}%`,
                severe: `${(prediction.severeHypoxia * 100).toFixed(1)}%`
            }
        };
    },

    generateAISummary(analysisHistory) {
        if (!analysisHistory.length) return null;

        const totalPredictions = analysisHistory.length;
        const classCount = analysisHistory.reduce((acc, entry) => {
            acc[entry.predictedClass] = (acc[entry.predictedClass] || 0) + 1;
            return acc;
        }, {});

        const avgConfidence = analysisHistory.reduce((sum, entry) =>
            sum + entry.confidence, 0) / totalPredictions;

        return {
            totalPredictions,
            classDistribution: {
                normal: ((classCount['Normal'] || 0) / totalPredictions * 100).toFixed(1),
                mildHypoxia: ((classCount['Mild Hypoxia'] || 0) / totalPredictions * 100).toFixed(1),
                severeHypoxia: ((classCount['Severe Hypoxia'] || 0) / totalPredictions * 100).toFixed(1)
            },
            averageConfidence: (avgConfidence * 100).toFixed(1)
        };
    },

    async testAIIntegration(aiManager) {
        console.log('Testing AI integration...');

        const testCases = [
            { spo2: 98, heartRate: 72, expected: 'Normal' },
            { spo2: 92, heartRate: 85, expected: 'Mild Hypoxia' },
            { spo2: 87, heartRate: 110, expected: 'Severe Hypoxia' }
        ];

        for (const testCase of testCases) {
            try {
                const prediction = await aiManager.predictHypoxiaStatus(testCase.spo2, testCase.heartRate);
                console.log(`Test: SpO2=${testCase.spo2}, HR=${testCase.heartRate}`);
                console.log(`Expected: ${testCase.expected}, Got: ${prediction.predictedClass}`);
                console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);
                console.log('---');
            } catch (error) {
                console.error('Test failed:', error);
            }
        }

        console.log('AI integration test completed');
    }
};

// Export for use in your application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { HypoxiaMonitorAppWithAI, AIConfig, AIUtils };
}