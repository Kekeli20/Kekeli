class FixedHypoxiaClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.analysisHistory = [];
        this.modelPath = './trained_model/model.json';
        this.preprocessingParams = null;
        this.scaler = null;
        this.inputShape = [8];
    }

    async loadModel() {
        try {
            console.log('Loading hypoxia classification model...');
            this.model = await tf.loadLayersModel(this.modelPath);
            console.log('Model loaded successfully');
            this.isModelLoaded = true;
            
            // Dispatch ready event
            window.dispatchEvent(new CustomEvent('aiModelReady', {
                detail: { isReady: true }
            }));
        } catch (error) {
            console.error('Failed to load model:', error);
            this.isModelLoaded = false;
            
            // Dispatch error event
            window.dispatchEvent(new CustomEvent('aiError', {
                detail: { message: 'Failed to load AI model: ' + error.message }
            }));
        }
    }

    async predictHypoxiaStatus(spo2, heartRate) {
        if (!this.isModelLoaded) {
            throw new Error('Model not loaded');
        }

        try {
            // Prepare input features (matching your training script)
            const features = tf.tensor2d([[
                spo2,
                heartRate,
                spo2 / (heartRate + 1e-8), // spo2_hr_ratio
                (spo2 * heartRate) / 1000,  // spo2_hr_product
                spo2 * spo2,                // spo2_squared
                heartRate * heartRate,      // hr_squared
                this.getBinnedValue(spo2, [0, 85, 90, 95, 100]), // spo2_binned
                this.getBinnedValue(heartRate, [0, 70, 90, 110, 200]) // hr_binned
            ]]);

            // Make prediction
            const prediction = this.model.predict(features);
            const probabilities = await prediction.data();
            
            // Clean up tensors
            features.dispose();
            prediction.dispose();

            // Map probabilities to classes
            const [normal, mildHypoxia, severeHypoxia] = probabilities;
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            
            const classes = ['Normal', 'Mild Hypoxia', 'Severe Hypoxia'];
            const predictedClass = classes[maxIndex];
            const confidence = probabilities[maxIndex];

            const result = {
                predictedClass,
                confidence,
                normal,
                mildHypoxia,
                severeHypoxia,
                timestamp: Date.now()
            };

            // Store in history
            this.analysisHistory.push(result);
            
            // Keep only last 100 predictions
            if (this.analysisHistory.length > 100) {
                this.analysisHistory = this.analysisHistory.slice(-100);
            }

            return result;
        } catch (error) {
            console.error('Prediction failed:', error);
            throw error;
        }
    }

    getBinnedValue(value, bins) {
        for (let i = 0; i < bins.length - 1; i++) {
            if (value >= bins[i] && value < bins[i + 1]) {
                return i;
            }
        }
        return bins.length - 1;
    }

    generateInsights(spo2, heartRate, prediction) {
        const insights = [];

        // SpO2 insights
        if (spo2 < 90) {
            insights.push({
                type: 'critical',
                message: 'Severe oxygen desaturation detected',
                recommendation: 'Immediate medical attention required'
            });
        } else if (spo2 < 95) {
            insights.push({
                type: 'warning',
                message: 'Mild oxygen desaturation',
                recommendation: 'Monitor closely and consider oxygen therapy'
            });
        } else if (spo2 >= 95) {
            insights.push({
                type: 'positive',
                message: 'Normal oxygen saturation',
                recommendation: 'Continue current care'
            });
        }

        // Heart rate insights
        if (heartRate < 60) {
            insights.push({
                type: 'warning',
                message: 'Bradycardia detected',
                recommendation: 'Monitor heart rate closely'
            });
        } else if (heartRate > 100) {
            insights.push({
                type: 'warning',
                message: 'Tachycardia detected',
                recommendation: 'Consider causes and treatment'
            });
        }

        // AI prediction insights
        if (prediction.confidence < 0.7) {
            insights.push({
                type: 'info',
                message: 'AI prediction confidence is low',
                recommendation: 'Consider manual assessment'
            });
        }

        return insights;
    }

    getRiskAssessment(spo2, heartRate, prediction) {
        let score = 0;
        const maxScore = 7;

        // SpO2 risk factors
        if (spo2 < 90) score += 3;
        else if (spo2 < 95) score += 1;

        // Heart rate risk factors
        if (heartRate < 50 || heartRate > 120) score += 2;
        else if (heartRate < 60 || heartRate > 100) score += 1;

        // AI prediction risk
        if (prediction.predictedClass === 'Severe Hypoxia') score += 2;
        else if (prediction.predictedClass === 'Mild Hypoxia') score += 1;

        let level = 'Low';
        if (score >= 5) level = 'Critical';
        else if (score >= 3) level = 'High';
        else if (score >= 1) level = 'Medium';

        return {
            score,
            maxScore,
            level
        };
    }

    getAnalysisHistory() {
        return this.analysisHistory;
    }

    clearAnalysisHistory() {
        this.analysisHistory = [];
    }

    exportPredictionData() {
        return JSON.stringify({
            modelInfo: {
                loaded: this.isModelLoaded,
                path: this.modelPath
            },
            analysisHistory: this.analysisHistory,
            exportTime: new Date().toISOString()
        }, null, 2);
    }
}

// Make available globally for browser use
window.HypoxiaClassifier = FixedHypoxiaClassifier;
window.FixedHypoxiaClassifier = FixedHypoxiaClassifier;