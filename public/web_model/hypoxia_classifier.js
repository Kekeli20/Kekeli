class HypoxiaClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.analysisHistory = [];
        this.modelPath = './trained_model/model.json'; // Updated path to match Python output
        this.preprocessingParams = null;
        this.scaler = null;
    }

    async loadModel() {
        try {
            console.log('Loading hypoxia classification model...');
            
            // Load preprocessing parameters first
            await this.loadPreprocessingParams();
            
            // Load the TensorFlow.js model
            this.model = await tf.loadLayersModel(this.modelPath);
            
            console.log('Model loaded successfully');
            console.log('Input shape:', this.model.inputShape);
            console.log('Output shape:', this.model.outputShape);
            
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

    async loadPreprocessingParams() {
        try {
            const response = await fetch('./trained_model/preprocessing_params.json');
            this.preprocessingParams = await response.json();
            
            // Initialize scaler with saved parameters
            if (this.preprocessingParams.scaler_center && this.preprocessingParams.scaler_scale) {
                this.scaler = {
                    center: this.preprocessingParams.scaler_center,
                    scale: this.preprocessingParams.scaler_scale
                };
            }
            
            console.log('Preprocessing parameters loaded:', this.preprocessingParams);
        } catch (error) {
            console.warn('Could not load preprocessing parameters, using defaults:', error);
            // Use default parameters if file not found
            this.initializeDefaultScaler();
        }
    }

    initializeDefaultScaler() {
        // Default scaler parameters - you should replace these with actual values from training
        this.scaler = {
            center: [0, 0, 0, 0, 0, 0, 0, 0], // 8 features
            scale: [1, 1, 1, 1, 1, 1, 1, 1]   // 8 features
        };
    }

    engineerFeatures(spo2, heartRate) {
        // Match the exact feature engineering from Python training script
        const features = [];
        
        // Basic features
        features.push(spo2);                                    // spo2
        features.push(heartRate);                               // heart_rate
        
        // Interaction features
        features.push(spo2 / (heartRate + 1e-8));              // spo2_hr_ratio
        features.push((spo2 * heartRate) / 1000);              // spo2_hr_product
        
        // Polynomial features
        features.push(spo2 * spo2);                             // spo2_squared
        features.push(heartRate * heartRate);                  // hr_squared
        
        // Binned features (matching Python pd.cut logic)
        features.push(this.getBinnedValue(spo2, [0, 85, 90, 95, 100]));     // spo2_binned
        features.push(this.getBinnedValue(heartRate, [0, 70, 90, 110, 200])); // hr_binned
        
        return features;
    }

    getBinnedValue(value, bins) {
        // Match pandas cut logic: bins=[0, 85, 90, 95, 100], labels=[0, 1, 2, 3]
        for (let i = 0; i < bins.length - 1; i++) {
            if (value >= bins[i] && value < bins[i + 1]) {
                return i;
            }
        }
        // Handle edge case for maximum value
        if (value >= bins[bins.length - 1]) {
            return bins.length - 2;
        }
        return 0; // Default to first bin for values below minimum
    }

    scaleFeatures(features) {
        if (!this.scaler) {
            console.warn('No scaler available, returning unscaled features');
            return features;
        }
        
        // Apply RobustScaler transformation: (x - center) / scale
        const scaledFeatures = features.map((feature, i) => {
            return (feature - this.scaler.center[i]) / this.scaler.scale[i];
        });
        
        return scaledFeatures;
    }

    async predictHypoxiaStatus(spo2, heartRate) {
        if (!this.isModelLoaded) {
            throw new Error('Model not loaded');
        }

        try {
            // Validate inputs
            if (spo2 < 0 || spo2 > 100 || heartRate < 0 || heartRate > 300) {
                throw new Error('Invalid vital signs values');
            }

            // Engineer features (must match Python exactly)
            const rawFeatures = this.engineerFeatures(spo2, heartRate);
            
            // Scale features
            const scaledFeatures = this.scaleFeatures(rawFeatures);
            
            console.log('Raw features:', rawFeatures);
            console.log('Scaled features:', scaledFeatures);

            // Create tensor with correct shape [1, 8]
            const inputTensor = tf.tensor2d([scaledFeatures], [1, 8]);
            
            console.log('Input tensor shape:', inputTensor.shape);

            // Make prediction
            const prediction = this.model.predict(inputTensor);
            const probabilities = await prediction.data();
            
            // Clean up tensors
            inputTensor.dispose();
            prediction.dispose();

            // Map probabilities to classes (matching Python label encoding)
            const [normal, mildHypoxia, severeHypoxia] = probabilities;
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            
            // Classes should match the label encoder from Python
            const classes = ['Normal', 'Mild Hypoxia', 'Severe Hypoxia'];
            const predictedClass = classes[maxIndex];
            const confidence = probabilities[maxIndex];

            const result = {
                predictedClass,
                confidence,
                normal,
                mildHypoxia,
                severeHypoxia,
                timestamp: Date.now(),
                rawFeatures,
                scaledFeatures
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

        // Feature-based insights
        if (prediction.rawFeatures) {
            const [spo2Val, hrVal, ratio, product] = prediction.rawFeatures;
            
            if (ratio < 1.0) {
                insights.push({
                    type: 'info',
                    message: 'SpO2 to heart rate ratio is low',
                    recommendation: 'May indicate compensatory response'
                });
            }
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
                path: this.modelPath,
                inputShape: this.model?.inputShape,
                outputShape: this.model?.outputShape
            },
            preprocessingParams: this.preprocessingParams,
            scalerInfo: this.scaler,
            analysisHistory: this.analysisHistory,
            exportTime: new Date().toISOString()
        }, null, 2);
    }

    // Debug method to test feature engineering
    debugFeatureEngineering(spo2, heartRate) {
        console.log('=== Feature Engineering Debug ===');
        console.log('Input values:', { spo2, heartRate });
        
        const rawFeatures = this.engineerFeatures(spo2, heartRate);
        console.log('Raw features:', rawFeatures);
        
        const scaledFeatures = this.scaleFeatures(rawFeatures);
        console.log('Scaled features:', scaledFeatures);
        
        console.log('Feature mapping:');
        const featureNames = ['spo2', 'heart_rate', 'spo2_hr_ratio', 'spo2_hr_product', 
                             'spo2_squared', 'hr_squared', 'spo2_binned', 'hr_binned'];
        
        rawFeatures.forEach((feature, i) => {
            console.log(`  ${featureNames[i]}: ${feature} -> ${scaledFeatures[i]}`);
        });
        
        return { rawFeatures, scaledFeatures };
    }
}

// Testing utility to verify the feature engineering matches Python
const FeatureEngineeringTest = {
    testCase: (spo2, heartRate) => {
        console.log(`\n=== Testing SpO2: ${spo2}, Heart Rate: ${heartRate} ===`);
        
        const classifier = new HypoxiaClassifier();
        classifier.initializeDefaultScaler(); // Use defaults for testing
        
        const result = classifier.debugFeatureEngineering(spo2, heartRate);
        
        // Expected calculations (verify these match your Python output)
        const expected = {
            spo2: spo2,
            heart_rate: heartRate,
            spo2_hr_ratio: spo2 / (heartRate + 1e-8),
            spo2_hr_product: (spo2 * heartRate) / 1000,
            spo2_squared: spo2 * spo2,
            hr_squared: heartRate * heartRate,
            spo2_binned: classifier.getBinnedValue(spo2, [0, 85, 90, 95, 100]),
            hr_binned: classifier.getBinnedValue(heartRate, [0, 70, 90, 110, 200])
        };
        
        console.log('Expected features:', Object.values(expected));
        console.log('Match:', JSON.stringify(result.rawFeatures) === JSON.stringify(Object.values(expected)));
        
        return result;
    }
};

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { HypoxiaClassifier, FeatureEngineeringTest };
}