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
            
            // Load preprocessing parameters first
            await this.loadPreprocessingParams();
            
            // Try the model format fix approach
            await this.loadModelWithFormatFix();
            
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

    async loadModelWithFormatFix() {
        // Load the model JSON and fix the format
        const response = await fetch(this.modelPath);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const originalModel = await response.json();
        console.log('Original model loaded, fixing format...');
        
        // Create a fixed version
        const fixedModel = this.createFixedModelFormat(originalModel);
        
        // Load weights separately
        const weightsPath = this.modelPath.replace('model.json', 'group1-shard1of1.bin');
        const weightsResponse = await fetch(weightsPath);
        if (!weightsResponse.ok) {
            throw new Error(`Could not load weights: ${weightsResponse.status}`);
        }
        
        const weightsBuffer = await weightsResponse.arrayBuffer();
        
        // Create memory IO with fixed format
        const memoryIO = tf.io.fromMemory({
            modelTopology: fixedModel.modelTopology,
            weightSpecs: fixedModel.weightsManifest[0].weights,
            weightData: weightsBuffer,
            format: fixedModel.format,
            generatedBy: fixedModel.generatedBy,
            convertedBy: fixedModel.convertedBy
        });
        
        // Load the model
        this.model = await tf.loadLayersModel(memoryIO);
        console.log('Model loaded with format fix');
    }

    createFixedModelFormat(originalModel) {
        // Deep copy the original model
        const fixed = JSON.parse(JSON.stringify(originalModel));
        
        // Fix the model topology to be compatible with older TensorFlow.js versions
        if (fixed.modelTopology && fixed.modelTopology.model_config) {
            const config = fixed.modelTopology.model_config.config;
            
            if (config && config.layers) {
                // Fix the input layer specifically
                for (let i = 0; i < config.layers.length; i++) {
                    const layer = config.layers[i];
                    
                    if (layer.class_name === 'InputLayer') {
                        // Convert batch_shape to the format TensorFlow.js expects
                        if (layer.config.batch_shape) {
                            layer.config.batch_input_shape = layer.config.batch_shape;
                            // Keep both for compatibility
                        }
                        
                        // Ensure dtype is simple string
                        layer.config.dtype = 'float32';
                        
                        // Remove complex dtype objects
                        delete layer.config.dtype;
                        layer.config.dtype = 'float32';
                        
                        console.log('Fixed input layer:', layer.config);
                    }
                    
                    // Fix all other layers that have complex dtype objects
                    if (layer.config && layer.config.dtype && typeof layer.config.dtype === 'object') {
                        layer.config.dtype = 'float32';
                    }
                    
                    // Remove trainable dtype objects from all layers
                    if (layer.config && layer.config.dtype && layer.config.dtype.module) {
                        layer.config.dtype = 'float32';
                    }
                }
            }
        }
        
        return fixed;
    }

    async loadPreprocessingParams() {
        try {
            const response = await fetch('./trained_model/preprocessing_params.json');
            if (response.ok) {
                this.preprocessingParams = await response.json();
                
                if (this.preprocessingParams.scaler_center && this.preprocessingParams.scaler_scale) {
                    this.scaler = {
                        center: this.preprocessingParams.scaler_center,
                        scale: this.preprocessingParams.scaler_scale
                    };
                }
                
                console.log('Preprocessing parameters loaded successfully');
            }
        } catch (error) {
            console.warn('Could not load preprocessing parameters, using defaults');
            this.initializeDefaultScaler();
        }
    }

    initializeDefaultScaler() {
        // Default scaler parameters
        this.scaler = {
            center: [90, 75, 1.2, 6.75, 8100, 5625, 2, 1],
            scale: [10, 20, 0.5, 2, 500, 1000, 1, 1]
        };
    }

    engineerFeatures(spo2, heartRate) {
        const features = [];
        
        features.push(spo2);                                    
        features.push(heartRate);                               
        features.push(spo2 / (heartRate + 1e-8));              
        features.push((spo2 * heartRate) / 1000);              
        features.push(spo2 * spo2);                             
        features.push(heartRate * heartRate);                  
        features.push(this.getBinnedValue(spo2, [0, 85, 90, 95, 100]));     
        features.push(this.getBinnedValue(heartRate, [0, 70, 90, 110, 200])); 
        
        return features;
    }

    getBinnedValue(value, bins) {
        for (let i = 0; i < bins.length - 1; i++) {
            if (value >= bins[i] && value < bins[i + 1]) {
                return i;
            }
        }
        if (value >= bins[bins.length - 1]) {
            return bins.length - 2;
        }
        return 0;
    }

    scaleFeatures(features) {
        if (!this.scaler) {
            console.warn('No scaler available, returning unscaled features');
            return features;
        }
        
        return features.map((feature, i) => {
            return (feature - this.scaler.center[i]) / this.scaler.scale[i];
        });
    }

    async predictHypoxiaStatus(spo2, heartRate) {
        if (!this.isModelLoaded) {
            throw new Error('Model not loaded');
        }

        try {
            // Validate inputs
            if (spo2 < 0 || spo2 > 100 || heartRate < 0 || heartRate > 300) {
                throw new Error(`Invalid vital signs: SpO2=${spo2}, HR=${heartRate}`);
            }

            // Engineer and scale features
            const rawFeatures = this.engineerFeatures(spo2, heartRate);
            const scaledFeatures = this.scaleFeatures(rawFeatures);
            
            console.log('Raw features:', rawFeatures);
            console.log('Scaled features:', scaledFeatures);
            
            // Create input tensor
            const inputTensor = tf.tensor2d([scaledFeatures], [1, 8]);
            
            console.log('Input tensor shape:', inputTensor.shape);

            // Make prediction
            const prediction = this.model.predict(inputTensor);
            const probabilities = await prediction.data();
            
            // Clean up
            inputTensor.dispose();
            prediction.dispose();

            // Process results
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
                timestamp: Date.now(),
                rawFeatures,
                scaledFeatures
            };

            this.analysisHistory.push(result);
            
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
        } else {
            insights.push({
                type: 'positive',
                message: 'Normal oxygen saturation',
                recommendation: 'Continue current care'
            });
        }

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

        if (spo2 < 90) score += 3;
        else if (spo2 < 95) score += 1;

        if (heartRate < 50 || heartRate > 120) score += 2;
        else if (heartRate < 60 || heartRate > 100) score += 1;

        if (prediction.predictedClass === 'Severe Hypoxia') score += 2;
        else if (prediction.predictedClass === 'Mild Hypoxia') score += 1;

        let level = 'Low';
        if (score >= 5) level = 'Critical';
        else if (score >= 3) level = 'High';
        else if (score >= 1) level = 'Medium';

        return { score, maxScore, level };
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
                inputShape: this.model?.inputShape,
                outputShape: this.model?.outputShape
            },
            preprocessingParams: this.preprocessingParams,
            scalerInfo: this.scaler,
            analysisHistory: this.analysisHistory,
            exportTime: new Date().toISOString()
        }, null, 2);
    }
}

// Alias for compatibility
const HypoxiaClassifier = FixedHypoxiaClassifier;