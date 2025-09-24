import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

class HypoxiaClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.analysisHistory = [];
        this.modelPath = './trained_model/model.json';
        this.preprocessingParams = null;
        this.scaler = null;
        this.inputShape = [8];
    }

    registerL2Regularizer() {
        class L2Regularizer {
            constructor(args) {
                this.l2 = args?.l2 ?? 0.01;
            }

            apply(x) {
                return tf.mul(this.l2, tf.sum(tf.square(x)));
            }

            getConfig() {
                return { l2: this.l2 };
            }
        }
        tf.serialization.registerClass(L2Regularizer);
        console.log("L2 regularizer registered");
    }

    async loadModel() {
        try {
            console.log('Loading hypoxia classification model...');
            await this.loadPreprocessingParams();
            this.registerL2Regularizer();
            this.model = await tf.loadLayersModel(this.modelPath);
            console.log('Model loaded successfully');
            console.log('Input shape:', this.model.inputShape);
            console.log('Output shape:', this.model.outputShape);
            this.isModelLoaded = true;
            window.dispatchEvent(new CustomEvent('aiModelReady', { detail: { isReady: true } }));
        } catch (error) {
            console.error('Failed to load model:', error);
            this.isModelLoaded = false;
            window.dispatchEvent(new CustomEvent('aiError', { detail: { message: 'Failed to load AI model: ' + error.message } }));
        }
    }

    async loadPreprocessingParams() {
        try {
            const response = await fetch('./trained_model/preprocessing_params.json');
            if (response.ok) {
                this.preprocessingParams = await response.json();
                this.scaler = {
                    center: this.preprocessingParams.scaler_center,
                    scale: this.preprocessingParams.scaler_scale
                };
                console.log('Preprocessing parameters loaded');
            } else {
                this.initializeDefaultScaler();
            }
        } catch (error) {
            console.warn('Could not load preprocessing parameters, using defaults');
            this.initializeDefaultScaler();
        }
    }

    initializeDefaultScaler() {
        this.scaler = {
            center: [90, 75, 1.2, 6.75, 8100, 5625, 2, 1],
            scale: [10, 20, 0.5, 2, 500, 1000, 1, 1]
        };
    }

    engineerFeatures(spo2, heartRate) {
        return [
            spo2,
            heartRate,
            spo2 / (heartRate + 1e-8),
            (spo2 * heartRate) / 1000,
            spo2 * spo2,
            heartRate * heartRate,
            this.getBinnedValue(spo2, [0, 85, 90, 95, 100]),
            this.getBinnedValue(heartRate, [0, 70, 90, 110, 200])
        ];
    }

    getBinnedValue(value, bins) {
        for (let i = 0; i < bins.length - 1; i++) {
            if (value >= bins[i] && value < bins[i + 1]) return i;
        }
        return value >= bins[bins.length - 1] ? bins.length - 2 : 0;
    }

    scaleFeatures(features) {
        return features.map((f, i) => (f - this.scaler.center[i]) / this.scaler.scale[i]);
    }

    async predictHypoxiaStatus(spo2, heartRate) {
        if (!this.isModelLoaded) throw new Error('Model not loaded');
        const rawFeatures = this.engineerFeatures(spo2, heartRate);
        const scaledFeatures = this.scaleFeatures(rawFeatures);
        const inputTensor = tf.tensor2d([scaledFeatures], [1, 8]);
        const prediction = this.model.predict(inputTensor);
        const probabilities = await prediction.data();
        inputTensor.dispose();
        prediction.dispose();

        const [normal, mildHypoxia, severeHypoxia] = probabilities;
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        const classes = ['Normal', 'Mild Hypoxia', 'Severe Hypoxia'];

        const result = {
            predictedClass: classes[maxIndex],
            confidence: probabilities[maxIndex],
            normal,
            mildHypoxia,
            severeHypoxia,
            timestamp: Date.now(),
            rawFeatures,
            scaledFeatures
        };

        this.analysisHistory.push(result);
        if (this.analysisHistory.length > 100) this.analysisHistory.shift();
        return result;
    }

    getAnalysisHistory() {
        return this.analysisHistory;
    }

    clearAnalysisHistory() {
        this.analysisHistory = [];
    }
}

export { HypoxiaClassifier };
