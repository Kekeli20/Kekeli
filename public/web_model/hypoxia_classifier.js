// hypoxia_classifier.js
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class HypoxiaClassifier {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.analysisHistory = [];
        this.modelPath = './trained_model/model.json';
        this.preprocessingParams = null;
        this.scaler = null;
        this.inputShape = [8];

        this.registerL2Regularizer();
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
            await this.loadModelWithFormatFix();

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

    async loadModelWithFormatFix() {
        const response = await fetch(this.modelPath);
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

        const originalModel = await response.json();
        console.log('Original model loaded, fixing format...');
        const fixedModel = this.createFixedModelFormat(originalModel);

        const weightsPath = this.modelPath.replace('model.json', 'group1-shard1of1.bin');
        const weightsResponse = await fetch(weightsPath);
        if (!weightsResponse.ok) throw new Error(`Could not load weights: ${weightsResponse.status}`);

        const weightsBuffer = await weightsResponse.arrayBuffer();
        const memoryIO = tf.io.fromMemory({
            modelTopology: fixedModel.modelTopology,
            weightSpecs: fixedModel.weightsManifest[0].weights,
            weightData: weightsBuffer,
            format: fixedModel.format,
            generatedBy: fixedModel.generatedBy,
            convertedBy: fixedModel.convertedBy
        });

        this.model = await tf.loadLayersModel(memoryIO);
        console.log('Model loaded with format fix');
    }

    createFixedModelFormat(originalModel) {
        const fixed = JSON.parse(JSON.stringify(originalModel));
        if (fixed.modelTopology && fixed.modelTopology.model_config) {
            const config = fixed.modelTopology.model_config.config;
            if (config && config.layers) {
                for (let layer of config.layers) {
                    if (layer.class_name === 'InputLayer' && layer.config.batch_shape) {
                        layer.config.batch_input_shape = layer.config.batch_shape;
                        layer.config.dtype = 'float32';
                    }
                    if (layer.config?.dtype && typeof layer.config.dtype === 'object') {
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
            if (value >= bins[i] && value < bins[i + 1]) return i;
        }
        return value >= bins[bins.length - 1] ? bins.length - 2 : 0;
    }

    scaleFeatures(features) {
        if (!this.scaler) return features;
        return features.map((feature, i) => (feature - this.scaler.center[i]) / this.scaler.scale[i]);
    }

    async predictHypoxiaStatus(spo2, heartRate) {
        if (!this.isModelLoaded) throw new Error('Model not loaded');

        if (spo2 < 0 || spo2 > 100 || heartRate < 0 || heartRate > 300) {
            throw new Error(`Invalid vital signs: SpO2=${spo2}, HR=${heartRate}`);
        }

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

        return {
            predictedClass: classes[maxIndex],
            confidence: probabilities[maxIndex],
            normal,
            mildHypoxia,
            severeHypoxia,
            timestamp: Date.now(),
            rawFeatures,
            scaledFeatures
        };
    }
}
