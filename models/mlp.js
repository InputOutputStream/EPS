import * as tf from '@tensorflow/tfjs';
import { BaseModel, ModelUtils } from './base.js';

export class MLPModel extends BaseModel {
    constructor(config = {}) {
        const defaultConfig = {
            hiddenLayers: [64, 32],
            activation: 'relu',
            dropout: 0.2,
            learningRate: 0.001,
            batchSize: 32,
            epochs: 100,
            outputActivation: 'softmax',
            numClasses: 3
        };
        
        super('MLP', { ...defaultConfig, ...config });
        this.buildModel();
    }

    buildModel() {
        const layers = [];
        
        // Input layer will be defined when we fit the model
        this.inputShape = null;
        
        // Hidden layers
        this.config.hiddenLayers.forEach((units, index) => {
            if (index === 0) {
                // First hidden layer - input shape will be set during fit
                layers.push(tf.layers.dense({
                    units: units,
                    activation: this.config.activation,
                    name: `hidden_${index + 1}`
                }));
            } else {
                layers.push(tf.layers.dense({
                    units: units,
                    activation: this.config.activation,
                    name: `hidden_${index + 1}`
                }));
            }
            
            // Add dropout after each hidden layer
            if (this.config.dropout > 0) {
                layers.push(tf.layers.dropout({
                    rate: this.config.dropout,
                    name: `dropout_${index + 1}`
                }));
            }
        });
        
        // Output layer
        layers.push(tf.layers.dense({
            units: this.config.numClasses,
            activation: this.config.outputActivation,
            name: 'output'
        }));

        this.layers = layers;
    }

    createModel(inputShape) {
        const model = tf.sequential();
        
        // Add input layer
        model.add(tf.layers.dense({
            inputShape: [inputShape],
            units: this.config.hiddenLayers[0],
            activation: this.config.activation,
            name: 'input_hidden'
        }));
        
        // Add dropout after first layer
        if (this.config.dropout > 0) {
            model.add(tf.layers.dropout({ rate: this.config.dropout }));
        }
        
        // Add remaining hidden layers
        for (let i = 1; i < this.config.hiddenLayers.length; i++) {
            model.add(tf.layers.dense({
                units: this.config.hiddenLayers[i],
                activation: this.config.activation,
                name: `hidden_${i + 1}`
            }));
            
            if (this.config.dropout > 0) {
                model.add(tf.layers.dropout({ rate: this.config.dropout }));
            }
        }
        
        // Output layer
        model.add(tf.layers.dense({
            units: this.config.numClasses,
            activation: this.config.outputActivation,
            name: 'output'
        }));

        // Compile model
        model.compile({
            optimizer: tf.train.adam(this.config.learningRate),
            loss: this.config.numClasses > 2 ? 'categoricalCrossentropy' : 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    async fit(X, y, validationSplit = 0.2) {
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const inputShape = XTensor.shape[1];
        
        // Create model with proper input shape
        this.model = this.createModel(inputShape);
        this.inputShape = inputShape;
        
        // Prepare labels
        let yTensor;
        if (this.config.numClasses > 2) {
            yTensor = y instanceof tf.Tensor ? y : ModelUtils.oneHotEncode(y, this.config.numClasses);
        } else {
            yTensor = y instanceof tf.Tensor ? y : tf.tensor1d(y);
        }

        // Training configuration
        const fitConfig = {
            epochs: this.config.epochs,
            batchSize: this.config.batchSize,
            validationSplit: validationSplit,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 10 === 0) {
                        console.log(`Epoch ${epoch + 1}/${this.config.epochs} - Loss: ${logs.loss.toFixed(4)} - Accuracy: ${logs.acc.toFixed(4)}`);
                    }
                }
            }
        };

        // Train model
        const history = await this.model.fit(XTensor, yTensor, fitConfig);
        this.trained = true;
        
        // Store training history
        this.trainingHistory = history;

        // Clean up tensors
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();

        return history;
    }

    async predict(X) {
        if (!this.trained || !this.model) {
            throw new Error('Model must be trained before prediction');
        }

        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const predictions = this.model.predict(XTensor);
        
        // Clean up input tensor if we created it
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        
        return predictions;
    }

    async predictClasses(X) {
        const predictions = await this.predict(X);
        const classes = ModelUtils.argMax(predictions);
        
        predictions.dispose();
        return classes;
    }

    async predictProbabilities(X) {
        return await this.predict(X);
    }

    updateHyperparameters(newConfig) {
        this.updateConfig(newConfig);
        
        // Rebuild model if structure changed
        if (newConfig.hiddenLayers || newConfig.dropout || newConfig.activation) {
            this.dispose();
            this.buildModel();
            this.trained = false;
        }
        
        // Update learning rate if model exists
        if (this.model && newConfig.learningRate) {
            this.model.compile({
                optimizer: tf.train.adam(newConfig.learningRate),
                loss: this.config.numClasses > 2 ? 'categoricalCrossentropy' : 'binaryCrossentropy',
                metrics: ['accuracy']
            });
        }
    }

    getModelSummary() {
        if (this.model) {
            this.model.summary();
            return this.model.layers.map(layer => ({
                name: layer.name,
                outputShape: layer.outputShape,
                params: layer.countParams()
            }));
        }
        return null;
    }

    async saveModel(path = 'models/mlp_model') {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        await this.model.save(`localstorage://${path}`);
        
        // Save config
        localStorage.setItem(`${path}_config`, JSON.stringify(this.config));
        
        console.log(`MLP model saved to ${path}`);
    }

    async loadModel(path = 'models/mlp_model') {
        try {
            this.model = await tf.loadLayersModel(`localstorage://${path}`);
            
            // Load config
            const configStr = localStorage.getItem(`${path}_config`);
            if (configStr) {
                this.config = JSON.parse(configStr);
            }
            
            this.trained = true;
            console.log(`MLP model loaded from ${path}`);
        } catch (error) {
            console.error('Error loading MLP model:', error);
        }
    }

    dispose() {
        super.dispose();
        this.layers = null;
        this.trainingHistory = null;
    }
}