import * as tf from '@tensorflow/tfjs';
import { BaseModel, ModelUtils } from './base.js';

/**
 * XGBoost-like implementation using TensorFlow.js
 * Simulates gradient boosting with neural network weak learners
 */
export class XGBoostModel extends BaseModel {
    constructor(config = {}) {
        const defaultConfig = {
            nEstimators: 100,
            maxDepth: 6,
            learningRate: 0.1,
            subsample: 0.8,
            colsampleByTree: 0.8,
            regAlpha: 0.0,
            regLambda: 1.0,
            minChildWeight: 1,
            gamma: 0.0,
            numClasses: 3,
            objective: 'multi:softprob'
        };
        
        super('XGBoost', { ...defaultConfig, ...config });
        this.estimators = [];
        this.featureImportances = null;
        this.basePrediction = null;
    }

    createWeakLearner(inputSize, depth) {
        // Create a simple neural network as weak learner
        const layers = [];
        
        // Calculate layer sizes based on depth
        const hiddenSize = Math.max(4, Math.floor(inputSize / (depth + 1)));
        
        layers.push(tf.layers.dense({
            inputShape: [inputSize],
            units: hiddenSize,
            activation: 'tanh',
            kernelRegularizer: tf.regularizers.l1l2({
                l1: this.config.regAlpha,
                l2: this.config.regLambda
            })
        }));
        
        // Add depth layers
        for (let i = 1; i < depth; i++) {
            const layerSize = Math.max(2, Math.floor(hiddenSize / (i + 1)));
            layers.push(tf.layers.dense({
                units: layerSize,
                activation: 'tanh',
                kernelRegularizer: tf.regularizers.l1l2({
                    l1: this.config.regAlpha,
                    l2: this.config.regLambda
                })
            }));
        }
        
        // Output layer
        layers.push(tf.layers.dense({
            units: this.config.numClasses,
            activation: 'linear' // Raw logits for gradient boosting
        }));
        
        const model = tf.sequential({ layers });
        
        model.compile({
            optimizer: tf.train.sgd(this.config.learningRate),
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    async subsampleData(X, y) {
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const yTensor = y instanceof tf.Tensor ? y : tf.tensor2d(y);
        
        const nSamples = XTensor.shape[0];
        const nFeatures = XTensor.shape[1];
        
        // Row subsampling
        const nRowSample = Math.floor(nSamples * this.config.subsample);
        const rowIndices = tf.util.createShuffledIndices(nSamples).slice(0, nRowSample);
        
        // Column subsampling
        const nColSample = Math.floor(nFeatures * this.config.colsampleByTree);
        const colIndices = tf.util.createShuffledIndices(nFeatures).slice(0, nColSample);
        
        // Sample data
        const XSampled = XTensor.gather(rowIndices).gather(colIndices, 1);
        const ySampled = yTensor.gather(rowIndices);
        
        // Clean up if we created tensors
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        
        return { XSampled, ySampled, rowIndices, colIndices };
    }

    async computeGradients(yTrue, yPred) {
        // Compute gradients for multi-class classification
        const softmaxPred = tf.softmax(yPred);
        const gradients = softmaxPred.sub(yTrue);
        
        // Compute hessians (second derivatives)
        const hessians = softmaxPred.mul(tf.scalar(1).sub(softmaxPred));
        
        return { gradients, hessians };
    }

    async fit(X, y, validationSplit = 0.1) {
        console.log(`Training XGBoost with ${this.config.nEstimators} estimators...`);
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const inputSize = XTensor.shape[1];
        
        // Convert labels to one-hot if needed
        let yTensor;
        if (y instanceof tf.Tensor) {
            yTensor = y;
        } else {
            yTensor = ModelUtils.oneHotEncode(y, this.config.numClasses);
        }
        
        // Initialize base prediction (mean of training labels)
        this.basePrediction = yTensor.mean(0);
        let currentPrediction = this.basePrediction.expandDims(0).tile([XTensor.shape[0], 1]);
        
        // Split validation data if needed
        let XTrain = XTensor;
        let yTrain = yTensor;
        let XVal = null;
        let yVal = null;
        
        if (validationSplit > 0) {
            const nVal = Math.floor(XTensor.shape[0] * validationSplit);
            const valIndices = tf.util.createShuffledIndices(XTensor.shape[0]).slice(0, nVal);
            const trainIndices = tf.util.createShuffledIndices(XTensor.shape[0]).slice(nVal);
            
            XVal = XTensor.gather(valIndices);
            yVal = yTensor.gather(valIndices);
            XTrain = XTensor.gather(trainIndices);
            yTrain = yTensor.gather(trainIndices);
        }
        
        // Boosting iterations
        const trainLosses = [];
        const valLosses = [];
        
        for (let i = 0; i < this.config.nEstimators; i++) {
            // Compute gradients and hessians
            const { gradients, hessians } = await this.computeGradients(yTrain, currentPrediction);
            
            // Subsample data
            const { XSampled, ySampled: gradSampled, colIndices } = 
                await this.subsampleData(XTrain, gradients);
            
            // Create and train weak learner
            const weakLearner = this.createWeakLearner(XSampled.shape[1], this.config.maxDepth);
            
            // Train on negative gradients (gradient boosting)
            await weakLearner.fit(XSampled, gradSampled.mul(tf.scalar(-1)), {
                epochs: 1,
                verbose: 0,
                batchSize: Math.min(256, XSampled.shape[0])
            });
            
            // Store estimator with column indices
            this.estimators.push({
                model: weakLearner,
                colIndices: colIndices,
                weight: this.config.learningRate
            });
            
            // Update predictions
            const weakPrediction = weakLearner.predict(XSampled);
            const fullPrediction = this.applyWeakLearner(XTrain, weakLearner, colIndices);
            
            const updatedPrediction = currentPrediction.add(
                fullPrediction.mul(tf.scalar(this.config.learningRate))
            );
            
            currentPrediction.dispose();
            currentPrediction = updatedPrediction;
            
            // Calculate training loss
            const trainLoss = tf.losses.softmaxCrossEntropy(yTrain, currentPrediction);
            const trainLossValue = await trainLoss.data();
            trainLosses.push(trainLossValue[0]);
            trainLoss.dispose();
            
            // Calculate validation loss if available
            if (XVal) {
                const valPred = await this.predict(XVal);
                const valLoss = tf.losses.softmaxCrossEntropy(yVal, valPred);
                const valLossValue = await valLoss.data();
                valLosses.push(valLossValue[0]);
                valLoss.dispose();
                valPred.dispose();
            }
            
            // Log progress
            if (i % 20 === 0) {
                const valInfo = XVal ? ` - Val Loss: ${valLosses[valLosses.length - 1].toFixed(4)}` : '';
                console.log(`Estimator ${i + 1}/${this.config.nEstimators} - Train Loss: ${trainLossValue[0].toFixed(4)}${valInfo}`);
            }
            
            // Clean up
            gradients.dispose();
            hessians.dispose();
            XSampled.dispose();
            gradSampled.dispose();
            weakPrediction.dispose();
            fullPrediction.dispose();
            
            // Early stopping check
            if (this.shouldEarlyStop(valLosses)) {
                console.log(`Early stopping at estimator ${i + 1}`);
                break;
            }
        }
        
        this.trained = true;
        
        // Calculate feature importances
        await this.calculateFeatureImportances(XTrain.shape[1]);
        
        // Clean up tensors
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        if (XVal) { XVal.dispose(); yVal.dispose(); }
        if (XTrain !== XTensor) { XTrain.dispose(); yTrain.dispose(); }
        currentPrediction.dispose();
        
        return { trainLosses, valLosses, nEstimators: this.estimators.length };
    }

    applyWeakLearner(X, weakLearner, colIndices) {
        // Apply weak learner to full dataset with column masking
        const XSubset = X.gather(colIndices, 1);
        const prediction = weakLearner.predict(XSubset);
        XSubset.dispose();
        return prediction;
    }

    async predict(X) {
        if (!this.trained || this.estimators.length === 0) {
            throw new Error('XGBoost model must be trained before prediction');
        }
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        
        // Start with base prediction
        let prediction = this.basePrediction.expandDims(0).tile([XTensor.shape[0], 1]);
        
        // Add predictions from all estimators
        for (const estimator of this.estimators) {
            const weakPred = this.applyWeakLearner(XTensor, estimator.model, estimator.colIndices);
            const weightedPred = weakPred.mul(tf.scalar(estimator.weight));
            
            const updatedPred = prediction.add(weightedPred);
            prediction.dispose();
            prediction = updatedPred;
            
            weakPred.dispose();
            weightedPred.dispose();
        }
        
        // Apply softmax for probabilities
        const probabilities = tf.softmax(prediction);
        
        // Clean up
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        prediction.dispose();
        
        return probabilities;
    }

    async predictClasses(X) {
        const probabilities = await this.predict(X);
        const classes = ModelUtils.argMax(probabilities);
        probabilities.dispose();
        return classes;
    }

    shouldEarlyStop(valLosses, patience = 10) {
        if (valLosses.length < patience * 2) return false;
        
        const recent = valLosses.slice(-patience);
        const earlier = valLosses.slice(-patience * 2, -patience);
        
        const recentMean = recent.reduce((a, b) => a + b) / recent.length;
        const earlierMean = earlier.reduce((a, b) => a + b) / earlier.length;
        
        return recentMean >= earlierMean;
    }

    async calculateFeatureImportances(nFeatures) {
        const importances = new Array(nFeatures).fill(0);
        
        // Calculate importance based on how often features are used
        for (const estimator of this.estimators) {
            const weight = estimator.weight / this.estimators.length;
            for (const colIndex of estimator.colIndices) {
                importances[colIndex] += weight;
            }
        }
        
        // Normalize
        const maxImportance = Math.max(...importances);
        this.featureImportances = importances.map(imp => imp / maxImportance);
        
        return this.featureImportances;
    }

    getFeatureImportances() {
        return this.featureImportances;
    }

    updateHyperparameters(newConfig) {
        this.updateConfig(newConfig);
        
        // Mark as untrained if significant parameters changed
        if (newConfig.nEstimators || newConfig.maxDepth || newConfig.learningRate) {
            this.trained = false;
            this.dispose();
            this.estimators = [];
        }
    }

    dispose() {
        // Dispose all weak learners
        this.estimators.forEach(estimator => {
            if (estimator.model) {
                estimator.model.dispose();
            }
        });
        
        if (this.basePrediction) {
            this.basePrediction.dispose();
        }
        
        this.estimators = [];
        this.featureImportances = null;
        
        super.dispose();
    }
}