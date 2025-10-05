import * as tf from '@tensorflow/tfjs';
import { BaseModel, ModelUtils } from './base.js';

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
        this.nColSample = null; // Store fixed number of subsampled features
    }

    createWeakLearner(inputSize, depth) {
        const layers = [];
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
        
        layers.push(tf.layers.dense({
            units: this.config.numClasses,
            activation: 'linear'
        }));
        
        const model = tf.sequential({ layers });
        
        model.compile({
            optimizer: tf.train.sgd(this.config.learningRate),
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    async subsampleData(X, y, fixedNColSample = null) {
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const yTensor = y instanceof tf.Tensor ? y : tf.tensor2d(y);
        
        const nSamples = XTensor.shape[0];
        const nFeatures = XTensor.shape[1];
        
        // Row subsampling
        const nRowSample = Math.floor(nSamples * this.config.subsample);
        const rowIndicesArray = tf.util.createShuffledIndices(nSamples);
        const rowIndicesFlat = Array.from(rowIndicesArray.slice(0, nRowSample));
        const rowIndices = tf.tensor1d(new Int32Array(rowIndicesFlat), 'int32');
        
        // Column subsampling
        const nColSample = fixedNColSample || Math.floor(nFeatures * this.config.colsampleByTree);
        const colIndicesArray = tf.util.createShuffledIndices(nFeatures);
        const colIndicesFlat = Array.from(colIndicesArray.slice(0, nColSample));
        const colIndices = tf.tensor1d(new Int32Array(colIndicesFlat), 'int32');
        // Create a persistent copy for storage in estimators
        const colIndicesStored = tf.tensor1d(new Int32Array(colIndicesFlat), 'int32');
        
        // Sample data
        const XSampled = XTensor.gather(rowIndices, 0).gather(colIndices, 1);
        const ySampled = yTensor.gather(rowIndices, 0);
        
        // Store nColSample for consistency
        if (!this.nColSample) {
            this.nColSample = nColSample;
        }
        
        // Clean up
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        rowIndices.dispose();
        colIndices.dispose();
        
        return { XSampled, ySampled, rowIndices, colIndices: colIndicesStored, nColSample };
    }

    async computeGradients(yTrue, yPred) {
        const softmaxPred = tf.softmax(yPred);
        const gradients = softmaxPred.sub(yTrue);
        const hessians = softmaxPred.mul(tf.scalar(1).sub(softmaxPred));
        
        return { gradients, hessians };
    }

    async fit(X, y, validationSplit = 0.1) {
        console.log(`Training XGBoost with ${this.config.nEstimators} estimators...`);
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const inputSize = XTensor.shape[1];
        
        let yTensor;
        if (y instanceof tf.Tensor) {
            yTensor = y;
        } else {
            yTensor = ModelUtils.oneHotEncode(y, this.config.numClasses);
        }
        
        let XTrain = XTensor;
        let yTrain = yTensor;
        let XVal = null;
        let yVal = null;
        let currentPrediction;
        
        if (validationSplit > 0) {
            const nVal = Math.floor(XTensor.shape[0] * validationSplit);
            const nTrain = XTensor.shape[0] - nVal;
            const indicesArray = tf.util.createShuffledIndices(XTensor.shape[0]);
            const valIndicesFlat = Array.from(indicesArray.slice(0, nVal));
            const valIndices = tf.tensor1d(new Int32Array(valIndicesFlat), 'int32');
            const trainIndicesFlat = Array.from(indicesArray.slice(nVal));
            const trainIndices = tf.tensor1d(new Int32Array(trainIndicesFlat), 'int32');
            
            XVal = XTensor.gather(valIndices, 0);
            yVal = yTensor.gather(valIndices, 0);
            XTrain = XTensor.gather(trainIndices, 0);
            yTrain = yTensor.gather(trainIndices, 0);
            
            // Initialize basePrediction and currentPrediction for training set
            this.basePrediction = yTrain.mean(0); // Shape: [3]
            currentPrediction = this.basePrediction.expandDims(0).tile([nTrain, 1]); // Shape: [nTrain, 3]
            
            valIndices.dispose();
            trainIndices.dispose();
        } else {
            // No validation split
            this.basePrediction = yTensor.mean(0); // Shape: [3]
            currentPrediction = this.basePrediction.expandDims(0).tile([XTensor.shape[0], 1]); // Shape: [nSamples, 3]
        }
        
        const trainLosses = [];
        const valLosses = [];
        
        for (let i = 0; i < this.config.nEstimators; i++) {
            const { gradients, hessians } = await this.computeGradients(yTrain, currentPrediction);
            const { XSampled, ySampled: gradSampled, colIndices, rowIndices, nColSample } = 
                await this.subsampleData(XTrain, gradients, this.nColSample);
            
            const weakLearner = this.createWeakLearner(nColSample, this.config.maxDepth);
            
            await weakLearner.fit(XSampled, gradSampled.mul(tf.scalar(-1)), {
                epochs: 1,
                verbose: 0,
                batchSize: Math.min(256, XSampled.shape[0])
            });
            
            this.estimators.push({
                model: weakLearner,
                colIndices,
                weight: this.config.learningRate
            });
            
            const weakPrediction = weakLearner.predict(XSampled);
            const fullPrediction = this.applyWeakLearner(XTrain, weakLearner, colIndices);
            
            const updatedPrediction = currentPrediction.add(
                fullPrediction.mul(tf.scalar(this.config.learningRate))
            );
            
            currentPrediction.dispose();
            currentPrediction = updatedPrediction;
            
            const trainLoss = tf.losses.softmaxCrossEntropy(yTrain, currentPrediction);
            const trainLossValue = await trainLoss.data();
            trainLosses.push(trainLossValue[0]);
            trainLoss.dispose();
            
            // Only compute validation loss after at least one weak learner is trained
            if (XVal && this.estimators.length > 0) {
                const valPred = await this.predict(XVal);
                const valLoss = tf.losses.softmaxCrossEntropy(yVal, valPred);
                const valLossValue = await valLoss.data();
                valLosses.push(valLossValue[0]);
                valLoss.dispose();
                valPred.dispose();
            }
            
            if (i % 20 === 0) {
                const valInfo = XVal && valLosses.length > 0 ? ` - Val Loss: ${valLosses[valLosses.length - 1].toFixed(4)}` : '';
                console.log(`Estimator ${i + 1}/${this.config.nEstimators} - Train Loss: ${trainLossValue[0].toFixed(4)}${valInfo}`);
            }
            
            gradients.dispose();
            hessians.dispose();
            XSampled.dispose();
            gradSampled.dispose();
            weakPrediction.dispose();
            fullPrediction.dispose();
            rowIndices.dispose();
            // Do not dispose colIndices here; it is stored in this.estimators
            if (this.shouldEarlyStop(valLosses)) {
                console.log(`Early stopping at estimator ${i + 1}`);
                break;
            }
        }
        
        this.trained = true;
        
        await this.calculateFeatureImportances(XTrain.shape[1]);
        
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        if (XVal) { XVal.dispose(); yVal.dispose(); }
        if (XTrain !== XTensor) { XTrain.dispose(); yTrain.dispose(); }
        currentPrediction.dispose();
        
        return { trainLosses, valLosses, nEstimators: this.estimators.length };
    }

    applyWeakLearner(X, weakLearner, colIndices) {
        // Debug: Verify colIndices is valid
        if (!colIndices || colIndices.isDisposed) {
            throw new Error('colIndices tensor is disposed or undefined');
        }
        const XSubset = X.gather(colIndices, 1);
        const prediction = weakLearner.predict(XSubset);
        XSubset.dispose();
        return prediction;
    }

    async predict(X) {
        if (!this.trained && this.estimators.length === 0) {
            throw new Error('XGBoost model must be trained before prediction');
        }
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        let prediction = this.basePrediction.expandDims(0).tile([XTensor.shape[0], 1]);
        
        for (const estimator of this.estimators) {
            const weakPred = this.applyWeakLearner(XTensor, estimator.model, estimator.colIndices);
            const weightedPred = weakPred.mul(tf.scalar(estimator.weight));
            
            const updatedPred = prediction.add(weightedPred);
            prediction.dispose();
            prediction = updatedPred;
            
            weakPred.dispose();
            weightedPred.dispose();
        }
        
        const probabilities = tf.softmax(prediction);
        
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
        
        for (const estimator of this.estimators) {
            const weight = estimator.weight / this.estimators.length;
            for (const colIndex of estimator.colIndices.dataSync()) {
                importances[colIndex] += weight;
            }
        }
        
        const maxImportance = Math.max(...importances);
        this.featureImportances = importances.map(imp => imp / maxImportance);
        
        return this.featureImportances;
    }

    getFeatureImportances() {
        return this.featureImportances;
    }

    updateHyperparameters(newConfig) {
        this.updateConfig(newConfig);
        
        if (newConfig.nEstimators || newConfig.maxDepth || newConfig.learningRate) {
            this.trained = false;
            this.dispose();
            this.estimators = [];
        }
    }

    dispose() {
        this.estimators.forEach(estimator => {
            if (estimator.model) {
                estimator.model.dispose();
            }
            if (estimator.colIndices && !estimator.colIndices.isDisposed) {
                estimator.colIndices.dispose();
            }
        });
        
        if (this.basePrediction && !this.basePrediction.isDisposed) {
            this.basePrediction.dispose();
        }
        
        this.estimators = [];
        this.featureImportances = null;
        this.nColSample = null;
        
        super.dispose();
    }
}