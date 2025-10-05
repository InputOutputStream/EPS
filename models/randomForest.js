import * as tf from '@tensorflow/tfjs';
import { BaseModel, ModelUtils } from './base.js';

export class RandomForestModel extends BaseModel {
    constructor(config = {}) {
        const defaultConfig = {
            nEstimators: 100,
            maxDepth: 5,
            minSamplesSplit: 2,
            minSamplesLeaf: 1,
            maxFeatures: 'sqrt',
            bootstrap: true,
            randomState: null,
            numClasses: 3,
            subsampleRatio: 0.5 // Reduced from 0.8
        };
        
        super('RandomForest', { ...defaultConfig, ...config });
        this.trees = [];
        this.featureImportances = null;
        this.outOfBagScore = null;
        this.nMaxFeatures = null;
    }

    createDecisionTree(inputSize, maxDepth) {
        const layers = [];
        const branchingFactor = 2;
        let currentWidth = inputSize;
        
        // Simplified network structure
        for (let depth = 0; depth < maxDepth; depth++) {
            const layerWidth = Math.max(
                this.config.numClasses,
                Math.floor(currentWidth / branchingFactor)
            );
            
            layers.push(tf.layers.dense({
                units: layerWidth,
                activation: depth === maxDepth - 1 ? 'softmax' : 'relu',
                inputShape: depth === 0 ? [inputSize] : undefined,
                kernelInitializer: 'glorotNormal',
                biasInitializer: 'zeros',
                name: `tree_layer_${depth}`
            }));
            
            currentWidth = layerWidth;
            
            if (layerWidth <= this.config.numClasses) break;
        }
        
        const model = tf.sequential({ layers });
        
        model.compile({
            optimizer: tf.train.adam(0.01),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    calculateMaxFeatures(totalFeatures) {
        if (this.config.maxFeatures === 'sqrt') {
            return Math.floor(Math.sqrt(totalFeatures));
        } else if (this.config.maxFeatures === 'log2') {
            return Math.floor(Math.log2(totalFeatures));
        } else if (typeof this.config.maxFeatures === 'number') {
            return Math.min(this.config.maxFeatures, totalFeatures);
        }
        return totalFeatures;
    }

    async bootstrapSample(X, y, fixedMaxFeatures = null) {
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const yTensor = y instanceof tf.Tensor ? y : tf.tensor2d(y);
        
        const nSamples = XTensor.shape[0];
        const nFeatures = XTensor.shape[1];
        
        let sampleIndices;
        if (this.config.bootstrap) {
            sampleIndices = [];
            const sampleSize = Math.floor(nSamples * this.config.subsampleRatio);
            for (let i = 0; i < sampleSize; i++) {
                sampleIndices.push(Math.floor(Math.random() * nSamples));
            }
        } else {
            sampleIndices = Array.from(tf.util.createShuffledIndices(nSamples));
        }
        
        const maxFeatures = fixedMaxFeatures || this.calculateMaxFeatures(nFeatures);
        const featureIndicesArray = tf.util.createShuffledIndices(nFeatures);
        const featureIndicesFlat = Array.from(featureIndicesArray.slice(0, maxFeatures));
        const sampleIndicesTensor = tf.tensor1d(new Int32Array(sampleIndices), 'int32');
        const featureIndicesTensor = tf.tensor1d(new Int32Array(featureIndicesFlat), 'int32');
        const featureIndicesStored = tf.tensor1d(new Int32Array(featureIndicesFlat), 'int32');
        
        const XBootstrap = XTensor.gather(sampleIndicesTensor).gather(featureIndicesTensor, 1);
        const yBootstrap = yTensor.gather(sampleIndicesTensor);
        
        let oobIndices = null;
        if (this.config.bootstrap) {
            const usedIndices = new Set(sampleIndices);
            oobIndices = [];
            for (let i = 0; i < nSamples; i++) {
                if (!usedIndices.has(i)) {
                    oobIndices.push(i);
                }
            }
        }
        
        if (!this.nMaxFeatures) {
            this.nMaxFeatures = maxFeatures;
        }
        
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        sampleIndicesTensor.dispose();
        featureIndicesTensor.dispose();
        
        return { 
            XBootstrap, 
            yBootstrap, 
            featureIndices: featureIndicesStored, 
            oobIndices,
            sampleIndices,
            maxFeatures
        };
    }

    async fit(X, y, validationSplit = 0.1) {
        console.log(`Training Random Forest with ${this.config.nEstimators} trees...`);
        
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
            
            valIndices.dispose();
            trainIndices.dispose();
        }
        
        this.trees = [];
        const oobPredictions = [];
        const oobIndicesAll = [];
        const trainLosses = [];
        const valLosses = [];
        
        // Increased batch size for more parallelism
        const batchSize = Math.min(20, this.config.nEstimators);
        const numBatches = Math.ceil(this.config.nEstimators / batchSize);
        
        for (let batch = 0; batch < numBatches; batch++) {
            const batchStart = batch * batchSize;
            const batchEnd = Math.min((batch + 1) * batchSize, this.config.nEstimators);
            const batchPromises = [];
            
            for (let i = batchStart; i < batchEnd; i++) {
                batchPromises.push(this.trainSingleTree(XTrain, yTrain, i));
            }
            
            const batchResults = await Promise.all(batchPromises);
            
            for (const result of batchResults) {
                if (result.success) {
                    console.log(`Trained Tree ${result.index + 1}`);
                    this.trees.push(result.tree);
                    if (result.oobPredictions) {
                        oobPredictions.push(result.oobPredictions);
                        oobIndicesAll.push(result.oobIndices);
                    }
                } else {
                    console.warn(`Tree ${result.index + 1} training failed:`, result.error);
                }
            }
            
            // Compute loss every few batches to reduce overhead
            if (this.trees.length > 0 && (batch + 1) % Math.ceil(numBatches / 5) === 0) {
                const trainPred = await this.predict(XTrain);
                const trainLoss = tf.losses.softmaxCrossEntropy(yTrain, trainPred);
                const trainLossValue = await trainLoss.data();
                trainLosses.push(trainLossValue[0]);
                trainLoss.dispose();
                trainPred.dispose();
                
                if (XVal) {
                    const valPred = await this.predict(XVal);
                    const valLoss = tf.losses.softmaxCrossEntropy(yVal, valPred);
                    const valLossValue = await valLoss.data();
                    valLosses.push(valLossValue[0]);
                    valLoss.dispose();
                    valPred.dispose();
                }
                
                console.log(`Batch ${batch + 1}/${numBatches} - Train Loss: ${trainLosses[trainLosses.length - 1].toFixed(4)}${XVal && valLosses.length > 0 ? ` - Val Loss: ${valLosses[valLosses.length - 1].toFixed(4)}` : ''}`);
            }
        }
        
        if (oobPredictions.length > 0) {
            this.outOfBagScore = await this.calculateOOBScore(oobPredictions, oobIndicesAll, yTrain);
            console.log(`Out-of-bag accuracy: ${this.outOfBagScore.toFixed(4)}`);
        }
        
        await this.calculateFeatureImportances(inputSize);
        
        this.trained = true;
        
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        if (XVal) { XVal.dispose(); yVal.dispose(); }
        if (XTrain !== XTensor) { XTrain.dispose(); yTrain.dispose(); }
        
        return {
            nTrees: this.trees.length,
            oobScore: this.outOfBagScore,
            featureImportances: this.featureImportances,
            trainLosses,
            valLosses
        };
    }

    async trainSingleTree(X, y, treeIndex) {
        try {
            const { XBootstrap, yBootstrap, featureIndices, oobIndices, maxFeatures } = 
                await this.bootstrapSample(X, y, this.nMaxFeatures);
            
            const tree = this.createDecisionTree(maxFeatures, this.config.maxDepth);
            
            // Reduced epochs
            await tree.fit(XBootstrap, yBootstrap, {
                epochs: 10, // Reduced from 50
                batchSize: Math.min(64, XBootstrap.shape[0]), // Increased batch size
                verbose: 0,
                shuffle: true
            });
            
            let oobPredictions = null;
            if (oobIndices && oobIndices.length > 0) {
                const oobIndicesTensor = tf.tensor1d(new Int32Array(oobIndices), 'int32');
                const XOob = X.gather(oobIndicesTensor).gather(featureIndices, 1);
                oobPredictions = await tree.predict(XOob);
                XOob.dispose();
                oobIndicesTensor.dispose();
            }
            
            XBootstrap.dispose();
            yBootstrap.dispose();
            
            return {
                success: true,
                tree: { model: tree, featureIndices },
                oobPredictions,
                oobIndices,
                index: treeIndex
            };
        } catch (error) {
            return {
                success: false,
                error: error.message,
                index: treeIndex
            };
        }
    }

    async predict(X) {
        if (!this.trained && this.trees.length === 0) {
            throw new Error('Random Forest must be trained before prediction');
        }
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const treePredictions = [];
        
        for (const tree of this.trees) {
            try {
                const featureIndicesTensor = tree.featureIndices;
                if (!featureIndicesTensor || featureIndicesTensor.isDisposed) {
                    throw new Error(`featureIndices tensor for tree ${this.trees.indexOf(tree)} is invalid`);
                }
                const XSubset = XTensor.gather(featureIndicesTensor, 1);
                const prediction = await tree.model.predict(XSubset);
                treePredictions.push(prediction);
                XSubset.dispose();
            } catch (error) {
                console.warn(`Error in tree ${this.trees.indexOf(tree)} prediction:`, error);
            }
        }
        
        if (treePredictions.length === 0) {
            throw new Error('No valid tree predictions available');
        }
        
        let sumPredictions = treePredictions[0];
        for (let i = 1; i < treePredictions.length; i++) {
            const temp = sumPredictions.add(treePredictions[i]);
            if (i > 1) sumPredictions.dispose();
            sumPredictions = temp;
        }
        
        const avgPredictions = sumPredictions.div(tf.scalar(treePredictions.length));
        
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        treePredictions.forEach((pred, i) => {
            if (i > 0) pred.dispose();
        });
        sumPredictions.dispose();
        
        return avgPredictions;
    }

    async predictClasses(X) {
        const probabilities = await this.predict(X);
        const classes = ModelUtils.argMax(probabilities);
        probabilities.dispose();
        return classes;
    }

    async calculateOOBScore(oobPredictions, oobIndicesAll, yTrue) {
        const nSamples = yTrue.shape[0];
        const nClasses = yTrue.shape[1];
        
        const oobVotes = new Array(nSamples).fill(null).map(() => 
            new Array(nClasses).fill(0)
        );
        const oobCounts = new Array(nSamples).fill(0);
        
        for (let treeIdx = 0; treeIdx < oobPredictions.length; treeIdx++) {
            const predictions = await oobPredictions[treeIdx].data();
            const indices = oobIndicesAll[treeIdx];
            
            for (let i = 0; i < indices.length; i++) {
                const sampleIdx = indices[i];
                for (let classIdx = 0; classIdx < nClasses; classIdx++) {
                    oobVotes[sampleIdx][classIdx] += predictions[i * nClasses + classIdx];
                }
                oobCounts[sampleIdx]++;
            }
            oobPredictions[treeIdx].dispose();
        }
        
        let correct = 0;
        let total = 0;
        const yTrueData = await yTrue.data();
        
        for (let i = 0; i < nSamples; i++) {
            if (oobCounts[i] > 0) {
                const avgVotes = oobVotes[i].map(vote => vote / oobCounts[i]);
                const predictedClass = avgVotes.indexOf(Math.max(...avgVotes));
                
                let trueClass = 0;
                for (let j = 0; j < nClasses; j++) {
                    if (yTrueData[i * nClasses + j] === 1) {
                        trueClass = j;
                        break;
                    }
                }
                
                if (predictedClass === trueClass) correct++;
                total++;
            }
        }
        
        return total > 0 ? correct / total : 0;
    }

    async calculateFeatureImportances(nFeatures) {
        const importances = new Array(nFeatures).fill(0);
        const featureUsageCounts = new Array(nFeatures).fill(0);
        
        for (const tree of this.trees) {
            const weight = 1.0 / this.trees.length;
            for (const featureIdx of tree.featureIndices.dataSync()) {
                importances[featureIdx] += weight;
                featureUsageCounts[featureIdx]++;
            }
        }
        
        for (let i = 0; i < nFeatures; i++) {
            if (featureUsageCounts[i] > 0) {
                importances[i] /= featureUsageCounts[i];
            }
        }
        
        const maxImportance = Math.max(...importances);
        this.featureImportances = maxImportance > 0 ? importances.map(imp => imp / maxImportance) : importances;
        
        return this.featureImportances;
    }

    getFeatureImportances() {
        return this.featureImportances;
    }

    getOutOfBagScore() {
        return this.outOfBagScore;
    }

    getTreeCount() {
        return this.trees.length;
    }

    updateHyperparameters(newConfig) {
        this.updateConfig(newConfig);
        
        const significantParams = [
            'nEstimators', 'maxDepth', 'minSamplesSplit', 
            'maxFeatures', 'bootstrap'
        ];
        
        if (significantParams.some(param => newConfig.hasOwnProperty(param))) {
            this.trained = false;
            this.dispose();
            this.trees = [];
            this.nMaxFeatures = null;
        }
    }

    async partialFit(X, y, nNewTrees = 10) {
        console.log(`Adding ${nNewTrees} new trees to existing forest...`);
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        let yTensor;
        if (y instanceof tf.Tensor) {
            yTensor = y;
        } else {
            yTensor = ModelUtils.oneHotEncode(y, this.config.numClasses);
        }
        
        const originalTreeCount = this.trees.length;
        
        for (let i = 0; i < nNewTrees; i++) {
            const result = await this.trainSingleTree(XTensor, yTensor, originalTreeCount + i);
            if (result.success) {
                this.trees.push(result.tree);
            }
        }
        
        await this.calculateFeatureImportances(XTensor.shape[1]);
        
        console.log(`Added ${this.trees.length - originalTreeCount} trees. Total: ${this.trees.length}`);
        
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        
        return {
            newTrees: this.trees.length - originalTreeCount,
            totalTrees: this.trees.length
        };
    }

    dispose() {
        this.trees.forEach(tree => {
            if (tree.model) {
                tree.model.dispose();
            }
            if (tree.featureIndices && !tree.featureIndices.isDisposed) {
                tree.featureIndices.dispose();
            }
        });
        
        this.trees = [];
        this.featureImportances = null;
        this.outOfBagScore = null;
        this.nMaxFeatures = null;
        
        super.dispose();
    }
}