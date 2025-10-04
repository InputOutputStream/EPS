import * as tf from '@tensorflow/tfjs';
import { BaseModel, ModelUtils } from './base.js';

/**
 * Random Forest implementation using ensemble of decision tree-like neural networks
 */
export class RandomForestModel extends BaseModel {
    constructor(config = {}) {
        const defaultConfig = {
            nEstimators: 100,
            maxDepth: 10,
            minSamplesSplit: 2,
            minSamplesLeaf: 1,
            maxFeatures: 'sqrt', // 'sqrt', 'log2', number, or null
            bootstrap: true,
            randomState: null,
            numClasses: 3,
            subsampleRatio: 0.8
        };
        
        super('RandomForest', { ...defaultConfig, ...config });
        this.trees = [];
        this.featureImportances = null;
        this.outOfBagScore = null;
    }

    createDecisionTree(inputSize, maxDepth) {
        // Create a neural network that mimics decision tree behavior
        const layers = [];
        
        // Calculate tree structure
        const branchingFactor = 2;
        let currentWidth = inputSize;
        
        // Create tree-like structure with decreasing layer sizes
        for (let depth = 0; depth < maxDepth; depth++) {
            const layerWidth = Math.max(
                this.config.numClasses,
                Math.floor(currentWidth / branchingFactor)
            );
            
            layers.push(tf.layers.dense({
                units: layerWidth,
                activation: depth === maxDepth - 1 ? 'linear' : 'relu',
                inputShape: depth === 0 ? [inputSize] : undefined,
                kernelInitializer: 'glorotNormal',
                biasInitializer: 'zeros',
                name: `tree_layer_${depth}`
            }));
            
            // Add dropout for regularization (simulates random splits)
            if (depth < maxDepth - 1) {
                layers.push(tf.layers.dropout({
                    rate: 0.1,
                    name: `tree_dropout_${depth}`
                }));
            }
            
            currentWidth = layerWidth;
            
            if (layerWidth <= this.config.numClasses) break;
        }
        
        // Ensure output layer has correct size
        if (layers[layers.length - 1].units !== this.config.numClasses) {
            layers.push(tf.layers.dense({
                units: this.config.numClasses,
                activation: 'softmax',
                name: 'tree_output'
            }));
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
        } else {
            return totalFeatures; // Use all features
        }
    }

    async bootstrapSample(X, y) {
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const yTensor = y instanceof tf.Tensor ? y : tf.tensor2d(y);
        
        const nSamples = XTensor.shape[0];
        const nFeatures = XTensor.shape[1];
        
        // Bootstrap sampling (sampling with replacement)
        let sampleIndices;
        if (this.config.bootstrap) {
            sampleIndices = [];
            const sampleSize = Math.floor(nSamples * this.config.subsampleRatio);
            for (let i = 0; i < sampleSize; i++) {
                sampleIndices.push(Math.floor(Math.random() * nSamples));
            }
        } else {
            // Use all samples without replacement
            sampleIndices = tf.util.createShuffledIndices(nSamples);
        }
        
        // Feature subsampling
        const maxFeatures = this.calculateMaxFeatures(nFeatures);
        const featureIndices = tf.util.createShuffledIndices(nFeatures).slice(0, maxFeatures);
        
        // Sample data
        const XBootstrap = XTensor.gather(sampleIndices).gather(featureIndices, 1);
        const yBootstrap = yTensor.gather(sampleIndices);
        
        // Out-of-bag samples for evaluation
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
        
        // Clean up if we created tensors
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        
        return { 
            XBootstrap, 
            yBootstrap, 
            featureIndices, 
            oobIndices,
            sampleIndices 
        };
    }

    async fit(X, y, validationSplit = 0.0) {
        console.log(`Training Random Forest with ${this.config.nEstimators} trees...`);
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const inputSize = XTensor.shape[1];
        
        // Convert labels to one-hot if needed
        let yTensor;
        if (y instanceof tf.Tensor) {
            yTensor = y;
        } else {
            yTensor = ModelUtils.oneHotEncode(y, this.config.numClasses);
        }
        
        this.trees = [];
        const oobPredictions = [];
        const oobIndicesAll = [];
        
        // Train trees in parallel batches to avoid memory issues
        const batchSize = Math.min(10, this.config.nEstimators);
        const numBatches = Math.ceil(this.config.nEstimators / batchSize);
        
        for (let batch = 0; batch < numBatches; batch++) {
            const batchStart = batch * batchSize;
            const batchEnd = Math.min((batch + 1) * batchSize, this.config.nEstimators);
            const batchPromises = [];
            
            for (let i = batchStart; i < batchEnd; i++) {
                batchPromises.push(this.trainSingleTree(XTensor, yTensor, i));
            }
            
            const batchResults = await Promise.all(batchPromises);
            
            // Collect results
            for (const result of batchResults) {
                if (result.success) {
                    this.trees.push(result.tree);
                    if (result.oobPredictions) {
                        oobPredictions.push(result.oobPredictions);
                        oobIndicesAll.push(result.oobIndices);
                    }
                } else {
                    console.warn(`Tree ${result.index} training failed:`, result.error);
                }
            }
            
            console.log(`Batch ${batch + 1}/${numBatches} completed. Trees trained: ${this.trees.length}`);
        }
        
        // Calculate out-of-bag score
        if (oobPredictions.length > 0) {
            this.outOfBagScore = await this.calculateOOBScore(
                oobPredictions, oobIndicesAll, yTensor
            );
            console.log(`Out-of-bag accuracy: ${this.outOfBagScore.toFixed(4)}`);
        }
        
        // Calculate feature importances
        await this.calculateFeatureImportances(inputSize);
        
        this.trained = true;
        
        // Clean up
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        
        return {
            nTrees: this.trees.length,
            oobScore: this.outOfBagScore,
            featureImportances: this.featureImportances
        };
    }

    async trainSingleTree(X, y, treeIndex) {
        try {
            // Bootstrap sample
            const { XBootstrap, yBootstrap, featureIndices, oobIndices } = 
                await this.bootstrapSample(X, y);
            
            // Create and train tree
            const tree = this.createDecisionTree(XBootstrap.shape[1], this.config.maxDepth);
            
            await tree.fit(XBootstrap, yBootstrap, {
                epochs: 50,
                batchSize: Math.min(32, XBootstrap.shape[0]),
                verbose: 0,
                shuffle: true
            });
            
            // Out-of-bag predictions
            let oobPredictions = null;
            if (oobIndices && oobIndices.length > 0) {
                const XOob = X.gather(oobIndices).gather(featureIndices, 1);
                oobPredictions = await tree.predict(XOob);
                XOob.dispose();
            }
            
            // Clean up bootstrap data
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
        if (!this.trained || this.trees.length === 0) {
            throw new Error('Random Forest must be trained before prediction');
        }
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const nSamples = XTensor.shape[0];
        
        // Collect predictions from all trees
        const treePredictions = [];
        
        for (const tree of this.trees) {
            try {
                const XSubset = XTensor.gather(tree.featureIndices, 1);
                const prediction = await tree.model.predict(XSubset);
                treePredictions.push(prediction);
                XSubset.dispose();
            } catch (error) {
                console.warn('Error in tree prediction:', error);
            }
        }
        
        if (treePredictions.length === 0) {
            throw new Error('No valid tree predictions available');
        }
        
        // Average predictions across all trees
        let sumPredictions = treePredictions[0];
        
        for (let i = 1; i < treePredictions.length; i++) {
            const temp = sumPredictions.add(treePredictions[i]);
            if (i > 1) sumPredictions.dispose();
            sumPredictions = temp;
        }
        
        const avgPredictions = sumPredictions.div(tf.scalar(treePredictions.length));
        
        // Clean up
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
        // Aggregate out-of-bag predictions
        const nSamples = yTrue.shape[0];
        const nClasses = yTrue.shape[1];
        
        // Initialize aggregation arrays
        const oobVotes = new Array(nSamples).fill(null).map(() => 
            new Array(nClasses).fill(0)
        );
        const oobCounts = new Array(nSamples).fill(0);
        
        // Aggregate votes from each tree
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
        }
        
        // Calculate accuracy for samples with OOB predictions
        let correct = 0;
        let total = 0;
        const yTrueData = await yTrue.data();
        
        for (let i = 0; i < nSamples; i++) {
            if (oobCounts[i] > 0) {
                // Average the votes
                const avgVotes = oobVotes[i].map(vote => vote / oobCounts[i]);
                const predictedClass = avgVotes.indexOf(Math.max(...avgVotes));
                
                // Get true class
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
        
        // Calculate importance based on feature usage and tree performance
        for (const tree of this.trees) {
            const weight = 1.0 / this.trees.length;
            
            for (const featureIdx of tree.featureIndices) {
                importances[featureIdx] += weight;
                featureUsageCounts[featureIdx]++;
            }
        }
        
        // Normalize by usage frequency
        for (let i = 0; i < nFeatures; i++) {
            if (featureUsageCounts[i] > 0) {
                importances[i] /= featureUsageCounts[i];
            }
        }
        
        // Final normalization
        const maxImportance = Math.max(...importances);
        if (maxImportance > 0) {
            this.featureImportances = importances.map(imp => imp / maxImportance);
        } else {
            this.featureImportances = importances;
        }
        
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
        
        // Mark as untrained if significant parameters changed
        const significantParams = [
            'nEstimators', 'maxDepth', 'minSamplesSplit', 
            'maxFeatures', 'bootstrap'
        ];
        
        const hasSignificantChange = significantParams.some(param => 
            newConfig.hasOwnProperty(param)
        );
        
        if (hasSignificantChange) {
            this.trained = false;
            this.dispose();
            this.trees = [];
        }
    }

    async partialFit(X, y, nNewTrees = 10) {
        // Add new trees to existing forest
        if (!this.trained) {
            return await this.fit(X, y);
        }
        
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
        
        // Recalculate feature importances
        await this.calculateFeatureImportances(XTensor.shape[1]);
        
        console.log(`Added ${this.trees.length - originalTreeCount} trees. Total: ${this.trees.length}`);
        
        // Clean up
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        
        return {
            newTrees: this.trees.length - originalTreeCount,
            totalTrees: this.trees.length
        };
    }

    dispose() {
        // Dispose all tree models
        this.trees.forEach(tree => {
            if (tree.model) {
                tree.model.dispose();
            }
        });
        
        this.trees = [];
        this.featureImportances = null;
        this.outOfBagScore = null;
        
        super.dispose();
    }
}