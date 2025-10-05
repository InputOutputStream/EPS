import * as tf from '@tensorflow/tfjs';
import { BaseModel, ModelUtils } from './base.js';

export class EnsembleModel extends BaseModel {
    constructor(baseModels = [], metaLearner = null, config = {}) {
        const defaultConfig = {
            stackingMethod: 'average', // 'average', 'weighted', 'meta'
            weights: null, // for weighted averaging
            metaLearnerConfig: {
                hiddenLayers: [16, 8],
                activation: 'relu',
                dropout: 0.1,
                learningRate: 0.01,
                epochs: 50
            }
        };
        
        super('Ensemble', { ...defaultConfig, ...config });
        
        this.baseModels = baseModels;
        this.metaLearner = metaLearner;
        this.baseModelPredictions = [];
        this.stackingTrained = false;
    }

    addBaseModel(model) {
        if (!(model instanceof BaseModel)) {
            throw new Error('Base model must extend BaseModel class');
        }
        this.baseModels.push(model);
    }

    removeBaseModel(modelName) {
        this.baseModels = this.baseModels.filter(model => model.name !== modelName);
    }

    async fitBaseModels(X, y, validationSplit = 0.2) {
        console.log(`Training ${this.baseModels.length} base models...`);
        
        const trainingPromises = this.baseModels.map(async (model, index) => {
            console.log(`Training ${model.name}...`);
            try {
                await model.fit(X, y, validationSplit);
                console.log(`${model.name} training completed`);
                return { success: true, modelIndex: index, model: model.name };
            } catch (error) {
                console.error(`Error training ${model.name}:`, error);
                return { success: false, modelIndex: index, model: model.name, error };
            }
        });

        const results = await Promise.all(trainingPromises);
        
        // Filter out failed models
        const successfulModels = results
            .filter(result => result.success)
            .map(result => result.modelIndex);
        
        this.baseModels = this.baseModels.filter((_, index) => 
            successfulModels.includes(index)
        );

        console.log(`Successfully trained ${this.baseModels.length} base models`);
        return results;
    }

    async generateMetaFeatures(X) {
        if (this.baseModels.length === 0) {
            throw new Error('No base models available for meta-feature generation');
        }

        const predictions = await Promise.all(
            this.baseModels.map(async (model) => {
                const pred = await model.predict(X);
                return pred;
            })
        );

        // Stack predictions horizontally
        const metaFeatures = tf.concat(predictions, 1);
        
        // Clean up individual predictions
        predictions.forEach(pred => pred.dispose());
        
        return metaFeatures;
    }

    async trainMetaLearner(X, y, validationSplit = 0.2) {
        if (this.config.stackingMethod !== 'meta') {
            console.log('Skipping meta-learner training (not using meta stacking)');
            return;
        }

        console.log('Generating meta-features for stacking...');
        const metaFeatures = await this.generateMetaFeatures(X);
        
        // Create meta-learner if not provided
        if (!this.metaLearner) {
            const { MLPModel } = await import('./mlp.js');
            this.metaLearner = new MLPModel({
                ...this.config.metaLearnerConfig,
                numClasses: 3 // 3-class classification
            });
        }

        console.log('Training meta-learner...');
        await this.metaLearner.fit(metaFeatures, y, validationSplit);
        
        this.stackingTrained = true;
        metaFeatures.dispose();
        
        console.log('Meta-learner training completed');
    }

    async fit(X, y, validationSplit = 0.2) {
        // Train base models
        await this.fitBaseModels(X, y, validationSplit);
        
        // Train meta-learner if using meta stacking
        if (this.config.stackingMethod === 'meta') {
            await this.trainMetaLearner(X, y, validationSplit);
        }
        
        this.trained = true;
        return { baseModels: this.baseModels.length, stackingTrained: this.stackingTrained };
    }

    async predict(X) {
        if (!this.trained) {
            throw new Error('Ensemble model must be trained before prediction');
        }

        if (this.baseModels.length === 0) {
            throw new Error('No trained base models available');
        }

        // Get predictions from all base models
        const basePredictions = await Promise.all(
            this.baseModels.map(async (model) => {
                return await model.predict(X);
            })
        );

        let finalPredictions;

        switch (this.config.stackingMethod) {
            case 'average':
                finalPredictions = this.averagePredictions(basePredictions);
                break;
                
            case 'weighted':
                finalPredictions = this.weightedAveragePredictions(basePredictions);
                break;
                
            case 'meta':
                if (!this.stackingTrained || !this.metaLearner) {
                    console.warn('Meta-learner not trained, falling back to averaging');
                    finalPredictions = this.averagePredictions(basePredictions);
                } else {
                    const metaFeatures = tf.concat(basePredictions, 1);
                    finalPredictions = await this.metaLearner.predict(metaFeatures);
                    metaFeatures.dispose();
                }
                break;
                
            default:
                finalPredictions = this.averagePredictions(basePredictions);
        }

        // Store base predictions for analysis
        this.baseModelPredictions = basePredictions.map((pred, index) => ({
            model: this.baseModels[index].name,
            predictions: pred
        }));

        // Clean up base predictions (except if stored for analysis)
        // basePredictions.forEach(pred => pred.dispose());

        return finalPredictions;
    }

    averagePredictions(predictions) {
        if (predictions.length === 1) {
            return predictions[0];
        }
        
        // Stack and average
        const stacked = tf.stack(predictions);
        const averaged = tf.mean(stacked, 0);
        
        stacked.dispose();
        return averaged;
    }

    weightedAveragePredictions(predictions) {
        if (!this.config.weights) {
            // Use model performance as weights
            const weights = this.baseModels.map(model => 
                model.getMetrics().accuracy || 1.0 / this.baseModels.length
            );
            this.config.weights = weights;
        }

        if (this.config.weights.length !== predictions.length) {
            console.warn('Weights length mismatch, using equal weights');
            return this.averagePredictions(predictions);
        }

        // Normalize weights
        const weightSum = this.config.weights.reduce((a, b) => a + b, 0);
        const normalizedWeights = this.config.weights.map(w => w / weightSum);

        // Apply weights
        let weightedSum = null;
        predictions.forEach((pred, index) => {
            const weighted = pred.mul(tf.scalar(normalizedWeights[index]));
            
            if (weightedSum === null) {
                weightedSum = weighted;
            } else {
                const temp = weightedSum.add(weighted);
                weightedSum.dispose();
                weightedSum = temp;
                weighted.dispose();
            }
        });

        return weightedSum;
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

    getBaseModelPredictions() {
        return this.baseModelPredictions;
    }

    getModelWeights() {
        return this.config.weights;
    }

    updateStackingMethod(method, weights = null) {
        this.config.stackingMethod = method;
        if (weights) {
            this.config.weights = weights;
        }
    }

    async evaluateBaseModels(X, y) {
        const results = {};
        
        for (const model of this.baseModels) {
            try {
                const metrics = await model.evaluate(X, y);
                results[model.name] = metrics;
            } catch (error) {
                console.error(`Error evaluating ${model.name}:`, error);
                results[model.name] = { error: error.message };
            }
        }
        
        return results;
    }

    async crossValidateEnsemble(X, y, folds = 5) {
        console.log(`Performing ${folds}-fold cross-validation...`);
        
        const XTensor = X instanceof tf.Tensor ? X : tf.tensor2d(X);
        const yTensor = y instanceof tf.Tensor ? y : tf.tensor1d(y);
        
        const n = XTensor.shape[0];
        const foldSize = Math.floor(n / folds);
        
        const cvResults = [];
        
        for (let fold = 0; fold < folds; fold++) {
            console.log(`Fold ${fold + 1}/${folds}`);
            
            // Create train/validation split
            const valStart = fold * foldSize;
            const valEnd = fold === folds - 1 ? n : (fold + 1) * foldSize;
            
            const trainIndices = [];
            const valIndices = [];
            
            for (let i = 0; i < n; i++) {
                if (i >= valStart && i < valEnd) {
                    valIndices.push(i);
                } else {
                    trainIndices.push(i);
                }
            }
            
            const XTrain = XTensor.gather(trainIndices);
            const yTrain = yTensor.gather(trainIndices);
            const XVal = XTensor.gather(valIndices);
            const yVal = yTensor.gather(valIndices);
            
            // Train ensemble on fold
            const foldEnsemble = new EnsembleModel(
                this.baseModels.map(model => Object.create(Object.getPrototypeOf(model))),
                null,
                this.config
            );
            
            try {
                await foldEnsemble.fit(XTrain, yTrain, 0.0);
                const metrics = await foldEnsemble.evaluate(XVal, yVal);
                cvResults.push(metrics);
            } catch (error) {
                console.error(`Error in fold ${fold + 1}:`, error);
                cvResults.push({ error: error.message });
            }
            
            // Clean up fold tensors
            XTrain.dispose();
            yTrain.dispose();
            XVal.dispose();
            yVal.dispose();
            foldEnsemble.dispose();
        }
        
        // Calculate average metrics
        const validResults = cvResults.filter(r => !r.error);
        const avgMetrics = {};
        
        if (validResults.length > 0) {
            ['accuracy', 'precision', 'recall', 'f1'].forEach(metric => {
                const values = validResults.map(r => r[metric]).filter(v => v !== undefined);
                if (values.length > 0) {
                    avgMetrics[metric] = {
                        mean: values.reduce((a, b) => a + b) / values.length,
                        std: Math.sqrt(values.map(v => Math.pow(v - avgMetrics[metric]?.mean || 0, 2))
                                     .reduce((a, b) => a + b) / values.length)
                    };
                }
            });
        }
        
        // Clean up tensors
        if (!(X instanceof tf.Tensor)) XTensor.dispose();
        if (!(y instanceof tf.Tensor)) yTensor.dispose();
        
        return { foldResults: cvResults, averageMetrics: avgMetrics };
    }

    getEnsembleInfo() {
        return {
            name: this.name,
            numBaseModels: this.baseModels.length,
            baseModelNames: this.baseModels.map(m => m.name),
            stackingMethod: this.config.stackingMethod,
            weights: this.config.weights,
            trained: this.trained,
            stackingTrained: this.stackingTrained,
            metaLearnerConfig: this.config.metaLearnerConfig
        };
    }

    dispose() {
        // Dispose base models
        this.baseModels.forEach(model => model.dispose());
        
        // Dispose meta-learner
        if (this.metaLearner) {
            this.metaLearner.dispose();
        }
        
        // Dispose stored predictions
        this.baseModelPredictions.forEach(pred => {
            if (pred.predictions && typeof pred.predictions.dispose === 'function') {
                pred.predictions.dispose();
            }
        });
        
        super.dispose();
    }
}

/**
 * Factory for creating common ensemble configurations
 */
export class EnsembleFactory {
    static createDiverseEnsemble(pcaParams) {
        const models = [];
        
        // Import models dynamically
        return import('./mlp.js').then(({ MLPModel }) => {
            return import('./xgboost.js').then(({ XGBoostModel }) => {
                return import('./randomForest.js').then(({ RandomForestModel }) => {
                    
                    // Deep MLP for complex patterns
                    models.push(new MLPModel({
                        hiddenLayers: [128, 64, 32],
                        activation: 'relu',
                        dropout: 0.3,
                        learningRate: 0.001,
                        epochs: 150
                    }));
                    
                    // Shallow MLP for simple patterns
                    models.push(new MLPModel({
                        hiddenLayers: [32, 16],
                        activation: 'tanh',
                        dropout: 0.1,
                        learningRate: 0.01,
                        epochs: 100
                    }));
                    
                    // XGBoost for gradient boosting
                    models.push(new XGBoostModel({
                        nEstimators: 100,
                        maxDepth: 6,
                        learningRate: 0.1,
                        subsample: 0.8
                    }));
                    
                    // Random Forest for bagging
                    models.push(new RandomForestModel({
                        nEstimators: 100,
                        maxDepth: 10,
                        minSamplesSplit: 5,
                        bootstrap: true
                    }));
                    
                    // Create ensemble with meta-learning
                    return new EnsembleModel(models, null, {
                        stackingMethod: 'meta',
                        metaLearnerConfig: {
                            hiddenLayers: [32, 16],
                            activation: 'relu',
                            dropout: 0.2,
                            learningRate: 0.005,
                            epochs: 75
                        }
                    });
                });
            });
        });
    }
    
    static createLightEnsemble() {
        const models = [];
        
        return import('./mlp.js').then(({ MLPModel }) => {
            // Lightweight models for fast prediction
            models.push(new MLPModel({
                hiddenLayers: [32],
                activation: 'relu',
                dropout: 0.1,
                learningRate: 0.01,
                epochs: 50
            }));
            
            models.push(new MLPModel({
                hiddenLayers: [16, 8],
                activation: 'tanh',
                dropout: 0.05,
                learningRate: 0.005,
                epochs: 75
            }));
            
            return new EnsembleModel(models, null, {
                stackingMethod: 'weighted'
            });
        });
    }
}