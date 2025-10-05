import { MLPModel } from '../models/mlp.js';
import { XGBoostModel } from '../models/xgboost.js';
import { RandomForestModel } from '../models/randomForest.js';
import { EnsembleModel } from '../models/ensemble.js';
import { ModelUtils } from '../models/base.js';
import { DataLoader } from '../loader/dataLoader.js';
import { DataProcessor } from '../processor/processor.js';
import { Visualizer } from '../visualization/charts.js';
import { TrainTestSplit } from '../train_test_split/tts.js';

export class ExoplanetDetectorApp {
    constructor() {
        this.currentMode = 'novice';
        this.models = new Map();
        this.activeModel = null;
        this.dataLoader = new DataLoader();
        this.visualizer = new Visualizer();
        this.dataProcessor = new DataProcessor();
        this.pcaParams = null;
        this.trainingData = null;
        this.testData = null;
        this.candidatesData = null;
        this.splitStats = null;
        this.isTraining = false;
        this.dataLoadedFromPCA = false;
        
        this.initializeModels();
        this.setupEventListeners();
    }

    async initialize() {
        console.log('Initializing EDA...');
        
        try {
            await this.loadPCAParameters();
            this.visualizer.initializePlots();
            this.setActiveModel('ensemble');
            console.log('Application initialized successfully');
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showNotification('Failed to initialize application: ' + error.message, 'error');
        }
    }

    initializeModels() {
        this.models.set('mlp-deep', new MLPModel({
            hiddenLayers: [128, 64, 32],
            activation: 'relu',
            dropout: 0.3,
            learningRate: 0.001,
            epochs: 150,
            batchSize: 32
        }));

        this.models.set('mlp-shallow', new MLPModel({
            hiddenLayers: [32, 16],
            activation: 'tanh',
            dropout: 0.1,
            learningRate: 0.01,
            epochs: 100,
            batchSize: 64
        }));

        this.models.set('xgboost', new XGBoostModel({
            nEstimators: 100,
            maxDepth: 6,
            learningRate: 0.1,
            subsample: 0.8,
            colsampleByTree: 0.8
        }));

        this.models.set('randomforest', new RandomForestModel({
            nEstimators: 100,
            maxDepth: 10,
            minSamplesSplit: 5,
            maxFeatures: 'sqrt',
            bootstrap: true
        }));

        const baseModels = [
            this.models.get('mlp-deep'),
            this.models.get('mlp-shallow'),
            this.models.get('xgboost'),
            this.models.get('randomforest')
        ];

        this.models.set('ensemble', new EnsembleModel(baseModels, null, {
            stackingMethod: 'meta',
            metaLearnerConfig: {
                hiddenLayers: [32, 16],
                activation: 'relu',
                dropout: 0.2,
                learningRate: 0.005,
                epochs: 75
            }
        }));
    }

    async loadPCAParameters() {
        try {
            this.pcaParams = await ModelUtils.loadPCAParams();
            if (this.pcaParams) {
                console.log('PCA parameters loaded successfully');
                this.pcaParams = ModelUtils.normalizePCAParams(this.pcaParams);
                
                if (!this.pcaParams.explained_variance_ratio || !Array.isArray(this.pcaParams.explained_variance_ratio)) {
                    throw new Error('PCA parameters missing explained_variance_ratio array after normalization');
                }
                
                if (this.pcaParams.explained_variance_ratio.length < 3) {
                    throw new Error(`PCA parameters have only ${this.pcaParams.explained_variance_ratio.length} components, need at least 3`);
                }
                
                this.updatePCADisplay();
            } else {
                console.warn('PCA parameters not found - will generate when dataset is loaded');
                this.pcaParams = null;
                this.showNotification('PCA parameters will be generated from your dataset', 'info');
            }
        } catch (error) {
            console.error('Error loading PCA parameters:', error.message);
            this.pcaParams = null;
            this.showNotification(`PCA parameters unavailable: ${error.message}`, 'warning');
        }
    }

    generateMockPCAParams(nFeatures, nComponents = 3) {
        this.pcaParams = {
            mean: Array(nFeatures).fill(0).map(() => Math.random() * 10),
            scale: Array(nFeatures).fill(0).map(() => Math.random() * 2 + 0.1),
            pca_components: Array(nComponents).fill(0).map(() => 
                Array(nFeatures).fill(0).map(() => Math.random() * 0.2 - 0.1)
            ),
            explained_variance_ratio: Array(nComponents).fill(0).map((_, i) => 1 / (i + 1) / (1 + 1/(i+1))),
            n_components: nComponents,
            feature_names: Array(nFeatures).fill(0).map((_, i) => `feature_${i + 1}`)
        };
        console.log(`Generated mock PCA parameters for ${nFeatures} features and ${nComponents} components`);
    }

    setActiveModel(modelName) {
        if (this.models.has(modelName)) {
            this.activeModel = this.models.get(modelName);
            this.updateModelDisplay(modelName);
            console.log(`Active model set to: ${modelName}`);
        } else {
            console.error(`Model ${modelName} not found`);
        }
    }

    switchMode(mode) {
        this.currentMode = mode;
        document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
        const targetBtn = document.querySelector(`.mode-btn[data-mode="${mode}"]`);
        if (targetBtn) targetBtn.classList.add('active');
        
        const body = document.body;
        const tutorial = document.getElementById('tutorial');
        
        if (mode === 'novice') {
            body.classList.add('novice-mode');
            if (tutorial) tutorial.style.display = 'block';
        } else {
            body.classList.remove('novice-mode');
            if (tutorial) tutorial.style.display = 'none';
        }
        
        console.log(`Switched to ${mode} mode`);
    }

    async trainModels(trainingData, valSplit = 0.2) {
        if (this.isTraining) {
            console.log('Training already in progress');
            this.showNotification('Training already in progress');
            return;
        }
        
        if (!trainingData || !trainingData.features || !trainingData.labels) {
            this.showNotification('No training data available. Please upload a dataset first.', 'error');
            this.updateStatus('No training data available. Please upload a dataset first.', false);
            return;
        }
        
        this.isTraining = true;
        this.updateStatus('Training models...', true);

        try {
            const { features, labels } = trainingData;

            // CRITICAL FIX: Data is already PCA-transformed from preprocessing
            // Just convert to tensor directly
            const pcaFeatures = tf.tensor2d(features);
            console.log(`Training with ${features.length} samples, ${features[0].length} features`);

            let metrics;
            if (this.activeModel instanceof EnsembleModel) {
                const results = await this.activeModel.fit(pcaFeatures, labels, valSplit);
                console.log('Ensemble training results:', results);
                metrics = await this.activeModel.evaluate(pcaFeatures, labels);
            } else {
                await this.activeModel.fit(pcaFeatures, labels, valSplit);
                metrics = await this.activeModel.evaluate(pcaFeatures, labels);
            }

            this.updateMetricsDisplay(metrics);
            this.updateStatus('Training completed successfully');
            await this.updateTrainingVisualizations(pcaFeatures, labels);

            pcaFeatures.dispose();

        } catch (error) {
            console.error('Training failed:', error);
            this.updateStatus('Training failed: ' + error.message);
            this.showNotification('Model training failed: ' + error.message, 'error');
        } finally {
            this.isTraining = false;
        }
    }

    async batchPredict(featuresArray) {
        if (!this.activeModel || !this.activeModel.trained) {
            throw new Error('No trained model available for prediction');
        }

        try {
            this.updateStatus('Running batch prediction...', true);
            
            if (!Array.isArray(featuresArray) || featuresArray.length === 0) {
                throw new Error('featuresArray must be a non-empty array');
            }

            // Features are already PCA-transformed
            const pcaFeatures = tf.tensor2d(featuresArray);
            
            const probabilities = await this.activeModel.predict(pcaFeatures);
            const classes = await this.activeModel.predictClasses(pcaFeatures);
            
            const probData = await probabilities.data();
            const classData = await classes.data();
            
            // Binary classification: only FALSE POSITIVE (0) or CONFIRMED (2)
            const results = [];
            const nClasses = probabilities.shape[1] || 3;
            
            for (let i = 0; i < featuresArray.length; i++) {
                const startIdx = i * nClasses;
                const probs = Array.from(probData.slice(startIdx, startIdx + nClasses));
                
                // For binary decision: compare FALSE POSITIVE vs CONFIRMED probabilities
                const fpProb = probs[0] || 0;
                const confirmedProb = probs[2] || 0;
                
                // Determine classification (skip candidate class)
                const classIdx = classData[i];
                const isFalsePositive = (classIdx === 0) || (fpProb > confirmedProb);
                
                results.push({
                    features: featuresArray[i],
                    result: {
                        classification: isFalsePositive ? 'FALSE POSITIVE' : 'CONFIRMED',
                        confidence: Math.max(fpProb, confirmedProb),
                        probabilities: {
                            falsePositive: fpProb,
                            confirmed: confirmedProb
                        }
                    }
                });
            }
            
            pcaFeatures.dispose();
            probabilities.dispose();
            classes.dispose();
            
            this.updateStatus('Batch prediction completed');
            return results;
            
        } catch (error) {
            console.error('Batch prediction failed:', error);
            this.updateStatus('Batch prediction failed');
            throw new Error('Batch prediction failed: ' + error.message);
        }
    }

    getClassificationLabel(classIndex) {
        // Binary classification: 0 = FALSE POSITIVE, 2 = CONFIRMED
        // We skip index 1 (CANDIDATE) as it's not a valid prediction
        if (classIndex === 0) return 'FALSE POSITIVE';
        if (classIndex === 2) return 'CONFIRMED';
        // Fallback for any unexpected indices - treat as binary
        return classIndex < 1 ? 'FALSE POSITIVE' : 'CONFIRMED';
    }

    async processFile(file) {
        try {
            const extension = file.name.split('.').pop().toLowerCase();
            let data;

            if (extension === 'csv') {
                const rawData = await this.dataLoader.loadCSV(file);
                if (!rawData.data || rawData.data.length === 0) {
                    throw new Error('No valid data rows loaded from CSV');
                }
                
                // CRITICAL FIX: Apply PCA during preprocessing
                data = await this.dataProcessor.preprocessData(rawData, true);
                
                if (!data.features || !Array.isArray(data.features) || data.features.length === 0 || !data.features[0]) {
                    throw new Error('Invalid preprocessed data: features array is missing or empty');
                }
                
                // Store PCA params from preprocessing
                if (this.dataProcessor.pcaModel) {
                    this.pcaParams = this.dataProcessor.exportParams();
                    this.updatePCADisplay();
                }
                
                const splitter = new TrainTestSplit(0.2, 42);
                const splits = splitter.split(data.features, data.labels);
                
                this.dataLoadedFromPCA = true; // Data is now PCA-transformed
                console.log(`Loaded CSV with ${data.features[0].length} PCA components`);
                console.log(`Split: ${splits.train.features.length} train, ${splits.test.features.length} test, ${splits.candidates.features.length} candidates`);
                
                this.trainingData = splits.train;
                this.testData = splits.test;
                this.candidatesData = splits.candidates;
                this.splitStats = splits.stats;
                
                this.updateStatus(`Data loaded: ${splits.stats.train.total} train, ${splits.stats.test.total} test, ${splits.stats.candidates} candidates`);
                this.updateSplitDisplay(splits.stats);
                
                if (this.visualizer) {
                    this.visualizer.generateSplitVisualization(splits.stats);
                }
                
                return splits;
            } else if (extension === 'npy' || extension === 'json') {
                const pcaData = await this.dataLoader.loadPCAData([file]);
                if (pcaData.pca_params) {
                    this.pcaParams = pcaData.pca_params.data;
                    console.log('Loaded PCA parameters');
                    return pcaData.pca_params;
                }
                
                this.dataLoadedFromPCA = true;
                this.trainingData = pcaData.train;
                this.testData = pcaData.test;
                this.candidatesData = pcaData.candidates;
                
                const inputFeatures = pcaData.train.features[0]?.length || 0;
                console.log(`Loaded PCA data with ${inputFeatures} components`);
                
                if (!this.pcaParams) {
                    await this.loadPCAParameters();
                }
                
                const pcaComponents = this.pcaParams?.n_components || Math.min(3, inputFeatures);
                if (inputFeatures !== pcaComponents) {
                    throw new Error(
                        `PCA data dimension mismatch: input has ${inputFeatures} components, expected ${pcaComponents}`
                    );
                }
                
                this.updateStatus(`PCA data loaded: ${pcaData.train.features.length} train samples`);
                return pcaData;
            } else {
                throw new Error(`Unsupported file format: ${extension}`);
            }
        } catch (error) {
            console.error('Error processing file:', error);
            this.showNotification(`Failed to process file: ${error.message}`, 'error');
            throw error;
        }
    }

    async loadPCAData(files) {
        try {
            const results = await this.dataLoader.loadPCADataset(files);
            
            if (results.errors.length > 0) {
                console.warn('Some files failed to load:', results.errors);
            }
            
            const features = results.X_train_pca?.data || [];
            const labels = results.y_train?.data || [];
            
            console.log("PCA Params: ", results.pca_params);
            if (results.pca_params) {
                this.pcaParams = results.pca_params.data;
                this.updatePCADisplay();
            }
            
            this.dataLoadedFromPCA = true;
            
            return {
                train: {
                    features: this.reshapeNPYData(features, results.X_train_pca?.shape),
                    labels: labels
                },
                test: results.X_test_pca ? {
                    features: this.reshapeNPYData(results.X_test_pca.data, results.X_test_pca.shape),
                    labels: results.y_test?.data || []
                } : null,
                candidates: results.X_candidates_pca ? {
                    features: this.reshapeNPYData(results.X_candidates_pca.data, results.X_candidates_pca.shape),
                    labels: results.y_candidates?.data || []
                } : null,
                featureNames: Array.from({length: this.pcaParams?.n_components || 3}, (_, i) => `PC${i+1}`)
            };
            
        } catch (error) {
            console.error('PCA data loading failed:', error);
            throw new Error('Failed to load PCA data: ' + error.message);
        }
    }

    reshapeNPYData(flatData, shape) {
        if (!shape || shape.length < 2) return flatData;
        
        const rows = shape[0];
        const cols = shape[1];
        const reshaped = [];
        
        for (let i = 0; i < rows; i++) {
            reshaped.push(flatData.slice(i * cols, (i + 1) * cols));
        }
        
        return reshaped;
    }

    updateStatus(message, processing = false) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.classList.toggle('processing', processing);
        }
        console.log('Status:', message);
    }

    updateMetricsDisplay(metrics) {
        const elements = {
            'accuracy': document.getElementById('accuracy'),
            'precision': document.getElementById('precision'),
            'recall': document.getElementById('recall'),
            'f1score': document.getElementById('f1score')
        };
        
        Object.entries(metrics).forEach(([key, value]) => {
            if (elements[key]) {
                elements[key].textContent = `${(value * 100).toFixed(1)}%`;
            }
        });
    }

    updatePCADisplay() {
        const varianceElement = document.getElementById('variance-explained');
        if (!varianceElement) {
            console.warn('Variance explained element not found in DOM');
            return;
        }
        
        if (!this.pcaParams) {
            varianceElement.innerHTML = '<small style="color: #ffa500;">PCA parameters not loaded</small>';
            return;
        }
        
        if (!this.pcaParams.explained_variance_ratio || !Array.isArray(this.pcaParams.explained_variance_ratio)) {
            varianceElement.innerHTML = '<small style="color: #ff0000;">Invalid PCA variance data</small>';
            console.error('PCA params missing explained_variance_ratio array');
            return;
        }
        
        const ratios = this.pcaParams.explained_variance_ratio;
        
        if (ratios.length < 3) {
            varianceElement.innerHTML = '<small style="color: #ffa500;">Insufficient PCA components (need at least 3)</small>';
            return;
        }
        
        try {
            const cumulative = ratios.reduce((acc, val, idx) => {
                acc.push(idx === 0 ? val : acc[idx-1] + val);
                return acc;
            }, []);
            
            varianceElement.innerHTML = 
                `PC1: ${(ratios[0] * 100).toFixed(1)}%, ` +
                `PC2: ${(ratios[1] * 100).toFixed(1)}%, ` +
                `PC3: ${(ratios[2] * 100).toFixed(1)}%<br>` +
                `<small>Cumulative: ${(cumulative[2] * 100).toFixed(1)}%</small>`;
        } catch (error) {
            varianceElement.innerHTML = '<small style="color: #ff0000;">Error displaying PCA variance</small>';
            console.error('Error in updatePCADisplay:', error);
        }
    }

    updateSplitDisplay(stats) {
        let splitElement = document.getElementById('split-stats');
        if (!splitElement) {
            const container = document.querySelector('.data-info') || document.querySelector('.container');
            if (container) {
                splitElement = document.createElement('div');
                splitElement.id = 'split-stats';
                splitElement.className = 'panel';
                container.appendChild(splitElement);
            } else {
                return;
            }
        }
        
        splitElement.innerHTML = `
            <h3>Data Split Summary</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" style="color: #00d4ff">${stats.train.total}</div>
                    <div class="metric-label">Training</div>
                    <div class="metric-detail">
                        <small>Confirmed: ${stats.train.confirmed} | FP: ${stats.train.falsePositive}</small>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-value" style="color: #00ff00">${stats.test.total}</div>
                    <div class="metric-label">Testing</div>
                    <div class="metric-detail">
                        <small>Confirmed: ${stats.test.confirmed} | FP: ${stats.test.falsePositive}</small>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-value" style="color: #ffa500">${stats.candidates}</div>
                    <div class="metric-label">Candidates</div>
                    <div class="metric-detail">
                        <small>For Prediction</small>
                    </div>
                </div>
            </div>
        `;
    }

    updateModelDisplay(modelName) {
        const modelButtons = document.querySelectorAll('.model-btn');
        modelButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.model === modelName);
        });
    }

    showNotification(message, type = 'info') {
        let notificationDiv = document.getElementById('notification-display');
        if (!notificationDiv) {
            notificationDiv = document.createElement('div');
            notificationDiv.id = 'notification-display';
            notificationDiv.className = 'notification-container';
            const container = document.querySelector('.container');
            if (container) {
                container.prepend(notificationDiv);
            } else {
                document.body.prepend(notificationDiv);
            }
        }
        
        const icons = { error: '✗', warning: '⚠', info: 'ℹ', success: '✓' };
        const colors = { error: '#ff4444', warning: '#ffa500', info: '#4444ff', success: '#00ff00' };
        
        notificationDiv.innerHTML = `
            <div class="notification-content" style="border-left: 4px solid ${colors[type]}">
                <span>${icons[type]} ${message}</span>
                <button onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        const duration = type === 'error' ? 10000 : type === 'warning' ? 7000 : 5000;
        setTimeout(() => {
            if (notificationDiv.parentElement) {
                notificationDiv.remove();
            }
        }, duration);
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    async updateTrainingVisualizations(features, labels) {
        if (this.visualizer) {
            await this.visualizer.updatePCAPlot(features, labels);
            
            if (this.trainingData && this.testData) {
                await this.visualizer.updatePCAPlotBySplit(
                    this.trainingData,
                    this.testData,
                    this.candidatesData
                );
            }
            
            await this.visualizer.updateFeatureImportance(this.activeModel, this.pcaParams?.n_components);
            await this.visualizer.updateModelComparison(this.models);
            
            if (this.splitStats) {
                this.visualizer.generateSplitVisualization(this.splitStats);
            }
        }
    }

    setupEventListeners() {
        window.switchMode = (mode) => this.switchMode(mode);
        window.setActiveModel = (modelName) => this.setActiveModel(modelName);
        window.trainModels = () => this.trainModels(this.trainingData);
        window.evaluateOnTest = () => this.evaluateOnTestSet();
        window.predictSingle = () => this.handleSinglePrediction();
        window.batchPredict = () => this.handleBatchPrediction();
        window.handleFileUpload = (event) => this.handleFileUpload(event);
        window.updatePCAComponents = (num) => this.updatePCAComponents(num);
        window.updateHyperparameter = (param, value, model) => 
            this.updateHyperparameters(model || 'mlp-deep', { [param]: value });
        window.crossValidate = (k = 5) => this.crossValidate(k);
    }

    async evaluateOnTestSet() {
        if (!this.activeModel || !this.activeModel.trained) {
            this.showNotification('Please train a model first', 'warning');
            return;
        }
        
        if (!this.testData || this.testData.features.length === 0) {
            this.showNotification('No test data available', 'warning');
            return;
        }
        
        try {
            this.updateStatus('Evaluating on test set...', true);
            
            // Data is already PCA-transformed
            const testFeatures = tf.tensor2d(this.testData.features);
            
            const metrics = await this.activeModel.evaluate(testFeatures, this.testData.labels);
            testFeatures.dispose();
            
            this.displayTestResults(metrics);
            this.updateStatus('Test evaluation completed');
            this.showNotification('Test evaluation completed successfully', 'success');
            
        } catch (error) {
            console.error('Test evaluation failed:', error);
            this.updateStatus('Test evaluation failed');
            this.showNotification(`Test evaluation failed: ${error.message}`, 'error');
        }
    }

    displayTestResults(metrics) {
        let resultsDiv = document.getElementById('test-results');
        if (!resultsDiv) {
            const container = document.querySelector('.metrics-panel') || document.querySelector('.container');
            if (container) {
                resultsDiv = document.createElement('div');
                resultsDiv.id = 'test-results';
                resultsDiv.className = 'panel';
                container.appendChild(resultsDiv);
            } else {
                return;
            }
        }
        
        resultsDiv.innerHTML = `
            <h3>Test Set Performance (${this.activeModel.name})</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">${(metrics.accuracy * 100).toFixed(1)}%</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${(metrics.precision * 100).toFixed(1)}%</div>
                    <div class="metric-label">Test Precision</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${(metrics.recall * 100).toFixed(1)}%</div>
                    <div class="metric-label">Test Recall</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${(metrics.f1score * 100).toFixed(1)}%</div>
                    <div class="metric-label">Test F1 Score</div>
                </div>
            </div>
            <p><small>Test metrics show how well the model generalizes to unseen data.</small></p>
        `;
    }

    extractFeaturesFromForm() {
        const ids = ['period', 'duration', 'depth', 'stellar_radius', 'stellar_temp', 'impact'];
        const features = [];
        
        ids.forEach(id => {
            const el = document.getElementById(id);
            if (!el) {
                features.push(null);
            } else {
                const v = el.value;
                const n = v === '' || v === null ? null : Number(v);
                features.push(Number.isNaN(n) ? null : n);
            }
        });
        
        return features;
    }

    async handleSinglePrediction() {
        const features = this.extractFeaturesFromForm();
        
        if (!this.validateFeatures(features)) {
            this.showNotification('Please enter at least orbital period, transit duration, or transit depth.', 'error');
            return;
        }
        
        try {
            const result = await this.predictSingle(features);
            this.displaySingleResult(result);
        } catch (error) {
            this.showNotification(error.message, 'error');
        }
    }

    async handleBatchPrediction() {
        if (!this.activeModel || !this.activeModel.trained) {
            this.showNotification('Please train a model first', 'warning');
            return;
        }
        
        if (!this.candidatesData || this.candidatesData.features.length === 0) {
            this.showNotification('No candidate data available. Please upload a dataset with candidates.', 'warning');
            return;
        }
        
        try {
            const results = await this.batchPredict(this.candidatesData.features);
            this.displayBatchResults(results, 'candidates');
            this.showNotification(`Predicted ${results.length} candidates successfully`, 'success');
        } catch (error) {
            this.showNotification(`Batch prediction failed: ${error.message}`, 'error');
        }
    }

    async handleFileUpload(event) {
        const files = event.target.files;
        if (files.length === 0) return;
        
        try {
            for (const file of files) {
                await this.processFile(file);
            }
        } catch (error) {
            this.showNotification('File upload failed: ' + error.message, 'error');
        }
    }

    displayBatchResults(results, type = 'test') {
        const confirmed = results.filter(r => r.result.classification === 'CONFIRMED').length;
        const falsePos = results.filter(r => r.result.classification === 'FALSE POSITIVE').length;
        
        const resultsDiv = document.getElementById('results');
        if (!resultsDiv) return;
        
        const typeLabel = type === 'candidates' ? 'Candidate Classification Results' : 'Batch Analysis Results';
        
        resultsDiv.innerHTML = `
            <div class="panel">
                <h3>${typeLabel} (${this.activeModel.name})</h3>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" style="color: #00ff00">${confirmed}</div>
                        <div class="metric-label">Confirmed Exoplanets</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" style="color: #ff0000">${falsePos}</div>
                        <div class="metric-label">False Positives</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${results.length}</div>
                        <div class="metric-label">Total Objects</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${((confirmed / results.length) * 100).toFixed(1)}%</div>
                        <div class="metric-label">Confirmation Rate</div>
                    </div>
                </div>
                <div class="visualization" id="batch-results-chart"></div>
                ${type === 'candidates' ? '<p><small>Binary classification: candidates are predicted as either <strong>Confirmed Exoplanets</strong> or <strong>False Positives</strong>.</small></p>' : ''}
            </div>
        `;
        
        this.visualizer.generateBatchChart(results);
    }

    updateHyperparameters(modelName = 'mlp-deep', newConfig = {}) {
        try {
            if (!this.models.has(modelName)) {
                this.showNotification(`Model ${modelName} not found`, 'error');
                return;
            }
            const model = this.models.get(modelName);

            if (typeof model.updateHyperparameters === 'function') {
                model.updateHyperparameters(newConfig);
            } else if (typeof model.updateConfig === 'function') {
                model.updateConfig({ ...model.config, ...newConfig });
            } else {
                model.config = { ...(model.config || {}), ...newConfig };
            }

            const mapping = {
                'n_estimators': ['n-estimators', 'n-estimators-value'],
                'learning_rate': ['learning-rate', 'learning-rate-value'],
                'hidden_layers': ['hidden-layers', 'hidden-layers-value'],
                'dropout': ['dropout', 'dropout-value']
            };

            Object.entries(newConfig).forEach(([k, v]) => {
                if (mapping[k]) {
                    const [inputId, spanId] = mapping[k];
                    const el = document.getElementById(inputId);
                    const span = document.getElementById(spanId);
                    if (el) el.value = v;
                    if (span) span.textContent = v;
                }
            });

            const significant = ['nEstimators', 'maxDepth', 'minSamplesSplit', 'maxFeatures', 'bootstrap', 'hiddenLayers', 'learningRate'];
            const hasSignificant = Object.keys(newConfig).some(k => significant.includes(k) || significant.includes(this._toCamelCase(k)));

            if (hasSignificant) {
                if (model.trained) {
                    model.trained = false;
                }
                if (typeof model.dispose === 'function') {
                    try { model.dispose(); } catch (e) { /* ignore */ }
                }
                this.showNotification(`Updated hyperparameters for ${modelName}. Retraining is recommended.`, 'info');
            } else {
                this.showNotification(`Updated hyperparameters for ${modelName}`, 'success');
            }

            this.updateModelDisplay(modelName);
        } catch (error) {
            console.error('updateHyperparameters error:', error);
            this.showNotification('Failed to update hyperparameters: ' + error.message, 'error');
        }
    }

    _toCamelCase(s) {
        return s.replace(/_([a-z])/g, (m, p1) => p1.toUpperCase());
    }

    async updatePCAComponents(num) {
        try {
            const n = Number(num);
            const span = document.getElementById('pca-components-value');
            if (span) span.textContent = `${n}`;

            // Update processor configuration
            this.dataProcessor.nComponents = n;

            if (!this.pcaParams) {
                this.showNotification('PCA parameters will be applied when data is loaded.', 'info');
                return;
            }

            this.pcaParams.n_components = n;
            this.updatePCADisplay();

            // If we have training data, refresh visualizations
            if (this.visualizer && this.trainingData) {
                try {
                    await this.visualizer.updatePCAPlotBySplit(
                        this.trainingData,
                        this.testData,
                        this.candidatesData
                    );
                } catch (err) {
                    console.warn('PCA visualization refresh failed:', err);
                }
            }

            this.showNotification(`PCA components set to ${n}. Reload data to apply.`, 'success');
        } catch (error) {
            console.error('updatePCAComponents error:', error);
            this.showNotification('Failed to update PCA components: ' + error.message, 'error');
        }
    }
    
    validateFeatures(features) {
        if (!Array.isArray(features)) return false;
        const core = [features[0], features[1], features[2]];
        return core.some(v => v !== null && v !== undefined && !Number.isNaN(Number(v)));
    }

    async predictSingle(features = null) {
        try {
            if (!features) features = this.extractFeaturesFromForm();

            if (!this.validateFeatures(features)) {
                throw new Error('Please enter at least orbital period, transit duration, or transit depth.');
            }

            if (!this.activeModel || !this.activeModel.trained) {
                throw new Error('No trained model available for prediction. Please train a model first.');
            }

            const inputArr = [features];
            
            if (this.dataProcessor.scaler && this.dataProcessor.pcaModel) {
                const transformed = this.dataProcessor.transform(inputArr);
                const inputTensor = tf.tensor2d(transformed);
                
                const probabilities = await this.activeModel.predict(inputTensor);
                const classes = await this.activeModel.predictClasses(inputTensor);

                const probData = await probabilities.data();
                const classData = await classes.data();

                const nClasses = probabilities.shape[1] || 3;
                const probs = Array.from(probData).slice(0, nClasses);
                
                // Binary classification: FALSE POSITIVE vs CONFIRMED
                const fpProb = probs[0] || 0;
                const confirmedProb = probs[2] || 0;
                const classIdx = classData[0];
                
                const isFalsePositive = (classIdx === 0) || (fpProb > confirmedProb);
                const confidence = Math.max(fpProb, confirmedProb);

                inputTensor.dispose();
                probabilities.dispose();
                classes.dispose();

                return {
                    features,
                    result: {
                        classification: isFalsePositive ? 'FALSE POSITIVE' : 'CONFIRMED',
                        confidence,
                        probabilities: {
                            falsePositive: fpProb,
                            confirmed: confirmedProb
                        }
                    }
                };
            } else {
                throw new Error('Model not properly trained. Please train the model with a dataset first.');
            }
        } catch (error) {
            console.error('predictSingle error:', error);
            throw error;
        }
    }
    
    displaySingleResult(result) {
        try {
            const resultsDiv = document.getElementById('results');
            if (!resultsDiv) return;

            let singleDiv = document.getElementById('single-result');
            if (!singleDiv) {
                singleDiv = document.createElement('div');
                singleDiv.id = 'single-result';
                singleDiv.className = 'panel';
                resultsDiv.prepend(singleDiv);
            }

            const probs = result.result.probabilities;
            const conf = (result.result.confidence * 100).toFixed(1) + '%';
            const classification = result.result.classification;
            
            // Color based on classification
            const classColor = classification === 'CONFIRMED' ? '#00ff00' : '#ff0000';
            const classIcon = classification === 'CONFIRMED' ? '✓' : '✗';

            singleDiv.innerHTML = `
                <h3>Object Analysis (${this.activeModel?.name || 'Model'})</h3>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value" style="color: ${classColor}">${classIcon} ${classification}</div>
                        <div class="metric-label">Classification</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${conf}</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                </div>
                <div style="margin-top:15px;">
                    <strong>Probabilities:</strong>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px;">
                        <div>
                            <span style="color: #ff0000;">False Positive:</span> 
                            <strong>${(probs.falsePositive * 100).toFixed(1)}%</strong>
                        </div>
                        <div>
                            <span style="color: #00ff00;">Confirmed:</span> 
                            <strong>${(probs.confirmed * 100).toFixed(1)}%</strong>
                        </div>
                    </div>
                </div>
                <div style="margin-top:12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <small><strong>Input Features:</strong> ${result.features.map((v, i) => {
                        const labels = ['Period', 'Duration', 'Depth', 'Stellar R', 'Stellar T', 'Impact'];
                        return `${labels[i]}: ${v === null ? 'N/A' : v.toFixed(3)}`;
                    }).join(' | ')}</small>
                </div>
            `;

            this.showNotification(`Classified as ${classification} with ${conf} confidence`, 
                classification === 'CONFIRMED' ? 'success' : 'info');
        } catch (error) {
            console.error('displaySingleResult error:', error);
            this.showNotification('Failed to display prediction result', 'error');
        }
    }

    async crossValidate(k = 5) {
        try {
            if (!this.trainingData || !this.trainingData.features || this.trainingData.features.length === 0) {
                this.showNotification('No training data available for cross-validation', 'warning');
                return;
            }
            if (!this.activeModel) {
                this.showNotification('Select a model before running cross-validation', 'warning');
                return;
            }

            this.updateStatus('Running cross-validation...', true);

            const X = this.trainingData.features;
            const y = this.trainingData.labels;
            const n = X.length;
            const foldSize = Math.max(1, Math.floor(n / k));

            const metricsAcc = { accuracy: [], precision: [], recall: [], f1score: [] };

            for (let fold = 0; fold < k; fold++) {
                const start = fold * foldSize;
                const end = Math.min(n, start + foldSize);

                const X_val = X.slice(start, end);
                const y_val = y.slice(start, end);

                const X_train = X.slice(0, start).concat(X.slice(end));
                const y_train = y.slice(0, start).concat(y.slice(end));

                // Clone model
                let clone;
                try {
                    const Ctor = this.activeModel.constructor;
                    const cfg = JSON.parse(JSON.stringify(this.activeModel.config || {}));
                    clone = new Ctor(cfg);
                } catch (err) {
                    console.warn('Failed to clone model; using the same instance (may overwrite).', err);
                    clone = this.activeModel;
                }

                // Data already PCA-transformed
                const XtrainTensor = tf.tensor2d(X_train);
                const XvalTensor = tf.tensor2d(X_val);

                // Fit clone
                try {
                    await clone.fit(XtrainTensor, y_train, 0.1);
                    const foldMetrics = await clone.evaluate(XvalTensor, y_val);
                    metricsAcc.accuracy.push(foldMetrics.accuracy || 0);
                    metricsAcc.precision.push(foldMetrics.precision || 0);
                    metricsAcc.recall.push(foldMetrics.recall || 0);
                    metricsAcc.f1score.push(foldMetrics.f1score || foldMetrics.f1 || 0);
                } catch (err) {
                    console.warn('Cross-validation fold failed:', err);
                } finally {
                    if (XtrainTensor) XtrainTensor.dispose();
                    if (XvalTensor) XvalTensor.dispose();
                    if (clone && clone !== this.activeModel && typeof clone.dispose === 'function') {
                        try { clone.dispose(); } catch (e) {}
                    }
                }
            }

            // Aggregate metrics
            const avg = arr => arr.length ? (arr.reduce((a,b)=>a+b,0)/arr.length) : 0;
            const summary = {
                accuracy: avg(metricsAcc.accuracy),
                precision: avg(metricsAcc.precision),
                recall: avg(metricsAcc.recall),
                f1score: avg(metricsAcc.f1score)
            };

            // Display results
            this.displayCrossValResults(summary, k);
            this.updateStatus('Cross-validation completed');
            this.showNotification(`Cross-validation completed (${k} folds)`, 'success');

        } catch (error) {
            console.error('crossValidate error:', error);
            this.updateStatus('Cross-validation failed');
            this.showNotification('Cross-validation failed: ' + error.message, 'error');
        }
    }

    displayCrossValResults(summary, k) {
        try {
            let cvDiv = document.getElementById('crossval-results');
            if (!cvDiv) {
                cvDiv = document.createElement('div');
                cvDiv.id = 'crossval-results';
                cvDiv.className = 'panel';
                const container = document.querySelector('.container');
                if (container) {
                    container.insertBefore(cvDiv, container.firstChild);
                }
            }
            cvDiv.innerHTML = `
                <h3>Cross-validation (${k} folds) - Summary</h3>
                <div class="metrics">
                    <div class="metric"><div class="metric-value">${(summary.accuracy*100).toFixed(1)}%</div><div class="metric-label">Mean Accuracy</div></div>
                    <div class="metric"><div class="metric-value">${(summary.precision*100).toFixed(1)}%</div><div class="metric-label">Mean Precision</div></div>
                    <div class="metric"><div class="metric-value">${(summary.recall*100).toFixed(1)}%</div><div class="metric-label">Mean Recall</div></div>
                    <div class="metric"><div class="metric-value">${(summary.f1score*100).toFixed(1)}%</div><div class="metric-label">Mean F1</div></div>
                </div>
            `;
        } catch (error) {
            console.error('displayCrossValResults error:', error);
        }
    }

    dispose() {
        this.models.forEach(model => model.dispose());
        this.models.clear();
        
        if (this.visualizer) {
            this.visualizer.dispose();
        }
        
        console.log('Application disposed');
    }
}