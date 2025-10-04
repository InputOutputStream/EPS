import { MLPModel } from '../models/mlp.js';
import { XGBoostModel } from '../models/xgboost.js';
import { RandomForestModel } from '../models/randomForest.js';
import { EnsembleModel, EnsembleFactory } from '../models/ensemble.js';
import { ModelUtils } from '../models/base.js';
import { DataLoader } from '../loader/dataLoader.js';
import { Visualizer } from '../visualization/charts.js';

export class ExoplanetDetectorApp {
    constructor() {
        this.currentMode = 'novice';
        this.models = new Map();
        this.activeModel = null;
        this.dataLoader = new DataLoader();
        this.visualizer = new Visualizer();
        this.pcaParams = null;
        this.trainingData = null;
        this.isTraining = false;
        
        this.initializeModels();
        this.setupEventListeners();
    }

    async initialize() {
        console.log('Initializing EDA...');
        
        try {
            // Load PCA parameters
            await this.loadPCAParameters();
            
            // Initialize visualizations
            this.visualizer.initializePlots();
            
            // Set default active model
            this.setActiveModel('ensemble');
            
            console.log('Application initialized successfully');
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('Failed to initialize application: ' + error.message);
        }
    }

    initializeModels() {
        // Deep MLP Model
        this.models.set('mlp-deep', new MLPModel({
            hiddenLayers: [128, 64, 32],
            activation: 'relu',
            dropout: 0.3,
            learningRate: 0.001,
            epochs: 150,
            batchSize: 32
        }));

        // Shallow MLP Model  
        this.models.set('mlp-shallow', new MLPModel({
            hiddenLayers: [32, 16],
            activation: 'tanh',
            dropout: 0.1,
            learningRate: 0.01,
            epochs: 100,
            batchSize: 64
        }));

        // XGBoost Model
        this.models.set('xgboost', new XGBoostModel({
            nEstimators: 100,
            maxDepth: 6,
            learningRate: 0.1,
            subsample: 0.8,
            colsampleByTree: 0.8
        }));

        // Random Forest Model
        this.models.set('randomforest', new RandomForestModel({
            nEstimators: 100,
            maxDepth: 10,
            minSamplesSplit: 5,
            maxFeatures: 'sqrt',
            bootstrap: true
        }));

        // Ensemble Model
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

    async loadPCAParameters() { // There is something to fo here
        try {
            this.pcaParams = await ModelUtils.loadPCAParams();
            if (this.pcaParams) {
                console.log('PCA parameters loaded successfully');
                this.updatePCADisplay();
            } else {
                console.warn('PCA parameters not found, using mock data');
                this.generateMockPCAParams();
            }
        } catch (error) {
            console.error('Error loading PCA parameters:', error);
            this.generateMockPCAParams();
        }
    }

    generateMockPCAParams() {
        // Generate mock PCA parameters for testing
        this.pcaParams = {
            pca_components: [
                [0.4, 0.3, 0.2, 0.1, 0.05, -0.05],
                [-0.3, 0.4, 0.1, 0.2, 0.15, 0.1],
                [0.1, -0.2, 0.5, 0.2, 0.1, 0.3]
            ],
            explained_variance_ratio: [0.45, 0.25, 0.15],
            mean: [2.5, 8.0, 3.2, 1.0, 5500, 0.5],
            scale: [1.2, 5.0, 1.5, 0.8, 1000, 0.3],
            n_components: 3,
            feature_names: ['period', 'duration', 'depth', 'stellar_radius', 'stellar_temp', 'impact_parameter'] //Thats not all there are more features
        };
        
        console.log('Using mock PCA parameters');
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
        event.target.classList.add('active');
        
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
            return;
        }

        this.isTraining = true;
        this.updateStatus('Training models...', true);

        try {
            const { features, labels } = trainingData;
            
            // Apply PCA transformation
            const pcaFeatures = ModelUtils.applyPCA(features, this.pcaParams);
            
            console.log(`Training with ${features.length} samples, ${pcaFeatures.shape[1]} PCA features`);
            
            // Train active model
            if (this.activeModel instanceof EnsembleModel) {
                const results = await this.activeModel.fit(pcaFeatures, labels, valSplit);
                console.log('Ensemble training results:', results);
            } else {
                await this.activeModel.fit(pcaFeatures, labels, valSplit);
            }
            
            // Evaluate model performance
            const metrics = await this.activeModel.evaluate(pcaFeatures, labels);
            this.updateMetricsDisplay(metrics);
            
            this.updateStatus('Training completed successfully');
            
            // Update visualizations
            await this.updateTrainingVisualizations(pcaFeatures, labels);
            
            pcaFeatures.dispose();
            
        } catch (error) {
            console.error('Training failed:', error);
            this.updateStatus('Training failed: ' + error.message);
            this.showError('Model training failed: ' + error.message);
        } finally {
            this.isTraining = false;
        }
    }

    async predictSingle(features) {
        if (!this.activeModel || !this.activeModel.trained) {
            throw new Error('No trained model available for prediction');
        }

        try {
            // Apply PCA transformation
            const pcaFeatures = ModelUtils.applyPCA([features], this.pcaParams);
            
            // Get prediction
            const probabilities = await this.activeModel.predict(pcaFeatures);
            const classes = await this.activeModel.predictClasses(pcaFeatures);
            
            const probData = await probabilities.data();
            const classData = await classes.data();
            
            const result = {
                classification: this.getClassificationLabel(classData[0]),
                confidence: Math.max(...probData),
                probabilities: Array.from(probData),
                features: features
            };
            
            // Clean up tensors
            pcaFeatures.dispose();
            probabilities.dispose();
            classes.dispose();
            
            return result;
            
        } catch (error) {
            console.error('Prediction failed:', error);
            throw new Error('Prediction failed: ' + error.message);
        }
    }

    async batchPredict(featuresArray) {
        if (!this.activeModel || !this.activeModel.trained) {
            throw new Error('No trained model available for prediction');
        }

        try {
            this.updateStatus('Running batch prediction...', true);
            
            // Apply PCA transformation
            const pcaFeatures = ModelUtils.applyPCA(featuresArray, this.pcaParams);
            
            // Get predictions
            const probabilities = await this.activeModel.predict(pcaFeatures);
            const classes = await this.activeModel.predictClasses(pcaFeatures);
            
            const probData = await probabilities.data();
            const classData = await classes.data();
            
            // Format results
            const results = [];
            for (let i = 0; i < featuresArray.length; i++) {
                const startIdx = i * 3; // 3 classes: Confirmed, Candidate, FalseP
                results.push({
                    features: featuresArray[i],
                    result: {
                        classification: this.getClassificationLabel(classData[i]),
                        confidence: Math.max(...probData.slice(startIdx, startIdx + 3)),
                        probabilities: Array.from(probData.slice(startIdx, startIdx + 3))
                    }
                });
            }
            
            // Clean up tensors
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


    /**😑😑😑😑😑😑😑😑😑😑😑😑😑
     * 
     * Add a model whose role is to get in candidates and predict if they are confirmed or not 
     */

    getClassificationLabel(classIndex) {
        const labels = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'];
        return labels[classIndex] || 'UNKNOWN';
    }

    async updateHyperparameters(modelName, params) {
        const model = this.models.get(modelName);
        if (model) {
            model.updateHyperparameters(params);
            console.log(`Updated hyperparameters for ${modelName}:`, params);
            
            // Update display
            this.updateModelConfigDisplay(modelName, model.config);
        }
    }

    async updatePCAComponents(numComponents) {  
        if (this.pcaParams && numComponents > 0 && numComponents <= this.pcaParams.feature_names.length) {
            this.pcaParams.n_components = numComponents;
            console.log(`PCA components updated to ${numComponents}`);
            this.updatePCADisplay();
            
            // If training data exists, re-train models with new PCA settings
            if (this.trainingData) {
                await this.trainModels(this.trainingData);
            }
        } else {
            console.warn('Invalid number of PCA components:', numComponents);
        }   
    }

    async processFile(file) {
        try {
            this.updateStatus('Processing uploaded file...', true);
            
            const processedData = await this.dataLoader.loadFile(file);
            this.trainingData = processedData;
            console.log('File processed:', processedData);
            this.updateStatus(`Processed ${processedData.rowCount} records`);
            
            // Automatically start training if in researcher mode
            if (this.currentMode === 'researcher' && processedData.features.rowCount > 0) {
                await this.trainModels(processedData);
            }
            
            return processedData;
            
        } catch (error) {
            console.error('File processing failed:', error);
            this.updateStatus('File processing failed');
            throw error;
        }
    }

    // UI Update Methods
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
        if (!this.pcaParams) return;
        
        const varianceElement = document.getElementById('variance-explained');
        if (varianceElement) {
            const ratios = this.pcaParams.explained_variance_ratio;
            const cumulative = ratios.reduce((acc, val, idx) => {
                acc.push(idx === 0 ? val : acc[idx-1] + val);
                return acc;
            }, []);
            
            varianceElement.innerHTML = 
                `PC1: ${(ratios[0] * 100).toFixed(1)}%, ` +
                `PC2: ${(ratios[1] * 100).toFixed(1)}%, ` +
                `PC3: ${(ratios[2] * 100).toFixed(1)}%<br>` +
                `<small>Cumulative: ${(cumulative[2] * 100).toFixed(1)}%</small>`;
        }
    }

    updateModelDisplay(modelName) {
        // Update UI to reflect active model
        const modelButtons = document.querySelectorAll('.model-btn');
        modelButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.model === modelName);
        });
    }

    updateModelConfigDisplay(modelName, config) // TO DO
    {
        updateModelDisplay(modelName);

    }

    showError(message) {
        // Create or update error display
        let errorDiv = document.getElementById('error-display');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'error-display';
            errorDiv.className = 'error-message';
            document.querySelector('.container').prepend(errorDiv);
        }
        
        errorDiv.innerHTML = `
            <div class="error-content">
                <span>❌ Error: ${message}</span>
                <button onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 10000);
    }

    async updateTrainingVisualizations(features, labels) {
        // Update various charts and visualizations
        if (this.visualizer) {
            await this.visualizer.updatePCAPlot(features, labels);
            await this.visualizer.updateFeatureImportance(this.activeModel);
            await this.visualizer.updateModelComparison(this.models);
        }
    }

    setupEventListeners() {
        // Set up global event listeners
        window.switchMode = (mode) => this.switchMode(mode);
        window.setActiveModel = (modelName) => this.setActiveModel(modelName);
        window.trainModels = () => this.trainModels(this.trainingData);
        window.predictSingle = () => this.handleSinglePrediction();
        window.batchPredict = () => this.handleBatchPrediction();
        window.handleFileUpload = (event) => this.handleFileUpload(event);
        window.updatePCAComponents = (num) => updatePCAComponents(num);
        window.updateHyperparameter = (param, value, model) => 
            this.updateHyperparameters(model || 'mlp-deep', { [param]: value });
    }

    async handleSinglePrediction() {
        const features = this.extractFeaturesFromForm();
        
        if (!this.validateFeatures(features)) {
            this.showError('Please enter at least orbital period, transit duration, or transit depth.');
            return;
        }
        
        try {
            const result = await this.predictSingle(features);
            this.displaySingleResult(result);
        } catch (error) {
            this.showError(error.message);
        }
    }

    async handleBatchPrediction() {
        if (!this.trainingData || this.trainingData.features.length === 0) {
            // Generate mock data for demo
            // Dont forget to remove all the mock stuffs
            const mockFeatures = this.dataProcessor.generateMockDataset(100);
            const results = await this.batchPredict(mockFeatures);
            this.displayBatchResults(results);
        } else {
            const results = await this.batchPredict(this.trainingData.features.slice(0, 50));
            this.displayBatchResults(results);
        }
    }

    async handleFileUpload(event) {
        const files = event.target.files;
        if (files.length === 0) return;
        
        try {
            const file = files[0];
            await this.processFile(file);
        } catch (error) {
            this.showError('File upload failed: ' + error.message);
        }
    }

    extractFeaturesFromForm() {
        return {
            period: parseFloat(document.getElementById('period')?.value) || null,
            duration: parseFloat(document.getElementById('duration')?.value) || null,
            depth: parseFloat(document.getElementById('depth')?.value) || null,
            stellar_radius: parseFloat(document.getElementById('stellar_radius')?.value) || null,
            stellar_temp: parseFloat(document.getElementById('stellar_temp')?.value) || null,
            impact: parseFloat(document.getElementById('impact')?.value) || null
        };
    }

    validateFeatures(features) {
        return features.period || features.duration || features.depth;
    }

    displaySingleResult(result) {
        const resultsDiv = document.getElementById('results');
        if (!resultsDiv) return;
        
        const className = result.classification === 'CONFIRMED' ? 'confirmed' : 
                         result.classification === 'CANDIDATE' ? 'candidate' : 'false-positive';
        
        resultsDiv.innerHTML = `
            <div class="panel">
                <h3>🎯 Prediction Result (${this.activeModel.name})</h3>
                <div class="prediction-result ${className}">
                    <h2>${result.classification}</h2>
                    <p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
                    <div class="probability-breakdown">
                        <small>
                            False Positive: ${(result.probabilities[0] * 100).toFixed(1)}% | 
                            Candidate: ${(result.probabilities[1] * 100).toFixed(1)}% | 
                            Confirmed: ${(result.probabilities[2] * 100).toFixed(1)}%
                        </small>
                    </div>
                </div>
                <div class="metrics">
                    <div class="metric"><div class="metric-value">${result.features.period?.toFixed(2) || 'N/A'}</div><div class="metric-label">Period (days)</div></div>
                    <div class="metric"><div class="metric-value">${result.features.duration?.toFixed(2) || 'N/A'}</div><div class="metric-label">Duration (hrs)</div></div>
                    <div class="metric"><div class="metric-value">${result.features.depth?.toFixed(0) || 'N/A'}</div><div class="metric-label">Depth (ppm)</div></div>
                    <div class="metric"><div class="metric-value">${result.features.stellar_radius?.toFixed(2) || 'N/A'}</div><div class="metric-label">Stellar R☉</div></div>
                </div>
            </div>
        `;
    }

    displayBatchResults(results) {
        const confirmed = results.filter(r => r.result.classification === 'CONFIRMED').length;
        const candidates = results.filter(r => r.result.classification === 'CANDIDATE').length;
        const falsePos = results.filter(r => r.result.classification === 'FALSE POSITIVE').length;
        
        const resultsDiv = document.getElementById('results');
        if (!resultsDiv) return;
        
        resultsDiv.innerHTML = `
            <div class="panel">
                <h3>📊 Batch Analysis Results (${this.activeModel.name})</h3>
                <div class="metrics">
                    <div class="metric"><div class="metric-value" style="color: #00ff00">${confirmed}</div><div class="metric-label">Confirmed</div></div>
                    <div class="metric"><div class="metric-value" style="color: #ffa500">${candidates}</div><div class="metric-label">Candidates</div></div>
                    <div class="metric"><div class="metric-value" style="color: #ff0000">${falsePos}</div><div class="metric-label">False Positives</div></div>
                    <div class="metric"><div class="metric-value">${results.length}</div><div class="metric-label">Total Objects</div></div>
                </div>
                <div class="visualization" id="batch-results-chart"></div>
            </div>
        `;
        
        // Update batch chart
        this.visualizer.generateBatchChart(results);
    }

    dispose() {
        // Clean up all models
        this.models.forEach(model => model.dispose());
        this.models.clear();
        
        if (this.visualizer) {
            this.visualizer.dispose();
        }
        
        console.log('Application disposed');
    }
}