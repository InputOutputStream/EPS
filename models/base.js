import * as tf from '@tensorflow/tfjs';

/**
 * Base class for all ML models
 */
export class BaseModel {
    constructor(name, config = {}) {
        this.name = name;
        this.config = config;
        this.model = null;
        this.trained = false;
        this.metrics = {};
    }

    async fit(X, y) {
        throw new Error('fit method must be implemented by subclass');
    }

    async predict(X) {
        if (!this.trained) {
            throw new Error(`Model ${this.name} is not trained`);
        }
        throw new Error('predict method must be implemented by subclass');
    }

    async evaluate(X, y) {
        const predictions = await this.predict(X);
        return this.calculateMetrics(predictions, y);
    }

    calculateMetrics(predictions, actual) {
        // Convert to arrays if tensors
        const pred = predictions instanceof tf.Tensor ? predictions.arraySync() : predictions;
        const act = actual instanceof tf.Tensor ? actual.arraySync() : actual;

        const n = pred.length;
        let correct = 0;
        let tp = 0, fp = 0, tn = 0, fn = 0;

        for (let i = 0; i < n; i++) {
            const predClass = Array.isArray(pred[i]) ? pred[i].indexOf(Math.max(...pred[i])) : Math.round(pred[i]);
            const actualClass = Array.isArray(act[i]) ? act[i].indexOf(Math.max(...act[i])) : act[i];
            
            if (predClass === actualClass) correct++;
            
            // Binary classification metrics (assuming class 2 is positive)
            if (actualClass === 2 && predClass === 2) tp++;
            else if (actualClass !== 2 && predClass === 2) fp++;
            else if (actualClass !== 2 && predClass !== 2) tn++;
            else if (actualClass === 2 && predClass !== 2) fn++;
        }

        const accuracy = correct / n;
        const precision = tp / (tp + fp) || 0;
        const recall = tp / (tp + fn) || 0;
        const f1 = 2 * (precision * recall) / (precision + recall) || 0;

        this.metrics = { accuracy, precision, recall, f1 };
        return this.metrics;
    }

    getMetrics() {
        return this.metrics;
    }

    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }

    dispose() {
        if (this.model && typeof this.model.dispose === 'function') {
            this.model.dispose();
        }
    }
}

/**
 * Utility functions for model operations
 */
export class ModelUtils {
    static oneHotEncode(labels, numClasses = 3) {
        return tf.oneHot(tf.tensor1d(labels, 'int32'), numClasses);
    }

    static softmax(logits) {
        return tf.softmax(logits);
    }

    static argMax(predictions) {
        return tf.argMax(predictions, -1);
    }

    static async loadPCAParams(path = '/dataset/output/pca_params.json') {
        try {
            const response = await fetch(path);
            return await response.json();
        } catch (error) {
            console.error('Error loading PCA parameters:', error);
            return null;
        }
    }

    static applyPCA(data, pcaParams) {
        if (!pcaParams) {
            throw new Error('PCA parameters not available');
        }

        // Validate pcaParams structure
        if (!pcaParams.mean || !Array.isArray(pcaParams.mean)) {
            throw new Error('Invalid PCA parameters: mean is missing or not an array');
        }

        // Handle both naming conventions:
        // - DataProcessor uses: { mean, std, components }
        // - Pre-loaded PCA uses: { mean, scale, pca_components }
        const scaleArray = pcaParams.scale || pcaParams.std;
        const componentsArray = pcaParams.pca_components || pcaParams.components;

        if (!scaleArray || !Array.isArray(scaleArray)) {
            throw new Error('Invalid PCA parameters: scale/std is missing or not an array');
        }
        if (!componentsArray || !Array.isArray(componentsArray)) {
            throw new Error('Invalid PCA parameters: components is missing or not an array');
        }

        console.log('PCA Transform Info:', {
            dataShape: [data.length, data[0]?.length],
            meanLength: pcaParams.mean.length,
            scaleLength: scaleArray.length,
            componentsShape: [componentsArray.length, componentsArray[0]?.length]
        });

        try {
            const X = tf.tensor2d(data);
            const mean = tf.tensor1d(pcaParams.mean);
            const scale = tf.tensor1d(scaleArray);
            const components = tf.tensor2d(componentsArray);

            // Standardize: (X - mean) / scale
            const X_scaled = X.sub(mean).div(scale);
            
            // Apply PCA transformation: X_scaled * components^T
            const X_pca = X_scaled.matMul(components.transpose());

            // Clean up intermediate tensors
            X.dispose();
            X_scaled.dispose();
            mean.dispose();
            scale.dispose();
            components.dispose();

            console.log('PCA transformation successful:', X_pca.shape);
            return X_pca;

        } catch (error) {
            console.error('PCA transformation error:', error);
            throw new Error(`PCA transformation failed: ${error.message}`);
        }
    }

    /**
     * Normalize PCA parameters to a consistent format
     */
    static normalizePCAParams(pcaParams) {
        if (!pcaParams) return null;

        return {
            mean: pcaParams.mean,
            scale: pcaParams.scale || pcaParams.std,
            pca_components: pcaParams.pca_components || pcaParams.components,
            explained_variance_ratio: pcaParams.explained_variance_ratio,
            n_components: pcaParams.n_components,
            feature_names: pcaParams.feature_names || pcaParams.selected_features
        };
    }
}