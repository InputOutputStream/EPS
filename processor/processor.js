import * as tf from '@tensorflow/tfjs';

export class DataProcessor {
    constructor(nComponents = 3, varianceThreshold = 0.01) {
        this.nComponents = nComponents;
        this.varianceThreshold = varianceThreshold;
        
        // NASA-prioritized features
        this.criticalFeatures = [
            'disposition', 'pl_orbper', 'pl_rade', 'pl_trandur',
            'st_teff', 'st_rad', 'discoverymethod'
        ];
        
        this.nasaPriorityFeatures = {
            'pl_orbper': 1.0,
            'pl_trandur': 0.95,
            'pl_trandep': 0.90,
            'pl_rade': 0.95,
            'pl_masse': 0.85,
            'st_teff': 0.90,
            'st_rad': 0.85,
            'st_mass': 0.80,
            'pl_orbincl': 0.75,
            'pl_imppar': 0.70
        };
        
        this.categoricalColumns = [
            'discoverymethod', 'disc_locale', 'disc_facility',
            'disc_telescope', 'disc_instrument', 'st_spectype'
        ];
        
        this.columnsToDropList = [
            'rowid', 'pl_name', 'hostname', 'pl_letter', 'k2_name',
            'epic_hostname', 'epic_candname', 'hd_name', 'hip_name',
            'pl_refname', 'disc_refname', 'pl_pubdate', 'releasedate',
            'rastr', 'decstr', 'st_refname', 'sy_refname',
            'rowupdate', 'default_flag'
        ];
        
        this.scaler = null;
        this.pcaModel = null;
        this.selectedFeatures = null;
        this.featureImportance = null;
        this.labelEncoders = {};
        this.calculatedReputations = {};
    }

    async preprocessData(csvData, applyPCA = true) {
        console.log('\n=== PREPROCESSING PIPELINE ===');
        
        let data = csvData.data;
        
        // 1. Calculate reputations
        this.calculateReputations(data);
        
        // 2. Filter critical missing
        data = this.filterCriticalMissing(data);
        
        // 3. Add reputation features
        data = this.addReputationFeatures(data);
        
        // 4. Encode categorical
        data = this.encodeCategorical(data);
        
        // 5. Drop unnecessary columns
        data = this.removeDropColumns(data);
        
        // 6. Extract labels BEFORE removing disposition
        const labels = this.extractLabels(data);
        data = data.map(row => {
            const { disposition, ...rest } = row;
            return rest;
        });
        
        // 7. Handle missing values
        data = this.handleMissingValues(data);
        
        // 8. Convert to matrix
        const { features, featureNames } = this.toMatrix(data);
        
        // 9. Feature selection
        const { X, selectedFeatures } = await this.selectFeatures(features, labels, featureNames);
        
        // 10. Handle outliers
        const XClean = this.handleOutliers(X);
        
        // 11. Log transform skewed
        const XTransformed = this.logTransformSkewed(XClean, selectedFeatures);
        
        // 12. Apply PCA if requested (CRITICAL FIX)
        let finalFeatures;
        if (applyPCA) {
            console.log('\n=== APPLYING PCA TRANSFORMATION ===');
            finalFeatures = await this.fitTransform(XTransformed);
            console.log(`✓ PCA applied: ${XTransformed.length} samples × ${XTransformed[0].length} features → ${finalFeatures.length} × ${finalFeatures[0].length} components`);
        } else {
            finalFeatures = XTransformed;
            console.log(`✓ Skipping PCA: ${XTransformed.length} samples × ${XTransformed[0].length} features`);
        }
        
        console.log(`\n✓ Final: ${finalFeatures.length} records × ${finalFeatures[0].length} features`);
        
        return {
            features: finalFeatures,
            labels,
            featureNames: applyPCA ? 
                Array.from({length: this.nComponents}, (_, i) => `PC${i+1}`) : 
                selectedFeatures,
            rawFeatures: XTransformed,
            rawFeatureNames: selectedFeatures,
            pcaApplied: applyPCA
        };
    }

    calculateReputations(data) {
        console.log('\nCalculating data-driven reputations...');
        
        const successMap = { 'CONFIRMED': 1, 'Confirmed': 1 };
        
        // Telescope reputation
        if (data[0].disc_telescope) {
            const telescopeStats = {};
            data.forEach(row => {
                const tel = row.disc_telescope || 'unknown';
                if (!telescopeStats[tel]) telescopeStats[tel] = { sum: 0, count: 0 };
                telescopeStats[tel].count++;
                telescopeStats[tel].sum += successMap[row.disposition] || 0;
            });
            
            const maxCount = Math.max(...Object.values(telescopeStats).map(s => s.count));
            this.calculatedReputations.telescope = {};
            Object.entries(telescopeStats).forEach(([tel, stats]) => {
                const successRate = stats.sum / stats.count;
                const volume = stats.count / maxCount;
                this.calculatedReputations.telescope[tel] = successRate * 0.7 + volume * 0.3;
            });
            
            console.log(`  Telescopes: ${Object.keys(this.calculatedReputations.telescope).length} analyzed`);
        }
        
        // Method reputation
        if (data[0].discoverymethod) {
            const methodStats = {};
            data.forEach(row => {
                const method = row.discoverymethod || 'unknown';
                if (!methodStats[method]) methodStats[method] = { sum: 0, count: 0 };
                methodStats[method].count++;
                methodStats[method].sum += successMap[row.disposition] || 0;
            });
            
            const maxCount = Math.max(...Object.values(methodStats).map(s => s.count));
            this.calculatedReputations.method = {};
            Object.entries(methodStats).forEach(([method, stats]) => {
                const successRate = stats.sum / stats.count;
                const volume = stats.count / maxCount;
                this.calculatedReputations.method[method] = successRate * 0.8 + volume * 0.2;
            });
            
            console.log(`  Methods: ${Object.keys(this.calculatedReputations.method).length} analyzed`);
        }
    }

    filterCriticalMissing(data) {
        const initial = data.length;
        const criticalPresent = this.criticalFeatures.filter(col => 
            data[0].hasOwnProperty(col)
        );
        
        const topCritical = criticalPresent.slice(0, 3);
        const filtered = data.filter(row => 
            topCritical.every(col => row[col] != null && row[col] !== '')
        );
        
        console.log(`Critical filter: ${initial} → ${filtered.length} records`);
        return filtered;
    }

    addReputationFeatures(data) {
        console.log('\nApplying calculated reputations...');
        
        return data.map(row => {
            const newRow = { ...row };
            
            if (this.calculatedReputations.telescope && row.disc_telescope) {
                newRow.telescope_reputation = 
                    this.calculatedReputations.telescope[row.disc_telescope] || 0.5;
            }
            
            if (this.calculatedReputations.method && row.discoverymethod) {
                newRow.method_reputation = 
                    this.calculatedReputations.method[row.discoverymethod] || 0.5;
            }
            
            // Composite confidence
            const repValues = [];
            if (newRow.telescope_reputation) repValues.push(newRow.telescope_reputation);
            if (newRow.method_reputation) repValues.push(newRow.method_reputation);
            if (repValues.length > 0) {
                newRow.confidence_score = repValues.reduce((a, b) => a + b, 0) / repValues.length;
            }
            
            return newRow;
        });
    }

    encodeCategorical(data) {
        console.log('\nEncoding categorical features...');
        
        return data.map(row => {
            const newRow = { ...row };
            
            this.categoricalColumns.forEach(col => {
                if (row[col] != null) {
                    const value = String(row[col]);
                    
                    if (!this.labelEncoders[col]) {
                        this.labelEncoders[col] = { mapping: {}, counter: 0 };
                    }
                    
                    if (!this.labelEncoders[col].mapping[value]) {
                        this.labelEncoders[col].mapping[value] = this.labelEncoders[col].counter++;
                    }
                    
                    newRow[`${col}_encoded`] = this.labelEncoders[col].mapping[value];
                    delete newRow[col];
                }
            });
            
            return newRow;
        });
    }

    removeDropColumns(data) {
        const toDrop = this.columnsToDropList.filter(col => data[0].hasOwnProperty(col));
        console.log(`Dropped ${toDrop.length} identifier columns`);
        
        return data.map(row => {
            const newRow = { ...row };
            toDrop.forEach(col => delete newRow[col]);
            return newRow;
        });
    }

    extractLabels(data) {
        const labelMap = {
            'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0,
            'Confirmed': 2, 'Candidate': 1, 'False Positive': 0
        };
        
        const labels = data.map(row => labelMap[row.disposition] || 0);
        
        const counts = [0, 0, 0];
        labels.forEach(l => counts[l]++);
        console.log(`\nLabels: FP=${counts[0]}, Candidate=${counts[1]}, Confirmed=${counts[2]}`);
        
        return labels;
    }

    handleMissingValues(data) {
        console.log('\nHandling missing values...');
        
        const keys = Object.keys(data[0]).filter(k => k !== 'disposition');
        const toRemove = new Set();
        
        keys.forEach(col => {
            const values = data.map(row => row[col]);
            const missing = values.filter(v => v == null || v === '').length;
            const missingPct = (missing / data.length) * 100;
            
            const isPriority = Object.keys(this.nasaPriorityFeatures).some(p => col.includes(p));
            
            if (missingPct > 70 && !isPriority) {
                toRemove.add(col);
            } else if (missingPct > 0) {
                const numeric = values.filter(v => v != null && !isNaN(v));
                const median = this.median(numeric);
                
                data.forEach(row => {
                    if (row[col] == null || row[col] === '') {
                        row[col] = median || 0;
                    }
                });
            }
        });
        
        return data.map(row => {
            const newRow = { ...row };
            toRemove.forEach(col => delete newRow[col]);
            return newRow;
        });
    }

    toMatrix(data) {
        const keys = Object.keys(data[0]);
        const numericKeys = keys.filter(k => {
            const val = data[0][k];
            return typeof val === 'number' || !isNaN(parseFloat(val));
        });
        
        const features = data.map(row => 
            numericKeys.map(k => parseFloat(row[k]) || 0)
        );
        
        return { features, featureNames: numericKeys };
    }

    async selectFeatures(features, labels, featureNames) {
        console.log('\nFeature selection (NASA-weighted)...');
        
        // Variance threshold
        const variances = this.calculateVariances(features);
        const selectedIdx = variances
            .map((v, i) => ({ v, i }))
            .filter(x => x.v > this.varianceThreshold)
            .map(x => x.i);
        
        let selectedFeatures = selectedIdx.map(i => featureNames[i]);
        let X = features.map(row => selectedIdx.map(i => row[i]));
        
        console.log(`  Variance filter: ${featureNames.length} → ${selectedFeatures.length}`);
        
        // Mutual information with NASA weighting
        const miScores = await this.mutualInformation(X, labels);
        
        const weightedScores = miScores.map((score, i) => {
            const feat = selectedFeatures[i];
            const nasaWeight = Math.max(
                ...Object.entries(this.nasaPriorityFeatures)
                    .filter(([k]) => feat.includes(k))
                    .map(([, v]) => v),
                1.0
            );
            return score * nasaWeight;
        });
        
        this.featureImportance = Object.fromEntries(
            selectedFeatures.map((f, i) => [f, weightedScores[i]])
        );
        
        // Select top features
        const topN = Math.min(50, selectedFeatures.length);
        const sortedIdx = weightedScores
            .map((score, i) => ({ score, i }))
            .sort((a, b) => b.score - a.score)
            .slice(0, topN)
            .map(x => x.i);
        
        selectedFeatures = sortedIdx.map(i => selectedFeatures[i]);
        X = X.map(row => sortedIdx.map(i => row[i]));
        
        console.log(`  NASA-weighted MI: Top ${selectedFeatures.length} features`);
        console.log(`  Top 5: ${selectedFeatures.slice(0, 5).join(', ')}`);
        
        this.selectedFeatures = selectedFeatures;
        return { X, selectedFeatures };
    }

    calculateVariances(features) {
        const n = features.length;
        const m = features[0].length;
        const variances = [];
        
        for (let j = 0; j < m; j++) {
            const values = features.map(row => row[j]);
            const mean = values.reduce((a, b) => a + b, 0) / n;
            const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / n;
            variances.push(variance);
        }
        
        return variances;
    }

    async mutualInformation(X, y) {
        const n = X.length;
        const m = X[0].length;
        const miScores = [];
        
        for (let j = 0; j < m; j++) {
            const values = X.map(row => row[j]);
            
            // Discretize feature
            const sorted = [...values].sort((a, b) => a - b);
            const q1 = sorted[Math.floor(n * 0.25)];
            const q2 = sorted[Math.floor(n * 0.5)];
            const q3 = sorted[Math.floor(n * 0.75)];
            
            const discretized = values.map(v => {
                if (v <= q1) return 0;
                if (v <= q2) return 1;
                if (v <= q3) return 2;
                return 3;
            });
            
            // Calculate MI
            let mi = 0;
            for (let xi = 0; xi < 4; xi++) {
                for (let yi = 0; yi < 3; yi++) {
                    const pxy = discretized.filter((x, i) => x === xi && y[i] === yi).length / n;
                    const px = discretized.filter(x => x === xi).length / n;
                    const py = y.filter(yy => yy === yi).length / n;
                    
                    if (pxy > 0 && px > 0 && py > 0) {
                        mi += pxy * Math.log2(pxy / (px * py));
                    }
                }
            }
            
            miScores.push(Math.max(0, mi));
        }
        
        return miScores;
    }

    handleOutliers(X) {
        const XClean = X.map(row => [...row]);
        const m = X[0].length;
        
        for (let j = 0; j < m; j++) {
            const values = X.map(row => row[j]);
            const sorted = [...values].sort((a, b) => a - b);
            const n = sorted.length;
            const q1 = sorted[Math.floor(n * 0.25)];
            const q3 = sorted[Math.floor(n * 0.75)];
            const iqr = q3 - q1;
            const lower = q1 - 3 * iqr;
            const upper = q3 + 3 * iqr;
            const median = this.median(values);
            
            XClean.forEach((row, i) => {
                if (row[j] < lower || row[j] > upper) {
                    row[j] = median;
                }
            });
        }
        
        return XClean;
    }

    logTransformSkewed(X, features) {
        const skewCols = ['pl_orbper', 'pl_trandep', 'pl_masse', 'pl_massj'];
        
        const XTransformed = X.map(row => [...row]);
        
        features.forEach((feat, j) => {
            if (skewCols.some(skew => feat.includes(skew))) {
                const allPositive = X.every(row => row[j] > 0);
                if (allPositive) {
                    XTransformed.forEach(row => {
                        row[j] = Math.log10(row[j] + 1e-10);
                    });
                }
            }
        });
        
        return XTransformed;
    }

    async fitTransform(X) {
        console.log('Fitting PCA transform...');
        
        // Standardize
        const { mean, std } = this.calculateStats(X);
        this.scaler = { mean, std };
        
        const XScaled = X.map(row => 
            row.map((val, j) => (val - mean[j]) / (std[j] || 1))
        );
        
        // PCA via covariance matrix and eigendecomposition
        const n = XScaled.length;
        
        // Compute covariance matrix
        const XTensor = tf.tensor2d(XScaled);
        const XTranspose = tf.transpose(XTensor);
        const covMatrix = tf.matMul(XTranspose, XTensor).div(n - 1);
        
        const covData = await covMatrix.array();
        
        // Simplified eigendecomposition
        const { eigenvalues, eigenvectors } = this.computeEigenPairs(covData, this.nComponents);
        
        // Sort by eigenvalues (descending)
        const sortedPairs = eigenvalues
            .map((val, i) => ({ val, vec: eigenvectors[i] }))
            .sort((a, b) => b.val - a.val)
            .slice(0, this.nComponents);
        
        const sortedEigenvalues = sortedPairs.map(p => p.val);
        const components = sortedPairs.map(p => p.vec);
        
        // Calculate explained variance
        const totalVariance = eigenvalues.reduce((a, b) => a + b, 0);
        const explainedVariance = sortedEigenvalues.map(v => v / totalVariance);
        
        // Transform data
        const componentsTensor = tf.tensor2d(components);
        const XPca = tf.matMul(XTensor, tf.transpose(componentsTensor));
        
        this.pcaModel = {
            components: components,
            explainedVariance: explainedVariance
        };
        
        const XPcaArray = await XPca.array();
        
        // Cleanup
        XTensor.dispose();
        XTranspose.dispose();
        covMatrix.dispose();
        componentsTensor.dispose();
        XPca.dispose();
        
        console.log(`✓ PCA variance: ${explainedVariance.map(v => (v * 100).toFixed(1) + '%').join(', ')}`);
        
        return XPcaArray;
    }

    computeEigenPairs(matrix, nComponents) {
        const n = matrix.length;
        const eigenvalues = [];
        const eigenvectors = [];
        
        // Power iteration method for dominant eigenvectors
        for (let k = 0; k < Math.min(nComponents, n); k++) {
            // Random initialization
            let vector = Array(n).fill(0).map(() => Math.random() - 0.5);
            
            // Power iteration
            for (let iter = 0; iter < 100; iter++) {
                // Matrix-vector multiplication
                const newVector = Array(n).fill(0);
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        newVector[i] += matrix[i][j] * vector[j];
                    }
                }
                
                // Orthogonalize against previous eigenvectors
                for (let prev = 0; prev < k; prev++) {
                    const dot = newVector.reduce((sum, val, i) => sum + val * eigenvectors[prev][i], 0);
                    for (let i = 0; i < n; i++) {
                        newVector[i] -= dot * eigenvectors[prev][i];
                    }
                }
                
                // Normalize
                const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
                if (norm > 1e-10) {
                    vector = newVector.map(v => v / norm);
                }
            }
            
            // Calculate eigenvalue (Rayleigh quotient)
            const Av = Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Av[i] += matrix[i][j] * vector[j];
                }
            }
            const eigenvalue = vector.reduce((sum, val, i) => sum + val * Av[i], 0);
            
            eigenvalues.push(Math.max(0, eigenvalue));
            eigenvectors.push(vector);
            
            // Deflate matrix for next iteration
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    matrix[i][j] -= eigenvalue * vector[i] * vector[j];
                }
            }
        }
        
        return { eigenvalues, eigenvectors };
    }

    transform(X) {
        if (!this.scaler || !this.pcaModel) {
            throw new Error('Processor not fitted');
        }
        
        // Standardize
        const XScaled = X.map(row => 
            row.map((val, j) => (val - this.scaler.mean[j]) / (this.scaler.std[j] || 1))
        );
        
        // Apply PCA
        return XScaled.map(row => 
            this.pcaModel.components.map(component =>
                row.reduce((sum, val, i) => sum + val * component[i], 0)
            )
        );
    }

    calculateStats(X) {
        const n = X.length;
        const m = X[0].length;
        
        const mean = [];
        const std = [];
        
        for (let j = 0; j < m; j++) {
            const values = X.map(row => row[j]);
            const m_val = values.reduce((a, b) => a + b, 0) / n;
            const variance = values.reduce((sum, v) => sum + Math.pow(v - m_val, 2), 0) / n;
            
            mean.push(m_val);
            std.push(Math.sqrt(variance));
        }
        
        return { mean, std };
    }

    median(values) {
        if (values.length === 0) return 0;
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 
            ? (sorted[mid - 1] + sorted[mid]) / 2 
            : sorted[mid];
    }

    exportParams() {
        return {
            mean: this.scaler?.mean || [],
            std: this.scaler?.std || [],
            components: this.pcaModel?.components || [],
            explained_variance_ratio: this.pcaModel?.explainedVariance || [],
            n_components: this.nComponents,
            selected_features: this.selectedFeatures || [],
            feature_importance: this.featureImportance || {}
        };
    }
}