import Papa from 'papaparse';

export class DataProcessor {
    constructor() {
        this.columnMappings = {
            // Kepler mappings
            'pl_orbper': 'period',
            'pl_trandur': 'duration',
            'pl_trandep': 'depth', 
            'st_rad': 'stellar_radius',
            'st_teff': 'stellar_temp',
            'pl_imppar': 'impact',
            'pl_rade': 'planet_radius',
            'disposition': 'classification',
            
            // TESS mappings
            'Period': 'period',
            'Duration': 'duration',
            'Depth': 'depth',
            'Stellar_Radius': 'stellar_radius',
            'Stellar_Temperature': 'stellar_temp',
            'Impact_Parameter': 'impact',
            'Planet_Radius': 'planet_radius',
            'Disposition': 'classification',
            
            // K2 mappings
            'koi_period': 'period',
            'koi_duration': 'duration',
            'koi_depth': 'depth',
            'koi_srad': 'stellar_radius',
            'koi_steff': 'stellar_temp',
            'koi_impact': 'impact',
            'koi_prad': 'planet_radius',
            'koi_disposition': 'classification'
        };
        
        this.requiredFeatures = ['period', 'duration', 'depth', 'stellar_radius', 'stellar_temp', 'impact'];
        this.classificationMap = {
            'CONFIRMED': 2,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0,
            'Confirmed': 2,
            'Candidate': 1,
            'False Positive': 0,
            'PC': 1, // Planet Candidate
            'FP': 0, // False Positive
            'CP': 2, // Confirmed Planet
            'KP': 2  // Known Planet
        };
    }

    async loadFile(file) {
        return new Promise((resolve, reject) => {
            const fileExtension = file.name.split('.').pop().toLowerCase();
            
            if (fileExtension === 'csv') {
                this.loadCSV(file, resolve, reject);
            } else if (fileExtension === 'json') {
                this.loadJSON(file, resolve, reject);
            } else {
                reject(new Error(`Unsupported file format: ${fileExtension}`));
            }
        });
    }

    loadCSV(file, resolve, reject) {
        Papa.parse(file, {
            complete: (results) => {
                if (results.errors.length > 0) {
                    console.warn('CSV parsing warnings:', results.errors);
                }
                resolve(results.data);
            },
            error: (error) => {
                reject(new Error(`CSV parsing failed: ${error.message}`));
            },
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            transformHeader: (header) => header.trim()
        });
    }

    loadJSON(file, resolve, reject) {
        const reader = new FileReader();
        
        reader.onload = (event) => {
            try {
                const data = JSON.parse(event.target.result);
                resolve(Array.isArray(data) ? data : [data]);
            } catch (error) {
                reject(new Error(`JSON parsing failed: ${error.message}`));
            }
        };
        
        reader.onerror = () => {
            reject(new Error('Failed to read file'));
        };
        
        reader.readAsText(file);
    }

    preprocessData(rawData) {
        console.log(`Preprocessing ${rawData.length} raw records...`);
        
        // Map column names
        const mappedData = this.mapColumnNames(rawData);
        
        // Extract features and labels
        const { features, labels } = this.extractFeaturesAndLabels(mappedData);
        
        // Clean and normalize features
        const cleanedFeatures = this.cleanFeatures(features);
        
        // Filter out invalid records
        const { validFeatures, validLabels } = this.filterValidRecords(cleanedFeatures, labels);
        
        console.log(`Preprocessing complete: ${validFeatures.length} valid records`);
        
        return {
            features: validFeatures,
            labels: validLabels,
            originalCount: rawData.length,
            validCount: validFeatures.length,
            featureNames: this.requiredFeatures
        };
    }

    mapColumnNames(data) {
        return data.map(record => {
            const mappedRecord = {};
            
            // Map known columns
            Object.entries(record).forEach(([key, value]) => {
                const mappedKey = this.columnMappings[key] || key.toLowerCase().replace(/\s+/g, '_');
                mappedRecord[mappedKey] = value;
            });
            
            return mappedRecord;
        });
    }

    extractFeaturesAndLabels(data) {
        const features = [];
        const labels = [];
        
        data.forEach(record => {
            // Extract feature vector
            const featureVector = this.requiredFeatures.map(feature => {
                let value = record[feature];
                
                // Handle different data types
                if (typeof value === 'string') {
                    value = parseFloat(value);
                }
                
                return isNaN(value) ? null : value;
            });
            
            // Extract label
            let label = null;
            if (record.classification) {
                label = this.classificationMap[record.classification];
                if (label === undefined) {
                    // Try to infer from string patterns
                    const classification = record.classification.toString().toUpperCase();
                    if (classification.includes('CONFIRMED') || classification.includes('CP')) {
                        label = 2;
                    } else if (classification.includes('CANDIDATE') || classification.includes('PC')) {
                        label = 1;
                    } else if (classification.includes('FALSE') || classification.includes('FP')) {
                        label = 0;
                    }
                }
            }
            
            features.push(featureVector);
            labels.push(label);
        });
        
        return { features, labels };
    }

    cleanFeatures(features) {
        const cleanedFeatures = [];
        const featureStats = this.calculateFeatureStatistics(features);
        
        features.forEach(featureVector => {
            const cleanedVector = featureVector.map((value, index) => {
                if (value === null || isNaN(value)) {
                    // Use median for missing values
                    return featureStats[index].median;
                }
                
                // Handle outliers (beyond 3 standard deviations)
                const stats = featureStats[index];
                if (Math.abs(value - stats.mean) > 3 * stats.std) {
                    return stats.median; // Replace outliers with median
                }
                
                // Apply log transformation to skewed features
                if (['period', 'depth'].includes(this.requiredFeatures[index])) {
                    return Math.log10(Math.max(value, 1e-10));
                }
                
                return value;
            });
            
            cleanedFeatures.push(cleanedVector);
        });
        
        return cleanedFeatures;
    }

    calculateFeatureStatistics(features) {
        const numFeatures = features[0].length;
        const stats = [];
        
        for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            const values = features
                .map(vector => vector[featureIndex])
                .filter(value => value !== null && !isNaN(value));
            
            if (values.length === 0) {
                stats.push({ mean: 0, std: 1, median: 0, min: 0, max: 1 });
                continue;
            }
            
            values.sort((a, b) => a - b);
            
            const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
            const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
            const std = Math.sqrt(variance);
            const median = values[Math.floor(values.length / 2)];
            const min = values[0];
            const max = values[values.length - 1];
            
            stats.push({ mean, std, median, min, max });
        }
        
        return stats;
    }

    filterValidRecords(features, labels) {
        const validIndices = [];
        
        features.forEach((featureVector, index) => {
            // Check if at least 3 features are valid
            const validFeatureCount = featureVector.filter(value => 
                value !== null && !isNaN(value)
            ).length;
            
            if (validFeatureCount >= 3) {
                validIndices.push(index);
            }
        });
        
        const validFeatures = validIndices.map(i => features[i]);
        const validLabels = validIndices.map(i => labels[i]);
        
        return { validFeatures, validLabels };
    }

    generateMockDataset(numSamples = 1000) {
        console.log(`Generating ${numSamples} mock samples...`);
        
        const features = [];
        
        for (let i = 0; i < numSamples; i++) {
            let featureVector;
            const objectType = Math.random();
            
            if (objectType < 0.4) {
                // Confirmed exoplanets
                featureVector = [
                    Math.log10(Math.random() * 2000 + 1), // log period
                    Math.random() * 20 + 2,               // duration hours  
                    Math.log10(Math.random() * 10000 + 100), // log depth ppm
                    Math.random() * 2 + 0.5,              // stellar radius
                    Math.random() * 3000 + 4000,          // stellar temp
                    Math.random() * 0.8                   // impact parameter
                ];
            } else if (objectType < 0.7) {
                // Planet candidates
                featureVector = [
                    Math.log10(Math.random() * 1000 + 1),
                    Math.random() * 15 + 3,
                    Math.log10(Math.random() * 5000 + 50),
                    Math.random() * 3 + 0.3,
                    Math.random() * 4000 + 3500,
                    Math.random() * 1.0
                ];
            } else {
                // False positives
                featureVector = [
                    Math.log10(Math.random() * 5000 + 0.1),
                    Math.random() * 50 + 0.5,
                    Math.log10(Math.random() * 50000 + 10),
                    Math.random() * 5 + 0.1,
                    Math.random() * 5000 + 2500,
                    Math.random() * 1.2
                ];
            }
            
            // Add correlations and noise
            featureVector[0] += Math.random() * 0.2 - 0.1; // period noise
            featureVector[1] += featureVector[0] * 0.5 + Math.random() * 2 - 1; // duration correlation
            featureVector[2] += Math.random() * 0.3 - 0.15; // depth noise
            
            features.push(featureVector);
        }
        
        return features;
    }

    generateMockLabels(numSamples = 1000) {
        const labels = [];
        
        for (let i = 0; i < numSamples; i++) {
            const rand = Math.random();
            if (rand < 0.4) {
                labels.push(2); // Confirmed
            } else if (rand < 0.7) {
                labels.push(1); // Candidate  
            } else {
                labels.push(0); // False Positive
            }
        }
        
        return labels;
    }

    async saveProcessedData(data, filename = 'processed_exoplanet_data.json') {
        const dataStr = JSON.stringify(data, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        
        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
        
        console.log(`Processed data saved as ${filename}`);
    }

    validateDataset(data) {
        const issues = [];
        
        if (!data.features || !Array.isArray(data.features)) {
            issues.push('Features array is missing or invalid');
        }
        
        if (data.features && data.features.length === 0) {
            issues.push('No feature data available');
        }
        
        if (data.features && data.features.length > 0) {
            const expectedLength = this.requiredFeatures.length;
            const invalidVectors = data.features.filter(vector => 
                !Array.isArray(vector) || vector.length !== expectedLength
            );
            
            if (invalidVectors.length > 0) {
                issues.push(`${invalidVectors.length} feature vectors have incorrect length`);
            }
        }
        
        if (data.labels && data.features && data.labels.length !== data.features.length) {
            issues.push('Feature and label arrays have different lengths');
        }
        
        return {
            valid: issues.length === 0,
            issues: issues
        };
    }

    getDatasetSummary(data) {
        if (!data.features) return null;
        
        const summary = {
            totalRecords: data.features.length,
            featureCount: this.requiredFeatures.length,
            featureNames: this.requiredFeatures
        };
        
        if (data.labels) {
            const labelCounts = data.labels.reduce((counts, label) => {
                counts[label] = (counts[label] || 0) + 1;
                return counts;
            }, {});
            
            summary.labelDistribution = {
                'False Positive': labelCounts[0] || 0,
                'Candidate': labelCounts[1] || 0,
                'Confirmed': labelCounts[2] || 0
            };
        }
        
        return summary;
    }
}