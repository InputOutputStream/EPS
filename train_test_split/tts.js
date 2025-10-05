
export class TrainTestSplit {
    constructor(testSize = 0.2, randomSeed = 42) {
        if (testSize <= 0 || testSize >= 1) {
            throw new Error('testSize must be between 0 and 1');
        }
        this.testSize = testSize;
        this.randomSeed = randomSeed;
        this.rng = this.seededRandom(randomSeed);
    }

    seededRandom(seed) {
        let state = seed;
        return () => {
            state = (state * 1664525 + 1013904223) % 4294967296;
            return state / 4294967296;
        };
    }

    /**
     * Split data into train/test/candidates
     * @param {Array} features - Feature matrix [[f1, f2, ...], ...]
     * @param {Array} labels - Label array (0=FP, 1=CANDIDATE, 2=CONFIRMED)
     * @returns {Object} {train, test, candidates, stats}
     */
    split(features, labels) {
        // Validation
        if (!Array.isArray(features) || !Array.isArray(labels)) {
            throw new Error('Features and labels must be arrays');
        }
        
        if (features.length !== labels.length) {
            throw new Error(`Feature and label counts don't match: ${features.length} vs ${labels.length}`);
        }
        
        if (features.length === 0) {
            throw new Error('Cannot split empty dataset');
        }

        // Separate by label type
        const confirmed = [];
        const falsePositive = [];
        const candidates = [];

        features.forEach((feat, idx) => {
            const label = labels[idx];
            
            if (!Array.isArray(feat)) {
                throw new Error(`Invalid feature at index ${idx}: expected array, got ${typeof feat}`);
            }
            
            const item = { features: feat, label };

            if (label === 2) {
                confirmed.push(item);
            } else if (label === 0) {
                falsePositive.push(item);
            } else if (label === 1) {
                candidates.push(item);
            } else {
                console.warn(`Unknown label ${label} at index ${idx}, skipping`);
            }
        });

        // Validation
        if (confirmed.length === 0 && falsePositive.length === 0) {
            throw new Error('No confirmed or false positive data found for training');
        }

        console.log(`📊 Split input: ${confirmed.length} confirmed, ${falsePositive.length} FP, ${candidates.length} candidates`);

        // Shuffle with stratification
        const shuffledConfirmed = this.shuffle([...confirmed]);
        const shuffledFP = this.shuffle([...falsePositive]);

        // Split confirmed
        const confirmedSplit = Math.floor(confirmed.length * (1 - this.testSize));
        const trainConfirmed = shuffledConfirmed.slice(0, confirmedSplit);
        const testConfirmed = shuffledConfirmed.slice(confirmedSplit);

        // Split false positives
        const fpSplit = Math.floor(falsePositive.length * (1 - this.testSize));
        const trainFP = shuffledFP.slice(0, fpSplit);
        const testFP = shuffledFP.slice(fpSplit);

        // Ensure we have at least some data in each split
        if (trainConfirmed.length === 0 && trainFP.length === 0) {
            throw new Error('Training set would be empty with current split ratio');
        }

        // Combine and shuffle train/test
        const trainData = this.shuffle([...trainConfirmed, ...trainFP]);
        const testData = this.shuffle([...testConfirmed, ...testFP]);

        console.log(`✅ Split complete: ${trainData.length} train, ${testData.length} test, ${candidates.length} candidates`);

        const stats = {
            train: this.getStats(trainData.map(d => d.label)),
            test: this.getStats(testData.map(d => d.label)),
            candidates: candidates.length
        };

        return {
            train: {
                features: trainData.map(d => d.features),
                labels: trainData.map(d => d.label)
            },
            test: {
                features: testData.map(d => d.features),
                labels: testData.map(d => d.label)
            },
            candidates: {
                features: candidates.map(d => d.features),
                labels: candidates.map(d => d.label)
            },
            stats
        };
    }

    shuffle(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(this.rng() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    getStats(labels) {
        const counts = { 0: 0, 2: 0 }; // Only FP and Confirmed in train/test
        labels.forEach(l => {
            if (counts.hasOwnProperty(l)) {
                counts[l]++;
            }
        });
        
        return {
            falsePositive: counts[0],
            confirmed: counts[2],
            total: labels.length
        };
    }

    /**
     * K-fold stratified split for cross-validation
     * @param {Array} features - Feature matrix
     * @param {Array} labels - Label array
     * @param {number} k - Number of folds
     * @returns {Array} Array of fold objects
     */
    kFold(features, labels, k = 5) {
        if (k < 2) {
            throw new Error('k must be at least 2');
        }
        
        if (features.length < k) {
            throw new Error(`Not enough samples (${features.length}) for ${k} folds`);
        }

        const confirmed = [];
        const falsePositive = [];

        features.forEach((feat, idx) => {
            const label = labels[idx];
            if (label === 2) {
                confirmed.push({ features: feat, label });
            } else if (label === 0) {
                falsePositive.push({ features: feat, label });
            }
        });

        if (confirmed.length < k || falsePositive.length < k) {
            console.warn(`Warning: Not enough samples in one or both classes for ${k} folds`);
        }

        const shuffledConfirmed = this.shuffle(confirmed);
        const shuffledFP = this.shuffle(falsePositive);

        const folds = [];
        const confirmedFoldSize = Math.ceil(confirmed.length / k);
        const fpFoldSize = Math.ceil(falsePositive.length / k);

        for (let i = 0; i < k; i++) {
            const confirmedStart = i * confirmedFoldSize;
            const confirmedEnd = Math.min((i + 1) * confirmedFoldSize, confirmed.length);
            const confirmedFold = shuffledConfirmed.slice(confirmedStart, confirmedEnd);
            
            const fpStart = i * fpFoldSize;
            const fpEnd = Math.min((i + 1) * fpFoldSize, falsePositive.length);
            const fpFold = shuffledFP.slice(fpStart, fpEnd);

            const fold = this.shuffle([...confirmedFold, ...fpFold]);
            
            if (fold.length > 0) {
                folds.push({
                    features: fold.map(d => d.features),
                    labels: fold.map(d => d.label),
                    stats: this.getStats(fold.map(d => d.label))
                });
            }
        }

        console.log(`🔀 Created ${folds.length} folds for cross-validation`);
        return folds;
    }

    /**
     * Export split to JSON format
     */
    exportToJSON(splitData) {
        return {
            train: {
                features: splitData.train.features,
                labels: splitData.train.labels,
                count: splitData.train.features.length
            },
            test: {
                features: splitData.test.features,
                labels: splitData.test.labels,
                count: splitData.test.features.length
            },
            candidates: {
                features: splitData.candidates.features,
                labels: splitData.candidates.labels,
                count: splitData.candidates.features.length
            },
            stats: splitData.stats,
            metadata: {
                testSize: this.testSize,
                randomSeed: this.randomSeed,
                createdAt: new Date().toISOString()
            }
        };
    }
}

// Helper function for quick splitting
export function trainTestSplit(features, labels, testSize = 0.2, randomSeed = 42) {
    const splitter = new TrainTestSplit(testSize, randomSeed);
    return splitter.split(features, labels);
}