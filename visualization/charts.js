import * as Plotly from 'plotly.js-dist';

export class Visualizer {
    constructor() {
        this.plotConfigs = {
            defaultLayout: {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#ffffff' },
                showlegend: true
            },
            colors: {
                'CONFIRMED': '#00ff00',
                'CANDIDATE': '#ffa500',
                'FALSE POSITIVE': '#ff0000',
                primary: '#00d4ff',
                secondary: '#ff0080'
            }
        };

        // Known plot ids to purge on dispose
        this.knownPlotIds = new Set([
            'pca-biplot',
            'feature-importance',
            'model-comparison',
            'roc-curve',
            'light-curve',
            'batch-chart',
            'batch-results-chart',
            'split-visualization'
        ]);
    }

    initializePlots() {
        this.generateROCCurve();
        this.generateFeatureImportance();
        this.generateSampleLightCurve();
        console.log('Visualization plots initialized');
    }

    // ---------- Helpers ----------
    async _toArrayIfTensor(maybeTensor) {
        try {
            if (!maybeTensor) return null;
            if (typeof maybeTensor.array === 'function') {
                // async tensor
                return await maybeTensor.array();
            } else if (typeof maybeTensor.arraySync === 'function') {
                return maybeTensor.arraySync();
            } else {
                return maybeTensor;
            }
        } catch (err) {
            console.warn('Failed to convert tensor to array:', err);
            return maybeTensor;
        }
    }

    _labelsToIndices(labels) {
        // labels may be: [0,1,2,...] or one-hot arrays [[0,1,0],...]
        if (!labels) return [];
        if (!Array.isArray(labels)) return labels;

        // detect one-hot: first element is array
        if (Array.isArray(labels[0])) {
            return labels.map(l => {
                const idx = l.findIndex(v => v === 1);
                return idx >= 0 ? idx : l.indexOf(Math.max(...l));
            });
        } else {
            return labels;
        }
    }

    _ensure3dPoint(pt) {
        if (!pt) return [0,0,0];
        return [pt[0] ?? 0, pt[1] ?? 0, pt[2] ?? 0];
    }

    // ---------- PCA / scatter ----------
    async updatePCAPlot(features, labels) {
        try {
            const featuresArray = await this._toArrayIfTensor(features) || [];
            const labelsArray = await this._toArrayIfTensor(labels) || [];
            
            const traces = this.createClassTraces(featuresArray, labelsArray);
            
            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: 'PCA Visualization (3D)',
                scene: {
                    xaxis: { title: 'PC1' },
                    yaxis: { title: 'PC2' },
                    zaxis: { title: 'PC3' },
                    bgcolor: 'rgba(0,0,0,0)'
                },
                margin: { l: 0, r: 0, b: 0, t: 40 }
            };
            
            const element = document.getElementById('pca-biplot');
            if (element) await Plotly.newPlot(element, traces, layout, {responsive: true});
            
        } catch (error) {
            console.error('Error updating PCA plot:', error);
        }
    }

    createClassTraces(features, labels) {
        const classData = {
            0: { name: 'FALSE POSITIVE', color: this.plotConfigs.colors['FALSE POSITIVE'], points: [] },
            1: { name: 'CANDIDATE', color: this.plotConfigs.colors['CANDIDATE'], points: [] },
            2: { name: 'CONFIRMED', color: this.plotConfigs.colors['CONFIRMED'], points: [] }
        };
        
        const labelIndices = this._labelsToIndices(labels);
        
        features.forEach((feature, index) => {
            const label = labelIndices[index];
            if (classData[label]) {
                classData[label].points.push(feature);
            }
        });
        
        const traces = [];
        Object.entries(classData).forEach(([classId, classInfo]) => {
            if (classInfo.points.length > 0) {
                const trace = {
                    x: classInfo.points.map(p => p[0]),
                    y: classInfo.points.map(p => p[1]),
                    z: classInfo.points.map(p => p[2] || 0),
                    mode: 'markers',
                    type: 'scatter3d',
                    name: classInfo.name,
                    marker: {
                        color: classInfo.color,
                        size: 4,
                        opacity: 0.8,
                        symbol: 'circle'
                    }
                };
                traces.push(trace);
            }
        });
        
        return traces;
    }

    // ---------- Feature importance / model comparison (existing) ----------
    async updateFeatureImportance(model, nComponents = null) {
        try {
            let importances = null;
            let featureNames = [];
            
            if (model && typeof model.getFeatureImportances === 'function') {
                importances = model.getFeatureImportances();
                
                const nFeatures = importances.length;
                
                if (nComponents && nFeatures === nComponents) {
                    featureNames = Array.from({length: nFeatures}, (_, i) => `PC${i + 1}`);
                } else {
                    featureNames = Array.from({length: nFeatures}, (_, i) => `Feature ${i + 1}`);
                }
            } else {
                importances = [0.45, 0.25, 0.15, 0.08, 0.04, 0.03];
                featureNames = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'];
            }
            
            const maxDisplay = 15;
            if (importances.length > maxDisplay) {
                const indexed = importances.map((val, idx) => ({val, idx}));
                indexed.sort((a, b) => b.val - a.val);
                const topIndices = indexed.slice(0, maxDisplay).map(x => x.idx);
                
                importances = topIndices.map(idx => importances[idx]);
                featureNames = topIndices.map(idx => featureNames[idx]);
            }
            
            const trace = {
                x: importances,
                y: featureNames,
                type: 'bar',
                orientation: 'h',
                marker: { 
                    color: importances,
                    colorscale: 'Viridis',
                    showscale: true,
                    colorbar: { title: 'Importance' }
                }
            };
            
            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: `Feature Importance - ${model?.name || 'Model'}`,
                xaxis: { title: 'Importance Score' },
                yaxis: { title: 'Features', automargin: true },
                height: Math.max(400, featureNames.length * 25)
            };
            
            const element = document.getElementById('feature-importance');
            if (element) {
                await Plotly.newPlot(element, [trace], layout, {responsive: true});
            }
            
        } catch (error) {
            console.error('Error updating feature importance:', error);
        }
    }

    async updateModelComparison(modelsMap) {
        try {
            const modelNames = [];
            const accuracies = [];
            const precisions = [];
            const recalls = [];
            const f1Scores = [];
            
            const modelsArray = modelsMap instanceof Map ? 
                Array.from(modelsMap.entries()) : 
                Object.entries(modelsMap);
            
            for (const [name, model] of modelsArray) {
                if (model && model.trained) {
                    const metrics = model.getMetrics();
                    modelNames.push(name);
                    accuracies.push(metrics.accuracy || 0);
                    precisions.push(metrics.precision || 0);
                    recalls.push(metrics.recall || 0);
                    f1Scores.push(metrics.f1 || 0);
                }
            }
            
            if (modelNames.length === 0) {
                console.log('No trained models to compare');
                return;
            }
            
            const traces = [
                {
                    x: modelNames,
                    y: accuracies,
                    name: 'Accuracy',
                    type: 'bar',
                    marker: { color: this.plotConfigs.colors.primary }
                },
                {
                    x: modelNames,
                    y: precisions,
                    name: 'Precision', 
                    type: 'bar',
                    marker: { color: this.plotConfigs.colors.secondary }
                },
                {
                    x: modelNames,
                    y: recalls,
                    name: 'Recall',
                    type: 'bar',
                    marker: { color: '#00ff00' }
                },
                {
                    x: modelNames,
                    y: f1Scores,
                    name: 'F1 Score',
                    type: 'bar',
                    marker: { color: '#ffaa00' }
                }
            ];
            
            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: 'Model Performance Comparison',
                xaxis: { title: 'Models', automargin: true },
                yaxis: { title: 'Score', range: [0, 1] },
                barmode: 'group',
                height: 400
            };
            
            const element = document.getElementById('model-comparison');
            if (element) {
                await Plotly.newPlot(element, traces, layout, {responsive: true});
            }
            
        } catch (error) {
            console.error('Error updating model comparison:', error);
        }
    }

    // ---------- ROC, sample light curve (existing) ----------
    generateROCCurve(metrics = { recall: 0.981 }) {
        const x = [];
        const y = [];
        
        for (let i = 0; i <= 100; i++) {
            const fpr = i / 100;
            const tpr = Math.min(1, metrics.recall * (1 - Math.exp(-fpr * 5)));
            x.push(fpr);
            y.push(tpr);
        }
        
        const traces = [
            {
                x: x,
                y: y,
                type: 'scatter',
                mode: 'lines',
                name: 'ROC Curve',
                line: { color: this.plotConfigs.colors.primary, width: 3 }
            },
            {
                x: [0, 1],
                y: [0, 1],
                type: 'scatter',
                mode: 'lines',
                name: 'Random Classifier',
                line: { color: 'rgba(255,255,255,0.3)', width: 2, dash: 'dash' },
                showlegend: false
            }
        ];
        
        const layout = {
            ...this.plotConfigs.defaultLayout,
            title: 'ROC Curve',
            xaxis: { title: 'False Positive Rate', range: [0, 1] },
            yaxis: { title: 'True Positive Rate', range: [0, 1] },
            height: 320
        };
        
        const element = document.getElementById('roc-curve');
        if (element) {
            Plotly.newPlot(element, traces, layout, {responsive: true});
        }
    }

    generateFeatureImportance() {
        const features = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'];
        const importance = [0.45, 0.25, 0.15, 0.08, 0.04, 0.03];
        
        const trace = {
            x: importance,
            y: features,
            type: 'bar',
            orientation: 'h',
            marker: { 
                color: importance,
                colorscale: 'Viridis',
                showscale: true
            }
        };
        
        const layout = {
            ...this.plotConfigs.defaultLayout,
            title: 'PCA Component Importance',
            xaxis: { title: 'Explained Variance Ratio' },
            yaxis: { title: 'Principal Components' },
            height: 300
        };
        
        const element = document.getElementById('feature-importance');
        if (element) {
            Plotly.newPlot(element, [trace], layout, {responsive: true});
        }
    }

    // ---------- Split visualization (NEW) ----------
    generateSplitVisualization(splits_stats) {
        try {
            if (!splits_stats) {
                console.warn('No split stats provided to generateSplitVisualization');
                return;
            }

            // build grouped bar for train/test confirmed vs false positive
            const categories = ['Confirmed', 'False Positive'];
            const trainVals = [splits_stats.train.confirmed || 0, splits_stats.train.falsePositive || 0];
            const testVals = [splits_stats.test.confirmed || 0, splits_stats.test.falsePositive || 0];
            const candidateCount = splits_stats.candidates || 0;

            const traces = [
                {
                    x: categories,
                    y: trainVals,
                    name: 'Train',
                    type: 'bar',
                    marker: { color: this.plotConfigs.colors.primary }
                },
                {
                    x: categories,
                    y: testVals,
                    name: 'Test',
                    type: 'bar',
                    marker: { color: this.plotConfigs.colors.secondary }
                }
            ];

            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: 'Data Split - Class Distribution',
                barmode: 'group',
                height: 300
            };

            // ensure container exists or create
            let element = document.getElementById('split-visualization');
            if (!element) {
                const container = document.querySelector('.data-info') || document.querySelector('.container');
                if (container) {
                    const div = document.createElement('div');
                    div.id = 'split-visualization';
                    div.className = 'panel';
                    container.appendChild(div);
                    element = div;
                }
            }

            if (element) {
                Plotly.newPlot(element, traces, layout, {responsive: true});
            }

            // Donut summarizing counts (candidates & totals)
            const donutElementId = 'split-summary-donut';
            let donutElement = document.getElementById(donutElementId);
            if (!donutElement && element && element.parentElement) {
                const d = document.createElement('div');
                d.id = donutElementId;
                d.style.width = '220px';
                d.style.height = '220px';
                d.style.display = 'inline-block';
                d.style.marginLeft = '20px';
                element.parentElement.appendChild(d);
                donutElement = d;
            }

            if (donutElement) {
                const totalTrain = splits_stats.train.total || (trainVals[0] + trainVals[1]);
                const totalTest = splits_stats.test.total || (testVals[0] + testVals[1]);
                const values = [totalTrain, totalTest, candidateCount];
                const labels = ['Train', 'Test', 'Candidates'];
                const donutTrace = {
                    labels,
                    values,
                    type: 'pie',
                    hole: 0.5,
                    marker: { colors: [this.plotConfigs.colors.primary, this.plotConfigs.colors.secondary, this.plotConfigs.colors.CANDIDATE] },
                    textinfo: 'label+value'
                };
                const donutLayout = { ...this.plotConfigs.defaultLayout, title: 'Split Summary', height: 240, width: 240 };
                Plotly.newPlot(donutElement, [donutTrace], donutLayout, {responsive: true});
            }

        } catch (error) {
            console.error('Error generating split visualization:', error);
        }
    }

    // ---------- PCA plot by split (NEW) ----------
    async updatePCAPlotBySplit(trainingData, testData, candidatesData) {
        try {
            // Accept either {features, labels} or arrays directly
            const trainFeat = await this._toArrayIfTensor(trainingData?.features || trainingData || []);
            const trainLab = await this._toArrayIfTensor(trainingData?.labels || []);
            const testFeat = await this._toArrayIfTensor(testData?.features || testData || []);
            const testLab = await this._toArrayIfTensor(testData?.labels || []);
            const candFeat = await this._toArrayIfTensor(candidatesData?.features || candidatesData || []);
            const candLab = await this._toArrayIfTensor(candidatesData?.labels || []);

            const trainLabels = this._labelsToIndices(trainLab);
            const testLabels = this._labelsToIndices(testLab);
            const candLabels = this._labelsToIndices(candLab);

            const buildTraceFromSet = (featArr, labArr, setName, markerSpec) => {
                if (!featArr || featArr.length === 0) return null;
                // group by class to have consistent legend
                const groups = {};
                featArr.forEach((pt, i) => {
                    const cls = labArr?.[i] ?? -1;
                    if (!groups[cls]) groups[cls] = [];
                    groups[cls].push(this._ensure3dPoint(pt));
                });
                const traces = [];
                Object.entries(groups).forEach(([cls, pts]) => {
                    const name = cls === '0' ? `${setName} - FALSE POSITIVE` : cls === '1' ? `${setName} - CANDIDATE` : cls === '2' ? `${setName} - CONFIRMED` : `${setName} - Unknown`;
                    traces.push({
                        x: pts.map(p => p[0]),
                        y: pts.map(p => p[1]),
                        z: pts.map(p => p[2]),
                        mode: 'markers',
                        type: 'scatter3d',
                        name,
                        marker: {
                            size: markerSpec.size,
                            symbol: markerSpec.symbol,
                            color: markerSpec.colorMap?.[cls] ?? markerSpec.color,
                            opacity: markerSpec.opacity
                        },
                        showlegend: true
                    });
                });
                return traces;
            };

            const trainTraces = buildTraceFromSet(trainFeat, trainLabels, 'Train', {
                size: 3, symbol: 'circle', color: this.plotConfigs.colors.primary, opacity: 0.7,
                colorMap: { 0: this.plotConfigs.colors['FALSE POSITIVE'], 1: this.plotConfigs.colors['CANDIDATE'], 2: this.plotConfigs.colors['CONFIRMED'] }
            }) || [];

            const testTraces = buildTraceFromSet(testFeat, testLabels, 'Test', {
                size: 4, symbol: 'diamond', color: this.plotConfigs.colors.secondary, opacity: 0.9,
                colorMap: { 0: this.plotConfigs.colors['FALSE POSITIVE'], 1: this.plotConfigs.colors['CANDIDATE'], 2: this.plotConfigs.colors['CONFIRMED'] }
            }) || [];

            const candTraces = buildTraceFromSet(candFeat, candLabels, 'Candidates', {
                size: 3, symbol: 'cross', color: '#ffffff', opacity: 0.85,
                colorMap: { 0: this.plotConfigs.colors['FALSE POSITIVE'], 1: this.plotConfigs.colors['CANDIDATE'], 2: this.plotConfigs.colors['CONFIRMED'] }
            }) || [];

            const traces = [...trainTraces, ...testTraces, ...candTraces];
            if (traces.length === 0) {
                console.warn('No PCA points to plot in updatePCAPlotBySplit');
                return;
            }

            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: 'PCA by Split (3D)',
                scene: { xaxis: { title: 'PC1' }, yaxis: { title: 'PC2' }, zaxis: { title: 'PC3' }, bgcolor: 'rgba(0,0,0,0)' },
                margin: { t: 40 }
            };

            const element = document.getElementById('pca-biplot');
            if (element) {
                await Plotly.newPlot(element, traces, layout, {responsive: true});
            }

        } catch (error) {
            console.error('Error in updatePCAPlotBySplit:', error);
        }
    }

    // ---------- Sample light curve (existing) ----------
    generateSampleLightCurve(params = { period: 365, depth: 1000, duration: 13 }) {
        const time = [];
        const flux = [];
        const { period, depth, duration } = params;
        
        for (let i = 0; i < 200; i++) {
            const t = i * period / 200;
            let f = 1.0;
            
            const phase = (t % period) / period;
            if (Math.abs(phase - 0.5) < (duration/24) / (2 * period)) {
                f -= depth / 1000000;
            }
            
            f += (Math.random() - 0.5) * 0.0001;
            f += 0.00005 * Math.sin(2 * Math.PI * t / (period * 0.3));
            
            time.push(t);
            flux.push(f);
        }
        
        const trace = {
            x: time,
            y: flux,
            type: 'scatter',
            mode: 'lines',
            name: 'Simulated Light Curve',
            line: { color: this.plotConfigs.colors.primary, width: 2 }
        };
        
        const layout = {
            ...this.plotConfigs.defaultLayout,
            title: 'Exoplanet Transit Light Curve',
            xaxis: { title: 'Time (days)' },
            yaxis: { title: 'Relative Flux' },
            annotations: [{
                x: period / 2,
                y: flux[Math.floor(flux.length / 2)] - depth / 2000000,
                text: 'Transit Event',
                arrowhead: 2,
                arrowcolor: this.plotConfigs.colors.secondary,
                font: { color: this.plotConfigs.colors.secondary }
            }],
            height: 320
        };
        
        const element = document.getElementById('light-curve');
        if (element) {
            Plotly.newPlot(element, [trace], layout, {responsive: true});
        }
    }

    // ---------- Batch chart (slightly more flexible) ----------
    generateBatchChart(results) {
        try {
            const classifications = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'];
            const counts = classifications.map(classification => 
                results.filter(r => r.result.classification === classification).length
            );
            
            const colors = [
                this.plotConfigs.colors['FALSE POSITIVE'],
                this.plotConfigs.colors['CANDIDATE'],
                this.plotConfigs.colors['CONFIRMED']
            ];

            const trace = {
                labels: classifications,
                values: counts,
                type: 'pie',
                marker: { 
                    colors: colors,
                    line: { color: '#ffffff', width: 2 }
                },
                textfont: { color: '#ffffff', size: 14 },
                textinfo: 'label+percent+value',
                hovertemplate: '<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            };
            
            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: 'Classification Distribution',
                showlegend: true,
                legend: {
                    orientation: 'v',
                    x: 1,
                    y: 0.5
                },
                height: 360
            };

            // app.js sometimes renders into 'batch-results-chart' - prefer that if present
            const preferredIds = ['batch-results-chart', 'batch-chart'];
            let plotted = false;
            for (const id of preferredIds) {
                const el = document.getElementById(id);
                if (el) {
                    Plotly.newPlot(el, [trace], layout, {responsive: true});
                    plotted = true;
                    break;
                }
            }

            if (!plotted) {
                console.warn('No container found for batch chart (tried batch-results-chart and batch-chart)');
            }

        } catch (error) {
            console.error('Error generating batch chart:', error);
        }
    }

    // ---------- Resize & dispose ----------
    resizePlots() {
        const plotIds = ['pca-biplot', 'feature-importance', 'model-comparison', 'roc-curve', 'light-curve', 'batch-chart', 'batch-results-chart', 'split-visualization'];
        plotIds.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                try {
                    Plotly.Plots.resize(element);
                } catch (err) {
                    console.warn('Plotly resize failed for', id, err);
                }
            }
        });
    }

    dispose(){
        // purge known Plotly plots
        this.knownPlotIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                try {
                    Plotly.purge(el);
                } catch (err) {
                    // fallback: remove children
                    while (el.firstChild) el.removeChild(el.firstChild);
                }
            }
        });

        this.knownPlotIds.clear();
        console.log('Visualizer disposed');
    }
}

export const visualizer = new Visualizer();
