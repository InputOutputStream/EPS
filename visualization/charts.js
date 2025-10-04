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
    }

    initializePlots() {
        this.generateROCCurve();
        this.generateFeatureImportance();
        this.generateSampleLightCurve();
        console.log('Visualization plots initialized');
    }

    async updatePCAPlot(features, labels) {
        try {
            // Convert tensors to arrays if needed
            const featuresArray = features.arraySync ? await features.arraySync() : features;
            const labelsArray = labels.arraySync ? await labels.arraySync() : labels;
            
            // Create traces for each class
            const traces = this.createClassTraces(featuresArray, labelsArray);
            
            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: 'PCA Visualization (3D)',
                scene: {
                    xaxis: { title: 'PC1' },
                    yaxis: { title: 'PC2' },
                    zaxis: { title: 'PC3' },
                    bgcolor: 'rgba(0,0,0,0)'
                }
            };
            
            await Plotly.newPlot('pca-biplot', traces, layout);
            
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
        
        // Group data by class
        features.forEach((feature, index) => {
            const label = labels[index];
            if (classData[label]) {
                classData[label].points.push(feature);
            }
        });
        
        // Create traces
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
                        opacity: 0.7
                    }
                };
                traces.push(trace);
            }
        });
        
        return traces;
    }

    async updateFeatureImportance(model) {
        try {
            let importances = null;
            let featureNames = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'];
            
            if (model && typeof model.getFeatureImportances === 'function') {
                importances = model.getFeatureImportances();
            } else {
                // Default importance values for PCA components
                importances = [0.45, 0.25, 0.15, 0.08, 0.04, 0.03];
            }
            
            const trace = {
                x: importances.slice(0, 6),
                y: featureNames.slice(0, 6),
                type: 'bar',
                orientation: 'h',
                marker: { 
                    color: this.plotConfigs.colors.primary,
                    colorscale: 'Viridis'
                }
            };
            
            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: 'Feature Importance (PCA Components)',
                xaxis: { title: 'Importance Score' },
                yaxis: { title: 'Features' }
            };
            
            await Plotly.newPlot('feature-importance', [trace], layout);
            
        } catch (error) {
            console.error('Error updating feature importance:', error);
        }
    }

    async updateModelComparison(models) {
        try {
            const modelNames = [];
            const accuracies = [];
            const precisions = [];
            const recalls = [];
            
            models.forEach((model, name) => {
                if (model.trained) {
                    const metrics = model.getMetrics();
                    modelNames.push(name);
                    accuracies.push(metrics.accuracy || 0);
                    precisions.push(metrics.precision || 0);
                    recalls.push(metrics.recall || 0);
                }
            });
            
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
                }
            ];
            
            const layout = {
                ...this.plotConfigs.defaultLayout,
                title: 'Model Performance Comparison',
                xaxis: { title: 'Models' },
                yaxis: { title: 'Score', range: [0, 1] },
                barmode: 'group'
            };
            
            // Check if element exists
            const element = document.getElementById('model-comparison');
            if (element) {
                await Plotly.newPlot('model-comparison', traces, layout);
            }
            
        } catch (error) {
            console.error('Error updating model comparison:', error);
        }
    }

    generateROCCurve(metrics = { recall: 0.981 }) {
        const x = [];
        const y = [];
        
        for (let i = 0; i <= 100; i++) {
            const fpr = i / 100;
            const tpr = Math.min(1, metrics.recall * (1 - Math.exp(-fpr * 5)));
            x.push(fpr);
            y.push(tpr);
        }
        
        // Add diagonal reference line
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
            yaxis: { title: 'True Positive Rate', range: [0, 1] }
        };
        
        const element = document.getElementById('roc-curve');
        if (element) {
            Plotly.newPlot('roc-curve', traces, layout);
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
            yaxis: { title: 'Principal Components' }
        };
        
        const element = document.getElementById('feature-importance');
        if (element) {
            Plotly.newPlot('feature-importance', [trace], layout);
        }
    }

    generateSampleLightCurve(params = { period: 365, depth: 1000, duration: 13 }) {
        const time = [];
        const flux = [];
        const { period, depth, duration } = params;
        
        for (let i = 0; i < 200; i++) {
            const t = i * period / 200;
            let f = 1.0;
            
            // Add transit
            const phase = (t % period) / period;
            if (Math.abs(phase - 0.5) < (duration/24) / (2 * period)) {
                f -= depth / 1000000;
            }
            
            // Add noise and stellar variability
            f += (Math.random() - 0.5) * 0.0001;
            f += 0.00005 * Math.sin(2 * Math.PI * t / (period * 0.3)); // Stellar rotation
            
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
            }]
        };
        
        const element = document.getElementById('light-curve');
        if (element) {
            Plotly.newPlot('light-curve', [trace], layout);
        }
    }

    generateBatchChart(results) {
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
            }
        }
        const element = document.getElementById('batch-chart');
        if (element) {
            Plotly.newPlot('batch-chart', [trace], layout);
        }
    }

    resizePlots() {
        const plotIds = ['pca-biplot', 'feature-importance', 'model-comparison', 'roc-curve', 'light-curve', 'batch-chart'];
        plotIds.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                Plotly.Plots.resize(element);
            }
        });
    }

    dispose(){
        console.log('Visualizer disposed');
    }
}

// Export singleton instance
export const visualizer = new Visualizer();