import Papa from 'papaparse';

export class DataLoader {
    constructor() {
        this.supportedFormats = ['csv', 'json'];
    }

    async loadFile(file) {
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        if (!this.supportedFormats.includes(fileExtension)) {
            throw new Error(`Unsupported file format: ${fileExtension}. Supported: ${this.supportedFormats.join(', ')}`);
        }
        
        return fileExtension === 'csv' 
            ? this.loadCSV(file) 
            : this.loadJSON(file);
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.warn('CSV parsing warnings:', results.errors);
                    }
                    resolve({
                        data: results.data,
                        meta: results.meta,
                        format: 'csv',
                        filename: file.name,
                        size: file.size,
                        rowCount: results.data.length
                    });
                },
                error: (error) => {
                    reject(new Error(`CSV parsing failed: ${error.message}`));
                },
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true,
                transformHeader: (header) => header.trim()
            });
        });
    }

    async loadJSON(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const data = JSON.parse(event.target.result);
                    const dataArray = Array.isArray(data) ? data : [data];
                    
                    resolve({
                        data: dataArray,
                        meta: {
                            fields: dataArray.length > 0 ? Object.keys(dataArray[0]) : []
                        },
                        format: 'json',
                        filename: file.name,
                        size: file.size,
                        rowCount: dataArray.length
                    });
                } catch (error) {
                    reject(new Error(`JSON parsing failed: ${error.message}`));
                }
            };
            
            reader.onerror = () => {
                reject(new Error('Failed to read file'));
            };
            
            reader.readAsText(file);
        });
    }

    async loadMultipleFiles(files) {
        const results = await Promise.allSettled(
            Array.from(files).map(file => this.loadFile(file))
        );
        
        return {
            successful: results
                .filter(r => r.status === 'fulfilled')
                .map(r => r.value),
            failed: results
                .filter(r => r.status === 'rejected')
                .map(r => r.reason.message)
        };
    }

    getDataSummary(loadedData) {
        if (!loadedData || !loadedData.data) return null;
        
        const { data, meta } = loadedData;
        
        return {
            filename: loadedData.filename,
            format: loadedData.format,
            fileSize: `${(loadedData.size / 1024).toFixed(2)} KB`,
            rowCount: data.length,
            columnCount: meta.fields?.length || 0,
            columns: meta.fields || Object.keys(data[0] || {}),
            sampleRow: data[0] || null
        };
    }

    exportToJSON(loadedData, filename = 'export.json') {
        if (!loadedData || !loadedData.data) {
            throw new Error('No data to export');
        }
        
        const exportData = {
            metadata: {
                format: loadedData.format,
                filename: loadedData.filename,
                size: loadedData.size,
                rowCount: loadedData.rowCount,
                exportDate: new Date().toISOString()
            },
            data: loadedData.data,
            features: loadedData.meta?.fields || []
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        this.downloadFile(dataStr, filename, 'application/json');
    }

    exportToCSV(loadedData, filename = 'export.csv') {
        if (!loadedData || !loadedData.data) {
            throw new Error('No data to export');
        }
        
        const csv = Papa.unparse(loadedData.data, {
            header: true,
            skipEmptyLines: true
        });
        
        this.downloadFile(csv, filename, 'text/csv');
    }

    // Méthode utilitaire pour le téléchargement
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }
}
