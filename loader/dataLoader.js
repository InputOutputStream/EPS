import Papa from 'papaparse';

export class DataLoader {
    constructor() {
        this.supportedFormats = ['csv', 'json', 'npy'];
        this.splitData = {
            train: null,
            test: null,
            candidates: null
        };
    }

    async loadFile(file) {
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        if (!this.supportedFormats.includes(fileExtension)) {
            throw new Error(`Unsupported file format: ${fileExtension}`);
        }
        
        const result = await (async () => {
            switch (fileExtension) {
                case 'csv':
                    return this.loadCSV(file);
                case 'json':
                    return this.loadJSON(file);
                case 'npy':
                    return this.loadNPY(file);
                default:
                    throw new Error(`Unsupported format: ${fileExtension}`);
            }
        })();

        return {
            ...result,
            isPCAData: fileExtension === 'npy' || (fileExtension === 'json' && !result.isPCAParams)
        };
    }

    async loadNPY(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = async (event) => {
                try {
                    const arrayBuffer = event.target.result;
                    const npyData = this.parseNPY(arrayBuffer);
                    
                    resolve({
                        data: npyData.data,
                        shape: npyData.shape,
                        dtype: npyData.dtype,
                        format: 'npy',
                        filename: file.name,
                        size: file.size,
                        type: this.inferNPYType(file.name)
                    });
                } catch (error) {
                    reject(new Error(`NPY parsing failed: ${error.message}`));
                }
            };
            
            reader.onerror = () => reject(new Error('Failed to read NPY file'));
            reader.readAsArrayBuffer(file);
        });
    }

    parseNPY(arrayBuffer) {
        const view = new DataView(arrayBuffer);
        
        const magic = String.fromCharCode(view.getUint8(0)) + 
                     String.fromCharCode(view.getUint8(1)) + 
                     String.fromCharCode(view.getUint8(2)) + 
                     String.fromCharCode(view.getUint8(3)) + 
                     String.fromCharCode(view.getUint8(4)) + 
                     String.fromCharCode(view.getUint8(5));
        
        if (magic !== '\x93NUMPY') {
            throw new Error('Invalid NPY file format');
        }
        
        const major = view.getUint8(6);
        const minor = view.getUint8(7);
        
        let headerLen;
        if (major === 1) {
            headerLen = view.getUint16(8, true);
        } else if (major === 2 || major === 3) {
            headerLen = view.getUint32(8, true);
        } else {
            throw new Error('Unsupported NPY version');
        }
        
        const headerStart = major === 1 ? 10 : 12;
        const headerBytes = new Uint8Array(arrayBuffer, headerStart, headerLen);
        const headerStr = new TextDecoder().decode(headerBytes);
        
        const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
        const dtypeMatch = headerStr.match(/'descr':\s*'([^']*)'/);
        
        if (!shapeMatch || !dtypeMatch) {
            throw new Error('Invalid NPY header format');
        }
        
        const shape = shapeMatch[1]
            .split(',')
            .map(s => s.trim())
            .filter(s => s)
            .map(Number);
        
        const dtype = dtypeMatch[1];
        
        const dataStart = headerStart + headerLen;
        const dataBytes = new Uint8Array(arrayBuffer, dataStart);
        
        let data;
        const totalElements = shape.reduce((a, b) => a * b, 1);
        
        if (dtype.includes('f4') || dtype === '<f4') {
            data = new Float32Array(dataBytes.buffer, dataBytes.byteOffset, totalElements);
        } else if (dtype.includes('f8') || dtype === '<f8') {
            data = new Float64Array(dataBytes.buffer, dataBytes.byteOffset, totalElements);
        } else if (dtype.includes('i4') || dtype === '<i4') {
            data = new Int32Array(dataBytes.buffer, dataBytes.byteOffset, totalElements);
        } else if (dtype.includes('i8') || dtype === '<i8') {
            data = new BigInt64Array(dataBytes.buffer, dataBytes.byteOffset, totalElements);
        } else {
            throw new Error(`Unsupported dtype: ${dtype}`);
        }
        
        return {
            data: Array.from(data),
            shape,
            dtype
        };
    }

    inferNPYType(filename) {
        const lower = filename.toLowerCase();
        if (lower.includes('x_train_pca') || lower.includes('x_train')) return 'X_train_pca';
        if (lower.includes('x_test_pca') || lower.includes('x_test')) return 'X_test_pca';
        if (lower.includes('x_candidates_pca') || lower.includes('x_candidates')) return 'X_candidates_pca';
        if (lower.includes('y_train')) return 'y_train';
        if (lower.includes('y_test')) return 'y_test';
        if (lower.includes('y_candidates')) return 'y_candidates';
        if (lower.includes('pca_params')) return 'pca_params';
        return 'unknown';
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                complete: (results) => {
                    try {
                        if (results.errors.length > 0) {
                            console.warn('CSV parsing warnings:', results.errors);
                        }

                        const { data, meta } = results;
                        if (!data || data.length === 0) {
                            throw new Error('No data rows found in CSV');
                        }

                        const header = meta.fields || Object.keys(data[0] || {});
                        if (header.length < 2) {
                            throw new Error('CSV must have at least one feature and one label column');
                        }

                        const filteredData = data.filter(row => row && Object.keys(row).length > 0);

                        if (filteredData.length === 0) {
                            throw new Error('No valid rows in CSV after filtering');
                        }

                        resolve({
                            data: filteredData,
                            format: 'csv',
                            filename: file.name,
                            size: file.size,
                            rowCount: filteredData.length,
                            header
                        });
                    } catch (error) {
                        reject(new Error(`CSV parsing failed: ${error.message}`));
                    }
                },
                error: (error) => {
                    reject(new Error(`CSV parsing failed: ${error.message}`));
                },
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true,
                transformHeader: (header) => header.trim(),
                comments: '#'
            });
        });
    }

    async loadJSON(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (event) => {
                try {
                    const data = JSON.parse(event.target.result);
                    
                    const isPCAParams = file.name.includes('pca_params');
                    const isSplitStats = file.name.includes('split_stats');
                    
                    resolve({
                        data: data,
                        format: 'json',
                        filename: file.name,
                        size: file.size,
                        type: isPCAParams ? 'pca_params' : (isSplitStats ? 'split_stats' : 'generic'),
                        isPCAParams,
                        isSplitStats
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

    async loadSplitDataset(files) {
        const results = {
            X_train_pca: null,
            X_test_pca: null,
            X_candidates_pca: null,
            y_train: null,
            y_test: null,
            y_candidates: null,
            pca_params: null,
            split_stats: null,
            errors: []
        };
        
        for (const file of files) {
            try {
                const loaded = await this.loadFile(file);
                
                if (loaded.format === 'npy') {
                    const fileType = loaded.type;
                    if (fileType && results.hasOwnProperty(fileType)) {
                        results[fileType] = loaded;
                    } else {
                        console.warn(`Unknown NPY file type: ${fileType} (${file.name})`);
                    }
                } else if (loaded.format === 'json') {
                    if (loaded.isPCAParams) {
                        results.pca_params = loaded;
                    } else if (loaded.isSplitStats) {
                        results.split_stats = loaded;
                    }
                }
            } catch (error) {
                results.errors.push({
                    filename: file.name,
                    error: error.message
                });
            }
        }
        
        if (!results.X_train_pca || !results.y_train) {
            console.warn('⚠️ Missing training data in split dataset');
        }
        
        if (results.X_test_pca && !results.y_test) {
            console.warn('⚠️ Test features found but no test labels');
        }
        
        console.log('📦 Split dataset loaded:', {
            train: results.X_train_pca ? '✓' : '✗',
            test: results.X_test_pca ? '✓' : '✗',
            candidates: results.X_candidates_pca ? '✓' : '✗',
            pca_params: results.pca_params ? '✓' : '✗'
        });
        
        return results;
    }

    async loadPCADataset(files) {
        return this.loadSplitDataset(files);
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
        if (!loadedData) return null;
        
        if (loadedData.format === 'npy') {
            return {
                filename: loadedData.filename,
                format: 'NumPy Array',
                fileSize: `${(loadedData.size / 1024).toFixed(2)} KB`,
                shape: loadedData.shape.join(' × '),
                dtype: loadedData.dtype,
                totalElements: loadedData.shape.reduce((a, b) => a * b, 1),
                type: loadedData.type
            };
        }
        
        if (loadedData.format === 'json' && loadedData.isPCAParams) {
            return {
                filename: loadedData.filename,
                format: 'PCA Parameters',
                fileSize: `${(loadedData.size / 1024).toFixed(2)} KB`,
                components: loadedData.data.n_components,
                varianceExplained: loadedData.data.explained_variance_ratio,
                totalVariance: loadedData.data.explained_variance_ratio?.reduce((a, b) => a + b, 0).toFixed(4)
            };
        }
        
        if (loadedData.format === 'csv') {
            return {
                filename: loadedData.filename,
                format: 'CSV',
                fileSize: `${(loadedData.size / 1024).toFixed(2)} KB`,
                rowCount: loadedData.rowCount,
                columnCount: loadedData.header?.length || 0,
                columns: loadedData.header || []
            };
        }
        
        return {
            filename: loadedData.filename,
            format: loadedData.format.toUpperCase(),
            fileSize: `${(loadedData.size / 1024).toFixed(2)} KB`
        };
    }

    exportToJSON(data, filename = 'export.json') {
        const dataStr = JSON.stringify(data, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }

    exportToCSV(data, filename = 'export.csv') {
        const csv = Papa.unparse(data);
        const blob = new Blob([csv], { type: 'text/csv' });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
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
}