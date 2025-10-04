import Papa from 'papaparse';

export class DataLoader {
    constructor() {
        this.supportedFormats = ['csv', 'json', 'npy'];
    }

    async loadFile(file) {
        const fileExtension = file.name.split('.').pop().toLowerCase();
        
        if (!this.supportedFormats.includes(fileExtension)) {
            throw new Error(`Unsupported file format: ${fileExtension}. Supported: ${this.supportedFormats.join(', ')}`);
        }
        
        switch(fileExtension) {
            case 'csv': return this.loadCSV(file);
            case 'json': return this.loadJSON(file);
            case 'npy': return this.loadNPY(file);
            default: throw new Error(`Unsupported format: ${fileExtension}`);
        }
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
        
        // Vérifier le magic number de NumPy
        const magic = String.fromCharCode(view.getUint8(0)) + 
                     String.fromCharCode(view.getUint8(1)) + 
                     String.fromCharCode(view.getUint8(2)) + 
                     String.fromCharCode(view.getUint8(3)) + 
                     String.fromCharCode(view.getUint8(4)) + 
                     String.fromCharCode(view.getUint8(5));
        
        if (magic !== '\x93NUMPY') {
            throw new Error('Invalid NPY file format');
        }
        
        // Lire la version
        const major = view.getUint8(6);
        const minor = view.getUint8(7);
        
        // Lire la longueur du header
        let headerLen;
        if (major === 1) {
            headerLen = view.getUint16(8, true);
        } else if (major === 2 || major === 3) {
            headerLen = view.getUint32(8, true);
        } else {
            throw new Error('Unsupported NPY version');
        }
        
        // Lire le header
        const headerStart = major === 1 ? 10 : 12;
        const headerBytes = new Uint8Array(arrayBuffer, headerStart, headerLen);
        const headerStr = new TextDecoder().decode(headerBytes);
        
        // Parser le header (format Python dict)
        const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
        const dtypeMatch = headerStr.match(/'descr':\s*'([^']*)'/);
        const fortranMatch = headerStr.match(/'fortran_order':\s*(True|False)/);
        
        if (!shapeMatch || !dtypeMatch) {
            throw new Error('Invalid NPY header format');
        }
        
        const shape = shapeMatch[1]
            .split(',')
            .map(s => s.trim())
            .filter(s => s)
            .map(Number);
        
        const dtype = dtypeMatch[1];
        const fortranOrder = fortranMatch ? fortranMatch[1] === 'True' : false;
        
        // Lire les données
        const dataStart = headerStart + headerLen;
        const dataBytes = new Uint8Array(arrayBuffer, dataStart);
        
        // Convertir selon le dtype
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
            dtype,
            fortranOrder
        };
    }

    inferNPYType(filename) {
        const lower = filename.toLowerCase();
        if (lower.includes('x_train')) return 'X_train_pca';
        if (lower.includes('x_test')) return 'X_test_pca';
        if (lower.includes('y_train')) return 'y_train';
        if (lower.includes('y_test')) return 'y_test';
        return 'unknown';
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
                    
                    // Vérifier si c'est le fichier pca_params.json
                    const isPCAParams = file.name.includes('pca_params');
                    
                    resolve({
                        data: data,
                        format: 'json',
                        filename: file.name,
                        size: file.size,
                        type: isPCAParams ? 'pca_params' : 'generic',
                        isPCAParams
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

    async loadPCADataset(files) {
        const results = {
            X_train_pca: null,
            X_test_pca: null,
            y_train: null,
            y_test: null,
            pca_params: null,
            errors: []
        };
        
        for (const file of files) {
            try {
                const loaded = await this.loadFile(file);
                
                if (loaded.format === 'npy') {
                    results[loaded.type] = loaded;
                } else if (loaded.format === 'json' && loaded.isPCAParams) {
                    results.pca_params = loaded;
                }
            } catch (error) {
                results.errors.push({
                    filename: file.name,
                    error: error.message
                });
            }
        }
        
        return results;
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
            const { data, meta } = loadedData;
            return {
                filename: loadedData.filename,
                format: 'CSV',
                fileSize: `${(loadedData.size / 1024).toFixed(2)} KB`,
                rowCount: data.length,
                columnCount: meta.fields?.length || 0,
                columns: meta.fields || Object.keys(data[0] || {})
            };
        }
        
        return null;
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
}