import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: 'index.html'
      }
    }
  },
  server: {
    port: 3000,
    open: true
  },
  resolve: {
    alias: {
      '@': '/'
    }
  },
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs',
      'plotly.js-dist',
      'd3',
      'papaparse'
    ]
  }
});