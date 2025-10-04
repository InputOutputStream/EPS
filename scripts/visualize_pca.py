import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

def visualize_pca(data_path='dataset/output/'):
    """Visualize PCA results"""
    
    # Load data
    X_train = np.load(f'{data_path}/X_train_pca.npy')
    X_test = np.load(f'{data_path}/X_test_pca.npy')
    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    
    # Load metadata
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    n_components = X_train.shape[1]
    
    print(f"PCA Visualization")
    print(f"={'='*50}")
    print(f"Training: {X_train.shape}")
    print(f"Test: {X_test.shape}")
    print(f"Components: {n_components}")
    print(f"Variance explained: {metadata['explained_variance']}")
    print(f"Cumulative: {metadata['cumulative_variance']}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Color mapping
    colors = {0: 'red', 1: 'orange', 2: 'green'}
    labels_map = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
    
    if n_components >= 2:
        # 2D scatter
        ax1 = plt.subplot(2, 3, 1)
        for label in [0, 1, 2]:
            mask = y_train == label
            ax1.scatter(X_train[mask, 0], X_train[mask, 1], 
                       c=colors[label], label=labels_map[label], 
                       alpha=0.6, s=30)
        ax1.set_xlabel(f'PC1 ({metadata["explained_variance"][0]:.3%})')
        ax1.set_ylabel(f'PC2 ({metadata["explained_variance"][1]:.3%})')
        ax1.set_title('PCA: PC1 vs PC2 (Train)')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    if n_components >= 3:
        # 3D scatter
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        for label in [0, 1, 2]:
            mask = y_train == label
            ax2.scatter(X_train[mask, 0], X_train[mask, 1], X_train[mask, 2],
                       c=colors[label], label=labels_map[label], 
                       alpha=0.6, s=30)
        ax2.set_xlabel(f'PC1 ({metadata["explained_variance"][0]:.2%})')
        ax2.set_ylabel(f'PC2 ({metadata["explained_variance"][1]:.2%})')
        ax2.set_zlabel(f'PC3 ({metadata["explained_variance"][2]:.2%})')
        ax2.set_title('PCA: 3D View (Train)')
        ax2.legend()
    
    # Variance explained
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(range(1, n_components + 1), metadata['explained_variance'])
    ax3.plot(range(1, n_components + 1), metadata['cumulative_variance'], 
             'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Variance Explained')
    ax3.set_title('Variance Explained by Component')
    ax3.grid(alpha=0.3)
    ax3.legend(['Cumulative', 'Individual'])
    
    # Test set visualization
    if n_components >= 2:
        ax4 = plt.subplot(2, 3, 4)
        for label in [0, 1, 2]:
            mask = y_test == label
            ax4.scatter(X_test[mask, 0], X_test[mask, 1], 
                       c=colors[label], label=labels_map[label], 
                       alpha=0.6, s=30)
        ax4.set_xlabel(f'PC1')
        ax4.set_ylabel(f'PC2')
        ax4.set_title('PCA: PC1 vs PC2 (Test)')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    # Class distribution
    ax5 = plt.subplot(2, 3, 5)
    train_counts = [np.sum(y_train == i) for i in range(3)]
    test_counts = [np.sum(y_test == i) for i in range(3)]
    x = np.arange(3)
    width = 0.35
    ax5.bar(x - width/2, train_counts, width, label='Train', alpha=0.8)
    ax5.bar(x + width/2, test_counts, width, label='Test', alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['FP', 'Candidate', 'Confirmed'])
    ax5.set_ylabel('Count')
    ax5.set_title('Class Distribution')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Feature importance (top 10)
    if 'feature_importance' in metadata and metadata['feature_importance']:
        ax6 = plt.subplot(2, 3, 6)
        features = list(metadata['feature_importance'].keys())[:10]
        scores = [metadata['feature_importance'][f] for f in features]
        y_pos = np.arange(len(features))
        ax6.barh(y_pos, scores)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels([f[:30] for f in features], fontsize=8)
        ax6.set_xlabel('MI Score')
        ax6.set_title('Top 10 Features (Mutual Information)')
        ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = f'{data_path}/pca_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', default='dataset/output/')
    args = parser.parse_args()
    
    visualize_pca(args.data_path)