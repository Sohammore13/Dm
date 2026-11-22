"""
Experiment 9: Agglomerative Hierarchical Clustering with Dendrogram Visualization
Objective: Perform bottom-up hierarchical clustering on a real-world dataset, construct a dendrogram to identify cluster hierarchies, and evaluate clustering structure across linkage metrics.

================================================================================
-- AGGLOMERATIVE HIERARCHICAL CLUSTERING CONCEPTS
================================================================================
1. Bottom-up approach: Each sample starts as an individual singleton cluster.
2. At each iteration, the two clusters with the minimum inter-cluster distance are merged.
3. Linkage Criteria:
   - Ward's Method: Minimizes total within-cluster variance.
   - Complete Linkage: Maximum distance between points in two clusters.
   - Average Linkage: Mean distance between all pairs of points.
4. Dendrogram: Tree diagram displaying sequence of cluster merges and vertical 
   distance thresholds indicating cluster separation.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

def main():
    print("=" * 70)
    print("Experiment 9: Agglomerative Hierarchical Clustering & Dendrogram")
    print("=" * 70)

    # 1. Getting dataset from library itself
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names
    target_names = iris.target_names
    print(f"[+] Loaded 'Iris' dataset from sklearn: {X.shape[0]} samples, {X.shape[1]} features.")

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Compute Linkage Matrices using Scipy (Ward, Complete, Average)
    linkage_ward = linkage(X_scaled, method='ward')
    linkage_complete = linkage(X_scaled, method='complete')
    linkage_average = linkage(X_scaled, method='average')

    # 3. Fit Agglomerative Clustering model with 3 clusters
    n_clusters = 3
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = agg_cluster.fit_predict(X_scaled)

    # Evaluate with Silhouette Score
    sil_score = silhouette_score(X_scaled, labels)
    print(f"[+] Agglomerative Clustering (K={n_clusters}, Ward Linkage) Silhouette Score: {sil_score:.4f}")

    # 4. Dendrogram & Cluster Scatter Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Dendrogram
    dendrogram(
        linkage_ward,
        truncate_mode='lastp',
        p=25,
        leaf_rotation=90.,
        leaf_font_size=9.,
        show_contracted=True,
        ax=axes[0]
    )
    # Draw horizontal threshold line for 3 clusters
    axes[0].axhline(y=10.0, color='r', linestyle='--', label='Cut Threshold (3 Clusters)')
    axes[0].set_title("Hierarchical Dendrogram (Ward Linkage)", fontsize=13)
    axes[0].set_xlabel("Cluster Index / Truncated Sample Size", fontsize=11)
    axes[0].set_ylabel("Euclidean Distance (Variance)", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Subplot 2: Cluster Scatter Plot (Petal Length vs Petal Width)
    scatter = axes[1].scatter(X[:, 2], X[:, 3], c=labels, cmap='viridis', s=60, edgecolors='k', alpha=0.85)
    axes[1].set_title(f"Agglomerative Clustering Results (Clusters = {n_clusters})", fontsize=13)
    axes[1].set_xlabel(feature_names[2], fontsize=11)
    axes[1].set_ylabel(feature_names[3], fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label("Hierarchical Cluster Label")

    plt.tight_layout()
    plt.savefig("exp09_hierarchical_dendrogram.png")
    print("[+] Dendrogram and cluster visualization saved to: exp09_hierarchical_dendrogram.png")

if __name__ == "__main__":
    main()
