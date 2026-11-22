"""
Experiment 8: Apply K-Means and use the Elbow Plot for Optimal K
Objective: Partition unlabeled data points using K-Means clustering, calculate Within-Cluster Sum of Squares (Inertia), construct the Elbow plot, validate with Silhouette Score, and visualize clusters.

================================================================================
-- K-MEANS ALGORITHM & ELBOW METHOD PRINCIPLE
================================================================================
1. Objective: Minimize Within-Cluster Sum of Squares (WCSS / Inertia):
   WCSS = sum_{k=1}^K sum_{x_i in S_k} ||x_i - mu_k||^2
2. Elbow Method: Plot WCSS versus K. The "elbow" point represents the optimal K
   where increasing clusters yields diminishing reduction in variance.
3. Silhouette Score: S(i) = (b(i) - a(i)) / max(a(i), b(i))
   - a(i): Mean intra-cluster distance
   - b(i): Mean nearest-cluster distance
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    print("=" * 70)
    print("Experiment 8: K-Means Clustering and Elbow Method Analysis")
    print("=" * 70)

    # 1. Getting dataset from library itself
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names
    print(f"[+] Loaded 'Iris' dataset from sklearn: {X.shape[0]} samples, {X.shape[1]} features.")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Iterate K from 1 to 10 to compute WCSS (Inertia) & Silhouette Scores
    wcss = []
    silhouette_scores = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        if k > 1:
            sil_score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(None)

    # Print summary table
    print("\nK-Means Inertia & Silhouette Analysis:")
    print(f"{'K':<5} {'Inertia (WCSS)':<18} {'Silhouette Score':<18}")
    print("-" * 42)
    for k in K_range:
        sil_str = f"{silhouette_scores[k-1]:.4f}" if silhouette_scores[k-1] is not None else "N/A"
        print(f"{k:<5} {wcss[k-1]:<18.2f} {sil_str:<18}")

    # Optimal K determination (highest silhouette score)
    optimal_k = K_range[1:][np.argmax(silhouette_scores[1:])]
    print(f"\n[+] Optimal K indicated by Silhouette Analysis: K = {optimal_k}")

    # 3. Fit optimal K-Means model
    best_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
    labels = best_kmeans.fit_predict(X_scaled)

    # 4. Visualization: Elbow Curve & Cluster Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: Elbow Curve
    axes[0].plot(K_range, wcss, marker='o', color='crimson', linewidth=2, markersize=7)
    axes[0].axvline(x=optimal_k, color='blue', linestyle='--', label=f'Optimal K = {optimal_k}')
    axes[0].set_title("Elbow Method for Optimal K", fontsize=13)
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=11)
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    # Subplot 2: Cluster Scatter Plot (Sepal Length vs Petal Length)
    scatter = axes[1].scatter(X[:, 0], X[:, 2], c=labels, cmap='viridis', s=60, edgecolors='k', alpha=0.85)
    # Project centroids back to original scale
    original_centroids = scaler.inverse_transform(best_kmeans.cluster_centers_)
    axes[1].scatter(original_centroids[:, 0], original_centroids[:, 2], c='red', s=200, marker='X', label='Centroids')
    axes[1].set_title(f"K-Means Clustering Visualization (K={optimal_k})", fontsize=13)
    axes[1].set_xlabel(feature_names[0], fontsize=11)
    axes[1].set_ylabel(feature_names[2], fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("exp08_kmeans_elbow_plot.png")
    print("[+] Elbow curve and cluster plot saved to: exp08_kmeans_elbow_plot.png")

if __name__ == "__main__":
    main()
