"""
Experiment 10: Apply DBSCAN Algorithm for Clustering and Noise/Outlier Identification
Objective: Apply Density-Based Spatial Clustering of Applications with Noise (DBSCAN), discover arbitrary-shaped clusters, estimate optimal epsilon (eps) using k-distance graph, and separate core samples, border samples, and noise/outliers.

================================================================================
-- DBSCAN ALGORITHM FOUNDATIONS
================================================================================
1. Parameters:
   - eps (epsilon): Radius of neighborhood around a data point.
   - min_samples: Minimum number of points within eps-neighborhood.
2. Point Classification:
   - Core Point: Has at least min_samples within distance eps.
   - Border Point: Within distance eps of a Core Point, but has < min_samples.
   - Noise/Outlier: Neither a Core Point nor reachable from any Core Point (labeled as -1).
3. Advantage over K-Means:
   - Discovers non-spherical, arbitrary geometric clusters (e.g. intertwined moons/rings).
   - Inherently robust to noise and outlier contamination without forcing them into clusters.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def main():
    print("=" * 70)
    print("Experiment 10: DBSCAN Density-Based Clustering & Outlier Detection")
    print("=" * 70)

    # 1. Getting dataset from library itself (Synthetic non-linear data with injected noise)
    np.random.seed(42)
    X, y = make_moons(n_samples=350, noise=0.08, random_state=42)

    # Add 25 random uniform outliers to test noise identification
    outliers = np.random.uniform(low=-1.5, high=2.5, size=(25, 2))
    X_full = np.vstack([X, outliers])

    print(f"[+] Loaded complex non-linear dataset: {len(X_full)} total points (including 25 outlier points).")

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # 2. Optimal Epsilon Estimation using k-Nearest Neighbors Distance Graph
    k = 5
    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = nearest_neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    # Sort k-distances
    k_distances = np.sort(distances[:, k-1])

    # 3. Fit DBSCAN Model
    # eps = 0.25 chosen from knee/elbow of k-distance plot; min_samples = 5
    dbscan = DBSCAN(eps=0.25, min_samples=5)
    db_labels = dbscan.fit_predict(X_scaled)

    # Core point mask
    core_samples_mask = np.zeros_like(db_labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    # Analysis of Clusters and Outliers
    unique_labels = set(db_labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = np.sum(db_labels == -1)

    print("\n--- DBSCAN Clustering Results ---")
    print(f"[+] Estimated number of clusters: {n_clusters}")
    print(f"[+] Number of detected noise/outlier points: {n_noise} ({(n_noise/len(X_full))*100:.1f}%)")

    # Silhouette score excluding noise points
    non_noise_mask = db_labels != -1
    if len(set(db_labels[non_noise_mask])) > 1:
        score = silhouette_score(X_scaled[non_noise_mask], db_labels[non_noise_mask])
        print(f"[+] Silhouette Score (excluding noise): {score:.4f}")

    # Compare with K-Means (which fails on arbitrary/crescent shapes)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)

    # 4. Visualization: k-distance graph, K-Means failure, and DBSCAN success
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: k-NN Distance graph for Epsilon selection
    axes[0].plot(k_distances, color='navy', linewidth=2)
    axes[0].axhline(y=0.25, color='red', linestyle='--', label='Selected eps = 0.25')
    axes[0].set_title(f"{k}-NN Distance Graph for Optimal Epsilon", fontsize=12)
    axes[0].set_xlabel("Points sorted by distance", fontsize=10)
    axes[0].set_ylabel(f"{k}-NN Distance", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Plot 2: K-Means Failure on Non-Spherical Data
    axes[1].scatter(X_full[:, 0], X_full[:, 1], c=km_labels, cmap='coolwarm', s=45, edgecolors='k')
    axes[1].set_title("K-Means (Fails to capture non-convex shapes & forces outliers)", fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    # Plot 3: DBSCAN Success (Separating shapes and highlighting noise in black)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k_label, col in zip(unique_labels, colors):
        if k_label == -1:
            col = [0, 0, 0, 1] # Black for noise/outliers

        class_member_mask = (db_labels == k_label)

        # Plot core points
        xy_core = X_full[class_member_mask & core_samples_mask]
        axes[2].plot(xy_core[:, 0], xy_core[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=8, label=f'Cluster {k_label}' if k_label != -1 else 'Noise')

        # Plot non-core / border points
        xy_border = X_full[class_member_mask & ~core_samples_mask]
        axes[2].plot(xy_border[:, 0], xy_border[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=4)

    axes[2].set_title(f"DBSCAN Result ({n_clusters} Clusters, {n_noise} Outliers in Black)", fontsize=11)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig("exp10_dbscan_clustering_outliers.png")
    print("[+] DBSCAN comparison chart saved to: exp10_dbscan_clustering_outliers.png")

if __name__ == "__main__":
    main()
