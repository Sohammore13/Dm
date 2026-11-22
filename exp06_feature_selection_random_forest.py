"""
Experiment 6: Feature Selection with Random Forest for High-Dimensional Data
Objective: Rank and select features from high-dimensional datasets using Random Forest feature importance (MDI), and evaluate classification performance before and after feature reduction.

================================================================================
-- FEATURE SELECTION PRINCIPLE IN ENSEMBLES
================================================================================
- Random Forest calculates Feature Importance via Mean Decrease in Impurity (MDI).
- For each feature X_j, the importance is the total decrease in node impurity 
  weighted by the probability of reaching that node, averaged over all trees:
  Imp(X_j) = (1 / N_trees) * sum_T sum_{t in T, split on X_j} p(t) * Delta_i(t)
- Features below a threshold (e.g. median or mean) are eliminated to reduce variance,
  prevent overfitting, and accelerate downstream model inference.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 70)
    print("Experiment 6: Feature Selection with Random Forest")
    print("=" * 70)

    # 1. Getting dataset from library itself
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    print(f"[+] Loaded Breast Cancer dataset from sklearn: {X.shape[0]} samples, {X.shape[1]} features.")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2. Baseline Random Forest on ALL features
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)

    y_pred_baseline = rf_baseline.predict(X_test)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    print(f"\n[Baseline Model] Number of features: {X_train.shape[1]}")
    print(f"[Baseline Model] Test Accuracy     : {acc_baseline * 100:.2f}%")

    # 3. Extract and Rank Feature Importances
    importances = rf_baseline.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Most Important Features identified by Random Forest:")
    print(feature_imp_df.head(10).to_string(index=False))

    # 4. Feature Selection using SelectFromModel (Selecting features with importance >= mean)
    selector = SelectFromModel(rf_baseline, threshold='mean', prefit=True)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]

    print(f"\n[+] Features selected (Importance >= mean): {len(selected_features)} out of {X.shape[1]}")
    print(f"Selected Feature Names:\n{list(selected_features)}")

    # Transform datasets to reduced feature subspace
    X_train_reduced = selector.transform(X_train)
    X_test_reduced = selector.transform(X_test)

    # 5. Train Random Forest on REDUCED feature set
    rf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_reduced.fit(X_train_reduced, y_train)

    y_pred_reduced = rf_reduced.predict(X_test_reduced)
    acc_reduced = accuracy_score(y_test, y_pred_reduced)

    print(f"\n[Reduced Model] Number of features: {X_train_reduced.shape[1]}")
    print(f"[Reduced Model] Test Accuracy     : {acc_reduced * 100:.2f}%")

    print("\nClassification Report (Reduced Feature Space):")
    print(classification_report(y_test, y_pred_reduced, target_names=data.target_names))

    # 6. Visualization: Plot Top 15 Feature Importances
    plt.figure(figsize=(12, 6))
    top_15 = feature_imp_df.head(15)
    plt.barh(top_15['Feature'][::-1], top_15['Importance'][::-1], color='teal')
    plt.xlabel("Mean Decrease in Impurity (Feature Importance)")
    plt.title("Top 15 Features Selected by Random Forest Classifier")
    plt.tight_layout()
    plt.savefig("exp06_rf_feature_importance.png")
    print("[+] Feature importance chart saved to: exp06_rf_feature_importance.png")

if __name__ == "__main__":
    main()
