"""
Experiment 7: Implement XGBoost on Structured Data and Evaluate Performance
Objective: Apply Extreme Gradient Boosting (XGBoost) on structured tabular data, configure hyperparameters, and evaluate multi-metric classification performance.

================================================================================
-- XGBOOST OBJECTIVE FUNCTION & GRADIENT BOOSTING MECHANISM
================================================================================
Obj^(t) = sum_{i=1}^n l(y_i, y_hat_i^(t-1) + f_t(x_i)) + Omega(f_t)
Where:
- l: Differentiable convex loss function (e.g., binary cross-entropy)
- Omega(f_t) = gamma * T + 0.5 * lambda * sum(w_j^2): Regularization term penalizing tree complexity
- Second-order Taylor expansion utilizes both First (gradient g_i) and Second 
  order gradients (hessian h_i) for optimal split finding.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb

def main():
    print("=" * 70)
    print("Experiment 7: XGBoost Classification on Structured Data")
    print("=" * 70)

    # 1. Getting dataset from library itself
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    print(f"[+] Loaded structured dataset: {X.shape[0]} records, {X.shape[1]} features.")
    print(f"[+] Target Classes: {list(data.target_names)} (0: Malignant, 1: Benign)")

    # Train / Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Initialize and Train XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )

    # Cross-validation
    cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"[+] 5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Fit Model
    xgb_clf.fit(X_train, y_train)

    # 3. Model Predictions & Probability Estimation
    y_pred = xgb_clf.predict(X_test)
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]

    # 4. Comprehensive Performance Evaluation
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 50)
    print("XGBOOST PERFORMANCE EVALUATION")
    print("=" * 50)
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Precision     : {prec * 100:.2f}%")
    print(f"Recall        : {rec * 100:.2f}%")
    print(f"F1-Score      : {f1 * 100:.2f}%")
    print(f"ROC-AUC Score : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # 5. Feature Importance Visualization using XGBoost built-in plot
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(xgb_clf, max_num_features=12, importance_type='gain', title='Top 12 Features by Information Gain')
    plt.tight_layout()
    plt.savefig("exp07_xgboost_feature_importance.png")
    print("[+] XGBoost Feature Importance plot saved to: exp07_xgboost_feature_importance.png")

if __name__ == "__main__":
    main()
