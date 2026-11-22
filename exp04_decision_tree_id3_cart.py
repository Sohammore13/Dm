"""
Experiment 4: Implement Decision Tree Classifier (ID3, CART)
Objective: Construct decision tree classifiers using ID3 (Entropy / Information Gain) and CART (Gini Impurity) criteria, and evaluate their classification performance.

================================================================================
-- CONCEPTUAL EQUIVALENCE: ID3 VS CART
================================================================================
1. ID3 (Iterative Dichotomiser 3):
   - Impurity Measure: Entropy H(S) = - sum(p_i * log2(p_i))
   - Splitting Criterion: Information Gain IG(S, A) = H(S) - sum(|S_v|/|S| * H(S_v))
   - Scikit-Learn equivalent: DecisionTreeClassifier(criterion='entropy')

2. CART (Classification and Regression Trees):
   - Impurity Measure: Gini Index G(S) = 1 - sum(p_i^2)
   - Splitting Criterion: Gini Gain / Reduction in Impurity
   - Scikit-Learn equivalent: DecisionTreeClassifier(criterion='gini')
================================================================================
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=" * 70)
    print("Experiment 4: Decision Tree Classifier (ID3 vs CART)")
    print("=" * 70)

    # 1. Getting dataset from library itself
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    print(f"[+] Loaded 'Iris' dataset from sklearn: {X.shape[0]} samples, {X.shape[1]} features.")

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # -------------------------------------------------------------
    # 1. ID3 IMPLEMENTATION (Criterion = 'entropy')
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. MODEL 1: ID3 DECISION TREE (Information Gain / Entropy)")
    print("=" * 50)
    id3_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    id3_clf.fit(X_train, y_train)

    y_pred_id3 = id3_clf.predict(X_test)
    acc_id3 = accuracy_score(y_test, y_pred_id3)
    print(f"[+] ID3 Model Accuracy: {acc_id3 * 100:.2f}%")
    print("\nID3 Classification Report:")
    print(classification_report(y_test, y_pred_id3, target_names=target_names))

    print("ID3 Tree Rules:")
    print(export_text(id3_clf, feature_names=feature_names))

    # -------------------------------------------------------------
    # 2. CART IMPLEMENTATION (Criterion = 'gini')
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. MODEL 2: CART DECISION TREE (Gini Impurity)")
    print("=" * 50)
    cart_clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    cart_clf.fit(X_train, y_train)

    y_pred_cart = cart_clf.predict(X_test)
    acc_cart = accuracy_score(y_test, y_pred_cart)
    print(f"[+] CART Model Accuracy: {acc_cart * 100:.2f}%")
    print("\nCART Classification Report:")
    print(classification_report(y_test, y_pred_cart, target_names=target_names))

    print("CART Tree Rules:")
    print(export_text(cart_clf, feature_names=feature_names))

    # -------------------------------------------------------------
    # 3. VISUALIZATION OF DECISION TREES
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plot_tree(id3_clf, feature_names=feature_names, class_names=target_names, filled=True, ax=axes[0])
    axes[0].set_title(f"ID3 Decision Tree (Entropy)\nAccuracy: {acc_id3*100:.2f}%")

    plot_tree(cart_clf, feature_names=feature_names, class_names=target_names, filled=True, ax=axes[1])
    axes[1].set_title(f"CART Decision Tree (Gini)\nAccuracy: {acc_cart*100:.2f}%")

    plt.tight_layout()
    plt.savefig("exp04_decision_trees_comparison.png")
    print("\n[+] Decision Tree comparison plot saved to: exp04_decision_trees_comparison.png")

if __name__ == "__main__":
    main()
