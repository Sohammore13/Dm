"""
Experiment 14: Supervised, Unsupervised & Semi-Supervised Learning using Data Mining Tools (WEKA, RapidMiner, Orange)
Objective: Implement learning paradigms (Supervised, Unsupervised, Semi-Supervised) and prepare interoperable workflows/datasets for industry data mining tools like WEKA, RapidMiner, and Orange.

================================================================================
-- WEKA / RAPIDMINER / ORANGE WORKFLOW SPECIFICATIONS
================================================================================
1. WEKA (Waikato Environment for Knowledge Analysis):
   - Dataset Format: Attribute-Relation File Format (.arff)
   - CLI Command for Supervised J48 Decision Tree:
     java -cp weka.jar weka.classifiers.trees.J48 -t iris.arff -x 10
   - CLI Command for Unsupervised SimpleKMeans:
     java -cp weka.jar weka.clusterers.SimpleKMeans -N 3 -t iris.arff
   - GUI Workflow: Explorer -> Preprocess -> Open File (arff) -> Classify (J48/NaiveBayes) -> Cluster (SimpleKMeans)

2. RapidMiner:
   - Visual operator flow: [Retrieve Data] -> [Split Data 70/30] -> [Apply Model (Decision Tree/SVM)] -> [Performance]
   - Semi-supervised: [Transductive Support Vector Machine (TSVM)]

3. Orange Data Mining:
   - Visual canvas workflow:
     [File Widget] ---> [Data Sampler] ---> [Tree / SVM / Random Forest] ---> [Predictions]
            |                                       ^
            +---------------------------------------+
     [File Widget] ---> [Distances] ---> [Hierarchical Clustering] / [k-Means]
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

def export_to_weka_arff(df, target_col, filename="iris_dataset.arff"):
    """
    Exports DataFrame to standard WEKA ARFF (Attribute-Relation File Format).
    """
    with open(filename, 'w') as f:
        f.write("@relation iris_data_mining\n\n")
        for col in df.columns:
            if col == target_col:
                unique_classes = df[col].unique()
                classes_str = "{" + ",".join([str(c) for c in unique_classes]) + "}"
                f.write(f"@attribute {col} {classes_str}\n")
            else:
                f.write(f"@attribute {col} numeric\n")
        f.write("\n@data\n")
        for _, row in df.iterrows():
            line = ",".join([str(row[c]) for c in df.columns])
            f.write(line + "\n")
    print(f"[+] Exported dataset in WEKA ARFF format to: {filename}")

def export_rapidminer_xml_process(filename="rapidminer_process.rmp"):
    """
    Generates a RapidMiner process execution XML file.
    """
    rmp_content = """<?xml version="1.0" encoding="UTF-8"?>
<process version="9.10.000">
  <context>
    <input/>
    <output/>
    <macros/>
  </context>
  <operator activated="true" class="process" expanded="true" name="Process">
    <process expanded="true">
      <operator activated="true" class="retrieve" name="Retrieve Iris"/>
      <operator activated="true" class="split_data" name="Split Data (70/30)"/>
      <operator activated="true" class="concurrency:parallel_random_forest" name="Random Forest"/>
      <operator activated="true" class="apply_model" name="Apply Model"/>
      <operator activated="true" class="performance" name="Performance Evaluation"/>
    </process>
  </operator>
</process>"""
    with open(filename, 'w') as f:
        f.write(rmp_content)
    print(f"[+] Exported RapidMiner Process Workflow to: {filename}")

def main():
    print("=" * 70)
    print("Experiment 14: Data Mining Tools & Supervised/Unsupervised/Semi-Supervised")
    print("=" * 70)

    # 1. Getting dataset from library itself
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = [f.replace(" ", "_").replace("(", "").replace(")", "") for f in iris.feature_names]
    target_names = list(iris.target_names)

    df = pd.DataFrame(X, columns=feature_names)
    df['class'] = [target_names[i] for i in y]

    # Export formats for external data mining tools
    export_to_weka_arff(df, target_col='class')
    export_rapidminer_xml_process()

    # -------------------------------------------------------------
    # PARADIGM 1: SUPERVISED LEARNING (Random Forest Classifier)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. SUPERVISED LEARNING PARADIGM (Full Supervision)")
    print("=" * 50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_sup = rf.predict(X_test)

    acc_sup = accuracy_score(y_test, y_pred_sup)
    print(f"[+] Supervised Random Forest Accuracy: {acc_sup * 100:.2f}%")
    print(classification_report(y_test, y_pred_sup, target_names=target_names))

    # -------------------------------------------------------------
    # PARADIGM 2: UNSUPERVISED LEARNING (K-Means Clustering)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. UNSUPERVISED LEARNING PARADIGM (Zero Labels Used)")
    print("=" * 50)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, cluster_labels)

    print(f"[+] Unsupervised K-Means Silhouette Score: {sil_score:.4f}")
    c_df = pd.crosstab(y, cluster_labels, rownames=['True Class'], colnames=['K-Means Cluster'])
    print("Contingency Table (True Labels vs Discovered Clusters):")
    print(c_df)

    # -------------------------------------------------------------
    # PARADIGM 3: SEMI-SUPERVISED LEARNING (Label Spreading)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. SEMI-SUPERVISED LEARNING PARADIGM (80% Unlabeled, 20% Labeled)")
    print("=" * 50)
    # Mask 80% of labels by setting them to -1 (unlabeled marker)
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y)) < 0.80
    y_semi = np.copy(y)
    y_semi[random_unlabeled_points] = -1

    n_labeled = np.sum(y_semi != -1)
    n_unlabeled = np.sum(y_semi == -1)
    print(f"[+] Labeled Samples Available   : {n_labeled} ({n_labeled/len(y)*100:.1f}%)")
    print(f"[+] Unlabeled Samples to Infer : {n_unlabeled} ({n_unlabeled/len(y)*100:.1f}%)")

    # Fit Label Spreading algorithm (affinity graph diffusion)
    label_spread = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.8)
    label_spread.fit(X, y_semi)

    # Predict previously unlabeled points
    y_predicted_semi = label_spread.transduction_
    semi_accuracy_on_unlabeled = accuracy_score(y[random_unlabeled_points], y_predicted_semi[random_unlabeled_points])

    print(f"[+] Label Spreading Accuracy on the 80% UNLABELED samples: {semi_accuracy_on_unlabeled * 100:.2f}%")
    print("\nSummary of Classification on Unlabeled Samples:")
    print(classification_report(y[random_unlabeled_points], y_predicted_semi[random_unlabeled_points], target_names=target_names))

if __name__ == "__main__":
    main()
