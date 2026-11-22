# Data Mining Laboratory Experiments (1 to 14)

This repository contains complete, verified, standalone implementations for all 14 Data Mining Lab experiments mapped directly to your course syllabus.

Each experiment:
1. Loads datasets directly from Python libraries (`scikit-learn`, `seaborn`, `pandas`).
2. Contains native **SQL DDL/DML and OLAP queries in docstring comments** wherever required.
3. Has been tested and verified to execute without errors.

---

## Experiment Mapping & Quick Reference

| Exp # | Experiment Title | Script File | Dataset Source | Key Techniques / Topics |
|:---:|:---|:---|:---|:---|
| **1** | Design & Implement Star Schema | [exp01_star_schema.py](file:///c:/Users/soham/Desktop/DM/codes/exp01_star_schema.py) | Library Dataset (`tips`) / SQLite | DDL/DML Star Schema, Fact & Dim tables, Analytical SQL joins |
| **2** | OLAP Operations (Roll-up, Drill-down, Slice, Dice) | [exp02_olap_operations.py](file:///c:/Users/soham/Desktop/DM/codes/exp02_olap_operations.py) | Library Dataset (`flights`) | SQL `ROLLUP`, `CUBE`, Slicing, Dicing, Hierarchical Drill-down |
| **3** | Data Cleaning, Integration, Transformation & Visualization | [exp03_data_cleaning_transformation.py](file:///c:/Users/soham/Desktop/DM/codes/exp03_data_cleaning_transformation.py) | Library Dataset (`titanic`) | Imputation, IQR Outliers, `pd.merge`, Z-score, Min-Max, Boxplots |
| **4** | Decision Tree Classifier (ID3 & CART) | [exp04_decision_tree_id3_cart.py](file:///c:/Users/soham/Desktop/DM/codes/exp04_decision_tree_id3_cart.py) | `sklearn.datasets.load_iris` | Entropy (Information Gain), Gini Impurity, Tree Rules, Plot |
| **5** | Spam Detection using Naïve Bayes | [exp05_spam_detection_naive_bayes.py](file:///c:/Users/soham/Desktop/DM/codes/exp05_spam_detection_naive_bayes.py) | Text Corpus (`sklearn`) | TF-IDF Vectorization, MultinomialNB, Precision/Recall/F1 |
| **6** | Feature Selection with Random Forest | [exp06_feature_selection_random_forest.py](file:///c:/Users/soham/Desktop/DM/codes/exp06_feature_selection_random_forest.py) | `sklearn.datasets.load_breast_cancer` | Mean Decrease in Impurity (MDI), `SelectFromModel`, High-Dim Reduction |
| **7** | XGBoost on Structured Data | [exp07_xgboost_structured_data.py](file:///c:/Users/soham/Desktop/DM/codes/exp07_xgboost_structured_data.py) | `sklearn.datasets.load_breast_cancer` | Gradient Boosting, 5-Fold CV, ROC-AUC, Log-Loss, Feature Importance |
| **8** | K-Means Clustering & Elbow Plot | [exp08_kmeans_elbow_analysis.py](file:///c:/Users/soham/Desktop/DM/codes/exp08_kmeans_elbow_analysis.py) | `sklearn.datasets.load_iris` | Inertia (WCSS), Elbow Curve, Silhouette Coefficient, Centroids |
| **9** | Agglomerative Hierarchical Clustering & Dendrogram | [exp09_hierarchical_clustering_dendrogram.py](file:///c:/Users/soham/Desktop/DM/codes/exp09_hierarchical_clustering_dendrogram.py) | `sklearn.datasets.load_iris` | Linkage (Ward/Complete/Avg), Dendrogram Threshold Cut, Scatter |
| **10** | DBSCAN Clustering & Outlier Detection | [exp10_dbscan_clustering_outliers.py](file:///c:/Users/soham/Desktop/DM/codes/exp10_dbscan_clustering_outliers.py) | `sklearn.datasets.make_moons` | Density Clustering, k-NN Distance graph (eps), Core vs Noise (-1) |
| **11** | Market Basket Analysis using Apriori | [exp11_apriori_market_basket_analysis.py](file:///c:/Users/soham/Desktop/DM/codes/exp11_apriori_market_basket_analysis.py) | `mlxtend.preprocessing` | Frequent Itemsets, Support, Confidence, Lift > 1.0, Conviction |
| **12** | FP-Growth Pattern Mining | [exp12_fp_growth_pattern_mining.py](file:///c:/Users/soham/Desktop/DM/codes/exp12_fp_growth_pattern_mining.py) | Transaction Dataset | Frequent Pattern Tree (FP-Tree), Candidate-free, Apriori Benchmark |
| **13** | PCY, Multihash & Multistage on Large Dataset | [exp13_pcy_multihash_multistage.py](file:///c:/Users/soham/Desktop/DM/codes/exp13_pcy_multihash_multistage.py) | Large Transaction Logs | Hash Buckets, Bitmaps, Multihash filtering, Multistage 3-pass C2 reduction |
| **14** | Data Mining Tools & Learning Paradigms | [exp14_data_mining_tools_paradigms.py](file:///c:/Users/soham/Desktop/DM/codes/exp14_data_mining_tools_paradigms.py) | `sklearn.datasets.load_iris` | WEKA ARFF, RapidMiner XML, Orange flow, Supervised, Unsupervised, Semi-Supervised (`LabelSpreading`) |

---

## How to Run Any Experiment

Run directly with Python:
```powershell
python exp01_star_schema.py
python exp02_olap_operations.py
python exp03_data_cleaning_transformation.py
python exp04_decision_tree_id3_cart.py
python exp05_spam_detection_naive_bayes.py
python exp06_feature_selection_random_forest.py
python exp07_xgboost_structured_data.py
python exp08_kmeans_elbow_analysis.py
python exp09_hierarchical_clustering_dendrogram.py
python exp10_dbscan_clustering_outliers.py
python exp11_apriori_market_basket_analysis.py
python exp12_fp_growth_pattern_mining.py
python exp13_pcy_multihash_multistage.py
python exp14_data_mining_tools_paradigms.py
```
