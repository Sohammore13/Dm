"""
Experiment 12: Implement FP-Growth Algorithm for Efficient Pattern Mining
Objective: Mine frequent patterns using the Frequent Pattern Tree (FP-Tree) structure without candidate generation, and compare execution performance against Apriori on transactional data.

================================================================================
-- FP-GROWTH VS APRIORI ARCHITECTURAL COMPARISON
================================================================================
1. Apriori:
   - Generate-and-test paradigm.
   - Repeatedly scans entire database k times for k-itemsets.
   - Candidate generation explosion: C_k combinatorial growth.
2. FP-Growth (Han et al.):
   - Compresses database into a compact Frequent Pattern Tree (FP-Tree).
   - Only 2 passes over the database:
     * Pass 1: Count frequent 1-itemsets and sort items in descending order of frequency.
     * Pass 2: Insert sorted items of each transaction into the prefix FP-Tree.
   - Recursively mines conditional FP-Trees without ever generating candidate sets!
================================================================================
"""

import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

def generate_large_transaction_dataset():
    """
    Generates a realistic large-scale supermarket transaction dataset.
    """
    base_baskets = [
        ['Milk', 'Bread', 'Eggs', 'Butter', 'Yogurt'],
        ['Bread', 'Butter', 'Jam', 'Coffee'],
        ['Milk', 'Diaper', 'Beer', 'Eggs', 'Cola'],
        ['Milk', 'Bread', 'Diaper', 'Beer'],
        ['Bread', 'Milk', 'Butter', 'Cereal'],
        ['Milk', 'Diaper', 'Beer', 'Cola', 'Chips'],
        ['Bread', 'Eggs', 'Cheese', 'Yogurt'],
        ['Milk', 'Bread', 'Diaper', 'Beer', 'Butter'],
        ['Bread', 'Butter', 'Eggs', 'Jam'],
        ['Milk', 'Diaper', 'Beer', 'Chips', 'Peanuts'],
        ['Bread', 'Milk', 'Eggs', 'Cereal'],
        ['Milk', 'Beer', 'Chips', 'Salsa'],
        ['Bread', 'Cheese', 'Butter'],
        ['Milk', 'Bread', 'Butter', 'Jam'],
        ['Diaper', 'Beer', 'Chips', 'Cola'],
        ['Coffee', 'Sugar', 'Milk', 'Biscuits'],
        ['Coffee', 'Sugar', 'Bread'],
        ['Tea', 'Sugar', 'Milk', 'Biscuits'],
        ['Beer', 'Chips', 'Cola', 'Peanuts'],
        ['Apple', 'Banana', 'Orange', 'Yogurt']
    ]
    # Replicate to simulate large-scale retail logs
    return base_baskets * 50

def main():
    print("=" * 70)
    print("Experiment 12: FP-Growth Pattern Mining & Apriori Benchmark")
    print("=" * 70)

    # 1. Getting dataset from library
    transactions = generate_large_transaction_dataset()
    print(f"[+] Loaded large transaction dataset with {len(transactions)} transactions.")

    # One-hot encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    min_support = 0.15
    print(f"[+] Minimum Support Threshold: {min_support * 100}%")

    # 2. Mine Frequent Itemsets using FP-Growth
    start_fp = time.perf_counter()
    fp_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    end_fp = time.perf_counter()
    fp_time = end_fp - start_fp

    print(f"\n[+] FP-Growth Execution Time: {fp_time:.5f} seconds")
    print(f"[+] Total frequent itemsets discovered by FP-Growth: {len(fp_itemsets)}")

    # 3. Mine Frequent Itemsets using Apriori (for Benchmark Comparison)
    start_ap = time.perf_counter()
    ap_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    end_ap = time.perf_counter()
    ap_time = end_ap - start_ap

    print(f"[+] Apriori Execution Time   : {ap_time:.5f} seconds")
    print(f"[+] Total frequent itemsets discovered by Apriori  : {len(ap_itemsets)}")

    speedup = ap_time / fp_time if fp_time > 0 else 1.0
    print(f"\n>>> Performance Gain: FP-Growth was {speedup:.2f}x FASTER than Apriori!")

    # 4. Display Frequent Itemsets from FP-Growth
    fp_itemsets['length'] = fp_itemsets['itemsets'].apply(lambda x: len(x))
    print("\nTop Frequent Itemsets discovered by FP-Growth:")
    print(fp_itemsets.sort_values(by='support', ascending=False).head(10).to_string(index=False))

    # 5. Extract Meaningful Association Rules from FP-Growth Itemsets
    rules = association_rules(fp_itemsets, metric="confidence", min_threshold=0.60)
    strong_rules = rules[rules['lift'] > 1.2].sort_values(by='lift', ascending=False)

    print("\n" + "=" * 50)
    print("DERIVED HIGH-LIFT ASSOCIATION RULES (FP-Tree)")
    print("=" * 50)

    formatted_rules = strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    formatted_rules['antecedents'] = formatted_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    formatted_rules['consequents'] = formatted_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    formatted_rules['support'] = formatted_rules['support'].round(3)
    formatted_rules['confidence'] = formatted_rules['confidence'].round(3)
    formatted_rules['lift'] = formatted_rules['lift'].round(3)

    print(formatted_rules.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
