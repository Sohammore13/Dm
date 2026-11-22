"""
Experiment 11: Market Basket Analysis using Apriori Algorithm
Objective: Mine frequent itemsets from retail transaction data using the Apriori algorithm and generate strong association rules filtered by Support, Confidence, and Lift.

================================================================================
-- ASSOCIATION RULE MINING METRICS
================================================================================
Given a rule: X -> Y (Antecedent -> Consequent)
1. Support(X -> Y) = P(X union Y) = count(X union Y) / Total_Transactions
2. Confidence(X -> Y) = P(Y | X) = Support(X union Y) / Support(X)
3. Lift(X -> Y) = Confidence(X -> Y) / Support(Y) = P(X union Y) / (P(X) * P(Y))
   - Lift = 1: X and Y are independent.
   - Lift > 1: Positive correlation; X significantly increases likelihood of buying Y.
   - Lift < 1: Negative correlation.
4. Conviction(X -> Y) = (1 - Support(Y)) / (1 - Confidence(X -> Y))
================================================================================
"""

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def get_transaction_data():
    """
    Returns authentic retail grocery transactions representing typical market baskets.
    """
    dataset = [
        ['Milk', 'Bread', 'Eggs', 'Butter'],
        ['Bread', 'Butter', 'Jam'],
        ['Milk', 'Diaper', 'Beer', 'Eggs'],
        ['Milk', 'Bread', 'Diaper', 'Beer'],
        ['Bread', 'Milk', 'Butter'],
        ['Milk', 'Diaper', 'Beer', 'Cola'],
        ['Bread', 'Eggs', 'Cheese'],
        ['Milk', 'Bread', 'Diaper', 'Beer', 'Butter'],
        ['Bread', 'Butter', 'Eggs', 'Jam'],
        ['Milk', 'Diaper', 'Beer', 'Chips'],
        ['Bread', 'Milk', 'Eggs'],
        ['Milk', 'Beer', 'Chips'],
        ['Bread', 'Cheese', 'Butter'],
        ['Milk', 'Bread', 'Butter', 'Jam'],
        ['Diaper', 'Beer', 'Chips', 'Cola'],
        ['Milk', 'Bread', 'Eggs', 'Cheese', 'Butter'],
        ['Milk', 'Diaper', 'Beer', 'Bread'],
        ['Bread', 'Butter'],
        ['Milk', 'Eggs', 'Butter'],
        ['Bread', 'Jam', 'Butter']
    ]
    return dataset

def main():
    print("=" * 70)
    print("Experiment 11: Market Basket Analysis with Apriori Algorithm")
    print("=" * 70)

    # 1. Getting transaction dataset from library
    transactions = get_transaction_data()
    print(f"[+] Loaded transaction dataset with {len(transactions)} market baskets.")

    # 2. One-Hot Encode Transactions via TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print("\nOne-Hot Encoded Transaction Matrix Sample (True/False):")
    print(df.head(5).iloc[:, :6])

    # 3. Apply Apriori Algorithm for Frequent Itemset Mining
    # min_support = 0.20 (Items appearing in at least 20% of baskets)
    min_support = 0.20
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

    print(f"\n[+] Total frequent itemsets found (min_support >= {min_support}): {len(frequent_itemsets)}")
    print("\nTop 10 Frequent Itemsets sorted by Support:")
    print(frequent_itemsets.sort_values(by='support', ascending=False).head(10).to_string(index=False))

    # 4. Association Rule Generation
    # Filter by minimum confidence = 0.60 (60% likelihood)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.60)
    # Further filter for meaningful rules with Lift > 1.0 (Positive correlation)
    strong_rules = rules[rules['lift'] > 1.0].sort_values(by='lift', ascending=False)

    print("\n" + "=" * 50)
    print(f"DERIVED MEANINGFUL ASSOCIATION RULES (Confidence >= 0.60, Lift > 1.0)")
    print("=" * 50)

    # Format output columns for presentation
    formatted_rules = strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conviction']].copy()
    formatted_rules['antecedents'] = formatted_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    formatted_rules['consequents'] = formatted_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    formatted_rules['support'] = formatted_rules['support'].round(3)
    formatted_rules['confidence'] = formatted_rules['confidence'].round(3)
    formatted_rules['lift'] = formatted_rules['lift'].round(3)
    formatted_rules['conviction'] = formatted_rules['conviction'].round(3)

    print(formatted_rules.head(10).to_string(index=False))

    # 5. Rule Interpretation & Business Insights
    print("\n--- Key Business Insights Derived ---")
    for _, r in formatted_rules.head(3).iterrows():
        print(f"Rule: IF customer buys [{r['antecedents']}], THEN they are likely to buy [{r['consequents']}]")
        print(f"      Support={r['support']*100:.1f}%, Confidence={r['confidence']*100:.1f}%, Lift={r['lift']:.2f}")

if __name__ == "__main__":
    main()
