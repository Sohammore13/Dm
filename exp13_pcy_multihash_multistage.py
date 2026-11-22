"""
Experiment 13: Implement Multistage and Multihash PCY on Large Dataset
Objective: Implement the Park-Chen-Yu (PCY) frequent itemset mining framework, along with Multihash PCY and Multistage PCY enhancements to reduce candidate pairs (C_2) on large datasets.

================================================================================
-- PCY, MULTIHASH & MULTISTAGE ALGORITHMIC FOUNDATIONS
================================================================================
1. Standard PCY (Park, Chen, Yu):
   - In Pass 1: Count frequent 1-itemsets AND hash all pairs in each basket into a 
     hash table of buckets.
   - Summarize hash table into a 1-bit-per-bucket Bitmap (1 = Frequent bucket, 0 = Infrequent).
   - In Pass 2: Candidate pairs C_2 require:
     * Both items {i, j} are frequent singletons (in L_1).
     * Pair hashes to a frequent bucket: Bitmap[h(i, j)] == 1.

2. Multihash PCY:
   - Divides available hash table memory across k independent hash functions (h1, h2).
   - In Pass 1: Hashes each pair with BOTH h1 and h2 into separate bucket arrays.
   - In Pass 2: Candidate pair requires:
     * i in L_1 AND j in L_1
     * Bitmap1[h1(i, j)] == 1 AND Bitmap2[h2(i, j)] == 1 (Joint filtering).

3. Multistage PCY:
   - Pass 1: Count singletons & hash pairs to Bitmap 1 using h1.
   - Pass 2: Re-scan transactions. Only hash pairs that PASSED Bitmap 1 into a NEW 
     hash table using h2 to produce Bitmap 2.
   - Pass 3: Count only pairs surviving BOTH Bitmap 1 and Bitmap 2.
================================================================================
"""

import itertools
from collections import defaultdict

def generate_large_baskets(num_baskets=1000, num_items=50):
    """
    Generates synthetic market basket data containing controlled frequent item patterns.
    """
    import random
    random.seed(42)
    baskets = []
    # Injected correlated patterns (e.g. items 1, 2, 3 frequently bought together)
    frequent_core = [1, 2, 3]
    frequent_pair = [10, 11]

    for _ in range(num_baskets):
        basket_size = random.randint(3, 10)
        basket = set(random.sample(range(1, num_items + 1), basket_size))
        # Inject patterns with probability
        if random.random() < 0.35:
            basket.update(frequent_core)
        if random.random() < 0.40:
            basket.update(frequent_pair)
        baskets.append(sorted(list(basket)))
    return baskets

# Independent hash functions for PCY
def hash_func_1(i, j, num_buckets):
    return (i * 31 + j * 17 + 7) % num_buckets

def hash_func_2(i, j, num_buckets):
    return (i * 59 + j * 43 + 19) % num_buckets

def main():
    print("=" * 70)
    print("Experiment 13: PCY, Multihash PCY & Multistage PCY Implementation")
    print("=" * 70)

    # 1. Generate / Load dataset
    num_baskets = 1200
    num_items = 40
    baskets = generate_large_baskets(num_baskets=num_baskets, num_items=num_items)
    min_support = 300  # Support threshold count (25% of 1200)
    num_buckets = 500

    print(f"[+] Loaded dataset: {len(baskets)} transactions with {num_items} distinct items.")
    print(f"[+] Minimum Support Threshold (Count): {min_support} occurrences")
    print(f"[+] Hash Buckets Count: {num_buckets}\n")

    # =============================================================
    # PASS 1: Count Singletons (L1) for all methods
    # =============================================================
    item_counts = defaultdict(int)
    for b in baskets:
        for item in b:
            item_counts[item] += 1

    L1 = {item for item, count in item_counts.items() if count >= min_support}
    all_possible_pairs = len(L1) * (len(L1) - 1) // 2
    print(f"[+] Pass 1 Complete: {len(L1)} Frequent 1-itemsets (L1) found.")
    print(f"[+] Theoretical Apriori Candidate 2-itemsets (L1 x L1): {all_possible_pairs} pairs")

    # =============================================================
    # 1. STANDARD PCY IMPLEMENTATION
    # =============================================================
    print("\n" + "-" * 50)
    print("1. STANDARD PCY ALGORITHM")
    print("-" * 50)
    bucket_counts_pcy = defaultdict(int)
    for b in baskets:
        for i, j in itertools.combinations(b, 2):
            if i in L1 and j in L1:
                b_idx = hash_func_1(i, j, num_buckets)
                bucket_counts_pcy[b_idx] += 1

    bitmap_pcy = {b_idx: (cnt >= min_support) for b_idx, cnt in bucket_counts_pcy.items()}
    frequent_buckets_pcy = sum(1 for v in bitmap_pcy.values() if v)
    print(f"[PCY] Frequent Buckets in Bitmap: {frequent_buckets_pcy} / {num_buckets}")

    # Pass 2 Candidate generation
    candidates_pcy = []
    for i, j in itertools.combinations(sorted(list(L1)), 2):
        if bitmap_pcy.get(hash_func_1(i, j, num_buckets), False):
            candidates_pcy.append((i, j))
    print(f"[PCY] Surviving Candidate Pairs (C2): {len(candidates_pcy)} (Filtered out {all_possible_pairs - len(candidates_pcy)} pairs)")

    # =============================================================
    # 2. MULTIHASH PCY IMPLEMENTATION
    # =============================================================
    print("\n" + "-" * 50)
    print("2. MULTIHASH PCY ALGORITHM (2 Independent Hash Functions)")
    print("-" * 50)
    bucket_counts_mh1 = defaultdict(int)
    bucket_counts_mh2 = defaultdict(int)

    # In Pass 1, hash into BOTH hash tables simultaneously
    for b in baskets:
        for i, j in itertools.combinations(b, 2):
            if i in L1 and j in L1:
                b1 = hash_func_1(i, j, num_buckets)
                b2 = hash_func_2(i, j, num_buckets)
                bucket_counts_mh1[b1] += 1
                bucket_counts_mh2[b2] += 1

    bitmap_mh1 = {b: (cnt >= min_support) for b, cnt in bucket_counts_mh1.items()}
    bitmap_mh2 = {b: (cnt >= min_support) for b, cnt in bucket_counts_mh2.items()}

    # Candidate filtering in Pass 2 requires passing BOTH bitmaps
    candidates_multihash = []
    for i, j in itertools.combinations(sorted(list(L1)), 2):
        if bitmap_mh1.get(hash_func_1(i, j, num_buckets), False) and \
           bitmap_mh2.get(hash_func_2(i, j, num_buckets), False):
            candidates_multihash.append((i, j))

    print(f"[Multihash] Bitmap 1 Frequent Buckets: {sum(bitmap_mh1.values())}")
    print(f"[Multihash] Bitmap 2 Frequent Buckets: {sum(bitmap_mh2.values())}")
    print(f"[Multihash] Surviving Candidate Pairs (C2): {len(candidates_multihash)} pairs")

    # =============================================================
    # 3. MULTISTAGE PCY IMPLEMENTATION
    # =============================================================
    print("\n" + "-" * 50)
    print("3. MULTISTAGE PCY ALGORITHM (Sequential Multi-Pass Refinement)")
    print("-" * 50)
    # Pass 1 produces Bitmap 1 (same as standard PCY)
    # Pass 2: Hash ONLY candidate pairs that passed Bitmap 1 into a NEW Hash Table using h2
    bucket_counts_stage2 = defaultdict(int)
    for b in baskets:
        for i, j in itertools.combinations(b, 2):
            if i in L1 and j in L1:
                if bitmap_pcy.get(hash_func_1(i, j, num_buckets), False):
                    b2 = hash_func_2(i, j, num_buckets)
                    bucket_counts_stage2[b2] += 1

    bitmap_stage2 = {b: (cnt >= min_support) for b, cnt in bucket_counts_stage2.items()}
    print(f"[Multistage] Pass 2 Bitmap 2 Frequent Buckets: {sum(bitmap_stage2.values())}")

    # Pass 3: Candidate counting requires surviving both Stage 1 and Stage 2
    candidates_multistage = []
    for i, j in itertools.combinations(sorted(list(L1)), 2):
        if bitmap_pcy.get(hash_func_1(i, j, num_buckets), False) and \
           bitmap_stage2.get(hash_func_2(i, j, num_buckets), False):
            candidates_multistage.append((i, j))
    print(f"[Multistage] Surviving Candidate Pairs (C2) after Stage 2: {len(candidates_multistage)} pairs")

    # =============================================================
    # FINAL VERIFICATION & ACTUAL FREQUENT 2-ITEMSETS
    # =============================================================
    actual_pair_counts = defaultdict(int)
    for b in baskets:
        for pair in candidates_multistage:
            if pair[0] in b and pair[1] in b:
                actual_pair_counts[pair] += 1

    L2 = {pair: count for pair, count in actual_pair_counts.items() if count >= min_support}

    print("\n" + "=" * 50)
    print("COMPARATIVE EFFICIENCY SUMMARY")
    print("=" * 50)
    print(f"{'Method':<20} {'Candidate Pairs (C2)':<25} {'Reduction %':<15}")
    print("-" * 60)
    print(f"{'Apriori (L1 x L1)':<20} {all_possible_pairs:<25} {'0.0%':<15}")
    print(f"{'Standard PCY':<20} {len(candidates_pcy):<25} {((all_possible_pairs - len(candidates_pcy))/all_possible_pairs)*100:.1f}%")
    print(f"{'Multihash PCY':<20} {len(candidates_multihash):<25} {((all_possible_pairs - len(candidates_multihash))/all_possible_pairs)*100:.1f}%")
    print(f"{'Multistage PCY':<20} {len(candidates_multistage):<25} {((all_possible_pairs - len(candidates_multistage))/all_possible_pairs)*100:.1f}%")

    print("\nActual True Frequent 2-Itemsets (L2):")
    for pair, count in sorted(L2.items(), key=lambda x: x[1], reverse=True):
        print(f"  Itemset {pair}: Count = {count} / {num_baskets}")

if __name__ == "__main__":
    main()
