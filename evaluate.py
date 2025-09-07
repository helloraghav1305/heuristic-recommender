from collections import defaultdict
import numpy as np
from app import recommend_heuristic, ratings

def build_user_likes(ratings_df, threshold=4):
    likes = defaultdict(set)
    for _, row in ratings_df.iterrows():
        if row['rating'] >= threshold:
            likes[row['user']].add(row['title'])
    return likes

user_likes = build_user_likes(ratings)

def precision_at_k(recommended, relevant, k=5):
    if not recommended: return 0.0
    rec_k = recommended[:5]
    n = sum(1 for r in rec_k if r in relevant)
    return n/k

def evaluate_precision_at_k(k=5, sample_size=200):
    sampled_users = np.random.choice(list(user_likes.keys()),
                                     size=min(sample_size, len(user_likes)),
                                     replace=False)
    precision = []
    for user in sampled_users:
        likes_list = list(user_likes[user])
        if len(likes_list) < 3:
            continue
        seed = set(likes_list[:3])
        gt = set(likes_list) - seed
        recs = recommend_heuristic(seed, top_n=k)
        p = precision_at_k(recs, gt, k=k)
        precision.append(p)

    return np.mean(precision) if precision else 0.0

avg_precision_5 = evaluate_precision_at_k(k=5, sample_size=500)
print(f"Mean Precision@5 (heuristic): {avg_precision_5:.3f}")