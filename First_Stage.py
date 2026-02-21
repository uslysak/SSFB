import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
"""
Seed-based community partitioning.

Description:
- Input: graph G, node embeddings, seed nodes (seed_nodes)
- Output: non-overlapping initial communities
"""
def cosine_similarity(u, v, embeddings):

    vec_u = embeddings[u]
    vec_v = embeddings[v]
    num = np.dot(vec_u, vec_v)
    denom = np.linalg.norm(vec_u) * np.linalg.norm(vec_v)
    if denom == 0:
        return 0.0
    return num / denom


def seed_based_communities(G, embeddings, seed_nodes,
                           threshold=0.3, normalize=True):


    seed_attraction = {seed: {} for seed in seed_nodes}


    all_nodes = list(G.nodes())
    for seed in seed_nodes:

        cos_scores = [cosine_similarity(node, seed, embeddings) for node in all_nodes]

        if normalize:

            scaler = MinMaxScaler()
            cos_norm = scaler.fit_transform(np.array(cos_scores).reshape(-1, 1)).flatten()
            scores = cos_norm
        else:

            scores = np.array(cos_scores)


        for i, node in enumerate(all_nodes):
            seed_attraction[seed][node] = float(scores[i])



    assigned_nodes = set(seed_nodes)
    communities = {seed: {seed} for seed in seed_nodes}
    assignments = {seed: seed for seed in seed_nodes}


    for node in all_nodes:
        if node in assigned_nodes:
            continue

        max_score = -1e9
        best_seed = None
        for seed in seed_nodes:
            score = seed_attraction[seed][node]
            if score > max_score and score >= threshold:
                max_score = score
                best_seed = seed

        if best_seed is not None:
            communities[best_seed].add(node)
            assigned_nodes.add(node)
            assignments[node] = best_seed




    return communities, assignments
