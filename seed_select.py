from __future__ import division

import metis
"""
Seed Node Selection via METIS Partition + Intra-cluster Centrality Scoring
1) Partition graph G into K clusters using METIS.
2) For each cluster C, compute a cluster-based score for every node 𝑣∈𝐶.
3) Choose one seed per cluster: seed_C = argmax score(v)

"""






def metis_partition(G, num_clusters):
    edgecuts, parts = metis.part_graph(G, nparts=num_clusters)
    return dict(zip(G.nodes(), parts))


def get_community(G, nb_cluster, part):
    nodes = [node for node, c in part.items() if c == nb_cluster]
    return G.subgraph(nodes)


def get_clusters_node(part, cluster):
    return [node for node in part if part[node] == cluster]


def Graclus_centers(G, num_clusters):
    seeds = []
    part = metis_partition(G, num_clusters)


    clusters = sorted(set(part.values()))
    #print("Clusters found:", clusters)

    for cluster in clusters:

        nodes_in_cluster = get_clusters_node(part, cluster)
        subGraph = G.subgraph(nodes_in_cluster)



        GammaC = {v: list(subGraph.neighbors(v)) for v in subGraph.nodes()}
        kC = {v: len(GammaC[v]) for v in subGraph.nodes()}

        f1 = {}
        for v in subGraph.nodes():
            k_total = G.degree[v]
            if k_total > 0:
                f1[v] = kC[v] / float(k_total)
            else:
                f1[v] = 0.0

        NTS_C = {}
        for v in subGraph.nodes():
            if kC[v] == 0:
                NTS_C[v] = 0.0
                continue

            neigh = GammaC[v]
            denom = sum(kC[u] for u in neigh)

            if denom == 0:
                NTS_C[v] = float(kC[v])
                continue

            Nv = set(neigh)
            common_sum = 0
            for u in neigh:
                common_sum += len(Nv & set(GammaC[u]))

            NTS_C[v] = kC[v] + (kC[v] * common_sum) / (2.0 * denom)

        max_NTS = max(NTS_C.values()) if NTS_C else 1.0

        g = {}
        for v in subGraph.nodes():
            if max_NTS > 0:
                g[v] = NTS_C[v] / float(max_NTS)
            else:
                g[v] = 0.0

        scores = {}
        for v in subGraph.nodes():
            scores[v] = f1[v] * 1 + g[v] * 1
            #print(f"Vertex {v}: f1={f1[v]:.4f}, g={g[v]:.4f}, score={scores[v]:.4f}")

        if scores:
            best_seed = max(scores, key=scores.get)
            seeds.append(best_seed)
            #print(f"--> Chosen seed for cluster {cluster}: {best_seed}")

    return seeds



def select_seed_nodes(G, num_clusters):

    return Graclus_centers(G, num_clusters)




