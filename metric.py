import collections

def cal_EQ(cover, G):

    if isinstance(cover, dict):
        communities = list(cover.values())
    else:
        communities = cover

    m = len(G.edges(None, False))
    vertex_community = collections.defaultdict(set)


    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            vertex_community[node].add(community_id)

    total = 0.0

    for community in communities:
        for i in community:
            o_i = len(vertex_community[i])
            k_i = len(G[i])

            for j in community:
                o_j = len(vertex_community[j])
                k_j = len(G[j])
                t = 0.0


                if G.has_edge(i, j):
                    t += 1.0 / (o_i * o_j)


                t -= (k_i * k_j) / (2 * m * o_i * o_j)
                total += t


    return round(total / (2 * m), 4)