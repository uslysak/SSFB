import random

import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import Counter

"""
Utilities: helper functions.
"""


def batch_input_generator(a_walk, random_walk_length, window_size):

    seq_1 = [a_walk[j] for j in range(random_walk_length-window_size)]
    seq_2 = [a_walk[j] for j in range(window_size, random_walk_length)]
    return np.array(seq_1 + seq_2)

def batch_label_generator(a_walk, random_walk_length, window_size):

    grams_1 = [a_walk[j+1:j+1+window_size] for j in range(random_walk_length-window_size)]
    grams_2 = [a_walk[j-window_size:j] for j in range(window_size, random_walk_length)]
    return np.array(grams_1 + grams_2)


def gamma_incrementer(step, gamma_0, gamma_final, current_gamma, num_steps):
    if step > 1:
        exponent = (0-np.log10(gamma_0))/float(num_steps)
        current_gamma = current_gamma * (10 **exponent)*(gamma_final-gamma_0)
        current_gamma = current_gamma + gamma_0
    return current_gamma
class RandomWalker:

    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.nodes = nx.nodes(self.G)
        print("Edge weighting.\n")
        for edge in tqdm(self.G.edges()):
            self.G[edge[0]][edge[1]]["weight"] = 1.0
            self.G[edge[1]][edge[0]]["weight"] = 1.0
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def get_walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def count_frequency_values(self, walks):

        raw_counts = [node for walk in walks for node in walk]
        counts = Counter(raw_counts)
        self.degrees = [counts[i] for i in range(0,len(self.nodes))]
        return self.degrees

    def simulate_walks(self, num_walks, walk_length):

        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            print(" ")
            print("Random walk series " + str(walk_iter+1) + ". initiated.")
            print(" ")
            random.shuffle(nodes)
            for node in tqdm(nodes):
                walks.append(self.get_walk(walk_length=walk_length, start_node=node))

        return walks, self.count_frequency_values(walks)

    def get_alias_edge(self, src, dst):

        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"]/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"]/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):

        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        print("")
        print("Preprocesing.\n")
        for node in tqdm(G.nodes()):
             unnormalized_probs = [G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))]
             norm_const = sum(unnormalized_probs)
             normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
             alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in tqdm(G.edges()):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

def alias_setup(probs):

    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    smaller = []
    larger = []

    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):

    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def epoch_printer(repetition):

    print("")
    print("Epoch " + str(repetition+1) + ". initiated.")
    print("")

