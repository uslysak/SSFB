

import pandas as pd
import networkx as nx

def graph_reader(input_path):
    edges = pd.read_csv(input_path)
    G = nx.from_edgelist(edges.values.tolist())
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G
G = graph_reader('./data/Dolphin.csv')