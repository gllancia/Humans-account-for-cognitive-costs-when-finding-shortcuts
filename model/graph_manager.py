import networkx as nx
import json
import numpy as np

def get_adjacency_matrix(G):
    return np.array(nx.linalg.graphmatrix.adjacency_matrix(G).todense())

def load_graph(filename, filedir="../data/maps/processed_maps/"):
    with open(filedir+filename) as f:
        graph = json.load(f)
    
    G = nx.Graph()
    G.add_nodes_from(np.array(list(graph["pos"].keys()),dtype="int"))
    G.add_edges_from(graph["edgelist"])
    
    G.pos = {int(key) : graph["pos"][key] for key in graph["pos"].keys()}
    
    G.adjacency_matrix = get_adjacency_matrix(G)
    
    G.start = graph.get("start")
    G.goals = graph.get("goals")
    
    G.grid_height = graph.get("height")
    G.grid_width = graph.get("width")
    
    return G