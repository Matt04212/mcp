import numpy as np
import random
import matplotlib.pyplot as plt

class hypergraph:
    def __init__(self, num_edges, num_nodes):
        self.num_edge = num_edges
        self.num_node = num_nodes

    def elements(self):
        self.edges = []
        self.nodes = []
        for i in range(self.num_edge):
            self.edges.append(i)
        for i in range(self.num_node):
            self.nodes.append(i)

    def distribution(self):
        edge_size = np.ceil(np.random.exponential(scale=20.0, size=self.num_edge)).astype(int)
        self.edge_size = np.clip(edge_size, 1, self.num_node)
        edges_dict = {}
        nodes_dict = {node: set() for node in self.nodes}
        for edge, size in zip(self.edges, self.edge_size):
            c = set(random.sample(self.nodes, size))
            edges_dict[edge] = c
            for node in c:
                nodes_dict[node].add(edge)

        for node in nodes_dict:
            if not nodes_dict[node]:
                r = random.choice(self.edges)
                edges_dict[r].add(node)
                nodes_dict[node].add(r)





