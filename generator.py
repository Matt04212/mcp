import numpy as np
import matplotlib.pyplot as plt

class hypergraph:
    def __init__(self, num_edges, num_nodes):
        self.edge = num_edges
        self.node = num_nodes

    def elements(self):
        self.edges = []
        self.nodes = []
        for i in range(self.edge):
            self.edges.append(i)
        return self.edges
        for i in range(self.node):
            self.nodes.append(i)
        return self.nodes

    def distribution(self):
        edge_size = np.ceil(np.random.exponential(scale=25.0, size=self.edge)).astype(int)
        edge_size = np.clip(edge_size, 1, self.node)
        plt.hist(edge_size)
        plt.show()
        print(edge_size)


