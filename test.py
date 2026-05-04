from hypergraph import Hypergraph
from analysis import plot_graph, analyze_graph
from algo import hmetis_set_cover

distributions = ['gamma']


for dist in distributions:
    hg = Hypergraph(10000, 20000)
    hg.generate(distribution=dist)

    plot_graph(hg, dist)


