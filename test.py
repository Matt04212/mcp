from hypergraph import Hypergraph
from analysis import plot_graph, analyze_graph
from algo import hmetis_set_cover

distributions = ['dis']


for dist in distributions:
    hg = Hypergraph(30000, 60000)
    hg.generate(distribution=dist)
    stats, _, _ = analyze_graph(hg, dist)
    print(stats)
    #plot_graph(hg, dist)

