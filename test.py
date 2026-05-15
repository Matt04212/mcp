from hypergraph import Hypergraph
from analysis import analyze_graph, plot_graph

dist = 'uniform'

hg = Hypergraph(10000, 20000)
hg.generate(distribution=dist)

a, _, _ = analyze_graph(hg, dist)
print(a)
#plot_graph(hg, dist)