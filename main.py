from hypergraph import Hypergraph
from algo import algo
import matplotlib.pyplot as plt

hg = Hypergraph(10000,10000)
hg.generate()
hg.output("data1.hgr")

solution = algo(hg, 4)
