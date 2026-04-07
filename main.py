from generator import hypergraph
from algo import algo
import matplotlib.pyplot as plt

hg = hypergraph(1000,1000)
hg.generate()
hg.output("data1.hgr")

output = algo(hg)
