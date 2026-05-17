from algo import hmetis_partitioned_greedy_mcp, pure_greedy_mcp
from evaluation import evaluate_mcp
import pandas as pd
from analysis import plot_graph, analyze_graph
from hypergraph import Hypergraph

algos = {
    'greedy': pure_greedy_mcp,
    'hmeits': hmetis_partitioned_greedy_mcp
}

dist = [
    'beta_right',
    #'beta_left',
    #'beta_bell',
    #'uniform'
]

size = [(20000, 40000)]

result = evaluate_mcp(
    algos=algos,
    distributions=dist,
    size=size,
    filename='data1.hgr',
    budget_ratio=0.1,
    n_runs=1
)

df = pd.DataFrame(result)
print(df.to_string(index=False))



