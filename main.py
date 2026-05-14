from algo import (hmetis_set_cover, pure_greedy_set_cover, hmetis_mcp, pure_greedy_mcp, pure_greedy_mcp_0,
hmetis_set_cover_greedy_first, hmetis_mcp_early)
from evaluation import evaluate_mcp, evaluate_optimal
import pandas as pd


"""distributions1 = [
    'dis'
]

size1 = [
    (1000, 2000),
    #(35000, 70000),
    #(40000, 80000),
    #(45000, 90000),
    #(50000, 100000),
    #(55000, 110000),
    #(60000, 120000)
]

all_results = []
all_stats = []


algos1 = {
    #'hmetis': hmetis_mcp,
    #'hmetis_first': hmetis_mcp_early,
    'hmetis_swap': hmetis_mcp_swap,
    #'pure_greedy_0' : pure_greedy_mcp_0,
    'pure_greedy': pure_greedy_mcp

}
results1 = evaluate_mcp(
    algos=algos1,
    filename="data1.hgr",
    size=size1,
    distributions=distributions1,
    n_runs=10,
    budget_ratio = 0.1
)"""

#df1 = pd.DataFrame(results1)
#print(df1.to_string(index=False))

distributions2 = [
    #'beta_right',
    #'beta_bell',
    #'beta_left'
    #'uniform',
    'dis',
    #'dis2',
    #clustered'
]

size2 = [
    (160, 200)
]


algos2 = {
    'hmetis': hmetis_mcp,
    #'hmetis_first': hmetis_mcp_early,
    #'pure_greedy_0' : pure_greedy_mcp_0,
    'pure_greedy': pure_greedy_mcp
}

results2 = evaluate_optimal(
    algos=algos2,
    filename="data2.hgr",
    size=size2,
    distributions=distributions2,
    n_runs=10,
    budget_ratio = 0.1
)

df2 = pd.DataFrame(results2)
print(df2.to_string(index=False))

