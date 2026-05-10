from algo import hmetis_set_cover, pure_greedy_set_cover, hmetis_mcp, pure_greedy_mcp
from evaluation import evaluate_mcp, evaluate_set_cover
import pandas as pd



"""distributions1 = [
    #'exponential',
    #'dis',
    'dis2'
]

size1 = [
    (80, 100)
]

all_results = []
all_stats = []

algos1 = {
    'hmetis': hmetis_mcp,
    'pure_greedy': pure_greedy_mcp

}
results1 = evaluate_mcp(
    algos=algos1,
    filename="data1.hgr",
    size=size1,
    distributions=distributions1,
    n_runs=1,
    budget_ratio = 0.1
)
df1 = pd.DataFrame(results1)
print(df1.to_string(index=False))"""


distributions2 = [
    #'gamma',
    #'reverse_gamma',
    #'uniform',
    'dis',
    #'dis2'
]

size2 = [
    (30000, 60000),
    (35000, 70000),
    (40000, 80000),
    (45000, 90000),
    (50000, 100000)
]


algos2 = {
    #'hmetis': hmetis_set_cover,
    'hmetis': hmetis_set_cover,
    'pure_greedy': pure_greedy_set_cover
}

results2 = evaluate_set_cover(
    algos=algos2,
    filename="data2.hgr",
    size=size2,
    distributions=distributions2,
    n_runs=2
)

df2 = pd.DataFrame(results2)
print(df2.to_string(index=False))

"""df = pd.DataFrame(all_stats).set_index('distribution').T
print(df.to_string())"""


