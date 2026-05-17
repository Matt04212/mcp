import pandas as pd

from algo import hmetis_partitioned_greedy_mcp, pure_greedy_mcp
from evaluation import evaluate_parallel_mcp


DIST = [
    'beta_right',
    #'beta_bell',
    #'beta_left',
    #'uniform',
]

SIZE = [
    (10000, 20000),
]

NPARTS = [2]

N_RUNS = 1
BUDGET_RATIO = 0.10
BUDGET_MODE = 'equal'
TIMEOUT = 60
SEED_BASE = 700_000
FILENAME = 'parallel_greedy_beta.hgr'


summary_rows, detail_rows = evaluate_parallel_mcp(
    filename=FILENAME,
    size=SIZE,
    distributions=DIST,
    n_runs=N_RUNS,
    budget_ratio=BUDGET_RATIO,
    whole_algo=pure_greedy_mcp,
    partition_algo=hmetis_partitioned_greedy_mcp,
    nparts_list=NPARTS,
    budget_mode=BUDGET_MODE,
    timeout=TIMEOUT,
    seed_base=SEED_BASE,
    return_details=True,
)


summary_df = pd.DataFrame(summary_rows)
detail_df = pd.DataFrame(detail_rows)

print("\n=== Summary ===")
print(
    summary_df[
        [
            'distribution',
            'size',
            'budget',
            'nparts',
            'runs',
            'quality_ratio',
            'quality_loss_pct',
            'whole_time(s)',
            'partition_time(s)',
            'partition_wins',
            'ties',
            'partition_losses',
        ]
    ].to_string(index=False)
)

summary_df.to_csv('parallel_greedy_beta_summary.csv', index=False)
detail_df.to_csv('parallel_greedy_beta_results.csv', index=False)

print("\nSaved detailed rows to parallel_greedy_beta_results.csv")
print("Saved summary rows to parallel_greedy_beta_summary.csv")
