from datetime import datetime
from pathlib import Path
from tempfile import gettempdir

import pandas as pd

from algo import (
    hmetis_partitioned_greedy_mcp,
    hype_partitioned_greedy_mcp,
    pure_greedy_mcp,
    random_partitioned_greedy_mcp,
)
from evaluation import evaluate_parallel_mcp


DIST = [
    #'beta_right',
    #'beta_bell',
    #'beta_left',
    #'uniform',
    'spatial_right_skewed',
    #'spatial_bell',
    #'spatial_left_skewed',
    #'spatial_uniform'
]

SIZE = [
    (2500, 5000),
    #(5000, 10000),
    #(7500, 15000),
    #(10000, 20000),
    #(20000, 40000),
    #(30000, 60000),
    #(50000, 100000),
    #(70000, 1400000),
    #(100000, 200000),
    #(150000, 300000),
    #(300000, 600000)
]

NPARTS = [2]
PARTITION_METHODS = {
    'hmetis': {
        'enabled': True,
        'func': hmetis_partitioned_greedy_mcp,
        'kwargs': {
            'hmetis_params': {
                'ubfactor': 5,
                'nruns': 1,
                'ctype': 5,
                'rtype': 3,
                'vcycle': 0,
                'reconst': 0,
                'dbglvl': 0,
            },
        },
    },
    'random': {
        'enabled': True,
        'func': random_partitioned_greedy_mcp,
        'kwargs': {},
    },
    'hype': {
        'enabled': True,
        'func': hype_partitioned_greedy_mcp,
        'kwargs': {
            'hype_params': {
                'fringe_size': 10,
                'fringe_candidates': 2,
            },
        },
    },
}

N_RUNS = 1
BUDGET_RATIO = 0.10
BUDGET_MODE = 'equal'
TIMEOUT = 60
SEED_BASE = 700_000
RUN_WHOLE_BASELINE = True


RUN_STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FILENAME = str(Path(gettempdir()) / f'parallel_greedy_beta_{RUN_STAMP}.hgr')


ACTIVE_PARTITION_METHODS = {
    name: cfg['func']
    for name, cfg in PARTITION_METHODS.items()
    if cfg.get('enabled')
}
ACTIVE_PARTITION_METHOD_KWARGS = {
    name: cfg.get('kwargs', {})
    for name, cfg in PARTITION_METHODS.items()
    if cfg.get('enabled')
}

if not ACTIVE_PARTITION_METHODS:
    raise ValueError("enable at least one partition method in PARTITION_METHODS")


summary_rows = evaluate_parallel_mcp(
    filename=FILENAME,
    size=SIZE,
    distributions=DIST,
    n_runs=N_RUNS,
    budget_ratio=BUDGET_RATIO,
    whole_algo=pure_greedy_mcp if RUN_WHOLE_BASELINE else None,
    nparts_list=NPARTS,
    partition_algos=ACTIVE_PARTITION_METHODS,
    partition_algo_kwargs=ACTIVE_PARTITION_METHOD_KWARGS,
    budget_mode=BUDGET_MODE,
    timeout=TIMEOUT,
    seed_base=SEED_BASE,
    return_details=False,
)


summary_df = pd.DataFrame(summary_rows)
display_columns = [
    'distribution',
    'partition_method',
    'size',
    'budget',
    'nparts',
    'runs',
]
if RUN_WHOLE_BASELINE:
    display_columns.extend([
        'quality_ratio',
        'quality_loss_pct',
        'whole_time(s)',
        'partition_time(s)',
        'time_saved(s)',
        'time_change_pct',
        'time_ratio',
        'estimated_parallel_time(s)',
        'estimated_parallel_time_saved(s)',
        'estimated_parallel_time_ratio',
    ])
else:
    display_columns.extend([
        'partition_time(s)',
        'partition_partition_time(s)',
        'partition_local_greedy_time(s)',
        'partition_estimated_parallel_local_time(s)',
        'partition_merge_time(s)',
    ])

print("\n=== Summary ===")
print(summary_df[display_columns].to_string(index=False))

summary_path = RESULTS_DIR / f'parallel_greedy_beta_summary_{RUN_STAMP}.csv'

summary_df.to_csv(summary_path, index=False)

print(f"Saved summary rows to {summary_path}")
