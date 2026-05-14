import time
import numpy as np
from hypergraph import Hypergraph
from optimal import optimal_solu


# ─────────────────────────────────────────────
#  Averaging helpers
# ─────────────────────────────────────────────

def _safe_mean(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return round(np.mean(vals), 4) if vals else None

def _safe_std(runs, key):
    vals = [r[key] for r in runs if r.get(key) is not None]
    return round(np.std(vals), 4) if vals else None


def _average_metrics(runs, nhedges, nvtxs, dist, algo_name, n_runs):
    """
    Aggregation for MCP runs.
    Includes overall metrics + per-stage (early/mid/late) coverage and time.
    Stage keys come from _StageTrackerMCP.result():
      early_coverage, early_time, mid_coverage, mid_time, late_coverage, late_time
    """
    # flatten stage dicts into each run for easy averaging
    for r in runs:
        stage = r.get('stage') or {}
        for k, v in stage.items():
            r[k] = v

    stage_keys = ['early_coverage', 'early_time',
                  'mid_coverage',   'mid_time',
                  'late_coverage',  'late_time']

    result = {
        'size':              (nhedges, nvtxs),
        'distribution':      dist,
        'algo_name':         algo_name,
        'time(s)':           _safe_mean(runs, 'time(s)'),
        'performance':       _safe_mean(runs, 'performance'),
        'std_performance':   _safe_std(runs,  'performance'),
        'write_time(s)':     _safe_mean(runs, 'write_time(s)'),
        'partition_time(s)': _safe_mean(runs, 'partition_time(s)'),
    }

    for k in stage_keys:
        result[k]          = _safe_mean(runs, k)
        result[f'std_{k}'] = _safe_std(runs,  k)

    return result

def _average_metrics_optimal(runs, nhedges, nvtxs, dist, algo_name, n_runs):
    """Aggregation for MCP runs."""
    return {
        'size':             (nhedges, nvtxs),
        'distribution':     dist,
        'algo_name':        algo_name,
        'time(s)':          _safe_mean(runs, 'time(s)'),
        'fraction_optimal': round(sum(r['is_optimal'] for r in runs) / n_runs, 4),
        'performance':      _safe_mean(runs, 'performance'),
        'std_performance':  _safe_std(runs,  'performance'),
        'write_time(s)':    _safe_mean(runs, 'write_time(s)'),
        'partition_time(s)':_safe_mean(runs, 'partition_time(s)'),
    }


def _average_metrics_set_cover(runs, nhedges, nvtxs, dist, algo_name, n_runs):
    """
    Aggregation for set-cover runs.
    Includes overall metrics + per-stage (early/mid/late) edges and time.
    Stage keys come from _StageTracker.result():
      early_edges, early_time, mid_edges, mid_time, late_edges, late_time
    """
    # flatten stage dicts into each run for easy averaging
    for r in runs:
        stage = r.get('stage') or {}
        for k, v in stage.items():
            r[k] = v

    stage_keys = ['early_edges', 'early_time',
                  'mid_edges',   'mid_time',
                  'late_edges',  'late_time']

    result = {
        'size':              (nhedges, nvtxs),
        'distribution':      dist,
        'algo_name':         algo_name,
        'time(s)':           _safe_mean(runs, 'time(s)'),
        'edges_used':        _safe_mean(runs, 'edges_used'),
        'std_edges_used':    _safe_std(runs,  'edges_used'),
        'write_time(s)':     _safe_mean(runs, 'write_time(s)'),
        'partition_time(s)': _safe_mean(runs, 'partition_time(s)'),
    }

    for k in stage_keys:
        result[k]          = _safe_mean(runs, k)
        result[f'std_{k}'] = _safe_std(runs,  k)

    return result




# ─────────────────────────────────────────────
#  Single-run wrappers
# ─────────────────────────────────────────────

def run_single(hg, algo_func, filename, budget, **kwargs):
    """Run one MCP algo on one graph — returns metrics dict."""
    t          = time.time()
    result     = algo_func(hg, budget=budget, filename=filename, **kwargs)
    total_time = time.time() - t

    # result = (size, covered_vertices, removed_edges, write_time, partition_time, stages)
    return {
        'size':             result[0],
        'covered_vertices': result[1],
        'solution':         result[2],
        'time(s)':          round(total_time, 4),
        'write_time(s)':    round(result[3], 4) if result[3] is not None else None,
        'partition_time(s)':round(result[4], 4) if result[4] is not None else None,
        'stage':            result[5],
    }


def run_single_set_cover(hg, algo_func, filename, **kwargs):
    """Run one set-cover algo on one graph — returns metrics dict."""
    t          = time.time()
    result     = algo_func(hg, filename=filename, **kwargs)
    total_time = time.time() - t

    # result = (size, covered_vertices, removed_edges, write_time, partition_time, stages)
    return {
        'size':             result[0],
        'edges_used':       len(result[2]),
        'solution':         result[2],
        'time(s)':          round(total_time, 4),
        'write_time(s)':    round(result[3], 4) if result[3] is not None else None,
        'partition_time(s)':round(result[4], 4) if result[4] is not None else None,
        'stage':            result[5],
    }


# ─────────────────────────────────────────────
#  Evaluation loops
# ─────────────────────────────────────────────

def evaluate_mcp(algos, filename, size, distributions, n_runs, budget_ratio, **kwargs):
    """Maximum Coverage Problem — fixed budget, maximize coverage."""
    all_results = []
    for nhedges, nvtxs in size:
        budget = int(budget_ratio * nhedges)
        for dist in distributions:
            run_results = {algo_name: [] for algo_name in algos}

            for run in range(n_runs):
                hg = Hypergraph(nhedges, nvtxs)
                hg.generate(distribution=dist)
                hg.output('original.hgr')

                for algo_name, algo_func in algos.items():
                    metrics = run_single(hg, algo_func, filename, budget=budget, **kwargs)
                    weighted_coverage = round(
                        sum(hg.vtx_weights[v] for v in metrics['covered_vertices']), 6
                    )
                    metrics['performance'] = round(weighted_coverage, 4)
                    run_results[algo_name].append(metrics)

                print(f"  [{dist} {nhedges},{nvtxs}] run {run+1}/{n_runs} done")

            for algo_name in algos:
                runs = run_results[algo_name]
                all_results.append(
                    _average_metrics(runs, nhedges, nvtxs, dist, algo_name, n_runs)
                )

    return all_results

def evaluate_optimal(algos, filename, size, distributions, n_runs, budget_ratio, **kwargs):
    """Maximum Coverage Problem — fixed budget, maximize coverage."""
    all_results = []
    for nhedges, nvtxs in size:
        budget = int(budget_ratio * nhedges)
        for dist in distributions:
            run_results = {algo_name: [] for algo_name in algos}

            for run in range(n_runs):
                hg = Hypergraph(nhedges, nvtxs)
                hg.generate(distribution=dist)
                hg.output('original.hgr')

                o = optimal_solu(hg, budget=budget)

                for algo_name, algo_func in algos.items():
                    metrics = run_single(hg, algo_func, filename, budget=budget, **kwargs)
                    weighted_coverage = round(
                        sum(hg.vtx_weights[v] for v in metrics['covered_vertices']), 6
                    )
                    metrics['performance'] = round(weighted_coverage/o, 4)
                    metrics['is_optimal'] = weighted_coverage >= o
                    run_results[algo_name].append(metrics)

                print(f"  [{dist} {nhedges},{nvtxs}] run {run+1}/{n_runs} done")

            for algo_name in algos:
                runs = run_results[algo_name]
                all_results.append(
                    _average_metrics_optimal(runs, nhedges, nvtxs, dist, algo_name, n_runs)
                )

    return all_results


def evaluate_set_cover(algos, filename, size, distributions, n_runs_map=None, n_runs=5, **kwargs):
    """
    Set Cover Problem — no budget, minimize edges to cover all vertices.

    n_runs_map: dict mapping (nhedges, nvtxs) -> int, for adaptive run counts.
                e.g. {(20000,40000): 10, (50000,100000): 5}
                Falls back to n_runs if a size is not in the map.
    """
    all_results = []
    for nhedges, nvtxs in size:
        # adaptive run count: more runs for small graphs, fewer for large
        runs_this_size = (n_runs_map or {}).get((nhedges, nvtxs), n_runs)

        for dist in distributions:
            run_results = {algo_name: [] for algo_name in algos}

            for run in range(runs_this_size):
                hg = Hypergraph(nhedges, nvtxs)
                hg.generate(distribution=dist)
                hg.output('original.hgr')

                for algo_name, algo_func in algos.items():
                    metrics = run_single_set_cover(hg, algo_func, filename, **kwargs)
                    run_results[algo_name].append(metrics)

                print(f"  [{dist} {nhedges},{nvtxs}] run {run+1}/{runs_this_size} done")

            for algo_name in algos:
                runs = run_results[algo_name]
                avg  = _average_metrics_set_cover(
                    runs, nhedges, nvtxs, dist, algo_name, runs_this_size
                )
                all_results.append(avg)

    return all_results