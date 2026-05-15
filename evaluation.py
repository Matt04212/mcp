import time
import numpy as np
from hypergraph import Hypergraph
from optimal import optimal_solu


# ─────────────────────────────────────────────
#  Overlap helpers
# ─────────────────────────────────────────────

def _compute_overlap(hg, removed_edges):
    """
    Compute overlap metrics for a solution.
    Returns dict with overlap_ratio and pairwise_overlap_avg.
    pairwise is O(k^2) — fine for small budgets, skip for large.
    """
    selected = list(removed_edges)
    if not selected:
        return {'overlap_ratio': None, 'pairwise_overlap_avg': None,
                'multi_covered_vertices': None}

    all_covered = set()
    total_raw   = 0
    for e in selected:
        verts = hg.hedges_dict[e]
        all_covered.update(verts)
        total_raw += len(verts)

    overlap_count = total_raw - len(all_covered)
    overlap_ratio = overlap_count / total_raw if total_raw > 0 else 0

    # pairwise — skip if budget too large (>500) to avoid O(k^2) slowdown
    if len(selected) <= 500:
        pairwise = []
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                pairwise.append(
                    len(hg.hedges_dict[selected[i]] & hg.hedges_dict[selected[j]])
                )
        pairwise_avg = float(np.mean(pairwise)) if pairwise else 0.0
    else:
        pairwise_avg = None

    vertex_cover_count = {}
    for e in selected:
        for v in hg.hedges_dict[e]:
            vertex_cover_count[v] = vertex_cover_count.get(v, 0) + 1
    multi_covered = sum(1 for c in vertex_cover_count.values() if c > 1)

    return {
        'overlap_ratio':          round(overlap_ratio, 4),
        'pairwise_overlap_avg':   round(pairwise_avg, 4) if pairwise_avg is not None else None,
        'multi_covered_vertices': multi_covered,
    }


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
    for r in runs:
        stage = r.get('stage') or {}
        for k, v in stage.items():
            r[k] = v

    stage_keys = ['early_coverage', 'early_time',
                  'mid_coverage',   'mid_time',
                  'late_coverage',  'late_time']

    overlap_keys = ['overlap_ratio', 'pairwise_overlap_avg', 'multi_covered_vertices']

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

    for k in overlap_keys:
        result[k]          = _safe_mean(runs, k)
        result[f'std_{k}'] = _safe_std(runs,  k)

    return result


def _average_metrics_optimal(runs, nhedges, nvtxs, dist, algo_name, n_runs):
    overlap_keys = ['overlap_ratio', 'pairwise_overlap_avg', 'multi_covered_vertices']

    result = {
        'size':              (nhedges, nvtxs),
        'distribution':      dist,
        'algo_name':         algo_name,
        'time(s)':           _safe_mean(runs, 'time(s)'),
        'fraction_optimal':  round(sum(r['is_optimal'] for r in runs) / n_runs, 4),
        'performance':       _safe_mean(runs, 'performance'),
        'std_performance':   _safe_std(runs,  'performance'),
        'write_time(s)':     _safe_mean(runs, 'write_time(s)'),
        'partition_time(s)': _safe_mean(runs, 'partition_time(s)'),
    }

    for k in overlap_keys:
        result[k]          = _safe_mean(runs, k)
        result[f'std_{k}'] = _safe_std(runs,  k)

    return result


def _average_metrics_set_cover(runs, nhedges, nvtxs, dist, algo_name, n_runs):
    for r in runs:
        stage = r.get('stage') or {}
        for k, v in stage.items():
            r[k] = v

    stage_keys   = ['early_edges', 'early_time',
                    'mid_edges',   'mid_time',
                    'late_edges',  'late_time']
    overlap_keys = ['overlap_ratio', 'pairwise_overlap_avg', 'multi_covered_vertices']

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

    for k in overlap_keys:
        result[k]          = _safe_mean(runs, k)
        result[f'std_{k}'] = _safe_std(runs,  k)

    return result


# ─────────────────────────────────────────────
#  Single-run wrappers
# ─────────────────────────────────────────────

def run_single(hg, algo_func, filename, budget, **kwargs):
    t          = time.time()
    result     = algo_func(hg, budget=budget, filename=filename, **kwargs)
    total_time = time.time() - t

    overlap = _compute_overlap(hg, result[2])

    metrics = {
        'size':              result[0],
        'covered_vertices':  result[1],
        'solution':          result[2],
        'time(s)':           round(total_time, 4),
        'write_time(s)':     round(result[3], 4) if result[3] is not None else None,
        'partition_time(s)': round(result[4], 4) if result[4] is not None else None,
        'stage':             result[5],
    }
    metrics.update(overlap)
    return metrics


def run_single_set_cover(hg, algo_func, filename, **kwargs):
    t          = time.time()
    result     = algo_func(hg, filename=filename, **kwargs)
    total_time = time.time() - t

    overlap = _compute_overlap(hg, result[2])

    metrics = {
        'size':              result[0],
        'edges_used':        len(result[2]),
        'solution':          result[2],
        'time(s)':           round(total_time, 4),
        'write_time(s)':     round(result[3], 4) if result[3] is not None else None,
        'partition_time(s)': round(result[4], 4) if result[4] is not None else None,
        'stage':             result[5],
    }
    metrics.update(overlap)
    return metrics


# ─────────────────────────────────────────────
#  Evaluation loops
# ─────────────────────────────────────────────

def evaluate_mcp(algos, filename, size, distributions, n_runs, budget_ratio, **kwargs):
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
                    metrics['performance']  = round(weighted_coverage / o, 4)
                    metrics['is_optimal']   = weighted_coverage >= o
                    run_results[algo_name].append(metrics)

                print(f"  [{dist} {nhedges},{nvtxs}] run {run+1}/{n_runs} done")

            for algo_name in algos:
                runs = run_results[algo_name]
                all_results.append(
                    _average_metrics_optimal(runs, nhedges, nvtxs, dist, algo_name, n_runs)
                )

    return all_results


def evaluate_set_cover(algos, filename, size, distributions, n_runs_map=None, n_runs=5, **kwargs):
    all_results = []
    for nhedges, nvtxs in size:
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
                all_results.append(
                    _average_metrics_set_cover(
                        runs, nhedges, nvtxs, dist, algo_name, runs_this_size
                    )
                )

    return all_results