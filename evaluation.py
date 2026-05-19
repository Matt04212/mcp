import time
import numpy as np
from hypergraph import Hypergraph


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


def _edge_size_meta(hg):
    sizes = [len(hg.hedges_dict[e]) for e in hg.hedges]
    return {
        'min_edge_size': min(sizes),
        'mean_edge_size': round(float(np.mean(sizes)), 4),
        'median_edge_size': round(float(np.median(sizes)), 4),
        'max_edge_size': max(sizes),
    }


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


def evaluate_parallel_mcp(
    filename,
    size,
    distributions,
    n_runs,
    budget_ratio,
    nparts_list,
    whole_algo=None,
    partition_algo=None,
    partition_algos=None,
    partition_algo_kwargs=None,
    seed_base=1000,
    budget_mode='equal',
    timeout=120,
    return_details=False,
    **kwargs,
):
    """
    Compare whole-graph greedy against one-shot partition-then-local-greedy.

    The graph family comes directly from Hypergraph.generate(...), so
    distributions should use names such as:
      beta_right, beta_left, beta_bell, uniform
    """
    summary_rows = []
    detail_rows = []
    if partition_algos is None:
        if partition_algo is None:
            raise ValueError("provide partition_algo or partition_algos")
        partition_algos = {partition_algo.__name__: partition_algo}
    partition_algo_kwargs = partition_algo_kwargs or {}

    for nhedges, nvtxs in size:
        budget = max(1, int(round(budget_ratio * nhedges)))

        for dist_idx, dist in enumerate(distributions):
            by_method_nparts = {
                (method_name, nparts): []
                for method_name in partition_algos
                for nparts in nparts_list
            }

            for run in range(1, n_runs + 1):
                seed = seed_base + 1_000_000 * dist_idx + 10_000 * run + 10 * nhedges + nvtxs
                np.random.seed(seed)
                hg = Hypergraph(nhedges, nvtxs)
                hg.generate(distribution=dist)

                graph_meta = _edge_size_meta(hg)
                if whole_algo is not None:
                    whole_metrics = run_single(hg, whole_algo, filename, budget=budget, **kwargs)
                    whole_weighted = round(
                        sum(hg.vtx_weights[v] for v in whole_metrics['covered_vertices']), 6
                    )
                else:
                    whole_metrics = {
                        'covered_vertices': set(),
                        'time(s)': None,
                        'overlap_ratio': None,
                    }
                    whole_weighted = None

                for method_name, partition_func in partition_algos.items():
                    method_kwargs = partition_algo_kwargs.get(method_name, {})
                    for nparts in nparts_list:
                        partition_metrics = run_single(
                            hg,
                            partition_func,
                            filename,
                            budget=budget,
                            nparts=nparts,
                            timeout=timeout,
                            budget_mode=budget_mode,
                            partition_seed=seed,
                            **method_kwargs,
                            **kwargs,
                        )
                        partition_weighted = round(
                            sum(hg.vtx_weights[v] for v in partition_metrics['covered_vertices']), 6
                        )
                        partition_stage = partition_metrics.get('stage') or {}

                        row = {
                            'size': (nhedges, nvtxs),
                            'distribution': dist,
                            'partition_method': method_name,
                            'run': run,
                            'seed': seed,
                            'budget': budget,
                            'budget_ratio': budget_ratio,
                            'nparts': nparts,
                            'budget_mode': budget_mode,
                            'whole_coverage': len(whole_metrics['covered_vertices']) if whole_algo is not None else None,
                            'whole_weighted_coverage': whole_weighted,
                            'whole_time(s)': whole_metrics['time(s)'],
                            'whole_overlap_ratio': whole_metrics['overlap_ratio'],
                            'partition_coverage': len(partition_metrics['covered_vertices']),
                            'partition_weighted_coverage': partition_weighted,
                            'partition_time(s)': partition_metrics['time(s)'],
                            'partition_write_time(s)': partition_metrics['write_time(s)'],
                            'partition_partition_time(s)': partition_metrics['partition_time(s)'],
                            'partition_overlap_ratio': partition_metrics['overlap_ratio'],
                            'partition_local_greedy_time(s)': partition_stage.get('partition_local_greedy_time(s)'),
                            'partition_local_wall_time(s)': partition_stage.get('partition_local_wall_time(s)'),
                            'partition_estimated_parallel_local_time(s)': partition_stage.get('partition_estimated_parallel_local_time(s)'),
                            'partition_merge_time(s)': partition_stage.get('partition_merge_time(s)'),
                            'partition_count': partition_stage.get('partition_count'),
                            'requested_nparts': partition_stage.get('requested_nparts'),
                            'partition_edge_counts': partition_stage.get('partition_edge_counts'),
                            'partition_budgets': partition_stage.get('partition_budgets'),
                            'partition_selected_counts': partition_stage.get('partition_selected_counts'),
                            'partition_local_times': partition_stage.get('partition_local_times'),
                            'quality_ratio': round(partition_weighted / whole_weighted, 6)
                            if whole_weighted and whole_weighted > 0 else None,
                            'quality_gap': round(whole_weighted - partition_weighted, 6)
                            if whole_weighted is not None else None,
                            'quality_loss_pct': round(
                                100.0 * (whole_weighted - partition_weighted) / whole_weighted, 4
                            ) if whole_weighted and whole_weighted > 0 else None,
                            'time_saved(s)': round(
                                whole_metrics['time(s)'] - partition_metrics['time(s)'], 6
                            ) if whole_metrics['time(s)'] is not None else None,
                            'time_change_pct': round(
                                100.0 * (whole_metrics['time(s)'] - partition_metrics['time(s)'])
                                / whole_metrics['time(s)'],
                                4,
                            ) if whole_metrics['time(s)'] and whole_metrics['time(s)'] > 0 else None,
                            'time_ratio': round(
                                partition_metrics['time(s)'] / whole_metrics['time(s)'], 6
                            ) if whole_metrics['time(s)'] and whole_metrics['time(s)'] > 0 else None,
                            'estimated_parallel_time(s)': round(
                                (partition_metrics['partition_time(s)'] or 0.0)
                                + (partition_stage.get('partition_estimated_parallel_local_time(s)') or 0.0)
                                + (partition_stage.get('partition_merge_time(s)') or 0.0),
                                6,
                            ),
                            **graph_meta,
                        }
                        row['estimated_parallel_time_saved(s)'] = round(
                            whole_metrics['time(s)'] - row['estimated_parallel_time(s)'],
                            6,
                        ) if whole_metrics['time(s)'] is not None else None
                        row['estimated_parallel_time_change_pct'] = round(
                            100.0 * row['estimated_parallel_time_saved(s)'] / whole_metrics['time(s)'],
                            4,
                        ) if whole_metrics['time(s)'] and whole_metrics['time(s)'] > 0 else None
                        row['estimated_parallel_time_ratio'] = round(
                            row['estimated_parallel_time(s)'] / whole_metrics['time(s)'],
                            6,
                        ) if whole_metrics['time(s)'] and whole_metrics['time(s)'] > 0 else None

                        by_method_nparts[(method_name, nparts)].append(row)
                        detail_rows.append(row)

                print(f"  [{dist} {nhedges},{nvtxs}] run {run}/{n_runs} done")

            for method_name in partition_algos:
                for nparts in nparts_list:
                    runs = by_method_nparts[(method_name, nparts)]
                    summary_rows.append({
                        'size': (nhedges, nvtxs),
                        'distribution': dist,
                        'partition_method': method_name,
                        'nparts': nparts,
                        'budget': budget,
                        'budget_ratio': budget_ratio,
                        'budget_mode': budget_mode,
                        'runs': len(runs),
                        'whole_weighted_coverage': _safe_mean(runs, 'whole_weighted_coverage'),
                        'partition_weighted_coverage': _safe_mean(runs, 'partition_weighted_coverage'),
                        'quality_ratio': _safe_mean(runs, 'quality_ratio'),
                        'std_quality_ratio': _safe_std(runs, 'quality_ratio'),
                        'quality_gap': _safe_mean(runs, 'quality_gap'),
                        'quality_loss_pct': _safe_mean(runs, 'quality_loss_pct'),
                        'whole_time(s)': _safe_mean(runs, 'whole_time(s)'),
                        'partition_time(s)': _safe_mean(runs, 'partition_time(s)'),
                        'time_saved(s)': _safe_mean(runs, 'time_saved(s)'),
                        'time_change_pct': _safe_mean(runs, 'time_change_pct'),
                        'time_ratio': _safe_mean(runs, 'time_ratio'),
                        'partition_write_time(s)': _safe_mean(runs, 'partition_write_time(s)'),
                        'partition_partition_time(s)': _safe_mean(runs, 'partition_partition_time(s)'),
                        'partition_local_greedy_time(s)': _safe_mean(runs, 'partition_local_greedy_time(s)'),
                        'partition_local_wall_time(s)': _safe_mean(runs, 'partition_local_wall_time(s)'),
                        'partition_estimated_parallel_local_time(s)': _safe_mean(runs, 'partition_estimated_parallel_local_time(s)'),
                        'partition_merge_time(s)': _safe_mean(runs, 'partition_merge_time(s)'),
                        'estimated_parallel_time(s)': _safe_mean(runs, 'estimated_parallel_time(s)'),
                        'estimated_parallel_time_saved(s)': _safe_mean(runs, 'estimated_parallel_time_saved(s)'),
                        'estimated_parallel_time_change_pct': _safe_mean(runs, 'estimated_parallel_time_change_pct'),
                        'estimated_parallel_time_ratio': _safe_mean(runs, 'estimated_parallel_time_ratio'),
                        'whole_overlap_ratio': _safe_mean(runs, 'whole_overlap_ratio'),
                        'partition_overlap_ratio': _safe_mean(runs, 'partition_overlap_ratio'),
                        'min_edge_size': _safe_mean(runs, 'min_edge_size'),
                        'mean_edge_size': _safe_mean(runs, 'mean_edge_size'),
                        'median_edge_size': _safe_mean(runs, 'median_edge_size'),
                        'max_edge_size': _safe_mean(runs, 'max_edge_size'),
                    })

    if return_details:
        return summary_rows, detail_rows
    return summary_rows


def evaluate_optimal(algos, filename, size, distributions, n_runs, budget_ratio, **kwargs):
    from optimal import optimal_solu

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
