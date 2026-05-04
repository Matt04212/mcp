import time
from optimal import optimal_solu
from analysis import analyze_graph
from hypergraph import Hypergraph
import numpy as np
def _average_metrics(runs, nhedges, nvtxs, dist, algo_name, n_runs):
    return {
        'size': (nhedges, nvtxs),
        'distribution': dist,
        'algo_name': algo_name,
        'time(s)': round(np.mean([r['time(s)'] for r in runs]), 4),
        'fraction_optimal': round(sum(r['is_optimal'] for r in runs) / n_runs, 4),
        'performance': round(np.mean([r['performance'] for r in runs]), 4),
        'std_performance': round(np.std([r['performance'] for r in runs]), 4),
        'write_time(s)': round(np.mean([r['write_time(s)'] for r in runs
                               if r['write_time(s)'] is not None]), 4)
                         if any(r['write_time(s)'] is not None for r in runs) else None,
        'partition_time(s)': round(np.mean([r['partition_time(s)'] for r in runs
                                   if r['partition_time(s)'] is not None]), 4)
                             if any(r['partition_time(s)'] is not None for r in runs) else None,
    }

def _average_metrics_set_cover(runs, nhedges, nvtxs, dist, algo_name, n_runs):
    return {
        'size': (nhedges, nvtxs),
        'distribution': dist,
        'algo_name': algo_name,
        'time(s)': round(np.mean([r['time(s)'] for r in runs]), 4),
        'edges_used': round(np.mean([r['edges_used'] for r in runs]), 2),
        'std_edges_used': round(np.std([r['edges_used'] for r in runs]), 4),
        'write_time(s)': round(np.mean([r['write_time(s)'] for r in runs
                               if r['write_time(s)'] is not None]), 4)
                         if any(r['write_time(s)'] is not None for r in runs) else None,
        'partition_time(s)': round(np.mean([r['partition_time(s)'] for r in runs
                                   if r['partition_time(s)'] is not None]), 4)
                             if any(r['partition_time(s)'] is not None for r in runs) else None,
    }

def run_single(hg, algo_func, filename, budget, **kwargs):
    """runs one algo on one graph, returns metrics"""
    t = time.time()
    result = algo_func(hg, budget=budget, filename=filename, **kwargs)
    total_time = time.time() - t
    return {
        'size': result[0],
        'covered_vertices': result[1],
        'solution': result[2],
        'time(s)': round(total_time, 4),
        'write_time(s)': round(result[3], 4) if len(result) > 3 else None,
        'partition_time(s)': round(result[4], 4) if len(result) > 4 else None,
    }

def run_single_set_cover(hg, algo_func, filename, **kwargs):
    t = time.time()
    result = algo_func(hg, filename=filename, **kwargs)  # no budget
    total_time = time.time() - t
    return {
        'size': result[0],
        'edges_used': len(result[2]),
        'solution': result[2],
        'time(s)': round(total_time, 4),
        'write_time(s)': round(result[3], 4) if len(result) > 3 else None,
        'partition_time(s)': round(result[4], 4) if len(result) > 4 else None,
    }

def evaluate_mcp(algos, filename, size, distributions, n_runs, budget, **kwargs):
    """Maximum Coverage Problem - fixed budget, maximize coverage"""
    all_results = []
    for nhedges, nvtxs in size:
        for dist in distributions:
            run_results = {algo_name: [] for algo_name in algos}
            for run in range(n_runs):
                hg = Hypergraph(nhedges, nvtxs)
                hg.generate(distribution=dist)
                hg.output('original.hgr')
                o = optimal_solu(hg, budget)

                for algo_name, algo_func in algos.items():
                    metrics = run_single(hg, algo_func, filename, budget=budget, **kwargs)
                    weighted_coverage = round(sum(hg.vtx_weights[v] for v in metrics['covered_vertices']), 4)
                    metrics['performance'] = round(weighted_coverage / o, 4)
                    metrics['is_optimal'] = weighted_coverage >= o - 1e-6
                    run_results[algo_name].append(metrics)

            for algo_name in algos:
                runs = run_results[algo_name]
                all_results.append(_average_metrics(runs, nhedges, nvtxs, dist, algo_name, n_runs))

    return all_results


def evaluate_set_cover(algos, filename, size, distributions, n_runs, **kwargs):
    """Set Cover Problem - no budget, minimize edges to cover all vertices"""
    all_results = []
    for nhedges, nvtxs in size:
        for dist in distributions:
            run_results = {algo_name: [] for algo_name in algos}
            for run in range(n_runs):
                hg = Hypergraph(nhedges, nvtxs)
                hg.generate(distribution=dist)
                hg.output('original.hgr')

                for algo_name, algo_func in algos.items():
                    metrics = run_single_set_cover(hg, algo_func, filename, **kwargs)
                    run_results[algo_name].append(metrics)

            for algo_name in algos:
                runs = run_results[algo_name]
                avg = _average_metrics_set_cover(runs, nhedges, nvtxs, dist, algo_name, n_runs)
                all_results.append(avg)

    return all_results
