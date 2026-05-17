import contextlib
import io
import random
import time

import numpy as np
import pandas as pd

from algo import (
    hmetis_mcp,
    hmetis_mcp_refine,
    pure_greedy_mcp,
    pure_oblswap_mcp,
    pure_tabu_mcp,
)
from evaluation import _compute_overlap
from graph_generators import build_dis2_graph, build_natural_hierarchy_graph
from optimal import optimal_solu_detail


"""
Clean MCP experiment runner.

Paper-style sizing:
    F = number of hedges/sets/facilities
    |U| = number of vertices/elements/users/locations
    F = 0.5|U| or 0.8|U|
    budget = 0.1F or 0.2F

Graph families:
    dis2:
        geometric/spatial graph from hypergraph.py
    natural_hierarchy:
        stochastic community/subcommunity graph

Algorithms:
    pure_greedy
    original_hmetis
    hmetis_refine
    pure_tabu

Change the CONFIG below to run larger/smaller experiments.
"""


CONFIG = {
    # Default graph list. Individual scenarios can override this with their
    # own "graph_types" field.
    "graph_types": ["dis2_weighted"],
    "results_file": "mcp_two_graph_compare_results.csv",
    "summary_file": "mcp_two_graph_compare_summary.csv",
    "hmetis_file": "mcp_two_graph_compare.hgr",

    # LP is useful for small instances only. For bigger cases, compare final
    # coverage and runtime instead.
    "solve_lp_when_enabled": True,
    "lp_time_limit_s": 20,
    "lp_max_nvtxs": 500,
    "lp_max_nhedges": 400,
}


SCENARIOS = [
    # Paper-style dis2 instances. LP is enabled so performance and fraction
    # optimal are computed like the paper.
    {"name": "paper_dis2_100_F0.5_B0.1", "graph_types": ["dis2_weighted"], "nvtxs": 100, "f_ratio": 0.5, "budget_ratio": 0.1, "runs": 10, "solve_lp": True, "nparts": 16},
    {"name": "paper_dis2_150_F0.5_B0.1", "graph_types": ["dis2_weighted"], "nvtxs": 150, "f_ratio": 0.5, "budget_ratio": 0.1, "runs": 10, "solve_lp": True, "nparts": 16},
    {"name": "paper_dis2_200_F0.5_B0.1", "graph_types": ["dis2_weighted"], "nvtxs": 200, "f_ratio": 0.5, "budget_ratio": 0.1, "runs": 10, "solve_lp": True, "nparts": 16},

    # Structured hierarchy instances. These are the main cases for evaluating
    # whether hMETIS-refine captures a meaningful fraction of Tabu's gain with
    # less runtime.
    {"name": "hier_1200_F0.5_B0.1", "graph_types": ["natural_hierarchy"], "nvtxs": 1200, "f_ratio": 0.5, "budget_ratio": 0.1, "runs": 10, "solve_lp": False, "nparts": 16},
    {"name": "hier_2400_F0.5_B0.1", "graph_types": ["natural_hierarchy"], "nvtxs": 2400, "f_ratio": 0.5, "budget_ratio": 0.1, "runs": 10, "solve_lp": False, "nparts": 16},
]


GRAPH_BUILDERS = {
    "dis2_uniform": {
        "fn": build_dis2_graph,
        "kwargs": {"rmax": "auto", "weight_mode": "uniform"},
    },
    "dis2_unweighted": {
        "fn": build_dis2_graph,
        "kwargs": {"rmax": "auto", "weight_mode": "uniform"},
    },
    "dis2_weighted": {
        "fn": build_dis2_graph,
        "kwargs": {"rmax": 0.1, "weight_mode": "paper_1_10"},
    },
    "natural_hierarchy": {
        "fn": build_natural_hierarchy_graph,
        "kwargs": {},
    },
}


ALGORITHMS = {
    "pure_greedy": {
        "kind": "greedy",},
    "original_hmetis": {
        "kind": "hmetis",
        "nparts": "scenario",},
    "hmetis_refine": {
        "kind": "refine",
        "nparts": "scenario",
        "top_per_partition": 10,
        "min_gain_ratio": 0.0,
        "refine_top_per_partition": 10,
        "refine_rounds": 5,
        "max_swaps_per_round": None,
        "use_greedy_seed": False,
    },
    "pure_tabu": {
        "kind": "tabu",
        "max_iter": 240,
        # Paper parameters: L = 50 and NT = 50 for MCP.
        "tabu_tenure": 50,
        # None means full one-swap neighborhood, matching the paper more
        # closely. Use a number if larger instances become too slow.
        "candidate_pool_size": None,
        "random_candidate_size": 0,
        "no_improve_limit": 50,
    },
}


ALGO_ORDER = {
    "pure_greedy": 1,
    "pure_tabu": 2,
    "original_hmetis": 3,
    "hmetis_refine": 4,
}


def scenario_nparts(scenario):
    if "nparts" in scenario and scenario["nparts"] is not None:
        return scenario["nparts"]
    if scenario["nhedges"] <= 200:
        return 4
    if scenario["nhedges"] <= 600:
        return 8
    return 16


def resolved_algo_config(config, scenario):
    resolved = dict(config)
    if resolved.get("nparts") == "scenario":
        resolved["nparts"] = scenario_nparts(scenario)
    return resolved


def build_graph(graph_type, nhedges, nvtxs, seed):
    builder = GRAPH_BUILDERS[graph_type]
    hg, meta = builder["fn"](
        nhedges=nhedges,
        nvtxs=nvtxs,
        seed=seed,
        **builder["kwargs"],
    )
    meta["graph_type"] = graph_type
    return hg, meta


def covered_weight(hg, covered_vertices):
    return sum(hg.vtx_weights[v] for v in covered_vertices)


def run_algo(hg, budget, algo_name, config, seed):
    start = time.time()
    filename = CONFIG["hmetis_file"]

    with contextlib.redirect_stdout(io.StringIO()):
        kind = config["kind"]
        cfg = {k: v for k, v in config.items() if k != "kind"}

        if kind == "greedy":
            result = pure_greedy_mcp(hg, budget=budget, filename=filename)
        elif kind == "hmetis":
            result = hmetis_mcp(
                hg,
                budget=budget,
                filename=filename,
                timeout=45,
                **cfg,
            )
        elif kind == "refine":
            result = hmetis_mcp_refine(
                hg,
                budget=budget,
                filename=filename,
                timeout=45,
                verbose=False,
                **cfg,
            )
        elif kind == "oblswap":
            result = pure_oblswap_mcp(
                hg,
                budget=budget,
                filename=filename,
                verbose=False,
                **cfg,
            )
        elif kind == "tabu":
            result = pure_tabu_mcp(
                hg,
                budget=budget,
                filename=filename,
                seed=seed,
                verbose=False,
                **cfg,
            )
        else:
            raise ValueError(f"unknown algorithm kind: {kind}")

    overlap = _compute_overlap(hg, result[2])
    return {
        "coverage": len(result[1]),
        "weighted_coverage": round(covered_weight(hg, result[1]), 6),
        "coverage_fraction": round(len(result[1]) / hg.nvtxs, 5),
        "solution_size": len(result[2]),
        "overlap_ratio": overlap["overlap_ratio"],
        "multi_covered_vertices": overlap["multi_covered_vertices"],
        "time_s": round(time.time() - start, 4),
    }


def iter_scenarios():
    for scenario_idx, size_cfg in enumerate(SCENARIOS, start=1):
        nvtxs = size_cfg["nvtxs"]
        f_ratio = size_cfg["f_ratio"]
        budget_ratio = size_cfg["budget_ratio"]
        nhedges = max(1, int(f_ratio * nvtxs))
        budget = max(1, int(budget_ratio * nhedges))
        for graph_type in size_cfg.get("graph_types", CONFIG["graph_types"]):
            for run in range(1, size_cfg.get("runs", 1) + 1):
                seed = 200_000 + 10_000 * run + 1_000 * scenario_idx + nvtxs
                scenario = {
                    "scenario_name": size_cfg["name"],
                    "graph_type": graph_type,
                    "nvtxs": nvtxs,
                    "nhedges": nhedges,
                    "f_ratio": f_ratio,
                    "budget_ratio": budget_ratio,
                    "budget": budget,
                    "run": run,
                    "seed": seed,
                    "solve_lp": size_cfg.get("solve_lp", False),
                    "skip_algorithms": size_cfg.get("skip_algorithms", []),
                }
                scenario["nparts"] = scenario_nparts({**scenario, **size_cfg})
                yield scenario


def maybe_solve_lp(hg, scenario):
    if not CONFIG["solve_lp_when_enabled"] or not scenario.get("solve_lp"):
        return {"optimal_value": None, "optimal_status": None, "lp_time_s": None}
    if hg.nvtxs > CONFIG["lp_max_nvtxs"] or hg.nhedges > CONFIG["lp_max_nhedges"]:
        return {"optimal_value": None, "optimal_status": "skipped_size", "lp_time_s": None}

    start = time.time()
    result = optimal_solu_detail(
        hg,
        budget=scenario["budget"],
        time_limit=CONFIG["lp_time_limit_s"],
        msg=False,
    )
    return {
        "optimal_value": result["objective"],
        "optimal_status": result["status"],
        "lp_time_s": round(time.time() - start, 4),
    }


def reached_optimal(coverage, optimal_value):
    if optimal_value is None:
        return None
    return int(abs(coverage - optimal_value) <= 1e-6)


def main():
    rows = []

    for scenario in iter_scenarios():
        random.seed(scenario["seed"])
        np.random.seed(scenario["seed"])
        hg, graph_meta = build_graph(
            graph_type=scenario["graph_type"],
            nhedges=scenario["nhedges"],
            nvtxs=scenario["nvtxs"],
            seed=scenario["seed"],
        )
        lp_meta = maybe_solve_lp(hg, scenario)

        run_metrics = {}
        for algo_name, algo_config in ALGORITHMS.items():
            if algo_name in scenario.get("skip_algorithms", []):
                print(
                    f"[skip] graph={scenario['graph_type']} "
                    f"scenario={scenario['scenario_name']} "
                    f"run={scenario['run']} {algo_name}",
                    flush=True,
                )
                continue
            resolved_config = resolved_algo_config(algo_config, scenario)
            metrics = run_algo(
                hg=hg,
                budget=scenario["budget"],
                algo_name=algo_name,
                config=resolved_config,
                seed=scenario["seed"],
            )
            run_metrics[algo_name] = metrics
            print(
                f"[done] graph={scenario['graph_type']} "
                f"scenario={scenario['scenario_name']} "
                f"|U|={scenario['nvtxs']} F={scenario['nhedges']} "
                f"budget={scenario['budget']} nparts={scenario['nparts']} "
                f"run={scenario['run']} "
                f"{algo_name} coverage={metrics['coverage']} "
                f"weighted={metrics['weighted_coverage']} "
                f"time={metrics['time_s']}s",
                flush=True,
            )

        greedy = run_metrics["pure_greedy"]
        tabu = run_metrics.get("pure_tabu")
        tabu_gain = None
        if tabu is not None:
            tabu_gain = tabu["weighted_coverage"] - greedy["weighted_coverage"]

        for algo_name, metrics in run_metrics.items():
            hmetis_gain_capture = None
            if algo_name == "hmetis_refine" and tabu_gain and tabu_gain > 0:
                hmetis_gain_capture = round(
                    (metrics["weighted_coverage"] - greedy["weighted_coverage"]) / tabu_gain,
                    5,
                )

            rows.append({
                **scenario,
                **graph_meta,
                **lp_meta,
                "algo_name": algo_name,
                "algo_order": ALGO_ORDER[algo_name],
                "coverage_delta_greedy": metrics["coverage"] - greedy["coverage"],
                "weighted_delta_greedy": round(
                    metrics["weighted_coverage"] - greedy["weighted_coverage"],
                    6,
                ),
                "overlap_delta_greedy": round(
                    metrics["overlap_ratio"] - greedy["overlap_ratio"], 5
                ),
                "coverage_delta_tabu": (
                    metrics["coverage"] - tabu["coverage"] if tabu is not None else None
                ),
                "weighted_delta_tabu": (
                    round(metrics["weighted_coverage"] - tabu["weighted_coverage"], 6)
                    if tabu is not None else None
                ),
                "hmetis_gain_capture_of_tabu": hmetis_gain_capture,
                # Paper-style performance: solution quality divided by optimal
                # solution quality, when LP was solved for this instance.
                "performance": (
                    round(metrics["weighted_coverage"] / lp_meta["optimal_value"], 5)
                    if lp_meta["optimal_value"] else None
                ),
                "reached_optimal": reached_optimal(
                    metrics["weighted_coverage"],
                    lp_meta["optimal_value"],
                ),
                **metrics,
            })

    df = pd.DataFrame(rows)
    df.to_csv(CONFIG["results_file"], index=False)

    summary = (
        df
        .groupby(
            [
                "scenario_name",
                "graph_type",
                "nvtxs",
                "nhedges",
                "budget",
                "nparts",
                "f_ratio",
                "budget_ratio",
                "graph_family",
                "weight_mode",
                "algo_order",
                "algo_name",
            ],
            as_index=False,
        )
        .agg(
            runs=("run", "count"),
            avg_coverage=("coverage", "mean"),
            avg_weighted_coverage=("weighted_coverage", "mean"),
            avg_delta_greedy=("coverage_delta_greedy", "mean"),
            avg_weighted_delta_greedy=("weighted_delta_greedy", "mean"),
            wins_vs_greedy=("coverage_delta_greedy", lambda s: int((s > 0).sum())),
            ties_vs_greedy=("coverage_delta_greedy", lambda s: int((s == 0).sum())),
            losses_vs_greedy=("coverage_delta_greedy", lambda s: int((s < 0).sum())),
            weighted_wins_vs_greedy=("weighted_delta_greedy", lambda s: int((s > 0).sum())),
            weighted_ties_vs_greedy=("weighted_delta_greedy", lambda s: int((s == 0).sum())),
            weighted_losses_vs_greedy=("weighted_delta_greedy", lambda s: int((s < 0).sum())),
            avg_overlap_delta_greedy=("overlap_delta_greedy", "mean"),
            avg_time_s=("time_s", "mean"),
            avg_hmetis_gain_capture=("hmetis_gain_capture_of_tabu", "mean"),
            avg_performance=("performance", "mean"),
            std_performance=("performance", "std"),
            fraction_optimal=("reached_optimal", "mean"),
            optimal_runs=("reached_optimal", "count"),
            lp_status=("optimal_status", lambda s: ",".join(sorted({str(v) for v in s.dropna()}))),
            avg_lp_time_s=("lp_time_s", "mean"),
        )
        .sort_values(
            [
                "graph_type",
                "nvtxs",
                "nhedges",
                "budget",
                "nparts",
                "algo_order",
            ],
            ascending=[True, True, True, True, True, True],
        )
    )
    summary = summary.drop(columns=["algo_order"])

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(CONFIG["summary_file"], index=False)
    print(f"\nSaved detailed rows to {CONFIG['results_file']}")
    print(f"Saved averaged summary to {CONFIG['summary_file']}")


#if __name__ == "__main__":
#    main()
