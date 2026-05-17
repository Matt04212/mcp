import argparse
import contextlib
import io
import random
import time

import numpy as np
import pandas as pd

from algo import hmetis_partitioned_greedy_mcp, pure_greedy_mcp
from hypergraph import Hypergraph


PRESETS = {
    "smoke": {
        "nvtxs": [600],
        "f_ratios": [0.5],
        "budget_ratio": 0.1,
        "runs": 2,
        "nparts": [4, 8],
        "max_edge_ratio": 0.16,
    },
    "large": {
        "nvtxs": [5000, 10000],
        "f_ratios": [0.5],
        "budget_ratio": 0.1,
        "runs": 5,
        "nparts": [4, 8, 16, 32],
        "max_edge_ratio": 0.12,
    },
}


DEFAULT_CONFIG = {
    "nvtxs_list": [5000],
    "f_ratios": [0.5],
    "budget_ratio": 0.02,
    "runs": 2,
    "nparts_list": [4, 8, 16],
    "distributions": ["beta_right", "beta_left", "beta_bell", "uniform"],
    "max_edge_ratio": 0.12,
    "min_edge_size": 1,
    "budget_mode": "equal",
    "timeout": 60,
    "seed_base": 700_000,
    "output_prefix": "parallel_greedy_beta",
    "hmetis_file": "parallel_greedy_beta.hgr",
}


SUPPORTED_DISTRIBUTIONS = {"beta_right", "beta_left", "beta_bell", "uniform"}


def parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def covered_weight(hg, covered_vertices):
    return sum(hg.vtx_weights[vtx] for vtx in covered_vertices)


def edge_size_meta(hg):
    sizes = [len(vertices) for vertices in hg.hedges_dict.values()]
    return {
        "graph_family": "hypergraph_generate",
        "min_edge_size": min(sizes),
        "mean_edge_size": round(float(np.mean(sizes)), 2),
        "median_edge_size": round(float(np.median(sizes)), 2),
        "std_edge_size": round(float(np.std(sizes)), 2),
        "max_edge_size": max(sizes),
    }


def build_distribution_graph(nhedges, nvtxs, seed, distribution, min_edge_size, max_edge_ratio):
    random.seed(seed)
    np.random.seed(seed)

    hg = Hypergraph(nhedges, nvtxs)
    max_size = max(min_edge_size, min(nvtxs, int(round(max_edge_ratio * nvtxs))))

    kwargs = {}
    if distribution == "uniform":
        kwargs["low"] = min_edge_size
        kwargs["high"] = max_size
    else:
        kwargs["max_size"] = max_size

    hg.generate(distribution=distribution, **kwargs)
    meta = {
        "graph_type": distribution,
        "distribution": distribution,
        "generator_max_size": max_size,
        **edge_size_meta(hg),
    }
    return hg, meta


def compute_overlap(hg, selected_edges):
    selected_edges = list(selected_edges)
    if not selected_edges:
        return {"overlap_ratio": None, "multi_covered_vertices": None}

    all_covered = set()
    total_raw = 0
    cover_counts = {}
    for edge in selected_edges:
        vertices = hg.hedges_dict[edge]
        all_covered.update(vertices)
        total_raw += len(vertices)
        for vertex in vertices:
            cover_counts[vertex] = cover_counts.get(vertex, 0) + 1

    overlap_count = total_raw - len(all_covered)
    return {
        "overlap_ratio": round(overlap_count / total_raw, 4) if total_raw else 0,
        "multi_covered_vertices": sum(1 for count in cover_counts.values() if count > 1),
    }


def run_solution(hg, algo_func, budget, filename, **kwargs):
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        result = algo_func(hg, budget=budget, filename=filename, **kwargs)
    elapsed = time.time() - start

    covered_vertices = result[1]
    selected_edges = result[2]
    overlap = compute_overlap(hg, selected_edges)
    stage = result[5] or {}
    return {
        "coverage": len(covered_vertices),
        "weighted_coverage": round(covered_weight(hg, covered_vertices), 6),
        "coverage_fraction": round(len(covered_vertices) / hg.nvtxs, 6),
        "solution_size": len(selected_edges),
        "time_s": round(elapsed, 4),
        "write_time_s": round(result[3], 4) if result[3] is not None else None,
        "partition_time_s": round(result[4], 4) if result[4] is not None else None,
        "overlap_ratio": overlap["overlap_ratio"],
        "multi_covered_vertices": overlap["multi_covered_vertices"],
        **stage,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare whole-graph greedy against independent hMETIS-partition "
            "greedy on beta edge-size hypergraphs."
        )
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="large")
    parser.add_argument("--nvtxs", type=parse_int_list)
    parser.add_argument("--f-ratios", type=parse_float_list)
    parser.add_argument("--budget-ratio", type=float)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--nparts", type=parse_int_list)
    parser.add_argument("--max-edge-ratio", type=float)
    parser.add_argument("--min-edge-size", type=int, default=1)
    parser.add_argument("--budget-mode", choices=["equal", "proportional"], default="equal")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed-base", type=int, default=700_000)
    parser.add_argument(
        "--distributions",
        type=parse_str_list,
        default=parse_str_list("beta_right,beta_left,beta_bell,uniform"),
    )
    parser.add_argument("--output-prefix", default="parallel_greedy_beta")
    parser.add_argument("--hmetis-file", default="parallel_greedy_beta.hgr")
    return parser


def resolve_config(args):
    preset = PRESETS[args.preset]
    config = {
        "nvtxs": args.nvtxs or preset["nvtxs"],
        "f_ratios": args.f_ratios or preset["f_ratios"],
        "budget_ratio": args.budget_ratio
        if args.budget_ratio is not None
        else preset["budget_ratio"],
        "runs": args.runs if args.runs is not None else preset["runs"],
        "nparts": args.nparts or preset["nparts"],
        "max_edge_ratio": args.max_edge_ratio
        if args.max_edge_ratio is not None
        else preset["max_edge_ratio"],
    }

    unknown = sorted(set(args.distributions) - SUPPORTED_DISTRIBUTIONS)
    if unknown:
        known = ", ".join(sorted(SUPPORTED_DISTRIBUTIONS))
        raise ValueError(f"unknown distributions {unknown}; expected one of {known}")
    config["distributions"] = args.distributions
    return config


def summarize(df):
    group_cols = [
        "distribution",
        "nvtxs",
        "nhedges",
        "budget",
        "budget_ratio",
        "nparts",
        "budget_mode",
        "max_edge_ratio",
    ]
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            runs=("run", "count"),
            avg_global_coverage=("global_coverage", "mean"),
            avg_partitioned_coverage=("partitioned_coverage", "mean"),
            avg_quality_ratio=("quality_ratio", "mean"),
            std_quality_ratio=("quality_ratio", "std"),
            avg_quality_loss=("quality_loss", "mean"),
            avg_quality_loss_pct=("quality_loss_pct", "mean"),
            partitioned_wins=("quality_delta", lambda s: int((s > 0).sum())),
            ties=("quality_delta", lambda s: int((s == 0).sum())),
            partitioned_losses=("quality_delta", lambda s: int((s < 0).sum())),
            avg_global_time_s=("global_time_s", "mean"),
            avg_partitioned_time_s=("partitioned_time_s", "mean"),
            avg_partition_time_s=("hmetis_partition_time_s", "mean"),
            avg_global_overlap=("global_overlap_ratio", "mean"),
            avg_partitioned_overlap=("partitioned_overlap_ratio", "mean"),
            avg_mean_edge_size=("mean_edge_size", "mean"),
            avg_median_edge_size=("median_edge_size", "mean"),
            avg_max_edge_size=("max_edge_size", "mean"),
            avg_mean_partition_edges=("mean_partition_edges", "mean"),
        )
        .sort_values(["nvtxs", "distribution", "nparts"])
    )
    return summary


def _validate_config(config):
    unknown = sorted(set(config["distributions"]) - SUPPORTED_DISTRIBUTIONS)
    if unknown:
        known = ", ".join(sorted(SUPPORTED_DISTRIBUTIONS))
        raise ValueError(f"unknown distributions {unknown}; expected one of {known}")


def run_experiment(config):
    _validate_config(config)
    rows = []

    for nvtxs in config["nvtxs_list"]:
        for f_ratio in config["f_ratios"]:
            nhedges = max(1, int(round(f_ratio * nvtxs)))
            budget = max(1, int(round(config["budget_ratio"] * nhedges)))

            for dist_idx, distribution in enumerate(config["distributions"]):
                for run in range(1, config["runs"] + 1):
                    seed = (
                        config["seed_base"]
                        + 1_000_000 * dist_idx
                        + 100_000 * run
                        + 10 * nvtxs
                        + int(1000 * f_ratio)
                    )
                    random.seed(seed)
                    np.random.seed(seed)
                    hg, graph_meta = build_distribution_graph(
                        nhedges=nhedges,
                        nvtxs=nvtxs,
                        seed=seed,
                        distribution=distribution,
                        min_edge_size=config["min_edge_size"],
                        max_edge_ratio=config["max_edge_ratio"],
                    )

                    global_metrics = run_solution(
                        hg,
                        pure_greedy_mcp,
                        budget=budget,
                        filename=config["hmetis_file"],
                    )

                    for nparts in config["nparts_list"]:
                        row_base = {
                            "distribution": distribution,
                            "run": run,
                            "seed": seed,
                            "nvtxs": nvtxs,
                            "nhedges": nhedges,
                            "f_ratio": f_ratio,
                            "budget": budget,
                            "budget_ratio": config["budget_ratio"],
                            "nparts": nparts,
                            "budget_mode": config["budget_mode"],
                            "min_edge_size_config": config["min_edge_size"],
                            "max_edge_ratio": config["max_edge_ratio"],
                            **graph_meta,
                        }

                        try:
                            partitioned_metrics = run_solution(
                                hg,
                                hmetis_partitioned_greedy_mcp,
                                budget=budget,
                                filename=config["hmetis_file"],
                                nparts=nparts,
                                timeout=config["timeout"],
                                budget_mode=config["budget_mode"],
                            )
                            global_quality = global_metrics["weighted_coverage"]
                            partitioned_quality = partitioned_metrics["weighted_coverage"]
                            quality_delta = round(partitioned_quality - global_quality, 6)
                            quality_loss = round(global_quality - partitioned_quality, 6)
                            quality_ratio = (
                                round(partitioned_quality / global_quality, 6)
                                if global_quality > 0
                                else None
                            )
                            quality_loss_pct = (
                                round(100.0 * quality_loss / global_quality, 4)
                                if global_quality > 0
                                else None
                            )
                            status = "ok"
                        except Exception as exc:
                            partitioned_metrics = {}
                            quality_delta = None
                            quality_loss = None
                            quality_ratio = None
                            quality_loss_pct = None
                            status = f"failed: {exc}"

                        rows.append(
                            {
                                **row_base,
                                "status": status,
                                "global_coverage": global_metrics["coverage"],
                                "global_weighted_coverage": global_metrics["weighted_coverage"],
                                "global_coverage_fraction": global_metrics["coverage_fraction"],
                                "global_solution_size": global_metrics["solution_size"],
                                "global_time_s": global_metrics["time_s"],
                                "global_overlap_ratio": global_metrics["overlap_ratio"],
                                "partitioned_coverage": partitioned_metrics.get("coverage"),
                                "partitioned_weighted_coverage": partitioned_metrics.get("weighted_coverage"),
                                "partitioned_coverage_fraction": partitioned_metrics.get("coverage_fraction"),
                                "partitioned_solution_size": partitioned_metrics.get("solution_size"),
                                "partitioned_time_s": partitioned_metrics.get("time_s"),
                                "partitioned_overlap_ratio": partitioned_metrics.get("overlap_ratio"),
                                "hmetis_write_time_s": partitioned_metrics.get("write_time_s"),
                                "hmetis_partition_time_s": partitioned_metrics.get("partition_time_s"),
                                "partition_count": partitioned_metrics.get("partition_count"),
                                "min_partition_edges": partitioned_metrics.get("min_partition_edges"),
                                "mean_partition_edges": partitioned_metrics.get("mean_partition_edges"),
                                "max_partition_edges": partitioned_metrics.get("max_partition_edges"),
                                "min_partition_budget": partitioned_metrics.get("min_partition_budget"),
                                "max_partition_budget": partitioned_metrics.get("max_partition_budget"),
                                "quality_delta": quality_delta,
                                "quality_loss": quality_loss,
                                "quality_ratio": quality_ratio,
                                "quality_loss_pct": quality_loss_pct,
                            }
                        )

                        print(
                            "[done] "
                            f"dist={distribution} |U|={nvtxs} F={nhedges} "
                            f"B={budget} run={run}/{config['runs']} nparts={nparts} "
                            f"global={global_metrics['weighted_coverage']} "
                            f"partitioned={partitioned_metrics.get('weighted_coverage')} "
                            f"ratio={quality_ratio} loss_pct={quality_loss_pct} "
                            f"status={status}",
                            flush=True,
                        )

    df = pd.DataFrame(rows)
    details_file = f"{config['output_prefix']}_results.csv"
    summary_file = f"{config['output_prefix']}_summary.csv"
    df.to_csv(details_file, index=False)

    ok_df = df[df["status"] == "ok"].copy()
    summary = summarize(ok_df) if not ok_df.empty else pd.DataFrame()
    if not summary.empty:
        summary.to_csv(summary_file, index=False)

    return {
        "rows": df,
        "summary": summary,
        "details_file": details_file,
        "summary_file": summary_file,
    }


def main():
    args = build_parser().parse_args()
    config = resolve_config(args)
    runner_config = {
        "nvtxs_list": config["nvtxs"],
        "f_ratios": config["f_ratios"],
        "budget_ratio": config["budget_ratio"],
        "runs": config["runs"],
        "nparts_list": config["nparts"],
        "distributions": config["distributions"],
        "max_edge_ratio": config["max_edge_ratio"],
        "min_edge_size": args.min_edge_size,
        "budget_mode": args.budget_mode,
        "timeout": args.timeout,
        "seed_base": args.seed_base,
        "output_prefix": args.output_prefix,
        "hmetis_file": args.hmetis_file,
    }
    result = run_experiment(runner_config)

    if result["summary"].empty:
        print("\nNo successful partitioned-greedy runs.")
        print(f"Saved detailed rows to {result['details_file']}")
        return

    print("\n=== Summary ===")
    print(
        result["summary"][
            [
                "distribution",
                "nvtxs",
                "nhedges",
                "budget",
                "nparts",
                "runs",
                "avg_quality_ratio",
                "avg_quality_loss_pct",
                "partitioned_wins",
                "ties",
                "partitioned_losses",
                "avg_global_time_s",
                "avg_partitioned_time_s",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved detailed rows to {result['details_file']}")
    print(f"Saved averaged summary to {result['summary_file']}")


if __name__ == "__main__":
    main()
