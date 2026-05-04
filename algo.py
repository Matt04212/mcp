import subprocess, time, heapq
from greedy import greedy
import numpy as np
from subgraph import write_hgr
from selection import selection


def hmetis_mcp(hg, budget, filename, nparts=2, **kwargs):
    removed_edges = set()
    covered_vertices = set()
    iteration = 0
    write_time = 0
    partition_time = 0

    while len(removed_edges) < budget:
        w = time.time()
        e_map_inv = write_hgr(hg, covered_vertices, removed_edges, filename)
        write_time += time.time() - w

        p = time.time()
        subprocess.run(f"./hmetis {filename} {nparts} 5 1 1 1 1 0 0",
                      shell=True)
        partition_time += time.time() - p

        with open(f"{filename}.part.{nparts}", "r") as f:
            line = f.read().splitlines()

        if len(line) != len(e_map_inv):
            break

        edge_partitions = {}
        for edge, part in enumerate(line, start=1):
            original_edge = e_map_inv[edge]
            part = int(part)
            edge_partitions.setdefault(part, set()).add(original_edge)

        prev_len = len(removed_edges)
        for n, part_edges in edge_partitions.items():
            if len(removed_edges) >= budget:
                break
            best_edge = max(
                (e for e in part_edges if e not in removed_edges),
                key=lambda e: sum(hg.vtx_weights[v]
                                  for v in hg.hedges_dict[e] - covered_vertices),
                default=None
            )
            if best_edge is not None:
                newly_covered = hg.hedges_dict[best_edge] - covered_vertices
                if newly_covered:
                    covered_vertices.update(newly_covered)
                    removed_edges.add(best_edge)

        if len(removed_edges) == prev_len:
            break

        iteration += 1
        print(f"iter={iteration}, covered={len(covered_vertices)}, removed={len(removed_edges)}")

    weighted_coverage = sum(hg.vtx_weights[v] for v in covered_vertices)
    return (hg.nhedges, hg.nvtxs), weighted_coverage, list(removed_edges), write_time, partition_time


def hmetis_set_cover(hg, filename, nparts=2, **kwargs):
    removed_edges = set()
    covered_vertices = set()
    iteration = 0
    write_time = 0
    partition_time = 0

    while len(covered_vertices) < hg.nvtxs:

        live_count = hg.nvtxs - len(covered_vertices)
        """if live_count < 3000:
            greedy(hg, covered_vertices, removed_edges)

            break"""
        w = time.time()
        e_map_inv = write_hgr(hg, covered_vertices, removed_edges, filename)
        write_time += time.time() - w

        p = time.time()
        subprocess.run(f"./hmetis {filename} {nparts} 5 1 2 1 0 0 0",
                      shell=True)
        partition_time += time.time() - p

        with open(f"{filename}.part.{nparts}", "r") as f:
            line = f.read().splitlines()

        if len(line) != len(e_map_inv):
            # handle remaining uncovered vertices
            for v in hg.vtxs:
                if v not in covered_vertices:
                    for hedge in hg.vtxs_dict[v]:
                        if hedge not in removed_edges:
                            newly_covered = hg.hedges_dict[hedge] - covered_vertices
                            covered_vertices.update(newly_covered)
                            removed_edges.add(hedge)
                            break
            break

        edge_partitions = {}
        for edge, part in enumerate(line, start=1):
            original_edge = e_map_inv[edge]
            part = int(part)
            edge_partitions.setdefault(part, set()).add(original_edge)

        prev_len = len(removed_edges)
        for n, part_edges in edge_partitions.items():
            if len(covered_vertices) >= hg.nvtxs:
                break
            best_edge = max(
                (e for e in part_edges if e not in removed_edges),
                key=lambda e: sum(hg.vtx_weights[v]
                                  for v in hg.hedges_dict[e] - covered_vertices),
                default=None
            )
            if best_edge is not None:
                newly_covered = hg.hedges_dict[best_edge] - covered_vertices
                if newly_covered:
                    covered_vertices.update(newly_covered)
                    removed_edges.add(best_edge)  # bug here — should be best_edge

        if len(removed_edges) == prev_len:
            break

        iteration += 1
        print(f"iter={iteration}, covered={len(covered_vertices)}, removed={len(removed_edges)}")

    weighted_coverage = sum(hg.vtx_weights[v] for v in covered_vertices)
    return (hg.nhedges, hg.nvtxs), weighted_coverage, list(removed_edges), write_time, partition_time








