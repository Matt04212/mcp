import subprocess, time
import numpy as np
from greedy import greedy
from subgraph2 import HgrWriter, write_hgr


def _build_scores(hg, covered_vertices, removed_edges):
    scores = np.zeros(hg.nhedges + 1, dtype=np.float64)
    for e in hg.hedges:
        if e in removed_edges:
            scores[e] = -1.0
        else:
            scores[e] = sum(hg.vtx_weights[v]
                            for v in hg.hedges_dict[e] - covered_vertices)
    return scores


def _update_scores(hg, scores, newly_covered, removed_edges):
    affected_edges = set()
    for v in newly_covered:
        affected_edges.update(hg.vtxs_dict[v])
    for e in affected_edges:
        if e in removed_edges:
            scores[e] = -1.0
        else:
            scores[e] -= sum(
                hg.vtx_weights[v] for v in newly_covered if e in hg.vtxs_dict[v]
            )


def _best_in_partition(part_edges, removed_edges, scores):
    candidates = np.array([e for e in part_edges if e not in removed_edges],
                          dtype=np.int32)
    if len(candidates) == 0:
        return None
    best_idx = np.argmax(scores[candidates])
    best_edge = candidates[best_idx]
    return int(best_edge) if scores[best_edge] > 0 else None


def _parse_partitions(line, e_map_inv):
    partitions = {}
    for edge, part in enumerate(line, start=1):
        original_edge = e_map_inv[edge]
        partitions.setdefault(int(part), set()).add(original_edge)
    return partitions


def hmetis_mcp2(hg, budget, filename, nparts=2, **kwargs):
    removed_edges = set()
    covered_vertices = set()
    iteration = 0
    write_time = 0
    partition_time = 0

    scores = _build_scores(hg, covered_vertices, removed_edges)
    writer = HgrWriter(hg)                          # built once, updated incrementally

    while len(removed_edges) < budget:
        w = time.time()
        e_map_inv = writer.write(filename)          # single bulk write
        write_time += time.time() - w

        p = time.time()
        subprocess.run(f"./hmetis {filename} {nparts} 5 1 1 1 1 0 0", shell=True)
        partition_time += time.time() - p

        with open(f"{filename}.part.{nparts}") as f:
            line = f.read().splitlines()

        if len(line) != len(e_map_inv):
            break

        partitions = _parse_partitions(line, e_map_inv)

        prev_len = len(removed_edges)
        for part_edges in partitions.values():
            if len(removed_edges) >= budget:
                break
            best_edge = _best_in_partition(part_edges, removed_edges, scores)
            if best_edge is not None:
                newly_covered = hg.hedges_dict[best_edge] - covered_vertices
                if newly_covered:
                    covered_vertices.update(newly_covered)
                    removed_edges.add(best_edge)
                    scores[best_edge] = -1.0
                    _update_scores(hg, scores, newly_covered, removed_edges)
                    writer.update(best_edge, newly_covered)   # incremental patch

        if len(removed_edges) == prev_len:
            break

        iteration += 1
        print(f"iter={iteration}, covered={len(covered_vertices)}, removed={len(removed_edges)}")

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time


def hmetis_set_cover2(hg, filename, nparts=2, **kwargs):
    removed_edges = set()
    covered_vertices = set()
    iteration = 0
    write_time = 0
    partition_time = 0

    scores = _build_scores(hg, covered_vertices, removed_edges)
    writer = HgrWriter(hg)                          # built once, updated incrementally

    while len(covered_vertices) < hg.nvtxs:
        w = time.time()
        e_map_inv = writer.write(filename)          # single bulk write
        write_time += time.time() - w

        p = time.time()
        subprocess.run(f"./hmetis {filename} {nparts} 5 1 2 1 0 0 0", shell=True)
        partition_time += time.time() - p

        with open(f"{filename}.part.{nparts}") as f:
            line = f.read().splitlines()

        if len(line) != len(e_map_inv):
            # hMETIS couldn't partition — greedily cover remaining vertices
            for v in hg.vtxs:
                if v in covered_vertices:
                    continue
                for hedge in hg.vtxs_dict[v]:
                    if hedge not in removed_edges:
                        newly_covered = hg.hedges_dict[hedge] - covered_vertices
                        covered_vertices.update(newly_covered)
                        removed_edges.add(hedge)
                        scores[hedge] = -1.0
                        _update_scores(hg, scores, newly_covered, removed_edges)
                        writer.update(hedge, newly_covered)
                        break
            break

        parse = time.time()
        partitions = _parse_partitions(line, e_map_inv)
        parse_time = time.time() - parse

        select = time.time()
        prev_len = len(removed_edges)
        for part_edges in partitions.values():
            if len(covered_vertices) >= hg.nvtxs:
                break
            best_edge = _best_in_partition(part_edges, removed_edges, scores)
            if best_edge is not None:
                newly_covered = hg.hedges_dict[best_edge] - covered_vertices
                if newly_covered:
                    covered_vertices.update(newly_covered)
                    removed_edges.add(best_edge)
                    scores[best_edge] = -1.0
                    _update_scores(hg, scores, newly_covered, removed_edges)
                    writer.update(best_edge, newly_covered)   # incremental patch
        select_time = time.time() - select

        if len(removed_edges) == prev_len:
            break

        iteration += 1
        print(f"iter={iteration}, "
              f"covered={len(covered_vertices)}, "
              f"removed={len(removed_edges)},"
              f"parse_time={parse_time}, select_time={select_time}"al)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time







