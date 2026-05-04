import subprocess, time
from greedy import greedy
from subgraph import write_hgr


def _uncovered_weight(hg, edge, covered_vertices):
    """Weighted gain of selecting an edge given current coverage."""
    return sum(hg.vtx_weights[v] for v in hg.hedges_dict[edge] - covered_vertices)


def _best_in_partition(hg, part_edges, covered_vertices, removed_edges):
    """Return the edge with highest uncovered weight from a partition."""
    return max(
        (e for e in part_edges if e not in removed_edges),
        key=lambda e: _uncovered_weight(hg, e, covered_vertices),
        default=None,
    )


def _parse_partitions(line, e_map_inv):
    """Map hMETIS output lines to {partition_id: {original_edge, ...}}."""
    partitions = {}
    for edge, part in enumerate(line, start=1):
        original_edge = e_map_inv[edge]
        partitions.setdefault(int(part), set()).add(original_edge)
    return partitions


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
            best_edge = _best_in_partition(hg, part_edges, covered_vertices, removed_edges)
            if best_edge is not None:
                newly_covered = hg.hedges_dict[best_edge] - covered_vertices
                if newly_covered:
                    covered_vertices.update(newly_covered)
                    removed_edges.add(best_edge)

        if len(removed_edges) == prev_len:
            break

        iteration += 1
        print(f"iter={iteration}, covered={len(covered_vertices)}, removed={len(removed_edges)}")

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time


def hmetis_set_cover(hg, filename, nparts=2, **kwargs):
    removed_edges = set()
    covered_vertices = set()
    iteration = 0
    write_time = 0
    partition_time = 0

    while len(covered_vertices) < hg.nvtxs:
        w = time.time()
        e_map_inv = write_hgr(hg, covered_vertices, removed_edges, filename)
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
                        break
            break

        partitions = _parse_partitions(line, e_map_inv)

        prev_len = len(removed_edges)
        for part_edges in partitions.values():
            if len(covered_vertices) >= hg.nvtxs:
                break
            best_edge = _best_in_partition(hg, part_edges, covered_vertices, removed_edges)
            if best_edge is not None:
                newly_covered = hg.hedges_dict[best_edge] - covered_vertices
                if newly_covered:
                    covered_vertices.update(newly_covered)
                    removed_edges.add(best_edge)

        if len(removed_edges) == prev_len:
            break

        iteration += 1
        print(f"iter={iteration}, "
              f"covered={len(covered_vertices)}, "
              f"removed={len(removed_edges)}"
              )


    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time








