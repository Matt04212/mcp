
import heapq
import importlib.util
import os
import random
import subprocess, time
from collections import deque
from pathlib import Path
import numpy as np
from subgraph2 import HgrWriter, write_hgr


HMETIS_UBFACTOR = "5"
HMETIS_NRUNS = "1"
HMETIS_CTYPE = "5"
HMETIS_RTYPE = "3"
HMETIS_VCYCLE = "0"
HMETIS_RECONST = "0"
HMETIS_DBGLVL = "0"
DEFAULT_HMETIS_PARAMS = {
    "ubfactor": HMETIS_UBFACTOR,
    "nruns": HMETIS_NRUNS,
    "ctype": HMETIS_CTYPE,
    "rtype": HMETIS_RTYPE,
    "vcycle": HMETIS_VCYCLE,
    "reconst": HMETIS_RECONST,
    "dbglvl": HMETIS_DBGLVL,
}

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


def _selection_value(hg, edge, covered_vertices, scores, overlap_penalty):
    return scores[edge] - overlap_penalty * len(hg.hedges_dict[edge] & covered_vertices)


def _global_best_edge(hg, removed_edges, scores):
    best_edge = None
    best_score = 0.0
    for edge in hg.hedges:
        if edge in removed_edges:
            continue
        if scores[edge] > best_score:
            best_edge = edge
            best_score = scores[edge]
    return best_edge, best_score


def _apply_selected_edge(hg, edge, covered_vertices, removed_edges, scores, writer):
    newly_covered = hg.hedges_dict[edge] - covered_vertices
    if not newly_covered:
        return False

    covered_vertices.update(newly_covered)
    removed_edges.add(edge)
    scores[edge] = -1.0
    _update_scores(hg, scores, newly_covered, removed_edges)
    writer.update(edge, newly_covered)
    return True


def _best_in_partition(part_edges, removed_edges, scores, hg=None,
                       covered_vertices=None, overlap_penalty=0.0):
    candidates = np.array([e for e in part_edges if e not in removed_edges],
                          dtype=np.int32)
    if len(candidates) == 0:
        return None

    if overlap_penalty:
        best_edge = max(
            candidates,
            key=lambda e: (
                _selection_value(hg, int(e), covered_vertices, scores, overlap_penalty),
                scores[int(e)],
            ),
        )
    else:
        best_idx  = np.argmax(scores[candidates])
        best_edge = candidates[best_idx]

    return int(best_edge) if scores[best_edge] > 0 else None


def _top_partition_candidates(hg, part_edges, removed_edges, covered_vertices,
                              scores, top_per_partition, overlap_penalty):
    candidates = [edge for edge in part_edges
                  if edge not in removed_edges and scores[edge] > 0]
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda edge: (
            _selection_value(hg, edge, covered_vertices, scores, overlap_penalty),
            scores[edge],
        ),
        reverse=True,
    )
    return ranked[:top_per_partition]


def _top_global_candidates(hg, removed_edges, scores, limit):
    candidates = [edge for edge in hg.hedges
                  if edge not in removed_edges and scores[edge] > 0]
    candidates.sort(key=lambda edge: scores[edge], reverse=True)
    return candidates[:limit]


def _future_conflict_weight(hg, edge, peer_edges, covered_vertices, limit):
    total = 0.0
    used = 0
    edge_vertices = hg.hedges_dict[edge] - covered_vertices
    for peer in peer_edges:
        if peer == edge:
            continue
        overlap = edge_vertices & (hg.hedges_dict[peer] - covered_vertices)
        if overlap:
            total += sum(hg.vtx_weights[vtx] for vtx in overlap)
        used += 1
        if used >= limit:
            break
    return total


def _solution_cover_counts(hg, selected_edges):
    counts = {}
    for edge in selected_edges:
        for vtx in hg.hedges_dict[edge]:
            counts[vtx] = counts.get(vtx, 0) + 1
    return counts


def _coverage_weight(hg, cover_counts):
    return sum(hg.vtx_weights[vtx] for vtx, count in cover_counts.items() if count > 0)


def _replacement_gain(hg, cover_counts, in_edge, out_edge):
    out_vertices = hg.hedges_dict[out_edge]
    lost_weight = 0.0
    for vtx in out_vertices:
        if cover_counts.get(vtx, 0) == 1:
            lost_weight += hg.vtx_weights[vtx]

    gain_weight = 0.0
    for vtx in hg.hedges_dict[in_edge]:
        count_after_removal = cover_counts.get(vtx, 0)
        if vtx in out_vertices:
            count_after_removal -= 1
        if count_after_removal <= 0:
            gain_weight += hg.vtx_weights[vtx]

    return gain_weight - lost_weight


def _hmetis_candidate_pool(hg, filename, selected_edges, covered_vertices,
                           nparts, top_per_partition, overlap_penalty,
                           timeout, use_uncovered_only):
    hmetis_covered = covered_vertices if use_uncovered_only else set()
    e_map_inv = write_hgr(hg, hmetis_covered, selected_edges, filename)

    cur_nparts = min(nparts, len(e_map_inv))
    if cur_nparts <= 1:
        return [], 0.0, 0.0

    p = time.time()
    ok = _run_hmetis(filename, cur_nparts, timeout=timeout)
    partition_time = time.time() - p
    if not ok:
        return [], 0.0, partition_time

    with open(f"{filename}.part.{cur_nparts}") as f:
        line = f.read().splitlines()
    if len(line) != len(e_map_inv):
        return [], 0.0, partition_time

    p2 = time.time()
    partitions = _parse_partitions(line, e_map_inv)
    parse_time = time.time() - p2

    scores = _build_scores(hg, covered_vertices, selected_edges)
    candidates = set()
    for part_edges in partitions.values():
        candidates.update(
            _top_partition_candidates(
                hg, part_edges, selected_edges, covered_vertices,
                scores, top_per_partition, overlap_penalty
            )
        )

    return list(candidates), parse_time, partition_time


def _hmetis_fixed_partitions(hg, filename, nparts, timeout, hmetis_params=None):
    e_map_inv = write_hgr(hg, set(), set(), filename)
    cur_nparts = min(nparts, len(e_map_inv))
    if cur_nparts <= 1:
        return {}, 0.0, 0.0

    p = time.time()
    ok = _run_hmetis(filename, cur_nparts, timeout=timeout, hmetis_params=hmetis_params)
    partition_time = time.time() - p
    if not ok:
        return {}, 0.0, partition_time

    p2 = time.time()
    partitions = _read_partition_file(
        f"{filename}.part.{cur_nparts}",
        e_map_inv,
        label="hMETIS",
    )
    parse_time = time.time() - p2
    if partitions is None:
        return {}, 0.0, partition_time
    return partitions, parse_time, partition_time


def _random_fixed_partitions(hg, nparts, seed=None):
    cur_nparts = min(nparts, hg.nhedges)
    if cur_nparts <= 1:
        return {0: set(hg.hedges)}, 0.0, 0.0

    rng = random.Random(seed)
    edges = list(hg.hedges)
    rng.shuffle(edges)

    partitions = {part: set() for part in range(cur_nparts)}
    for idx, edge in enumerate(edges):
        partitions[idx % cur_nparts].add(edge)
    return partitions, 0.0, 0.0


def _overlap_greedy_fixed_partitions(hg, nparts, seed=None):
    cur_nparts = min(nparts, hg.nhedges)
    if cur_nparts <= 1:
        return {0: set(hg.hedges)}, 0.0, 0.0

    rng = random.Random(seed)
    edges = list(hg.hedges)
    rng.shuffle(edges)
    edges.sort(key=lambda edge: (-len(hg.hedges_dict[edge]), edge))

    partitions = {part: set() for part in range(cur_nparts)}
    part_vertices = {part: set() for part in range(cur_nparts)}
    part_loads = {part: 0 for part in range(cur_nparts)}
    target_load = max(1.0, len(edges) / cur_nparts)

    for edge in edges:
        edge_vertices = hg.hedges_dict[edge]
        best_part = None
        best_score = None
        best_load = None

        for part in range(cur_nparts):
            overlap = len(edge_vertices & part_vertices[part])
            load_penalty = part_loads[part] / target_load
            score = overlap - load_penalty
            if (
                best_score is None
                or score > best_score
                or (np.isclose(score, best_score) and part_loads[part] < best_load)
                or (
                    np.isclose(score, best_score)
                    and part_loads[part] == best_load
                    and part < best_part
                )
            ):
                best_part = part
                best_score = score
                best_load = part_loads[part]

        partitions[best_part].add(edge)
        part_vertices[best_part].update(edge_vertices)
        part_loads[best_part] += 1

    return partitions, 0.0, 0.0


def _write_patoh_hypergraph(hg, filename):
    live_edges = list(hg.hedges)
    e_map = {edge: idx + 1 for idx, edge in enumerate(live_edges)}
    e_map_inv = {idx + 1: edge for idx, edge in enumerate(live_edges)}

    nets = []
    pin_count = 0
    for vtx in hg.vtxs:
        incident = [e_map[edge] for edge in hg.vtxs_dict[vtx] if edge in e_map]
        if incident:
            nets.append(sorted(incident))
            pin_count += len(incident)

    with open(filename, "w") as f:
        f.write(f"1 {len(live_edges)} {len(nets)} {pin_count}\n")
        for net in nets:
            f.write(" ".join(map(str, net)) + "\n")

    return e_map_inv


def _kahypar_spec():
    return importlib.util.find_spec("kahypar")


def _resolve_kahypar_config(kahypar_module, config_path=None, preset="km1_kKaHyPar_sea20.ini"):
    if config_path:
        resolved = Path(config_path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"KaHyPar config file not found: {resolved}")
        return str(resolved)

    module_path = Path(kahypar_module.__file__).resolve()
    candidate_paths = [
        module_path.parent / "config" / preset,
        module_path.parent.parent / "config" / preset,
        module_path.parent.parent.parent / "config" / preset,
        Path.cwd() / "config" / preset,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not locate a KaHyPar config .ini file automatically. "
        "Provide kahypar_config in test.py."
    )


def _build_kahypar_hypergraph(hg, num_blocks):
    live_edges = list(hg.hedges)
    e_map = {edge: idx for idx, edge in enumerate(live_edges)}
    e_map_inv = {idx: edge for idx, edge in enumerate(live_edges)}

    hyperedge_indices = [0]
    hyperedges = []
    edge_weights = []
    for vtx in hg.vtxs:
        incident = [e_map[edge] for edge in hg.vtxs_dict[vtx] if edge in e_map]
        if not incident:
            continue
        hyperedges.extend(sorted(incident))
        hyperedge_indices.append(len(hyperedges))
        edge_weights.append(1)

    node_weights = [1] * len(live_edges)
    return (
        e_map_inv,
        len(live_edges),
        len(edge_weights),
        hyperedge_indices,
        hyperedges,
        edge_weights,
        node_weights,
    )


def _kahypar_fixed_partitions(
    hg,
    nparts,
    seed=None,
    epsilon=0.03,
    config_path=None,
    config_preset="km1_kKaHyPar_sea20.ini",
):
    spec = _kahypar_spec()
    if spec is None:
        raise ModuleNotFoundError(
            "kahypar Python package is not installed in the active environment"
        )

    import kahypar

    cur_nparts = min(nparts, hg.nhedges)
    if cur_nparts <= 1:
        return {0: set(hg.hedges)}, 0.0, 0.0

    (
        e_map_inv,
        num_nodes,
        num_nets,
        hyperedge_indices,
        hyperedges,
        edge_weights,
        node_weights,
    ) = _build_kahypar_hypergraph(hg, cur_nparts)

    if num_nodes <= 0:
        return {}, 0.0, 0.0

    hypergraph = kahypar.Hypergraph(
        num_nodes,
        num_nets,
        hyperedge_indices,
        hyperedges,
        cur_nparts,
        edge_weights,
        node_weights,
    )

    context = kahypar.Context()
    context.loadINIconfiguration(
        _resolve_kahypar_config(
            kahypar,
            config_path=config_path,
            preset=config_preset,
        )
    )
    context.setK(cur_nparts)
    context.setEpsilon(float(epsilon))
    if seed is not None and hasattr(context, "setSeed"):
        context.setSeed(int(seed))

    start = time.time()
    kahypar.partition(hypergraph, context)
    partition_time = time.time() - start

    partitions = {}
    for node_id in range(num_nodes):
        part = int(hypergraph.blockID(node_id))
        partitions.setdefault(part, set()).add(e_map_inv[node_id])

    return partitions, 0.0, partition_time


def _run_patoh(binary_path, filename, nparts, timeout=120, seed=None, extra_args=None):
    part_file = f"{filename}.part.{nparts}"
    if os.path.exists(part_file):
        os.remove(part_file)

    cmd = [binary_path, filename, str(nparts)]
    if seed is not None:
        cmd.append(f"SD={int(seed)}")
    for arg in extra_args or []:
        cmd.append(str(arg))

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"  [warn] PaToH timed out after {timeout}s")
        return False

    if not os.path.exists(part_file):
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = f": {stderr}" if stderr else ""
            if not detail and stdout:
                detail = f": {stdout.splitlines()[-1]}"
            print(f"  [warn] PaToH exited with code {result.returncode}{detail}")
        print("  [warn] PaToH did not produce partition file")
        return False

    return True


def _patoh_fixed_partitions(hg, filename, nparts, timeout, binary_path, seed=None, extra_args=None):
    if not binary_path:
        raise ValueError("PaToH binary path is required")
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"PaToH binary not found: {binary_path}")

    e_map_inv = _write_patoh_hypergraph(hg, filename)
    cur_nparts = min(nparts, len(e_map_inv))
    if cur_nparts <= 1:
        return {}, 0.0, 0.0

    start = time.time()
    ok = _run_patoh(
        binary_path,
        filename,
        cur_nparts,
        timeout=timeout,
        seed=seed,
        extra_args=extra_args,
    )
    partition_time = time.time() - start
    if not ok:
        return {}, 0.0, partition_time

    parse_start = time.time()
    partitions = _read_partition_file(
        f"{filename}.part.{cur_nparts}",
        e_map_inv,
        label="PaToH",
    )
    parse_time = time.time() - parse_start
    if partitions is None:
        return {}, 0.0, partition_time
    return partitions, parse_time, partition_time


def _hype_fixed_partitions(hg, nparts, seed=None, fringe_size=10, fringe_candidates=2):
    cur_nparts = min(nparts, hg.nhedges)
    if cur_nparts <= 1:
        return {0: set(hg.hedges)}, 0.0, 0.0

    rng = random.Random(seed)
    s = max(1, int(fringe_size))
    r = max(1, int(fringe_candidates))

    target_sizes = [hg.nhedges // cur_nparts] * cur_nparts
    for idx in range(hg.nhedges % cur_nparts):
        target_sizes[idx] += 1

    neighbor_cache = {}
    net_size_cache = {net: len(hg.vtxs_dict[net]) for net in hg.vtxs}
    unassigned = set(hg.hedges)
    unassigned_list = list(hg.hedges)
    unassigned_pos = {edge: idx for idx, edge in enumerate(unassigned_list)}
    partitions = {part: set() for part in range(cur_nparts)}

    def neighbors(edge):
        if edge not in neighbor_cache:
            nbrs = set()
            for net in hg.hedges_dict[edge]:
                nbrs.update(hg.vtxs_dict[net])
            nbrs.discard(edge)
            neighbor_cache[edge] = nbrs
        return neighbor_cache[edge]

    def external_neighbors_score(edge, fringe):
        return len(neighbors(edge) - fringe)

    def remove_unassigned(edge):
        if edge not in unassigned:
            return
        unassigned.remove(edge)
        idx = unassigned_pos.pop(edge)
        last_edge = unassigned_list.pop()
        if idx < len(unassigned_list):
            unassigned_list[idx] = last_edge
            unassigned_pos[last_edge] = idx

    def random_unassigned_vertex():
        if not unassigned_list:
            return None
        return unassigned_list[rng.randrange(len(unassigned_list))]

    for part in range(cur_nparts):
        target = target_sizes[part]
        if target <= 0 or not unassigned:
            continue

        core = set()
        fringe = set()
        cache = {}
        incident_nets = set()
        ordered_nets = []
        ordered_nets_dirty = False

        seed_edge = random_unassigned_vertex()
        if seed_edge is None:
            break
        core.add(seed_edge)
        partitions[part].add(seed_edge)
        remove_unassigned(seed_edge)
        incident_nets.update(hg.hedges_dict[seed_edge])
        ordered_nets = sorted(
            incident_nets,
            key=lambda net: (net_size_cache[net], net),
        )

        while len(core) < target and unassigned:
            # Algorithm 2: determine r fringe candidate vertices
            fringe_candidates_set = set()
            if ordered_nets_dirty:
                ordered_nets = sorted(
                    incident_nets,
                    key=lambda net: (net_size_cache[net], net),
                )
                ordered_nets_dirty = False

            for net in ordered_nets:
                for edge in hg.vtxs_dict[net]:
                    if edge in fringe or edge in core or edge not in unassigned:
                        continue
                    fringe_candidates_set.add(edge)
                    if len(fringe_candidates_set) >= r:
                        break
                if len(fringe_candidates_set) >= r:
                    break

            # Cache dext(v, F_i) once per partition, as in the paper's pseudocode.
            for edge in fringe_candidates_set:
                if edge not in cache:
                    cache[edge] = external_neighbors_score(edge, fringe)

            # Update fringe: keep the best s vertices from F_i U F_cand.
            ranked_fringe = sorted(
                fringe | fringe_candidates_set,
                key=lambda edge: (cache.get(edge, float("inf")), edge),
            )
            fringe = set(ranked_fringe[:s])

            if not fringe:
                random_edge = random_unassigned_vertex()
                if random_edge is None:
                    break
                fringe = {random_edge}
                if random_edge not in cache:
                    cache[random_edge] = external_neighbors_score(random_edge, set())

            # Algorithm 3: move vertex with minimal cached external score into core.
            chosen = min(fringe, key=lambda edge: (cache.get(edge, float("inf")), edge))
            fringe.remove(chosen)
            core.add(chosen)
            partitions[part].add(chosen)
            remove_unassigned(chosen)

            new_nets = hg.hedges_dict[chosen] - incident_nets
            if new_nets:
                incident_nets.update(new_nets)
                ordered_nets_dirty = True

        # Any leftover unassigned vertices are handled by later partitions / final sweep.

    if unassigned:
        ordered_parts = sorted(partitions, key=lambda part: len(partitions[part]))
        for idx, edge in enumerate(sorted(unassigned)):
            partitions[ordered_parts[idx % len(ordered_parts)]].add(edge)

    return partitions, 0.0, 0.0


def _hype_fixed_partitions_old(hg, nparts, seed=None, fringe_size=10, fringe_candidates=2):
    cur_nparts = min(nparts, hg.nhedges)
    if cur_nparts <= 1:
        return {0: set(hg.hedges)}, 0.0, 0.0

    rng = random.Random(seed)
    s = max(1, int(fringe_size))
    r = max(1, int(fringe_candidates))

    target_sizes = [hg.nhedges // cur_nparts] * cur_nparts
    for idx in range(hg.nhedges % cur_nparts):
        target_sizes[idx] += 1

    neighbor_cache = {}
    net_size_cache = {net: len(hg.vtxs_dict[net]) for net in hg.vtxs}
    unassigned = set(hg.hedges)
    partitions = {part: set() for part in range(cur_nparts)}

    def neighbors(edge):
        if edge not in neighbor_cache:
            nbrs = set()
            for net in hg.hedges_dict[edge]:
                nbrs.update(hg.vtxs_dict[net])
            nbrs.discard(edge)
            neighbor_cache[edge] = nbrs
        return neighbor_cache[edge]

    def external_neighbors_score(edge, fringe):
        return len(neighbors(edge) - fringe)

    def random_unassigned_vertex():
        if not unassigned:
            return None
        return rng.choice(tuple(unassigned))

    for part in range(cur_nparts):
        target = target_sizes[part]
        if target <= 0 or not unassigned:
            continue

        core = set()
        fringe = set()
        cache = {}

        seed_edge = random_unassigned_vertex()
        if seed_edge is None:
            break
        core.add(seed_edge)
        partitions[part].add(seed_edge)
        unassigned.remove(seed_edge)

        while len(core) < target and unassigned:
            fringe_candidates_set = set()
            incident_nets = set()
            for edge in core:
                incident_nets.update(hg.hedges_dict[edge])
            ordered_nets = sorted(
                incident_nets,
                key=lambda net: (net_size_cache[net], net),
            )

            for net in ordered_nets:
                for edge in hg.vtxs_dict[net]:
                    if edge in fringe or edge in core or edge not in unassigned:
                        continue
                    fringe_candidates_set.add(edge)
                    if len(fringe_candidates_set) >= r:
                        break
                if len(fringe_candidates_set) >= r:
                    break

            for edge in fringe_candidates_set:
                if edge not in cache:
                    cache[edge] = external_neighbors_score(edge, fringe)

            ranked_fringe = sorted(
                fringe | fringe_candidates_set,
                key=lambda edge: (cache.get(edge, float("inf")), edge),
            )
            fringe = set(ranked_fringe[:s])

            if not fringe:
                random_edge = random_unassigned_vertex()
                if random_edge is None:
                    break
                fringe = {random_edge}
                if random_edge not in cache:
                    cache[random_edge] = external_neighbors_score(random_edge, set())

            chosen = min(fringe, key=lambda edge: (cache.get(edge, float("inf")), edge))
            fringe.remove(chosen)
            core.add(chosen)
            partitions[part].add(chosen)
            unassigned.remove(chosen)

    if unassigned:
        ordered_parts = sorted(partitions, key=lambda part: len(partitions[part]))
        for idx, edge in enumerate(sorted(unassigned)):
            partitions[ordered_parts[idx % len(ordered_parts)]].add(edge)

    return partitions, 0.0, 0.0


def _fixed_partitions(hg, filename, nparts, timeout, method="hmetis", seed=None, **partition_kwargs):
    start = time.time()

    if method == "hmetis":
        partitions, parse_time, partition_time = _hmetis_fixed_partitions(
            hg,
            filename,
            nparts,
            timeout,
            hmetis_params=partition_kwargs.get("hmetis_params"),
        )
        write_time = max(0.0, time.time() - start - partition_time - parse_time)
        return partitions, write_time, partition_time

    if method == "random":
        partitions, _, _ = _random_fixed_partitions(hg, nparts, seed=seed)
        return partitions, 0.0, time.time() - start

    if method in {"overlap_greedy", "streaming_overlap"}:
        partitions, _, _ = _overlap_greedy_fixed_partitions(hg, nparts, seed=seed)
        return partitions, 0.0, time.time() - start

    if method == "patoh":
        partitions, parse_time, partition_time = _patoh_fixed_partitions(
            hg,
            filename,
            nparts,
            timeout,
            binary_path=partition_kwargs.get("patoh_binary"),
            seed=seed,
            extra_args=partition_kwargs.get("patoh_args"),
        )
        write_time = max(0.0, time.time() - start - partition_time - parse_time)
        return partitions, write_time, partition_time

    if method == "kahypar":
        partitions, parse_time, partition_time = _kahypar_fixed_partitions(
            hg,
            nparts,
            seed=seed,
            epsilon=partition_kwargs.get("kahypar_epsilon", 0.03),
            config_path=partition_kwargs.get("kahypar_config"),
            config_preset=partition_kwargs.get(
                "kahypar_config_preset",
                "km1_kKaHyPar_sea20.ini",
            ),
        )
        write_time = max(0.0, time.time() - start - partition_time - parse_time)
        return partitions, write_time, partition_time

    if method == "hype":
        hype_params = partition_kwargs.get("hype_params") or {}
        partitions, _, _ = _hype_fixed_partitions(
            hg,
            nparts,
            seed=seed,
            fringe_size=hype_params.get("fringe_size", 10),
            fringe_candidates=hype_params.get("fringe_candidates", 2),
        )
        return partitions, 0.0, time.time() - start

    if method == "hype_old":
        hype_params = partition_kwargs.get("hype_params") or {}
        partitions, _, _ = _hype_fixed_partitions_old(
            hg,
            nparts,
            seed=seed,
            fringe_size=hype_params.get("fringe_size", 10),
            fringe_candidates=hype_params.get("fringe_candidates", 2),
        )
        return partitions, 0.0, time.time() - start

    raise ValueError(f"unknown partition method: {method}")


def _allocate_partition_budgets(partitions, budget, mode="equal"):
    part_ids = sorted(partitions)
    if not part_ids or budget <= 0:
        return {part: 0 for part in part_ids}

    if mode == "equal":
        ordered = sorted(part_ids, key=lambda part: (-len(partitions[part]), part))
        base = budget // len(ordered)
        remainder = budget % len(ordered)
        return {
            part: base + (1 if idx < remainder else 0)
            for idx, part in enumerate(ordered)
        }

    if mode == "proportional":
        total_edges = sum(len(partitions[part]) for part in part_ids)
        if total_edges <= 0:
            return {part: 0 for part in part_ids}
        raw = {
            part: budget * len(partitions[part]) / total_edges
            for part in part_ids
        }
        budgets = {part: int(np.floor(value)) for part, value in raw.items()}
        remaining = budget - sum(budgets.values())
        ordered = sorted(part_ids, key=lambda part: (raw[part] - budgets[part], len(partitions[part])), reverse=True)
        for part in ordered[:remaining]:
            budgets[part] += 1
        return budgets

    raise ValueError(f"unknown budget allocation mode: {mode}")


def _local_greedy_on_edges(hg, part_edges, budget):
    part_edges = set(part_edges)
    selected_edges = set()
    covered_vertices = set()

    if budget <= 0 or not part_edges:
        return covered_vertices, selected_edges

    scores = np.full(hg.nhedges + 1, -1.0, dtype=np.float64)
    heap = []
    for edge in part_edges:
        score = sum(hg.vtx_weights[vtx] for vtx in hg.hedges_dict[edge])
        scores[edge] = score
        heapq.heappush(heap, (-score, edge))

    while len(selected_edges) < budget:
        best_edge = None
        best_score = 0.0
        while heap:
            neg_score, edge = heapq.heappop(heap)
            if edge in selected_edges:
                continue
            if not np.isclose(-neg_score, scores[edge]):
                continue
            best_edge = edge
            best_score = scores[edge]
            break

        if best_edge is None or best_score <= 0:
            break

        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        if not newly_covered:
            scores[best_edge] = -1.0
            selected_edges.add(best_edge)
            continue

        covered_vertices.update(newly_covered)
        selected_edges.add(best_edge)
        scores[best_edge] = -1.0

        for vtx in newly_covered:
            weight = hg.vtx_weights[vtx]
            for edge in hg.vtxs_dict[vtx]:
                if edge in part_edges and edge not in selected_edges:
                    scores[edge] -= weight
                    heapq.heappush(heap, (-scores[edge], edge))

    return covered_vertices, selected_edges


def _run_partitioned_greedy_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    budget_mode="equal",
    fallback_to_greedy=False,
    partition_method="hmetis",
    partition_seed=None,
    **kwargs,
):
    start_time = time.time()
    partitions, write_time, partition_time = _fixed_partitions(
        hg,
        filename,
        nparts,
        timeout,
        method=partition_method,
        seed=partition_seed,
        **kwargs,
    )

    if not partitions:
        if fallback_to_greedy:
            return pure_greedy_mcp(hg, budget=budget, filename=filename)
        raise RuntimeError(f"{partition_method} partitioning failed for partitioned greedy")

    budgets = _allocate_partition_budgets(partitions, budget, mode=budget_mode)
    selected_edges = set()
    partition_edge_counts = {
        part: len(part_edges) for part, part_edges in partitions.items()
    }
    partition_selected_counts = {}
    local_greedy_time = 0.0
    local_partition_times = {}
    local_wall_start = time.time()

    for part, part_edges in partitions.items():
        local_start = time.time()
        _, local_selected = _local_greedy_on_edges(
            hg,
            part_edges,
            budgets.get(part, 0),
        )
        part_time = time.time() - local_start
        local_greedy_time += part_time
        local_partition_times[part] = round(part_time, 4)
        partition_selected_counts[part] = len(local_selected)
        selected_edges.update(local_selected)

    local_wall_time = time.time() - local_wall_start

    merge_start = time.time()
    covered_vertices = set()
    for edge in selected_edges:
        covered_vertices.update(hg.hedges_dict[edge])
    merge_time = time.time() - merge_start

    part_sizes = [len(edges) for edges in partitions.values()]
    nonzero_budgets = [value for value in budgets.values() if value > 0]
    estimated_parallel_local_time = max(local_partition_times.values()) if local_partition_times else 0.0
    stages = {
        "partition_method": partition_method,
        "partition_count": len(partitions),
        "budget_mode": budget_mode,
        "requested_nparts": nparts,
        "min_partition_edges": min(part_sizes) if part_sizes else None,
        "mean_partition_edges": round(float(np.mean(part_sizes)), 4) if part_sizes else None,
        "max_partition_edges": max(part_sizes) if part_sizes else None,
        "min_partition_budget": min(nonzero_budgets) if nonzero_budgets else 0,
        "max_partition_budget": max(nonzero_budgets) if nonzero_budgets else 0,
        "total_assigned_budget": int(sum(budgets.values())),
        "requested_budget": budget,
        "selected_edges": len(selected_edges),
        "partition_edge_counts": str(
            [(part, partition_edge_counts[part]) for part in sorted(partitions)]
        ),
        "partition_budgets": str(
            [(part, budgets.get(part, 0)) for part in sorted(partitions)]
        ),
        "partition_selected_counts": str(
            [(part, partition_selected_counts.get(part, 0)) for part in sorted(partitions)]
        ),
        "partition_local_greedy_time(s)": round(local_greedy_time, 4),
        "partition_local_wall_time(s)": round(local_wall_time, 4),
        "partition_estimated_parallel_local_time(s)": round(estimated_parallel_local_time, 4),
        "partition_merge_time(s)": round(merge_time, 4),
        "partition_local_times": str(
            [(part, local_partition_times.get(part, 0.0)) for part in sorted(partitions)]
        ),
        "wall_time_s": round(time.time() - start_time, 4),
    }

    return (hg.nhedges, hg.nvtxs), covered_vertices, selected_edges, write_time, partition_time, stages


def _parse_partitions(line, e_map_inv):
    partitions = {}
    for edge, part in enumerate(line, start=1):
        original_edge = e_map_inv[edge]
        partitions.setdefault(int(part), set()).add(original_edge)
    return partitions


def _read_partition_file(part_file, e_map_inv, label):
    partitions = {}
    count = 0
    with open(part_file) as f:
        for count, raw in enumerate(f, start=1):
            part = raw.strip()
            if not part:
                continue
            original_edge = e_map_inv[count]
            partitions.setdefault(int(part), set()).add(original_edge)

    if count != len(e_map_inv):
        print(
            f"  [warn] {label} partition line count mismatch: "
            f"expected {len(e_map_inv)}, got {count}"
        )
        return None
    return partitions


def _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget=None):
    """
    Proper greedy fallback ??picks globally best edge each step.
    Used when hMETIS fails or times out mid-run.
    """
    while len(covered_vertices) < hg.nvtxs:
        if budget is not None and len(removed_edges) >= budget:
            break
        best_edge  = None
        best_score = -np.inf
        for e in hg.hedges:
            if e in removed_edges:
                continue
            s = scores[e]
            if s > best_score:
                best_score = s
                best_edge  = e
        if best_edge is None or best_score <= 0:
            break
        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        if newly_covered:
            covered_vertices.update(newly_covered)
            removed_edges.add(best_edge)
            scores[best_edge] = -1.0
            _update_scores(hg, scores, newly_covered, removed_edges)


def _run_hmetis(filename, nparts, timeout=120, hmetis_params=None):
    part_file = f"{filename}.part.{nparts}"
    params = dict(DEFAULT_HMETIS_PARAMS)
    params.update({k: str(v) for k, v in (hmetis_params or {}).items()})

    # remove stale partition file so we can detect if hMETIS fails to write
    if os.path.exists(part_file):
        os.remove(part_file)

    try:
        result = subprocess.run(
            [
                "./hmetis",
                filename,
                str(nparts),
                params["ubfactor"],
                params["nruns"],
                params["ctype"],
                params["rtype"],
                params["vcycle"],
                params["reconst"],
                params["dbglvl"],
            ],
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"  [warn] hMETIS timed out after {timeout}s")
        return False

    # check if hMETIS actually wrote the file
    if not os.path.exists(part_file):
        if result.returncode != 0:
            print(f"  [warn] hMETIS exited with code {result.returncode}")
        print(f"  [warn] hMETIS did not produce partition file ??likely crashed")
        return False

    return True


def _select_and_update(hg, partitions, covered_vertices, removed_edges,
                       scores, writer, stop_condition, overlap_penalty=0.0):
    """
    Pick at most one edge per partition, updating all state after each pick.

    The partition file order is not meaningful. Because each chosen edge changes
    every remaining edge's marginal gain, selecting partitions in file order can
    choose a weak, high-overlap edge before a strong candidate. Recomputing each
    partition's representative and taking the best representative globally keeps
    the hMETIS diversity constraint without letting arbitrary partition order
    decide the solution.

    stop_condition: a callable() -> bool that signals early exit.
    """
    remaining_parts = dict(partitions)

    while remaining_parts and not stop_condition():
        best_part = None
        best_edge = None
        best_score = -np.inf

        for part, part_edges in remaining_parts.items():
            edge = _best_in_partition(
                part_edges, removed_edges, scores, hg, covered_vertices, overlap_penalty
            )
            if edge is None:
                continue

            selection_score = _selection_value(
                hg, edge, covered_vertices, scores, overlap_penalty
            )
            if selection_score > best_score:
                best_part = part
                best_edge = edge
                best_score = selection_score

        if best_edge is None:
            break

        remaining_parts.pop(best_part)
        if stop_condition():
            break

        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        if newly_covered:
            covered_vertices.update(newly_covered)
            removed_edges.add(best_edge)
            scores[best_edge] = -1.0
            _update_scores(hg, scores, newly_covered, removed_edges)
            writer.update(best_edge, newly_covered)


class _StageTrackerMCP:
    """
    Records coverage and elapsed time at budget/3 and 2*budget/3 milestones.

    FIX: also records edges_used at each snapshot so comparisons across
    algorithms are honest — hMetis selects nparts edges per call and can
    overshoot the threshold by up to nparts-1 edges, while greedy hits it
    exactly. Reporting edges_used alongside coverage makes this visible.

    result() now includes edges_used at each stage boundary so the caller
    can normalise coverage-per-edge when comparing algorithms.
    """

    def __init__(self, budget, start_time):
        self.start_time = start_time
        self.thirds     = [budget // 3, 2 * (budget // 3)]
        self._hit       = [False, False]
        self.snapshots  = [None, None]

    def check(self, removed_edges, covered_vertices):
        n = len(removed_edges)
        for i, threshold in enumerate(self.thirds):
            if not self._hit[i] and n >= threshold:
                self._hit[i] = True
                self.snapshots[i] = {
                    'edges_used':   n,                   # actual count (may exceed threshold)
                    'threshold':    threshold,            # what we were aiming for
                    'overshoot':    n - threshold,        # how far past threshold we are
                    'coverage':     len(covered_vertices),
                    'time_elapsed': round(time.time() - self.start_time, 4),
                }

    def result(self, final_edges, final_coverage, final_time):
        s33 = self.snapshots[0]
        s66 = self.snapshots[1]

        return {
            # ── early stage ──────────────────────────────────────────────────
            'early_coverage':   s33['coverage']                             if s33 else None,
            'early_edges_used': s33['edges_used']                           if s33 else None,
            'early_overshoot':  s33['overshoot']                            if s33 else None,
            'early_time':       s33['time_elapsed']                         if s33 else None,

            # ── mid stage ────────────────────────────────────────────────────
            'mid_coverage':     (s66['coverage']     - s33['coverage'])     if (s33 and s66) else None,
            'mid_edges_used':   (s66['edges_used']   - s33['edges_used'])   if (s33 and s66) else None,
            'mid_overshoot':    s66['overshoot']                            if s66 else None,
            'mid_time':         (s66['time_elapsed'] - s33['time_elapsed']) if (s33 and s66) else None,

            # ── late stage ───────────────────────────────────────────────────
            'late_coverage':    (final_coverage - s66['coverage'])          if s66 else None,
            'late_edges_used':  (final_edges    - s66['edges_used'])        if s66 else None,
            'late_time':        (final_time     - s66['time_elapsed'])      if s66 else None,
        }


class _StageTracker:
    """
    Records edges_used and elapsed time at 33% and 66% coverage milestones.
    Stages:
      early : 0  -> 33%
      mid   : 33 -> 66%
      late  : 66 -> 100%  (derived from final totals minus mid snapshot)
    Call .check() after every selection step.
    """
    MILESTONES = [0.33, 0.66]

    def __init__(self, nvtxs, start_time):
        self.nvtxs      = nvtxs
        self.start_time = start_time
        self._hit       = [False, False]
        self.snapshots  = [None, None]

    def check(self, covered_vertices, removed_edges):
        ratio = len(covered_vertices) / self.nvtxs
        for i, threshold in enumerate(self.MILESTONES):
            if not self._hit[i] and ratio >= threshold:
                self._hit[i] = True
                self.snapshots[i] = {
                    'edges_used':   len(removed_edges),
                    'time_elapsed': round(time.time() - self.start_time, 4),
                }

    def result(self, final_edges, final_time):
        """
        Return dict with per-stage edges and time.
        late_edges / late_time are derived from final minus mid snapshot.
        """
        s33 = self.snapshots[0]
        s66 = self.snapshots[1]

        early_edges = s33['edges_used']   if s33 else None
        early_time  = s33['time_elapsed'] if s33 else None

        mid_edges   = (s66['edges_used']   - s33['edges_used'])   if (s33 and s66) else None
        mid_time    = (s66['time_elapsed'] - s33['time_elapsed']) if (s33 and s66) else None

        late_edges  = (final_edges - s66['edges_used'])   if s66 else None
        late_time   = (final_time  - s66['time_elapsed']) if s66 else None

        return {
            'early_edges': early_edges, 'early_time': early_time,
            'mid_edges':   mid_edges,   'mid_time':   mid_time,
            'late_edges':  late_edges,  'late_time':  late_time,
        }


def hmetis_mcp(hg, budget, filename, nparts=4, timeout=120,
               nparts_mid=None, nparts_late=None, overlap_penalty=0.0, **kwargs):
    """
    nparts       ??partitions for early stage  (0        -> budget/3)
    nparts_mid   ??partitions for mid stage    (budget/3 -> 2*budget/3), defaults to nparts
    nparts_late  ??partitions for late stage   (2*budget/3 -> budget),   defaults to nparts
    """
    nparts_mid  = nparts_mid  if nparts_mid  is not None else nparts
    nparts_late = nparts_late if nparts_late is not None else nparts

    stage_thresholds = [budget // 3, 2 * (budget // 3)]
    nparts_schedule  = [nparts, nparts_mid, nparts_late]

    removed_edges    = set()
    covered_vertices = set()
    iteration        = 0
    write_time       = 0
    partition_time   = 0
    start_time       = time.time()

    scores  = _build_scores(hg, covered_vertices, removed_edges)
    writer  = HgrWriter(hg)
    tracker = _StageTrackerMCP(budget, start_time)

    def _current_nparts():
        n = len(removed_edges)
        if n < stage_thresholds[0]:
            return nparts_schedule[0]
        elif n < stage_thresholds[1]:
            return nparts_schedule[1]
        else:
            return nparts_schedule[2]

    while len(removed_edges) < budget:
        cur_nparts = _current_nparts()

        if (hg.nhedges - len(removed_edges)) < cur_nparts:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        w = time.time()
        e_map_inv = writer.write(filename)
        write_time += time.time() - w

        p  = time.time()
        ok = _run_hmetis(filename, cur_nparts, timeout=timeout)
        partition_time += time.time() - p
        if not ok:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        with open(f"{filename}.part.{cur_nparts}") as f:
            line = f.read().splitlines()
        if len(line) != len(e_map_inv):
            _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        p2 = time.time()
        partitions = _parse_partitions(line, e_map_inv)
        parse_time = time.time() - p2

        prev_len = len(removed_edges)

        s = time.time()
        _select_and_update(
            hg, partitions, covered_vertices, removed_edges, scores, writer,
            stop_condition=lambda: len(removed_edges) >= budget,
            overlap_penalty=overlap_penalty,
        )
        select_time = time.time() - s

        tracker.check(removed_edges, covered_vertices)

        if len(removed_edges) == prev_len:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        iteration += 1
        print(f"iter={iteration}, nparts={cur_nparts}, covered={len(covered_vertices)}, "
              f"removed={len(removed_edges)}, parse={parse_time:.4f}s, "
              f"select={select_time:.4f}s")

    final_time     = round(time.time() - start_time, 4)
    final_coverage = len(covered_vertices)
    stages         = tracker.result(len(removed_edges), final_coverage, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time, stages


def hmetis_partitioned_greedy_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    budget_mode="equal",
    fallback_to_greedy=False,
    **kwargs,
):
    """
    Partition edges once with hMETIS, then run greedy independently per partition.

    Each partition has its own local covered set. The final solution is the
    union of all local choices, evaluated on the original whole graph. This
    models embarrassingly parallel greedy with no cross-partition coordination.
    """
    return _run_partitioned_greedy_mcp(
        hg,
        budget,
        filename,
        nparts=nparts,
        timeout=timeout,
        budget_mode=budget_mode,
        fallback_to_greedy=fallback_to_greedy,
        partition_method="hmetis",
        **kwargs,
    )


def random_partitioned_greedy_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    budget_mode="equal",
    partition_seed=None,
    **kwargs,
):
    return _run_partitioned_greedy_mcp(
        hg,
        budget,
        filename,
        nparts=nparts,
        timeout=timeout,
        budget_mode=budget_mode,
        partition_method="random",
        partition_seed=partition_seed,
        **kwargs,
    )


def overlap_greedy_partitioned_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    budget_mode="equal",
    partition_seed=None,
    **kwargs,
):
    return _run_partitioned_greedy_mcp(
        hg,
        budget,
        filename,
        nparts=nparts,
        timeout=timeout,
        budget_mode=budget_mode,
        partition_method="overlap_greedy",
        partition_seed=partition_seed,
        **kwargs,
    )


def patoh_partitioned_greedy_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    budget_mode="equal",
    partition_seed=None,
    patoh_binary="./patoh",
    patoh_args=None,
    **kwargs,
):
    return _run_partitioned_greedy_mcp(
        hg,
        budget,
        filename,
        nparts=nparts,
        timeout=timeout,
        budget_mode=budget_mode,
        partition_method="patoh",
        partition_seed=partition_seed,
        patoh_binary=patoh_binary,
        patoh_args=patoh_args,
        **kwargs,
    )


def kahypar_partitioned_greedy_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    budget_mode="equal",
    partition_seed=None,
    kahypar_config=None,
    kahypar_config_preset="km1_kKaHyPar_sea20.ini",
    kahypar_epsilon=0.03,
    **kwargs,
):
    return _run_partitioned_greedy_mcp(
        hg,
        budget,
        filename,
        nparts=nparts,
        timeout=timeout,
        budget_mode=budget_mode,
        partition_method="kahypar",
        partition_seed=partition_seed,
        kahypar_config=kahypar_config,
        kahypar_config_preset=kahypar_config_preset,
        kahypar_epsilon=kahypar_epsilon,
        **kwargs,
    )


def hype_partitioned_greedy_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    budget_mode="equal",
    partition_seed=None,
    **kwargs,
):
    return _run_partitioned_greedy_mcp(
        hg,
        budget,
        filename,
        nparts=nparts,
        timeout=timeout,
        budget_mode=budget_mode,
        partition_method="hype",
        partition_seed=partition_seed,
        **kwargs,
    )


def hype_old_partitioned_greedy_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    budget_mode="equal",
    partition_seed=None,
    **kwargs,
):
    return _run_partitioned_greedy_mcp(
        hg,
        budget,
        filename,
        nparts=nparts,
        timeout=timeout,
        budget_mode=budget_mode,
        partition_method="hype_old",
        partition_seed=partition_seed,
        **kwargs,
    )


def hmetis_one_shot_pool_mcp(
    hg,
    budget,
    filename,
    nparts=8,
    timeout=120,
    top_per_partition=3,
    global_top_k=None,
    pool_factor=2.0,
    overlap_penalty=0.0,
    hmetis_params=None,
    **kwargs,
):
    """
    Partition once with hMETIS, collect a small candidate pool from each
    partition, then run greedy globally on just that reduced pool.

    This keeps hMETIS as a one-time structural preprocessor instead of a hard
    local-budget wall, which often aligns better with MCP than independent
    per-partition greedy.
    """
    start_time = time.time()
    partitions, parse_time, partition_time = _hmetis_fixed_partitions(
        hg,
        filename,
        nparts,
        timeout,
        hmetis_params=hmetis_params,
    )
    write_time = parse_time

    if not partitions:
        return pure_greedy_mcp(hg, budget=budget, filename=filename)

    scores = _build_scores(hg, set(), set())
    candidate_edges = set()
    for part_edges in partitions.values():
        candidate_edges.update(
            _top_partition_candidates(
                hg,
                part_edges,
                set(),
                set(),
                scores,
                top_per_partition,
                overlap_penalty,
            )
        )

    target_pool_size = max(budget, int(np.ceil(pool_factor * budget)))
    if global_top_k is None:
        global_top_k = target_pool_size
    if global_top_k > 0 and len(candidate_edges) < target_pool_size:
        candidate_edges.update(_top_global_candidates(hg, set(), scores, global_top_k))

    greedy_start = time.time()
    covered_vertices, selected_edges = _local_greedy_on_edges(hg, candidate_edges, budget)
    candidate_greedy_time = time.time() - greedy_start

    stages = {
        "partition_method": "hmetis_one_shot_pool",
        "partition_count": len(partitions),
        "requested_nparts": nparts,
        "candidate_pool_size": len(candidate_edges),
        "top_per_partition": top_per_partition,
        "global_top_k": global_top_k,
        "pool_factor": pool_factor,
        "partition_edge_counts": str(
            [(part, len(partitions[part])) for part in sorted(partitions)]
        ),
        "partition_local_greedy_time(s)": round(candidate_greedy_time, 4),
        "partition_local_wall_time(s)": round(candidate_greedy_time, 4),
        "partition_estimated_parallel_local_time(s)": round(candidate_greedy_time, 4),
        "partition_merge_time(s)": 0.0,
        "partition_local_times": "[]",
        "wall_time_s": round(time.time() - start_time, 4),
    }

    return (hg.nhedges, hg.nvtxs), covered_vertices, selected_edges, write_time, partition_time, stages


def hmetis_mcp_pool(hg, budget, filename, nparts=8, timeout=120,
                    top_per_partition=3, min_gain_ratio=0.7,
                    max_picks_per_round=None, max_per_partition_per_round=1,
                    overlap_penalty=0.0, greedy_fill=True, verbose=True,
                    **kwargs):
    """
    hMETIS-guided candidate-pool heuristic for MCP.

    hMETIS still supplies the diversity signal: each partition contributes a
    small candidate list. Selection is then global over that pool using current
    marginal gains, and weak partition representatives can be skipped. This
    avoids forcing a low-gain set just because its partition has not been used.
    """
    removed_edges    = set()
    covered_vertices = set()
    iteration        = 0
    write_time       = 0.0
    partition_time   = 0.0
    start_time       = time.time()

    scores  = _build_scores(hg, covered_vertices, removed_edges)
    writer  = HgrWriter(hg)
    tracker = _StageTrackerMCP(budget, start_time)

    while len(removed_edges) < budget:
        cur_nparts = min(nparts, hg.nhedges - len(removed_edges))
        if cur_nparts <= 1:
            if greedy_fill:
                _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        w = time.time()
        e_map_inv = writer.write(filename)
        write_time += time.time() - w

        p = time.time()
        ok = _run_hmetis(filename, cur_nparts, timeout=timeout)
        partition_time += time.time() - p
        if not ok:
            if greedy_fill:
                _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        with open(f"{filename}.part.{cur_nparts}") as f:
            line = f.read().splitlines()
        if len(line) != len(e_map_inv):
            if greedy_fill:
                _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        p2 = time.time()
        partitions = _parse_partitions(line, e_map_inv)
        parse_time = time.time() - p2

        candidate_parts = {}
        for part, part_edges in partitions.items():
            top_edges = _top_partition_candidates(
                hg, part_edges, removed_edges, covered_vertices, scores,
                top_per_partition, overlap_penalty
            )
            for edge in top_edges:
                candidate_parts[edge] = part

        picks_limit = max_picks_per_round
        if picks_limit is None:
            picks_limit = max(1, min(cur_nparts, budget - len(removed_edges)))

        prev_len = len(removed_edges)
        picks_this_round = 0
        part_pick_counts = {}

        s = time.time()
        while candidate_parts and len(removed_edges) < budget and picks_this_round < picks_limit:
            global_best_edge, global_best_score = _global_best_edge(hg, removed_edges, scores)
            if global_best_edge is None or global_best_score <= 0:
                break

            threshold = min_gain_ratio * global_best_score
            best_edge = None
            best_value = -np.inf

            for edge, part in list(candidate_parts.items()):
                if edge in removed_edges or scores[edge] <= 0:
                    candidate_parts.pop(edge, None)
                    continue
                if part_pick_counts.get(part, 0) >= max_per_partition_per_round:
                    continue
                if scores[edge] < threshold:
                    continue

                value = _selection_value(hg, edge, covered_vertices, scores, overlap_penalty)
                if value > best_value or (
                    value == best_value and scores[edge] > (scores[best_edge] if best_edge else -np.inf)
                ):
                    best_edge = edge
                    best_value = value

            if best_edge is None:
                if not greedy_fill:
                    break
                best_edge = global_best_edge
                best_part = None
            else:
                best_part = candidate_parts.get(best_edge)

            if _apply_selected_edge(hg, best_edge, covered_vertices, removed_edges, scores, writer):
                picks_this_round += 1
                if best_part is not None:
                    part_pick_counts[best_part] = part_pick_counts.get(best_part, 0) + 1
                    if part_pick_counts[best_part] >= max_per_partition_per_round:
                        for edge, part in list(candidate_parts.items()):
                            if part == best_part:
                                candidate_parts.pop(edge, None)
                candidate_parts.pop(best_edge, None)
                tracker.check(removed_edges, covered_vertices)
            else:
                candidate_parts.pop(best_edge, None)

        select_time = time.time() - s

        if len(removed_edges) == prev_len:
            if greedy_fill:
                _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        iteration += 1
        if verbose:
            print(f"[pool] iter={iteration}, nparts={cur_nparts}, candidates={len(candidate_parts)}, "
                  f"picked={len(removed_edges) - prev_len}, covered={len(covered_vertices)}, "
                  f"removed={len(removed_edges)}, parse={parse_time:.4f}s, "
                  f"select={select_time:.4f}s")

    final_time     = round(time.time() - start_time, 4)
    final_coverage = len(covered_vertices)
    stages         = tracker.result(len(removed_edges), final_coverage, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time, stages


def hmetis_guided_greedy(hg, budget, filename, nparts=8, timeout=120,
                         top_per_partition=3, global_top_k=30,
                         min_gain_ratio=0.70, refresh_interval=None,
                         future_top_k=80, future_window=40,
                         future_overlap_penalty=0.06,
                         covered_overlap_penalty=0.80,
                         partition_reuse_penalty=0.08,
                         greedy_fill=True, verbose=True, **kwargs):
    """
    Greedy with hMETIS-guided state awareness.

    hMETIS supplies a diverse candidate pool from the residual uncovered graph.
    Selection still uses current marginal gain, but adds two state terms:
    overlap with already-covered vertices and an estimate of how much choosing
    this edge would reduce strong future candidates.
    """
    removed_edges = set()
    covered_vertices = set()
    iteration = 0
    write_time = 0.0
    partition_time = 0.0
    start_time = time.time()

    scores = _build_scores(hg, covered_vertices, removed_edges)
    writer = HgrWriter(hg)
    tracker = _StageTrackerMCP(budget, start_time)

    while len(removed_edges) < budget:
        cur_nparts = min(nparts, hg.nhedges - len(removed_edges))
        if cur_nparts <= 1:
            if greedy_fill:
                _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        w = time.time()
        e_map_inv = writer.write(filename)
        write_time += time.time() - w

        p = time.time()
        ok = _run_hmetis(filename, cur_nparts, timeout=timeout)
        partition_time += time.time() - p
        if not ok:
            if greedy_fill:
                _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        with open(f"{filename}.part.{cur_nparts}") as f:
            line = f.read().splitlines()
        if len(line) != len(e_map_inv):
            if greedy_fill:
                _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        p2 = time.time()
        partitions = _parse_partitions(line, e_map_inv)
        parse_time = time.time() - p2

        candidate_parts = {}
        for part, part_edges in partitions.items():
            top_edges = _top_partition_candidates(
                hg, part_edges, removed_edges, covered_vertices, scores,
                top_per_partition, 0.0
            )
            for edge in top_edges:
                candidate_parts[edge] = part

        for edge in _top_global_candidates(hg, removed_edges, scores, global_top_k):
            candidate_parts.setdefault(edge, None)

        picks_limit = refresh_interval
        if picks_limit is None:
            picks_limit = max(1, min(cur_nparts, budget - len(removed_edges)))

        prev_len = len(removed_edges)
        picks_this_round = 0
        part_pick_counts = {}

        s = time.time()
        while candidate_parts and len(removed_edges) < budget and picks_this_round < picks_limit:
            global_best_edge, global_best_score = _global_best_edge(hg, removed_edges, scores)
            if global_best_edge is None or global_best_score <= 0:
                break

            for edge in _top_global_candidates(hg, removed_edges, scores, global_top_k):
                candidate_parts.setdefault(edge, None)

            peer_edges = _top_global_candidates(hg, removed_edges, scores, future_top_k)
            threshold = min_gain_ratio * global_best_score
            best_edge = None
            best_value = -np.inf
            best_raw_score = -np.inf

            for edge, part in list(candidate_parts.items()):
                if edge in removed_edges or scores[edge] <= 0:
                    candidate_parts.pop(edge, None)
                    continue
                if scores[edge] < threshold:
                    continue

                covered_overlap = sum(
                    hg.vtx_weights[vtx]
                    for vtx in hg.hedges_dict[edge] & covered_vertices
                )
                future_conflict = _future_conflict_weight(
                    hg, edge, peer_edges, covered_vertices, future_window
                )
                part_reuse = part_pick_counts.get(part, 0) if part is not None else 0
                part_penalty = partition_reuse_penalty * global_best_score * part_reuse

                value = (
                    scores[edge]
                    - covered_overlap_penalty * covered_overlap
                    - future_overlap_penalty * future_conflict
                    - part_penalty
                )

                if value > best_value or (value == best_value and scores[edge] > best_raw_score):
                    best_edge = edge
                    best_value = value
                    best_raw_score = scores[edge]

            if best_edge is None:
                if not greedy_fill:
                    break
                best_edge = global_best_edge

            best_part = candidate_parts.get(best_edge)
            if _apply_selected_edge(hg, best_edge, covered_vertices, removed_edges, scores, writer):
                picks_this_round += 1
                if best_part is not None:
                    part_pick_counts[best_part] = part_pick_counts.get(best_part, 0) + 1
                candidate_parts.pop(best_edge, None)
                tracker.check(removed_edges, covered_vertices)
            else:
                candidate_parts.pop(best_edge, None)

        select_time = time.time() - s

        if len(removed_edges) == prev_len:
            if greedy_fill:
                _greedy_fallback(hg, covered_vertices, removed_edges, scores, budget)
            break

        iteration += 1
        if verbose:
            print(f"[guided] iter={iteration}, nparts={cur_nparts}, "
                  f"picked={len(removed_edges) - prev_len}, covered={len(covered_vertices)}, "
                  f"removed={len(removed_edges)}, parse={parse_time:.4f}s, "
                  f"select={select_time:.4f}s")

    final_time = round(time.time() - start_time, 4)
    final_coverage = len(covered_vertices)
    stages = tracker.result(len(removed_edges), final_coverage, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time, stages


def hmetis_mcp_refine(hg, budget, filename, nparts=8, timeout=120,
                      top_per_partition=3, min_gain_ratio=0.7,
                      overlap_penalty=0.0, refine_nparts=None,
                      refine_top_per_partition=6, refine_rounds=5,
                      max_swaps_per_round=None, min_swap_gain=1e-9,
                      refine_use_uncovered_only=False, use_greedy_seed=True,
                      verbose=True,
                      **kwargs):
    """
    Seed with the better of the hMETIS candidate-pool solution and Greedy, then
    use hMETIS again to propose replacement candidates. A swap is accepted only
    when it increases actual MCP coverage, so refinement cannot reduce solution
    quality or fall below Greedy when use_greedy_seed is enabled.
    """
    start_time = time.time()

    seed = hmetis_mcp_pool(
        hg,
        budget=budget,
        filename=filename,
        nparts=nparts,
        timeout=timeout,
        top_per_partition=top_per_partition,
        min_gain_ratio=min_gain_ratio,
        overlap_penalty=overlap_penalty,
        greedy_fill=True,
        verbose=False,
    )

    selected_edges = set(seed[2])
    covered_vertices = set(seed[1])
    write_time = seed[3] or 0.0
    partition_time = seed[4] or 0.0
    stages = seed[5]

    if use_greedy_seed:
        greedy_seed = pure_greedy_mcp(hg, budget=budget, filename=filename)
        greedy_edges = set(greedy_seed[2])
        greedy_covered = set(greedy_seed[1])
        current_value = sum(hg.vtx_weights[vtx] for vtx in covered_vertices)
        greedy_value = sum(hg.vtx_weights[vtx] for vtx in greedy_covered)
        if greedy_value > current_value:
            selected_edges = greedy_edges
            covered_vertices = greedy_covered
            stages = greedy_seed[5]

    if len(covered_vertices) >= hg.nvtxs:
        return (hg.nhedges, hg.nvtxs), covered_vertices, selected_edges, write_time, partition_time, stages

    refine_nparts = refine_nparts if refine_nparts is not None else nparts
    total_swaps = 0

    for round_idx in range(refine_rounds):
        if not selected_edges:
            break

        w = time.time()
        candidates, parse_time, p_time = _hmetis_candidate_pool(
            hg,
            filename,
            selected_edges,
            covered_vertices,
            refine_nparts,
            refine_top_per_partition,
            overlap_penalty,
            timeout,
            refine_use_uncovered_only,
        )
        write_time += time.time() - w - p_time
        partition_time += p_time

        candidates = [edge for edge in candidates if edge not in selected_edges]
        if not candidates:
            break

        swaps_this_round = 0
        improved = True

        while improved and candidates:
            if max_swaps_per_round is not None and swaps_this_round >= max_swaps_per_round:
                break

            improved = False
            cover_counts = _solution_cover_counts(hg, selected_edges)
            best_in = None
            best_out = None
            best_gain = min_swap_gain

            for in_edge in candidates:
                if in_edge in selected_edges:
                    continue
                for out_edge in selected_edges:
                    gain = _replacement_gain(hg, cover_counts, in_edge, out_edge)
                    if gain > best_gain:
                        best_gain = gain
                        best_in = in_edge
                        best_out = out_edge

            if best_in is not None:
                selected_edges.remove(best_out)
                selected_edges.add(best_in)
                candidates.remove(best_in)
                cover_counts = _solution_cover_counts(hg, selected_edges)
                covered_vertices = set(cover_counts)
                swaps_this_round += 1
                total_swaps += 1
                improved = True

        if verbose:
            print(f"[refine] round={round_idx + 1}, candidates={len(candidates)}, "
                  f"swaps={swaps_this_round}, covered={len(covered_vertices)}, "
                  f"parse={parse_time:.4f}s")

        if swaps_this_round == 0:
            break

    final_time = round(time.time() - start_time, 4)
    if verbose:
        print(f"[refine] done, swaps={total_swaps}, covered={len(covered_vertices)}, "
              f"selected={len(selected_edges)}, time={final_time:.4f}s")

    return (hg.nhedges, hg.nvtxs), covered_vertices, selected_edges, write_time, partition_time, stages


def hmetis_single_partition_refine(hg, budget, filename, nparts=8, timeout=120,
                                   top_per_partition=6, min_gain_ratio=0.0,
                                   overlap_penalty=0.0,
                                   refine_top_per_partition=6,
                                   refine_rounds=5,
                                   max_swaps_per_round=None,
                                   min_swap_gain=1e-9,
                                   use_greedy_seed=False,
                                   verbose=True,
                                   **kwargs):
    """
    Partition once with hMETIS, then reuse those fixed partitions for both
    construction and swap refinement.

    The intent is to test whether hMETIS has value as a one-time structural
    preprocessor, avoiding the repeated external partitioning overhead of
    hmetis_mcp_refine.
    """
    start_time = time.time()
    write_time = 0.0
    partition_time = 0.0

    partitions, parse_time, p_time = _hmetis_fixed_partitions(
        hg, filename, nparts, timeout
    )
    partition_time += p_time
    write_time += parse_time

    if not partitions:
        return pure_greedy_mcp(hg, budget=budget, filename=filename)

    selected_edges = set()
    covered_vertices = set()
    scores = _build_scores(hg, covered_vertices, selected_edges)
    tracker = _StageTrackerMCP(budget, start_time)

    candidate_parts = {}
    global_best_edge, global_best_score = _global_best_edge(hg, selected_edges, scores)
    threshold = min_gain_ratio * global_best_score if global_best_edge is not None else 0.0
    for part, part_edges in partitions.items():
        top_edges = _top_partition_candidates(
            hg, part_edges, selected_edges, covered_vertices, scores,
            top_per_partition, overlap_penalty
        )
        for edge in top_edges:
            if scores[edge] >= threshold:
                candidate_parts[edge] = part

    while candidate_parts and len(selected_edges) < budget:
        best_edge = None
        best_value = -np.inf

        for edge in list(candidate_parts):
            if edge in selected_edges or scores[edge] <= 0:
                candidate_parts.pop(edge, None)
                continue
            value = _selection_value(hg, edge, covered_vertices, scores, overlap_penalty)
            if value > best_value:
                best_edge = edge
                best_value = value

        if best_edge is None:
            break

        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        if newly_covered:
            covered_vertices.update(newly_covered)
            selected_edges.add(best_edge)
            scores[best_edge] = -1.0
            _update_scores(hg, scores, newly_covered, selected_edges)
            tracker.check(selected_edges, covered_vertices)
        candidate_parts.pop(best_edge, None)

    while len(selected_edges) < budget:
        best_edge, best_score = _global_best_edge(hg, selected_edges, scores)
        if best_edge is None or best_score <= 0:
            break
        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        if not newly_covered:
            break
        covered_vertices.update(newly_covered)
        selected_edges.add(best_edge)
        scores[best_edge] = -1.0
        _update_scores(hg, scores, newly_covered, selected_edges)
        tracker.check(selected_edges, covered_vertices)

    if use_greedy_seed:
        greedy_seed = pure_greedy_mcp(hg, budget=budget, filename=filename)
        greedy_edges = set(greedy_seed[2])
        greedy_covered = set(greedy_seed[1])
        current_value = sum(hg.vtx_weights[vtx] for vtx in covered_vertices)
        greedy_value = sum(hg.vtx_weights[vtx] for vtx in greedy_covered)
        if greedy_value > current_value:
            selected_edges = greedy_edges
            covered_vertices = greedy_covered

    for round_idx in range(refine_rounds):
        cover_counts = _solution_cover_counts(hg, selected_edges)
        scores = _build_scores(hg, set(cover_counts), selected_edges)
        candidates = set()
        for part_edges in partitions.values():
            candidates.update(
                _top_partition_candidates(
                    hg, part_edges, selected_edges, set(cover_counts),
                    scores, refine_top_per_partition, overlap_penalty
                )
            )

        candidates = [edge for edge in candidates if edge not in selected_edges]
        if not candidates:
            break

        swaps_this_round = 0
        improved = True
        while improved and candidates:
            if max_swaps_per_round is not None and swaps_this_round >= max_swaps_per_round:
                break

            improved = False
            cover_counts = _solution_cover_counts(hg, selected_edges)
            best_in = None
            best_out = None
            best_gain = min_swap_gain

            for in_edge in candidates:
                if in_edge in selected_edges:
                    continue
                for out_edge in selected_edges:
                    gain = _replacement_gain(hg, cover_counts, in_edge, out_edge)
                    if gain > best_gain:
                        best_gain = gain
                        best_in = in_edge
                        best_out = out_edge

            if best_in is not None:
                selected_edges.remove(best_out)
                selected_edges.add(best_in)
                candidates.remove(best_in)
                covered_vertices = set(_solution_cover_counts(hg, selected_edges))
                swaps_this_round += 1
                improved = True

        if verbose:
            print(f"[single-refine] round={round_idx + 1}, swaps={swaps_this_round}, "
                  f"covered={len(covered_vertices)}")

    final_time = round(time.time() - start_time, 4)
    final_coverage = len(covered_vertices)
    stages = tracker.result(len(selected_edges), final_coverage, final_time)
    return (hg.nhedges, hg.nvtxs), covered_vertices, selected_edges, write_time, partition_time, stages


def hmetis_mcp_early(hg, budget, filename, nparts=8, timeout=120,
                     overlap_penalty=0.0, **kwargs):
    """
    Phase 1 (0 -> budget/3)        : hMetis-guided selection
    Phase 2 (budget/3 -> budget)   : pure greedy fallback
    """
    removed_edges    = set()
    covered_vertices = set()
    iteration        = 0
    write_time       = 0
    partition_time   = 0
    start_time       = time.time()

    scores  = _build_scores(hg, covered_vertices, removed_edges)
    writer  = HgrWriter(hg)
    tracker = _StageTrackerMCP(budget, start_time)

    one_third = budget // 3

    while len(removed_edges) < one_third:
        if (hg.nhedges - len(removed_edges)) < nparts:
            break

        w = time.time()
        e_map_inv = writer.write(filename)
        write_time += time.time() - w

        p  = time.time()
        ok = _run_hmetis(filename, nparts, timeout=timeout)
        partition_time += time.time() - p
        if not ok:
            break

        with open(f"{filename}.part.{nparts}") as f:
            line = f.read().splitlines()
        if len(line) != len(e_map_inv):
            break

        p2 = time.time()
        partitions = _parse_partitions(line, e_map_inv)
        parse_time = time.time() - p2

        prev_len = len(removed_edges)

        s = time.time()
        _select_and_update(
            hg, partitions, covered_vertices, removed_edges, scores, writer,
            stop_condition=lambda: len(removed_edges) >= budget,
            overlap_penalty=overlap_penalty,
        )
        select_time = time.time() - s

        tracker.check(removed_edges, covered_vertices)

        if len(removed_edges) == prev_len:
            break

        iteration += 1
        print(f"[hmetis] iter={iteration}, covered={len(covered_vertices)}, "
              f"removed={len(removed_edges)}, parse={parse_time:.4f}s, "
              f"select={select_time:.4f}s")

    print(f"[hmetis_mcp_early] Phase 1 done ??removed={len(removed_edges)}, "
          f"covered={len(covered_vertices)}")

    # Phase 2: greedy
    while len(removed_edges) < budget:
        best_edge  = None
        best_score = 0.0
        for e in hg.hedges:
            if e in removed_edges:
                continue
            if scores[e] > best_score:
                best_score = scores[e]
                best_edge  = e
        if best_edge is None or best_score <= 0:
            break
        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        covered_vertices.update(newly_covered)
        removed_edges.add(best_edge)
        scores[best_edge] = -1.0
        _update_scores(hg, scores, newly_covered, removed_edges)
        tracker.check(removed_edges, covered_vertices)

    final_time     = round(time.time() - start_time, 4)
    final_coverage = len(covered_vertices)
    stages         = tracker.result(len(removed_edges), final_coverage, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time, stages


def pure_greedy_mcp_0(hg, budget, filename=None, **kwargs):
    covered_vertices = set()
    removed_edges    = set()
    start_time       = time.time()
    tracker          = _StageTrackerMCP(budget, start_time)

    while len(removed_edges) < budget:
        best_score = 0
        best_edge  = None
        for hedge in hg.hedges:
            if hedge in removed_edges:
                continue
            score = sum(hg.vtx_weights[v]
                        for v in hg.hedges_dict[hedge] - covered_vertices)
            if score > best_score:
                best_score = score
                best_edge  = hedge
        if best_edge is None or best_score == 0:
            break
        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        covered_vertices.update(newly_covered)
        removed_edges.add(best_edge)
        tracker.check(removed_edges, covered_vertices)

    final_time     = round(time.time() - start_time, 4)
    final_coverage = len(covered_vertices)
    stages         = tracker.result(len(removed_edges), final_coverage, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, None, None, stages

def pure_greedy_mcp(hg, budget, filename=None, **kwargs):
    covered_vertices = set()
    removed_edges    = set()
    start_time       = time.time()
    tracker          = _StageTrackerMCP(budget, start_time)

    scores = _build_scores(hg, covered_vertices, removed_edges)  # build once

    while len(removed_edges) < budget:
        best_score = 0
        best_edge  = None
        for hedge in hg.hedges:
            if hedge in removed_edges:
                continue
            if scores[hedge] > best_score:
                best_score = scores[hedge]
                best_edge  = hedge
        if best_edge is None or best_score == 0:
            break
        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        covered_vertices.update(newly_covered)
        removed_edges.add(best_edge)
        scores[best_edge] = -1.0
        _update_scores(hg, scores, newly_covered, removed_edges)  # incremental update
        tracker.check(removed_edges, covered_vertices)

    final_time     = round(time.time() - start_time, 4)
    final_coverage = len(covered_vertices)
    stages         = tracker.result(len(removed_edges), final_coverage, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, None, None, stages


def pure_oblswap_mcp(hg, budget, filename=None, max_iter=200,
                     candidate_pool_size=None, min_swap_gain=1e-9,
                     verbose=False, **kwargs):
    """
    Oblivious 1-swap local search.

    Starts from Greedy and repeatedly performs the best improving one-in,
    one-out swap according to the true MCP objective w(Y). This is the k=1
    version of OblSwap from the paper.
    """
    start_time = time.time()
    tracker = _StageTrackerMCP(budget, start_time)

    _, covered_vertices, selected_edges, *_ = pure_greedy_mcp(hg, budget, filename)
    selected_edges = set(selected_edges)
    cover_counts = _solution_cover_counts(hg, selected_edges)
    tracker.check(selected_edges, set(cover_counts))

    for iteration in range(1, max_iter + 1):
        if not selected_edges:
            break

        scores = _build_scores(hg, set(cover_counts), selected_edges)
        candidates = [edge for edge in hg.hedges
                      if edge not in selected_edges and scores[edge] > 0]
        if not candidates:
            break

        if candidate_pool_size is not None:
            candidates = sorted(
                candidates,
                key=lambda edge: scores[edge],
                reverse=True,
            )[:candidate_pool_size]

        best_in = None
        best_out = None
        best_gain = min_swap_gain

        for in_edge in candidates:
            for out_edge in selected_edges:
                gain = _replacement_gain(hg, cover_counts, in_edge, out_edge)
                if gain > best_gain:
                    best_gain = gain
                    best_in = in_edge
                    best_out = out_edge

        if best_in is None:
            break

        selected_edges.remove(best_out)
        selected_edges.add(best_in)
        cover_counts = _solution_cover_counts(hg, selected_edges)
        covered_vertices = set(cover_counts)
        tracker.check(selected_edges, covered_vertices)

        if verbose:
            print(f"[oblswap] iter={iteration}, gain={best_gain:.4f}, "
                  f"covered={len(covered_vertices)}")

    final_time = round(time.time() - start_time, 4)
    final_coverage = len(covered_vertices)
    stages = tracker.result(len(selected_edges), final_coverage, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, selected_edges, None, None, stages


def pure_tabu_mcp(hg, budget, filename=None, max_iter=250, tabu_tenure=50,
                  candidate_pool_size=None, random_candidate_size=0,
                  no_improve_limit=50, allow_non_improving=True, seed=0,
                  verbose=False, **kwargs):
    """
    Tabu 1-swap local search for the standard MCP.

    This follows the paper's Tabu structure for the standard cardinality MCP:
    start from Greedy, repeatedly choose the best non-tabu neighbor, store recent
    solutions in the tabu list, keep the best feasible solution, and terminate
    after the best solution has not strictly improved for no_improve_limit
    iterations. In the paper's notation, tabu_tenure is L and
    no_improve_limit is NT.
    """
    start_time = time.time()
    tracker = _StageTrackerMCP(budget, start_time)
    rng = np.random.default_rng(seed)

    _, covered_vertices, selected_edges, *_ = pure_greedy_mcp(hg, budget, filename)
    selected_edges = set(selected_edges)
    cover_counts = _solution_cover_counts(hg, selected_edges)
    current_value = _coverage_weight(hg, cover_counts)
    best_value = current_value
    best_edges = set(selected_edges)
    best_covered = set(cover_counts)
    tabu_list = deque()
    tabu_set = set()
    no_improve = 0

    tracker.check(selected_edges, best_covered)

    for iteration in range(1, max_iter + 1):
        if no_improve >= no_improve_limit:
            break

        eligible = [edge for edge in hg.hedges if edge not in selected_edges]
        if not eligible or not selected_edges:
            break

        if candidate_pool_size is None:
            candidates = eligible
        else:
            scores = _build_scores(hg, set(cover_counts), selected_edges)
            ranked = sorted(eligible, key=lambda edge: scores[edge], reverse=True)
            candidates = ranked[:candidate_pool_size]
            if random_candidate_size > 0 and len(ranked) > candidate_pool_size:
                tail = ranked[candidate_pool_size:]
                sample_size = min(random_candidate_size, len(tail))
                sampled = rng.choice(tail, size=sample_size, replace=False)
                candidates.extend(int(edge) for edge in sampled)

        best_move = None
        best_move_gain = -np.inf
        best_move_value = -np.inf
        best_move_solution = None

        for in_edge in candidates:
            for out_edge in selected_edges:
                neighbor = frozenset((selected_edges - {out_edge}) | {in_edge})
                if neighbor in tabu_set:
                    continue
                gain = _replacement_gain(hg, cover_counts, in_edge, out_edge)
                if not allow_non_improving and gain <= 0:
                    continue
                value = current_value + gain
                if value > best_move_value:
                    best_move_value = value
                    best_move_gain = gain
                    best_move = (in_edge, out_edge)
                    best_move_solution = neighbor

        if best_move is None:
            break

        in_edge, out_edge = best_move
        selected_edges.remove(out_edge)
        selected_edges.add(in_edge)

        tabu_list.append(best_move_solution)
        tabu_set.add(best_move_solution)
        if len(tabu_list) > tabu_tenure:
            expired = tabu_list.popleft()
            tabu_set.remove(expired)

        cover_counts = _solution_cover_counts(hg, selected_edges)
        current_value = _coverage_weight(hg, cover_counts)

        if current_value > best_value:
            best_value = current_value
            best_edges = set(selected_edges)
            best_covered = set(cover_counts)
            no_improve = 0
            tracker.check(best_edges, best_covered)
        else:
            no_improve += 1

        if verbose and (iteration == 1 or iteration % 25 == 0 or best_move_gain > 0):
            print(f"[tabu] iter={iteration}, gain={best_move_gain:.4f}, "
                  f"current={current_value:.4f}, best={best_value:.4f}, "
                  f"no_improve={no_improve}")

    final_time = round(time.time() - start_time, 4)
    stages = tracker.result(len(best_edges), len(best_covered), final_time)
    return (hg.nhedges, hg.nvtxs), best_covered, best_edges, None, None, stages


#  Set Cover


def hmetis_set_cover(hg, filename, nparts=2, timeout=120, **kwargs):
    removed_edges    = set()
    covered_vertices = set()
    iteration        = 0
    write_time       = 0
    partition_time   = 0
    start_time       = time.time()

    scores  = _build_scores(hg, covered_vertices, removed_edges)
    writer  = HgrWriter(hg)
    tracker = _StageTracker(hg.nvtxs, start_time)

    while len(covered_vertices) < hg.nvtxs:
        if (hg.nhedges - len(removed_edges)) < nparts:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores)
            tracker.check(covered_vertices, removed_edges)
            break

        w = time.time()
        e_map_inv = writer.write(filename)
        write_time += time.time() - w

        p  = time.time()
        ok = _run_hmetis(filename, nparts, timeout=timeout)
        p_time = time.time() - p          # fix: capture before accumulating
        partition_time += p_time

        if not ok:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores)
            tracker.check(covered_vertices, removed_edges)
            break

        with open(f"{filename}.part.{nparts}") as f:
            line = f.read().splitlines()

        if len(line) != len(e_map_inv):
            print(f"  missing: {len(e_map_inv) - len(line)}")
            break

        p2 = time.time()
        partitions = _parse_partitions(line, e_map_inv)
        parse_time = time.time() - p2

        prev_len = len(removed_edges)

        s = time.time()
        _select_and_update(
            hg, partitions, covered_vertices, removed_edges, scores, writer,
            stop_condition=lambda: len(covered_vertices) >= hg.nvtxs
        )
        select_time = time.time() - s

        tracker.check(covered_vertices, removed_edges)

        iteration += 1
        print(f"iter={iteration}, covered={len(covered_vertices)}, removed={len(removed_edges)}, "
              f"parse={parse_time:.4f}s, select={select_time:.4f}s, partition={p_time:.4f}s")

    final_time   = round(time.time() - start_time, 4)
    final_edges  = len(removed_edges)
    stages       = tracker.result(final_edges, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time, stages


HALF_THRESHOLD = 0.50


def pure_greedy_set_cover(hg, filename=None, **kwargs):
    covered_vertices = set()
    removed_edges    = set()
    start_time       = time.time()
    tracker          = _StageTracker(hg.nvtxs, start_time)

    scores = _build_scores(hg, covered_vertices, removed_edges)  # build once

    while len(covered_vertices) < hg.nvtxs:
        best_edge  = None
        best_score = 0.0
        for e in hg.hedges:
            if e in removed_edges:
                continue
            if scores[e] > best_score:
                best_score = scores[e]
                best_edge  = e
        if best_edge is None or best_score <= 0:
            break
        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        covered_vertices.update(newly_covered)
        removed_edges.add(best_edge)
        scores[best_edge] = -1.0
        _update_scores(hg, scores, newly_covered, removed_edges)  # incremental update
        tracker.check(covered_vertices, removed_edges)

    final_time  = round(time.time() - start_time, 4)
    final_edges = len(removed_edges)
    stages      = tracker.result(final_edges, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, None, None, stages


def hmetis_set_cover_greedy_first(hg, filename, nparts=2, timeout=120, **kwargs):
    """
    Phase 1 (0% -> 50%): incremental greedy ??same logic as pure_greedy_set_cover
    Phase 2 (50% -> 100%): hMetis-guided selection
    """
    removed_edges    = set()
    covered_vertices = set()
    write_time       = 0.0
    partition_time   = 0.0
    start_time       = time.time()
    iteration        = 0

    scores  = _build_scores(hg, covered_vertices, removed_edges)
    writer  = HgrWriter(hg)
    tracker = _StageTracker(hg.nvtxs, start_time)

    half = int(HALF_THRESHOLD * hg.nvtxs)

    #  Phase 1: incremental greedy
    while len(covered_vertices) < half:
        best_edge  = None
        best_score = 0.0
        for e in hg.hedges:
            if e in removed_edges:
                continue
            if scores[e] > best_score:
                best_score = scores[e]
                best_edge  = e
        if best_edge is None or best_score <= 0:
            break
        newly_covered = hg.hedges_dict[best_edge] - covered_vertices
        if newly_covered:
            covered_vertices.update(newly_covered)
            removed_edges.add(best_edge)
            scores[best_edge] = -1.0
            _update_scores(hg, scores, newly_covered, removed_edges)
            writer.update(best_edge, newly_covered)
        tracker.check(covered_vertices, removed_edges)

    print(f"[greedy_first] Phase 1 done ??covered={len(covered_vertices)}, "
          f"edges_used={len(removed_edges)}")

    #  Phase 2: hMetis-guided
    while len(covered_vertices) < hg.nvtxs:
        if (hg.nhedges - len(removed_edges)) < nparts:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores)
            tracker.check(covered_vertices, removed_edges)
            break

        w = time.time()
        e_map_inv = writer.write(filename)
        write_time += time.time() - w

        p  = time.time()
        ok = _run_hmetis(filename, nparts, timeout=timeout)
        partition_time += time.time() - p

        if not ok:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores)
            tracker.check(covered_vertices, removed_edges)
            break

        with open(f"{filename}.part.{nparts}") as f:
            line = f.read().splitlines()

        if len(line) != len(e_map_inv):
            break

        partitions = _parse_partitions(line, e_map_inv)
        _select_and_update(
            hg, partitions, covered_vertices, removed_edges, scores, writer,
            stop_condition=lambda: len(covered_vertices) >= hg.nvtxs,
        )
        tracker.check(covered_vertices, removed_edges)

        iteration += 1
        print(f"  [hmetis] iter={iteration}, covered={len(covered_vertices)}, "
              f"edges_used={len(removed_edges)}")

    final_time  = round(time.time() - start_time, 4)
    final_edges = len(removed_edges)
    stages      = tracker.result(final_edges, final_time)
    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time, stages
