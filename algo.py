

import subprocess, time
import numpy as np
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


def _parse_partitions(line, e_map_inv):
    partitions = {}
    for edge, part in enumerate(line, start=1):
        original_edge = e_map_inv[edge]
        partitions.setdefault(int(part), set()).add(original_edge)
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


def _run_hmetis(filename, nparts, timeout=120):
    import os
    part_file = f"{filename}.part.{nparts}"

    # remove stale partition file so we can detect if hMETIS fails to write
    if os.path.exists(part_file):
        os.remove(part_file)

    try:
        result = subprocess.run(
            ["./hmetis", filename, str(nparts), "3", "5", "5", "2", "3", "0", "0"],
            timeout=timeout, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        print(f"  [warn] hMETIS timed out after {timeout}s")
        return False

    # check if hMETIS actually wrote the file
    if not os.path.exists(part_file):
        if result.returncode != 0:
            stderr = result.stderr.strip()
            detail = f": {stderr}" if stderr else ""
            print(f"  [warn] hMETIS exited with code {result.returncode}{detail}")
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
                      refine_use_uncovered_only=False, verbose=True,
                      **kwargs):
    """
    Seed with the hMETIS candidate-pool heuristic, then use hMETIS again to
    propose replacement candidates. A swap is accepted only when it increases
    actual MCP coverage, so refinement cannot reduce solution quality.
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


def pure_tabu_mcp(hg, budget, filename=None, max_iter=250, tabu_tenure=25,
                  candidate_pool_size=120, random_candidate_size=30,
                  no_improve_limit=80, allow_non_improving=True, seed=0,
                  verbose=False, **kwargs):
    """
    Tabu-style 1-swap local search baseline for MCP.

    Starts from greedy, repeatedly swaps one selected edge with one unselected
    edge, and keeps the best solution ever seen. The tabu list prevents recently
    removed edges from immediately returning, unless the move beats the best
    solution found so far.
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
    tabu_until = {}
    no_improve = 0

    tracker.check(selected_edges, best_covered)

    for iteration in range(1, max_iter + 1):
        if no_improve >= no_improve_limit:
            break

        scores = _build_scores(hg, set(cover_counts), selected_edges)
        eligible = [edge for edge in hg.hedges
                    if edge not in selected_edges and scores[edge] > 0]
        if not eligible or not selected_edges:
            break

        ranked = sorted(eligible, key=lambda edge: scores[edge], reverse=True)
        candidates = ranked[:candidate_pool_size]
        if random_candidate_size > 0 and len(ranked) > candidate_pool_size:
            tail = ranked[candidate_pool_size:]
            sample_size = min(random_candidate_size, len(tail))
            sampled = rng.choice(tail, size=sample_size, replace=False)
            candidates.extend(int(edge) for edge in sampled)

        best_move = None
        best_move_gain = -np.inf

        for in_edge in candidates:
            for out_edge in selected_edges:
                gain = _replacement_gain(hg, cover_counts, in_edge, out_edge)
                aspiration = current_value + gain > best_value
                if tabu_until.get(in_edge, 0) > iteration and not aspiration:
                    continue
                if not allow_non_improving and gain <= 0:
                    continue
                if gain > best_move_gain:
                    best_move_gain = gain
                    best_move = (in_edge, out_edge)

        if best_move is None:
            break

        in_edge, out_edge = best_move
        selected_edges.remove(out_edge)
        selected_edges.add(in_edge)
        tabu_until[out_edge] = iteration + tabu_tenure

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
