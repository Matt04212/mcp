import subprocess, time
import numpy as np
from greedy import greedy
from subgraph2 import HgrWriter, write_hgr


# ─────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────

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
    best_idx  = np.argmax(scores[candidates])
    best_edge = candidates[best_idx]
    return int(best_edge) if scores[best_edge] > 0 else None


def _parse_partitions(line, e_map_inv):
    partitions = {}
    for edge, part in enumerate(line, start=1):
        original_edge = e_map_inv[edge]
        partitions.setdefault(int(part), set()).add(original_edge)
    return partitions


def _greedy_fallback(hg, covered_vertices, removed_edges, scores):
    """
    Proper greedy fallback — picks globally best edge each step.
    Used when hMETIS fails or times out mid-run.
    """
    while len(covered_vertices) < hg.nvtxs:
        best_edge  = None
        best_score = 0.0
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


def _run_hmetis(filename, nparts, ub_flag, timeout=120):
    """
    Run hMETIS with a timeout guard.
    Returns True on success, False if it timed out.
    ub_flag: 1 for MCP mode, 2 for set-cover mode.
    """
    cmd = f"./hmetis {filename} {nparts} 5 1 {ub_flag} 1 0 0 0"
    try:
        subprocess.run(cmd, shell=True, timeout=timeout, capture_output=True)
        return True
    except subprocess.TimeoutExpired:
        print(f"  [warn] hMETIS timed out after {timeout}s — switching to greedy")
        return False


def _select_and_update(hg, partitions, covered_vertices, removed_edges,
                       scores, writer, stop_condition):
    """
    Pick best edge per partition, update all state.
    stop_condition: a callable() -> bool that signals early exit.
    """
    for part_edges in partitions.values():
        if stop_condition():
            break
        best_edge = _best_in_partition(part_edges, removed_edges, scores)
        if best_edge is not None:
            newly_covered = hg.hedges_dict[best_edge] - covered_vertices
            if newly_covered:
                covered_vertices.update(newly_covered)
                removed_edges.add(best_edge)
                scores[best_edge] = -1.0
                _update_scores(hg, scores, newly_covered, removed_edges)
                writer.update(best_edge, newly_covered)


# ─────────────────────────────────────────────
#  Stage checkpointing for set cover
# ─────────────────────────────────────────────

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
        self.snapshots  = [None, None]   # snapshot at 33% and 66%

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


# ─────────────────────────────────────────────
#  MCP
# ─────────────────────────────────────────────

def hmetis_mcp(hg, budget, filename, nparts=2, timeout=120, **kwargs):
    removed_edges    = set()
    covered_vertices = set()
    iteration        = 0
    write_time       = 0
    partition_time   = 0

    scores = _build_scores(hg, covered_vertices, removed_edges)
    writer = HgrWriter(hg)

    while len(removed_edges) < budget:
        if (hg.nhedges - len(removed_edges)) < nparts:
            break

        w = time.time()
        e_map_inv = writer.write(filename)
        write_time += time.time() - w

        p  = time.time()
        ok = _run_hmetis(filename, nparts, ub_flag=1, timeout=timeout)
        partition_time += time.time() - p
        if not ok:
            break

        with open(f"{filename}.part.{nparts}") as f:
            line = f.read().splitlines()
        if len(line) != len(e_map_inv):
            break

        partitions = _parse_partitions(line, e_map_inv)
        prev_len   = len(removed_edges)

        _select_and_update(
            hg, partitions, covered_vertices, removed_edges, scores, writer,
            stop_condition=lambda: len(removed_edges) >= budget
        )

        if len(removed_edges) == prev_len:
            break

        iteration += 1
        print(f"iter={iteration}, covered={len(covered_vertices)}, removed={len(removed_edges)}")

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time


def pure_greedy_mcp(hg, budget, filename=None, **kwargs):
    covered_vertices = set()
    removed_edges    = set()

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

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, None, None


# ─────────────────────────────────────────────
#  Set Cover
# ─────────────────────────────────────────────

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
        ok = _run_hmetis(filename, nparts, ub_flag=2, timeout=timeout)
        partition_time += time.time() - p

        if not ok:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores)
            tracker.check(covered_vertices, removed_edges)
            break

        with open(f"{filename}.part.{nparts}") as f:
            line = f.read().splitlines()

        if len(line) != len(e_map_inv):
            _greedy_fallback(hg, covered_vertices, removed_edges, scores)
            tracker.check(covered_vertices, removed_edges)
            break

        partitions = _parse_partitions(line, e_map_inv)
        prev_len   = len(removed_edges)

        _select_and_update(
            hg, partitions, covered_vertices, removed_edges, scores, writer,
            stop_condition=lambda: len(covered_vertices) >= hg.nvtxs
        )

        tracker.check(covered_vertices, removed_edges)

        if len(removed_edges) == prev_len:
            _greedy_fallback(hg, covered_vertices, removed_edges, scores)
            tracker.check(covered_vertices, removed_edges)
            break

        iteration += 1
        print(f"iter={iteration}, covered={len(covered_vertices)}, removed={len(removed_edges)}")

    final_time   = round(time.time() - start_time, 4)
    final_edges  = len(removed_edges)
    stages       = tracker.result(final_edges, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, write_time, partition_time, stages


def pure_greedy_set_cover(hg, filename=None, **kwargs):
    covered_vertices = set()
    removed_edges    = set()
    start_time       = time.time()
    tracker          = _StageTracker(hg.nvtxs, start_time)

    while len(covered_vertices) < hg.nvtxs:
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
        tracker.check(covered_vertices, removed_edges)

    final_time  = round(time.time() - start_time, 4)
    final_edges = len(removed_edges)
    stages      = tracker.result(final_edges, final_time)

    return (hg.nhedges, hg.nvtxs), covered_vertices, removed_edges, None, None, stages








