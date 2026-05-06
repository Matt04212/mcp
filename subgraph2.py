import io


class HgrWriter:
    """
    Incremental hMETIS file writer.

    Key insight: store each vertex's row as a frozenset of *original* edge IDs,
    not hMETIS vertex IDs. Translate to hMETIS IDs only at write() time using
    the current live-edge ordering. This avoids the stale-ID bug caused by
    contiguous renumbering after each removal.
    """

    def __init__(self, hg):
        self.hg = hg
        self._removed = set()
        self._covered = set()

        # For each vertex: set of incident original edge IDs still live.
        # None means the vertex is covered (row omitted from file).
        # This is the stable representation — never stores hMETIS IDs.
        self._incident = {}
        for vtx in hg.vtxs:
            self._incident[vtx] = set(hg.vtxs_dict[vtx])   # copy so we can mutate

    def update(self, newly_removed_edge, newly_covered_vertices):
        """
        Call this after each selection, before the next write().
        Updates _incident incrementally — O(pins of removed edge + newly covered).
        """
        # 1. Remove the edge from every vertex's incident set
        for vtx in self.hg.hedges_dict[newly_removed_edge]:
            if vtx in self._incident and self._incident[vtx] is not None:
                self._incident[vtx].discard(newly_removed_edge)

        self._removed.add(newly_removed_edge)

        # 2. Mark newly covered vertices as inactive
        self._covered.update(newly_covered_vertices)
        for vtx in newly_covered_vertices:
            self._incident[vtx] = None

    def write(self, filename):
        """
        Write the current subgraph to disk and return e_map_inv.
        Translates original edge IDs -> hMETIS vertex IDs fresh each call,
        but assembles the file in one bulk write.
        """
        # Build current e_map from live edges (contiguous 1-indexed)
        live_edges = [e for e in self.hg.hedges if e not in self._removed]
        e_map = {e: i + 1 for i, e in enumerate(live_edges)}
        e_map_inv = {i + 1: e for i, e in enumerate(live_edges)}

        buf = io.StringIO()
        # Write header placeholder — fill after counting valid rows
        rows = []
        for vtx in self.hg.vtxs:
            inc = self._incident[vtx]
            if inc is None:          # covered
                continue
            mapped = [e_map[e] for e in inc]   # translate to hMETIS IDs
            if mapped:               # skip isolated vertices
                rows.append(" ".join(map(str, mapped)) + "\n")

        buf.write(f"{len(rows)} {len(live_edges)}\n")
        buf.writelines(rows)

        with open(filename, 'w') as f:
            f.write(buf.getvalue())

        return e_map_inv


# ---------------------------------------------------------------------------
# Backwards-compatible one-shot function (greedy fallback paths)
# ---------------------------------------------------------------------------

def write_hgr(hg, covered_vertices, removed_edges, filename):
    live_edges = [e for e in hg.hedges if e not in removed_edges]
    e_map = {e: i + 1 for i, e in enumerate(live_edges)}
    e_map_inv = {i + 1: e for i, e in enumerate(live_edges)}

    buf = io.StringIO()
    rows = []
    for vtx in hg.vtxs:
        if vtx in covered_vertices:
            continue
        incident = [e_map[e] for e in hg.vtxs_dict[vtx] if e not in removed_edges]
        if incident:
            rows.append(" ".join(map(str, incident)) + "\n")

    buf.write(f"{len(rows)} {len(live_edges)}\n")
    buf.writelines(rows)

    with open(filename, 'w') as f:
        f.write(buf.getvalue())

    return e_map_inv