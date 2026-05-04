def write_hgr(hg, covered_vertices, removed_edges, filename):
    # edges become vertices, vertices become hyperedges
    live_edges = [e for e in hg.hedges if e not in removed_edges]

    # map original edge ID to hMETIS vertex ID (1-indexed)
    e_map = {e: i + 1 for i, e in enumerate(live_edges)}
    e_map_inv = {i + 1: e for i, e in enumerate(live_edges)}

    # each original vertex becomes a hyperedge
    # connecting all live edges that contain it
    valid_hedges = []
    for vtx in hg.vtxs:
        if vtx in covered_vertices:  # vertex already covered, skip
            continue
        incident_live_edges = [e_map[e] for e in hg.vtxs_dict[vtx]
                               if e not in removed_edges]
        if len(incident_live_edges) >= 0:  # only meaningful if connects 2+ edges
            valid_hedges.append(incident_live_edges)

    with open(filename, 'w') as f:
        f.write(f"{len(valid_hedges)} {len(live_edges)}\n")
        for hedge in valid_hedges:
            f.write(" ".join(str(v) for v in hedge) + "\n")

    return e_map_inv