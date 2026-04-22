def write_hgr(hg, covered_vertices, removed_edges, filename):
    # filter to vertices that appear in at least one live edge
    live_vtxs = [v for v in hg.vtxs
                 if v not in covered_vertices
                 and any(e not in removed_edges for e in hg.vtxs_dict[v])]
    # orginial -> hmetis id
    v_map = {v: i + 1 for i, v in enumerate(live_vtxs)}
    # inverse
    v_map_inv = {i + 1: v for i, v in enumerate(live_vtxs)}

    valid_hedges = []
    for hedge in hg.hedges_dict:
        if hedge in removed_edges:
            continue
        live_vtxs_in_edge = [v_map[v] for v in hg.hedges_dict[hedge]
                             if v not in covered_vertices]
        if not live_vtxs_in_edge:
            continue
        valid_hedges.append(live_vtxs_in_edge)

    with open(filename, 'w') as f:
        f.write(f"{len(valid_hedges)} {len(live_vtxs)}\n")
        for edge in valid_hedges:
            f.write(" ".join(str(v) for v in edge) + "\n")

    return v_map_inv