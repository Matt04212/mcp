import subprocess

def write_hgr(hg, covered_vertices, removed_edges, filename):
    live_vtxs = [v for v in hg.vtxs if v not in covered_vertices]
    #orginial -> hmetis id
    v_map = {v: i+1 for i, v in enumerate(live_vtxs)}
    #inverse
    v_map_inv = {i+1: v for i, v in enumerate(live_vtxs)}

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


def algo(hg, nparts, filename):
    covered_vertices = set()
    removed_edges = set()
    solu = []

    while len(covered_vertices) < hg.nvtxs:
        live_vtxs = [v for v in hg.vtxs if v not in covered_vertices]

        if len(live_vtxs) < nparts:
            # too small to partition, pick greedy
            for hedge in hg.hedges:
                if hedge not in removed_edges:
                    newly_coverd = hg.hedges_dict[hedge] - covered_vertices
                    if newly_coverd:
                        covered_vertices.update(newly_coverd)
                        removed_edges.add(best_edge)
                        solu.append(best_edge)

            break

        v_map_inv= write_hgr(hg, covered_vertices, removed_edges, filename)

        subprocess.run(f"./hmetis {filename} {nparts} 5 1 2 3 0 0 0", shell=True)

        with open(f"{filename}.part.{nparts}", "r") as f:
            line = f.read().splitlines()

        partitions = {}
        for vtx, part in enumerate(line, start=1):
            #print(f"partition file lines: {len(line)}, v_map_inv size: {len(v_map_inv)}")
            original_vtx = v_map_inv[vtx]
            part = int(part)

            if part not in partitions:
                partitions[part] = set()

            partitions[part].add(original_vtx)

        for n in partitions:
            best_score = 0
            best_edge = None
            candidate_hedge = set()
            for v in partitions[n]:
                for hedge in hg.vtxs_dict[v]:
                    if hedge not in removed_edges:
                        candidate_hedge.add(hedge)

            for hedge in candidate_hedge:
                score = len(hg.hedges_dict[hedge].intersection(partitions[n]))
                if score > best_score:
                    best_score = score
                    best_edge = hedge
                else:
                    continue

            if best_edge is None:
                continue

            newly_coverd = hg.hedges_dict[best_edge] - covered_vertices
            covered_vertices.update(newly_coverd)
            removed_edges.add(best_edge)
            solu.append(best_edge)

    return solu











