import subprocess

def write_hgr(hg, covered_vertices, removed_edges, filename):
    live_vtxs = [v for v in hg.vtxs if v not in covered_vertices]
    #orginial -> hmetis id
    v_map = {v: i+1 for i, v in enumerate(live_vtxs)}
    #inverse
    v_map_inv = {i+1: v for i, v in enumerate(live_vtxs)}

    with open(filename, 'w') as f:
        f.write(f"{hg.nhedges} {hg.nvtxs}\n")

        for hedge in hg.hedges_dict:
            vtxs = hg.hedges_dict[hedge]
            line = " ".join(str(n) for n in vtxs)
            f.write(line + "\n")

    return v_map_inv, filename


def algo(hg, nparts, filename):
    covered_vertices = set()
    removed_edges = set()
    solu = []

    v_map_inv, filename = write_hgr(hg, covered_vertices, removed_edges, filename)

    subprocess.run(f"./hmetis {filename} {nparts} 5 10 2 1 0 0 0", shell=True)

    with open(f"{filename}.part.{nparts}", "r") as f:
        line = f.read().splitlines()

    partitions = {}
    for vtx, part in enumerate(line, start=1):
        original_vtx = v_map_inv[vtx]
        part = int(part)

        if part not in partitions:
            partitions[part] = set()

        partitions[part].add(original_vtx)

    for n in range(nparts):
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

        newly_coverd = hg.hedges_dict[best_edge] - covered_vertices
        covered_vertices.update(newly_coverd)
        removed_edges.add(best_edge)
        solu.append(best_edge)

    











