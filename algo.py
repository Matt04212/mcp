import subprocess

def algo(hg, nparts):
    covered_vertices = set()
    removed_edges = set()
    solu = []

    subprocess.run(f"./hmetis data1.hgr {nparts} 5 10 2 1 0 0 0", shell=True)

    with open(f'data1.hgr.part.{nparts}', "r") as f:
        line = f.read().splitlines()

    partitions = {}
    for vtx, part in enumerate(line, start=1):
        part = int(part)

        if part not in partitions:
            partitions[part] = set()

        partitions[part].add(vtx)

    for n in range(nparts):
        best_score = 0
        best_edge = None
        candidate_hedge = set()
        for v in partitions[n]:
            for hedge in hg.vtxs_dict[v]:
                if hedge not in removed_edges:
                    candidate_hedge.add(hedge)

        for hedge in candidate_hedge:
            score = len(hg.hedges_dict[hedge].intersection(partition[n]))
            if score > best_score:
                best_score = score
                best_edge = hedge
            else:
                continue

        print("edge:", best_edge)
        solu.add(best_edge)

    











