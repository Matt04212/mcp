import subprocess

def algo(hg, nparts):
    subprocess.run(f"./hmetis data1.hgr {nparts} 5 10 2 1 0 0 0", shell=True)

    with open(f'data1.hgr.part.{nparts}', "r") as f:
        line = f.read().splitlines()

    partitions = {}
    for vtx, part in enumerate(line, start=1):
        part = int(part)

        if part not in partitions:
            partitions[part] = set()

        partitions[part].update(hg.vtxs_dict[vtx])

    solu = set()
    for n in range(nparts):
        best_score = 0
        best_edge = None
        for hedge in hg.hedges:
            score = len(hg.hedges_dict[hedge].intersection(partitions[n]))
            if score > best_score and hedge not in solu:
                best_score = score
                best_edge = hedge
            else:
                continue

        print("edge:", best_edge)
        solu.add(best_edge)

    











