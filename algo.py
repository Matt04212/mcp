import subprocess
import time
import heapq
from greedy import greedy
from selection import selection
from subgraph import write_hgr
import numpy as np

def algo(hg, nparts, filename):
    covered_vertices = set()
    removed_edges = set()
    solu = []

    iteration = 0
    last_hmetis_covered = 0
    threshhold = int(0.05 * hg.nvtxs)

    while len(covered_vertices) <  hg.nvtxs:

        t = time.time()
        # use greedy if small enough
        live_vtxs = [v for v in hg.vtxs if v not in covered_vertices]
        """if len(live_vtxs) < 500:
            greedy(hg, covered_vertices, removed_edges, solu)

            break"""

        #if len(covered_vertices) - last_hmetis_covered >= threshhold:


        v_map_inv= write_hgr(hg, covered_vertices, removed_edges, filename, live_vtxs)
        subprocess.run(f"./hmetis {filename} {nparts} 5 1 2 1 0 0 0", shell=True)
        last_hmetis_covered = len(covered_vertices)


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

        #pick top-k
        selection(hg, covered_vertices, removed_edges, solu, partitions)
        """for n in partitions:
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
            solu.append(best_edge)"""

        iteration += 1
        print(f"iter{iteration}:covered = {len(covered_vertices)}, "
              f"removed_edges = {len(removed_edges)}, time = {time.time() - t:.2f}s")

        if iteration == 100:
            break

    return solu











