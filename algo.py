import subprocess, time, heapq
from greedy import greedy
import numpy as np
from subgraph import write_hgr
from selection import selection

def algo(hg, nparts, filename):
    covered_vertices = set()
    removed_edges = set()
    solu = []
    last_hmetis_covered = 0
    threshhold = int(0.05 * hg.nvtxs)
    iteration = 0
    while len(covered_vertices) <  hg.nvtxs:

        t = time.time()
        # use greedy if small enough
        live_count = hg.nvtxs - len(covered_vertices)
        """if live_count < 400:
            greedy(hg, covered_vertices, removed_edges, solu)

            break"""

        # write subgraph only if last covered >= threshold
        if iteration == 0 or len(covered_vertices) - last_hmetis_covered >= threshhold:


            v_map_inv= write_hgr(hg, covered_vertices, removed_edges, filename)
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
            print(f"number of partitions: {len(partitions)}")
            for n in partitions:
                print(f"partition {n}: {len(partitions[n])} vertices")


        #pick top-k
        selection(hg, covered_vertices, removed_edges, solu, partitions, 2)

        iteration += 1
        print(f"iter{iteration}:covered = {len(covered_vertices)}, "
              f"removed_edges = {len(removed_edges)}, time = {time.time() - t:.2f}s")

        if iteration == 10:
            break

    return len(removed_edges), len(covered_vertices)











