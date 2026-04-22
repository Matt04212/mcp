import heapq
def selection(hg, covered_vertices, removed_edges, solu, partitions, k):

    # remove covered vertices from partitions so reuse is clean
    for n in partitions:
        print(f"part{n} : {len(partitions[n])}")
        partitions[n] -= covered_vertices
        candidate_hedge = set()
        for v in partitions[n]:
            for hedge in hg.vtxs_dict[v]:
                if hedge not in removed_edges:
                    candidate_hedge.add(hedge)

        top_k = heapq.nlargest(k, candidate_hedge,
                               key=lambda h: len((hg.hedges_dict[h] & partitions[n]) - covered_vertices))

        for hedge in top_k:
            newly_covered = hg.hedges_dict[hedge] - covered_vertices
            if newly_covered:
                covered_vertices.update(newly_covered)
                removed_edges.add(hedge)
                solu.append(hedge)